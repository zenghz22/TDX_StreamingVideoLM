"""video_token_prune_td.py
Question-agnostic visual token pruning for TDX streaming video encoding.

两级剪枝，均无需训练，与问题内容无关：

Level 0 — Temporal Frame Deduplication（帧级时序去冗余）
  在 processor 调用之前，对每个 chunk 的帧序列做过滤：
  - 相邻帧像素空间余弦相似度 > threshold → 跳过该帧
  - 直接减少帧数，从根源降低 visual token 总量
  - 实现：纯 numpy，零额外依赖，零模型调用

Level 1 — Spatial Token Pruning（ViT 输出后 token 空间剪枝）
  通过 forward hook 挂在 model.vision_tower 上：
  - 捕获 ViT 输出 [F, 729, D]（SigLIP SO400M, 27×27 grid）
  - 对每帧做 token 重要性排序，保留 top-k
  - 重要性度量（question-agnostic）：
    a) Feature norm：token 的 L2 范数（背景/均匀区域 norm 低）
    b) Feature variance：token 在 channel 维的方差（信息量代理）
    c) Spatial diversity (ToMe-lite)：简化版双边匹配，去除高度相似 token
  - 剪枝后输出 [F, k, D]，后续 2D pool 和 projector 自适应处理

设计原则
--------
- ctx=None 或 disabled=True 时所有操作 no-op，透明降级
- 两级独立，可单独启用
- 不修改模型权重，不需要反向传播
- CPU 友好：避免 O(N²) 全局相似度，改用局部窗口或 top-k norm

集成方式（在 kvcache_generate_td.py 的 encode 循环中）
------------------------------------------------------
    from video_token_prune_td import PruneContext, temporal_filter_chunk, install_spatial_hook, remove_spatial_hook

    prune_ctx = PruneContext(
        temporal_threshold=0.95,   # 帧相似度阈值，越高保留越多帧
        spatial_keep_ratio=0.5,    # 每帧保留比例，0.5 = 196→98 tokens
        spatial_metric="norm",     # "norm" | "var" | "tome"
    )

    # 在 encode 循环外安装 hook（一次）
    hook_handle = install_spatial_hook(model, prune_ctx)

    for i in range(num_chunks):
        chunk = video[i * chunk_size : ...]

        # Level 0: temporal dedup（修改 chunk，之后正常走 processor）
        chunk = temporal_filter_chunk(chunk, prune_ctx, chunk_idx=i)
        if len(chunk) == 0:
            continue  # 极端情况：整个 chunk 被过滤掉

        model_inputs = processor(text=[encode_text], videos=[chunk], ...)
        # Level 1: 在 model forward 内部自动触发 hook

    # 在 encode 完成后移除 hook
    remove_spatial_hook(hook_handle)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PruneContext：统一配置对象
# ---------------------------------------------------------------------------

@dataclass
class PruneContext:
    """
    控制两级剪枝行为的配置对象。

    Parameters
    ----------
    enabled : bool
        总开关。False 时所有操作 no-op。

    --- Level 0: Temporal Frame Deduplication ---
    temporal_enabled : bool
        是否启用帧级时序去冗余。
    temporal_threshold : float
        相邻帧余弦相似度阈值。>=threshold 的帧被跳过。
        推荐范围 [0.90, 0.98]。视频越静态越低。
    temporal_min_frames : int
        每个 chunk 至少保留的帧数，避免过度过滤。
    temporal_downsample : int
        计算相似度前将帧 resize 到 (H/d, W/d)，加速计算。

    --- Level 1: Spatial Token Pruning ---
    spatial_enabled : bool
        是否启用空间 token 剪枝（需安装 hook）。
    spatial_keep_ratio : float
        每帧保留的 token 比例。0.5 = 196→98 tokens/frame。
    spatial_metric : str
        重要性度量方式：
        "norm" - token L2 范数（最快，CPU 友好）
        "var"  - token channel 方差（稍慢但更鲁棒）
        "tome" - 简化版 bipartite matching（最准但最慢）
    spatial_min_tokens : int
        每帧至少保留的 token 数，避免过度剪枝。

    --- Logging ---
    log_stats : bool
        是否打印剪枝统计信息。
    """
    enabled: bool = True

    # Level 0
    temporal_enabled: bool = True
    temporal_threshold: float = 0.95
    temporal_min_frames: int = 2
    temporal_downsample: int = 8

    # Level 1
    spatial_enabled: bool = True
    spatial_keep_ratio: float = 0.5
    spatial_metric: str = "norm"   # "norm" | "var" | "tome"
    spatial_min_tokens: int = 49   # 7×7

    log_stats: bool = True

    # Internal: stats accumulators
    _temporal_stats: List[dict] = field(default_factory=list, repr=False)
    _spatial_stats: List[dict] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        n_t = len(self._temporal_stats)
        n_s = len(self._spatial_stats)

        lines = ["[PruneContext] Pruning summary:"]

        if n_t > 0:
            orig  = sum(s["original_frames"] for s in self._temporal_stats)
            kept  = sum(s["kept_frames"] for s in self._temporal_stats)
            ratio = kept / orig if orig > 0 else 1.0
            lines.append(
                f"  Temporal (Level 0): {orig} frames → {kept} frames "
                f"({ratio:.1%} kept, threshold={self.temporal_threshold})"
            )
            lines.append(
                f"    Effect: fewer frames enter processor → fewer pixel_values → "
                f"ViT processes fewer images → proportionally fewer KV tokens in LLM"
            )
        else:
            lines.append("  Temporal (Level 0): not applied")

        if n_s > 0:
            orig  = sum(s["original_tokens"] for s in self._spatial_stats)
            kept  = sum(s["kept_tokens"] for s in self._spatial_stats)
            ratio = kept / orig if orig > 0 else 1.0
            tpf_b = self._spatial_stats[0].get("tokens_per_frame_before", "?")
            tpf_a = self._spatial_stats[0].get("tokens_per_frame_after", "?")
            lines.append(
                f"  Spatial  (Level 1): {orig} tokens → {kept} tokens "
                f"({ratio:.1%} kept, metric={self.spatial_metric})"
            )
            lines.append(
                f"    Per frame: {tpf_b} → {tpf_a} tokens "
                f"(hook on multi_modal_projector, after 2D pool)"
            )
            lines.append(
                f"    Effect: fewer tokens enter LLM attention → "
                f"proportionally smaller KV cache per chunk"
            )
        else:
            lines.append("  Spatial  (Level 1): not applied")

        # combined estimate
        if n_t > 0 and n_s > 0:
            t_r = (sum(s["kept_frames"] for s in self._temporal_stats) /
                   max(1, sum(s["original_frames"] for s in self._temporal_stats)))
            s_r = (sum(s["kept_tokens"] for s in self._spatial_stats) /
                   max(1, sum(s["original_tokens"] for s in self._spatial_stats)))
            combined = t_r * s_r
            lines.append(f"  Combined KV cache reduction: ~{1-combined:.0%} fewer entries")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Level 0: Temporal Frame Deduplication
# ---------------------------------------------------------------------------

def _frame_to_feature(frame: np.ndarray, downsample: int) -> np.ndarray:
    """将单帧降采样后展平为 1D 特征向量（纯 numpy，无 ViT）。"""
    # frame: [H, W, C] uint8 or float32
    h, w = frame.shape[:2]
    nh, nw = max(1, h // downsample), max(1, w // downsample)
    # 简单块平均降采样（等价于 PIL.resize BILINEAR）
    bh, bw = h // nh, w // nw
    f = frame[:nh * bh, :nw * bw].reshape(nh, bh, nw, bw, -1).mean(axis=(1, 3))
    feat = f.flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm
    return feat


def temporal_filter_chunk(
    chunk: np.ndarray,
    ctx: Optional[PruneContext],
    *,
    chunk_idx: int = 0,
) -> np.ndarray:
    """
    对单个 chunk 的帧序列做时序去冗余。

    Parameters
    ----------
    chunk : np.ndarray
        shape [F, H, W, C]，uint8 或 float32。
    ctx : PruneContext | None
        None 或 enabled=False 时原样返回 chunk。
    chunk_idx : int
        仅用于日志。

    Returns
    -------
    np.ndarray
        过滤后的帧序列，shape [F', H, W, C]，F' <= F。
    """
    if ctx is None or not ctx.enabled or not ctx.temporal_enabled:
        return chunk

    F = len(chunk)
    if F <= ctx.temporal_min_frames:
        return chunk

    kept_indices = [0]  # 始终保留第一帧
    prev_feat = _frame_to_feature(chunk[0], ctx.temporal_downsample)

    for i in range(1, F):
        curr_feat = _frame_to_feature(chunk[i], ctx.temporal_downsample)
        sim = float(np.dot(prev_feat, curr_feat))  # 两者已归一化，直接点积=cosine sim

        if sim < ctx.temporal_threshold:
            kept_indices.append(i)
            prev_feat = curr_feat  # 更新参考帧为最近一个被保留的帧
        # else: 帧太相似，跳过

    # 保证 min_frames
    if len(kept_indices) < ctx.temporal_min_frames:
        # 均匀补充
        step = max(1, F // ctx.temporal_min_frames)
        extra = [j for j in range(0, F, step) if j not in kept_indices]
        kept_indices = sorted(set(kept_indices) | set(extra[:ctx.temporal_min_frames]))

    kept_indices = sorted(kept_indices)
    result = chunk[np.array(kept_indices)]

    ratio = len(result) / F if F > 0 else 1.0
    stat = {
        "chunk_idx": chunk_idx,
        "original_frames": F,
        "kept_frames": len(result),
        "kept_ratio": ratio,
        "kept_indices": kept_indices,
    }
    ctx._temporal_stats.append(stat)

    if ctx.log_stats:
        logger.info(
            f"[prune:temporal] chunk {chunk_idx}: "
            f"{F} → {len(result)} frames ({ratio:.0%} kept, "
            f"threshold={ctx.temporal_threshold}, "
            f"kept_indices={kept_indices})"
        )
    return result


# ---------------------------------------------------------------------------
# Level 1: Spatial Token Pruning via ViT Output Hook
# ---------------------------------------------------------------------------

def _score_by_norm(features: torch.Tensor) -> torch.Tensor:
    """
    重要性 = token 的 L2 范数。

    背景/均匀区域对应低范数 token，信息量少。
    形状：[N, D] → [N]
    快速，适合 CPU。
    """
    return features.float().norm(dim=-1)


def _score_by_var(features: torch.Tensor) -> torch.Tensor:
    """
    重要性 = token 在 channel 维的方差。

    高方差意味着 token 在不同 channel 上有差异化激活，信息更丰富。
    形状：[N, D] → [N]
    """
    return features.float().var(dim=-1)


def _score_by_tome_lite(features: torch.Tensor, r: int) -> torch.Tensor:
    """
    简化版 Token Merging (ToMe) 得分：
    对每个 token 计算与其"最近邻"的相似度，
    相似度越高说明越冗余（得分越低）。

    原始 ToMe 做双边匹配，这里做单边近似：
    - 取前 sqrt(N) 个 token 作为 "anchors"
    - 每个 token 与最相似的 anchor 计算 cosine sim
    - 得分 = 1 - max_sim（冗余度越低得分越高）

    形状：[N, D] → [N]
    """
    N, D = features.shape
    if N <= r + 1:
        return torch.ones(N, device=features.device)

    feats_norm = F.normalize(features.float(), dim=-1)
    # 随机采样 anchors（防止 O(N²)）
    n_anchors = min(64, N // 2)
    anchor_idx = torch.randperm(N, device=features.device)[:n_anchors]
    anchors = feats_norm[anchor_idx]  # [n_anchors, D]

    # 每个 token 与所有 anchor 的 cosine sim
    sim = (feats_norm @ anchors.T)  # [N, n_anchors]
    max_sim, _ = sim.max(dim=-1)     # [N]
    return 1.0 - max_sim             # 越不像 anchor → 越独特 → 得分越高


def _prune_frame_tokens(
    frame_features: torch.Tensor,
    keep: int,
    metric: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单帧的 token 序列打分并保留 top-keep 个。

    Parameters
    ----------
    frame_features : [N, D]
    keep : int  要保留的 token 数
    metric : str

    Returns
    -------
    pruned_features : [keep, D]
    kept_indices    : [keep] long tensor，原始索引（有序）
    """
    N, D = frame_features.shape
    keep = min(keep, N)

    if metric == "norm":
        scores = _score_by_norm(frame_features)
    elif metric == "var":
        scores = _score_by_var(frame_features)
    elif metric == "tome":
        r = N - keep  # 要丢弃的数量
        scores = _score_by_tome_lite(frame_features, r)
    else:
        raise ValueError(f"Unknown spatial_metric: {metric!r}, expected 'norm'|'var'|'tome'")

    # 保留得分最高的 keep 个（topk 稳定性用 stable sort）
    topk_indices = scores.topk(keep, largest=True, sorted=True).indices
    kept_indices, _ = topk_indices.sort()  # 保持空间顺序
    return frame_features[kept_indices], kept_indices


class _SpatialPruneHook:
    """
    挂在 model.multi_modal_projector 上的 forward hook。

    ── 为什么挂在 projector 而非 vision_tower ──
    vision_tower (SigLIP) 输出 [B, 729, D_vit]，B = 总帧数（batch×frames）。
    之后 LLaVA-OneVision 的 multi_modal_projector 内部有 get_2dPool（stride=2），
    把 729 → 196（14×14 grid），再通过 MLP 投影到 D_llm。
    2D pool 里有 sqrt(N) 的硬编码 reshape；如果在 pool 之前改变 token 数，
    非方形 shape 会直接触发 RuntimeError。

    正确插入点：projector 的 forward 输出，形状 [total_tokens, D_llm]，
    其中 total_tokens = total_frames × 196。
    在这里剪枝，每帧从 196 → k，真正减少进入 LLM 的 visual token 数，
    等比例减少 KV cache 大小。

    debug 模式下会打印实际捕获的 tensor 形状，方便验证 hook 生效位置。
    """

    def __init__(self, ctx: PruneContext, num_frames_per_chunk: int = 16):
        self.ctx = ctx
        self.num_frames_per_chunk = num_frames_per_chunk  # 每次 forward 的帧数（temporal filter 后的实际帧数）
        self._call_count = 0

    def update_num_frames(self, n: int):
        """encode 循环每次 forward 前更新本次实际帧数（temporal filter 后可能改变）。"""
        self.num_frames_per_chunk = n

    def __call__(self, module, input, output):
        ctx = self.ctx
        self._call_count += 1

        if not ctx.enabled or not ctx.spatial_enabled:
            return output

        # projector 输出：[total_tokens, D_llm]，total_tokens = frames × tokens_per_frame
        if isinstance(output, tuple):
            features = output[0]
        else:
            features = output

        if features.dim() != 2:
            logger.warning(
                f"[prune:spatial] Unexpected projector output dim={features.dim()} "
                f"shape={tuple(features.shape)}, expected 2D. Skipping. "
                f"(call #{self._call_count})"
            )
            return output

        total_tokens, D = features.shape
        F = self.num_frames_per_chunk

        # 推断每帧 token 数（应为 196 = 14×14，pool stride=2 from 729）
        if F == 0 or total_tokens % F != 0:
            logger.warning(
                f"[prune:spatial] total_tokens={total_tokens} not divisible by F={F}. "
                f"Skipping pruning. (call #{self._call_count})"
            )
            return output

        tokens_per_frame = total_tokens // F
        keep = max(ctx.spatial_min_tokens, round(tokens_per_frame * ctx.spatial_keep_ratio))

        logger.debug(
            f"[prune:spatial] call #{self._call_count}: "
            f"projector out shape={tuple(features.shape)}, "
            f"F={F}, tokens_per_frame={tokens_per_frame}, keep={keep}"
        )

        if keep >= tokens_per_frame:
            logger.info(
                f"[prune:spatial] keep={keep} >= tokens_per_frame={tokens_per_frame}, "
                f"no pruning applied."
            )
            return output

        # reshape → [F, tokens_per_frame, D]，逐帧剪枝，再 flatten
        features_3d = features.view(F, tokens_per_frame, D)
        pruned_frames = []
        for f_idx in range(F):
            pruned_feat, kept_idx = _prune_frame_tokens(
                features_3d[f_idx],
                keep=keep,
                metric=ctx.spatial_metric,
            )
            pruned_frames.append(pruned_feat)

        pruned_3d = torch.stack(pruned_frames, dim=0)          # [F, keep, D]
        pruned_flat = pruned_3d.view(F * keep, D)              # [F×keep, D]

        stat = {
            "original_tokens": total_tokens,
            "kept_tokens": F * keep,
            "tokens_per_frame_before": tokens_per_frame,
            "tokens_per_frame_after": keep,
            "keep_ratio_actual": keep / tokens_per_frame,
        }
        ctx._spatial_stats.append(stat)

        logger.info(
            f"[prune:spatial] chunk forward #{self._call_count}: "
            f"{F} frames × {tokens_per_frame} → {F} frames × {keep} tokens "
            f"({total_tokens} → {F*keep}, ratio={keep/tokens_per_frame:.1%}, "
            f"metric={ctx.spatial_metric})"
        )

        if isinstance(output, tuple):
            return (pruned_flat,) + output[1:]
        return pruned_flat


def install_spatial_hook(model, ctx: Optional[PruneContext]):
    """
    在 model.multi_modal_projector 上安装空间 token 剪枝 hook。

    挂载点选择 multi_modal_projector（pool+MLP 之后），原因：
    - vision_tower 输出 [B, 729, D_vit]；projector 内部 2D pool（sqrt(N) reshape）
      要求 N=729，提前改变 token 数会触发 RuntimeError。
    - projector 输出 [total_tokens, D_llm]，total_tokens = frames×196，
      在此剪枝直接减少进入 LLM 的 visual token 数，是真正有效的节约点。

    Returns
    -------
    hook_obj : _SpatialPruneHook（调用者需在每次 forward 前调用 hook_obj.update_num_frames(F)）
    handle   : RemovableHook
    两者均为 None 时表示未安装。
    """
    if ctx is None or not ctx.enabled or not ctx.spatial_enabled:
        return None, None

    hook_obj = _SpatialPruneHook(ctx)

    proj = getattr(model, "multi_modal_projector", None)
    if proj is None:
        logger.warning(
            "[prune:spatial] model.multi_modal_projector not found. "
            "Trying vision_tower as fallback (may not work correctly)."
        )
        vt = getattr(model, "vision_tower", None)
        if vt is None:
            logger.warning("[prune:spatial] Neither projector nor vision_tower found. Hook not installed.")
            return None, None
        handle = vt.register_forward_hook(hook_obj)
        logger.warning("[prune:spatial] Fallback: hook on vision_tower (spatial pruning may be ineffective).")
        return hook_obj, handle

    handle = proj.register_forward_hook(hook_obj)
    logger.info(
        f"[prune:spatial] Hook installed on multi_modal_projector "
        f"(keep_ratio={ctx.spatial_keep_ratio}, metric={ctx.spatial_metric}). "
        f"Visual tokens per frame: 196 → {max(ctx.spatial_min_tokens, round(196*ctx.spatial_keep_ratio))}"
    )
    return hook_obj, handle


def remove_spatial_hook(handle) -> None:
    """移除之前安装的 hook（handle 为 install_spatial_hook 返回的第二个值）。"""
    if handle is not None:
        handle.remove()
        logger.info("[prune:spatial] Hook removed from multi_modal_projector.")


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

def make_prune_context(
    temporal_threshold: float = 0.95,
    spatial_keep_ratio: float = 0.5,
    spatial_metric: str = "norm",
    temporal_only: bool = False,
    spatial_only: bool = False,
) -> PruneContext:
    """
    快捷构建 PruneContext。

    典型配置：
      轻量（省内存 ~30%）：temporal=0.92, spatial_keep=0.7, metric="norm"
      中等（省内存 ~50%）：temporal=0.95, spatial_keep=0.5, metric="norm"
      激进（省内存 ~70%）：temporal=0.97, spatial_keep=0.3, metric="var"
    """
    return PruneContext(
        temporal_enabled=not spatial_only,
        temporal_threshold=temporal_threshold,
        spatial_enabled=not temporal_only,
        spatial_keep_ratio=spatial_keep_ratio,
        spatial_metric=spatial_metric,
    )


# ---------------------------------------------------------------------------
# 内存节省估算
# ---------------------------------------------------------------------------

def estimate_savings(
    num_frames: int,
    chunk_size: int = 16,
    tokens_per_frame: int = 196,
    ctx: Optional[PruneContext] = None,
) -> dict:
    """
    粗估剪枝后的 token 和 KV cache 节省量（不运行模型）。

    KV cache 大小 ∝ num_layers × 2 × total_tokens × (num_heads × head_dim)
    此函数只估算 total_tokens 减少量。
    """
    if ctx is None:
        keep_t = 1.0
        keep_s = 1.0
    else:
        # 时序去冗余的近似保留率（难以精确预测，给个上界估计）
        keep_t = 1.0 - (1.0 - ctx.temporal_threshold) * 5 if ctx.temporal_enabled else 1.0
        keep_t = max(0.1, min(1.0, keep_t))
        keep_s = ctx.spatial_keep_ratio if ctx.spatial_enabled else 1.0

    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    orig_tokens = num_chunks * chunk_size * tokens_per_frame
    pruned_tokens = orig_tokens * keep_t * keep_s

    return {
        "original_total_tokens": int(orig_tokens),
        "estimated_pruned_tokens": int(pruned_tokens),
        "estimated_reduction_ratio": 1.0 - pruned_tokens / orig_tokens,
        "temporal_keep_estimate": keep_t,
        "spatial_keep": keep_s,
    }


# ---------------------------------------------------------------------------
# 集成示例（可直接复制到 kvcache_generate_td.py 的 encode_video 函数中）
# ---------------------------------------------------------------------------

INTEGRATION_EXAMPLE = """
# ===== 在 kvcache_generate_td.py 的 encode_video 函数中接入 =====

from video_token_prune_td import (
    make_prune_context, temporal_filter_chunk,
    install_spatial_hook, remove_spatial_hook,
)

# --- 在 encode_video 函数的参数中加 prune_ctx=None ---
def encode_video(video, processor, model, ..., prune_ctx=None):

    # 安装 Level 1 hook（在循环外，一次性）
    spatial_hook = install_spatial_hook(model, prune_ctx)

    try:
        for i in range(num_chunks):
            chunk = video[i * chunk_size : ...]

            # Level 0: 时序帧过滤（纯 numpy，在 processor 之前）
            chunk = temporal_filter_chunk(chunk, prune_ctx, chunk_idx=i)
            if len(chunk) == 0:
                logger.warning(f"[prune] chunk {i} 全部帧被过滤，跳过")
                continue

            # 之后正常走 processor + model forward（Level 1 hook 自动生效）
            model_inputs = processor(text=[encode_text], videos=[chunk], ...)
            ...
    finally:
        remove_spatial_hook(spatial_hook)

    if prune_ctx is not None and prune_ctx.log_stats:
        print(prune_ctx.summary())

# --- 调用时传入 prune_ctx ---
prune_ctx = make_prune_context(
    temporal_threshold=0.95,
    spatial_keep_ratio=0.5,
    spatial_metric="norm",
)
encode_video_managed(video, processor, model, ..., prune_ctx=prune_ctx)
"""