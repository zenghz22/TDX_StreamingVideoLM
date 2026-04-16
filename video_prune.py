"""video_token_prune_td.py
Question-agnostic visual token pruning for TDX streaming video encoding.

设计说明
--------
针对 LLaVA-OneVision 架构，实际可行的剪枝只有以下两种方式：

Level 0 — Temporal Frame Deduplication（帧级时序去冗余）  ✅ 有效
  在 processor 调用之前，对每个 chunk 的帧序列做相似度过滤。
  减少帧数 → processor 产生更少 pixel_values → ViT 处理更少图像 →
  model_inputs["input_ids"] 变短 → attention_mask 正确缩短 →
  LLM 生成更少 KV token。全链路安全。

Level 1 — Pixel-level Spatial Downscale（像素级降分辨率）  ✅ 有效
  在 processor 调用之前，将每帧 resize 到更小的分辨率（如 224×224）。
  小分辨率 → ViT patch 数减少 → 每帧 visual token 数减少 →
  同样在 processor 构建 attention_mask 之前完成，全链路安全。

❌ 为什么不能做 mid-forward hook 剪枝：
  LLaVA-OneVision 的 processor 在调用 model.forward 之前根据帧数和
  tokens_per_frame 预构建 attention_mask（shape = [1, past+current]），
  current_text_len = F × 196 × 1 是硬编码的。
  若在 forward 途中（projector hook 或 vision_tower hook）改变 token 数，
  attention_mask 维度不匹配，必然崩溃或产生错误结果。

  multi_modal_projector 输出是 [F, 729, D_llm]（仅 MLP，未 pool），
  之后主模型的 get_2dPool 方法才做 27×27 → 14×14。
  hook 返回 [F, k, D_llm] 后续 get_2dPool 也会 shape 错误。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PruneContext
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
    temporal_keep_ratio : float
        目标保留帧比例（0, 1]）。0.6 = 每个 chunk 保留约 60% 的帧。
        内部根据帧间相似度分布自动推算阈值（见 temporal_filter_chunk 说明）。
        推荐范围 [0.4, 0.9]；越低压缩越激进，静态视频效果越好。
    temporal_min_frames : int
        每个 chunk 至少保留的帧数，防止过度过滤。
    temporal_downsample : int
        计算相似度时的降采样倍数（越大越快，建议 4~16）。

    --- Level 1: Pixel-level Spatial Downscale ---
    spatial_enabled : bool
    spatial_ratio : float | None
        像素保留比例（0, 1]）。0.5 = 面积缩小到 50%，
        等价于长宽均乘以 sqrt(0.5) ≈ 0.707，保留宽高比。
        None 表示不做空间降采样。
        效果：ViT patch 数 ∝ spatial_ratio，KV 等比减少。
        推荐范围 [0.3, 0.7]；低于 0.3 视觉细节损失较大。
    """
    enabled: bool = True

    temporal_enabled: bool = True
    temporal_keep_ratio: float = 0.6      # 用户直接设置保留比例
    temporal_min_frames: int = 2
    temporal_downsample: int = 8

    spatial_enabled: bool = False         # 默认关闭，需要显式开启
    spatial_ratio: Optional[float] = None # None = 不做空间降分辨率

    log_stats: bool = True

    _temporal_stats: List[dict] = field(default_factory=list, repr=False)
    _spatial_stats: List[dict] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        lines = ["[PruneContext] ─── Pruning summary ───"]

        # Temporal
        n_t = len(self._temporal_stats)
        if n_t > 0:
            orig  = sum(s["original_frames"] for s in self._temporal_stats)
            kept  = sum(s["kept_frames"]     for s in self._temporal_stats)
            ratio = kept / orig if orig > 0 else 1.0
            lines.append(
                f"  Level 0 temporal:  {orig} frames → {kept} frames "
                f"({ratio:.1%} kept, target={self.temporal_keep_ratio:.0%})"
            )
            lines.append(
                "    Mechanism: fewer frames → fewer pixel_values → fewer ViT images "
                "→ shorter input_ids → smaller KV cache  [full-pipeline effective]"
            )
        else:
            lines.append("  Level 0 temporal:  not applied")

        # Spatial
        n_s = len(self._spatial_stats)
        if n_s > 0:
            orig  = sum(s["original_pixels"] for s in self._spatial_stats)
            kept  = sum(s["resized_pixels"]  for s in self._spatial_stats)
            ratio = kept / orig if orig > 0 else 1.0
            s0 = self._spatial_stats[0]
            lines.append(
                f"  Level 1 spatial:   {s0['original_H']}×{s0['original_W']} "
                f"→ {s0['new_H']}×{s0['new_W']} per frame  "
                f"(spatial_ratio={self.spatial_ratio}, area={ratio:.1%} kept)"
            )
            lines.append(
                f"    Mechanism: scale={s0['scale_factor']:.3f}×, "
                f"~{s0['orig_patches_est']} → ~{s0['new_patches_est']} ViT patches per frame  "
                f"[full-pipeline effective, aspect ratio preserved]"
            )
        else:
            lines.append("  Level 1 spatial:   not applied (spatial_enabled=False or spatial_size=None)")

        # Combined estimate
        if n_t > 0 or n_s > 0:
            t_r = (kept / orig) if n_t > 0 else 1.0
            # for t_r we already have it if n_t > 0, recalc:
            if n_t > 0:
                t_orig = sum(s["original_frames"] for s in self._temporal_stats)
                t_kept = sum(s["kept_frames"]     for s in self._temporal_stats)
                t_r = t_kept / t_orig if t_orig > 0 else 1.0
            else:
                t_r = 1.0

            if n_s > 0:
                s_orig = sum(s["original_pixels"] for s in self._spatial_stats)
                s_kept = sum(s["resized_pixels"]  for s in self._spatial_stats)
                s_r = s_kept / s_orig if s_orig > 0 else 1.0
            else:
                s_r = 1.0

            # KV cache reduction = temporal × (spatial token ratio ≈ (target/384)²)
            combined = t_r * s_r
            lines.append(
                f"  Estimated KV cache reduction:  ~{1 - combined:.0%} fewer entries "
                f"(temporal: {1-t_r:.0%}, spatial: {1-s_r:.0%})"
            )

        lines.append("─" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Level 0: Temporal Frame Deduplication
# ---------------------------------------------------------------------------

def _frame_to_feature(frame: np.ndarray, downsample: int) -> np.ndarray:
    """将单帧降采样后展平为归一化 1D 特征（纯 numpy）。"""
    h, w = frame.shape[:2]
    nh = max(1, h // downsample)
    nw = max(1, w // downsample)
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
    对单个 chunk 的帧序列做时序去冗余（在 processor 调用之前）。

    设计：用户设置 temporal_keep_ratio（保留比例），内部自动推算阈值。

    推算方式
    --------
    1. 计算每帧与其前一帧（frame[i-1]，不是上一个保留帧）的余弦相似度，
       得到 F-1 个相似度值。
    2. 将帧按"对前一帧的相似度"升序排列——最不相似（信息量最多）的帧
       优先保留。
    3. 始终保留 frame 0。保留另外 target_keep-1 个最不相似的帧。
    4. 日志中报告"有效阈值"= 被保留帧里最高的相似度，即自动推算的
       cosine threshold。这可以帮助用户理解 keep_ratio 对应的实际阈值。

    Parameters
    ----------
    chunk : np.ndarray  [F, H, W, C]  uint8 or float32
    ctx   : PruneContext | None

    Returns
    -------
    np.ndarray  [F', H, W, C]，F' ≤ F，保持原始时序顺序
    """
    if ctx is None or not ctx.enabled or not ctx.temporal_enabled:
        return chunk

    F = len(chunk)
    target_keep = max(ctx.temporal_min_frames, round(F * ctx.temporal_keep_ratio))
    target_keep = min(target_keep, F)  # 不超过总帧数

    if target_keep >= F:
        if ctx.log_stats:
            logger.debug(
                f"[prune:temporal] chunk {chunk_idx}: "
                f"target_keep={target_keep} >= F={F}, no filtering."
            )
        return chunk

    # ── 计算每帧与其前一帧（i-1）的余弦相似度 ──────────────────────
    features = [_frame_to_feature(chunk[i], ctx.temporal_downsample) for i in range(F)]
    # adj_sims[i] = sim(frame[i], frame[i-1])，i = 1..F-1
    adj_sims = [float(np.dot(features[i - 1], features[i])) for i in range(1, F)]

    # ── 选出最不相似的 target_keep-1 帧（加 frame 0 = target_keep 帧）──
    # 按 adj_sim 升序排列（越低越不相似，越值得保留）
    ranked = sorted(range(1, F), key=lambda i: adj_sims[i - 1])
    kept_rest = sorted(ranked[:target_keep - 1])   # 保留时序顺序
    kept_indices = [0] + kept_rest

    result = chunk[np.array(kept_indices)]
    actual_ratio = len(result) / F

    # ── 有效阈值：被保留帧里的最高相似度（即自动推算的 cosine threshold）
    effective_threshold = (
        max(adj_sims[i - 1] for i in kept_rest)
        if kept_rest else 0.0
    )

    # ── 相似度分布摘要 ─────────────────────────────────────────────
    sim_min  = min(adj_sims)
    sim_max  = max(adj_sims)
    sim_mean = sum(adj_sims) / len(adj_sims)
    skipped_sims = [adj_sims[i - 1] for i in ranked[target_keep - 1:]]
    skipped_sim_min = min(skipped_sims) if skipped_sims else float("nan")

    stat = {
        "chunk_idx":        chunk_idx,
        "original_frames":  F,
        "kept_frames":      len(result),
        "kept_ratio":       actual_ratio,
        "target_keep_ratio": ctx.temporal_keep_ratio,
        "effective_threshold": effective_threshold,
        "kept_indices":     kept_indices,
        "adj_sims":         adj_sims,
    }
    ctx._temporal_stats.append(stat)

    if ctx.log_stats:
        logger.info(
            f"[prune:temporal] chunk {chunk_idx}: "
            f"{F} → {len(result)} frames  "
            f"(target={ctx.temporal_keep_ratio:.0%}, actual={actual_ratio:.0%})  "
            f"effective_threshold={effective_threshold:.4f}  "
            f"adj_sim range=[{sim_min:.3f}, {sim_max:.3f}] mean={sim_mean:.3f}  "
            f"skipped_sim_min={skipped_sim_min:.3f}  "
            f"kept_indices={kept_indices}"
        )

    return result


# ---------------------------------------------------------------------------
# Level 1: Pixel-level Spatial Downscale
# ---------------------------------------------------------------------------

def spatial_downscale_chunk(
    chunk: np.ndarray,
    ctx: Optional[PruneContext],
    *,
    chunk_idx: int = 0,
) -> np.ndarray:
    """
    对单个 chunk 的每帧做等比例降分辨率（在 processor 调用之前）。

    设计：用户设置 spatial_ratio（像素面积保留比例），内部等比缩放，
    保留原始宽高比。

    缩放公式
    --------
    scale = sqrt(spatial_ratio)          # 线性尺寸缩放因子
    new_H = round(H * scale)
    new_W = round(W * scale)

    示例（spatial_ratio=0.5，原始 672×896）：
      scale = sqrt(0.5) ≈ 0.707
      new_H = round(672 × 0.707) = 475
      new_W = round(896 × 0.707) = 634
      → 475×634，保留宽高比 3:4，面积 ≈ 50% of original

    与直接设定固定分辨率的区别
    --------------------------
    - 保留宽高比，不引入形变
    - 与输入视频分辨率无关，用户只需指定压缩倍率
    - ViT patches ∝ spatial_ratio（面积），KV 等比减少

    Parameters
    ----------
    chunk : np.ndarray  [F, H, W, C]
    ctx   : PruneContext | None

    Returns
    -------
    np.ndarray  [F, new_H, new_W, C]，宽高比与输入相同
    """
    if ctx is None or not ctx.enabled or not ctx.spatial_enabled:
        return chunk
    if ctx.spatial_ratio is None or ctx.spatial_ratio >= 1.0:
        return chunk

    try:
        from PIL import Image
    except ImportError:
        logger.warning("[prune:spatial] Pillow not installed, spatial downscale skipped.")
        return chunk

    F, H, W, C = chunk.shape
    scale  = ctx.spatial_ratio ** 0.5            # sqrt(area_ratio) = linear scale
    new_H  = max(1, round(H * scale))
    new_W  = max(1, round(W * scale))

    if new_H == H and new_W == W:
        return chunk

    resized_frames = []
    for f_idx in range(F):
        img = Image.fromarray(chunk[f_idx].astype(np.uint8))
        img = img.resize((new_W, new_H), Image.BILINEAR)  # PIL: (W, H)
        resized_frames.append(np.array(img))

    result = np.stack(resized_frames, axis=0)   # [F, new_H, new_W, C]

    # 实际面积比（因 round() 会有微小偏差）
    actual_area_ratio = (new_H * new_W) / (H * W)
    # ViT patch 数估算（patch_size=14，取最近 14 的倍数）
    orig_patches = (H // 14) * (W // 14)
    new_patches  = (new_H // 14) * (new_W // 14)
    token_ratio  = new_patches / orig_patches if orig_patches > 0 else 1.0

    stat = {
        "chunk_idx":          chunk_idx,
        "original_H":         H, "original_W": W,
        "new_H":              new_H, "new_W": new_W,
        "spatial_ratio":      ctx.spatial_ratio,
        "scale_factor":       scale,
        "actual_area_ratio":  actual_area_ratio,
        "original_pixels":    H * W,
        "resized_pixels":     new_H * new_W,
        "orig_patches_est":   orig_patches,
        "new_patches_est":    new_patches,
    }
    ctx._spatial_stats.append(stat)

    if ctx.log_stats:
        logger.info(
            f"[prune:spatial] chunk {chunk_idx}: "
            f"{H}×{W} → {new_H}×{new_W} per frame  "
            f"(scale={scale:.3f}, area={actual_area_ratio:.0%} of original)  "
            f"ViT patches: ~{orig_patches} → ~{new_patches}  "
            f"(token ratio≈{token_ratio:.0%})  "
            f"aspect ratio preserved: {W/H:.3f} → {new_W/new_H:.3f}"
        )

    return result


# ---------------------------------------------------------------------------
# stub: install_spatial_hook（保留接口但禁用，输出明确警告）
# ---------------------------------------------------------------------------

def install_spatial_hook(model, ctx: Optional[PruneContext]):
    """
    ⚠️  Hook-based mid-forward spatial pruning is NOT compatible with
    LLaVA-OneVision's pre-built attention_mask.

    Reason:
      - multi_modal_projector outputs [F, 729, D_llm] (MLP only, no pooling yet)
      - Downstream get_2dPool expects shape[1]==729 for sqrt() reshape
      - attention_mask is pre-built by processor as [1, past + F×196],
        changing token count mid-forward causes shape mismatch

    Alternative: use spatial_downscale_chunk() BEFORE processor call.
    This function returns (None, None) and does nothing.
    """
    if ctx is not None and ctx.spatial_enabled and ctx.spatial_size is None:
        logger.warning(
            "[prune:spatial] install_spatial_hook() called but hook-based spatial pruning "
            "is disabled (incompatible with LLaVA-OV attention_mask pre-computation). "
            "Use spatial_downscale_chunk() before processor call instead, "
            "or set spatial_size= in PruneContext."
        )
    return None, None


def remove_spatial_hook(handle) -> None:
    """No-op stub (hook was never installed)."""
    pass


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------

def make_prune_context(
    temporal_keep_ratio: float = 0.6,
    spatial_ratio: Optional[float] = None,
) -> PruneContext:
    """
    快捷构建 PruneContext。

    Parameters
    ----------
    temporal_keep_ratio : float
        目标帧保留比例。0.6 = 每 chunk 保留约 60% 的帧。
        推荐范围：
          0.8 ~ 0.9  轻量，适合快节奏视频
          0.5 ~ 0.7  中等，适合大多数场景
          0.3 ~ 0.5  激进，适合几乎静止的监控类视频
    spatial_ratio : float | None
        像素面积保留比例。0.5 = 面积缩小到 50%，长宽 ×sqrt(0.5)≈0.707。
        推荐范围：
          0.7  轻量，细节损失极小
          0.5  中等，推荐默认值
          0.3  激进，远景/背景类任务
    """
    return PruneContext(
        temporal_enabled=True,
        temporal_keep_ratio=temporal_keep_ratio,
        spatial_enabled=(spatial_ratio is not None),
        spatial_ratio=spatial_ratio,
    )


# ---------------------------------------------------------------------------
# 估算节省量
# ---------------------------------------------------------------------------

def estimate_savings(
    num_frames: int,
    chunk_size: int = 16,
    temporal_keep_ratio: float = 0.6,
    spatial_ratio: Optional[float] = None,
    tokens_per_frame_full: int = 196,
) -> dict:
    """粗估两级剪枝后的 KV cache 节省量（不运行模型）。"""
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    orig_tokens = num_chunks * chunk_size * tokens_per_frame_full

    kept_frames = num_frames * temporal_keep_ratio

    # spatial: ViT patches ∝ area = spatial_ratio
    if spatial_ratio is not None:
        tokens_per_frame_after = int(tokens_per_frame_full * spatial_ratio)
    else:
        tokens_per_frame_after = tokens_per_frame_full

    pruned_tokens = kept_frames * tokens_per_frame_after
    reduction = 1.0 - pruned_tokens / orig_tokens if orig_tokens > 0 else 0.0

    return {
        "original_total_tokens":  int(orig_tokens),
        "estimated_pruned_tokens": int(pruned_tokens),
        "estimated_reduction":    reduction,
        "temporal_keep_ratio":    temporal_keep_ratio,
        "spatial_ratio":          spatial_ratio,
        "tokens_per_frame_before": tokens_per_frame_full,
        "tokens_per_frame_after": tokens_per_frame_after,
    }