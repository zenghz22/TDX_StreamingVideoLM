"""
kvcache_select_td.py — Prompt-aware KV Cache chunk 选择器

位置：插在 encode 与 decode 之间，与两者完全解耦。

输入：
  - kv_cache_dir   : encode 产生的 chunk 目录（含 manifest.json）
  - question       : 用户问题字符串
  - processor      : LlavaOnevisionProcessor
  - model          : 已加载的 LlavaOnevision 模型
  - top_k          : 返回相似度最高的 k 个 chunk

输出：
  - List[int]      : 选中的 chunk_index 列表（升序排列）

原理（参考 ReKV ICLR'25 Internal Retrieval）：
  select 阶段直接从每个 chunk shard 提取逐层 K 向量均值，并计算问题逐层 Q 向量：
  1. 从 chunk_i.safetensors 读取每层 K，做 token 均值得到 K_i(layer)
  2. 对 question 做一次轻量 forward（use_cache=False，取每层 Q）
  3. 对 Q 按 KV 头维度聚合，并进行 RoPE 位置对齐
  4. 逐层 cosine(Q_layer, K_layer) 后取层均值，返回 top_k 个 chunk

为什么用模型内部 Q/K 向量而不是 CLIP embedding？
  Q/K 向量与模型的注意力空间对齐——query 在 decode 时正是用这些 K 向量
  做 attention，因此 QK-space cosine similarity 直接度量"这个 chunk 对
  这个 question 有多大注意力贡献"，比跨模型的 CLIP 空间更准确。
"""

import json
import os
from typing import List

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from kvpack_mmap_td import KVPackReader, has_kvpack


# ---------------------------------------------------------------------------
# 从 manifest 加载 chunk 元数据
# ---------------------------------------------------------------------------

def _load_manifest_chunks(kv_cache_dir: str):
    """
    从 manifest.json 读取 chunk 记录。

    Returns
    -------
    chunks : List[dict] 按 chunk_index 升序
    """
    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found: {kv_cache_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = sorted(manifest.get("chunks", []), key=lambda c: int(c.get("chunk_index", c.get("frame_index", -1))))
    if not chunks:
        raise ValueError("No chunk records in manifest.")

    return chunks


def _load_chunk_layer_key_vecs(kv_cache_dir: str):
    """
    ReKV internal 风格：直接从每个 chunk shard 提取逐层 K 向量均值。

    Returns
    -------
    chunk_indices : List[int]
    layer_key_vecs: torch.Tensor  [N, L, Hkv, D]
    """
    # 优先读取 encode 阶段写好的轻量检索索引，避免逐 shard 扫描
    index_path = os.path.join(kv_cache_dir, "retrieval_index.safetensors")
    if os.path.exists(index_path):
        with safe_open(index_path, framework="pt", device="cpu") as f:
            if "chunk_indices" in f.keys() and "layer_key_vecs" in f.keys():
                chunk_indices = [int(v) for v in f.get_tensor("chunk_indices").tolist()]
                layer_key_vecs = f.get_tensor("layer_key_vecs").float()
                return chunk_indices, layer_key_vecs

    if not has_kvpack(kv_cache_dir):
        raise FileNotFoundError(
            f"Only kvpack_mmap_v1 is supported now; kvpack_index.json not found in {kv_cache_dir}"
        )
    reader = KVPackReader(kv_cache_dir)
    try:
        frame_ids = sorted(reader.frames.keys())
        n_layers = int(reader.common_metadata.get("num_layers", 0))
        if n_layers <= 0:
            n_layers = max((l for l, _ in reader.by_layer_frame.keys()), default=-1) + 1
        layer_vecs = []
        for fi in frame_ids:
            per_layer = []
            for li in range(n_layers):
                k, _, _ = reader.read_layer_frame(li, fi, map_location="cpu")
                per_layer.append(k[0].mean(dim=1).float())  # [Hkv, D]
            layer_vecs.append(torch.stack(per_layer, dim=0))
        return frame_ids, torch.stack(layer_vecs, dim=0)
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# 计算 question 的 query_vec（Q 向量，带视频末尾位置上下文）
# ---------------------------------------------------------------------------

def _group_query_to_kv_heads(query_states: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """
    将 query heads 聚合到 kv heads 维度，便于与 chunk K summary 对齐。

    query_states: [batch, num_heads, seq, head_dim]
    return      : [num_kv_heads, head_dim] （对 batch/seq 求均值）
    """
    # [batch, num_heads, seq, head_dim]
    _, num_heads, _, head_dim = query_states.shape
    if num_heads == num_kv_heads:
        grouped = query_states
    else:
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})."
            )
        group_size = num_heads // num_kv_heads
        # [batch, num_kv_heads, group_size, seq, head_dim] -> mean(group_size)
        grouped = query_states.view(
            query_states.shape[0], num_kv_heads, group_size, query_states.shape[2], head_dim
        ).mean(dim=2)

    return grouped.mean(dim=0).mean(dim=1).float()  # [num_kv_heads, head_dim]


def _load_tail_kv_as_past(kv_cache_dir: str, map_location="cpu"):
    """
    按 manifest 中 window_size 加载视频尾部若干 chunk 作为 retrieval 的 past context。
    若无 window_size，则退化为只加载最后一个 chunk（兼容旧数据）。
    """
    from kvcache_retrieve_td import _load_single_safetensors_kv, _concat_kv_segments

    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = sorted(manifest.get("chunks", []), key=lambda c: int(c["chunk_index"]))
    if not chunks:
        raise ValueError(f"No chunks found in manifest: {manifest_path}")

    common_metadata = manifest.get("common_metadata", {}) or {}
    window_size = common_metadata.get("window_size")
    if window_size is None:
        selected = [chunks[-1]]
    else:
        selected = chunks[-int(window_size):]

    kv_segments = []
    for c in selected:
        shard_path = os.path.join(kv_cache_dir, c["file"])
        shard_kv, _ = _load_single_safetensors_kv(shard_path, map_location=map_location)
        kv_segments.append(shard_kv)

    past_kv = _concat_kv_segments(kv_segments)
    full_merged_seq_len = common_metadata.get("full_merged_seq_len")
    return past_kv, full_merged_seq_len


def _compute_query_vec(question: str, processor, model) -> torch.Tensor:
    """
    用 hook 捕获 question tokens 的 pre-RoPE Q 向量。

    为什么必须 pre-RoPE：
      retrieval_index 里的 K 向量是 pre-RoPE（encode 阶段 hook 捕获）。
      Q 也必须 pre-RoPE，两者才在同一纯内容语义空间。
      post-RoPE Q 与 post-RoPE K（不同绝对位置）做 cosine sim，
      结果由绝对位置差主导，与问题内容无关 → 单调性伪信号。
    """
    device = next(model.parameters()).device

    suffix = "\n问题：" + question
    inputs = processor(text=[suffix], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    captured_q = []   # [Hkv, D] per layer
    handles = []

    def _make_q_hook():
        def _hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None:
                with torch.no_grad():
                    q = module.q_proj(hidden.float())
                    B, T, _ = q.shape
                    Hq  = module.num_heads
                    Hkv = module.num_key_value_heads
                    D   = module.head_dim
                    q = q.view(B, T, Hq, D)
                    # GQA fold: Hq → Hkv（与 encode 侧的 Hkv 对齐）
                    if Hq != Hkv:
                        g = Hq // Hkv
                        q = q.view(B, T, Hkv, g, D).mean(dim=3)  # [B, T, Hkv, D]
                    captured_q.append(q[0].mean(dim=0).detach().cpu())  # [Hkv, D]
            return args, kwargs
        return _hook

    for layer in model.language_model.model.layers:
        h = layer.self_attn.register_forward_pre_hook(_make_q_hook(), with_kwargs=True)
        handles.append(h)

    try:
        with torch.no_grad():
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    return torch.stack(captured_q, dim=0).float()   # [L, Hkv, D]


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def select_chunks(
    kv_cache_dir: str,
    question: str,
    processor,
    model,
    top_k: int = 8,
) -> List[int]:
    """
    Prompt-aware chunk 选择，返回与 question 最相关的 top_k 个 chunk_index。

    - 不再强制包含 chunk 0：prefix_cache.safetensors 已承载 encode_prefix
      语义锚点，chunk 0 现在只是普通视频 chunk。
    - 不再要求连续：decode 侧的 position_ids 修正确保 Q 始终在
      full_merged_seq_len 处，K 向量保留 encode 时的绝对位置，
      cos(pos_q - pos_k) 对任意非连续 chunk 组合都正确。

    返回按 chunk_index 升序排列的列表（保留时序，便于 attention mask 构建）。
    """
    all_indices, chunk_layer_key_vecs = _load_chunk_layer_key_vecs(kv_cache_dir)
    n_chunks = len(all_indices)

    if n_chunks == 0:
        return []
    if top_k >= n_chunks:
        print(f"[select_chunks] top_k={top_k} >= total chunks={n_chunks}, returning all.")
        return all_indices

    # 计算 question 的逐层 pre-RoPE Q 向量
    query_layer_vecs = _compute_query_vec(question, processor, model)   # [L, Hkv, D]

    # per-layer cosine similarity（pre-RoPE Q · pre-RoPE K，纯内容相关性）
    q = query_layer_vecs.reshape(query_layer_vecs.shape[0], -1)               # [L, D']
    c = chunk_layer_key_vecs.reshape(n_chunks, q.shape[0], -1)                # [N, L, D']
    q_norm = F.normalize(q, dim=1)
    c_norm = F.normalize(c, dim=2)
    sim = (c_norm * q_norm.unsqueeze(0)).sum(dim=2).mean(dim=1)               # [N]

    # 直接取 top-k（离散，无连续约束）
    topk_local = torch.topk(sim, k=top_k, largest=True).indices.tolist()
    result = sorted(all_indices[i] for i in topk_local)

    print(f"[select_chunks] mode=global  top_k={top_k}  selected={result}")
    return result

# ---------------------------------------------------------------------------
# Per-layer 独立检索（ReKV Internal Retrieval 的完整实现）
# ---------------------------------------------------------------------------

def select_chunks_per_layer(
    kv_cache_dir: str,
    question: str,
    processor,
    model,
    top_k: int = 8,
) -> List[List[int]]:
    """
    ReKV ICLR'25 Section 3 Internal Retrieval 的直接实现：
    每个 attention layer 用自己的 Q 和 K 向量独立选出 top_k 个最相关 chunk，
    不同层的选择结果可以（且通常会）不同。

    原论文原文：
        "internal retrieval [...] operates independently within each self-attention
         layer, allowing different layers to retrieve different video blocks.
         This allows for a broader capture of video context."
        "The average of the key vectors of a frame is computed as its representative
         frame vector [...] concatenated across heads into a single vector."

    与 select_chunks() 的区别
    -------------------------
    select_chunks()       : mean over layers → 单一全局 chunk 选择（所有层共享）
    select_chunks_per_layer: per-layer 独立 → List[L] 每层各自的 top_k chunk 列表

    参数
    ----
    top_k : 每层各自选 top_k 个 chunk（不同层可能不同，但数目相同）

    返回
    ----
    per_layer_chunk_ids : List[List[int]]
        长度 = 模型层数（如 28）
        per_layer_chunk_ids[L] = 第 L 层选出的 chunk_index 列表（升序）

    使用方式
    --------
    per_layer_ids = select_chunks_per_layer(kv_dir, q, proc, model, top_k=8)
    answer = decode_kvcache(..., per_layer_chunk_indices=per_layer_ids)
    """
    all_indices, chunk_layer_key_vecs = _load_chunk_layer_key_vecs(kv_cache_dir)
    # chunk_layer_key_vecs: [N, L, Hkv, D]
    n_chunks = len(all_indices)

    if n_chunks == 0:
        return []
    if top_k >= n_chunks:
        print(f"[select_per_layer] top_k={top_k} >= total chunks={n_chunks}, returning all for every layer.")
        return [list(all_indices) for _ in range(chunk_layer_key_vecs.shape[1])]

    # Q 向量：[L, Hkv, D]（pre-RoPE，与 retrieval_index 的 K 向量对齐）
    query_layer_vecs = _compute_query_vec(question, processor, model)  # [L, Hkv, D]
    n_layers = query_layer_vecs.shape[0]

    # ── per-layer cosine similarity ──────────────────────────────────────
    # q: [L, D']   where D' = Hkv × D（flatten heads）
    # c: [N, L, D']
    D_prime = query_layer_vecs.shape[1] * query_layer_vecs.shape[2]
    q = query_layer_vecs.reshape(n_layers, D_prime)               # [L, D']
    c = chunk_layer_key_vecs.reshape(n_chunks, n_layers, D_prime) # [N, L, D']

    q_norm = F.normalize(q, dim=1)                                # [L, D']
    c_norm = F.normalize(c, dim=2)                                # [N, L, D']

    # sim_per_layer[n, l] = cosine(chunk_n, layer_l)
    # c_norm: [N, L, D']  q_norm: [L, D'] → broadcast → element-wise → sum over D' → [N, L]
    sim_per_layer = (c_norm * q_norm.unsqueeze(0)).sum(dim=2)     # [N, L]

    # ── 逐层独立 top-k 选择 ───────────────────────────────────────────────
    per_layer_chunk_ids = []
    for L_idx in range(n_layers):
        layer_sim = sim_per_layer[:, L_idx]  # [N]
        k = min(top_k, n_chunks)
        topk_local = torch.topk(layer_sim, k=k, largest=True).indices.tolist()
        layer_result = sorted(all_indices[i] for i in topk_local)
        per_layer_chunk_ids.append(layer_result)

    # ── 打印：每层各自选中了哪些 chunk（验证 per-layer 检索确实各层不同）
    all_selected = [set(ids) for ids in per_layer_chunk_ids]
    union_count = len(set().union(*all_selected))
    inter_count = len(set.intersection(*all_selected)) if all_selected else 0
    print(
        f"[select_per_layer] mode=per-layer  top_k={top_k}  "
        f"union_chunks={union_count} (across all layers)  "
        f"intersection_chunks={inter_count} (in every layer)"
    )
    # 逐层显示选中的 chunk 列表，每层一行
    for L_idx in range(n_layers):
        print(f"  L{L_idx:02d}: {per_layer_chunk_ids[L_idx]}")

    return per_layer_chunk_ids
