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
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


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

    chunks = sorted(manifest.get("chunks", []), key=lambda c: int(c["chunk_index"]))
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
    from kvcache_retrieve_td import _load_single_safetensors_kv

    chunks = _load_manifest_chunks(kv_cache_dir)
    chunk_indices = []
    layer_vecs = []
    for c in chunks:
        chunk_idx = int(c["chunk_index"])
        shard_path = os.path.join(kv_cache_dir, c["file"])
        shard_kv, _ = _load_single_safetensors_kv(shard_path, map_location="cpu")
        # shard_kv: tuple[L] of (k,v), k shape=[B,Hkv,T,D]
        per_layer = []
        for layer_k, _ in shard_kv:
            per_layer.append(layer_k[0].mean(dim=1).float())  # [Hkv, D]
        chunk_indices.append(chunk_idx)
        layer_vecs.append(torch.stack(per_layer, dim=0))      # [L, Hkv, D]

    return chunk_indices, torch.stack(layer_vecs, dim=0)      # [N, L, Hkv, D]


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


def _compute_query_vec(question: str, processor, model, kv_cache_dir: str) -> torch.Tensor:
    """
    以最后一个 chunk 的 KV 作为 past_key_values，对 question 做 forward，
    提取各层新增的 Q 向量（即 question tokens 的 Q），跨层跨 token 均值后返回。

    为什么必须带 past context：
      Qwen2 使用 RoPE，Q/K 向量都依赖 token 的绝对位置。
      encode 时 video chunk i 的 K 向量处于位置 pos≈i*chunk_merged_len。
      若 question standalone forward（past=None），其 Q 向量从 pos=0 开始，
      与 video K 向量的位置差距极大，cosine similarity 由位置主导而非内容，
      导致无论问什么问题相似度都单调递减。

      以最后一个 chunk 的 KV 作为 past context forward question，
      question tokens 的位置 ID 从 video 末尾（pos≈N）开始，
      与各 video chunk 的位置差距变为有意义的时序距离，
      cosine(Q_question, K_chunk_i) 才能反映问题与各段视频的语义相关性。
      这与 ReKV 论文的内部检索方式一致。
    """
    from kvcache_retrieve_td import move_to_device, _to_model_cache

    # 加载尾部窗口 KV 作为 past context（与 encode window 设置对齐）
    tail_kv, full_merged_seq_len = _load_tail_kv_as_past(kv_cache_dir, map_location="cpu")

    device = next(model.parameters()).device
    tail_kv = move_to_device(tail_kv, device)
    local_past_seq_len = int(tail_kv[0][0].shape[-2]) if tail_kv else 0
    model_past_key_values = _to_model_cache(tail_kv)

    suffix = "\n问题：" + question
    inputs = processor(text=[suffix], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    # attention_mask 覆盖 [past | question]
    current_len = inputs["input_ids"].shape[1]
    inputs["attention_mask"] = torch.ones(
        (1, local_past_seq_len + current_len),
        dtype=inputs["attention_mask"].dtype,
        device=device,
    )

    # ---------------- ReKV Internal-style: 用 question 的 Q 向量做检索 ----------------
    # 1) 先做一次 forward，拿到每层 hidden_states（layer 输入）
    # 2) 对每层 hidden_states 通过 self_attn.q_proj 得到 Q
    # 3) 用与解码一致的 position_ids 做 RoPE（关键）
    # 4) 将 Q heads 聚合到 KV heads 维度，再跨层平均
    # 绝对位置起点优先使用 manifest 里真实 full_merged_seq_len，避免 window 模式下位置回绕
    absolute_start = int(full_merged_seq_len) if full_merged_seq_len is not None else local_past_seq_len
    position_ids = torch.arange(absolute_start, absolute_start + current_len, device=device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=model_past_key_values,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            position_ids=position_ids,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Failed to get hidden_states for retrieval query vector computation.")

    q_means = []
    layers = model.language_model.model.layers
    for layer_idx, layer in enumerate(layers):
        layer_input = hidden_states[layer_idx]  # 第 layer_idx 层的输入
        attn = layer.self_attn

        # [B, Hq, T, D]
        q_states = attn.q_proj(layer_input).view(
            layer_input.shape[0], layer_input.shape[1], attn.num_heads, attn.head_dim
        ).transpose(1, 2)

        # 为了与 K-summary 对齐，按 KV 头数聚合 query 头
        # RoPE 需要和解码时位置一致：position_ids 从 past_seq_len 开始
        if hasattr(attn, "k_proj") and hasattr(attn, "rotary_emb"):
            k_dummy = attn.k_proj(layer_input).view(
                layer_input.shape[0], layer_input.shape[1], attn.num_key_value_heads, attn.head_dim
            ).transpose(1, 2)

            # Qwen2Attention.rotary_emb 在不同版本 transformers 参数签名略有差异：
            # 这里优先按 (x, position_ids) 调用；失败则退回不带 position_ids。
            try:
                cos, sin = attn.rotary_emb(k_dummy, position_ids)
            except TypeError:
                cos, sin = attn.rotary_emb(k_dummy)

            q_states, _ = apply_rotary_pos_emb(q_states, k_dummy, cos, sin)

        q_grouped = _group_query_to_kv_heads(q_states, attn.num_key_value_heads)  # [Hkv, D]
        q_means.append(q_grouped)

    query_layer_vecs = torch.stack(q_means, dim=0).cpu().float()  # [L, Hkv, D]
    del outputs, hidden_states, tail_kv
    return query_layer_vecs


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def select_chunks(
    kv_cache_dir: str,
    question: str,
    processor,
    model,
    top_k: int = 5,
    always_include_first: bool = True,
) -> List[int]:
    """
    Prompt-aware chunk 选择，返回与 question 最相关的 top_k 个 chunk_index。

    Parameters
    ----------
    kv_cache_dir : str
        encode 产生的 chunk 目录。
    question : str
        用户问题。
    processor / model :
        已加载的 LlavaOnevision processor 和 model。
    top_k : int
        返回相似度最高的 k 个 chunk（不含强制包含的第一块）。
    always_include_first : bool
        是否强制包含 chunk 0。
        chunk 0 含 ENCODE_PREFIX + 完整视觉语境，是整个视频理解的语义锚点，
        建议始终保留。默认 True。

    Returns
    -------
    List[int]  按 chunk_index 升序排列（保留视频时序）。
    """
    # 1. 加载每个 chunk 的逐层 K 向量均值
    all_indices, chunk_layer_key_vecs = _load_chunk_layer_key_vecs(kv_cache_dir)   # [N, L, Hkv, D]
    n_chunks = len(all_indices)

    if top_k >= n_chunks:
        print(f"[select_chunks] top_k={top_k} >= total chunks={n_chunks}, returning all.")
        return all_indices

    # 2. 计算 question 的逐层 Q 向量（以视频末尾位置作为绝对坐标）
    query_layer_vecs = _compute_query_vec(question, processor, model, kv_cache_dir)  # [L, Hkv, D]

    # 3. ReKV-style per-layer QK 相似度：
    #    逐层 cosine 后取平均，减少“跨层平均后被位置主导”的现象。
    q = query_layer_vecs.reshape(query_layer_vecs.shape[0], -1)                # [L, D']
    c = chunk_layer_key_vecs.reshape(chunk_layer_key_vecs.shape[0], q.shape[0], -1)  # [N, L, D']
    # 轻量去中心化：减去 chunk 集合均值，降低单调时间位置偏置
    c_center = c - c.mean(dim=0, keepdim=True)
    q_center = q - c.mean(dim=0)
    q_norm = F.normalize(q_center, dim=1)                     # [L, D']
    c_norm = F.normalize(c_center, dim=2)                     # [N, L, D']
    sim = (c_norm * q_norm.unsqueeze(0)).sum(dim=2).mean(dim=1)  # [N]

    print("[select_chunks] Cosine similarities:")
    for idx, s in zip(all_indices, sim.tolist()):
        print(f"  chunk {idx:3d}: {s:.4f}")

    # 4. 选 top_k（从 summary_vecs 中排序）
    topk_local = torch.topk(sim, k=min(top_k, n_chunks), largest=True).indices.tolist()
    selected = set(all_indices[i] for i in topk_local)

    # 5. 强制包含 chunk 0
    if always_include_first and 0 in all_indices:
        selected.add(0)

    result = sorted(selected)
    print(f"[select_chunks] Selected chunks: {result} (top_k={top_k}, always_first={always_include_first})")
    return result
