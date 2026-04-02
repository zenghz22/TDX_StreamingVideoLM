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

原理（来自 ReKV ICLR'25 Eq.2）：
  encode 阶段已在 manifest.json 的每个 chunk record 里保存了
  "summary_vec"——该 chunk 所有 delta K 向量的跨层跨 token 均值。

  select 阶段：
  1. 对 question tokens 做一次轻量 forward（use_cache=False，只拿 K 向量）
  2. 同样计算跨层跨 token 均值，得到 query_vec
  3. 对每个 chunk 的 summary_vec 计算 cosine similarity
  4. 返回 top_k 个 chunk_index（按升序保留原始时序）

为什么用模型内部 K 向量而不是 CLIP embedding？
  K 向量与模型的注意力空间对齐——query 在 decode 时正是用这些 K 向量
  做 attention，因此 K-space cosine similarity 直接度量"这个 chunk 对
  这个 question 有多大注意力贡献"，比跨模型的 CLIP 空间更准确。
"""

import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 从 manifest 加载 summary_vec
# ---------------------------------------------------------------------------

def _load_summary_vecs(kv_cache_dir: str):
    """
    从 manifest.json 读取每个 chunk 的 summary_vec。

    Returns
    -------
    chunk_indices : List[int]     按 chunk_index 升序
    summary_vecs  : torch.Tensor  shape = [num_chunks, vec_dim]，fp32
    """
    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found: {kv_cache_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = sorted(manifest.get("chunks", []), key=lambda c: int(c["chunk_index"]))
    if not chunks:
        raise ValueError("No chunk records in manifest.")

    indices = []
    vecs = []
    for c in chunks:
        sv = c.get("summary_vec")
        if sv is None:
            raise ValueError(
                f"chunk {c['chunk_index']} has no summary_vec. "
                "Please re-encode with the updated kvcache_generate_td.py."
            )
        indices.append(int(c["chunk_index"]))
        vecs.append(torch.tensor(sv, dtype=torch.float32))

    return indices, torch.stack(vecs, dim=0)   # [N, D]


# ---------------------------------------------------------------------------
# 计算 question 的 query_vec（K 向量，带视频末尾位置上下文）
# ---------------------------------------------------------------------------

def _compute_query_vec(question: str, processor, model, kv_cache_dir: str) -> torch.Tensor:
    """
    以最后一个 chunk 的 KV 作为 past_key_values，对 question 做 forward，
    提取各层新增的 K 向量（即 question tokens 的 K），跨层跨 token 均值后返回。

    为什么必须带 past context：
      Qwen2 使用 RoPE，K 向量的值依赖 token 的绝对位置。
      encode 时 video chunk i 的 K 向量处于位置 pos≈i*chunk_merged_len。
      若 question standalone forward（past=None），其 K 向量从 pos=0 开始，
      与 video K 向量的位置差距极大，cosine similarity 由位置主导而非内容，
      导致无论问什么问题相似度都单调递减。

      以最后一个 chunk 的 KV 作为 past context forward question，
      question tokens 的位置 ID 从 video 末尾（pos≈N）开始，
      与各 video chunk 的位置差距变为有意义的时序距离，
      cosine(K_question, K_chunk_i) 才能反映问题与各段视频的语义相关性。
      这与 ReKV 论文的内部检索方式一致。
    """
    from kvcache_retrieve_td import _load_single_safetensors_kv, _concat_kv_segments, move_to_device

    # 加载最后一个 chunk 的 KV 作为 past context
    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    chunks_sorted = sorted(manifest["chunks"], key=lambda c: int(c["chunk_index"]))
    last_chunk = chunks_sorted[-1]
    last_shard_path = os.path.join(kv_cache_dir, last_chunk["file"])
    last_kv, _ = _load_single_safetensors_kv(last_shard_path, map_location="cpu")

    device = next(model.parameters()).device
    last_kv = move_to_device(last_kv, device)
    past_seq_len = int(last_kv[0][0].shape[-2])

    suffix = "\n问题：" + question
    inputs = processor(text=[suffix], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    # attention_mask 覆盖 [past | question]
    current_len = inputs["input_ids"].shape[1]
    inputs["attention_mask"] = torch.ones(
        (1, past_seq_len + current_len),
        dtype=inputs["attention_mask"].dtype,
        device=device,
    )

    with torch.no_grad():
        outputs = model(**inputs, past_key_values=last_kv, use_cache=True, return_dict=True)

    # past_key_values 的 seq 维度 = past_seq_len + question_merged_len
    # question 的 K 向量 = [past_seq_len:] 的部分
    full_kv = outputs.past_key_values
    k_means = []
    for layer_k, _ in full_kv:
        # layer_k: [batch, num_kv_heads, full_seq, head_dim]
        q_k = layer_k[0, :, past_seq_len:, :].float()  # [num_kv_heads, question_seq, head_dim]
        k_means.append(q_k.mean(dim=1))                 # [num_kv_heads, head_dim]

    query_vec = torch.stack(k_means, dim=0).mean(dim=0).flatten().cpu()
    del outputs, full_kv, last_kv
    return query_vec.float()


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
    # 1. 加载所有 chunk 的 summary_vec
    all_indices, summary_vecs = _load_summary_vecs(kv_cache_dir)   # [N, D]
    n_chunks = len(all_indices)

    if top_k >= n_chunks:
        print(f"[select_chunks] top_k={top_k} >= total chunks={n_chunks}, returning all.")
        return all_indices

    # 2. 计算 question 的 K 向量（以视频末尾为位置上下文）
    query_vec = _compute_query_vec(question, processor, model, kv_cache_dir)

    # 3. Q_question · K_chunk —— 真实 attention score 的代理
    # summary_vecs[i] 是 chunk i 的 K 向量均值（encode 时存入 manifest）
    # query_vec 是 question 的 Q 向量均值
    # 点积度量"question 会对这个 chunk 投入多少 attention"
    # 归一化后等价于 cosine，但保留幅度信息也可选择不归一化
    query_norm = F.normalize(query_vec.unsqueeze(0), dim=1)    # [1, D]
    chunk_norm = F.normalize(summary_vecs, dim=1)              # [N, D]
    sim = (query_norm * chunk_norm).sum(dim=1)                  # [N]

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