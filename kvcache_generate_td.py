"""LLaVA-OneVision 视频预处理与 KV cache 生成/保存。"""

import time
import json
import os
from typing import Optional

import gc
import torch
import numpy as np
from decord import VideoReader, cpu
from safetensors.torch import save_file
from transformers import LlavaOnevisionForConditionalGeneration as LlavaOV
from transformers import LlavaOnevisionProcessor
from kvpack_mmap_td import KVPackWriter, KVPackReader

# 统一前缀：编码和解码都基于同一语境，避免 KV 与后续文本提示错位。
VIDEO_PLACEHOLDER = "<video>"


def _build_encode_text(encode_prefix):
    """确保文本中包含视频占位符，否则会出现 tokens/features 不匹配。"""
    text = (encode_prefix or "").strip()
    if VIDEO_PLACEHOLDER in text:
        return text
    if text:
        return VIDEO_PLACEHOLDER + "\n" + text
    return VIDEO_PLACEHOLDER


def _get_cache_seq_len(kv_cache):
    try:
        return int(kv_cache[0][0].shape[-2])
    except Exception:
        return 0


def load_model(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", load_weights=False):
    """加载 processor；可选加载模型权重。"""
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    model = None
    if load_weights:
        model = LlavaOV.from_pretrained(model_name, trust_remote_code=True)
    return processor, model


def load_video(video_path, sample_fps=10):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = max(1, round(vr.get_avg_fps()))
    step = max(1, int(fps / sample_fps))
    frame_idx = list(range(0, len(vr), step))
    print(
        f"Original FPS: {fps}, Sampled FPS: {fps / step}, "
        f"Total frames: {len(vr)}, Sampled frames: {len(frame_idx)}"
    )
    return vr.get_batch(frame_idx).asnumpy()


def detach_kv_to_cpu(kv_cache):
    """将 KV cache 递归搬到 CPU，便于存盘。"""
    if isinstance(kv_cache, torch.Tensor):
        return kv_cache.detach().to("cpu").contiguous()
    if isinstance(kv_cache, tuple):
        return tuple(detach_kv_to_cpu(v) for v in kv_cache)
    if isinstance(kv_cache, list):
        return [detach_kv_to_cpu(v) for v in kv_cache]
    return kv_cache


def save_kv_cache(kv_cache, kv_cache_path, model=None, extra_metadata=None):
    """保存 KV cache 与元信息（safetensors）。"""
    kv_cache_cpu = detach_kv_to_cpu(kv_cache)

    if not isinstance(kv_cache_cpu, (tuple, list)):
        raise TypeError("KV cache format is invalid: expected tuple/list of per-layer KV pairs.")

    metadata = {
        "saved_at": int(time.time()),
        "num_layers": len(kv_cache_cpu) if hasattr(kv_cache_cpu, "__len__") else None,
    }

    if model is not None:
        metadata["model_name_or_path"] = getattr(model.config, "_name_or_path", None)
        metadata["num_hidden_layers"] = getattr(model.config, "num_hidden_layers", None)

    try:
        first_k = kv_cache_cpu[0][0]
        metadata["layer0_key_shape"] = tuple(first_k.shape)
        metadata["past_seq_len"] = int(first_k.shape[-2])
        metadata["dtype"] = str(first_k.dtype)
    except Exception:
        pass

    if extra_metadata:
        metadata.update(extra_metadata)

    tensors = {}
    for layer_idx, layer_kv in enumerate(kv_cache_cpu):
        if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) != 2:
            raise TypeError(f"Layer {layer_idx} KV format is invalid: expected (key, value).")

        layer_k, layer_v = layer_kv
        if not isinstance(layer_k, torch.Tensor) or not isinstance(layer_v, torch.Tensor):
            raise TypeError(f"Layer {layer_idx} KV should be torch.Tensor.")

        tensors[f"layer_{layer_idx}.k"] = layer_k.contiguous()
        tensors[f"layer_{layer_idx}.v"] = layer_v.contiguous()

    # safetensors metadata 仅支持 str->str
    metadata_str = {k: json.dumps(v, ensure_ascii=False) for k, v in metadata.items()}
    save_file(tensors, kv_cache_path, metadata=metadata_str)
    print(f"KV cache saved to {kv_cache_path}.")
    print(f"KV metadata: {metadata}")
    return metadata


def _write_chunk_manifest(kv_cache_dir, chunks, common_metadata):
    manifest = {
        "format": "kvcache_safetensors_chunks_v1",
        "num_chunks": len(chunks),
        "chunks": chunks,
        "common_metadata": common_metadata or {},
    }
    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"KV chunk manifest saved to {manifest_path}")


def _write_retrieval_index(kv_cache_dir, chunk_indices, layer_key_vecs, summary_vecs, common_metadata):
    """
    将 select 所需的轻量向量索引写入独立文件（方案 A + C）：
      - chunk_indices : [N]
      - layer_key_vecs: [N, L, Hkv, D]
      - summary_vecs  : [N, Hkv*D]
    """
    if not chunk_indices:
        return

    index_path = os.path.join(kv_cache_dir, "retrieval_index.safetensors")
    tensors = {
        "chunk_indices": torch.tensor(chunk_indices, dtype=torch.int64).contiguous(),
        "layer_key_vecs": torch.stack(layer_key_vecs, dim=0).float().contiguous(),
        "summary_vecs": torch.stack(summary_vecs, dim=0).float().contiguous(),
    }
    metadata = {
        "format": "retrieval_index_v1",
        "num_chunks": len(chunk_indices),
        "common_metadata": common_metadata or {},
    }
    metadata_str = {k: json.dumps(v, ensure_ascii=False) for k, v in metadata.items()}
    save_file(tensors, index_path, metadata=metadata_str)
    print(f"Retrieval index saved to {index_path}")


# ---------------------------------------------------------------------------
# 前缀编码与保存（prefix_cache.safetensors）
# ---------------------------------------------------------------------------

def _encode_and_save_prefix(encode_prefix, processor, model, kv_cache_dir, device, crypto_ctx=None):
    """
    单独 forward encode_prefix 文字（无视频），将 prefix KV 存为
    prefix_cache.safetensors。

    crypto_ctx : CryptoContext | None
        仅为接口兼容保留。prefix cache 始终保持明文，不参与加密。
    """
    if not encode_prefix or not encode_prefix.strip():
        return None, 0

    text = encode_prefix.strip()
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, past_key_values=None, use_cache=True, return_dict=True)

    prefix_kv = tuple(
        (layer_kv[0].detach().cpu().contiguous(),
         layer_kv[1].detach().cpu().contiguous())
        for layer_kv in outputs.past_key_values
    )
    prefix_seq_len = int(prefix_kv[0][0].shape[-2])
    del outputs
    gc.collect()

    print(f"[encode_prefix] Encoded {prefix_seq_len} prefix text tokens.")

    if kv_cache_dir is not None:
        prefix_path = os.path.join(kv_cache_dir, "prefix_cache.safetensors")
        save_kv_cache(prefix_kv, prefix_path, model=model,
                      extra_metadata={"is_prefix": True,
                                      "prefix_seq_len": prefix_seq_len,
                                      "prefix_text": text})
        print(f"[encode_prefix] Saved to {prefix_path}")

        # 设计约束：prefix_cache 体积很小，且属于非核心视频 KV，
        # 为便于调试与稳定加载，始终保持明文，不执行加密。

    return prefix_kv, prefix_seq_len


# ---------------------------------------------------------------------------
# encode_video — 带自适应 PV 差分存储
# ---------------------------------------------------------------------------

def encode_video(
    video,
    processor,
    model=None,
    chunk_size=64,
    encode_prefix=None,
    stage_mark=None,
    kv_cache_dir=None,
    prune_ctx=None,               # 可选：video_token_prune_td.PruneContext
    crypto_ctx=None,              # 可选：kvcache_crypto_td.CryptoContext
    max_in_memory: int = 64,
    window_size: Optional[int] = None,
    max_consecutive_p: int = 0,         # 每层最大连续 P 帧数（0=禁用差分）
    delta_threshold: float = 1e-3,      # V 差分量化阈值
    delta_ratio_threshold: float = 0.75 # 自适应回退 I 帧的稀疏率上限
):
    """
    将视频分块编码成 KV cache。

    Parameters
    ----------
    prune_ctx : PruneContext | None
        视觉 token 剪枝上下文（video_token_prune_td.PruneContext）。
        None 时不剪枝。支持两级：帧级时序去冗余 + ViT 输出后空间剪枝。
    crypto_ctx : CryptoContext | None
        KV cache 加密上下文（kvcache_crypto_td.CryptoContext）。
        None 或 enabled=False 时不加密。每个 chunk shard 保存后立即加密，
        删除明文，decode 侧透明解密。
    max_consecutive_p : int
        每层允许的最大连续 PV 帧数，到达后自动插入 I 帧（0 禁用差分压缩）。
    delta_threshold : float
        PV 帧 V 差分量化阈值，绝对值小于此值置零。
    delta_ratio_threshold : float
        V 差分非零比例阈值，若超过此值则放弃使用 PV 帧，直接存 I 帧。
    """
    num_frames = video.shape[0]
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    chunk_records = []
    common_metadata = {}
    retrieval_chunk_indices = []
    retrieval_layer_key_vecs = []
    retrieval_summary_vecs = []

    pack_writer = None
    if kv_cache_dir is not None:
        os.makedirs(kv_cache_dir, exist_ok=True)
        pack_writer = KVPackWriter(kv_cache_dir)

    # 帧级缓存：frame_idx -> tuple[L](k,v)
    frame_cache = {}
    frame_order = []
    encrypt_payload_fn = None
    if crypto_ctx is not None and getattr(crypto_ctx, "enabled", False):
        from kvcache_crypto_td import layer_frame_block_id, encrypt_bytes_to_blob
        def _encrypt_payload(payload_bytes: bytes, header: dict) -> bytes:
            n_layers = int(common_metadata.get("num_layers", getattr(model.config, "num_hidden_layers", 0)))
            block_id = layer_frame_block_id(
                frame_index=int(header["frame_index"]),
                layer_index=int(header["layer_index"]),
                num_layers=n_layers,
            )
            aad = {
                "frame_index": int(header["frame_index"]),
                "layer_index": int(header["layer_index"]),
                "seq_start": int(header["seq_start"]),
                "seq_end": int(header["seq_end"]),
                "dtype": str(header["dtype"]),
            }
            return encrypt_bytes_to_blob(
                payload_bytes,
                crypto_ctx.master_key,
                chunk_index=block_id,
                aad=aad,
            )
        encrypt_payload_fn = _encrypt_payload

    # ---- 前缀单独编码 ----
    prefix_kv = None
    prefix_seq_len = 0
    if model is not None:
        device_tmp = next(model.parameters()).device
        prefix_kv, prefix_seq_len = _encode_and_save_prefix(
            encode_prefix, processor, model, kv_cache_dir, device_tmp,
            crypto_ctx=crypto_ctx,
        )
    full_merged_seq_len = int(prefix_seq_len)

    # ---- 剪枝说明 ────────────────────────────────────────────────────
    # Level 0 temporal: 在 processor 之前减少帧数（有效）
    # Level 1 spatial:  在 processor 之前降低分辨率（有效）
    # ⚠️  mid-forward hook spatial 已移除（不兼容 LLaVA-OV attention_mask 预构建）
    # ────────────────────────────────────────────────────────────────

    # 各层连续 P 帧计数，将在第一次获得 delta_kv 后初始化
    layer_p_count = None

    try:
        for i in range(num_chunks):
            chunk = video[i * chunk_size : min((i + 1) * chunk_size, num_frames)]
            if stage_mark is not None:
                stage_mark(f"chunk_{i}_start")

            frames_before_prune = len(chunk)

            # ---- 剪枝 ----
            if prune_ctx is not None:
                from video_prune import temporal_filter_chunk, spatial_downscale_chunk
                chunk = temporal_filter_chunk(chunk, prune_ctx, chunk_idx=i)
                if len(chunk) == 0:
                    print(f"[prune] chunk {i}: all frames filtered, skipping.")
                    continue
                chunk = spatial_downscale_chunk(chunk, prune_ctx, chunk_idx=i)

            pixel_values = processor.video_processor(chunk, return_tensors="pt").pixel_values_videos.to("cpu")
            print(f"[chunk {i}] pixel_values_videos shape: {tuple(pixel_values.shape)}")

            if model is None:
                continue

            encode_text = VIDEO_PLACEHOLDER

            model_inputs = processor(
                text=[encode_text],
                videos=[chunk],
                return_tensors="pt",
            )

            device = next(model.parameters()).device
            model_inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in model_inputs.items()
            }

            # ------------------------------------------------------------
            # 构建 past_kv
            from kvcache_retrieve_td import _concat_kv_segments as _cat_kv
            def _prepend_prefix(base_kv, base_seq_len):
                if prefix_kv is None:
                    return (base_kv, base_seq_len)
                if base_kv is None:
                    pkv = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
                    return (pkv, prefix_seq_len)
                pkv = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
                merged = _cat_kv([pkv, base_kv])
                return (merged, int(merged[0][0].shape[-2]))

            history_frames = list(range(i))
            if window_size is not None and window_size > 0:
                history_frames = history_frames[-int(window_size):]
            base_kv = None
            if history_frames:
                segments = []
                miss_frames = [fi for fi in history_frames if fi not in frame_cache]
                if miss_frames and kv_cache_dir is not None:
                    reader = KVPackReader(kv_cache_dir)
                    try:
                        n_layers = int(common_metadata.get("num_layers", 0))
                        if n_layers <= 0:
                            n_layers = getattr(model.config, "num_hidden_layers", 0)
                        for fi in miss_frames:
                            per_layer = []
                            for li in range(n_layers):
                                k, v, _ = reader.read_layer_frame(li, fi, map_location="cpu")
                                per_layer.append((k, v))
                            frame_cache[fi] = tuple(per_layer)
                    finally:
                        reader.close()
                for fi in history_frames:
                    segments.append(frame_cache[fi])
                base_kv = _cat_kv(segments) if segments else None

            base_seq = _get_cache_seq_len(base_kv) if base_kv is not None else 0
            past_kv, past_kv_seq_len = _prepend_prefix(base_kv, base_seq)
            current_text_len = model_inputs["input_ids"].shape[1]
            if past_kv_seq_len > 0:
                model_inputs["attention_mask"] = torch.ones(
                    (model_inputs["input_ids"].shape[0], past_kv_seq_len + current_text_len),
                    dtype=model_inputs["attention_mask"].dtype,
                    device=device,
                )

            # ------------------------------------------------------------
            # 捕获 pre-RoPE K（用于检索索引）
            _pre_rope_k_layers = []
            _hook_handles_k = []

            def _make_k_capture():
                def _hook(module, args, kwargs):
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    if hidden is not None:
                        with torch.no_grad():
                            k = module.k_proj(hidden.float())
                            B, T, _ = k.shape
                            Hkv = module.num_key_value_heads
                            D   = module.head_dim
                            k = k.view(B, T, Hkv, D)
                            _pre_rope_k_layers.append(k[0].mean(dim=0).detach().cpu())
                    return args, kwargs
                return _hook

            for _layer in model.language_model.model.layers:
                _h = _layer.self_attn.register_forward_pre_hook(
                    _make_k_capture(), with_kwargs=True
                )
                _hook_handles_k.append(_h)

            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    past_key_values=past_kv if past_kv else None,
                    use_cache=True,
                    return_dict=True,
                )

            for _h in _hook_handles_k:
                _h.remove()
            _hook_handles_k.clear()

            full_kv_after = outputs.past_key_values

            # ------------------------------------------------------------
            # 提取本 chunk 的 delta KV
            if full_kv_after:
                full_kv_seq_len = int(full_kv_after[0][0].shape[-2])
                delta_kv = tuple(
                    (
                        layer_kv[0][:, :, past_kv_seq_len:, :].detach().cpu().contiguous(),
                        layer_kv[1][:, :, past_kv_seq_len:, :].detach().cpu().contiguous(),
                    )
                    for layer_kv in full_kv_after
                )
                delta_seq_len = int(delta_kv[0][0].shape[-2])

                # chunk summary
                if not _pre_rope_k_layers:
                    raise RuntimeError(f"[chunk {i}] failed to capture pre-RoPE K vectors.")
                chunk_layer_key_vec = torch.stack(_pre_rope_k_layers, dim=0).float()
                _pre_rope_k_layers.clear()
                chunk_summary_vec = chunk_layer_key_vec.mean(dim=0).flatten().float()

                full_merged_seq_len += delta_seq_len
            else:
                delta_kv = None
                full_kv_seq_len = 0
                delta_seq_len = 0
                chunk_summary_vec = None

            # ------------------------------------------------------------
            # 保存 delta 到磁盘 & 更新状态
            if kv_cache_dir is not None and delta_kv is not None:
                if not common_metadata:
                    common_metadata = {
                        "encode_prefix": encode_prefix,
                        "chunk_size": chunk_size,
                        "num_frames": int(num_frames),
                        "num_layers": len(delta_kv),
                        "model_name_or_path": getattr(model.config, "_name_or_path", None),
                        "window_size": window_size,
                        "storage_format": "kvpack_mmap_v1",
                        "max_consecutive_p": int(max_consecutive_p),
                        "delta_threshold": float(delta_threshold),
                        "delta_ratio_threshold": float(delta_ratio_threshold),
                    }

                # 初始化各层 P 计数
                if layer_p_count is None:
                    layer_p_count = [0] * len(delta_kv)

                total_payload_bytes = 0
                total_orig_bytes = 0
                p_layer_count_this_frame = 0
                i_layers = []                         # <-- 新增

                for layer_idx, (layer_k, layer_v) in enumerate(delta_kv):
                    use_p = False
                    ref_v = None
                    ref_frame = None

                    if max_consecutive_p > 0 and i > 0 and layer_p_count[layer_idx] < max_consecutive_p:
                        prev_kv = frame_cache.get(i - 1)
                        if prev_kv is not None:
                            ref_v = prev_kv[layer_idx][1]
                            delta_v = (layer_v.float() - ref_v.float()).numpy()
                            nnz_v = int(np.sum(np.abs(delta_v.ravel()) > delta_threshold))
                            ratio_v = nnz_v / delta_v.size if delta_v.size > 0 else 1.0
                            if ratio_v <= delta_ratio_threshold:
                                use_p = True
                                ref_frame = i - 1

                    if use_p:
                        rec = pack_writer.append_p_block(      
                            frame_index=i,
                            layer_index=layer_idx,
                            seq_start=int(past_kv_seq_len),
                            seq_end=int(full_kv_seq_len),
                            key_tensor=layer_k,
                            value_tensor=layer_v,
                            ref_value_tensor=ref_v,
                            ref_frame_index=ref_frame,
                            delta_threshold=delta_threshold,
                            encrypt_fn=encrypt_payload_fn,
                        )
                        layer_p_count[layer_idx] += 1
                        p_layer_count_this_frame += 1
                    else:
                        rec = pack_writer.append_block(        
                            frame_index=i,
                            layer_index=layer_idx,
                            seq_start=int(past_kv_seq_len),
                            seq_end=int(full_kv_seq_len),
                            key_tensor=layer_k,
                            value_tensor=layer_v,
                            encrypt_fn=encrypt_payload_fn,
                        )
                        layer_p_count[layer_idx] = 0
                        i_layers.append(layer_idx)

                    total_payload_bytes += int(rec["payload_len"])
                    total_orig_bytes += layer_k.numel() * layer_k.element_size() * 2

                    rec["chunk_index"] = int(i)
                    rec["block_index"] = len(chunk_records)
                    rec["delta_seq_len"] = int(delta_seq_len)
                    chunk_records.append(rec)

                ratio = total_payload_bytes / total_orig_bytes if total_orig_bytes > 0 else 1.0
                total_layers = len(delta_kv)
                print(
                    f"[delta:B] frame={i}  "
                    f"P-layers={p_layer_count_this_frame}/{total_layers}  "
                    f"payload={total_payload_bytes//1024}KB / orig={total_orig_bytes//1024}KB  "
                    f"ratio={ratio:.3f}"
                )
                if i_layers:                            # <-- 新增
                    print(f"  [delta:B] frame={i} I-layers ({len(i_layers)}): {i_layers}")

                retrieval_chunk_indices.append(i)
                retrieval_layer_key_vecs.append(chunk_layer_key_vec.cpu().contiguous())
                retrieval_summary_vecs.append(chunk_summary_vec.cpu().contiguous())
                frame_cache[i] = tuple(
                    (k.cpu().contiguous(), v.cpu().contiguous()) for k, v in delta_kv
                )
                frame_order.append(i)
                while len(frame_order) > max_in_memory:
                    old = frame_order.pop(0)
                    if old in frame_cache:
                        del frame_cache[old]

            # 释放大对象
            del outputs, full_kv_after
            if past_kv is not None:
                del past_kv
            del model_inputs, pixel_values
            gc.collect()

    finally:
        if prune_ctx is not None and prune_ctx.log_stats:
            print(prune_ctx.summary())

    if kv_cache_dir is not None:
        common_metadata["full_merged_seq_len"] = int(full_merged_seq_len)
        if pack_writer is not None:
            pack_writer.write_index(common_metadata)
            pack_writer.close()
        _write_chunk_manifest(kv_cache_dir, chunk_records, common_metadata)
        _write_retrieval_index(
            kv_cache_dir,
            retrieval_chunk_indices,
            retrieval_layer_key_vecs,
            retrieval_summary_vecs,
            common_metadata,
        )

    return None