"""LLaVA-OneVision 视频预处理与 KV cache 生成/保存。"""

import time
import json
import os
from typing import Optional

import gc
import torch
from decord import VideoReader, cpu
from safetensors.torch import save_file
from transformers import LlavaOnevisionForConditionalGeneration as LlavaOV
from transformers import LlavaOnevisionProcessor

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

def _encode_and_save_prefix(encode_prefix, processor, model, kv_cache_dir, device):
    """
    单独 forward encode_prefix 文字（无视频），将 prefix KV 存为
    prefix_cache.safetensors。

    好处：
    - 各 chunk shard 只含视频 visual token 的 delta KV（不含前缀文字 token）
    - decode 时始终加载轻量 prefix_cache，再拼接连续窗口
    - 彻底解除"chunk 0 必须被选"的约束，避免非连续 chunk 的 RoPE 位置错乱
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

    return prefix_kv, prefix_seq_len


# ---------------------------------------------------------------------------
# encode_video — 支持 KVCacheManager（可选）
# ---------------------------------------------------------------------------

def encode_video(
    video,
    processor,
    model=None,
    chunk_size=64,
    encode_prefix=None,
    stage_mark=None,
    kv_cache_dir=None,
    manager=None,           # 可选：KVCacheManager 实例
):
    """
    将视频分块编码成 KV cache。

    Parameters
    ----------
    manager : KVCacheManager | None
        若提供，则由 manager 负责内存调度（LRU + 可选滑动窗口）。
        若为 None，退回原有行为（全量 KV 常驻内存）。

    行为说明
    --------
    无 manager（原有行为）
        每次 forward 传入全量累积 KV，内存随 chunk 数线性增长。

    有 manager，window_size=None（全局注意力 + LRU）
        每次 forward 传入全量历史 KV，但超出 max_in_memory 的 chunk
        会被 LRU evict 到磁盘，需要时再 reload。适合 chunk 数量少的情况。

    有 manager，window_size=W（滑动窗口）
        每次 forward 只传入最近 W 个 chunk 的 KV，窗口外的 chunk 永久驱逐。
        内存严格限制为 O(W)，但远程上下文不可见。
        Position IDs 通过 language_model 上的 pre-hook 修正（确保 RoPE 正确）。
    """
    kv_cache = []            # 原有模式下的全量累积 KV
    num_frames = video.shape[0]
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    chunk_records = []
    common_metadata = {}
    retrieval_chunk_indices = []
    retrieval_layer_key_vecs = []
    retrieval_summary_vecs = []

    # 安装 sliding window position hook（仅 manager + window_size 模式）
    use_manager = manager is not None and model is not None
    if use_manager and manager.window_size is not None:
        manager.install_position_hook(model.language_model)
        print(
            f"[encode_video] Sliding window mode: window_size={manager.window_size}, "
            f"max_in_memory={manager.max_in_memory}"
        )

    if kv_cache_dir is not None:
        os.makedirs(kv_cache_dir, exist_ok=True)

    # ---- 前缀单独编码 ----
    # encode_prefix 文字 token 的 KV 存为 prefix_cache.safetensors；
    # 各 chunk shard 只存视频 visual token 的 delta KV。
    prefix_kv = None
    prefix_seq_len = 0
    if model is not None:
        device_tmp = next(model.parameters()).device
        prefix_kv, prefix_seq_len = _encode_and_save_prefix(
            encode_prefix, processor, model, kv_cache_dir, device_tmp
        )
        # manager 的 _full_merged_seq_len 从 prefix_seq_len 开始累积
        if use_manager and prefix_seq_len > 0:
            manager._full_merged_seq_len = prefix_seq_len

    try:
        for i in range(num_chunks):
            chunk = video[i * chunk_size : min((i + 1) * chunk_size, num_frames)]
            if stage_mark is not None:
                stage_mark(f"chunk_{i}_start")

            pixel_values = processor.video_processor(chunk, return_tensors="pt").pixel_values_videos.to("cpu")
            print(f"[chunk {i}] pixel_values_videos shape: {tuple(pixel_values.shape)}")

            if model is None:
                kv_cache.append({"chunk_index": i, "pixel_values_shape": tuple(pixel_values.shape)})
                continue

            # 所有 chunk 都只用 VIDEO_PLACEHOLDER；
            # encode_prefix 已单独 forward 并存为 prefix_cache.safetensors。
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

            # ----------------------------------------------------------------
            # 决定传给 model 的 past_key_values 及 attention_mask
            # 所有模式均将 prefix_kv 拼在窗口/全量 KV 之前。
            # ----------------------------------------------------------------
            from kvcache_retrieve_td import _concat_kv_segments as _cat_kv
            def _prepend_prefix(base_kv, base_seq_len):
                """将 prefix_kv 拼到 base_kv 前面，返回 (merged_kv, merged_seq_len)。"""
                if prefix_kv is None:
                    return (base_kv, base_seq_len)
                if base_kv is None:
                    pkv = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
                    return (pkv, prefix_seq_len)
                pkv = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
                merged = _cat_kv([pkv, base_kv])
                return (merged, int(merged[0][0].shape[-2]))

            if use_manager:
                # manager 模式：获取窗口（或全量）历史 KV，再拼 prefix
                window_kv, window_seq_len = manager.get_past_kv_for_forward(i)
                past_kv, past_kv_seq_len = _prepend_prefix(window_kv, window_seq_len)
                current_text_len = model_inputs["input_ids"].shape[1]
                if past_kv_seq_len > 0:
                    model_inputs["attention_mask"] = torch.ones(
                        (model_inputs["input_ids"].shape[0], past_kv_seq_len + current_text_len),
                        dtype=model_inputs["attention_mask"].dtype,
                        device=device,
                    )
            else:
                # 原有模式：全量累积 KV + prefix
                base_kv = kv_cache if kv_cache else None
                base_seq = _get_cache_seq_len(kv_cache)
                past_kv, past_kv_seq_len = _prepend_prefix(base_kv, base_seq)
                current_text_len = model_inputs["input_ids"].shape[1]
                if past_kv_seq_len > 0:
                    model_inputs["attention_mask"] = torch.ones(
                        (model_inputs["input_ids"].shape[0], past_kv_seq_len + current_text_len),
                        dtype=model_inputs["attention_mask"].dtype,
                        device=device,
                    )

            print(f"[chunk {i}] encode_text: {repr(encode_text)}")
            print(f"[chunk {i}] past_kv_seq_len (passed to model): {past_kv_seq_len}")
            print(f"[chunk {i}] current_text_len: {current_text_len}")
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"[chunk {i}] model_inputs[{k}] shape: {tuple(v.shape)}")

            # ----------------------------------------------------------------
            # 捕获 pre-RoPE K 向量（用于 retrieval_index，内容相关性检索）
            # 注意：delta_kv 里的 K 是 post-RoPE（绝对位置已烘入），
            # 不同 chunk 的绝对位置差距数万，cosine sim 会被位置主导。
            # 用 hook 在 k_proj 输出后、apply_rotary_pos_emb 之前捕获，
            # 得到纯内容相关的 K 向量，可与 select 侧的 pre-RoPE Q 比较。
            # ----------------------------------------------------------------
            _pre_rope_k_layers = []   # [Hkv, D] per layer
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

            # position hook 用完立即关闭（避免影响后续操作）
            if use_manager:
                manager.deactivate_position_hook()

            full_kv_after = outputs.past_key_values

            # ----------------------------------------------------------------
            # 提取本 chunk 的 delta KV（新增的 token 部分）
            # delta = [past_kv_seq_len:] 即本 chunk 新增的 merged token 的 KV
            # 注意：.detach().cpu().contiguous() 会创建全新 tensor，与 full_kv_after
            # 无共享存储，之后可以安全释放 outputs / full_kv_after。
            # ----------------------------------------------------------------
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

                # ---- chunk summary vector（pre-RoPE K，供 select 检索）----
                if _pre_rope_k_layers:
                    chunk_layer_key_vec = torch.stack(_pre_rope_k_layers, dim=0).float()  # [L, Hkv, D]
                else:
                    # fallback: post-RoPE K mean（hook 未捕获时退化）
                    k_means = []
                    for layer_k, _ in delta_kv:
                        k_means.append(layer_k[0].mean(dim=1).float())
                    chunk_layer_key_vec = torch.stack(k_means, dim=0).float()
                _pre_rope_k_layers.clear()

                chunk_summary_vec = chunk_layer_key_vec.mean(dim=0).flatten().float()

                print(f"[chunk {i}] full_kv_seq_len: {full_kv_seq_len}")
                print(f"[chunk {i}] delta_seq_len (this chunk merged): {delta_seq_len}")
                print(f"[chunk {i}] summary_vec shape: {tuple(chunk_summary_vec.shape)}")

                if use_manager:
                    manager.update_full_merged_seq_len(delta_seq_len)

            else:
                delta_kv = None
                full_kv_seq_len = 0
                delta_seq_len = 0
                chunk_summary_vec = None

            # ----------------------------------------------------------------
            # 显式释放大对象引用，让 GC 能立即回收 tensor 内存
            # outputs      → 持有整个 model forward 结果（含全量 KV）
            # full_kv_after → 全量累积 KV 的直接引用
            # past_kv      → 本轮传给 model 的窗口/全量 KV concat 结果
            # model_inputs → 含 pixel_values 等大型 tensor
            # ----------------------------------------------------------------
            # 原有模式：更新全量 kv_cache（必须先于 full_kv_after 释放）
            if not use_manager:
                kv_cache = full_kv_after

            # ----------------------------------------------------------------
            # 保存 delta 到磁盘 & 更新状态
            # ----------------------------------------------------------------
            if kv_cache_dir is not None and delta_kv is not None:
                shard_file = f"chunk_{i:05d}.safetensors"
                shard_path = os.path.join(kv_cache_dir, shard_file)

                # 保存本 chunk 的 delta KV
                shard_metadata = save_kv_cache(
                    delta_kv,
                    shard_path,
                    model=model,
                    extra_metadata={
                        "encode_prefix": encode_prefix,
                        "chunk_index": i,
                        "chunk_size": chunk_size,
                        "num_frames": int(num_frames),
                        "is_delta_chunk": True,
                        "seq_start": int(past_kv_seq_len),
                        "seq_end": int(full_kv_seq_len),
                        "window_size": manager.window_size if use_manager else None,
                        "max_in_memory": manager.max_in_memory if use_manager else None,
                    },
                )

                if not common_metadata:
                    common_metadata = {
                        "encode_prefix": encode_prefix,
                        "chunk_size": chunk_size,
                        "num_frames": int(num_frames),
                        "model_name_or_path": shard_metadata.get("model_name_or_path"),
                        "window_size": manager.window_size if use_manager else None,
                    }

                chunk_records.append(
                    {
                        "chunk_index": i,
                        "file": shard_file,
                        "delta_seq_len": delta_seq_len,
                        "seq_start": int(past_kv_seq_len),
                        "seq_end": int(full_kv_seq_len),
                        "past_seq_len": shard_metadata.get("past_seq_len"),
                        "layer0_key_shape": shard_metadata.get("layer0_key_shape"),
                        # ReKV-style summary vector：跨层 K 均值，用于 prompt-aware 检索
                        #"summary_vec": chunk_summary_vec.tolist() if chunk_summary_vec is not None else None,
                    }
                )
                retrieval_chunk_indices.append(i)
                retrieval_layer_key_vecs.append(chunk_layer_key_vec.cpu().contiguous())
                retrieval_summary_vecs.append(chunk_summary_vec.cpu().contiguous())

                # 注册到 manager（manager 负责内存调度）
                if use_manager:
                    manager.register_chunk(
                        chunk_idx=i,
                        file_name=shard_file,
                        delta_kv=delta_kv,
                        seq_start=int(past_kv_seq_len),
                        seq_end=int(full_kv_seq_len),
                    )

            del outputs, full_kv_after
            if use_manager and past_kv is not None:
                del past_kv
            del model_inputs, pixel_values
            gc.collect()

            # 打印 manager 状态（便于调试）
            if use_manager:
                status = manager.memory_status()
                print(
                    f"[chunk {i}] Manager: in_memory={status['in_memory']}, "
                    f"full_merged_seq_len={status['full_merged_seq_len']}"
                )

    finally:
        # 无论是否发生异常，都清理 position hook
        if use_manager and manager.window_size is not None:
            manager.remove_position_hook()

    if kv_cache_dir is not None:
        # 把真实的 full_merged_seq_len 写入 common_metadata，
        # retrieve 侧直接读取，无需从 delta_seq_len 反推。
        if use_manager:
            common_metadata["full_merged_seq_len"] = manager._full_merged_seq_len
        else:
            # 无 manager 时，全量模式下 full_merged_seq_len = 最后一个 chunk 的 seq_end
            if chunk_records:
                common_metadata["full_merged_seq_len"] = chunk_records[-1]["seq_end"]
        _write_chunk_manifest(kv_cache_dir, chunk_records, common_metadata)
        _write_retrieval_index(
            kv_cache_dir,
            retrieval_chunk_indices,
            retrieval_layer_key_vecs,
            retrieval_summary_vecs,
            common_metadata,
        )

    # manager 模式下返回 None（KV 由 manager 管理，不在内存中累积）
    if use_manager:
        manager.print_stats()
        return None

    return kv_cache