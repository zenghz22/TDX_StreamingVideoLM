"""LLaVA-OneVision 视频预处理与 KV cache 生成/保存。"""

import time

import torch
from decord import VideoReader, cpu
from transformers import LlavaOnevisionForConditionalGeneration as LlavaOV
from transformers import LlavaOnevisionProcessor

# 统一前缀：编码和解码都基于同一语境，避免 KV 与后续文本提示错位。
ENCODE_PREFIX = "请先理解视频内容，并记住关键事件与人物。"
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
    """保存 KV cache 与元信息。"""
    kv_cache_cpu = detach_kv_to_cpu(kv_cache)
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

    payload = {"kv_cache": kv_cache_cpu, "metadata": metadata}
    torch.save(payload, kv_cache_path)
    print(f"KV cache saved to {kv_cache_path}.")
    print(f"KV metadata: {metadata}")


def encode_video(video, processor, model=None, chunk_size=64, encode_prefix=ENCODE_PREFIX, stage_mark=None):
    """将视频分块编码成 KV cache。"""
    kv_cache = []
    num_frames = video.shape[0]
    num_chunks = (num_frames + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk = video[i * chunk_size : min((i + 1) * chunk_size, num_frames)]
        if stage_mark is not None:
            stage_mark(f"chunk_{i}_start")        

        pixel_values = processor.video_processor(chunk, return_tensors="pt").pixel_values_videos.to("cpu")
        print(f"[chunk {i}] pixel_values_videos shape: {tuple(pixel_values.shape)}")

        if model is None:
            kv_cache.append({"chunk_index": i, "pixel_values_shape": tuple(pixel_values.shape)})
            continue

        if i == 0:
            encode_text = _build_encode_text(encode_prefix)
        else:
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

        past_seq_len = _get_cache_seq_len(kv_cache)
        current_seq_len = model_inputs["input_ids"].shape[1]
        if past_seq_len > 0:
            model_inputs["attention_mask"] = torch.ones(
                (model_inputs["input_ids"].shape[0], past_seq_len + current_seq_len),
                dtype=model_inputs["attention_mask"].dtype,
                device=device,
            )

        print(f"[chunk {i}] encode_text: {repr(encode_text)}")
        print(f"[chunk {i}] past_seq_len: {past_seq_len}, current_seq_len: {current_seq_len}")
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"[chunk {i}] model_inputs[{k}] shape: {tuple(v.shape)}")

        #if stage_mark is not None:
        #    stage_mark(f"chunk_{i}_prefill_start")

        with torch.no_grad():
            outputs = model(
                **model_inputs,
                past_key_values=kv_cache if kv_cache else None,
                use_cache=True,
                return_dict=True,
            )

        #if stage_mark is not None:
        #    stage_mark(f"chunk_{i}_prefill_end")

        kv_cache = outputs.past_key_values

        if kv_cache:
            first_layer_k = kv_cache[0][0]
            first_layer_v = kv_cache[0][1]
            print(f"[chunk {i}] kv_cache layer0 key shape: {tuple(first_layer_k.shape)}")
            print(f"[chunk {i}] kv_cache layer0 value shape: {tuple(first_layer_v.shape)}")
   
        #if stage_mark is not None:
        #    stage_mark(f"chunk_{i}_end")

    return kv_cache