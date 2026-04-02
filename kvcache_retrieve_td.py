"""LLaVA-OneVision KV cache 加载与解码。"""

import json
import os
import pickle

import torch
from safetensors import safe_open


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def _resolve_chunk_files_from_dir(kv_cache_dir, chunk_index=None, chunk_indices=None):
    manifest_path = os.path.join(kv_cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in KV cache dir: {kv_cache_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunk records found in manifest: {manifest_path}")

    chunks_sorted = sorted(chunks, key=lambda x: int(x["chunk_index"]))

    # 方案 B：用户指定任意 chunk 子集
    if chunk_indices is not None:
        idx_map = {int(c["chunk_index"]): c for c in chunks_sorted}
        missing = [i for i in chunk_indices if i not in idx_map]
        if missing:
            raise ValueError(f"chunk_indices {missing} not found in manifest: {manifest_path}")
        selected_chunks = sorted([idx_map[i] for i in chunk_indices], key=lambda c: int(c["chunk_index"]))
        selected = selected_chunks[-1]
    else:
        if chunk_index is None:
            selected = chunks_sorted[-1]
            selected_chunk_index = int(selected["chunk_index"])
        else:
            selected_chunk_index = int(chunk_index)
            idx_map = {int(c["chunk_index"]): c for c in chunks_sorted}
            if selected_chunk_index not in idx_map:
                raise ValueError(f"chunk_index={selected_chunk_index} not found in {manifest_path}")
            selected = idx_map[selected_chunk_index]

        selected_chunks = [c for c in chunks_sorted if int(c["chunk_index"]) <= selected_chunk_index]
        if not selected_chunks:
            raise ValueError(f"No chunks available before chunk_index={selected_chunk_index} in {manifest_path}")

    resolved_files = []
    for c in selected_chunks:
        chunk_file = c.get("file")
        if not chunk_file:
            raise ValueError(f"Invalid chunk entry in manifest: {c}")
        resolved_files.append(os.path.join(kv_cache_dir, chunk_file))

    return resolved_files, manifest, selected, selected_chunks


def _load_single_safetensors_kv(kv_cache_path, map_location="cpu"):
    kv_layers = {}
    metadata = {}
    with safe_open(kv_cache_path, framework="pt", device=map_location) as f:
        raw_metadata = f.metadata() or {}
        for k, v in raw_metadata.items():
            try:
                metadata[k] = json.loads(v)
            except (TypeError, json.JSONDecodeError):
                metadata[k] = v

        for key in f.keys():
            if key.startswith("layer_") and (key.endswith(".k") or key.endswith(".v")):
                layer_str, kv_suffix = key.split(".")
                layer_idx = int(layer_str.split("_")[1])
                kv_layers.setdefault(layer_idx, {})[kv_suffix] = f.get_tensor(key)

    kv_cache = []
    for layer_idx in sorted(kv_layers.keys()):
        layer = kv_layers[layer_idx]
        if "k" not in layer or "v" not in layer:
            raise ValueError(f"Incomplete KV for layer {layer_idx} in {kv_cache_path}")
        kv_cache.append((layer["k"], layer["v"]))
    return tuple(kv_cache), metadata


def _concat_kv_segments(kv_segments):
    if not kv_segments:
        return tuple()
    n_layers = len(kv_segments[0])
    full_kv = []
    for layer_idx in range(n_layers):
        k_parts = [seg[layer_idx][0] for seg in kv_segments]
        v_parts = [seg[layer_idx][1] for seg in kv_segments]
        full_kv.append((torch.cat(k_parts, dim=-2), torch.cat(v_parts, dim=-2)))
    return tuple(full_kv)


def load_kv_cache(kv_cache_path, map_location="cpu", chunk_index=None, chunk_indices=None):
    """从磁盘加载 KV cache 与元信息（优先 safetensors）。"""
    manifest = None
    selected_chunk = None
    resolved_path = kv_cache_path
    selected_chunks = None
    if os.path.isdir(kv_cache_path):
        resolved_files, manifest, selected_chunk, selected_chunks = _resolve_chunk_files_from_dir(
            kv_cache_path, chunk_index=chunk_index, chunk_indices=chunk_indices
        )
        kv_segments = []
        metadata = {}
        for shard_path in resolved_files:
            shard_kv, shard_metadata = _load_single_safetensors_kv(shard_path, map_location=map_location)
            kv_segments.append(shard_kv)
            metadata = shard_metadata
        kv_cache = _concat_kv_segments(kv_segments)
        full_seq_len = int(kv_cache[0][0].shape[-2]) if kv_cache else 0
        if "past_seq_len" in metadata:
            metadata["past_seq_len_shard"] = metadata["past_seq_len"]
        metadata["past_seq_len"] = full_seq_len
        metadata["chunk_manifest"] = manifest
        metadata["selected_chunk"] = selected_chunk
        metadata["loaded_chunks"] = selected_chunks
        metadata["loaded_from_dir"] = kv_cache_path
        loaded_indices = [int(c["chunk_index"]) for c in selected_chunks]
        resolved_path = f"{kv_cache_path} (chunks {loaded_indices})"
        print(f"Loaded KV cache from {resolved_path}")
        if metadata:
            print(f"KV metadata: {metadata}")
        return kv_cache, metadata

    if resolved_path.endswith(".safetensors"):
        kv_cache, metadata = _load_single_safetensors_kv(resolved_path, map_location=map_location)
    else:
        # 兼容旧版 pt 存档（便于平滑迁移）
        try:
            payload = torch.load(resolved_path, map_location=map_location, weights_only=True)
        except TypeError:
            payload = torch.load(resolved_path, map_location=map_location)
        except pickle.UnpicklingError as exc:
            print("Warning: weights_only=True load failed, fallback to weights_only=False for trusted local cache.")
            print(f"Details: {exc}")
            payload = torch.load(resolved_path, map_location=map_location, weights_only=False)

        if isinstance(payload, dict) and "kv_cache" in payload:
            kv_cache = payload["kv_cache"]
            metadata = payload.get("metadata", {})
        else:
            kv_cache = payload
            metadata = {}

    print(f"Loaded KV cache from {resolved_path}")
    if metadata:
        print(f"KV metadata: {metadata}")
    return kv_cache, metadata


def _build_decode_suffix(question):
    # 与编码前缀配套的续写格式，避免 chat-template 与 KV 上下文不匹配。
    return "\nQuestion: " + question + "\nAnswer: "


def _get_past_seq_len(kv_cache, metadata):
    actual_seq_len = 0
    try:
        actual_seq_len = int(kv_cache[0][0].shape[-2])
    except Exception:
        actual_seq_len = 0

    meta_seq_len = metadata.get("past_seq_len")
    if meta_seq_len is not None:
        meta_seq_len = int(meta_seq_len)
        # 若 metadata 与实际 KV 长度不一致（例如分片 metadata 是 delta 长度），以实际长度为准。
        if actual_seq_len > 0 and meta_seq_len != actual_seq_len:
            print(
                f"Warning: metadata past_seq_len={meta_seq_len} mismatches actual KV len={actual_seq_len}, "
                "using actual KV len."
            )
            return actual_seq_len
        return meta_seq_len
    return actual_seq_len


def decode_kvcache(
    kv_cache_path,
    question,
    processor,
    model,
    chunk_index=None,
    chunk_indices=None,
    max_new_tokens=128,
    min_new_tokens=8,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    decode_strategy="sample",
):
    """加载 KV cache，直接文本解码，跳过视频预处理与编码。

    chunk_indices : list[int] | None
        指定只加载哪些 chunk 的 KV（例如 [0, 3, 7]）。
        None 表示加载全部（原有行为）。
        加载的 chunk 越少，decode 阶段峰值内存越低。
    decode_strategy : str
        生成策略：
          - "sample": 采样（temperature/top_p/repetition_penalty 生效）
          - "greedy": 贪心（忽略 top_p，temperature/repetition_penalty 不生效）
    """
    kv_cache, metadata = load_kv_cache(
        kv_cache_path, map_location="cpu",
        chunk_index=chunk_index, chunk_indices=chunk_indices,
    )
    if not kv_cache:
        raise ValueError("KV cache is empty.")

    model_name = getattr(model.config, "_name_or_path", None)
    expected_model = metadata.get("model_name_or_path")
    if expected_model and model_name and expected_model != model_name:
        raise ValueError(f"Model mismatch: cache={expected_model}, current={model_name}")

    device = next(model.parameters()).device
    kv_cache = move_to_device(kv_cache, device)

    suffix = _build_decode_suffix(question)
    model_inputs = processor(text=[suffix], return_tensors="pt")
    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in model_inputs.items()}

    past_seq_len = _get_past_seq_len(kv_cache, metadata)
    current_seq_len = model_inputs["input_ids"].shape[1]
    full_attention_mask = torch.ones(
        (model_inputs["input_ids"].shape[0], past_seq_len + current_seq_len),
        dtype=model_inputs["attention_mask"].dtype,
        device=device,
    )
    model_inputs["attention_mask"] = full_attention_mask

    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"decode model_inputs[{k}] shape: {tuple(v.shape)}")

    print(f"decode past_seq_len: {past_seq_len}, current_seq_len: {current_seq_len}")

    if decode_strategy not in {"sample", "greedy"}:
        raise ValueError(f"Unsupported decode_strategy={decode_strategy}, expected one of ['sample', 'greedy'].")

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            past_key_values=kv_cache,
            use_cache=True,
            return_dict=True,
        )

        past = outputs.past_key_values
        generated = []
        generated_token_ids = []

        logits = outputs.logits[:, -1, :]
        if decode_strategy == "greedy":
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token)
        generated_token_ids.append(int(next_token.item()))

        for step_idx in range(max_new_tokens - 1):
            step_attention_mask = torch.ones(
                (1, past_seq_len + current_seq_len + len(generated)),
                dtype=model_inputs["attention_mask"].dtype,
                device=device,
            )
            step_outputs = model(
                input_ids=next_token,
                attention_mask=step_attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = step_outputs.past_key_values
            logits = step_outputs.logits[:, -1, :]

            if decode_strategy == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # repetition penalty: 降低已生成 token 再次被选中的概率。
                if repetition_penalty > 1.0:
                    for tid in set(generated_token_ids):
                        logits[:, tid] = logits[:, tid] / repetition_penalty

                logits = logits / max(temperature, 1e-5)

                # nucleus sampling (top-p)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_remove = cum_probs > top_p
                    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                    sorted_remove[..., 0] = False
                    sorted_logits[sorted_remove] = -1e10
                    logits = torch.full_like(logits, -1e10).scatter(1, sorted_indices, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            generated_token_ids.append(int(next_token.item()))

            eos_id = getattr(model.generation_config, "eos_token_id", None)
            if eos_id is not None and int(next_token.item()) == int(eos_id) and (step_idx + 1) >= min_new_tokens:
                break

    answer_ids = torch.cat(generated, dim=1)[0]
    answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer:
        raw_answer = processor.decode(answer_ids, skip_special_tokens=False)
        print("Warning: decoded empty answer.")
        print(f"Decoded (skip_special_tokens=True): {repr(answer)}")
        print(f"Decoded (skip_special_tokens=False): {repr(raw_answer)}")
    return answer
