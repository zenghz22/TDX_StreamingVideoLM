"""LLaVA-OneVision KV cache 加载与解码。"""

import json
import os

import torch
from safetensors import safe_open
from kvpack_mmap_td import KVPackReader, has_kvpack


def _to_model_cache(past_key_values):
    """
    兼容 transformers 新旧 cache API：
    - 旧版：past_key_values 为 tuple/list
    - 新版（如较新 Qwen2）：要求传入 Cache 对象（如 DynamicCache）
    """
    if past_key_values is None:
        return None

    if not isinstance(past_key_values, (tuple, list)):
        return past_key_values

    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(past_key_values)
    except Exception:
        # 旧版 transformers 无 cache_utils 或不支持 from_legacy_cache，回退 legacy tuple
        return past_key_values


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

        # RoPE 位置一致性保证：每个 chunk 的 K/V 向量在 encode 时绑定了其绝对位置。
        # 若直接拼接非连续的 chunk（如 [0, 38, 39]），chunk 38 的 K 向量仍编码
        # 位置 ~109364，但拼接后模型看到的相对位置是 ~551，两者相差 10 万+，
        # RoPE 旋转角度完全错误，导致 attention 崩溃，产生乱码输出。
        # 解决方案：始终加载连续范围 [min_idx, max_idx] 的全部 chunk。
        min_idx = min(chunk_indices)
        max_idx = max(chunk_indices)
        selected_chunks = [c for c in chunks_sorted
                           if min_idx <= int(c["chunk_index"]) <= max_idx]
        selected = selected_chunks[-1]
        print(f"[load_kv_cache] chunk_indices {sorted(chunk_indices)} → "
              f"expanded to contiguous range [{min_idx}, {max_idx}] "
              f"({len(selected_chunks)} chunks) for RoPE consistency.")
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


def _assemble_per_layer_kv(
    kv_cache_dir: str,
    per_layer_chunk_indices,   # List[List[int]]，长度=层数
    *,
    crypto_ctx=None,
    map_location: str = "cpu",
):
    """
    ReKV per-layer 独立检索的加载层：每层从各自选出的 chunk 集合里拼装 KV。

    原理
    ----
    past_key_values 是 tuple[L] of (K_L, V_L)，HuggingFace 每层独立传入自己的 KV。
    因此不同层携带不同 chunk 的内容完全合法，无需修改 attention 计算。

    唯一约束：所有层的 T（序列维度）必须相同，因为 attention_mask 是全局共享的。
    - 不开 temporal prune：所有 chunk T_chunk 相同（16帧×196tokens=3136），
      k 个 chunk → T = k × 3136，自动满足。
    - 开 temporal prune：不同 chunk T_chunk 可能不同，此时做 zero-padding 并输出警告。
      zero-padded 的 K/V 对 attention 影响极小（Q·0=0，softmax 结果接近 0），
      但若精度敏感，建议 eval 时关闭 temporal prune。

    参数
    ----
    per_layer_chunk_indices : List[List[int]]
        per_layer_chunk_indices[L] = 第 L 层选出的 chunk_index 列表

    返回
    ----
    per_layer_kv : tuple[L] of (K_L, V_L)   - 可直接传入 _to_model_cache()
    past_seq_len : int                        - 最大序列长度（含 padding）
    """
    if not has_kvpack(kv_cache_dir):
        raise FileNotFoundError(f"kvpack_index.json not found in {kv_cache_dir}")
    reader = KVPackReader(kv_cache_dir)
    try:
        n_layers = len(per_layer_chunk_indices)
        per_layer_kv = []
        seq_lens = []
        for L_idx in range(n_layers):
            selected = per_layer_chunk_indices[L_idx]
            if not selected:
                raise ValueError(f"layer {L_idx} has empty chunk selection")
            k_list = []
            v_list = []
            for frame_idx in selected:
                k, v, _ = reader.read_layer_frame(L_idx, int(frame_idx), map_location=map_location)
                k_list.append(k)
                v_list.append(v)
            K_cat = torch.cat(k_list, dim=-2)
            V_cat = torch.cat(v_list, dim=-2)
            per_layer_kv.append((K_cat, V_cat))
            seq_lens.append(K_cat.shape[-2])
        union_ids = sorted(set((L, ci) for L, ids in enumerate(per_layer_chunk_indices) for ci in ids))
    finally:
        reader.close()

    # ── 3. 处理不等长（temporal prune 场景）─────────────────────────────────
    unique_lens = set(seq_lens)
    if len(unique_lens) > 1:
        max_len = max(seq_lens)
        min_len = min(seq_lens)
        print(
            f"[assemble_per_layer_kv] WARNING: layers have unequal seq_len "
            f"(min={min_len}, max={max_len}). "
            f"Zero-padding shorter layers to {max_len}. "
            f"Consider disabling temporal pruning for eval to avoid this."
        )
        new_kv = []
        for (K, V), T in zip(per_layer_kv, seq_lens):
            if T < max_len:
                pad = max_len - T
                K = torch.cat([K, K.new_zeros(*K.shape[:-2], pad, K.shape[-1])], dim=-2)
                V = torch.cat([V, V.new_zeros(*V.shape[:-2], pad, V.shape[-1])], dim=-2)
            new_kv.append((K, V))
        per_layer_kv = new_kv
        past_seq_len = max_len
    else:
        past_seq_len = seq_lens[0]

    print(
        f"[assemble_per_layer_kv] Loaded {len(union_ids)} unique chunk files, "
        f"assembled {n_layers} layers × {past_seq_len} tokens."
    )
    return tuple(per_layer_kv), past_seq_len


def load_kv_cache(kv_cache_path, map_location="cpu", chunk_index=None, chunk_indices=None):
    """从磁盘加载 KV cache 与元信息（优先 safetensors）。"""
    if not os.path.isdir(kv_cache_path) or not has_kvpack(kv_cache_path):
        raise FileNotFoundError(
            f"Only kvpack_mmap_v1 is supported now; kvpack_index.json not found in {kv_cache_path}"
        )
    reader = KVPackReader(kv_cache_path)
    try:
        frame_ids = sorted(reader.frames.keys())
        if chunk_indices is not None:
            min_idx = min(chunk_indices)
            max_idx = max(chunk_indices)
            selected_frames = [f for f in frame_ids if min_idx <= f <= max_idx]
        else:
            max_frame = frame_ids[-1] if chunk_index is None else int(chunk_index)
            selected_frames = [f for f in frame_ids if f <= max_frame]

        n_layers = int(reader.common_metadata.get("num_layers", 0))
        if n_layers <= 0:
            n_layers = max((l for l, _ in reader.by_layer_frame.keys()), default=-1) + 1

        per_layer = []
        for layer_idx in range(n_layers):
            k_list, v_list = [], []
            for frame_idx in selected_frames:
                k, v, _ = reader.read_layer_frame(layer_idx, frame_idx, map_location=map_location)
                k_list.append(k)
                v_list.append(v)
            per_layer.append((torch.cat(k_list, dim=-2), torch.cat(v_list, dim=-2)))
        kv_cache = tuple(per_layer)
        full_seq_len = int(kv_cache[0][0].shape[-2]) if kv_cache else 0
        metadata = {
            "past_seq_len": full_seq_len,
            "loaded_from_dir": kv_cache_path,
            "loaded_chunks": selected_frames,
            "full_merged_seq_len": int(reader.common_metadata.get("full_merged_seq_len", full_seq_len)),
            "storage_format": "kvpack_mmap_v1",
        }
        print(f"Loaded KV cache from {kv_cache_path} (frames {selected_frames})")
        return kv_cache, metadata
    finally:
        reader.close()


def _load_prefix_cache(kv_cache_dir: str, map_location: str = "cpu"):
    """
    加载 prefix_cache.safetensors（encode_prefix 文字 token 的 KV）。
    若文件不存在返回 (None, 0)，兼容旧数据。
    """
    prefix_path = os.path.join(kv_cache_dir, "prefix_cache.safetensors")
    if not os.path.exists(prefix_path):
        return None, 0
    prefix_kv, meta = _load_single_safetensors_kv(prefix_path, map_location=map_location)
    prefix_seq_len = int(prefix_kv[0][0].shape[-2]) if prefix_kv else 0
    print(f"[load_prefix_cache] Loaded prefix KV: {prefix_seq_len} tokens from {prefix_path}")
    return prefix_kv, prefix_seq_len


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
    per_layer_chunk_indices=None,
    max_new_tokens=128,
    min_new_tokens=8,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    decode_strategy="sample",
    suffix=None,
    crypto_ctx=None,
):
    """加载 KV cache，直接文本解码，跳过视频预处理与编码。

    chunk_indices : list[int] | None
        指定只加载哪些 chunk 的 KV（例如 [0, 3, 7]）。
        None 表示加载全部（原有行为）。
    per_layer_chunk_indices : list[list[int]] | None
        ReKV per-layer 独立检索模式。由 select_chunks_per_layer() 生成。
        提供时优先使用，忽略 chunk_indices 参数。
        per_layer_chunk_indices[L] = 第 L 层选出的 chunk 列表。
    decode_strategy : str
        "sample" 或 "greedy"。
    suffix : str | None
        直接作为 decode 的文本输入，绕过 _build_decode_suffix。
    crypto_ctx : CryptoContext | None
        KV cache 解密上下文。
    """
    device = next(model.parameters()).device

    # ── 类型检查：自动识别 per-layer 列表 vs 全局 chunk 列表 ──────────────────
    # select_chunks()          → List[int]        → 全局模式（分支 B）
    # select_chunks_per_layer()→ List[List[int]]  → per-layer 模式（分支 A）
    # 如果用户误把 List[int] 传给 per_layer_chunk_indices，自动降级到分支 B。
    if per_layer_chunk_indices is not None:
        if len(per_layer_chunk_indices) == 0:
            per_layer_chunk_indices = None   # 空列表 → 全局模式
        elif isinstance(per_layer_chunk_indices[0], int):
            print(
                "[decode] WARNING: per_layer_chunk_indices received a flat List[int] "
                "(looks like output of select_chunks, not select_chunks_per_layer). "
                "Treating as chunk_indices and falling back to global mode."
            )
            chunk_indices = list(per_layer_chunk_indices)
            per_layer_chunk_indices = None

    # ── 分支 A：per-layer 独立检索模式 ────────────────────────────────────
    if per_layer_chunk_indices is not None:
        print(f"[decode] Per-layer retrieval mode: {len(per_layer_chunk_indices)} layers")

        # 加载 prefix（所有层共用）
        prefix_kv, prefix_seq_len = _load_prefix_cache(
            kv_cache_path, map_location="cpu"
        ) if os.path.isdir(kv_cache_path) else (None, 0)

        # 逐层加载各自的 chunk KV
        video_kv, video_seq_len = _assemble_per_layer_kv(
            kv_cache_path, per_layer_chunk_indices,
            crypto_ctx=crypto_ctx, map_location="cpu"
        )

        # 拼 prefix + video，逐层拼接（所有层 prefix 相同）
        if prefix_kv is not None:
            prefix_dev = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
            video_dev  = tuple((k.to(device), v.to(device)) for k, v in video_kv)
            kv_cache = _concat_kv_segments([prefix_dev, video_dev])
            print(f"[decode] prefix={prefix_seq_len} + video={video_seq_len} = "
                  f"{prefix_seq_len + video_seq_len} tokens per layer")
        else:
            kv_cache = move_to_device(video_kv, device)
            prefix_seq_len = 0

        past_seq_len = kv_cache[0][0].shape[-2]   # 实际加载的 KV 总长

        # per-layer 模式也需要 full_merged_seq_len，确保 Q 位置 > 所有 K 位置
        # 原因：K 向量保留了 encode 时的绝对位置（post-RoPE baked-in）；
        # 若 Q < 某 K 位置，相对距离为负，破坏因果注意力 → 模型立即输出 <|im_end|>
        manifest_path = os.path.join(kv_cache_path, "manifest.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as _f:
                _manifest = json.load(_f)
            _common = _manifest.get("common_metadata", {}) or {}
            full_merged_seq_len_for_pos = int(_common.get("full_merged_seq_len", past_seq_len))
        except Exception:
            full_merged_seq_len_for_pos = past_seq_len
        metadata = {"full_merged_seq_len": full_merged_seq_len_for_pos}

    # ── 分支 B：原有全局 chunk 模式（向后兼容）─────────────────────────────
    else:
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

        if os.path.isdir(kv_cache_path):
            prefix_kv, prefix_seq_len = _load_prefix_cache(
                kv_cache_path, map_location="cpu"
            )
            if prefix_kv is not None and kv_cache:
                prefix_kv_dev = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
                kv_cache = _concat_kv_segments([prefix_kv_dev, kv_cache])
                print(f"[decode] Prepended prefix ({prefix_seq_len} tokens) → "
                      f"total KV seq_len={kv_cache[0][0].shape[-2]}")
            elif prefix_kv is not None:
                kv_cache = tuple((k.to(device), v.to(device)) for k, v in prefix_kv)
        else:
            prefix_seq_len = 0

        kv_cache = move_to_device(kv_cache, device)
        past_seq_len = kv_cache[0][0].shape[-2]

    model_past_key_values = _to_model_cache(kv_cache)

    suffix = suffix if suffix is not None else _build_decode_suffix(question)
    model_inputs = processor(text=[suffix], return_tensors="pt")
    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in model_inputs.items()}

    # past_seq_len 已在两个分支中分别设置（per-layer 或 全局 chunk 路径）
    # 这里统一从实际 KV tensor 形状取，确保一致
    if per_layer_chunk_indices is not None:
        # per-layer 路径：past_seq_len 已由 _assemble_per_layer_kv 确定
        pass   # past_seq_len already set above
    else:
        past_seq_len = _get_past_seq_len(kv_cache, metadata)
    current_seq_len = model_inputs["input_ids"].shape[1]
    full_attention_mask = torch.ones(
        (model_inputs["input_ids"].shape[0], past_seq_len + current_seq_len),
        dtype=model_inputs["attention_mask"].dtype,
        device=device,
    )
    model_inputs["attention_mask"] = full_attention_mask

    # ---- position_ids ----
    # 我们的 K 向量是 post-RoPE（encode 时 baked-in 绝对位置）。
    # Q 必须放在所有 K 位置之后，否则相对距离为负，破坏因果注意力。
    #
    # ReKV 原论文用的是 pre-RoPE K + decode 时重新施加连续 RoPE，
    # 但我们存的是 post-RoPE K，因此不能直接套用 ReKV 的 past_seq_len 策略。
    # 必须用 full_merged_seq_len（encode 时的完整视频总长）作为 Q 起始位置，
    # 确保 Q > 所有 K 的 baked-in 绝对位置。
    full_merged_seq_len = int(metadata.get("full_merged_seq_len", past_seq_len))
    position_ids = torch.arange(
        full_merged_seq_len,
        full_merged_seq_len + current_seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    model_inputs["position_ids"] = position_ids
    print(
        f"[decode] position_ids: start={full_merged_seq_len}  "
        f"loaded_kv={past_seq_len}, full_merged={full_merged_seq_len}  "
        f"Q-K max relative dist ≈ {full_merged_seq_len}"
    )

    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"decode model_inputs[{k}] shape: {tuple(v.shape)}")

    print(f"decode past_seq_len: {past_seq_len}, current_seq_len: {current_seq_len}")

    if decode_strategy not in {"sample", "greedy"}:
        raise ValueError(f"Unsupported decode_strategy={decode_strategy}, expected one of ['sample', 'greedy'].")

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            past_key_values=model_past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past = outputs.past_key_values
        generated = []
        generated_token_ids = []

        # next_token_pos：下一个待生成 token 的位置（紧接在 prefill 之后）
        # 与 position_ids 起始位置保持一致，从 full_merged_seq_len 处延伸
        next_token_pos = full_merged_seq_len + current_seq_len

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
            step_position_ids = torch.tensor(
                [[next_token_pos]], dtype=torch.long, device=device
            )
            step_outputs = model(
                input_ids=next_token,
                attention_mask=step_attention_mask,
                position_ids=step_position_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            next_token_pos += 1
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
