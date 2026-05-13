import os
import gc
import json
import time
import argparse
import logging
import sys

from kvcache_generate_td import load_model, load_video, encode_video
from kvcache_retrieve_td import decode_kvcache
from kvcache_select_td import select_chunks, select_chunks_per_layer
from kvpack_cache_td import (
    DECODE_POLICY_NONE, DECODE_POLICY_LRU,
    DECODE_POLICY_PER_LAYER_PREFIX, DECODE_POLICY_MID_LAYER_PREFIX,
    DECODE_POLICY_SHALLOW_FIRST, DECODE_POLICY_DEEPEST_FIRST,
    VALID_POLICY_CODES,
)
from zhz_hardware_eval_utils import *
from zhz_model_eval_utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TD-side video encoding and decoding")
    parser.add_argument("--mode", type=str, default="encode_decode",
                        choices=["encode_decode", "decode", "encode"])
    parser.add_argument("--plot_file", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--encode_memory", type=int, default=64)
    parser.add_argument("--encode_window", type=int, default=0)



    parser.add_argument("--decode_select", type=int, default=0,
                        help="Number of chunks to select (0=all, use per-layer if >0).")



    # 加密参数
    parser.add_argument("--encrypt", action="store_true",
                        help="Enable KV cache encryption.")
    parser.add_argument("--key_file", type=str, default="../data/master.key",
                        help="Master key file path.")

    # TD-resident block cache 参数(decode 阶段使用)
    parser.add_argument("--decode_memory", type=int, default=0,
                        help="TD block cache budget in (layer×frame) blocks. "
                             "0 = disable cache (equivalent to policy 0/NONE).")
    parser.add_argument("--decode_policy", type=int, default=DECODE_POLICY_NONE,
                        choices=list(VALID_POLICY_CODES),
                        help="0=NONE (no cache), 1=LRU, 2=PER_LAYER_PREFIX, "
                             "3=MID_LAYER_PREFIX, 4=SHALLOW_FIRST, 5=DEEPEST_FIRST")
    parser.add_argument("--decode_mid_layer_range", type=str, default="10,19",
                        help="For policy 3/MID_LAYER_PREFIX: 'lo,hi' inclusive "
                             "layer indices (0-indexed). Default '10,19' covers "
                             "10 middle layers.")

    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/haimian_7.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    question      = "Who is in the video, and what are they doing?"
    questions_file = "../data/haimian_7.json"
    encode_prefix = "You are a helpful assistant. Please understand the video content and prepare to answer single-choice questions."



    # Crypto context
    crypto_ctx = None
    if args.encrypt:
        from kvcache_crypto_td import CryptoContext
        crypto_ctx = CryptoContext.from_key_file(args.key_file, create=True)
        logger.info(f"[crypto] Encryption enabled, key file: {args.key_file}")

    # td_cache 在 main 作用域定义,跨 encode/decode 共享
    # (decode-only 模式下保持 None,因 num_chunks 需要 video 来推算,且单 decode
    #  无窗口内复用,cache 收益主要来自 encode 端 pre-warm)
    td_cache = None

    with measure_resources(args.mode, logger=logger, plot_file=args.plot_file, plot_lable=True) as monitor:
        # ── 编码阶段 ──
        if args.mode in ("encode_decode", "encode"):
            monitor["mark"]("load_model_encode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])

            video = load_video(video_path, sample_fps=0.5)

            # ── 构造 TD block cache(若启用) ──
            # 必须在 encode_video 之前创建,以便 encode 阶段同步 pre-warm
            if args.decode_policy != DECODE_POLICY_NONE and args.decode_memory > 0:
                from kvpack_cache_td import make_td_cache
                num_layers = int(model.language_model.config.num_hidden_layers)
                num_chunks = (len(video) + args.chunk_size - 1) // args.chunk_size
                try:
                    lo_str, hi_str = args.decode_mid_layer_range.split(",")
                    mid_lo, mid_hi = int(lo_str), int(hi_str)
                except ValueError:
                    raise ValueError(
                        f"--decode_mid_layer_range must be 'lo,hi' "
                        f"(got {args.decode_mid_layer_range!r})"
                    )
                td_cache = make_td_cache(
                    args.decode_policy, args.decode_memory,
                    num_layers=num_layers, num_chunks=num_chunks,
                    mid_layer_lo=mid_lo, mid_layer_hi=mid_hi,
                    logger_obj=logger,
                )

            monitor["mark"]("kvcache_encode_start")
            encode_video(
                video=video,
                processor=processor,
                model=model,
                chunk_size=args.chunk_size,
                encode_prefix=encode_prefix,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                crypto_ctx=crypto_ctx,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
                td_cache=td_cache,
            )
            monitor["mark"]("kvcache_encode_done")

            if td_cache is not None:
                logger.info(
                    f"[td_cache] encode pre-warm done. "
                    f"resident={len(td_cache)}/{td_cache.capacity} "
                    f"admits={td_cache.monitor.admits} "
                    f"rejects={td_cache.monitor.rejects}"
                )

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        # ── 解码阶段 ──
        if args.mode in ("encode_decode", "decode"):
            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model)

            # ── 加载问题列表 ─────────────────────────────────────────
            # 优先从 --questions_file 读;读不到时 fallback 到内置单问题。
            questions = []
            if questions_file and os.path.exists(questions_file):
                try:
                    with open(questions_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        questions = [q for q in data if isinstance(q, str) and q.strip()]
                    elif isinstance(data, dict) and "questions" in data:
                        questions = [q for q in data["questions"]
                                      if isinstance(q, str) and q.strip()]
                    else:
                        raise ValueError(
                            "expected JSON array of strings or "
                            "{'questions': [...]} object"
                        )
                    if not questions:
                        raise ValueError("no valid question strings found")
                    logger.info(
                        f"[questions] loaded {len(questions)} questions "
                        f"from {questions_file}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[questions] failed to load {questions_file}: {e}; "
                        f"falling back to single built-in question"
                    )
                    questions = [question]
            else:
                if questions_file:
                    logger.info(
                        f"[questions] file not found: {questions_file}; "
                        f"using built-in single question"
                    )
                questions = [question]

            if args.mode == "decode" and args.decode_policy != DECODE_POLICY_NONE \
                    and args.decode_memory > 0:
                logger.warning(
                    "[td_cache] policy != NONE but mode == 'decode' (no encode in "
                    "this run); cache starts empty. Cross-question reuse may still "
                    "build hit rate over multiple questions; use mode=encode_decode "
                    "for additional pre-warm benefit."
                )

            answers = []

            # ── 多问答 for 循环 (monitor 仍在外层) ─────────────────────
            for q_idx, q_text in enumerate(questions):
                logger.info(
                    f"\n========== Question {q_idx + 1}/{len(questions)} =========="
                )
                logger.info(f"[Q{q_idx + 1}] {q_text}")

                text_content = f"Question: {q_text}\nAnswer:"
                conversation_context = [{
                    "role": "user",
                    "content": [{"type": "text", "text": text_content}],
                }]
                prompt = processor.apply_chat_template(
                    conversation_context,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                logger.info(f"decode_select={args.decode_select}")

                # 每个问题要重新跑 select (不同 question 的 Q vector 不同)
                if args.decode_select > 0:
                    decode_chunk_ids = select_chunks_per_layer(
                        kv_cache_path,
                        q_text,
                        processor,
                        model,
                        top_k=args.decode_select,
                        crypto_ctx=crypto_ctx,
                    )
                    logger.info(
                        f"Decoding with top-{args.decode_select} per-layer chunks."
                    )
                else:
                    decode_chunk_ids = None
                    logger.info("Decoding with all chunks.")

                monitor["mark"](f"kvcache_decode_q{q_idx + 1}")
                answer = decode_kvcache(
                    kv_cache_path,
                    q_text,
                    processor,
                    model,
                    max_new_tokens=32,
                    min_new_tokens=1,
                    temperature=0.0,
                    decode_strategy="sample",
                    suffix=prompt,
                    crypto_ctx=crypto_ctx,
                    per_layer_chunk_indices=decode_chunk_ids,
                    td_cache=td_cache,
                )

                answers.append(answer)
                logger.info(f"[A{q_idx + 1}] {answer}")

            # ── 全部 decode 完成,打 cache 总汇总 ─────────────────────
            logger.info(f"\n========== All {len(questions)} questions answered ==========")
            for idx, (q, a) in enumerate(zip(questions, answers)):
                logger.info(f"  Q{idx + 1}: {q}")
                logger.info(f"  A{idx + 1}: {a}")

            if td_cache is not None:
                td_cache.log_summary()

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        if crypto_ctx is not None:
            crypto_ctx.cleanup_tmp()