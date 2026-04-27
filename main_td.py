import os
import gc
import time
import argparse
import logging
import sys

from kvcache_generate_td import load_model, load_video, encode_video
from kvcache_retrieve_td import decode_kvcache
from kvcache_select_td import select_chunks, select_chunks_per_layer
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

    # 差分压缩参数（方向 B）
    parser.add_argument("--delta_max_p", type=int, default=0,
                        help="Max consecutive P-frames per layer (0=disable delta).")
    parser.add_argument("--delta_threshold", type=float, default=1e-3,
                        help="V delta sparse quantization threshold.")
    parser.add_argument("--delta_ratio_threshold", type=float, default=0.75,
                        help="Max V delta nnz ratio to keep P-frame, else fallback to I.")

    parser.add_argument("--decode_select", type=int, default=0,
                        help="Number of chunks to select (0=all, use per-layer if >0).")

    # 剪枝参数
    parser.add_argument("--prune", action="store_true",
                        help="Enable video token pruning.")
    parser.add_argument("--prune_temporal", type=float, default=0.0,
                        help="Temporal frame keep ratio (0~1).")
    parser.add_argument("--prune_spatial", type=float, default=0.0,
                        help="Spatial downscale ratio (0~1).")

    # 加密参数
    parser.add_argument("--encrypt", action="store_true",
                        help="Enable KV cache encryption.")
    parser.add_argument("--key_file", type=str, default="../data/master.key",
                        help="Master key file path.")

    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/haimian_7.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    question      = "Who is in the video, and what are they doing?"
    encode_prefix = "You are a helpful assistant. Please understand the video content and prepare to answer single-choice questions."

    # Prune context
    prune_ctx = None
    if args.prune and (args.prune_temporal > 0 or args.prune_spatial > 0):
        from video_prune import PruneContext
        prune_ctx = PruneContext(
            enabled=True,
            temporal_enabled=(args.prune_temporal > 0),
            temporal_keep_ratio=args.prune_temporal if args.prune_temporal > 0 else 0.6,
            spatial_enabled=(args.prune_spatial > 0),
            spatial_ratio=args.prune_spatial if args.prune_spatial > 0 else None,
            log_stats=True,
        )
        logger.info(f"[prune] temporal={args.prune_temporal}, spatial={args.prune_spatial}")

    # Crypto context
    crypto_ctx = None
    if args.encrypt:
        from kvcache_crypto_td import CryptoContext
        crypto_ctx = CryptoContext.from_key_file(args.key_file, create=True)
        logger.info(f"[crypto] Encryption enabled, key file: {args.key_file}")

    with measure_resources(args.mode, logger=logger, plot_file=args.plot_file, plot_lable=True) as monitor:
        # ── 编码阶段 ──
        if args.mode in ("encode_decode", "encode"):
            monitor["mark"]("load_model_encode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])

            video = load_video(video_path, sample_fps=0.5)

            monitor["mark"]("kvcache_encode_start")
            encode_video(
                video=video,
                processor=processor,
                model=model,
                chunk_size=args.chunk_size,
                encode_prefix=encode_prefix,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                prune_ctx=prune_ctx,
                crypto_ctx=crypto_ctx,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
                max_consecutive_p=args.delta_max_p,
                delta_threshold=args.delta_threshold,
                delta_ratio_threshold=args.delta_ratio_threshold,
            )
            monitor["mark"]("kvcache_encode_done")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        # ── 解码阶段 ──
        if args.mode in ("encode_decode", "decode"):
            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model)

            text_content = f"Question: {question}\nAnswer:"
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

            if args.decode_select > 0:
                decode_chunk_ids = select_chunks_per_layer(
                    kv_cache_path,
                    question,
                    processor,
                    model,
                    top_k=args.decode_select,
                    crypto_ctx=crypto_ctx,
                )
                logger.info(f"Decoding with top-{args.decode_select} per-layer chunks.")
            else:
                decode_chunk_ids = None
                logger.info("Decoding with all chunks.")

            monitor["mark"]("kvcache_decode")
            answer = decode_kvcache(
                kv_cache_path,
                question,
                processor,
                model,
                max_new_tokens=32,
                min_new_tokens=1,
                temperature=0.0,
                decode_strategy="sample",
                suffix=prompt,
                crypto_ctx=crypto_ctx,
                per_layer_chunk_indices=decode_chunk_ids,
            )

            logger.info(f"Model answer: {answer}")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        if crypto_ctx is not None:
            crypto_ctx.cleanup_tmp()