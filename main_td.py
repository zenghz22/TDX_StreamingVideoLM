import os
import gc
import time
import argparse

from kvcache_generate_td import load_model, load_video, encode_video
from kvcache_manager_td import KVCacheManager
from kvcache_retrieve_td import decode_kvcache
from kvcache_select_td import select_chunks
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
    parser.add_argument("--encode_memory", type=int, default=64)
    parser.add_argument("--encode_window", type=int, default=0)
    parser.add_argument("--decode_select", type=int, default=0)

    # ── 剪枝参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--prune_temporal", type=float, default=0.0,
                        help="帧级时序去冗余阈值（cosine sim）。0 = 不启用。推荐 0.92-0.97")
    parser.add_argument("--prune_spatial", type=float, default=0.0,
                        help="空间 token 保留比例（0~1）。0 = 不启用。推荐 0.4-0.7")
    parser.add_argument("--prune_metric", type=str, default="norm",
                        choices=["norm", "var", "tome"],
                        help="空间剪枝重要性度量方式")

    # ── 加密参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--encrypt", action="store_true",
                        help="启用 KV cache 加密（encode 时加密，decode 时自动解密）")
    parser.add_argument("--key_file", type=str, default="../data/master.key",
                        help="master key 文件路径（首次运行自动生成）")

    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/haimian_7.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    #question      = "Who is the green toy character appearing in the video?"
    #question      = "What is the material of the spoon at the end of the video?"
    #question      = "How are you feeling after watching the video?"
    question      = "Who is the main character in the video?"
    encode_prefix = "Please understand the video and be ready to answer the single-choice question based on the video content. "

    # ── 构建 PruneContext（encode 侧使用）────────────────────────────────
    prune_ctx = None
    if args.prune and (args.prune_temporal > 0 or args.prune_spatial > 0):
        try:
            from video_token_prune_td import PruneContext
            prune_ctx = PruneContext(
                enabled=True,
                temporal_enabled=(args.prune_temporal > 0),
                temporal_threshold=args.prune_temporal if args.prune_temporal > 0 else 0.95,
                spatial_enabled=(args.prune_spatial > 0),
                spatial_keep_ratio=args.prune_spatial if args.prune_spatial > 0 else 0.5,
                spatial_metric=args.prune_metric,
                log_stats=True,
            )
            logger.info(
                f"[prune] Enabled: temporal={args.prune_temporal}, "
                f"spatial={args.prune_spatial} ({args.prune_metric})"
            )
        except ImportError:
            logger.warning("[prune] video_token_prune_td not found, pruning disabled.")

    # ── 构建 CryptoContext（encode/decode 双侧使用）───────────────────────
    crypto_ctx = None
    if args.encrypt:
        try:
            from kvcache_crypto_td import CryptoContext
            crypto_ctx = CryptoContext.from_key_file(args.key_file, create=True)
            logger.info(f"[crypto] Encryption enabled. Key file: {args.key_file}")
        except ImportError:
            logger.warning("[crypto] kvcache_crypto_td not found, encryption disabled.")
        except Exception as e:
            logger.warning(f"[crypto] Failed to load key: {e}. Encryption disabled.")

    with measure_resources(args.mode, logger=logger, plot_file=args.plot_file) as monitor:
        # ── 编码阶段 ──────────────────────────────────────────────────────────
        if args.mode in ("encode_decode", "encode"):

            monitor["mark"]("load_model_encode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])

            video = load_video(video_path, sample_fps=0.5)

            monitor["mark"]("kvcache_encode_start")
            manager = KVCacheManager(
                kv_cache_dir=kv_cache_path,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
                device="cpu",
            )
            encode_video(
                video=video,
                processor=processor,
                model=model,
                chunk_size=16,
                encode_prefix=encode_prefix,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                manager=manager,
                prune_ctx=prune_ctx,
                crypto_ctx=crypto_ctx,
            )
            monitor["mark"]("kvcache_encode_done")

            remove_timing_hooks_from_model()
            del model, manager
            gc.collect()
            time.sleep(10)

        # ── 解码阶段 ──────────────────────────────────────────────────────────
        if args.mode in ("encode_decode", "decode"):
            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model)

            if args.decode_select > 0:
                decode_chunk_ids = select_chunks(
                    kv_cache_path, question, processor, model, top_k=args.decode_select,
                )
                logger.info(f"Decoding with top-{args.decode_select} selected chunks.")
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
                decode_strategy="greedy",
                chunk_indices=decode_chunk_ids,
                suffix=None,
                crypto_ctx=crypto_ctx,  # ← 自动解密
            )

            print(f"model answer: {answer}")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

            # 解码完成后清理解密临时文件
            if crypto_ctx is not None:
                crypto_ctx.cleanup_tmp()