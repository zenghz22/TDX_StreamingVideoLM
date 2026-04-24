import os
import gc
import time
import argparse

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
    parser.add_argument("--decode_select", type=int, default=0)

    # ── 剪枝参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--prune", action="store_true",
                        help="启用视频 token 剪枝（encode 时启用，decode 时自动兼容）")
    parser.add_argument("--prune_temporal", type=float, default=0.0,
                        help="帧级时序去冗余：目标保留帧比例（0~1）。"
                             "0 = 不启用。0.6 = 每 chunk 保留约 60%% 的帧。"
                             "内部自动推算 cosine 阈值并在日志中打印。推荐 0.4~0.8。")
    parser.add_argument("--prune_spatial", type=float, default=0.0,
                        help="空间降分辨率：像素面积保留比例（0~1）。"
                             "0 = 不启用。0.5 = 面积缩小至 50%%（长宽各 ×√0.5≈0.707），保留宽高比。"
                             "推荐 0.3~0.7。")
    # 在目前的模型llava-onevision下，空间剪枝是无效的，因为分辨率不随着输入视频分辨率的变化而变化，而是固定的384*384

    # ── 加密参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--encrypt", action="store_true",
                        help="启用 KV cache 加密（encode 时加密，decode 时自动解密）")
    parser.add_argument("--key_file", type=str, default="../data/master.key",
                        help="master key 文件路径（首次运行自动生成）")

    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/haimian_7.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    question      = "Who is in the video, and what are they doing?"
    encode_prefix = "You are a helpful assistant. Please understand the video content and prepare to answer single-choice questions."

    # ── 构建 PruneContext（encode 侧使用）────────────────────────────────
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
        logger.info(
            f"[prune] Enabled: temporal_keep_ratio={args.prune_temporal}, "
            f"spatial_ratio={args.prune_spatial}"
        )

    # ── 构建 CryptoContext（encode/decode 双侧使用）───────────────────────
    crypto_ctx = None
    if args.encrypt:
        from kvcache_crypto_td import CryptoContext
        crypto_ctx = CryptoContext.from_key_file(args.key_file, create=True)
        logger.info(f"[crypto] Encryption enabled. Key file: {args.key_file}")

    with measure_resources(args.mode, logger=logger, plot_file=args.plot_file, plot_lable=True) as monitor:
        # ── 编码阶段 ──────────────────────────────────────────────────────────
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
            )
            monitor["mark"]("kvcache_encode_done")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        # ── 解码阶段 ──────────────────────────────────────────────────────────
        if args.mode in ("encode_decode", "decode"):
            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model)

            text_content = f"Question: {question}\nAnswer:"
            conversation_context = [{
                "role": "user",
                "content": [{
                    "type": "text", "text": text_content
                }],
            }]
            prompt = processor.apply_chat_template(
                conversation_context,
                add_generation_prompt = True,
                tokenize=False,
            )

            print("======================================================")
            print(f"decode_select：{args.decode_select}")
            print("添加chat_template后Prompt为：")
            print(prompt)

            if args.decode_select > 0:
                #decode_chunk_ids = select_chunks(
                decode_chunk_ids = select_chunks_per_layer(
                    kv_cache_path, 
                    question, 
                    processor, 
                    model, 
                    top_k=args.decode_select,
                    crypto_ctx=crypto_ctx,   # 方向D：retrieval_index 可能已加密
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
                decode_strategy="sample",
                #chunk_indices=decode_chunk_ids,
                suffix=prompt,
                crypto_ctx=crypto_ctx,  # ← 自动解密
                per_layer_chunk_indices = decode_chunk_ids,
            )

            print(f"model answer: {answer}")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)

        # 解码完成后清理解密临时文件
        if crypto_ctx is not None:
            crypto_ctx.cleanup_tmp()
