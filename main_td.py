import os
import gc
import time
import argparse

from kvcache_generate_td import load_model, load_video
from kvcache_manager_td import encode_video_managed
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

    #log_system_info(logger)

    parser = argparse.ArgumentParser(description="TD-side video encoding and decoding")
    parser.add_argument("--mode", type=str, default="encode_decode",choices=["encode_decode","decode","encode"])
    parser.add_argument("--plot_file", type=str, default=None)
    parser.add_argument("--encode_memory", type=int, default=64)
    parser.add_argument("--encode_window", type=int, default=0)
    parser.add_argument("--decode_select", type=int, default=0)
    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/haimian_7.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    #question      = "Who is the green toy character appearing in the video?"
    #question      = "What is the tool used to extract the yolk from the egg?"
    #question      = "What is the material of the spoon at the end of the video?"
    question      = "How are you feeling after watching the video?"
    encode_prefix = "Please understand the video and be ready to answer the single-choice question based on the video content. "

    if args.mode == "encode_decode" or args.mode == "encode":
        with measure_resources("Encode Video", logger=logger, plot_file=args.plot_file) as monitor:
            # ── 编码阶段 ──────────────────────────────────────────────────────
            monitor["mark"]("load_model_encode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])

            video = load_video(video_path, sample_fps=1)

            monitor["mark"]("kvcache_encode_start")
            manager = encode_video_managed(
                video,
                processor,
                model=model,
                chunk_size=16,
                encode_prefix=encode_prefix,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
            )
            monitor["mark"]("kvcache_encode_done")


            remove_timing_hooks_from_model()
            del model, manager
            gc.collect()
            time.sleep(10)

    if args.mode == "encode_decode" or args.mode == "decode":
        with measure_resources("Decode Video", logger=logger, plot_file=args.plot_file) as monitor:
        # —— 解码阶段 ──────────────────────────────────────────────────────
            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model)

            if args.decode_select > 0:
                top_k = args.decode_select
                decode_chunk_ids = select_chunks(
                    kv_cache_path, 
                    question, 
                    processor, 
                    model, 
                    top_k=top_k, 
                )
                logger.info(f"Decoding with top-{top_k} selected chunks based on question relevance.")
            elif args.decode_select == 0:
                decode_chunk_ids = None  # None 表示加载全部 chunk
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
                chunk_indices=decode_chunk_ids,
                suffix = None
                )

            print(f"model answer: {answer}")

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)