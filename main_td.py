import os
import gc
import time
import argparse

from kvcache_generate_td import ENCODE_PREFIX, load_model, load_video
from kvcache_manager_td import encode_video_managed
from kvcache_retrieve_td import decode_kvcache
from kvcache_select_td import select_chunks
from zhz_hardware_eval_utils import *
from zhz_model_eval_utils import *

# ── 超参数配置 ────────────────────────────────────────────────────────────
# max_in_memory: 内存中最多保留的 chunk KV 数。
#   全量 attention 模式下，forward 时内存 ∝ i，该值只控制 forward 间隙期。
#   sliding window 模式下，forward 时内存 ≤ max_in_memory，严格界定。
#
# window_size: sliding window 大小（以 chunk 为单位）。
#   None  → 全量 attention（精度最高，内存无界）
#   W > 0 → 局部 attention（精度略降，内存严格有界）
#   推荐：max_in_memory = window_size + 1，给 delta 留一个空位。

# select_chunks: decode 时是否先选 chunk。
#  0 → 不选，加载全部 chunk（原有行为）
#  K > 0 → 选 top-K 个 chunk，峰值内存随 K 线性降低

# decode 阶段：指定只加载哪些 chunk 的 KV
# None  → 加载全部 chunk（原有行为）
# [...] → 只加载指定 chunk，峰值内存随列表长度线性降低
# 示例：DECODE_CHUNK_INDICES = [10, 11, 12, 13, 14]  # 只看视频后半段


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)




if __name__ == "__main__":

    log_system_info(logger)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/muffin.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    question      = "Who is the green toy character appearing in the video?"
    #question      = "How many eggs are there in the video?"
    #question      = "What is the tool used to extract the yolk from the egg?"

    parser = argparse.ArgumentParser(description="TD-side video encoding and decoding")
    parser.add_argument("--mode", type=str, default="encode_decode",choices=["encode_decode","decode","encode"])
    parser.add_argument("--plot_file", type=str, default=None)
    parser.add_argument("--encode_memory", type=int, default=64)
    parser.add_argument("--encode_window", type=int, default=0)
    parser.add_argument("--decode_indices", type=str, default="full")
    parser.add_argument("--decode_select", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "encode_decode" or args.mode == "encode":
        with measure_resources("Encode Video", logger=logger, plot_file=args.plot_file) as monitor:
            # ── 编码阶段 ──────────────────────────────────────────────────────
            monitor["mark"]("load_model_encode")
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])

            video = load_video(video_path, sample_fps=30)

            monitor["mark"]("kvcache_encode_start")
            manager = encode_video_managed(
                video,
                processor,
                model=model,
                chunk_size=16,
                encode_prefix=ENCODE_PREFIX,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
            )
            monitor["mark"]("kvcache_encode_done")

            #logger.info(f"Encode finished. Manager stats: {manager.stats()}")

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

            if args.decode_indices != "full":
                decode_chunk_ids = [int(idx) for idx in args.decode_indices.split(",")]
                logger.info(f"Decoding with specified chunk indices: {decode_chunk_ids}")

            elif args.decode_select > 0:
                monitor["mark"]("kvcache_select")
                decode_chunk_ids = select_chunks(kv_cache_path, question, processor, model, top_k=args.decode_select, always_include_first=True)
                logger.info(f"Decoding with top-{args.decode_select} selected chunks based on question relevance.")
            else:
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
                decode_strategy="greedy",
                chunk_indices=decode_chunk_ids,
                )

            print("Answer:", answer)

            remove_timing_hooks_from_model()
            del model
            gc.collect()
