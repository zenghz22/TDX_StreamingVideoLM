import os
import gc
import time
import argparse

from kvcache_generate_td import ENCODE_PREFIX, load_model, load_video
from kvcache_manager_td import encode_video_managed
from kvcache_retrieve_td import decode_kvcache
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
    question      = "What is the chef doing?"

    parser = argparse.ArgumentParser(description="TD-side video encoding and decoding")
    parser.add_argument("--max_in_memory", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--plot_file", type=str, default=None)    
    args = parser.parse_args()

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
            #max_in_memory=MAX_IN_MEMORY,
            #window_size=WINDOW_SIZE,
            max_in_memory=args.max_in_memory,
            window_size=args.window_size if args.window_size > 0 else None,
        )
        monitor["mark"]("kvcache_encode_done")

        #logger.info(f"Encode finished. Manager stats: {manager.stats()}")

        remove_timing_hooks_from_model()
        del model, manager
        gc.collect()
        time.sleep(10)

    # ── 解码阶段（不变）──────────────────────────────────────────────
    monitor["mark"]("load_model_decode")
    processor, model = load_model(model_path, load_weights=True)
    inject_timing_hook_to_model(model)

    monitor["mark"]("kvcache_decode")
    answer = decode_kvcache(kv_cache_path, question, processor, model)

    print("Answer:", answer)

    remove_timing_hooks_from_model()
    del model
    gc.collect()