import logging
import os
import sys

from kvcache_generate_td import ENCODE_PREFIX, encode_video, load_model, load_video, save_kv_cache
from kvcache_retrieve_td import decode_kvcache
from zhz_hardware_eval_utils import log_system_info, measure_resources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    log_system_info(logger)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path = "/home/zenghanzhang/tdx-streamvideo/data/muffin.mp4"
    kv_cache_path = "kv_cache.pt"
    question = "What is the chef doing?"

    processor, model = load_model(model_path, load_weights=True)
    print("Processor loaded successfully.")

    video = load_video(video_path, sample_fps=2)
    print("Video loaded successfully.")

    with measure_resources("Encode Video", logger=logger, plot_file="Encode Video.png") as monitor:
        kv_cache = encode_video(video, processor, model=model, encode_prefix=ENCODE_PREFIX)
        print("Preprocess done.")
        save_kv_cache(kv_cache, kv_cache_path, model=model, extra_metadata={"encode_prefix": ENCODE_PREFIX})

    with measure_resources("Decode KVcache", logger=logger, plot_file="Decode KVcache.png") as monitor:
        answer = decode_kvcache(kv_cache_path, question, processor, model)
        print("Answer:", answer)
