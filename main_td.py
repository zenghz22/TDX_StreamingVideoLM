import os
import gc
import time

from kvcache_generate_td import ENCODE_PREFIX, encode_video, load_model, load_video
from kvcache_retrieve_td import decode_kvcache
from zhz_hardware_eval_utils import *
from zhz_model_eval_utils import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    log_system_info(logger)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path = "/home/zenghanzhang/tdx-streamvideo/data/muffin.mp4"
    kv_cache_path = "../data/kv_cache_chunks"
    question = "What is the chef doing?"  # 示例问题
    with measure_resources("Encode Video", logger=logger, plot_file = "device.png") as monitor:
        # 加载模型
        monitor["mark"]("load_model_encode")
        processor, model = load_model(model_path, load_weights=True)
        inject_timing_hook_to_model(model, event_callback=monitor["mark"])

        # 加载视频
        video = load_video(video_path, sample_fps=4)

        # kvcache编码与保存
        kv_cache = encode_video(
            video,
            processor,
            model=model,
            chunk_size=16,
            encode_prefix=ENCODE_PREFIX,
            stage_mark=monitor["mark"],
            kv_cache_dir=kv_cache_path,
        )

        del kv_cache
        #print_timing_stats(model)
        # 卸载模型
        remove_timing_hooks_from_model()
        del model
        gc.collect()
        time.sleep(10)

        ##################################################################
        # 加载模型
        monitor["mark"]("load_model_decode")
        processor, model = load_model(model_path, load_weights=True)
        inject_timing_hook_to_model(model)


        # kvcache提取与解码
        monitor["mark"]("kvcache_decode")
        answer = decode_kvcache(kv_cache_path, question, processor, model)

        print("Answer:", answer)
        #print_timing_stats(model)

        # 卸载模型
        remove_timing_hooks_from_model()
        del model
        gc.collect()
