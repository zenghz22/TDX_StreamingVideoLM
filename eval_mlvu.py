'''评测MLVU数据集在设定不同Window下的精确度'''
import os
import gc
import time
import argparse
import json
import csv
import re

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

    #log_system_info(logger)

    parser = argparse.ArgumentParser(description="TD-side video encoding and decoding")
    parser.add_argument("--mode", type=str, default="encode_decode",choices=["encode_decode","decode","encode"])
    parser.add_argument("--plot_file", type=str, default=None)
    parser.add_argument("--encode_memory", type=int, default=64)
    parser.add_argument("--encode_window", type=int, default=0)
    parser.add_argument("--decode_select", type=str, default="0",
                        help="逗号分隔的 decode_select 值列表，如 '0,2,4,8,12'")
    parser.add_argument("--anon_index",type=int, default=0)
    args = parser.parse_args()
    # 解析 decode_select_list
    decode_select_list = [int(x) for x in args.decode_select.split(",")]

    #annotation_path = "/home/zenghanzhang/STC/data/mlvu_zhz_sample5/mlvu_zhz_sample5.json"
    annotation_path = "/home/zenghanzhang/STC/data/mlvu_zhz/mlvu_zhz.json"
    # 加载annotation文件，索引其中的第args.anno_index项字典
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    anon = annotations[args.anon_index]

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    kv_cache_path = "../data/kv_cache_chunks"

    video_id = anon["video_id"]
    video_path = anon["video_path"]
    duration = anon["duration"]
    encode_prefix = "Please understand the video and be ready to answer the single-choice question based on the video content. " 
    
    # ── 编码阶段 ──────────────────────────────────────────────────────
    if args.mode == "encode_decode" or args.mode == "encode":
        with measure_resources("Encode", logger=logger, plot_file=args.plot_file) as monitor:
            #'''
            processor, model = load_model(model_path, load_weights=True)
            inject_timing_hook_to_model(model, event_callback=monitor["mark"])
            
            video = load_video(video_path, sample_fps=1)
            monitor["mark"]("kvcache_encode_start")
            manager = KVCacheManager(
                kv_cache_dir = kv_cache_path,
                max_in_memory = args.encode_memory,
                window_size = args.encode_window if args.encode_window > 0 else None,
                device = "cpu",
            )
            encode_video(
                video = video,
                processor = processor,
                model = model,
                chunk_size = 16,
                encode_prefix=encode_prefix,
                stage_mark=monitor["mark"],
                kv_cache_dir=kv_cache_path,
                manager=manager,
            )
            monitor["mark"]("kvcache_encode_done")

            remove_timing_hooks_from_model()
            del model, manager
            gc.collect()
            time.sleep(10)
            #'''

    # —— 解码阶段 ──────────────────────────────────────────────────────
    # 模型只加载一次，在所有 decode_select 设置和所有 conversation 之间共享。
    # 每次 decode_kvcache 调用结束后 past_key_values 是局部变量自然释放；
    # 每个 decode_select 设置之间只需 reset hooks + gc，无需重载模型。
    if args.mode == "encode_decode" or args.mode == "decode":
        with measure_resources("Decode", logger=logger, plot_file=args.plot_file) as monitor:

            monitor["mark"]("load_model_decode")
            processor, model = load_model(model_path, load_weights=True)

            for decode_select in decode_select_list:

                # ---- 每个 decode_select 开始：注入 hooks ----
                inject_timing_hook_to_model(model)
                logger.info(f"===== decode_select={decode_select} =====")

                for conversation in anon["conversations"]:
                    question = conversation["question"]
                    choices = conversation["choices"]
                    choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
                    formatted_choices = "\n".join([
                        f"({choice_letters[i]}) {choice}"
                        for i, choice in enumerate(choices)
                    ])

                    correct_answer = conversation["answer"]
                    correct_letter = choice_letters[choices.index(correct_answer)]
                    question_type = conversation["question_type"]
                    prompt = f"Question:{question}\nOptions:{formatted_choices}\nOnly give the best option with one letter.\nAnswer:"

                    if decode_select > 0:
                        top_k = decode_select
                        decode_chunk_ids = select_chunks(
                            kv_cache_path,
                            question,
                            processor,
                            model,
                            top_k=top_k,
                        )
                        logger.info(f"Decoding with top-{top_k} selected chunks.")
                    else:
                        decode_chunk_ids = None
                        logger.info("Decoding with all chunks.")

                    monitor["mark"]("kvcache_decode")
                    model_answer = decode_kvcache(
                        kv_cache_path,
                        prompt,
                        processor,
                        model,
                        max_new_tokens=8,
                        min_new_tokens=1,
                        temperature=0.0,
                        decode_strategy="sample",
                        chunk_indices=decode_chunk_ids,
                        suffix=None,
                    )
                    print(f"model answer: {model_answer}")
                    print(choice_letters)
                    print(choices)

                    model_answer_stripped = model_answer.strip()
                    if model_answer_stripped.lower().startswith("assistant"):
                        model_answer_stripped = model_answer_stripped[len("assistant"):].strip()
                    model_answer_stripped = model_answer_stripped.replace("\n", "").strip()
                    letter_match = re.search(r'\b([A-H])\b', model_answer_stripped)
                    model_letter = letter_match.group(1) if letter_match else None
                    is_correct = (model_letter == correct_letter)

                    result = {
                        "anon_index": args.anon_index,
                        "encode_window": args.encode_window,
                        "decode_select": decode_select,
                        "video_id": video_id,
                        "question": question,
                        "choices": choices,
                        "model_answer": model_answer_stripped,
                        "correct_letter": correct_letter,
                        "decode_chunk_indices": decode_chunk_ids if decode_chunk_ids is not None else "all",
                        "is_correct": is_correct
                    }

                    csv_file_path = f"../results/csv/mlvu_results_W{args.encode_window}_S{decode_select}.csv"
                    with open(csv_file_path, "a", newline='') as csvfile:
                        fieldnames = ["anon_index", "encode_window", "decode_select", "video_id", "question", "choices", "model_answer", "correct_letter", "decode_chunk_indices", "is_correct"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if csvfile.tell() == 0:
                            writer.writeheader()
                        writer.writerow(result)

                # ---- 每个 decode_select 结束：移除 hooks，释放 decode 残留内存 ----
                remove_timing_hooks_from_model()
                gc.collect()
                logger.info(f"===== decode_select={decode_select} done, hooks cleared =====")

            # 所有 setting 跑完后再卸载模型
            del model
            gc.collect()
            time.sleep(5)