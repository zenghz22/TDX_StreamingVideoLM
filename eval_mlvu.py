'''评测MLVU数据集在设定不同Window下的精确度'''
import os
import gc
import time
import argparse
import json
import csv
import re

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
    parser.add_argument("--anon_index",type=int, default=0)
    args = parser.parse_args()

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
            video = load_video(video_path, sample_fps=1)
            
            manager = encode_video_managed(
                video,
                processor,
                model=model,
                chunk_size=16,
                encode_prefix=encode_prefix,
                kv_cache_dir=kv_cache_path,
                max_in_memory=args.encode_memory,
                window_size=args.encode_window if args.encode_window > 0 else None,
            )
            del model, manager
            gc.collect()
            time.sleep(10)
            #'''

    # —— 解码阶段 ──────────────────────────────────────────────────────
    if args.mode == "encode_decode" or args.mode == "decode":
        with measure_resources("Decode", logger=logger, plot_file=args.plot_file) as monitor:
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

                processor, model = load_model(model_path, load_weights=True)
            
                # 三种解码模式：指定chunk indices、select top-k chunk、使用全部chunk
                #格式 "full" "select{数量}" "{指定chunk编号(用,分隔)}"

                if args.decode_select > 0:
                    top_k = args.decode_select
                    decode_chunk_ids = select_chunks(
                        kv_cache_path, 
                        question, 
                        processor, 
                        model, 
                        top_k=top_k, 
                        always_include_first=False)
                    logger.info(f"Decoding with top-{top_k} selected chunks based on question relevance.")
                elif args.decode_select == 0:
                    decode_chunk_ids = None  # None 表示加载全部 chunk
                    logger.info("Decoding with all chunks.")

                model_answer = decode_kvcache(
                    kv_cache_path, 
                    prompt, 
                    processor, 
                    model,
                    max_new_tokens=8,
                    min_new_tokens=8,
                    temperature=0.0,
                    decode_strategy="sample",
                    chunk_indices=decode_chunk_ids,
                    suffix=prompt,
                    )
                print(f"model answer: {model_answer}")
                print(choice_letters)
                print(choices)

                # 在csv文件中记录问题、模型答案、正确答案、是否正确、使用的chunk indices
                model_answer_stripped = model_answer.strip()
                # 去除前缀"assistant\n"
                if model_answer_stripped.lower().startswith("assistant"):
                    model_answer_stripped = model_answer_stripped[len("assistant"):].strip()    
                # 去除换行符号
                model_answer_stripped = model_answer_stripped.replace("\n", "").strip()
                # 从答案中提取首个大写字母（模型可能输出 "(A)" 或 "A" 或 "Answer: A"）
                letter_match = re.search(r'\b([A-H])\b', model_answer_stripped)
                model_letter = letter_match.group(1) if letter_match else None
                is_correct = (model_letter == correct_letter)

                # 在csv文件中记录问题、模型答案、正确答案、是否正确、使用的chunk indices
                result = {
                    "anon_index": args.anon_index,
                    "video_id": video_id,
                    "question": question,
                    "choices": choices,
                    "model_answer": model_answer_stripped,
                    "correct_letter": correct_letter,
                    "decode_chunk_indices": decode_chunk_ids if decode_chunk_ids is not None else "all",
                    "is_correct": is_correct
                }
                with open("../data/mlvu_results.csv", "a", newline='') as csvfile:
                    fieldnames = ["anon_index", "video_id", "question", "choices", "model_answer", "correct_letter","decode_chunk_indices", "is_correct" ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if csvfile.tell() == 0:  # 如果文件是空的，写入表头
                        writer.writeheader()
                    writer.writerow(result)

            remove_timing_hooks_from_model()
            del model
            gc.collect()
            time.sleep(10)