"""main_baseline_naive.py — 实验 5.2 对照组：朴素全驻留基线（naive full-residence）

目的
----
不使用编码-解码解耦、滑动窗口、外存化、桥接、加密，直接用标准 HuggingFace
推理路径，把整段视频 + 问题一次性喂入 model.generate，use_cache=True 使
全部 KV 常驻 TD 内存。随视频帧数增长，峰值常驻内存线性上升，长视频触发 OOM。

这是 5.2 节"内存可行性"的对照组：与本工作（编码-解码解耦 + 立即下沉，常驻平稳）
形成对比，证明"不采用我们的方法就会 OOM"。

用法（按帧数扫描，定位 OOM 临界点）
----------------------------------
    python main_baseline_naive.py --num_frames 256  --plot_file baseline_256.json
    python main_baseline_naive.py --num_frames 512  --plot_file baseline_512.json
    python main_baseline_naive.py --num_frames 1024 --plot_file baseline_1024.json
    ...
每个帧数单独一个进程跑。成功跑完的最大帧数 = 可行性前沿；再大一档即 OOM。
注意：要真正触发 OOM，请用足够长的视频（如 30min+ 的 MLVU 长视频），
      使 num_frames 能达到内存临界点（按 5.2.1 估算约 700~1800 帧，取决于实测每帧驻留）。
"""

import os
import gc
import argparse
import logging
import sys

import torch

from kvcache_generate_td import load_model, load_video, VIDEO_PLACEHOLDER
from zhz_hardware_eval_utils import *   # measure_resources

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5.2 control group: naive full-residence baseline (no offload/window/bridge)."
    )
    parser.add_argument("--sample_fps", type=float, default=1,
                        help="抽帧率（与本工作实验保持一致）。")
    parser.add_argument("--num_frames", type=int, default=64,
                        help="截断到前 N 帧（0=全部）。用于扫描 OOM 临界点。")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--plot_file", type=str, default="/home/tdx/tee/tdx-streamvideo/results/plots/exp5.2_baseline.png",
                        help="hardware monitor 输出（每个帧数用不同文件名，便于汇总曲线）。")
    args = parser.parse_args()

    model_path    = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    video_path    = "../data/needle_4.mp4"   # 真正测 OOM 时换成长视频
    question      = "What is the man in the black silhouette wearing on the lake shore?"
    encode_prefix = ("You are a helpful assistant. Please understand the video "
                     "content and prepare to answer single-choice questions.")

    # ── 全程挂载 hardware monitor（峰值 RSS 等），label 用帧数便于汇总 ──
    with measure_resources("baseline_naive", logger=logger,
                           plot_file=args.plot_file, plot_lable=False) as monitor:

        # 1) 加载模型（约 14GB bf16 权重常驻）
        monitor["mark"]("load_model")
        processor, model = load_model(model_path, load_weights=True)
        device = next(model.parameters()).device

        # 2) 加载视频；可截断帧数以扫描临界点。load_video 返回 numpy (T,H,W,C)
        monitor["mark"]("load_video")
        video = load_video(video_path, sample_fps=args.sample_fps)
        if args.num_frames > 0:
            video = video[:args.num_frames]
        n_frames = int(video.shape[0])
        logger.info(f"[baseline] frames={n_frames}  (naive full-residence: no offload/window/bridge/crypto)")

        # 3) 整段视频 + 问题，标准推理，全 KV 常驻
        #    文本用 <video> 占位符 + 问题，与 encode_video 里能跑通的处理器调用一致。
        #    （答案质量不是本对照组关注点，5.2 关注的是内存；若要可读答案可改用
        #     processor.apply_chat_template 构造带角色标记的 prompt。）
        text = f"{VIDEO_PLACEHOLDER}\n{encode_prefix}\nQuestion: {question}\nAnswer:"
        model_inputs = processor(text=[text], videos=[video], return_tensors="pt")
        model_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in model_inputs.items()
        }

        monitor["mark"]("prefill_generate_start")
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,    # 贪心，确定性
                    use_cache=True,     # ← 关键：全 KV 常驻，这就是"朴素全驻留"
                )
            in_len = model_inputs["input_ids"].shape[1]
            try:
                answer = processor.batch_decode(
                    output_ids[:, in_len:], skip_special_tokens=True
                )[0]
            except Exception:
                answer = processor.tokenizer.decode(
                    output_ids[0][in_len:], skip_special_tokens=True
                )
            monitor["mark"]("generate_done")
            logger.info(f"[baseline] frames={n_frames}  OK  answer={answer!r}")

        except (MemoryError, RuntimeError) as e:
            # 可被 Python 捕获的 OOM/分配失败（CPU 上常见 bad_alloc / cannot allocate）
            monitor["mark"]("OOM")
            logger.error(f"[baseline] frames={n_frames}  OOM/RuntimeError: {e}")
            # 注意：若被内核 OOM-killer 直接 SIGKILL，进程会硬退出，本异常无法捕获，
            #       届时请用 `dmesg | grep -i oom` 与 monitor 的连续采样日志观察峰值。

        del model
        gc.collect()