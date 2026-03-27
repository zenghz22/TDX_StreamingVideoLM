import sys
import logging
import os
import threading
import time
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib.ticker import MaxNLocator


class ResourceMonitor:
    """资源监控器，用于测量 CPU 和内存使用情况。"""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process()
        self.cpu_percentages = []
        self.timestamps = []
        self.memory_usages = []
        self.start_time = None
        self.end_time = None
        self.initial_memory = None
        self.final_memory = None
        self.monitoring = False

    def start(self):
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.cpu_percentages = []
        self.timestamps = []
        self.memory_usages = []
        self.monitoring = True
        self.events = []

        self.timestamps.append(0.0)
        self.cpu_percentages.append(0.0)
        self.memory_usages.append(self.initial_memory)

        self.process.cpu_percent(interval=None)

    def stop(self):
        if not self.monitoring:
            return

        self.end_time = time.time()
        self.final_memory = self.process.memory_info().rss / 1024 / 1024
        self.monitoring = False

    def sample(self):
        if self.monitoring:
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            self.cpu_percentages.append(self.process.cpu_percent(interval=None))
            self.memory_usages.append(self.process.memory_info().rss / 1024 / 1024)

    def mark_event(self, label, payload=None):
        if not self.monitoring:
            return
        if payload is None:
            payload = {}
        t = time.time() - self.start_time
        self.events.append({"t": t, "label": str(label), "payload": payload})

    def get_results(self):
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("监控尚未开始或已停止")

        duration = self.end_time - self.start_time
        avg_cpu = np.mean(self.cpu_percentages) if self.cpu_percentages else 0
        max_cpu = np.max(self.cpu_percentages) if self.cpu_percentages else 0
        memory_used = self.final_memory - self.initial_memory

        return {
            "duration": duration,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "memory_initial_mb": self.initial_memory,
            "memory_final_mb": self.final_memory,
            "memory_used_mb": memory_used,
            "timestamps": self.timestamps,
            "cpu_percentages": self.cpu_percentages,
            "memory_usages": self.memory_usages,
            "events": self.events,
        }


@contextmanager
def measure_resources(name="Task", interval=0.1, logger=None, plot_file=None):
    """测量代码块的资源使用情况。"""
    if logger is None:
        logger = logging.getLogger(__name__)

    monitor = ResourceMonitor(interval)
    monitor.start()

    def sampling_task():
        while monitor.monitoring:
            monitor.sample()
            time.sleep(interval)

    sampling_thread = threading.Thread(target=sampling_task)
    sampling_thread.daemon = True
    sampling_thread.start()

    results = {"monitor": monitor, "stats": None, "mark": monitor.mark_event}

    try:
        yield results
    finally:
        monitor.stop()
        sampling_thread.join(timeout=0.5)
        results["stats"] = monitor.get_results()

        logger.info(f"--- {name} Resource Usage Statistics ---")
        logger.info(f"Execution time: {results['stats']['duration']:.2f} seconds")
        logger.info(f"Average CPU usage: {results['stats']['avg_cpu_percent']:.2f}%")
        logger.info(f"Peak CPU usage: {results['stats']['max_cpu_percent']:.2f}%")
        logger.info(f"Initial memory: {results['stats']['memory_initial_mb']:.2f} MB")
        logger.info(f"Final memory: {results['stats']['memory_final_mb']:.2f} MB")
        logger.info(f"Memory increase: {results['stats']['memory_used_mb']:.2f} MB")

        if plot_file is not None:
            if plot_file is True:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c for c in name if c.isalnum() or c in ["_", "-"])
                plot_file = f"resource_usage_{safe_name}_{timestamp}.png"

            plot_resource_usage(results["stats"], plot_file, name)
            logger.info(f"Resource usage chart saved to: {plot_file}")


def plot_resource_usage(stats, output_file, task_name="Task"):
    """绘制资源使用曲线图（仅 memory 图）。"""
    matplotlib.use("Agg")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 仅绘制内存曲线
    ax.plot(stats["timestamps"], stats["memory_usages"], "r-", linewidth=2)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Memory Usage (MB)", color="r", fontsize=12)
    ax.set_title(f"{task_name} - Memory Usage", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="y", labelcolor="r")

    # 将 CPU + Memory 统计信息统一放在 memory 图中
    summary_text = (
        f"Execution time: {stats['duration']:.2f} seconds\n"
        f"Avg CPU: {stats['avg_cpu_percent']:.2f}%\n"
        f"Peak CPU: {stats['max_cpu_percent']:.2f}%\n"
        f"Initial memory: {stats['memory_initial_mb']:.2f} MB\n"
        f"Final memory: {stats['memory_final_mb']:.2f} MB\n"
        f"Memory increase: {stats['memory_used_mb']:.2f} MB"
    )
    ax.text(
        0.66,
        0.12,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#d62728"),
    )

    # 将所有阶段标线和标签都绘制在 memory 图中
    events = stats.get("events", [])
    for idx, event in enumerate(events):
        t = event.get("t", 0.0)
        label = event.get("label", "event")
        color = "#2cc411"
        if "prefill" in label:
            color = "#ab23cd"
        if "visual" in label:
            color = "#EDBE33"
        if "load" in label:
            color = "#b01318"
        y = 0.98
        if  "prefill" in label:
            y = 0.65
        if  "visual" in label:
            y = 0.3
        ax.axvline(t, color=color, linestyle=":", linewidth=3, alpha=0.8)

        ax.text(
            t,
            ax.get_ylim()[1] * y,
            label,
            rotation=90,
            fontsize=15,
            va="top",
            ha="right",
            color=color,
            alpha=0.9,
        )

    system_info = get_system_info()
    system_text = (
        f"System: {system_info['cpu_count']} logical cores | "
        f"CPU freq: {system_info['cpu_freq_min']:.0f}-{system_info['cpu_freq_max']:.0f} MHz | "
        f"Total memory: {system_info['total_memory_mb']:.0f} MB"
    )
    plt.figtext(
        0.5,
        0.01,
        system_text,
        ha="center",
        fontsize=15,
        bbox=dict(facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_system_info():
    """获取系统基本信息。"""
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    memory = psutil.virtual_memory()

    return {
        "cpu_count": cpu_count,
        "cpu_freq_min": cpu_freq.min if cpu_freq else 0,
        "cpu_freq_max": cpu_freq.max if cpu_freq else 0,
        "total_memory_mb": memory.total / 1024 / 1024,
    }


def log_system_info(logger=None):
    """记录系统信息。"""
    if logger is None:
        logger = logging.getLogger(__name__)

    system_info = get_system_info()
    logger.info(
        f"System info: {system_info['cpu_count']} logical cores, "
        f"CPU frequency range: {system_info['cpu_freq_min']:.2f}-{system_info['cpu_freq_max']:.2f} MHz, "
        f"Total memory: {system_info['total_memory_mb']:.2f} MB"
    )