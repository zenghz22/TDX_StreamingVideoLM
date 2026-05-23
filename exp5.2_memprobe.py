"""mem_probe.py — 外部内存采样器，专为捕获 OOM 前的完整内存曲线。

独立进程启动并采样目标的 RSS（读 /proc/<pid>/status），每条样本即时 flush+fsync
落盘到 CSV。目标被内核 OOM-killer 杀掉（SIGKILL）后，本进程不受影响，CSV 仍保留
到死亡前的完整曲线 —— 解决"画图随目标一起被杀、图出不来"的问题。

用法
----
  # 把要监控的命令放在 -- 之后
  python mem_probe.py --out trace_1024.csv --interval 0.2 -- \
      python main_baseline_naive.py --num_frames 1024

  # 扫描多档（各自一个 CSV）
  for n in 256 512 1024 1536 2048; do
    python mem_probe.py --out trace_$n.csv -- \
        python main_baseline_naive.py --num_frames $n
  done

输出
----
  CSV: t_sec, rss_mb, hwm_mb   （hwm = 内核记录的峰值 RSS, VmHWM）
  退出时打印目标退出码与峰值；退出码为负=被信号杀（-9=SIGKILL，极可能 OOM）。
"""

import argparse
import csv
import os
import subprocess
import sys
import time


def read_rss_kb(pid: int):
    """从 /proc/<pid>/status 读 VmRSS 与 VmHWM（kB）。进程已死返回 (None,None)。"""
    try:
        rss = hwm = None
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
                if rss is not None and hwm is not None:
                    break
        return rss, hwm
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    ap.add_argument("--interval", type=float, default=0.2, help="采样间隔（秒）")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <要监控的命令...>")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: python mem_probe.py --out X.csv [--interval S] -- <command...>",
              file=sys.stderr)
        sys.exit(2)

    print(f"[mem_probe] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    peak_mb = 0.0

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "rss_mb", "hwm_mb"])
        fh.flush(); os.fsync(fh.fileno())

        while True:
            rc = proc.poll()
            rss_kb, hwm_kb = read_rss_kb(proc.pid)
            if rss_kb is not None:
                rss_mb = rss_kb / 1024.0
                hwm_mb = (hwm_kb if hwm_kb is not None else rss_kb) / 1024.0
                peak_mb = max(peak_mb, hwm_mb)
                w.writerow([f"{time.time() - t0:.3f}", f"{rss_mb:.1f}", f"{hwm_mb:.1f}"])
                fh.flush()
                os.fsync(fh.fileno())   # ← 关键：每条样本落盘，目标被 SIGKILL 也不丢
            if rc is not None:
                break                   # 目标已退出（正常结束 / 被杀）
            time.sleep(args.interval)

    rc = proc.returncode
    print(f"[mem_probe] target exit={rc}  peak_rss={peak_mb/1024:.2f} GB "
          f"({peak_mb:.0f} MB)  trace={args.out}")
    if rc is not None and rc < 0:
        sig = -rc
        print(f"[mem_probe] target killed by signal {sig} "
              f"({'SIGKILL=OOM-killer' if sig == 9 else 'signal'}). "
              f"确认: dmesg | grep -i 'killed process'")


if __name__ == "__main__":
    main()