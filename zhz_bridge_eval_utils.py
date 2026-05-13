"""zhz_bridge_eval_utils.py

SHM 桥接 I/O 监控工具：TD 侧和 Host 侧共用的桥接性能采样器。

设计目标
--------
- 每次 op 拆三个阶段记录（wait_idle / data_copy / wait_response），
  方便定位瓶颈在数据搬运还是同步等待
- 提供 percentile（p50/p95/p99），不只是 min/avg/max
- 支持周期性进度日志，避免长 encode 静默
- 任务结束时自动 log_summary，并按 op 类型 (OFFLOAD/FETCH/FINALIZE) 分组
- 可选 JSONL 详细日志落盘，便于 offline 分析
- TD/Host 两端共用同一份数据格式，便于事后做差值对齐

典型用法
--------
TD 侧：

    from zhz_bridge_eval_utils import BridgeMonitor

    monitor = BridgeMonitor("encode_offload", side="td",
                            logger=logger, progress_every_n=50)
    ...
    with monitor.measure(op="OFFLOAD",
                         layer_index=L, frame_index=F,
                         payload_bytes=len(payload)) as rec:
        # ... 用户在 with 内填充 rec.t_wait_idle_us / t_data_copy_us / ...
        # 退出 with 时自动 record() 并打印进度（若到阈值）
    ...
    monitor.log_summary()   # 任务结束时手动打一次最终汇总
    monitor.close()

Host 侧用法相同，将 side 改为 "host"。
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 单次操作的多阶段计时记录
# ---------------------------------------------------------------------------

@dataclass
class BridgeOpRecord:
    """单次 bridge I/O 操作的计时数据。

    阶段语义（TD 侧）
        wait_idle      : 等待对端进入 IDLE，准备接收请求
        data_copy      : 把 payload + 控制字段写入 SHM 数据页 / 读取响应
        wait_response  : 等待对端处理完毕（这通常就是对端真正干活的时间）

    阶段语义（Host 侧）
        wait_idle      : 上一次 op 完成到本次 op 起手的轮询空转
        data_copy      : 从 SHM 数据页读 / 向其写、可能含磁盘 mmap I/O
        wait_response  : Host 侧记为 0，无意义
    """
    op: str = ""                         # "OFFLOAD" / "FETCH" / "FINALIZE"
    side: str = "td"                     # "td" / "host"
    layer_index: int = -1
    frame_index: int = -1
    payload_bytes: int = 0
    t_wait_idle_us: float = 0.0
    t_data_copy_us: float = 0.0
    t_wait_response_us: float = 0.0
    t_total_us: float = 0.0              # 端到端，包括以上三段
    error: bool = False
    error_msg: str = ""
    timestamp: float = 0.0               # epoch seconds


# ---------------------------------------------------------------------------
# Monitor 主体
# ---------------------------------------------------------------------------

class BridgeMonitor:
    """
    一次 encode / decode 过程中 bridge 操作的累积监控器。

    可以同时使用多个实例（如 encode_offload / decode_fetch），互不干扰。

    Parameters
    ----------
    name : str
        阶段名，出现在所有日志前缀里
    side : str
        "td" 或 "host"
    logger : Logger | None
        日志器；默认用模块 logger
    jsonl_path : str | None
        若提供，把每条 op record 序列化为一行 JSON 写入该路径
    progress_every_n : int
        每 N 条记录打一行进度（0 = 关闭）
    progress_every_sec : float
        每多少秒打一行进度（0 = 关闭）
    """

    def __init__(
        self,
        name: str,
        *,
        side: str = "td",
        logger: Optional[logging.Logger] = None,
        jsonl_path: Optional[str] = None,
        progress_every_n: int = 0,
        progress_every_sec: float = 0.0,
    ):
        self.name = name
        self.side = side
        self.logger = logger or logging.getLogger(__name__)
        self.records: List[BridgeOpRecord] = []
        self._by_op: Dict[str, List[BridgeOpRecord]] = {}
        self._by_layer: Dict[int, List[BridgeOpRecord]] = {}
        self.errors = 0

        self.progress_every_n = int(progress_every_n)
        self.progress_every_sec = float(progress_every_sec)
        self._t_start = time.time()
        self._t_last_progress = self._t_start

        # JSONL 详细日志（可选）
        self._jsonl_fh = None
        if jsonl_path:
            jsonl_dir = os.path.dirname(jsonl_path)
            if jsonl_dir:
                os.makedirs(jsonl_dir, exist_ok=True)
            try:
                self._jsonl_fh = open(jsonl_path, "w", encoding="utf-8")
                # 写一行 meta，便于后续解析时识别
                self._jsonl_fh.write(
                    json.dumps({
                        "_meta": True,
                        "monitor_name": name,
                        "side": side,
                        "started_at": self._t_start,
                    }, ensure_ascii=False) + "\n"
                )
                self._jsonl_fh.flush()
            except Exception as e:
                self.logger.warning(
                    f"[bridge/{name}] cannot open jsonl '{jsonl_path}': {e}"
                )
                self._jsonl_fh = None

        self.logger.info(
            f"[bridge/{self.name}/{self.side}] monitor started  "
            f"jsonl={jsonl_path or 'off'}  "
            f"progress_every_n={progress_every_n}  "
            f"progress_every_sec={progress_every_sec}"
        )

    # ------------------------------------------------------------------
    # 记录入口
    # ------------------------------------------------------------------

    def record(self, rec: BridgeOpRecord) -> None:
        """直接登记一条已填充字段的 BridgeOpRecord。"""
        if rec.timestamp == 0.0:
            rec.timestamp = time.time()
        self.records.append(rec)
        self._by_op.setdefault(rec.op, []).append(rec)
        if rec.layer_index >= 0:
            self._by_layer.setdefault(rec.layer_index, []).append(rec)
        if rec.error:
            self.errors += 1

        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.write(
                    json.dumps(asdict(rec), ensure_ascii=False) + "\n"
                )
                self._jsonl_fh.flush()
            except Exception:
                pass

        self._maybe_progress()

    @contextmanager
    def measure(
        self,
        op: str,
        *,
        layer_index: int = -1,
        frame_index: int = -1,
        payload_bytes: int = 0,
    ):
        """
        Context manager 模式。退出时自动设置 t_total_us（如果调用方没设置）
        并 record。即使 with 块抛异常，也会把 error 标记后再 raise。
        """
        rec = BridgeOpRecord(
            op=op,
            side=self.side,
            layer_index=int(layer_index),
            frame_index=int(frame_index),
            payload_bytes=int(payload_bytes),
        )
        t0 = time.perf_counter_ns()
        try:
            yield rec
        except Exception as e:
            rec.error = True
            rec.error_msg = str(e)[:200]
            if rec.t_total_us == 0.0:
                rec.t_total_us = (time.perf_counter_ns() - t0) / 1000.0
            self.record(rec)
            raise
        else:
            if rec.t_total_us == 0.0:
                rec.t_total_us = (time.perf_counter_ns() - t0) / 1000.0
            self.record(rec)

    # ------------------------------------------------------------------
    # 周期性进度
    # ------------------------------------------------------------------

    def _maybe_progress(self) -> None:
        triggered = False
        if self.progress_every_n > 0 and len(self.records) % self.progress_every_n == 0:
            triggered = True
        if self.progress_every_sec > 0:
            now = time.time()
            if now - self._t_last_progress >= self.progress_every_sec:
                triggered = True
                self._t_last_progress = now
        if triggered:
            self._log_progress()

    def _log_progress(self) -> None:
        if not self.records:
            return
        n = len(self.records)
        total_bytes = sum(r.payload_bytes for r in self.records)
        total_us = sum(r.t_total_us for r in self.records)
        bw_mbps = (total_bytes / (1024 ** 2)) / (total_us / 1e6) if total_us > 0 else 0.0
        elapsed = time.time() - self._t_start
        op_counts = " ".join(f"{op}={len(recs)}" for op, recs in sorted(self._by_op.items()))
        self.logger.info(
            f"[bridge/{self.name}/{self.side}] progress  "
            f"n={n} ({op_counts})  "
            f"cum={total_bytes/1024/1024:.2f}MB  "
            f"bw={bw_mbps:.2f}MB/s  "
            f"wall={elapsed:.1f}s  errors={self.errors}"
        )

    # ------------------------------------------------------------------
    # 汇总统计
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        """无依赖的简单插值法 percentile。values 非空才有意义。"""
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * q
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    def _stats(self, records: List[BridgeOpRecord]) -> Dict[str, Any]:
        """计算一组 records 的统计字典。"""
        if not records:
            return {"n": 0}
        bytes_list = [r.payload_bytes for r in records]
        lat_list = [r.t_total_us for r in records]
        total_bytes = sum(bytes_list)
        total_us = sum(lat_list)
        return {
            "n": len(records),
            "bytes_total_mb": round(total_bytes / (1024 ** 2), 3),
            "time_total_s": round(total_us / 1e6, 3),
            "bw_mbps": round(
                (total_bytes / (1024 ** 2)) / (total_us / 1e6)
                if total_us > 0 else 0.0, 2
            ),
            "size_min_kb": round(min(bytes_list) / 1024, 2),
            "size_avg_kb": round(sum(bytes_list) / len(bytes_list) / 1024, 2),
            "size_max_kb": round(max(bytes_list) / 1024, 2),
            "lat_min_ms": round(min(lat_list) / 1000, 3),
            "lat_avg_ms": round(sum(lat_list) / len(lat_list) / 1000, 3),
            "lat_p50_ms": round(self._percentile(lat_list, 0.50) / 1000, 3),
            "lat_p95_ms": round(self._percentile(lat_list, 0.95) / 1000, 3),
            "lat_p99_ms": round(self._percentile(lat_list, 0.99) / 1000, 3),
            "lat_max_ms": round(max(lat_list) / 1000, 3),
        }

    def _phase_stats(self, records: List[BridgeOpRecord]) -> Dict[str, float]:
        """阶段分解：wait_idle / data_copy / wait_response 的均值与 p95。"""
        if not records:
            return {}
        n = len(records)
        wait_idle = [r.t_wait_idle_us for r in records]
        data_copy = [r.t_data_copy_us for r in records]
        wait_resp = [r.t_wait_response_us for r in records]
        return {
            "wait_idle_avg_ms":     round(sum(wait_idle) / n / 1000, 3),
            "wait_idle_p95_ms":     round(self._percentile(wait_idle, 0.95) / 1000, 3),
            "data_copy_avg_ms":     round(sum(data_copy) / n / 1000, 3),
            "data_copy_p95_ms":     round(self._percentile(data_copy, 0.95) / 1000, 3),
            "wait_response_avg_ms": round(sum(wait_resp) / n / 1000, 3),
            "wait_response_p95_ms": round(self._percentile(wait_resp, 0.95) / 1000, 3),
        }

    def summary(self) -> Dict[str, Any]:
        """返回结构化汇总字典（可直接 json.dumps）。"""
        wall_s = time.time() - self._t_start
        out: Dict[str, Any] = {
            "name": self.name,
            "side": self.side,
            "wall_clock_s": round(wall_s, 3),
            "errors": self.errors,
            "overall": self._stats(self.records),
            "by_op": {op: self._stats(recs) for op, recs in sorted(self._by_op.items())},
            "phases_by_op": {op: self._phase_stats(recs) for op, recs in sorted(self._by_op.items())},
        }
        if self._by_layer:
            out["by_layer"] = {
                str(L): {
                    "n": len(recs),
                    "bytes_mb": round(sum(r.payload_bytes for r in recs) / (1024 ** 2), 3),
                    "lat_avg_ms": round(sum(r.t_total_us for r in recs) / len(recs) / 1000, 3),
                }
                for L, recs in sorted(self._by_layer.items())
            }
        return out

    def log_summary(self) -> None:
        """打印多行人类可读汇总（写入 self.logger 的 INFO 级别）。"""
        s = self.summary()
        tag = f"bridge/{self.name}/{self.side}"
        sep = "=" * 70

        log = self.logger.info
        log(sep)
        log(f"  Bridge Monitor Summary — [{tag}]")
        log(sep)
        log(
            f"  wall_clock={s['wall_clock_s']}s  "
            f"errors={s['errors']}  "
            f"total_ops={s['overall'].get('n', 0)}  "
            f"total_bytes={s['overall'].get('bytes_total_mb', 0)}MB"
        )

        # 按 op 类型分组打印吞吐/延迟
        log(f"  {'op':<10} {'n':>5} {'bw(MB/s)':>10} {'size(KB) min/avg/max':>24} {'lat(ms) p50/p95/p99/max':>26}")
        for op, st in s["by_op"].items():
            if st.get("n", 0) == 0:
                continue
            size_str = f"{st['size_min_kb']}/{st['size_avg_kb']}/{st['size_max_kb']}"
            lat_str  = f"{st['lat_p50_ms']}/{st['lat_p95_ms']}/{st['lat_p99_ms']}/{st['lat_max_ms']}"
            log(
                f"  {op:<10} {st['n']:>5d} {st['bw_mbps']:>10.2f} {size_str:>24} {lat_str:>26}"
            )

        # 阶段分解
        log("  Phase breakdown (avg / p95, milliseconds):")
        log(f"  {'op':<10} {'wait_idle':>20} {'data_copy':>20} {'wait_response':>20}")
        for op, ph in s["phases_by_op"].items():
            if not ph:
                continue
            wi = f"{ph['wait_idle_avg_ms']}/{ph['wait_idle_p95_ms']}"
            dc = f"{ph['data_copy_avg_ms']}/{ph['data_copy_p95_ms']}"
            wr = f"{ph['wait_response_avg_ms']}/{ph['wait_response_p95_ms']}"
            log(f"  {op:<10} {wi:>20} {dc:>20} {wr:>20}")

        # by_layer（layer 维度热度，便于发现热层）
        if "by_layer" in s and s["by_layer"]:
            log(f"  Per-layer hotness (top 5 by traffic):")
            ranked = sorted(s["by_layer"].items(),
                             key=lambda kv: kv[1]["bytes_mb"], reverse=True)[:5]
            for L, stat in ranked:
                log(
                    f"    L={L:>3}  n={stat['n']:>4d}  "
                    f"bytes={stat['bytes_mb']:>7.2f}MB  "
                    f"lat_avg={stat['lat_avg_ms']:>7.3f}ms"
                )

        log(sep)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.log_summary()
        finally:
            self.close()
        return False


# ---------------------------------------------------------------------------
# 便捷工厂：未启用时返回 no-op 监控器
# ---------------------------------------------------------------------------

class _DummyMonitor:
    """关闭监控时的占位实现，所有方法都是 no-op。"""

    name = "disabled"
    side = "off"
    records: List[BridgeOpRecord] = []
    errors = 0

    def record(self, rec):
        pass

    @contextmanager
    def measure(self, op, *, layer_index=-1, frame_index=-1, payload_bytes=0):
        yield BridgeOpRecord(
            op=op, layer_index=layer_index,
            frame_index=frame_index, payload_bytes=payload_bytes,
        )

    def summary(self):
        return {}

    def log_summary(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def make_monitor(
    name: str,
    *,
    side: str = "td",
    enabled: bool = True,
    **kwargs,
):
    """
    工厂函数。

    enabled=False 时返回 _DummyMonitor（接口完全兼容，0 开销）。
    便于在 bridge 模块里写：
        self._monitor = make_monitor("encode_offload", side="td",
                                      enabled=BRIDGE_MONITOR_ENABLED)
    然后无脑用 self._monitor.measure(...)。
    """
    if not enabled:
        return _DummyMonitor()
    return BridgeMonitor(name, side=side, **kwargs)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("bridge_monitor_selftest")

    monitor = BridgeMonitor(
        "selftest", side="td", logger=logger,
        progress_every_n=10,
    )

    # 模拟 30 个 OFFLOAD + 5 个 FETCH + 1 个 FINALIZE
    for i in range(30):
        with monitor.measure(op="OFFLOAD",
                              layer_index=i % 28,
                              frame_index=i,
                              payload_bytes=random.randint(50_000, 200_000)) as r:
            r.t_wait_idle_us     = random.uniform(50, 200)
            r.t_data_copy_us     = random.uniform(500, 2000)
            r.t_wait_response_us = random.uniform(1000, 8000)
            time.sleep(0.001)   # 模拟真实开销

    for i in range(5):
        with monitor.measure(op="FETCH",
                              layer_index=i, frame_index=i,
                              payload_bytes=random.randint(80_000, 150_000)) as r:
            r.t_wait_idle_us     = random.uniform(20, 100)
            r.t_data_copy_us     = random.uniform(200, 800)
            r.t_wait_response_us = random.uniform(3000, 15000)

    with monitor.measure(op="FINALIZE", payload_bytes=2000) as r:
        r.t_wait_idle_us = 30
        r.t_data_copy_us = 100
        r.t_wait_response_us = 5000

    monitor.log_summary()
    monitor.close()