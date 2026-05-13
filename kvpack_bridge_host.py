"""kvpack_bridge_host.py
宿主进程（Host 侧）。

支持双向操作：
  OFFLOAD  (encode)：TD 推送 block 字节 → 宿主追加写入 kvpack.bin
  FETCH    (decode)：TD 请求 (layer, frame) → 宿主从 kvpack.bin 读字节返回

共享内存布局（总大小 = PAGESIZE + MAX_BLOCK_BYTES）：
  控制页：
    [0]     status      uint8
              0 = IDLE
              1 = OFFLOAD_READY   TD 已将 block 写入数据页，请求宿主存盘
              2 = HOST_BUSY
              3 = FETCH_READY     TD 请求读取 (layer, frame)
              4 = RESPONSE_READY  宿主已将响应写入数据页
              5 = FINALIZE_READY  TD 请求宿主写 kvpack_index.json（encode 结束）
              6 = SHUTDOWN        TD 请求宿主退出
    [1:5]   frame_index int32 LE
    [5:9]   layer_index int32 LE
    [9:17]  data_length uint64 LE  (OFFLOAD 时=block字节数, FETCH 时=响应字节数)
    [17:21] error_flag  uint32 LE
    [21:PAGESIZE] padding
  数据页：
    [0:data_length] block 字节（密文）

监控
----
Host 进程使用 zhz_bridge_eval_utils.BridgeMonitor 记录每次处理的 op，
进程退出时（SIGINT / SIGTERM / SHUTDOWN）自动打印汇总。

阶段语义（host 侧与 TD 侧含义略不同）
  wait_idle      : 上次 op 完成后空转 polling 的时长（busy-wait 时间）
  data_copy      : SHM 数据页读 / 写 + 磁盘 I/O 的总耗时
  wait_response  : 在 host 侧无意义，固定为 0

环境变量与 TD 侧共用：
  BRIDGE_MONITOR_JSONL_DIR : 若设置，写 ${dir}/host_<op_class>_host.jsonl
  BRIDGE_MONITOR_DISABLE   : "1" 关闭监控
"""

from __future__ import annotations

import argparse
import json
import logging
import mmap
import os
import signal
import struct
import sys
import time

from zhz_bridge_eval_utils import make_monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kvpack_bridge_host")


# ── 协议常量 ────────────────────────────────────────────────────────────────
SHM_FILE_PATH   = "/dev/shm/kvbridge_shmem"
PAGESIZE        = os.sysconf("SC_PAGE_SIZE")

OFF_STATUS      = 0
OFF_FRAME       = 1
OFF_LAYER       = 5
OFF_DATA_LEN    = 9
OFF_ERR_FLAG    = 17

STATUS_IDLE             = 0
STATUS_OFFLOAD_READY    = 1   # encode: TD 推数据
STATUS_HOST_BUSY        = 2
STATUS_FETCH_READY      = 3   # decode: TD 请求数据
STATUS_RESPONSE_READY   = 4   # decode: 宿主返回数据
STATUS_FINALIZE_READY   = 5   # encode 结束: TD 请求写 index
STATUS_SHUTDOWN         = 6

MAX_BLOCK_BYTES = 4 * 1024 * 1024
SHM_TOTAL_SIZE  = PAGESIZE + MAX_BLOCK_BYTES

FMT_FRAME   = struct.Struct("<i")
FMT_LAYER   = struct.Struct("<i")
FMT_DATALEN = struct.Struct("<Q")
FMT_ERR     = struct.Struct("<I")

# ── 监控开关（与 TD 侧同一组 env） ─────────────────────────────────────────
_MONITOR_ENABLED = os.environ.get("BRIDGE_MONITOR_DISABLE", "0") != "1"
_MONITOR_JSONL_DIR = os.environ.get("BRIDGE_MONITOR_JSONL_DIR", "").strip() or None


def _make_jsonl_path(name: str) -> str | None:
    if not _MONITOR_JSONL_DIR:
        return None
    return os.path.join(_MONITOR_JSONL_DIR, f"{name}_host.jsonl")


def _try_prepare_fetch_from_disk(kv_dir: str):
    """
    尝试从现有 kvpack.bin + kvpack_index.json 恢复 fetch 服务，
    使 decode-only 场景无需依赖本次进程内 encode/finalize。
    """
    bin_path = os.path.join(kv_dir, "kvpack.bin")
    index_path = os.path.join(kv_dir, "kvpack_index.json")
    if not (os.path.exists(bin_path) and os.path.exists(index_path)):
        return None, None, {}

    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("blocks", [])
    block_index = {}
    for rec in records:
        block_index[(int(rec["layer_index"]), int(rec["frame_index"]))] = rec

    bin_size = os.path.getsize(bin_path)
    if bin_size <= 0:
        logger.info(
            f"[host] preload skipped: {bin_path} is empty (size=0). "
            "Waiting for FINALIZE to build fetch state."
        )
        return None, None, {}

    fetch_fh = open(bin_path, "rb")
    fetch_mm = mmap.mmap(fetch_fh.fileno(), 0, access=mmap.ACCESS_READ)
    logger.info(
        f"[host] preload fetch state from disk: blocks={len(records)} "
        f"size={bin_size}B path={bin_path}"
    )
    return fetch_fh, fetch_mm, block_index


def _create_shm(path: str, size: int) -> None:
    with open(path, "w+b") as f:
        f.write(b"\x00" * size)
    os.chmod(path, 0o600)
    logger.info(f"[host] Shared memory created: {path}  ({size // 1024} KB)")


def run_host(kv_dir: str, verbose: bool = False) -> None:
    os.makedirs(kv_dir, exist_ok=True)
    _create_shm(SHM_FILE_PATH, SHM_TOTAL_SIZE)

    shm_fd = os.open(SHM_FILE_PATH, os.O_RDWR)
    mm = mmap.mmap(shm_fd, SHM_TOTAL_SIZE, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    mm[OFF_STATUS] = STATUS_IDLE
    mm.flush()

    # ── 监控器：三类 op 共用一个 monitor，按 op 字段分组统计 ────────────────
    monitor = make_monitor(
        "host_io",
        side="host",
        enabled=_MONITOR_ENABLED,
        logger=logger,
        jsonl_path=_make_jsonl_path("host_io"),
        progress_every_n=100,
        progress_every_sec=30.0,   # 长任务每 30s 至少打一行进度
    )

    # ── offload 侧：顺序追加写文件句柄 ──────────────────────────────────
    bin_path   = os.path.join(kv_dir, "kvpack.bin")
    index_path = os.path.join(kv_dir, "kvpack_index.json")
    offload_f  = None
    records    = []   # 接收 TD 推来的 index 记录

    # ── fetch 侧：decode 阶段 mmap 读句柄（优先尝试从磁盘恢复）────────────
    fetch_fh, fetch_mm, block_index = _try_prepare_fetch_from_disk(kv_dir)

    req_offload = 0
    req_fetch   = 0
    t_start     = time.time()
    t_last_idle = time.perf_counter_ns()   # 用于估算每次 op 的轮询空转时间

    logger.info(f"[host] kv_dir={kv_dir}  ready, waiting for requests...")
    if fetch_mm is None:
        logger.info("[host] fetch state not ready yet; waiting for FINALIZE or existing files.")
    else:
        logger.info(f"[host] fetch state ready from disk, indexed blocks={len(block_index)}")

    # ── SIGTERM 友好关闭：把汇总打到 log 后再退出 ─────────────────────────
    _shutdown_requested = {"flag": False}

    def _on_signal(signum, _frame):
        logger.info(f"[host] received signal={signum}, will shutdown gracefully.")
        _shutdown_requested["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    # SIGINT 仍走 KeyboardInterrupt 路径，但也设一下确保 print 顺序

    try:
        while True:
            if _shutdown_requested["flag"]:
                logger.info("[host] graceful shutdown triggered by signal.")
                break

            status = mm[OFF_STATUS]

            if status == STATUS_IDLE:
                continue   # busy-wait
            if status == STATUS_RESPONSE_READY:
                # 仅 TD 侧应消费 RESPONSE_READY 并回写 IDLE。
                # host 看到该状态时只能等待，不能主动改写。
                if verbose:
                    logger.debug("[host] observed RESPONSE_READY; waiting TD ack")
                continue

            # 测量从上次 IDLE 到本次发现请求的"轮询空转时长"
            t_op_seen = time.perf_counter_ns()
            wait_idle_us = (t_op_seen - t_last_idle) / 1000.0

            mm[OFF_STATUS] = STATUS_HOST_BUSY
            mm.flush()

            frame   = FMT_FRAME.unpack_from(mm, OFF_FRAME)[0]
            layer   = FMT_LAYER.unpack_from(mm, OFF_LAYER)[0]
            dlen    = FMT_DATALEN.unpack_from(mm, OFF_DATA_LEN)[0]

            # ── OFFLOAD：encode 阶段 TD 推一个 block ─────────────────────
            if status == STATUS_OFFLOAD_READY:
                op_name = "OFFLOAD"
                with monitor.measure(
                    op=op_name, layer_index=int(layer),
                    frame_index=int(frame), payload_bytes=int(dlen),
                ) as op_rec:
                    op_rec.t_wait_idle_us = wait_idle_us
                    t_proc0 = time.perf_counter_ns()
                    try:
                        if offload_f is None:
                            os.makedirs(kv_dir, exist_ok=True)
                            offload_f = open(bin_path, "wb")
                            logger.info(f"[host] OFFLOAD sink opened: {bin_path}")
                        raw = bytes(mm[PAGESIZE : PAGESIZE + dlen])
                        file_offset = offload_f.tell()
                        offload_f.write(raw)
                        offload_f.flush()

                        # TD 把 rec JSON 拼在 raw 末尾。raw = block_bytes + b"\x00RECJSON:" + rec_json
                        SEP = b"\x00RECJSON:"
                        sep_pos = raw.rfind(SEP)
                        if sep_pos >= 0:
                            block_bytes = raw[:sep_pos]
                            rec = json.loads(raw[sep_pos + len(SEP):])
                            rec["offset"] = int(file_offset)
                            records.append(rec)
                            # 覆盖写，纠正 offset 后的真实 block_bytes
                            offload_f.seek(file_offset)
                            offload_f.write(block_bytes)
                            offload_f.flush()
                            # offload_f 末尾位置已被覆盖写，需要 seek 回末尾
                            offload_f.seek(0, 2)

                        req_offload += 1
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 0)
                        if verbose:
                            logger.info(
                                f"[host] offload (L={layer},F={frame}) "
                                f"{dlen//1024}KB  offset={file_offset}"
                            )
                    except Exception as e:
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                        logger.error(f"[host] OFFLOAD ERROR: {e}")
                        raise   # 让 monitor.measure 把 error 标记下来

                    op_rec.t_data_copy_us = (time.perf_counter_ns() - t_proc0) / 1000.0
                    # wait_response 在 host 侧无意义
                    op_rec.t_wait_response_us = 0.0

                mm[OFF_STATUS] = STATUS_IDLE   # offload 无需 RESPONSE_READY
                mm.flush()

            # ── FINALIZE：encode 结束，写 kvpack_index.json ───────────────
            elif status == STATUS_FINALIZE_READY:
                op_name = "FINALIZE"
                with monitor.measure(
                    op=op_name, layer_index=-1, frame_index=-1,
                    payload_bytes=int(dlen),
                ) as op_rec:
                    op_rec.t_wait_idle_us = wait_idle_us
                    t_proc0 = time.perf_counter_ns()
                    try:
                        if offload_f is None:
                            os.makedirs(kv_dir, exist_ok=True)
                            offload_f = open(bin_path, "wb")
                            logger.info(
                                f"[host] FINALIZE with empty OFFLOAD stream, created {bin_path}"
                            )
                        offload_f.flush()
                        os.fsync(offload_f.fileno())
                        offload_f.close()
                        offload_f = None

                        # 读取 TD 传来的 common_metadata
                        common_meta = json.loads(mm[PAGESIZE : PAGESIZE + dlen])
                        payload = {
                            "format": "kvpack_mmap_v1",
                            "data_file": "kvpack.bin",
                            "num_blocks": len(records),
                            "blocks": records,
                            "common_metadata": common_meta,
                        }
                        with open(index_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)

                        # 重开 fetch 侧 mmap（供后续 decode 用）
                        fetch_fh = open(bin_path, "rb")
                        fetch_mm = mmap.mmap(fetch_fh.fileno(), 0,
                                             access=mmap.ACCESS_READ)
                        block_index = {}
                        for rec in records:
                            k = (int(rec["layer_index"]), int(rec["frame_index"]))
                            block_index[k] = rec

                        logger.info(
                            f"[host] FINALIZE done: {len(records)} blocks, "
                            f"index written to {index_path}"
                        )
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 0)
                    except Exception as e:
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                        logger.error(f"[host] FINALIZE ERROR: {e}")
                        raise

                    op_rec.t_data_copy_us = (time.perf_counter_ns() - t_proc0) / 1000.0
                    op_rec.t_wait_response_us = 0.0

                mm[OFF_STATUS] = STATUS_RESPONSE_READY
                mm.flush()

                # FINALIZE 之后，把累积监控立刻 dump 一次（encode 结束）
                logger.info("[host] encode phase done, snapshot summary:")
                monitor.log_summary()

            # ── FETCH：decode 阶段 TD 请求 (layer, frame) block ──────────
            elif status == STATUS_FETCH_READY:
                op_name = "FETCH"
                # 占位 payload_bytes=0，处理完再回填
                with monitor.measure(
                    op=op_name, layer_index=int(layer),
                    frame_index=int(frame), payload_bytes=0,
                ) as op_rec:
                    op_rec.t_wait_idle_us = wait_idle_us
                    t_proc0 = time.perf_counter_ns()
                    btotal = 0
                    try:
                        if fetch_mm is None:
                            raise RuntimeError("kvpack.bin not ready (encode not finalized)")

                        key = (int(layer), int(frame))
                        if key not in block_index:
                            raise KeyError(f"block (L={layer},F={frame}) not found")

                        rec = block_index[key]
                        from kvpack_mmap_td import BLOCK_HEAD
                        off      = int(rec["offset"])
                        hlen     = int(rec["header_len"])
                        plen     = int(rec["payload_len"])
                        btotal   = BLOCK_HEAD.size + hlen + plen

                        raw_block = bytes(fetch_mm[off : off + btotal])
                        mm[PAGESIZE : PAGESIZE + btotal] = raw_block
                        FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, btotal)
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 0)

                        req_fetch += 1
                        if verbose:
                            logger.info(
                                f"[host] fetch (L={layer},F={frame}) "
                                f"{btotal//1024}KB"
                            )
                    except Exception as e:
                        err_bytes = str(e).encode()[:MAX_BLOCK_BYTES]
                        mm[PAGESIZE : PAGESIZE + len(err_bytes)] = err_bytes
                        FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, len(err_bytes))
                        FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                        logger.error(
                            f"[host] FETCH ERROR (L={layer},F={frame}): {e}"
                        )
                        raise

                    op_rec.t_data_copy_us = (time.perf_counter_ns() - t_proc0) / 1000.0
                    op_rec.t_wait_response_us = 0.0
                    op_rec.payload_bytes = int(btotal)

                mm[OFF_STATUS] = STATUS_RESPONSE_READY
                mm.flush()

            elif status == STATUS_SHUTDOWN:
                logger.info("[host] Received SHUTDOWN signal.")
                mm[OFF_STATUS] = STATUS_IDLE
                mm.flush()
                break

            # 本次 op 完成，重置"上次 IDLE"基准
            t_last_idle = time.perf_counter_ns()

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        logger.info(
            f"[host] KeyboardInterrupt. offload={req_offload}, "
            f"fetch={req_fetch}, elapsed={elapsed:.1f}s"
        )
    except Exception as e:
        logger.exception(f"[host] fatal error: {e}")
    finally:
        # ── 最终汇总 ──
        try:
            logger.info("[host] final monitor summary:")
            monitor.log_summary()
            monitor.close()
        except Exception:
            pass

        if fetch_mm:
            try:
                fetch_mm.close()
            except Exception:
                pass
        if fetch_fh:
            try:
                fetch_fh.close()
            except Exception:
                pass
        try:
            if offload_f is not None:
                offload_f.close()
        except Exception:
            pass
        try:
            mm.close()
        except Exception:
            pass
        try:
            os.close(shm_fd)
        except Exception:
            pass
        try:
            os.unlink(SHM_FILE_PATH)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv_dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_host(args.kv_dir, verbose=args.verbose)