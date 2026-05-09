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
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time

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

MAX_BLOCK_BYTES = 4 * 1024 * 1024   # 4 MB，宽裕支持 PV+I
SHM_TOTAL_SIZE  = PAGESIZE + MAX_BLOCK_BYTES

FMT_FRAME   = struct.Struct("<i")
FMT_LAYER   = struct.Struct("<i")
FMT_DATALEN = struct.Struct("<Q")
FMT_ERR     = struct.Struct("<I")


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

    fetch_fh = open(bin_path, "rb")
    fetch_mm = mmap.mmap(fetch_fh.fileno(), 0, access=mmap.ACCESS_READ)
    print(f"[host] preload fetch state from disk: blocks={len(records)} path={bin_path}")
    return fetch_fh, fetch_mm, block_index


def _create_shm(path: str, size: int) -> None:
    with open(path, "w+b") as f:
        f.write(b"\x00" * size)
    os.chmod(path, 0o600)
    print(f"[host] Shared memory created: {path}  ({size // 1024} KB)")


def run_host(kv_dir: str, verbose: bool = False) -> None:
    os.makedirs(kv_dir, exist_ok=True)
    _create_shm(SHM_FILE_PATH, SHM_TOTAL_SIZE)

    shm_fd = os.open(SHM_FILE_PATH, os.O_RDWR)
    mm = mmap.mmap(shm_fd, SHM_TOTAL_SIZE, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    mm[OFF_STATUS] = STATUS_IDLE
    mm.flush()

    # ── offload 侧：顺序追加写文件句柄 ──────────────────────────────────
    bin_path   = os.path.join(kv_dir, "kvpack.bin")
    index_path = os.path.join(kv_dir, "kvpack_index.json")
    offload_f  = open(bin_path, "wb")   # 追加写
    records    = []   # 接收 TD 推来的 index 记录

    # ── fetch 侧：decode 阶段 mmap 读句柄（优先尝试从磁盘恢复）────────────
    fetch_fh, fetch_mm, block_index = _try_prepare_fetch_from_disk(kv_dir)

    req_offload = 0
    req_fetch   = 0
    t_start     = time.time()

    print(f"[host] kv_dir={kv_dir}  ready, waiting for requests...")
    if fetch_mm is None:
        print("[host] fetch state not ready yet; waiting for FINALIZE or existing files.")
    else:
        print(f"[host] fetch state ready from disk, indexed blocks={len(block_index)}")

    try:
        while True:
            status = mm[OFF_STATUS]

            if status == STATUS_IDLE:
                continue   # busy-wait

            mm[OFF_STATUS] = STATUS_HOST_BUSY
            mm.flush()

            frame   = FMT_FRAME.unpack_from(mm, OFF_FRAME)[0]
            layer   = FMT_LAYER.unpack_from(mm, OFF_LAYER)[0]
            dlen    = FMT_DATALEN.unpack_from(mm, OFF_DATA_LEN)[0]

            # ── OFFLOAD：encode 阶段 TD 推一个 block ─────────────────────
            if status == STATUS_OFFLOAD_READY:
                try:
                    raw = bytes(mm[PAGESIZE : PAGESIZE + dlen])
                    file_offset = offload_f.tell()
                    offload_f.write(raw)
                    offload_f.flush()

                    # TD 会同时把 rec JSON 放在 raw 末尾，约定格式：
                    # raw = block_bytes + b"\x00RECJSON:" + rec_json_bytes
                    SEP = b"\x00RECJSON:"
                    sep_pos = raw.rfind(SEP)
                    if sep_pos >= 0:
                        block_bytes = raw[:sep_pos]
                        rec = json.loads(raw[sep_pos + len(SEP):])
                        rec["offset"] = int(file_offset)   # 实际写入位置
                        records.append(rec)
                        # 覆盖写，纠正 offset
                        offload_f.seek(file_offset)
                        offload_f.write(block_bytes)
                        offload_f.flush()

                    req_offload += 1
                    FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 0)
                    if verbose:
                        print(f"[host] offload (L={layer},F={frame}) "
                              f"{dlen//1024}KB  offset={file_offset}")
                except Exception as e:
                    FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                    print(f"[host] OFFLOAD ERROR: {e}", file=sys.stderr)

                mm[OFF_STATUS] = STATUS_IDLE   # offload 无需响应数据
                mm.flush()

            # ── FINALIZE：encode 结束，写 kvpack_index.json ───────────────
            elif status == STATUS_FINALIZE_READY:
                try:
                    offload_f.flush()
                    os.fsync(offload_f.fileno())
                    offload_f.close()

                    # 读取 TD 传来的 common_metadata（放在数据页）
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

                    # 打开 fetch 侧的 mmap（供后续 decode 使用）
                    fetch_fh = open(bin_path, "rb")
                    fetch_mm = mmap.mmap(fetch_fh.fileno(), 0,
                                         access=mmap.ACCESS_READ)
                    # 重建 block_index
                    block_index = {}
                    for rec in records:
                        k = (int(rec["layer_index"]), int(rec["frame_index"]))
                        block_index[k] = rec

                    print(f"[host] FINALIZE done: {len(records)} blocks, "
                          f"index written to {index_path}")
                    FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 0)
                except Exception as e:
                    FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                    print(f"[host] FINALIZE ERROR: {e}", file=sys.stderr)

                mm[OFF_STATUS] = STATUS_RESPONSE_READY
                mm.flush()

            # ── FETCH：decode 阶段 TD 请求 (layer, frame) block ──────────
            elif status == STATUS_FETCH_READY:
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
                    if not verbose:
                        print(f"[host] fetch ok (L={layer},F={frame}) bytes={btotal}")
                    if verbose:
                        print(f"[host] fetch (L={layer},F={frame}) {btotal//1024}KB")
                except Exception as e:
                    err_bytes = str(e).encode()[:MAX_BLOCK_BYTES]
                    mm[PAGESIZE : PAGESIZE + len(err_bytes)] = err_bytes
                    FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, len(err_bytes))
                    FMT_ERR.pack_into(mm, OFF_ERR_FLAG, 1)
                    print(f"[host] FETCH ERROR (L={layer},F={frame}): {e}",
                          file=sys.stderr)

                mm[OFF_STATUS] = STATUS_RESPONSE_READY
                mm.flush()

            elif status == STATUS_SHUTDOWN:
                print("[host] Received SHUTDOWN signal.")
                mm[OFF_STATUS] = STATUS_IDLE
                mm.flush()
                break

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\n[host] Shutting down. offload={req_offload}, "
              f"fetch={req_fetch}, elapsed={elapsed:.1f}s")
    finally:
        if fetch_mm:
            fetch_mm.close()
        if fetch_fh:
            fetch_fh.close()
        try:
            offload_f.close()
        except Exception:
            pass
        mm.close()
        os.close(shm_fd)
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