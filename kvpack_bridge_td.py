"""kvpack_bridge_td.py
主机进程（TD 侧）。

encode 阶段：每写完一个 block，通过 OFFLOAD 协议推送给宿主存档；
             encode 结束后发 FINALIZE 通知宿主写 kvpack_index.json。

decode 阶段：通过 FETCH 协议向宿主请求 (layer, frame) block，
             收到密文后在 TD 侧解密，返回 (K, V) tensor。

对外接口：
  KVPackBridgeWriter  : 替代 KVPackWriter，encode 阶段使用
  KVPackBridgeClient  : 替代 KVPackReader，decode 阶段使用
  is_bridge_available : 检测宿主是否就绪

监控
----
本文件使用 zhz_bridge_eval_utils.BridgeMonitor 记录每次 OFFLOAD / FETCH
/ FINALIZE 的三阶段计时（wait_idle / data_copy / wait_response），并在
encode 完成（write_index）或 decode 关闭（close）时打印分组汇总。

环境变量
  BRIDGE_MONITOR_JSONL_DIR : 若设置，每个 monitor 会把详细记录写到
      ${BRIDGE_MONITOR_JSONL_DIR}/<name>_<side>.jsonl
  BRIDGE_MONITOR_DISABLE   : 若设置为 "1"，关闭监控（接口仍兼容，0 开销）
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import time
from typing import Optional, Tuple

import numpy as np
import torch

from zhz_bridge_eval_utils import make_monitor

logger = logging.getLogger(__name__)

# ── 协议常量（与 kvpack_bridge_host.py 完全一致）────────────────────────────
#SHM_FILE_PATH   = "/dev/shm/kvbridge_shmem"
SHM_FILE_PATH   = "/sys/bus/pci/devices/0000:00:03.0/resource2"   # TDX 特定路径
PAGESIZE        = os.sysconf("SC_PAGE_SIZE")

OFF_STATUS      = 0
OFF_FRAME       = 1
OFF_LAYER       = 5
OFF_DATA_LEN    = 9
OFF_ERR_FLAG    = 17

STATUS_IDLE             = 0
STATUS_OFFLOAD_READY    = 1
STATUS_HOST_BUSY        = 2
STATUS_FETCH_READY      = 3
STATUS_RESPONSE_READY   = 4
STATUS_FINALIZE_READY   = 5
STATUS_SHUTDOWN         = 6

MAX_BLOCK_BYTES = 4 * 1024 * 1024
SHM_TOTAL_SIZE  = PAGESIZE + MAX_BLOCK_BYTES

FMT_FRAME   = struct.Struct("<i")
FMT_LAYER   = struct.Struct("<i")
FMT_DATALEN = struct.Struct("<Q")
FMT_ERR     = struct.Struct("<I")

REQUEST_TIMEOUT = 60.0

SEP = b"\x00RECJSON:"   # block 字节与 rec JSON 的分隔符


# ── 全局监控开关 ───────────────────────────────────────────────────────────

_MONITOR_ENABLED = os.environ.get("BRIDGE_MONITOR_DISABLE", "0") != "1"
_MONITOR_JSONL_DIR = os.environ.get("BRIDGE_MONITOR_JSONL_DIR", "").strip() or None


def _make_jsonl_path(name: str, side: str) -> Optional[str]:
    """根据 env 拼一个 monitor 专属的 jsonl 路径，未配置则返回 None。"""
    if not _MONITOR_JSONL_DIR:
        return None
    return os.path.join(_MONITOR_JSONL_DIR, f"{name}_{side}.jsonl")


def is_bridge_available() -> bool:
    if not os.path.exists(SHM_FILE_PATH):
        return False
    try:
        fd = os.open(SHM_FILE_PATH, os.O_RDONLY)
        mm = mmap.mmap(fd, PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        status = mm[OFF_STATUS]
        mm.close(); os.close(fd)
        # host 可能短暂处于 busy/response 状态；只要不是 SHUTDOWN 都视为可用
        return status != STATUS_SHUTDOWN
    except Exception:
        return False


def _open_shm():
    fd = os.open(SHM_FILE_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_TOTAL_SIZE, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    return fd, mm


def _wait_idle(mm: mmap.mmap, timeout: float) -> None:
    t0 = time.time()
    while mm[OFF_STATUS] != STATUS_IDLE:
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"Bridge host not responding, status={int(mm[OFF_STATUS])}"
            )


def _wait_response(mm: mmap.mmap, timeout: float) -> None:
    t0 = time.time()
    while mm[OFF_STATUS] != STATUS_RESPONSE_READY:
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"Bridge host response timed out, status={int(mm[OFF_STATUS])}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Encode 侧：KVPackBridgeWriter
# ══════════════════════════════════════════════════════════════════════════════

class KVPackBridgeWriter:
    """
    替代 KVPackWriter，encode 阶段每个 block 写完后立即 offload 给宿主。
    接口与 KVPackWriter 完全相同（append_block / write_index / close）。

    监控
    ----
    每次 _offload_block 在 BridgeMonitor 中记录三段时间：
      - t_wait_idle_us     : 等宿主回到 IDLE
      - t_data_copy_us     : 把 payload 写入 SHM 数据页 + 控制字段 + flush
      - t_wait_response_us : 宿主完成磁盘写入并回 IDLE 的耗时
    write_index 另外记录一次 FINALIZE op，最后调用 log_summary。
    """

    def __init__(self, kv_cache_dir: str):
        if not is_bridge_available():
            raise RuntimeError(
                "Bridge host not running. Start kvpack_bridge_host.py first."
            )
        self._kv_dir = kv_cache_dir
        self._fd, self._mm = _open_shm()
        self.records = []   # 本地记录 rec 元数据（不含原始字节）
        self._monitor = make_monitor(
            "encode_offload",
            side="td",
            enabled=_MONITOR_ENABLED,
            logger=logger,
            jsonl_path=_make_jsonl_path("encode_offload", "td"),
            progress_every_n=50,
            progress_every_sec=0.0,
        )
        # 复用 KVPackWriter 的序列化逻辑（写到内存 BytesIO，不落盘）
        from kvpack_mmap_td import KVPackWriter as _W
        import io
        self._dummy_writer = _W.__new__(_W)
        self._dummy_writer._f = io.BytesIO()
        self._dummy_writer.records = []
        logger.info(
            f"[bridge/td] KVPackBridgeWriter ready, offloading to host. "
            f"kv_dir={kv_cache_dir}"
        )

    def _offload_block(self, block_bytes: bytes, rec: dict) -> None:
        """
        把一个 block 的字节推送给宿主，等待宿主存盘确认。
        payload = block_bytes + SEP + rec_json
        """
        mm = self._mm
        rec_json = json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode()
        payload  = block_bytes + SEP + rec_json

        if len(payload) > MAX_BLOCK_BYTES:
            raise ValueError(
                f"Block payload {len(payload)} bytes exceeds "
                f"MAX_BLOCK_BYTES={MAX_BLOCK_BYTES}"
            )

        _wait_idle(mm, REQUEST_TIMEOUT)
        t0 = time.time()
        print(f"[bridge/td] OFFLOAD send L={rec.get('layer_index')} F={rec.get('frame_index')} bytes={len(payload)}")
        mm[PAGESIZE : PAGESIZE + len(payload)] = payload
        FMT_FRAME.pack_into(mm, OFF_FRAME, rec.get("frame_index", 0))
        FMT_LAYER.pack_into(mm, OFF_LAYER, rec.get("layer_index", 0))
        FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, len(payload))
        mm.flush()
        mm[OFF_STATUS] = STATUS_OFFLOAD_READY
        mm.flush()

        # 对 OFFLOAD，宿主直接回到 IDLE（不需要等 RESPONSE_READY）
        t0 = time.time()
        while mm[OFF_STATUS] not in (STATUS_IDLE,):
            if mm[OFF_STATUS] == STATUS_HOST_BUSY:
                pass   # 宿主正在写，继续等
            if time.time() - t0 > REQUEST_TIMEOUT:
                raise TimeoutError("OFFLOAD timed out")

        err = FMT_ERR.unpack_from(mm, OFF_ERR_FLAG)[0]
        self._stats.record(len(payload), time.time() - t0)
        if err:
            raise RuntimeError("Host FINALIZE failed")
        # ── 关键修复:TD 侧也落一份本地 kvpack_index.json ──────────────
        # 真机下 TD 与 host 是两套文件系统;host 写的索引在 host 侧,
        # 而 TD 解码(KVPackBridgeClient)需在本地读取该索引以获得块结构与
        # common_metadata。TD 已在 self.records 持有全部块记录,直接落盘即可。
        # 注:解码按 (layer,frame) 取回、由 host 解析真实 offset,本地索引的
        #     offset 字段仅为占位,不参与取回,无需与 host 一致。
        print(f"[bridge/td] FINALIZE done: {len(self.records)} blocks offloaded.")

    def append_block(self, *, frame_index, layer_index, seq_start, seq_end,
                     key_tensor, value_tensor, encrypt_fn=None) -> dict:
        """与 KVPackWriter.append_block 接口完全相同。"""
        dw = self._dummy_writer
        dw._f.seek(0)
        dw._f.truncate(0)
        rec = dw.append_block(
            frame_index=frame_index, layer_index=layer_index,
            seq_start=seq_start, seq_end=seq_end,
            key_tensor=key_tensor, value_tensor=value_tensor,
            encrypt_fn=encrypt_fn,
        )
        dw._f.seek(0)
        block_bytes = dw._f.read()
        dw.records.clear()

        self._offload_block(block_bytes, rec)
        self.records.append(rec)
        return rec

    def write_index(self, common_metadata: dict) -> None:
        """encode 结束：通知宿主写 kvpack_index.json，并在 TD 侧落一份本地索引。"""
        mm = self._mm
        meta_bytes = json.dumps(common_metadata, ensure_ascii=False).encode()
        if len(meta_bytes) > MAX_BLOCK_BYTES:
            raise ValueError("common_metadata too large for bridge")

        _wait_idle(mm, REQUEST_TIMEOUT)
        t0 = time.time()
        print(f"[bridge/td] FINALIZE send metadata bytes={len(meta_bytes)}")
        mm[PAGESIZE : PAGESIZE + len(meta_bytes)] = meta_bytes
        FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, len(meta_bytes))
        mm.flush()
        mm[OFF_STATUS] = STATUS_FINALIZE_READY
        mm.flush()

        _wait_response(mm, REQUEST_TIMEOUT)
        err = FMT_ERR.unpack_from(mm, OFF_ERR_FLAG)[0]
        finalize_sec = time.time() - t0
        mm[OFF_STATUS] = STATUS_IDLE
        mm.flush()
        if err:
            raise RuntimeError("Host FINALIZE failed")

        # ── 关键修复:TD 侧也落一份本地 kvpack_index.json（解码自给自足）──
        self._write_local_index(common_metadata)

        print(f"[bridge/td] FINALIZE done: {len(self.records)} blocks offloaded.")
        print(self._stats.summary())
        print(f"[bridge/stats:finalize] metadata={len(meta_bytes)}B time={finalize_sec*1000:.2f}ms")

    def _write_local_index(self, common_metadata: dict) -> None:
        os.makedirs(self._kv_dir, exist_ok=True)
        local_index_path = os.path.join(self._kv_dir, "kvpack_index.json")
        payload = {
            "format": "kvpack_mmap_v1",
            "data_file": "kvpack.bin",   # 实际驻留 host;TD 侧仅作标识
            "num_blocks": len(self.records),
            "blocks": self.records,
            "common_metadata": common_metadata,
        }
        with open(local_index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[bridge/td] local kvpack_index.json written: "
              f"{local_index_path} ({len(self.records)} blocks)")

    def close(self) -> None:
        try:
            self._monitor.close()
        except Exception:
            pass
        if self._mm:
            self._mm.close()
        if self._fd:
            os.close(self._fd)


# ══════════════════════════════════════════════════════════════════════════════
# Decode 侧：KVPackBridgeClient
# ══════════════════════════════════════════════════════════════════════════════

class KVPackBridgeClient:
    """替代 KVPackReader，decode 阶段通过 FETCH 协议向宿主请求 block。

    监控
    ----
    每次 read_layer_frame 在 BridgeMonitor 中记录三段时间：
      - t_wait_idle_us     : 等宿主 IDLE
      - t_data_copy_us     : 写请求字段（很小） + 后续从 SHM 读响应
      - t_wait_response_us : 宿主磁盘读取并写回响应的耗时
    close() 时自动 log_summary。
    """

    def __init__(self, kv_cache_dir: str, crypto_ctx=None,
                 timeout: float = REQUEST_TIMEOUT):
        if not is_bridge_available():
            raise RuntimeError("Bridge host not running.")
        self._crypto_ctx = crypto_ctx
        self._timeout    = timeout
        self._fd, self._mm = _open_shm()
        self._monitor = make_monitor(
            "decode_fetch",
            side="td",
            enabled=_MONITOR_ENABLED,
            logger=logger,
            jsonl_path=_make_jsonl_path("decode_fetch", "td"),
            progress_every_n=50,
            progress_every_sec=0.0,
        )

        index_path = os.path.join(kv_cache_dir, "kvpack_index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self._index = json.load(f)
        self.common_metadata = self._index.get("common_metadata", {}) or {}
        self._num_layers = int(self.common_metadata.get("num_layers", 28))

        self.by_layer_frame = {}
        self.frames = {}
        for b in self._index.get("blocks", []):
            key = (int(b["layer_index"]), int(b["frame_index"]))
            self.by_layer_frame[key] = b
            self.frames.setdefault(int(b["frame_index"]), []).append(b)

        logger.info(
            f"[bridge/td] KVPackBridgeClient: "
            f"{len(self.by_layer_frame)} blocks, "
            f"crypto={'on' if crypto_ctx and crypto_ctx.enabled else 'off'}, "
            f"num_layers={self._num_layers}, timeout={self._timeout}s"
        )

    def read_layer_frame(self, layer_index: int, frame_index: int,
                         *, map_location: str = "cpu",
                         decrypt_fn=None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        mm = self._mm
        key = (int(layer_index), int(frame_index))
        if key not in self.by_layer_frame:
            raise KeyError(f"block (L={layer_index},F={frame_index}) not in index")

        # 先用 0 占位 payload_bytes，读到响应后再回填
        with self._monitor.measure(
            op="FETCH",
            layer_index=int(layer_index), frame_index=int(frame_index),
            payload_bytes=0,
        ) as op_rec:
            # ── 阶段 1: wait_idle ─────────────────────────────────────────
            t0 = time.perf_counter_ns()
            _wait_idle(mm, self._timeout)
            t_after_wait = time.perf_counter_ns()
            op_rec.t_wait_idle_us = (t_after_wait - t0) / 1000.0

            # ── 阶段 2a: 写请求字段（仅 layer + frame）─────────────────────
            FMT_FRAME.pack_into(mm, OFF_FRAME, frame_index)
            FMT_LAYER.pack_into(mm, OFF_LAYER, layer_index)
            mm.flush()
            mm[OFF_STATUS] = STATUS_FETCH_READY
            mm.flush()
            t_after_send = time.perf_counter_ns()
            t_send_us = (t_after_send - t_after_wait) / 1000.0

            # ── 阶段 3: wait_response（宿主磁盘读取并写回响应）───────────
            _wait_response(mm, self._timeout)
            t_after_resp = time.perf_counter_ns()
            op_rec.t_wait_response_us = (t_after_resp - t_after_send) / 1000.0

            # ── 阶段 2b: 从 SHM 数据页读响应字节 ─────────────────────────
            dlen = FMT_DATALEN.unpack_from(mm, OFF_DATA_LEN)[0]
            err  = FMT_ERR.unpack_from(mm, OFF_ERR_FLAG)[0]
            raw  = bytes(mm[PAGESIZE : PAGESIZE + dlen])
            mm[OFF_STATUS] = STATUS_IDLE
            mm.flush()
            t_end = time.perf_counter_ns()
            t_recv_us = (t_end - t_after_resp) / 1000.0

            # data_copy 包含 TD→SHM 写请求 + SHM→TD 读响应（两次小拷贝）
            op_rec.t_data_copy_us = t_send_us + t_recv_us
            op_rec.t_total_us     = (t_end - t0) / 1000.0
            op_rec.payload_bytes  = int(dlen)

            if err:
                raise RuntimeError(
                    f"Host FETCH error: {raw.decode('utf-8', errors='replace')}"
                )

        return self._parse_raw(raw, layer_index, frame_index, map_location)

    def _parse_raw(self, raw: bytes, layer_index: int, frame_index: int,
                   map_location: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        from kvpack_mmap_td import BLOCK_HEAD, _DTYPE_TO_NP
        magic, hlen, plen = BLOCK_HEAD.unpack_from(raw, 0)
        if magic != b"KVPBLK01":
            raise ValueError(f"Bad magic: {magic!r}")
        header = json.loads(raw[BLOCK_HEAD.size : BLOCK_HEAD.size + hlen])
        payload = raw[BLOCK_HEAD.size + hlen : BLOCK_HEAD.size + hlen + plen]

        if bool(header.get("encrypted", False)):
            if self._crypto_ctx is None or not self._crypto_ctx.enabled:
                raise RuntimeError("Block encrypted but no crypto_ctx")
            from kvcache_crypto_td import layer_frame_block_id, decrypt_blob_to_bytes
            block_id = layer_frame_block_id(
                frame_index=int(header["frame_index"]),
                layer_index=int(header["layer_index"]),
                num_layers=self._num_layers,
            )
            aad = {k: header[k] for k in
                   ("frame_index", "layer_index", "seq_start", "seq_end", "dtype")}
            payload = decrypt_blob_to_bytes(
                payload, self._crypto_ctx.master_key,
                expected_chunk_index=block_id, expected_aad=aad,
            )

        block_type = header.get("block_type", "I")
        dtype_key  = header["dtype"]
        np_dtype   = _DTYPE_TO_NP.get(dtype_key, np.float32)

        if block_type == "I":
            k_nbytes = int(header["k_nbytes"])
            k_buf = np.frombuffer(payload, dtype=np_dtype,
                                  count=k_nbytes // np.dtype(np_dtype).itemsize)
            v_buf = np.frombuffer(payload, dtype=np_dtype,
                                  count=(len(payload)-k_nbytes)//np.dtype(np_dtype).itemsize,
                                  offset=k_nbytes)
            k = torch.from_numpy(k_buf.copy()).reshape(header["k_shape"])
            v = torch.from_numpy(v_buf.copy()).reshape(header["v_shape"])
            if dtype_key == "torch.bfloat16":
                k, v = k.view(torch.bfloat16), v.view(torch.bfloat16)

        else:
            raise ValueError(
                f"Unsupported block_type={block_type!r}; "
                f"only 'I' is supported in mainline mode"
            )

        return k.to(map_location), v.to(map_location), header

    def close(self) -> None:
        try:
            # decode 结束，打整体汇总
            self._monitor.log_summary()
            self._monitor.close()
        except Exception:
            pass
        if self._mm:
            self._mm.close()
        if self._fd:
            os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()