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
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
from typing import Optional, Tuple

import numpy as np
import torch

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


class _BridgeStats:
    """轻量运行时监控：累计时延、吞吐、单次传输大小。"""
    def __init__(self, stage: str):
        self.stage = stage
        self.n = 0
        self.bytes_total = 0
        self.seconds_total = 0.0
        self.max_bytes = 0
        self.min_bytes = None

    def record(self, size_bytes: int, seconds: float):
        self.n += 1
        self.bytes_total += int(size_bytes)
        self.seconds_total += float(seconds)
        self.max_bytes = max(self.max_bytes, int(size_bytes))
        self.min_bytes = int(size_bytes) if self.min_bytes is None else min(self.min_bytes, int(size_bytes))

    def summary(self) -> str:
        if self.n == 0:
            return f"[bridge/stats:{self.stage}] no transfers"
        avg_bytes = self.bytes_total / self.n
        avg_ms = (self.seconds_total / self.n) * 1000
        bw_mbps = (self.bytes_total / (1024 * 1024)) / self.seconds_total if self.seconds_total > 0 else 0.0
        return (
            f"[bridge/stats:{self.stage}] n={self.n}  total={self.bytes_total/1024/1024:.2f}MB  "
            f"time={self.seconds_total:.3f}s  bw={bw_mbps:.2f}MB/s  "
            f"avg={avg_bytes/1024:.1f}KB/{avg_ms:.2f}ms  "
            f"min={self.min_bytes/1024:.1f}KB  max={self.max_bytes/1024:.1f}KB"
        )


def is_bridge_available() -> bool:
    if not os.path.exists(SHM_FILE_PATH):
        return False
    try:
        fd = os.open(SHM_FILE_PATH, os.O_RDONLY)
        mm = mmap.mmap(fd, PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        status = mm[OFF_STATUS]
        mm.close(); os.close(fd)
        # host may transiently stay in busy/response states; treat any non-shutdown as available
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
            raise TimeoutError(f"Bridge host not responding, status={int(mm[OFF_STATUS])}")


def _wait_response(mm: mmap.mmap, timeout: float) -> None:
    t0 = time.time()
    while mm[OFF_STATUS] != STATUS_RESPONSE_READY:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Bridge host response timed out, status={int(mm[OFF_STATUS])}")


# ══════════════════════════════════════════════════════════════════════════════
# Encode 侧：KVPackBridgeWriter
# ══════════════════════════════════════════════════════════════════════════════

class KVPackBridgeWriter:
    """
    替代 KVPackWriter，encode 阶段每个 block 写完后立即 offload 给宿主。
    接口与 KVPackWriter 完全相同（append_block / append_p_block / write_index / close）。

    原理：
      - 调用 append_block / append_p_block 时，先在 TD 侧完成序列化（含加密），
        然后通过 OFFLOAD 协议将字节推送给宿主，宿主追加写入 kvpack.bin。
      - encode 完成后调用 write_index，通过 FINALIZE 协议通知宿主写 kvpack_index.json。
      - TD 侧不保存任何 block 字节到本地磁盘，实现真正的 offload。
    """

    def __init__(self, kv_cache_dir: str):
        if not is_bridge_available():
            raise RuntimeError(
                "Bridge host not running. Start kvpack_bridge_host.py first."
            )
        self._kv_dir = kv_cache_dir
        self._fd, self._mm = _open_shm()
        self.records = []   # 本地记录 rec 元数据（不含原始字节）
        self._stats = _BridgeStats("encode_offload")
        # 复用 KVPackWriter 的序列化逻辑
        from kvpack_mmap_td import KVPackWriter as _W
        # 用一个内存 dummy 文件模拟写，只取序列化字节
        import io
        self._dummy_writer = _W.__new__(_W)
        self._dummy_writer._f = io.BytesIO()
        self._dummy_writer.records = []
        print(f"[bridge/td] KVPackBridgeWriter ready, offloading to host. kv_dir={kv_cache_dir}")

    def _offload_block(self, block_bytes: bytes, rec: dict) -> None:
        """
        把一个 block 的字节推送给宿主，等待宿主存盘确认。
        payload = block_bytes + SEP + rec_json
        宿主会从中分离出 block_bytes 和 rec，确定实际文件偏移后写入。
        """
        mm = self._mm
        rec_json = json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode()
        payload  = block_bytes + SEP + rec_json

        if len(payload) > MAX_BLOCK_BYTES:
            raise ValueError(
                f"Block payload {len(payload)} bytes exceeds MAX_BLOCK_BYTES={MAX_BLOCK_BYTES}"
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
        # 借用 dummy writer 序列化
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
        if self._mm:
            self._mm.close()
        if self._fd:
            os.close(self._fd)


# ══════════════════════════════════════════════════════════════════════════════
# Decode 侧：KVPackBridgeClient（与之前相同，fetch 协议）
# ══════════════════════════════════════════════════════════════════════════════

class KVPackBridgeClient:
    """替代 KVPackReader，decode 阶段通过 FETCH 协议向宿主请求 block。"""

    def __init__(self, kv_cache_dir: str, crypto_ctx=None,
                 timeout: float = REQUEST_TIMEOUT):
        if not is_bridge_available():
            raise RuntimeError("Bridge host not running.")
        self._crypto_ctx = crypto_ctx
        self._timeout    = timeout
        self._fd, self._mm = _open_shm()
        self._stats = _BridgeStats("decode_fetch")

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

        print(f"[bridge/td] KVPackBridgeClient: "
              f"{len(self.by_layer_frame)} blocks, "
              f"crypto={'on' if crypto_ctx and crypto_ctx.enabled else 'off'}, "
              f"num_layers={self._num_layers}, timeout={self._timeout}s")

    def read_layer_frame(self, layer_index: int, frame_index: int,
                         *, map_location: str = "cpu",
                         decrypt_fn=None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        mm = self._mm
        key = (int(layer_index), int(frame_index))
        if key not in self.by_layer_frame:
            raise KeyError(f"block (L={layer_index},F={frame_index}) not in index")

        t0 = time.time()
        _wait_idle(mm, self._timeout)
        print(f"[bridge/td] FETCH request L={layer_index} F={frame_index}")

        FMT_FRAME.pack_into(mm, OFF_FRAME, frame_index)
        FMT_LAYER.pack_into(mm, OFF_LAYER, layer_index)
        mm.flush()
        mm[OFF_STATUS] = STATUS_FETCH_READY
        mm.flush()

        _wait_response(mm, self._timeout)

        dlen  = FMT_DATALEN.unpack_from(mm, OFF_DATA_LEN)[0]
        err   = FMT_ERR.unpack_from(mm, OFF_ERR_FLAG)[0]
        raw   = bytes(mm[PAGESIZE : PAGESIZE + dlen])
        mm[OFF_STATUS] = STATUS_IDLE
        mm.flush()

        if err:
            raise RuntimeError(f"Host FETCH error: {raw.decode('utf-8', errors='replace')}")

        t_ms = (time.time() - t0) * 1000
        self._stats.record(dlen, t_ms / 1000)
        print(f"[bridge/td] fetch (L={layer_index},F={frame_index}) "
              f"{dlen//1024}KB  {t_ms:.1f}ms")

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
            raise ValueError(f"Unsupported block_type={block_type!r}; only 'I' is supported in mainline mode")

        return k.to(map_location), v.to(map_location), header

    def close(self) -> None:
        print(self._stats.summary())
        if self._mm:
            self._mm.close()
        if self._fd:
            os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
