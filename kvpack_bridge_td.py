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
SHM_FILE_PATH   = "/dev/shm/kvbridge_shmem"
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


def is_bridge_available() -> bool:
    if not os.path.exists(SHM_FILE_PATH):
        return False
    try:
        fd = os.open(SHM_FILE_PATH, os.O_RDONLY)
        mm = mmap.mmap(fd, PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        status = mm[OFF_STATUS]
        mm.close(); os.close(fd)
        return status == STATUS_IDLE
    except Exception:
        return False


def _open_shm():
    fd = os.open(SHM_FILE_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_TOTAL_SIZE, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    return fd, mm


def _wait_idle(mm: mmap.mmap, timeout: float) -> None:
    t0 = time.time()
    while mm[OFF_STATUS] not in (STATUS_IDLE, STATUS_RESPONSE_READY):
        if time.time() - t0 > timeout:
            raise TimeoutError("Bridge host not responding")


def _wait_response(mm: mmap.mmap, timeout: float) -> None:
    t0 = time.time()
    while mm[OFF_STATUS] != STATUS_RESPONSE_READY:
        if time.time() - t0 > timeout:
            raise TimeoutError("Bridge host response timed out")


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
        # 复用 KVPackWriter 的序列化逻辑
        from kvpack_mmap_td import KVPackWriter as _W
        # 用一个内存 dummy 文件模拟写，只取序列化字节
        import io
        self._dummy_writer = _W.__new__(_W)
        self._dummy_writer._f = io.BytesIO()
        self._dummy_writer.records = []
        print(f"[bridge/td] KVPackBridgeWriter ready, offloading to host.")

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
        if err:
            raise RuntimeError(
                f"Host OFFLOAD error for (L={rec.get('layer_index')}, "
                f"F={rec.get('frame_index')})"
            )

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

    def append_p_block(self, *, frame_index, layer_index, seq_start, seq_end,
                       key_tensor, value_tensor, ref_value_tensor,
                       ref_frame_index, delta_threshold=1e-3, encrypt_fn=None) -> dict:
        """与 KVPackWriter.append_p_block 接口完全相同。"""
        dw = self._dummy_writer
        dw._f.seek(0)
        dw._f.truncate(0)
        rec = dw.append_p_block(
            frame_index=frame_index, layer_index=layer_index,
            seq_start=seq_start, seq_end=seq_end,
            key_tensor=key_tensor, value_tensor=value_tensor,
            ref_value_tensor=ref_value_tensor,
            ref_frame_index=ref_frame_index,
            delta_threshold=delta_threshold,
            encrypt_fn=encrypt_fn,
        )
        dw._f.seek(0)
        block_bytes = dw._f.read()
        dw.records.clear()

        self._offload_block(block_bytes, rec)
        self.records.append(rec)
        return rec

    def write_index(self, common_metadata: dict) -> None:
        """encode 结束：通知宿主写 kvpack_index.json。"""
        mm = self._mm
        meta_bytes = json.dumps(common_metadata, ensure_ascii=False).encode()
        if len(meta_bytes) > MAX_BLOCK_BYTES:
            raise ValueError("common_metadata too large for bridge")

        _wait_idle(mm, REQUEST_TIMEOUT)
        mm[PAGESIZE : PAGESIZE + len(meta_bytes)] = meta_bytes
        FMT_DATALEN.pack_into(mm, OFF_DATA_LEN, len(meta_bytes))
        mm.flush()
        mm[OFF_STATUS] = STATUS_FINALIZE_READY
        mm.flush()

        _wait_response(mm, REQUEST_TIMEOUT)
        err = FMT_ERR.unpack_from(mm, OFF_ERR_FLAG)[0]
        mm[OFF_STATUS] = STATUS_IDLE
        mm.flush()
        if err:
            raise RuntimeError("Host FINALIZE failed")
        print(f"[bridge/td] FINALIZE done: {len(self.records)} blocks offloaded.")

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
              f"crypto={'on' if crypto_ctx and crypto_ctx.enabled else 'off'}")

    def read_layer_frame(self, layer_index: int, frame_index: int,
                         *, map_location: str = "cpu",
                         decrypt_fn=None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        mm = self._mm
        key = (int(layer_index), int(frame_index))
        if key not in self.by_layer_frame:
            raise KeyError(f"block (L={layer_index},F={frame_index}) not in index")

        t0 = time.time()
        _wait_idle(mm, self._timeout)

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
        print(f"[bridge/td] fetch (L={layer_index},F={frame_index}) "
              f"{dlen//1024}KB  {t_ms:.1f}ms")

        return self._parse_raw(raw, layer_index, frame_index, map_location)

    def _parse_raw(self, raw: bytes, layer_index: int, frame_index: int,
                   map_location: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        from kvpack_mmap_td import BLOCK_HEAD, _DTYPE_TO_NP, _unpack_sparse_delta
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

        elif block_type in ("PV", "P"):
            k_nbytes  = int(header["k_nbytes"])
            mask_v_b  = int(header["mask_v_bytes"])
            k_count   = k_nbytes // np.dtype(np_dtype).itemsize
            k_buf     = np.frombuffer(payload, dtype=np_dtype, count=k_count)
            k = torch.from_numpy(k_buf.copy()).reshape(header["k_shape"])
            if dtype_key == "torch.bfloat16":
                k = k.view(torch.bfloat16)
            _, v_ref, _ = self.read_layer_frame(
                layer_index, int(header["ref_frame"]), map_location="cpu"
            )
            mask_v  = payload[k_nbytes : k_nbytes + mask_v_b]
            nz_v    = payload[k_nbytes + mask_v_b :]
            delta_v = _unpack_sparse_delta(mask_v, nz_v, tuple(header["v_shape"]),
                                           np.float32, int(header["nnz_v"]))
            v = (v_ref.float() + torch.from_numpy(delta_v.astype(np.float32))).to(
                {"torch.float32": torch.float32,
                 "torch.float16": torch.float16,
                 "torch.bfloat16": torch.bfloat16}.get(dtype_key, torch.float32)
            )
        else:
            raise ValueError(f"Unknown block_type={block_type!r}")

        return k.to(map_location), v.to(map_location), header

    def close(self) -> None:
        if self._mm:
            self._mm.close()
        if self._fd:
            os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()