"""mmap-friendly KV pack format for layer-frame blocks.

Format v1 (append-only)
-----------------------
Data file: kvpack.bin
Each block layout:
    magic(8) = b'KVPBLK01'
    header_len(uint32 LE)
    payload_len(uint64 LE)
    header_json (utf-8)
    payload bytes = key_bytes || value_bytes

header_json fields:
    frame_index, layer_index, seq_start, seq_end,
    dtype, k_shape, v_shape, k_nbytes

Index file: kvpack_index.json
    stores all block offsets and common metadata.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

BLOCK_MAGIC = b"KVPBLK01"
BLOCK_HEAD = struct.Struct("<8sIQ")  # magic, header_len, payload_len

_DTYPE_TO_NP = {
    "torch.float16": np.float16,
    "torch.float32": np.float32,
    "torch.bfloat16": np.uint16,  # raw preserve, consumer may cast if needed
    "torch.int8": np.int8,
    "torch.int16": np.int16,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
}


@dataclass
class BlockRecord:
    frame_index: int
    layer_index: int
    seq_start: int
    seq_end: int
    offset: int
    total_len: int
    header_len: int
    payload_len: int
    dtype: str
    k_shape: List[int]
    v_shape: List[int]
    k_nbytes: int


class KVPackWriter:
    def __init__(self, kv_cache_dir: str):
        os.makedirs(kv_cache_dir, exist_ok=True)
        self.data_path = os.path.join(kv_cache_dir, "kvpack.bin")
        self.index_path = os.path.join(kv_cache_dir, "kvpack_index.json")
        self._f = open(self.data_path, "wb")
        self.records: List[dict] = []

    def close(self):
        if not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()

    def append_block(
        self,
        *,
        frame_index: int,
        layer_index: int,
        seq_start: int,
        seq_end: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        encrypt_fn=None,
    ) -> dict:
        k = key_tensor.detach().to("cpu").contiguous()
        v = value_tensor.detach().to("cpu").contiguous()
        if str(k.dtype) != str(v.dtype):
            raise TypeError("K/V dtype mismatch in block")

        k_bytes = memoryview(k.numpy()).tobytes(order="C")
        v_bytes = memoryview(v.numpy()).tobytes(order="C")
        header = {
            "frame_index": int(frame_index),
            "layer_index": int(layer_index),
            "seq_start": int(seq_start),
            "seq_end": int(seq_end),
            "dtype": str(k.dtype),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "k_nbytes": len(k_bytes),
            "encrypted": False,
        }
        payload = k_bytes + v_bytes
        if encrypt_fn is not None:
            payload = encrypt_fn(payload, header)
            header["encrypted"] = True
        header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        offset = self._f.tell()
        self._f.write(BLOCK_HEAD.pack(BLOCK_MAGIC, len(header_bytes), len(payload)))
        self._f.write(header_bytes)
        self._f.write(payload)

        rec = {
            **header,
            "offset": int(offset),
            "header_len": len(header_bytes),
            "payload_len": len(payload),
            "total_len": int(BLOCK_HEAD.size + len(header_bytes) + len(payload)),
        }
        self.records.append(rec)
        return rec

    def write_index(self, common_metadata: dict):
        payload = {
            "format": "kvpack_mmap_v1",
            "data_file": os.path.basename(self.data_path),
            "num_blocks": len(self.records),
            "blocks": self.records,
            "common_metadata": common_metadata or {},
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class KVPackReader:
    def __init__(self, kv_cache_dir: str):
        self.index_path = os.path.join(kv_cache_dir, "kvpack_index.json")
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.data_path = os.path.join(kv_cache_dir, self.index.get("data_file", "kvpack.bin"))
        self._fh = open(self.data_path, "rb")
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.common_metadata = self.index.get("common_metadata", {}) or {}

        self.by_layer_frame: Dict[Tuple[int, int], dict] = {}
        self.frames: Dict[int, List[dict]] = {}
        for b in self.index.get("blocks", []):
            key = (int(b["layer_index"]), int(b["frame_index"]))
            self.by_layer_frame[key] = b
            self.frames.setdefault(int(b["frame_index"]), []).append(b)

    def close(self):
        try:
            self._mmap.close()
        finally:
            self._fh.close()

    def _read_header(self, rec: dict) -> dict:
        off = int(rec["offset"])
        head = self._mmap[off: off + BLOCK_HEAD.size]
        magic, hlen, plen = BLOCK_HEAD.unpack(head)
        if magic != BLOCK_MAGIC:
            raise ValueError(f"Invalid block magic at offset={off}")
        hstart = off + BLOCK_HEAD.size
        header = json.loads(self._mmap[hstart: hstart + hlen].decode("utf-8"))
        if int(plen) != int(rec["payload_len"]):
            raise ValueError("payload_len mismatch between index and block")
        return header

    def read_layer_frame(self, layer_index: int, frame_index: int, *, map_location: str = "cpu", decrypt_fn=None):
        rec = self.by_layer_frame[(int(layer_index), int(frame_index))]
        header = self._read_header(rec)
        dtype_key = header["dtype"]
        if dtype_key not in _DTYPE_TO_NP:
            raise TypeError(f"Unsupported dtype in pack: {dtype_key}")
        np_dtype = _DTYPE_TO_NP[dtype_key]

        off = int(rec["offset"])
        hlen = int(rec["header_len"])
        pstart = off + BLOCK_HEAD.size + hlen
        k_nbytes = int(header["k_nbytes"])
        payload_len = int(rec["payload_len"])

        kv_bytes = self._mmap[pstart: pstart + payload_len]
        if bool(header.get("encrypted", False)):
            if decrypt_fn is None:
                raise RuntimeError(
                    f"Encrypted kvpack block requires decrypt_fn: layer={layer_index}, frame={frame_index}"
                )
            kv_bytes = decrypt_fn(bytes(kv_bytes), header)
        k_buf = np.frombuffer(kv_bytes, dtype=np_dtype, count=k_nbytes // np.dtype(np_dtype).itemsize, offset=0)
        v_off = k_nbytes
        v_count = (payload_len - k_nbytes) // np.dtype(np_dtype).itemsize
        v_buf = np.frombuffer(kv_bytes, dtype=np_dtype, count=v_count, offset=v_off)

        k = torch.from_numpy(k_buf.copy()).reshape(header["k_shape"])
        v = torch.from_numpy(v_buf.copy()).reshape(header["v_shape"])
        # bfloat16 raw restore
        if dtype_key == "torch.bfloat16":
            k = k.view(torch.bfloat16)
            v = v.view(torch.bfloat16)
        return k.to(map_location), v.to(map_location), header


def has_kvpack(kv_cache_dir: str) -> bool:
    return os.path.exists(os.path.join(kv_cache_dir, "kvpack_index.json"))
