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


# ---------------------------------------------------------------------------
# Delta 压缩：I 帧（关键帧）存完整 KV，P 帧存稀疏差分
#
# 格式扩展（仅在 header_json 加字段，不改 BLOCK_MAGIC/BLOCK_HEAD）：
#   block_type  : "I" = 关键帧（完整存储）
#                 "P" = 差分帧（相对于参考 I 帧的稀疏差）
#   ref_frame   : int，P 帧的参考帧索引（仅 P 帧有）
#   delta_threshold : float，生成 P 帧时使用的量化阈值
#   nnz_k / nnz_v   : int，K/V 差分中非零元素个数
#
# P 帧 payload 布局（序列化后传给 encrypt_fn，格式对加密透明）：
#   [mask_k_bytes (uint8，bit per element) |
#    mask_v_bytes (uint8，bit per element) |
#    nonzero_k_values (raw float bytes)    |
#    nonzero_v_values (raw float bytes)]
#
# 重建：result = I_frame_tensor.clone(); result[mask] += delta_values
# ---------------------------------------------------------------------------

def _pack_sparse_delta(
    delta: np.ndarray,
    threshold: float,
) -> Tuple[bytes, bytes, int]:
    """
    将差分 tensor（numpy，任意形状）量化为稀疏格式。

    Returns
    -------
    mask_bytes  : bit-packed 掩码（uint8），1 表示该位置非零
    value_bytes : 非零值的原始字节（保持原始 dtype）
    nnz         : 非零元素数量
    """
    flat = delta.ravel()
    mask_bool = np.abs(flat) > threshold    # bool array [N]
    nnz = int(mask_bool.sum())

    # bit-pack 掩码（每 8 个 bool 压成 1 个 uint8）
    pad = (8 - len(mask_bool) % 8) % 8
    if pad:
        mask_bool = np.concatenate([mask_bool, np.zeros(pad, dtype=bool)])
    mask_uint8 = np.packbits(mask_bool)
    mask_bytes = mask_uint8.tobytes()

    # 仅存非零值（使用原始 dtype，不降精度）
    nz_vals = flat[mask_bool[:len(flat)]] if nnz > 0 else flat[:0]
    value_bytes = nz_vals.tobytes()

    return mask_bytes, value_bytes, nnz


def _unpack_sparse_delta(
    mask_bytes: bytes,
    value_bytes: bytes,
    shape: tuple,
    np_dtype,
    nnz: int,
) -> np.ndarray:
    """
    从稀疏格式重建完整的差分 numpy array（与参考帧形状相同）。
    """
    n_elements = int(np.prod(shape))
    mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_bool  = np.unpackbits(mask_uint8)[:n_elements].astype(bool)

    delta_flat = np.zeros(n_elements, dtype=np_dtype)
    if nnz > 0:
        nz_vals = np.frombuffer(value_bytes, dtype=np_dtype)
        delta_flat[mask_bool] = nz_vals
    return delta_flat.reshape(shape)


def compute_delta_compression_ratio(
    k: torch.Tensor,
    k_ref: torch.Tensor,
    threshold: float,
) -> float:
    """
    估算当前帧相对参考帧的差分稀疏率（非零比例），
    用于在 encode 时决定是否值得存 P 帧。

    返回 nnz / total（越小越稀疏，压缩率越高）。
    """
    delta = (k.float() - k_ref.float()).numpy()
    nnz = int(np.sum(np.abs(delta.ravel()) > threshold))
    return nnz / delta.size if delta.size > 0 else 1.0


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
            "block_type": "I",   # 关键帧
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

    def append_p_block(
        self,
        *,
        frame_index: int,
        layer_index: int,
        seq_start: int,
        seq_end: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        ref_value_tensor: torch.Tensor,
        ref_frame_index: int,
        delta_threshold: float = 1e-3,
        encrypt_fn=None,
    ) -> dict:
        """
        写入一个 PV 帧（V 差分帧）。

        K：施加过 RoPE，位置信息已 baked-in，帧间差异由位置差主导，完整存储。
        V：不施加 RoPE，纯语义内容，相邻静态帧高度相似，存稀疏差分。

        payload 布局（加密前）：
          [k_bytes (完整 K) | mask_v (bit-packed uint8) | nz_v_values (稀疏非零值)]

        预期压缩效果（静态/慢速视频）：
          V 差分 nnz 比例 5-15% -> 每个 PV 帧约为 I 帧的 52-57%
        """
        k = key_tensor.detach().to("cpu").contiguous()
        v = value_tensor.detach().to("cpu").float().contiguous()
        v_ref = ref_value_tensor.detach().to("cpu").float().contiguous()

        orig_dtype = str(key_tensor.dtype)
        np_dtype_k = _DTYPE_TO_NP.get(orig_dtype, np.float32)

        # K：完整字节（保持原始 dtype，不做任何差分）
        if orig_dtype == "torch.bfloat16":
            k_bytes = memoryview(k.view(torch.int16).numpy()).tobytes(order="C")
        else:
            k_bytes = memoryview(k.numpy()).tobytes(order="C")

        # V：float32 空间计算差分，稀疏化后存储
        delta_v = (v.numpy() - v_ref.numpy()).astype(np.float32)
        mask_v, nz_v, nnz_v = _pack_sparse_delta(delta_v, delta_threshold)

        orig_size = key_tensor.numel() * key_tensor.element_size() * 2
        pv_size = len(k_bytes) + len(mask_v) + len(nz_v)
        ratio = pv_size / orig_size if orig_size > 0 else 1.0

        header = {
            "frame_index":       int(frame_index),
            "layer_index":       int(layer_index),
            "seq_start":         int(seq_start),
            "seq_end":           int(seq_end),
            "dtype":             orig_dtype,
            "k_shape":           list(key_tensor.shape),
            "v_shape":           list(value_tensor.shape),
            "k_nbytes":          len(k_bytes),
            "encrypted":         False,
            "block_type":        "PV",
            "ref_frame":         int(ref_frame_index),
            "delta_threshold":   float(delta_threshold),
            "nnz_v":             int(nnz_v),
            "mask_v_bytes":      len(mask_v),
            "nz_v_bytes":        len(nz_v),
            "delta_dtype":       "float32",
            "compression_ratio": round(ratio, 4),
        }

        payload = k_bytes + mask_v + nz_v
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
            "offset":      int(offset),
            "header_len":  len(header_bytes),
            "payload_len": len(payload),
            "total_len":   int(BLOCK_HEAD.size + len(header_bytes) + len(payload)),
        }
        self.records.append(rec)
        return rec

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
        block_type = header.get("block_type", "I")   # 默认 I 帧（兼容旧数据）

        # ── I 帧：原有完整读取逻辑 ────────────────────────────────────────────
        if block_type == "I":
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
            payload_len_eff = len(kv_bytes)
            if payload_len_eff < k_nbytes:
                raise ValueError(
                    f"Decrypted payload smaller than key bytes: payload={payload_len_eff}, k_nbytes={k_nbytes}"
                )
            k_buf = np.frombuffer(kv_bytes, dtype=np_dtype, count=k_nbytes // np.dtype(np_dtype).itemsize, offset=0)
            v_off = k_nbytes
            v_count = (payload_len_eff - k_nbytes) // np.dtype(np_dtype).itemsize
            v_buf = np.frombuffer(kv_bytes, dtype=np_dtype, count=v_count, offset=v_off)

            k = torch.from_numpy(k_buf.copy()).reshape(header["k_shape"])
            v = torch.from_numpy(v_buf.copy()).reshape(header["v_shape"])
            if dtype_key == "torch.bfloat16":
                k = k.view(torch.bfloat16)
                v = v.view(torch.bfloat16)
            return k.to(map_location), v.to(map_location), header

        # ── PV 帧：K 完整读取，V 从参考帧重建 ───────────────────────────────
        elif block_type in ("PV", "P"):
            ref_frame = int(header["ref_frame"])
            # 仅递归读参考帧的 V（K 不需要参考帧）
            _, v_ref, _ = self.read_layer_frame(
                layer_index, ref_frame, map_location="cpu", decrypt_fn=decrypt_fn
            )

            off = int(rec["offset"])
            hlen = int(rec["header_len"])
            pstart = off + BLOCK_HEAD.size + hlen
            payload_len = int(rec["payload_len"])

            raw = self._mmap[pstart: pstart + payload_len]
            if bool(header.get("encrypted", False)):
                if decrypt_fn is None:
                    raise RuntimeError(
                        f"Encrypted PV-frame block requires decrypt_fn: "
                        f"layer={layer_index}, frame={frame_index}"
                    )
                raw = decrypt_fn(bytes(raw), header)
            else:
                raw = bytes(raw)

            dtype_key  = header["dtype"]
            np_dtype_k = _DTYPE_TO_NP.get(dtype_key, np.float32)
            k_nbytes   = int(header["k_nbytes"])
            mask_v_b   = int(header["mask_v_bytes"])
            k_shape    = tuple(header["k_shape"])
            v_shape    = tuple(header["v_shape"])

            # K：直接从 raw 读取完整字节，不做差分重建
            k_count = k_nbytes // np.dtype(np_dtype_k).itemsize
            k_buf = np.frombuffer(raw, dtype=np_dtype_k, count=k_count, offset=0)
            k = torch.from_numpy(k_buf.copy()).reshape(k_shape)
            if dtype_key == "torch.bfloat16":
                k = k.view(torch.bfloat16)

            # V：解包稀疏差分，与参考帧 V 相加重建
            mask_v_bytes = raw[k_nbytes : k_nbytes + mask_v_b]
            nz_v_bytes   = raw[k_nbytes + mask_v_b :]
            delta_v_np   = _unpack_sparse_delta(
                mask_v_bytes, nz_v_bytes, v_shape, np.float32, int(header["nnz_v"])
            )
            v_out = v_ref.float() + torch.from_numpy(delta_v_np.astype(np.float32))

            dtype_map = {
                "torch.float32":  torch.float32,
                "torch.float16":  torch.float16,
                "torch.bfloat16": torch.bfloat16,
            }
            target_dtype = dtype_map.get(dtype_key, torch.float32)
            v_out = v_out.to(target_dtype)

            return k.to(map_location), v_out.to(map_location), header

        else:
            raise ValueError(f"Unknown block_type={block_type!r} at layer={layer_index}, frame={frame_index}")


def has_kvpack(kv_cache_dir: str) -> bool:
    return os.path.exists(os.path.join(kv_cache_dir, "kvpack_index.json"))