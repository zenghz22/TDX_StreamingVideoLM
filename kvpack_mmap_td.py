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

__all__ = [
    "KVPackWriter",
    "KVPackReader",
    "has_kvpack",
    "BLOCK_MAGIC",
    "BLOCK_HEAD",
]


# ---------------------------------------------------------------------------
# Mainline mode: only I-frame KV blocks are supported.
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

def has_kvpack(kv_cache_dir: str) -> bool:
    return os.path.exists(os.path.join(kv_cache_dir, "kvpack_index.json"))
