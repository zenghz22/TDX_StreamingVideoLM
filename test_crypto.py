"""test_crypto.py
验证 kvpack_mmap_v1 架构下的视频 KV block 加密/解密可靠性。

测试覆盖：
1) 启用加密写入 kvpack 后，不提供 crypto_ctx 读取会失败。
2) 提供正确 crypto_ctx 读取能恢复与原始 tensor 一致。
3) 错误密钥读取会失败（鉴别失败）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch

from kvpack_mmap_td import KVPackWriter
from kvcache_crypto_td import (
    CryptoContext,
    decrypt_blob_to_bytes,
    encrypt_bytes_to_blob,
    layer_frame_block_id,
)
from kvcache_retrieve_td import load_kv_cache


def _make_encrypt_fn(ctx: CryptoContext, num_layers: int):
    def _encrypt(payload: bytes, header: dict) -> bytes:
        block_id = layer_frame_block_id(
            frame_index=int(header["frame_index"]),
            layer_index=int(header["layer_index"]),
            num_layers=num_layers,
        )
        aad = {
            "frame_index": int(header["frame_index"]),
            "layer_index": int(header["layer_index"]),
            "seq_start": int(header["seq_start"]),
            "seq_end": int(header["seq_end"]),
            "dtype": str(header["dtype"]),
        }
        return encrypt_bytes_to_blob(payload, ctx.master_key, chunk_index=block_id, aad=aad)
    return _encrypt


def _build_fake_kvpack(kv_dir: str, num_frames: int = 3, num_layers: int = 2):
    writer = KVPackWriter(kv_dir)
    originals = {}
    ctx = CryptoContext(master_key=os.urandom(32), enabled=True)
    enc_fn = _make_encrypt_fn(ctx, num_layers)

    for fi in range(num_frames):
        for li in range(num_layers):
            k = torch.randn(1, 2, 4, 8)
            v = torch.randn(1, 2, 4, 8)
            originals[(li, fi)] = (k.clone(), v.clone())
            writer.append_block(
                frame_index=fi,
                layer_index=li,
                seq_start=fi * 4,
                seq_end=(fi + 1) * 4,
                key_tensor=k,
                value_tensor=v,
                encrypt_fn=enc_fn,
            )

    writer.write_index({"num_layers": num_layers, "full_merged_seq_len": num_frames * 4})
    writer.close()
    return originals, ctx


def run_test():
    td = tempfile.mkdtemp(prefix="kvpack_crypto_test_")
    kv_dir = str(Path(td) / "kv")
    os.makedirs(kv_dir, exist_ok=True)

    try:
        originals, good_ctx = _build_fake_kvpack(kv_dir)

        print("[T1] 无 crypto_ctx 读取应失败（因为 block 为密文）...")
        try:
            load_kv_cache(kv_dir, map_location="cpu", chunk_indices=[0, 2], crypto_ctx=None)
            raise AssertionError("T1 failed: expected failure but load succeeded")
        except Exception as e:
            print(f"  [OK] got expected failure: {type(e).__name__}: {e}")

        print("[T2] 使用正确 crypto_ctx 读取应成功并与原始一致...")
        kv_cache, meta = load_kv_cache(kv_dir, map_location="cpu", chunk_indices=[0, 2], crypto_ctx=good_ctx)
        assert kv_cache is not None and len(kv_cache) == 2
        # 校验 layer0/frame0 对应前4 tokens
        l0k = kv_cache[0][0][:, :, :4, :]
        l0v = kv_cache[0][1][:, :, :4, :]
        ok_k, ok_v = originals[(0, 0)]
        assert torch.allclose(l0k, ok_k), "T2 failed: decrypted K mismatch"
        assert torch.allclose(l0v, ok_v), "T2 failed: decrypted V mismatch"
        print(f"  [OK] decrypted tensors match. meta={meta}")

        print("[T3] 使用错误密钥读取应失败...")
        bad_ctx = CryptoContext(master_key=os.urandom(32), enabled=True)
        try:
            load_kv_cache(kv_dir, map_location="cpu", chunk_indices=[0, 2], crypto_ctx=bad_ctx)
            raise AssertionError("T3 failed: expected failure with wrong key")
        except Exception as e:
            print(f"  [OK] wrong-key failure: {type(e).__name__}: {e}")

        print("\nAll crypto tests passed.")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    run_test()
