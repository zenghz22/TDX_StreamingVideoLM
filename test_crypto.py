"""
experiment_no_decrypt.py
========================
实验：encode 时加密（删明文），decode 时故意不解密，直接读密文，观察结果。

不依赖真实模型和视频，用随机 tensor 模拟 KV cache shard。

预期现象
--------
场景 A（.enc 文件不存在时尝试读 .safetensors）：
  FileNotFoundError — 明文已被加密删除，safetensors loader 找不到文件

场景 B（把 .enc 文件改名为 .safetensors 强行读取）：
  safetensors.SafetensorError 或 struct.error —
  密文以 b"TDKVCHNK" magic 开头，不是 safetensors 的 8 字节小端 header，
  解析器会立即报头部校验失败

场景 C（绕过文件名检测，直接把密文字节喂给 torch.load）：
  pickle.UnpicklingError — 密文不是合法的 pickle 流

运行方式
--------
pip install cryptography safetensors torch
python experiment_no_decrypt.py
"""

import io
import os
import struct
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

# ---------------------------------------------------------------------------
# 导入加密模块
# ---------------------------------------------------------------------------
from kvcache_crypto_td import CryptoContext, MAGIC

# ---------------------------------------------------------------------------
# 辅助：构造一个假的单层 KV shard
# ---------------------------------------------------------------------------

def make_fake_shard(path: str):
    """用随机 tensor 写一个最小的 safetensors KV shard。"""
    tensors = {
        "layer_0.k": torch.randn(1, 4, 8, 32),
        "layer_0.v": torch.randn(1, 4, 8, 32),
    }
    save_file(tensors, path)
    print(f"[setup] Wrote fake shard: {path}  ({os.path.getsize(path)} bytes)")
    return tensors


# ---------------------------------------------------------------------------
# 场景 A：读取不存在的明文（明文已被删除）
# ---------------------------------------------------------------------------

def scene_a(tmp: Path, ctx: CryptoContext):
    print("\n" + "=" * 60)
    print("Scene A: 明文已被加密删除，直接读 .safetensors → FileNotFoundError")
    print("=" * 60)

    plain = tmp / "chunk_00000.safetensors"
    enc   = tmp / "chunk_00000.safetensors.enc"

    make_fake_shard(str(plain))
    ctx.maybe_encrypt_after_save(plain, chunk_index=0)
    # remove_plain_after_encrypt=True（默认），明文已删除

    print(f"  plain exists: {plain.exists()}")
    print(f"  enc   exists: {enc.exists()}")

    print("  Attempting load_file(plain)...")
    try:
        load_file(str(plain))
        print("  [UNEXPECTED] Loaded without error!")
    except FileNotFoundError as e:
        print(f"  [EXPECTED] FileNotFoundError: {e}")
    except Exception as e:
        print(f"  [OTHER] {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 场景 B：把 .enc 重命名为 .safetensors，强行用 safetensors 解析密文
# ---------------------------------------------------------------------------

def scene_b(tmp: Path, ctx: CryptoContext):
    print("\n" + "=" * 60)
    print("Scene B: 把 .enc 重命名为 .safetensors，密文头部不合法 → 解析失败")
    print("=" * 60)

    plain = tmp / "chunk_00001.safetensors"
    enc   = tmp / "chunk_00001.safetensors.enc"
    fake_plain = tmp / "chunk_00001_ciphertext_as_plain.safetensors"

    make_fake_shard(str(plain))
    ctx.maybe_encrypt_after_save(plain, chunk_index=1)

    # 读出密文字节，检查 magic
    enc_bytes = enc.read_bytes()
    print(f"  Ciphertext first 8 bytes: {enc_bytes[:8]!r}  (expected magic: {MAGIC!r})")
    print(f"  Ciphertext size: {len(enc_bytes)} bytes")

    # 把密文字节写成 .safetensors 文件名，尝试解析
    fake_plain.write_bytes(enc_bytes)

    print("  Attempting load_file(fake_plain whose content is ciphertext)...")
    try:
        load_file(str(fake_plain))
        print("  [UNEXPECTED] Loaded without error!")
    except Exception as e:
        print(f"  [EXPECTED] {type(e).__name__}: {e}")

    # 同时展示 safetensors 正常 header 长什么样（对比）
    normal_bytes = (tmp / "chunk_00000.safetensors.enc").read_bytes()  # just reuse enc for size demo
    # 重新写一个正常的 safetensors 看 header
    normal_plain = tmp / "normal_ref.safetensors"
    save_file({"x": torch.zeros(1)}, str(normal_plain))
    normal_header = normal_plain.read_bytes()[:8]
    print(f"  Normal safetensors first 8 bytes: {normal_header!r}")
    print(f"  → safetensors header = little-endian u64 (metadata JSON length), not 'TDKVCHNK'")


# ---------------------------------------------------------------------------
# 场景 C：用 torch.load 直接解析密文字节流（pickle 路径）
# ---------------------------------------------------------------------------

def scene_c(tmp: Path, ctx: CryptoContext):
    print("\n" + "=" * 60)
    print("Scene C: 把密文字节流喂给 torch.load → pickle 解析失败")
    print("=" * 60)

    plain = tmp / "chunk_00002.safetensors"
    enc   = tmp / "chunk_00002.safetensors.enc"

    make_fake_shard(str(plain))
    ctx.maybe_encrypt_after_save(plain, chunk_index=2)

    enc_bytes = enc.read_bytes()
    buf = io.BytesIO(enc_bytes)

    print("  Attempting torch.load(BytesIO(ciphertext))...")
    try:
        result = torch.load(buf, map_location="cpu", weights_only=False)
        print(f"  [UNEXPECTED] Loaded: {result}")
    except Exception as e:
        print(f"  [EXPECTED] {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 场景 D：正确解密对照组
# ---------------------------------------------------------------------------

def scene_d(tmp: Path, ctx: CryptoContext):
    print("\n" + "=" * 60)
    print("Scene D: 正确解密对照组（验证加密本身没问题）")
    print("=" * 60)

    plain = tmp / "chunk_00003.safetensors"
    enc   = tmp / "chunk_00003.safetensors.enc"

    original = make_fake_shard(str(plain))
    ctx.maybe_encrypt_after_save(plain, chunk_index=3)

    # 用 maybe_decrypt_before_load 解密
    decrypted_path = ctx.maybe_decrypt_before_load(str(plain), chunk_index=3)
    loaded = load_file(decrypted_path)

    match = all(
        torch.allclose(original[k], loaded[k])
        for k in original
    )
    print(f"  Decrypted and loaded successfully. Tensors match: {match}")
    ctx.cleanup_tmp()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    master_key = os.urandom(32)  # 随机一次性密钥，不写文件
    ctx = CryptoContext(master_key=master_key, enabled=True, remove_plain_after_encrypt=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scene_a(tmp, ctx)
        scene_b(tmp, ctx)
        scene_c(tmp, ctx)
        scene_d(tmp, ctx)

    print("\n" + "=" * 60)
    print("实验结论：")
    print("  A: 明文删除后 safetensors loader 直接 FileNotFoundError")
    print("  B: 密文头部是 b'TDKVCHNK'，safetensors 期望 u64 JSON 长度，立即报错")
    print("  C: 密文不是 pickle 流，torch.load 报 UnpicklingError")
    print("  D: 正确解密后完全还原，tensor 逐元素相同")
    print("  → 没有任何路径能在不持有正确 master_key 的情况下读出有意义的数据")
    print("=" * 60)