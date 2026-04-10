'''对分块的离线KVcache进行加密与解密'''
"""kvcache_crypt_td.py

A minimal-intrusion encryption/decryption helper for video chunk KV-cache shards.

Design goals
------------
- Only touch video chunk KV-cache files (e.g. chunk_00001.safetensors).
- Keep the integration surface small:
    * after save_kv_cache(...) -> encrypt_chunk_file(...)
    * before load/reuse      -> decrypt_chunk_file(...)
- Support local-restorable keys (single-machine restore).
- Be easy to swap to a TDX sealing-key-backed master key later.

Recommended usage in your pipeline
----------------------------------
Write path:
    save_kv_cache(delta_kv, shard_path, ...)
    encrypt_chunk_file(shard_path, shard_path + ".enc", master_key, aad=...)
    os.remove(shard_path)

Read path:
    decrypt_chunk_file(enc_path, tmp_plain_path, master_key, expected_aad=...)
    # then load tmp_plain_path with safetensors/torch logic

File format
-----------
Encrypted file format is binary and self-contained:

    magic      : 8 bytes   b"TDKVCHNK"
    version    : u16       currently 1
    flags      : u16       reserved
    nonce_len  : u16       AEAD nonce length
    aad_len    : u32
    ct_len     : u64
    nonce      : bytes
    aad        : bytes
    ciphertext : bytes

Cryptography
------------
- AEAD: AES-256-GCM
- Key derivation: HKDF-SHA256 for per-chunk keys
- AAD: should bind chunk_index / seq range / model hash / prefix hash / dtype, etc.

Dependencies
------------
- cryptography
- torch (optional, only for in-memory helpers)

Notes
-----
- This module intentionally does NOT handle non-video caches or other files.
- In TDX, replace the master-key provider with a sealing-key-derived secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
except Exception as e:  # pragma: no cover
    AESGCM = None  # type: ignore
    HKDF = None  # type: ignore
    hashes = None  # type: ignore
    _CRYPTO_IMPORT_ERROR = e
else:
    _CRYPTO_IMPORT_ERROR = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

MAGIC = b"TDKVCHNK"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sHHHIQ")
# fields: magic, version, flags, nonce_len, aad_len, ct_len
DEFAULT_NONCE_LEN = 12
DEFAULT_KEY_LEN = 32  # AES-256


class KVCryptoError(RuntimeError):
    """Raised when encryption/decryption fails."""


@dataclass(frozen=True)
class ChunkAAD:
    """Recommended authenticated metadata for a chunk."""

    chunk_index: int
    seq_start: int = 0
    seq_end: int = 0
    chunk_size: int = 0
    num_frames: int = 0
    model_name_or_path: str = ""
    prefix_text: str = ""
    is_delta_chunk: bool = True
    version: str = "v1"

    def to_bytes(self) -> bytes:
        payload = {
            "chunk_index": self.chunk_index,
            "seq_start": self.seq_start,
            "seq_end": self.seq_end,
            "chunk_size": self.chunk_size,
            "num_frames": self.num_frames,
            "model_name_or_path": self.model_name_or_path,
            "prefix_text": self.prefix_text,
            "is_delta_chunk": self.is_delta_chunk,
            "version": self.version,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _require_crypto() -> None:
    if AESGCM is None or HKDF is None or hashes is None:
        raise KVCryptoError(
            "cryptography is required for kvcache_crypt_td.py. "
            f"Import error: {_CRYPTO_IMPORT_ERROR!r}"
        )


def load_or_create_local_master_key(key_path: Union[str, os.PathLike], *, create: bool = True) -> bytes:
    """Load a persistent local master key for single-machine restore.

    This is suitable for development/testing. In TDX, replace this function
    with a sealing-key-backed key loader.
    """
    key_path = Path(key_path)
    if key_path.exists():
        key = key_path.read_bytes()
        if len(key) != DEFAULT_KEY_LEN:
            raise KVCryptoError(f"Invalid key length in {key_path}: expected {DEFAULT_KEY_LEN}, got {len(key)}")
        return key

    if not create:
        raise FileNotFoundError(str(key_path))

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(DEFAULT_KEY_LEN)
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass
    return key


def derive_chunk_key(master_key: bytes, *, chunk_index: int, salt: Optional[bytes] = None, info: Optional[bytes] = None) -> bytes:
    """Derive a per-chunk AES key from a master key."""
    _require_crypto()
    if len(master_key) not in (16, 24, 32):
        raise KVCryptoError(f"master_key must be 16/24/32 bytes, got {len(master_key)}")

    if salt is None:
        # Deterministic but domain-separated salt for a given chunk index.
        salt = hashlib.sha256(f"kvcache-chunk-salt:{chunk_index}".encode("utf-8")).digest()
    if info is None:
        info = f"kvcache-chunk-key:{chunk_index}".encode("utf-8")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=DEFAULT_KEY_LEN,
        salt=salt,
        info=info,
    )
    return hkdf.derive(master_key)


def derive_master_key_from_seed(seed: bytes, *, context: str = "kvcache-td") -> bytes:
    """Derive a 32-byte master key from an arbitrary seed.

    Useful when your runtime can provide a per-boot secret, or when migrating
    the source of the master key to a TDX sealing secret.
    """
    _require_crypto()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=DEFAULT_KEY_LEN,
        salt=hashlib.sha256((context + ":salt").encode("utf-8")).digest(),
        info=(context + ":master-key").encode("utf-8"),
    )
    return hkdf.derive(seed)


# ---------------------------------------------------------------------------
# Generic AEAD file envelope
# ---------------------------------------------------------------------------

def _encrypt_bytes(
    plaintext: bytes,
    master_key: bytes,
    *,
    chunk_index: int,
    aad: bytes,
    nonce: Optional[bytes] = None,
) -> Tuple[bytes, bytes, bytes]:
    _require_crypto()
    if nonce is None:
        nonce = os.urandom(DEFAULT_NONCE_LEN)
    if len(nonce) != DEFAULT_NONCE_LEN:
        raise KVCryptoError(f"nonce must be {DEFAULT_NONCE_LEN} bytes, got {len(nonce)}")

    key = derive_chunk_key(master_key, chunk_index=chunk_index)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, aad, ciphertext


def _decrypt_bytes(
    nonce: bytes,
    aad: bytes,
    ciphertext: bytes,
    master_key: bytes,
    *,
    chunk_index: int,
) -> bytes:
    _require_crypto()
    key = derive_chunk_key(master_key, chunk_index=chunk_index)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def encrypt_bytes_to_blob(
    plaintext: bytes,
    master_key: bytes,
    *,
    chunk_index: int,
    aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
    nonce: Optional[bytes] = None,
) -> bytes:
    """Encrypt arbitrary bytes into a self-contained binary blob."""
    if aad is None:
        aad_bytes = b""
    elif isinstance(aad, ChunkAAD):
        aad_bytes = aad.to_bytes()
    elif isinstance(aad, dict):
        aad_bytes = json.dumps(aad, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif isinstance(aad, bytes):
        aad_bytes = aad
    else:
        raise TypeError("aad must be bytes, dict, ChunkAAD, or None")

    nonce, aad_bytes, ciphertext = _encrypt_bytes(plaintext, master_key, chunk_index=chunk_index, aad=aad_bytes, nonce=nonce)

    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        0,
        len(nonce),
        len(aad_bytes),
        len(ciphertext),
    )
    return header + nonce + aad_bytes + ciphertext


def decrypt_blob_to_bytes(
    blob: bytes,
    master_key: bytes,
    *,
    expected_chunk_index: int,
    expected_aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
) -> bytes:
    """Decrypt a blob produced by encrypt_bytes_to_blob()."""
    if len(blob) < HEADER_STRUCT.size:
        raise KVCryptoError("Encrypted blob too small")

    magic, version, _flags, nonce_len, aad_len, ct_len = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise KVCryptoError("Invalid magic header")
    if version != VERSION:
        raise KVCryptoError(f"Unsupported version: {version}")

    offset = HEADER_STRUCT.size
    nonce = blob[offset : offset + nonce_len]
    offset += nonce_len
    aad_bytes = blob[offset : offset + aad_len]
    offset += aad_len
    ciphertext = blob[offset : offset + ct_len]

    if len(nonce) != nonce_len or len(aad_bytes) != aad_len or len(ciphertext) != ct_len:
        raise KVCryptoError("Encrypted blob is truncated")

    if expected_aad is not None:
        if isinstance(expected_aad, ChunkAAD):
            expected = expected_aad.to_bytes()
        elif isinstance(expected_aad, dict):
            expected = json.dumps(expected_aad, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif isinstance(expected_aad, bytes):
            expected = expected_aad
        else:
            raise TypeError("expected_aad must be bytes, dict, ChunkAAD, or None")
        if expected != aad_bytes:
            raise KVCryptoError("AAD mismatch (refusing to decrypt)")

    return _decrypt_bytes(nonce, aad_bytes, ciphertext, master_key, chunk_index=expected_chunk_index)


# ---------------------------------------------------------------------------
# File helpers: this is what you will likely call from the existing pipeline.
# ---------------------------------------------------------------------------

def encrypt_chunk_file(
    src_plain_path: Union[str, os.PathLike],
    dst_enc_path: Union[str, os.PathLike],
    master_key: bytes,
    *,
    chunk_index: int,
    aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
    remove_source: bool = False,
) -> Dict[str, Any]:
    """Encrypt a chunk shard file (e.g. safetensors) into an encrypted file."""
    src_plain_path = Path(src_plain_path)
    dst_enc_path = Path(dst_enc_path)

    plaintext = src_plain_path.read_bytes()
    blob = encrypt_bytes_to_blob(plaintext, master_key, chunk_index=chunk_index, aad=aad)

    dst_enc_path.parent.mkdir(parents=True, exist_ok=True)
    dst_enc_path.write_bytes(blob)
    try:
        os.chmod(dst_enc_path, 0o600)
    except Exception:
        pass

    if remove_source:
        try:
            src_plain_path.unlink()
        except FileNotFoundError:
            pass

    return {
        "src_plain_path": str(src_plain_path),
        "dst_enc_path": str(dst_enc_path),
        "chunk_index": chunk_index,
        "plain_size": len(plaintext),
        "enc_size": len(blob),
    }


def decrypt_chunk_file(
    src_enc_path: Union[str, os.PathLike],
    dst_plain_path: Union[str, os.PathLike],
    master_key: bytes,
    *,
    expected_chunk_index: int,
    expected_aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
    remove_source: bool = False,
) -> Dict[str, Any]:
    """Decrypt an encrypted chunk file back to plaintext file."""
    src_enc_path = Path(src_enc_path)
    dst_plain_path = Path(dst_plain_path)

    blob = src_enc_path.read_bytes()
    plaintext = decrypt_blob_to_bytes(
        blob,
        master_key,
        expected_chunk_index=expected_chunk_index,
        expected_aad=expected_aad,
    )

    dst_plain_path.parent.mkdir(parents=True, exist_ok=True)
    dst_plain_path.write_bytes(plaintext)
    try:
        os.chmod(dst_plain_path, 0o600)
    except Exception:
        pass

    if remove_source:
        try:
            src_enc_path.unlink()
        except FileNotFoundError:
            pass

    return {
        "src_enc_path": str(src_enc_path),
        "dst_plain_path": str(dst_plain_path),
        "expected_chunk_index": expected_chunk_index,
        "plain_size": len(plaintext),
        "enc_size": len(blob),
    }


# ---------------------------------------------------------------------------
# In-memory helpers for direct integration (optional)
# ---------------------------------------------------------------------------

def encrypt_kv_cache_obj(
    kv_cache: Any,
    master_key: bytes,
    *,
    chunk_index: int,
    aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
) -> bytes:
    """Encrypt a Python KV cache object by serializing it with torch.save.

    This is optional. For minimal intrusion, file-level encryption is usually
    better because your current pipeline already writes safetensors chunk shards.
    """
    if torch is None:
        raise KVCryptoError("torch is required for in-memory KV cache helpers")

    import io

    buffer = io.BytesIO()
    torch.save(kv_cache, buffer)
    return encrypt_bytes_to_blob(buffer.getvalue(), master_key, chunk_index=chunk_index, aad=aad)


def decrypt_kv_cache_blob(
    blob: bytes,
    master_key: bytes,
    *,
    expected_chunk_index: int,
    expected_aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
) -> Any:
    """Decrypt a blob produced by encrypt_kv_cache_obj() back into a KV cache object."""
    if torch is None:
        raise KVCryptoError("torch is required for in-memory KV cache helpers")

    import io

    plaintext = decrypt_blob_to_bytes(
        blob,
        master_key,
        expected_chunk_index=expected_chunk_index,
        expected_aad=expected_aad,
    )
    buffer = io.BytesIO(plaintext)
    return torch.load(buffer, map_location="cpu")


# ---------------------------------------------------------------------------
# Convenience: file-name helpers for your chunk layout
# ---------------------------------------------------------------------------

def chunk_plain_path(kv_cache_dir: Union[str, os.PathLike], chunk_index: int) -> str:
    return str(Path(kv_cache_dir) / f"chunk_{chunk_index:05d}.safetensors")


def chunk_enc_path(kv_cache_dir: Union[str, os.PathLike], chunk_index: int) -> str:
    return str(Path(kv_cache_dir) / f"chunk_{chunk_index:05d}.safetensors.enc")


def encrypt_chunk_by_index(
    kv_cache_dir: Union[str, os.PathLike],
    chunk_index: int,
    master_key: bytes,
    *,
    aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
    remove_source: bool = True,
) -> Dict[str, Any]:
    plain = chunk_plain_path(kv_cache_dir, chunk_index)
    enc = chunk_enc_path(kv_cache_dir, chunk_index)
    return encrypt_chunk_file(plain, enc, master_key, chunk_index=chunk_index, aad=aad, remove_source=remove_source)


def decrypt_chunk_by_index(
    kv_cache_dir: Union[str, os.PathLike],
    chunk_index: int,
    master_key: bytes,
    *,
    aad: Union[bytes, ChunkAAD, Dict[str, Any], None] = None,
    remove_source: bool = False,
    write_plain_suffix: str = ".plain",
) -> Dict[str, Any]:
    enc = chunk_enc_path(kv_cache_dir, chunk_index)
    plain = str(Path(kv_cache_dir) / f"chunk_{chunk_index:05d}.safetensors{write_plain_suffix}")
    return decrypt_chunk_file(enc, plain, master_key, expected_chunk_index=chunk_index, expected_aad=aad, remove_source=remove_source)


# ---------------------------------------------------------------------------
# Optional self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    if torch is None:
        raise SystemExit("torch is required for self-test")

    import tempfile

    master = os.urandom(32)
    sample = {
        "layer_0.k": torch.randn(1, 2, 3, 4),
        "layer_0.v": torch.randn(1, 2, 3, 4),
    }
    aad = ChunkAAD(chunk_index=7, seq_start=64, seq_end=128, chunk_size=64, num_frames=256, model_name_or_path="demo")

    with tempfile.TemporaryDirectory() as td:
        plain = Path(td) / "chunk_00007.safetensors"
        enc = Path(td) / "chunk_00007.safetensors.enc"
        # lazy import to avoid adding safetensors dependency at import time
        from safetensors.torch import save_file, load_file

        save_file(sample, str(plain))
        encrypt_chunk_file(plain, enc, master, chunk_index=7, aad=aad, remove_source=False)
        decrypt_chunk_file(enc, plain.with_suffix(".dec.safetensors"), master, expected_chunk_index=7, expected_aad=aad)
        loaded = load_file(str(plain.with_suffix(".dec.safetensors")))
        assert torch.allclose(loaded["layer_0.k"], sample["layer_0.k"])
        assert torch.allclose(loaded["layer_0.v"], sample["layer_0.v"])
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
