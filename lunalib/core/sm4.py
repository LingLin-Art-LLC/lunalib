"""
SM4 block cipher with optional NumPy/CuPy acceleration for batch operations.
"""

from __future__ import annotations

import os
from lunalib.config import apply_profile
from typing import Iterable, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional
    cp = None

try:
    from .sm4_cuda.sm4_gpu import sm4_crypt_blocks_gpu, sm4_crypt_blocks_gpu_raw, CUDA_AVAILABLE as _SM4_CUDA_AVAILABLE
except Exception:
    sm4_crypt_blocks_gpu = None
    sm4_crypt_blocks_gpu_raw = None
    _SM4_CUDA_AVAILABLE = False


_SBOX = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
]

apply_profile()

_FK = [0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC]

_CK = [
    0x00070E15, 0x1C232A31, 0x383F464D, 0x545B6269,
    0x70777E85, 0x8C939AA1, 0xA8AFB6BD, 0xC4CBD2D9,
    0xE0E7EEF5, 0xFC030A11, 0x181F262D, 0x343B4249,
    0x50575E65, 0x6C737A81, 0x888F969D, 0xA4ABB2B9,
    0xC0C7CED5, 0xDCE3EAF1, 0xF8FF060D, 0x141B2229,
    0x30373E45, 0x4C535A61, 0x686F767D, 0x848B9299,
    0xA0A7AEB5, 0xBCC3CAD1, 0xD8DFE6ED, 0xF4FB0209,
    0x10171E25, 0x2C333A41, 0x484F565D, 0x646B7279,
]

_SBOX_NP = None
_SBOX_CP = None


def _rotl32(x, n, xp):
    return ((x << n) | (x >> (32 - n))) & xp.uint32(0xFFFFFFFF)


def _tau(x, xp, sbox):
    b0 = (x >> 24) & xp.uint32(0xFF)
    b1 = (x >> 16) & xp.uint32(0xFF)
    b2 = (x >> 8) & xp.uint32(0xFF)
    b3 = x & xp.uint32(0xFF)
    y0 = sbox[b0]
    y1 = sbox[b1]
    y2 = sbox[b2]
    y3 = sbox[b3]
    return (y0.astype(xp.uint32) << 24) | (y1.astype(xp.uint32) << 16) | (y2.astype(xp.uint32) << 8) | y3.astype(xp.uint32)


def _t(x, xp, sbox):
    b = _tau(x, xp, sbox)
    return b ^ _rotl32(b, 2, xp) ^ _rotl32(b, 10, xp) ^ _rotl32(b, 18, xp) ^ _rotl32(b, 24, xp)


def _t_key(x, xp, sbox):
    b = _tau(x, xp, sbox)
    return b ^ _rotl32(b, 13, xp) ^ _rotl32(b, 23, xp)


def _tau_scalar(x: int) -> int:
    b0 = _SBOX[(x >> 24) & 0xFF]
    b1 = _SBOX[(x >> 16) & 0xFF]
    b2 = _SBOX[(x >> 8) & 0xFF]
    b3 = _SBOX[x & 0xFF]
    return ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & 0xFFFFFFFF


def _t_scalar(x: int) -> int:
    b = _tau_scalar(x)
    r2 = ((b << 2) | (b >> 30)) & 0xFFFFFFFF
    r10 = ((b << 10) | (b >> 22)) & 0xFFFFFFFF
    r18 = ((b << 18) | (b >> 14)) & 0xFFFFFFFF
    r24 = ((b << 24) | (b >> 8)) & 0xFFFFFFFF
    return (b ^ r2 ^ r10 ^ r18 ^ r24) & 0xFFFFFFFF


def _t_key_scalar(x: int) -> int:
    b = _tau_scalar(x)
    r13 = ((b << 13) | (b >> 19)) & 0xFFFFFFFF
    r23 = ((b << 23) | (b >> 9)) & 0xFFFFFFFF
    return (b ^ r13 ^ r23) & 0xFFFFFFFF


def _expand_key(key: bytes):
    if len(key) != 16:
        raise ValueError("SM4 key must be 16 bytes")
    mk = [int.from_bytes(key[i:i+4], "big") for i in range(0, 16, 4)]
    k = [mk[i] ^ _FK[i] for i in range(4)]
    rk = []
    for i in range(32):
        tmp = k[i] ^ _t_key_scalar(k[i+1] ^ k[i+2] ^ k[i+3] ^ _CK[i])
        k.append(tmp)
        rk.append(tmp)
    return rk


def _crypt_blocks(blocks: bytes, rk: list[int], decrypt: bool = False, use_gpu: bool = False) -> bytes:
    if len(blocks) % 16 != 0:
        raise ValueError("Data length must be multiple of 16 bytes")

    blocks_count = len(blocks) // 16
    min_blocks = int(os.getenv("LUNALIB_SM4_MIN_BLOCKS", "8"))
    if blocks_count < min_blocks:
        out = bytearray()
        keys = rk[::-1] if decrypt else rk
        for i in range(0, len(blocks), 16):
            out.extend(_crypt_block(blocks[i:i+16], keys))
        return bytes(out)

    if use_gpu and os.getenv("LUNALIB_SM4_CUDA_KERNEL", "0") == "1" and _SM4_CUDA_AVAILABLE:
        try:
            return sm4_crypt_blocks_gpu(blocks, rk[::-1] if decrypt else rk)
        except Exception:
            pass

    if np is None:
        # Fallback to scalar path
        out = bytearray()
        keys = rk[::-1] if decrypt else rk
        for i in range(0, len(blocks), 16):
            out.extend(_crypt_block(blocks[i:i+16], keys))
        return bytes(out)

    xp = cp if (use_gpu and cp is not None) else np
    global _SBOX_NP, _SBOX_CP
    if xp is np:
        if _SBOX_NP is None:
            _SBOX_NP = np.asarray(_SBOX, dtype=np.uint8)
        sbox = _SBOX_NP
    else:
        if _SBOX_CP is None:
            _SBOX_CP = cp.asarray(_SBOX, dtype=cp.uint8)
        sbox = _SBOX_CP

    data = np.frombuffer(blocks, dtype=">u4").reshape(-1, 4)
    x = xp.asarray(data, dtype=xp.uint32)

    x0, x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    keys = rk[::-1] if decrypt else rk
    for r in keys:
        t = _t(x1 ^ x2 ^ x3 ^ xp.uint32(r), xp, sbox)
        x4 = x0 ^ t
        x0, x1, x2, x3 = x1, x2, x3, x4

    out_words = xp.stack([x3, x2, x1, x0], axis=1).astype(xp.uint32)
    if xp is cp:
        out_words = out_words.get()
    return out_words.astype(">u4").tobytes()


class _FakeNP:
    uint32 = int
    uint8 = int

    def __call__(self, value):
        return value




def _crypt_block(block: bytes, rk: list[int]) -> bytes:
    if len(block) != 16:
        raise ValueError("SM4 block must be 16 bytes")
    x = [int.from_bytes(block[i:i+4], "big") for i in range(0, 16, 4)]
    for r in rk:
        t = _t_scalar(x[1] ^ x[2] ^ x[3] ^ r)
        x = [x[1], x[2], x[3], x[0] ^ t]
    out = x[::-1]
    return b"".join(v.to_bytes(4, "big") for v in out)


class SM4Cipher:
    def __init__(self, key: bytes):
        self.key = key
        self.round_keys = _expand_key(key)

    def encrypt_block(self, block: bytes) -> bytes:
        return _crypt_block(block, self.round_keys)

    def decrypt_block(self, block: bytes) -> bytes:
        return _crypt_block(block, self.round_keys[::-1])

    def encrypt_ecb(self, data: bytes, use_gpu: bool = False) -> bytes:
        if not use_gpu and cp is not None and os.getenv("LUNALIB_SM4_USE_GPU", "0") == "1":
            use_gpu = True
        data = _pkcs7_pad(data)
        return _crypt_blocks(data, self.round_keys, decrypt=False, use_gpu=use_gpu)

    def decrypt_ecb(self, data: bytes, use_gpu: bool = False) -> bytes:
        if not use_gpu and cp is not None and os.getenv("LUNALIB_SM4_USE_GPU", "0") == "1":
            use_gpu = True
        plain = _crypt_blocks(data, self.round_keys, decrypt=True, use_gpu=use_gpu)
        return _pkcs7_unpad(plain)

    def encrypt_ctr(self, data: bytes, iv: bytes, use_gpu: bool = False) -> bytes:
        if not use_gpu and cp is not None and os.getenv("LUNALIB_SM4_USE_GPU", "0") == "1":
            use_gpu = True
        if len(iv) != 16:
            raise ValueError("IV must be 16 bytes")
        return _crypt_ctr(data, self.round_keys, iv, use_gpu=use_gpu)

    def decrypt_ctr(self, data: bytes, iv: bytes, use_gpu: bool = False) -> bytes:
        return self.encrypt_ctr(data, iv, use_gpu=use_gpu)


def sm4_encrypt_block(block: bytes, key: bytes) -> bytes:
    return SM4Cipher(key).encrypt_block(block)


def sm4_decrypt_block(block: bytes, key: bytes) -> bytes:
    return SM4Cipher(key).decrypt_block(block)


def sm4_encrypt_ecb(data: bytes, key: bytes, use_gpu: bool = False) -> bytes:
    return SM4Cipher(key).encrypt_ecb(data, use_gpu=use_gpu)


def sm4_decrypt_ecb(data: bytes, key: bytes, use_gpu: bool = False) -> bytes:
    return SM4Cipher(key).decrypt_ecb(data, use_gpu=use_gpu)


def sm4_encrypt_ctr(data: bytes, key: bytes, iv: bytes, use_gpu: bool = False) -> bytes:
    return SM4Cipher(key).encrypt_ctr(data, iv, use_gpu=use_gpu)


def sm4_decrypt_ctr(data: bytes, key: bytes, iv: bytes, use_gpu: bool = False) -> bytes:
    return SM4Cipher(key).decrypt_ctr(data, iv, use_gpu=use_gpu)


def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("Invalid padding")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("Invalid padding")
    return data[:-pad_len]


def _crypt_ctr(data: bytes, rk: list[int], iv: bytes, use_gpu: bool = False) -> bytes:
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")

    blocks = (len(data) + 15) // 16
    min_blocks = int(os.getenv("LUNALIB_SM4_CTR_MIN_BLOCKS", "8"))
    chunk_blocks = int(os.getenv("LUNALIB_SM4_CTR_CHUNK_BLOCKS", "131072"))
    counter = int.from_bytes(iv, "big")

    out = bytearray(len(data))
    offset = 0
    remaining = blocks
    use_gpu_xor = (
        use_gpu
        and cp is not None
        and _SM4_CUDA_AVAILABLE
        and os.getenv("LUNALIB_SM4_CTR_GPU_XOR", "0") == "1"
    )
    while remaining > 0:
        take = remaining if remaining < chunk_blocks else chunk_blocks
        if np is not None and take >= min_blocks:
            hi = (counter >> 64) & 0xFFFFFFFFFFFFFFFF
            lo = counter & 0xFFFFFFFFFFFFFFFF
            idx = np.arange(take, dtype=np.uint64)
            lo_arr = (np.uint64(lo) + idx).astype(np.uint64)
            carry = (lo_arr < np.uint64(lo)).astype(np.uint64)
            hi_arr = (np.uint64(hi) + carry) & np.uint64(0xFFFFFFFFFFFFFFFF)
            counters = np.empty((take, 2), dtype=">u8")
            counters[:, 0] = hi_arr
            counters[:, 1] = lo_arr
            counter_bytes = counters.tobytes()
        else:
            counters = []
            cur = counter
            for _ in range(take):
                counters.append(cur.to_bytes(16, "big"))
                cur = (cur + 1) & ((1 << 128) - 1)
            counter_bytes = b"".join(counters)

        if use_gpu_xor and sm4_crypt_blocks_gpu_raw is not None:
            ks_gpu = sm4_crypt_blocks_gpu_raw(counter_bytes, rk)
            chunk_len = min(len(data) - offset, take * 16)
            data_gpu = cp.frombuffer(data[offset:offset + chunk_len], dtype=cp.uint8)
            xor_gpu = cp.bitwise_xor(data_gpu, ks_gpu[:chunk_len])
            out[offset:offset + chunk_len] = cp.asnumpy(xor_gpu).tobytes()
        else:
            keystream = _crypt_blocks(counter_bytes, rk, decrypt=False, use_gpu=use_gpu)
            chunk_len = min(len(data) - offset, take * 16)
            out[offset:offset + chunk_len] = _xor_bytes(data[offset:offset + chunk_len], keystream[:chunk_len])

        counter = (counter + take) & ((1 << 128) - 1)
        offset += chunk_len
        remaining -= take

    return bytes(out)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("XOR inputs must have the same length")
    min_bytes = int(os.getenv("LUNALIB_SM4_XOR_MIN_BYTES", "4096"))
    if np is not None and len(a) >= min_bytes:
        arr_a = np.frombuffer(a, dtype=np.uint8)
        arr_b = np.frombuffer(b, dtype=np.uint8)
        return np.bitwise_xor(arr_a, arr_b).tobytes()
    return bytes(x ^ y for x, y in zip(a, b))
