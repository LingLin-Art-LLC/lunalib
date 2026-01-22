import os
from typing import List

try:
    import cupy as cp  # type: ignore
    CUDA_AVAILABLE = True
except Exception:
    cp = None
    CUDA_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
KERNEL_PATH = os.path.join(BASE_DIR, "sm4_kernel.cu")
_KERNEL = None


def _load_kernel(name: str) -> "cp.RawKernel":
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    global _KERNEL
    if _KERNEL is None:
        with open(KERNEL_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        _KERNEL = cp.RawKernel(src, name)
    return _KERNEL


def sm4_crypt_blocks_gpu_raw(blocks: bytes, rk: List[int]) -> "cp.ndarray":
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    if len(blocks) % 16 != 0:
        raise ValueError("Data length must be multiple of 16 bytes")

    kernel = _load_kernel("sm4_encrypt_kernel")
    rk_gpu = cp.asarray(rk, dtype=cp.uint32)
    chunk_bytes = int(os.getenv("LUNALIB_SM4_GPU_CHUNK_BYTES", "0"))

    if chunk_bytes <= 0 or len(blocks) <= chunk_bytes:
        blocks_count = len(blocks) // 16
        in_gpu = cp.asarray(bytearray(blocks), dtype=cp.uint8)
        out_gpu = cp.empty_like(in_gpu)
        threads = 128
        blocks_grid = (blocks_count + threads - 1) // threads
        kernel((blocks_grid,), (threads,), (in_gpu, rk_gpu, out_gpu, blocks_count))
        return out_gpu

    out = bytearray(len(blocks))
    offset = 0
    total = len(blocks)
    threads = 128
    while offset < total:
        chunk = blocks[offset:offset + chunk_bytes]
        blocks_count = len(chunk) // 16
        in_gpu = cp.asarray(bytearray(chunk), dtype=cp.uint8)
        out_gpu = cp.empty_like(in_gpu)
        blocks_grid = (blocks_count + threads - 1) // threads
        kernel((blocks_grid,), (threads,), (in_gpu, rk_gpu, out_gpu, blocks_count))
        out[offset:offset + len(chunk)] = cp.asnumpy(out_gpu).tobytes()
        offset += len(chunk)

    return cp.asarray(out, dtype=cp.uint8)


def sm4_crypt_blocks_gpu(blocks: bytes, rk: List[int]) -> bytes:
    out_gpu = sm4_crypt_blocks_gpu_raw(blocks, rk)
    return cp.asnumpy(out_gpu).tobytes()
