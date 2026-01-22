import os
from typing import Iterable, List

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    cp = None
    CUDA_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
KERNEL_PATH = os.path.join(BASE_DIR, "sm3_kernel.cu")
CONSTANTS_PATH = os.path.join(BASE_DIR, "sm3_kernel_constants.h")
UTILS_PATH = os.path.join(BASE_DIR, "sm3_kernel_utils.cu")


def _load_kernel(name: str) -> "cp.RawKernel":
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    with open(CONSTANTS_PATH, "r", encoding="utf-8") as f:
        constants_src = f.read()
    with open(UTILS_PATH, "r", encoding="utf-8") as f:
        utils_src = f.read()
    with open(KERNEL_PATH, "r", encoding="utf-8") as f:
        kernel_src = f.read()
    kernel_src = kernel_src.replace('#include "sm3_kernel_constants.h"', '')
    kernel_src = kernel_src.replace('#include "sm3_kernel_utils.cu"', '')
    src = "\n".join([constants_src, utils_src, kernel_src])
    return cp.RawKernel(src, name)


def _pad_message(message: bytes) -> bytes:
    """Pad a message to SM3 512-bit blocks."""
    bit_len = len(message) * 8
    padded = message + b"\x80"
    while (len(padded) % 64) != 56:
        padded += b"\x00"
    padded += bit_len.to_bytes(8, "big")
    return padded


def gpu_sm3_hash_blocks(messages: Iterable[bytes]) -> List[bytes]:
    """Hash a batch of messages on the GPU using SM3 (single-block optimized)."""
    return gpu_sm3_hash_messages(messages)


def gpu_sm3_hash_messages(messages: Iterable[bytes]) -> List[bytes]:
    """Hash a batch of messages (multi-block supported) on the GPU using SM3.

    Returns a list of 32-byte hashes.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")

    msgs = list(messages)
    if not msgs:
        return []

    kernel = _load_kernel("sm3_compress")

    padded_blocks = [
        [p[i:i+64] for i in range(0, len(p), 64)]
        for p in (_pad_message(m) for m in msgs)
    ]

    max_blocks = max(len(b) for b in padded_blocks)

    ivs = [
        [0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
         0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E]
        for _ in msgs
    ]

    threads = 256

    for block_index in range(max_blocks):
        active_indices = [i for i, blocks in enumerate(padded_blocks) if block_index < len(blocks)]
        if not active_indices:
            continue

        block_bytes = b"".join(padded_blocks[i][block_index] for i in active_indices)
        iv_in = [word for i in active_indices for word in ivs[i]]

        blocks_gpu = cp.asarray(bytearray(block_bytes), dtype=cp.uint8)
        iv_in_gpu = cp.asarray(iv_in, dtype=cp.uint32)
        iv_out_gpu = cp.empty(len(active_indices) * 8, dtype=cp.uint32)

        blocks_per_grid = (len(active_indices) + threads - 1) // threads
        kernel((blocks_per_grid,), (threads,), (blocks_gpu, iv_in_gpu, iv_out_gpu, len(active_indices)))

        iv_out = cp.asnumpy(iv_out_gpu).tolist()
        for idx_pos, msg_index in enumerate(active_indices):
            base = idx_pos * 8
            ivs[msg_index] = iv_out[base:base+8]

    results = []
    for iv in ivs:
        results.append(b"".join(int(v).to_bytes(4, "big") for v in iv))

    return results
