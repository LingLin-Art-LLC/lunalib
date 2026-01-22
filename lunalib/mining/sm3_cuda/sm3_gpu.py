import os
import time
from typing import Iterable, List, Optional

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
    iterations = int(os.getenv("LUNALIB_CUDA_HASHES_PER_THREAD", "1"))
    if iterations < 1:
        iterations = 1

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
        kernel((blocks_per_grid,), (threads,), (blocks_gpu, iv_in_gpu, iv_out_gpu, len(active_indices), int(iterations)))

        iv_out = cp.asnumpy(iv_out_gpu).tolist()
        for idx_pos, msg_index in enumerate(active_indices):
            base = idx_pos * 8
            ivs[msg_index] = iv_out[base:base+8]

    results = []
    for iv in ivs:
        results.append(b"".join(int(v).to_bytes(4, "big") for v in iv))

    return results


def gpu_sm3_mine_compact(
    base80: bytes,
    start_nonce: int,
    count: int,
    difficulty: int,
    iterations: Optional[int] = None,
    hashes_per_thread: Optional[int] = None,
) -> Optional[int]:
    """Mine using compact 88-byte header (80-byte base + 8-byte nonce) on GPU."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    if len(base80) != 80:
        raise ValueError("base80 must be exactly 80 bytes")
    if count <= 0:
        return None

    kernel = _load_kernel("sm3_mine_compact")
    threads = int(os.getenv("LUNALIB_CUDA_THREADS", "256"))
    if threads < 1:
        threads = 256
    blocks = (count + threads - 1) // threads
    max_blocks = int(os.getenv("LUNALIB_CUDA_BLOCKS", "0"))
    if max_blocks > 0:
        blocks = min(blocks, max_blocks)
    if hashes_per_thread is None:
        env_hpt = os.getenv("LUNALIB_CUDA_HASHES_PER_THREAD")
        if env_hpt:
            try:
                hashes_per_thread = int(env_hpt)
            except Exception:
                hashes_per_thread = None

    if iterations is None:
        if hashes_per_thread is not None:
            iterations = hashes_per_thread
        else:
            iterations = int(os.getenv("LUNALIB_CUDA_ITERS", "1"))

    if iterations < 1:
        iterations = 1

    base_gpu = cp.asarray(bytearray(base80), dtype=cp.uint8)
    found_gpu = cp.zeros(1, dtype=cp.uint32)
    nonce_gpu = cp.zeros(1, dtype=cp.uint64)

    kernel((blocks,), (threads,), (
        base_gpu,
        cp.uint64(start_nonce),
        cp.uint64(count),
        int(difficulty),
        int(iterations),
        found_gpu,
        nonce_gpu,
    ))

    found = int(cp.asnumpy(found_gpu)[0])
    if not found:
        return None
    return int(cp.asnumpy(nonce_gpu)[0])


def gpu_sm3_throughput_persistent(
    payload: bytes,
    seconds: float,
    device_id: int = 0,
    threads: Optional[int] = None,
    blocks: Optional[int] = None,
    slice_seconds: Optional[float] = None,
) -> tuple[int, float]:
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    if seconds <= 0:
        return 0, 0.0

    padded = _pad_message(payload)
    if len(padded) != 64:
        raise ValueError("payload must fit in a single SM3 block")

    kernel_clock = _load_kernel("sm3_throughput_timed")
    kernel_iters = _load_kernel("sm3_throughput_iters")

    if threads is None:
        threads = int(os.getenv("LUNALIB_CUDA_THREADS", "256"))
    if threads < 1:
        threads = 256
    if blocks is None:
        blocks = int(os.getenv("LUNALIB_CUDA_BLOCKS", "256"))
    if blocks < 1:
        blocks = 256

    cp.cuda.Device(device_id).use()
    block_gpu = cp.asarray(bytearray(padded), dtype=cp.uint8)
    if slice_seconds is None:
        slice_seconds = float(os.getenv("LUNALIB_CUDA_PERSISTENT_SLICE", "1.0"))
    if slice_seconds <= 0:
        slice_seconds = float(seconds)

    props = cp.cuda.runtime.getDeviceProperties(device_id)
    clock_khz = int(props.get("clockRate", 0))
    clock_khz = max(clock_khz, 1)

    total_count = 0
    total_elapsed = 0.0
    remaining = float(seconds)

    while remaining > 0:
        run_seconds = min(slice_seconds, remaining)
        counter_gpu = cp.zeros(1, dtype=cp.uint64)

        if os.name == "nt":
            calib_iters = int(os.getenv("LUNALIB_CUDA_PERSISTENT_CALIB_ITERS", "8192"))
            if calib_iters < 1:
                calib_iters = 8192
            min_calib_seconds = float(os.getenv("LUNALIB_CUDA_PERSISTENT_CALIB_MIN", "0.05"))
            max_calib_loops = int(os.getenv("LUNALIB_CUDA_PERSISTENT_CALIB_LOOPS", "5"))

            # Calibrate to estimate hashes/sec with a minimum timing window.
            calib_elapsed = 0.0
            calib_count = 0
            for _ in range(max_calib_loops):
                counter_gpu.fill(0)
                start = time.perf_counter()
                kernel_iters((blocks,), (threads,), (block_gpu, counter_gpu, cp.uint32(calib_iters)))
                cp.cuda.Stream.null.synchronize()
                loop_elapsed = time.perf_counter() - start
                loop_count = int(cp.asnumpy(counter_gpu)[0])
                calib_elapsed += loop_elapsed
                calib_count += loop_count
                if calib_elapsed >= min_calib_seconds:
                    break
                calib_iters *= 2

            rate = calib_count / max(calib_elapsed, 1e-6)
            total_threads = int(blocks) * int(threads)
            target_hashes = rate * run_seconds
            per_thread_iters = int(target_hashes / max(total_threads, 1))
            max_iters = int(os.getenv("LUNALIB_CUDA_PERSISTENT_MAX_ITERS", "1000000"))
            per_thread_iters = max(1, min(per_thread_iters, max_iters))

            counter_gpu.fill(0)
            start = time.perf_counter()
            kernel_iters((blocks,), (threads,), (block_gpu, counter_gpu, cp.uint32(per_thread_iters)))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            count = int(cp.asnumpy(counter_gpu)[0])
        else:
            stop_cycles = int(clock_khz * 1000 * run_seconds)
            start = time.perf_counter()
            kernel_clock((blocks,), (threads,), (block_gpu, counter_gpu, cp.uint64(stop_cycles)))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            count = int(cp.asnumpy(counter_gpu)[0])

        total_count += count
        total_elapsed += elapsed
        remaining -= run_seconds

    return total_count, total_elapsed


def gpu_sm3_throughput_iters(
    payload: bytes,
    iterations: int,
    device_id: int = 0,
    threads: Optional[int] = None,
    blocks: Optional[int] = None,
) -> tuple[int, float]:
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    if iterations <= 0:
        return 0, 0.0

    padded = _pad_message(payload)
    if len(padded) != 64:
        raise ValueError("payload must fit in a single SM3 block")

    kernel_iters = _load_kernel("sm3_throughput_iters")

    if threads is None:
        threads = int(os.getenv("LUNALIB_CUDA_THREADS", "256"))
    if threads < 1:
        threads = 256
    if blocks is None:
        blocks = int(os.getenv("LUNALIB_CUDA_BLOCKS", "256"))
    if blocks < 1:
        blocks = 256

    cp.cuda.Device(device_id).use()
    block_gpu = cp.asarray(bytearray(padded), dtype=cp.uint8)
    counter_gpu = cp.zeros(1, dtype=cp.uint64)

    start = time.perf_counter()
    kernel_iters((blocks,), (threads,), (block_gpu, counter_gpu, cp.uint32(iterations)))
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    count = int(cp.asnumpy(counter_gpu)[0])
    return count, elapsed
