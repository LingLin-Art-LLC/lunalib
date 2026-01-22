import argparse
import os
import secrets
import time

from lunalib.core import sm4 as sm4_module
from lunalib.core.sm4 import SM4Cipher


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return sm4_module._xor_bytes(a, b)


def _bench(mode: str, size_kb: int, iterations: int, use_gpu: bool) -> tuple[float, float]:
    key = secrets.token_bytes(16)
    iv = secrets.token_bytes(16)
    data = secrets.token_bytes(size_kb * 1024)
    xor_key = secrets.token_bytes(size_kb * 1024)
    cipher = SM4Cipher(key)

    # warmup
    if mode == "ecb":
        cipher.encrypt_ecb(data, use_gpu=use_gpu)
    elif mode == "ctr":
        cipher.encrypt_ctr(data, iv, use_gpu=use_gpu)
    else:
        _xor_bytes(data, xor_key)

    start = time.perf_counter()
    total_bytes = 0
    for i in range(iterations):
        if mode == "ecb":
            cipher.encrypt_ecb(data, use_gpu=use_gpu)
        elif mode == "ctr":
            cipher.encrypt_ctr(data, iv, use_gpu=use_gpu)
        else:
            _xor_bytes(data, xor_key)
        total_bytes += len(data)
        if (i + 1) % max(1, iterations // 5) == 0:
            elapsed = time.perf_counter() - start
            mb_s = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
            label = "GPU" if use_gpu else "CPU"
            print(f"{label} progress: {i + 1}/{iterations} | {mb_s:,.2f} MB/s", flush=True)

    elapsed = time.perf_counter() - start
    mb_s = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
    return elapsed, mb_s


def main() -> None:
    parser = argparse.ArgumentParser(description="SM4 benchmark (ECB/CTR) with optional GPU")
    parser.add_argument("--mode", choices=["ecb", "ctr", "xor"], default="ecb")
    parser.add_argument("--size-kb", type=int, default=1024, help="payload size in KB")
    parser.add_argument("--iterations", type=int, default=50, help="number of iterations")
    parser.add_argument("--use-gpu", action="store_true", help="use CuPy acceleration if available")
    args = parser.parse_args()

    print(f"Mode={args.mode} Size={args.size_kb}KB Iter={args.iterations} GPU={args.use_gpu}")
    elapsed, mb_s = _bench(args.mode, args.size_kb, args.iterations, args.use_gpu)
    label = "GPU" if args.use_gpu else "CPU"
    print(f"{label} total: {elapsed:.3f}s | {mb_s:,.2f} MB/s")


if __name__ == "__main__":
    main()
