import argparse
import os
import sys
import time
import secrets
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lunalib.core.sm2_cuda.sm2_gpu import sm2_sign_gpu
from lunalib.core.sm2 import SM2

try:
    from lunalib.core.sm2_c import sm2_ext as sm2_c_ext  # type: ignore
    _HAS_SM2_C = True
except Exception:
    sm2_c_ext = None
    _HAS_SM2_C = False

try:
    from gmssl import sm2 as gmssl_sm2  # type: ignore
    from gmssl import func as gmssl_func  # type: ignore
    _HAS_GMSSL = True
except Exception:
    gmssl_sm2 = None
    gmssl_func = None
    _HAS_GMSSL = False


def _bench_sign(batch_size: int, iterations: int, msg_size: int, use_gpu: bool, cpu_workers: int, backend: str, reuse_key: bool, precompute_k: bool, warmup_batches: int, use_threads: bool, reuse_pool: bool) -> Tuple[float, float]:
    message = secrets.token_bytes(msg_size)
    if reuse_key:
        key = secrets.token_bytes(32)
        priv_keys = [key for _ in range(batch_size)]
    else:
        priv_keys = [secrets.token_bytes(32) for _ in range(batch_size)]

    pool = None
    if not use_gpu and cpu_workers > 1 and reuse_pool:
        if use_threads:
            pool = ThreadPoolExecutor(max_workers=cpu_workers)
        else:
            pool = ProcessPoolExecutor(max_workers=cpu_workers, initializer=_init_worker, initargs=(backend,))

    if warmup_batches > 0:
        label = "GPU" if use_gpu else f"CPU({cpu_workers}t,{backend})"
        print(f"{label} warmup: {warmup_batches} batch(es) ...", flush=True)
        for _ in range(warmup_batches):
            if use_gpu:
                sm2_sign_gpu(message, priv_keys, use_gpu=True)
            else:
                _cpu_sign_parallel(message, priv_keys, cpu_workers, backend, precompute_k, use_threads, pool)

    start = time.perf_counter()
    total = 0
    for i in range(iterations):
        if use_gpu:
            sm2_sign_gpu(message, priv_keys, use_gpu=True)
        else:
            _cpu_sign_parallel(message, priv_keys, cpu_workers, backend, precompute_k, use_threads, pool)
        total += batch_size
        if (i + 1) % max(1, iterations // 10) == 0:
            elapsed = time.perf_counter() - start
            rate = total / elapsed if elapsed > 0 else 0.0
            label = "GPU" if use_gpu else f"CPU({cpu_workers}t,{backend})"
            print(f"{label} progress: {i + 1}/{iterations} | {rate:,.0f} sigs/sec", flush=True)
    elapsed = time.perf_counter() - start
    if pool is not None:
        pool.shutdown(wait=True)
    sigs_per_sec = total / elapsed if elapsed > 0 else 0.0
    return elapsed, sigs_per_sec


_SM2_WORKER = None
_BACKEND = "python"


def _init_worker(backend: str) -> None:
    global _SM2_WORKER
    global _BACKEND
    _BACKEND = backend
    if backend == "gmssl":
        if not _HAS_GMSSL:
            raise RuntimeError("gmssl backend requested but not installed")
        _SM2_WORKER = None
    elif backend == "cpython":
        if not _HAS_SM2_C:
            raise RuntimeError("CPython SM2 backend requested but not built")
        _SM2_WORKER = None
    else:
        _SM2_WORKER = SM2()


def _cpu_sign_chunk(args: tuple[bytes, list[bytes], str, bool], mute_output: bool = True) -> None:
    import contextlib
    message, keys, backend, precompute_k = args
    debug_enabled = bool(os.getenv("SM2_EXT_DEBUG"))
    if debug_enabled:
        print(f"[worker] start chunk size={len(keys)} backend={backend}", flush=True)
    def _do_work() -> None:
        if backend == "gmssl":
            k_list = None
            if precompute_k:
                k_list = [gmssl_func.random_hex(len(message)) for _ in keys]
            for i, k in enumerate(keys):
                sm2_obj = gmssl_sm2.CryptSM2(public_key='', private_key=k.hex())
                rk = k_list[i] if k_list else gmssl_func.random_hex(len(message))
                sm2_obj.sign(message, rk)
        elif backend == "cpython":
            if not _HAS_SM2_C:
                raise RuntimeError("CPython SM2 backend requested but not built")
            for i, k in enumerate(keys):
                sm2_c_ext.sign(message, k)
                if debug_enabled and i == 0:
                    print("[worker] first sign done", flush=True)
        else:
            sm2 = _SM2_WORKER if _SM2_WORKER is not None else SM2()
            for k in keys:
                sm2.sign(message, k.hex())

    if mute_output and not debug_enabled:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            _do_work()
    else:
        _do_work()
    if debug_enabled:
        print("[worker] chunk done", flush=True)


def _cpu_sign_single(args: tuple[bytes, bytes]) -> None:
    """Legacy helper to avoid multiprocessing pickling errors from old tasks."""
    message, key = args
    _cpu_sign_chunk((message, [key]))


def _cpu_sign_parallel(message: bytes, priv_keys: list[bytes], cpu_workers: int, backend: str, precompute_k: bool, use_threads: bool = False, pool=None) -> None:
    if cpu_workers <= 1:
        debug_enabled = bool(os.getenv("SM2_EXT_DEBUG"))
        if backend == "gmssl":
            if not _HAS_GMSSL:
                raise RuntimeError("gmssl backend requested but not installed")
            k_list = None
            if precompute_k:
                k_list = [gmssl_func.random_hex(len(message)) for _ in priv_keys]
            for i, k in enumerate(priv_keys):
                sm2_obj = gmssl_sm2.CryptSM2(public_key='', private_key=k.hex())
                rk = k_list.pop(0) if k_list else gmssl_func.random_hex(len(message))
                sm2_obj.sign(message, rk)
                if debug_enabled and i == 0:
                    print("[main] first gmssl sign done", flush=True)
        elif backend == "cpython":
            if not _HAS_SM2_C:
                raise RuntimeError("CPython SM2 backend requested but not built")
            for i, k in enumerate(priv_keys):
                sm2_c_ext.sign(message, k)
                if debug_enabled and i == 0:
                    print("[main] first cpython sign done", flush=True)
        else:
            sm2_sign_gpu(message, priv_keys, use_gpu=False)
        return
    # Chunk keys to reduce overhead per task
    total = len(priv_keys)
    workers = min(cpu_workers, total)
    chunk_size = max(1, total // workers)
    chunks = [priv_keys[i:i + chunk_size] for i in range(0, total, chunk_size)]
    total_chunks = len(chunks)
    print(f"CPU({workers}t,{backend}) chunk progress: 0/{total_chunks}", flush=True)
    if pool is not None:
        total_chunks = len(chunks)
        print(f"CPU({workers}t,{backend}) chunk progress: 0/{total_chunks}", flush=True)
        futures = [pool.submit(_cpu_sign_chunk, (message, chunk, backend, precompute_k)) for chunk in chunks]
        done = 0
        pending = set(futures)
        last_report = time.perf_counter()
        while pending:
            finished, pending = wait(pending, timeout=1)
            for _ in finished:
                done += 1
                print(f"CPU({workers}t,{backend}) chunk progress: {done}/{total_chunks}", flush=True)
            now = time.perf_counter()
            if now - last_report >= 5 and pending:
                print(f"CPU({workers}t,{backend}) still running... {done}/{total_chunks}", flush=True)
                last_report = now
        return
    if use_threads:
        # Single shared SM2 instance for python backend within this process
        global _SM2_WORKER
        if backend == "python":
            _SM2_WORKER = _SM2_WORKER or SM2()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_cpu_sign_chunk, (message, chunk, backend, precompute_k), False) for chunk in chunks]
            done = 0
            pending = set(futures)
            last_report = time.perf_counter()
            while pending:
                finished, pending = wait(pending, timeout=1)
                for _ in finished:
                    done += 1
                    print(f"CPU({workers}t,{backend}) chunk progress: {done}/{total_chunks}", flush=True)
                now = time.perf_counter()
                if now - last_report >= 5 and pending:
                    print(f"CPU({workers}t,{backend}) still running... {done}/{total_chunks}", flush=True)
                    last_report = now
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(backend,)) as pool:
        futures = [pool.submit(_cpu_sign_chunk, (message, chunk, backend, precompute_k)) for chunk in chunks]
        done = 0
        pending = set(futures)
        last_report = time.perf_counter()
        while pending:
            finished, pending = wait(pending, timeout=1)
            for _ in finished:
                done += 1
                print(f"CPU({workers}t,{backend}) chunk progress: {done}/{total_chunks}", flush=True)
            now = time.perf_counter()
            if now - last_report >= 5 and pending:
                print(f"CPU({workers}t,{backend}) still running... {done}/{total_chunks}", flush=True)
                last_report = now


def main() -> None:
    parser = argparse.ArgumentParser(description="SM2 signing benchmark (CPU vs GPU-assisted)")
    parser.add_argument("--batch", type=int, default=1024, help="signatures per batch")
    parser.add_argument("--iterations", type=int, default=50, help="number of batches")
    parser.add_argument("--message-size", type=int, default=64, help="message size in bytes")
    parser.add_argument("--backend", type=str, default=os.getenv("LUNALIB_SM2_BACKEND", "python"), choices=["python", "gmssl", "cpython", "phos"], help="CPU signing backend")
    parser.add_argument("--reuse-key", action="store_true", help="reuse a single private key for all signatures")
    parser.add_argument("--precompute-k", action="store_true", help="precompute k values for gmssl backend")
    parser.add_argument("--warmup-batches", type=int, default=1, help="warmup batches before timing")
    parser.add_argument("--quick", action="store_true", help="quick run: iterations=1, batch=256, warmup=0")
    parser.add_argument("--debug-ext", action="store_true", help="enable SM2_EXT_DEBUG for C backend")
    parser.add_argument("--cpu-optimized", action="store_true", help="apply CPU-optimized defaults (workers, batch, reuse-key)")
    env_workers = os.getenv("LUNALIB_SM2_CPU_WORKERS")
    default_workers = int(env_workers) if env_workers and env_workers.isdigit() else 16
    parser.add_argument("--cpu-workers", type=int, default=default_workers, help="CPU worker threads")
    parser.add_argument("--no-multiprocess", action="store_true", help="disable multiprocessing for CPU path")
    parser.add_argument("--threaded", action="store_true", help="use threads (single SM2 instance per process)")
    args = parser.parse_args()
    if args.debug_ext:
        os.environ["SM2_EXT_DEBUG"] = "1"
        print("SM2_EXT_DEBUG=1", flush=True)

    if args.quick:
        args.iterations = 1
        args.batch = 256
        args.warmup_batches = 0
        args.cpu_workers = 1

    if args.cpu_optimized:
        if args.backend in ("python",):
            args.backend = "phos"
        if not args.reuse_key:
            args.reuse_key = True
        if args.backend == "gmssl" and not args.precompute_k:
            args.precompute_k = True
        if args.batch < 4096:
            args.batch = 4096
        if args.iterations < 10:
            args.iterations = 10
        if args.cpu_workers <= 1 and not args.no_multiprocess:
            args.cpu_workers = max(2, os.cpu_count() or 2)

    os.environ.setdefault("LUNALIB_SM2_GPU", "0")

    backend = "cpython" if args.backend == "phos" else args.backend
    cpu_workers = 1 if args.no_multiprocess else args.cpu_workers
    cpu_time, cpu_rate = _bench_sign(args.batch, args.iterations, args.message_size, use_gpu=False, cpu_workers=cpu_workers, backend=backend, reuse_key=args.reuse_key, precompute_k=args.precompute_k, warmup_batches=args.warmup_batches, use_threads=args.threaded, reuse_pool=True)
    print(f"CPU({cpu_workers}t,{args.backend}): {cpu_time:.3f}s total | {cpu_rate:,.0f} sigs/sec")

    try:
        gpu_time, gpu_rate = _bench_sign(args.batch, args.iterations, args.message_size, use_gpu=True, cpu_workers=cpu_workers, backend=backend, reuse_key=args.reuse_key, precompute_k=args.precompute_k, warmup_batches=args.warmup_batches, use_threads=args.threaded, reuse_pool=False)
        print(f"GPU-assisted: {gpu_time:.3f}s total | {gpu_rate:,.0f} sigs/sec")
        if cpu_rate > 0:
            print(f"Speedup: {gpu_rate / cpu_rate:.2f}x")
    except Exception as e:
        print(f"GPU-assisted benchmark failed: {e}")


if __name__ == "__main__":
    main()
