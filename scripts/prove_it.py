import argparse
import json
import os
import subprocess
import time
import threading
from pathlib import Path
from statistics import mean, median
from typing import Dict, Optional, Tuple, List
from datetime import datetime


def _require_gmssl():
    try:
        from gmssl import sm3  # noqa: F401
        try:
            from gmssl import utils  # noqa: F401
        except Exception:
            from gmssl import func as utils  # noqa: F401
        return True
    except Exception:
        return False


def _sm3_hex(data: bytes) -> str:
    from gmssl import sm3
    try:
        from gmssl import utils
    except Exception:
        from gmssl import func as utils
    return sm3.sm3_hash(utils.bytes_to_list(data))


def _parse_int(value: str) -> int:
    if value.lower().startswith("0x"):
        return int(value, 16)
    return int(value)


def _build_compact_base80(previous_hash: str, index: int, difficulty: int, timestamp: float, miner: str) -> bytes:
    if len(previous_hash) != 64:
        raise ValueError("previous_hash must be 64 hex chars")
    prev_bytes = bytes.fromhex(previous_hash)
    index_bytes = int(index).to_bytes(4, "big", signed=False)
    difficulty_bytes = int(difficulty).to_bytes(4, "big", signed=False)
    timestamp_bytes = float(timestamp).hex()
    # Match lunalib.mining.miner._build_compact_base timestamp packing (big-endian double)
    import struct
    timestamp_packed = struct.pack(">d", float(timestamp))
    from lunalib.core.sm3 import sm3_digest
    miner_hash = sm3_digest(str(miner).encode())
    base = prev_bytes + index_bytes + difficulty_bytes + timestamp_packed + miner_hash
    if len(base) != 80:
        raise ValueError("compact base must be 80 bytes")
    return base


def construct_header_88(mining_data: Dict, nonce: int) -> bytes:
    base80 = _build_compact_base80(
        previous_hash=mining_data["previous_hash"],
        index=mining_data["index"],
        difficulty=mining_data["difficulty"],
        timestamp=mining_data["timestamp"],
        miner=mining_data["miner"],
    )
    return base80 + int(nonce).to_bytes(8, "big", signed=False)


def verify_hash(mining_data: Dict, nonce: int, difficulty: Optional[int] = None) -> Tuple[bool, str]:
    header = construct_header_88(mining_data, nonce)
    hash_hex = _sm3_hex(header)
    if difficulty is None:
        difficulty = int(mining_data.get("difficulty", 0))
    prefix = "0" * int(difficulty)
    ok = hash_hex.startswith(prefix)
    return ok, hash_hex


def read_mining_data(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    required = ["previous_hash", "index", "difficulty", "timestamp", "miner"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return payload


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        return False


def _write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def gpu_watch(duration: float = 10.0, interval: float = 0.5,
              stop_event: Optional[threading.Event] = None) -> Dict[int, Dict[str, List[float]]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    start = time.time()
    samples: Dict[int, Dict[str, List[float]]] = {}
    while time.time() - start < duration:
        if stop_event and stop_event.is_set():
            break
        try:
            out = subprocess.check_output(cmd, text=True)
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
            return samples
        now = time.time() - start
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            util = float(parts[1])
            power = float(parts[2])
            temp = float(parts[3])
            stats = samples.setdefault(idx, {"t": [], "util": [], "power": [], "temp": []})
            stats["t"].append(now)
            stats["util"].append(util)
            stats["power"].append(power)
            stats["temp"].append(temp)
        time.sleep(interval)

    print("GPU utilization summary:")
    for idx, stats in sorted(samples.items()):
        util_avg = mean(stats["util"]) if stats["util"] else 0.0
        power_avg = mean(stats["power"]) if stats["power"] else 0.0
        temp_avg = mean(stats["temp"]) if stats["temp"] else 0.0
        print(f"  GPU {idx}: util={util_avg:.1f}% power={power_avg:.1f}W temp={temp_avg:.1f}C")
    return samples


def _throughput_summary(samples: List[Tuple[float, float]]) -> Dict[str, float]:
    rates = [r for _, r in samples if r > 0]
    if not rates:
        return {
            "min": 0,
            "avg": 0,
            "median": 0,
            "p95": 0,
            "max": 0,
        }
    rates_sorted = sorted(rates)
    p95 = rates_sorted[int(0.95 * (len(rates_sorted) - 1))]
    return {
        "min": float(min(rates)),
        "avg": float(mean(rates)),
        "median": float(median(rates)),
        "p95": float(p95),
        "max": float(max(rates)),
    }


def _assert_hashrate_sane(hashrate: float, mode: str) -> None:
    if mode != "gpu":
        return
    max_rate = float(os.getenv("LUNALIB_SM3_HASHRATE_MAX", "1e11"))
    if hashrate > max_rate:
        raise RuntimeError(
            f"Invalid SM3 hashrate ({hashrate:.2e} H/s). "
            "Counting bug detected in GPU throughput path."
        )


def throughput_test(seconds: float,
                    warmup: float = 5.0,
                    batch_size: int = 100000,
                    sample_interval: float = 1.0,
                    mode: str = "gpu",
                    report_dir: Optional[Path] = None,
                    plot: bool = False,
                    benchmark: bool = False,
                    profile: str = "steady",
                    persistent: bool = False,
                    persistent_slice: Optional[float] = None) -> None:
    payload = b"SM3 BENCHMARK PAYLOAD"
    samples: List[Tuple[float, float]] = []
    if benchmark or profile == "single" or (persistent and profile == "steady"):
        sample_interval = 0.0
        plot = False

    final_hashrate: Optional[float] = None

    if mode == "cpu":
        from lunalib.core.sm3 import sm3_digest

        # Warmup
        warmup_end = time.perf_counter() + warmup
        while time.perf_counter() < warmup_end:
            sm3_digest(payload)

        if profile == "single":
            count = 0
            start = time.perf_counter()
            for _ in range(batch_size):
                sm3_digest(payload)
                count += 1
            elapsed = time.perf_counter() - start
            hashrate = count / max(elapsed, 1e-9)
            final_hashrate = hashrate
            print(f"CPU throughput (single): {hashrate:,.0f} H/s (hashes={count}, seconds={elapsed:.2f})")
        else:
            start = time.perf_counter()
            end = start + seconds
            count = 0
            last_sample = start
            while time.perf_counter() < end:
                sm3_digest(payload)
                count += 1
                if sample_interval > 0:
                    now = time.perf_counter()
                    if now - last_sample >= sample_interval:
                        elapsed = now - start
                        samples.append((elapsed, count / max(elapsed, 1e-9)))
                        last_sample = now

            elapsed = time.perf_counter() - start
            hashrate = count / max(elapsed, 1e-9)
            final_hashrate = hashrate
            print(f"CPU throughput: {hashrate:,.0f} H/s (hashes={count}, seconds={elapsed:.2f})")
    else:
        try:
            import cupy as cp
            from lunalib.mining.sm3_cuda.sm3_gpu import (
                gpu_sm3_hash_messages,
                gpu_sm3_throughput_persistent,
                gpu_sm3_throughput_iters,
                CUDA_AVAILABLE,
            )
        except Exception as e:
            print(f"GPU throughput test unavailable: {e}")
            return
        if not CUDA_AVAILABLE:
            print("GPU throughput test unavailable: CUDA not available")
            return

        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            print("GPU throughput test unavailable: no devices")
            return

        if persistent and profile == "steady" and os.name == "nt" and os.getenv("LUNALIB_PERSISTENT_OK") != "1":
            print("Persistent kernel disabled on Windows to avoid driver hangs. Use --throughput-persistent-force to override.")
            persistent = False

        persistent_enabled = bool(persistent and profile == "steady")
        if persistent_enabled and os.name == "nt":
            if persistent_slice is None:
                persistent_slice = 0.25
            elif persistent_slice > 0.25:
                print("Windows persistent slice capped at 0.25s to avoid long-running kernels.")
                persistent_slice = 0.25

        if persistent_enabled:
            counts = [0 for _ in range(device_count)]
            elapsed_list = [0.0 for _ in range(device_count)]
            iters = int(os.getenv("LUNALIB_CUDA_THROUGHPUT_ITERS", "4096"))
            if iters < 1:
                iters = 4096

            def worker(device_id: int):
                # Warmup with fixed-iteration kernels
                if warmup > 0:
                    warm_end = time.perf_counter() + warmup
                    while time.perf_counter() < warm_end:
                        gpu_sm3_throughput_iters(payload, iters, device_id=device_id)

                start = time.perf_counter()
                local_count = 0
                local_elapsed = 0.0
                while time.perf_counter() - start < seconds:
                    count, elapsed = gpu_sm3_throughput_iters(payload, iters, device_id=device_id)
                    local_count += count
                    local_elapsed += elapsed
                counts[device_id] = local_count
                elapsed_list[device_id] = local_elapsed

            threads = []
            for i in range(device_count):
                t = threading.Thread(target=worker, args=(i,), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            elapsed = max(elapsed_list) if elapsed_list else 0.0
            count = sum(counts)
            hashrate = count / max(elapsed, 1e-9)
            final_hashrate = hashrate
            _assert_hashrate_sane(hashrate, mode)
            print(f"GPU throughput (steady-persistent): {hashrate:,.0f} H/s (hashes={count}, seconds={elapsed:.2f}, devices={device_count})")
        else:
            total_hashes = 0
            total_lock = threading.Lock()
            stop_event = threading.Event()
            start_event = threading.Event()
            per_device_counts = [0 for _ in range(device_count)]

            def worker(device_id: int, batch: int):
                nonlocal total_hashes
                cp.cuda.Device(device_id).use()
                msgs = [payload] * batch
                if profile == "single":
                    start_event.wait()
                    if stop_event.is_set():
                        return
                    gpu_sm3_hash_messages(msgs)
                    per_device_counts[device_id] = batch
                    return
                while not stop_event.is_set():
                    gpu_sm3_hash_messages(msgs)
                    if not start_event.is_set():
                        continue
                    if sample_interval > 0:
                        with total_lock:
                            total_hashes += batch
                    else:
                        per_device_counts[device_id] += batch

            def sampler(start_time: float):
                last_sample = start_time
                while not stop_event.is_set():
                    time.sleep(sample_interval)
                    with total_lock:
                        count = total_hashes
                    now = time.perf_counter()
                    elapsed = now - start_time
                    if elapsed <= 0:
                        continue
                    samples.append((elapsed, count / elapsed))
                    last_sample = now

            # Start workers
            threads = []
            per_device = max(1, int(batch_size // device_count))
            for i in range(device_count):
                t = threading.Thread(target=worker, args=(i, per_device), daemon=True)
                threads.append(t)
                t.start()

            # Warmup
            if warmup > 0:
                time.sleep(warmup)

            # Reset counters and start timed run
            with total_lock:
                total_hashes = 0
            for i in range(device_count):
                per_device_counts[i] = 0
            start = time.perf_counter()
            start_event.set()
            sampler_thread = None
            if sample_interval > 0:
                sampler_thread = threading.Thread(target=sampler, args=(start,), daemon=True)
                sampler_thread.start()

            if profile == "single":
                for t in threads:
                    t.join()
                stop_event.set()
            else:
                time.sleep(seconds)
                stop_event.set()
                for t in threads:
                    t.join(timeout=2.0)

            if sampler_thread:
                sampler_thread.join(timeout=2.0)

            elapsed = time.perf_counter() - start
            if sample_interval > 0:
                with total_lock:
                    count = total_hashes
            else:
                count = sum(per_device_counts)
            hashrate = count / max(elapsed, 1e-9)
            final_hashrate = hashrate
            _assert_hashrate_sane(hashrate, mode)
            label = "single" if profile == "single" else "steady"
            print(f"GPU throughput ({label}): {hashrate:,.0f} H/s (hashes={count}, seconds={elapsed:.2f}, devices={device_count})")

    stats = _throughput_summary(samples)
    if final_hashrate is not None and stats["max"] == 0:
        stats = {
            "min": float(final_hashrate),
            "avg": float(final_hashrate),
            "median": float(final_hashrate),
            "p95": float(final_hashrate),
            "max": float(final_hashrate),
        }
    print(
        "Throughput summary: "
        f"min={stats['min']:,.0f} avg={stats['avg']:,.0f} "
        f"median={stats['median']:,.0f} p95={stats['p95']:,.0f} max={stats['max']:,.0f} H/s"
    )

    if report_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir.mkdir(parents=True, exist_ok=True)
        if samples and not benchmark:
            _write_csv(
                report_dir / f"throughput_{mode}_{ts}.csv",
                ["t_seconds", "hashrate"],
                [[t, r] for t, r in samples],
            )
        summary = {
            "mode": mode,
            "profile": profile,
            "duration_seconds": seconds,
            "warmup_seconds": warmup,
            "batch_size": batch_size,
            "persistent": bool(persistent_enabled),
            "persistent_slice_seconds": persistent_slice,
            "samples": len(samples),
            "persistent_iters": int(os.getenv("LUNALIB_CUDA_THROUGHPUT_ITERS", "4096")) if persistent_enabled else 0,
            "hashrate_min": stats["min"],
            "hashrate_avg": stats["avg"],
            "hashrate_median": stats["median"],
            "hashrate_p95": stats["p95"],
            "hashrate_max": stats["max"],
        }
        with (report_dir / f"throughput_summary_{mode}_{ts}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if plot and _try_import_matplotlib() and samples and not benchmark:
            import matplotlib.pyplot as plt

            t_vals = [t for t, _ in samples]
            r_vals = [r for _, r in samples]
            plt.figure(figsize=(10, 4))
            plt.plot(t_vals, r_vals, label=f"Throughput ({mode})")
            plt.xlabel("Time (s)")
            plt.ylabel("Hashrate (H/s)")
            plt.title("SM3 throughput over time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(report_dir / f"throughput_{mode}_{ts}.png", dpi=150)
            plt.close()
        elif plot and not _try_import_matplotlib():
            print("matplotlib not installed; throughput plot skipped")


def mine_for_seconds(seconds: float, difficulty: Optional[int] = None,
                     report_dir: Optional[Path] = None,
                     collect_gpu: bool = False,
                     gpu_interval: float = 0.5) -> None:
    from lunalib.mining.miner import Miner

    class MinimalConfig:
        def __init__(self):
            self.miner_address = os.getenv("LUNALIB_MINER_ADDRESS", "LUN_PROVE_TEST")
            self.node_url = os.getenv("LUNALIB_NODE_URL", "https://bank.linglin.art")
            if difficulty is not None:
                self.difficulty = int(difficulty)
            else:
                self.difficulty = int(os.getenv("LUNALIB_DIFFICULTY", "1"))
            self.use_gpu = os.getenv("LUNALIB_GPU_ENABLED", "1") == "1"
            self.use_cpu = os.getenv("LUNALIB_CPU_ENABLED", "1") == "1"
            self.cuda_batch_size = int(os.getenv("LUNALIB_CUDA_BATCH_SIZE", "1000000"))
            self.multi_gpu_enabled = os.getenv("LUNALIB_MULTI_GPU", "1") == "1"
            self.cuda_sm3_kernel = os.getenv("LUNALIB_CUDA_SM3", "1") == "1"

    class MinimalDataManager:
        def load_mining_history(self):
            return []

        def save_mining_history(self, history):
            pass

    config = MinimalConfig()
    data_manager = MinimalDataManager()
    miner = Miner(config, data_manager)

    start_time = time.time()
    hashrate_samples: List[Tuple[float, float]] = []
    blocks: List[Dict] = []

    def _on_hashrate(rate: float, engine: str):
        if engine == "gpu":
            hashrate_samples.append((time.time() - start_time, float(rate)))

    miner.hashrate_callback = _on_hashrate

    gpu_samples: Dict[int, Dict[str, List[float]]] = {}
    stop_event = threading.Event()
    gpu_thread = None
    if collect_gpu:
        gpu_thread = threading.Thread(
            target=lambda: gpu_samples.update(gpu_watch(seconds, gpu_interval, stop_event)),
            daemon=True,
        )
        gpu_thread.start()

    blocks_found = 0
    while time.time() - start_time < seconds:
        success, _, _ = miner.mine_block()
        if success:
            blocks_found += 1
            if _:
                blocks.append(_)

    stop_event.set()
    if gpu_thread:
        gpu_thread.join(timeout=2.0)

    print(f"Blocks/{int(seconds)}s: {blocks_found}")

    # Summary stats
    rates = [r for _, r in hashrate_samples if r > 0]
    if rates:
        rates_sorted = sorted(rates)
        p95 = rates_sorted[int(0.95 * (len(rates_sorted) - 1))]
        print(f"Hashrate summary (GPU): min={min(rates):,.0f} avg={mean(rates):,.0f} median={median(rates):,.0f} p95={p95:,.0f} max={max(rates):,.0f} H/s")
    else:
        print("Hashrate summary (GPU): no samples")

    if report_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save hashrate samples
        if hashrate_samples:
            _write_csv(
                report_dir / f"hashrate_{ts}.csv",
                ["t_seconds", "hashrate"],
                [[t, r] for t, r in hashrate_samples],
            )

        # Save GPU samples
        for idx, stats in gpu_samples.items():
            _write_csv(
                report_dir / f"gpu{idx}_samples_{ts}.csv",
                ["t_seconds", "util", "power", "temp"],
                list(zip(stats.get("t", []), stats.get("util", []), stats.get("power", []), stats.get("temp", []))),
            )

        # Save summary JSON
        summary = {
            "duration_seconds": seconds,
            "blocks_found": blocks_found,
            "hashrate_samples": len(hashrate_samples),
            "hashrate_min": min(rates) if rates else 0,
            "hashrate_avg": mean(rates) if rates else 0,
            "hashrate_median": median(rates) if rates else 0,
            "hashrate_max": max(rates) if rates else 0,
            "hashrate_p95": rates_sorted[int(0.95 * (len(rates_sorted) - 1))] if rates else 0,
            "gpu_samples": {idx: len(stats.get("t", [])) for idx, stats in gpu_samples.items()},
            "blocks": [{"index": b.get("index"), "hash": b.get("hash"), "nonce": b.get("nonce"), "difficulty": b.get("difficulty")} for b in blocks],
        }
        with (report_dir / f"summary_{ts}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Plots
        if _try_import_matplotlib():
            import matplotlib.pyplot as plt

            if hashrate_samples:
                t_vals = [t for t, _ in hashrate_samples]
                r_vals = [r for _, r in hashrate_samples]
                plt.figure(figsize=(10, 4))
                plt.plot(t_vals, r_vals, label="Hashrate (H/s)")
                plt.xlabel("Time (s)")
                plt.ylabel("Hashrate (H/s)")
                plt.title("Hashrate over time")
                plt.legend()
                plt.tight_layout()
                plt.savefig(report_dir / f"hashrate_{ts}.png", dpi=150)
                plt.close()

            for idx, stats in gpu_samples.items():
                t_vals = stats.get("t", [])
                if not t_vals:
                    continue
                plt.figure(figsize=(10, 4))
                plt.plot(t_vals, stats.get("util", []), label="Util (%)")
                plt.plot(t_vals, stats.get("power", []), label="Power (W)")
                plt.plot(t_vals, stats.get("temp", []), label="Temp (C)")
                plt.xlabel("Time (s)")
                plt.title(f"GPU {idx} utilization/power/temp")
                plt.legend()
                plt.tight_layout()
                plt.savefig(report_dir / f"gpu{idx}_{ts}.png", dpi=150)
                plt.close()
        else:
            print("matplotlib not installed; graphs skipped")


def main():
    parser = argparse.ArgumentParser(description="Proof/verification helper for SM3 mining claims")
    parser.add_argument("--mining-data", help="JSON file with previous_hash,index,difficulty,timestamp,miner")
    parser.add_argument("--nonce", help="Nonce to verify", type=_parse_int)
    parser.add_argument("--difficulty", help="Difficulty (leading hex zeros)", type=int)
    parser.add_argument("--hash-only", action="store_true", help="Only print hash")
    parser.add_argument("--gpu-watch", action="store_true", help="Sample nvidia-smi for utilization")
    parser.add_argument("--watch-seconds", type=float, default=10.0)
    parser.add_argument("--watch-interval", type=float, default=0.5)
    parser.add_argument("--mine-10s", action="store_true", help="Run 10s mining loop (may submit blocks)")
    parser.add_argument("--mine-seconds", type=float, help="Run mining loop for N seconds (may submit blocks)")
    parser.add_argument("--mine-difficulty", type=int, help="Override mining difficulty for mine loop")
    parser.add_argument("--report-dir", help="Directory to write CSV/JSON/plots")
    parser.add_argument("--plot", action="store_true", help="Enable plots (requires matplotlib)")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5, help="GPU sample interval (s)")
    parser.add_argument("--throughput-seconds", type=float, help="Run raw throughput test for N seconds")
    parser.add_argument("--throughput-warmup", type=float, default=5.0, help="Warmup seconds for throughput test")
    parser.add_argument("--throughput-batch", type=int, default=100000, help="Batch size for throughput test")
    parser.add_argument("--throughput-interval", type=float, default=1.0, help="Sample interval for throughput test")
    parser.add_argument("--throughput-mode", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--throughput-benchmark", action="store_true", help="Disable sampling/plotting for throughput benchmark")
    parser.add_argument("--throughput-profile", choices=["steady", "single", "both"], default="steady", help="Throughput profile: steady-state or single-shot")
    parser.add_argument("--throughput-persistent", action="store_true", help="Use persistent GPU kernel for steady-state throughput")
    parser.add_argument("--throughput-persistent-slice", type=float, default=1.0, help="Slice duration for persistent kernel (s)")
    parser.add_argument("--throughput-persistent-force", action="store_true", help="Force persistent kernel on Windows (may hang driver)")

    args = parser.parse_args()

    if not _require_gmssl():
        print("gmssl not installed. Install with: pip install gmssl")
        return

    if args.mining_data and args.nonce is not None:
        mining_data = read_mining_data(args.mining_data)
        ok, hash_hex = verify_hash(mining_data, args.nonce, args.difficulty)
        if args.hash_only:
            print(hash_hex)
        else:
            prefix = "0" * int(args.difficulty or mining_data.get("difficulty", 0))
            print(f"Hash: {hash_hex}")
            print(f"Starts with {len(prefix)} zeros: {ok}")

    if args.gpu_watch:
        gpu_watch(duration=args.watch_seconds, interval=args.watch_interval)

    report_dir = Path(args.report_dir) if args.report_dir else None

    if args.mine_10s:
        mine_for_seconds(
            10,
            args.mine_difficulty,
            report_dir=report_dir,
            collect_gpu=args.plot,
            gpu_interval=args.gpu_sample_interval,
        )

    if args.mine_seconds:
        mine_for_seconds(
            float(args.mine_seconds),
            args.mine_difficulty,
            report_dir=report_dir,
            collect_gpu=args.plot,
            gpu_interval=args.gpu_sample_interval,
        )

    if args.throughput_seconds:
        if args.throughput_persistent_force:
            os.environ["LUNALIB_PERSISTENT_OK"] = "1"
        profiles = [args.throughput_profile]
        if args.throughput_profile == "both":
            profiles = ["steady", "single"]
        for profile in profiles:
            throughput_test(
                float(args.throughput_seconds),
                warmup=float(args.throughput_warmup),
                batch_size=int(args.throughput_batch),
                sample_interval=float(args.throughput_interval),
                mode=args.throughput_mode,
                report_dir=report_dir,
                plot=bool(args.plot),
                benchmark=bool(args.throughput_benchmark),
                profile=str(profile),
                persistent=bool(args.throughput_persistent),
                persistent_slice=float(args.throughput_persistent_slice),
            )


if __name__ == "__main__":
    main()
