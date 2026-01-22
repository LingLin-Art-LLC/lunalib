import argparse
import contextlib
import os
import tempfile
import time

from lunalib.core.wallet import LunaWallet


def _suppress_output(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    devnull = open(os.devnull, "w")
    return contextlib.redirect_stdout(devnull)


def bench_create(wallet: LunaWallet, count: int, password: str, suppress: bool) -> tuple[float, float]:
    start = time.perf_counter()
    with _suppress_output(suppress):
        for i in range(count):
            wallet.create_new_wallet(f"bench_wallet_{i}", password)
    elapsed = time.perf_counter() - start
    rate = count / elapsed if elapsed > 0 else 0.0
    return elapsed, rate


def bench_unlock_lock(wallet: LunaWallet, address: str, password: str, cycles: int, suppress: bool) -> tuple[float, float]:
    start = time.perf_counter()
    with _suppress_output(suppress):
        for _ in range(cycles):
            wallet.unlock_wallet(address, password)
            wallet.lock_wallet(address)
    elapsed = time.perf_counter() - start
    rate = cycles / elapsed if elapsed > 0 else 0.0
    return elapsed, rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Wallet create/unlock/lock benchmark")
    parser.add_argument("--wallets", type=int, default=100, help="number of wallets to create")
    parser.add_argument("--cycles", type=int, default=100, help="unlock/lock cycles")
    parser.add_argument("--password", type=str, default="benchmark_password", help="wallet password")
    parser.add_argument("--no-suppress", action="store_true", help="do not suppress wallet debug output")
    args = parser.parse_args()

    suppress = not args.no_suppress

    with tempfile.TemporaryDirectory() as tmpdir:
        wallet = LunaWallet(data_dir=tmpdir)

        create_elapsed, create_rate = bench_create(wallet, args.wallets, args.password, suppress)
        print(f"Create: {create_elapsed:.3f}s total | {create_rate:,.2f} wallets/sec")

        # ensure one wallet exists for unlock/lock
        if not wallet.wallets:
            wallet_data = wallet.create_new_wallet("bench_wallet", args.password)
            address = wallet_data["address"]
        else:
            address = next(iter(wallet.wallets.keys()))

        lock_elapsed, lock_rate = bench_unlock_lock(wallet, address, args.password, args.cycles, suppress)
        print(f"Unlock+Lock: {lock_elapsed:.3f}s total | {lock_rate:,.2f} cycles/sec")


if __name__ == "__main__":
    main()
