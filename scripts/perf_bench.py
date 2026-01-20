import argparse
import hashlib
import time

from lunalib.core.sm2 import SM2
from lunalib.core.wallet import LunaWallet
from lunalib.transactions.transactions import TransactionManager


def bench_sm2(iterations: int):
    sm2 = SM2()
    private_key, public_key = sm2.generate_keypair()
    message = b"performance-test"

    start = time.perf_counter()
    for _ in range(iterations):
        signature = sm2.sign(message, private_key)
        sm2.verify(message, signature, public_key)
    elapsed = time.perf_counter() - start
    print(f"SM2 sign+verify: {iterations} iterations in {elapsed:.4f}s ({iterations/elapsed:.2f}/s)")


def bench_wallet_balance(tx_count: int):
    wallet = LunaWallet()
    address = "LUN_perf"
    wallet.address = address

    confirmed = []
    for i in range(tx_count):
        direction = "incoming" if i % 2 == 0 else "outgoing"
        confirmed.append({
            "type": "transaction",
            "direction": direction,
            "amount": 1.0,
            "fee": 0.001,
            "block_height": i,
            "timestamp": time.time(),
            "hash": f"tx_{direction}_{i}",
            "from": "LUN_a",
            "to": "LUN_b",
        })

    wallet._confirmed_tx_cache[address] = confirmed
    wallet._pending_tx_cache[address] = []

    start = time.perf_counter()
    balance = wallet._compute_confirmed_balance(confirmed)
    elapsed = time.perf_counter() - start
    print(f"Balance compute: {tx_count} txs in {elapsed:.6f}s | balance={balance}")


def bench_transaction_creation(iterations: int):
    tm = TransactionManager(network_endpoints=[])
    sm2 = SM2()
    private_key, public_key = sm2.generate_keypair()
    from_addr = "LUN_perf_sender"
    to_addr = "LUN_perf_receiver"

    start = time.perf_counter()
    for i in range(iterations):
        tx = tm.create_transaction(from_addr, to_addr, 1.0, private_key=private_key, memo=str(i))
        if not tx:
            raise RuntimeError("Transaction creation failed")
    elapsed = time.perf_counter() - start
    print(f"Transaction creation: {iterations} txs in {elapsed:.4f}s ({iterations/elapsed:.2f}/s)")


def bench_hashrate(iterations: int, difficulty: int):
    prefix = "0" * max(1, difficulty)
    nonce = 0
    start = time.perf_counter()
    found = 0
    while nonce < iterations:
        payload = f"bench|{nonce}|{time.time()}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        if digest.startswith(prefix):
            found += 1
        nonce += 1
    elapsed = time.perf_counter() - start
    print(
        f"Hashrate: {iterations} hashes in {elapsed:.4f}s "
        f"({iterations/elapsed:.2f} H/s), difficulty={difficulty}, hits={found}"
    )


def main():
    parser = argparse.ArgumentParser(description="LunaLib performance benchmarks")
    parser.add_argument("--sm2", type=int, default=50, help="SM2 sign+verify iterations")
    parser.add_argument("--tx", type=int, default=100, help="Transaction creation iterations")
    parser.add_argument("--balance", type=int, default=5000, help="Number of txs for balance compute")
    parser.add_argument("--hashes", type=int, default=50000, help="Number of hashes for hashrate test")
    parser.add_argument("--difficulty", type=int, default=2, help="Difficulty prefix for hash hits")
    args = parser.parse_args()

    bench_sm2(args.sm2)
    bench_wallet_balance(args.balance)
    bench_transaction_creation(args.tx)
    bench_hashrate(args.hashes, args.difficulty)


if __name__ == "__main__":
    main()
