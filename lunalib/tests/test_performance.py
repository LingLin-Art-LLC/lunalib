import time

import pytest

from lunalib.core.sm2 import SM2
from lunalib.core.wallet import LunaWallet


def _make_tx(direction, amount, block_height):
    return {
        "type": "transaction",
        "direction": direction,
        "amount": amount,
        "fee": 0.001,
        "block_height": block_height,
        "timestamp": time.time(),
        "hash": f"tx_{direction}_{block_height}",
        "from": "LUN_a",
        "to": "LUN_b",
    }


@pytest.mark.performance
def test_balance_computation_performance(benchmark=None):
    wallet = LunaWallet()
    address = "LUN_perf"
    wallet.address = address

    confirmed = []
    for i in range(5000):
        direction = "incoming" if i % 2 == 0 else "outgoing"
        confirmed.append(_make_tx(direction, 1.0, i))

    wallet._confirmed_tx_cache[address] = confirmed
    wallet._pending_tx_cache[address] = []

    def _run():
        return wallet._compute_confirmed_balance(confirmed)

    if benchmark is None:
        start = time.perf_counter()
        balance = _run()
        elapsed = time.perf_counter() - start
        print(f"[PERF] confirmed balance calc: {elapsed:.6f}s")
    else:
        balance = benchmark(_run)
    assert balance >= 0


@pytest.mark.performance
def test_sm2_sign_verify_performance(benchmark=None):
    sm2 = SM2()
    private_key, public_key = sm2.generate_keypair()
    message = b"performance-test"

    def _sign_verify():
        signature = sm2.sign(message, private_key)
        return sm2.verify(message, signature, public_key)

    if benchmark is None:
        start = time.perf_counter()
        ok = _sign_verify()
        elapsed = time.perf_counter() - start
        print(f"[PERF] SM2 sign+verify: {elapsed:.6f}s")
    else:
        ok = benchmark(_sign_verify)
    assert ok is True