import pytest
import tempfile
import os
import time
from lunalib.core.wallet import LunaWallet
from lunalib.mining.miner import GenesisMiner
from lunalib.gtx.genesis import GTXGenesis

try:
    import pytest_benchmark  # type: ignore
    _HAS_BENCHMARK = True
except Exception:
    _HAS_BENCHMARK = False

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def test_wallet(temp_dir):
    """Create a test wallet"""
    wallet = LunaWallet(data_dir=temp_dir)
    wallet_data = wallet.create_wallet("Test Wallet", "test_password")
    return wallet, wallet_data

@pytest.fixture
def test_miner():
    """Create a test miner"""
    return GenesisMiner()

@pytest.fixture
def test_gtx(temp_dir):
    """Create test GTX system"""
    return GTXGenesis()

@pytest.fixture
def sample_transaction_data(test_wallet):
    """Create sample transaction data"""
    wallet, wallet_data = test_wallet
    return {
        "from": wallet_data["address"],
        "to": "LUN_test_recipient_12345",
        "amount": 100.0,
        "memo": "Test transaction"
    }


if not _HAS_BENCHMARK:
    @pytest.fixture
    def benchmark():
        """Fallback benchmark fixture when pytest-benchmark isn't installed."""
        def _runner(func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[PERF] elapsed: {elapsed:.6f}s")
            return result
        return _runner