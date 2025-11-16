import pytest
import tempfile
import os
import time
from core.wallet import LunaWallet
from mining.miner import GenesisMiner
from gtx.genesis import GTXGenesis

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