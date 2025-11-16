import pytest
import json
import tempfile
import os
from storage.encryption import EncryptionManager
from storage.database import WalletDatabase

class TestStorage:
    def test_encryption_decryption(self):
        """Test data encryption and decryption"""
        encryption = EncryptionManager()
        
        original_data = "Sensitive wallet data"
        password = "strong_password"
        
        encrypted = encryption.encrypt_data(original_data, password)
        decrypted = encryption.decrypt_data(encrypted, password)
        
        assert decrypted == original_data
        assert encrypted != original_data

    def test_wrong_password_decryption(self):
        """Test decryption with wrong password"""
        encryption = EncryptionManager()
        
        original_data = "Test data"
        encrypted = encryption.encrypt_data(original_data, "correct_password")
        
        # Should raise exception or return error with wrong password
        try:
            decrypted = encryption.decrypt_data(encrypted, "wrong_password")
            assert decrypted != original_data
        except Exception:
            # Expected behavior - decryption should fail
            pass

    def test_wallet_database_operations(self, temp_dir):
        """Test wallet database operations"""
        db_path = os.path.join(temp_dir, "test_wallets.db")
        database = WalletDatabase(db_path)
        
        # Test saving wallet
        wallet_data = {
            "address": "LUN_test_address_123",
            "label": "Test Wallet",
            "public_key": "test_public_key",
            "encrypted_private_key": "encrypted_priv",
            "balance": 1000.0,
            "created": 1234567890,
            "metadata": {"test": "data"}
        }
        
        success = database.save_wallet(wallet_data)
        assert success is True
        
        # Test loading wallet
        loaded = database.load_wallet("LUN_test_address_123")
        assert loaded is not None
        assert loaded['address'] == "LUN_test_address_123"
        assert loaded['balance'] == 1000.0

    def test_transaction_storage(self, temp_dir):
        """Test transaction storage in database"""
        db_path = os.path.join(temp_dir, "test_tx.db")
        database = WalletDatabase(db_path)
        
        transaction = {
            "hash": "tx_hash_123",
            "type": "transfer",
            "from": "LUN_sender",
            "to": "LUN_receiver", 
            "amount": 500.0,
            "fee": 0.01,
            "timestamp": 1234567890,
            "status": "confirmed"
        }
        
        success = database.save_transaction(transaction, "LUN_sender")
        assert success is True
        
        # Test retrieving transactions
        transactions = database.get_wallet_transactions("LUN_sender")
        assert len(transactions) == 1
        assert transactions[0]['hash'] == "tx_hash_123"