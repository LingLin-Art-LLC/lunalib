import pytest
import os
import json
from core.wallet import LunaWallet

class TestLunaWallet:
    def test_wallet_creation(self, test_wallet):
        """Test basic wallet creation"""
        wallet, wallet_data = test_wallet
        
        assert 'address' in wallet_data
        assert 'public_key' in wallet_data
        assert 'private_key' in wallet_data
        assert wallet_data['label'] == "Test Wallet"
        assert wallet_data['balance'] == 0.0
        assert len(wallet_data['address']) > 0

    def test_wallet_encryption_decryption(self, temp_dir):
        """Test wallet encryption and decryption"""
        wallet = LunaWallet(data_dir=temp_dir)
        wallet_data = wallet.create_wallet("Encrypted Wallet", "test_password")
        
        # Test unlocking with correct password
        assert wallet.unlock_wallet(wallet_data['address'], "test_password")
        assert wallet.is_unlocked
        
        # Test unlocking with wrong password
        assert not wallet.unlock_wallet(wallet_data['address'], "wrong_password")

    def test_wallet_export_import(self, test_wallet):
        """Test wallet export and import functionality"""
        wallet, wallet_data = test_wallet

        # Export private key
        exported = wallet.export_private_key(wallet_data['address'], "test_password")
        assert exported is not None
        # FIXED: exported is the private key string, not a dict
        assert exported.startswith('priv_')  # Check it's a valid private key format
        assert len(exported) > 10

    def test_wallet_balance_operations(self, test_wallet):
        """Test wallet balance operations"""
        wallet, wallet_data = test_wallet
        
        # Initial balance should be 0
        wallet_info = wallet.get_wallet_info()
        assert wallet_info['balance'] == 0.0
        assert wallet_info['available_balance'] == 0.0

    def test_multiple_wallets(self, temp_dir):
        """Test managing multiple wallets"""
        wallet = LunaWallet(data_dir=temp_dir)

        # Create first wallet
        wallet1_data = wallet.create_wallet("Wallet 1", "pass1")
        print(f"Wallet 1 address: {wallet1_data['address']}")

        # Create second wallet (adds to collection without switching current)
        wallet2_data = wallet.create_new_wallet("Wallet 2", "pass2")
        print(f"Wallet 2 address: {wallet2_data['address']}")

        print(f"All wallets: {list(wallet.wallets.keys())}")
        
        assert len(wallet.wallets) == 2

    def test_wallet_persistence(self, temp_dir):
        """Test wallet data persistence"""
        # Create and save wallet
        wallet1 = LunaWallet(data_dir=temp_dir)
        wallet_data = wallet1.create_wallet("Persistent Wallet", "password123")

        # Save to file
        assert wallet1.save_to_file("test_wallet.json")
        
        # Create new wallet instance and load
        wallet2 = LunaWallet(data_dir=temp_dir)
        assert wallet2.load_from_file("test_wallet.json")
        
        # Now unlock with password
        assert wallet2.unlock_wallet(wallet_data['address'], "password123")
        assert wallet2.is_unlocked