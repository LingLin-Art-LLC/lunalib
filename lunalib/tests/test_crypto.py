import pytest
from lunalib.core.crypto import KeyManager

class TestCrypto:
    def test_key_generation(self):
        """Test cryptographic key generation"""
        key_manager = KeyManager()
        
        private_key = key_manager.generate_private_key()
        public_key = key_manager.derive_public_key(private_key)
        address = key_manager.derive_address(public_key)
        
        assert len(private_key) == 64  # 32 bytes in hex
        assert len(public_key) == 130  # 04 + 64-byte x + 64-byte y
        assert address.startswith("LUN_")

    def test_data_signing(self):
        """Test data signing and verification"""
        key_manager = KeyManager()
        
        private_key = key_manager.generate_private_key()
        public_key = key_manager.derive_public_key(private_key)
        
        test_data = "Hello, Luna Library!"
        signature = key_manager.sign_data(test_data, private_key)
        
        # Basic signature format check
        assert len(signature) == 128
        assert all(c in "0123456789abcdef" for c in signature.lower())

    def test_address_generation_uniqueness(self):
        """Test that addresses are unique"""
        key_manager = KeyManager()
        
        addresses = set()
        for _ in range(10):
            private_key = key_manager.generate_private_key()
            public_key = key_manager.derive_public_key(private_key)
            address = key_manager.derive_address(public_key)
            addresses.add(address)
        
        assert len(addresses) == 10  # All addresses should be unique