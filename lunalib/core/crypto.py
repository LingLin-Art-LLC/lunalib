import hashlib
import secrets
from typing import Optional

class KeyManager:
    """Manages cryptographic keys and signing"""
    
    def generate_private_key(self):
        """Generate a new private key"""
        return secrets.token_hex(32)
    
    def derive_public_key(self, private_key):
        """Derive public key from private key"""
        return hashlib.sha256(private_key.encode()).hexdigest()
    
    def derive_address(self, public_key):
        """Derive address from public key"""
        address_hash = hashlib.sha256(public_key.encode()).hexdigest()
        return f"LUN_{address_hash[:16]}_{secrets.token_hex(4)}"
    
    def sign_data(self, data, private_key):
        """Sign data with private key"""
        # In production, use proper ECDSA
        # For now, simplified implementation
        sign_string = data + private_key
        return hashlib.sha256(sign_string.encode()).hexdigest()
    
    def verify_signature(self, data, signature, public_key):
        """Verify signature with public key"""
        # Simplified verification
        expected_public = self.derive_public_key(public_key)
        return public_key == expected_public