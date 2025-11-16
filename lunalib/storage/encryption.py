import os
import json
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Optional
class EncryptionManager:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self):
        self.salt = b'luna_wallet_salt'  # In production, use random salt per wallet
    
    def encrypt_wallet(self, wallet_data: Dict, password: str) -> Dict:
        """Encrypt wallet data"""
        try:
            # Derive key from password
            key = self._derive_key(password)
            fernet = Fernet(key)
            
            # Encrypt private key separately
            private_key = wallet_data.get('private_key', '')
            if private_key:
                encrypted_private = fernet.encrypt(private_key.encode()).decode()
                wallet_data['encrypted_private_key'] = encrypted_private
                del wallet_data['private_key']
            
            # Encrypt entire wallet data
            wallet_json = json.dumps(wallet_data)
            encrypted_wallet = fernet.encrypt(wallet_json.encode()).decode()
            
            return {
                'encrypted_data': encrypted_wallet,
                'version': '1.0',
                'salt': base64.b64encode(self.salt).decode()
            }
            
        except Exception as e:
            print(f"Encryption error: {e}")
            return {}
    
    def decrypt_wallet(self, encrypted_data: Dict, password: str) -> Optional[Dict]:
        """Decrypt wallet data"""
        try:
            encrypted_wallet = encrypted_data.get('encrypted_data', '')
            if not encrypted_wallet:
                return None
            
            # Derive key from password
            key = self._derive_key(password)
            fernet = Fernet(key)
            
            # Decrypt wallet
            decrypted_data = fernet.decrypt(encrypted_wallet.encode())
            wallet_data = json.loads(decrypted_data.decode())
            
            # Decrypt private key if present
            encrypted_private = wallet_data.get('encrypted_private_key')
            if encrypted_private:
                private_key = fernet.decrypt(encrypted_private.encode()).decode()
                wallet_data['private_key'] = private_key
                del wallet_data['encrypted_private_key']
            
            return wallet_data
            
        except Exception as e:
            print(f"Decryption error: {e}")
            return None
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def verify_password(self, encrypted_data: Dict, password: str) -> bool:
        """Verify password without full decryption"""
        try:
            # Try to decrypt a small part to verify password
            key = self._derive_key(password)
            fernet = Fernet(key)
            
            # This will throw an exception if password is wrong
            fernet.decrypt(encrypted_data.get('encrypted_data', 'A')[:10].encode())
            return True
        except:
            return False
    
    def encrypt_data(self, data: str, password: str) -> str:
        """Encrypt arbitrary data"""
        key = self._derive_key(password)
        fernet = Fernet(key)
        return fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str, password: str) -> str:
        """Decrypt arbitrary data"""
        key = self._derive_key(password)
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data.encode()).decode()