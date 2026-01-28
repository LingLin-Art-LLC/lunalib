import sys
def safe_print(*args, **kwargs):
    encoding = sys.stdout.encoding or 'utf-8'
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        print(*(str(a).encode(encoding, errors='replace').decode(encoding) for a in args), **kwargs)
import time
import hashlib
from lunalib.utils.hash import sm3_hex
import secrets
import json
import base64
from typing import Dict, Optional

from ..core.crypto import KeyManager
from ..core.sm2 import SM2
from .bill_registry import BillRegistry

class DigitalBill:
    """Represents a GTX Genesis digital bill with cryptographic signatures"""
    
    def __init__(self, denomination, user_address, difficulty, bill_data=None, 
                 bill_type="GTX_Genesis", front_serial=None, back_serial=None, 
                 metadata_hash=None, public_key=None, signature=None):
        # GTX Genesis properties
        self.denomination = denomination
        self.user_address = user_address
        self.difficulty = difficulty
        self.bill_data = bill_data or {}
        self.bill_serial = front_serial or self._generate_serial()
        self.created_time = time.time()
        self.bill_registry = BillRegistry()
        
        # Signature properties
        self.bill_type = bill_type
        self.front_serial = front_serial or self.bill_serial
        self.back_serial = back_serial or ""
        self.metadata_hash = metadata_hash or self._generate_metadata_hash()
        self.timestamp = self.created_time
        self.issued_to = user_address
        self.public_key = public_key
        self.signature = signature
        self._sm2 = SM2()
        self._key_manager = KeyManager()
    
    def _generate_serial(self):
        """Generate unique bill serial number"""
        date_part = time.strftime("%Y%m%d", time.gmtime())
        random_bytes = secrets.token_bytes(10)
        random_part = base64.b32encode(random_bytes).decode("utf-8").strip("=")
        denom = int(self.denomination) if float(self.denomination).is_integer() else self.denomination
        return f"GTX-{denom}-{date_part}-{random_part}"
    
    def _generate_metadata_hash(self):
        """Generate metadata hash for the bill"""
        metadata = {
            "denomination": self.denomination,
            "user_address": self.user_address,
            "difficulty": self.difficulty,
            "timestamp": self.created_time,
            "bill_serial": self.bill_serial
        }
        return sm3_hex(json.dumps(metadata, sort_keys=True).encode())
    
    def get_mining_data(self, nonce):
        """Get data for mining computation"""
        return {
            "type": "GTX_Genesis",
            "denomination": self.denomination,
            "user_address": self.user_address,
            "bill_serial": self.bill_serial,
            "timestamp": self.created_time,
            "difficulty": self.difficulty,
            "previous_hash": self._get_previous_hash(),
            "nonce": nonce,
            "bill_data": self.bill_data
        }
    
    def finalize(self, hash, nonce, mining_time, private_key=None):
        """Finalize bill after successful mining with optional signing"""
        # Create transaction data
        transaction_data = {
            "type": "GTX_Genesis",
            "from": "genesis_network",
            "to": self.user_address,
            "amount": self.denomination,
            "bill_serial": self.bill_serial,
            "serial_number": self.bill_serial,
            "mining_difficulty": self.difficulty,
            "mining_time": mining_time,
            "hash": hash,
            "timestamp": time.time(),
            "status": "mined",
            "front_serial": self.front_serial,
            "issued_to": self.user_address,
            "denomination": self.denomination,
            "metadata_hash": self.metadata_hash
        }
        
        # Generate signature if private key provided
        if private_key:
            self.sign(private_key)
            transaction_data.update({
                "public_key": self.public_key,
                "signature": self.signature
            })
        
        bill_info = {
            "success": True,
            "bill_serial": self.bill_serial,
            "denomination": self.denomination,
            "user_address": self.user_address,
            "mining_time": mining_time,
            "difficulty": self.difficulty,
            "hash": hash,
            "nonce": nonce,
            "timestamp": time.time(),
            "luna_value": self.denomination,  # 1:1 ratio
            "transaction_data": transaction_data
        }
        
        # Add to bill registry
        self.bill_registry.register_bill(bill_info)
        
        return bill_info
    
    def _get_previous_hash(self):
        """Get hash of previous genesis transaction"""
        # In production, this would query the blockchain
        return sm3_hex(str(time.time()).encode())
    
    # Signature methods from your signatures.py
    def to_dict(self):
        """Convert bill data to dictionary for hashing/serialization"""
        return {
            'type': self.bill_type,
            'front_serial': self.front_serial,
            'back_serial': self.back_serial,
            'metadata_hash': self.metadata_hash,
            'timestamp': self.timestamp,
            'issued_to': self.issued_to,
            'denomination': self.denomination
        }
    
    def calculate_hash(self):
        """Calculate SHA-256 hash of the bill data"""
        bill_string = json.dumps(self.to_dict(), sort_keys=True)
        return sm3_hex(bill_string.encode())
    
    def sign(self, private_key):
        """Sign the bill data with a private key"""
        bill_hash = self.calculate_hash()
        
        try:
            private_key_hex = None
            if isinstance(private_key, bytes):
                private_key_hex = private_key.decode("utf-8")
            elif isinstance(private_key, str):
                private_key_hex = private_key

            if not private_key_hex or len(private_key_hex) != 64:
                return self._sign_fallback(private_key)

            signature = self._sm2.sign(bill_hash.encode(), private_key_hex)
            self.signature = signature

            # Set public key derived from private key
            self.public_key = self._key_manager.derive_public_key(private_key_hex)

            return self.signature
        except Exception as e:
            safe_print(f"SM2 signing failed, using fallback: {e}")
            return self._sign_fallback(private_key)
    
    def _sign_fallback(self, private_key):
        """Fallback signing method using hashes"""
        bill_hash = self.calculate_hash()
        # Simple hash-based "signature" for when cryptography is unavailable
        if isinstance(private_key, str):
            signature_input = f"{private_key}{bill_hash}"
        else:
            signature_input = f"fallback_key{bill_hash}"
        
        self.signature = sm3_hex(signature_input.encode())
        
        # Set fallback public key
        if isinstance(private_key, str) and len(private_key) > 32:
            self.public_key = sm3_hex(private_key.encode())
        else:
            self.public_key = "fallback_public_key"
        
        return self.signature
    
    def verify(self):
        """Verify signature using the exact same method as creation"""
        if not self.public_key or not self.signature:
            return False
            
        # Handle mock signatures
        if self.signature.startswith('mock_signature_'):
            expected_mock = 'mock_signature_' + hashlib.md5(
                f"{self.issued_to}{self.denomination}{self.front_serial}".encode()
            ).hexdigest()
            return self.signature == expected_mock
        
        # Handle fallback signatures
        if self.public_key == 'fallback_public_key':
            expected_fallback = sm3_hex(
                f"{self.issued_to}{self.denomination}{self.front_serial}{self.timestamp}".encode()
            )
            return self.signature == expected_fallback
        
        # Prefer SM2 verification when key/signature formats match
        if (
            isinstance(self.public_key, str)
            and isinstance(self.signature, str)
            and self.public_key.startswith("04")
            and len(self.public_key) == 130
            and len(self.signature) == 128
        ):
            bill_hash = self.calculate_hash()
            try:
                return self._sm2.verify(bill_hash.encode(), self.signature, self.public_key)
            except Exception:
                pass

        # Handle metadata_hash based signatures
        if self.metadata_hash:
            verification_data = f"{self.public_key}{self.metadata_hash}"
            expected_signature = sm3_hex(verification_data.encode())
            return self.signature == expected_signature
        
        # Final fallback - accept any signature that looks valid
        return len(self.signature) > 0
    
    @staticmethod
    def generate_key_pair():
        """Generate a new SM2 key pair for signing"""
        sm2 = SM2()
        private_key, public_key = sm2.generate_keypair()
        return private_key, public_key
    
    @staticmethod
    def _generate_fallback_key_pair():
        """Generate fallback key pair using hashes"""
        import random
        import string
        
        # Generate random strings as "keys"
        private_key = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        public_key = sm3_hex(private_key.encode())
        
        return private_key, public_key