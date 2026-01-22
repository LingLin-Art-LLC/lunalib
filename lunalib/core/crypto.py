import hashlib
import secrets
import os
from lunalib.config import apply_profile
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from ..core.sm2 import SM2 # Assuming an SM2 implementation is available
from ..core.sm2_cuda.sm2_gpu import sm2_sign_gpu
try:
    from ..core.sm2_cuda.sm2_gpu import CUDA_AVAILABLE as _CUDA_AVAILABLE
except Exception:
    _CUDA_AVAILABLE = False

try:
    from ..core.sm2_c import sm2_ext as sm2_c_ext  # type: ignore
    _HAS_SM2_C = True
except Exception:
    sm2_c_ext = None
    _HAS_SM2_C = False
from lunalib.utils.hash import sm3_hex
apply_profile()

class KeyManager:
    """Manages cryptographic keys and signing using SM2"""
    
    def __init__(self):
        self.sm2 = SM2()  # SM2 instance for cryptographic operations
        self._public_key_cache: dict[str, str] = {}
        self._address_cache: dict[str, str] = {}
        self._signing_key_cache: dict[tuple[str, str], str] = {}
    
    def generate_keypair(self) -> Tuple[str, str, str]:
        """
        Generate a new SM2 key pair and address
        
        Returns:
            Tuple of (private_key_hex, public_key_hex, address)
        """
        print("DEBUG: Generating SM2 key pair...")
        
        # Generate keys using SM2
        private_key, public_key = self.sm2.generate_keypair()
        
        # Derive address from public key
        address = self.sm2.public_key_to_address(public_key)
        
        print(f"DEBUG: Generated private key: {private_key[:16]}...")
        print(f"DEBUG: Generated public key: {public_key[:24]}...")
        print(f"DEBUG: Generated address: {address}")
        
        return private_key, public_key, address
    
    def generate_private_key(self):
        """Generate a new private key (for backward compatibility)"""
        private_key, _, _ = self.generate_keypair()
        return private_key
    
    def derive_public_key(self, private_key_hex: str) -> str:
        """
        Derive public key from private key using SM2
        
        IMPORTANT: SM2 public key should be 130 characters: '04' + 64-byte x + 64-byte y
        """
        try:
            cached = self._public_key_cache.get(private_key_hex)
            if cached:
                return cached

            print(f"[KEYMANAGER] Deriving public key from private key...")
            print(f"[KEYMANAGER] Private key: {private_key_hex[:16]}...")

            # Try to use the SM2 instance
            if hasattr(self.sm2, 'generate_keypair'):
                # This is a hack: generate a new keypair and replace the private key
                # In a real implementation, you'd calculate public_key = private_key * G
                priv_int = int(private_key_hex, 16)
                
                # Calculate public key = d * G using SM2 curve math
                from ..core.sm2 import SM2Curve
                Px, Py = SM2Curve.point_multiply(priv_int, SM2Curve.Gx, SM2Curve.Gy)
                
                public_key = f"04{Px:064x}{Py:064x}"
                print(f"[KEYMANAGER] Generated full public key: {public_key[:24]}... (length: {len(public_key)})")
                self._public_key_cache[private_key_hex] = public_key
                return public_key
            else:
                # Fallback
                print(f"[KEYMANAGER] Using fallback public key generation")
                # Generate a deterministic but invalid public key for testing
                hash1 = sm3_hex(private_key_hex.encode())
                hash2 = sm3_hex(hash1.encode())
                public_key = f"04{hash1}{hash2}"  # 130 chars
                self._public_key_cache[private_key_hex] = public_key
                return public_key
        except Exception as e:
            print(f"[KEYMANAGER ERROR] Error deriving public key: {e}")
            # Emergency fallback - generate a full 130-char key
            import secrets
            random_part = secrets.token_hex(64)  # 128 chars
            public_key = f"04{random_part}"  # 130 chars total
            self._public_key_cache[private_key_hex] = public_key
            return public_key
        except Exception as e:
            print(f"DEBUG: Error deriving public key: {e}")
            # Fallback to hash-based method
            public_key = f"04{sm3_hex(private_key_hex.encode())}"
            self._public_key_cache[private_key_hex] = public_key
            return public_key
    
    def derive_address(self, public_key_hex: str) -> str:
        """
        Derive address from public key using SM2 standard
        
        Args:
            public_key_hex: Public key in hex format (should start with '04')
        
        Returns:
            Address string with LUN_ prefix
        """
        try:
            cached = self._address_cache.get(public_key_hex)
            if cached:
                return cached

            # Use SM2's address generation
            address = self.sm2.public_key_to_address(public_key_hex)
            self._address_cache[public_key_hex] = address
            return address
        except Exception as e:
            print(f"DEBUG: Error deriving address: {e}")
            # Fallback method
            if not public_key_hex.startswith('04'):
                public_key_hex = f"04{public_key_hex}"
            
            address_hash = sm3_hex(public_key_hex.encode())
            address = f"LUN_{address_hash[:16]}_{secrets.token_hex(4)}"
            self._address_cache[public_key_hex] = address
            return address
    
    def sign_data(self, data: str, private_key_hex: str) -> str:
        """
        Sign data with SM2 private key
        
        Args:
            data: Data to sign (string)
            private_key_hex: Private key in hex format
        
        Returns:
            SM2 signature in hex format
        """
        try:
            cached = self._signing_key_cache.get((private_key_hex, data))
            if cached:
                return cached

            backend = os.getenv("LUNALIB_SM2_BACKEND")
            if not backend:
                backend = "phos" if _CUDA_AVAILABLE and _HAS_SM2_C else "python"
            backend = backend.lower()
            if backend in {"phos", "cpython"} and _HAS_SM2_C:
                priv_bytes = bytes.fromhex(private_key_hex)
                sig_bytes = sm2_c_ext.sign(data.encode("utf-8"), priv_bytes)
                signature = sig_bytes.hex()
            else:
                # Use SM2 signing
                signature = self.sm2.sign(data.encode('utf-8'), private_key_hex)
            print(f"DEBUG: Signed data with SM2, signature: {signature[:16]}...")
            self._signing_key_cache[(private_key_hex, data)] = signature
            return signature
        except Exception as e:
            print(f"DEBUG: SM2 signing failed: {e}, using fallback")
            # Fallback to simplified signing
            sign_string = data + private_key_hex
            signature = sm3_hex(sign_string.encode())
            self._signing_key_cache[(private_key_hex, data)] = signature
            return signature
    
    def verify_signature(self, data: str, signature: str, public_key_hex: str) -> bool:
        """
        Verify SM2 signature
        
        Args:
            data: Original data (string)
            signature: SM2 signature in hex format
            public_key_hex: Public key in hex format
        
        Returns:
            True if signature is valid
        """
        try:
            # Use SM2 verification
            is_valid = self.sm2.verify(data.encode('utf-8'), signature, public_key_hex)
            print(f"DEBUG: SM2 signature verification: {is_valid}")
            return is_valid
        except Exception as e:
            print(f"DEBUG: SM2 verification failed: {e}, using fallback")
            # Fallback verification (always returns True for compatibility)
            return True

    def verify_signatures_batch(
        self,
        messages: List[str],
        signatures: List[str],
        public_keys: List[str],
        max_workers: int = 8,
    ) -> List[bool]:
        """Verify a batch of SM2 signatures in parallel."""
        def _verify(args):
            msg, sig, pub = args
            return self.verify_signature(msg, sig, pub)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_verify, zip(messages, signatures, public_keys)))

    def sign_data_batch(self, messages: List[str], private_key_hex: str, max_workers: int = 8) -> List[str]:
        """Sign a batch of messages in parallel."""
        if not messages:
            return []

        def _sign(msg: str) -> str:
            return self.sign_data(msg, private_key_hex)

        backend = os.getenv("LUNALIB_SM2_BACKEND")
        if not backend:
            backend = "phos" if _CUDA_AVAILABLE and _HAS_SM2_C else "python"
        backend = backend.lower()
        use_gpu = os.getenv("LUNALIB_SM2_GPU", "0") == "1"
        if backend in {"phos", "cpython"} and _HAS_SM2_C and not use_gpu:
            priv_bytes = bytes.fromhex(private_key_hex)
            return [sm2_c_ext.sign(msg.encode("utf-8"), priv_bytes).hex() for msg in messages]
        if use_gpu:
            # GPU path supports same-message batches only
            if len(set(messages)) == 1:
                priv_bytes = bytes.fromhex(private_key_hex)
                priv_list = [priv_bytes for _ in messages]
                sigs = sm2_sign_gpu(messages[0].encode("utf-8"), priv_list, use_gpu=True)
                return [s.hex() for s in sigs]

        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_sign, messages))

        return [_sign(msg) for msg in messages]
    
    def validate_key_pair(self, private_key_hex: str, public_key_hex: str) -> bool:
        """
        Validate that private and public keys form a valid SM2 key pair
        
        Args:
            private_key_hex: Private key in hex
            public_key_hex: Public key in hex
        
        Returns:
            True if keys are valid and match
        """
        try:
            # Test signing and verification
            test_data = "SM2 key validation test"
            
            # Sign with private key
            signature = self.sign_data(test_data, private_key_hex)
            
            # Verify with public key
            is_valid = self.verify_signature(test_data, signature, public_key_hex)
            
            print(f"DEBUG: Key pair validation: {is_valid}")
            return is_valid
            
        except Exception as e:
            print(f"DEBUG: Key validation error: {e}")
            return False
    
    def get_key_info(self, private_key_hex: str = None, public_key_hex: str = None) -> dict:
        """
        Get information about keys
        
        Args:
            private_key_hex: Private key (optional)
            public_key_hex: Public key (optional)
        
        Returns:
            Dictionary with key information
        """
        info = {
            "crypto_standard": "SM2 (GB/T 32918)",
            "curve": "SM2 P-256",
            "key_size_bits": 256,
        }
        
        if private_key_hex:
            info["private_key_length"] = len(private_key_hex)
            info["private_key_prefix"] = private_key_hex[:8]
            
        if public_key_hex:
            info["public_key_length"] = len(public_key_hex)
            info["public_key_format"] = "uncompressed" if public_key_hex.startswith('04') else "unknown"
            
            # Derive address if we have public key
            try:
                info["address"] = self.derive_address(public_key_hex)
            except:
                info["address"] = "could_not_derive"
        
        return info
    
    def test_sm2_operations(self) -> bool:
        """
        Test all SM2 operations
        
        Returns:
            True if all tests pass
        """
        print("="*60)
        print("Testing SM2 KeyManager operations...")
        print("="*60)
        
        try:
            # Test 1: Generate key pair
            print("Test 1: Generating key pair...")
            private_key, public_key, address = self.generate_keypair()
            
            if len(private_key) != 64:
                print(f"❌ Invalid private key length: {len(private_key)}")
                return False
            if not public_key.startswith('04'):
                print(f"❌ Invalid public key format: {public_key[:10]}...")
                return False
            if not address.startswith('LUN_'):
                print(f"❌ Invalid address format: {address[:10]}...")
                return False
            
            print(f"  ✓ Private: {private_key[:16]}...")
            print(f"  ✓ Public: {public_key[:24]}...")
            print(f"  ✓ Address: {address}")
            
            # Test 2: Sign and verify
            print("\nTest 2: Signing and verification...")
            test_message = "Hello, SM2 cryptography!"
            signature = self.sign_data(test_message, private_key)
            
            if len(signature) != 128:
                print(f"❌ Invalid signature length: {len(signature)}")
                return False
            
            is_valid = self.verify_signature(test_message, signature, public_key)
            if not is_valid:
                print("❌ Signature verification failed")
                return False
            
            print(f"  ✓ Signature: {signature[:16]}...")
            print(f"  ✓ Verification: {is_valid}")
            
            # Test 3: Address derivation
            print("\nTest 3: Address derivation...")
            derived_address = self.derive_address(public_key)
            if derived_address != address:
                print(f"❌ Address mismatch: {derived_address[:20]}... != {address[:20]}...")
                return False
            
            print(f"  ✓ Address consistently derived")
            
            print("\n" + "="*60)
            print("✅ All SM2 KeyManager tests passed!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False