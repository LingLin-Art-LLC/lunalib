import time
import hashlib
from typing import Dict, Tuple

class TransactionSecurity:
    """Enhanced transaction security system with SM2 support"""
    
    def __init__(self):
        self.min_transaction_amount = 0.000001
        self.max_transaction_amount = 100000000
        self.required_fee = 0.00001
        self.rate_limits = {}
        self.blacklisted_addresses = set()
        
        # Try to import SM2 KeyManager
        try:
            from ..core.crypto import KeyManager as SM2KeyManager
            self.key_manager = SM2KeyManager()
            self.sm2_available = True
            print("[SECURITY] SM2 KeyManager loaded successfully")
        except ImportError as e:
            self.key_manager = None
            self.sm2_available = False
            print(f"[SECURITY] SM2 KeyManager not available: {e}")
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Comprehensive transaction security validation with SM2"""
        tx_type = transaction.get("type", "").lower()
        
        if tx_type == "gtx_genesis":
            return self._validate_genesis_transaction(transaction)
        elif tx_type == "reward":
            return self._validate_reward_transaction(transaction)
        elif tx_type == "transfer":
            return self._validate_transfer_transaction(transaction)
        else:
            return False, f"Unknown transaction type: {tx_type}"
    
    def _validate_genesis_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate GTX Genesis transaction"""
        required_fields = ["bill_serial", "denomination", "mining_difficulty", "hash", "nonce"]
        for field in required_fields:
            if field not in transaction:
                return False, f"Missing GTX field: {field}"
        
        # Validate denomination
        denomination = transaction.get("denomination")
        valid_denominations = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
        if denomination not in valid_denominations:
            return False, f"Invalid denomination: {denomination}"
        
        # Validate mining proof
        if not self._validate_mining_proof(transaction):
            return False, "Invalid mining proof"
        
        return True, "Valid GTX Genesis transaction"
    
    def _validate_reward_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate reward transaction"""
        required_fields = ["from", "to", "amount", "block_height", "hash"]
        for field in required_fields:
            if field not in transaction:
                return False, f"Missing reward field: {field}"
        
        # Only network can create rewards
        if transaction.get("from") != "network":
            return False, "Unauthorized reward creation"
        
        return True, "Valid reward transaction"
    
    def _validate_transfer_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transfer transaction with SM2 signature"""
        required_fields = ["from", "to", "amount", "signature", "public_key", "nonce"]
        for field in required_fields:
            if field not in transaction:
                return False, f"Missing field: {field}"
        
        # Amount validation
        amount = transaction.get("amount", 0)
        if amount < self.min_transaction_amount:
            return False, f"Amount below minimum: {self.min_transaction_amount}"
        if amount > self.max_transaction_amount:
            return False, f"Amount above maximum: {self.max_transaction_amount}"
        
        # Fee validation
        fee = transaction.get("fee", 0)
        if fee < self.required_fee:
            return False, f"Insufficient fee: {fee} (required: {self.required_fee})"
        
        # SM2 Signature validation
        if not self._validate_signature_sm2(transaction):
            return False, "Invalid SM2 signature"
        
        # Anti-spam checks
        from_address = transaction.get("from", "")
        if not self._check_rate_limit(from_address):
            return False, "Rate limit exceeded"
        
        if self._is_blacklisted(from_address):
            return False, "Address is blacklisted"
        
        return True, "Valid transfer transaction"
    
    def _validate_signature_sm2(self, transaction: Dict) -> bool:
        """Validate transaction signature using SM2"""
        try:
            signature = transaction.get("signature", "")
            public_key = transaction.get("public_key", "")
            tx_type = transaction.get("type", "").lower()
            
            # Skip system transactions
            if tx_type in ["gtx_genesis", "reward"]:
                return True
            
            # For unsigned test transactions
            if signature in ["system", "unsigned", "test"]:
                print(f"[SECURITY] Skipping signature check for system/unsigned transaction")
                return True
            
            # Check SM2 signature length (should be 128 hex chars = 64 bytes)
            if len(signature) != 128:
                print(f"[SECURITY] Invalid SM2 signature length: {len(signature)} (expected 128)")
                return False
            
            # Check if all characters are valid hex
            if not all(c in "0123456789abcdefABCDEF" for c in signature):
                print(f"[SECURITY] Signature contains non-hex characters")
                return False
            
            # Check public key format (should start with '04' for uncompressed)
            if not public_key.startswith('04'):
                print(f"[SECURITY] Invalid public key format: {public_key[:20]}...")
                return False
            
            # Use KeyManager for verification if available
            if self.sm2_available and self.key_manager:
                # Create signing data string
                signing_data = self._get_signing_data(transaction)
                
                # Verify signature
                is_valid = self.key_manager.verify_signature(signing_data, signature, public_key)
                print(f"[SECURITY] SM2 signature verification: {is_valid}")
                return is_valid
            
            # Fallback: Basic format check if SM2 not available
            print(f"[SECURITY] SM2 not available, using basic signature validation")
            return len(signature) == 128 and signature.startswith(('04', '03', '02'))
            
        except Exception as e:
            print(f"[SECURITY] Signature validation error: {e}")
            return False
    
    def _get_signing_data(self, transaction: Dict) -> str:
        """Create data string for signing"""
        # Create copy without signature and hash
        tx_copy = {k: v for k, v in transaction.items() 
                  if k not in ['signature', 'hash']}
        
        # Sort keys and convert to string
        import json
        return json.dumps(tx_copy, sort_keys=True)
    
    def _validate_mining_proof(self, transaction: Dict) -> bool:
        """Validate mining proof-of-work"""
        try:
            difficulty = transaction.get("mining_difficulty", 0)
            bill_hash = transaction.get("hash", "")
            
            # Verify difficulty requirement
            target = "0" * difficulty
            return bill_hash.startswith(target)
        except:
            return False
    
    def _validate_signature(self, transaction: Dict) -> bool:
        """Legacy signature validation (for backward compatibility)"""
        print(f"[SECURITY] Using legacy signature validation")
        return self._validate_signature_sm2(transaction)
    
    def _check_rate_limit(self, address: str) -> bool:
        """Check transaction rate limiting"""
        now = time.time()
        
        if address not in self.rate_limits:
            self.rate_limits[address] = []
        
        # Remove old entries (older than 1 minute)
        self.rate_limits[address] = [t for t in self.rate_limits[address] if now - t < 60]
        
        # Check if over limit (10 transactions per minute)
        if len(self.rate_limits[address]) >= 10:
            return False
        
        self.rate_limits[address].append(now)
        return True
    
    def _is_blacklisted(self, address: str) -> bool:
        """Check if address is blacklisted"""
        return address.lower() in self.blacklisted_addresses
    
    def blacklist_address(self, address: str):
        """Add address to blacklist"""
        self.blacklisted_addresses.add(address.lower())
    
    def calculate_security_score(self, transaction: Dict) -> int:
        """Calculate security score for transaction with SM2 bonus"""
        score = 0
        
        # Signature strength (SM2 gives higher score)
        signature = transaction.get("signature", "")
        if len(signature) == 128:  # SM2 signature
            score += 60  # Higher score for SM2
        elif len(signature) == 64:  # Legacy ECDSA
            score += 40
        
        # Public key presence
        public_key = transaction.get("public_key", "")
        if public_key and public_key.startswith('04'):
            score += 30  # SM2 uncompressed format
        
        # Timestamp freshness
        timestamp = transaction.get("timestamp", 0)
        if time.time() - timestamp < 600:  # 10 minutes
            score += 20
        
        # Nonce uniqueness
        if transaction.get("nonce"):
            score += 10
        
        # Additional security features
        if transaction.get("security_hash"):
            score += 10
        
        return min(score, 100)