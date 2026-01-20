import time
import sys
import os
from lunalib.utils.console import print_info, print_warn, print_error, print_success, print_debug

# --- Unicode-safe print for Windows console ---
def safe_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "ERROR" in msg:
        print_error(msg)
    elif "WARNING" in msg:
        print_warn(msg)
    elif "DEBUG" in msg:
        print_debug(msg)
    else:
        print_info(msg)
import hashlib
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

class TransactionSecurity:
    """Enhanced transaction security system with SM2 support"""
    
    def __init__(self):
        self.min_transaction_amount = 0.000001
        self.max_transaction_amount = 100000000
        self.required_fee = 0.00001
        self.rate_limits = {}
        self.blacklisted_addresses = set()
        self.stats = {
            "validate_count": 0,
            "validate_seconds": 0.0,
        }
        self.verbose = bool(int(os.getenv("LUNALIB_DEBUG", "0")))
        
        # Try to import SM2 KeyManager
        try:
            from ..core.crypto import KeyManager as SM2KeyManager
            self.key_manager = SM2KeyManager()
            self.sm2_available = True
            if self.verbose:
                safe_print("[SECURITY] SM2 KeyManager loaded successfully")
        except ImportError as e:
            self.key_manager = None
            self.sm2_available = False
            if self.verbose:
                safe_print(f"[SECURITY] SM2 KeyManager not available: {e}")
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Comprehensive transaction security validation with SM2"""
        start = time.perf_counter()
        tx_type = transaction.get("type", "").lower()

        fast_ok, fast_msg = self._fast_reject(transaction)
        if not fast_ok:
            elapsed = time.perf_counter() - start
            self.stats["validate_count"] += 1
            self.stats["validate_seconds"] += elapsed
            return False, fast_msg

        if tx_type == "gtx_genesis":
            result = self._validate_genesis_transaction(transaction)
        elif tx_type == "reward":
            result = self._validate_reward_transaction(transaction)
        elif tx_type == "transfer":
            result = self._validate_transfer_transaction(transaction)
        else:
            result = (False, f"Unknown transaction type: {tx_type}")

        elapsed = time.perf_counter() - start
        self.stats["validate_count"] += 1
        self.stats["validate_seconds"] += elapsed
        return result

    def _fast_reject(self, transaction: Dict) -> Tuple[bool, str]:
        """Cheap checks to reject malformed transactions before crypto."""
        tx_type = (transaction.get("type") or "").lower()

        # Common required fields
        for field in ("type", "timestamp"):
            if field not in transaction:
                return False, f"Missing required field: {field}"

        # Timestamp sanity (no far future)
        try:
            ts = float(transaction.get("timestamp"))
            if ts > time.time() + 300:
                return False, "Timestamp too far in future"
        except Exception:
            return False, "Invalid timestamp"

        # Type-specific cheap checks
        if tx_type in ("transfer", "transaction"):
            for field in ("from", "to", "amount"):
                if field not in transaction:
                    return False, f"Missing field: {field}"

            from_addr = transaction.get("from", "")
            to_addr = transaction.get("to", "")
            if not from_addr.startswith("LUN_") or not to_addr.startswith("LUN_"):
                return False, "Invalid address format"

            try:
                amount = float(transaction.get("amount", 0))
            except Exception:
                return False, "Invalid amount"
            if amount <= 0:
                return False, "Amount must be positive"

            sig = transaction.get("signature", "")
            pub = transaction.get("public_key", "")
            if not sig or not pub:
                return False, "Missing signature or public key"
            if len(sig) != 128:
                return False, "Invalid signature length"

        return True, "OK"

    def validate_transaction_security_batch(self, transactions: list[Dict], max_workers: int = 8) -> list[Tuple[bool, str]]:
        """Validate multiple transactions in parallel."""
        fast_results: list[Tuple[bool, str]] = []
        pending = []

        for tx in transactions:
            ok, msg = self._fast_reject(tx)
            if ok:
                pending.append(tx)
            fast_results.append((ok, msg))

        validated: list[Tuple[bool, str]] = []
        if pending:
            # Batch verify signatures for transfers
            transfers = [tx for tx in pending if (tx.get("type") or "").lower() in ("transfer", "transaction")]
            non_transfers = [tx for tx in pending if tx not in transfers]

            sig_ok_map = {}
            if transfers and self.sm2_available and self.key_manager:
                messages = [self._get_signing_data(tx) for tx in transfers]
                signatures = [tx.get("signature", "") for tx in transfers]
                public_keys = [tx.get("public_key", "") for tx in transfers]
                results = self.key_manager.verify_signatures_batch(messages, signatures, public_keys, max_workers=max_workers)
                sig_ok_map = {id(tx): ok for tx, ok in zip(transfers, results)}

            def _validate(tx: Dict) -> Tuple[bool, str]:
                tx_type = (tx.get("type") or "").lower()
                if tx_type in ("transfer", "transaction"):
                    if transfers and id(tx) in sig_ok_map and not sig_ok_map[id(tx)]:
                        return False, "Invalid SM2 signature"
                    return self._validate_transfer_transaction_no_sig(tx)
                return self.validate_transaction_security(tx)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                validated = list(pool.map(_validate, pending))

        # Merge fast + validated in order
        output = []
        pending_idx = 0
        for ok, msg in fast_results:
            if ok:
                output.append(validated[pending_idx])
                pending_idx += 1
            else:
                output.append((False, msg))

        return output

    def get_stats(self) -> Dict:
        return self.stats.copy()
    
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

    def _validate_transfer_transaction_no_sig(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transfer transaction without signature check (assumed verified)."""
        required_fields = ["from", "to", "amount", "signature", "public_key", "nonce"]
        for field in required_fields:
            if field not in transaction:
                return False, f"Missing field: {field}"

        amount = transaction.get("amount", 0)
        if amount < self.min_transaction_amount:
            return False, f"Amount below minimum: {self.min_transaction_amount}"
        if amount > self.max_transaction_amount:
            return False, f"Amount above maximum: {self.max_transaction_amount}"

        fee = transaction.get("fee", 0)
        if fee < self.required_fee:
            return False, f"Insufficient fee: {fee} (required: {self.required_fee})"

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
                if self.verbose:
                    safe_print(f"[SECURITY] Skipping signature check for system/unsigned transaction")
                return True
            
            # Check SM2 signature length (should be 128 hex chars = 64 bytes)
            if len(signature) != 128:
                if self.verbose:
                    safe_print(f"[SECURITY] Invalid SM2 signature length: {len(signature)} (expected 128)")
                return False
            
            # Check if all characters are valid hex
            if not all(c in "0123456789abcdefABCDEF" for c in signature):
                if self.verbose:
                    safe_print(f"[SECURITY] Signature contains non-hex characters")
                return False
            
            # Check public key format (should start with '04' for uncompressed)
            if not public_key.startswith('04'):
                if self.verbose:
                    safe_print(f"[SECURITY] Invalid public key format: {public_key[:20]}...")
                return False
            
            # Use KeyManager for verification if available
            if self.sm2_available and self.key_manager:
                # Create signing data string
                signing_data = self._get_signing_data(transaction)
                
                # Verify signature
                is_valid = self.key_manager.verify_signature(signing_data, signature, public_key)
                if self.verbose:
                    safe_print(f"[SECURITY] SM2 signature verification: {is_valid}")
                return is_valid
            
            # Fallback: Basic format check if SM2 not available
            if self.verbose:
                safe_print(f"[SECURITY] SM2 not available, using basic signature validation")
            return len(signature) == 128 and signature.startswith(('04', '03', '02'))
            
        except Exception as e:
            if self.verbose:
                safe_print(f"[SECURITY] Signature validation error: {e}")
            return False
    
    def _get_signing_data(self, transaction: Dict) -> str:
        """Create data string for signing"""
        cached = transaction.get("_signing_data")
        if cached:
            return cached

        # Create copy without signature, hash, and public_key (must match creation)
        tx_copy = {k: v for k, v in transaction.items()
                  if k not in ['signature', 'hash', 'public_key']}

        for key in ['amount', 'fee']:
            if key in tx_copy:
                tx_copy[key] = float(tx_copy[key])
        if 'timestamp' in tx_copy:
            tx_copy['timestamp'] = int(tx_copy['timestamp'])

        import json
        signing = json.dumps(tx_copy, sort_keys=True, separators=(',', ':'))
        transaction["_signing_data"] = signing
        transaction["_signing_hash"] = hashlib.sha256(signing.encode()).hexdigest()
        return signing
    
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
        safe_print(f"[SECURITY] Using legacy signature validation")
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