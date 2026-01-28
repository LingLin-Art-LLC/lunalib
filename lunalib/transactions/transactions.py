import time
import sys
import os
from lunalib.utils.console import print_info, print_warn, print_error, print_success, print_debug

# --- Unicode-safe print for Windows console ---
def safe_print(*args, **kwargs):
    # 既存のsafe_print呼び出しを用途別に色分けprintへ置換するためのラッパー
    msg = " ".join(str(a) for a in args)
    if "ERROR" in msg:
        print_error(msg)
    elif "WARNING" in msg:
        print_warn(msg)
    elif "DEBUG" in msg:
        print_debug(msg)
    else:
        print_info(msg)
import json
from typing import Dict, Optional, Tuple, List
from ..core.mempool import MempoolManager
from .validator import TransactionValidator
from lunalib.utils.hash import sm3_hex
from lunalib.utils.validation import is_valid_address, sanitize_memo

# Import REAL SM2 KeyManager from crypto module
try:
    from ..core.crypto import KeyManager as SM2KeyManager
    SM2_AVAILABLE = True
    if os.getenv("LUNALIB_DEBUG", "0") == "1":
        safe_print("DEBUG: Using SM2 KeyManager from crypto module")
except ImportError as e:
    SM2_AVAILABLE = False
    safe_print(f"WARNING: SM2 KeyManager not available: {e}")

class TransactionSecurity:
    """Transaction security validation and risk assessment"""
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction structure"""
        required_fields = ['type', 'from', 'to', 'amount', 'timestamp', 'hash']
        for field in required_fields:
            if field not in transaction:
                return False, f'Missing required field: {field}'
        
        # Validate amount
        if transaction['amount'] <= 0 and transaction['type'] != 'reward':
            return False, 'Amount must be positive'
            
        return True, 'Valid'
    
    def assess_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        amount = transaction.get('amount', 0)
        tx_type = transaction.get('type', 'transfer')
        
        if tx_type in ['gtx_genesis', 'reward']:
            return 'very_low', 'System transaction'
        
        if amount > 1000000:
            return 'high', 'Very large transaction'
        elif amount > 100000:
            return 'medium', 'Large transaction'
        else:
            return 'low', 'Normal transaction'

class FeeCalculator:
    """Simple fee calculation"""
    
    def __init__(self):
        self.fee_config = {
            'transfer': 0.001,
            'reward': 0.0,
            'gtx_genesis': 0.0
        }
    
    def get_fee(self, transaction_type: str) -> float:
        """Get fee for transaction type"""
        return self.fee_config.get(transaction_type, 0.001)

class TransactionManager:
    """Handles transaction creation, signing, validation, and broadcasting"""
    
    def __init__(self, network_endpoints: List[str] = None):
        self.security = TransactionSecurity()
        self.fee_calculator = FeeCalculator()
        self.mempool_manager = MempoolManager(network_endpoints)
        self.verbose = bool(int(os.getenv("LUNALIB_DEBUG", "0")))
        self._public_key_cache: Dict[str, str] = {}
        
        # Initialize SM2 KeyManager if available
        if SM2_AVAILABLE:
            self.key_manager = SM2KeyManager()
        else:
            safe_print("ERROR: SM2 KeyManager not available - cannot sign transactions")
            self.key_manager = None
    
    def create_transaction(self, from_address: str, to_address: str, amount: float, 
                        private_key: Optional[str] = None, memo: str = "",
                        transaction_type: str = "transfer", public_key: Optional[str] = None) -> Dict:
        """Create and sign a transaction"""

        if transaction_type in ("transfer", "transaction"):
            if not is_valid_address(from_address) or not is_valid_address(to_address):
                raise ValueError("Invalid from/to address")
            if float(amount) <= 0:
                raise ValueError("Amount must be positive")
        
        # Calculate fee
        fee = self.fee_calculator.get_fee(transaction_type)
        
        transaction = {
            "type": transaction_type,
            "from": from_address,
            "to": to_address,
            "amount": float(amount),
            "fee": fee,
            "nonce": int(time.time() * 1000),
            "timestamp": int(time.time()),
            "memo": sanitize_memo(memo),
            "version": "2.0"  # Version 2.0 = SM2 signatures
        }
        
        # Sign transaction with SM2 if private key provided
        if private_key and self.key_manager:
            try:
                # Sign the transaction data
                tx_string = self._get_signing_data(transaction)
                transaction["_signing_data"] = tx_string
                transaction["_signing_hash"] = sm3_hex(tx_string.encode())
                
                if self.verbose:
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Signing data: {tx_string}")
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Private key available: {bool(private_key)}")
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Private key length: {len(private_key)}")
                
                signature = self.key_manager.sign_data(tx_string, private_key)

                # Get public key from cache or provided value
                if not public_key:
                    public_key = self._public_key_cache.get(private_key)
                    if not public_key:
                        public_key = self.key_manager.derive_public_key(private_key)
                        self._public_key_cache[private_key] = public_key
                
                if self.verbose:
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Generated signature: {signature}")
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Generated public key: {public_key}")
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Signature length: {len(signature)}")
                    safe_print(f"[TRANSACTIONS CREATE DEBUG] Public key length: {len(public_key)}")
                
                transaction["signature"] = signature
                transaction["public_key"] = public_key
                
                # Optionally test verification (can be expensive)
                if self.verbose or bool(int(os.getenv("LUNALIB_TX_SELF_VERIFY", "0"))):
                    test_verify = self.key_manager.verify_signature(tx_string, signature, public_key)
                    if self.verbose:
                        safe_print(f"[TRANSACTIONS CREATE DEBUG] Immediate self-verification: {test_verify}")
                    if not test_verify and self.verbose:
                        safe_print(f"[TRANSACTIONS CREATE ERROR] Signature doesn't verify immediately!")
                        safe_print(f"[TRANSACTIONS CREATE ERROR] This suggests an SM2 implementation issue")
                    
            except Exception as e:
                if self.verbose:
                    safe_print(f"[TRANSACTIONS CREATE ERROR] Signing failed: {e}")
                import traceback
                traceback.print_exc()
                transaction["signature"] = "unsigned"
                transaction["public_key"] = "unsigned"
        else:
            # For unsigned transactions (rewards, gtx_genesis)
            transaction["signature"] = "unsigned"
            transaction["public_key"] = "unsigned"
        
        # Calculate transaction hash (must be last)
        transaction["hash"] = self._calculate_transaction_hash(transaction)
        
        return transaction

    def create_transactions_batch(
        self,
        from_address: str,
        to_address: str,
        amounts: List[float],
        private_key: Optional[str] = None,
        memo_prefix: str = "",
        transaction_type: str = "transfer",
        public_key: Optional[str] = None,
        parallel_sign: bool = True,
        sign_workers: int = 8,
    ) -> List[Dict]:
        """Create and sign a batch of transactions efficiently."""
        transactions: List[Dict] = []
        if not amounts:
            return transactions

        fee = self.fee_calculator.get_fee(transaction_type)
        now = int(time.time())

        if private_key and self.key_manager and not public_key:
            public_key = self._public_key_cache.get(private_key)
            if not public_key:
                public_key = self.key_manager.derive_public_key(private_key)
                self._public_key_cache[private_key] = public_key

        for idx, amount in enumerate(amounts):
            memo = f"{memo_prefix}{idx}" if memo_prefix else ""
            transaction = {
                "type": transaction_type,
                "from": from_address,
                "to": to_address,
                "amount": float(amount),
                "fee": fee,
                "nonce": int(time.time() * 1000) + idx,
                "timestamp": now,
                "memo": memo,
                "version": "2.0",
            }
            transactions.append(transaction)

        if private_key and self.key_manager:
            signing_payloads = []
            for tx in transactions:
                tx_string = self._get_signing_data(tx)
                tx["_signing_data"] = tx_string
                tx["_signing_hash"] = sm3_hex(tx_string.encode())
                signing_payloads.append(tx_string)

            if parallel_sign:
                signatures = self.key_manager.sign_data_batch(
                    signing_payloads, private_key, max_workers=sign_workers
                )
            else:
                signatures = [self.key_manager.sign_data(p, private_key) for p in signing_payloads]

            for tx, signature in zip(transactions, signatures):
                tx["signature"] = signature
                tx["public_key"] = public_key or ""

        return transactions
    
    def send_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Send transaction to mempool for broadcasting"""
        try:
            # Validate transaction first
            is_valid, message = self.validate_transaction(transaction)
            if not is_valid:
                return False, f"Validation failed: {message}"
            
            # Signature is validated inside validate_transaction (TransactionSecurity)
            
            # Add to mempool
            success = self.mempool_manager.add_transaction(transaction)
            if success:
                return True, f"Transaction added to mempool: {transaction.get('hash', '')[:16]}..."
            else:
                return False, "Failed to add transaction to mempool"
                
        except Exception as e:
            return False, f"Error sending transaction: {str(e)}"
    
    # TRANSFER TRANSACTION
    def create_transfer(self, from_address: str, to_address: str, amount: float,
                       private_key: str, memo: str = "") -> Dict:
        """Create a transfer transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            private_key=private_key,
            memo=memo,
            transaction_type="transfer"
        )
    
    # SYSTEM TRANSACTIONS (unsigned)
    def create_gtx_transaction(self, bill_info: Dict) -> Dict:
        """Create GTX Genesis transaction from mined bill"""
        tx_data = bill_info.get("transaction_data") or {}
        bill_serial = bill_info.get("bill_serial") or bill_info.get("serial") or tx_data.get("bill_serial") or ""
        denomination = bill_info.get("denomination", tx_data.get("denomination", 0))
        mining_difficulty = bill_info.get("difficulty", tx_data.get("mining_difficulty", 0))
        nonce = bill_info.get("nonce", tx_data.get("nonce", 0))
        timestamp = bill_info.get("timestamp", tx_data.get("timestamp", int(time.time())))
        to_address = (
            bill_info.get("user_address")
            or tx_data.get("to")
            or tx_data.get("issued_to")
            or bill_info.get("issued_to")
            or "unknown"
        )
        from_address = tx_data.get("from") or bill_info.get("from") or "genesis_network"
        issued_to = bill_info.get("issued_to") or tx_data.get("issued_to") or to_address
        transaction = {
            "type": "gtx_genesis",
            "from": from_address,
            "to": to_address,
            "amount": float(denomination or 0),
            "fee": 0.0,
            "timestamp": int(timestamp),
            "bill_serial": bill_serial,
            "serial_number": bill_serial,
            "issued_to": issued_to,
            "denomination": denomination,
            "nonce": nonce,
            "mining_difficulty": mining_difficulty,
            "signature": "system",
            "public_key": "system",
            "version": "2.0"
        }
        transaction["hash"] = self._calculate_transaction_hash(transaction)
        return transaction
    
    def create_reward_transaction(self, to_address: str, amount: float, 
                                block_height: int) -> Dict:
        """Create reward transaction"""
        transaction = {
            "type": "reward",
            "from": "ling country",
            "to": to_address,
            "amount": float(amount),
            "fee": 0.0,
            "block_height": block_height,
            "timestamp": int(time.time()),
            "signature": "ling country",
            "public_key": "ling country",
            "version": "2.0"
        }
        transaction["hash"] = self._calculate_transaction_hash(transaction)
        return transaction
    
    # VALIDATION METHODS
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction using security module"""
        return self.security.validate_transaction(transaction)
    
    def verify_transaction_signature(self, transaction: Dict) -> bool:
        """Verify transaction signature with SM2"""
        try:
            signature = transaction.get("signature", "")
            tx_type = transaction.get("type", "").lower()
            
            # System transactions are always valid
            if signature in ["system", "unsigned", "test"]:
                if self.verbose:
                    safe_print(f"[TRANSACTIONS] Skipping signature check for {signature} transaction")
                return True
            
            if not self.key_manager:
                if self.verbose:
                    safe_print("[TRANSACTIONS] No key manager available for verification")
                return False
            
            # Check SM2 signature format
            if len(signature) != 128:
                if self.verbose:
                    safe_print(f"[TRANSACTIONS] Invalid SM2 signature length: {len(signature)} (expected 128)")
                return False
            
            # Get signing data (without public_key!)
            sign_data = transaction.get("_signing_data") or self._get_signing_data(transaction)
            if "_signing_data" not in transaction:
                transaction["_signing_data"] = sign_data
                transaction["_signing_hash"] = sm3_hex(sign_data.encode())
            public_key = transaction.get("public_key", "")
            
            if self.verbose:
                safe_print(f"[TRANSACTIONS VERIFY] Signing data length: {len(sign_data)}")
                safe_print(f"[TRANSACTIONS VERIFY] Signing data (first 100 chars): {sign_data[:100]}")
            
            # Try to verify with KeyManager
            if self.verbose:
                safe_print(f"[TRANSACTIONS] Attempting verification...")
            is_valid = self.key_manager.verify_signature(sign_data, signature, public_key)
            
            if self.verbose:
                safe_print(f"[TRANSACTIONS] SM2 signature verification result: {is_valid}")
            
            return is_valid
            
        except Exception as e:
            if self.verbose:
                safe_print(f"[TRANSACTIONS] Verification error: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _debug_signature_issue(self, transaction: Dict, sign_data: str, signature: str, public_key: str):
        """Debug why signature verification is failing"""
        safe_print("\n" + "="*60)
        safe_print("DEBUGGING SIGNATURE ISSUE")
        safe_print("="*60)
        
        # 1. Check if we can sign and verify a simple test
        safe_print("\n1. Testing SM2 with simple message...")
        test_message = "Simple test message"
        test_private = self.key_manager.generate_private_key()
        test_public = self.key_manager.derive_public_key(test_private)
        test_sig = self.key_manager.sign_data(test_message, test_private)
        test_valid = self.key_manager.verify_signature(test_message, test_sig, test_public)
        safe_print(f"   Simple test verification: {test_valid}")
        
        # 2. Try to recreate what was signed during transaction creation
        safe_print("\n2. Reconstructing original transaction data...")
        # Create the exact transaction data that should have been signed
        reconstructed = {
            "amount": float(transaction["amount"]),
            "fee": float(transaction["fee"]),
            "from": transaction["from"],
            "memo": transaction.get("memo", ""),
            "timestamp": int(transaction["timestamp"]),
            "to": transaction["to"],
            "type": transaction["type"],
            "version": transaction.get("version", "2.0")
        }
        
        import json
        reconstructed_json = json.dumps(reconstructed, sort_keys=True)
        safe_print(f"   Reconstructed JSON: {reconstructed_json}")
        safe_print(f"   Current signing data: {sign_data}")
        safe_print(f"   Are they equal? {reconstructed_json == sign_data}")
        safe_print(f"   Length difference: {len(reconstructed_json)} vs {len(sign_data)}")
        
        # 3. Check for whitespace differences
        safe_print("\n3. Checking for whitespace differences...")
        safe_print(f"   Reconstructed has spaces: {' ' in reconstructed_json}")
        safe_print(f"   Sign data has spaces: {' ' in sign_data}")
        
        # 4. Check float formatting
        safe_print("\n4. Checking float formatting...")
        safe_print(f"   Amount in tx: {transaction['amount']} (type: {type(transaction['amount'])})")
        safe_print(f"   Amount in reconstructed: {reconstructed['amount']} (type: {type(reconstructed['amount'])})")
        
        # 5. Try different JSON serialization options
        safe_print("\n5. Trying different JSON formats...")
        formats = [
            ("Compact", lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'))),
            ("Default", lambda x: json.dumps(x, sort_keys=True)),
            ("Indented", lambda x: json.dumps(x, sort_keys=True, indent=2)),
        ]
        
        for name, formatter in formats:
            formatted = formatter(reconstructed)
            is_valid_test = self.key_manager.verify_signature(formatted, signature, public_key)
            safe_print(f"   {name} format: {is_valid_test} (length: {len(formatted)})")
        
        safe_print("="*60 + "\n")
    def assess_transaction_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        return self.security.assess_risk(transaction)
    
    # MEMPOOL MANAGEMENT
    def get_pending_transactions(self, address: str = None) -> List[Dict]:
        """Get pending transactions from mempool"""
        return self.mempool_manager.get_pending_transactions(address)
    
    def is_transaction_pending(self, tx_hash: str) -> bool:
        """Check if transaction is pending in mempool"""
        return self.mempool_manager.is_transaction_pending(tx_hash)
    
    def _get_signing_data(self, transaction: Dict) -> str:
        """Create data string for signing - MUST match signing process exactly"""
        # IMPORTANT: During signing, the transaction does NOT have 'public_key' yet!
        # It's added AFTER signing. So we must exclude it during verification too.
        
        # Fields that should NOT be included in signing data:
        exclude_fields = ['signature', 'hash', 'public_key']
        
        # Create copy without excluded fields
        tx_copy = {k: v for k, v in transaction.items() 
                if k not in exclude_fields}
        
        # Use the EXACT same format as during creation
        import json
        
        # Ensure consistent data types
        for key in ['amount', 'fee']:
            if key in tx_copy:
                tx_copy[key] = float(tx_copy[key])
        
        if 'timestamp' in tx_copy:
            tx_copy['timestamp'] = int(tx_copy['timestamp'])
        
        # Use COMPACT JSON (no extra spaces) - this is critical!
        return json.dumps(tx_copy, sort_keys=True, separators=(',', ':'))   
    def _calculate_transaction_hash(self, transaction: Dict) -> str:
        """Calculate transaction hash"""
        # Remove existing hash if present
        tx_copy = transaction.copy()
        tx_copy.pop("hash", None)
        
        # Convert to JSON and hash
        data_string = json.dumps(tx_copy, sort_keys=True)
        return sm3_hex(data_string.encode())
    
    def _generate_reward_hash(self, to_address: str, amount: float, 
                            block_height: int) -> str:
        """Generate unique hash for reward transaction"""
        data = f"reward_{to_address}_{amount}_{block_height}_{time.time_ns()}"
        return sm3_hex(data.encode())
    
    def stop(self):
        """Stop the transaction manager"""
        self.mempool_manager.stop()

# Create a global instance for convenience
_transaction_manager = None

def get_transaction_manager() -> TransactionManager:
    """Get or create global transaction manager instance"""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager

# Convenience functions
def create_transfer(from_address: str, to_address: str, amount: float,
                   private_key: str, memo: str = "") -> Dict:
    """Create transfer transaction (convenience function)"""
    mgr = get_transaction_manager()
    return mgr.create_transfer(from_address, to_address, amount, private_key, memo)

def send_transaction(transaction: Dict) -> Tuple[bool, str]:
    """Send transaction (convenience function)"""
    mgr = get_transaction_manager()
    return mgr.send_transaction(transaction)