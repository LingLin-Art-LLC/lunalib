import time
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List
from .security import TransactionSecurity

class TransactionValidator:
    """Network-level transaction validation"""
    
    def __init__(self):
        self.security = TransactionSecurity()
        self.recent_transactions = set()
        self.max_recent_size = 10000
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Main transaction validation entry point"""
        
        # Check for duplicate transaction
        tx_hash = transaction.get("hash")
        if tx_hash in self.recent_transactions:
            return False, "Duplicate transaction detected"
        
        # Security validation
        is_valid, message = self.security.validate_transaction_security(transaction)
        if not is_valid:
            return False, message
        
        # Add to recent transactions
        self._add_to_recent(tx_hash)
        
        return True, message
    
    def validate_transaction_batch(self, transactions: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate multiple transactions with tqdm progress bar"""
        from tqdm import tqdm
        from lunalib.utils.console import print_info, print_error
        all_valid = True
        results: list[Tuple[bool, str]] = [None] * len(transactions)
        pending: list[Tuple[int, Dict]] = []

        for idx, tx in enumerate(transactions):
            tx_hash = tx.get("hash")
            if tx_hash in self.recent_transactions:
                results[idx] = (False, "Duplicate transaction detected")
            else:
                pending.append((idx, tx))

        if pending:
            def _validate(item: Tuple[int, Dict]) -> Tuple[bool, str]:
                _, tx = item
                return self.security.validate_transaction_security(tx)

            if sys.platform != "emscripten":
                max_workers = min(8, len(pending))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    validated = list(pool.map(_validate, pending))
            else:
                validated = [_validate(item) for item in pending]

            for (idx, tx), (ok, msg) in zip(pending, validated):
                results[idx] = (ok, msg)
                if ok:
                    self._add_to_recent(tx.get("hash"))

        output_messages: list[str] = []
        for ok, msg in tqdm(results, desc="Validating transactions", ncols=80):
            if ok:
                print_info(msg)
            else:
                print_error(msg)
                all_valid = False
            output_messages.append(msg)

        return all_valid, output_messages
    
    def verify_transaction_inclusion(self, transaction_hash: str, block_height: int) -> bool:
        """Verify transaction is included in blockchain"""
        # This would typically query the blockchain
        # For now, we'll assume if it passed validation it's included
        return transaction_hash in self.recent_transactions
    
    def get_transaction_risk_level(self, transaction: Dict) -> str:
        """Assess transaction risk level"""
        amount = transaction.get("amount", 0)
        security_score = self.security.calculate_security_score(transaction)
        
        if amount > 1000000 and security_score < 80:
            return "HIGH"
        elif amount > 10000 and security_score < 60:
            return "MEDIUM"
        elif security_score < 40:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _add_to_recent(self, tx_hash: str):
        """Add transaction to recent set with size management"""
        self.recent_transactions.add(tx_hash)
        
        # Manage set size
        if len(self.recent_transactions) > self.max_recent_size:
            # Remove oldest entries (this is simplified)
            self.recent_transactions = set(list(self.recent_transactions)[-self.max_recent_size:])