import time
import hashlib
import json
from typing import Dict, Optional, Tuple, List

class TransactionSecurity:
    """Transaction security validation and risk assessment"""
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction structure"""
        required_fields = ['from', 'to', 'amount', 'timestamp']
        for field in required_fields:
            if field not in transaction:
                return False, f'Missing required field: {field}'
        
        # Validate amount
        if transaction['amount'] <= 0:
            return False, 'Amount must be positive'
            
        return True, 'Valid'
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Enhanced security validation"""
        required_fields = ['from', 'to', 'amount', 'timestamp', 'type']
        for field in required_fields:
            if field not in transaction:
                return False, f'Missing required field: {field}'
        
        # Validate amount
        if transaction['amount'] <= 0:
            return False, 'Invalid amount'
            
        return True, 'Secure'
    
    def assess_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        amount = transaction.get('amount', 0)
        
        if amount > 100000:
            return 'high', 'Large transaction amount'
        elif amount > 10000:
            return 'medium', 'Medium transaction amount'
        else:
            return 'low', 'Normal transaction'

class KeyManager:
    """Key management for transaction signing"""
    
    def derive_public_key(self, private_key: str) -> str:
        """Derive public key from private key"""
        if not private_key:
            return "default_public_key"
        return f"pub_{private_key[-16:]}"
    
    def sign_data(self, data: str, private_key: str) -> str:
        """Sign data with private key"""
        if not private_key:
            return "default_signature"
        return hashlib.sha256(f"{data}{private_key}".encode()).hexdigest()

class FeePoolManager:
    """Decentralized fee pool management"""
    
    def __init__(self):
        self.fee_pool_address = self._generate_fee_pool_address()
        self.pending_fees = 0.0
        self.distribution_blocks = 100  # Distribute fees every 100 blocks
        self.last_distribution_block = 0
        
    def _generate_fee_pool_address(self) -> str:
        """Generate deterministic fee pool address"""
        base_data = "LUNA_FEE_POOL_V1"
        return hashlib.sha256(base_data.encode()).hexdigest()[:32]
    
    def collect_fee(self, fee_amount: float, transaction_hash: str) -> bool:
        """Collect fee into the pool"""
        if fee_amount > 0:
            self.pending_fees += fee_amount
            return True
        return False
    
    def should_distribute(self, current_block_height: int) -> bool:
        """Check if it's time to distribute fees"""
        return (current_block_height - self.last_distribution_block) >= self.distribution_blocks
    
    def calculate_rewards(self, stakers: List[Dict], total_stake: float) -> List[Dict]:
        """Calculate rewards for stakers based on their stake"""
        if total_stake <= 0 or self.pending_fees <= 0:
            return []
            
        rewards = []
        for staker in stakers:
            stake_amount = staker.get('stake', 0)
            if stake_amount > 0:
                share = stake_amount / total_stake
                reward = self.pending_fees * share
                rewards.append({
                    'address': staker['address'],
                    'reward': reward,
                    'stake_share': share
                })
        
        return rewards
    
    def create_distribution_transactions(self, current_block_height: int, 
                                      stakers: List[Dict], total_stake: float) -> List[Dict]:
        """Create fee distribution transactions to stakers"""
        if not self.should_distribute(current_block_height) or self.pending_fees <= 0:
            return []
        
        rewards = self.calculate_rewards(stakers, total_stake)
        distribution_txs = []
        
        for reward_info in rewards:
            distribution_tx = {
                "type": "fee_distribution",
                "from": self.fee_pool_address,
                "to": reward_info['address'],
                "amount": reward_info['reward'],
                "fee": 0.0,
                "block_height": current_block_height,
                "distribution_cycle": current_block_height // self.distribution_blocks,
                "stake_share": reward_info['stake_share'],
                "timestamp": time.time(),
                "hash": self._generate_distribution_hash(reward_info['address'], reward_info['reward'], current_block_height)
            }
            distribution_txs.append(distribution_tx)
        
        # Reset pending fees after distribution
        self.pending_fees = 0.0
        self.last_distribution_block = current_block_height
        
        return distribution_txs
    
    def _generate_distribution_hash(self, address: str, amount: float, block_height: int) -> str:
        """Generate unique hash for distribution transaction"""
        data = f"fee_dist_{address}_{amount}_{block_height}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()

class FeeCalculator:
    """Configurable fee calculation system"""
    
    def __init__(self, fee_pool_manager: FeePoolManager):
        self.fee_pool_manager = fee_pool_manager
        self.fee_config = {
            'transfer': 0.00001,      # Default transfer fee
            'gtx_genesis': 0.0,       # No fee for GTX genesis
            'reward': 0.0,            # No fee for rewards
            'stake': 0.0001,          # Staking fee
            'unstake': 0.0001,        # Unstaking fee
            'delegate': 0.00005,      # Delegation fee
            'fee_distribution': 0.0,  # No fee for fee distributions
        }
        
        # Dynamic fee tiers based on transaction amount
        self.amount_tiers = {
            'micro': (0, 1, 0.000001),
            'small': (1, 100, 0.00001),
            'medium': (100, 10000, 0.0001),
            'large': (10000, 100000, 0.001),
            'xlarge': (100000, float('inf'), 0.01)
        }
    
    def set_fee(self, transaction_type: str, fee_amount: float):
        """Set custom fee for a transaction type"""
        self.fee_config[transaction_type] = max(0.0, fee_amount)
    
    def get_fee(self, transaction_type: str, amount: float = 0.0) -> float:
        """Get fee for transaction type and amount"""
        base_fee = self.fee_config.get(transaction_type, 0.00001)
        
        # Apply amount-based fee scaling
        for tier_name, (min_amt, max_amt, tier_fee) in self.amount_tiers.items():
            if min_amt <= amount < max_amt:
                return max(base_fee, tier_fee)
        
        return base_fee
    
    def calculate_network_fee(self, transaction_size: int, priority: str = 'normal') -> float:
        """Calculate fee based on transaction size and priority"""
        base_fee_per_byte = 0.0000001  # Base fee per byte
        
        priority_multipliers = {
            'low': 0.5,
            'normal': 1.0,
            'high': 2.0,
            'urgent': 5.0
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        return transaction_size * base_fee_per_byte * multiplier
    
    def process_transaction_fee(self, transaction: Dict) -> bool:
        """Process and collect transaction fee"""
        fee = transaction.get('fee', 0)
        if fee > 0:
            return self.fee_pool_manager.collect_fee(fee, transaction.get('hash', ''))
        return True

class TransactionManager:
    """Handles transaction creation, signing, and validation"""
    
    def __init__(self):
        self.security = TransactionSecurity()
        self.key_manager = KeyManager()
        self.fee_pool_manager = FeePoolManager()
        self.fee_calculator = FeeCalculator(self.fee_pool_manager)
    
    def create_transaction(self, from_address: str, to_address: str, amount: float, 
                         private_key: Optional[str] = None, memo: str = "",
                         transaction_type: str = "transfer", fee_override: Optional[float] = None,
                         priority: str = 'normal') -> Dict:
        """Create and sign a transaction with configurable fees"""
        
        # Calculate fee
        if fee_override is not None:
            fee = max(0.0, fee_override)
        else:
            fee = self.fee_calculator.get_fee(transaction_type, amount)
        
        transaction = {
            "type": transaction_type,
            "from": from_address,
            "to": to_address,
            "amount": float(amount),
            "fee": fee,
            "nonce": int(time.time() * 1000),
            "timestamp": time.time(),
            "memo": memo,
            "priority": priority,
        }
        
        # Only add cryptographic fields if private key is provided
        if private_key:
            transaction["public_key"] = self.key_manager.derive_public_key(private_key)
            sign_data = self._get_signing_data(transaction)
            signature = self.key_manager.sign_data(sign_data, private_key)
            transaction["signature"] = signature
        else:
            # For unsigned transactions (like rewards or test transactions)
            transaction["public_key"] = "unsigned"
            transaction["signature"] = "unsigned"
        
        transaction["hash"] = self._calculate_transaction_hash(transaction)
        
        # Automatically collect fee
        self.fee_calculator.process_transaction_fee(transaction)
        
        return transaction
    
    def distribute_fees(self, current_block_height: int, stakers: List[Dict], total_stake: float) -> List[Dict]:
        """Distribute collected fees to stakers"""
        return self.fee_pool_manager.create_distribution_transactions(
            current_block_height, stakers, total_stake
        )
    
    def get_fee_pool_balance(self) -> float:
        """Get current fee pool balance"""
        return self.fee_pool_manager.pending_fees
    
    def get_fee_pool_address(self) -> str:
        """Get fee pool address"""
        return self.fee_pool_manager.fee_pool_address
    
    def create_gtx_transaction(self, bill_info: Dict) -> Dict:
        """Create GTX Genesis transaction from mined bill"""
        # Ensure we return a proper transaction structure
        if "transaction_data" in bill_info:
            return bill_info["transaction_data"]
        else:
            # Fallback: create basic transaction from bill info
            return {
                "type": "gtx_genesis",
                "from": "mining",
                "to": bill_info.get("owner_address", "unknown"),
                "amount": bill_info.get("denomination", 0),
                "fee": 0.0,  # No fee for GTX genesis
                "timestamp": time.time(),
                "hash": f"gtx_{hashlib.sha256(json.dumps(bill_info).encode()).hexdigest()[:16]}"
            }
    
    def create_reward_transaction(self, to_address: str, amount: float, block_height: int) -> Dict:
        """Create block reward transaction"""
        transaction = {
            "type": "reward",
            "from": "network",
            "to": to_address,
            "amount": float(amount),
            "fee": 0.0,  # No fee for rewards
            "block_height": block_height,
            "timestamp": time.time(),
            "hash": self._generate_reward_hash(to_address, amount, block_height)
        }
        
        return transaction
    
    def create_stake_transaction(self, from_address: str, amount: float, 
                               private_key: Optional[str] = None) -> Dict:
        """Create staking transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address="staking_pool",
            amount=amount,
            private_key=private_key,
            transaction_type="stake"
        )
    
    def create_unstake_transaction(self, from_address: str, amount: float,
                                 private_key: Optional[str] = None) -> Dict:
        """Create unstaking transaction"""
        return self.create_transaction(
            from_address="staking_pool",
            to_address=from_address,
            amount=amount,
            private_key=private_key,
            transaction_type="unstake"
        )
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction using security module"""
        return self.security.validate_transaction(transaction)
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction security"""
        return self.security.validate_transaction_security(transaction)
    
    def assess_transaction_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        return self.security.assess_risk(transaction)
    
    def set_transaction_fee(self, transaction_type: str, fee_amount: float):
        """Set custom fee for transaction type"""
        self.fee_calculator.set_fee(transaction_type, fee_amount)
    
    def calculate_network_fee(self, transaction_size: int, priority: str = 'normal') -> float:
        """Calculate network fee based on size and priority"""
        return self.fee_calculator.calculate_network_fee(transaction_size, priority)
    
    def _get_signing_data(self, transaction: Dict) -> str:
        """Create data string for signing"""
        parts = [
            transaction["from"],
            transaction["to"],
            str(transaction["amount"]),
            str(transaction.get("nonce", 0)),
            str(transaction["timestamp"]),
            transaction.get("memo", ""),
            str(transaction.get("fee", 0)),
            transaction.get("type", "transfer")
        ]
        return "".join(parts)
    
    def _calculate_transaction_hash(self, transaction: Dict) -> str:
        """Calculate transaction hash"""
        # Create a copy without signature for consistent hashing
        tx_copy = transaction.copy()
        tx_copy.pop("signature", None)
        data_string = json.dumps(tx_copy, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()  # FIXED: data_string instead of data
    
    def _generate_reward_hash(self, to_address: str, amount: float, block_height: int) -> str:
        """Generate unique hash for reward transaction"""
        data = f"reward_{to_address}_{amount}_{block_height}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()

# Additional validator class for backward compatibility
class TransactionValidator:
    """Transaction validator for risk assessment and validation"""
    
    def __init__(self):
        self.security = TransactionSecurity()
    
    def assess_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        return self.security.assess_risk(transaction)
    
    def validate(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction"""
        return self.security.validate_transaction(transaction)