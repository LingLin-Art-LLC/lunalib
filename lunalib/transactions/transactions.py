# lunalib/transactions/transactions.py
import time
import hashlib
import json
from typing import Dict, Optional, Tuple, List
from ..core.mempool import MempoolManager

class TransactionSecurity:
    """Transaction security validation and risk assessment"""
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction structure"""
        required_fields = ['type', 'from', 'to', 'amount', 'timestamp']
        for field in required_fields:
            if field not in transaction:
                return False, f'Missing required field: {field}'
        
        # Validate amount
        if transaction['amount'] <= 0:
            return False, 'Amount must be positive'
            
        return True, 'Valid'
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Enhanced security validation"""
        required_fields = ['type', 'from', 'to', 'amount', 'timestamp', 'hash']
        for field in required_fields:
            if field not in transaction:
                return False, f'Missing required field: {field}'
        
        # Validate amount
        if transaction['amount'] <= 0:
            return False, 'Invalid amount'
        
        # Validate transaction type
        valid_types = ['transfer', 'stake', 'unstake', 'delegate', 'reward', 
                      'fee_distribution', 'gtx_genesis', 'validator_join', 
                      'validator_leave', 'governance_vote']
        if transaction['type'] not in valid_types:
            return False, f'Invalid transaction type: {transaction["type"]}'
            
        return True, 'Secure'
    
    def assess_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        amount = transaction.get('amount', 0)
        tx_type = transaction.get('type', 'transfer')
        
        if tx_type in ['gtx_genesis', 'reward', 'fee_distribution']:
            return 'very_low', 'System transaction'
        
        if amount > 1000000:
            return 'high', 'Very large transaction amount'
        elif amount > 100000:
            return 'high', 'Large transaction amount'
        elif amount > 10000:
            return 'medium', 'Medium transaction amount'
        elif amount > 1000:
            return 'low', 'Small transaction amount'
        else:
            return 'very_low', 'Normal transaction'

class KeyManager:
    """Key management for transaction signing"""
    
    def derive_public_key(self, private_key: str) -> str:
        """Derive public key from private key"""
        if not private_key:
            return "unsigned_public_key"
        return f"pub_{hashlib.sha256(private_key.encode()).hexdigest()[:32]}"
    
    def sign_data(self, data: str, private_key: str) -> str:
        """Sign data with private key"""
        if not private_key:
            return "unsigned_signature"
        return hashlib.sha256(f"{data}{private_key}".encode()).hexdigest()
    
    def verify_signature(self, data: str, signature: str, public_key: str) -> bool:
        """Verify signature (simplified - in production use proper ECDSA)"""
        if signature == "unsigned_signature" or public_key == "unsigned_public_key":
            return True  # Allow unsigned system transactions
        
        # Simplified verification - in real implementation, use proper cryptographic verification
        expected_signature = hashlib.sha256(f"{data}{public_key}".encode()).hexdigest()
        return signature == expected_signature

class FeePoolManager:
    """Decentralized fee pool management"""
    
    def __init__(self):
        self.fee_pool_address = self._generate_fee_pool_address()
        self.pending_fees = 0.0
        self.distribution_blocks = 100  # Distribute fees every 100 blocks
        self.last_distribution_block = 0
        self.collected_fees = []  # Track fee collection history
        
    def _generate_fee_pool_address(self) -> str:
        """Generate deterministic fee pool address"""
        base_data = "LUNA_FEE_POOL_V2"
        return hashlib.sha256(base_data.encode()).hexdigest()[:32]
    
    def collect_fee(self, fee_amount: float, transaction_hash: str) -> bool:
        """Collect fee into the pool"""
        if fee_amount > 0:
            self.pending_fees += fee_amount
            self.collected_fees.append({
                'amount': fee_amount,
                'tx_hash': transaction_hash,
                'timestamp': time.time()
            })
            print(f"DEBUG: Collected fee {fee_amount} from transaction {transaction_hash}")
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
                    'stake_share': share,
                    'stake_amount': stake_amount
                })
        
        return rewards
    
    def create_distribution_transactions(self, current_block_height: int, 
                                      stakers: List[Dict], total_stake: float) -> List[Dict]:
        """Create fee distribution transactions to stakers"""
        if not self.should_distribute(current_block_height) or self.pending_fees <= 0:
            return []
        
        rewards = self.calculate_rewards(stakers, total_stake)
        distribution_txs = []
        
        total_distributed = 0
        for reward_info in rewards:
            if reward_info['reward'] > 0:
                distribution_tx = {
                    "type": "fee_distribution",
                    "from": self.fee_pool_address,
                    "to": reward_info['address'],
                    "amount": reward_info['reward'],
                    "fee": 0.0,
                    "block_height": current_block_height,
                    "distribution_cycle": current_block_height // self.distribution_blocks,
                    "stake_share": reward_info['stake_share'],
                    "stake_amount": reward_info['stake_amount'],
                    "timestamp": time.time(),
                    "hash": self._generate_distribution_hash(reward_info['address'], reward_info['reward'], current_block_height)
                }
                distribution_txs.append(distribution_tx)
                total_distributed += reward_info['reward']
        
        # Reset pending fees after distribution
        self.pending_fees = 0.0
        self.last_distribution_block = current_block_height
        
        print(f"DEBUG: Distributed {total_distributed} fees to {len(distribution_txs)} stakers")
        return distribution_txs
    
    def _generate_distribution_hash(self, address: str, amount: float, block_height: int) -> str:
        """Generate unique hash for distribution transaction"""
        data = f"fee_dist_{address}_{amount}_{block_height}_{time.time_ns()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_fee_statistics(self) -> Dict:
        """Get fee pool statistics"""
        total_collected = sum(fee['amount'] for fee in self.collected_fees)
        return {
            'pending_fees': self.pending_fees,
            'total_collected': total_collected,
            'distribution_count': len(self.collected_fees),
            'last_distribution_block': self.last_distribution_block,
            'pool_address': self.fee_pool_address
        }

class FeeCalculator:
    """Configurable fee calculation system"""
    
    def __init__(self, fee_pool_manager: FeePoolManager):
        self.fee_pool_manager = fee_pool_manager
        self.fee_config = {
            'transfer': 0.00001,          # Default transfer fee
            'stake': 0.0001,              # Staking fee
            'unstake': 0.0001,            # Unstaking fee
            'delegate': 0.00005,          # Delegation fee
            'validator_join': 0.001,      # Validator registration fee
            'validator_leave': 0.0001,    # Validator exit fee
            'governance_vote': 0.00001,   # Governance voting fee
            'gtx_genesis': 0.0,           # No fee for GTX genesis
            'reward': 0.0,                # No fee for rewards
            'fee_distribution': 0.0,      # No fee for fee distributions
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
    """Handles transaction creation, signing, validation, and broadcasting"""
    
    def __init__(self, network_endpoints: List[str] = None):
        self.security = TransactionSecurity()
        self.key_manager = KeyManager()
        self.fee_pool_manager = FeePoolManager()
        self.fee_calculator = FeeCalculator(self.fee_pool_manager)
        self.mempool_manager = MempoolManager(network_endpoints)
    
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
            "version": "1.0"
        }
        
        # Only add cryptographic fields if private key is provided
        if private_key:
            transaction["public_key"] = self.key_manager.derive_public_key(private_key)
            sign_data = self._get_signing_data(transaction)
            signature = self.key_manager.sign_data(sign_data, private_key)
            transaction["signature"] = signature
        else:
            # For unsigned transactions (like rewards or system transactions)
            transaction["public_key"] = "unsigned"
            transaction["signature"] = "unsigned"
        
        transaction["hash"] = self._calculate_transaction_hash(transaction)
        
        # Automatically collect fee
        self.fee_calculator.process_transaction_fee(transaction)
        
        return transaction
    
    def send_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Send transaction to mempool for broadcasting"""
        try:
            # Validate transaction first
            is_valid, message = self.validate_transaction(transaction)
            if not is_valid:
                return False, f"Validation failed: {message}"
            
            # Add to mempool
            success = self.mempool_manager.add_transaction(transaction)
            if success:
                return True, f"Transaction added to mempool: {transaction.get('hash')}"
            else:
                return False, "Failed to add transaction to mempool"
                
        except Exception as e:
            return False, f"Error sending transaction: {str(e)}"
    
    # TRANSFER TRANSACTIONS
    def create_transfer(self, from_address: str, to_address: str, amount: float,
                       private_key: str, memo: str = "", fee_override: Optional[float] = None) -> Dict:
        """Create a transfer transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            private_key=private_key,
            memo=memo,
            transaction_type="transfer",
            fee_override=fee_override
        )
    
    # STAKING TRANSACTIONS
    def create_stake(self, from_address: str, amount: float, private_key: str) -> Dict:
        """Create staking transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address="staking_pool",
            amount=amount,
            private_key=private_key,
            transaction_type="stake"
        )
    
    def create_unstake(self, from_address: str, amount: float, private_key: str) -> Dict:
        """Create unstaking transaction"""
        return self.create_transaction(
            from_address="staking_pool",
            to_address=from_address,
            amount=amount,
            private_key=private_key,
            transaction_type="unstake"
        )
    
    def create_delegate(self, from_address: str, to_validator: str, amount: float, 
                       private_key: str) -> Dict:
        """Create delegation transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address=to_validator,
            amount=amount,
            private_key=private_key,
            transaction_type="delegate"
        )
    
    # VALIDATOR TRANSACTIONS
    def create_validator_join(self, from_address: str, validator_info: Dict, 
                             private_key: str) -> Dict:
        """Create validator registration transaction"""
        transaction = self.create_transaction(
            from_address=from_address,
            to_address="validator_registry",
            amount=validator_info.get('stake', 0),
            private_key=private_key,
            transaction_type="validator_join"
        )
        # Add validator-specific info
        transaction.update({
            "validator_name": validator_info.get('name', ''),
            "validator_url": validator_info.get('url', ''),
            "commission_rate": validator_info.get('commission_rate', 0.0)
        })
        return transaction
    
    def create_validator_leave(self, from_address: str, private_key: str) -> Dict:
        """Create validator exit transaction"""
        return self.create_transaction(
            from_address=from_address,
            to_address="validator_exit",
            amount=0,  # No amount for exit
            private_key=private_key,
            transaction_type="validator_leave"
        )
    
    # GOVERNANCE TRANSACTIONS
    def create_governance_vote(self, from_address: str, proposal_id: str, 
                              vote: str, private_key: str) -> Dict:
        """Create governance vote transaction"""
        transaction = self.create_transaction(
            from_address=from_address,
            to_address="governance",
            amount=0,  # No amount for voting
            private_key=private_key,
            transaction_type="governance_vote"
        )
        # Add governance-specific info
        transaction.update({
            "proposal_id": proposal_id,
            "vote": vote,  # 'yes', 'no', 'abstain'
            "voting_power": 1.0  # Could be based on stake
        })
        return transaction
    
    # SYSTEM TRANSACTIONS
    def create_gtx_transaction(self, bill_info: Dict) -> Dict:
        """Create GTX Genesis transaction from mined bill"""
        return {
            "type": "gtx_genesis",
            "from": "mining",
            "to": bill_info.get("owner_address", "unknown"),
            "amount": bill_info.get("denomination", 0),
            "fee": 0.0,
            "timestamp": time.time(),
            "bill_serial": bill_info.get("serial", ""),
            "mining_difficulty": bill_info.get("difficulty", 0),
            "hash": f"gtx_{hashlib.sha256(json.dumps(bill_info).encode()).hexdigest()[:16]}",
            "public_key": "system",
            "signature": "system"
        }
    
    def create_reward_transaction(self, to_address: str, amount: float, 
                                block_height: int, reward_type: str = "block") -> Dict:
        """Create reward transaction"""
        transaction = {
            "type": "reward",
            "from": "network",
            "to": to_address,
            "amount": float(amount),
            "fee": 0.0,
            "block_height": block_height,
            "reward_type": reward_type,  # block, fee, staking
            "timestamp": time.time(),
            "hash": self._generate_reward_hash(to_address, amount, block_height, reward_type),
            "public_key": "system",
            "signature": "system"
        }
        return transaction
    
    def create_fee_distribution(self, to_address: str, amount: float, 
                              distribution_cycle: int, stake_share: float) -> Dict:
        """Create fee distribution transaction"""
        return {
            "type": "fee_distribution",
            "from": self.fee_pool_manager.fee_pool_address,
            "to": to_address,
            "amount": float(amount),
            "fee": 0.0,
            "distribution_cycle": distribution_cycle,
            "stake_share": stake_share,
            "timestamp": time.time(),
            "hash": self.fee_pool_manager._generate_distribution_hash(to_address, amount, distribution_cycle),
            "public_key": "system",
            "signature": "system"
        }
    
    # FEE DISTRIBUTION
    def distribute_fees(self, current_block_height: int, stakers: List[Dict], 
                       total_stake: float) -> List[Dict]:
        """Distribute collected fees to stakers"""
        return self.fee_pool_manager.create_distribution_transactions(
            current_block_height, stakers, total_stake
        )
    
    # VALIDATION METHODS
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction using security module"""
        return self.security.validate_transaction(transaction)
    
    def validate_transaction_security(self, transaction: Dict) -> Tuple[bool, str]:
        """Validate transaction security"""
        return self.security.validate_transaction_security(transaction)
    
    def assess_transaction_risk(self, transaction: Dict) -> Tuple[str, str]:
        """Assess transaction risk level"""
        return self.security.assess_risk(transaction)
    
    def verify_transaction_signature(self, transaction: Dict) -> bool:
        """Verify transaction signature"""
        try:
            sign_data = self._get_signing_data(transaction)
            signature = transaction.get("signature", "")
            public_key = transaction.get("public_key", "")
            return self.key_manager.verify_signature(sign_data, signature, public_key)
        except:
            return False
    
    # MEMPOOL MANAGEMENT
    def get_pending_transactions(self, address: str = None) -> List[Dict]:
        """Get pending transactions from mempool"""
        return self.mempool_manager.get_pending_transactions(address)
    
    def is_transaction_pending(self, tx_hash: str) -> bool:
        """Check if transaction is pending in mempool"""
        return self.mempool_manager.is_transaction_pending(tx_hash)
    
    def is_transaction_confirmed(self, tx_hash: str) -> bool:
        """Check if transaction has been confirmed"""
        return self.mempool_manager.is_transaction_confirmed(tx_hash)
    
    # FEE MANAGEMENT
    def set_transaction_fee(self, transaction_type: str, fee_amount: float):
        """Set custom fee for transaction type"""
        self.fee_calculator.set_fee(transaction_type, fee_amount)
    
    def calculate_network_fee(self, transaction_size: int, priority: str = 'normal') -> float:
        """Calculate network fee based on size and priority"""
        return self.fee_calculator.calculate_network_fee(transaction_size, priority)
    
    def get_fee_pool_balance(self) -> float:
        """Get current fee pool balance"""
        return self.fee_pool_manager.pending_fees
    
    def get_fee_pool_address(self) -> str:
        """Get fee pool address"""
        return self.fee_pool_manager.fee_pool_address
    
    def get_fee_statistics(self) -> Dict:
        """Get fee pool statistics"""
        return self.fee_pool_manager.get_fee_statistics()
    
    # UTILITY METHODS
    def _get_signing_data(self, transaction: Dict) -> str:
        """Create data string for signing"""
        parts = [
            transaction["type"],
            transaction["from"],
            transaction["to"],
            str(transaction["amount"]),
            str(transaction.get("nonce", 0)),
            str(transaction["timestamp"]),
            transaction.get("memo", ""),
            str(transaction.get("fee", 0)),
            transaction.get("version", "1.0")
        ]
        return "|".join(parts)
    
    def _calculate_transaction_hash(self, transaction: Dict) -> str:
        """Calculate transaction hash"""
        # Create a copy without signature for consistent hashing
        tx_copy = transaction.copy()
        tx_copy.pop("signature", None)
        tx_copy.pop("hash", None)
        data_string = json.dumps(tx_copy, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _generate_reward_hash(self, to_address: str, amount: float, 
                            block_height: int, reward_type: str) -> str:
        """Generate unique hash for reward transaction"""
        data = f"reward_{reward_type}_{to_address}_{amount}_{block_height}_{time.time_ns()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def stop(self):
        """Stop the transaction manager"""
        self.mempool_manager.stop()

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