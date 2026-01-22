# lunalib/mining/miner.py
import time
import sys
import os
# --- Unicode-safe print for Windows console ---
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', 'utf-8')
        print(*(str(a).encode(encoding, errors='replace').decode(encoding) for a in args), **kwargs)
import hashlib
from lunalib.core.sm3 import sm3_hex
import json
import threading
from typing import Dict, Optional, List, Union, Callable
from ..mining.difficulty import DifficultySystem
from ..gtx.digital_bill import DigitalBill
from ..transactions.transactions import TransactionManager
from ..core.blockchain import BlockchainManager
from ..core.mempool import MempoolManager
from ..mining.cuda_manager import CUDAManager

class GenesisMiner:
    """Mines GTX Genesis bills AND regular transfer transactions with configurable difficulty"""
    
    def __init__(self, network_endpoints: List[str] = None):
        self.difficulty_system = DifficultySystem()
        self.transaction_manager = TransactionManager(network_endpoints)
        self.blockchain_manager = BlockchainManager(network_endpoints[0] if network_endpoints else "https://bank.linglin.art")
        self.mempool_manager = MempoolManager(network_endpoints)
        
        self.mining_active = False
        self.current_thread = None
        self.mining_stats = {
            "bills_mined": 0,
            "blocks_mined": 0,
            "total_mining_time": 0,
            "total_hash_attempts": 0
        }
        
        safe_print("🔧 GenesisMiner initialized with integrated lunalib components")
    
    def mine_bill(self, denomination: float, user_address: str, bill_data: Dict = None) -> Dict:
        """Mine a GTX Genesis bill using DigitalBill system"""
        try:
            if denomination not in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]:
                return {"success": False, "error": "Invalid denomination"}

            difficulty = self.difficulty_system.get_bill_difficulty(denomination)
            
            # Create digital bill using GTX system
            digital_bill = DigitalBill(
                denomination=denomination,
                user_address=user_address,
                difficulty=difficulty,
                bill_data=bill_data or {}
            )
            
            safe_print(f"⛏️ Mining GTX ${denomination:,} Bill - Difficulty: {difficulty} zeros")
            
            start_time = time.time()
            mining_result = self._perform_bill_mining(digital_bill, difficulty)
            
            if mining_result["success"]:
                mining_time = time.time() - start_time
                
                # Finalize the bill
                bill = digital_bill.finalize(
                    hash=mining_result["hash"],
                    nonce=mining_result["nonce"],
                    mining_time=mining_time
                )
                
                # Update mining statistics
                self.mining_stats["bills_mined"] += 1
                self.mining_stats["total_mining_time"] += mining_time
                self.mining_stats["total_hash_attempts"] += mining_result["nonce"]
                
                safe_print(f"✅ Successfully mined GTX ${denomination:,} bill!")
                safe_print(f"⏱️ Mining time: {mining_time:.2f}s")
                safe_print(f" Hash attempts: {mining_result['nonce']:,}")
                safe_print(f"🔗 Bill hash: {mining_result['hash'][:32]}...")
                
                # Convert to GTX Genesis transaction
                gtx_transaction = self._create_gtx_genesis_transaction(bill)
                return {
                    "success": True,
                    "type": "bill",
                    "bill": bill,
                    "transaction": gtx_transaction,
                    "mining_time": mining_time,
                    "hash_attempts": mining_result["nonce"]
                }
            else:
                return {"success": False, "error": "Bill mining failed"}
                
        except Exception as e:
            print(f"[X]Error mining bill: {e}")
            return {"success": False, "error": str(e)}
    
    def mine_transaction_block(self, miner_address: str, previous_hash: str = None, block_height: int = None) -> Dict:
        """Mine a block containing transactions from mempool"""
        try:
            # Get current blockchain state if not provided
            if previous_hash is None or block_height is None:
                current_height = self.blockchain_manager.get_blockchain_height()
                latest_block = self.blockchain_manager.get_latest_block()
                block_height = current_height + 1
                previous_hash = latest_block.get('hash', '0' * 64) if latest_block else '0' * 64
            
            # Get transactions from mempool
            pending_txs = self.mempool_manager.get_pending_transactions()
            transactions = pending_txs[:10]  # Limit block size
            
            if not transactions:
                return {"success": False, "error": "No transactions in mempool"}
            
            # Calculate block difficulty
            difficulty = self.difficulty_system.get_transaction_block_difficulty(transactions)
            
            safe_print(f"⛏️ Mining Transaction Block #{block_height} - Difficulty: {difficulty} zeros")
            safe_print(f"📦 Transactions: {len(transactions)} | Previous Hash: {previous_hash[:16]}...")
            
            # Create block structure for mining
            block_data = {
                "index": block_height,
                "previous_hash": previous_hash,
                "timestamp": time.time(),
                "transactions": transactions,
                "miner": miner_address,
                "difficulty": difficulty,
                "nonce": 0,
                "version": "1.0"
            }
            
            start_time = time.time()
            mining_result = self._perform_block_mining(block_data, difficulty)
            
            if mining_result["success"]:
                mining_time = time.time() - start_time
                
                # Add hash and nonce to block_data for validation
                block_data["hash"] = mining_result["hash"]
                block_data["nonce"] = mining_result["nonce"]
                
                # Create reward transaction WITH VALIDATION
                reward_tx = self._create_mining_reward_transaction(
                    miner_address=miner_address,
                    block_height=block_height,
                    transactions=transactions,
                    block_data=block_data  # Pass block data for validation
                )
                
                # Add reward transaction
                block_data["transactions"].append(reward_tx)
                
                # Calculate merkleroot for submission
                merkleroot = self._calculate_merkleroot(transactions)  # Without reward
                
                # Finalize block
                block = {
                    **block_data,
                    "hash": mining_result["hash"],
                    "nonce": mining_result["nonce"],
                    "merkleroot": merkleroot,
                    "transactions_hash": merkleroot,
                    "mining_time": mining_time,
                    "reward": reward_tx["amount"],
                    "transaction_count": len(block_data["transactions"]),
                    "timestamp": block_data["timestamp"]  # Ensure timestamp is included
                }
                
                # Update mining statistics
                self.mining_stats["blocks_mined"] += 1
                self.mining_stats["total_mining_time"] += mining_time
                self.mining_stats["total_hash_attempts"] += mining_result["nonce"]
                
                safe_print(f"✅ Successfully mined and validated Transaction Block #{block_height}!")
                safe_print(f"⏱️ Mining time: {mining_time:.2f}s")
                safe_print(f"💰 Block reward: {block['reward']:.6f} LUN")
                safe_print(f" Transactions: {block['transaction_count']}")
                safe_print(f"🔗 Block hash: {mining_result['hash'][:32]}...")
                
                # Submit block to blockchain
                submission_success = self.blockchain_manager.submit_mined_block(block)
                if submission_success:
                    safe_print("✅ Block successfully submitted to blockchain!")
                    # Clear mined transactions from local mempool
                    self._clear_mined_transactions(transactions)
                else:
                    safe_print("⚠️ Block mined but submission failed")
                
                return {
                    "success": True,
                    "type": "block",
                    "block": block,
                    "submitted": submission_success,
                    "mining_time": mining_time,
                    "hash_attempts": mining_result["nonce"]
                }
            else:
                return {"success": False, "error": "Block mining failed"}
                
        except Exception as e:
            print(f"❌ Error mining block: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _perform_bill_mining(self, digital_bill: DigitalBill, difficulty: int) -> Dict:
        """Perform proof-of-work mining for GTX bills"""
        target = "0" * difficulty
        nonce = 0
        start_time = time.time()
        last_update = start_time
        
        while self.mining_active:
            mining_data = digital_bill.get_mining_data(nonce)
            data_string = json.dumps(mining_data, sort_keys=True)
            bill_hash = sm3_hex(data_string.encode())
            
            if bill_hash.startswith(target):
                mining_time = time.time() - start_time
                return {
                    "success": True,
                    "hash": bill_hash,
                    "nonce": nonce,
                    "mining_time": mining_time
                }
            
            nonce += 1
            
            # Progress updates every 5 seconds
            current_time = time.time()
            if current_time - last_update >= 5:
                hashrate = nonce / (current_time - start_time)
                print(f"⏳ Bill mining: {nonce:,} attempts | Rate: {hashrate:,.0f} H/s")
                last_update = current_time
        
        return {"success": False, "error": "Mining stopped"}
    
    def _perform_block_mining(self, block_data: Dict, difficulty: int) -> Dict:
        """Perform proof-of-work mining for transaction blocks"""
        target = "0" * difficulty
        nonce = 0
        start_time = time.time()
        last_update = start_time
        
        while self.mining_active:
            # Update nonce for this attempt
            block_data["nonce"] = nonce
            
            # Create block hash using canonical server format
            block_hash = self._calculate_block_hash(
                block_data.get("index"),
                block_data.get("previous_hash"),
                block_data.get("timestamp"),
                block_data.get("transactions", []),
                nonce,
                block_data.get("miner"),
                block_data.get("difficulty", difficulty),
            )
            
            if block_hash.startswith(target):
                mining_time = time.time() - start_time
                return {
                    "success": True,
                    "hash": block_hash,
                    "nonce": nonce,
                    "mining_time": mining_time
                }
            
            nonce += 1
            
            # Progress updates every 5 seconds
            current_time = time.time()
            if current_time - last_update >= 5:
                hashrate = nonce / (current_time - start_time)
                print(f"Block mining: {nonce:,} attempts | Rate: {hashrate:,.0f} H/s")
                last_update = current_time
        
        return {"success": False, "error": "Mining stopped"}
    
    def _create_gtx_genesis_transaction(self, bill: Dict) -> Dict:
        """Create GTX Genesis transaction from mined bill"""
        return self.transaction_manager.create_gtx_transaction(bill)
    
    def _create_mining_reward_transaction(self, miner_address: str, block_height: int, 
                                        transactions: List[Dict], block_data: Dict = None) -> Dict:
        """Create mining reward transaction with validation of the mining proof"""
        # Calculate reward
        difficulty = block_data.get('difficulty', 0) if block_data else 0
        non_reward_txs = [tx for tx in transactions if tx.get('type') != 'reward']
        is_empty_block = not non_reward_txs

        reward_mode = self._resolve_reward_mode()
        if is_empty_block:
            total_reward = float(difficulty)
        else:
            if reward_mode == "linear":
                total_reward = float(difficulty)
            else:
                total_reward = self.difficulty_system.calculate_block_reward(difficulty)
        
        # If block_data is provided, validate the mining proof
        if block_data:
            print("🔍 Validating mining proof before creating reward...")
            
            # Extract mining proof components
            block_hash = block_data.get('hash', '')
            difficulty = block_data.get('difficulty', 0)
            nonce = block_data.get('nonce', 0)
            timestamp = block_data.get('timestamp', time.time())
            previous_hash = block_data.get('previous_hash', '0' * 64)
            miner = block_data.get('miner', miner_address)
            
            # Calculate merkleroot from transactions
            merkleroot = self._calculate_merkleroot(transactions)
            
            print(f"  Mining proof components:")
            print(f"  Block hash: {block_hash[:16]}...")
            print(f"  Difficulty: {difficulty}")
            print(f"  Nonce: {nonce}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Previous hash: {previous_hash[:16]}...")
            print(f"  Miner: {miner}")
            print(f"  Merkleroot: {merkleroot[:16]}...")
            
            # Validate difficulty requirement
            if not block_hash.startswith('0' * difficulty):
                print(f"❌ FAIL: Hash doesn't start with {difficulty} zeros")
                raise ValueError(f"Invalid mining proof: Hash doesn't meet difficulty requirement")
            
            # Try multiple validation methods
            validation_passed = False
            
            # Method 1: Original format (what server likely expects)
            original_string = f"{previous_hash}{timestamp}{merkleroot}{miner}{nonce}"
            original_hash = sm3_hex(original_string.encode())
            
            print(f"🔍 Original format validation:")
            print(f"  String: {original_string[:80]}...")
            print(f"  Calculated: {original_hash[:16]}...")
            
            if original_hash == block_hash:
                validation_passed = True
                print("✅ Original format validation passed")
            
            # Method 2: JSON format (what miner might be using)
            if not validation_passed:
                mining_json = {
                    "index": block_height,
                    "previous_hash": previous_hash,
                    "timestamp": timestamp,
                    "transactions": transactions,
                    "miner": miner,
                    "difficulty": difficulty,
                    "nonce": nonce,
                    "version": "1.0"
                }
                
                json_string = json.dumps(mining_json, sort_keys=True)
                json_hash = sm3_hex(json_string.encode())
                
                print(f"🔍 JSON format validation:")
                print(f"  String: {json_string[:100]}...")
                print(f"  Calculated: {json_hash[:16]}...")
                
                if json_hash == block_hash:
                    validation_passed = True
                    print("✅ JSON format validation passed")
            
            # Method 3: JSON without transactions (for empty blocks)
            if not validation_passed and len(transactions) == 0:
                mining_json_empty = {
                    "index": block_height,
                    "previous_hash": previous_hash,
                    "timestamp": timestamp,
                    "transactions": [],
                    "miner": miner,
                    "difficulty": difficulty,
                    "nonce": nonce,
                    "version": "1.0"
                }
                
                json_string_empty = json.dumps(mining_json_empty, sort_keys=True)
                json_hash_empty = sm3_hex(json_string_empty.encode())
                
                print(f"🔍 JSON empty format validation:")
                print(f"  Calculated: {json_hash_empty[:16]}...")
                
                if json_hash_empty == block_hash:
                    validation_passed = True
                    print("✅ JSON empty format validation passed")
            
            if not validation_passed:
                print("❌ All validation methods failed")
                raise ValueError("Invalid mining proof: Hash verification failed")
            
            print("✅ Mining proof validation successful!")
        
        # Create the reward transaction
        return self.transaction_manager.create_reward_transaction(
            to_address=miner_address,
            amount=total_reward,
            block_height=block_height,
            reward_type="block"
        )

    def _resolve_reward_mode(self) -> str:
        """Resolve reward mode, preferring daemon configuration when available."""
        env_mode = os.getenv("LUNALIB_BLOCK_REWARD_MODE")
        if env_mode:
            return str(env_mode).lower().strip()

        stats = {}
        try:
            stats = self.blockchain_manager.get_server_stats()
        except Exception:
            stats = {}

        mode = (
            stats.get("block_reward_mode")
            or stats.get("reward_mode")
            or stats.get("blockRewardMode")
        )
        if mode:
            return str(mode).lower().strip()

        return "exponential"

    def _calculate_merkleroot(self, transactions: List[Dict]) -> str:
        """Calculate merkle root from transactions"""
        if not transactions:
            return "0" * 64
        
        tx_hashes = []
        for tx in transactions:
            if 'hash' in tx:
                tx_hashes.append(tx['hash'])
            else:
                tx_string = json.dumps(tx, sort_keys=True)
                tx_hashes.append(sm3_hex(tx_string.encode()))
        
        # Simple merkle root calculation
        while len(tx_hashes) > 1:
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]
                new_hash = sm3_hex(combined.encode())
                new_hashes.append(new_hash)
            tx_hashes = new_hashes
        
        return tx_hashes[0] if tx_hashes else "0" * 64
    
    def _clear_mined_transactions(self, mined_transactions: List[Dict]):
        """Remove mined transactions from local mempool"""
        for tx in mined_transactions:
            tx_hash = tx.get('hash')
            if tx_hash:
                self.mempool_manager.remove_transaction(tx_hash)
        
        print(f"Cleared {len(mined_transactions)} mined transactions from mempool")
    
    def start_auto_bill_mining(self, denominations: List[float], user_address: str, 
                             callback: Callable = None) -> bool:
        """Start auto-mining multiple GTX bills"""
        if self.mining_active:
            print("Mining already active")
            return False
            
        self.mining_active = True
        
        def auto_mine():
            results = []
            for denomination in denominations:
                if not self.mining_active:
                    break
                    
                print(f"Starting auto-mining for ${denomination:,} bill...")
                result = self.mine_bill(denomination, user_address)
                results.append(result)
                
                if callback:
                    callback(result)
                
                # Brief pause between bills
                time.sleep(1)
            
            print("Auto bill mining completed")
            return results
        
        self.current_thread = threading.Thread(target=auto_mine, daemon=True)
        self.current_thread.start()
        print(f"Started auto-mining {len(denominations)} bills")
        return True
    
    def start_continuous_block_mining(self, miner_address: str, callback: Callable = None) -> bool:
        """Start continuous transaction block mining"""
        if self.mining_active:
            print("⚠️ Mining already active")
            return False
            
        self.mining_active = True
        
        def continuous_mine():
            block_height = self.blockchain_manager.get_blockchain_height() + 1
            latest_block = self.blockchain_manager.get_latest_block()
            previous_hash = latest_block.get('hash', '0' * 64) if latest_block else '0' * 64
            
            while self.mining_active:
                # Check mempool for transactions
                pending_count = len(self.mempool_manager.get_pending_transactions())
                
                if pending_count > 0:
                    print(f"🔄 Mining block #{block_height} with {pending_count} pending transactions...")
                    
                    result = self.mine_transaction_block(miner_address, previous_hash, block_height)
                    
                    if result.get("success"):
                        if callback:
                            callback(result)
                        
                        # Update for next block
                        block_height += 1
                        previous_hash = result["block"]["hash"]
                    
                    # Brief pause between blocks
                    time.sleep(2)
                else:
                    print("⏳ No transactions in mempool, waiting...")
                    time.sleep(10)  # Wait longer if no transactions
        
        self.current_thread = threading.Thread(target=continuous_mine, daemon=True)
        self.current_thread.start()
        print("Started continuous block mining")
        return True
    
    def start_hybrid_mining(self, miner_address: str, bill_denominations: List[float] = None, 
                          callback: Callable = None) -> bool:
        """Start hybrid mining - both GTX bills and transaction blocks"""
        if self.mining_active:
            print("Mining already active")
            return False
            
        self.mining_active = True
        
        def hybrid_mine():
            # Mine GTX bills first if denominations provided
            if bill_denominations:
                for denomination in bill_denominations:
                    if not self.mining_active:
                        break
                    
                    print(f"Mining GTX ${denomination:,} bill...")
                    bill_result = self.mine_bill(denomination, miner_address)
                    
                    if callback:
                        callback({"type": "bill", "data": bill_result})
                    
                    time.sleep(1)
            
            # Switch to continuous block mining
            block_height = self.blockchain_manager.get_blockchain_height() + 1
            latest_block = self.blockchain_manager.get_latest_block()
            previous_hash = latest_block.get('hash', '0' * 64) if latest_block else '0' * 64
            
            while self.mining_active:
                pending_count = len(self.mempool_manager.get_pending_transactions())
                
                if pending_count > 0:
                    print(f"🔄 Mining transaction block #{block_height}...")
                    
                    block_result = self.mine_transaction_block(miner_address, previous_hash, block_height)
                    
                    if block_result.get("success"):
                        if callback:
                            callback({"type": "block", "data": block_result})
                        
                        block_height += 1
                        previous_hash = block_result["block"]["hash"]
                    
                    time.sleep(2)
                else:
                    print("⏳ No transactions, checking again in 10s...")
                    time.sleep(10)
        
        self.current_thread = threading.Thread(target=hybrid_mine, daemon=True)
        self.current_thread.start()
        print("Started hybrid mining (bills + blocks)")
        return True
    
    def get_mining_stats(self) -> Dict:
        """Get comprehensive mining statistics"""
        pending_txs = self.mempool_manager.get_pending_transactions()
        
        return {
            "mining_active": self.mining_active,
            "bills_mined": self.mining_stats["bills_mined"],
            "blocks_mined": self.mining_stats["blocks_mined"],
            "total_mining_time": self.mining_stats["total_mining_time"],
            "total_hash_attempts": self.mining_stats["total_hash_attempts"],
            "mempool_size": len(pending_txs),
            "pending_transactions": [
                {
                    "hash": tx.get('hash', '')[:16] + '...',
                    "from": tx.get('from', ''),
                    "to": tx.get('to', ''),
                    "amount": tx.get('amount', 0),
                    "fee": tx.get('fee', 0),
                    "type": tx.get('type', 'unknown')
                }
                for tx in pending_txs[:5]  # Show first 5
            ],
            "average_hashrate": (
                self.mining_stats["total_hash_attempts"] / self.mining_stats["total_mining_time"]
                if self.mining_stats["total_mining_time"] > 0 else 0
            )
        }
    
    def stop_mining(self):
        """Stop all mining activities"""
        self.mining_active = False
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=5)
        print("Mining stopped")
        
        stats = self.get_mining_stats()
        print(f"Final statistics:")
        print(f"Bills mined: {stats['bills_mined']}")
        print(f"Blocks mined: {stats['blocks_mined']}")
        print(f"Total mining time: {stats['total_mining_time']:.2f}s")
        print(f"Average hashrate: {stats['average_hashrate']:,.0f} H/s")
        print(f"Mempool size: {stats['mempool_size']} transactions")
    
    def submit_transaction(self, transaction: Dict) -> bool:
        """Submit transaction to mempool for mining"""
        try:
            success = self.mempool_manager.add_transaction(transaction)
            if success:
                print(f"📨 Added transaction to mining mempool: {transaction.get('hash', '')[:16]}...")
            return success
        except Exception as e:
            print(f"Error submitting transaction: {e}")
            return False
    
    def get_network_status(self) -> Dict:
        """Get current network and blockchain status"""
        try:
            height = self.blockchain_manager.get_blockchain_height()
            connected = self.blockchain_manager.check_network_connection()
            mempool_size = len(self.mempool_manager.get_pending_transactions())
            
            return {
                "network_connected": connected,
                "blockchain_height": height,
                "mempool_size": mempool_size,
                "mining_active": self.mining_active
            }
        except Exception as e:
            return {
                "network_connected": False,
                "error": str(e)
            }

class Miner:
    # Add: allow user to set number of CPU workers
    def set_cpu_workers(self, num_workers: int):
        self.cpu_workers = max(1, int(num_workers))

    def get_cpu_workers(self):
        return getattr(self, 'cpu_workers', 1)
    """
    Robust miner class that integrates with BlockchainManager, MempoolManager, 
    DifficultySystem, and security validation to mine all transaction types.
    """

    def __init__(self, config, data_manager, mining_started_callback=None, mining_completed_callback=None, block_mined_callback=None, block_added_callback=None):
        self.config = config
        self.data_manager = data_manager
        self.is_mining = False
        self.blocks_mined = 0
        self.total_reward = 0.0
        self.mining_started_callback = mining_started_callback
        self.mining_completed_callback = mining_completed_callback
        self.block_mined_callback = block_mined_callback
        self.block_added_callback = block_added_callback  # 新規: ブロックがチェーンに追加された時のコールバック
        self.mining_status_callbacks: List[Callable] = []

        self.mining_history = self.data_manager.load_mining_history()

        # Ensure hashrate_callback is always defined
        self.hashrate_callback = None

        # Mining engine toggles (default to enabled when unspecified)
        def _resolve_flag(flag_names: tuple[str, ...], default: bool) -> bool:
            for name in flag_names:
                if hasattr(config, name):
                    value = getattr(config, name)
                    if value is None:
                        continue
                    return bool(value)
            return default

        gpu_flags = ("enable_gpu_mining", "gpu_mining", "use_gpu", "enable_cuda")
        self.gpu_enabled = _resolve_flag(gpu_flags, True)
        cpu_flags = ("enable_cpu_mining", "cpu_mining", "use_cpu")
        self.cpu_enabled = _resolve_flag(cpu_flags, True)
        sm3_kernel_flags = ("cuda_sm3_kernel", "use_sm3_kernel", "gpu_sm3_kernel")
        self.cuda_sm3_kernel = _resolve_flag(sm3_kernel_flags, True)

        # Initialize lunalib components
        self.blockchain_manager = BlockchainManager(endpoint_url=config.node_url)
        self.mempool_manager = MempoolManager([config.node_url])
        self.difficulty_system = DifficultySystem()
        self.multi_gpu_enabled = getattr(self, 'multi_gpu_enabled', False)
        if hasattr(self, 'config') and hasattr(self.config, 'multi_gpu_enabled'):
            self.multi_gpu_enabled = bool(getattr(self.config, 'multi_gpu_enabled', False))
        elif os.getenv('LUNALIB_MULTI_GPU', '0') == '1':
            self.multi_gpu_enabled = True
        self.cuda_manager = CUDAManager() if self.gpu_enabled else None
        if self.cuda_manager:
            self.cuda_manager.use_sm3_kernel = self.cuda_sm3_kernel
        
        # Import security components
        try:
            from ..transactions.security import SecurityManager
            from ..transactions.validator import TransactionValidator
            self.security_manager = SecurityManager()
            self.transaction_validator = TransactionValidator()
        except ImportError:
            self.security_manager = None
            self.transaction_validator = None

        self.current_hash = ""
        self.current_nonce = 0
        self.hash_rate = 0
        self.mining_thread = None
        self.should_stop_mining = False

        # Telemetry for recent mining attempts
        self.last_cpu_hashrate = 0.0
        self.last_cpu_attempts = 0
        self.last_cpu_duration = 0.0
        self.last_engine_used = None
        self.last_gpu_hashrate = 0.0
        self.last_gpu_attempts = 0
        self.last_gpu_duration = 0.0
        self._last_hashing_status = 0.0
        self._reward_mode_cache = {"mode": None, "ts": 0.0}

    def _resolve_reward_mode(self) -> str:
        """Resolve reward mode, preferring daemon configuration when available."""
        for attr in ("block_reward_mode", "reward_mode"):
            if hasattr(self.config, attr):
                value = getattr(self.config, attr)
                if value:
                    return str(value).lower().strip()

        env_mode = os.getenv("LUNALIB_BLOCK_REWARD_MODE")
        if env_mode:
            return str(env_mode).lower().strip()

        ttl = float(os.getenv("LUNALIB_REWARD_MODE_TTL", "30"))
        now = time.time()
        if now - self._reward_mode_cache.get("ts", 0.0) >= ttl:
            try:
                stats = self.blockchain_manager.get_server_stats()
            except Exception:
                stats = {}
            mode = (
                stats.get("block_reward_mode")
                or stats.get("reward_mode")
                or stats.get("blockRewardMode")
            )
            if mode:
                self._reward_mode_cache = {"mode": str(mode).lower().strip(), "ts": now}
        return self._reward_mode_cache.get("mode") or "exponential"

    def on_mining_status(self, callback: Callable) -> None:
        """Register a callback for mining status updates."""
        self.mining_status_callbacks.append(callback)

    def _emit_status(self, phase: str, message: str = "", payload: Optional[Dict] = None) -> None:
        data = {
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
            "engine": self.last_engine_used,
            "hash_rate": self.hash_rate or self.last_cpu_hashrate or self.last_gpu_hashrate,
        }
        if payload:
            data.update(payload)
        if bool(int(os.getenv("LUNALIB_MINER_CONCISE", "1"))):
            labels = {
                "next": "Next Block...",
                "initializing": "Initializing...",
                "downloading": "Downloading Block...",
                "mempool": "Downloading Mempool...",
                "validating": "Validating...",
                "hashing": "Hashing...",
                "submitting": "Submitting...",
                "success": "Success",
                "failure": "Failure",
            }
            label = labels.get(phase, phase)
            extra = f" {message}" if message else ""
            print(f"{label}{extra}")
        for cb in self.mining_status_callbacks:
            try:
                cb(data)
            except Exception:
                pass

    def mine_block(self) -> tuple[bool, str, Optional[Dict]]:
        """
        Mine a block from the mempool with proper validation and difficulty calculation.
        Returns: (success, message, block_data)
        """
        safe_print("[DEBUG] Entered mine_block()")
        try:
            self._emit_status("next")
            self._emit_status("downloading", "Fetching latest block")
            # Get the latest block from the blockchain
            latest_block = self.blockchain_manager.get_latest_block()
            if not latest_block:
                return False, "Could not get latest block from server", None

            current_index = latest_block.get('index', 0)
            previous_hash = latest_block.get('hash', '0' * 64)
            new_index = current_index + 1

            # Get fresh transactions from mempool
            mempool = self._get_fresh_mempool()
            self._emit_status("mempool", f"Fetched {len(mempool)} pending txs")
            
            # Validate all transactions
            self._emit_status("validating", "Validating mempool transactions")
            valid_transactions = self._validate_transactions(mempool)
            
            # Calculate block difficulty based on transactions
            block_difficulty = self._calculate_block_difficulty(valid_transactions)
            
            # Calculate block reward:
            # - Empty blocks use LINEAR system: difficulty 1 = 1 LKC, difficulty 2 = 2 LKC, etc.
            # - Blocks with transactions use EXPONENTIAL system: 10^(difficulty-1)
            if not valid_transactions:
                # Empty block: linear reward
                total_reward = float(block_difficulty)
                reward_tx = self._create_empty_block_reward(new_index, block_difficulty, total_reward)
                valid_transactions = [reward_tx]
            else:
                # Regular block: reward mode can be linear (difficulty) or exponential
                reward_mode = self._resolve_reward_mode()
                if reward_mode == "linear":
                    total_reward = float(block_difficulty)
                else:
                    total_reward = self._calculate_exponential_block_reward(block_difficulty)
                reward_tx = self._create_block_reward(new_index, block_difficulty, total_reward)
                valid_transactions = valid_transactions + [reward_tx]

            # Create block data
            block_data = {
                'index': new_index,
                'previous_hash': previous_hash,
                'timestamp': time.time(),
                'transactions': valid_transactions,
                'miner': self.config.miner_address,
                'difficulty': block_difficulty,
                'nonce': 0,
                'reward': total_reward,
                'hash': ''
            }

            self._emit_status("hashing", "Starting proof-of-work", {
                "difficulty": block_difficulty,
                "index": new_index,
                "transaction_count": len(valid_transactions),
            })


            cuda_available = self.gpu_enabled and self.cuda_manager and self.cuda_manager.cuda_available
            cpu_available = self.cpu_enabled

            safe_print(f"[DEBUG] (mine_block) gpu_enabled={self.gpu_enabled}, cpu_enabled={self.cpu_enabled}, cuda_manager={self.cuda_manager}, cuda_available={cuda_available}, cpu_available={cpu_available}")

            # If both are enabled, prefer GPU if available, else CPU
            if cuda_available:
                safe_print("[DEBUG] Mining engine selected: GPU (CUDA)")
                self.last_engine_used = "gpu"
                try:
                    cuda_result = self._cuda_mine(block_data, block_difficulty)
                except Exception as e:
                    safe_print(f"[FATAL] Exception calling _cuda_mine: {e}")
                    cuda_result = None
                if cuda_result:
                    return self._finalize_block(cuda_result, 'cuda', total_reward)
                else:
                    safe_print("[DEBUG] GPU mining failed or not successful, falling back to CPU if available...")
                    # Fallback to CPU if allowed
                    if cpu_available:
                        self.last_engine_used = "cpu"
                        try:
                            cpu_result = self._cpu_mine(block_data, block_difficulty)
                        except Exception as e:
                            safe_print(f"[FATAL] Exception calling _cpu_mine: {e}")
                            cpu_result = None
                        if cpu_result:
                            return self._finalize_block(cpu_result, 'cpu', total_reward)
                        else:
                            safe_print("[FATAL] CPU mining also failed after GPU fallback!")
                    else:
                        safe_print("[FATAL] No CPU available to fallback after GPU mining failed!")
            elif cpu_available:
                safe_print("[DEBUG] Mining engine selected: CPU")
                self.last_engine_used = "cpu"
                try:
                    cpu_result = self._cpu_mine(block_data, block_difficulty)
                except Exception as e:
                    safe_print(f"[FATAL] Exception calling _cpu_mine: {e}")
                    cpu_result = None
                if cpu_result:
                    return self._finalize_block(cpu_result, 'cpu', total_reward)
                else:
                    safe_print("[FATAL] CPU mining failed!")
            else:
                safe_print("[FATAL] No mining engine available! Both GPU and CPU mining are disabled.")

            self._emit_status("failure", "Mining disabled or timeout - no solution found")
            return False, "Mining disabled or timeout - no solution found", None

        except Exception as e:
            safe_print(f"[DEBUG] Exception in mine_block: {e}")
            self._emit_status("failure", f"Mining error: {str(e)}")
            return False, f"Mining error: {str(e)}", None

    def _get_fresh_mempool(self) -> List[Dict]:
        """Get fresh mempool transactions with validation"""
        try:
            mempool = self.mempool_manager.get_pending_transactions()
            if not mempool:
                mempool = self.blockchain_manager.get_mempool()
            return mempool if mempool else []
        except Exception as e:
            safe_print(f"Error fetching mempool: {e}")
            return []

    def _validate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Validate transactions using security manager"""
        valid_transactions = []
        
        for tx in transactions:
            try:
                # Basic validation
                if not self._validate_transaction_structure(tx):
                    continue
                
                # Security validation if available
                if self.transaction_validator:
                    if not self.transaction_validator.validate_transaction(tx):
                        continue
                
                valid_transactions.append(tx)
            except Exception as e:
                safe_print(f"Transaction validation error: {e}")
                continue
        
        return valid_transactions

    def _validate_transaction_structure(self, tx: Dict) -> bool:
        """Basic transaction structure validation"""
        required_fields = ['type', 'timestamp']
        
        for field in required_fields:
            if field not in tx:
                return False
        
        tx_type = tx.get('type')
        
        if tx_type == 'transaction':
            if not all(k in tx for k in ['from', 'to', 'amount']):
                return False
            # Reject self-transfers (from == to) as they fail server validation
            if tx.get('from') == tx.get('to'):
                safe_print(f"⚠️  Filtering out self-transfer: {tx.get('from')} → {tx.get('to')}")
                return False
        elif tx_type == 'genesis_bill':
            if 'denomination' not in tx:
                return False
        elif tx_type == 'reward':
            if not all(k in tx for k in ['to', 'amount']):
                return False
        
        return True

    def _calculate_block_difficulty(self, transactions: List[Dict]) -> int:
        """Calculate block difficulty based on transactions using DifficultySystem"""
        if not transactions:
            return self.config.difficulty

        max_difficulty = self.config.difficulty

        for tx in transactions:
            tx_type = tx.get('type')
            
            if tx_type == 'genesis_bill':
                denomination = tx.get('denomination', 0)
                tx_difficulty = self.difficulty_system.get_bill_difficulty(denomination)
                max_difficulty = max(max_difficulty, tx_difficulty)
            elif tx_type == 'transaction':
                amount = tx.get('amount', 0)
                tx_difficulty = self.difficulty_system.get_transaction_difficulty(amount)
                max_difficulty = max(max_difficulty, tx_difficulty)
            elif tx_type == 'reward':
                max_difficulty = max(max_difficulty, 1)

        return max(max_difficulty, self.config.difficulty)

    def _calculate_block_reward(self, transactions: List[Dict]) -> float:
        """Calculate total block reward based on transactions using DifficultySystem"""
        total_reward = 0.0

        for tx in transactions:
            tx_type = tx.get('type')
            
            if tx_type == 'genesis_bill':
                denomination = tx.get('denomination', 0)
                # Use difficulty system to calculate proper reward
                bill_difficulty = self.difficulty_system.get_bill_difficulty(denomination)
                avg_mining_time = 15.0  # Will be updated with actual time
                bill_reward = self.difficulty_system.calculate_mining_reward(denomination, avg_mining_time)
                total_reward += bill_reward
            elif tx_type == 'transaction':
                fee = tx.get('fee', 0)
                total_reward += fee
            elif tx_type == 'reward':
                reward_amount = tx.get('amount', 0)
                total_reward += reward_amount

        # Minimum reward for empty blocks
        if total_reward == 0:
            total_reward = 1.0

        return total_reward
    
    def _calculate_exponential_block_reward(self, difficulty: int) -> float:
        """Calculate block reward using exponential difficulty system
        
        Uses the new exponential reward system:
        difficulty 1 = 1 LKC
        difficulty 2 = 10 LKC  
        difficulty 3 = 100 LKC
        difficulty 9 = 100,000,000 LKC
        """
        return self.difficulty_system.calculate_block_reward(difficulty)

    def _create_empty_block_reward(self, block_index: int, difficulty: int, reward_amount: float) -> Dict:
        """Create reward transaction for empty blocks using LINEAR reward system (difficulty = reward)"""
        return {
            'type': 'reward',
            'from': 'network',
            'to': self.config.miner_address,
            'amount': reward_amount,
            'timestamp': time.time(),
            'block_height': block_index,
            'difficulty': difficulty,
            'hash': f"reward_{block_index}_{int(time.time())}",
            'description': f'Empty block mining reward (Linear: Difficulty {difficulty} = {reward_amount} LKC)',
            'is_empty_block': True
        }

    def _create_block_reward(self, block_index: int, difficulty: int, reward_amount: float) -> Dict:
        """Create reward transaction for non-empty blocks using EXPONENTIAL reward system."""
        reward_mode = self._resolve_reward_mode()
        if reward_mode == "linear":
            return {
                'type': 'reward',
                'from': 'network',
                'to': self.config.miner_address,
                'amount': reward_amount,
                'timestamp': time.time(),
                'block_height': block_index,
                'difficulty': difficulty,
                'hash': f"reward_{block_index}_{int(time.time())}",
                'description': f'Block mining reward (Linear: Difficulty {difficulty} = {reward_amount} LKC)',
                'is_empty_block': False
            }
        else:
            return {
                'type': 'reward',
                'from': 'network',
                'to': self.config.miner_address,
                'amount': reward_amount,
                'timestamp': time.time(),
                'block_height': block_index,
                'difficulty': difficulty,
                'hash': f"reward_{block_index}_{int(time.time())}",
                'description': f'Block mining reward (Exponential: Difficulty {difficulty} = {reward_amount} LKC)',
                'is_empty_block': False
            }

    def _cuda_mine(self, block_data: Dict, difficulty: int) -> Optional[Dict]:
        """Mine using CUDA acceleration"""
        safe_print("[DEBUG] TOP OF _cuda_mine: function entered")
        if not self.gpu_enabled:
            safe_print("[DEBUG] _cuda_mine called but gpu_enabled is False (early return)")
            return None
        if not self.cuda_manager:
            safe_print("[DEBUG] _cuda_mine: self.cuda_manager is None (early return)")
            return None
        if not getattr(self.cuda_manager, 'cuda_available', False):
            safe_print("[DEBUG] _cuda_mine: cuda_manager.cuda_available is False (early return)")
            return None
        try:
            safe_print("[DEBUG] Entering _cuda_mine: Attempting CUDA mining...")
            if self.hashrate_callback:
                self.hashrate_callback(0.0, 'gpu')
            safe_print(f"[DEBUG] About to call cuda_manager.cuda_mine_batch with difficulty={difficulty}")
            status_interval = float(os.getenv("LUNALIB_MINER_STATUS_INTERVAL", "5"))
            batch_size = int(getattr(self.config, "cuda_batch_size", 100000) or 100000)
            env_batch = os.getenv("LUNALIB_CUDA_BATCH_SIZE")
            if env_batch:
                try:
                    batch_size = int(env_batch)
                except Exception:
                    pass

            def _progress(update: Dict):
                now = time.time()
                self.hash_rate = float(update.get("hashrate", 0.0))
                self.last_gpu_hashrate = self.hash_rate
                self.last_gpu_attempts = int(update.get("attempts", 0))
                self.last_gpu_duration = float(update.get("duration", 0.0))
                if now - self._last_hashing_status >= status_interval:
                    self._last_hashing_status = now
                    rate = self.hash_rate
                    self._emit_status("hashing", f"{rate:,.0f} H/s")
                    safe_print(f"[DEBUG] [GPU] Hashrate update: {rate:,.0f} H/s, attempts={self.last_gpu_attempts}, duration={self.last_gpu_duration}")
                    if self.hashrate_callback:
                        self.hashrate_callback(rate, 'gpu')

            cuda_result = None
            try:
                if self.multi_gpu_enabled:
                    safe_print("[DEBUG] Multi-GPU mining enabled. Using cuda_mine_multi_gpu_batch.")
                    cuda_result = self.cuda_manager.cuda_mine_multi_gpu_batch(
                        block_data, difficulty, batch_size=batch_size, progress_callback=_progress
                    )
                else:
                    cuda_result = self.cuda_manager.cuda_mine_batch(
                        block_data, difficulty, batch_size=batch_size, progress_callback=_progress
                    )
                safe_print(f"[DEBUG] cuda_manager mining returned: {cuda_result}")
            except Exception as ce:
                safe_print(f"[DEBUG] Exception in cuda_manager mining: {ce}")
                return None
            if cuda_result and cuda_result.get('success'):
                mining_time = float(cuda_result.get("mining_time", 0))
                nonce = int(cuda_result.get("nonce", 0))
                if mining_time > 0:
                    self.hash_rate = nonce / mining_time
                    self.last_gpu_hashrate = self.hash_rate
                    self.last_gpu_attempts = nonce
                    self.last_gpu_duration = mining_time
                    self.last_cpu_hashrate = 0.0
                    self.last_cpu_attempts = 0
                    self.last_cpu_duration = 0.0
                    safe_print(f"[DEBUG] [GPU] Final hashrate: {self.hash_rate:,.0f} H/s, nonce={nonce}, mining_time={mining_time}")
                    if self.hashrate_callback:
                        self.hashrate_callback(self.hash_rate, 'gpu')
                self.current_nonce = nonce
                self.current_hash = cuda_result.get('hash', '')
                block_data['hash'] = cuda_result['hash']
                block_data['nonce'] = cuda_result['nonce']
                return block_data
        except Exception as e:
            safe_print(f"[DEBUG] Exception in _cuda_mine: {e}")
        return None

    def _cpu_mine(self, block_data: Dict, difficulty: int) -> Optional[Dict]:
        """Mine using CPU"""
        import concurrent.futures
        if not self.cpu_enabled:
            safe_print("[DEBUG] _cpu_mine called but cpu_enabled is False")
            return None
        safe_print("[DEBUG] Entering _cpu_mine: Using CPU mining (multi-threaded)...")
        if self.hashrate_callback:
            self.hashrate_callback(0.0, 'cpu')
        start_time = time.time()
        target = "0" * difficulty
        num_workers = self.get_cpu_workers()
        max_nonce = 1000000
        found = threading.Event()
        result = [None]

        def worker(start_nonce, step):
            nonce = start_nonce
            hash_count = 0
            last_hash_update = time.time()
            while not found.is_set() and nonce < max_nonce:
                block_hash = self._calculate_block_hash(
                    block_data['index'],
                    block_data['previous_hash'],
                    block_data['timestamp'],
                    block_data['transactions'],
                    nonce,
                    block_data['miner'],
                    difficulty
                )
                if block_hash.startswith(target):
                    elapsed = time.time() - start_time
                    safe_print(f"[DEBUG] [CPU] Block found: nonce={nonce}, hash={block_hash}, elapsed={elapsed:.2f}s")
                    block_data['hash'] = block_hash
                    block_data['nonce'] = nonce
                    result[0] = block_data.copy()
                    found.set()
                    return
                nonce += step
                hash_count += 1
                # Hashrate update
                current_time = time.time()
                if current_time - last_hash_update >= 1:
                    rate = hash_count / (current_time - last_hash_update)
                    last_hash_update = current_time
                    hash_count = 0
                    safe_print(f"[DEBUG] [CPU] Worker {start_nonce}: {rate:,.0f} H/s at nonce {nonce}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i, num_workers) for i in range(num_workers)]
            concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            found.set()
        elapsed = time.time() - start_time
        if result[0]:
            self.last_cpu_attempts = result[0]['nonce']
            self.last_cpu_duration = elapsed
            self.last_cpu_hashrate = result[0]['nonce'] / elapsed if elapsed > 0 else 0.0
            safe_print(f"[DEBUG] [CPU] Final hashrate: {self.last_cpu_hashrate:,.0f} H/s, nonce={result[0]['nonce']}, elapsed={elapsed:.2f}s")
            if self.hashrate_callback:
                self.hashrate_callback(self.last_cpu_hashrate, 'cpu')
            return result[0]
        return None
    def profile_hash_function(self, n=10000):
        import time
        block_data = {
            'index': 0,
            'previous_hash': '0'*64,
            'timestamp': time.time(),
            'transactions': [],
            'miner': 'PROFILE',
            'difficulty': 1,
            'nonce': 0,
            'reward': 1.0,
            'hash': ''
        }
        start = time.perf_counter()
        for i in range(n):
            self._calculate_block_hash(
                block_data['index'],
                block_data['previous_hash'],
                block_data['timestamp'],
                block_data['transactions'],
                i,
                block_data['miner'],
                1
            )
        elapsed = time.perf_counter() - start
        print(f"[PROFILE] {n} hashes in {elapsed:.4f}s | {n/elapsed:,.0f} H/s")
        if n/elapsed < 10000:
            print("[PROFILE] WARNING: Hash function is slow! Consider using a C/CUDA-accelerated version.")

    def _calculate_block_hash(self, index: int, previous_hash: str, timestamp: float, 
                             transactions: List[Dict], nonce: int, miner: str, difficulty: int) -> str:
        """Calculate SHA-256 hash of a block matching server validation"""
        try:
            block_data = {
                "difficulty": int(difficulty),
                "index": int(index),
                "miner": str(miner),
                "nonce": int(nonce),
                "previous_hash": str(previous_hash),
                "timestamp": float(timestamp),
                "transactions": [],  # Empty for mining proof
                "version": "1.0"
            }

            block_string = json.dumps(block_data, sort_keys=True)
            calculated_hash = sm3_hex(block_string.encode())
            return calculated_hash

        except Exception as e:
            safe_print(f"Hash calculation error: {e}")
            return "0" * 64

    def _finalize_block(self, block_data: Dict, method: str, total_reward: float) -> tuple[bool, str, Dict]:
        """Finalize mined block with proper record keeping and blockchain submission"""
        mining_time = time.time() - block_data.get('timestamp', time.time())

        self._emit_status("validating", "Validating mined block", {
            "index": block_data.get("index"),
            "hash": (block_data.get("hash", "")[:16] + "...") if block_data.get("hash") else "",
        })

        # Validate block before submission
        validation_result = self._validate_mined_block(block_data)
        if not validation_result[0]:
            safe_print(f"❌ Block validation failed: {validation_result[1]}")
            self._emit_status("failure", f"Validation failed: {validation_result[1]}")
            return False, f"Block validation failed: {validation_result[1]}", None

        # Block reward is already calculated based on difficulty (exponential system)
        final_reward = block_data['reward']

        # Submit block to blockchain
        try:
            self._emit_status("submitting", "Submitting mined block", {
                "index": block_data.get("index"),
                "method": method,
            })
            submission_success = self.blockchain_manager.submit_mined_block(block_data)

            if not submission_success:
                safe_print(f"⚠️  Block #{block_data['index']} submission failed")
                self._emit_status("failure", f"Submission failed for block #{block_data['index']}")
                return False, f"Block #{block_data['index']} mined but submission failed", None

            safe_print(f"✅ Block #{block_data['index']} submitted successfully (Reward: {final_reward} LKC)")
            self._emit_status("success", f"Block #{block_data['index']} submitted", {
                "reward": final_reward,
            })

            # 新規: ブロックがチェーンに追加されたことを通知
            if self.block_added_callback:
                try:
                    self.block_added_callback(block_data)
                except Exception as e:
                    safe_print(f"[WARN] block_added_callback error: {e}")

            # Clear mined transactions from mempool
            self._clear_transactions_from_mempool(block_data['transactions'])

        except Exception as e:
            safe_print(f"❌ Block submission error: {e}")
            self._emit_status("failure", f"Submission error: {str(e)}")
            return False, f"Block submission error: {str(e)}", None

        # Record mining history
        mining_record = {
            'block_index': block_data['index'],
            'timestamp': time.time(),
            'mining_time': mining_time,
            'difficulty': block_data['difficulty'],
            'nonce': block_data['nonce'],
            'hash': block_data['hash'],
            'method': method,
            'reward': final_reward,
            'status': 'success'
        }
        self.mining_history.append(mining_record)
        self.save_mining_history()

        self.blocks_mined += 1
        self.total_reward += final_reward

        if self.mining_completed_callback:
            self.mining_completed_callback(True, f"Block #{block_data['index']} mined - Reward: {final_reward}")

        if self.block_mined_callback:
            self.block_mined_callback(block_data)

        # --- Update bills.db, mined blocks, and mining stats ---
        try:
            # Update bills.db if available in data_manager
            if hasattr(self.data_manager, "update_bills_db"):
                self.data_manager.update_bills_db(block_data)
            # Update mined blocks record
            if hasattr(self.data_manager, "record_mined_block"):
                self.data_manager.record_mined_block(block_data)
            # Update mining stats
            if hasattr(self.data_manager, "update_mining_stats"):
                self.data_manager.update_mining_stats(self.get_mining_stats())
        except Exception as e:
            safe_print(f"[WARN] Failed to update mining records: {e}")

        return True, f"Block #{block_data['index']} mined - Reward: {final_reward}", block_data
    
    def _validate_mined_block(self, block: Dict) -> tuple:
        """Validate mined block before submission
        
        Returns: (is_valid, error_message)
        """
        # Validate structure
        is_valid, error = self.difficulty_system.validate_block_structure(block)
        if not is_valid:
            return False, error
        
        # Validate hash meets difficulty
        block_hash = block.get('hash', '')
        difficulty = block.get('difficulty', 0)
        
        if not self.difficulty_system.validate_block_hash(block_hash, difficulty):
            return False, f"Hash does not meet difficulty {difficulty} requirement"
        
        # Validate previous hash (get from blockchain)
        try:
            latest_block = self.blockchain_manager.get_latest_block()
            if latest_block:
                expected_prev_hash = latest_block.get('hash', '')
                if block.get('previous_hash') != expected_prev_hash:
                    return False, f"Previous hash mismatch: expected {expected_prev_hash[:16]}..., got {block.get('previous_hash', '')[:16]}..."
        except Exception as e:
            safe_print(f"⚠️  Could not validate previous hash: {e}")
        
        # Validate reward matches difficulty
        # Empty blocks use LINEAR system (difficulty = reward)
        # Regular blocks use EXPONENTIAL system (10^(difficulty-1))
        transactions = block.get('transactions', [])
        reward_tx = next((tx for tx in transactions if tx.get('type') == 'reward'), None)
        non_reward_txs = [tx for tx in transactions if tx.get('type') != 'reward']
        is_empty_block = not non_reward_txs or (len(transactions) == 1 and reward_tx and reward_tx.get('is_empty_block', False))
        
        if is_empty_block:
            # Empty block: linear reward
            expected_reward = float(difficulty)
        else:
            # Regular block: exponential reward
            expected_reward = self.difficulty_system.calculate_block_reward(difficulty)
        
        reward_tx_amount = reward_tx.get('amount', 0) if reward_tx else None
        actual_reward = block.get('reward', reward_tx_amount if reward_tx_amount is not None else 0)
        
        # Allow some tolerance for floating point comparison
        if abs(actual_reward - expected_reward) > 0.01:
            reward_type = "Linear" if is_empty_block else "Exponential"
            return False, f"Reward mismatch ({reward_type}): expected {expected_reward} LKC for difficulty {difficulty}, got {actual_reward} LKC"
        
        reward_type = "Linear" if is_empty_block else "Exponential"
        safe_print(f"✅ Block validation passed ({reward_type}): Hash meets difficulty {difficulty}, Reward: {actual_reward} LKC")
        return True, ""
    
    def _clear_transactions_from_mempool(self, transactions: List[Dict]):
        """Remove mined transactions from mempool"""
        try:
            for tx in transactions:
                # Skip reward transactions (they were created during mining)
                if tx.get('type') == 'reward' and tx.get('from') == 'network':
                    continue
                
                tx_hash = tx.get('hash')
                if tx_hash:
                    # Remove from mempool manager if available
                    try:
                        self.mempool_manager.remove_transaction(tx_hash)
                    except:
                        pass  # Silent fail if method doesn't exist
            
            safe_print(f"🧹 Cleared {len(transactions)} transactions from mempool")
            
        except Exception as e:
            safe_print(f"⚠️  Error clearing mempool: {e}")

    def _calculate_final_reward(self, transactions: List[Dict], actual_mining_time: float) -> float:
        """Calculate final reward using actual mining time"""
        total_reward = 0.0

        for tx in transactions:
            tx_type = tx.get('type')
            
            if tx_type == 'genesis_bill':
                denomination = tx.get('denomination', 0)
                bill_reward = self.difficulty_system.calculate_mining_reward(denomination, actual_mining_time)
                total_reward += bill_reward
            elif tx_type == 'transaction':
                total_reward += tx.get('fee', 0)
            elif tx_type == 'reward':
                total_reward += tx.get('amount', 0)

        if total_reward == 0:
            total_reward = 1.0

        return total_reward

    def save_mining_history(self):
        """Save mining history to storage"""
        self.data_manager.save_mining_history(self.mining_history)

    def start_mining(self):
        """Start the mining process"""
        if self.is_mining:
            return

        self.is_mining = True
        self.should_stop_mining = False

        if self.mining_started_callback:
            self.mining_started_callback()

        if not self.mining_thread or not self.mining_thread.is_alive():
            self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
            self.mining_thread.start()

    def _mining_loop(self):
        """Background mining loop that updates hashrate telemetry."""
        status_interval = float(os.getenv("LUNALIB_MINER_STATUS_INTERVAL", "5"))
        last_status = time.time()

        while self.is_mining and not self.should_stop_mining:
            success, message, block = self.mine_block()

            if success:
                if self.block_mined_callback:
                    try:
                        self.block_mined_callback(block)
                    except Exception:
                        pass
            else:
                # Avoid tight loop if mining is disabled/unavailable
                time.sleep(0.25)

            now = time.time()
            if now - last_status >= status_interval:
                last_status = now
                engine = self.last_engine_used or "none"
                rate = self.hash_rate or self.last_cpu_hashrate
                safe_print(f"[MINER] engine={engine} rate={rate:,.0f} H/s nonce={self.current_nonce}")

    def stop_mining(self):
        """Stop the mining process"""
        self.is_mining = False
        self.should_stop_mining = True
        if self.mining_thread and self.mining_thread.is_alive():
            self.mining_thread.join()

    def get_mining_stats(self):
        """Return the current mining statistics"""
        return {
            "blocks_mined": self.blocks_mined,
            "total_reward": self.total_reward,
            "current_hash": self.current_hash,
            "current_nonce": self.current_nonce,
            "hash_rate": self.hash_rate,
            "mining_history": len(self.mining_history),
            "last_engine_used": self.last_engine_used,
            "last_cpu_hashrate": self.last_cpu_hashrate,
            "last_cpu_attempts": self.last_cpu_attempts,
            "last_cpu_duration": self.last_cpu_duration
            ,
            "last_gpu_hashrate": self.last_gpu_hashrate,
            "last_gpu_attempts": self.last_gpu_attempts,
            "last_gpu_duration": self.last_gpu_duration
        }