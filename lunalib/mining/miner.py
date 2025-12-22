# lunalib/mining/miner.py
import time
import hashlib
import json
import threading
from typing import Dict, Optional, List, Union, Callable
from ..mining.difficulty import DifficultySystem
from ..gtx.digital_bill import DigitalBill
from ..transactions.transactions import TransactionManager
from ..core.blockchain import BlockchainManager
from ..core.mempool import MempoolManager

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
        
        print("🔧 GenesisMiner initialized with integrated lunalib components")
    
    def mine_bill(self, denomination: float, user_address: str, bill_data: Dict = None) -> Dict:
        """Mine a GTX Genesis bill using DigitalBill system"""
        try:
            difficulty = self.difficulty_system.get_bill_difficulty(denomination)
            
            # Create digital bill using GTX system
            digital_bill = DigitalBill(
                denomination=denomination,
                user_address=user_address,
                difficulty=difficulty,
                bill_data=bill_data or {}
            )
            
            print(f"⛏️ Mining GTX ${denomination:,} Bill - Difficulty: {difficulty} zeros")
            
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
                
                print(f"✅ Successfully mined GTX ${denomination:,} bill!")
                print(f"⏱️ Mining time: {mining_time:.2f}s")
                print(f"📊 Hash attempts: {mining_result['nonce']:,}")
                print(f"🔗 Bill hash: {mining_result['hash'][:32]}...")
                
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
            
            print(f"⛏️ Mining Transaction Block #{block_height} - Difficulty: {difficulty} zeros")
            print(f"📦 Transactions: {len(transactions)} | Previous Hash: {previous_hash[:16]}...")
            
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
                
                print(f"✅ Successfully mined and validated Transaction Block #{block_height}!")
                print(f"⏱️ Mining time: {mining_time:.2f}s")
                print(f"💰 Block reward: {block['reward']:.6f} LUN")
                print(f"📊 Transactions: {block['transaction_count']}")
                print(f"🔗 Block hash: {mining_result['hash'][:32]}...")
                
                # Submit block to blockchain
                submission_success = self.blockchain_manager.submit_mined_block(block)
                if submission_success:
                    print("✅ Block successfully submitted to blockchain!")
                    # Clear mined transactions from local mempool
                    self._clear_mined_transactions(transactions)
                else:
                    print("⚠️ Block mined but submission failed")
                
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
            bill_hash = hashlib.sha256(data_string.encode()).hexdigest()
            
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
            
            # Create block hash
            block_string = json.dumps(block_data, sort_keys=True)
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
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
        base_reward = 1.0  # Base block reward
        total_fees = sum(tx.get('fee', 0) for tx in transactions)
        total_reward = base_reward + total_fees
        
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
            
            print(f"📊 Mining proof components:")
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
            original_hash = hashlib.sha256(original_string.encode()).hexdigest()
            
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
                json_hash = hashlib.sha256(json_string.encode()).hexdigest()
                
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
                json_hash_empty = hashlib.sha256(json_string_empty.encode()).hexdigest()
                
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
                tx_hashes.append(hashlib.sha256(tx_string.encode()).hexdigest())
        
        # Simple merkle root calculation
        while len(tx_hashes) > 1:
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
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