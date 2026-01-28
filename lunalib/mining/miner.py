# lunalib/mining/miner.py
import time
import sys
import os
import ctypes
import copy
# --- Unicode-safe print for Windows console ---
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', 'utf-8')
        print(*(str(a).encode(encoding, errors='replace').decode(encoding) for a in args), **kwargs)
import hashlib
import struct
from lunalib.core.sm3 import sm3_hex, sm3_digest, sm3_compact_hash, sm3_mine_compact, sm3_set_abort
import json
import threading
from typing import Dict, Optional, List, Union, Callable, Tuple, Any
from ..mining.difficulty import DifficultySystem
from ..gtx.digital_bill import DigitalBill
from ..transactions.transactions import TransactionManager
from ..core.blockchain import BlockchainManager
from ..core.mempool import MempoolManager
from ..mining.cuda_manager import CUDAManager


def _parse_cpu_list(value: str | None) -> List[int]:
    if not value:
        return []
    cores: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            cores.extend(range(start, end + 1))
        else:
            try:
                cores.append(int(part))
            except ValueError:
                continue
    return [c for c in cores if c >= 0]


def _pin_current_thread(core_id: int) -> bool:
    if core_id < 0:
        return False
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            mask = ctypes.c_size_t(1 << core_id)
            handle = kernel32.GetCurrentThread()
            result = kernel32.SetThreadAffinityMask(handle, mask)
            return bool(result)
        except Exception:
            return False
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core_id})
            return True
        except Exception:
            return False
    return False


def _get_linux_numa_cpulist(node: int) -> List[int]:
    path = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _parse_cpu_list(handle.read().strip())
    except Exception:
        return []


def _get_windows_numa_cpulist(node: int) -> List[int]:
    try:
        kernel32 = ctypes.windll.kernel32
        mask = ctypes.c_ulonglong()
        if not kernel32.GetNumaNodeProcessorMask(ctypes.c_ubyte(node), ctypes.byref(mask)):
            return []
        value = mask.value
        return [i for i in range(64) if (value >> i) & 1]
    except Exception:
        return []


def _get_numa_cpulist(node: int) -> List[int]:
    if os.name == "nt":
        return _get_windows_numa_cpulist(node)
    if os.name == "posix":
        return _get_linux_numa_cpulist(node)
    return []


def validate_mining_proof_internal(block: Dict) -> Dict[str, Any]:
    """Validate mining proof for a block using Lunalib canonical hashing."""
    if not isinstance(block, dict):
        return {"valid": False, "message": "Block must be a dict"}

    block_hash = str(block.get("hash", ""))
    try:
        difficulty = int(block.get("difficulty", 0))
    except Exception:
        return False, "Invalid difficulty"

    if not block_hash or difficulty < 0:
        return {"valid": False, "message": "Missing hash or difficulty"}

    target = "0" * difficulty
    if difficulty > 0 and not block_hash.startswith(target):
        return {"valid": False, "message": "Hash does not meet difficulty"}

    try:
        pow_payload = {
            "difficulty": int(difficulty),
            "index": int(block.get("index", 0)),
            "miner": str(block.get("miner", "")),
            "nonce": int(block.get("nonce", 0)),
            "previous_hash": str(block.get("previous_hash", "")),
            "timestamp": float(block.get("timestamp", 0.0)),
            "transactions": [],
            "version": "1.0",
        }
        pow_string = json.dumps(pow_payload, sort_keys=True)
        pow_hash = sm3_hex(pow_string.encode())
        if pow_hash == block_hash:
            return {"valid": True, "message": "OK"}
    except Exception as e:
        return {"valid": False, "message": f"SM3 hash validation error: {e}"}

    try:
        if os.getenv("LUNALIB_MINING_HASH_MODE", "json").lower() == "compact":
            previous_hash = str(block.get("previous_hash", ""))
            if len(previous_hash) == 64:
                prev_bytes = bytes.fromhex(previous_hash)
                miner_hash = sm3_digest(str(block.get("miner", "")).encode())
                base = (
                    prev_bytes
                    + int(block.get("index", 0)).to_bytes(4, "big", signed=False)
                    + int(difficulty).to_bytes(4, "big", signed=False)
                    + struct.pack(">d", float(block.get("timestamp", 0.0)))
                    + miner_hash
                )
                if len(base) == 80:
                    nonce = int(block.get("nonce", 0))
                    compact_hash = sm3_compact_hash(base, nonce).hex()
                    if compact_hash == block_hash:
                        return {"valid": True, "message": "OK"}
    except Exception as e:
        return {"valid": False, "message": f"Compact hash validation error: {e}"}

    return {"valid": False, "message": "SM3 hash mismatch"}

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
            enforced = os.getenv("LUNALIB_MINER_ADDRESS")
            if enforced and user_address and user_address != enforced:
                safe_print("[WARN] Miner address override denied; using configured miner address.")
                user_address = enforced
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
            enforced = os.getenv("LUNALIB_MINER_ADDRESS")
            if enforced and miner_address and miner_address != enforced:
                safe_print("[WARN] Miner address override denied; using configured miner address.")
                miner_address = enforced
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

                try:
                    non_reward_txs = [tx for tx in transactions if tx.get('type') != 'reward']
                    fees_total = sum(
                        float(tx.get("fee", 0) or 0)
                        for tx in non_reward_txs
                        if str(tx.get("type") or "").lower() == "transaction"
                    )
                    gtx_denom_total = 0.0
                    for tx in non_reward_txs:
                        if str(tx.get("type") or "").lower() in {"gtx_genesis", "genesis_bill"}:
                            denom = float(tx.get("amount", tx.get("denomination", 0)) or 0)
                            gtx_denom_total += self.difficulty_system.gtx_reward_units(denom)
                    tx_count = len(non_reward_txs)
                    expected_reward = self._calculate_expected_block_reward(
                        difficulty,
                        block_height,
                        tx_count,
                        fees_total,
                        gtx_denom_total,
                        False,
                    )
                    reward_tx["amount"] = expected_reward
                    reward_tx["block_height"] = block_height
                    reward_tx["difficulty"] = difficulty
                    reward_tx["timestamp"] = block_data.get("timestamp")
                    reward_tx["signature"] = reward_tx.get("signature") or "ling country"
                    reward_tx["public_key"] = reward_tx.get("public_key") or "ling country"
                    reward_tx["version"] = reward_tx.get("version") or "2.0"
                    tx_copy = {k: v for k, v in reward_tx.items() if k != "hash"}
                    reward_tx["hash"] = sm3_hex(json.dumps(tx_copy, sort_keys=True).encode())
                except Exception:
                    pass
                
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
        """Create mining reward transaction with daemon-compatible formatting."""
        difficulty = block_data.get('difficulty', 0) if block_data else 0
        non_reward_txs = [tx for tx in transactions if tx.get('type') != 'reward']
        is_empty_block = not non_reward_txs
        fees_total = sum(
            float(tx.get("fee", 0) or 0)
            for tx in non_reward_txs
            if str(tx.get("type") or "").lower() == "transaction"
        )
        gtx_denom_total = sum(
            float(tx.get("amount", tx.get("denomination", 0)) or 0)
            for tx in non_reward_txs
            if str(tx.get("type") or "").lower() in {"gtx_genesis", "genesis_bill"}
        )
        tx_count = len(non_reward_txs)
        block_ts = float(block_data.get('timestamp', time.time())) if block_data else time.time()

        total_reward = self._calculate_expected_block_reward(
            difficulty,
            block_height,
            tx_count,
            fees_total,
            gtx_denom_total,
            is_empty_block,
        )
        
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

            # Method 4: Compact mining format
            if not validation_passed:
                compact_hash = self._calculate_block_hash_compact(
                    block_height, previous_hash, timestamp, nonce, miner, difficulty
                )
                print("🔍 Compact format validation:")
                print(f"  Calculated: {compact_hash[:16]}...")
                if compact_hash == block_hash:
                    validation_passed = True
                    print("✅ Compact format validation passed")
            
            if not validation_passed:
                print("❌ All validation methods failed")
                raise ValueError("Invalid mining proof: Hash verification failed")
            
            print("✅ Mining proof validation successful!")
        
        if is_empty_block:
            return self._create_empty_block_reward(block_height, difficulty, total_reward, timestamp=block_ts)
        return self._create_block_reward(block_height, difficulty, total_reward, timestamp=block_ts)

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
        enforced = os.getenv("LUNALIB_MINER_ADDRESS")
        if enforced and user_address and user_address != enforced:
            safe_print("[WARN] Miner address override denied; using configured miner address.")
            user_address = enforced
            
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
        enforced = os.getenv("LUNALIB_MINER_ADDRESS")
        if enforced and miner_address and miner_address != enforced:
            safe_print("[WARN] Miner address override denied; using configured miner address.")
            miner_address = enforced
            
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
        enforced = os.getenv("LUNALIB_MINER_ADDRESS")
        if enforced and miner_address and miner_address != enforced:
            safe_print("[WARN] Miner address override denied; using configured miner address.")
            miner_address = enforced
            
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
        self.miner_address = getattr(config, "miner_address", "") or os.getenv("LUNALIB_MINER_ADDRESS", "")
        self.mining_started_callback = mining_started_callback
        self.mining_completed_callback = mining_completed_callback
        self.block_mined_callback = block_mined_callback
        self.block_added_callback = block_added_callback  # 新規: ブロックがチェーンに追加された時のコールバック
        self.mining_status_callbacks: List[Callable] = []

        # Mining performance defaults
        env_cpu_workers = os.getenv("LUNALIB_CPU_WORKERS")
        config_cpu_workers = getattr(config, "cpu_workers", None)
        if env_cpu_workers is not None:
            try:
                self.set_cpu_workers(int(env_cpu_workers))
            except Exception:
                self.set_cpu_workers(os.cpu_count() or 1)
        elif config_cpu_workers is not None:
            self.set_cpu_workers(config_cpu_workers)
        else:
            self.set_cpu_workers(os.cpu_count() or 1)

        env_cpu_max_nonce = os.getenv("LUNALIB_CPU_MAX_NONCE")
        config_cpu_max_nonce = getattr(config, "cpu_max_nonce", None)
        try:
            if env_cpu_max_nonce is not None:
                self.cpu_max_nonce = int(env_cpu_max_nonce)
            elif config_cpu_max_nonce is not None:
                self.cpu_max_nonce = int(config_cpu_max_nonce)
            else:
                self.cpu_max_nonce = 200_000_000
        except Exception:
            self.cpu_max_nonce = 200_000_000

        env_hash_mode = os.getenv("LUNALIB_MINING_HASH_MODE")
        config_hash_mode = getattr(config, "mining_hash_mode", None)
        self.mining_hash_mode = (env_hash_mode or config_hash_mode or "compact").lower()

        env_balance_mode = os.getenv("LUNALIB_LOAD_BALANCE")
        config_balance_mode = getattr(config, "load_balance_mode", None)
        self.load_balance_mode = (env_balance_mode or config_balance_mode or "prefer_gpu").lower()

        env_cpu_c_chunk = os.getenv("LUNALIB_CPU_C_CHUNK")
        try:
            self.cpu_c_chunk = int(env_cpu_c_chunk) if env_cpu_c_chunk else 200_000
        except Exception:
            self.cpu_c_chunk = 200_000

        self.mining_history = self._merge_mining_history(self.data_manager.load_mining_history())

        # Ensure hashrate_callback is always defined
        self.hashrate_callback = None
        self.cpu_hashrate_callbacks: List[Callable] = []
        self.gpu_hashrate_callbacks: List[Callable] = []

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
        self.peak_hashrate = 0.0
        self.mined_rewards: List[Dict] = []
        self.mined_bills: List[Dict] = []
        self.mined_blocks: List[Dict] = []
        self._mined_rewards_limit = int(os.getenv("LUNALIB_MINER_REWARD_CACHE_LIMIT", "2000"))
        self._mined_bills_limit = int(os.getenv("LUNALIB_MINER_BILL_CACHE_LIMIT", "500"))
        self._mined_blocks_limit = int(os.getenv("LUNALIB_MINER_BLOCK_CACHE_LIMIT", "500"))
        self.mined_block_callbacks: List[Callable] = []
        self.mined_bill_callbacks: List[Callable] = []
        self.loading_blocks_callbacks: List[Callable] = []
        self.loading_bills_callbacks: List[Callable] = []
        self._abort_event = threading.Event()
        self._load_cached_records()

    def _normalize_address(self, addr: str) -> str:
        if not addr:
            return ""
        addr_str = str(addr).strip("'\" ").lower()
        return addr_str[4:] if addr_str.startswith("lun_") else addr_str

    def _resolve_miner_address(self, requested: Optional[str] = None) -> str:
        configured = self.miner_address or ""
        if configured:
            if requested and self._normalize_address(requested) != self._normalize_address(configured):
                safe_print("[WARN] Miner address override denied; using configured miner address.")
            return configured
        return requested or ""

    def _cache_mined_reward(self, reward_tx: Dict) -> None:
        if not reward_tx:
            return
        target = reward_tx.get("to") or ""
        if self._normalize_address(target) != self._normalize_address(self.miner_address):
            return
        self.mined_rewards.append(reward_tx)
        if self._mined_rewards_limit > 0 and len(self.mined_rewards) > self._mined_rewards_limit:
            self.mined_rewards = self.mined_rewards[-self._mined_rewards_limit :]
        try:
            if hasattr(self.data_manager, "save_mined_rewards"):
                self.data_manager.save_mined_rewards(self.mined_rewards)
        except Exception:
            pass

    def _cache_mined_bill(self, bill_tx: Dict, persist: bool = True, emit: bool = True) -> None:
        if not bill_tx:
            return
        self.mined_bills.append(bill_tx)
        if self._mined_bills_limit > 0 and len(self.mined_bills) > self._mined_bills_limit:
            self.mined_bills = self.mined_bills[-self._mined_bills_limit :]
        if emit:
            self._emit_mined_bill(bill_tx)
        if persist:
            try:
                if hasattr(self.data_manager, "save_mined_bills"):
                    self.data_manager.save_mined_bills(self.mined_bills)
            except Exception:
                pass

    def _push_wallet_confirmed(self, txs: List[Dict]) -> None:
        if not txs:
            return
        try:
            from ..core.wallet_manager import get_wallet_manager
            mgr = get_wallet_manager()
            address = self.miner_address
            if not address:
                return
            mgr.register_wallets([address])
            mgr.sync_wallets_from_sources({address: txs}, {address: []})
        except Exception:
            pass

    def _handle_mined_block_records(self, block_data: Dict) -> None:
        if not isinstance(block_data, dict):
            return
        self._cache_mined_block(block_data)
        txs = block_data.get("transactions", []) or []
        reward_txs: List[Dict] = []
        for tx in txs:
            if not isinstance(tx, dict):
                continue
            if str(tx.get("type") or "").lower() != "reward":
                continue
            target = tx.get("to") or ""
            if self._normalize_address(target) == self._normalize_address(self.miner_address):
                reward_txs.append(tx)

        if not reward_txs and block_data.get("reward") and self.miner_address:
            reward_txs.append({
                "type": "reward",
                "from": "network",
                "to": self.miner_address,
                "amount": block_data.get("reward"),
                "timestamp": block_data.get("timestamp"),
                "block_height": block_data.get("index"),
                "difficulty": block_data.get("difficulty"),
                "hash": f"reward_{block_data.get('index')}_{str(block_data.get('hash', ''))[:8]}",
                "description": f"Mining reward for block #{block_data.get('index')}",
                "status": "confirmed",
                "direction": "incoming",
                "effective_amount": block_data.get("reward"),
                "fee": 0,
            })

        for reward_tx in reward_txs:
            self._cache_mined_reward(reward_tx)
        if reward_txs:
            self._push_wallet_confirmed(reward_txs)

        for tx in txs:
            if not isinstance(tx, dict):
                continue
            tx_type = str(tx.get("type") or "").lower()
            if tx_type in {"gtx_genesis", "genesis_bill"}:
                self._cache_mined_bill(tx)
                self._register_bill_from_tx(tx, block_data)

    def _cache_mined_block(self, block_data: Dict, persist: bool = True, emit: bool = True) -> None:
        if not isinstance(block_data, dict):
            return
        key = self._history_key(block_data)
        if key is None:
            return
        for entry in self.mined_blocks:
            if self._history_key(entry) == key:
                return
        self.mined_blocks.append(block_data)
        if self._mined_blocks_limit > 0 and len(self.mined_blocks) > self._mined_blocks_limit:
            self.mined_blocks = self.mined_blocks[-self._mined_blocks_limit :]
        if emit:
            self._emit_mined_block(block_data)
        if persist:
            try:
                if hasattr(self.data_manager, "record_mined_block"):
                    self.data_manager.record_mined_block(block_data)
            except Exception:
                pass

    def _emit_mined_block(self, block_data: Dict) -> None:
        for cb in self.mined_block_callbacks:
            try:
                cb(block_data)
            except Exception:
                pass

    def _emit_mined_bill(self, bill_tx: Dict) -> None:
        for cb in self.mined_bill_callbacks:
            try:
                cb(bill_tx)
            except Exception:
                pass

    def _emit_loading_blocks(self, current: int, total: int, item: Optional[Dict] = None) -> None:
        for cb in self.loading_blocks_callbacks:
            try:
                cb(current, total, item)
            except Exception:
                pass

    def _emit_loading_bills(self, current: int, total: int, item: Optional[Dict] = None) -> None:
        for cb in self.loading_bills_callbacks:
            try:
                cb(current, total, item)
            except Exception:
                pass

    def _load_cached_records(self) -> None:
        try:
            if hasattr(self.data_manager, "load_mined_blocks"):
                blocks = self.data_manager.load_mined_blocks() or []
                total = len(blocks)
                for idx, block in enumerate(blocks, start=1):
                    self._cache_mined_block(block, persist=False, emit=True)
                    self._emit_loading_blocks(idx, total, block)
        except Exception:
            pass

        try:
            if hasattr(self.data_manager, "load_mined_bills"):
                bills = self.data_manager.load_mined_bills() or []
                total = len(bills)
                for idx, bill in enumerate(bills, start=1):
                    self._cache_mined_bill(bill, persist=False, emit=True)
                    self._emit_loading_bills(idx, total, bill)
        except Exception:
            pass

    def _register_bill_from_tx(self, tx: Dict, block_data: Optional[Dict] = None) -> None:
        try:
            from ..gtx.bill_registry import BillRegistry
        except Exception:
            return
        if not isinstance(tx, dict):
            return
        tx_type = str(tx.get("type") or "").lower()
        if tx_type not in {"gtx_genesis", "genesis_bill"}:
            return

        bill_serial = tx.get("bill_serial") or tx.get("serial") or tx.get("bill_id")
        if not bill_serial:
            return
        denomination = tx.get("denomination") or tx.get("amount") or 0
        user_address = tx.get("user_address") or tx.get("to") or ""
        difficulty = tx.get("mining_difficulty") or (block_data or {}).get("difficulty", 0)
        mining_time = (block_data or {}).get("mining_time", 0.0)
        tx_hash = tx.get("hash") or ""
        timestamp = tx.get("timestamp") or (block_data or {}).get("timestamp") or time.time()
        bill_data = tx.get("bill_data") or {}

        bill_info = {
            "bill_serial": bill_serial,
            "denomination": float(denomination or 0),
            "user_address": user_address,
            "hash": tx_hash,
            "mining_time": float(mining_time or 0.0),
            "difficulty": int(difficulty or 0),
            "luna_value": float(denomination or 0),
            "timestamp": float(timestamp),
            "bill_data": bill_data,
        }

        try:
            BillRegistry().register_bill(bill_info)
        except Exception:
            pass

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

    def on_cpu_hashrate(self, callback: Callable) -> None:
        """Register a callback for CPU hashrate updates."""
        self.cpu_hashrate_callbacks.append(callback)

    def on_gpu_hashrate(self, callback: Callable) -> None:
        """Register a callback for GPU hashrate updates."""
        self.gpu_hashrate_callbacks.append(callback)

    def _emit_cpu_hashrate(self, rate: float) -> None:
        for cb in self.cpu_hashrate_callbacks:
            try:
                cb(rate)
            except Exception:
                pass

    def _emit_gpu_hashrate(self, rate: float) -> None:
        for cb in self.gpu_hashrate_callbacks:
            try:
                cb(rate)
            except Exception:
                pass

    def on_mined_block(self, callback: Callable) -> None:
        """Register a callback when a block is cached."""
        self.mined_block_callbacks.append(callback)

    def on_mined_bill(self, callback: Callable) -> None:
        """Register a callback when a GTX bill is cached."""
        self.mined_bill_callbacks.append(callback)

    def on_loading_mined_blocks(self, callback: Callable) -> None:
        """Register a callback for mined blocks loading progress."""
        self.loading_blocks_callbacks.append(callback)

    def on_loading_mined_bills(self, callback: Callable) -> None:
        """Register a callback for mined bills loading progress."""
        self.loading_bills_callbacks.append(callback)

    def _update_peak_hashrate(self, rate: float) -> None:
        try:
            if rate and rate > self.peak_hashrate:
                self.peak_hashrate = float(rate)
        except Exception:
            pass

    def _get_mempool_size(self) -> int:
        try:
            pending = self.mempool_manager.get_pending_transactions(fetch_remote=True)
            return len(pending)
        except Exception:
            try:
                return int(self.mempool_manager.get_mempool_size())
            except Exception:
                return 0

    def _emit_status(self, phase: str, message: str = "", payload: Optional[Dict] = None) -> None:
        data = {
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
            "engine": self.last_engine_used,
            "hash_rate": self.hash_rate or self.last_cpu_hashrate or self.last_gpu_hashrate,
            "mempool_size": self._get_mempool_size(),
            "peak_hashrate": self.peak_hashrate,
            "blocks_mined": self.blocks_mined,
            "total_reward": self.total_reward,
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

            non_reward_txs = [tx for tx in valid_transactions if tx.get('type') != 'reward']
            fees_total = sum(
                float(tx.get("fee", 0) or 0)
                for tx in non_reward_txs
                if str(tx.get("type") or "").lower() == "transaction"
            )
            gtx_denom_total = 0.0
            for tx in non_reward_txs:
                if str(tx.get("type") or "").lower() in {"gtx_genesis", "genesis_bill"}:
                    denom = float(tx.get("amount", tx.get("denomination", 0)) or 0)
                    gtx_denom_total += self.difficulty_system.gtx_reward_units(denom)
            tx_count = len(non_reward_txs)
            
            # Calculate block reward:
            # - Empty blocks use LINEAR system: difficulty 1 = 1 LKC, difficulty 2 = 2 LKC, etc.
            # - Blocks with transactions use EXPONENTIAL system: 10^(difficulty-1)
            is_empty_block = not valid_transactions
            total_reward = self._calculate_expected_block_reward(
                block_difficulty,
                new_index,
                tx_count,
                fees_total,
                gtx_denom_total,
                is_empty_block,
            )
            if is_empty_block:
                reward_tx = self._create_empty_block_reward(new_index, block_difficulty, total_reward)
                valid_transactions = [reward_tx]
            else:
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

            if cuda_available and cpu_available and self.load_balance_mode == "parallel":
                safe_print("[DEBUG] Mining engine selected: Hybrid (CPU+GPU)")
                self.last_engine_used = "hybrid"

                class _CombinedAbort:
                    def __init__(self, a, b):
                        self._a = a
                        self._b = b

                    def is_set(self):
                        return (self._a.is_set() if self._a else False) or (self._b.is_set() if self._b else False)

                local_abort = threading.Event()
                combined_abort = _CombinedAbort(self._abort_event, local_abort)

                block_cpu = copy.deepcopy(block_data)
                block_gpu = copy.deepcopy(block_data)

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fut_gpu = executor.submit(self._cuda_mine, block_gpu, block_difficulty, combined_abort)
                    fut_cpu = executor.submit(self._cpu_mine, block_cpu, block_difficulty, combined_abort)

                    winner = None
                    while True:
                        done, pending = concurrent.futures.wait(
                            [fut_gpu, fut_cpu],
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for f in done:
                            try:
                                res = f.result()
                            except Exception:
                                res = None
                            if res:
                                winner = res
                        if winner:
                            local_abort.set()
                            break
                        if fut_gpu.done() and fut_cpu.done():
                            break

                if winner:
                    engine = "cuda" if winner is block_gpu else "cpu"
                    method = self._format_mining_method(engine)
                    return self._finalize_block(winner, method, total_reward)

                safe_print("[FATAL] Hybrid mining failed (CPU+GPU)")
                self._emit_status("failure", "Hybrid mining failed")
                return False, "Hybrid mining failed", None

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
                    return self._finalize_block(cuda_result, self._format_mining_method('cuda'), total_reward)
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
                            return self._finalize_block(cpu_result, self._format_mining_method('cpu'), total_reward)
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
                    return self._finalize_block(cpu_result, self._format_mining_method('cpu'), total_reward)
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
                    is_valid, _ = self.transaction_validator.validate_transaction(tx)
                    if not is_valid:
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
        
        tx_type = str(tx.get('type') or '').lower()
        
        if tx_type in {'transaction', 'transfer'}:
            if not all(k in tx for k in ['from', 'to', 'amount']):
                return False
            # Reject self-transfers (from == to) as they fail server validation
            if tx.get('from') == tx.get('to'):
                safe_print(f"⚠️  Filtering out self-transfer: {tx.get('from')} → {tx.get('to')}")
                return False
        elif tx_type in {'genesis_bill', 'gtx_genesis'}:
            if 'denomination' not in tx and 'amount' not in tx:
                return False
            serial = tx.get('bill_serial') or tx.get('front_serial') or tx.get('serial') or tx.get('serial_number')
            if not serial:
                return False
            if 'hash' not in tx:
                return False
            to_addr = tx.get('to') or tx.get('issued_to') or tx.get('owner_address')
            if not to_addr:
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

    def _format_mining_method(self, engine: str) -> str:
        engine_lower = str(engine or "").lower()
        if engine_lower in {"cuda", "gpu"}:
            total = 0
            if self.cuda_manager and getattr(self.cuda_manager, "cuda_available", False):
                total = int(getattr(self.cuda_manager, "device_count", 0) or 0)
            used = total if getattr(self, "multi_gpu_enabled", False) and total > 0 else (1 if total != 0 else 1)
            if total <= 0:
                total = used
            return f"GPU[{used}/{total}]"
        if engine_lower == "cpu":
            try:
                total = int(os.cpu_count() or 1)
            except Exception:
                total = 1
            used = int(self.get_cpu_workers()) if hasattr(self, "get_cpu_workers") else total
            if used <= 0:
                used = total
            if total <= 0:
                total = used
            return f"CPU[{used}/{total}]"
        return str(engine)
    
    def _calculate_exponential_block_reward(
        self,
        difficulty: int,
        block_height: int | None = None,
        tx_count: int = 0,
        fees_total: float = 0.0,
        gtx_denom_total: float = 0.0,
    ) -> float:
        """Calculate block reward using exponential difficulty system
        
        Uses the new exponential reward system:
        difficulty 1 = 1 LKC
        difficulty 2 = 10 LKC  
        difficulty 3 = 100 LKC
        difficulty 9 = 100,000,000 LKC
        """
        return self.difficulty_system.calculate_block_reward(
            difficulty,
            block_height=block_height,
            tx_count=tx_count,
            fees_total=fees_total,
            gtx_denom_total=gtx_denom_total,
        )

    def _calculate_expected_block_reward(
        self,
        difficulty: int,
        block_height: int | None,
        tx_count: int,
        fees_total: float,
        gtx_denom_total: float,
        is_empty_block: bool,
    ) -> float:
        base_reward = None
        reward_mode = self._resolve_reward_mode()
        if is_empty_block or reward_mode == "linear":
            base_reward = float(difficulty or 0)
        
        reward = self.difficulty_system.calculate_block_reward(
            difficulty,
            block_height=block_height,
            tx_count=0 if is_empty_block else tx_count,
            fees_total=0.0 if is_empty_block else fees_total,
            gtx_denom_total=0.0 if is_empty_block else gtx_denom_total,
            base_reward=base_reward,
        )
        if is_empty_block:
            try:
                empty_mult = float(os.getenv("LUNALIB_EMPTY_BLOCK_MULT", "0.0001"))
            except Exception:
                empty_mult = 0.0001
            reward = max(0.0, reward * empty_mult)
        return reward

    def _create_empty_block_reward(self, block_index: int, difficulty: int, reward_amount: float, timestamp: Optional[float] = None) -> Dict:
        """Create reward transaction for empty blocks using LINEAR reward system (difficulty = reward)"""
        try:
            empty_mult = float(os.getenv("LUNALIB_EMPTY_BLOCK_MULT", "0.0001"))
        except Exception:
            empty_mult = 0.0001
        if timestamp is None:
            timestamp = time.time()
        tx = {
            'type': 'reward',
            'from': 'ling country',
            'to': self.config.miner_address,
            'amount': reward_amount,
            'fee': 0.0,
            'timestamp': timestamp,
            'block_height': block_index,
            'difficulty': difficulty,
            'signature': 'Ling Country',
            'public_key': 'Ling Country',
            'version': '2.0',
            'hash': '',
            'description': f'Empty block mining reward (Difficulty {difficulty} x {empty_mult} = {reward_amount} LKC)',
            'is_empty_block': True
        }
        tx['hash'] = sm3_hex(json.dumps({k: v for k, v in tx.items() if k != 'hash'}, sort_keys=True).encode())
        return tx

    def _create_block_reward(self, block_index: int, difficulty: int, reward_amount: float, timestamp: Optional[float] = None) -> Dict:
        """Create reward transaction for non-empty blocks using EXPONENTIAL reward system."""
        if reward_amount is None or reward_amount <= 0:
            reward_amount = self._calculate_expected_block_reward(
                difficulty,
                block_index,
                0,
                0.0,
                0.0,
                False,
            )
        if timestamp is None:
            timestamp = time.time()
        reward_mode = self._resolve_reward_mode()
        if reward_mode == "linear":
            tx = {
                'type': 'reward',
                'from': 'ling country',
                'to': self.config.miner_address,
                'amount': reward_amount,
                'fee': 0.0,
                'timestamp': timestamp,
                'block_height': block_index,
                'difficulty': difficulty,
                'signature': 'Ling Country',
                'public_key': 'Ling Country',
                'version': '2.0',
                'hash': '',
                'description': f'Block mining reward (Difficulty {difficulty} = {reward_amount} LKC)',
                'is_empty_block': False
            }
            tx['hash'] = sm3_hex(json.dumps({k: v for k, v in tx.items() if k != 'hash'}, sort_keys=True).encode())
            return tx
        else:
            tx = {
                'type': 'reward',
                'from': 'ling country',
                'to': self.config.miner_address,
                'amount': reward_amount,
                'fee': 0.0,
                'timestamp': timestamp,
                'block_height': block_index,
                'difficulty': difficulty,
                'signature': 'ling country',
                'public_key': 'ling country',
                'version': '2.0',
                'hash': '',
                'description': f'Block mining reward (Difficulty {difficulty} = {reward_amount} LKC)',
                'is_empty_block': False
            }
            tx['hash'] = sm3_hex(json.dumps({k: v for k, v in tx.items() if k != 'hash'}, sort_keys=True).encode())
            return tx

    def _cuda_mine(self, block_data: Dict, difficulty: int, stop_event: Optional[threading.Event] = None) -> Optional[Dict]:
        """Mine using CUDA acceleration"""
        safe_print("[DEBUG] TOP OF _cuda_mine: function entered")
        if not self.gpu_enabled:
            safe_print("[DEBUG] _cuda_mine called but gpu_enabled is False (early return)")
            return None
        if (stop_event and stop_event.is_set()) or self._abort_event.is_set() or self.should_stop_mining:
            safe_print("[DEBUG] _cuda_mine aborted before start")
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
            self._emit_gpu_hashrate(0.0)
            safe_print(f"[DEBUG] About to call cuda_manager.cuda_mine_batch with difficulty={difficulty}")
            status_interval = float(os.getenv("LUNALIB_MINER_STATUS_INTERVAL", "5"))
            batch_size = int(getattr(self.config, "cuda_batch_size", 1000000) or 1000000)
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
                self._update_peak_hashrate(self.hash_rate)
                self.last_gpu_attempts = int(update.get("attempts", 0))
                self.last_gpu_duration = float(update.get("duration", 0.0))
                if now - self._last_hashing_status >= status_interval:
                    self._last_hashing_status = now
                    rate = self.hash_rate
                    self._emit_status("hashing", f"{rate:,.0f} H/s")
                    safe_print(f"[DEBUG] [GPU] Hashrate update: {rate:,.0f} H/s, attempts={self.last_gpu_attempts}, duration={self.last_gpu_duration}")
                    if self.hashrate_callback:
                        self.hashrate_callback(rate, 'gpu')
                    self._emit_gpu_hashrate(rate)

            cuda_result = None
            try:
                if self.multi_gpu_enabled:
                    safe_print("[DEBUG] Multi-GPU mining enabled. Using cuda_mine_multi_gpu_batch.")
                    cuda_result = self.cuda_manager.cuda_mine_multi_gpu_batch(
                        block_data, difficulty, batch_size=batch_size, progress_callback=_progress, stop_event=(stop_event or self._abort_event)
                    )
                else:
                    cuda_result = self.cuda_manager.cuda_mine_batch(
                        block_data, difficulty, batch_size=batch_size, progress_callback=_progress, stop_event=(stop_event or self._abort_event)
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
                    self._update_peak_hashrate(self.hash_rate)
                    self.last_gpu_attempts = nonce
                    self.last_gpu_duration = mining_time
                    self.last_cpu_hashrate = 0.0
                    self.last_cpu_attempts = 0
                    self.last_cpu_duration = 0.0
                    safe_print(f"[DEBUG] [GPU] Final hashrate: {self.hash_rate:,.0f} H/s, nonce={nonce}, mining_time={mining_time}")
                    if self.hashrate_callback:
                        self.hashrate_callback(self.hash_rate, 'gpu')
                self._emit_gpu_hashrate(self.hash_rate)
                self.current_nonce = nonce
                self.current_hash = cuda_result.get('hash', '')
                block_data['hash'] = cuda_result['hash']
                block_data['nonce'] = cuda_result['nonce']
                return block_data
        except Exception as e:
            safe_print(f"[DEBUG] Exception in _cuda_mine: {e}")
        return None

    def _cpu_mine(self, block_data: Dict, difficulty: int, stop_event: Optional[threading.Event] = None) -> Optional[Dict]:
        """Mine using CPU"""
        import concurrent.futures
        if not self.cpu_enabled:
            safe_print("[DEBUG] _cpu_mine called but cpu_enabled is False")
            return None
        if (stop_event and stop_event.is_set()) or self._abort_event.is_set() or self.should_stop_mining:
            safe_print("[DEBUG] _cpu_mine aborted before start")
            return None
        safe_print("[DEBUG] Entering _cpu_mine: Using CPU mining (multi-threaded)...")
        if self.hashrate_callback:
            self.hashrate_callback(0.0, 'cpu')
        self._emit_cpu_hashrate(0.0)
        start_time = time.time()
        target = "0" * difficulty
        num_workers = self.get_cpu_workers()
        max_nonce = getattr(self, "cpu_max_nonce", 1_000_000)
        found = threading.Event()
        result = [None]

        pinning_enabled = bool(int(os.getenv("LUNALIB_CPU_PINNING", "0")))
        pin_list = _parse_cpu_list(os.getenv("LUNALIB_CPU_PIN_LIST"))
        if bool(int(os.getenv("LUNALIB_NUMA", "0"))):
            try:
                node_id = int(os.getenv("LUNALIB_NUMA_NODE", "0"))
            except Exception:
                node_id = 0
            pin_list = _get_numa_cpulist(node_id) or pin_list
        if not pin_list:
            pin_list = list(range(os.cpu_count() or 1))

        hash_mode = (os.getenv("LUNALIB_MINING_HASH_MODE") or getattr(self, "mining_hash_mode", "json")).lower()
        base80 = None
        if hash_mode == "compact":
            base80 = self._build_compact_base80(
                block_data['index'],
                block_data['previous_hash'],
                block_data['timestamp'],
                block_data['miner'],
                difficulty,
            )

        use_c_mine = base80 is not None and callable(sm3_mine_compact)
        config_c_threads = getattr(self.config, "cpu_c_threads", None)
        env_c_threads = os.getenv("LUNALIB_CPU_C_THREADS")
        try:
            if config_c_threads is not None:
                c_threads = int(config_c_threads)
            elif env_c_threads is not None:
                c_threads = int(env_c_threads)
            else:
                c_threads = 1
        except Exception:
            c_threads = 1
        if c_threads < 1:
            c_threads = 1

        def worker(start_nonce, step):
            if pinning_enabled and pin_list:
                core = pin_list[start_nonce % len(pin_list)]
                _pin_current_thread(core)
            if use_c_mine:
                chunk = getattr(self, "cpu_c_chunk", 200_000)
                if chunk < 1:
                    chunk = 200_000
                nonce = start_nonce * chunk
                while not found.is_set() and nonce < max_nonce and not self._abort_event.is_set() and not self.should_stop_mining and not (stop_event and stop_event.is_set()):
                    count = min(chunk, max_nonce - nonce)
                    t0 = time.time()
                    if c_threads > 1:
                        found_nonce = sm3_mine_compact(base80, nonce, count, difficulty, c_threads)
                    else:
                        found_nonce = sm3_mine_compact(base80, nonce, count, difficulty)
                    elapsed = time.time() - t0
                    if elapsed > 0:
                        rate = count / elapsed
                        safe_print(f"[DEBUG] [CPU] Worker {start_nonce}: {rate:,.0f} H/s at nonce {nonce}")
                        if self.hashrate_callback:
                            try:
                                self.hashrate_callback(rate, 'cpu')
                            except Exception:
                                pass
                        self._emit_cpu_hashrate(rate)
                    if found_nonce is not None:
                        block_hash = self._calculate_block_hash(
                            block_data['index'],
                            block_data['previous_hash'],
                            block_data['timestamp'],
                            block_data['transactions'],
                            found_nonce,
                            block_data['miner'],
                            difficulty
                        )
                        elapsed_total = time.time() - start_time
                        safe_print(f"[DEBUG] [CPU] Block found: nonce={found_nonce}, hash={block_hash}, elapsed={elapsed_total:.2f}s")
                        block_data['hash'] = block_hash
                        block_data['nonce'] = int(found_nonce)
                        result[0] = block_data.copy()
                        found.set()
                        return
                    nonce += chunk * step
                return

            nonce = start_nonce
            hash_count = 0
            last_hash_update = time.time()
            while not found.is_set() and nonce < max_nonce and not self._abort_event.is_set() and not self.should_stop_mining and not (stop_event and stop_event.is_set()):
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
                    if self.hashrate_callback:
                        try:
                            self.hashrate_callback(rate, 'cpu')
                        except Exception:
                            pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i, num_workers) for i in range(num_workers)]
            concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            found.set()
        elapsed = time.time() - start_time
        if result[0]:
            self.last_cpu_attempts = result[0]['nonce']
            self.last_cpu_duration = elapsed
            self.last_cpu_hashrate = result[0]['nonce'] / elapsed if elapsed > 0 else 0.0
            self._update_peak_hashrate(self.last_cpu_hashrate)
            safe_print(f"[DEBUG] [CPU] Final hashrate: {self.last_cpu_hashrate:,.0f} H/s, nonce={result[0]['nonce']}, elapsed={elapsed:.2f}s")
            if self.hashrate_callback:
                self.hashrate_callback(self.last_cpu_hashrate, 'cpu')
            self._emit_cpu_hashrate(self.last_cpu_hashrate)
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
            env_hash_mode = os.getenv("LUNALIB_MINING_HASH_MODE")
            hash_mode = (env_hash_mode or getattr(self, "mining_hash_mode", "json")).lower()
            if hash_mode == "compact":
                return self._calculate_block_hash_compact(index, previous_hash, timestamp, nonce, miner, difficulty)
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

    def _calculate_block_hash_compact(self, index: int, previous_hash: str, timestamp: float,
                                      nonce: int, miner: str, difficulty: int) -> str:
        """Compact mining hash: fixed 88-byte header (80-byte base + 8-byte nonce)."""
        try:
            base = self._build_compact_base80(index, previous_hash, timestamp, miner, difficulty)
            if not base:
                return "0" * 64
            try:
                return sm3_compact_hash(base, int(nonce)).hex()
            except Exception:
                pass
            nonce_bytes = int(nonce).to_bytes(8, "big", signed=False)
            return sm3_digest(base + nonce_bytes).hex()
        except Exception:
            return "0" * 64

    def _build_compact_base80(self, index: int, previous_hash: str, timestamp: float,
                              miner: str, difficulty: int) -> Optional[bytes]:
        if len(previous_hash) != 64:
            return None
        try:
            prev_bytes = bytes.fromhex(previous_hash)
        except Exception:
            return None
        miner_hash = sm3_digest(str(miner).encode())
        base = (
            prev_bytes
            + int(index).to_bytes(4, "big", signed=False)
            + int(difficulty).to_bytes(4, "big", signed=False)
            + struct.pack(">d", float(timestamp))
            + miner_hash
        )
        return base if len(base) == 80 else None

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
            if os.getenv("LUNALIB_SKIP_SUBMIT", "0") == "1":
                safe_print("[INFO] LUNALIB_SKIP_SUBMIT=1, skipping block submission")
                submission_success = True
            else:
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

        # 新規: ブロックがチェーンに追加されたことを通知
        self._safe_callback(self.block_added_callback, "block_added_callback", block_data)

        self.blocks_mined += 1
        self.total_reward += final_reward

        if self.mining_completed_callback:
            self.mining_completed_callback(True, f"Block #{block_data['index']} mined - Reward: {final_reward}")

        self._safe_callback(self.block_mined_callback, "block_mined_callback", block_data)

        self.save_mining_history()

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

        # Cache rewards/bills and push rewards into wallet state
        self._handle_mined_block_records(block_data)

        return True, f"Block #{block_data['index']} mined - Reward: {final_reward}", block_data
    
    def _validate_mined_block(self, block: Dict) -> tuple:
        """Validate mined block before submission
        
        Returns: (is_valid, error_message)
        """
        try:
            from lunalib.core.daemon import BlockchainDaemon
            daemon = BlockchainDaemon(
                self.blockchain_manager,
                self.mempool_manager,
                security_manager=self.security_manager,
                max_workers=1,
            )
            result = daemon.validate_block(block)
            if result.get("valid"):
                return True, ""
            message = result.get("message", "Block validation failed")
            errors = result.get("errors") or []
            if errors:
                message = f"{message}: {errors}"
            return False, message
        except Exception as e:
            return False, f"Daemon-style validation failed: {e}"
        
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
        tx_count = len(non_reward_txs)
        fees_total = sum(
            float(tx.get("fee", 0) or 0)
            for tx in non_reward_txs
            if str(tx.get("type") or "").lower() == "transaction"
        )
        gtx_denom_total = 0.0
        for tx in non_reward_txs:
            if str(tx.get("type") or "").lower() in {"gtx_genesis", "genesis_bill"}:
                denom = float(tx.get("amount", tx.get("denomination", 0)) or 0)
                gtx_denom_total += self.difficulty_system.gtx_reward_units(denom)
        
        expected_reward = self._calculate_expected_block_reward(
            difficulty,
            block.get("index"),
            tx_count,
            fees_total,
            gtx_denom_total,
            is_empty_block,
        )
        
        reward_tx_amount = reward_tx.get('amount', 0) if reward_tx else None
        actual_reward = block.get('reward', reward_tx_amount if reward_tx_amount is not None else 0)
        
        # Allow some tolerance for floating point comparison
        if abs(actual_reward - expected_reward) > 0.01:
            reward_type = "Linear" if is_empty_block or self._resolve_reward_mode() == "linear" else "Exponential"
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

    def _safe_callback(self, callback: Optional[Callable], name: str, *args, **kwargs) -> bool:
        """Safely invoke callback and return success status."""
        if not callback:
            return False
        try:
            callback(*args, **kwargs)
            return True
        except Exception as e:
            safe_print(f"[WARN] {name} error: {e}")
            return False

    def _history_key(self, entry: Dict) -> Optional[tuple]:
        """Build a unique key for mining history entries."""
        if not isinstance(entry, dict):
            return None
        block_index = entry.get("block_index", entry.get("index"))
        block_hash = entry.get("hash")
        if block_index is None or not block_hash:
            return None
        return (block_index, block_hash)

    def _merge_mining_history(self, *histories: List[Dict]) -> List[Dict]:
        """Merge histories and deduplicate by block_index + hash."""
        merged: Dict[tuple, Dict] = {}
        for history in histories:
            if not history:
                continue
            for entry in history:
                key = self._history_key(entry)
                if key is None:
                    continue
                existing = merged.get(key)
                if not existing:
                    merged[key] = entry
                    continue
                existing_ts = float(existing.get("timestamp", 0) or 0)
                entry_ts = float(entry.get("timestamp", 0) or 0)
                if entry_ts >= existing_ts:
                    merged[key] = entry

        return sorted(merged.values(), key=lambda e: float(e.get("timestamp", 0) or 0))

    def save_mining_history(self):
        """Save mining history to storage"""
        try:
            persisted = []
            if hasattr(self.data_manager, "load_mining_history"):
                persisted = self.data_manager.load_mining_history()
            self.mining_history = self._merge_mining_history(persisted, self.mining_history)
            self.data_manager.save_mining_history(self.mining_history)
        except Exception as e:
            safe_print(f"[WARN] save_mining_history failed: {e}")

    def get_mining_history(self) -> List[Dict]:
        """Return merged mining history (deduplicated)."""
        try:
            persisted = []
            if hasattr(self.data_manager, "load_mining_history"):
                persisted = self.data_manager.load_mining_history()
            self.mining_history = self._merge_mining_history(persisted, self.mining_history)
        except Exception as e:
            safe_print(f"[WARN] get_mining_history failed: {e}")
        return list(self.mining_history)

    def start_mining(self):
        """Start the mining process"""
        if self.is_mining:
            return

        self.is_mining = True
        self.should_stop_mining = False
        self._abort_event.clear()
        sm3_set_abort(False)
        if self.cuda_manager and hasattr(self.cuda_manager, "reset_abort"):
            try:
                self.cuda_manager.reset_abort()
            except Exception:
                pass

        if self.mining_started_callback:
            self.mining_started_callback()

        if not self.mining_thread or not self.mining_thread.is_alive():
            self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
            self.mining_thread.start()

    def _mining_loop(self):
        """Background mining loop that updates hashrate telemetry."""
        status_interval = float(os.getenv("LUNALIB_MINER_STATUS_INTERVAL", "5"))
        last_status = time.time()

        while self.is_mining and not self.should_stop_mining and not self._abort_event.is_set():
            success, message, block = self.mine_block()

            if success:
                pass
            else:
                # Avoid tight loop if mining is disabled/unavailable
                time.sleep(0.25)

            now = time.time()
            if now - last_status >= status_interval:
                last_status = now
                engine = self.last_engine_used or "none"
                rate = self.hash_rate or self.last_cpu_hashrate
                self._update_peak_hashrate(rate)
                mempool_size = self._get_mempool_size()
                safe_print(f"[MINER] engine={engine} rate={rate:,.0f} H/s nonce={self.current_nonce}")
                safe_print(
                    "[STATS] "
                    f"mempool={mempool_size} "
                    f"peak={self.peak_hashrate:,.0f} H/s "
                    f"blocks={self.blocks_mined} "
                    f"reward={self.total_reward:,.6f}"
                )

    def stop_mining(self):
        """Stop the mining process"""
        self.is_mining = False
        self.should_stop_mining = True
        self._abort_event.set()
        sm3_set_abort(True)
        if self.cuda_manager and hasattr(self.cuda_manager, "abort"):
            try:
                self.cuda_manager.abort()
            except Exception:
                pass
        if self.mining_thread and self.mining_thread.is_alive():
            self.mining_thread.join()

    def abort_mining_now(self) -> None:
        """Abort mining immediately (best-effort for CPU/CUDA)."""
        self.is_mining = False
        self.should_stop_mining = True
        self.mining_active = False
        self._abort_event.set()
        sm3_set_abort(True)
        if self.cuda_manager and hasattr(self.cuda_manager, "abort"):
            try:
                self.cuda_manager.abort()
            except Exception:
                pass

    def get_mining_stats(self):
        """Return the current mining statistics"""
        return {
            "blocks_mined": self.blocks_mined,
            "total_reward": self.total_reward,
            "current_hash": self.current_hash,
            "current_nonce": self.current_nonce,
            "hash_rate": self.hash_rate,
            "peak_hashrate": self.peak_hashrate,
            "mempool_size": self._get_mempool_size(),
            "mining_history": len(self.mining_history),
            "miner_address": self.miner_address,
            "mined_rewards": len(self.mined_rewards),
            "mined_bills": len(self.mined_bills),
            "last_engine_used": self.last_engine_used,
            "last_cpu_hashrate": self.last_cpu_hashrate,
            "last_cpu_attempts": self.last_cpu_attempts,
            "last_cpu_duration": self.last_cpu_duration
            ,
            "last_gpu_hashrate": self.last_gpu_hashrate,
            "last_gpu_attempts": self.last_gpu_attempts,
            "last_gpu_duration": self.last_gpu_duration
        }