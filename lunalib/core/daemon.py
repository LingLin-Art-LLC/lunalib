# lunalib/core/daemon.py
import time
import threading
import os
import gzip
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable
from lunalib.utils.validation import is_valid_address
from lunalib.core.sm3 import sm3_hex, sm3_compact_hash, sm3_digest
import json
from datetime import datetime

import requests

try:
    import msgpack  # type: ignore
    _HAS_MSGPACK = True
except Exception:
    msgpack = None
    _HAS_MSGPACK = False


class BlockchainDaemon:
    """
    Primary blockchain daemon that manages the authoritative blockchain state.
    Validates all transactions, manages peer registry, and serves as source of truth.
    """
    
    def __init__(self, blockchain_manager, mempool_manager, security_manager=None, max_workers=5):
        self.blockchain = blockchain_manager
        self.mempool = mempool_manager
        self.security = security_manager
        self.batch_workers = max_workers
        
        # Initialize difficulty system for validation
        from ..mining.difficulty import DifficultySystem
        self.difficulty_system = DifficultySystem()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DaemonWorker")
        self._async_tasks = {}  # Track async tasks
        self._peer_session = requests.Session()
        self.use_msgpack = bool(int(os.getenv("LUNALIB_USE_MSGPACK", "0"))) and _HAS_MSGPACK
        self.p2p_gzip = bool(int(os.getenv("LUNALIB_P2P_GZIP", "1")))
        self.batch_workers = max_workers
        self._seen_tx_max = int(os.getenv("LUNALIB_SEEN_TX_MAX", "50000"))
        self._seen_tx_hashes = set()
        self._seen_tx_order = []
        self._trusted_peers = set(
            p.strip() for p in os.getenv("LUNALIB_TRUSTED_PEERS", "").split(",") if p.strip()
        )
        
        # Peer registry
        self.peers = {}  # {node_id: peer_info}
        self.peer_lock = threading.Lock()
        
        # Daemon state
        self.is_running = False
        self.validation_thread = None
        self.cleanup_thread = None
        
        # Statistics
        self.stats = {
            'blocks_validated': 0,
            'transactions_validated': 0,
            'peers_registered': 0,
            'start_time': time.time(),
            'tx_validation_seconds': 0.0,
            'tx_validation_count': 0
        }

    def _remember_seen_tx(self, tx_hash: str) -> bool:
        if not tx_hash:
            return False
        if tx_hash in self._seen_tx_hashes:
            return False
        self._seen_tx_hashes.add(tx_hash)
        self._seen_tx_order.append(tx_hash)
        if len(self._seen_tx_order) > self._seen_tx_max:
            old = self._seen_tx_order.pop(0)
            self._seen_tx_hashes.discard(old)
        return True

    def _is_trusted_peer(self, peer_id: Optional[str]) -> bool:
        if not peer_id:
            return False
        return peer_id in self._trusted_peers

    def _parse_amount(self, value, default: float = 0.0) -> float:
        """Parse numeric amounts that may include unit suffixes like '3.0JS:3'."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                return default
        try:
            return float(value)
        except Exception:
            text = str(value)
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match:
                try:
                    return float(match.group(0))
                except Exception:
                    return default
        return default

    def _is_tx_in_recent_chain(self, tx_hash: str) -> bool:
        """Check if a transaction hash already exists in recent blockchain history."""
        if not tx_hash:
            return False
        try:
            depth = int(os.getenv("LUNALIB_TX_DEDUP_DEPTH", "1000"))
        except Exception:
            depth = 1000
        if depth <= 0:
            return False
        try:
            current_height = self.blockchain.get_blockchain_height()
        except Exception:
            return False
        start_height = max(0, current_height - depth + 1)
        for height in range(current_height, start_height - 1, -1):
            try:
                block = self.blockchain.get_block(height)
            except Exception:
                block = None
            if not block:
                continue
            try:
                if hasattr(self.blockchain, "_extract_block_transactions"):
                    txs = self.blockchain._extract_block_transactions(block)
                else:
                    txs = block.get("transactions", [])
            except Exception:
                txs = block.get("transactions", []) if isinstance(block, dict) else []
            for tx in txs or []:
                if not isinstance(tx, dict):
                    continue
                if tx.get("hash") == tx_hash:
                    return True
        return False

    def _is_gtx_serial_in_recent_chain(self, bill_serial: str) -> bool:
        """Check if a GTX bill serial already exists in recent blockchain history."""
        if not bill_serial:
            return False
        try:
            depth = int(os.getenv("LUNALIB_GTX_DEDUP_DEPTH", "2000"))
        except Exception:
            depth = 2000
        if depth <= 0:
            return False
        try:
            current_height = self.blockchain.get_blockchain_height()
        except Exception:
            return False
        start_height = max(0, current_height - depth + 1)
        for height in range(current_height, start_height - 1, -1):
            try:
                block = self.blockchain.get_block(height)
            except Exception:
                block = None
            if not block:
                continue
            try:
                if hasattr(self.blockchain, "_extract_block_transactions"):
                    txs = self.blockchain._extract_block_transactions(block)
                else:
                    txs = block.get("transactions", [])
            except Exception:
                txs = block.get("transactions", []) if isinstance(block, dict) else []
            for tx in txs or []:
                if not isinstance(tx, dict):
                    continue
                tx_type = str(tx.get("type") or "").lower()
                if tx_type not in {"gtx_genesis", "genesis_bill"}:
                    continue
                existing = (
                    tx.get("bill_serial")
                    or tx.get("front_serial")
                    or tx.get("serial")
                    or tx.get("serial_number")
                )
                if existing == bill_serial:
                    return True
        return False
    
    def start(self):
        """Start the blockchain daemon"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🏛️  Blockchain Daemon starting...")
        
        # Start background threads
        self.validation_thread = threading.Thread(target=self._validation_loop, daemon=True)
        self.validation_thread.start()
        
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        print("✅ Blockchain Daemon started")
    
    def stop(self):
        """Stop the blockchain daemon"""
        self.is_running = False
        
        if self.validation_thread:
            self.validation_thread.join(timeout=5)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Shutdown executor
        print("🛑 Shutting down daemon executor...")
        self.executor.shutdown(wait=True)
        
        print("🛑 Blockchain Daemon stopped")
    
    def register_peer(self, peer_info: Dict) -> Dict:
        """
        Register a new peer node.
        Returns: {'success': bool, 'node_id': str, 'message': str}
        """
        try:
            node_id = peer_info.get('node_id')
            if not node_id:
                return {'success': False, 'message': 'Missing node_id'}
            
            with self.peer_lock:
                self.peers[node_id] = {
                    'node_id': node_id,
                    'registered_at': time.time(),
                    'last_seen': time.time(),
                    'capabilities': peer_info.get('capabilities', []),
                    'url': peer_info.get('url'),
                    'version': peer_info.get('version', 'unknown')
                }
                
                self.stats['peers_registered'] += 1
            
            print(f"👤 New peer registered: {node_id}")
            return {
                'success': True,
                'node_id': node_id,
                'message': 'Peer registered successfully'
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def unregister_peer(self, node_id: str) -> Dict:
        """
        Unregister a peer node.
        Returns: {'success': bool, 'message': str}
        """
        try:
            with self.peer_lock:
                if node_id in self.peers:
                    del self.peers[node_id]
                    print(f"👋 Peer unregistered: {node_id}")
                    return {'success': True, 'message': 'Peer unregistered'}
                else:
                    return {'success': False, 'message': 'Peer not found'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def get_peer_list(self, exclude_node_id: Optional[str] = None) -> List[Dict]:
        """
        Get list of registered peers.
        exclude_node_id: Optional node ID to exclude from results
        """
        with self.peer_lock:
            peer_list = []
            for node_id, peer_info in self.peers.items():
                if exclude_node_id and node_id == exclude_node_id:
                    continue
                
                peer_list.append({
                    'node_id': peer_info['node_id'],
                    'url': peer_info.get('url'),
                    'last_seen': peer_info['last_seen'],
                    'capabilities': peer_info.get('capabilities', [])
                })
            
            return peer_list
    
    def update_peer_heartbeat(self, node_id: str):
        """Update peer's last seen timestamp"""
        with self.peer_lock:
            if node_id in self.peers:
                self.peers[node_id]['last_seen'] = time.time()
    
    def validate_block_async(self, block: Dict, callback: Callable = None) -> str:
        """Async version: Validate block in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"validate_block_{block.get('index', 'unknown')}_{int(time.time())}"
        
        def _validate_task():
            try:
                result = self.validate_block(block)
                if callback:
                    callback(success=result['valid'], result=result, error=None if result['valid'] else result['message'])
                return result
            except Exception as e:
                print(f"❌ Async validation error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return {'valid': False, 'message': str(e), 'errors': [str(e)]}
        
        future = self.executor.submit(_validate_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async block validation: {task_id}")
        return task_id
    
    def validate_block(self, block: Dict) -> Dict:
        """
        Validate a block submitted by a peer or miner.
        Returns: {'valid': bool, 'message': str, 'errors': List[str]}
        """
        errors = []
        
        try:
            # Basic structure validation using difficulty system
            is_valid, error_msg = self.difficulty_system.validate_block_structure(block)
            if not is_valid:
                errors.append(error_msg)
                return {'valid': False, 'message': 'Invalid block structure', 'errors': errors}
            
            # Validate block index
            latest_block = self.blockchain.get_latest_block()
            if latest_block:
                expected_index = latest_block.get('index', 0) + 1
                if block['index'] != expected_index:
                    errors.append(f"Invalid block index: expected {expected_index}, got {block['index']}")
            
            # Validate previous hash
            if latest_block and block['previous_hash'] != latest_block.get('hash'):
                expected_hash = latest_block.get('hash', '')
                actual_hash = block.get('previous_hash', '')
                errors.append(f"Previous hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}...")
            
            # Validate hash meets difficulty requirement using difficulty system
            difficulty = block.get('difficulty', 0)
            block_hash = block.get('hash', '')
            
            if not self.difficulty_system.validate_block_hash(block_hash, difficulty):
                errors.append(f"Hash doesn't meet difficulty {difficulty} requirement (needs {difficulty} leading zeros)")

            # Validate PoW hash using SM3 (miner/daemon canonical format)
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
                if pow_hash != block_hash:
                    compact_ok = False
                    try:
                        prev_hash = str(block.get("previous_hash", ""))
                        if len(prev_hash) == 64:
                            prev_bytes = bytes.fromhex(prev_hash)
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
                                compact_ok = compact_hash == block_hash
                    except Exception:
                        compact_ok = False
                    if not compact_ok:
                        errors.append(f"SM3 hash mismatch: expected {pow_hash[:16]}..., got {str(block_hash)[:16]}...")
            except Exception as e:
                errors.append(f"SM3 hash validation error: {e}")
            
            # Validate reward matches difficulty/height/base reward
            transactions = block.get('transactions', [])
            reward_tx = next((tx for tx in transactions if tx.get('type') == 'reward'), None)
            non_reward_txs = [tx for tx in transactions if tx.get('type') != 'reward']
            is_empty_block = not non_reward_txs or (len(transactions) == 1 and reward_tx and reward_tx.get('is_empty_block', False))
            tx_count = len(non_reward_txs)

            # Reject duplicate transactions (within block or already confirmed)
            seen_hashes: set[str] = set()
            seen_gtx_serials: set[str] = set()
            for tx in non_reward_txs:
                if not isinstance(tx, dict):
                    continue
                tx_hash = tx.get("hash")
                if not tx_hash:
                    errors.append("Transaction missing hash")
                    continue
                if tx_hash in seen_hashes:
                    errors.append(f"Duplicate transaction hash in block: {tx_hash[:16]}...")
                    continue
                seen_hashes.add(tx_hash)
                if hasattr(self.mempool, "is_transaction_confirmed") and self.mempool.is_transaction_confirmed(tx_hash):
                    errors.append(f"Transaction already confirmed: {tx_hash[:16]}...")
                    continue
                if self._is_tx_in_recent_chain(tx_hash):
                    errors.append(f"Transaction already in blockchain: {tx_hash[:16]}...")
                    continue

                tx_type = str(tx.get("type") or "").lower()
                if tx_type in {"gtx_genesis", "genesis_bill"}:
                    bill_serial = (
                        tx.get("bill_serial")
                        or tx.get("front_serial")
                        or tx.get("serial")
                        or tx.get("serial_number")
                    )
                    if not bill_serial:
                        errors.append("GTX bill missing serial")
                        continue
                    if bill_serial in seen_gtx_serials:
                        errors.append(f"Duplicate GTX bill serial in block: {bill_serial}")
                        continue
                    seen_gtx_serials.add(bill_serial)
                    if self._is_gtx_serial_in_recent_chain(bill_serial):
                        errors.append(f"GTX bill already in blockchain: {bill_serial}")
                        continue

            fees_total = 0.0
            for tx in non_reward_txs:
                try:
                    fee_value = tx.get('fee', None)
                    if fee_value is None:
                        fee_value = tx.get('gas', 0)
                    fees_total += self._parse_amount(fee_value or 0)
                except Exception:
                    continue
            gtx_denom_total = 0.0
            for tx in non_reward_txs:
                try:
                    if str(tx.get("type") or "").lower() in {"gtx_genesis", "genesis_bill"}:
                        denom = self._parse_amount(tx.get("amount", tx.get("denomination", 0)) or 0)
                        gtx_denom_total += self.difficulty_system.gtx_reward_units(denom)
                except Exception:
                    continue
            
            reward_mode = os.getenv("LUNALIB_BLOCK_REWARD_MODE", "exponential").lower().strip()
            base_reward = None
            if is_empty_block or reward_mode == "linear":
                base_reward = float(difficulty or 0)
                reward_type = "Linear"
            else:
                reward_type = "Exponential"

            reward_block_height = block.get("index")
            if reward_tx and reward_tx.get("block_height") is not None:
                reward_block_height = reward_tx.get("block_height")

            expected_reward = self.difficulty_system.calculate_block_reward(
                difficulty,
                block_height=reward_block_height,
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
                expected_reward = max(0.0, expected_reward * empty_mult)
            
            if reward_tx:
                required_reward_fields = [
                    "type",
                    "from",
                    "to",
                    "amount",
                    "fee",
                    "timestamp",
                    "block_height",
                    "difficulty",
                    "signature",
                    "public_key",
                    "version",
                    "hash",
                ]
                missing = [f for f in required_reward_fields if f not in reward_tx]
                if missing:
                    errors.append(f"Reward transaction missing fields: {missing}")

                if reward_tx.get("block_height") != block.get("index"):
                    errors.append("Reward transaction block_height mismatch")
                if reward_tx.get("difficulty") != difficulty:
                    errors.append("Reward transaction difficulty mismatch")

                try:
                    if abs(float(reward_tx.get("timestamp", 0)) - float(block.get("timestamp", 0))) > 5:
                        errors.append("Reward timestamp does not match block timestamp")
                except Exception:
                    errors.append("Invalid reward timestamp")

                try:
                    tx_copy = {k: v for k, v in reward_tx.items() if k != "hash"}
                    expected_hash = sm3_hex(json.dumps(tx_copy, sort_keys=True).encode())
                    if str(reward_tx.get("hash")) != expected_hash:
                        errors.append("Reward transaction hash mismatch")
                except Exception as e:
                    errors.append(f"Reward hash validation error: {e}")

            reward_tx_amount = self._parse_amount(reward_tx.get('amount', 0)) if reward_tx else None
            actual_reward = self._parse_amount(block.get('reward', reward_tx_amount if reward_tx_amount is not None else 0))
            if reward_tx_amount is not None and block.get('reward') is not None and abs(self._parse_amount(block.get('reward', 0)) - self._parse_amount(reward_tx_amount)) > 0.01:
                errors.append("Reward transaction amount does not match block reward")
            
            # Allow small tolerance for floating point comparison
            if abs(actual_reward - expected_reward) > 0.01:
                errors.append(f"Reward mismatch ({reward_type}): expected {expected_reward} LKC for difficulty {difficulty}, got {actual_reward} LKC")

            if reward_tx:
                allowed_senders = {
                    "network",
                    "block_reward",
                    "mining_reward",
                    "coinbase",
                    "ling country",
                    "ling country mines",
                    "foreign exchange",
                }
                sender = str(reward_tx.get("from") or "").lower().strip()
                sig = str(reward_tx.get("signature") or "").lower().strip()
                pub = str(reward_tx.get("public_key") or "").lower().strip()
                if sender and sender not in allowed_senders:
                    errors.append(f"Invalid reward sender: {reward_tx.get('from')}. Must be one of {sorted(allowed_senders)}")
                if sig and sig not in allowed_senders:
                    errors.append("Invalid reward signature")
                if pub and pub not in allowed_senders:
                    errors.append("Invalid reward public key")
            
            # Validate transactions
            txs = block.get('transactions', [])
            if txs:
                batch = self.validate_transactions_batch(txs)
                for res in batch.get("results", []):
                    if not res.get("valid"):
                        errors.append(f"Invalid transaction: {res.get('message')}")
            
            if errors:
                return {'valid': False, 'message': 'Block validation failed', 'errors': errors}
            
            self.stats['blocks_validated'] += 1
            print(f"✅ Block #{block['index']} validated: Difficulty {difficulty}, Reward {actual_reward} LKC, Hash {block_hash[:16]}...")
            return {'valid': True, 'message': 'Block is valid', 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}', 'errors': [str(e)]}
    
    def validate_transaction(self, transaction: Dict) -> Dict:
        """
        Validate a transaction submitted by a peer.
        Returns: {'valid': bool, 'message': str, 'errors': List[str]}
        """
        errors = []
        
        try:
            start = time.perf_counter()
            # Early reject (cheap checks)
            tx_type = (transaction.get('type') or '').lower()
            tx_hash = transaction.get('hash', '')
            if tx_hash:
                if hasattr(self.mempool, "is_transaction_pending") and self.mempool.is_transaction_pending(tx_hash):
                    return {'valid': True, 'message': 'Transaction already pending', 'errors': []}
                if hasattr(self.mempool, "is_transaction_confirmed") and self.mempool.is_transaction_confirmed(tx_hash):
                    return {'valid': True, 'message': 'Transaction already confirmed', 'errors': []}
            if tx_type in ("transfer", "transaction"):
                for field in ('from', 'to', 'amount', 'timestamp', 'signature', 'public_key'):
                    if field not in transaction:
                        return {'valid': False, 'message': f'Missing field: {field}', 'errors': [f'Missing field: {field}']}
                if not is_valid_address(str(transaction.get('from', ''))) or not is_valid_address(str(transaction.get('to', ''))):
                    return {'valid': False, 'message': 'Invalid address format', 'errors': ['Invalid address format']}
                try:
                    if self._parse_amount(transaction.get('amount', 0)) <= 0:
                        return {'valid': False, 'message': 'Invalid amount', 'errors': ['Invalid amount']}
                except Exception:
                    return {'valid': False, 'message': 'Invalid amount', 'errors': ['Invalid amount']}
            # Basic validation
            tx_type = transaction.get('type')
            if not tx_type:
                errors.append("Missing transaction type")
            
            # Type-specific validation
            if tx_type == 'transaction':
                required = ['from', 'to', 'amount', 'timestamp']
                for field in required:
                    if field not in transaction:
                        errors.append(f"Missing field: {field}")
                
                # Validate amount
                amount = self._parse_amount(transaction.get('amount', 0))
                if amount <= 0:
                    errors.append("Invalid amount: must be positive")
            
            elif tx_type == 'genesis_bill':
                if 'denomination' not in transaction:
                    errors.append("Missing denomination for genesis bill")
                if not is_valid_address(str(transaction.get('to', ''))):
                    errors.append("Invalid address format")
            
            elif tx_type == 'reward':
                required = ['to', 'amount']
                for field in required:
                    if field not in transaction:
                        errors.append(f"Missing field: {field}")
                if not is_valid_address(str(transaction.get('to', ''))):
                    errors.append("Invalid address format")
            
            # Security validation if available
            if self.security and not errors:
                try:
                    ok, msg = self.security.validate_transaction_security(transaction)
                    if not ok:
                        errors.append(msg)
                except Exception as e:
                    errors.append(f"Security validation error: {e}")
            
            if errors:
                return {'valid': False, 'message': 'Transaction validation failed', 'errors': errors}
            
            self.stats['transactions_validated'] += 1
            self.stats['tx_validation_count'] += 1
            self.stats['tx_validation_seconds'] += (time.perf_counter() - start)
            return {'valid': True, 'message': 'Transaction is valid', 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}', 'errors': [str(e)]}

    def validate_transactions_batch(self, transactions: List[Dict], max_workers: Optional[int] = None) -> Dict:
        """Validate a batch of transactions in parallel.

        Returns: {'accepted': int, 'total': int, 'results': List[Dict]}
        """
        start = time.perf_counter()
        workers = max_workers or self.batch_workers or 1

        if self.security and hasattr(self.security, "validate_transaction_security_batch"):
            sec_results = self.security.validate_transaction_security_batch(transactions, max_workers=workers)
            results = [
                {"valid": ok, "message": msg, "errors": [] if ok else [msg]}
                for ok, msg in sec_results
            ]
        else:
            results = []

            def _validate(tx):
                return self.validate_transaction(tx)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for res in pool.map(_validate, transactions):
                    results.append(res)

        accepted = sum(1 for res in results if res.get("valid"))
        elapsed = time.perf_counter() - start
        self.stats['transactions_validated'] += accepted
        self.stats['tx_validation_count'] += len(transactions)
        self.stats['tx_validation_seconds'] += elapsed
        return {"accepted": accepted, "total": len(transactions), "results": results}
    
    def process_incoming_block_async(self, block: Dict, from_peer: Optional[str] = None, callback: Callable = None) -> str:
        """Async version: Process incoming block in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"process_block_{block.get('index', 'unknown')}_{int(time.time())}"
        
        def _process_task():
            try:
                result = self.process_incoming_block(block, from_peer)
                if callback:
                    callback(success=result.get('success', False), result=result, error=None if result.get('success') else result.get('message'))
                return result
            except Exception as e:
                print(f"❌ Async block processing error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return {'success': False, 'message': str(e)}
        
        future = self.executor.submit(_process_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async block processing: {task_id}")
        return task_id
    
    def process_incoming_block(self, block: Dict, from_peer: Optional[str] = None) -> Dict:
        """
        Process an incoming block from P2P network.
        Validates and adds to blockchain if valid.
        """
        try:
            # Validate block
            validation = self.validate_block(block)
            
            if not validation['valid']:
                print(f"❌ Invalid block from peer {from_peer}: {validation['message']}")
                return validation
            
            # Add to blockchain
            success = self.blockchain.submit_mined_block(block)
            
            if success:
                print(f"✅ Block #{block['index']} accepted from peer {from_peer}")

                # Clear mined transactions from mempool
                try:
                    txs = block.get("transactions", []) if isinstance(block, dict) else []
                    if hasattr(self.mempool, "remove_transaction"):
                        for tx in txs:
                            tx_hash = tx.get("hash") if isinstance(tx, dict) else None
                            if tx_hash:
                                self.mempool.remove_transaction(tx_hash)
                except Exception as e:
                    print(f"⚠️  Failed to clear mempool for block #{block.get('index')}: {e}")
                
                # Broadcast to other peers
                self._broadcast_block_to_peers(block, exclude=from_peer)
                
                return {'success': True, 'message': 'Block accepted and propagated'}
            else:
                return {'success': False, 'message': 'Block submission failed'}
                
        except Exception as e:
            return {'success': False, 'message': f'Processing error: {str(e)}'}
    
    def process_incoming_transaction(self, transaction: Dict, from_peer: Optional[str] = None) -> Dict:
        """
        Process an incoming transaction from P2P network.
        Validates and adds to mempool if valid.
        """
        try:
            tx_hash = transaction.get("hash", "")
            if tx_hash and not self._remember_seen_tx(tx_hash):
                return {'success': True, 'message': 'Duplicate transaction ignored'}

            # Validate transaction
            if self._is_trusted_peer(from_peer):
                validation = {'valid': True, 'message': 'Trusted peer', 'errors': []}
            else:
                validation = self.validate_transaction(transaction)
            
            if not validation['valid']:
                print(f"❌ Invalid transaction from peer {from_peer}: {validation['message']}")
                return validation
            
            # Add to mempool
            if hasattr(self.mempool, "add_transaction"):
                self.mempool.add_transaction(transaction)
            print(f"✅ Transaction accepted from peer {from_peer}")
            
            # Broadcast to other peers
            self._broadcast_transaction_to_peers(transaction, exclude=from_peer)
            
            return {'success': True, 'message': 'Transaction accepted and propagated'}
            
        except Exception as e:
            return {'success': False, 'message': f'Processing error: {str(e)}'}

    def process_incoming_transactions_batch(self, transactions: List[Dict], from_peer: Optional[str] = None) -> Dict:
        """Process a batch of incoming transactions from P2P network."""
        try:
            filtered = []
            for tx in transactions:
                tx_hash = tx.get("hash", "")
                if tx_hash and not self._remember_seen_tx(tx_hash):
                    continue
                if tx_hash and hasattr(self.mempool, "is_transaction_pending") and self.mempool.is_transaction_pending(tx_hash):
                    continue
                if tx_hash and hasattr(self.mempool, "is_transaction_confirmed") and self.mempool.is_transaction_confirmed(tx_hash):
                    continue
                filtered.append(tx)

            if self._is_trusted_peer(from_peer):
                accepted_txs = filtered
            else:
                validation = self.validate_transactions_batch(filtered, max_workers=self.batch_workers)
                accepted_txs = [
                    tx for tx, res in zip(filtered, validation["results"]) if res.get("valid")
                ]

            if accepted_txs:
                if hasattr(self.mempool, "add_transactions_batch_validated"):
                    self.mempool.add_transactions_batch_validated(accepted_txs)
                elif hasattr(self.mempool, "add_transactions_batch"):
                    self.mempool.add_transactions_batch(accepted_txs)
                else:
                    for tx in accepted_txs:
                        if hasattr(self.mempool, "add_transaction"):
                            self.mempool.add_transaction(tx)
            if from_peer:
                if accepted_txs:
                    self._broadcast_transactions_to_peers_batch(accepted_txs, exclude=from_peer)

            return {
                "success": True,
                "accepted": len(accepted_txs),
                "total": len(transactions),
            }
        except Exception as e:
            return {"success": False, "message": f"Processing error: {str(e)}"}

    def _encode_payload(self, payload: Dict, gzip_body: bool = False):
        if self.use_msgpack:
            raw = msgpack.packb(payload, use_bin_type=True)
            headers = {"Content-Type": "application/msgpack"}
        else:
            raw = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}

        if gzip_body and self.p2p_gzip:
            raw = gzip.compress(raw)
            headers["Content-Encoding"] = "gzip"

        return raw, headers

    def _broadcast_transactions_to_peers_batch(self, transactions: List[Dict], exclude: Optional[str] = None):
        """Broadcast a batch of transactions to peers in one request per peer."""
        if not transactions:
            return

        with self.peer_lock:
            targets = [
                peer_info
                for node_id, peer_info in self.peers.items()
                if not exclude or node_id != exclude
            ]

        if not targets:
            return

        payload = {"transactions": transactions}
        body, headers = self._encode_payload(payload, gzip_body=True)

        def _send(peer_info: Dict):
            try:
                peer_url = peer_info.get("url") or peer_info.get("peer_url")
                if peer_url:
                    self._peer_session.post(
                        f"{peer_url}/api/transactions/new/batch",
                        data=body,
                        headers=headers,
                        timeout=5,
                    )
            except Exception:
                pass

        if self.executor:
            for peer_info in targets:
                self.executor.submit(_send, peer_info)
        else:
            for peer_info in targets:
                _send(peer_info)
    
    def _broadcast_block_to_peers(self, block: Dict, exclude: Optional[str] = None):
        """Broadcast block to all registered peers except excluded one"""
        with self.peer_lock:
            for node_id, peer_info in self.peers.items():
                if node_id == exclude:
                    continue
                
                # Send block to peer (implementation depends on transport)
                # This would typically use HTTP POST or WebSocket
                pass
    
    def _broadcast_transaction_to_peers(self, transaction: Dict, exclude: Optional[str] = None):
        """Broadcast transaction to all registered peers except excluded one"""
        with self.peer_lock:
            for node_id, peer_info in self.peers.items():
                if node_id == exclude:
                    continue
                
                # Send transaction to peer
                pass
    
    def _validation_loop(self):
        """Background thread for continuous validation"""
        while self.is_running:
            try:
                # Periodic mempool validation
                # Remove invalid transactions from mempool
                time.sleep(30)
                
            except Exception as e:
                print(f"❌ Validation loop error: {e}")
                time.sleep(10)
    
    def _cleanup_loop(self):
        """Background thread for peer cleanup"""
        while self.is_running:
            try:
                current_time = time.time()
                timeout = 300  # 5 minutes
                
                with self.peer_lock:
                    inactive_peers = []
                    for node_id, peer_info in self.peers.items():
                        if current_time - peer_info['last_seen'] > timeout:
                            inactive_peers.append(node_id)
                    
                    # Remove inactive peers
                    for node_id in inactive_peers:
                        del self.peers[node_id]
                        print(f"🧹 Removed inactive peer: {node_id}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Cleanup loop error: {e}")
                time.sleep(60)
    
    def get_task_status(self, task_id: str) -> Dict:
        """Check status of an async task
        
        Returns: {'status': 'running'|'completed'|'failed'|'not_found', 'result': any, 'error': str}
        """
        if task_id not in self._async_tasks:
            return {'status': 'not_found', 'result': None, 'error': 'Task not found'}
        
        future = self._async_tasks[task_id]
        
        if future.running():
            return {'status': 'running', 'result': None, 'error': None}
        elif future.done():
            try:
                result = future.result(timeout=0)
                return {'status': 'completed', 'result': result, 'error': None}
            except Exception as e:
                return {'status': 'failed', 'result': None, 'error': str(e)}
        else:
            return {'status': 'pending', 'result': None, 'error': None}
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs"""
        return [task_id for task_id, future in self._async_tasks.items() if future.running()]
    
    def cleanup_completed_tasks(self):
        """Remove completed tasks from tracking"""
        completed = [task_id for task_id, future in self._async_tasks.items() if future.done()]
        for task_id in completed:
            del self._async_tasks[task_id]
        if completed:
            print(f"🧹 Cleaned up {len(completed)} completed tasks")
    
    def get_stats(self) -> Dict:
        """Get daemon statistics"""
        uptime = time.time() - self.stats['start_time']
        
        with self.peer_lock:
            peer_count = len(self.peers)
        
        return {
            'uptime_seconds': uptime,
            'blocks_validated': self.stats['blocks_validated'],
            'transactions_validated': self.stats['transactions_validated'],
            'peers_registered': self.stats['peers_registered'],
            'active_peers': peer_count,
            'mempool_size': len(self.mempool.get_pending_transactions()) if self.mempool else 0,
            'tx_validation_seconds': self.stats.get('tx_validation_seconds', 0.0),
            'tx_validation_count': self.stats.get('tx_validation_count', 0)
        }
    
    def get_blockchain_state(self) -> Dict:
        """Get current blockchain state for peers"""
        latest_block = self.blockchain.get_latest_block()
        height = self.blockchain.get_blockchain_height()
        
        return {
            'height': height,
            'latest_block': latest_block,
            'mempool_size': len(self.mempool.get_pending_transactions()) if self.mempool else 0,
            'timestamp': time.time()
        }
