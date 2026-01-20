# lunalib/core/mempool.py - Updated version

import time
import requests
import threading
import sys
import os
from queue import Queue
from typing import Dict, List, Optional, Set
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import gzip
import os

try:
    import msgpack  # type: ignore
    _HAS_MSGPACK = True
except Exception:
    msgpack = None
    _HAS_MSGPACK = False

class MempoolManager:
    """Manages transaction mempool and network broadcasting"""
    
    def __init__(self, network_endpoints: List[str] = None):
        self.network_endpoints = network_endpoints or ["https://bank.linglin.art"]
        self.local_mempool = {}  # {tx_hash: transaction}
        self.pending_broadcasts = Queue()
        self.confirmed_transactions: Set[str] = set()
        self.max_mempool_size = 10000
        self.mempool_ttl = int(os.getenv("LUNALIB_MEMPOOL_TTL", "3600"))
        self.broadcast_retries = 3
        self.is_running = True
        self._threading_enabled = sys.platform != "emscripten"
        self._broadcast_workers = int(os.getenv("LUNALIB_BROADCAST_PARALLEL", "8"))
        self._broadcast_pool = ThreadPoolExecutor(max_workers=self._broadcast_workers) if self._threading_enabled else None
        self._session = requests.Session()
        self._pool_size = int(os.getenv("LUNALIB_HTTP_POOL", "16"))
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self._pool_size,
            pool_maxsize=self._pool_size,
            max_retries=0,
            pool_block=False,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({"Connection": "keep-alive", "User-Agent": "LunaWallet/1.0"})
        self._connection_lock = threading.Lock()
        self._last_connection_test = 0.0
        self._last_connection_ok = False
        self._connection_test_ttl = int(os.getenv("LUNALIB_CONN_TEST_TTL", "5"))
        self._mempool_order = deque()
        self._addr_index = defaultdict(set)
        self._last_remote_fetch = 0.0
        self._remote_refresh = int(os.getenv("LUNALIB_MEMPOOL_REFRESH", "10"))
        self.mobile_mode = bool(int(os.getenv("LUNALIB_MOBILE_MODE", "0")))
        self._broadcast_batch_size = int(
            os.getenv("LUNALIB_BROADCAST_BATCH_SIZE", "100" if self.mobile_mode else "25")
        )
        self._broadcast_batch_window = float(
            os.getenv("LUNALIB_BROADCAST_BATCH_WINDOW", "0.1" if self.mobile_mode else "0.02")
        )
        self.stats = {
            "broadcast_count": 0,
            "broadcast_success": 0,
            "broadcast_seconds": 0.0,
            "batch_broadcast_count": 0,
            "batch_broadcast_success": 0,
            "batch_broadcast_seconds": 0.0,
        }
        self.verbose = bool(int(os.getenv("LUNALIB_DEBUG", "0")))
        self.use_msgpack = bool(int(os.getenv("LUNALIB_USE_MSGPACK", "0"))) and _HAS_MSGPACK
        
        # Start background broadcasting thread (disabled on Pyodide)
        if self._threading_enabled:
            self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
            self.broadcast_thread.start()

    # ----------------------
    # Address normalization
    # ----------------------
    def _normalize_address(self, addr: str) -> str:
        """Normalize addresses (lowercase, strip, drop LUN_)."""
        if not addr:
            return ''
        addr_str = str(addr).strip("'\" ").lower()
        return addr_str[4:] if addr_str.startswith('lun_') else addr_str

    def _log_debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _index_tx(self, tx_hash: str, tx: Dict) -> None:
        if not tx_hash:
            return
        from_norm = self._normalize_address(tx.get('from') or tx.get('sender'))
        to_norm = self._normalize_address(tx.get('to') or tx.get('receiver'))
        if from_norm:
            self._addr_index[from_norm].add(tx_hash)
        if to_norm:
            self._addr_index[to_norm].add(tx_hash)

    def _deindex_tx(self, tx_hash: str) -> None:
        if not tx_hash:
            return
        for addr in list(self._addr_index.keys()):
            hashes = self._addr_index.get(addr)
            if not hashes:
                self._addr_index.pop(addr, None)
                continue
            if tx_hash in hashes:
                hashes.discard(tx_hash)
                if not hashes:
                    self._addr_index.pop(addr, None)

    
    def add_transaction(self, transaction: Dict) -> bool:
        """Add transaction to local mempool and broadcast to network"""
        try:
            tx_hash = transaction.get('hash')
            if not tx_hash:
                if self.verbose:
                    print("DEBUG: Transaction missing hash")
                return False
            
            # Check if transaction already exists or is confirmed
            if tx_hash in self.local_mempool or tx_hash in self.confirmed_transactions:
                if self.verbose:
                    print(f"DEBUG: Transaction already processed: {tx_hash}")
                return True
            
            # Validate basic transaction structure
            if not self._validate_transaction_basic(transaction):
                if self.verbose:
                    print("DEBUG: Transaction validation failed")
                return False
            
            # Add to local mempool
            self.local_mempool[tx_hash] = {
                'transaction': transaction,
                'timestamp': time.time(),
                'broadcast_attempts': 0,
                'last_broadcast': 0
            }
            self._mempool_order.append(tx_hash)
            self._index_tx(tx_hash, transaction)
            self._prune_mempool()
            if self.verbose:
                print(f"DEBUG: Added transaction to mempool: {tx_hash}")
            
            # Queue for broadcasting
            if self._threading_enabled:
                self.pending_broadcasts.put(transaction)
                if self.verbose:
                    print(f"DEBUG: Queued transaction for broadcasting: {tx_hash}")
            else:
                if self.verbose:
                    print(f"DEBUG: Broadcasting inline (no threads available): {tx_hash}")
                self.broadcast_transaction(transaction)
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"DEBUG: Error adding transaction to mempool: {e}")
            return False

    def add_transactions_batch(self, transactions: List[Dict]) -> Dict[str, int]:
        """Add a batch of transactions to the local mempool and broadcast as batch."""
        accepted = 0
        for tx in transactions:
            tx_hash = tx.get("hash")
            if not tx_hash:
                continue
            if tx_hash in self.local_mempool or tx_hash in self.confirmed_transactions:
                continue
            if not self._validate_transaction_basic(tx):
                continue
            self.local_mempool[tx_hash] = {
                "transaction": tx,
                "timestamp": time.time(),
                "broadcast_attempts": 0,
                "last_broadcast": 0,
            }
            self._mempool_order.append(tx_hash)
            self._index_tx(tx_hash, tx)
            accepted += 1

        if accepted:
            self._prune_mempool()

        if transactions:
            if self._threading_enabled:
                self.pending_broadcasts.put(transactions)
            else:
                self.broadcast_transactions_batch(transactions)

        return {"accepted": accepted, "total": len(transactions)}

    def add_transactions_batch_validated(self, transactions: List[Dict]) -> Dict[str, int]:
        """Add a batch of already-validated transactions without revalidation."""
        accepted = 0
        for tx in transactions:
            tx_hash = tx.get("hash")
            if not tx_hash:
                continue
            if tx_hash in self.local_mempool or tx_hash in self.confirmed_transactions:
                continue
            self.local_mempool[tx_hash] = {
                "transaction": tx,
                "timestamp": time.time(),
                "broadcast_attempts": 0,
                "last_broadcast": 0,
            }
            self._mempool_order.append(tx_hash)
            self._index_tx(tx_hash, tx)
            accepted += 1

        if accepted:
            self._prune_mempool()

        if transactions:
            if self._threading_enabled:
                self.pending_broadcasts.put(transactions)
            else:
                self.broadcast_transactions_batch(transactions)

        return {"accepted": accepted, "total": len(transactions)}
    
    def broadcast_transaction(self, transaction: Dict) -> bool:
        """Broadcast transaction to network endpoints - SIMPLIFIED FOR YOUR FLASK APP"""
        tx_hash = transaction.get('hash')
        print(f"DEBUG: Broadcasting transaction to mempool: {tx_hash}")
        
        start = time.perf_counter()
        use_gzip = bool(int(os.getenv("LUNALIB_HTTP_GZIP", "1")))

        def _post(endpoint: str) -> bool:
            for attempt in range(self.broadcast_retries):
                try:
                    # Use the correct endpoint for your Flask app
                    broadcast_endpoint = f"{endpoint}/mempool/add"

                    print(f"DEBUG: Attempt {attempt + 1} to {broadcast_endpoint}")
                    print(f"DEBUG: Transaction type: {transaction.get('type')}")
                    print(f"DEBUG: From: {transaction.get('from')}")
                    print(f"DEBUG: To: {transaction.get('to')}")
                    print(f"DEBUG: Amount: {transaction.get('amount')}")

                    headers = {
                        'Content-Type': 'application/json',
                        'User-Agent': 'LunaWallet/1.0',
                        'Connection': 'keep-alive',
                    }

                    if use_gzip:
                        payload = gzip.compress(json.dumps(transaction).encode("utf-8"))
                        headers['Content-Encoding'] = 'gzip'
                        response = self._session.post(
                            broadcast_endpoint,
                            data=payload,
                            headers=headers,
                            timeout=10
                        )
                    else:
                        response = self._session.post(
                            broadcast_endpoint,
                            json=transaction,  # Send the transaction directly
                            headers=headers,
                            timeout=10
                        )

                    print(f"DEBUG: Response status: {response.status_code}")

                    if response.status_code in [200, 201]:
                        result = response.json()
                        print(f"DEBUG: Response data: {result}")

                        if result.get('success'):
                            print(f"✅ Successfully added to mempool via {broadcast_endpoint}")
                            return True
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            print(f"❌ Mempool rejected transaction: {error_msg}")
                    else:
                        print(f"❌ HTTP error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    print(f"❌ Cannot connect to {endpoint}")
                except requests.exceptions.Timeout:
                    print(f"❌ Request timeout to {endpoint}")
                except Exception as e:
                    print(f"❌ Exception during broadcast: {e}")

                # Wait before retry
                if attempt < self.broadcast_retries - 1:
                    print(f"DEBUG: Waiting before retry...")
                    time.sleep(2)
            return False

        if self._broadcast_pool:
            results = list(self._broadcast_pool.map(_post, self.network_endpoints))
            success = any(results)
        else:
            success = False
            for endpoint in self.network_endpoints:
                if _post(endpoint):
                    success = True

        if success:
            print(f"✅ Transaction {tx_hash} successfully broadcasted")
        else:
            print(f"❌ All broadcast attempts failed for transaction {tx_hash}")

        elapsed = time.perf_counter() - start
        self.stats["broadcast_count"] += 1
        self.stats["broadcast_success"] += 1 if success else 0
        self.stats["broadcast_seconds"] += elapsed

        return success

    def broadcast_transactions_batch(self, transactions: List[Dict]) -> int:
        """Broadcast a batch of transactions to network endpoints."""
        if not transactions:
            return 0

        accepted = 0
        payload = {"transactions": transactions}
        if self.use_msgpack:
            raw = msgpack.packb(payload, use_bin_type=True)
            headers = {"Content-Type": "application/msgpack"}
            gz = gzip.compress(raw)
            headers["Content-Encoding"] = "gzip"
        else:
            raw = json.dumps(payload).encode("utf-8")
            gz = gzip.compress(raw)
            headers = {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            }
        start = time.perf_counter()

        def _post(endpoint: str) -> int:
            batch_endpoint = f"{endpoint}/mempool/add/batch"
            try:
                response = self._session.post(batch_endpoint, data=gz, headers=headers, timeout=10)
                if response.status_code in [200, 201]:
                    return int(response.json().get("accepted", 0))
            except Exception:
                pass
            return 0

        def _post_single(endpoint: str, tx: Dict) -> bool:
            try:
                response = self._session.post(
                    f"{endpoint}/mempool/add",
                    json=tx,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                return response.status_code in [200, 201]
            except Exception:
                return False

        if self._broadcast_pool:
            for count in self._broadcast_pool.map(_post, self.network_endpoints):
                accepted = max(accepted, count)
        else:
            for endpoint in self.network_endpoints:
                accepted = max(accepted, _post(endpoint))

        # Fallback: if batch not accepted anywhere, try single-transaction posts
        if accepted == 0:
            for endpoint in self.network_endpoints:
                single_ok = 0
                for tx in transactions:
                    if _post_single(endpoint, tx):
                        single_ok += 1
                if single_ok > 0:
                    accepted = max(accepted, single_ok)
                    break

        elapsed = time.perf_counter() - start
        self.stats["batch_broadcast_count"] += 1
        self.stats["batch_broadcast_success"] += accepted
        self.stats["batch_broadcast_seconds"] += elapsed
        return accepted

    def get_stats(self) -> Dict:
        return self.stats.copy()
    
    def test_connection(self) -> bool:
        """Test connection to network endpoints"""
        now = time.time()
        with self._connection_lock:
            if self._connection_test_ttl > 0 and (now - self._last_connection_test) < self._connection_test_ttl:
                return self._last_connection_ok

            for endpoint in self.network_endpoints:
                try:
                    self._log_debug(f"DEBUG: Testing connection to {endpoint}")
                    # Test with a simple health check or mempool status
                    test_endpoints = [
                        f"{endpoint}/system/health",
                        f"{endpoint}/mempool/status",
                        f"{endpoint}/"
                    ]

                    for test_endpoint in test_endpoints:
                        try:
                            with self._session.get(test_endpoint, timeout=5) as response:
                                self._log_debug(
                                    f"DEBUG: Connection test response from {test_endpoint}: {response.status_code}"
                                )
                                if response.status_code == 200:
                                    print(f"✅ Successfully connected to {endpoint}")
                                    self._last_connection_test = now
                                    self._last_connection_ok = True
                                    return True
                        except Exception as e:
                            self._log_debug(f"DEBUG: Connection test error from {test_endpoint}: {e}")
                            continue

                except Exception as e:
                    self._log_debug(f"DEBUG: Connection test failed for {endpoint}: {e}")

            print("❌ All connection tests failed")
            self._last_connection_test = now
            self._last_connection_ok = False
            return False
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction from mempool by hash"""
        if tx_hash in self.local_mempool:
            return self.local_mempool[tx_hash]['transaction']
        return None
    
    def _maybe_fetch_remote_mempool(self):
        """Fetch mempool from remote endpoints and merge into local cache."""
        now = time.time()
        if self._remote_refresh > 0 and (now - self._last_remote_fetch) < self._remote_refresh:
            return
        self._last_remote_fetch = now
        for endpoint in self.network_endpoints:
            try:
                resp = self._session.get(f"{endpoint}/mempool", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for tx in data:
                            tx_hash = tx.get('hash')
                            if tx_hash and tx_hash not in self.local_mempool and tx_hash not in self.confirmed_transactions:
                                self.local_mempool[tx_hash] = {
                                    'transaction': tx,
                                    'timestamp': time.time(),
                                    'broadcast_attempts': 0,
                                    'last_broadcast': 0
                                }
                                self._index_tx(tx_hash, tx)
                else:
                    print(f"DEBUG: Remote mempool fetch HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"DEBUG: Remote mempool fetch error from {endpoint}: {e}")

    def get_pending_transactions(self, address: str = None, fetch_remote: bool = True) -> List[Dict]:
        """Get all pending transactions, optionally filtered by address; can fetch remote first."""
        if fetch_remote:
            self._maybe_fetch_remote_mempool()

        target_norm = self._normalize_address(address) if address else None
        transactions = []
        if target_norm:
            for tx_hash in self._addr_index.get(target_norm, set()):
                tx_data = self.local_mempool.get(tx_hash)
                if tx_data:
                    transactions.append(tx_data['transaction'])
        else:
            for tx_data in self.local_mempool.values():
                transactions.append(tx_data['transaction'])
                
        max_tx = int(os.getenv("LUNALIB_MEMPOOL_TX_LIMIT", "2000"))
        if max_tx > 0 and len(transactions) > max_tx:
            transactions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            transactions = transactions[:max_tx]
        print(f"[MEMPOOL] get_pending_transactions for {address}: {len(transactions)} txs returned")
        return transactions

    def get_pending_transactions_for_addresses(self, addresses: List[str], fetch_remote: bool = True) -> Dict[str, List[Dict]]:
        """Get pending transactions mapped per address in one pass; can fetch remote first."""
        if not addresses:
            return {}

        if fetch_remote:
            self._maybe_fetch_remote_mempool()

        norm_to_original: Dict[str, str] = {}
        for addr in addresses:
            norm = self._normalize_address(addr)
            if norm:
                norm_to_original[norm] = addr

        results: Dict[str, List[Dict]] = {addr: [] for addr in addresses}

        for norm, original in norm_to_original.items():
            for tx_hash in self._addr_index.get(norm, set()):
                tx_data = self.local_mempool.get(tx_hash)
                if tx_data:
                    results[original].append(tx_data['transaction'])

        max_tx = int(os.getenv("LUNALIB_MEMPOOL_TX_LIMIT", "2000"))
        if max_tx > 0:
            for addr, txs in results.items():
                if len(txs) > max_tx:
                    txs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    results[addr] = txs[:max_tx]

        return {addr: txs for addr, txs in results.items() if txs}
    
    def remove_transaction(self, tx_hash: str):
        """Remove transaction from mempool (usually after confirmation)"""
        if tx_hash in self.local_mempool:
            del self.local_mempool[tx_hash]
            self.confirmed_transactions.add(tx_hash)
            self._deindex_tx(tx_hash)
            try:
                self._mempool_order.remove(tx_hash)
            except ValueError:
                pass
            print(f"DEBUG: Removed transaction from mempool: {tx_hash}")
    
    def is_transaction_pending(self, tx_hash: str) -> bool:
        """Check if transaction is pending in mempool"""
        return tx_hash in self.local_mempool
    
    def is_transaction_confirmed(self, tx_hash: str) -> bool:
        """Check if transaction has been confirmed"""
        return tx_hash in self.confirmed_transactions
    
    def get_mempool_size(self) -> int:
        """Get current mempool size"""
        return len(self.local_mempool)
    
    def clear_mempool(self):
        """Clear all transactions from mempool"""
        self.local_mempool.clear()
        self._mempool_order.clear()
        self._addr_index.clear()
        print("DEBUG: Cleared mempool")
    
    
    
    def _validate_transaction_basic(self, transaction: Dict) -> bool:
        """Basic transaction validation"""
        required_fields = ['type', 'from', 'to', 'amount', 'timestamp', 'hash']
        
        for field in required_fields:
            if field not in transaction:
                print(f"DEBUG: Missing required field: {field}")
                return False
        
        # Validate amount
        try:
            amount = float(transaction['amount'])
            if amount <= 0:
                print("DEBUG: Invalid amount (must be positive)")
                return False
        except (ValueError, TypeError):
            print("DEBUG: Invalid amount format")
            return False
        
        # Validate timestamp (not too far in future)
        try:
            timestamp = float(transaction['timestamp'])
            if timestamp > time.time() + 300:  # 5 minutes in future
                print("DEBUG: Transaction timestamp too far in future")
                return False
        except (ValueError, TypeError):
            print("DEBUG: Invalid timestamp format")
            return False
        
        # Validate addresses (basic format check)
        from_addr = transaction.get('from', '')
        to_addr = transaction.get('to', '')
        
        if not from_addr or not to_addr:
            print("DEBUG: Missing from or to address")
            return False

        # Early reject: address format + signature/public key for transfers
        tx_type = (transaction.get("type") or "").lower()
        if tx_type in ("transfer", "transaction"):
            if not from_addr.startswith("LUN_") or not to_addr.startswith("LUN_"):
                print("DEBUG: Invalid address format")
                return False
            signature = transaction.get("signature", "")
            public_key = transaction.get("public_key", "")
            if not signature or not public_key:
                print("DEBUG: Missing signature or public key")
                return False
            if len(signature) != 128:
                print("DEBUG: Invalid signature length")
                return False
        
        print(f"✅ Transaction validation passed: {transaction.get('type')} from {from_addr} to {to_addr}")
        return True
    
    def _broadcast_worker(self):
        """Background worker to process pending broadcasts"""
        while self.is_running:
            try:
                self._prune_mempool()
                # Get next item to broadcast (blocking)
                item = self.pending_broadcasts.get(timeout=1.0)

                batch = item if isinstance(item, list) else [item]
                deadline = time.perf_counter() + self._broadcast_batch_window

                while len(batch) < self._broadcast_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        next_item = self.pending_broadcasts.get(timeout=remaining)
                    except Exception:
                        break
                    if isinstance(next_item, list):
                        batch.extend(next_item)
                    else:
                        batch.append(next_item)

                if batch:
                    if len(batch) > self._broadcast_batch_size:
                        for start in range(0, len(batch), self._broadcast_batch_size):
                            chunk = batch[start : start + self._broadcast_batch_size]
                            self.broadcast_transactions_batch(chunk)
                    else:
                        self.broadcast_transactions_batch(batch)

                    for transaction in batch:
                        tx_hash = transaction.get("hash")
                        if tx_hash in self.local_mempool:
                            self.local_mempool[tx_hash]["broadcast_attempts"] += 1
                            self.local_mempool[tx_hash]["last_broadcast"] = time.time()

                    self.pending_broadcasts.task_done()
                    time.sleep(0.05)
                    
            except Exception as e:
                # Queue.get() timed out or other error
                continue
    
    def stop(self):
        """Stop the mempool manager"""
        self.is_running = False
        print("DEBUG: Mempool manager stopped")

    def _prune_mempool(self):
        """Prune expired or excess mempool entries."""
        now = time.time()

        # Prune by TTL
        if self.mempool_ttl > 0:
            while self._mempool_order:
                tx_hash = self._mempool_order[0]
                tx_data = self.local_mempool.get(tx_hash)
                if not tx_data:
                    self._mempool_order.popleft()
                    continue
                if now - tx_data.get('timestamp', now) > self.mempool_ttl:
                    self._mempool_order.popleft()
                    self.local_mempool.pop(tx_hash, None)
                    self._deindex_tx(tx_hash)
                    continue
                break

        # Prune by size
        while len(self.local_mempool) > self.max_mempool_size and self._mempool_order:
            tx_hash = self._mempool_order.popleft()
            self.local_mempool.pop(tx_hash, None)
            self._deindex_tx(tx_hash)