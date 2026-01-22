from tqdm import tqdm
# blockchain.py - Updated version

from ..storage.cache import BlockchainCache
import requests
import time
import os
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple, Callable, Union
import gzip
import re


class BlockchainManager:
    """Manages blockchain interactions and scanning with transaction broadcasting"""
    
    def __init__(self, endpoint_url="https://bank.linglin.art", max_workers=10):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.cache = BlockchainCache()
        self.network_connected = False
        self._stop_events = []  # Track background monitors so they can be stopped
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BlockchainWorker")
        self._async_tasks = {}  # Track async tasks by ID
        self._task_callbacks = {}  # Callbacks for task completion
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------
    def _normalize_address(self, addr: str) -> str:
        """Normalize LUN addresses for comparison (lowercase, strip, drop prefix)."""
        if not addr:
            return ''
        addr_str = str(addr).strip("'\" ").lower()
        return addr_str[4:] if addr_str.startswith('lun_') else addr_str

    def _extract_miner_address(self, block: Dict) -> str:
        """Extract miner address from a block using known aliases."""
        return (
            block.get('miner')
            or block.get('mined_by')
            or block.get('miner_address')
            or block.get('mined_by_address')
            or block.get('minerAddress')
            or block.get('minedBy')
            or ''
        )

    def _extract_reward_amount(self, block: Dict) -> float:
        """Extract reward amount from block metadata using known aliases."""
        reward_raw = (
            block.get('reward')
            or block.get('reward_amount')
            or block.get('block_reward')
            or block.get('mining_reward')
            or block.get('miner_reward')
            or block.get('rewardAmount')
            or block.get('blockReward')
        )
        return self._parse_amount(reward_raw or 0)
        
    def broadcast_transaction_async(self, transaction: Dict, callback: Callable = None) -> str:
        """Async version: Broadcast transaction in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"broadcast_{transaction.get('hash', 'unknown')}_{int(time.time())}"
        
        def _broadcast_task():
            try:
                success, message = self.broadcast_transaction(transaction)
                if callback:
                    callback(success=success, result=message, error=None if success else message)
                return (success, message)
            except Exception as e:
                print(f"❌ Async broadcast error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return (False, str(e))
        
        future = self.executor.submit(_broadcast_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async broadcast: {task_id}")
        return task_id
    
    def broadcast_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Broadcast transaction to mempool with enhanced error handling"""
        try:
            print(f"🔄 Broadcasting transaction to {self.endpoint_url}/mempool/add")
            print(f"   Transaction type: {transaction.get('type', 'unknown')}")
            print(f"   From: {transaction.get('from', 'unknown')}")
            print(f"   To: {transaction.get('to', 'unknown')}")
            print(f"   Amount: {transaction.get('amount', 'unknown')}")
            
            # Ensure transaction has required fields
            if not self._validate_transaction_before_broadcast(transaction):
                return False, "Transaction validation failed"
            
            response = self._session.post(
                f'{self.endpoint_url}/mempool/add',
                json=transaction,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            print(f"📡 Broadcast response: HTTP {response.status_code}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                if result.get('success'):
                    tx_hash = result.get('transaction_hash', transaction.get('hash', 'unknown'))
                    print(f"✅ Transaction broadcast successful! Hash: {tx_hash}")
                    return True, f"Transaction broadcast successfully: {tx_hash}"
                else:
                    error_msg = result.get('error', 'Unknown error from server')
                    print(f"❌ Broadcast failed: {error_msg}")
                    return False, f"Server rejected transaction: {error_msg}"
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Network error: {error_msg}")
                return False, error_msg
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to blockchain server"
            print(f"❌ {error_msg}")
            return False, error_msg
        except requests.exceptions.Timeout:
            error_msg = "Broadcast request timed out"
            print(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _validate_transaction_before_broadcast(self, transaction: Dict) -> bool:
        """Validate transaction before broadcasting"""
        required_fields = ['type', 'from', 'to', 'amount', 'timestamp', 'hash', 'signature']
        
        for field in required_fields:
            if field not in transaction:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate addresses
        if not transaction['from'].startswith('LUN_'):
            print(f"❌ Invalid from address format: {transaction['from']}")
            return False
            
        if not transaction['to'].startswith('LUN_'):
            print(f"❌ Invalid to address format: {transaction['to']}")
            return False
        
        # Validate amount
        try:
            amount = float(transaction['amount'])
            if amount <= 0:
                print(f"❌ Invalid amount: {amount}")
                return False
        except (ValueError, TypeError):
            print(f"❌ Invalid amount format: {transaction['amount']}")
            return False
        
        # Validate signature
        if not transaction.get('signature') or len(transaction['signature']) < 10:
            print(f"❌ Invalid or missing signature")
            return False
        
        # Validate hash
        if not transaction.get('hash') or len(transaction['hash']) < 10:
            print(f"❌ Invalid or missing transaction hash")
            return False
        
        print("✅ Transaction validation passed")
        return True

    def get_transaction_status(self, tx_hash: str) -> Dict:
        """Check transaction status (pending/confirmed)"""
        try:
            # First check mempool for pending transactions
            mempool_txs = self.get_mempool()
            for tx in mempool_txs:
                if tx.get('hash') == tx_hash:
                    return {
                        'status': 'pending',
                        'message': 'Transaction is in mempool waiting to be mined',
                        'confirmations': 0
                    }
            
            # Then check blockchain for confirmed transactions
            current_height = self.get_blockchain_height()
            for height in range(max(0, current_height - 100), current_height + 1):
                block = self.get_block(height)
                if block:
                    for tx in block.get('transactions', []):
                        if tx.get('hash') == tx_hash:
                            confirmations = current_height - height + 1
                            return {
                                'status': 'confirmed',
                                'message': f'Transaction confirmed in block {height}',
                                'confirmations': confirmations,
                                'block_height': height
                            }
            
            return {
                'status': 'unknown',
                'message': 'Transaction not found in mempool or recent blocks'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking transaction status: {str(e)}'
            }

    def get_blockchain_height(self) -> int:
        """Get current blockchain height - FIXED VERSION"""
        try:
            # Get the actual latest block to determine height
            response = self._session.get(f'{self.endpoint_url}/blockchain/blocks', timeout=10)
            if response.status_code == 200:
                data = response.json()
                blocks = data.get('blocks', [])
                
                if blocks:
                    # The height is the index of the latest block
                    latest_block = blocks[-1]
                    latest_index = latest_block.get('index', len(blocks) - 1)
                    print(f"🔍 Server has {len(blocks)} blocks, latest index: {latest_index}")
                    print(f"🔍 Latest block hash: {latest_block.get('hash', '')[:32]}...")
                    return latest_index
                return 0
                    
        except Exception as e:
            print(f"Blockchain height error: {e}")
            
        return 0

    def get_latest_block(self) -> Optional[Dict]:
        """Get the actual latest block from server"""
        try:
            response = self._session.get(f'{self.endpoint_url}/blockchain/blocks', timeout=10)
            if response.status_code == 200:
                data = response.json()
                blocks = data.get('blocks', [])
                if blocks:
                    return blocks[-1]
        except Exception as e:
            print(f"Get latest block error: {e}")
        return None

    def get_server_stats(self) -> Dict:
        """Fetch server stats/configuration if exposed by the daemon."""
        endpoints = [
            f"{self.endpoint_url}/api/blockchain-stats",
            f"{self.endpoint_url}/blockchain/stats",
            f"{self.endpoint_url}/blockchain/status",
        ]
        for url in endpoints:
            try:
                response = self._session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data if isinstance(data, dict) else {"data": data}
            except Exception:
                continue
        return {}

    def scan_chain(self, peer_urls: Optional[List[str]] = None) -> Optional[Dict]:
        """Download full blockchain, favoring primary and falling back to peers."""
        def _fetch(base_url: str) -> Optional[Dict]:
            for url in (f"{base_url}/blockchain", f"{base_url}/api/blockchain/full"):
                try:
                    response = self._session.get(url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ Downloaded blockchain: {len(data.get('blocks', []))} blocks")
                        return data
                except Exception:
                    continue
            return None

        data = _fetch(self.endpoint_url)
        if data:
            return data

        if peer_urls:
            for peer_url in peer_urls:
                data = _fetch(peer_url.rstrip('/'))
                if data:
                    print(f"✅ Downloaded blockchain from peer: {peer_url}")
                    return data

        print("❌ Failed to download blockchain from primary and peers")
        return None
    
    def _looks_like_hash(self, value: str) -> bool:
        if not value or len(value) != 64:
            return False
        return all(ch in "0123456789abcdefABCDEF" for ch in value)

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

    def _is_reward_tx(self, tx: Dict) -> bool:
        tx_type = str(tx.get("type") or "").lower()
        desc = str(tx.get("description") or "").lower()
        return (
            tx_type in {"reward", "coinbase", "mining_reward"}
            or str(tx.get("hash", "")).startswith("reward_")
            or (str(tx.get("from", "")) == "network" and ("reward" in desc or "mining" in desc))
            or (str(tx.get("from", "")) == "network" and tx.get("block_height") is not None)
            or ("reward" in tx)
        )

    def _is_gtx_genesis_tx(self, tx: Dict) -> bool:
        tx_type = str(tx.get("type") or "").lower()
        return tx_type in {"gtx_genesis", "genesis_bill", "gtxgenesis"}

    def _filter_transactions(
        self,
        txs: List[Dict],
        include_rewards: bool,
        include_transfers: bool,
        include_gtx_genesis: bool,
    ) -> List[Dict]:
        filtered: List[Dict] = []
        for tx in txs:
            if self._is_reward_tx(tx):
                if include_rewards:
                    filtered.append(tx)
                continue
            if self._is_gtx_genesis_tx(tx):
                if include_gtx_genesis:
                    filtered.append(tx)
                continue
            if include_transfers:
                filtered.append(tx)
        return filtered

    def _fetch_blocks_list(self) -> List[Dict]:
        try:
            response = self._session.get(f'{self.endpoint_url}/blockchain/blocks', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                return data.get('blocks', [])
        except Exception as e:
            print(f"Get blocks list error: {e}")
        return []

    def get_block_by_height(self, height: int) -> Optional[Dict]:
        """Get block by height (index)"""
        # Check cache first
        cached_block = self.cache.get_block(height)
        if cached_block:
            return cached_block

        try:
            response = self._session.get(f'{self.endpoint_url}/blockchain/block/{height}', timeout=10)
            if response.status_code == 200:
                block = response.json()
                self.cache.save_block(height, block.get('hash', ''), block)
                return block
            if response.status_code == 404:
                # Fallback: some servers expect hash at /blockchain/block/<value>
                blocks = self._fetch_blocks_list()
                if 0 <= height < len(blocks):
                    block = blocks[height]
                    self.cache.save_block(height, block.get('hash', ''), block)
                    return block
                for block in blocks:
                    block_index = block.get('index') if isinstance(block, dict) else None
                    if block_index == height:
                        self.cache.save_block(height, block.get('hash', ''), block)
                        return block
        except Exception as e:
            print(f"Get block error: {e}")

        return None

    def get_block_by_hash(self, block_hash: str) -> Optional[Dict]:
        """Get block by hash"""
        if not block_hash:
            return None

        try:
            response = self._session.get(f'{self.endpoint_url}/blockchain/block/{block_hash}', timeout=10)
            if response.status_code == 200:
                block = response.json()
                height = block.get('index', block.get('height'))
                if isinstance(height, int):
                    self.cache.save_block(height, block.get('hash', ''), block)
                return block
        except Exception as e:
            print(f"Get block by hash error: {e}")

        # Fallback: scan blocks list
        blocks = self._fetch_blocks_list()
        for block in blocks:
            if isinstance(block, dict) and block.get('hash') == block_hash:
                height = block.get('index', block.get('height'))
                if isinstance(height, int):
                    self.cache.save_block(height, block.get('hash', ''), block)
                return block

        return None

    def get_block(self, block_id: Union[int, str]) -> Optional[Dict]:
        """Get block by height (int) or hash (str)."""
        if isinstance(block_id, str):
            value = block_id.strip()
            if self._looks_like_hash(value):
                return self.get_block_by_hash(value)
            if value.isdigit():
                return self.get_block_by_height(int(value))
            return self.get_block_by_hash(value)

        return self.get_block_by_height(int(block_id))
    
    def get_blocks_range_async(self, start_height: int, end_height: int, callback: Callable = None) -> str:
        """Async version: Get range of blocks in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"blocks_range_{start_height}_{end_height}_{int(time.time())}"
        
        def _fetch_task():
            try:
                result = self.get_blocks_range(start_height, end_height)
                if callback:
                    callback(success=True, result=result, error=None)
                return result
            except Exception as e:
                print(f"❌ Async blocks fetch error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return None
        
        future = self.executor.submit(_fetch_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async blocks fetch: {task_id}")
        return task_id
    
    def get_blocks_range(self, start_height: int, end_height: int, cache_only: bool = False) -> List[Dict]:
        """Get range of blocks (uses cache and fetches only missing blocks)"""
        blocks = []

        # Check cache first
        cached_blocks = self.cache.get_block_range(start_height, end_height)
        cached_by_height: Dict[int, Dict] = {}
        for block in cached_blocks:
            height = block.get('index', block.get('height'))
            if isinstance(height, int):
                cached_by_height[height] = block

        tail_refresh = int(os.getenv("LUNALIB_CACHE_TAIL_REFRESH", "25"))
        if tail_refresh < 0:
            tail_refresh = 0
        if tail_refresh:
            tail_start = max(start_height, end_height - tail_refresh + 1)
            for height in range(tail_start, end_height + 1):
                cached_by_height.pop(height, None)
                try:
                    self.cache.delete_block(height)
                except Exception:
                    pass

        expected_count = (end_height - start_height + 1)
        if len(cached_by_height) == expected_count:
            return [cached_by_height[h] for h in range(start_height, end_height + 1)]

        if cache_only:
            return [cached_by_height[h] for h in range(start_height, end_height + 1) if h in cached_by_height]

        missing_heights = [
            h for h in range(start_height, end_height + 1) if h not in cached_by_height
        ]

        try:
            if not cached_by_height:
                response = self._session.get(
                    f'{self.endpoint_url}/blockchain/range?start={start_height}&end={end_height}',
                    timeout=30
                )
                if response.status_code == 200:
                    blocks = response.json().get('blocks', [])
                    # Cache the blocks
                    for block in blocks:
                        height = block.get('index', 0)
                        self.cache.save_block(height, block.get('hash', ''), block)
                    return blocks

            # Fallback: fetch missing blocks individually (optionally in parallel)
            fetch_delay = float(os.getenv("LUNALIB_BLOCK_FETCH_DELAY", "0"))
            max_workers = int(os.getenv("LUNALIB_BLOCK_FETCH_WORKERS", "4"))
            if max_workers < 1:
                max_workers = 1

            def _fetch_one(height: int) -> Optional[Dict]:
                cached = self.cache.get_block(height)
                if cached:
                    return cached
                if fetch_delay > 0:
                    time.sleep(fetch_delay)
                try:
                    response = requests.get(f'{self.endpoint_url}/blockchain/block/{height}', timeout=10)
                    if response.status_code == 200:
                        block = response.json()
                        self.cache.save_block(height, block.get('hash', ''), block)
                        return block
                except Exception:
                    return None
                return None

            if max_workers == 1 or len(missing_heights) < 2:
                for height in tqdm(missing_heights, desc="Get Blocks", leave=False):
                    block = _fetch_one(height)
                    if block:
                        cached_by_height[height] = block
            else:
                futures = [self.executor.submit(_fetch_one, height) for height in missing_heights]
                for height, future in tqdm(list(zip(missing_heights, futures)), desc="Get Blocks", leave=False):
                    try:
                        block = future.result()
                    except Exception:
                        block = None
                    if block:
                        cached_by_height[height] = block

        except Exception as e:
            print(f"Get blocks range error: {e}")

        return [cached_by_height[h] for h in range(start_height, end_height + 1) if h in cached_by_height]
    
    def get_mempool(self) -> List[Dict]:
        """Get current mempool transactions"""
        try:
            response = self._session.get(f'{self.endpoint_url}/mempool', timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Mempool error: {e}")
            
        return []
    
    def check_network_connection(self) -> bool:
        """Check if network is accessible"""
        try:
            response = self._session.get(f'{self.endpoint_url}/system/health', timeout=5)
            self.network_connected = response.status_code == 200
            return self.network_connected
        except:
            self.network_connected = False
            return False
    
    def scan_transactions_for_address(
        self,
        address: str,
        start_height: int = 0,
        end_height: int = None,
        cache_only: bool = False,
        max_range: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Scan blockchain for transactions involving an address"""
        if end_height is None:
            end_height = self.get_blockchain_height()

        if max_range and end_height - start_height + 1 > max_range:
            start_height = max(0, end_height - max_range + 1)

        print(f"[SCAN] Scanning transactions for {address} from block {start_height} to {end_height}")

        transactions = []
        batch_size = int(os.getenv("LUNALIB_SCAN_BATCH_SIZE", "200"))
        if batch_size < 1:
            batch_size = 200
        total_batches = ((end_height - start_height) // batch_size) + 1
        batch_index = 0
        skip_full_fetch = os.getenv("LUNALIB_SCAN_SKIP_FULL_BLOCK_FETCH", "0") == "1"
        for batch_start in tqdm(range(start_height, end_height + 1, batch_size), desc=f"Scan {address}", total=total_batches):
            batch_end = min(batch_start + batch_size - 1, end_height)
            if progress_callback:
                progress_callback({
                    "stage": "scan",
                    "scope": "address",
                    "address": address,
                    "batch_index": batch_index,
                    "batch_total": total_batches,
                    "start_height": start_height,
                    "end_height": end_height,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "cache_only": cache_only,
                })
            blocks = self.get_blocks_range(batch_start, batch_end, cache_only=cache_only)
            for block in blocks:
                if not isinstance(block, dict):
                    resolved = None
                    if isinstance(block, (int, str)):
                        resolved = self.get_block(block)
                    if isinstance(resolved, dict):
                        block = resolved
                    else:
                        continue

                height = block.get('index', block.get('height'))
                has_txs = isinstance(block.get('transactions'), list)
                has_meta = bool(self._extract_miner_address(block)) or (self._extract_reward_amount(block) > 0)
                if not skip_full_fetch and (not has_txs or not has_meta):
                    full_block = None
                    if cache_only:
                        if isinstance(height, int):
                            full_block = self.cache.get_block(height)
                    else:
                        if isinstance(height, int):
                            full_block = self.get_block(height)
                        elif block.get('hash'):
                            full_block = self.get_block(block.get('hash'))
                    if isinstance(full_block, dict):
                        block = full_block
                block_transactions = self._find_address_transactions(block, address)
                transactions.extend(block_transactions)
            batch_index += 1
            if progress_callback:
                progress_callback({
                    "stage": "scan",
                    "scope": "address",
                    "address": address,
                    "batch_index": batch_index,
                    "batch_total": total_batches,
                    "start_height": start_height,
                    "end_height": end_height,
                    "scanned_end": batch_end,
                    "tx_count": len(transactions),
                    "cache_only": cache_only,
                })

        max_tx = int(os.getenv("LUNALIB_SCAN_TX_LIMIT", "5000"))
        if max_tx > 0 and len(transactions) > max_tx:
            transactions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            transactions = transactions[:max_tx]

        print(f"[SCAN] Found {len(transactions)} total transactions for {address}")
        return transactions
    
    def scan_transactions_for_address_async(self, address: str, callback: Callable = None, 
                                           start_height: int = 0, end_height: int = None) -> str:
        """Async version: Scan blockchain in background thread, call callback when done
        
        Returns: task_id that can be used to check status
        """
        task_id = f"scan_{address}_{int(time.time())}"
        
        def _scan_task():
            try:
                result = self.scan_transactions_for_address(address, start_height, end_height)
                if callback:
                    callback(success=True, result=result, error=None)
                return result
            except Exception as e:
                print(f"❌ Async scan error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return None
        
        future = self.executor.submit(_scan_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async scan task: {task_id}")
        return task_id

    def scan_transactions_for_addresses(
        self,
        addresses: List[str],
        start_height: int = 0,
        end_height: int = None,
        cache_only: bool = False,
        max_range: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, List[Dict]]:
        """Scan the blockchain once for multiple addresses (rewards and transfers)."""
        if not addresses:
            return {}

        if end_height is None:
            end_height = self.get_blockchain_height()

        if end_height < start_height:
            return {addr: [] for addr in addresses}

        if max_range and end_height - start_height + 1 > max_range:
            start_height = max(0, end_height - max_range + 1)

        print(f"[MULTI-SCAN] Scanning {len(addresses)} addresses from block {start_height} to {end_height}")

        # Map normalized address -> original address for quick lookup
        normalized_map = {}
        for addr in addresses:
            norm = self._normalize_address(addr)
            if norm:
                normalized_map[norm] = addr

        results: Dict[str, List[Dict]] = {addr: [] for addr in addresses}

        batch_size = int(os.getenv("LUNALIB_SCAN_BATCH_SIZE", "200"))
        if batch_size < 1:
            batch_size = 200
        total_batches = ((end_height - start_height) // batch_size) + 1
        batch_index = 0
        skip_full_fetch = os.getenv("LUNALIB_SCAN_SKIP_FULL_BLOCK_FETCH", "0") == "1"
        for batch_start in tqdm(range(start_height, end_height + 1, batch_size), desc="Multi-Scan", total=total_batches):
            batch_end = min(batch_start + batch_size - 1, end_height)
            if progress_callback:
                progress_callback({
                    "stage": "scan",
                    "scope": "multi",
                    "addresses": len(addresses),
                    "batch_index": batch_index,
                    "batch_total": total_batches,
                    "start_height": start_height,
                    "end_height": end_height,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "cache_only": cache_only,
                })
            blocks = self.get_blocks_range(batch_start, batch_end, cache_only=cache_only)
            for block in blocks:
                if not isinstance(block, dict):
                    resolved = None
                    if isinstance(block, (int, str)):
                        resolved = self.get_block(block)
                    if isinstance(resolved, dict):
                        block = resolved
                    else:
                        continue

                height = block.get('index', block.get('height'))
                has_txs = isinstance(block.get('transactions'), list)
                has_meta = bool(self._extract_miner_address(block)) or (self._extract_reward_amount(block) > 0)
                if not skip_full_fetch and (not has_txs or not has_meta):
                    full_block = None
                    if cache_only:
                        if isinstance(height, int):
                            full_block = self.cache.get_block(height)
                    else:
                        if isinstance(height, int):
                            full_block = self.get_block(height)
                        elif block.get('hash'):
                            full_block = self.get_block(block.get('hash'))
                    if isinstance(full_block, dict):
                        block = full_block
                collected = self._collect_transactions_for_addresses(block, normalized_map)
                # Fallback: ensure miner reward is captured even if block shape differs
                try:
                    miner_raw = self._extract_miner_address(block)
                    miner_norm = self._normalize_address(miner_raw or '')
                    reward_amount = self._extract_reward_amount(block)
                    if miner_norm in normalized_map and reward_amount > 0:
                        target_addr = normalized_map[miner_norm]
                        reward_hash = f"reward_{block.get('index')}_{block.get('hash', '')[:8]}"
                        existing = collected.get(target_addr, [])
                        if not any(tx.get('hash') == reward_hash for tx in existing):
                            reward_tx = {
                                'type': 'reward',
                                'from': 'network',
                                'to': target_addr,
                                'amount': reward_amount,
                                'block_height': block.get('index'),
                                'timestamp': block.get('timestamp'),
                                'hash': reward_hash,
                                'status': 'confirmed',
                                'description': f"Mining reward for block #{block.get('index')}",
                                'direction': 'incoming',
                                'effective_amount': reward_amount,
                                'fee': 0
                            }
                            collected.setdefault(target_addr, []).append(reward_tx)
                except Exception:
                    pass
                for original_addr, txs in collected.items():
                    if txs:
                        results[original_addr].extend(txs)
            batch_index += 1
            if progress_callback:
                total_txs = sum(len(txs) for txs in results.values())
                progress_callback({
                    "stage": "scan",
                    "scope": "multi",
                    "addresses": len(addresses),
                    "batch_index": batch_index,
                    "batch_total": total_batches,
                    "start_height": start_height,
                    "end_height": end_height,
                    "scanned_end": batch_end,
                    "tx_count": total_txs,
                    "cache_only": cache_only,
                })

        max_tx = int(os.getenv("LUNALIB_SCAN_TX_LIMIT", "5000"))
        if max_tx > 0:
            for addr in addresses:
                txs = results.get(addr, [])
                if len(txs) > max_tx:
                    txs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    results[addr] = txs[:max_tx]

        # Summary
        total_txs = sum(len(txs) for txs in results.values())
        print(f"[MULTI-SCAN] Found {total_txs} total transactions")
        for addr in addresses:
            print(f"  - {addr}: {len(results[addr])} transactions")

        return results

    def scan_transactions_for_addresses_filtered(
        self,
        addresses: List[str],
        start_height: int = 0,
        end_height: int = None,
        include_rewards: bool = True,
        include_transfers: bool = True,
        include_gtx_genesis: bool = True,
        cache_only: bool = False,
        max_range: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, List[Dict]]:
        """Scan blockchain and filter results by transaction category."""
        results = self.scan_transactions_for_addresses(
            addresses,
            start_height,
            end_height,
            cache_only=cache_only,
            max_range=max_range,
            progress_callback=progress_callback,
        )
        if include_rewards and include_transfers and include_gtx_genesis:
            return results
        filtered: Dict[str, List[Dict]] = {}
        for addr, txs in results.items():
            filtered_txs = self._filter_transactions(
                txs,
                include_rewards=include_rewards,
                include_transfers=include_transfers,
                include_gtx_genesis=include_gtx_genesis,
            )
            if filtered_txs:
                filtered[addr] = filtered_txs
        return filtered

    def scan_transactions_for_address_filtered(
        self,
        address: str,
        start_height: int = 0,
        end_height: int = None,
        include_rewards: bool = True,
        include_transfers: bool = True,
        include_gtx_genesis: bool = True,
        cache_only: bool = False,
        max_range: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Scan blockchain for a single address with category filters."""
        results = self.scan_transactions_for_addresses_filtered(
            [address],
            start_height=start_height,
            end_height=end_height,
            include_rewards=include_rewards,
            include_transfers=include_transfers,
            include_gtx_genesis=include_gtx_genesis,
            cache_only=cache_only,
            max_range=max_range,
            progress_callback=progress_callback,
        )
        return results.get(address, [])
    
    def scan_transactions_for_addresses_async(self, addresses: List[str], callback: Callable = None,
                                             start_height: int = 0, end_height: int = None) -> str:
        """Async version: Scan blockchain for multiple addresses in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"multi_scan_{int(time.time())}"
        
        def _scan_task():
            try:
                result = self.scan_transactions_for_addresses(addresses, start_height, end_height)
                if callback:
                    callback(success=True, result=result, error=None)
                return result
            except Exception as e:
                print(f"❌ Async multi-scan error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return None
        
        future = self.executor.submit(_scan_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async multi-scan task: {task_id}")
        return task_id

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
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running async task
        
        Returns: True if cancelled, False otherwise
        """
        if task_id in self._async_tasks:
            future = self._async_tasks[task_id]
            if future.cancel():
                del self._async_tasks[task_id]
                print(f"✅ Task cancelled: {task_id}")
                return True
        return False
    
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
    
    def shutdown(self):
        """Shutdown the thread pool executor"""
        print("🛑 Shutting down BlockchainManager executor...")
        self.executor.shutdown(wait=True)
        print("✅ Executor shutdown complete")
    
    def monitor_addresses(self, addresses: List[str], on_update, poll_interval: int = 15):
        """Start background monitor for addresses; returns a stop event."""
        import threading
        from lunalib.core.mempool import MempoolManager

        stop_event = threading.Event()
        self._stop_events.append(stop_event)

        mempool = MempoolManager()
        chain_height = self.get_blockchain_height()
        cached_height = self.cache.get_highest_cached_height()
        last_height = cached_height if 0 <= cached_height < chain_height else chain_height

        def _emit_update(confirmed_map: Dict[str, List[Dict]], pending_map: Dict[str, List[Dict]], source: str):
            try:
                if on_update:
                    on_update({
                        'confirmed': confirmed_map or {},
                        'pending': pending_map or {},
                        'source': source
                    })
            except Exception as e:
                print(f"Monitor callback error: {e}")

        def _monitor_loop():
            nonlocal last_height

            # Initial emission: existing chain + current mempool
            initial_confirmed = self.scan_transactions_for_addresses(addresses, 0, last_height)
            initial_pending = mempool.get_pending_transactions_for_addresses(addresses)
            _emit_update(initial_confirmed, initial_pending, source="initial")

            while not stop_event.wait(poll_interval):
                try:
                    current_height = self.get_blockchain_height()
                    if current_height > last_height:
                        new_confirmed = self.scan_transactions_for_addresses(addresses, last_height + 1, current_height)
                        if any(new_confirmed.values()):
                            _emit_update(new_confirmed, {}, source="blockchain")
                        last_height = current_height

                    pending_now = mempool.get_pending_transactions_for_addresses(addresses)
                    if any(pending_now.values()):
                        _emit_update({}, pending_now, source="mempool")

                except Exception as e:
                    print(f"Monitor loop error: {e}")

        thread = threading.Thread(target=_monitor_loop, daemon=True)
        thread.start()
        return stop_event

    def submit_mined_block_async(self, block_data: Dict, callback: Callable = None) -> str:
        """Async version: Submit mined block in background thread
        
        Returns: task_id that can be used to check status
        """
        task_id = f"submit_block_{block_data.get('index', 'unknown')}_{int(time.time())}"
        
        def _submit_task():
            try:
                success = self.submit_mined_block(block_data)
                if callback:
                    callback(success=success, result=block_data if success else None, error=None if success else "Submission failed")
                return success
            except Exception as e:
                print(f"❌ Async block submission error: {e}")
                if callback:
                    callback(success=False, result=None, error=str(e))
                return False
        
        future = self.executor.submit(_submit_task)
        self._async_tasks[task_id] = future
        print(f"🔄 Started async block submission: {task_id}")
        return task_id
    
    def submit_mined_block(self, block_data: Dict) -> bool:
        """Submit a mined block to the network with built-in validation"""
        try:
            print(f"🔄 Preparing to submit block #{block_data.get('index')}...")
            
            # Step 1: Validate block structure before submission
            validation_result = self._validate_block_structure(block_data)
            if not validation_result['valid']:
                print(f"❌ Block validation failed:")
                for issue in validation_result['issues']:
                    print(f"   - {issue}")
                return False
            
            print(f"✅ Block structure validation passed")
            print(f"   Block #{block_data.get('index')} | Hash: {block_data.get('hash', '')[:16]}...")
            print(f"   Transactions: {len(block_data.get('transactions', []))} | Difficulty: {block_data.get('difficulty')}")
            
            # Step 2: Submit to the correct endpoint (gzip first, then plain JSON fallback)
            timeout = float(os.getenv("LUNALIB_BLOCK_SUBMIT_TIMEOUT", "30"))

            def _post(payload: bytes, headers: Dict[str, str]) -> Optional[requests.Response]:
                try:
                    return self._session.post(
                        f'{self.endpoint_url}/blockchain/submit-block',
                        data=payload,
                        headers=headers,
                        timeout=timeout
                    )
                except requests.exceptions.RequestException as e:
                    print(f"❌ Block submit request error: {e}")
                    return None

            raw = json.dumps(block_data).encode("utf-8")
            gz = gzip.compress(raw)
            # Try gzip first, with correct headers
            response = _post(
                gz,
                {
                    'Content-Type': 'application/json',
                    'Content-Encoding': 'gzip',
                    'Accept-Encoding': 'gzip, deflate',
                }
            )

            # Fallback to plain JSON if gzip fails or non-2xx
            if response is None or response.status_code not in [200, 201]:
                if response is not None:
                    print(f"⚠️ Gzip submit failed: HTTP {response.status_code} - {response.text}")
                response = _post(
                    raw,
                    {
                        'Content-Type': 'application/json',
                        'Accept-Encoding': 'gzip, deflate',
                    }
                )

            # Step 3: Handle response
            if response and response.status_code in [200, 201]:
                try:
                    result = response.json()
                except Exception:
                    print(f"🎉 Block #{block_data.get('index')} submitted (non-JSON response)")
                    return True

                if result.get('success'):
                    print(f"🎉 Block #{block_data.get('index')} successfully added to blockchain!")
                    print(f"   Block hash: {result.get('block_hash', '')[:16]}...")
                    print(f"   Transactions count: {result.get('transactions_count', 0)}")
                    print(f"   Miner: {result.get('miner', 'unknown')}")
                    return True

                error_msg = result.get('error', 'Unknown error')
                print(f"❌ Block submission rejected: {error_msg}")
                return False

            # If submission failed, check if block landed anyway (race/timeout)
            if self._confirm_block_submission(block_data):
                print(f"✅ Block #{block_data.get('index')} confirmed on chain after submit retry")
                return True

            if response is not None:
                print(f"❌ HTTP error {response.status_code}: {response.text}")
            return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error submitting block: {e}")
            return False
        except Exception as e:
            print(f"💥 Unexpected error submitting block: {e}")
            return False

    def _confirm_block_submission(self, block_data: Dict) -> bool:
        """Confirm block made it on-chain after a submit timeout or error."""
        try:
            retries = int(os.getenv("LUNALIB_SUBMIT_CONFIRM_RETRIES", "3"))
            delay = float(os.getenv("LUNALIB_SUBMIT_CONFIRM_DELAY", "1.0"))
            target_hash = block_data.get("hash")
            target_index = block_data.get("index")

            for _ in range(max(1, retries)):
                if target_hash:
                    block = self.get_block_by_hash(target_hash)
                    if block and block.get("hash") == target_hash:
                        return True
                if isinstance(target_index, int):
                    block = self.get_block_by_height(target_index)
                    if block and block.get("hash") == target_hash:
                        return True
                time.sleep(delay)
        except Exception:
            return False
        return False

    def _validate_block_structure(self, block_data: Dict) -> Dict:
        """Internal: Validate block structure before submission"""
        issues = []
        
        # Check required fields
        required_fields = ["index", "previous_hash", "timestamp", "transactions", "miner", "difficulty", "nonce", "hash"]
        missing_fields = [field for field in required_fields if field not in block_data]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Check data types
        if not isinstance(block_data.get('index'), int) or block_data.get('index') < 0:
            issues.append("Index must be a non-negative integer")
        
        if not isinstance(block_data.get('transactions'), list):
            issues.append("Transactions must be a list")
        
        if not isinstance(block_data.get('difficulty'), int) or block_data.get('difficulty') < 0:
            issues.append("Difficulty must be a non-negative integer")
        
        if not isinstance(block_data.get('nonce'), int) or block_data.get('nonce') < 0:
            issues.append("Nonce must be a non-negative integer")
        
        # Check hash meets difficulty requirement
        block_hash = block_data.get('hash', '')
        difficulty = block_data.get('difficulty', 0)
        if difficulty > 0 and not block_hash.startswith('0' * difficulty):
            issues.append(f"Hash doesn't meet difficulty {difficulty}: {block_hash[:16]}...")
        
        # Check hash length (should be 64 chars for SHA-256)
        if len(block_hash) != 64:
            issues.append(f"Hash should be 64 characters, got {len(block_hash)}")
        
        # Check previous hash format
        previous_hash = block_data.get('previous_hash', '')
        if len(previous_hash) != 64 and previous_hash != '0' * 64:  # Allow genesis block
            issues.append(f"Previous hash should be 64 characters, got {len(previous_hash)}")
        
        # Check timestamp is reasonable
        current_time = time.time()
        block_time = block_data.get('timestamp', 0)
        if block_time > current_time + 300:  # 5 minutes in future
            issues.append(f"Block timestamp is in the future")
        if block_time < current_time - 86400:  # 24 hours in past  
            issues.append(f"Block timestamp is too far in the past")
        
        # Validate transactions structure
        transactions = block_data.get('transactions', [])
        for i, tx in enumerate(transactions):
            if not isinstance(tx, dict):
                issues.append(f"Transaction {i} is not a dictionary")
                continue
            
            tx_type = tx.get('type')
            if not tx_type:
                issues.append(f"Transaction {i} missing 'type' field")
            
            # Basic transaction validation
            if tx_type == 'GTX_Genesis':
                required_tx_fields = ['serial_number', 'denomination', 'issued_to', 'timestamp', 'hash']
                missing_tx_fields = [field for field in required_tx_fields if field not in tx]
                if missing_tx_fields:
                    issues.append(f"GTX_Genesis transaction {i} missing fields: {missing_tx_fields}")
            
            elif tx_type == 'reward':
                required_tx_fields = ['to', 'amount', 'timestamp', 'hash']
                missing_tx_fields = [field for field in required_tx_fields if field not in tx]
                if missing_tx_fields:
                    issues.append(f"Reward transaction {i} missing fields: {missing_tx_fields}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'block_info': {
                'index': block_data.get('index'),
                'hash_preview': block_data.get('hash', '')[:16] + '...',
                'transaction_count': len(transactions),
                'difficulty': block_data.get('difficulty'),
                'miner': block_data.get('miner'),
                'nonce': block_data.get('nonce')
            }
        }

    def _collect_transactions_for_addresses(self, block: Dict, normalized_map: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Collect transactions in a block for multiple addresses in one pass."""
        results: Dict[str, List[Dict]] = {original: [] for original in normalized_map.values()}

        # Mining reward via block metadata (support aliases)
        miner_raw = self._extract_miner_address(block)
        miner_norm = self._normalize_address(miner_raw or '')
        if miner_norm in normalized_map:
            reward_amount = self._extract_reward_amount(block)
            if reward_amount > 0:
                target_addr = normalized_map[miner_norm]
                reward_tx = {
                    'type': 'reward',
                    'from': 'network',
                    'to': target_addr,
                    'amount': reward_amount,
                    'block_height': block.get('index'),
                    'timestamp': block.get('timestamp'),
                    'hash': f"reward_{block.get('index')}_{block.get('hash', '')[:8]}",
                    'status': 'confirmed',
                    'description': f"Mining reward for block #{block.get('index')}",
                    'direction': 'incoming',
                    'effective_amount': reward_amount,
                    'fee': 0
                }
                results[target_addr].append(reward_tx)

        # Regular transactions
        for tx_index, tx in enumerate(block.get('transactions', [])):
            tx_type = (tx.get('type') or 'transfer').lower()
            from_norm = self._normalize_address(tx.get('from') or tx.get('sender') or '')
            to_norm = self._normalize_address(tx.get('to') or tx.get('receiver') or '')

            desc = str(tx.get('description', '') or '').lower()
            reward_hint = (
                tx_type in {'reward', 'coinbase', 'mining_reward'}
                or (tx.get('hash', '') or '').startswith('reward_')
                or (tx.get('from') == 'network' and ('reward' in desc or 'mining' in desc))
                or (tx.get('from') == 'network' and tx.get('block_height') is not None)
                or ('reward' in tx)
            )

            # Explicit reward transaction (support aliases)
            if reward_hint:
                reward_to = tx.get('to') or tx.get('receiver') or tx.get('issued_to') or tx.get('owner_address') or tx.get('to_address')
                reward_to_norm = self._normalize_address(reward_to or '')
                if reward_to_norm in normalized_map:
                    target_addr = normalized_map[reward_to_norm]
                    amount = self._parse_amount(tx.get('amount', tx.get('denomination', tx.get('reward', 0)) or 0) or 0)
                    enhanced = tx.copy()
                    enhanced.update({
                        'to': reward_to,
                        'block_height': block.get('index'),
                        'status': 'confirmed',
                        'tx_index': tx_index,
                        'direction': 'incoming',
                        'effective_amount': amount,
                        'amount': amount,
                        'fee': 0,
                    })
                    enhanced.setdefault('from', 'network')
                    results[target_addr].append(enhanced)
                    continue

            # Incoming transfer
            if to_norm in normalized_map:
                target_addr = normalized_map[to_norm]
                amount = self._parse_amount(tx.get('amount', 0) or 0)
                fee = self._parse_amount(tx.get('fee', 0) or tx.get('gas', 0) or 0)
                enhanced = tx.copy()
                enhanced.update({
                    'block_height': block.get('index'),
                    'status': 'confirmed',
                    'tx_index': tx_index,
                    'direction': 'incoming',
                    'effective_amount': amount,
                    'amount': amount,
                    'fee': fee
                })
                results[target_addr].append(enhanced)

            # Outgoing transfer
            if from_norm in normalized_map:
                target_addr = normalized_map[from_norm]
                amount = self._parse_amount(tx.get('amount', 0) or 0)
                fee = self._parse_amount(tx.get('fee', 0) or tx.get('gas', 0) or 0)
                enhanced = tx.copy()
                enhanced.update({
                    'block_height': block.get('index'),
                    'status': 'confirmed',
                    'tx_index': tx_index,
                    'direction': 'outgoing',
                    'effective_amount': -(amount + fee),
                    'amount': amount,
                    'fee': fee
                })
                results[target_addr].append(enhanced)

        # Trim empty entries
        return {addr: txs for addr, txs in results.items() if txs}

    def _find_address_transactions(self, block: Dict, address: str) -> List[Dict]:
        """Find transactions in block that involve the address - FIXED REWARD DETECTION"""
        transactions = []
        address_lower = address.lower().strip('"\'')  # Remove quotes if present
        
        print(f"🔍 Scanning block #{block.get('index')} for address: {address}")
        print(f"   Block data: {block}")
        
        # ==================================================================
        # 1. CHECK BLOCK MINING REWARD (from block metadata)
        # ==================================================================
        miner = self._extract_miner_address(block) or ''
        # Clean the miner address (remove quotes, trim)
        miner_clean = str(miner).strip('"\' ')
        
        print(f"   Miner in block: '{miner_clean}'")
        print(f"   Our address: '{address_lower}'")
        print(f"   Block reward: {self._extract_reward_amount(block)}")
        
        # Function to normalize addresses for comparison
        def normalize_address(addr):
            if not addr:
                return ''
            # Remove LUN_ prefix and quotes, convert to lowercase
            addr_str = str(addr).strip('"\' ').lower()
            # Remove 'lun_' prefix if present
            if addr_str.startswith('lun_'):
                addr_str = addr_str[4:]
            return addr_str
        
        # Normalize both addresses
        miner_normalized = normalize_address(miner_clean)
        address_normalized = normalize_address(address_lower)
        
        print(f"   Miner normalized: '{miner_normalized}'")
        print(f"   Address normalized: '{address_normalized}'")
        
        # Check if this block was mined by our address
        if miner_normalized == address_normalized and miner_normalized:
            reward_amount = self._extract_reward_amount(block)
            if reward_amount > 0:
                reward_tx = {
                    'type': 'reward',
                    'from': 'network',
                    'to': address,
                    'amount': reward_amount,
                    'block_height': block.get('index'),
                    'timestamp': block.get('timestamp'),
                    'hash': f"reward_{block.get('index')}_{block.get('hash', '')[:8]}",
                    'status': 'confirmed',
                    'description': f'Mining reward for block #{block.get("index")}',
                    'direction': 'incoming',
                    'effective_amount': reward_amount,
                    'fee': 0
                }
                transactions.append(reward_tx)
                print(f"🎁 FOUND MINING REWARD: {reward_amount} LUN for block #{block.get('index')}")
                print(f"   Miner match: '{miner_clean}' == '{address}'")
        else:
            print(f"   Not our block - Miner: '{miner_clean}', Our address: '{address}'")
        
        # ==================================================================
        # 2. CHECK ALL TRANSACTIONS IN THE BLOCK
        # ==================================================================
        block_transactions = block.get('transactions', [])
        print(f"   Block has {len(block_transactions)} transactions")
        
        for tx_index, tx in enumerate(block_transactions):
            enhanced_tx = tx.copy()
            enhanced_tx['block_height'] = block.get('index')
            enhanced_tx['status'] = 'confirmed'
            enhanced_tx['tx_index'] = tx_index
            
            # Get transaction type
            tx_type = tx.get('type', 'transfer').lower()
            
            # Helper function for address matching with normalization
            def addresses_match(addr1, addr2):
                if not addr1 or not addr2:
                    return False
                
                # Normalize both addresses
                addr1_norm = normalize_address(addr1)
                addr2_norm = normalize_address(addr2)
                
                # Check if they match
                return addr1_norm == addr2_norm
            
            # ==================================================================
            # A) REWARD TRANSACTIONS (explicit reward transactions)
            # ==================================================================
            tx_type = tx.get('type', 'transfer').lower()
            if tx_type in {'reward', 'coinbase', 'mining_reward'}:
                reward_to_address = tx.get('to') or tx.get('receiver') or tx.get('issued_to') or tx.get('owner_address') or tx.get('to_address') or ''
                # Compare the reward's destination with our wallet address
                if addresses_match(reward_to_address, address):
                    amount = self._parse_amount(tx.get('amount', tx.get('denomination', tx.get('reward', 0)) or 0))
                    enhanced_tx['direction'] = 'incoming'
                    enhanced_tx['effective_amount'] = amount
                    enhanced_tx['fee'] = 0
                    enhanced_tx.setdefault('from', 'network') # Ensure sender is set
                    transactions.append(enhanced_tx)
                    print(f"✅ Found mining reward: {amount} LUN (to: {reward_to_address})")
                    continue  # Move to next transaction
            
            # ==================================================================
            # B) REGULAR TRANSFERS
            # ==================================================================
            from_addr = tx.get('from') or tx.get('sender') or ''
            to_addr = tx.get('to') or tx.get('receiver') or ''
            
            # Check if transaction involves our address
            is_incoming = addresses_match(to_addr, address)
            is_outgoing = addresses_match(from_addr, address)
            
            if is_incoming:
                amount = self._parse_amount(tx.get('amount', 0))
                fee = self._parse_amount(tx.get('fee', 0) or tx.get('gas', 0) or 0)
                
                enhanced_tx['direction'] = 'incoming'
                enhanced_tx['effective_amount'] = amount
                enhanced_tx['amount'] = amount
                enhanced_tx['fee'] = fee
                
                transactions.append(enhanced_tx)
                print(f"⬆️ Found incoming transaction: {amount} LUN")
                
            elif is_outgoing:
                amount = self._parse_amount(tx.get('amount', 0))
                fee = self._parse_amount(tx.get('fee', 0) or tx.get('gas', 0) or 0)
                
                enhanced_tx['direction'] = 'outgoing'
                enhanced_tx['effective_amount'] = -(amount + fee)
                enhanced_tx['amount'] = amount
                enhanced_tx['fee'] = fee
                
                transactions.append(enhanced_tx)
                print(f"⬇️ Found outgoing transaction: {amount} LUN + {fee} fee")
        
        print(f" Scan complete for block #{block.get('index')}: {len(transactions)} transactions found")
        return transactions
    def _handle_regular_transfers(self, tx: Dict, address_lower: str) -> Dict:
        """Handle regular transfer transactions that might be in different formats"""
        enhanced_tx = tx.copy()
        
        # Try to extract addresses from various possible field names
        possible_from_fields = ['from', 'sender', 'from_address', 'source', 'payer']
        possible_to_fields = ['to', 'receiver', 'to_address', 'destination', 'payee']
        possible_amount_fields = ['amount', 'value', 'quantity', 'transfer_amount']
        possible_fee_fields = ['fee', 'gas', 'transaction_fee', 'gas_fee']
        
        # Find from address
        from_addr = ''
        for field in possible_from_fields:
            if field in tx:
                from_addr = (tx.get(field) or '').lower()
                break
        
        # Find to address
        to_addr = ''
        for field in possible_to_fields:
            if field in tx:
                to_addr = (tx.get(field) or '').lower()
                break
        
        # Find amount
        amount = 0
        for field in possible_amount_fields:
            if field in tx:
                amount = self._parse_amount(tx.get(field, 0))
                break
        
        # Find fee
        fee = 0
        for field in possible_fee_fields:
            if field in tx:
                fee = self._parse_amount(tx.get(field, 0))
                break
        
        # Set direction
        if from_addr == address_lower:
            enhanced_tx['direction'] = 'outgoing'
            enhanced_tx['effective_amount'] = -(amount + fee)
            enhanced_tx['from'] = from_addr
            enhanced_tx['to'] = to_addr
            enhanced_tx['amount'] = amount
            enhanced_tx['fee'] = fee
        elif to_addr == address_lower:
            enhanced_tx['direction'] = 'incoming'
            enhanced_tx['effective_amount'] = amount
            enhanced_tx['from'] = from_addr
            enhanced_tx['to'] = to_addr
            enhanced_tx['amount'] = amount
            enhanced_tx['fee'] = fee
        else:
            # If we can't determine direction from addresses, check other fields
            enhanced_tx['direction'] = 'unknown'
            enhanced_tx['effective_amount'] = amount
        
        # Set type if not present
        if not enhanced_tx.get('type'):
            enhanced_tx['type'] = 'transfer'
        
        return enhanced_tx