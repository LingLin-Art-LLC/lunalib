# lunalib/core/p2p.py
import time
import requests
import threading
import json
import sys
import os
import random
import gzip
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Callable
from queue import Queue
from collections import deque
import hashlib
from lunalib.utils.hash import sm3_hex

try:
    import msgpack  # type: ignore
    _HAS_MSGPACK = True
except Exception:
    msgpack = None
    _HAS_MSGPACK = False


class P2PClient:
    """
    P2P client for blockchain synchronization with hybrid primary/peer architecture.
    Downloads initial state from primary node, then syncs via P2P with periodic validation.
    """
    
    def __init__(
        self,
        primary_node_url: str,
        node_id: Optional[str] = None,
        peer_url: Optional[str] = None,
        peer_seed_urls: Optional[List[str]] = None,
        prefer_peers: bool = False,
    ):
        self.primary_node = primary_node_url
        self.node_id = node_id or self._generate_node_id()
        self.peer_url = peer_url or self._generate_peer_url()
        self.peer_seed_urls = peer_seed_urls or []
        self.prefer_peers = prefer_peers
        self.peers = []
        self.last_primary_check = 0
        self.last_peer_update = 0
        self.peer_update_interval = 300  # 5 minutes
        self.primary_check_interval = 3600  # 1 hour
        self.gossip_fanout = int(os.getenv("LUNALIB_GOSSIP_FANOUT", "8"))
        self._seen_tx_max = int(os.getenv("LUNALIB_SEEN_TX_MAX", "50000"))
        self._seen_block_max = int(os.getenv("LUNALIB_SEEN_BLOCK_MAX", "5000"))
        self.use_msgpack = bool(int(os.getenv("LUNALIB_USE_MSGPACK", "0"))) and _HAS_MSGPACK
        self.p2p_gzip = bool(int(os.getenv("LUNALIB_P2P_GZIP", "1")))
        self._seen_txs = set()
        self._seen_blocks = set()
        self._seen_tx_order = deque()
        self._seen_block_order = deque()
        
        # Callbacks for events
        self.on_new_block_callback = None
        self.on_new_transaction_callback = None
        self.on_peer_update_callback = None
        
        # P2P message queue
        self.message_queue = Queue()
        self.is_running = False
        self.sync_thread = None
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import socket
        hostname = socket.gethostname()
        timestamp = str(time.time())
        return sm3_hex(f"{hostname}{timestamp}".encode())[:16]
    
    def _generate_peer_url(self) -> str:
        """Generate peer URL (defaults to localhost, should be overridden for public nodes)"""
        import socket
        try:
            # Try to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return f"http://{local_ip}:8080"
        except:
            return "http://localhost:8080"

    def _remember_seen(self, key: str, cache: set, order: deque, max_size: int) -> bool:
        if not key:
            return True
        if key in cache:
            return False
        cache.add(key)
        order.append(key)
        if len(order) > max_size:
            old = order.popleft()
            cache.discard(old)
        return True

    def _select_peers(self) -> List[Dict]:
        peers = list(self.peers)
        fanout = max(0, self.gossip_fanout)
        if fanout and len(peers) > fanout:
            return random.sample(peers, fanout)
        return peers

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
    
    def start(self):
        """Start P2P client"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initial setup
        self._initial_sync()
        self._register_with_primary()
        self._update_peer_list()
        
        # Start background sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        print(f"✅ P2P Client started (Node ID: {self.node_id})")
    
    def stop(self):
        """Stop P2P client"""
        self.is_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        print("🛑 P2P Client stopped")
    
    def _initial_sync(self):
        """Download initial blockchain, favoring primary but falling back to peers."""
        def _peer_urls():
            urls = []
            for peer in self.peers:
                peer_url = peer.get("url") or peer.get("peer_url")
                if peer_url:
                    urls.append(peer_url)
            urls.extend(self.peer_seed_urls)
            return list(dict.fromkeys(urls))

        if self.prefer_peers:
            data = self._fetch_chain_from_peers(_peer_urls())
            if data:
                return data

        data = self._fetch_chain_from_primary()
        if data:
            return data

        return self._fetch_chain_from_peers(_peer_urls())

    def _fetch_chain_from_primary(self):
        try:
            print(f"📥 Initial sync from primary node: {self.primary_node}")
            return self._fetch_chain_from_base(self.primary_node)
        except Exception as e:
            print(f"❌ Primary sync failed: {e}")
            return None

    def _fetch_chain_from_peers(self, peer_urls: List[str]):
        if not peer_urls:
            return None
        print(f"📥 Initial sync from peers: {len(peer_urls)} candidates")
        for peer_url in peer_urls:
            try:
                data = self._fetch_chain_from_base(peer_url)
                if data:
                    print(f"✅ Downloaded blockchain from peer: {peer_url}")
                    return data
            except Exception:
                continue
        print("❌ Peer sync failed")
        return None

    def _fetch_chain_from_base(self, base_url: str) -> Optional[Dict]:
        endpoints = [
            f"{base_url}/blockchain",
            f"{base_url}/api/blockchain/full",
        ]
        for url in endpoints:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                blockchain_data = response.json()
                if isinstance(blockchain_data, list):
                    print(f"✅ Downloaded blockchain: {len(blockchain_data)} blocks")
                    return {"blocks": blockchain_data}
                print(
                    f"✅ Downloaded blockchain: {len(blockchain_data.get('blocks', []))} blocks"
                )
                return blockchain_data
        return None
    
    def _register_with_primary(self):
        """Register this node with the primary daemon"""
        try:
            peer_info = {
                'node_id': self.node_id,
                'peer_url': self.peer_url,
                'timestamp': time.time(),
                'capabilities': ['sync', 'relay']
            }
            
            response = requests.post(
                f"{self.primary_node}/api/peers/register",
                json=peer_info,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Registered with primary node as peer: {self.node_id} ({self.peer_url})")
                return True
            else:
                print(f"⚠️  Registration failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def _update_peer_list(self):
        """Get updated peer list from primary daemon"""
        try:
            response = requests.get(f"{self.primary_node}/api/peers/list", timeout=10)
            
            if response.status_code == 200:
                peer_data = response.json()
                new_peers = peer_data.get('peers', []) if isinstance(peer_data, dict) else peer_data

                normalized = []
                for peer in new_peers or []:
                    if isinstance(peer, str):
                        normalized.append({"node_id": None, "url": peer})
                    elif isinstance(peer, dict):
                        normalized.append(peer)

                # Filter out self
                self.peers = [p for p in normalized if p.get('node_id') != self.node_id]
                
                print(f"📋 Updated peer list: {len(self.peers)} peers")
                self.last_peer_update = time.time()
                
                if self.on_peer_update_callback:
                    self.on_peer_update_callback(self.peers)
                
                return self.peers
                
        except Exception as e:
            print(f"❌ Peer list update failed: {e}")
            return []
    
    def _sync_loop(self):
        """Background sync loop for P2P updates"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Periodic peer list update
                if current_time - self.last_peer_update > self.peer_update_interval:
                    self._update_peer_list()
                
                # Periodic primary node validation
                if current_time - self.last_primary_check > self.primary_check_interval:
                    self._validate_with_primary()
                
                # Sync with primary/base before peers
                self._sync_from_primary()

                # Sync with peers
                self._sync_from_peers()
                
                # Process message queue
                self._process_messages()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Sync loop error: {e}")
                time.sleep(10)

    def _sync_from_primary(self):
        """Sync latest block from primary/base URL before peers."""
        try:
            response = requests.get(f"{self.primary_node}/api/blockchain/latest", timeout=5)
            if response.status_code == 200:
                latest_block = response.json()
                if self.on_new_block_callback:
                    self.on_new_block_callback(latest_block)
        except Exception:
            pass
    
    def _sync_from_peers(self):
        """Sync new blocks and transactions from peers"""
        for peer in self.peers[:5]:  # Sync with first 5 peers
            try:
                peer_url = peer.get('url') or peer.get('peer_url')
                if not peer_url:
                    continue
                
                # Get latest block from peer
                response = requests.get(f"{peer_url}/api/blockchain/latest", timeout=5)
                if response.status_code == 200:
                    latest_block = response.json()
                    
                    if self.on_new_block_callback:
                        self.on_new_block_callback(latest_block)
                
            except Exception as e:
                # Silent fail for peer sync
                pass
    
    def _validate_with_primary(self):
        """Validate local blockchain state with primary node"""
        try:
            print("🔍 Validating with primary node...")
            
            response = requests.get(f"{self.primary_node}/api/blockchain/latest", timeout=10)
            if response.status_code == 200:
                primary_latest = response.json()
                primary_hash = primary_latest.get('hash')
                primary_height = primary_latest.get('index')
                
                print(f"✅ Primary validation: Block #{primary_height}, Hash: {primary_hash[:16]}...")
                self.last_primary_check = time.time()
                
                return primary_latest
                
        except Exception as e:
            print(f"❌ Primary validation failed: {e}")
            return None
    
    def _process_messages(self):
        """Process incoming P2P messages"""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                msg_type = message.get('type')
                
                if msg_type == 'new_block':
                    if self.on_new_block_callback:
                        self.on_new_block_callback(message.get('data'))
                elif msg_type == 'new_transaction':
                    if self.on_new_transaction_callback:
                        self.on_new_transaction_callback(message.get('data'))
                        
            except Exception as e:
                print(f"❌ Message processing error: {e}")
    
    def broadcast_block(self, block: Dict):
        """Broadcast new block to peers"""
        block_hash = block.get("hash") or block.get("block_hash")
        if block_hash and not self._remember_seen(block_hash, self._seen_blocks, self._seen_block_order, self._seen_block_max):
            return

        targets = self._select_peers()
        if not targets:
            return

        body, headers = self._encode_payload(block, gzip_body=False)

        def _send(peer_info):
            try:
                peer_url = peer_info.get('url') or peer_info.get('peer_url')
                if peer_url:
                    requests.post(
                        f"{peer_url}/api/blocks/new",
                        data=body,
                        headers=headers,
                        timeout=3
                    )
            except Exception:
                pass

        max_workers = min(8, len(targets)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_send, targets))
    
    def broadcast_transaction(self, transaction: Dict):
        """Broadcast new transaction to peers"""
        tx_hash = transaction.get("hash")
        if tx_hash and not self._remember_seen(tx_hash, self._seen_txs, self._seen_tx_order, self._seen_tx_max):
            return

        targets = self._select_peers()
        if not targets:
            return

        body, headers = self._encode_payload(transaction, gzip_body=False)

        def _send(peer_info):
            try:
                peer_url = peer_info.get('url') or peer_info.get('peer_url')
                if peer_url:
                    requests.post(
                        f"{peer_url}/api/transactions/new",
                        data=body,
                        headers=headers,
                        timeout=3
                    )
            except Exception:
                pass

        max_workers = min(8, len(targets)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_send, targets))

    def broadcast_transactions_batch(self, transactions: List[Dict]):
        """Broadcast a batch of transactions to peers."""
        if not transactions:
            return

        fresh: List[Dict] = []
        for tx in transactions:
            tx_hash = tx.get("hash")
            if tx_hash and not self._remember_seen(tx_hash, self._seen_txs, self._seen_tx_order, self._seen_tx_max):
                continue
            fresh.append(tx)

        if not fresh:
            return

        targets = self._select_peers()
        if not targets:
            return

        payload = {"transactions": fresh}
        body, headers = self._encode_payload(payload, gzip_body=True)

        def _send(peer_info):
            try:
                peer_url = peer_info.get('url') or peer_info.get('peer_url')
                if peer_url:
                    requests.post(
                        f"{peer_url}/api/transactions/new/batch",
                        data=body,
                        headers=headers,
                        timeout=5
                    )
            except Exception:
                pass

        max_workers = min(8, len(targets)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_send, targets))
    
    def get_peers(self) -> List[Dict]:
        """Get current peer list"""
        return self.peers
    
    def set_callbacks(self, on_new_block=None, on_new_transaction=None, on_peer_update=None):
        """Set event callbacks"""
        self.on_new_block_callback = on_new_block
        self.on_new_transaction_callback = on_new_transaction
        self.on_peer_update_callback = on_peer_update


class HybridBlockchainClient:
    """
    Hybrid blockchain client that combines primary node trust with P2P scalability.
    - Initial sync from primary node
    - P2P updates from peers
    - Periodic validation against primary node
    """
    
    def __init__(self, primary_node_url: str, blockchain_manager, mempool_manager, peer_url: Optional[str] = None):
        self.primary_node = primary_node_url
        self.blockchain = blockchain_manager
        self.mempool = mempool_manager
        
        # Initialize P2P client
        self.p2p = P2PClient(primary_node_url, peer_url=peer_url)
        self.p2p.set_callbacks(
            on_new_block=self._handle_new_block,
            on_new_transaction=self._handle_new_transaction,
            on_peer_update=self._handle_peer_update
        )
    
    def start(self):
        """Start hybrid client"""
        print("🚀 Starting Hybrid Blockchain Client...")
        self.p2p.start()
    
    def stop(self):
        """Stop hybrid client"""
        self.p2p.stop()
    
    def _handle_new_block(self, block: Dict):
        """Handle new block from P2P network"""
        try:
            # Validate block before adding
            if self._validate_block_with_primary(block):
                print(f"📦 New block from P2P: #{block.get('index')}")
                # Add to local blockchain
                # self.blockchain.add_block(block)
        except Exception as e:
            print(f"❌ Block handling error: {e}")
    
    def _handle_new_transaction(self, transaction: Dict):
        """Handle new transaction from P2P network"""
        try:
            # Validate transaction with primary node
            if self._validate_transaction_with_primary(transaction):
                print(f"💳 New transaction from P2P: {transaction.get('hash', 'unknown')[:16]}...")
                # Add to local mempool
                # self.mempool.add_transaction(transaction)
        except Exception as e:
            print(f"❌ Transaction handling error: {e}")
    
    def _handle_peer_update(self, peers: List[Dict]):
        """Handle peer list update"""
        print(f"👥 Peer list updated: {len(peers)} peers available")
    
    def _validate_block_with_primary(self, block: Dict) -> bool:
        """Validate block against primary node"""
        try:
            response = requests.post(
                f"{self.primary_node}/api/blocks/validate",
                json=block,
                timeout=5
            )
            return response.status_code == 200 and response.json().get('valid', False)
        except Exception:
            print("⚠️  Primary validation unavailable; accepting P2P block as unverified")
            return True
    
    def _validate_transaction_with_primary(self, transaction: Dict) -> bool:
        """Validate transaction against primary node"""
        try:
            response = requests.post(
                f"{self.primary_node}/api/transactions/validate",
                json=transaction,
                timeout=5
            )
            return response.status_code == 200 and response.json().get('valid', False)
        except Exception:
            print("⚠️  Primary validation unavailable; accepting P2P transaction as unverified")
            return True
    
    def broadcast_block(self, block: Dict):
        """Broadcast block to P2P network"""
        self.p2p.broadcast_block(block)
    
    def broadcast_transaction(self, transaction: Dict):
        """Broadcast transaction to P2P network"""
        self.p2p.broadcast_transaction(transaction)

    def broadcast_transactions_batch(self, transactions: List[Dict]):
        """Broadcast batch of transactions to P2P network"""
        self.p2p.broadcast_transactions_batch(transactions)
    
    def get_peers(self) -> List[Dict]:
        """Get current peer list"""
        return self.p2p.get_peers()
