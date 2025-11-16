from storage.cache import BlockchainCache
import requests
import time
import json
from typing import Dict, List, Optional, Tuple


class BlockchainManager:
    """Manages blockchain interactions and scanning"""
    
    def __init__(self, endpoint_url="https://bank.linglin.art"):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.cache = BlockchainCache()
        self.network_connected = False
        
    def get_blockchain_height(self) -> int:
        """Get current blockchain height"""
        try:
            # Try height endpoint first
            response = requests.get(f'{self.endpoint_url}/blockchain/height', timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('height', 0)
            
            # Fallback to blocks endpoint
            response = requests.get(f'{self.endpoint_url}/blockchain/blocks', timeout=10)
            if response.status_code == 200:
                data = response.json()
                blocks = data.get('blocks', [])
                return len(blocks)
                
        except Exception as e:
            print(f"Blockchain height error: {e}")
            
        return 0
    
    def get_block(self, height: int) -> Optional[Dict]:
        """Get block by height"""
        # Check cache first
        cached_block = self.cache.get_block(height)
        if cached_block:
            return cached_block
            
        try:
            response = requests.get(f'{self.endpoint_url}/blockchain/block/{height}', timeout=10)
            if response.status_code == 200:
                block = response.json()
                self.cache.save_block(height, block.get('hash', ''), block)
                return block
        except Exception as e:
            print(f"Get block error: {e}")
            
        return None
    
    def get_blocks_range(self, start_height: int, end_height: int) -> List[Dict]:
        """Get range of blocks"""
        blocks = []
        
        # Check cache first
        cached_blocks = self.cache.get_block_range(start_height, end_height)
        if len(cached_blocks) == (end_height - start_height + 1):
            return cached_blocks
            
        try:
            response = requests.get(
                f'{self.endpoint_url}/blockchain/range?start={start_height}&end={end_height}',
                timeout=30
            )
            if response.status_code == 200:
                blocks = response.json().get('blocks', [])
                # Cache the blocks
                for block in blocks:
                    height = block.get('index', 0)
                    self.cache.save_block(height, block.get('hash', ''), block)
            else:
                # Fallback: get blocks individually
                for height in range(start_height, end_height + 1):
                    block = self.get_block(height)
                    if block:
                        blocks.append(block)
                    time.sleep(0.01)  # Be nice to the API
                        
        except Exception as e:
            print(f"Get blocks range error: {e}")
            
        return blocks
    
    def get_mempool(self) -> List[Dict]:
        """Get current mempool transactions"""
        try:
            response = requests.get(f'{self.endpoint_url}/mempool', timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Mempool error: {e}")
            
        return []
    
    def broadcast_transaction(self, transaction: Dict) -> bool:
        """Broadcast transaction to network"""
        try:
            response = requests.post(
                f'{self.endpoint_url}/mempool/add',
                json=transaction,
                timeout=30
            )
            return response.status_code == 201
        except Exception as e:
            print(f"Broadcast error: {e}")
            return False
    
    def check_network_connection(self) -> bool:
        """Check if network is accessible"""
        try:
            response = requests.get(f'{self.endpoint_url}/health', timeout=5)
            self.network_connected = response.status_code == 200
            return self.network_connected
        except:
            self.network_connected = False
            return False
    
    def scan_transactions_for_address(self, address: str, start_height: int = 0, end_height: int = None) -> List[Dict]:
        """Scan blockchain for transactions involving an address"""
        if end_height is None:
            end_height = self.get_blockchain_height()
            
        transactions = []
        
        # Scan in batches for efficiency
        batch_size = 100
        for batch_start in range(start_height, end_height + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_height)
            blocks = self.get_blocks_range(batch_start, batch_end)
            
            for block in blocks:
                block_transactions = self._find_address_transactions(block, address)
                transactions.extend(block_transactions)
                
        return transactions
    
    def _find_address_transactions(self, block: Dict, address: str) -> List[Dict]:
        """Find transactions in block that involve the address"""
        transactions = []
        address_lower = address.lower()
        
        # Check block reward
        miner = block.get('miner', '').lower()
        if miner == address_lower:
            reward_tx = {
                'type': 'reward',
                'from': 'network',
                'to': address,
                'amount': block.get('reward', 0),
                'block_height': block.get('index'),
                'timestamp': block.get('timestamp'),
                'hash': f"reward_{block.get('index')}_{address}",
                'status': 'confirmed'
            }
            transactions.append(reward_tx)
        
        # Check regular transactions
        for tx in block.get('transactions', []):
            from_addr = (tx.get('from') or '').lower()
            to_addr = (tx.get('to') or '').lower()
            
            if from_addr == address_lower or to_addr == address_lower:
                enhanced_tx = tx.copy()
                enhanced_tx['block_height'] = block.get('index')
                enhanced_tx['status'] = 'confirmed'
                transactions.append(enhanced_tx)
                
        return transactions