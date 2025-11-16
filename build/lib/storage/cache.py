import os
import sqlite3
import pickle
import gzip
import time
from typing import Dict, List, Optional

class BlockchainCache:
    """Caches blockchain data to avoid repeated network requests"""
    
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".luna_wallet")
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "blockchain_cache.db")
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                height INTEGER PRIMARY KEY,
                hash TEXT UNIQUE,
                block_data BLOB,
                timestamp REAL,
                last_accessed REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mempool (
                tx_hash TEXT PRIMARY KEY,
                tx_data BLOB,
                received_time REAL,
                address_involved TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_block(self, height: int, block_hash: str, block_data: Dict):
        """Save block to cache"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            compressed_data = gzip.compress(pickle.dumps(block_data))
            
            cursor.execute('''
                INSERT OR REPLACE INTO blocks 
                (height, hash, block_data, timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?)
            ''', (height, block_hash, compressed_data, time.time(), time.time()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def get_block(self, height: int) -> Optional[Dict]:
        """Get block from cache"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('SELECT block_data FROM blocks WHERE height = ?', (height,))
            result = cursor.fetchone()
            
            if result:
                # Update access time
                cursor.execute('UPDATE blocks SET last_accessed = ? WHERE height = ?', 
                             (time.time(), height))
                conn.commit()
                conn.close()
                
                return pickle.loads(gzip.decompress(result[0]))
                
            conn.close()
        except Exception as e:
            print(f"Cache read error: {e}")
            
        return None
    
    def get_block_range(self, start_height: int, end_height: int) -> List[Dict]:
        """Get multiple blocks from cache"""
        blocks = []
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT height, block_data FROM blocks 
                WHERE height BETWEEN ? AND ? 
                ORDER BY height
            ''', (start_height, end_height))
            
            results = cursor.fetchall()
            conn.close()
            
            for height, block_data in results:
                try:
                    block = pickle.loads(gzip.decompress(block_data))
                    blocks.append(block)
                except:
                    continue
                    
        except Exception as e:
            print(f"Block range cache error: {e}")
            
        return blocks
    
    def get_highest_cached_height(self) -> int:
        """Get the highest block height in cache"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('SELECT MAX(height) FROM blocks')
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] is not None else -1
        except:
            return -1
    
    def clear_old_blocks(self, max_age_hours=24):
        """Clear blocks older than specified hours"""
        try:
            cutoff = time.time() - (max_age_hours * 3600)
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM blocks WHERE last_accessed < ?', (cutoff,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache cleanup error: {e}")