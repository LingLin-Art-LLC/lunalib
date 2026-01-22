import os
import sys

# --- Unicode-safe print for Windows console ---
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', 'utf-8')
        print(*(str(a).encode(encoding, errors='replace').decode(encoding) for a in args), **kwargs)
if sys.platform != "emscripten":
    import sqlite3
else:
    sqlite3 = None
import pickle
import gzip
import time
import base64
from typing import Dict, List, Optional
from .indexeddb import IndexedDBStore

class BlockchainCache:
    """Caches blockchain data to avoid repeated network requests"""
    
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".luna_wallet")
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "blockchain_cache.db")
        self._use_indexeddb = sys.platform == "emscripten"
        self._use_memory_db = False
        self._idb = None
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache database"""
        if self._use_indexeddb:
            self._idb = IndexedDBStore(
                db_name="lunalib",
                stores=["blocks", "mempool", "meta"],
            )
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            conn = sqlite3.connect(self.cache_file)
        except sqlite3.OperationalError:
            self.cache_file = ":memory:"
            self._use_memory_db = True
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
            if self._use_indexeddb and self._idb:
                payload = {
                    "height": height,
                    "hash": block_hash,
                    "block_data": base64.b64encode(gzip.compress(pickle.dumps(block_data))).decode("utf-8"),
                    "timestamp": time.time(),
                    "last_accessed": time.time(),
                }
                return self._idb.put("blocks", str(height), payload)
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
            safe_print(f"Cache save error: {e}")
    
    def get_block(self, height: int) -> Optional[Dict]:
        """Get block from cache"""
        try:
            if self._use_indexeddb and self._idb:
                payload = self._idb.get("blocks", str(height))
                if payload:
                    payload["last_accessed"] = time.time()
                    self._idb.put("blocks", str(height), payload)
                    raw = base64.b64decode(payload.get("block_data", "").encode("utf-8"))
                    return pickle.loads(gzip.decompress(raw))
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
            safe_print(f"Cache read error: {e}")
            
        return None

    def delete_block(self, height: int) -> None:
        """Delete a cached block by height."""
        try:
            if self._use_indexeddb and self._idb:
                self._idb.delete("blocks", str(height))
                return
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM blocks WHERE height = ?', (height,))
            conn.commit()
            conn.close()
        except Exception as e:
            safe_print(f"Cache delete error: {e}")
    
    def get_block_range(self, start_height: int, end_height: int) -> List[Dict]:
        """Get multiple blocks from cache"""
        blocks = []
        try:
            if self._use_indexeddb and self._idb:
                items = self._idb.get_all_items("blocks")
                for item in items:
                    payload = item.get("value")
                    if not payload:
                        continue
                    height = int(payload.get("height", -1))
                    if start_height <= height <= end_height:
                        try:
                            raw = base64.b64decode(payload.get("block_data", "").encode("utf-8"))
                            block = pickle.loads(gzip.decompress(raw))
                            blocks.append(block)
                        except Exception:
                            continue
                blocks.sort(key=lambda b: b.get("height", 0))
                return blocks
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
            safe_print(f"Block range cache error: {e}")
            
        return blocks
    
    def get_highest_cached_height(self) -> int:
        """Get the highest block height in cache"""
        try:
            if self._use_indexeddb and self._idb:
                items = self._idb.get_all_items("blocks")
                heights = [int(item["value"].get("height", -1)) for item in items if item.get("value")]
                return max(heights) if heights else -1
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
            if self._use_indexeddb and self._idb:
                items = self._idb.get_all_items("blocks")
                for item in items:
                    payload = item.get("value")
                    if not payload:
                        continue
                    if payload.get("last_accessed", 0) < cutoff:
                        self._idb.delete("blocks", item.get("key"))
                return
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM blocks WHERE last_accessed < ?', (cutoff,))
            conn.commit()
            conn.close()
        except Exception as e:
            safe_print(f"Cache cleanup error: {e}")