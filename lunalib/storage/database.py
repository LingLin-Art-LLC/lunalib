import os
import sqlite3
import json
import time
from typing import Dict, List, Optional, Any

class WalletDatabase:
    """Manages wallet data storage"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(os.path.expanduser("~"), ".luna_wallet", "wallets.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize wallet database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Wallets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                label TEXT,
                public_key TEXT,
                encrypted_private_key TEXT,
                balance REAL DEFAULT 0.0,
                created REAL,
                last_accessed REAL,
                metadata TEXT
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                tx_hash TEXT PRIMARY KEY,
                wallet_address TEXT,
                tx_type TEXT,
                from_address TEXT,
                to_address TEXT,
                amount REAL,
                fee REAL,
                timestamp REAL,
                block_height INTEGER,
                status TEXT,
                memo TEXT,
                raw_data TEXT,
                FOREIGN KEY (wallet_address) REFERENCES wallets (address)
            )
        ''')
        
        # Pending transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_transactions (
                tx_hash TEXT PRIMARY KEY,
                wallet_address TEXT,
                from_address TEXT,
                to_address TEXT,
                amount REAL,
                fee REAL,
                created_time REAL,
                status TEXT DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                last_retry REAL,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_wallet(self, wallet_data: Dict) -> bool:
        """Save wallet to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO wallets 
                (address, label, public_key, encrypted_private_key, balance, created, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                wallet_data['address'],
                wallet_data.get('label', ''),
                wallet_data.get('public_key', ''),
                wallet_data.get('encrypted_private_key', ''),
                wallet_data.get('balance', 0.0),
                wallet_data.get('created', time.time()),
                time.time(),
                json.dumps(wallet_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Save wallet error: {e}")
            return False
    
    def load_wallet(self, address: str) -> Optional[Dict]:
        """Load wallet from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM wallets WHERE address = ?', (address,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'address': result[0],
                    'label': result[1],
                    'public_key': result[2],
                    'encrypted_private_key': result[3],
                    'balance': result[4],
                    'created': result[5],
                    'last_accessed': result[6],
                    'metadata': json.loads(result[7]) if result[7] else {}
                }
                
        except Exception as e:
            print(f"Load wallet error: {e}")
            
        return None
    
    def save_transaction(self, transaction: Dict, wallet_address: str) -> bool:
        """Save transaction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (tx_hash, wallet_address, tx_type, from_address, to_address, amount, fee, 
                 timestamp, block_height, status, memo, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.get('hash', ''),
                wallet_address,
                transaction.get('type', 'transfer'),
                transaction.get('from', ''),
                transaction.get('to', ''),
                transaction.get('amount', 0),
                transaction.get('fee', 0),
                transaction.get('timestamp', time.time()),
                transaction.get('block_height'),
                transaction.get('status', 'confirmed'),
                transaction.get('memo', ''),
                json.dumps(transaction)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Save transaction error: {e}")
            return False
    
    def get_wallet_transactions(self, wallet_address: str, limit: int = 100) -> List[Dict]:
        """Get transactions for a wallet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT raw_data FROM transactions 
                WHERE wallet_address = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (wallet_address, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            transactions = []
            for result in results:
                try:
                    tx = json.loads(result[0])
                    transactions.append(tx)
                except:
                    continue
                    
            return transactions
            
        except Exception as e:
            print(f"Get transactions error: {e}")
            return []
    
    def save_pending_transaction(self, transaction: Dict, wallet_address: str) -> bool:
        """Save pending transaction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pending_transactions 
                (tx_hash, wallet_address, from_address, to_address, amount, fee, 
                 created_time, status, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.get('hash', ''),
                wallet_address,
                transaction.get('from', ''),
                transaction.get('to', ''),
                transaction.get('amount', 0),
                transaction.get('fee', 0),
                time.time(),
                'pending',
                json.dumps(transaction)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Save pending transaction error: {e}")
            return False