import os
import json
import sqlite3
from typing import Dict, List, Optional

class BillRegistry:
    """Manages bill database with verification links and metadata"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(os.path.expanduser("~"), ".luna_wallet", "bills.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize bill database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bills (
                bill_serial TEXT PRIMARY KEY,
                denomination INTEGER,
                user_address TEXT,
                hash TEXT,
                mining_time REAL,
                difficulty INTEGER,
                luna_value REAL,
                timestamp REAL,
                verification_url TEXT,
                image_url TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_bill(self, bill_info):
        """Register a new bill in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate verification URL
        verification_url = f"https://bank.linglin.art/verify/{bill_info['hash']}"
        image_url = f"https://bank.linglin.art/bills/{bill_info['bill_serial']}.png"
        
        cursor.execute('''
            INSERT OR REPLACE INTO bills 
            (bill_serial, denomination, user_address, hash, mining_time, 
             difficulty, luna_value, timestamp, verification_url, image_url, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bill_info['bill_serial'],
            bill_info['denomination'],
            bill_info['user_address'],
            bill_info['hash'],
            bill_info['mining_time'],
            bill_info['difficulty'],
            bill_info['luna_value'],
            bill_info['timestamp'],
            verification_url,
            image_url,
            json.dumps(bill_info.get('bill_data', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_bill(self, bill_serial):
        """Retrieve bill information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bills WHERE bill_serial = ?', (bill_serial,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'bill_serial': result[0],
                'denomination': result[1],
                'user_address': result[2],
                'hash': result[3],
                'mining_time': result[4],
                'difficulty': result[5],
                'luna_value': result[6],
                'timestamp': result[7],
                'verification_url': result[8],
                'image_url': result[9],
                'metadata': json.loads(result[10]) if result[10] else {},
                'status': result[11]
            }
        return None
    
    def get_user_bills(self, user_address):
        """Get all bills for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bills WHERE user_address = ? ORDER BY timestamp DESC', (user_address,))
        results = cursor.fetchall()
        conn.close()
        
        bills = []
        for result in results:
            bills.append({
                'bill_serial': result[0],
                'denomination': result[1],
                'user_address': result[2],
                'hash': result[3],
                'mining_time': result[4],
                'difficulty': result[5],
                'luna_value': result[6],
                'timestamp': result[7],
                'verification_url': result[8],
                'image_url': result[9],
                'metadata': json.loads(result[10]) if result[10] else {},
                'status': result[11]
            })
        
        return bills