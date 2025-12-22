# lunalib/core/mempool.py - Updated version

import time
import requests
import threading
from queue import Queue
from typing import Dict, List, Optional, Set
import json
import hashlib

class MempoolManager:
    """Manages transaction mempool and network broadcasting"""
    
    def __init__(self, network_endpoints: List[str] = None):
        self.network_endpoints = network_endpoints or ["https://bank.linglin.art"]
        self.local_mempool = {}  # {tx_hash: transaction}
        self.pending_broadcasts = Queue()
        self.confirmed_transactions: Set[str] = set()
        self.max_mempool_size = 10000
        self.broadcast_retries = 3
        self.is_running = True
        
        # Start background broadcasting thread
        self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
        self.broadcast_thread.start()

    
    def add_transaction(self, transaction: Dict) -> bool:
        """Add transaction to local mempool and broadcast to network"""
        try:
            tx_hash = transaction.get('hash')
            if not tx_hash:
                print("DEBUG: Transaction missing hash")
                return False
            
            # Check if transaction already exists or is confirmed
            if tx_hash in self.local_mempool or tx_hash in self.confirmed_transactions:
                print(f"DEBUG: Transaction already processed: {tx_hash}")
                return True
            
            # Validate basic transaction structure
            if not self._validate_transaction_basic(transaction):
                print("DEBUG: Transaction validation failed")
                return False
            
            # Add to local mempool
            self.local_mempool[tx_hash] = {
                'transaction': transaction,
                'timestamp': time.time(),
                'broadcast_attempts': 0,
                'last_broadcast': 0
            }
            print(f"DEBUG: Added transaction to mempool: {tx_hash}")
            
            # Queue for broadcasting
            self.pending_broadcasts.put(transaction)
            print(f"DEBUG: Queued transaction for broadcasting: {tx_hash}")
            
            return True
            
        except Exception as e:
            print(f"DEBUG: Error adding transaction to mempool: {e}")
            return False
    
    def broadcast_transaction(self, transaction: Dict) -> bool:
        """Broadcast transaction to network endpoints - SIMPLIFIED FOR YOUR FLASK APP"""
        tx_hash = transaction.get('hash')
        print(f"DEBUG: Broadcasting transaction to mempool: {tx_hash}")
        
        success = False
        for endpoint in self.network_endpoints:
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
                        'User-Agent': 'LunaWallet/1.0'
                    }
                    
                    # Send transaction directly to mempool endpoint
                    response = requests.post(
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
                            success = True
                            break
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
        
        if success:
            print(f"✅ Transaction {tx_hash} successfully broadcasted")
        else:
            print(f"❌ All broadcast attempts failed for transaction {tx_hash}")
            
        return success
    
    def test_connection(self) -> bool:
        """Test connection to network endpoints"""
        for endpoint in self.network_endpoints:
            try:
                print(f"DEBUG: Testing connection to {endpoint}")
                # Test with a simple health check or mempool status
                test_endpoints = [
                    f"{endpoint}/system/health",
                    f"{endpoint}/mempool/status", 
                    f"{endpoint}/"
                ]
                
                for test_endpoint in test_endpoints:
                    try:
                        response = requests.get(test_endpoint, timeout=5)
                        print(f"DEBUG: Connection test response from {test_endpoint}: {response.status_code}")
                        if response.status_code == 200:
                            print(f"✅ Successfully connected to {endpoint}")
                            return True
                    except:
                        continue
                        
            except Exception as e:
                print(f"DEBUG: Connection test failed for {endpoint}: {e}")
        
        print("❌ All connection tests failed")
        return False
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction from mempool by hash"""
        if tx_hash in self.local_mempool:
            return self.local_mempool[tx_hash]['transaction']
        return None
    
    def get_pending_transactions(self, address: str = None) -> List[Dict]:
        """Get all pending transactions, optionally filtered by address"""
        transactions = []
        for tx_data in self.local_mempool.values():
            tx = tx_data['transaction']
            if address is None or tx.get('from') == address or tx.get('to') == address:
                transactions.append(tx)
        return transactions
    
    def remove_transaction(self, tx_hash: str):
        """Remove transaction from mempool (usually after confirmation)"""
        if tx_hash in self.local_mempool:
            del self.local_mempool[tx_hash]
            self.confirmed_transactions.add(tx_hash)
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
        
        print(f"✅ Transaction validation passed: {transaction.get('type')} from {from_addr} to {to_addr}")
        return True
    
    def _broadcast_worker(self):
        """Background worker to process pending broadcasts"""
        while self.is_running:
            try:
                # Get next transaction to broadcast (blocking)
                transaction = self.pending_broadcasts.get(timeout=1.0)
                
                if transaction:
                    tx_hash = transaction.get('hash')
                    print(f"DEBUG: Processing broadcast for transaction: {tx_hash}")
                    
                    # Broadcast the transaction
                    success = self.broadcast_transaction(transaction)
                    
                    # Update broadcast attempts in local mempool
                    if tx_hash in self.local_mempool:
                        self.local_mempool[tx_hash]['broadcast_attempts'] += 1
                        self.local_mempool[tx_hash]['last_broadcast'] = time.time()
                    
                    # Mark task as done
                    self.pending_broadcasts.task_done()
                    
                    # Small delay between broadcasts
                    time.sleep(0.5)
                    
            except Exception as e:
                # Queue.get() timed out or other error
                continue
    
    def stop(self):
        """Stop the mempool manager"""
        self.is_running = False
        print("DEBUG: Mempool manager stopped")