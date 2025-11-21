# wallet.py
import time
import hashlib
import json
from cryptography.fernet import Fernet
import base64
import os

class LunaWallet:
    """Luna wallet implementation with proper key management and balance tracking"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.wallets = {}  # Main wallet storage: {address: wallet_data}
        self.current_wallet_address = None  # Track which wallet is active
        
        # Initialize with an empty current wallet state
        self._reset_current_wallet()
        
    def _reset_current_wallet(self):
        """Reset current wallet to empty state"""
        self.address = None
        self.balance = 0.0  # Total balance (confirmed transactions)
        self.available_balance = 0.0  # Available balance (total - pending outgoing)
        self.created = time.time()
        self.private_key = None
        self.public_key = None
        self.encrypted_private_key = None
        self.label = "New Wallet"
        self.is_locked = True

    def _generate_address(self):
        """Generate unique wallet address"""
        import secrets
        import time
        # Use cryptographically secure random data for uniqueness
        random_data = secrets.token_hex(32)
        timestamp_ns = time.time_ns()  # More precise timestamp
        base_data = f"LUN_{timestamp_ns}_{random_data}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:32]

    def calculate_available_balance(self) -> float:
        """Calculate available balance (total balance minus pending outgoing transactions)"""
        try:
            from lunalib.core.mempool import MempoolManager
            from lunalib.core.blockchain import BlockchainManager
            
            # Get total balance from blockchain
            total_balance = self._get_total_balance_from_blockchain()
            
            # Get pending outgoing transactions from mempool
            mempool = MempoolManager()
            pending_txs = mempool.get_pending_transactions(self.address)
            
            # Sum pending outgoing amounts
            pending_outgoing = 0.0
            for tx in pending_txs:
                if tx.get('from') == self.address:
                    pending_outgoing += float(tx.get('amount', 0)) + float(tx.get('fee', 0))
            
            available_balance = max(0.0, total_balance - pending_outgoing)
            
            # Update both current wallet and wallets collection
            self.available_balance = available_balance
            if self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address]['available_balance'] = available_balance
            
            print(f"DEBUG: Available balance calculated - Total: {total_balance}, Pending Out: {pending_outgoing}, Available: {available_balance}")
            return available_balance
            
        except Exception as e:
            print(f"DEBUG: Error calculating available balance: {e}")
            return self.balance  # Fallback to total balance

    def _get_total_balance_from_blockchain(self) -> float:
        """Get total balance by scanning blockchain for confirmed transactions"""
        try:
            from lunalib.core.blockchain import BlockchainManager
            
            blockchain = BlockchainManager()
            transactions = blockchain.scan_transactions_for_address(self.address)
            
            total_balance = 0.0
            for tx in transactions:
                tx_type = tx.get('type', '')
                
                # Handle incoming transactions
                if tx.get('to') == self.address:
                    if tx_type in ['transfer', 'reward', 'fee_distribution', 'gtx_genesis']:
                        total_balance += float(tx.get('amount', 0))
                
                # Handle outgoing transactions  
                elif tx.get('from') == self.address:
                    if tx_type in ['transfer', 'stake', 'delegate']:
                        total_balance -= float(tx.get('amount', 0))
                        total_balance -= float(tx.get('fee', 0))
            
            return max(0.0, total_balance)
            
        except Exception as e:
            print(f"DEBUG: Error getting blockchain balance: {e}")
            return self.balance

    def refresh_balance(self) -> bool:
        """Refresh both total and available balance from blockchain and mempool"""
        try:
            total_balance = self._get_total_balance_from_blockchain()
            available_balance = self.calculate_available_balance()
            
            # Update wallet state
            self.balance = total_balance
            self.available_balance = available_balance
            
            # Update in wallets collection
            if self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address]['balance'] = total_balance
                self.wallets[self.current_wallet_address]['available_balance'] = available_balance
            
            print(f"DEBUG: Balance refreshed - Total: {total_balance}, Available: {available_balance}")
            return True
            
        except Exception as e:
            print(f"DEBUG: Error refreshing balance: {e}")
            return False

    def get_available_balance(self) -> float:
        """Get current wallet available balance"""
        return self.available_balance

    def _get_total_balance_from_blockchain(self) -> float:
        """Get total balance by scanning blockchain for confirmed transactions"""
        try:
            from lunalib.core.blockchain import BlockchainManager
            
            blockchain = BlockchainManager()
            transactions = blockchain.scan_transactions_for_address(self.address)
            
            total_balance = 0.0
            for tx in transactions:
                tx_type = tx.get('type', '')
                
                # Handle incoming transactions
                if tx.get('to') == self.address:
                    if tx_type in ['transfer', 'reward', 'fee_distribution', 'gtx_genesis']:
                        total_balance += float(tx.get('amount', 0))
                
                # Handle outgoing transactions  
                elif tx.get('from') == self.address:
                    if tx_type in ['transfer', 'stake', 'delegate']:
                        total_balance -= float(tx.get('amount', 0))
                        total_balance -= float(tx.get('fee', 0))
            
            return max(0.0, total_balance)
            
        except Exception as e:
            print(f"DEBUG: Error getting blockchain balance: {e}")
            return self.balance

    def refresh_balance(self) -> bool:
        """Refresh both total and available balance from blockchain and mempool"""
        try:
            total_balance = self._get_total_balance_from_blockchain()
            available_balance = self.calculate_available_balance()
            
            # Update wallet state
            self.balance = total_balance
            self.available_balance = available_balance
            
            # Update in wallets collection
            if self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address]['balance'] = total_balance
                self.wallets[self.current_wallet_address]['available_balance'] = available_balance
            
            print(f"DEBUG: Balance refreshed - Total: {total_balance}, Available: {available_balance}")
            return True
            
        except Exception as e:
            print(f"DEBUG: Error refreshing balance: {e}")
            return False

    def send_transaction(self, to_address: str, amount: float, memo: str = "", password: str = None) -> bool:
        """Send transaction using lunalib transactions with proper mempool submission"""
        try:
            print(f"DEBUG: send_transaction called - to: {to_address}, amount: {amount}, memo: {memo}")
            
            # Refresh balances first to get latest state
            self.refresh_balance()
            
            # Check available balance before proceeding
            if amount > self.available_balance:
                print(f"DEBUG: Insufficient available balance: {self.available_balance} < {amount}")
                return False
            
            # Check if wallet is unlocked
            if self.is_locked or not self.private_key:
                print("DEBUG: Wallet is locked or no private key available")
                return False
            
            # Import transaction manager
            from lunalib.transactions.transactions import TransactionManager
            
            # Create transaction manager
            tx_manager = TransactionManager()
            
            # Create and sign transaction
            transaction = tx_manager.create_transaction(
                from_address=self.address,
                to_address=to_address,
                amount=amount,
                private_key=self.private_key,
                memo=memo,
                transaction_type="transfer"
            )
            
            print(f"DEBUG: Transaction created: {transaction.get('hash')}")
            
            # Validate transaction
            is_valid, message = tx_manager.validate_transaction(transaction)
            if not is_valid:
                print(f"DEBUG: Transaction validation failed: {message}")
                return False
            
            # Send to mempool for broadcasting
            success, message = tx_manager.send_transaction(transaction)
            if success:
                print(f"DEBUG: Transaction sent to mempool: {message}")
                
                # Update available balance immediately (deduct pending transaction)
                fee = transaction.get('fee', 0)
                self.available_balance -= (amount + fee)
                if self.current_wallet_address in self.wallets:
                    self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
                
                print(f"DEBUG: Available balance updated - new available: {self.available_balance}")
                return True
            else:
                print(f"DEBUG: Failed to send transaction to mempool: {message}")
                return False
                
        except Exception as e:
            print(f"DEBUG: Error in send_transaction: {e}")
            import traceback
            traceback.print_exc()
            return False

    def send_transaction_from(self, from_address: str, to_address: str, amount: float, memo: str = "", password: str = None) -> bool:
        """Send transaction from specific address"""
        try:
            print(f"DEBUG: send_transaction_from called - from: {from_address}, to: {to_address}, amount: {amount}")
            
            # Switch to the specified wallet if different from current
            if from_address != self.current_wallet_address:
                if from_address in self.wallets:
                    # Switch to the wallet first
                    wallet_data = self.wallets[from_address]
                    self._set_current_wallet(wallet_data)
                    
                    # If password provided, unlock the wallet
                    if password:
                        unlock_success = self.unlock_wallet(from_address, password)
                        if not unlock_success:
                            print("DEBUG: Failed to unlock wallet for sending")
                            return False
                else:
                    print(f"DEBUG: Wallet not found: {from_address}")
                    return False
            
            # Now use the regular send_transaction method
            return self.send_transaction(to_address, amount, memo, password)
            
        except Exception as e:
            print(f"DEBUG: Error in send_transaction_from: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_transaction_history(self) -> dict:
        """Get complete transaction history (both pending and confirmed)"""
        try:
            from lunalib.core.blockchain import BlockchainManager
            from lunalib.core.mempool import MempoolManager
            
            blockchain = BlockchainManager()
            mempool = MempoolManager()
            
            # Get confirmed transactions from blockchain
            confirmed_txs = blockchain.scan_transactions_for_address(self.address)
            
            # Get pending transactions from mempool
            pending_txs = mempool.get_pending_transactions(self.address)
            
            return {
                'confirmed': confirmed_txs,
                'pending': pending_txs,
                'total_confirmed': len(confirmed_txs),
                'total_pending': len(pending_txs)
            }
        except Exception as e:
            print(f"DEBUG: Error getting transaction history: {e}")
            return {'confirmed': [], 'pending': [], 'total_confirmed': 0, 'total_pending': 0}

    def _generate_private_key(self):
        """Generate private key"""
        return f"priv_{hashlib.sha256(str(time.time()).encode()).hexdigest()}"
    
    def _derive_public_key(self, private_key=None):
        """Derive public key from private key"""
        priv_key = private_key or self.private_key
        if not priv_key:
            return None
        return f"pub_{priv_key[-16:]}"
    
    def get_wallet_info(self):
        """Get complete wallet information for current wallet"""
        if not self.address:
            return None
        
        # Refresh balances to ensure they're current
        self.refresh_balance()
        
        return {
            'address': self.address,
            'balance': self.balance,
            'available_balance': self.available_balance,
            'created': self.created,
            'private_key': self.private_key,
            'public_key': self.public_key,
            'encrypted_private_key': self.encrypted_private_key,
            'label': self.label,
            'is_locked': self.is_locked
        }

    def create_new_wallet(self, name, password):
        """Create a new wallet and add to collection without switching"""
        # Generate new wallet data
        address = self._generate_address()
        private_key = self._generate_private_key()
        public_key = f"pub_{private_key[-16:]}"
        
        # Encrypt private key
        key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
        fernet = Fernet(key)
        encrypted_private_key = fernet.encrypt(private_key.encode())
        
        # Create new wallet data
        new_wallet_data = {
            'address': address,
            'balance': 0.0,
            'available_balance': 0.0,
            'created': time.time(),
            'private_key': private_key,
            'public_key': public_key,
            'encrypted_private_key': encrypted_private_key,
            'label': name,
            'is_locked': True
        }
        
        # CRITICAL: Add to wallets collection
        self.wallets[address] = new_wallet_data
        
        print(f"DEBUG: Created new wallet {address}, total wallets: {len(self.wallets)}")
        
        return new_wallet_data

    def create_wallet(self, name, password):
        """Create a new wallet and set it as current"""
        # Generate new wallet data
        address = self._generate_address()
        private_key = self._generate_private_key()
        public_key = f"pub_{private_key[-16:]}"
        
        # Encrypt private key
        key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
        fernet = Fernet(key)
        encrypted_private_key = fernet.encrypt(private_key.encode())
        
        # Create wallet data
        wallet_data = {
            'address': address,
            'balance': 0.0,
            'available_balance': 0.0,
            'created': time.time(),
            'private_key': private_key,
            'public_key': public_key,
            'encrypted_private_key': encrypted_private_key,
            'label': name,
            'is_locked': True
        }
        
        # CRITICAL: Add to wallets collection
        self.wallets[address] = wallet_data
        
        # Set as current wallet
        self._set_current_wallet(wallet_data)
        
        print(f"DEBUG: Created and switched to wallet {address}, total wallets: {len(self.wallets)}")
        
        return wallet_data

    def _set_current_wallet(self, wallet_data):
        """Set the current wallet from wallet data"""
        self.current_wallet_address = wallet_data['address']
        self.address = wallet_data['address']
        self.balance = wallet_data['balance']
        self.available_balance = wallet_data['available_balance']
        self.created = wallet_data['created']
        self.private_key = wallet_data['private_key']
        self.public_key = wallet_data['public_key']
        self.encrypted_private_key = wallet_data['encrypted_private_key']
        self.label = wallet_data['label']
        self.is_locked = wallet_data.get('is_locked', True)

    def switch_wallet(self, address, password=None):
        """Switch to a different wallet in the collection"""
        if address in self.wallets:
            wallet_data = self.wallets[address]
            self._set_current_wallet(wallet_data)
            
            # Refresh balances for the new wallet
            self.refresh_balance()
            
            # If password provided, unlock the wallet
            if password:
                return self.unlock_wallet(address, password)
            
            return True
        return False

    def unlock_wallet(self, address, password):
        """Unlock wallet with password"""
        if address not in self.wallets:
            return False
            
        wallet_data = self.wallets[address]
        
        try:
            if wallet_data.get('encrypted_private_key'):
                key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
                fernet = Fernet(key)
                decrypted_key = fernet.decrypt(wallet_data['encrypted_private_key'])
                wallet_data['private_key'] = decrypted_key.decode()
                wallet_data['is_locked'] = False
                
                # If this is the current wallet, update current state
                if self.current_wallet_address == address:
                    self.private_key = wallet_data['private_key']
                    self.is_locked = False
                
                return True
        except:
            pass
        return False
    
    @property
    def is_unlocked(self):
        """Check if current wallet is unlocked"""
        if not self.current_wallet_address:
            return False
        wallet_data = self.wallets.get(self.current_wallet_address, {})
        return not wallet_data.get('is_locked', True)
    
    def export_private_key(self, address, password):
        """Export private key with password decryption"""
        if address not in self.wallets:
            return None
            
        wallet_data = self.wallets[address]
        
        try:
            if wallet_data.get('encrypted_private_key'):
                key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
                fernet = Fernet(key)
                decrypted_key = fernet.decrypt(wallet_data['encrypted_private_key'])
                return decrypted_key.decode()
        except:
            pass
        return None
    
    def import_wallet(self, wallet_data, password=None):
        """Import wallet from data"""
        if isinstance(wallet_data, dict):
            address = wallet_data.get('address')
            if not address:
                return False
                
            # Add to wallets collection
            self.wallets[address] = wallet_data.copy()
            
            # Set as current wallet
            self._set_current_wallet(wallet_data)
            
            # Refresh balances for imported wallet
            self.refresh_balance()
            
            if password and wallet_data.get('encrypted_private_key'):
                return self.unlock_wallet(address, password)
            
            return True
        return False
    
    def update_balance(self, new_balance):
        """Update current wallet balance (use refresh_balance instead for accurate tracking)"""
        self.balance = float(new_balance)
        self.available_balance = float(new_balance)
        
        # Also update in wallets collection
        if self.current_wallet_address and self.current_wallet_address in self.wallets:
            self.wallets[self.current_wallet_address]['balance'] = self.balance
            self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
        
        return True
    
    def get_balance(self):
        """Get current wallet total balance"""
        return self.balance
    
    def get_available_balance(self):
        """Get current wallet available balance"""
        return self.available_balance
    
    def get_wallet_by_address(self, address):
        """Get wallet by address from wallets collection"""
        return self.wallets.get(address)
    
    def list_wallets(self):
        """List all wallets in collection"""
        return list(self.wallets.keys())
    
    def get_current_wallet_info(self):
        """Get current wallet information"""
        if not self.current_wallet_address:
            return None
        
        # Refresh balances to ensure they're current
        self.refresh_balance()
        
        return self.wallets.get(self.current_wallet_address)
    
    def save_to_file(self, filename=None):
        """Save wallet to file"""
        if not self.data_dir:
            return False
            
        if filename is None:
            filename = f"wallet_{self.address}.json"
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Ensure directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Prepare encrypted private key for serialization
            encrypted_key_data = None
            if self.encrypted_private_key:
                # Ensure it's bytes before encoding
                if isinstance(self.encrypted_private_key, bytes):
                    encrypted_key_data = base64.b64encode(self.encrypted_private_key).decode('utf-8')
                else:
                    encrypted_key_data = base64.b64encode(self.encrypted_private_key.encode()).decode('utf-8')
            
            # Prepare wallets for serialization (remove any non-serializable data)
            serializable_wallets = {}
            for addr, wallet_info in self.wallets.items():
                serializable_wallet = wallet_info.copy()
                # Ensure encrypted_private_key is serializable
                if serializable_wallet.get('encrypted_private_key') and isinstance(serializable_wallet['encrypted_private_key'], bytes):
                    serializable_wallet['encrypted_private_key'] = base64.b64encode(
                        serializable_wallet['encrypted_private_key']
                    ).decode('utf-8')
                serializable_wallets[addr] = serializable_wallet
            
            wallet_data = {
                'address': self.address,
                'balance': self.balance,
                'available_balance': self.available_balance,
                'created': self.created,
                'public_key': self.public_key,
                'encrypted_private_key': encrypted_key_data,
                'label': self.label,
                'is_locked': self.is_locked,
                'wallets': serializable_wallets,
                'current_wallet_address': self.current_wallet_address
            }
            
            with open(filepath, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving wallet: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_from_file(self, filename, password=None):
        """Load wallet from file"""
        if not self.data_dir:
            return False
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                wallet_data = json.load(f)
            
            # Load wallets collection
            self.wallets = wallet_data.get('wallets', {})
            
            # Load current wallet address
            self.current_wallet_address = wallet_data.get('current_wallet_address')
            
            # If we have a current wallet, load its data
            if self.current_wallet_address and self.current_wallet_address in self.wallets:
                current_wallet_data = self.wallets[self.current_wallet_address]
                self._set_current_wallet(current_wallet_data)
                
                # Handle encrypted private key
                encrypted_key = wallet_data.get('encrypted_private_key')
                if encrypted_key:
                    self.encrypted_private_key = base64.b64decode(encrypted_key.encode())
                    # Also update in wallets collection
                    if self.current_wallet_address in self.wallets:
                        self.wallets[self.current_wallet_address]['encrypted_private_key'] = self.encrypted_private_key
            
            # Refresh balances after loading
            self.refresh_balance()
            
            # If password provided and we have encrypted key, unlock
            if password and self.encrypted_private_key and self.current_wallet_address:
                return self.unlock_wallet(self.current_wallet_address, password)
            
            return True
        except Exception as e:
            print(f"Error loading wallet: {e}")
            return False