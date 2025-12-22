import time
import hashlib
import json
import os
import secrets
import binascii
import base64
import threading
import concurrent.futures
from typing import Optional, Callable
from cryptography.fernet import Fernet
from ..core.crypto import KeyManager


class LunaWallet:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.wallets = {}
        self.current_wallet_address = None
        self.key_manager = KeyManager()
        
        # Threading for asynchronous balance loading
        self.balance_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.balance_callbacks = []
        self.balance_loading = False
        self.last_balance_update = 0
        self.balance_update_interval = 30  # seconds
        self._reset_current_wallet()
        
        
    def _reset_current_wallet(self):
        """Reset current wallet to empty state"""
        self.address = None
        self.balance = 0.0  # Total balance (confirmed transactions)
        self.available_balance = 0.0  # Available balance (total - pending outgoing)
        self.created = time.time()
        self.private_key = None  # Will store REAL SM2 private key
        self.public_key = None   # Will store REAL SM2 public key
        self.encrypted_private_key = None
        self.label = "New Wallet"
        self.is_locked = True
    
    # ============================================================================
    # REAL CRYPTOGRAPHIC KEY GENERATION (SM2 EAST ASIAN STANDARD)
    # ============================================================================
    
 
    
    def _sign_transaction_data(self, transaction_data):
        """Sign transaction data with SM2 private key"""
        if not self.private_key or self.is_locked:
            print("DEBUG: Cannot sign - wallet locked or no private key")
            return None
        
        # Convert transaction data to string for signing
        tx_string = json.dumps(transaction_data, sort_keys=True)
        
        # Sign with SM2
        signature = self.sm2.sign_message(self.private_key, tx_string)
        
        if signature:
            print(f"DEBUG: Transaction signed successfully, signature: {signature[:16]}...")
        else:
            print("DEBUG: Failed to sign transaction")
        
        return signature
    
    def _verify_wallet_integrity(self) -> bool:
        """Basic cryptographic key verification"""
        # Check if keys exist
        if not self.private_key or not self.public_key:
            return False
        
        # Validate key formats
        if len(self.private_key) != 64:  # 256-bit hex
            return False
        
        if not self.public_key.startswith('04') or len(self.public_key) != 130:
            return False
        
        return True
    
    def _generate_key_pair(self):
        """Generate REAL SM2 key pair"""
        return self.key_manager.generate_keypair()

    def create_wallet(self, name, password):
        """Create a new wallet and set it as current"""
        print(f"DEBUG: Creating wallet '{name}'")
        
        # Generate REAL SM2 keys
        private_key, public_key, address = self._generate_key_pair()
        
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
            'private_key': private_key,  # REAL SM2 private key
            'public_key': public_key,    # REAL SM2 public key
            'encrypted_private_key': encrypted_private_key,
            'label': name,
            'is_locked': True,
            'crypto_standard': 'SM2_GB/T_32918'
        }
        
        # Add to wallets collection
        self.wallets[address] = wallet_data
        
        # Set as current wallet
        self._set_current_wallet(wallet_data)
        
        print(f"DEBUG: Created and set current wallet {address}")
        
        return wallet_data

    
    def _generate_address(self, public_key_hex):
        """Generate address from SM2 public key"""
        return self.key_manager.derive_address(public_key_hex)
    def create_new_wallet(self, name, password):
        """
        Create a new wallet without switching to it
        Returns: wallet_data dict
        """
        print(f"DEBUG: Creating additional wallet '{name}'")
        
        # Generate REAL cryptographic keys
        private_key_hex, public_key_hex, address = self._generate_key_pair()
        
        # Encrypt private key
        key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
        fernet = Fernet(key)
        encrypted_private_key = fernet.encrypt(private_key_hex.encode())
        
        # Create wallet data
        new_wallet_data = {
            'address': address,
            'balance': 0.0,
            'available_balance': 0.0,
            'created': time.time(),
            'private_key': private_key_hex,
            'public_key': public_key_hex,
            'encrypted_private_key': encrypted_private_key,
            'label': name,
            'is_locked': True,
            'crypto_standard': 'SM2_GB/T_32918'
        }
        
        # Add to wallets collection
        self.wallets[address] = new_wallet_data
        
        print(f"DEBUG: Created wallet {address}, total wallets: {len(self.wallets)}")
        
        return new_wallet_data
    
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
        
        print(f"DEBUG: Set current wallet to {self.address}")
    
    def unlock_wallet(self, address, password):
        """Unlock wallet with password"""
        if address not in self.wallets:
            print(f"DEBUG: Wallet {address} not found in collection")
            return False
        
        wallet_data = self.wallets[address]
        
        try:
            if wallet_data.get('encrypted_private_key'):
                # Decrypt private key
                key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
                fernet = Fernet(key)
                decrypted_key = fernet.decrypt(wallet_data['encrypted_private_key'])
                
                # Update wallet data
                wallet_data['private_key'] = decrypted_key.decode()
                wallet_data['is_locked'] = False
                
                # If this is the current wallet, update current state
                if self.current_wallet_address == address:
                    self.private_key = wallet_data['private_key']
                    self.is_locked = False
                
                print(f"DEBUG: Wallet {address} unlocked successfully")
                print(f"DEBUG: Private key available: {bool(self.private_key)}")
                
                # Verify cryptographic integrity after unlock
                if self._verify_wallet_integrity():
                    print(f"DEBUG: Wallet cryptographic integrity verified")
                else:
                    print(f"DEBUG: WARNING: Wallet unlocked but cryptographic verification failed")
                
                return True
        except Exception as e:
            print(f"DEBUG: Unlock failed: {e}")
        
        return False
    
    def switch_wallet(self, address, password=None):
        """Switch to a different wallet in the collection"""
        if address in self.wallets:
            wallet_data = self.wallets[address]
            self._set_current_wallet(wallet_data)
            
            # If password provided, unlock the wallet
            if password:
                return self.unlock_wallet(address, password)
            
            return True
        
        print(f"DEBUG: Cannot switch to {address} - not in wallet collection")
        return False
    
    # ============================================================================
    # BALANCE AND TRANSACTION METHODS
    # ============================================================================
    
    def calculate_available_balance(self) -> float:
        """Calculate available balance (total balance minus pending outgoing transactions)"""
        try:
            # Get total balance from blockchain
            total_balance = self._get_total_balance_from_blockchain()
            
            # Get pending outgoing amount
            pending_outgoing = self._get_pending_balance()
            
            # Also check for any incoming pending transactions
            incoming_pending = self._get_pending_incoming_balance()
            
            available_balance = max(0.0, total_balance - pending_outgoing)
            
            # Update both current wallet and wallets collection
            self.available_balance = available_balance
            self.balance = total_balance  # Also update total balance
            
            if self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address]['available_balance'] = available_balance
                self.wallets[self.current_wallet_address]['balance'] = total_balance
            
            print(f"DEBUG: Balance calculated - Total: {total_balance}, Pending Out: {pending_outgoing}, Available: {available_balance}")
            
            if incoming_pending > 0:
                print(f"DEBUG: Also {incoming_pending} LUN incoming (pending)")
            
            return available_balance
            
        except Exception as e:
            print(f"DEBUG: Error calculating available balance: {e}")
            return self.balance  # Fallback to total balance
    def _get_pending_balance(self) -> float:
        """Get total pending balance from mempool (outgoing pending transactions)"""
        try:
            from lunalib.core.mempool import MempoolManager
            
            mempool = MempoolManager()
            
            # Get pending transactions for this address
            pending_txs = mempool.get_pending_transactions(self.address)
            
            total_pending_outgoing = 0.0
            
            for tx in pending_txs:
                # Only count outgoing transactions (where we're the sender)
                if tx.get('from') == self.address:
                    amount = float(tx.get('amount', 0))
                    fee = float(tx.get('fee', 0) or tx.get('gas', 0) or 0)
                    total_pending_outgoing += amount + fee
                    print(f"🔍 Found pending outgoing: {amount} + {fee} fee = {amount + fee}")
            
            return total_pending_outgoing
            
        except Exception as e:
            print(f"DEBUG: Error calculating pending balance: {e}")
            return 0.0
    
    def _get_pending_incoming_balance(self) -> float:
        """Get total pending incoming balance from mempool"""
        try:
            from lunalib.core.mempool import MempoolManager
            
            mempool = MempoolManager()
            
            # Get pending transactions for this address
            pending_txs = mempool.get_pending_transactions(self.address)
            
            total_pending_incoming = 0.0
            
            for tx in pending_txs:
                # Only count incoming transactions (where we're the receiver)
                if tx.get('to') == self.address:
                    amount = float(tx.get('amount', 0))
                    total_pending_incoming += amount
                    print(f"🔍 Found pending incoming: +{amount}")
            
            return total_pending_incoming
            
        except Exception as e:
            print(f"DEBUG: Error calculating pending incoming balance: {e}")
            return 0.0
    
    def _get_total_balance_from_blockchain(self) -> float:
        """Get total balance by scanning blockchain for confirmed transactions"""
        try:
            from lunalib.core.blockchain import BlockchainManager
            
            blockchain = BlockchainManager()
            transactions = blockchain.scan_transactions_for_address(self.address)
            
            total_balance = 0.0
            for tx in transactions:
                tx_type = tx.get('type', '').lower()
                direction = tx.get('direction', '')
                
                # For reward transactions (always incoming)
                if tx_type == 'reward' or tx.get('from') == 'network':
                    total_balance += float(tx.get('amount', 0))
                    print(f"💰 Reward added: +{tx.get('amount', 0)}")
                
                # For regular transactions
                elif direction == 'incoming':
                    total_balance += float(tx.get('amount', 0))
                    print(f"💰 Incoming added: +{tx.get('amount', 0)}")
                
                elif direction == 'outgoing':
                    total_balance -= float(tx.get('amount', 0))
                    total_balance -= float(tx.get('fee', 0))
                    print(f"💰 Outgoing deducted: -{tx.get('amount', 0)} -{tx.get('fee', 0)} fee")
            
            return max(0.0, total_balance)
            
        except Exception as e:
            print(f"DEBUG: Error getting blockchain balance: {e}")
            return self.balance
    
    def register_balance_callback(self, callback: Callable[[float, float], None]):
        """Register a callback to be called when balance is updated asynchronously"""
        self.balance_callbacks.append(callback)
    
    def start_async_balance_loading(self):
        """Start asynchronous balance loading in background thread"""
        if self.balance_loading:
            return  # Already loading
        
        if time.time() - self.last_balance_update < self.balance_update_interval:
            return  # Too soon since last update
        
        self.balance_loading = True
        
        def _async_balance_update():
            try:
                # Load balances asynchronously
                total_balance = self._get_total_balance_from_blockchain()
                available_balance = self.calculate_available_balance()
                
                # Update wallet state
                self.balance = total_balance
                self.available_balance = available_balance
                self.last_balance_update = time.time()
                
                # Update in wallets collection
                if self.current_wallet_address in self.wallets:
                    self.wallets[self.current_wallet_address]['balance'] = total_balance
                    self.wallets[self.current_wallet_address]['available_balance'] = available_balance
                
                # Notify callbacks
                for callback in self.balance_callbacks:
                    try:
                        callback(total_balance, available_balance)
                    except Exception as e:
                        print(f"Error in balance callback: {e}")
                
                print(f"DEBUG: Async balance updated - Total: {total_balance}, Available: {available_balance}")
                
            except Exception as e:
                print(f"DEBUG: Error in async balance loading: {e}")
            finally:
                self.balance_loading = False
        
        # Start in background thread
        self.balance_thread_pool.submit(_async_balance_update)
    
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
        """Send transaction using REAL SM2 signatures"""
        try:
            print(f"[SM2] Sending {amount} to {to_address}")
            
            # 1. Basic validation
            if self.is_locked or not self.private_key:
                print("[SM2] Wallet locked or no private key")
                return False
            
            # 2. Check key integrity
            if not self._verify_wallet_integrity():
                print("[SM2] Invalid cryptographic keys")
                return False
            
            # 3. Balance check
            self.refresh_balance()
            if amount > self.get_available_balance():
                print(f"[SM2] Insufficient balance: {self.get_available_balance()} < {amount}")
                return False
            
            # 4. Create and sign transaction
            from lunalib.transactions.transactions import TransactionManager
            tx_manager = TransactionManager()
            
            transaction = tx_manager.create_transaction(
                from_address=self.address,
                to_address=to_address,
                amount=amount,
                private_key=self.private_key,
                memo=memo,
                transaction_type="transfer"
            )
            
            print(f"[SM2] Transaction created: {transaction.get('hash', 'no_hash')[:16]}")
            
            # 5. Validate
            is_valid, message = tx_manager.validate_transaction(transaction)
            if not is_valid:
                print(f"[SM2] Validation failed: {message}")
                return False
            
            # 6. Broadcast
            success, message = tx_manager.send_transaction(transaction)
            if success:
                print(f"[SM2] Transaction broadcast: {message}")
                
                # Update balance immediately
                total_cost = amount + transaction.get('fee', 0)
                self.available_balance -= total_cost
                if self.current_wallet_address in self.wallets:
                    self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
                
                # Save state
                self.save_wallet_data()
                
                # Trigger async balance refresh to get accurate data
                self.start_async_balance_loading()
                
                return True
            else:
                print(f"[SM2] Broadcast failed: {message}")
                return False
                
        except Exception as e:
            print(f"[SM2] Error: {e}")
            return False
    
    def save_wallet_data(self):
        """Save wallet data to file"""
        try:
            print(f"[WALLET] Saving wallet data...")
            
            # Prepare wallet data
            wallet_data = {
                'version': '2.0',
                'address': self.address,
                'public_key': self.public_key,
                'private_key': self.private_key,  # WARNING: In production, encrypt this!
                'balance': self.balance,
                'available_balance': self.available_balance,
                'transactions': self.transactions,
                'created_at': self.created_at,
                'last_sync': time.time(),
                'network': self.network,
                'key_type': 'SM2'  # Indicate SM2 key type
            }
            
            # Determine save path
            if hasattr(self, 'wallet_file'):
                save_path = self.wallet_file
            else:
                # Default save location
                import os
                wallet_dir = os.path.expanduser('~/.lunawallet')
                os.makedirs(wallet_dir, exist_ok=True)
                save_path = os.path.join(wallet_dir, f'{self.address}.json')
            
            # Save to file
            import json
            with open(save_path, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            print(f"[WALLET] Wallet data saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"[WALLET ERROR] Failed to save wallet data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_wallet_data(self, wallet_file=None):
        """Load wallet data from file"""
        try:
            print(f"[WALLET] Loading wallet data...")
            
            if wallet_file:
                self.wallet_file = wallet_file
                load_path = wallet_file
            elif hasattr(self, 'wallet_file'):
                load_path = self.wallet_file
            else:
                # Try to find wallet file
                import os
                wallet_dir = os.path.expanduser('~/.lunawallet')
                # Look for any .json file
                wallet_files = [f for f in os.listdir(wallet_dir) if f.endswith('.json')]
                if not wallet_files:
                    print("[WALLET] No wallet file found")
                    return False
                load_path = os.path.join(wallet_dir, wallet_files[0])
                self.wallet_file = load_path
            
            # Load from file
            import json
            with open(load_path, 'r') as f:
                wallet_data = json.load(f)
            
            # Restore wallet data
            self.address = wallet_data.get('address', '')
            self.public_key = wallet_data.get('public_key', '')
            self.private_key = wallet_data.get('private_key', '')
            self.balance = wallet_data.get('balance', 0.0)
            self.available_balance = wallet_data.get('available_balance', 0.0)
            self.transactions = wallet_data.get('transactions', [])
            self.created_at = wallet_data.get('created_at', time.time())
            self.network = wallet_data.get('network', 'mainnet')
            
            print(f"[WALLET] Wallet loaded: {self.address}")
            print(f"[WALLET] Balance: {self.balance}")
            
            # Initialize SM2 if we have keys
            if self.public_key and self.private_key:
                self._initialize_sm2()
            
            # Start async balance loading
            self.start_async_balance_loading()
            
            return True
            
        except Exception as e:
            print(f"[WALLET ERROR] Failed to load wallet data: {e}")
            return False
    
    def _initialize_sm2(self):
        """Initialize SM2 crypto if keys exist"""
        try:
            if self.public_key and self.private_key:
                from ..core.crypto import KeyManager
                self.key_manager = KeyManager()
                print(f"[WALLET] SM2 initialized with existing keys")
                return True
        except Exception as e:
            print(f"[WALLET ERROR] Failed to initialize SM2: {e}")
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
    
    # ============================================================================
    # WALLET INFO AND UTILITIES
    # ============================================================================
    
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
            
            # Check if wallet uses SM2 cryptography
            if wallet_data.get('crypto_standard') != 'SM2_GB/T_32918':
                print(f"DEBUG: WARNING: Importing wallet without SM2 cryptography standard")
            
            # Add to wallets collection
            self.wallets[address] = wallet_data.copy()
            
            # Set as current wallet
            self._set_current_wallet(wallet_data)
            
            if password and wallet_data.get('encrypted_private_key'):
                return self.unlock_wallet(address, password)
            
            return True
        return False
    
    def update_balance(self, new_balance):
        """Update current wallet balance"""
        self.balance = float(new_balance)
        self.available_balance = float(new_balance)
        
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
        
        return self.wallets.get(self.current_wallet_address)
    
    def get_wallet_info(self):
        """Get complete wallet information for current wallet"""
        if not self.address:
            return None
        
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
            'is_locked': self.is_locked,
            'crypto_standard': 'SM2_GB/T_32918'
        }
    
    def save_to_file(self, filename=None):
        """Save wallet to file"""
        if not self.data_dir:
            return False
            
        if filename is None:
            filename = f"wallet_{self.address}.json"
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            
            encrypted_key_data = None
            if self.encrypted_private_key:
                if isinstance(self.encrypted_private_key, bytes):
                    encrypted_key_data = base64.b64encode(self.encrypted_private_key).decode('utf-8')
                else:
                    encrypted_key_data = base64.b64encode(self.encrypted_private_key.encode()).decode('utf-8')
            
            serializable_wallets = {}
            for addr, wallet_info in self.wallets.items():
                serializable_wallet = wallet_info.copy()
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
                'current_wallet_address': self.current_wallet_address,
                'crypto_standard': 'SM2_GB/T_32918'
            }
            
            with open(filepath, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            print(f"DEBUG: Wallet saved to {filepath}")
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
            
            # Check crypto standard
            crypto_standard = wallet_data.get('crypto_standard')
            if crypto_standard != 'SM2_GB/T_32918':
                print(f"DEBUG: WARNING: Loading wallet with different crypto standard: {crypto_standard}")
            
            # Load wallets collection
            self.wallets = wallet_data.get('wallets', {})
            
            # Load current wallet address
            self.current_wallet_address = wallet_data.get('current_wallet_address')
            
            # If we have a current wallet, load its data
            if self.current_wallet_address and self.current_wallet_address in self.wallets:
                current_wallet_data = self.wallets[self.current_wallet_address]
                self._set_current_wallet(current_wallet_data)
                
                encrypted_key = wallet_data.get('encrypted_private_key')
                if encrypted_key:
                    self.encrypted_private_key = base64.b64decode(encrypted_key.encode())
                    if self.current_wallet_address in self.wallets:
                        self.wallets[self.current_wallet_address]['encrypted_private_key'] = self.encrypted_private_key
            
            self.refresh_balance()
            
            if password and self.encrypted_private_key and self.current_wallet_address:
                return self.unlock_wallet(self.current_wallet_address, password)
            
            print(f"DEBUG: Wallet loaded from {filepath}")
            print(f"DEBUG: Total wallets: {len(self.wallets)}")
            print(f"DEBUG: Current wallet: {self.current_wallet_address}")
            
            return True
        except Exception as e:
            print(f"Error loading wallet: {e}")
            return False
    
    def debug_crypto_info(self):
        """Debug cryptographic information"""
        print("\n" + "="*60)
        print("SM2 CRYPTOGRAPHY DEBUG INFO")
        print("="*60)
        
        if self.current_wallet_address:
            print(f"Current Address: {self.address}")
            
            # Check address format
            if self.address.startswith('LUN_'):
                print("✅ Address has correct LUN_ prefix")
            else:
                print("❌ Address missing LUN_ prefix")
            
            # Check private key
            if self.private_key:
                print(f"Private Key: {self.private_key[:16]}...")
                if len(self.private_key) == 64:
                    print(f"✅ Private key is 256-bit (64 hex chars)")
                else:
                    print(f"❌ Private key has wrong length: {len(self.private_key)} chars")
            else:
                print("❌ No private key available")
            
            # Check public key
            if self.public_key:
                print(f"Public Key: {self.public_key[:16]}...")
                if len(self.public_key) >= 128:
                    print(f"✅ Public key is valid length")
                else:
                    print(f"❌ Public key too short: {len(self.public_key)} chars")
            else:
                print("❌ No public key available")
            
            # Check cryptographic standard
            wallet_data = self.wallets.get(self.address, {})
            if wallet_data.get('crypto_standard') == 'SM2_GB/T_32918':
                print("✅ Using SM2 GB/T 32918 East Asian cryptography standard")
            else:
                print("❌ Not using SM2 cryptography standard")
            
            # Test signing
            if self.private_key and not self.is_locked:
                test_sig = self._sign_transaction_data({"test": "data"})
                if test_sig:
                    print(f"✅ Can sign transactions, signature: {test_sig[:16]}...")
                else:
                    print("❌ Cannot sign transactions")
        else:
            print("❌ No wallet selected")
        
        print("="*60)