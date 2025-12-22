# wallet.py
import time
import hashlib
import json
from cryptography.fernet import Fernet
import base64
import os

class LunaWallet:
    """Luna wallet implementation with proper key management"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.wallets = {}  # Main wallet storage: {address: wallet_data}
        self.current_wallet_address = None  # Track which wallet is active
        
        # Initialize with an empty current wallet state
        self._reset_current_wallet()
        
    def _reset_current_wallet(self):
        """Reset current wallet to empty state"""
        self.address = None
        self.balance = 0.0
        self.available_balance = 0.0
        self.created = time.time()
        self.private_key = None
        self.public_key = None
        self.encrypted_private_key = None
        self.label = "New Wallet"
        self.is_locked = True
        self.pending_transactions = {}  # {tx_hash: amount}

    def _generate_address(self):
        """Generate wallet address"""
        base_data = f"LUN_{int(time.time()*1000)}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:32]
    
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
        return {
            'address': self.address,
            'balance': self.balance,
            'available_balance': self.available_balance,
            'pending_amount': self.get_pending_amount(),
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
            'is_locked': True,
            'pending_transactions': {}
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
            'is_locked': True,
            'pending_transactions': {}
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
        self.pending_transactions = wallet_data.get('pending_transactions', {})
        self._recalculate_available_balance()
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
                
            # Ensure pending_transactions exists
            if 'pending_transactions' not in wallet_data:
                wallet_data['pending_transactions'] = {}
            
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
        self._recalculate_available_balance()
        
        # Also update in wallets collection
        if self.current_wallet_address and self.current_wallet_address in self.wallets:
            self.wallets[self.current_wallet_address]['balance'] = self.balance
            self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
        
        return True
    
    def _recalculate_available_balance(self):
        """Recalculate available balance based on pending transactions"""
        pending_amount = sum(self.pending_transactions.values())
        self.available_balance = max(0.0, self.balance - pending_amount)
    
    def add_pending_transaction(self, tx_hash: str, amount: float) -> bool:
        """Add a transaction to pending list (e.g., when sent to mempool)"""
        if amount > self.available_balance:
            return False
        self.pending_transactions[tx_hash] = float(amount)
        self._recalculate_available_balance()
        
        # Update in wallets collection
        if self.current_wallet_address and self.current_wallet_address in self.wallets:
            self.wallets[self.current_wallet_address]['pending_transactions'] = self.pending_transactions
            self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
        
        return True
    
    def confirm_pending_transaction(self, tx_hash: str) -> bool:
        """Confirm a pending transaction (e.g., added to blockchain)"""
        if tx_hash in self.pending_transactions:
            del self.pending_transactions[tx_hash]
            self._recalculate_available_balance()
            
            # Update in wallets collection
            if self.current_wallet_address and self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address]['pending_transactions'] = self.pending_transactions
                self.wallets[self.current_wallet_address]['available_balance'] = self.available_balance
            
            return True
        return False
    
    def get_pending_amount(self) -> float:
        """Get total amount in pending transactions"""
        return sum(self.pending_transactions.values())
    
    def get_pending_transactions(self) -> dict:
        """Get all pending transactions"""
        return self.pending_transactions.copy()
    
    def clear_pending_transaction(self, tx_hash: str) -> bool:
        """Remove a pending transaction (e.g., if it failed)"""
        return self.confirm_pending_transaction(tx_hash)
    
    def get_balance(self):
        """Get current wallet balance"""
        return self.balance
    
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
                'pending_amount': self.get_pending_amount(),
                'pending_transactions': self.pending_transactions,
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
            
            # Ensure all wallets have pending_transactions field
            for addr, w_data in self.wallets.items():
                if 'pending_transactions' not in w_data:
                    w_data['pending_transactions'] = {}
            
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
            
            # If password provided and we have encrypted key, unlock
            if password and self.encrypted_private_key and self.current_wallet_address:
                return self.unlock_wallet(self.current_wallet_address, password)
            
            return True
        except Exception as e:
            print(f"Error loading wallet: {e}")
            return False