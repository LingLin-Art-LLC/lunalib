# lunalib/wallet_manager.py
"""
Unified Wallet State Manager

Provides a single source of truth for wallet balances, transaction history,
pending transactions, transfers, and rewards. Scans the blockchain once and
efficiently distributes transactions to multiple wallets while maintaining
real-time updates available to the UI.
"""

import threading
import time
import os
import re
from collections import deque
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime
from lunalib.utils.console import print_info, print_success, print_warn
from lunalib.utils.formatting import format_amount
from lunalib.utils.validation import is_valid_address


class TransactionType(Enum):
    """Types of transactions"""
    TRANSFER = "transfer"
    REWARD = "reward"
    GTX_GENESIS = "gtx_genesis"
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class TransactionStatus(Enum):
    """Status of a transaction"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class Transaction:
    """Represents a single transaction"""
    hash: str
    type: str
    from_address: str
    to_address: str
    amount: float
    fee: float = 0.0
    timestamp: int = 0
    status: str = TransactionStatus.PENDING.value
    block_height: Optional[int] = None
    confirmations: int = 0
    memo: str = ""
    direction: str = ""  # 'incoming', 'outgoing', or '' for self
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary"""
        return asdict(self)


@dataclass
class WalletBalance:
    """Wallet balance information"""
    total_balance: float = 0.0
    available_balance: float = 0.0
    pending_incoming: float = 0.0
    pending_outgoing: float = 0.0
    confirmed_balance: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data.update({
            "total_balance_display": format_amount(self.total_balance),
            "available_balance_display": format_amount(self.available_balance),
            "pending_incoming_display": format_amount(self.pending_incoming),
            "pending_outgoing_display": format_amount(self.pending_outgoing),
            "confirmed_balance_display": format_amount(self.confirmed_balance),
        })
        return data


@dataclass
class WalletState:
    """Complete state for a single wallet"""
    address: str
    balance: WalletBalance = field(default_factory=WalletBalance)
    confirmed_transactions: List[Transaction] = field(default_factory=list)
    pending_transactions: List[Transaction] = field(default_factory=list)
    confirmed_transfers: List[Transaction] = field(default_factory=list)
    pending_transfers: List[Transaction] = field(default_factory=list)
    rewards: List[Transaction] = field(default_factory=list)
    genesis_transactions: List[Transaction] = field(default_factory=list)
    last_updated: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, converting transactions to dicts"""
        return {
            'address': self.address,
            'balance': self.balance.to_dict(),
            'confirmed_transactions': [tx.to_dict() for tx in self.confirmed_transactions],
            'pending_transactions': [tx.to_dict() for tx in self.pending_transactions],
            'confirmed_transfers': [tx.to_dict() for tx in self.confirmed_transfers],
            'pending_transfers': [tx.to_dict() for tx in self.pending_transfers],
            'rewards': [tx.to_dict() for tx in self.rewards],
            'genesis_transactions': [tx.to_dict() for tx in self.genesis_transactions],
            'last_updated': self.last_updated
        }


class WalletStateManager:
    """
    Unified wallet manager that:
    1. Scans blockchain once for all wallets
    2. Efficiently categorizes transactions (transfers, rewards, genesis)
    3. Tracks pending transactions from mempool
    4. Maintains real-time balance calculations
    5. Provides immediate UI updates via callbacks
    """
    
    def __init__(self):
        self.wallet_states: Dict[str, WalletState] = {}
        self.state_lock = threading.RLock()
        self.balance_callbacks: List[Callable] = []
        self.transaction_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []
        # Cache settings
        self.max_confirmed_cache = int(os.getenv("LUNALIB_CONFIRMED_CACHE", "5000"))
        self.max_pending_cache = int(os.getenv("LUNALIB_PENDING_CACHE", "2000"))
        # Cache for pending transactions to avoid duplicate processing
        self.processed_pending_hashes: Set[str] = set()
        self.processed_confirmed_hashes: Set[str] = set()
        self._pending_hash_order = deque()
        self._confirmed_hash_order = deque()
        # Sync control
        self.last_sync_time = 0
        self.sync_in_progress = False
        self.last_blockchain_height = 0
        self._last_balance_snapshot: Dict[str, Dict] = {}
        self._last_tx_hashes: Dict[str, Dict[str, Set[str]]] = {}
        self._callback_debounce = float(os.getenv("LUNALIB_UI_DEBOUNCE", "0.25"))
        self._pending_balance_changes: Dict[str, Dict] = {}
        self._pending_tx_changes: Dict[str, Dict] = {}
        self._pending_event_changes: Dict[str, Dict] = {}
        self._callback_timer: Optional[threading.Timer] = None
        self._seen_tx_cache = int(os.getenv("LUNALIB_SEEN_TX_CACHE", "20000"))
        self._seen_tx_hashes: Set[str] = set()
        self._seen_tx_order = deque()
        self._confirmed_signature: Dict[str, str] = {}
        self._mempool_signature: Dict[str, str] = {}

    def _track_hash(self, tx_hash: str, cache: Set[str], order: deque, max_size: int) -> None:
        if not tx_hash or tx_hash in cache:
            return
        cache.add(tx_hash)
        order.append(tx_hash)
        while len(order) > max_size:
            old = order.popleft()
            cache.discard(old)

    def _track_seen_hash(self, tx_hash: str) -> None:
        if not tx_hash or tx_hash in self._seen_tx_hashes:
            return
        self._seen_tx_hashes.add(tx_hash)
        self._seen_tx_order.append(tx_hash)
        while len(self._seen_tx_order) > self._seen_tx_cache:
            old = self._seen_tx_order.popleft()
            self._seen_tx_hashes.discard(old)

    def _merge_transactions(self, existing: List[Transaction], new: List[Transaction], max_items: int) -> List[Transaction]:
        if not new and not existing:
            return []

        merged: Dict[str, Transaction] = {tx.hash: tx for tx in existing if tx.hash}
        for tx in new:
            if tx.hash:
                merged[tx.hash] = tx
        sorted_list = sorted(merged.values(), key=lambda t: t.timestamp or 0, reverse=True)
        if max_items > 0:
            sorted_list = sorted_list[:max_items]
        return sorted_list

    def _trim_transactions(self, items: List[Transaction], max_items: int) -> List[Transaction]:
        if max_items <= 0:
            return items
        return sorted(items, key=lambda t: t.timestamp or 0, reverse=True)[:max_items]
        
    # =========================================================================
    # Address normalization
    # =========================================================================
    
    def _normalize_address(self, addr: str) -> str:
        """Normalize address for comparison"""
        if not addr:
            return ''
        addr_str = str(addr).strip("'\" ").lower()
        return addr_str[4:] if addr_str.startswith('lun_') else addr_str
    
    def _addresses_match(self, addr1: str, addr2: str) -> bool:
        """Check if two addresses match (after normalization)"""
        return self._normalize_address(addr1) == self._normalize_address(addr2)

    def _parse_amount(self, value, default: float = 0.0) -> float:
        """Parse numeric amounts that may include unit suffixes like '3.0JS:3'."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                return default
        try:
            return float(value)
        except Exception:
            text = str(value)
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match:
                try:
                    return float(match.group(0))
                except Exception:
                    return default
        return default
    
    # =========================================================================
    # Wallet management
    # =========================================================================
    
    def register_wallet(self, address: str) -> WalletState:
        """Register a new wallet to track"""
        if not is_valid_address(address):
            print_warn(f"⚠️  Invalid wallet address rejected: {address}")
            return WalletState(address=address)
        with self.state_lock:
            if address not in self.wallet_states:
                self.wallet_states[address] = WalletState(address=address)
                print_success(f"📱 Registered wallet: {address}")
            return self.wallet_states[address]
    
    def register_wallets(self, addresses: List[str]) -> Dict[str, WalletState]:
        """Register multiple wallets"""
        states = {}
        for addr in addresses:
            states[addr] = self.register_wallet(addr)
        return states
    
    def get_wallet_state(self, address: str) -> Optional[WalletState]:
        """Get current state of a wallet"""
        with self.state_lock:
            return self.wallet_states.get(address)
    
    def get_all_wallet_states(self) -> Dict[str, WalletState]:
        """Get states for all registered wallets"""
        with self.state_lock:
            return dict(self.wallet_states)
    
    # =========================================================================
    # Transaction categorization and processing
    # =========================================================================
    
    def _categorize_confirmed_transaction(self, tx: Transaction, address: str) -> List[str]:
        """
        Categorize a confirmed transaction and return which lists to add it to.
        Returns list of category names.
        """
        categories = ['confirmed_transactions']
        
        tx_type_lower = (tx.type or '').lower()
        
        # Categorize by type
        reward_sender = str(tx.from_address or '').lower()
        if (tx_type_lower == 'reward' or reward_sender in {'ling country', 'ling country mines', 'foreign exchange', 'network', 'block_reward', 'mining_reward', 'coinbase'}) and self._addresses_match(tx.to_address, address):
            categories.append('rewards')
        elif tx_type_lower == 'gtx_genesis':
            categories.append('genesis_transactions')
        elif tx_type_lower in ['transfer', '']:
            categories.append('confirmed_transfers')
        
        return categories
    
    def _categorize_pending_transaction(self, tx: Transaction, address: str) -> List[str]:
        """
        Categorize a pending transaction and return which lists to add it to.
        Returns list of category names.
        """
        categories = ['pending_transactions']
        
        tx_type_lower = (tx.type or '').lower()
        
        # Categorize by type
        if tx_type_lower == 'reward' and self._addresses_match(tx.to_address, address):
            categories.append('rewards')
        elif tx_type_lower == 'gtx_genesis':
            categories.append('genesis_transactions')
        elif tx_type_lower in ['transfer', '']:
            categories.append('pending_transfers')
        
        return categories
    
    def _process_blockchain_transactions(self, raw_transactions: Dict[str, List[Dict]]) -> Dict[str, List[Transaction]]:
        """
        Convert raw blockchain transactions to Transaction objects and
        categorize them by address and type.
        
        Returns mapping of address -> list of Transaction objects
        """
        processed = {}
        
        norm_cache = {addr: self._normalize_address(addr) for addr in raw_transactions.keys()}

        for address, raw_txs in raw_transactions.items():
            transactions = []
            address_norm = norm_cache.get(address, self._normalize_address(address))
            
            for raw_tx in raw_txs:
                try:
                    tx_hash = raw_tx.get('hash', '')
                    if tx_hash and (tx_hash in self.processed_confirmed_hashes or tx_hash in self._seen_tx_hashes):
                        continue

                    # Determine transaction direction
                    direction = ""
                    from_norm = self._normalize_address(raw_tx.get('from'))
                    to_norm = self._normalize_address(raw_tx.get('to'))
                    if from_norm == address_norm:
                        direction = 'outgoing'
                    if to_norm == address_norm:
                        direction = 'incoming'
                    
                    # Create Transaction object
                    tx = Transaction(
                        hash=tx_hash,
                        type=raw_tx.get('type', 'transfer'),
                        from_address=raw_tx.get('from', ''),
                        to_address=raw_tx.get('to', ''),
                        amount=self._parse_amount(raw_tx.get('amount', 0)),
                        fee=self._parse_amount(raw_tx.get('fee', 0)),
                        timestamp=int(raw_tx.get('timestamp', 0)),
                        status=TransactionStatus.CONFIRMED.value,
                        block_height=raw_tx.get('block_height'),
                        confirmations=raw_tx.get('confirmations', 1),
                        memo=raw_tx.get('memo', ''),
                        direction=direction
                    )
                    
                    # Skip if we've already processed this
                    if tx.hash not in self.processed_confirmed_hashes:
                        transactions.append(tx)
                        self._track_hash(tx.hash, self.processed_confirmed_hashes, self._confirmed_hash_order, self.max_confirmed_cache)
                        self._track_seen_hash(tx.hash)
                        
                except Exception as e:
                    print(f"⚠️  Error processing transaction {raw_tx.get('hash')}: {e}")
                    continue
            
            if transactions:
                processed[address] = transactions
        
        return processed
    
    def _process_mempool_transactions(self, raw_transactions: Dict[str, List[Dict]]) -> Dict[str, List[Transaction]]:
        """
        Convert raw mempool transactions to Transaction objects.
        
        Returns mapping of address -> list of Transaction objects
        """
        processed = {}
        
        norm_cache = {addr: self._normalize_address(addr) for addr in raw_transactions.keys()}

        for address, raw_txs in raw_transactions.items():
            transactions = []
            address_norm = norm_cache.get(address, self._normalize_address(address))
            
            seen_hashes: Set[str] = set()
            for raw_tx in raw_txs:
                try:
                    tx_hash = raw_tx.get('hash', '')
                    if tx_hash and (tx_hash in self.processed_pending_hashes or tx_hash in self._seen_tx_hashes):
                        continue

                    # Determine transaction direction
                    direction = ""
                    from_norm = self._normalize_address(raw_tx.get('from'))
                    to_norm = self._normalize_address(raw_tx.get('to'))
                    if from_norm == address_norm:
                        direction = 'outgoing'
                    if to_norm == address_norm:
                        direction = 'incoming'
                    
                    # Create Transaction object
                    tx = Transaction(
                        hash=tx_hash,
                        type=raw_tx.get('type', 'transfer'),
                        from_address=raw_tx.get('from', ''),
                        to_address=raw_tx.get('to', ''),
                        amount=self._parse_amount(raw_tx.get('amount', 0)),
                        fee=self._parse_amount(raw_tx.get('fee', 0)),
                        timestamp=int(raw_tx.get('timestamp', 0)),
                        status=TransactionStatus.PENDING.value,
                        memo=raw_tx.get('memo', ''),
                        direction=direction
                    )
                    
                    if tx.hash and tx.hash in seen_hashes:
                        continue
                    transactions.append(tx)
                    seen_hashes.add(tx.hash)
                    self._track_hash(tx.hash, self.processed_pending_hashes, self._pending_hash_order, self.max_pending_cache)
                    self._track_seen_hash(tx.hash)
                        
                except Exception as e:
                    print(f"⚠️  Error processing pending transaction {raw_tx.get('hash')}: {e}")
                    continue
            
            if transactions:
                processed[address] = transactions
        
        return processed
    
    # =========================================================================
    # Balance calculation
    # =========================================================================
    
    def _calculate_balance_from_transactions(self, address: str, 
                                           confirmed_txs: List[Transaction],
                                           pending_txs: List[Transaction]) -> WalletBalance:
        """Calculate wallet balance from confirmed and pending transactions"""
        
        confirmed_balance = 0.0
        pending_incoming = 0.0
        pending_outgoing = 0.0
        
        # Process confirmed transactions
        for tx in confirmed_txs:
            if self._addresses_match(tx.from_address, address):
                # Outgoing: subtract amount and fee
                confirmed_balance -= (tx.amount + tx.fee)
            elif self._addresses_match(tx.to_address, address):
                # Incoming: add amount
                confirmed_balance += tx.amount
            elif (tx.type == 'reward' or str(tx.from_address or '').lower() in {'ling country', 'ling country mines', 'foreign exchange', 'network', 'block_reward', 'mining_reward', 'coinbase'}) and self._addresses_match(tx.to_address, address):
                # Reward received
                confirmed_balance += tx.amount
        
        # Process pending transactions
        for tx in pending_txs:
            if self._addresses_match(tx.from_address, address):
                # Pending outgoing
                pending_outgoing += (tx.amount + tx.fee)
            elif self._addresses_match(tx.to_address, address):
                # Pending incoming
                pending_incoming += tx.amount
        
        # Calculate derived balances
        confirmed_balance = max(0.0, confirmed_balance)
        total_balance = confirmed_balance + pending_incoming
        available_balance = max(0.0, confirmed_balance - pending_outgoing)
        
        return WalletBalance(
            total_balance=total_balance,
            available_balance=available_balance,
            pending_incoming=pending_incoming,
            pending_outgoing=pending_outgoing,
            confirmed_balance=confirmed_balance
        )
    
    # =========================================================================
    # Sync operations (blockchain + mempool)
    # =========================================================================
    
    def sync_wallets_from_sources(self, blockchain_transactions: Dict[str, List[Dict]],
                                   mempool_transactions: Dict[str, List[Dict]]):
        """
        Main sync method: takes raw blockchain and mempool data and updates all wallets.
        
        This is designed to be called with data from:
        - blockchain.scan_transactions_for_addresses()
        - mempool.get_pending_transactions_for_addresses()
        
        The method efficiently:
        1. Processes and categorizes transactions
        2. Updates all wallet states
        3. Calculates balances
        4. Triggers callbacks for UI updates
        """
        
        with self.state_lock:
            print_info(f"\n🔄 Syncing wallets...")
            sync_start = time.time()
            
            # Process blockchain transactions (diff-based)
            confirmed_changed: Dict[str, List[Dict]] = {}
            for addr in self.wallet_states.keys():
                txs = blockchain_transactions.get(addr, [])
                hashes = sorted([tx.get("hash", "") for tx in txs if tx.get("hash")])
                signature = "|".join(hashes)
                if self._confirmed_signature.get(addr) != signature:
                    confirmed_changed[addr] = txs
                    self._confirmed_signature[addr] = signature

            confirmed_map = self._process_blockchain_transactions(confirmed_changed)
            for addr in self.wallet_states.keys():
                if addr not in confirmed_changed:
                    confirmed_map[addr] = self.wallet_states[addr].confirmed_transactions

            # Process mempool transactions (diff-based)
            mempool_changed: Dict[str, List[Dict]] = {}
            for addr in self.wallet_states.keys():
                txs = mempool_transactions.get(addr, [])
                hashes = sorted([tx.get("hash", "") for tx in txs if tx.get("hash")])
                signature = "|".join(hashes)
                if self._mempool_signature.get(addr) != signature:
                    mempool_changed[addr] = txs
                    self._mempool_signature[addr] = signature

            pending_map = self._process_mempool_transactions(mempool_changed)
            for addr in self.wallet_states.keys():
                if addr not in mempool_changed:
                    pending_map[addr] = self.wallet_states[addr].pending_transactions
            
            # Get all addresses we're tracking
            all_addresses = set(self.wallet_states.keys())
            
            # Track changes for event-driven UI updates
            balance_changes: Dict[str, Dict] = {}
            tx_changes: Dict[str, Dict] = {}
            event_changes: Dict[str, Dict] = {}

            # Update each wallet
            for address in all_addresses:
                state = self.wallet_states[address]
                
                # Get transactions for this wallet
                confirmed_txs = confirmed_map.get(address, [])
                pending_txs = pending_map.get(address, [])

                # Merge confirmed history (mini cache)
                state.confirmed_transactions = self._merge_transactions(
                    state.confirmed_transactions, confirmed_txs, self.max_confirmed_cache
                )

                # Pending should reflect current mempool; keep a small cache
                state.pending_transactions = self._trim_transactions(
                    pending_txs, self.max_pending_cache
                )

                # Remove any pending that are now confirmed
                confirmed_hashes = {tx.hash for tx in state.confirmed_transactions if tx.hash}
                state.pending_transactions = [
                    tx for tx in state.pending_transactions if tx.hash not in confirmed_hashes
                ]
                
                # Clear specialized lists and repopulate
                state.confirmed_transfers = []
                state.pending_transfers = []
                state.rewards = []
                state.genesis_transactions = []
                
                # Categorize confirmed transactions
                for tx in state.confirmed_transactions:
                    categories = self._categorize_confirmed_transaction(tx, address)
                    for category in categories:
                        if category == 'confirmed_transfers':
                            state.confirmed_transfers.append(tx)
                        elif category == 'rewards':
                            state.rewards.append(tx)
                        elif category == 'genesis_transactions':
                            state.genesis_transactions.append(tx)
                
                # Categorize pending transactions
                for tx in state.pending_transactions:
                    categories = self._categorize_pending_transaction(tx, address)
                    for category in categories:
                        if category == 'pending_transfers':
                            state.pending_transfers.append(tx)
                        elif category == 'rewards':
                            if tx not in state.rewards:
                                state.rewards.append(tx)
                        elif category == 'genesis_transactions':
                            if tx not in state.genesis_transactions:
                                state.genesis_transactions.append(tx)
                
                # Calculate new balance
                state.balance = self._calculate_balance_from_transactions(
                    address, state.confirmed_transactions, state.pending_transactions
                )
                
                # Update timestamp
                state.last_updated = time.time()

                # Detect balance changes
                current_balance = state.balance.to_dict()
                last_balance = self._last_balance_snapshot.get(address)
                if last_balance != current_balance:
                    balance_changes[address] = current_balance

                # Detect transaction changes
                current_confirmed = {tx.hash for tx in state.confirmed_transactions if tx.hash}
                current_pending = {tx.hash for tx in state.pending_transactions if tx.hash}
                last_sets = self._last_tx_hashes.get(address, {"confirmed": set(), "pending": set()})
                last_confirmed = last_sets.get("confirmed", set())
                last_pending = last_sets.get("pending", set())

                new_confirmed_hashes = current_confirmed - last_confirmed
                new_pending_hashes = current_pending - last_pending
                promoted_hashes = current_confirmed & last_pending
                cleared_pending_hashes = last_pending - current_pending - promoted_hashes

                if new_confirmed_hashes or new_pending_hashes or promoted_hashes or cleared_pending_hashes:
                    tx_changes[address] = {
                        'confirmed': [tx.to_dict() for tx in state.confirmed_transactions],
                        'pending': [tx.to_dict() for tx in state.pending_transactions],
                        'transfers': {
                            'confirmed': [tx.to_dict() for tx in state.confirmed_transfers],
                            'pending': [tx.to_dict() for tx in state.pending_transfers],
                        },
                        'rewards': [tx.to_dict() for tx in state.rewards],
                    }

                    event_changes[address] = {
                        'new_confirmed': [tx.to_dict() for tx in state.confirmed_transactions if tx.hash in new_confirmed_hashes],
                        'new_pending': [tx.to_dict() for tx in state.pending_transactions if tx.hash in new_pending_hashes],
                        'promoted': [tx.to_dict() for tx in state.confirmed_transactions if tx.hash in promoted_hashes],
                        'pending_cleared': list(cleared_pending_hashes),
                    }

                # Update last snapshots
                self._last_balance_snapshot[address] = current_balance
                self._last_tx_hashes[address] = {
                    "confirmed": current_confirmed,
                    "pending": current_pending,
                }
            
            sync_time = time.time() - sync_start
            print_success(f"✅ Sync complete in {sync_time:.2f}s")
            
            # Trigger callbacks only when changes occur (debounced)
            if balance_changes or tx_changes or event_changes:
                self._queue_callback_updates(balance_changes, tx_changes, event_changes)
    
    def sync_wallets_background(self, get_blockchain_data: Callable,
                               get_mempool_data: Callable,
                               poll_interval: int = 30):
        """
        Start a background thread that periodically syncs wallet data.
        
        Parameters:
            get_blockchain_data: Callable that returns Dict[str, List[Dict]]
            get_mempool_data: Callable that returns Dict[str, List[Dict]]
            poll_interval: Seconds between syncs
        """
        
        def sync_loop():
            while True:
                try:
                    addresses = list(self.wallet_states.keys())
                    if addresses:
                        blockchain_txs = get_blockchain_data(addresses)
                        mempool_txs = get_mempool_data(addresses)
                        self.sync_wallets_from_sources(blockchain_txs, mempool_txs)
                    
                    time.sleep(poll_interval)
                except Exception as e:
                    print(f"⚠️  Background sync error: {e}")
                    time.sleep(poll_interval)
        
        thread = threading.Thread(target=sync_loop, daemon=True)
        thread.start()
        print_info(f"🔄 Started background sync thread (interval: {poll_interval}s)")
    
    # =========================================================================
    # Callbacks for UI updates
    # =========================================================================
    
    def on_balance_update(self, callback: Callable):
        """Register callback for balance updates"""
        self.balance_callbacks.append(callback)
    
    def on_transaction_update(self, callback: Callable):
        """Register callback for transaction updates"""
        self.transaction_callbacks.append(callback)

    def on_event_update(self, callback: Callable):
        """Register callback for change-only event updates"""
        self.event_callbacks.append(callback)
    
    def _trigger_balance_updates(self, balance_data: Optional[Dict] = None):
        """Trigger all balance update callbacks"""
        if balance_data is None:
            with self.state_lock:
                balance_data = {addr: state.balance.to_dict() for addr, state in self.wallet_states.items()}
        
        for callback in self.balance_callbacks:
            try:
                callback(balance_data)
            except Exception as e:
                print(f"⚠️  Balance callback error: {e}")
    
    def _trigger_transaction_updates(self, transaction_data: Optional[Dict] = None):
        """Trigger all transaction update callbacks"""
        if transaction_data is None:
            with self.state_lock:
                transaction_data = {
                    addr: {
                        'confirmed': [tx.to_dict() for tx in state.confirmed_transactions],
                        'pending': [tx.to_dict() for tx in state.pending_transactions],
                        'transfers': {
                            'confirmed': [tx.to_dict() for tx in state.confirmed_transfers],
                            'pending': [tx.to_dict() for tx in state.pending_transfers],
                        },
                        'rewards': [tx.to_dict() for tx in state.rewards],
                    }
                    for addr, state in self.wallet_states.items()
                }
        
        for callback in self.transaction_callbacks:
            try:
                callback(transaction_data)
            except Exception as e:
                print(f"⚠️  Transaction callback error: {e}")

    def _trigger_event_updates(self, event_data: Dict):
        """Trigger all change-only event callbacks"""
        for callback in self.event_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                print(f"⚠️  Event callback error: {e}")

    def _queue_callback_updates(
        self,
        balance_changes: Dict[str, Dict],
        tx_changes: Dict[str, Dict],
        event_changes: Dict[str, Dict],
    ):
        """Coalesce UI callbacks to reduce sync chatter."""
        with self.state_lock:
            if balance_changes:
                self._pending_balance_changes.update(balance_changes)
            if tx_changes:
                self._pending_tx_changes.update(tx_changes)
            if event_changes:
                self._pending_event_changes.update(event_changes)

            if self._callback_timer and self._callback_timer.is_alive():
                return

            self._callback_timer = threading.Timer(self._callback_debounce, self._flush_callback_updates)
            self._callback_timer.daemon = True
            self._callback_timer.start()

    def _flush_callback_updates(self):
        """Flush debounced callback payloads."""
        with self.state_lock:
            balance_payload = self._pending_balance_changes
            tx_payload = self._pending_tx_changes
            event_payload = self._pending_event_changes
            self._pending_balance_changes = {}
            self._pending_tx_changes = {}
            self._pending_event_changes = {}

        if balance_payload:
            self._trigger_balance_updates(balance_payload)
        if tx_payload:
            self._trigger_transaction_updates(tx_payload)
        if event_payload:
            self._trigger_event_updates(event_payload)
    
    # =========================================================================
    # Query methods for UI
    # =========================================================================
    
    def get_balance(self, address: str) -> Optional[Dict]:
        """Get balance for a specific wallet"""
        state = self.get_wallet_state(address)
        if state:
            return state.balance.to_dict()
        return None
    
    def get_all_balances(self) -> Dict[str, Dict]:
        """Get balances for all wallets"""
        with self.state_lock:
            balances = {}
            for addr, state in self.wallet_states.items():
                balances[addr] = state.balance.to_dict()
            return balances
    
    def get_transactions(self, address: str, transaction_type: str = 'all') -> List[Dict]:
        """
        Get transactions for a wallet.
        
        Types: 'all', 'confirmed', 'pending', 'transfers', 'rewards', 'genesis'
        """
        state = self.get_wallet_state(address)
        if not state:
            return []
        
        result = []
        
        if transaction_type in ['all', 'confirmed']:
            result.extend([tx.to_dict() for tx in state.confirmed_transactions])
        
        if transaction_type in ['all', 'pending']:
            result.extend([tx.to_dict() for tx in state.pending_transactions])
        
        if transaction_type in ['transfers']:
            result.extend([tx.to_dict() for tx in state.confirmed_transfers])
            result.extend([tx.to_dict() for tx in state.pending_transfers])
        
        if transaction_type in ['all', 'rewards']:
            result.extend([tx.to_dict() for tx in state.rewards])
        
        if transaction_type in ['all', 'genesis']:
            result.extend([tx.to_dict() for tx in state.genesis_transactions])
        
        # Sort by timestamp (most recent first)
        result.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return result
    
    def get_wallet_summary(self, address: str) -> Optional[Dict]:
        """Get complete summary for a wallet (balance + all transaction types)"""
        state = self.get_wallet_state(address)
        if not state:
            return None
        
        return {
            'address': address,
            'balance': state.balance.to_dict(),
            'transaction_counts': {
                'confirmed': len(state.confirmed_transactions),
                'pending': len(state.pending_transactions),
                'transfers_confirmed': len(state.confirmed_transfers),
                'transfers_pending': len(state.pending_transfers),
                'rewards': len(state.rewards),
                'genesis': len(state.genesis_transactions),
            },
            'transactions': {
                'confirmed': [tx.to_dict() for tx in state.confirmed_transactions[-10:]],  # Last 10
                'pending': [tx.to_dict() for tx in state.pending_transactions],
                'rewards': [tx.to_dict() for tx in state.rewards[-10:]],
            },
            'last_updated': state.last_updated
        }
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all registered wallets"""
        with self.state_lock:
            summaries = {}
            for address in self.wallet_states:
                summary = self.get_wallet_summary(address)
                if summary:
                    summaries[address] = summary
            return summaries
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def clear_all_caches(self):
        """Clear all transaction caches (for testing)"""
        with self.state_lock:
            self.processed_pending_hashes.clear()
            self.processed_confirmed_hashes.clear()
    
    def reset_wallet(self, address: str):
        """Reset a wallet to empty state"""
        with self.state_lock:
            if address in self.wallet_states:
                self.wallet_states[address] = WalletState(address=address)
    
    def remove_wallet(self, address: str):
        """Remove a wallet from tracking"""
        with self.state_lock:
            if address in self.wallet_states:
                del self.wallet_states[address]


# Global singleton instance
_wallet_state_manager = WalletStateManager()


def get_wallet_manager() -> WalletStateManager:
    """Get the global wallet state manager instance"""
    return _wallet_state_manager
