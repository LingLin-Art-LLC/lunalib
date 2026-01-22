# lunalib/wallet_sync_helper.py
"""
Wallet Sync Helper

Provides integration between LunaWallet, BlockchainManager, MempoolManager,
and the WalletStateManager for seamless real-time balance and transaction updates.
"""

import os
from typing import List, Dict, Optional, Callable
from .wallet_manager import get_wallet_manager


class WalletSyncHelper:
    """
    Helper class to sync LunaWallet with WalletStateManager using data from
    BlockchainManager and MempoolManager.
    """
    
    def __init__(self, wallet=None, blockchain=None, mempool=None):
        """
        Initialize sync helper.
        
        Parameters:
            wallet: LunaWallet instance
            blockchain: BlockchainManager instance
            mempool: MempoolManager instance
        """
        self.wallet = wallet
        self.blockchain = blockchain
        self.mempool = mempool
        self.state_manager = get_wallet_manager()
        
    def register_wallets_from_lunawallet(self) -> Dict:
        """Register all wallets from LunaWallet into the state manager"""
        if not self.wallet:
            print("⚠️  No wallet instance provided")
            return {}
        
        addresses = list(self.wallet.wallets.keys())
        if not addresses:
            print("⚠️  No wallets registered in LunaWallet")
            return {}
        
        print(f"📱 Registering {len(addresses)} wallets with state manager...")
        states = self.state_manager.register_wallets(addresses)
        print(f"✅ Registered {len(states)} wallets")
        
        return states
    
    def sync_wallets_now(self, on_scan_progress: Optional[Callable] = None) -> Dict:
        """
        Perform a single synchronization of all registered wallets.
        
        Returns: Dictionary of wallet addresses and their updated summaries
        """
        if not self.blockchain or not self.mempool:
            print("⚠️  Blockchain or mempool not provided")
            return {}
        
        addresses = list(self.state_manager.wallet_states.keys())
        if not addresses:
            print("⚠️  No wallets registered")
            return {}
        
        print(f"🔄 Syncing {len(addresses)} wallets...")
        
        try:
            # Get data from blockchain and mempool
            end_height = self.blockchain.get_blockchain_height()
            lookback = int(os.getenv("LUNALIB_WALLET_SYNC_LOOKBACK", "50"))
            if lookback < 0:
                lookback = 0
            cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "1") == "1"
            max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "500"))
            if end_height <= self.state_manager.last_blockchain_height:
                if lookback > 0:
                    start_height = max(0, end_height - lookback + 1)
                else:
                    start_height = end_height
            else:
                start_height = max(0, self.state_manager.last_blockchain_height + 1)
                if lookback > 0:
                    start_height = min(start_height, max(0, end_height - lookback + 1))

            blockchain_txs = self.blockchain.scan_transactions_for_addresses_filtered(
                addresses,
                start_height=start_height,
                end_height=end_height,
                include_rewards=True,
                include_transfers=True,
                include_gtx_genesis=False,
                cache_only=cache_only,
                max_range=max_range if max_range > 0 else None,
                progress_callback=on_scan_progress,
            )
            mempool_txs = self.mempool.get_pending_transactions_for_addresses(addresses)
            
            # Sync the state manager
            self.state_manager.sync_wallets_from_sources(blockchain_txs, mempool_txs)
            self.state_manager.last_blockchain_height = end_height
            
            # Update LunaWallet balances if available
            if self.wallet:
                self._update_lunawallet_balances()
            
            # Return summaries
            return self.state_manager.get_all_summaries()
            
        except Exception as e:
            print(f"❌ Sync error: {e}")
            return {}
    
    def _update_lunawallet_balances(self):
        """Update LunaWallet instance balances from state manager"""
        if not self.wallet:
            return
        
        balances = self.state_manager.get_all_balances()
        
        for address, balance_data in balances.items():
            if address in self.wallet.wallets:
                wallet_data = self.wallet.wallets[address]
                wallet_data['balance'] = balance_data['confirmed_balance']
                wallet_data['available_balance'] = balance_data['available_balance']
        
        # Update current wallet if one is selected
        if self.wallet.current_wallet_address in balances:
            balance_data = balances[self.wallet.current_wallet_address]
            self.wallet.balance = balance_data['confirmed_balance']
            self.wallet.available_balance = balance_data['available_balance']
    
    def start_continuous_sync(self, poll_interval: int = 30,
                             on_balance_update: Optional[Callable] = None,
                             on_transaction_update: Optional[Callable] = None,
                             on_scan_progress: Optional[Callable] = None) -> None:
        """
        Start continuous synchronization in the background.
        
        Parameters:
            poll_interval: Seconds between syncs
            on_balance_update: Callback function(balance_data) for balance updates
            on_transaction_update: Callback function(transaction_data) for transaction updates
        """
        
        if on_balance_update:
            self.state_manager.on_balance_update(on_balance_update)
        
        if on_transaction_update:
            self.state_manager.on_transaction_update(on_transaction_update)
        
        def get_blockchain_data(addresses):
            try:
                end_height = self.blockchain.get_blockchain_height()
                lookback = int(os.getenv("LUNALIB_WALLET_SYNC_LOOKBACK", "50"))
                if lookback < 0:
                    lookback = 0
                cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "1") == "1"
                max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "500"))
                if end_height <= self.state_manager.last_blockchain_height:
                    if lookback > 0:
                        start_height = max(0, end_height - lookback + 1)
                    else:
                        start_height = end_height
                else:
                    start_height = max(0, self.state_manager.last_blockchain_height + 1)
                    if lookback > 0:
                        start_height = min(start_height, max(0, end_height - lookback + 1))
                data = self.blockchain.scan_transactions_for_addresses_filtered(
                    addresses,
                    start_height=start_height,
                    end_height=end_height,
                    include_rewards=True,
                    include_transfers=True,
                    include_gtx_genesis=False,
                    cache_only=cache_only,
                    max_range=max_range if max_range > 0 else None,
                    progress_callback=on_scan_progress,
                )
                self.state_manager.last_blockchain_height = end_height
                return data
            except Exception as e:
                print(f"⚠️  Blockchain scan error: {e}")
                return {}
        
        def get_mempool_data(addresses):
            try:
                return self.mempool.get_pending_transactions_for_addresses(addresses)
            except Exception as e:
                print(f"⚠️  Mempool fetch error: {e}")
                return {}
        
        self.state_manager.sync_wallets_background(
            get_blockchain_data,
            get_mempool_data,
            poll_interval
        )
    
    def get_wallet_balance(self, address: str) -> Optional[Dict]:
        """Get current balance for a wallet"""
        return self.state_manager.get_balance(address)
    
    def get_wallet_transactions(self, address: str, tx_type: str = 'all') -> List[Dict]:
        """Get transactions for a wallet"""
        return self.state_manager.get_transactions(address, tx_type)
    
    def get_wallet_summary(self, address: str) -> Optional[Dict]:
        """Get complete summary for a wallet"""
        return self.state_manager.get_wallet_summary(address)
    
    def get_all_wallets_summary(self) -> Dict:
        """Get summaries for all wallets"""
        return self.state_manager.get_all_summaries()


# Convenience function to create sync helper
def create_wallet_sync_helper(wallet=None, blockchain=None, mempool=None) -> WalletSyncHelper:
    """Create a new WalletSyncHelper instance"""
    return WalletSyncHelper(wallet, blockchain, mempool)
