"""
Luna Library - Complete cryptocurrency wallet and mining system
Main library entry point exposing all core functionality
"""

from core.wallet import LunaWallet
from mining.miner import GenesisMiner
from gtx.genesis import GTXGenesis
from transactions.transactions import TransactionManager
from core.blockchain import BlockchainManager
from core.mempool import MempoolManager


class LunaLib:
    """Main library class exposing all Luna cryptocurrency functionality"""
    
    # Expose all main classes as class attributes
    Wallet = LunaWallet
    Miner = GenesisMiner
    GTX = GTXGenesis
    Transaction = TransactionManager
    Blockchain = BlockchainManager
    Mempool = MempoolManager
    
    @staticmethod
    def get_version():
        """Get the current library version"""
        return "1.0.0"
    
    @staticmethod
    def get_available_classes():
        """Get list of all available classes in the library"""
        return {
            'Wallet': 'LunaWallet - Cryptocurrency wallet management',
            'Miner': 'GenesisMiner - Mining operations', 
            'GTX': 'GTXGenesis - GTX token operations',
            'Transaction': 'TransactionManager - Transaction handling',
            'Blockchain': 'BlockchainManager - Blockchain operations with endpoint support',
            'Mempool': 'MempoolManager - Memory Pool management and endpoint'
        }


# Convenience functions for direct import with proper constructors
def create_wallet():
    """Create a new Luna wallet"""
    return LunaWallet()

def create_miner():
    """Create a new Genesis miner"""
    return GenesisMiner()

def create_blockchain_manager(endpoint_url="https://bank.linglin.art"):
    """Create a blockchain manager with optional endpoint URL"""
    return BlockchainManager(endpoint_url)

def create_mempool_manager(endpoint_url="https://bank.linglin.art"):
    """Create a blockchain manager with optional endpoint URL"""
    return MempoolManager(network_endpoints=[endpoint_url])

def get_transaction_manager():
    """Get transaction manager instance"""
    return TransactionManager()


# Export the same classes as __init__.py for consistency
__all__ = [
    'LunaLib',
    'LunaWallet', 
    'GenesisMiner', 
    'GTXGenesis', 
    'TransactionManager',
    'BlockchainManager',
    'MempoolManager',
    'create_wallet',
    'create_miner',
    'create_blockchain_manager',
    'create_mempool_manager',
    'get_transaction_manager'
]

# Direct exports
LunaWallet = LunaWallet
GenesisMiner = GenesisMiner
GTXGenesis = GTXGenesis
TransactionManager = TransactionManager
BlockchainManager = BlockchainManager
MempoolManager = MempoolManager