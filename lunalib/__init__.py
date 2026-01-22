"""
Luna Library - Complete cryptocurrency wallet and mining system
"""
import os

from .core.wallet import LunaWallet
from .mining.miner import GenesisMiner
from .gtx.genesis import GTXGenesis
from .transactions.transactions import TransactionManager
from .core.blockchain import BlockchainManager
from .core.mempool import MempoolManager
from .core.wallet_manager import WalletStateManager, get_wallet_manager

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("lunalib")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"
__all__ = [
    'LunaWallet', 
    'GenesisMiner', 
    'GTXGenesis', 
    'TransactionManager',
    'BlockchainManager',
    'MempoolManager',
    'WalletStateManager',
    'get_wallet_manager'
]