def update_all_balances(self):
    """Unified balance update: recalculates and persists balances."""
    # 残高計算前にキャッシュを最新化
    self._refresh_tx_caches()
    available = self.calculate_available_balance()
    # 計算値を必ずwalletsコレクションにも反映
    if self.current_wallet_address in self.wallets:
        self.wallets[self.current_wallet_address]["balance"] = self.balance
        self.wallets[self.current_wallet_address]["available_balance"] = self.available_balance
        w = self.wallets[self.current_wallet_address]
        if hasattr(self, 'db') and self.db:
            self.db.save_wallet(
                self.current_wallet_address,
                w.get("label", ""),
                w.get("public_key", ""),
                w.get("encrypted_private_key", b""),
                w.get("is_locked", True),
                w.get("created", 0),
                self.balance,
                available
            )
    from lunalib.utils.console import print_info
    print_info(f"[WALLET] update_all_balances: balance={self.balance}, available={self.available_balance}")

def _refresh_tx_caches(self):
    """Force refresh of confirmed and pending tx caches for the current wallet."""
    if not self.current_wallet_address:
        return
    from lunalib.core.blockchain import BlockchainManager
    from lunalib.core.mempool import MempoolManager
    blockchain = BlockchainManager()
    mempool = MempoolManager()
    cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
    max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "500"))
    confirmed = blockchain.scan_transactions_for_address_filtered(
        self.current_wallet_address,
        include_rewards=True,
        include_transfers=True,
        include_gtx_genesis=False,
        cache_only=cache_only,
        max_range=max_range if max_range > 0 else None,
    )
    pending = mempool.get_pending_transactions(self.current_wallet_address, fetch_remote=True)
    pending = self._filter_zero_amount_transfers(pending)
    norm = self._normalize_address(self.current_wallet_address)
    self._confirmed_tx_cache[self.current_wallet_address] = confirmed
    self._pending_tx_cache[self.current_wallet_address] = pending
    if norm:
        self._confirmed_tx_cache[norm] = confirmed
        self._pending_tx_cache[norm] = pending

import time
import json
import os
import secrets
import binascii
import base64
import hmac
import threading
from collections import OrderedDict
import concurrent.futures
from typing import Optional, Callable, Dict, List, Tuple
from ..core.crypto import KeyManager
from ..core.sm4 import SM4Cipher
from lunalib.config import apply_profile
from .wallet_db import WalletDB
from lunalib.utils.hash import derive_key_sm3, hmac_sm3
from lunalib.utils.formatting import format_amount

_WALLET_SALT = b"luna_wallet_salt"
_KDF_CACHE_LOCK = threading.Lock()
_KDF_CACHE: "OrderedDict[Tuple[bytes, int], bytes]" = OrderedDict()

apply_profile()


def _kdf_cache_key(password: str, iterations: int) -> Tuple[bytes, int]:
    try:
        pwd_bytes = password.encode()
    except Exception:
        pwd_bytes = str(password).encode()
    digest = hmac_sm3(pwd_bytes, _WALLET_SALT)
    return digest, iterations


def _derive_wallet_key(password: str) -> bytes:
    iterations = int(os.getenv("LUNALIB_WALLET_KDF_ITERATIONS", "100000"))
    if iterations < 1:
        iterations = 1

    cache_enabled = os.getenv("LUNALIB_WALLET_KDF_CACHE", "0") == "1"
    if not cache_enabled:
        return derive_key_sm3(password, _WALLET_SALT, iterations=iterations, dklen=32)

    cache_size = int(os.getenv("LUNALIB_WALLET_KDF_CACHE_SIZE", "16"))
    if cache_size < 1:
        cache_size = 1

    cache_key = _kdf_cache_key(password, iterations)
    with _KDF_CACHE_LOCK:
        cached = _KDF_CACHE.get(cache_key)
        if cached is not None:
            _KDF_CACHE.move_to_end(cache_key)
            return cached

    derived = derive_key_sm3(password, _WALLET_SALT, iterations=iterations, dklen=32)
    with _KDF_CACHE_LOCK:
        _KDF_CACHE[cache_key] = derived
        if len(_KDF_CACHE) > cache_size:
            _KDF_CACHE.popitem(last=False)
    return derived


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        counter_bytes = counter.to_bytes(4, "big")
        output.extend(hmac_sm3(key, nonce + counter_bytes))
        counter += 1
    return bytes(output[:length])


def _normalize_token_bytes(token) -> bytes:
    if isinstance(token, bytes):
        return token
    if isinstance(token, str):
        try:
            return base64.b64decode(token.encode())
        except Exception:
            return token.encode()
    return bytes(token)


def _encrypt_with_password(plaintext: bytes, password: str) -> bytes:
    key = _derive_wallet_key(password)
    nonce = os.urandom(16)
    cipher_mode = os.getenv("LUNALIB_WALLET_CIPHER", "wl3").lower()
    if cipher_mode == "sm4":
        sm4_key = key[:16]
        use_gpu = os.getenv("LUNALIB_SM4_USE_GPU", "0") == "1"
        ciphertext = SM4Cipher(sm4_key).encrypt_ctr(plaintext, nonce, use_gpu=use_gpu)
        mac = hmac_sm3(key, nonce + ciphertext)
        return b"WL4" + nonce + ciphertext + mac
    stream = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    mac = hmac_sm3(key, nonce + ciphertext)
    return b"WL3" + nonce + ciphertext + mac


def _decrypt_with_password(token, password: str, derived_key: bytes = None) -> bytes:
    token_bytes = _normalize_token_bytes(token)
    if token_bytes.startswith(b"gAAAA"):
        raise ValueError("Legacy Fernet token not supported without cryptography")
    if token_bytes.startswith(b"WL4"):
        nonce = token_bytes[3:19]
        mac = token_bytes[-32:]
        ciphertext = token_bytes[19:-32]
        key = derived_key or _derive_wallet_key(password)
        expected_mac = hmac_sm3(key, nonce + ciphertext)
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("Invalid password or corrupted data")
        sm4_key = key[:16]
        use_gpu = os.getenv("LUNALIB_SM4_USE_GPU", "0") == "1"
        return SM4Cipher(sm4_key).decrypt_ctr(ciphertext, nonce, use_gpu=use_gpu)
    if not token_bytes.startswith(b"WL3"):
        raise ValueError("Unsupported wallet encryption format")
    nonce = token_bytes[3:19]
    mac = token_bytes[-32:]
    ciphertext = token_bytes[19:-32]
    key = derived_key or _derive_wallet_key(password)
    expected_mac = hmac_sm3(key, nonce + ciphertext)
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("Invalid password or corrupted data")
    stream = _keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


class LunaWallet:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.expanduser("~/.lunawallet")
        from lunalib.utils.console import print_info
        print_info(f"[LunaWallet] data_dir: {self.data_dir}")
        self.db = WalletDB(self.data_dir)
        self.wallets = {}  # address -> wallet dict (in-memory cache)
        self.current_wallet_address = None
        self.key_manager = KeyManager()
        self.balance_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.balance_callbacks = []
        self.balance_loading = False
        self.last_balance_update = 0
        self.balance_update_interval = float(os.getenv("LUNALIB_BALANCE_UPDATE_INTERVAL", "5"))
        self._reset_current_wallet()
        self._confirmed_tx_cache: Dict[str, List[Dict]] = {}
        self._pending_tx_cache: Dict[str, List[Dict]] = {}
        self._ui_callbacks: List[Callable] = []
        self._ui_event_callbacks: List[Callable] = []
        self._ui_debounce = float(os.getenv("LUNALIB_UI_DEBOUNCE", "0.25"))
        self._ui_pending_payload: Dict[str, Dict] = {}
        self._ui_pending_events: Dict[str, Dict] = {}
        self._ui_timer: Optional[threading.Timer] = None
        self._ui_handler_registered = False
        self._ui_event_handler_registered = False
        self._session_kdf_cache: Dict[str, bytes] = {}
        self._load_wallets_from_db()

    def _load_wallets_from_db(self):
        """Load all wallets from the database into self.wallets."""
        self.wallets = {}
        for row in self.db.list_wallets():
            addr, label, is_locked, balance, available_balance = row
            w = self.db.load_wallet(addr)
            if w:
                self.wallets[addr] = w
        # Set current_wallet_address to the first wallet if any
        if self.wallets and not self.current_wallet_address:
            self.current_wallet_address = next(iter(self.wallets))

    def lock_wallet(self, address=None):
        """Lock the wallet (removes private key from memory and updates db)"""
        addr = address or self.current_wallet_address
        if not addr or addr not in self.wallets:
            return {"success": False, "error": "Wallet not found"}
        self.wallets[addr]["is_locked"] = True
        self.wallets[addr]["private_key"] = None
        self._session_kdf_cache.pop(addr, None)
        if addr == self.current_wallet_address:
            self.private_key = None
            self.is_locked = True
        # Update db
        w = self.wallets[addr]
        self.db.save_wallet(
            addr, w.get("label", ""), w.get("public_key", ""), w.get("encrypted_private_key", b""), True, w.get("created", time.time()), w.get("balance", 0.0), w.get("available_balance", 0.0)
        )
        return {"success": True}

    def unlock_wallet(self, address, password):
        """Unlock wallet with password using SM2. Returns dict with success/error."""
        if address not in self.wallets:
            return {"success": False, "error": f"Wallet {address} not found"}
        wallet_data = self.wallets[address]
        if not wallet_data.get("is_locked") and wallet_data.get("private_key"):
            return {"success": True}
        try:
            if wallet_data.get("encrypted_private_key"):
                cache_enabled = os.getenv("LUNALIB_WALLET_SESSION_CACHE", "0") == "1"
                derived_key = self._session_kdf_cache.get(address) if cache_enabled else None
                if cache_enabled and derived_key is None:
                    derived_key = _derive_wallet_key(password)
                decrypted_key = _decrypt_with_password(
                    wallet_data["encrypted_private_key"], password, derived_key=derived_key
                )
                wallet_data["private_key"] = decrypted_key.decode()
                wallet_data["is_locked"] = False
                if cache_enabled and derived_key is not None:
                    self._session_kdf_cache[address] = derived_key
                if self.current_wallet_address == address:
                    self.private_key = wallet_data["private_key"]
                    self.is_locked = False
                # Update db
                self.db.save_wallet(
                    address, wallet_data.get("label", ""), wallet_data.get("public_key", ""), wallet_data.get("encrypted_private_key", b""), False, wallet_data.get("created", time.time()), wallet_data.get("balance", 0.0), wallet_data.get("available_balance", 0.0)
                )
                if self._verify_wallet_integrity():
                    return {"success": True}
                else:
                    return {"success": False, "error": "Cryptographic verification failed"}
        except Exception as e:
            return {"success": False, "error": f"Unlock failed: {e}"}
        return {"success": False, "error": "Unlock failed"}

    def get_wallet_index(self):
        """Return a list of all wallets with address, label, is_locked, balance, available_balance."""
        self._load_wallets_from_db()
        index = []
        for addr, data in self.wallets.items():
            index.append({
                "address": addr,
                "label": data.get("label", ""),
                "is_locked": data.get("is_locked", True),
                "balance": data.get("balance", 0.0),
                "available_balance": data.get("available_balance", 0.0),
                "balance_display": format_amount(data.get("balance", 0.0)),
                "available_balance_display": format_amount(data.get("available_balance", 0.0)),
            })
        return index

    def get_wallet_info(self, address=None):
        """Get info for a single wallet (address, label, balances, is_locked, created, public_key)."""
        addr = address or self.current_wallet_address
        if not addr or addr not in self.wallets:
            return {"success": False, "error": "Wallet not found"}
        data = self.wallets[addr]
        return {
            "address": addr,
            "label": data.get("label", ""),
            "is_locked": data.get("is_locked", True),
            "balance": data.get("balance", 0.0),
            "available_balance": data.get("available_balance", 0.0),
            "balance_display": format_amount(data.get("balance", 0.0)),
            "available_balance_display": format_amount(data.get("available_balance", 0.0)),
            "created": data.get("created", 0),
            "public_key": data.get("public_key", ""),
        }

    def get_wallet_balances(self, address=None):
        """最新のavailable/pending残高をWalletStateManagerから取得して返す。"""
        addr = address or self.current_wallet_address
        if not addr:
            return {"success": False, "error": "Wallet not found"}
        try:
            from .wallet_manager import get_wallet_manager
            state_manager = get_wallet_manager()
            # 最新の残高情報を取得
            balance_data = state_manager.get_balance(addr)
            if not balance_data:
                return {"success": False, "error": "No balance data found"}
            return {
                "address": addr,
                "balance": balance_data.get("confirmed_balance", 0.0),
                "available_balance": balance_data.get("available_balance", 0.0),
                "pending_in": balance_data.get("pending_incoming", 0.0),
                "pending_out": balance_data.get("pending_outgoing", 0.0),
                "total_balance": balance_data.get("total_balance", 0.0),
                "balance_display": format_amount(balance_data.get("confirmed_balance", 0.0)),
                "available_balance_display": format_amount(balance_data.get("available_balance", 0.0)),
                "pending_in_display": format_amount(balance_data.get("pending_incoming", 0.0)),
                "pending_out_display": format_amount(balance_data.get("pending_outgoing", 0.0)),
                "total_balance_display": format_amount(balance_data.get("total_balance", 0.0)),
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get balance: {e}"}

    def get_wallet_transaction_history(self, address=None):
        """Get confirmed and pending transactions for a wallet (includes rewards)."""
        addr = address or self.current_wallet_address
        if not addr:
            return {"success": False, "error": "No wallet selected"}

        norm_addr = self._normalize_address(addr)

        confirmed = self._confirmed_tx_cache.get(norm_addr, [])
        pending = self._pending_tx_cache.get(norm_addr, [])

        if not confirmed:
            from lunalib.core.blockchain import BlockchainManager

            blockchain = BlockchainManager()
            cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
            max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "0"))
            confirmed = blockchain.scan_transactions_for_address_filtered(
                addr,
                include_rewards=True,
                include_transfers=True,
                include_gtx_genesis=False,
                cache_only=cache_only,
                max_range=max_range if max_range > 0 else None,
            )
            self._confirmed_tx_cache[norm_addr] = confirmed

        if not pending:
            from lunalib.core.mempool import MempoolManager

            mempool = MempoolManager()
            pending = mempool.get_pending_transactions(addr, fetch_remote=True)
            self._pending_tx_cache[norm_addr] = pending

        reward_txs = [
            tx
            for tx in confirmed
                if tx.get("type", "").lower() in ["reward", "mining", "gtx_genesis"]
                or str(tx.get("from") or "").lower() in {"ling country", "ling country mines", "foreign exchange", "network", "block_reward", "mining_reward", "coinbase"}
        ]

        return {
            "address": addr,
            "confirmed": confirmed,
            "pending": pending,
            "reward_transactions": reward_txs,
            "total_confirmed": len(confirmed),
            "total_pending": len(pending),
            "total_rewards": len(reward_txs),
        }

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
        self.balance_update_interval = float(os.getenv("LUNALIB_BALANCE_UPDATE_INTERVAL", "5"))
        self._reset_current_wallet()
        self._confirmed_tx_cache: Dict[str, List[Dict]] = {}
        self._pending_tx_cache: Dict[str, List[Dict]] = {}
        self._session_kdf_cache: Dict[str, bytes] = {}

    # ----------------------
    # Address normalization
    # ----------------------
    def _normalize_address(self, addr: str) -> str:
        if not addr:
            return ""
        addr_str = str(addr).strip("'\" ").lower()
        return addr_str[4:] if addr_str.startswith("lun_") else addr_str

    def _is_zero_amount_transfer(self, transaction: Dict) -> bool:
        tx_type = (transaction.get("type") or "").lower()
        if tx_type not in ("transfer", "transaction"):
            return False
        amount = transaction.get(
            "amount",
            transaction.get(
                "transfer_amount",
                transaction.get("value", transaction.get("denomination", transaction.get("quantity"))),
            ),
        )
        if amount is None:
            return False
        try:
            return float(amount) <= 0
        except (TypeError, ValueError):
            return False

    def _filter_zero_amount_transfers(self, transactions: List[Dict]) -> List[Dict]:
        if not transactions:
            return transactions
        return [tx for tx in transactions if not self._is_zero_amount_transfer(tx)]

    def _reset_current_wallet(self):
        """Reset current wallet to empty state"""
        self.address = None
        self.balance = 0.0  # Total balance (confirmed transactions)
        self.available_balance = 0.0  # Available balance (total - pending outgoing)
        self.created = time.time()
        self.private_key = None  # Will store REAL SM2 private key
        self.public_key = None  # Will store REAL SM2 public key
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
            print(
                f"DEBUG: Transaction signed successfully, signature: {signature[:16]}..."
            )
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

        if not self.public_key.startswith("04") or len(self.public_key) != 130:
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
        encrypted_private_key = _encrypt_with_password(private_key.encode(), password)

        # Create wallet data
        wallet_data = {
            "address": address,
            "balance": 0.0,
            "available_balance": 0.0,
            "created": time.time(),
            "private_key": private_key,  # REAL SM2 private key
            "public_key": public_key,  # REAL SM2 public key
            "encrypted_private_key": encrypted_private_key,
            "label": name,
            "is_locked": True,
            "crypto_standard": "SM2_GB/T_32918",
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
        encrypted_private_key = _encrypt_with_password(private_key_hex.encode(), password)

        # Create wallet data
        new_wallet_data = {
            "address": address,
            "balance": 0.0,
            "available_balance": 0.0,
            "created": time.time(),
            "private_key": private_key_hex,
            "public_key": public_key_hex,
            "encrypted_private_key": encrypted_private_key,
            "label": name,
            "is_locked": True,
            "crypto_standard": "SM2_GB/T_32918",
        }

        # Add to wallets collection
        self.wallets[address] = new_wallet_data

        print(f"DEBUG: Created wallet {address}, total wallets: {len(self.wallets)}")

        return new_wallet_data

    def _set_current_wallet(self, wallet_data):
        """Set the current wallet from wallet data"""
        self.current_wallet_address = wallet_data["address"]
        self.address = wallet_data["address"]
        self.balance = wallet_data["balance"]
        self.available_balance = wallet_data["available_balance"]
        self.created = wallet_data["created"]
        self.private_key = wallet_data["private_key"]
        self.public_key = wallet_data["public_key"]
        self.encrypted_private_key = wallet_data["encrypted_private_key"]
        self.label = wallet_data["label"]
        self.is_locked = wallet_data.get("is_locked", True)

        print(f"DEBUG: Set current wallet to {self.address}")

    def unlock_wallet(self, address, password):
        """Unlock wallet with password"""
        if address not in self.wallets:
            print(f"DEBUG: Wallet {address} not found in collection")
            return False

        wallet_data = self.wallets[address]
        if not wallet_data.get("is_locked") and wallet_data.get("private_key"):
            return True

        try:
            if wallet_data.get("encrypted_private_key"):
                cache_enabled = os.getenv("LUNALIB_WALLET_SESSION_CACHE", "0") == "1"
                derived_key = self._session_kdf_cache.get(address) if cache_enabled else None
                if cache_enabled and derived_key is None:
                    derived_key = _derive_wallet_key(password)
                # Decrypt private key
                decrypted_key = _decrypt_with_password(
                    wallet_data["encrypted_private_key"], password, derived_key=derived_key
                )

                # Update wallet data
                wallet_data["private_key"] = decrypted_key.decode()
                wallet_data["is_locked"] = False
                if cache_enabled and derived_key is not None:
                    self._session_kdf_cache[address] = derived_key

                # If this is the current wallet, update current state
                if self.current_wallet_address == address:
                    self.private_key = wallet_data["private_key"]
                    self.is_locked = False

                print(f"DEBUG: Wallet {address} unlocked successfully")
                print(f"DEBUG: Private key available: {bool(self.private_key)}")

                # Verify cryptographic integrity after unlock
                if self._verify_wallet_integrity():
                    print(f"DEBUG: Wallet cryptographic integrity verified")
                else:
                    print(
                        f"DEBUG: WARNING: Wallet unlocked but cryptographic verification failed"
                    )

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
            from lunalib.core.blockchain import BlockchainManager
            blockchain = BlockchainManager()
            current_height = blockchain.get_blockchain_height()

            addr = self.current_wallet_address or self.address
            if not addr:
                return self.balance
            norm_addr = self._normalize_address(addr)

            # Get confirmed transactions for this wallet
            confirmed = self._confirmed_tx_cache.get(norm_addr, [])
            if not confirmed and addr in self._confirmed_tx_cache:
                confirmed = self._confirmed_tx_cache.get(addr, [])
            total_balance = self._compute_confirmed_balance(confirmed, current_height)

            # Get pending transactions for this wallet
            pending = self._pending_tx_cache.get(norm_addr, [])
            if not pending and addr in self._pending_tx_cache:
                pending = self._pending_tx_cache.get(addr, [])
            pending_out, pending_in = self._compute_pending_totals(pending, addr, current_height)

            available_balance = max(0.0, total_balance + pending_in - pending_out)

            # Update both current wallet and wallets collection
            self.available_balance = available_balance
            self.balance = total_balance  # Also update total balance

            if self.current_wallet_address in self.wallets:
                self.wallets[self.current_wallet_address][
                    "available_balance"
                ] = available_balance
                self.wallets[self.current_wallet_address]["balance"] = total_balance

            print(
                f"DEBUG: Balance calculated - Total: {total_balance}, Pending Out: {pending_out}, Pending In: {pending_in}, Available: {available_balance}"
            )

            if pending_in > 0:
                print(f"DEBUG: Also {pending_in} LUN incoming (pending, including rewards)")

            return available_balance

        except Exception as e:
            print(f"DEBUG: Error calculating available balance: {e}")
            return self.balance  # Fallback to total balance

    def _compute_confirmed_balance(self, transactions: List[Dict]) -> float:
        """Compute confirmed balance from a list of transactions, using confirmation count for rewards."""
        total_balance = 0.0
        current_height = None
        import inspect
        required_conf = int(os.getenv("LUNALIB_REWARD_CONFIRMATIONS", "1"))
        # Try to get current_height from caller if passed
        frame = inspect.currentframe().f_back
        if frame and 'current_height' in frame.f_locals:
            current_height = frame.f_locals['current_height']
        for tx in transactions:
            tx_type = (tx.get("type") or "").lower()
            direction = tx.get("direction", "")
            amount = float(tx.get("amount", 0) or 0)
            block_height = tx.get("block_height")
            confirmations = None
            if block_height is not None and current_height is not None:
                try:
                    confirmations = int(current_height) - int(block_height) + 1
                except Exception:
                    confirmations = None

            if tx_type == "reward" or str(tx.get("from") or "").lower() in {"ling country", "ling country mines", "foreign exchange", "network", "block_reward", "mining_reward", "coinbase"}:
                print(f"[DEBUG] reward tx: {tx}, confirmations={confirmations}")
                # block_heightやcurrent_heightがNoneならとりあえず加算
                if confirmations is None or confirmations >= required_conf:
                    total_balance += amount
            elif direction == "incoming":
                total_balance += amount
            elif direction == "outgoing":
                fee = float(tx.get("fee", 0) or 0)
                total_balance -= amount
                total_balance -= fee

        print(f"[DEBUG] computed confirmed balance: {total_balance}")
        return max(0.0, total_balance)

    def _compute_pending_totals(
        self, pending_txs: List[Dict], address: str, current_height: int = None
    ) -> Tuple[float, float]:
        """Return (pending_outgoing, pending_incoming) for an address, including pending rewards (<6 confs)."""
        pending_out = 0.0
        pending_in = 0.0

        required_conf = int(os.getenv("LUNALIB_REWARD_CONFIRMATIONS", "1"))

        target_norm = self._normalize_address(address)
        for tx in pending_txs:
            from_norm = self._normalize_address(tx.get("from") or tx.get("sender"))
            to_norm = self._normalize_address(tx.get("to") or tx.get("receiver"))
            tx_type = (tx.get("type") or "").lower()
            amount = float(tx.get("amount", 0) or 0)
            block_height = tx.get("block_height")
            confirmations = None
            if block_height is not None and current_height is not None:
                try:
                    confirmations = int(current_height) - int(block_height) + 1
                except Exception:
                    confirmations = None

            if from_norm == target_norm:
                fee = float(tx.get("fee", 0) or tx.get("gas", 0) or 0)
                pending_out += amount + fee
            if to_norm == target_norm:
                # For rewards, only count as pending if confirmations < 6
                if tx_type == "reward" or str(tx.get("from") or "").lower() in {"ling country", "ling country mines", "foreign exchange", "network", "block_reward", "mining_reward", "coinbase"}:
                    if confirmations is None or confirmations < required_conf:
                        pending_in += amount
                else:
                    pending_in += amount

        return pending_out, pending_in

    def _recompute_balances_from_cache(self) -> Dict[str, Dict[str, float]]:
        """Recompute balances for all wallets from cached tx sets."""
        updated: Dict[str, Dict[str, float]] = {}

        for addr, wallet_data in self.wallets.items():
            norm = self._normalize_address(addr)
            confirmed = self._confirmed_tx_cache.get(norm, []) if norm else self._confirmed_tx_cache.get(addr, [])
            pending = self._pending_tx_cache.get(norm, []) if norm else self._pending_tx_cache.get(addr, [])

            total_balance = max(0.0, self._compute_confirmed_balance(confirmed))
            pending_out, pending_in = self._compute_pending_totals(pending, addr)
            available_balance = max(0.0, total_balance + pending_in - pending_out)

            wallet_data["balance"] = total_balance
            wallet_data["available_balance"] = available_balance

            if addr == self.current_wallet_address:
                self.balance = total_balance
                self.available_balance = available_balance

            updated[addr] = {
                "balance": total_balance,
                "available_balance": available_balance,
                "pending_out": pending_out,
                "pending_in": pending_in,
            }

        return updated

    def refresh_all_wallet_balances(self) -> Dict[str, Dict[str, float]]:
        """Scan blockchain and mempool for all wallets, update balances and pending info."""
        addresses = list(self.wallets.keys())
        if not addresses:
            return {}

        try:
            from lunalib.core.blockchain import BlockchainManager
            from lunalib.core.mempool import MempoolManager

            blockchain = BlockchainManager()
            mempool = MempoolManager()
            cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
            max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "0"))

            confirmed_map = blockchain.scan_transactions_for_addresses_filtered(
                addresses,
                include_rewards=True,
                include_transfers=True,
                include_gtx_genesis=False,
                cache_only=cache_only,
                max_range=max_range if max_range > 0 else None,
            )
            pending_map = mempool.get_pending_transactions_for_addresses(addresses, fetch_remote=True)

            # Reset caches with normalized + original keys
            normalized_confirmed: Dict[str, List[Dict]] = {}
            for addr, txs in confirmed_map.items():
                normalized_confirmed[addr] = txs
                norm = self._normalize_address(addr)
                if norm:
                    normalized_confirmed[norm] = txs

            normalized_pending: Dict[str, List[Dict]] = {}
            for addr, txs in pending_map.items():
                filtered = self._filter_zero_amount_transfers(txs)
                normalized_pending[addr] = filtered
                norm = self._normalize_address(addr)
                if norm:
                    normalized_pending[norm] = filtered

            self._confirmed_tx_cache = normalized_confirmed
            self._pending_tx_cache = normalized_pending

            # Recompute balances and pending for all wallets
            updated: Dict[str, Dict[str, float]] = {}
            for addr in addresses:
                norm = self._normalize_address(addr)
                confirmed = normalized_confirmed.get(norm, []) if norm else normalized_confirmed.get(addr, [])
                pending = normalized_pending.get(norm, []) if norm else normalized_pending.get(addr, [])
                # 最新のブロック高を取得
                try:
                    current_height = blockchain.get_blockchain_height()
                except Exception:
                    current_height = None
                total_balance = max(0.0, self._compute_confirmed_balance(confirmed))
                pending_out, pending_in = self._compute_pending_totals(pending, addr, current_height)
                available_balance = max(0.0, total_balance + pending_in - pending_out)

                wallet_data = self.wallets.get(addr, {})
                wallet_data["balance"] = total_balance
                wallet_data["available_balance"] = available_balance
                wallet_data["pending_out"] = pending_out
                wallet_data["pending_in"] = pending_in

                if addr == self.current_wallet_address:
                    self.balance = total_balance
                    self.available_balance = available_balance

                updated[addr] = {
                    "balance": total_balance,
                    "available_balance": available_balance,
                    "pending_out": pending_out,
                    "pending_in": pending_in,
                    "balance_display": format_amount(total_balance),
                    "available_balance_display": format_amount(available_balance),
                    "pending_out_display": format_amount(pending_out),
                    "pending_in_display": format_amount(pending_in),
                }

            return updated

        except Exception as e:
            print(f"DEBUG: Error refreshing all wallet balances: {e}")
            return {}

    def sync_all_wallets_once(self) -> Dict[str, Dict[str, float]]:
        """Convenience: scan blockchain once and mempool once, update all wallet balances."""
        return self.refresh_all_wallet_balances()

    def get_all_wallets_overview(
        self, include_transactions: bool = True
    ) -> Dict[str, Dict]:
        """Return balances and (optionally) cached transactions for all wallets."""
        overview: Dict[str, Dict] = {}
        for addr, wallet_data in self.wallets.items():
            norm = self._normalize_address(addr)
            confirmed = self._confirmed_tx_cache.get(norm, []) if norm else self._confirmed_tx_cache.get(addr, [])
            pending = self._pending_tx_cache.get(norm, []) if norm else self._pending_tx_cache.get(addr, [])

            # pending_in/pending_outはwallet_dataに格納済み
            overview[addr] = {
                "balance": wallet_data.get("balance", 0.0),
                "available_balance": wallet_data.get("available_balance", 0.0),
                "pending_out": wallet_data.get("pending_out", 0.0),
                "pending_in": wallet_data.get("pending_in", 0.0),
                "balance_display": format_amount(wallet_data.get("balance", 0.0)),
                "available_balance_display": format_amount(wallet_data.get("available_balance", 0.0)),
                "pending_out_display": format_amount(wallet_data.get("pending_out", 0.0)),
                "pending_in_display": format_amount(wallet_data.get("pending_in", 0.0)),
            }

            if include_transactions:
                overview[addr]["confirmed_transactions"] = confirmed
                overview[addr]["pending_transactions"] = pending

        return overview

    def _apply_transaction_updates(
        self, confirmed_map: Dict[str, List[Dict]], pending_map: Dict[str, List[Dict]]
    ):
        """Update caches from monitor callbacks and recompute balances."""
        for addr, txs in confirmed_map.items():
            self._confirmed_tx_cache.setdefault(addr, []).extend(txs)
            norm = self._normalize_address(addr)
            if norm:
                self._confirmed_tx_cache.setdefault(norm, []).extend(txs)

        for addr, txs in pending_map.items():
            filtered = self._filter_zero_amount_transfers(txs)
            self._pending_tx_cache[addr] = filtered
            norm = self._normalize_address(addr)
            if norm:
                self._pending_tx_cache[norm] = filtered

        return self._recompute_balances_from_cache()

    def start_wallet_monitoring(self, poll_interval: int = None):
        """Start monitoring blockchain and mempool for all wallets."""
        if poll_interval is None:
            poll_interval = int(os.getenv("LUNALIB_WALLET_MONITOR_INTERVAL", "5"))
        addresses = list(self.wallets.keys())
        if not addresses:
            print("DEBUG: No wallets to monitor")
            return None

        from lunalib.core.blockchain import BlockchainManager

        blockchain = BlockchainManager()

        # Run an initial refresh to seed caches and balances
        self.refresh_all_wallet_balances()

        def _on_update(payload: Dict):
            confirmed_map = payload.get("confirmed", {}) or {}
            pending_map = payload.get("pending", {}) or {}
            self._apply_transaction_updates(confirmed_map, pending_map)

        self._monitor_stop_event = blockchain.monitor_addresses(
            addresses, _on_update, poll_interval=poll_interval
        )
        return self._monitor_stop_event

    def stop_wallet_monitoring(self):
        """Stop the background monitor if running."""
        stop_event = getattr(self, "_monitor_stop_event", None)
        if stop_event:
            stop_event.set()

    def start_sync_and_monitor(self, poll_interval: int = 15):
        """Run a one-time sync (blockchain + mempool) then start live monitoring."""
        self.sync_all_wallets_once()
        return self.start_wallet_monitoring(poll_interval=poll_interval)

    def get_all_balances_after_sync(
        self, include_transactions: bool = True
    ) -> Dict[str, Dict]:
        """One-shot sync then return per-wallet balances (and transactions if requested)."""
        self.sync_all_wallets_once()
        return self.get_all_wallets_overview(include_transactions=include_transactions)

    def _get_pending_balance(self) -> float:
        """Get total pending balance from mempool (outgoing pending transactions)"""
        try:
            from lunalib.core.mempool import MempoolManager

            mempool = MempoolManager()

            # Get pending transactions for this address
            pending_txs = mempool.get_pending_transactions(
                self.address, fetch_remote=True
            )

            total_pending_outgoing = 0.0

            for tx in pending_txs:
                # Only count outgoing transactions (where we're the sender)
                if tx.get("from") == self.address:
                    amount = float(tx.get("amount", 0))
                    fee = float(tx.get("fee", 0) or tx.get("gas", 0) or 0)
                    total_pending_outgoing += amount + fee
                    print(
                        f"🔍 Found pending outgoing: {amount} + {fee} fee = {amount + fee}"
                    )

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
            pending_txs = mempool.get_pending_transactions(
                self.address, fetch_remote=True
            )

            total_pending_incoming = 0.0

            for tx in pending_txs:
                # Only count incoming transactions (where we're the receiver)
                if tx.get("to") == self.address:
                    amount = float(tx.get("amount", 0))
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
            cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
            max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "500"))
            transactions = blockchain.scan_transactions_for_address_filtered(
                self.address,
                include_rewards=True,
                include_transfers=True,
                include_gtx_genesis=False,
                cache_only=cache_only,
                max_range=max_range if max_range > 0 else None,
            )
            total_balance = self._compute_confirmed_balance(transactions)
            return total_balance

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
                    self.wallets[self.current_wallet_address]["balance"] = total_balance
                    self.wallets[self.current_wallet_address][
                        "available_balance"
                    ] = available_balance

                # Notify callbacks
                for callback in self.balance_callbacks:
                    try:
                        callback(total_balance, available_balance)
                    except Exception as e:
                        print(f"Error in balance callback: {e}")

                print(
                    f"DEBUG: Async balance updated - Total: {total_balance}, Available: {available_balance}"
                )

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
                self.wallets[self.current_wallet_address]["balance"] = total_balance
                self.wallets[self.current_wallet_address][
                    "available_balance"
                ] = available_balance

            print(
                f"DEBUG: Balance refreshed - Total: {total_balance}, Available: {available_balance}"
            )
            return True

        except Exception as e:
            print(f"DEBUG: Error refreshing balance: {e}")
            return False

    def send_transaction(
        self, to_address: str, amount: float, memo: str = "", password: str = None
    ) -> bool:
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
                print(
                    f"[SM2] Insufficient balance: {self.get_available_balance()} < {amount}"
                )
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
                transaction_type="transfer",
            )

            print(
                f"[SM2] Transaction created: {transaction.get('hash', 'no_hash')[:16]}"
            )

            # 5. Validate
            is_valid, message = tx_manager.validate_transaction(transaction)
            if not is_valid:
                print(f"[SM2] Validation failed: {message}")
                return False

            # 6. Broadcast
            success, message = tx_manager.send_transaction(transaction)
            if success:
                print(f"[SM2] Transaction broadcast: {message}")

                # Add to local pending cache + history immediately
                pending_tx = transaction.copy()
                pending_tx["status"] = "pending"
                pending_tx["direction"] = "outgoing"
                pending_tx["effective_amount"] = -(
                    float(transaction.get("amount", 0)) + float(transaction.get("fee", 0) or 0)
                )
                addr = self.current_wallet_address or self.address
                if addr:
                    if self._is_zero_amount_transfer(pending_tx):
                        print("DEBUG: Skipping zero-amount pending transfer")
                    pending_list = self._pending_tx_cache.get(addr, [])
                    # Avoid duplicates by hash
                    tx_hash = pending_tx.get("hash")
                    if not any(t.get("hash") == tx_hash for t in pending_list):
                        pending_list.append(pending_tx)
                        self._pending_tx_cache[addr] = pending_list

                if hasattr(self, "transactions") and isinstance(self.transactions, list):
                    if not any(t.get("hash") == pending_tx.get("hash") for t in self.transactions):
                        self.transactions.append(pending_tx)

                # Update balance immediately
                total_cost = amount + transaction.get("fee", 0)
                self.available_balance -= total_cost
                if self.current_wallet_address in self.wallets:
                    self.wallets[self.current_wallet_address][
                        "available_balance"
                    ] = self.available_balance

                # Save state
                self.save_wallet_data()

                # Recompute balances using pending cache
                try:
                    self.calculate_available_balance()
                except Exception:
                    pass

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
                "version": "2.0",
                "address": self.address,
                "public_key": self.public_key,
                "private_key": self.private_key,  # WARNING: In production, encrypt this!
                "balance": self.balance,
                "available_balance": self.available_balance,
                "transactions": self.transactions,
                "created_at": self.created_at,
                "last_sync": time.time(),
                "network": self.network,
                "key_type": "SM2",  # Indicate SM2 key type
            }

            # Determine save path
            if hasattr(self, "wallet_file"):
                save_path = self.wallet_file
            else:
                # Default save location
                import os

                from lunalib.storage.database import get_default_wallet_dir
                legacy_dir = os.path.expanduser("~/.lunawallet")
                if os.path.isdir(legacy_dir):
                    wallet_dir = legacy_dir
                else:
                    wallet_dir = os.path.join(get_default_wallet_dir(), "wallets")
                os.makedirs(wallet_dir, exist_ok=True)
                save_path = os.path.join(wallet_dir, f"{self.address}.json")

            # Save to file
            import json

            with open(save_path, "w") as f:
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
            elif hasattr(self, "wallet_file"):
                load_path = self.wallet_file
            else:
                # Try to find wallet file
                import os

                from lunalib.storage.database import get_default_wallet_dir
                legacy_dir = os.path.expanduser("~/.lunawallet")
                if os.path.isdir(legacy_dir):
                    wallet_dir = legacy_dir
                else:
                    wallet_dir = os.path.join(get_default_wallet_dir(), "wallets")
                # Look for any .json file
                if not os.path.isdir(wallet_dir):
                    print("[WALLET] No wallet file found")
                    return False
                wallet_files = [
                    f for f in os.listdir(wallet_dir) if f.endswith(".json")
                ]
                if not wallet_files:
                    print("[WALLET] No wallet file found")
                    return False
                load_path = os.path.join(wallet_dir, wallet_files[0])
                self.wallet_file = load_path

            # Load from file
            import json

            with open(load_path, "r") as f:
                wallet_data = json.load(f)

            # Restore wallet data
            self.address = wallet_data.get("address", "")
            self.public_key = wallet_data.get("public_key", "")
            self.private_key = wallet_data.get("private_key", "")
            self.balance = wallet_data.get("balance", 0.0)
            self.available_balance = wallet_data.get("available_balance", 0.0)
            self.transactions = wallet_data.get("transactions", [])
            self.created_at = wallet_data.get("created_at", time.time())
            self.network = wallet_data.get("network", "mainnet")

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
        """Get complete transaction history (both pending and confirmed).
        Uses cached transactions if available, otherwise scans blockchain/mempool.
        Includes ALL reward transactions (mining rewards and explicit reward txs).
        """
        try:
            # Try to use cache first (from monitoring or sync)
            confirmed_txs = self._confirmed_tx_cache.get(self.address, [])
            pending_txs = self._pending_tx_cache.get(self.address, [])

            # If cache is empty, perform fresh scan
            if not confirmed_txs:
                from lunalib.core.blockchain import BlockchainManager
                from lunalib.core.mempool import MempoolManager

                blockchain = BlockchainManager()
                cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
                max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "0"))
                mempool = MempoolManager()

                # Get confirmed transactions from blockchain (includes mining rewards)
                confirmed_txs = blockchain.scan_transactions_for_address_filtered(
                    self.address,
                    include_rewards=True,
                    include_transfers=True,
                    include_gtx_genesis=False,
                    cache_only=cache_only,
                    max_range=max_range if max_range > 0 else None,
                )
                self._confirmed_tx_cache[self.address] = confirmed_txs

                # Get pending transactions from mempool
                pending_txs = mempool.get_pending_transactions(
                    self.address, fetch_remote=True
                )
                self._pending_tx_cache[self.address] = pending_txs

            # Count by type for debugging
            reward_count = sum(
                1
                for tx in confirmed_txs
                if tx.get("type", "").lower() in ["reward", "mining"]
                or tx.get("from") == "network"
            )

            return {
                "confirmed": confirmed_txs,
                "pending": pending_txs,
                "total_confirmed": len(confirmed_txs),
                "total_pending": len(pending_txs),
                "reward_count": reward_count,
            }
        except Exception as e:
            print(f"DEBUG: Error getting transaction history: {e}")
            return {
                "confirmed": [],
                "pending": [],
                "total_confirmed": 0,
                "total_pending": 0,
                "reward_count": 0,
            }

    def get_wallet_transactions(
        self, address: str = None, include_pending: bool = True
    ) -> Dict[str, List[Dict]]:
        """Get ALL transactions for a wallet including mining rewards, transfers, and pending.

        This is the comprehensive transaction getter that ensures ALL reward transactions
        (mining rewards, explicit reward transactions) are included.

        Args:
            address: wallet address to query (defaults to current wallet address)
            include_pending: whether to include pending mempool transactions (default: True)

        Returns:
            Dict with:
                - 'confirmed': list of all confirmed transactions (transfers + mining rewards)
                - 'pending': list of pending mempool transactions
                - 'reward_transactions': list of only reward/mining transactions
                - 'transfer_transactions': list of only transfer transactions
                - 'total_rewards': count of reward transactions
                - 'total_transfers': count of transfer transactions
        """
        if address is None:
            address = self.address

        if not address:
            return {
                "confirmed": [],
                "pending": [],
                "reward_transactions": [],
                "transfer_transactions": [],
                "incoming_transfers": [],
                "outgoing_transfers": [],
                "total_rewards": 0,
                "total_transfers": 0,
                "total_incoming": 0,
                "total_outgoing": 0,
            }

        try:
            # Normalize address for comparison
            norm_addr = self._normalize_address(address)

            # Use cache if available
            confirmed_txs = self._confirmed_tx_cache.get(norm_addr, [])
            pending_txs = (
                self._pending_tx_cache.get(norm_addr, []) if include_pending else []
            )

            # If cache is empty, perform fresh scan
            if not confirmed_txs:
                from lunalib.core.blockchain import BlockchainManager

                blockchain = BlockchainManager()
                cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
                max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "0"))
                confirmed_txs = blockchain.scan_transactions_for_address_filtered(
                    address,
                    include_rewards=True,
                    include_transfers=True,
                    include_gtx_genesis=False,
                    cache_only=cache_only,
                    max_range=max_range if max_range > 0 else None,
                )
                self._confirmed_tx_cache[norm_addr] = confirmed_txs

            if include_pending and not pending_txs:
                from lunalib.core.mempool import MempoolManager

                mempool = MempoolManager()
                pending_txs = mempool.get_pending_transactions(
                    address, fetch_remote=True
                )
                self._pending_tx_cache[norm_addr] = pending_txs

            # Separate rewards and transfers based on type and source
            # Rewards: explicitly marked as reward/mining type, or from Ling Country
            reward_txs = [
                tx
                for tx in confirmed_txs
                if tx.get("type", "").lower() in ["reward", "mining", "gtx_genesis"]
                or str(tx.get("from") or "").lower() == "ling country"
            ]

            # Transfers: anything that's NOT a reward (includes both incoming and outgoing)
            transfer_txs = [
                tx
                for tx in confirmed_txs
                if tx.get("type", "").lower() not in ["reward", "mining", "gtx_genesis"]
                and str(tx.get("from") or "").lower() != "ling country"
            ]

            # Separate incoming vs outgoing transfers
            incoming_transfers = [
                tx for tx in transfer_txs if tx.get("direction") == "incoming"
            ]
            outgoing_transfers = [
                tx for tx in transfer_txs if tx.get("direction") == "outgoing"
            ]

            result = {
                "confirmed": confirmed_txs,
                "pending": pending_txs,
                "reward_transactions": reward_txs,
                "transfer_transactions": transfer_txs,
                "incoming_transfers": incoming_transfers,
                "outgoing_transfers": outgoing_transfers,
                "total_rewards": len(reward_txs),
                "total_transfers": len(transfer_txs),
                "total_incoming": len(incoming_transfers),
                "total_outgoing": len(outgoing_transfers),
            }

            print(f"DEBUG: get_wallet_transactions({address}):")
            print(f"  - Mining Rewards: {len(reward_txs)}")
            print(f"  - Incoming Transfers: {len(incoming_transfers)}")
            print(f"  - Outgoing Transfers: {len(outgoing_transfers)}")
            print(f"  - Total Transfers: {len(transfer_txs)}")
            print(f"  - Pending: {len(pending_txs)}")
            print(f"  - Total Confirmed: {len(confirmed_txs)}")

            return result

        except Exception as e:
            print(f"DEBUG: Error getting wallet transactions: {e}")
            import traceback

            traceback.print_exc()
            return {
                "confirmed": [],
                "pending": [],
                "reward_transactions": [],
                "transfer_transactions": [],
                "incoming_transfers": [],
                "outgoing_transfers": [],
                "total_rewards": 0,
                "total_transfers": 0,
                "total_incoming": 0,
                "total_outgoing": 0,
            }

    # ============================================================================
    # WALLET INFO AND UTILITIES
    # ============================================================================

    @property
    def is_unlocked(self):
        """Check if current wallet is unlocked"""
        if not self.current_wallet_address:
            return False
        wallet_data = self.wallets.get(self.current_wallet_address, {})
        return not wallet_data.get("is_locked", True)

    def export_private_key(self, address, password):
        """Export private key with password decryption"""
        if address not in self.wallets:
            return None

        wallet_data = self.wallets[address]

        try:
            if wallet_data.get("encrypted_private_key"):
                cache_enabled = os.getenv("LUNALIB_WALLET_SESSION_CACHE", "0") == "1"
                derived_key = self._session_kdf_cache.get(address) if cache_enabled else None
                if cache_enabled and derived_key is None:
                    derived_key = _derive_wallet_key(password)
                decrypted_key = _decrypt_with_password(
                    wallet_data["encrypted_private_key"], password, derived_key=derived_key
                )
                if cache_enabled and derived_key is not None:
                    self._session_kdf_cache[address] = derived_key
                return decrypted_key.decode()
        except:
            pass
        return None

    def import_wallet(self, wallet_data, password=None):
        """Import wallet from data"""
        if isinstance(wallet_data, dict):
            from lunalib.utils.validation import validate_wallet_import

            ok, message = validate_wallet_import(wallet_data)
            if not ok:
                print(f"DEBUG: Wallet import validation failed: {message}")
                return False

            address = wallet_data.get("address")
            if not address:
                return False

            # Check if wallet uses SM2 cryptography
            if wallet_data.get("crypto_standard") != "SM2_GB/T_32918":
                print(
                    f"DEBUG: WARNING: Importing wallet without SM2 cryptography standard"
                )

            # Add to wallets collection
            self.wallets[address] = wallet_data.copy()

            # Set as current wallet
            self._set_current_wallet(wallet_data)

            if password and wallet_data.get("encrypted_private_key"):
                return self.unlock_wallet(address, password)

            return True
        return False

    def update_balance(self, new_balance):
        """Update current wallet balance"""
        self.balance = float(new_balance)
        self.available_balance = float(new_balance)

        if self.current_wallet_address and self.current_wallet_address in self.wallets:
            self.wallets[self.current_wallet_address]["balance"] = self.balance
            self.wallets[self.current_wallet_address][
                "available_balance"
            ] = self.available_balance

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
            "address": self.address,
            "balance": self.balance,
            "available_balance": self.available_balance,
            "created": self.created,
            "private_key": self.private_key,
            "public_key": self.public_key,
            "encrypted_private_key": self.encrypted_private_key,
            "label": self.label,
            "is_locked": self.is_locked,
            "crypto_standard": "SM2_GB/T_32918",
        }

    # ============================================================================
    # UNIFIED WALLET STATE MANAGER INTEGRATION
    # ============================================================================

    def sync_with_state_manager(self, blockchain=None, mempool=None) -> Dict:
        """
        Sync all registered wallets with the unified WalletStateManager.
        Scans blockchain once and merges mempool data for all wallets.

        Parameters:
            blockchain: BlockchainManager instance (required)
            mempool: MempoolManager instance (required)

        Returns: Dictionary of wallet summaries with balances and transactions
        """
        try:
            if not blockchain or not mempool:
                print("❌ blockchain and mempool instances required")
                return {}

            from .wallet_manager import get_wallet_manager

            state_manager = get_wallet_manager()

            # Register all wallets with state manager if not already done
            addresses = list(self.wallets.keys())
            state_manager.register_wallets(addresses)

            print(f"🔄 Syncing {len(addresses)} wallets...")

            # Get data from blockchain and mempool (single scan)
            end_height = blockchain.get_blockchain_height()
            lookback = int(os.getenv("LUNALIB_WALLET_SYNC_LOOKBACK", "50"))
            if lookback < 0:
                lookback = 0
            cache_only = os.getenv("LUNALIB_WALLET_SCAN_CACHE_ONLY", "0") == "1"
            max_range = int(os.getenv("LUNALIB_WALLET_SCAN_MAX_RANGE", "0"))
            if end_height <= state_manager.last_blockchain_height:
                if lookback > 0:
                    start_height = max(0, end_height - lookback + 1)
                    blockchain_txs = blockchain.scan_transactions_for_addresses_filtered(
                        addresses,
                        start_height=start_height,
                        end_height=end_height,
                        include_rewards=True,
                        include_transfers=True,
                        include_gtx_genesis=False,
                        cache_only=cache_only,
                        max_range=max_range if max_range > 0 else None,
                    )
                else:
                    blockchain_txs = {}
            else:
                start_height = max(0, state_manager.last_blockchain_height + 1)
                if lookback > 0:
                    start_height = min(start_height, max(0, end_height - lookback + 1))
                blockchain_txs = blockchain.scan_transactions_for_addresses_filtered(
                    addresses,
                    start_height=start_height,
                    end_height=end_height,
                    include_rewards=True,
                    include_transfers=True,
                    include_gtx_genesis=False,
                    cache_only=cache_only,
                    max_range=max_range if max_range > 0 else None,
                )
            mempool_txs = mempool.get_pending_transactions_for_addresses(addresses)

            # Sync state manager with the data
            state_manager.sync_wallets_from_sources(blockchain_txs, mempool_txs)
            state_manager.last_blockchain_height = end_height

            # Update LunaWallet balances from state manager
            balances = state_manager.get_all_balances()
            for address, balance_data in balances.items():
                if address in self.wallets:
                    self.wallets[address]["balance"] = balance_data["confirmed_balance"]
                    self.wallets[address]["available_balance"] = balance_data[
                        "available_balance"
                    ]

            # Update current wallet
            if self.current_wallet_address and self.current_wallet_address in balances:
                balance_data = balances[self.current_wallet_address]
                self.balance = balance_data["confirmed_balance"]
                self.available_balance = balance_data["available_balance"]

            # Return summaries
            summaries = state_manager.get_all_summaries()
            print(f"✅ Sync complete - {len(summaries)} wallets updated")

            return summaries

        except Exception as e:
            print(f"❌ Sync error: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def get_wallet_details(self, address: str = None) -> Optional[Dict]:
        """
        Get detailed information for a wallet including balance and transaction summary.
        If address is None, uses current wallet.
        """
        if address is None:
            address = self.current_wallet_address

        if not address:
            return None

        try:
            from .wallet_manager import get_wallet_manager

            state_manager = get_wallet_manager()

            summary = state_manager.get_wallet_summary(address)
            if summary:
                return summary

            # Fallback to basic wallet info
            if address in self.wallets:
                wallet_data = self.wallets[address]
                return {
                    "address": address,
                    "label": wallet_data.get("label", "Wallet"),
                    "balance": wallet_data.get("balance", 0.0),
                    "available_balance": wallet_data.get("available_balance", 0.0),
                    "is_locked": wallet_data.get("is_locked", True),
                }

            return None

        except Exception as e:
            print(f"⚠️  Error getting wallet details: {e}")
            return None

    def get_wallet_transactions(
        self, address: str = None, tx_type: str = "all"
    ) -> List[Dict]:
        """
        Get transactions for a wallet from the state manager.

        tx_type: 'all', 'confirmed', 'pending', 'transfers', 'rewards', 'genesis'
        """
        if address is None:
            address = self.current_wallet_address

        if not address:
            return []

        try:
            from .wallet_manager import get_wallet_manager

            state_manager = get_wallet_manager()

            return state_manager.get_transactions(address, tx_type)

        except Exception as e:
            print(f"⚠️  Error getting transactions: {e}")
            return []

    def register_wallet_ui_callback(self, callback: Callable) -> None:
        """
        Register a callback to receive real-time wallet balance updates.
        Callback will be called with: callback(balance_data_dict)
        """
        try:
            from .wallet_manager import get_wallet_manager

            state_manager = get_wallet_manager()
            self._ui_callbacks.append(callback)

            if not self._ui_handler_registered:
                def _ui_handler(balance_data: Dict):
                    self._queue_ui_update(balance_data)

                state_manager.on_balance_update(_ui_handler)
                self._ui_handler_registered = True
        except Exception as e:
            print(f"⚠️  Error registering callback: {e}")

    def register_wallet_event_callback(self, callback: Callable) -> None:
        """
        Register a callback to receive change-only transaction events.
        Callback will be called with: callback(event_data_dict)
        """
        try:
            from .wallet_manager import get_wallet_manager

            state_manager = get_wallet_manager()
            self._ui_event_callbacks.append(callback)

            if not self._ui_event_handler_registered:
                def _event_handler(event_data: Dict):
                    self._queue_ui_event_update(event_data)

                state_manager.on_event_update(_event_handler)
                self._ui_event_handler_registered = True
        except Exception as e:
            print(f"⚠️  Error registering event callback: {e}")

    def _queue_ui_update(self, balance_data: Dict) -> None:
        """Coalesce balance updates for UI callbacks."""
        if not balance_data:
            return

        if self._ui_pending_payload:
            self._ui_pending_payload.update(balance_data)
        else:
            self._ui_pending_payload = dict(balance_data)

        if self._ui_timer and self._ui_timer.is_alive():
            return

        self._ui_timer = threading.Timer(self._ui_debounce, self._flush_ui_update)
        self._ui_timer.daemon = True
        self._ui_timer.start()

    def _queue_ui_event_update(self, event_data: Dict) -> None:
        """Coalesce transaction event updates for UI callbacks."""
        if not event_data:
            return

        if self._ui_pending_events:
            for addr, payload in event_data.items():
                if addr in self._ui_pending_events:
                    existing = self._ui_pending_events[addr]
                    existing.setdefault('new_confirmed', []).extend(payload.get('new_confirmed', []))
                    existing.setdefault('new_pending', []).extend(payload.get('new_pending', []))
                    existing.setdefault('promoted', []).extend(payload.get('promoted', []))
                    existing.setdefault('pending_cleared', []).extend(payload.get('pending_cleared', []))
                else:
                    self._ui_pending_events[addr] = payload
        else:
            self._ui_pending_events = dict(event_data)

        if self._ui_timer and self._ui_timer.is_alive():
            return

        self._ui_timer = threading.Timer(self._ui_debounce, self._flush_ui_update)
        self._ui_timer.daemon = True
        self._ui_timer.start()

    def _flush_ui_update(self) -> None:
        payload = self._ui_pending_payload
        events = self._ui_pending_events
        self._ui_pending_payload = {}
        self._ui_pending_events = {}

        for callback in list(self._ui_callbacks):
            try:
                callback(payload)
            except Exception as e:
                print(f"⚠️  Error in UI callback: {e}")

        for callback in list(self._ui_event_callbacks):
            try:
                callback(events)
            except Exception as e:
                print(f"⚠️  Error in UI event callback: {e}")

    def start_continuous_sync(
        self, blockchain=None, mempool=None, poll_interval: int = 30
    ) -> None:
        """
        Start continuous synchronization in background thread.
        Syncs every poll_interval seconds.
        """
        if not blockchain or not mempool:
            print("❌ blockchain and mempool instances required")
            return

        try:
            from .wallet_manager import get_wallet_manager
            from .wallet_sync_helper import WalletSyncHelper

            state_manager = get_wallet_manager()

            # Register wallets
            addresses = list(self.wallets.keys())
            state_manager.register_wallets(addresses)

            def sync_callback(balance_data):
                """Update LunaWallet when state manager updates"""
                for address, balance_info in balance_data.items():
                    if address in self.wallets:
                        self.wallets[address]["balance"] = balance_info[
                            "confirmed_balance"
                        ]
                        self.wallets[address]["available_balance"] = balance_info[
                            "available_balance"
                        ]

                    if address == self.current_wallet_address:
                        self.balance = balance_info["confirmed_balance"]
                        self.available_balance = balance_info["available_balance"]

            # Create sync helper
            sync_helper = WalletSyncHelper(self, blockchain, mempool)

            # Start continuous sync with callback
            sync_helper.start_continuous_sync(
                poll_interval, on_balance_update=sync_callback
            )

            print(f"🔄 Started continuous sync (interval: {poll_interval}s)")

        except Exception as e:
            print(f"❌ Error starting continuous sync: {e}")

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
                    encrypted_key_data = base64.b64encode(
                        self.encrypted_private_key
                    ).decode("utf-8")
                else:
                    encrypted_key_data = base64.b64encode(
                        self.encrypted_private_key.encode()
                    ).decode("utf-8")

            serializable_wallets = {}
            for addr, wallet_info in self.wallets.items():
                serializable_wallet = wallet_info.copy()
                if serializable_wallet.get("encrypted_private_key") and isinstance(
                    serializable_wallet["encrypted_private_key"], bytes
                ):
                    serializable_wallet["encrypted_private_key"] = base64.b64encode(
                        serializable_wallet["encrypted_private_key"]
                    ).decode("utf-8")
                serializable_wallets[addr] = serializable_wallet

            wallet_data = {
                "address": self.address,
                "balance": self.balance,
                "available_balance": self.available_balance,
                "created": self.created,
                "public_key": self.public_key,
                "encrypted_private_key": encrypted_key_data,
                "label": self.label,
                "is_locked": self.is_locked,
                "wallets": serializable_wallets,
                "current_wallet_address": self.current_wallet_address,
                "crypto_standard": "SM2_GB/T_32918",
            }

            with open(filepath, "w") as f:
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
            with open(filepath, "r") as f:
                wallet_data = json.load(f)

            # Check crypto standard
            crypto_standard = wallet_data.get("crypto_standard")
            if crypto_standard != "SM2_GB/T_32918":
                print(
                    f"DEBUG: WARNING: Loading wallet with different crypto standard: {crypto_standard}"
                )

            # Load wallets collection
            self.wallets = wallet_data.get("wallets", {})

            # Load current wallet address
            self.current_wallet_address = wallet_data.get("current_wallet_address")

            # If we have a current wallet, load its data
            if (
                self.current_wallet_address
                and self.current_wallet_address in self.wallets
            ):
                current_wallet_data = self.wallets[self.current_wallet_address]
                self._set_current_wallet(current_wallet_data)

                encrypted_key = wallet_data.get("encrypted_private_key")
                if encrypted_key:
                    self.encrypted_private_key = base64.b64decode(
                        encrypted_key.encode()
                    )
                    if self.current_wallet_address in self.wallets:
                        self.wallets[self.current_wallet_address][
                            "encrypted_private_key"
                        ] = self.encrypted_private_key

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
        print("\n" + "=" * 60)
        print("SM2 CRYPTOGRAPHY DEBUG INFO")
        print("=" * 60)

        if self.current_wallet_address:
            print(f"Current Address: {self.address}")

            # Check address format
            if self.address.startswith("LUN_"):
                print("✅ Address has correct LUN_ prefix")
            else:
                print("❌ Address missing LUN_ prefix")

            # Check private key
            if self.private_key:
                print(f"Private Key: {self.private_key[:16]}...")
                if len(self.private_key) == 64:
                    print(f"✅ Private key is 256-bit (64 hex chars)")
                else:
                    print(
                        f"❌ Private key has wrong length: {len(self.private_key)} chars"
                    )
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
            if wallet_data.get("crypto_standard") == "SM2_GB/T_32918":
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

        print("=" * 60)
