import json
import re
from typing import Optional, Tuple, Dict, Any, List

_ADDRESS_RE = re.compile(r"^LUN_[A-Za-z0-9_]{1,64}$", re.IGNORECASE)
_HEX_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_GTX_SERIAL_RE = re.compile(r"^GTX-[0-9]+(?:\.[0-9]+)?-[0-9]{8}-[A-Z2-7]{10,32}$")

MAX_TEXT_LEN = 256
MAX_LABEL_LEN = 128
MAX_MEMO_LEN = 256
MAX_WALLET_IMPORT_BYTES = 256 * 1024
MAX_TX_BYTES = 64 * 1024
MAX_BLOCK_BYTES = 512 * 1024


def is_safe_text(value: Optional[str], max_len: int = MAX_TEXT_LEN) -> bool:
    if value is None:
        return False
    text = str(value)
    if len(text) == 0 or len(text) > max_len:
        return False
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127:
            return False
    return True


def strip_html(value: Optional[str]) -> str:
    if value is None:
        return ""
    return _HTML_TAG_RE.sub("", str(value))


def sanitize_memo(value: Optional[str], max_len: int = MAX_MEMO_LEN) -> str:
    """Strip HTML, keep BBCode, enforce length and remove control chars."""
    text = strip_html(value)
    if len(text) > max_len:
        text = text[:max_len]
    cleaned = []
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127:
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def is_valid_address(addr: Optional[str]) -> bool:
    if not is_safe_text(addr, max_len=80):
        return False
    return bool(_ADDRESS_RE.fullmatch(str(addr)))


def _is_hex(value: Optional[str], length: int) -> bool:
    if value is None:
        return False
    text = str(value).lower().strip()
    if len(text) != length:
        return False
    return bool(_HEX_RE.fullmatch(text))


def is_valid_private_key(key: Optional[str]) -> bool:
    return _is_hex(key, 64)


def is_valid_public_key(key: Optional[str]) -> bool:
    if not _is_hex(key, 130):
        return False
    return str(key).lower().startswith("04")


def is_valid_tx_hash(value: Optional[str]) -> bool:
    return _is_hex(value, 64)


def is_valid_gtx_serial(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(_GTX_SERIAL_RE.fullmatch(text))


def validate_wallet_import(wallet_data: Any) -> Tuple[bool, str]:
    if not isinstance(wallet_data, dict):
        return False, "Wallet payload must be an object"

    try:
        payload_size = len(json.dumps(wallet_data))
        if payload_size > MAX_WALLET_IMPORT_BYTES:
            return False, "Wallet payload too large"
    except Exception:
        return False, "Wallet payload not serializable"

    address = wallet_data.get("address")
    if not is_valid_address(address):
        return False, "Invalid wallet address"

    label = wallet_data.get("label")
    if label is not None and not is_safe_text(label, max_len=MAX_LABEL_LEN):
        return False, "Invalid wallet label"

    private_key = wallet_data.get("private_key")
    if private_key and not is_valid_private_key(private_key):
        return False, "Invalid private key"

    public_key = wallet_data.get("public_key")
    if public_key and not is_valid_public_key(public_key):
        return False, "Invalid public key"

    return True, "OK"


def validate_transaction_payload(transaction: Any, max_memo_len: int = MAX_MEMO_LEN) -> Tuple[bool, str]:
    if not isinstance(transaction, dict):
        return False, "Transaction payload must be an object"

    try:
        payload_size = len(json.dumps(transaction))
        if payload_size > MAX_TX_BYTES:
            return False, "Transaction payload too large"
    except Exception:
        return False, "Transaction payload not serializable"

    tx_type = str(transaction.get("type") or "").lower()
    if not tx_type:
        return False, "Missing transaction type"

    if tx_type in ("transfer", "transaction"):
        if not is_valid_address(transaction.get("from")):
            return False, "Invalid from address"
        if not is_valid_address(transaction.get("to")):
            return False, "Invalid to address"
        try:
            amount = float(transaction.get("amount", 0))
        except Exception:
            return False, "Invalid amount"
        if amount <= 0:
            return False, "Amount must be positive"

    if "memo" in transaction:
        transaction["memo"] = sanitize_memo(transaction.get("memo"), max_len=max_memo_len)

    if "hash" in transaction and not is_valid_tx_hash(transaction.get("hash")):
        return False, "Invalid transaction hash"

    return True, "OK"


def validate_gtx_genesis_payload(transaction: Any) -> Tuple[bool, str]:
    if not isinstance(transaction, dict):
        return False, "Genesis payload must be an object"
    required = ["denomination", "mining_difficulty", "hash", "nonce"]
    for field in required:
        if field not in transaction:
            return False, f"Missing GTX field: {field}"

    bill_serial = (
        transaction.get("bill_serial")
        or transaction.get("serial_number")
        or transaction.get("front_serial")
    )
    if not bill_serial or not is_safe_text(bill_serial, max_len=128):
        return False, "Invalid bill serial"
    if not is_valid_gtx_serial(bill_serial):
        return False, "Invalid GTX serial format"

    denom = transaction.get("denomination")
    valid_denominations = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    if denom not in valid_denominations:
        return False, "Invalid denomination"

    if not is_valid_tx_hash(transaction.get("hash")):
        return False, "Invalid hash"

    try:
        difficulty = int(transaction.get("mining_difficulty", 0))
        if difficulty < 0 or difficulty > 64:
            return False, "Invalid mining difficulty"
    except Exception:
        return False, "Invalid mining difficulty"

    return True, "OK"


def validate_block_payload(block: Any, max_tx_count: int = 5000) -> Tuple[bool, str]:
    if not isinstance(block, dict):
        return False, "Block payload must be an object"
    try:
        payload_size = len(json.dumps(block))
        if payload_size > MAX_BLOCK_BYTES:
            return False, "Block payload too large"
    except Exception:
        return False, "Block payload not serializable"

    for field in ("index", "previous_hash", "timestamp", "nonce", "difficulty", "hash"):
        if field not in block:
            return False, f"Missing field: {field}"

    if not is_valid_tx_hash(block.get("hash")):
        return False, "Invalid block hash"
    if not is_valid_tx_hash(block.get("previous_hash")):
        return False, "Invalid previous hash"

    txs = block.get("transactions", [])
    if not isinstance(txs, list):
        return False, "Invalid transactions list"
    if max_tx_count > 0 and len(txs) > max_tx_count:
        return False, "Too many transactions"

    return True, "OK"
