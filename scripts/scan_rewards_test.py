import argparse
import os
import time
from typing import Dict, List, Tuple

from lunalib.core.blockchain import BlockchainManager


def _normalize_address(addr: str) -> str:
    if not addr:
        return ""
    addr_str = str(addr).strip("'\" ").lower()
    return addr_str[4:] if addr_str.startswith("lun_") else addr_str


def _addresses_match(a: str, b: str) -> bool:
    return _normalize_address(a) == _normalize_address(b)


def _reward_hint(tx: Dict) -> bool:
    tx_type = str(tx.get("type") or "").lower()
    desc = str(tx.get("description") or "").lower()
    return (
        tx_type in {"reward", "coinbase", "mining_reward"}
        or str(tx.get("hash", "")).startswith("reward_")
        or (str(tx.get("from", "")) == "network" and ("reward" in desc or "mining" in desc))
        or (str(tx.get("from", "")) == "network" and tx.get("block_height") is not None)
        or ("reward" in tx)
    )


def _is_gtx_genesis(tx: Dict) -> bool:
    tx_type = str(tx.get("type") or "").lower()
    return tx_type in {"gtx_genesis", "genesis_bill", "gtxgenesis"}


def _extract_reward_amount(manager: BlockchainManager, block: Dict) -> float:
    return manager._extract_reward_amount(block)


def _extract_miner_address(manager: BlockchainManager, block: Dict) -> str:
    return manager._extract_miner_address(block)


def _collect_block_transactions(
    manager: BlockchainManager,
    block: Dict,
    address: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rewards: List[Dict] = []
    gtx_genesis: List[Dict] = []
    transfers: List[Dict] = []

    # Reward via block metadata
    miner = _extract_miner_address(manager, block)
    if _addresses_match(miner, address):
        reward_amount = _extract_reward_amount(manager, block)
        if reward_amount > 0:
            rewards.append({
                "type": "reward",
                "from": "network",
                "to": address,
                "amount": reward_amount,
                "block_height": block.get("index"),
                "timestamp": block.get("timestamp"),
                "hash": f"reward_{block.get('index')}_{str(block.get('hash', ''))[:8]}",
                "source": "block_metadata",
            })

    # Transactions in block
    for tx in block.get("transactions", []) or []:
        tx_type = str(tx.get("type") or "").lower()
        from_addr = tx.get("from") or tx.get("sender") or ""
        to_addr = tx.get("to") or tx.get("receiver") or ""

        if _reward_hint(tx):
            reward_to = (
                tx.get("to")
                or tx.get("receiver")
                or tx.get("issued_to")
                or tx.get("owner_address")
                or tx.get("to_address")
                or ""
            )
            if _addresses_match(reward_to, address):
                rewards.append({
                    **tx,
                    "block_height": block.get("index"),
                    "timestamp": tx.get("timestamp", block.get("timestamp")),
                    "source": "transaction",
                })
            continue

        if _is_gtx_genesis(tx):
            target = (
                tx.get("issued_to")
                or tx.get("owner_address")
                or tx.get("to")
                or tx.get("receiver")
                or ""
            )
            if _addresses_match(target, address):
                gtx_genesis.append({
                    **tx,
                    "block_height": block.get("index"),
                    "timestamp": tx.get("timestamp", block.get("timestamp")),
                })
            continue

        if _addresses_match(from_addr, address) or _addresses_match(to_addr, address):
            transfers.append({
                **tx,
                "block_height": block.get("index"),
                "timestamp": tx.get("timestamp", block.get("timestamp")),
            })

    return rewards, gtx_genesis, transfers


def _print_section(title: str, items: List[Dict], limit: int) -> None:
    print(f"{title}: {len(items)}")
    for tx in items[:limit]:
        tx_hash = str(tx.get("hash", ""))
        print(
            f"  - {tx_hash[:16]}... type={tx.get('type', '')} "
            f"amount={tx.get('amount', tx.get('denomination', ''))} "
            f"block={tx.get('block_height')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan rewards, transfers, and GTX genesis transactions.")
    parser.add_argument("--address", required=True, help="Target wallet/miner address")
    parser.add_argument("--node-url", default=os.getenv("LUNALIB_NODE_URL", "https://bank.linglin.art"))
    parser.add_argument("--start-height", type=int)
    parser.add_argument("--end-height", type=int)
    parser.add_argument("--lookback", type=int, default=200)
    parser.add_argument("--omit-gtx-genesis", action="store_true", help="Omit GTX genesis results")
    parser.add_argument("--omit-transfers", action="store_true", help="Omit transfer results")
    parser.add_argument("--limit", type=int, default=5, help="Max sample rows to print per section")

    args = parser.parse_args()

    manager = BlockchainManager(endpoint_url=args.node_url)

    end_height = args.end_height
    if end_height is None:
        end_height = manager.get_blockchain_height()

    if args.start_height is None:
        start_height = max(0, end_height - max(0, args.lookback) + 1)
    else:
        start_height = max(0, args.start_height)

    if end_height < start_height:
        print("Invalid height range")
        return

    print(f"[SCAN] Address={args.address}")
    print(f"[SCAN] Range={start_height}..{end_height} (lookback={args.lookback})")

    # Cache update is independent of filters
    blocks = manager.get_blocks_range(start_height, end_height)
    cached_height = manager.cache.get_highest_cached_height()
    print(f"[CACHE] highest_cached_height={cached_height}")

    wallet_rewards: List[Dict] = []
    wallet_transfers: List[Dict] = []
    node_rewards: List[Dict] = []
    node_genesis: List[Dict] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        rewards, gtx_genesis, transfers = _collect_block_transactions(manager, block, args.address)

        # Shared reward scanning for both node and wallet
        if rewards:
            wallet_rewards.extend(rewards)
            node_rewards.extend(rewards)

        # Node: rewards + gtx_genesis (unless omitted)
        if gtx_genesis and not args.omit_gtx_genesis:
            node_genesis.extend(gtx_genesis)

        # Wallet: rewards + transfers (unless omitted)
        if transfers and not args.omit_transfers:
            wallet_transfers.extend(transfers)

    print("\n[WALLET SCAN]")
    _print_section("Rewards", wallet_rewards, args.limit)
    _print_section("Transfers", wallet_transfers, args.limit)

    print("\n[NODE SCAN]")
    _print_section("Rewards", node_rewards, args.limit)
    _print_section("GTX Genesis", node_genesis, args.limit)

    print("\n[SUMMARY]")
    print(
        "wallet_rewards={wr} wallet_transfers={wt} node_rewards={nr} node_gtx_genesis={ng}"
        .format(
            wr=len(wallet_rewards),
            wt=len(wallet_transfers),
            nr=len(node_rewards),
            ng=len(node_genesis),
        )
    )


if __name__ == "__main__":
    main()
