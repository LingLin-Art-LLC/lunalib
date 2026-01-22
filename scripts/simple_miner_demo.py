import os
import time
from lunalib.mining.miner import Miner
from lunalib.core.sm3 import sm3_hex

# Dummy config for demonstration
class Config:
    node_url = "https://bank.linglin.art"
    miner_address = "LUN_BtAZyfkkTyS12fcDE9R8RygeyHWz3VMZE8"
    difficulty = 3
    enable_gpu_mining = True
    enable_cpu_mining = True
    cuda_batch_size = 100000

# Dummy DataManager (replace with your actual implementation)
class DummyDataManager:
    def load_mining_history(self): return []
    def save_mining_history(self, history): pass

def block_mined_callback(block):
    print("\n=== BLOCK MINED ===")
    print(f"Index: {block.get('index')}")
    print(f"Hash: {block.get('hash')}")
    print(f"Nonce: {block.get('nonce')}")
    print(f"Reward: {block.get('reward')}")
    print(f"Difficulty: {block.get('difficulty')}")
    print(f"Transactions: {len(block.get('transactions', []))}")
    print("===================\n")

def mining_status_callback(data):
    # Print detailed mining status updates
    phase = data.get("phase")
    msg = data.get("message", "")
    rate = data.get("hash_rate", 0)
    engine = data.get("engine", "?")
    print(f"[STATUS] {phase} | {msg} | Engine: {engine} | Hashrate: {rate:,.0f} H/s")

def main():
    config = Config()
    data_manager = DummyDataManager()
    miner = Miner(config, data_manager, block_mined_callback=block_mined_callback)
    miner.on_mining_status(mining_status_callback)

    print("Starting auto-mining (GPU preferred, fallback to CPU)...")
    miner.start_mining()

    # Let it run for a while (e.g., 30 seconds)
    time.sleep(30)
    miner.stop_mining()
    stats = miner.get_mining_stats()
    print("\n=== FINAL MINING STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("==========================\n")

if __name__ == "__main__":
    main()
