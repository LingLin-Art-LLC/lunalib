import os
import time
from lunalib.mining.miner import Miner

# Dummy config for demonstration
class Config:
    node_url = "https://bank.linglin.art"
    miner_address = "LUN_CCHhmzZBcbuSPrEHeBozWwJ6bCBwmhvUF6"
    difficulty = 3
    enable_gpu_mining = True
    enable_cpu_mining = True
    cuda_batch_size = 100000
    multi_gpu_enabled = True
    cuda_sm3_kernel = True

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
    os.environ.setdefault("LUNALIB_CUDA_SM3", "1")
    os.environ.setdefault("LUNALIB_MULTI_GPU", "1")
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
