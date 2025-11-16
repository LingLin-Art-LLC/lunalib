import time
import hashlib
import json
import threading
from typing import Dict, Optional
from mining.difficulty import DifficultySystem
from gtx.digital_bill import DigitalBill

class GenesisMiner:
    """Mines GTX Genesis bills with configurable difficulty"""
    
    def __init__(self):
        self.difficulty_system = DifficultySystem()
        self.mining_active = False
        self.current_thread = None
        
    def mine_bill(self, denomination, user_address, bill_data=None):
        """Mine a GTX Genesis bill"""
        difficulty = self.difficulty_system.get_bill_difficulty(denomination)
        
        digital_bill = DigitalBill(
            denomination=denomination,
            user_address=user_address,
            difficulty=difficulty,
            bill_data=bill_data or {}
        )
        
        print(f"⛏️ Mining GTX ${denomination:,} Bill - Difficulty: {difficulty} zeros")
        
        start_time = time.time()
        mining_result = self._perform_mining(digital_bill, difficulty)
        
        if mining_result["success"]:
            mining_time = time.time() - start_time
            bill = digital_bill.finalize(
                hash=mining_result["hash"],
                nonce=mining_result["nonce"],
                mining_time=mining_time
            )
            
            print(f"✅ Successfully mined GTX ${denomination:,} bill!")
            print(f"⏱️ Mining time: {mining_time:.2f}s")
            
            return bill
        else:
            return {"success": False, "error": "Mining failed"}
    
    def _perform_mining(self, digital_bill, difficulty):
        """Perform proof-of-work mining"""
        target = "0" * difficulty
        nonce = 0
        start_time = time.time()
        
        while self.mining_active:
            mining_data = digital_bill.get_mining_data(nonce)
            data_string = json.dumps(mining_data, sort_keys=True)
            bill_hash = hashlib.sha256(data_string.encode()).hexdigest()
            
            if bill_hash.startswith(target):
                mining_time = time.time() - start_time
                return {
                    "success": True,
                    "hash": bill_hash,
                    "nonce": nonce,
                    "mining_time": mining_time
                }
            
            nonce += 1
            
            # Progress updates
            if nonce % 100000 == 0:
                current_time = time.time()
                hashrate = nonce / (current_time - start_time)
                print(f"⏳ Attempts: {nonce:,} | Rate: {hashrate:,.0f} H/s")
        
        return {"success": False, "error": "Mining stopped"}
    
    def start_auto_mining(self, denominations, user_address, callback=None):
        """Start auto-mining multiple bills"""
        self.mining_active = True
        
        def auto_mine():
            results = []
            for denomination in denominations:
                if not self.mining_active:
                    break
                    
                result = self.mine_bill(denomination, user_address)
                results.append(result)
                
                if callback:
                    callback(result)
                
                time.sleep(1)
            
            return results
        
        self.current_thread = threading.Thread(target=auto_mine, daemon=True)
        self.current_thread.start()
        return True
    
    def stop_mining(self):
        """Stop all mining activities"""
        self.mining_active = False
        if self.current_thread:
            self.current_thread.join(timeout=5)
        print("🛑 Mining stopped")