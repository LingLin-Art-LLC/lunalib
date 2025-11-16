# difficulty.py
class DifficultySystem:
    """9-tier difficulty system for mining and transactions"""
    
    def get_bill_difficulty(self, denomination):
        """Get mining difficulty based on bill denomination - 9 tiers"""
        if denomination <= 0:
            return 1  # Handle negative/zero amounts
        elif denomination <= 1:
            return 1  # Trivial
        elif denomination <= 10:
            return 2  # Very Easy
        elif denomination <= 100:
            return 3  # Easy
        elif denomination <= 1000:
            return 4  # Moderate
        elif denomination <= 10000:
            return 5  # Standard
        elif denomination <= 100000:
            return 6  # Challenging
        elif denomination <= 1000000:
            return 7  # Hard
        elif denomination <= 10000000:
            return 8  # Very Hard
        else:
            return 9  # Extreme
    
    def get_transaction_difficulty(self, amount):
        """Get transaction difficulty based on amount - 9 tiers"""
        if amount <= 0:
            return 1  # Handle negative/zero amounts
        elif amount <= 0.001:
            return 1  # Trivial
        elif amount <= 0.01:
            return 2  # Very Easy
        elif amount <= 0.1:
            return 3  # Easy
        elif amount <= 1.0:
            return 4  # Moderate
        elif amount <= 10.0:
            return 5  # Standard
        elif amount <= 100.0:
            return 6  # Challenging
        elif amount <= 1000.0:
            return 7  # Hard
        elif amount <= 10000.0:
            return 8  # Very Hard
        else:
            return 9  # Extreme
    
    def calculate_mining_reward(self, denomination, mining_time):
        """Calculate mining reward with time-based bonus"""
        if denomination <= 0:
            return 0
            
        base_reward = denomination  # Full denomination as base reward
        
        # Time bonus - faster mining gets higher reward
        if mining_time < 10:
            time_bonus = 0.5  # 50% bonus for very fast mining
        elif mining_time < 30:
            time_bonus = 0.2  # 20% bonus for fast mining
        else:
            time_bonus = 0.0  # No bonus for slow mining
            
        return base_reward * (1 + time_bonus)
    
    def get_difficulty_name(self, difficulty_level):
        """Get human-readable name for difficulty level"""
        names = {
            1: "Trivial",
            2: "Very Easy", 
            3: "Easy",
            4: "Moderate",
            5: "Standard",
            6: "Challenging",
            7: "Hard",
            8: "Very Hard",
            9: "Extreme"
        }
        return names.get(difficulty_level, "Unknown")
    
    def get_difficulty_color(self, difficulty_level):
        """Get color representation for difficulty level"""
        colors = {
            1: "🟢",  # Green - Trivial
            2: "🟢",  # Green - Very Easy
            3: "🟡",  # Yellow - Easy
            4: "🟡",  # Yellow - Moderate
            5: "🟠",  # Orange - Standard
            6: "🟠",  # Orange - Challenging
            7: "🔴",  # Red - Hard
            8: "🔴",  # Red - Very Hard
            9: "💀"   # Skull - Extreme
        }
        return colors.get(difficulty_level, "⚫")
    
    def get_expected_mining_time(self, difficulty_level, hashrate=1000000):
        """Get expected mining time in seconds based on difficulty and hashrate"""
        if difficulty_level < 1 or difficulty_level > 9:
            return float('inf')
            
        # Rough estimate: each difficulty level increases time by ~16x
        base_time = 0.1  # base time for difficulty 1
        time_multiplier = 16 ** (difficulty_level - 1)
        return base_time * time_multiplier / max(1, hashrate / 1000000)