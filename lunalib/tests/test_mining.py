import pytest
import time
from unittest.mock import Mock, patch
from lunalib.mining.miner import GenesisMiner
from lunalib.mining.difficulty import DifficultySystem

class TestMining:
    def test_difficulty_system(self):
        """Test difficulty calculations with 9-tier system"""
        difficulty = DifficultySystem()
        
        # Test bill difficulties - 9 tiers
        assert difficulty.get_bill_difficulty(1) == 1        # Trivial
        assert difficulty.get_bill_difficulty(5) == 2        # Very Easy
        assert difficulty.get_bill_difficulty(50) == 3       # Easy
        assert difficulty.get_bill_difficulty(500) == 4      # Moderate
        assert difficulty.get_bill_difficulty(5000) == 5     # Standard
        assert difficulty.get_bill_difficulty(50000) == 6    # Challenging
        assert difficulty.get_bill_difficulty(500000) == 7   # Hard
        assert difficulty.get_bill_difficulty(5000000) == 8  # Very Hard
        assert difficulty.get_bill_difficulty(50000000) == 9 # Extreme
        
        # Test transaction difficulties - 9 tiers
        assert difficulty.get_transaction_difficulty(0.0005) == 1  # Trivial
        assert difficulty.get_transaction_difficulty(0.005) == 2   # Very Easy
        assert difficulty.get_transaction_difficulty(0.05) == 3    # Easy
        assert difficulty.get_transaction_difficulty(0.5) == 4     # Moderate
        assert difficulty.get_transaction_difficulty(5) == 5       # Standard
        assert difficulty.get_transaction_difficulty(50) == 6      # Challenging
        assert difficulty.get_transaction_difficulty(500) == 7     # Hard
        assert difficulty.get_transaction_difficulty(5000) == 8    # Very Hard
        assert difficulty.get_transaction_difficulty(50000) == 9   # Extreme
        
        # Test edge cases and boundary values
        assert difficulty.get_bill_difficulty(0) == 1              # Zero amount
        assert difficulty.get_bill_difficulty(-100) == 1           # Negative amount
        assert difficulty.get_bill_difficulty(999999999999) == 9   # Huge number
        
        # Test boundary values for each tier
        assert difficulty.get_bill_difficulty(1) == 1              # Tier 1 upper bound
        assert difficulty.get_bill_difficulty(10) == 2             # Tier 2 upper bound
        assert difficulty.get_bill_difficulty(100) == 3            # Tier 3 upper bound
        assert difficulty.get_bill_difficulty(1000) == 4           # Tier 4 upper bound
        assert difficulty.get_bill_difficulty(10000) == 5          # Tier 5 upper bound
        assert difficulty.get_bill_difficulty(100000) == 6         # Tier 6 upper bound
        assert difficulty.get_bill_difficulty(1000000) == 7        # Tier 7 upper bound
        assert difficulty.get_bill_difficulty(10000000) == 8       # Tier 8 upper bound
        
        # Test transaction edge cases
        assert difficulty.get_transaction_difficulty(0) == 1       # Zero amount
        assert difficulty.get_transaction_difficulty(-50) == 1     # Negative amount
        assert difficulty.get_transaction_difficulty(999999999) == 9 # Huge amount
        
        # Test difficulty names for all tiers
        assert difficulty.get_difficulty_name(1) == "Trivial"
        assert difficulty.get_difficulty_name(2) == "Very Easy"
        assert difficulty.get_difficulty_name(3) == "Easy"
        assert difficulty.get_difficulty_name(4) == "Moderate"
        assert difficulty.get_difficulty_name(5) == "Standard"
        assert difficulty.get_difficulty_name(6) == "Challenging"
        assert difficulty.get_difficulty_name(7) == "Hard"
        assert difficulty.get_difficulty_name(8) == "Very Hard"
        assert difficulty.get_difficulty_name(9) == "Extreme"
        assert difficulty.get_difficulty_name(999) == "Unknown"    # Invalid level
        
        # Test difficulty colors
        assert difficulty.get_difficulty_color(1) == "🟢"
        assert difficulty.get_difficulty_color(5) == "🟠"
        assert difficulty.get_difficulty_color(9) == "💀"
        
        # Test mining rewards with 9-tier system
        reward_fast = difficulty.calculate_mining_reward(1000, 5)   # 5 seconds - 50% bonus
        reward_medium = difficulty.calculate_mining_reward(1000, 15) # 15 seconds - 20% bonus
        reward_slow = difficulty.calculate_mining_reward(1000, 60)  # 60 seconds - no bonus
        
        assert reward_fast > reward_medium > reward_slow
        assert reward_fast == 1500.0  # 1000 * 1.5
        assert reward_medium == 1200.0  # 1000 * 1.2
        assert reward_slow == 1000.0  # 1000 * 1.0
        
        # Test expected mining time calculations for different tiers
        time_trivial = difficulty.get_expected_mining_time(1, 1000000)
        time_easy = difficulty.get_expected_mining_time(3, 1000000)
        time_standard = difficulty.get_expected_mining_time(5, 1000000)
        time_hard = difficulty.get_expected_mining_time(7, 1000000)
        time_extreme = difficulty.get_expected_mining_time(9, 1000000)
        
        # Higher difficulty should take exponentially longer
        assert time_extreme > time_hard > time_standard > time_easy > time_trivial
        
        # Test invalid difficulty levels
        assert difficulty.get_expected_mining_time(0, 1000000) == float('inf')
        assert difficulty.get_expected_mining_time(10, 1000000) == float('inf')

    def test_mining_reward_calculation(self):
        """Test mining reward calculations with 9-tier system"""
        difficulty = DifficultySystem()
        
        # Test base rewards for different denominations
        reward_small = difficulty.calculate_mining_reward(10, 10)    # Small bill, fast mining
        reward_medium = difficulty.calculate_mining_reward(1000, 10) # Medium bill, fast mining
        reward_large = difficulty.calculate_mining_reward(100000, 10) # Large bill, fast mining
        
        # Test time-based bonuses across different tiers
        reward_ultra_fast = difficulty.calculate_mining_reward(1000, 2)   # 2 seconds - 50% bonus
        reward_fast = difficulty.calculate_mining_reward(1000, 5)         # 5 seconds - 50% bonus
        reward_medium_time = difficulty.calculate_mining_reward(1000, 15) # 15 seconds - 20% bonus
        reward_slow = difficulty.calculate_mining_reward(1000, 35)        # 35 seconds - no bonus
        reward_very_slow = difficulty.calculate_mining_reward(1000, 70)   # 70 seconds - no bonus
        
        # Verify bonus tiers work correctly
        assert reward_ultra_fast == reward_fast == 1500.0  # Both get 50% bonus
        assert reward_medium_time == 1200.0  # 20% bonus
        assert reward_slow == reward_very_slow == 1000.0  # No bonus
        
        # Verify faster mining always gives better rewards
        assert reward_ultra_fast > reward_medium_time > reward_slow
        
        # Test edge cases
        zero_reward = difficulty.calculate_mining_reward(0, 5)
        negative_reward = difficulty.calculate_mining_reward(-100, 5)
        assert zero_reward == 0
        assert negative_reward == 0

    @patch('lunalib.mining.miner.GenesisMiner._perform_bill_mining')
    def test_bill_mining_success(self, mock_mining, test_miner, test_wallet):
        """Test successful bill mining across different tiers"""
        wallet, wallet_data = test_wallet
        
        # Test mining for different difficulty tiers
        test_cases = [
            (10, 2),    # Very Easy tier
            (100, 3),   # Easy tier
            (1000, 4),  # Moderate tier
            (10000, 5), # Standard tier
        ]
        
        for denomination, expected_difficulty in test_cases:
            # Mock successful mining
            mock_mining.return_value = {
                "success": True,
                "hash": "0" * expected_difficulty + "abc123",
                "nonce": 12345,
                "mining_time": 2.5
            }
            
            result = test_miner.mine_bill(denomination, wallet_data['address'])
            
            assert result['success'] is True
            assert result['bill']['denomination'] == denomination
            assert result['bill']['luna_value'] == denomination  # 1:1 ratio
            assert 'bill_serial' in result['bill']
            assert 'mining_time' in result

    @patch('lunalib.mining.miner.GenesisMiner._perform_bill_mining')
    def test_high_difficulty_mining(self, mock_mining, test_miner, test_wallet):
        """Test mining for high difficulty tiers"""
        wallet, wallet_data = test_wallet
        
        # Test high difficulty bills
        high_tier_cases = [
            (100000, 6),  # Challenging tier
            (1000000, 7), # Hard tier
            (10000000, 8), # Very Hard tier
        ]
        
        for denomination, expected_difficulty in high_tier_cases:
            mock_mining.return_value = {
                "success": True,
                "hash": "0" * expected_difficulty + "def456",
                "nonce": 99999,
                "mining_time": 30.0  # Longer mining time for higher difficulty
            }
            
            result = test_miner.mine_bill(denomination, wallet_data['address'])
            
            assert result['success'] is True
            assert result['bill']['denomination'] == denomination
            # High denomination bills might have additional validation
            assert result['bill']['luna_value'] >= denomination

    def test_mining_stop_functionality(self, test_miner):
        """Test mining stop functionality"""
        # Test starting and stopping mining
        test_miner.mining_active = True
        assert test_miner.mining_active is True
        
        test_miner.stop_mining()
        assert test_miner.mining_active is False
        
        # Test stopping when already stopped
        test_miner.stop_mining()
        assert test_miner.mining_active is False

    def test_invalid_denomination(self, test_miner, test_wallet):
        """Test mining with invalid denominations"""
        wallet, wallet_data = test_wallet
        
        invalid_denominations = [
            -100,      # Negative
            0,         # Zero
            999,       # Unusual amount
            -999999,   # Large negative
        ]
        
        for invalid_denom in invalid_denominations:
            result = test_miner.mine_bill(invalid_denom, wallet_data['address'])
            assert result['success'] is False
            assert 'error' in result

    @patch('lunalib.mining.miner.GenesisMiner._perform_bill_mining')
    def test_mining_failure(self, mock_mining, test_miner, test_wallet):
        """Test mining failure scenarios"""
        wallet, wallet_data = test_wallet
        
        # Mock mining failure
        mock_mining.return_value = {
            "success": False,
            "error": "Mining timeout"
        }
        
        result = test_miner.mine_bill(100, wallet_data['address'])
        
        assert result['success'] is False
        assert 'error' in result

    def test_difficulty_progression(self):
        """Test that difficulty progression makes sense"""
        difficulty = DifficultySystem()
        
        # Test that each tier is more difficult than the previous
        for i in range(1, 9):
            time_lower = difficulty.get_expected_mining_time(i, 1000000)
            time_higher = difficulty.get_expected_mining_time(i + 1, 1000000)
            assert time_higher > time_lower, f"Tier {i+1} should be harder than tier {i}"

    def test_mining_auto_stop(self, test_miner):
        """Test that mining can be stopped automatically"""
        test_miner.start_auto_bill_mining([100, 200, 300], "test_address")
        assert test_miner.mining_active is True
        
        # Stop mining and verify
        test_miner.stop_mining()
        assert test_miner.mining_active is False