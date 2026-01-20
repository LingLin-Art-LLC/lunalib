import pytest
import time
from lunalib.transactions.transactions import TransactionManager
from lunalib.transactions.security import TransactionSecurity

class TestTransactions:
    def test_transaction_creation(self, test_wallet, sample_transaction_data):
        """Test transaction creation and signing"""
        wallet, wallet_data = test_wallet
        tx_manager = TransactionManager()
        
        transaction = tx_manager.create_transaction(
            from_address=wallet_data['address'],
            to_address=sample_transaction_data['to'],
            amount=sample_transaction_data['amount'],
            private_key=wallet_data['private_key'],
            memo=sample_transaction_data['memo']
        )
        
        assert transaction['type'] == 'transfer'
        assert transaction['from'] == wallet_data['address']
        assert transaction['amount'] == 100.0
        assert 'signature' in transaction
        assert 'hash' in transaction

    def test_transaction_security_validation(self, test_wallet, sample_transaction_data):
        """Test transaction security validation"""
        wallet, wallet_data = test_wallet
        security = TransactionSecurity()
        tx_manager = TransactionManager()
        
        # Create valid transaction
        transaction = tx_manager.create_transaction(
            from_address=wallet_data['address'],
            to_address=sample_transaction_data['to'],
            amount=sample_transaction_data['amount'],
            private_key=wallet_data['private_key']
        )
        
        # Test validation
        is_valid, message = security.validate_transaction_security(transaction)
        assert is_valid is True
        assert "Valid" in message

    def test_invalid_transaction_validation(self):
        """Test validation of invalid transactions"""
        security = TransactionSecurity()
        # validator = TransactionValidator()
        
        # Test missing required fields
        invalid_tx = {"type": "transfer", "amount": 100}
        is_valid, message = security.validate_transaction_security(invalid_tx)
        assert is_valid is False
        assert "Missing required field" in message

    def test_gtx_transaction_creation(self):
        """Test GTX transaction creation"""
        tx_manager = TransactionManager()
        
        # Create a mock bill
        bill_info = {
            "owner_address": "LUN_test_address",
            "denomination": 1000
        }
        
        # Create GTX transaction
        gtx_tx = tx_manager.create_gtx_transaction(bill_info)
        
        assert gtx_tx['type'] == 'gtx_genesis'  # FIXED: Match actual type
        assert gtx_tx['amount'] == 1000
        assert gtx_tx['from'] == 'mining'

    def test_reward_transaction_creation(self):
        """Test reward transaction creation"""
        tx_manager = TransactionManager()
        
        reward_tx = tx_manager.create_reward_transaction(
            to_address="LUN_miner_123",
            amount=50.0,
            block_height=1000
        )
        
        assert reward_tx['type'] == 'reward'
        assert reward_tx['from'] == 'network'
        assert reward_tx['amount'] == 50.0
        assert reward_tx['block_height'] == 1000

    def test_transaction_risk_assessment(self, test_wallet, sample_transaction_data):
        """Test transaction risk assessment"""
        wallet, wallet_data = test_wallet
        tx_manager = TransactionManager()
        
        # Create high-value transaction
        transaction = tx_manager.create_transaction(
            from_address=wallet_data['address'],
            to_address=sample_transaction_data['to'],
            amount=1000000,  # High amount
            private_key=wallet_data['private_key']
        )
        
        risk_level, reason = tx_manager.security.assess_risk(transaction)
        assert risk_level in ["high", "medium", "low", "very_low"]
        assert isinstance(reason, str)