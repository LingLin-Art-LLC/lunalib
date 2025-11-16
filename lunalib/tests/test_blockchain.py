import pytest
from unittest.mock import Mock, patch
from core.blockchain import BlockchainManager

class TestBlockchain:
    @patch('core.blockchain.requests.get')
    def test_blockchain_height(self, mock_get):
        """Test blockchain height retrieval"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'height': 1500}
        
        blockchain = BlockchainManager()
        height = blockchain.get_blockchain_height()
        
        assert height == 1500

    @patch('core.blockchain.requests.get')
    def test_network_connection(self, mock_get):
        """Test network connection checking"""
        mock_get.return_value.status_code = 200
        
        blockchain = BlockchainManager()
        is_connected = blockchain.check_network_connection()
        
        assert is_connected is True

    def test_transaction_scanning(self):
        """Test transaction scanning functionality"""
        blockchain = BlockchainManager()
        
        # This would typically be mocked in a real test
        # For now, test the method exists and returns expected type
        transactions = blockchain.scan_transactions_for_address("LUN_test", 0, 10)
        assert isinstance(transactions, list)