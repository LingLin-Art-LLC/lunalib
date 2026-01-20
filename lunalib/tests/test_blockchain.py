import pytest
from unittest.mock import Mock, patch
from lunalib.core.blockchain import BlockchainManager

class TestBlockchain:
    @patch('lunalib.core.blockchain.requests.Session.get')
    def test_blockchain_height(self, mock_get):
        """Test blockchain height retrieval"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'blocks': [{'index': i, 'hash': '0'*64} for i in range(1501)]
        }
        
        blockchain = BlockchainManager()
        height = blockchain.get_blockchain_height()
        
        assert height == 1500

    @patch('lunalib.core.blockchain.requests.Session.get')
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