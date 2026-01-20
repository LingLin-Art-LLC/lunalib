import pytest
import tempfile
from lunalib.core.wallet import LunaWallet
from lunalib.mining.miner import GenesisMiner
from lunalib.gtx.genesis import GTXGenesis

class TestIntegration:
    def test_complete_workflow(self, temp_dir):
        """Test complete wallet -> mining -> GTX workflow"""
        # Create wallet
        wallet = LunaWallet(data_dir=temp_dir)
        wallet_data = wallet.create_wallet("Integration Test", "test_pass")

        # Initialize systems
        miner = GenesisMiner()
        gtx = GTXGenesis()

        # Mock mining a bill
        with pytest.MonkeyPatch().context() as m:
            m.setattr(miner, '_perform_bill_mining', lambda *args: {
                "success": True,
                "hash": "000integration",
                "nonce": 999,
                "mining_time": 1.0
            })

            # Mine a bill
            bill = miner.mine_bill(1000, wallet_data['address'])

            assert bill['success'] is True
            assert bill['bill']['denomination'] == 1000
            assert bill['bill']['luna_value'] == 1000

            # Create proper bill data for verification
            metadata_hash = "test_metadata_hash_12345"
            signature = metadata_hash  # This will pass METHOD 1 verification
            
            # Register the bill in GTX system for verification
            bill_info = {
                'bill_serial': bill['bill']['bill_serial'],
                'denomination': 1000,
                'user_address': wallet_data['address'],
                'hash': "000integration",
                'mining_time': 1.0,
                'difficulty': 4,
                'luna_value': 1000,
                'timestamp': bill['bill']['timestamp'],
                'bill_data': {
                    'public_key': 'test_public_key',
                    'signature': signature,  # Use the signature that matches metadata_hash
                    'metadata_hash': metadata_hash,  # This must match signature for METHOD 1
                    'issued_to': wallet_data['address'],
                    'front_serial': bill['bill']['bill_serial'],
                    'denomination': 1000,
                    'timestamp': bill['bill']['timestamp'],
                    'type': 'GTX_Genesis',
                    'back_serial': ''  # Add this to avoid errors
                }
            }
            gtx.bill_registry.register_bill(bill_info)

            # Verify the bill
            verification = gtx.verify_bill(bill['bill']['bill_serial'])
            assert verification['valid'] is True