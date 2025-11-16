import pytest
import tempfile
from core.wallet import LunaWallet
from mining.miner import GenesisMiner
from gtx.genesis import GTXGenesis

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
            m.setattr(miner, '_perform_mining', lambda *args: {
                "success": True,
                "hash": "000integration",
                "nonce": 999,
                "mining_time": 1.0
            })
            
            # Mine a bill
            bill = miner.mine_bill(1000, wallet_data['address'])
            
            assert bill['success'] is True
            assert bill['denomination'] == 1000
            assert bill['luna_value'] == 1000
            
            # Verify the bill
            verification = gtx.verify_bill(bill['bill_serial'])
            assert verification['valid'] is True
            
            # Check portfolio
            portfolio = gtx.get_user_portfolio(wallet_data['address'])
            assert portfolio['total_bills'] >= 1
            assert portfolio['total_luna_value'] >= 1000