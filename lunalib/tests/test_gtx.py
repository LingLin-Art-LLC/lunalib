import sys
def safe_print(*args, **kwargs):
    encoding = sys.stdout.encoding or 'utf-8'
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        print(*(str(a).encode(encoding, errors='replace').decode(encoding) for a in args), **kwargs)
import pytest
from lunalib.gtx.genesis import GTXGenesis
from lunalib.gtx.digital_bill import DigitalBill
import time
class TestGTXGenesis:
    def test_digital_bill_creation(self, test_gtx):
        """Test digital bill creation"""
        bill = test_gtx.create_genesis_bill(
            denomination=1000,
            user_address="LUN_test_address_123",
            custom_data={"issuer": "test"}
        )
        
        assert bill.denomination == 1000
        assert bill.user_address == "LUN_test_address_123"
        assert bill.bill_data["issuer"] == "test"
        assert bill.bill_serial.startswith("GTX1000_")

    def test_bill_mining_data(self, test_gtx):
        """Test bill mining data generation"""
        bill = test_gtx.create_genesis_bill(100, "LUN_test")
        mining_data = bill.get_mining_data(nonce=123)
        
        assert mining_data["type"] == "GTX_Genesis"
        assert mining_data["denomination"] == 100
        assert mining_data["nonce"] == 123
        assert "previous_hash" in mining_data

    def test_bill_finalization(self, test_gtx):
        """Test bill finalization after mining"""
        bill = test_gtx.create_genesis_bill(1000, "LUN_test")
        
        final_bill = bill.finalize(
            hash="000abc123",
            nonce=12345,
            mining_time=3.2
        )
        
        assert final_bill["success"] is True
        assert final_bill["hash"] == "000abc123"
        assert final_bill["mining_time"] == 3.2
        assert "transaction_data" in final_bill

    def test_bill_verification(self, test_gtx, temp_dir):
        """Test bill verification with proper bill data structure"""
        import time
        import hashlib
        from unittest.mock import patch

        # Create a complete bill_info structure that matches what register_bill expects
        bill_serial = "GTX_TEST_12345"
        timestamp = time.time()
        denomination = 100
        user_address = "LUN_test_verify"

        # Create metadata that will pass one of the verification methods
        metadata_hash = hashlib.sha256(f"test_metadata_{timestamp}".encode()).hexdigest()
        public_key = f"pub_test_key_{timestamp}"
        signature = metadata_hash  # This will pass METHOD 1 verification

        # Create the complete bill_info structure
        bill_info = {
            'bill_serial': bill_serial,
            'denomination': denomination,
            'user_address': user_address,
            'hash': hashlib.sha256(bill_serial.encode()).hexdigest(),
            'mining_time': 1.5,
            'difficulty': 4,
            'luna_value': denomination,
            'timestamp': timestamp,
            'bill_data': {
                'public_key': public_key,
                'signature': signature,
                'metadata_hash': metadata_hash,
                'issued_to': user_address,
                'denomination': denomination,
                'front_serial': bill_serial,
                'timestamp': timestamp,
                'type': 'GTX_Genesis',
                'back_serial': ''
            }
        }

        # Register the bill first
        test_gtx.bill_registry.register_bill(bill_info)

        # DEBUG: Check what get_bill returns
        bill_data_from_registry = test_gtx.bill_registry.get_bill(bill_serial)
        safe_print(f"DEBUG: Bill data from registry: {bill_data_from_registry}")

        # Test verification - should pass METHOD 1
        result = test_gtx.verify_bill(bill_serial)
        safe_print(f"DEBUG: Verification result: {result}")
        
        assert result["valid"] is True

    def test_invalid_bill_verification(self, test_gtx):
        """Test verification of non-existent bill"""
        result = test_gtx.verify_bill("GTX_invalid_serial")
        assert result["valid"] is False
        assert "error" in result

    def test_portfolio_management(self, test_gtx, temp_dir):
        """Test user portfolio functionality"""
        # Use a unique user address to avoid conflicts
        unique_user = f"LUN_test_user_{int(time.time())}"
        
        # Add some test bills to registry
        for denom in [100, 1000, 10000]:
            bill = test_gtx.create_genesis_bill(denom, unique_user)
            final_bill = bill.finalize(f"000hash{denom}", 123, 1.0)
            
            # Register the bill properly
            bill_info = {
                'bill_serial': final_bill['bill_serial'],
                'denomination': denom,
                'user_address': unique_user,
                'hash': f"000hash{denom}",
                'mining_time': 1.0,
                'difficulty': 4,
                'luna_value': denom,
                'timestamp': final_bill['timestamp'],
                'bill_data': final_bill
            }
            test_gtx.bill_registry.register_bill(bill_info)

        portfolio = test_gtx.get_user_portfolio(unique_user)

        assert portfolio["user_address"] == unique_user
        assert portfolio["total_bills"] == 3