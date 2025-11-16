import time
from typing import Optional, Dict, Any
import hashlib
import json

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

class CUDAManager:
    """Manages CUDA acceleration for mining operations"""
    
    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.device = None
        
        if self.cuda_available:
            self._initialize_cuda()
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            if not CUDA_AVAILABLE:
                return False
                
            if cp.cuda.runtime.getDeviceCount() > 0:
                print("✅ CUDA is available for accelerated mining")
                return True
            else:
                print("❌ CUDA drivers found but no GPU available")
                return False
        except Exception as e:
            print(f"❌ CUDA check failed: {e}")
            return False
    
    def _initialize_cuda(self):
        """Initialize CUDA device"""
        try:
            self.device = cp.cuda.Device(0)
            self.device.use()
            print(f"✅ Using CUDA device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        except Exception as e:
            print(f"❌ CUDA initialization failed: {e}")
            self.cuda_available = False
    
    def cuda_mine_batch(self, mining_data: Dict, difficulty: int, batch_size: int = 100000) -> Optional[Dict]:
        """Mine using CUDA acceleration"""
        if not self.cuda_available:
            return None
            
        try:
            target = "0" * difficulty
            nonce_start = 0
            start_time = time.time()
            
            while True:
                # Prepare batch data for GPU
                nonces = cp.arange(nonce_start, nonce_start + batch_size, dtype=cp.uint64)
                mining_strings = self._prepare_mining_batch(mining_data, nonces)
                
                # Compute hashes on GPU
                hashes = self._compute_hashes_gpu(mining_strings)
                
                # Check for successful hash
                for i, hash_hex in enumerate(hashes):
                    if hash_hex.startswith(target):
                        mining_time = time.time() - start_time
                        successful_nonce = nonce_start + i
                        
                        return {
                            "success": True,
                            "hash": hash_hex,
                            "nonce": int(successful_nonce),
                            "mining_time": mining_time,
                            "method": "cuda"
                        }
                
                nonce_start += batch_size
                
                # Progress update
                if nonce_start % (batch_size * 10) == 0:
                    current_time = time.time()
                    hashrate = nonce_start / (current_time - start_time)
                    print(f"⏳ CUDA: {nonce_start:,} attempts | {hashrate:,.0f} H/s")
                    
        except Exception as e:
            print(f"CUDA mining error: {e}")
            
        return None
    
    def _prepare_mining_batch(self, mining_data: Dict, nonces) -> Any:
        """Prepare batch mining data for GPU"""
        mining_strings = []
        
        for nonce in nonces:
            mining_data["nonce"] = int(nonce)
            data_string = json.dumps(mining_data, sort_keys=True)
            mining_strings.append(data_string.encode())
            
        return cp.array(mining_strings)
    
    def _compute_hashes_gpu(self, mining_strings) -> list:
        """Compute SHA256 hashes on GPU"""
        # Convert to CuPy array if needed
        if not isinstance(mining_strings, cp.ndarray):
            mining_strings = cp.array(mining_strings)
        
        # This is a simplified implementation
        # In a real implementation, you'd use proper CUDA kernels
        hashes = []
        for data in mining_strings:
            # For now, fall back to CPU hashing
            # A real implementation would use CUDA-accelerated hashing
            hash_obj = hashlib.sha256(data.tobytes())
            hashes.append(hash_obj.hexdigest())
            
        return hashes
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information"""
        if not self.cuda_available:
            return {"available": False}
            
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            return {
                "available": True,
                "device_name": props.get('name', 'Unknown'),
                "compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
                "total_memory": props.get('totalGlobalMem', 0),
                "multiprocessors": props.get('multiProcessorCount', 0)
            }
        except Exception as e:
            return {"available": False, "error": str(e)}