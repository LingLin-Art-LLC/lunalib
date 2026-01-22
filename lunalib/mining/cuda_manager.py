import time
import os
import struct
from typing import Optional, Dict, Any, Callable
import json
from lunalib.utils.hash import sm3_hex
from lunalib.core.sm3 import sm3_digest

try:
    from lunalib.mining.sm3_cuda.sm3_gpu import gpu_sm3_hash_messages, gpu_sm3_mine_compact
    _HAS_SM3_GPU = True
except Exception:
    gpu_sm3_hash_messages = None
    gpu_sm3_mine_compact = None
    _HAS_SM3_GPU = False

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

class CUDAManager:
    """Manages CUDA acceleration for mining operations"""
    
    def __init__(self):
        self.device_count = 0
        self.devices = []
        self.last_hashrate = 0.0
        self.last_attempts = 0
        self.last_duration = 0.0
        self.cuda_available = self._check_cuda()
        if self.cuda_available:
            self._initialize_cuda_multi()
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            if not CUDA_AVAILABLE:
                return False
            self.device_count = cp.cuda.runtime.getDeviceCount()
            if self.device_count > 0:
                print(f"✅ CUDA is available for accelerated mining ({self.device_count} device(s))")
                return True
            else:
                print("❌ CUDA drivers found but no GPU available")
                return False
        except Exception as e:
            print(f"❌ CUDA check failed: {e}")
            return False
    
    def _initialize_cuda_multi(self):
        """Initialize all available CUDA devices"""
        try:
            self.devices = []
            for i in range(self.device_count):
                dev = cp.cuda.Device(i)
                self.devices.append(dev)
                dev.use()
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"✅ CUDA device {i}: {props['name']}")
            self.device_count = len(self.devices)
        except Exception as e:
            print(f"❌ CUDA initialization failed: {e}")
            self.cuda_available = False

    def cuda_mine_multi_gpu_batch(self, mining_data: Dict, difficulty: int, batch_size: int = 1000000,
                                  progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Optional[Dict]:
        """Mine using all available CUDA devices in parallel"""
        import threading
        if not self.cuda_available or self.device_count < 2:
            print("[CUDA_DIAG] Multi-GPU requested but less than 2 devices available. Falling back to single GPU.")
            return self.cuda_mine_batch(mining_data, difficulty, batch_size, progress_callback)

        result_holder = {}
        stop_flag = threading.Event()

        # Weighted load balancing based on device properties
        weights = []
        total_weight = 0.0
        for i in range(self.device_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                sm_count = float(props.get("multiProcessorCount", 1))
                clock_khz = float(props.get("clockRate", 1))
                weight = max(1.0, sm_count * clock_khz)
            except Exception:
                weight = 1.0
            weights.append(weight)
            total_weight += weight

        total_batch = batch_size * self.device_count
        base_start = 0

        def mine_on_device(device_idx):
            try:
                self.devices[device_idx].use()
                print(f"[CUDA_DIAG] Mining on device {device_idx}")
                # Allocate per-device batch proportional to weight, with a minimum floor
                if total_weight > 0:
                    ratio = weights[device_idx] / total_weight
                else:
                    ratio = 1.0 / max(1, self.device_count)
                dev_batch = max(1, int(total_batch * ratio))

                # Compute nonce offset as prefix sum of previous device batches
                prefix = 0
                for i in range(device_idx):
                    if total_weight > 0:
                        r = weights[i] / total_weight
                    else:
                        r = 1.0 / max(1, self.device_count)
                    prefix += max(1, int(total_batch * r))

                start_nonce = base_start + prefix
                res = self.cuda_mine_batch(
                    mining_data,
                    difficulty,
                    dev_batch,
                    progress_callback,
                    start_nonce=start_nonce,
                    nonce_stride=total_batch,
                )
                if res and res.get("success"):
                    result_holder["result"] = res
                    stop_flag.set()
            except Exception as e:
                print(f"[CUDA_DIAG] Exception on device {device_idx}: {e}")

        threads = []
        for i in range(self.device_count):
            t = threading.Thread(target=mine_on_device, args=(i,))
            threads.append(t)
            t.start()

        while not stop_flag.is_set():
            for t in threads:
                t.join(timeout=0.1)
        # Stop all threads once a result is found
        print("[CUDA_DIAG] Multi-GPU mining finished.")
        return result_holder.get("result")
    
    def cuda_mine_batch(self, mining_data: Dict, difficulty: int, batch_size: int = 1000000,
                        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                        start_nonce: int = 0,
                        nonce_stride: Optional[int] = None) -> Optional[Dict]:
        """Mine using CUDA acceleration with CPU-side hash computation"""
        print("[CUDA_DIAG] Entered cuda_mine_batch")
        if not self.cuda_available:
            print("[CUDA_DIAG] cuda_available is False, returning None")
            return None
        try:
            print(f"[CUDA_DIAG] mining_data={mining_data}, difficulty={difficulty}, batch_size={batch_size}, start_nonce={start_nonce}")
            target = "0" * difficulty
            nonce_start = int(start_nonce)
            start_time = time.time()
            # Pre-compute the base data without nonce for efficiency
            base_data = {k: v for k, v in mining_data.items() if k != 'nonce'}
            progress_batches = int(os.getenv("LUNALIB_CUDA_PROGRESS_BATCHES", "5"))
            if progress_batches < 1:
                progress_batches = 1
            use_gpu_sm3 = _HAS_SM3_GPU
            cfg_flag = getattr(self, "use_sm3_kernel", None)
            if cfg_flag is None:
                use_gpu_sm3 = use_gpu_sm3 and os.getenv("LUNALIB_CUDA_SM3", "1") != "0"
            else:
                use_gpu_sm3 = use_gpu_sm3 and bool(cfg_flag)
            hash_mode = os.getenv("LUNALIB_MINING_HASH_MODE", "json").lower()
            if use_gpu_sm3:
                print("[CUDA_DIAG] Using GPU SM3 kernel for hashing")
            else:
                print("[CUDA_DIAG] Using CPU SM3 hashing fallback")
            if hash_mode == "compact":
                print("[CUDA_DIAG] Compact mining hash mode enabled (non-JSON). Ensure network supports this mode.")
            chunk_size = int(os.getenv("LUNALIB_CUDA_CHUNK_SIZE", "200000"))
            if chunk_size < 1:
                chunk_size = batch_size
            if chunk_size > batch_size:
                chunk_size = batch_size
            print(f"[CUDA_DIAG] progress_batches={progress_batches}")
            print(f"[CUDA_DIAG] chunk_size={chunk_size}")
            loop_count = 0
            attempts = 0
            while True:
                loop_count += 1
                if loop_count % 10 == 0:
                    print(f"[CUDA_DIAG] Loop iteration {loop_count}, nonce_start={nonce_start}")
                batch_end = nonce_start + batch_size
                print(f"[CUDA_DIAG] Generating nonces: {nonce_start} to {batch_end}")
                if hash_mode == "compact" and use_gpu_sm3 and gpu_sm3_mine_compact:
                    base80 = self._build_compact_base(mining_data)
                    if base80 is None:
                        print("[CUDA_DIAG] Compact base build failed, falling back to JSON hashing.")
                    else:
                        found_nonce = gpu_sm3_mine_compact(base80, nonce_start, batch_size, difficulty)
                        if found_nonce is not None:
                            mining_time = time.time() - start_time
                            hash_hex = self._compact_hash_from_base(base80, found_nonce)
                            print(f"[CUDA_DIAG] SUCCESS: Found valid hash at nonce {found_nonce}: {hash_hex}")
                            self.last_duration = mining_time
                            attempts += batch_size
                            self.last_attempts = attempts
                            self.last_hashrate = (attempts / mining_time) if mining_time > 0 else 0.0
                            if progress_callback:
                                progress_callback({
                                    "attempts": attempts,
                                    "hashrate": self.last_hashrate,
                                    "duration": mining_time,
                                })
                            return {
                                "success": True,
                                "hash": hash_hex,
                                "nonce": int(found_nonce),
                                "mining_time": mining_time,
                                "method": "cuda"
                            }
                else:
                    chunk_start = nonce_start
                    while chunk_start < batch_end:
                        chunk_end = min(chunk_start + chunk_size, batch_end)
                        nonces = list(range(chunk_start, chunk_end))
                        print(f"[CUDA_DIAG] Computing hashes in parallel for chunk size {len(nonces)}")
                        if use_gpu_sm3 and gpu_sm3_hash_messages:
                            hashes = self._compute_hashes_gpu(base_data, nonces)
                        else:
                            hashes = self._compute_hashes_parallel(base_data, nonces)
                        
                        # Check for successful hash
                        for i, hash_hex in enumerate(hashes):
                            if hash_hex.startswith(target):
                                mining_time = time.time() - start_time
                                successful_nonce = int(nonces[i])
                                print(f"[CUDA_DIAG] SUCCESS: Found valid hash at nonce {successful_nonce}: {hash_hex}")
                                self.last_duration = mining_time
                                attempts += batch_size
                                self.last_attempts = attempts
                                self.last_hashrate = (attempts / mining_time) if mining_time > 0 else 0.0
                                if progress_callback:
                                    progress_callback({
                                        "attempts": attempts,
                                        "hashrate": self.last_hashrate,
                                        "duration": mining_time,
                                    })
                                return {
                                    "success": True,
                                    "hash": hash_hex,
                                    "nonce": successful_nonce,
                                    "mining_time": mining_time,
                                    "method": "cuda"
                                }
                        chunk_start = chunk_end
                
                attempts += batch_size
                if nonce_stride is None or nonce_stride < 1:
                    nonce_start = batch_end
                else:
                    nonce_start += int(nonce_stride)
                print(f"[CUDA_DIAG] Incremented nonce_start to {nonce_start}")
                
                # Progress update
                if attempts % (batch_size * progress_batches) == 0:
                    current_time = time.time()
                    hashrate = attempts / (current_time - start_time)
                    self.last_hashrate = hashrate
                    self.last_attempts = attempts
                    self.last_duration = current_time - start_time
                    print(f"[CUDA_DIAG] Progress: {attempts:,} attempts | {hashrate:,.0f} H/s")
                    if progress_callback:
                        progress_callback({
                            "attempts": attempts,
                            "hashrate": hashrate,
                            "duration": current_time - start_time,
                        })
                
                # Timeout check
                if time.time() - start_time > 300:  # 5 minutes timeout
                    print("[CUDA_DIAG] Timeout reached, breaking loop.")
                    break
        except Exception as e:
            print(f"[CUDA_DIAG] Exception in cuda_mine_batch: {e}")
        print("[CUDA_DIAG] Exiting cuda_mine_batch, returning None.")
        return None

    def _build_compact_base(self, mining_data: Dict) -> Optional[bytes]:
        try:
            previous_hash = str(mining_data.get("previous_hash", ""))
            if len(previous_hash) != 64:
                return None
            prev_bytes = bytes.fromhex(previous_hash)
            index = int(mining_data.get("index", 0))
            difficulty = int(mining_data.get("difficulty", 0))
            timestamp = float(mining_data.get("timestamp", 0.0))
            miner = str(mining_data.get("miner", ""))
            miner_hash = sm3_digest(miner.encode())
            base = (
                prev_bytes
                + index.to_bytes(4, "big", signed=False)
                + difficulty.to_bytes(4, "big", signed=False)
                + struct.pack(">d", timestamp)
                + miner_hash
            )
            if len(base) != 80:
                return None
            return base
        except Exception:
            return None

    def _compact_hash_from_base(self, base80: bytes, nonce: int) -> str:
        nonce_bytes = int(nonce).to_bytes(8, "big", signed=False)
        return sm3_digest(base80 + nonce_bytes).hex()
    
    def _compute_hashes_parallel(self, base_data: Dict, nonces: list) -> list:
        """Compute SM3 hashes in parallel on CPU (string operations not supported on GPU)"""
        hashes = []
        
        for nonce in nonces:
            # Create mining data with current nonce
            mining_data = base_data.copy()
            mining_data["nonce"] = int(nonce)
            
            # Compute hash
            data_string = json.dumps(mining_data, sort_keys=True)
            hashes.append(sm3_hex(data_string.encode()))
            
        return hashes

    def _compute_hashes_gpu(self, base_data: Dict, nonces: list) -> list:
        """Compute SM3 hashes on GPU using custom CUDA kernel."""
        messages = []
        for nonce in nonces:
            mining_data = base_data.copy()
            mining_data["nonce"] = int(nonce)
            data_string = json.dumps(mining_data, sort_keys=True)
            messages.append(data_string.encode())
        try:
            hashes_bytes = gpu_sm3_hash_messages(messages) if gpu_sm3_hash_messages else []
        except Exception as e:
            print(f"[CUDA_DIAG] GPU SM3 hashing failed, falling back to CPU: {e}")
            return self._compute_hashes_parallel(base_data, nonces)
        return [hb.hex() for hb in hashes_bytes]
    
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