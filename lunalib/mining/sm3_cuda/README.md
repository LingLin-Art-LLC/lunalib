# SM3 CUDA Kernel for LunaLib Mining

This folder contains a custom CUDA implementation of the SM3 hash function for GPU-accelerated mining.

## Files
- `sm3_kernel.cu`: Main CUDA kernel (scaffold)
- `sm3_kernel_constants.h`: SM3 constants
- `sm3_kernel_utils.cu`: Utility functions (ROTL, P0, P1, FF, GG)
- `sm3_kernel.cuh`: Kernel header

## How to Use
1. Implement the SM3 logic in `sm3_kernel.cu` using the provided constants and utilities.
2. Compile the kernel with NVCC for use in C++ or integrate with Python using CuPy's RawKernel.
3. Prepare input data (block data, nonces) as byte arrays for batch hashing.
4. Launch the kernel to compute SM3 hashes in parallel on the GPU.

## Python (CuPy) Integration
Use [sm3_gpu.py](lunalib/mining/sm3_cuda/sm3_gpu.py) to run the kernel from Python:

- `gpu_sm3_hash_messages(messages)` supports multi-block messages.
- `gpu_sm3_hash_blocks(messages)` is an alias that uses the multi-block path.
- Messages are padded in Python, then processed block-by-block on the GPU with IV chaining.
- Returns a list of 32-byte SM3 hashes.

## References
- [SM3 Standard (IETF Draft)](https://tools.ietf.org/id/draft-oscca-cfrg-sm3-02.html)
- [CuPy RawKernel Documentation](https://docs.cupy.dev/en/stable/reference/kernel.html)

## Next Steps
- Fill in the SM3 logic in the kernel.
- Integrate with your mining pipeline.
- Benchmark and optimize for your GPU(s).
