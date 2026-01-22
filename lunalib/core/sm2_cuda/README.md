# SM2 CUDA (Scaffold)

This folder is a scaffold for SM2 GPU acceleration.

Planned order:
1. Signing kernel
2. Verification kernel
3. Key generation kernel

## Files
- `sm2_kernel.cu`: CUDA kernel stubs for SM2 signing/verification/keygen
- `sm2_gpu.py`: Python CuPy wrapper stubs
- `sm2_curve_params.h`: SM2 curve parameter scaffold
- `sm2_field.cu`: Field arithmetic scaffold (limb-based)

## Status
GPU kernels are not implemented yet. SM2 requires big integer arithmetic and elliptic-curve operations on GPU.

## CPU Fallback
`sm2_gpu.py` provides CPU fallback implementations for signing, verification, and key generation.
Set `LUNALIB_SM2_GPU=1` to force GPU (will raise NotImplementedError until kernels are implemented).

## Next Steps
- Implement field arithmetic mod p for SM2 curve
- Implement point add/double and scalar multiply
- Build signing kernel first, then verification, then keygen

## Current Limitation
`fe_mul_mod_p` is implemented with pseudo-Mersenne reduction for SM2 p.
`fe_inv_mod_p` uses exponentiation by $p-2$ (correct but not optimized).
Point operations are affine and unoptimized (slow, no Jacobian coordinates).

## SM2 GPU Signing
The current GPU signing path only computes $k\cdot G$ on the GPU. The final $(r,s)$ is computed on CPU.

## SM2 GPU Verification
The GPU verification path computes $x_1$ from $sG + tP$ on the GPU. The final check $R=(e+x_1)\bmod n$ is done on CPU.

## SM2 GPU Key Generation
GPU keygen treats the provided 32-byte entropy as the private key and computes the public key on GPU.
Use `sm2_keygen_gpu_with_pub()` to return (priv, pub) tuples (pub is 64 bytes x||y).
