// sm3_kernel_utils.cu
// Utility functions for SM3 CUDA kernel

__device__ __forceinline__ unsigned int ROTL(unsigned int x, unsigned int n) {
    unsigned int r = n & 31;
    if (r == 0) {
        return x;
    }
    return (x << r) | (x >> (32 - r));
}

__device__ __forceinline__ unsigned int P0(unsigned int x) {
    return x ^ ROTL(x, 9) ^ ROTL(x, 17);
}

__device__ __forceinline__ unsigned int P1(unsigned int x) {
    return x ^ ROTL(x, 15) ^ ROTL(x, 23);
}

__device__ __forceinline__ unsigned int FF(unsigned int x, unsigned int y, unsigned int z, int j) {
    return (j < 16) ? (x ^ y ^ z) : ((x & y) | (x & z) | (y & z));
}

__device__ __forceinline__ unsigned int GG(unsigned int x, unsigned int y, unsigned int z, int j) {
    return (j < 16) ? (x ^ y ^ z) : ((x & y) | (~x & z));
}
