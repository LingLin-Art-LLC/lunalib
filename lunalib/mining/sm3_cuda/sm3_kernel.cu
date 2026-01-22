// sm3_kernel.cu
// CUDA kernel for SM3 hash

#include "sm3_kernel_constants.h"
#include "sm3_kernel_utils.cu"

__device__ void sm3_compress_block(const unsigned char* block, unsigned int* V) {
    unsigned int W[68];
    unsigned int W1[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((unsigned int)block[4*i] << 24) |
               ((unsigned int)block[4*i+1] << 16) |
               ((unsigned int)block[4*i+2] << 8) |
               ((unsigned int)block[4*i+3]);
    }

    for (int i = 16; i < 68; i++) {
        W[i] = P1(W[i-16] ^ W[i-9] ^ ROTL(W[i-3], 15)) ^ ROTL(W[i-13], 7) ^ W[i-6];
    }

    for (int i = 0; i < 64; i++) {
        W1[i] = W[i] ^ W[i+4];
    }

    unsigned int A = V[0], B = V[1], C = V[2], D = V[3];
    unsigned int E = V[4], F = V[5], G = V[6], H = V[7];

    for (int j = 0; j < 64; j++) {
        unsigned int Tj = (j < 16) ? 0x79CC4519 : 0x7A879D8A;
        unsigned int SS1 = ROTL((ROTL(A, 12) + E + ROTL(Tj, j)) & 0xFFFFFFFF, 7);
        unsigned int SS2 = SS1 ^ ROTL(A, 12);
        unsigned int TT1 = (FF(A, B, C, j) + D + SS2 + W1[j]) & 0xFFFFFFFF;
        unsigned int TT2 = (GG(E, F, G, j) + H + SS1 + W[j]) & 0xFFFFFFFF;
        D = C;
        C = ROTL(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = ROTL(F, 19);
        F = E;
        E = P0(TT2);
    }

    V[0] ^= A; V[1] ^= B; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

extern "C" __global__
void sm3_hash(const unsigned char* input, unsigned char* output, int input_len) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int block_offset = idx * SM3_BLOCK_SIZE;

    if (block_offset + SM3_BLOCK_SIZE > input_len) return;

    const unsigned int IV[8] = {
        0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
        0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
    };

    unsigned char block[SM3_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < SM3_BLOCK_SIZE; i++) {
        block[i] = input[block_offset + i];
    }

    unsigned int V[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        V[i] = IV[i];
    }

    sm3_compress_block(block, V);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[idx * SM3_HASH_SIZE + 4*i + 0] = (V[i] >> 24) & 0xFF;
        output[idx * SM3_HASH_SIZE + 4*i + 1] = (V[i] >> 16) & 0xFF;
        output[idx * SM3_HASH_SIZE + 4*i + 2] = (V[i] >> 8) & 0xFF;
        output[idx * SM3_HASH_SIZE + 4*i + 3] = (V[i] >> 0) & 0xFF;
    }
}

extern "C" __global__
void sm3_compress(const unsigned char* blocks, const unsigned int* iv_in, unsigned int* iv_out, int block_count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= block_count) return;

    const unsigned char* block = blocks + idx * SM3_BLOCK_SIZE;

    unsigned int V[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        V[i] = iv_in[idx * 8 + i];
    }

    sm3_compress_block(block, V);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        iv_out[idx * 8 + i] = V[i];
    }
}
