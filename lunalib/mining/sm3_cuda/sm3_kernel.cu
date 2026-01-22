// sm3_kernel.cu
// CUDA kernel for SM3 hash

#include "sm3_kernel_constants.h"
#include "sm3_kernel_utils.cu"

__device__ unsigned int sm3_throughput_sink = 0;

__device__ __forceinline__ void sm3_compress_block(const unsigned char* block, unsigned int* V) {
    unsigned int W[68];
    unsigned int W1[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((unsigned int)block[4*i] << 24) |
               ((unsigned int)block[4*i+1] << 16) |
               ((unsigned int)block[4*i+2] << 8) |
               ((unsigned int)block[4*i+3]);
    }

    #pragma unroll
    for (int i = 16; i < 68; i++) {
        W[i] = P1(W[i-16] ^ W[i-9] ^ ROTL(W[i-3], 15)) ^ ROTL(W[i-13], 7) ^ W[i-6];
    }

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        W1[i] = W[i] ^ W[i+4];
    }

    unsigned int A = V[0], B = V[1], C = V[2], D = V[3];
    unsigned int E = V[4], F = V[5], G = V[6], H = V[7];

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        const unsigned int Tj = 0x79CC4519;
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

    #pragma unroll
    for (int j = 16; j < 64; j++) {
        const unsigned int Tj = 0x7A879D8A;
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

__device__ __forceinline__ void sm3_set_nonce(unsigned char* block, unsigned long long nonce) {
    block[0] = (unsigned char)((nonce >> 56) & 0xFF);
    block[1] = (unsigned char)((nonce >> 48) & 0xFF);
    block[2] = (unsigned char)((nonce >> 40) & 0xFF);
    block[3] = (unsigned char)((nonce >> 32) & 0xFF);
    block[4] = (unsigned char)((nonce >> 24) & 0xFF);
    block[5] = (unsigned char)((nonce >> 16) & 0xFF);
    block[6] = (unsigned char)((nonce >> 8) & 0xFF);
    block[7] = (unsigned char)((nonce >> 0) & 0xFF);
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
void sm3_compress(const unsigned char* blocks, const unsigned int* iv_in, unsigned int* iv_out, int block_count, int iterations) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= block_count) return;
    if (iterations < 1) iterations = 1;

    int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < iterations; i++) {
        int block_idx = idx + i * stride;
        if (block_idx >= block_count) break;

        const unsigned char* block = blocks + block_idx * SM3_BLOCK_SIZE;

        unsigned int V[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            V[j] = iv_in[block_idx * 8 + j];
        }

        sm3_compress_block(block, V);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            iv_out[block_idx * 8 + j] = V[j];
        }
    }
}

extern "C" __global__
void sm3_throughput_persistent(const unsigned char* block, unsigned long long* counter, volatile unsigned int* stop_flag) {
    unsigned long long local = 0;
    unsigned int mix = 0;
    unsigned char local_block[SM3_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < SM3_BLOCK_SIZE; i++) {
        local_block[i] = block[i];
    }
    unsigned long long nonce = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    while (!(*stop_flag)) {
        sm3_set_nonce(local_block, nonce);
        unsigned int V[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            V[i] = SM3_IV[i];
        }
        sm3_compress_block(local_block, V);
        mix ^= V[0];
        local++;
        nonce += stride;
    }
    if (local > 0) {
        atomicAdd(counter, local);
    }
    if (mix != 0) {
        atomicAdd(&sm3_throughput_sink, mix);
    }
}

extern "C" __global__
void sm3_throughput_timed(const unsigned char* block, unsigned long long* counter, unsigned long long stop_cycles) {
    unsigned long long local = 0;
    unsigned int mix = 0;
    unsigned char local_block[SM3_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < SM3_BLOCK_SIZE; i++) {
        local_block[i] = block[i];
    }
    unsigned long long nonce = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long start = clock64();
    while ((clock64() - start) < stop_cycles) {
        sm3_set_nonce(local_block, nonce);
        unsigned int V[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            V[i] = SM3_IV[i];
        }
        sm3_compress_block(local_block, V);
        mix ^= V[0];
        local++;
        nonce += stride;
    }
    if (local > 0) {
        atomicAdd(counter, local);
    }
    if (mix != 0) {
        atomicAdd(&sm3_throughput_sink, mix);
    }
}

extern "C" __global__
void sm3_throughput_iters(const unsigned char* block, unsigned long long* counter, unsigned int iterations) {
    unsigned long long local = 0;
    unsigned int mix = 0;
    unsigned char local_block[SM3_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < SM3_BLOCK_SIZE; i++) {
        local_block[i] = block[i];
    }
    unsigned long long nonce = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned int i = 0; i < iterations; i++) {
        sm3_set_nonce(local_block, nonce);
        unsigned int V[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            V[j] = SM3_IV[j];
        }
        sm3_compress_block(local_block, V);
        mix ^= V[0];
        local++;
        nonce += stride;
    }
    if (local > 0) {
        atomicAdd(counter, local);
    }
    if (mix != 0) {
        atomicAdd(&sm3_throughput_sink, mix);
    }
}

__device__ bool sm3_check_difficulty(const unsigned int* V, int difficulty) {
    if (difficulty <= 0) {
        return true;
    }
    int full_bytes = difficulty / 2;
    int half = difficulty & 1;
    int byte_index = 0;
    for (; byte_index < full_bytes; byte_index++) {
        int word_index = byte_index >> 2;
        int shift = 24 - ((byte_index & 3) << 3);
        unsigned int byte_val = (V[word_index] >> shift) & 0xFFU;
        if (byte_val != 0) {
            return false;
        }
    }
    if (half) {
        int word_index = byte_index >> 2;
        int shift = 24 - ((byte_index & 3) << 3);
        unsigned int byte_val = (V[word_index] >> shift) & 0xFFU;
        if ((byte_val & 0xF0U) != 0) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void sm3_hash_compact_88(const unsigned char* base80, unsigned long long nonce, unsigned int* V) {
    const unsigned int IV[8] = {
        0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
        0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
    };

    unsigned char block0[SM3_BLOCK_SIZE];
    unsigned char block1[SM3_BLOCK_SIZE];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        block0[i] = base80[i];
    }

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        block1[i] = 0;
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block1[i] = base80[64 + i];
    }

    block1[16] = (unsigned char)((nonce >> 56) & 0xFF);
    block1[17] = (unsigned char)((nonce >> 48) & 0xFF);
    block1[18] = (unsigned char)((nonce >> 40) & 0xFF);
    block1[19] = (unsigned char)((nonce >> 32) & 0xFF);
    block1[20] = (unsigned char)((nonce >> 24) & 0xFF);
    block1[21] = (unsigned char)((nonce >> 16) & 0xFF);
    block1[22] = (unsigned char)((nonce >> 8) & 0xFF);
    block1[23] = (unsigned char)((nonce >> 0) & 0xFF);

    block1[24] = 0x80;
    block1[56] = 0x00;
    block1[57] = 0x00;
    block1[58] = 0x00;
    block1[59] = 0x00;
    block1[60] = 0x00;
    block1[61] = 0x00;
    block1[62] = 0x02;
    block1[63] = 0xC0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        V[i] = IV[i];
    }

    sm3_compress_block(block0, V);
    sm3_compress_block(block1, V);
}

extern "C" __global__
void sm3_mine_compact(const unsigned char* base80, unsigned long long start_nonce, unsigned long long count,
                      int difficulty, int iterations, unsigned int* found, unsigned long long* out_nonce) {
    __shared__ unsigned char s_base80[80];

    if (threadIdx.x < 80) {
        s_base80[threadIdx.x] = base80[threadIdx.x];
    }
    __syncthreads();

    unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    if (idx >= count) return;
    if (*found) return;
    if (iterations < 1) iterations = 1;

    for (int i = 0; i < iterations; i++) {
        unsigned long long nonce = start_nonce + idx + (unsigned long long)i * stride;
        if (nonce >= start_nonce + count) break;
        if (*found) return;
        unsigned int V[8];
        sm3_hash_compact_88(s_base80, nonce, V);
        if (sm3_check_difficulty(V, difficulty)) {
            if (atomicCAS(found, 0U, 1U) == 0U) {
                *out_nonce = nonce;
            }
            return;
        }
    }
}
