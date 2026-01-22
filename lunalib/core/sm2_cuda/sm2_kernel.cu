// sm2_kernel.cu
// CUDA kernel scaffold for SM2 operations

#include "sm2_curve_params.h"
#include "sm2_field.cu"

__device__ __forceinline__ void load_scalar_be(const unsigned char* in, uint32_t out[8]) {
    // in: 32 bytes big-endian, out: 8 limbs little-endian
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int off = i * 4;
        uint32_t w = ((uint32_t)in[off] << 24) |
                     ((uint32_t)in[off + 1] << 16) |
                     ((uint32_t)in[off + 2] << 8) |
                     ((uint32_t)in[off + 3]);
        out[7 - i] = w;
    }
}

__device__ __forceinline__ void store_fe_be(const sm2_fe* a, unsigned char* out) {
    // a: 8 limbs little-endian -> out: 32 bytes big-endian
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t w = a->v[7 - i];
        out[i * 4 + 0] = (unsigned char)((w >> 24) & 0xFF);
        out[i * 4 + 1] = (unsigned char)((w >> 16) & 0xFF);
        out[i * 4 + 2] = (unsigned char)((w >> 8) & 0xFF);
        out[i * 4 + 3] = (unsigned char)(w & 0xFF);
    }
}

__device__ __forceinline__ void load_fe_be(const unsigned char* in, sm2_fe* out) {
    // in: 32 bytes big-endian, out: 8 limbs little-endian
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t w = ((uint32_t)in[i * 4] << 24) |
                     ((uint32_t)in[i * 4 + 1] << 16) |
                     ((uint32_t)in[i * 4 + 2] << 8) |
                     ((uint32_t)in[i * 4 + 3]);
        out->v[7 - i] = w;
    }
}

extern "C" __global__
void sm2_sign_kernel(const unsigned char* msg, int msg_len,
                     const unsigned char* priv_keys, int key_stride,
                     const unsigned char* nonces, int nonce_stride,
                     unsigned char* pub_out, int pub_stride,
                     int batch_count) {
    // This kernel computes k*G for each nonce and outputs (x1,y1) in pub_out.
    // Signature r,s are computed on CPU using e,d,k and x1.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_count) return;

    const unsigned char* nonce = nonces + idx * nonce_stride;
    uint32_t k[8];
    load_scalar_be(nonce, k);

    sm2_point G;
    point_set_generator(&G);
    sm2_point R;
    point_scalar_mul(&R, &G, k);

    unsigned char* out = pub_out + idx * pub_stride;
    store_fe_be(&R.x, out);
    store_fe_be(&R.y, out + 32);
}

extern "C" __global__
void sm2_verify_kernel(const unsigned char* msg, int msg_len,
                       const unsigned char* pub_keys, int key_stride,
                       const unsigned char* sigs, int sig_stride,
                       int* results, int batch_count) {
    // NOTE: legacy signature kept for compatibility; not used by Python wrapper.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_count) return;
    results[idx] = 0;
}

extern "C" __global__
void sm2_verify_x1_kernel(const unsigned char* pub_keys, int key_stride,
                          const unsigned char* scalars, int scalar_stride,
                          unsigned char* x_out, int out_stride,
                          int batch_count) {
    // scalars = [s||t] (64 bytes per item), pub_keys = x||y (64 bytes per item)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_count) return;

    const unsigned char* pk = pub_keys + idx * key_stride;
    const unsigned char* sc = scalars + idx * scalar_stride;

    sm2_point P;
    load_fe_be(pk, &P.x);
    load_fe_be(pk + 32, &P.y);
    P.infinity = 0;

    uint32_t s[8];
    uint32_t t[8];
    load_scalar_be(sc, s);
    load_scalar_be(sc + 32, t);

    sm2_point G;
    point_set_generator(&G);

    sm2_point sG;
    sm2_point tP;
    point_scalar_mul(&sG, &G, s);
    point_scalar_mul(&tP, &P, t);

    sm2_point R;
    point_add(&R, &sG, &tP);

    unsigned char* out = x_out + idx * out_stride;
    store_fe_be(&R.x, out);
}

extern "C" __global__
void sm2_keygen_kernel(const unsigned char* entropy, int entropy_stride,
                       unsigned char* priv_out, int priv_stride,
                       unsigned char* pub_out, int pub_stride,
                       int batch_count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_count) return;

    const unsigned char* ent = entropy + idx * entropy_stride;
    uint32_t d[8];
    load_scalar_be(ent, d);

    sm2_point G;
    point_set_generator(&G);
    sm2_point Q;
    point_scalar_mul(&Q, &G, d);

    // Write private key as provided (32 bytes)
    unsigned char* priv = priv_out + idx * priv_stride;
    #pragma unroll
    for (int i = 0; i < 32; i++) priv[i] = ent[i];

    // Write public key (x||y)
    unsigned char* pub = pub_out + idx * pub_stride;
    store_fe_be(&Q.x, pub);
    store_fe_be(&Q.y, pub + 32);
}
