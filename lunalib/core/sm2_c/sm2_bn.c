#include "sm2_bn.h"

int sm2_bn_from_bytes_be(sm2_bn256* out, const uint8_t* in32) {
    for (int i = 0; i < 8; i++) {
        int off = i * 4;
        uint32_t w = ((uint32_t)in32[off] << 24) |
                     ((uint32_t)in32[off + 1] << 16) |
                     ((uint32_t)in32[off + 2] << 8) |
                     ((uint32_t)in32[off + 3]);
        out->v[7 - i] = w;
    }
    return 0;
}

int sm2_bn_to_bytes_be(const sm2_bn256* in, uint8_t* out32) {
    for (int i = 0; i < 8; i++) {
        uint32_t w = in->v[7 - i];
        out32[i * 4 + 0] = (uint8_t)((w >> 24) & 0xFF);
        out32[i * 4 + 1] = (uint8_t)((w >> 16) & 0xFF);
        out32[i * 4 + 2] = (uint8_t)((w >> 8) & 0xFF);
        out32[i * 4 + 3] = (uint8_t)(w & 0xFF);
    }
    return 0;
}

void sm2_bn_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a->v[i] + b->v[i] + carry;
        r->v[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

void sm2_bn_sub(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = (uint64_t)a->v[i];
        uint64_t bv = (uint64_t)b->v[i] + borrow;
        if (av >= bv) {
            r->v[i] = (uint32_t)(av - bv);
            borrow = 0;
        } else {
            r->v[i] = (uint32_t)((av + (1ULL << 32)) - bv);
            borrow = 1;
        }
    }
}

int sm2_bn_cmp(const sm2_bn256* a, const sm2_bn256* b) {
    for (int i = 7; i >= 0; i--) {
        if (a->v[i] > b->v[i]) return 1;
        if (a->v[i] < b->v[i]) return -1;
    }
    return 0;
}

// SM2 prime p (little-endian limbs)
static const sm2_bn256 SM2_P = {{
    0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
}};

// SM2 order n
static const sm2_bn256 SM2_N = {{
    0x39D54123, 0x53BBF409, 0x21C6052B, 0x7203DF6B,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
}};

static void bn_copy(sm2_bn256* r, const sm2_bn256* a) {
    for (int i = 0; i < 8; i++) r->v[i] = a->v[i];
}

static int bn_is_zero(const sm2_bn256* a) {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++) acc |= a->v[i];
    return acc == 0;
}

static void bn_sub_p(sm2_bn256* r, const sm2_bn256* a) {
    sm2_bn_sub(r, a, &SM2_P);
}

void sm2_bn_mod_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    sm2_bn_add(r, a, b);
    if (sm2_bn_cmp(r, &SM2_P) >= 0) {
        sm2_bn256 tmp;
        bn_sub_p(&tmp, r);
        bn_copy(r, &tmp);
    }
}

void sm2_bn_mod_sub(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    sm2_bn256 tmp;
    sm2_bn_sub(&tmp, a, b);
    if (sm2_bn_cmp(&tmp, &SM2_P) < 0 && sm2_bn_cmp(a, b) < 0) {
        sm2_bn256 sum;
        sm2_bn_add(&sum, &tmp, &SM2_P);
        bn_copy(r, &sum);
        return;
    }
    bn_copy(r, &tmp);
}

static void bn_mul_raw(const sm2_bn256* a, const sm2_bn256* b, uint32_t out[16]) {
    uint64_t t[16] = {0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            t[i + j] += (uint64_t)a->v[i] * (uint64_t)b->v[j];
        }
    }
    for (int k = 0; k < 16; k++) {
        uint64_t carry = t[k] >> 32;
        out[k] = (uint32_t)t[k];
        if (k < 15) t[k + 1] += carry;
    }
}

static void bn_reduce_p(uint32_t in[16], sm2_bn256* r) {
    uint32_t tmp[16];
    for (int i = 0; i < 16; i++) tmp[i] = in[i];

    for (int round = 0; round < 3; round++) {
        int high_zero = 1;
        for (int i = 8; i < 16; i++) {
            if (tmp[i] != 0) { high_zero = 0; }
        }
        if (high_zero) break;

        long long acc[16] = {0};
        for (int i = 0; i < 8; i++) acc[i] = (long long)tmp[i];

        uint32_t h[8];
        for (int i = 0; i < 8; i++) h[i] = tmp[i + 8];

        for (int i = 0; i < 8; i++) acc[i] += (long long)h[i];
        for (int i = 0; i < 8; i++) acc[i + 7] += (long long)h[i];
        for (int i = 0; i < 8; i++) acc[i + 3] += (long long)h[i];
        for (int i = 0; i < 8; i++) acc[i + 2] -= (long long)h[i];

        long long carry = 0;
        for (int i = 0; i < 16; i++) {
            long long v = acc[i] + carry;
            tmp[i] = (uint32_t)(v & 0xFFFFFFFFLL);
            if (v >= 0) carry = v >> 32;
            else carry = -(((-v) + 0xFFFFFFFFLL) >> 32);
        }
    }

    for (int i = 0; i < 8; i++) r->v[i] = tmp[i];
    if (sm2_bn_cmp(r, &SM2_P) >= 0) {
        sm2_bn256 tmp;
        bn_sub_p(&tmp, r);
        bn_copy(r, &tmp);
    }
}

void sm2_bn_mod_mul(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    uint32_t prod[16];
    bn_mul_raw(a, b, prod);
    bn_reduce_p(prod, r);
}

static void bn_sqr(sm2_bn256* r, const sm2_bn256* a) {
    sm2_bn_mod_mul(r, a, a);
}

void sm2_bn_mod_inv(sm2_bn256* r, const sm2_bn256* a) {
    // p-2 exponent
    const uint32_t exp[8] = {
        0xFFFFFFFD, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
    };

    sm2_bn256 result;
    sm2_bn256 base;
    sm2_bn256 tmp;
    result.v[0] = 1;
    for (int i = 1; i < 8; i++) result.v[i] = 0;
    bn_copy(&base, a);

    for (int limb = 7; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            bn_sqr(&result, &result);
            if ((exp[limb] >> bit) & 1U) {
                sm2_bn_mod_mul(&tmp, &result, &base);
                bn_copy(&result, &tmp);
            }
        }
    }
    bn_copy(r, &result);
}

void sm2_bn_mod_n_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    sm2_bn_add(r, a, b);
    // r may be >= n; subtract until below
    while (sm2_bn_cmp(r, &SM2_N) >= 0) {
        sm2_bn256 tmp;
        sm2_bn_sub(&tmp, r, &SM2_N);
        bn_copy(r, &tmp);
    }
}

static void bn_mul_raw_n(const sm2_bn256* a, const sm2_bn256* b, uint32_t out[16]) {
    uint64_t t[16] = {0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            t[i + j] += (uint64_t)a->v[i] * (uint64_t)b->v[j];
        }
    }
    for (int k = 0; k < 16; k++) {
        uint64_t carry = t[k] >> 32;
        out[k] = (uint32_t)t[k];
        if (k < 15) t[k + 1] += carry;
    }
}

static int bn_highest_nonzero(const uint32_t* x, int len) {
    for (int i = len - 1; i >= 0; i--) {
        if (x[i] != 0) return i;
    }
    return -1;
}

static void bn_sub_shifted(uint32_t* x, const sm2_bn256* mod, int shift) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int idx = i + shift;
        uint64_t xv = (uint64_t)x[idx];
        uint64_t mv = (uint64_t)mod->v[i] + borrow;
        if (xv >= mv) {
            x[idx] = (uint32_t)(xv - mv);
            borrow = 0;
        } else {
            x[idx] = (uint32_t)((xv + (1ULL << 32)) - mv);
            borrow = 1;
        }
    }
    // propagate borrow
    int idx = 8 + shift;
    while (borrow && idx < 16) {
        uint64_t xv = (uint64_t)x[idx];
        if (xv >= 1) {
            x[idx] = (uint32_t)(xv - 1);
            borrow = 0;
        } else {
            x[idx] = 0xFFFFFFFFu;
            idx++;
        }
    }
}

static void bn_mod_n_reduce(uint32_t x[16], sm2_bn256* r) {
    int hi = bn_highest_nonzero(x, 16);
    while (hi >= 8) {
        int shift = hi - 7;
        // ensure no underflow; if underflow occurs, add back and shift down
        bn_sub_shifted(x, &SM2_N, shift);
        hi = bn_highest_nonzero(x, 16);
    }
    for (int i = 0; i < 8; i++) r->v[i] = x[i];
    while (sm2_bn_cmp(r, &SM2_N) >= 0) {
        sm2_bn256 tmp;
        sm2_bn_sub(&tmp, r, &SM2_N);
        bn_copy(r, &tmp);
    }
}

void sm2_bn_mod_n_mul(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b) {
    uint32_t prod[16];
    bn_mul_raw_n(a, b, prod);
    bn_mod_n_reduce(prod, r);
}

void sm2_bn_mod_n_inv(sm2_bn256* r, const sm2_bn256* a) {
    // n-2 exponent
    const uint32_t exp[8] = {
        0x39D54121, 0x53BBF409, 0x21C6052B, 0x7203DF6B,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
    };

    sm2_bn256 result;
    sm2_bn256 base;
    sm2_bn256 tmp;
    result.v[0] = 1;
    for (int i = 1; i < 8; i++) result.v[i] = 0;
    bn_copy(&base, a);

    for (int limb = 7; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            sm2_bn_mod_n_mul(&result, &result, &result);
            if ((exp[limb] >> bit) & 1U) {
                sm2_bn_mod_n_mul(&tmp, &result, &base);
                bn_copy(&result, &tmp);
            }
        }
    }
    bn_copy(r, &result);
}
