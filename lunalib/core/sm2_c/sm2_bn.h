#pragma once

#include <stdint.h>
#include <stddef.h>

// 256-bit big integer (little-endian limbs)
typedef struct {
    uint32_t v[8];
} sm2_bn256;

int sm2_bn_from_bytes_be(sm2_bn256* out, const uint8_t* in32);
int sm2_bn_to_bytes_be(const sm2_bn256* in, uint8_t* out32);

// Basic arithmetic
void sm2_bn_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_sub(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
int sm2_bn_cmp(const sm2_bn256* a, const sm2_bn256* b);

// Modular arithmetic (mod p)
void sm2_bn_mod_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_mod_sub(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_mod_mul(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_mod_inv(sm2_bn256* r, const sm2_bn256* a);

// Mod n (group order) helpers
void sm2_bn_mod_n_add(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_mod_n_mul(sm2_bn256* r, const sm2_bn256* a, const sm2_bn256* b);
void sm2_bn_mod_n_inv(sm2_bn256* r, const sm2_bn256* a);
