#include "sm2_ec.h"

static const sm2_bn256 SM2_A = {{
    0xFFFFFFFC, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
}};

static const sm2_bn256 SM2_GX = {{
    0x334C74C7, 0x715A4589, 0xF2660BE1, 0x8FE30BBF,
    0x6A39C994, 0x5F990446, 0x1F198119, 0x32C4AE2C
}};

static const sm2_bn256 SM2_GY = {{
    0x2139F0A0, 0x02DF32E5, 0xC62A4740, 0xD0A9877C,
    0x6B692153, 0x59BDCEE3, 0xF4F6779C, 0xBC3736A2
}};

static int bn_is_zero(const sm2_bn256* a) {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++) acc |= a->v[i];
    return acc == 0;
}

static int bn_eq(const sm2_bn256* a, const sm2_bn256* b) {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++) acc |= (a->v[i] ^ b->v[i]);
    return acc == 0;
}

static void bn_copy(sm2_bn256* r, const sm2_bn256* a) {
    for (int i = 0; i < 8; i++) r->v[i] = a->v[i];
}

void sm2_point_set_generator(sm2_point* p) {
    bn_copy(&p->x, &SM2_GX);
    bn_copy(&p->y, &SM2_GY);
    p->infinity = 0;
}

void sm2_point_double(sm2_point* r, const sm2_point* a) {
    if (a->infinity || bn_is_zero(&a->y)) {
        r->infinity = 1;
        return;
    }

    sm2_bn256 t1, t2, t3, lambda;
    sm2_bn_mod_mul(&t1, &a->x, &a->x); // x^2
    sm2_bn_mod_add(&t2, &t1, &t1);     // 2x^2
    sm2_bn_mod_add(&t1, &t2, &t1);     // 3x^2
    sm2_bn_mod_add(&t1, &t1, &SM2_A);  // 3x^2 + a

    sm2_bn_mod_add(&t2, &a->y, &a->y); // 2y
    sm2_bn_mod_inv(&t2, &t2);
    sm2_bn_mod_mul(&lambda, &t1, &t2);

    sm2_bn_mod_mul(&t3, &lambda, &lambda); // lambda^2
    sm2_bn_mod_sub(&t3, &t3, &a->x);
    sm2_bn_mod_sub(&t3, &t3, &a->x);       // x3
    bn_copy(&r->x, &t3);

    sm2_bn_mod_sub(&t1, &a->x, &r->x);
    sm2_bn_mod_mul(&t1, &lambda, &t1);
    sm2_bn_mod_sub(&t1, &t1, &a->y);
    bn_copy(&r->y, &t1);
    r->infinity = 0;
}

void sm2_point_add(sm2_point* r, const sm2_point* a, const sm2_point* b) {
    if (a->infinity) { *r = *b; return; }
    if (b->infinity) { *r = *a; return; }

    if (bn_eq(&a->x, &b->x)) {
        if (bn_eq(&a->y, &b->y)) {
            sm2_point_double(r, a);
        } else {
            r->infinity = 1;
        }
        return;
    }

    sm2_bn256 t1, t2, lambda;
    sm2_bn_mod_sub(&t1, &b->y, &a->y);
    sm2_bn_mod_sub(&t2, &b->x, &a->x);
    sm2_bn_mod_inv(&t2, &t2);
    sm2_bn_mod_mul(&lambda, &t1, &t2);

    sm2_bn256 x3, y3;
    sm2_bn_mod_mul(&x3, &lambda, &lambda);
    sm2_bn_mod_sub(&x3, &x3, &a->x);
    sm2_bn_mod_sub(&x3, &x3, &b->x);

    sm2_bn_mod_sub(&y3, &a->x, &x3);
    sm2_bn_mod_mul(&y3, &lambda, &y3);
    sm2_bn_mod_sub(&y3, &y3, &a->y);

    bn_copy(&r->x, &x3);
    bn_copy(&r->y, &y3);
    r->infinity = 0;
}

void sm2_point_scalar_mul(sm2_point* r, const sm2_point* p, const sm2_bn256* k) {
    sm2_point acc;
    acc.infinity = 1;
    sm2_point base = *p;

    for (int limb = 7; limb >= 0; limb--) {
        uint32_t w = k->v[limb];
        for (int bit = 31; bit >= 0; bit--) {
            sm2_point_double(&acc, &acc);
            if ((w >> bit) & 1U) {
                sm2_point tmp;
                sm2_point_add(&tmp, &acc, &base);
                acc = tmp;
            }
        }
    }
    *r = acc;
}
