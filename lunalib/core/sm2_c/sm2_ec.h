#pragma once

#include "sm2_bn.h"

typedef struct {
    sm2_bn256 x;
    sm2_bn256 y;
    int infinity;
} sm2_point;

void sm2_point_set_generator(sm2_point* p);
void sm2_point_add(sm2_point* r, const sm2_point* a, const sm2_point* b);
void sm2_point_double(sm2_point* r, const sm2_point* a);
void sm2_point_scalar_mul(sm2_point* r, const sm2_point* p, const sm2_bn256* k);
