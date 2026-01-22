// sm2_field.cu
// Field arithmetic scaffolding for SM2 (mod p)

#include "sm2_curve_params.h"

typedef struct {
	uint32_t v[8];
} sm2_fe;

__device__ __forceinline__ void fe_copy(sm2_fe* r, const sm2_fe* a) {
	#pragma unroll
	for (int i = 0; i < 8; i++) r->v[i] = a->v[i];
}

__device__ __forceinline__ void fe_set_u32(sm2_fe* r, uint32_t x) {
	r->v[0] = x;
	#pragma unroll
	for (int i = 1; i < 8; i++) r->v[i] = 0;
}

__device__ __forceinline__ bool fe_is_zero(const sm2_fe* a) {
	uint32_t acc = 0;
	#pragma unroll
	for (int i = 0; i < 8; i++) acc |= a->v[i];
	return acc == 0;
}

__device__ __forceinline__ bool fe_eq(const sm2_fe* a, const sm2_fe* b) {
	uint32_t acc = 0;
	#pragma unroll
	for (int i = 0; i < 8; i++) acc |= (a->v[i] ^ b->v[i]);
	return acc == 0;
}

__device__ __forceinline__ int fe_cmp(const sm2_fe* a, const uint32_t* b) {
	for (int i = 7; i >= 0; i--) {
		if (a->v[i] > b[i]) return 1;
		if (a->v[i] < b[i]) return -1;
	}
	return 0;
}

__device__ __forceinline__ void fe_add_raw(sm2_fe* r, const sm2_fe* a, const sm2_fe* b) {
	unsigned long long carry = 0ULL;
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		unsigned long long sum = (unsigned long long)a->v[i] + b->v[i] + carry;
		r->v[i] = (uint32_t)sum;
		carry = sum >> 32;
	}
}

__device__ __forceinline__ uint32_t fe_sub_raw(sm2_fe* r, const sm2_fe* a, const sm2_fe* b) {
	unsigned long long borrow = 0ULL;
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		unsigned long long av = (unsigned long long)a->v[i];
		unsigned long long bv = (unsigned long long)b->v[i] + borrow;
		if (av >= bv) {
			r->v[i] = (uint32_t)(av - bv);
			borrow = 0ULL;
		} else {
			r->v[i] = (uint32_t)((av + (1ULL << 32)) - bv);
			borrow = 1ULL;
		}
	}
	return (uint32_t)borrow;
}

__device__ __forceinline__ void fe_sub_p(sm2_fe* r, const sm2_fe* a) {
	sm2_fe p;
	#pragma unroll
	for (int i = 0; i < 8; i++) p.v[i] = SM2_P[i];
	fe_sub_raw(r, a, &p);
}

__device__ __forceinline__ void fe_add_mod_p(sm2_fe* r, const sm2_fe* a, const sm2_fe* b) {
	fe_add_raw(r, a, b);
	if (fe_cmp(r, SM2_P) >= 0) {
		sm2_fe tmp;
		fe_sub_p(&tmp, r);
		fe_copy(r, &tmp);
	}
}

__device__ __forceinline__ void fe_sub_mod_p(sm2_fe* r, const sm2_fe* a, const sm2_fe* b) {
	sm2_fe tmp;
	uint32_t borrow = fe_sub_raw(&tmp, a, b);
	if (borrow) {
		sm2_fe p;
		#pragma unroll
		for (int i = 0; i < 8; i++) p.v[i] = SM2_P[i];
		fe_add_raw(&tmp, &tmp, &p);
	}
	fe_copy(r, &tmp);
}

__device__ __forceinline__ void fe_neg_mod_p(sm2_fe* r, const sm2_fe* a) {
	if (fe_is_zero(a)) {
		fe_copy(r, a);
		return;
	}
	sm2_fe p;
	#pragma unroll
	for (int i = 0; i < 8; i++) p.v[i] = SM2_P[i];
	fe_sub_raw(r, &p, a);
}

// ---- Multiplication / reduction ----

__device__ __forceinline__ void fe_mul_raw(const sm2_fe* a, const sm2_fe* b, uint32_t out[16]) {
	unsigned long long t[16] = {0};
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		#pragma unroll
		for (int j = 0; j < 8; j++) {
			t[i + j] += (unsigned long long)a->v[i] * (unsigned long long)b->v[j];
		}
	}
	#pragma unroll
	for (int k = 0; k < 16; k++) {
		unsigned long long carry = t[k] >> 32;
		out[k] = (uint32_t)t[k];
		if (k < 15) t[k + 1] += carry;
	}
}

__device__ __forceinline__ void fe_reduce_p(uint32_t in[16], sm2_fe* r) {
	// SM2 p = 2^256 - 2^224 - 2^96 + 2^64 - 1
	// Reduction: L + H + (H<<224) + (H<<96) - (H<<64)
	uint32_t tmp[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) tmp[i] = in[i];

	for (int round = 0; round < 3; round++) {
		// Check if high limbs are zero
		bool high_zero = true;
		#pragma unroll
		for (int i = 8; i < 16; i++) {
			if (tmp[i] != 0) { high_zero = false; }
		}
		if (high_zero) break;

		long long acc[16];
		#pragma unroll
		for (int i = 0; i < 16; i++) acc[i] = 0;

		// L
		#pragma unroll
		for (int i = 0; i < 8; i++) acc[i] = (long long)tmp[i];

		// H = tmp[8..15]
		uint32_t h[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) h[i] = tmp[i + 8];

		// acc += H
		#pragma unroll
		for (int i = 0; i < 8; i++) acc[i] += (long long)h[i];
		// acc += H << 7
		#pragma unroll
		for (int i = 0; i < 8; i++) acc[i + 7] += (long long)h[i];
		// acc += H << 3
		#pragma unroll
		for (int i = 0; i < 8; i++) acc[i + 3] += (long long)h[i];
		// acc -= H << 2
		#pragma unroll
		for (int i = 0; i < 8; i++) acc[i + 2] -= (long long)h[i];

		// Normalize acc to 32-bit limbs
		long long carry = 0;
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			long long v = acc[i] + carry;
			tmp[i] = (uint32_t)(v & 0xFFFFFFFFLL);
			if (v >= 0) {
				carry = v >> 32;
			} else {
				carry = -(((-v) + 0xFFFFFFFFLL) >> 32);
			}
		}
		// carry is ignored beyond 512 bits (should be small and folded next round)
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) r->v[i] = tmp[i];

	// Final conditional subtraction if r >= p
	if (fe_cmp(r, SM2_P) >= 0) {
		sm2_fe reduced;
		fe_sub_p(&reduced, r);
		fe_copy(r, &reduced);
	}
}

__device__ __forceinline__ void fe_mul_mod_p(sm2_fe* r, const sm2_fe* a, const sm2_fe* b) {
	uint32_t prod[16];
	fe_mul_raw(a, b, prod);
	fe_reduce_p(prod, r);
}

__device__ __forceinline__ void fe_sqr_mod_p(sm2_fe* r, const sm2_fe* a) {
	fe_mul_mod_p(r, a, a);
}

// ---- Inversion (a^(p-2) mod p) ----
__device__ __forceinline__ void fe_inv_mod_p(sm2_fe* r, const sm2_fe* a) {
	// p-2 = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFD
	const uint32_t exp[8] = {
		0xFFFFFFFD, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
		0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
	};

	sm2_fe result;
	sm2_fe base;
	fe_set_u32(&result, 1);
	fe_copy(&base, a);

	for (int limb = 7; limb >= 0; limb--) {
		for (int bit = 31; bit >= 0; bit--) {
			fe_sqr_mod_p(&result, &result);
			uint32_t bitval = (exp[limb] >> bit) & 1U;
			if (bitval) {
				fe_mul_mod_p(&result, &result, &base);
			}
		}
	}
	fe_copy(r, &result);
}

// ---- Affine point operations ----
typedef struct {
	sm2_fe x;
	sm2_fe y;
	int infinity;
} sm2_point;

__device__ __forceinline__ void point_copy(sm2_point* r, const sm2_point* p) {
	fe_copy(&r->x, &p->x);
	fe_copy(&r->y, &p->y);
	r->infinity = p->infinity;
}

__device__ __forceinline__ void point_set_generator(sm2_point* r) {
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		r->x.v[i] = SM2_GX[i];
		r->y.v[i] = SM2_GY[i];
	}
	r->infinity = 0;
}

__device__ __forceinline__ void point_set_infinity(sm2_point* r) {
	fe_set_u32(&r->x, 0);
	fe_set_u32(&r->y, 0);
	r->infinity = 1;
}

__device__ __forceinline__ void point_double(sm2_point* r, const sm2_point* p) {
	if (p->infinity || fe_is_zero(&p->y)) {
		point_set_infinity(r);
		return;
	}

	sm2_fe t1, t2, t3, lambda;
	fe_sqr_mod_p(&t1, &p->x);           // x^2
	fe_add_mod_p(&t2, &t1, &t1);        // 2*x^2
	fe_add_mod_p(&t1, &t2, &t1);        // 3*x^2

	sm2_fe a;
	#pragma unroll
	for (int i = 0; i < 8; i++) a.v[i] = SM2_A[i];
	fe_add_mod_p(&t1, &t1, &a);         // 3*x^2 + a

	fe_add_mod_p(&t2, &p->y, &p->y);    // 2*y
	fe_inv_mod_p(&t2, &t2);             // (2*y)^-1
	fe_mul_mod_p(&lambda, &t1, &t2);    // lambda

	fe_sqr_mod_p(&t3, &lambda);         // lambda^2
	fe_sub_mod_p(&t3, &t3, &p->x);      // lambda^2 - x
	fe_sub_mod_p(&t3, &t3, &p->x);      // lambda^2 - 2x

	sm2_fe x3;
	fe_copy(&x3, &t3);

	fe_sub_mod_p(&t1, &p->x, &x3);      // x - x3
	fe_mul_mod_p(&t1, &lambda, &t1);    // lambda*(x - x3)
	fe_sub_mod_p(&t1, &t1, &p->y);      // y3

	fe_copy(&r->x, &x3);
	fe_copy(&r->y, &t1);
	r->infinity = 0;
}

__device__ __forceinline__ void point_add(sm2_point* r, const sm2_point* p, const sm2_point* q) {
	if (p->infinity) { point_copy(r, q); return; }
	if (q->infinity) { point_copy(r, p); return; }

	if (fe_eq(&p->x, &q->x)) {
		if (fe_eq(&p->y, &q->y)) {
			point_double(r, p);
		} else {
			point_set_infinity(r);
		}
		return;
	}

	sm2_fe t1, t2, lambda;
	fe_sub_mod_p(&t1, &q->y, &p->y);    // y2 - y1
	fe_sub_mod_p(&t2, &q->x, &p->x);    // x2 - x1
	fe_inv_mod_p(&t2, &t2);             // (x2 - x1)^-1
	fe_mul_mod_p(&lambda, &t1, &t2);    // lambda

	sm2_fe x3, y3;
	fe_sqr_mod_p(&x3, &lambda);         // lambda^2
	fe_sub_mod_p(&x3, &x3, &p->x);
	fe_sub_mod_p(&x3, &x3, &q->x);      // x3 = lambda^2 - x1 - x2

	fe_sub_mod_p(&y3, &p->x, &x3);      // x1 - x3
	fe_mul_mod_p(&y3, &lambda, &y3);    // lambda*(x1 - x3)
	fe_sub_mod_p(&y3, &y3, &p->y);      // y3

	fe_copy(&r->x, &x3);
	fe_copy(&r->y, &y3);
	r->infinity = 0;
}

__device__ __forceinline__ void point_scalar_mul(sm2_point* r, const sm2_point* p, const uint32_t k[8]) {
	sm2_point acc;
	point_set_infinity(&acc);
	sm2_point base;
	point_copy(&base, p);

	for (int limb = 7; limb >= 0; limb--) {
		uint32_t w = k[limb];
		for (int bit = 31; bit >= 0; bit--) {
			point_double(&acc, &acc);
			if ((w >> bit) & 1U) {
				sm2_point tmp;
				point_add(&tmp, &acc, &base);
				point_copy(&acc, &tmp);
			}
		}
	}
	point_copy(r, &acc);
}

// TODO: Implement point ops for full SM2
