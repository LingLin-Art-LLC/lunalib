// sm2_curve_params.h
// SM2 curve parameters (GM/T 0003.5-2012)

#ifndef SM2_UINT32_T
#define SM2_UINT32_T
typedef unsigned int uint32_t;
#endif

// Limb order: little-endian 32-bit words (v[0] = least significant word)

__device__ __constant__ uint32_t SM2_P[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
};

__device__ __constant__ uint32_t SM2_A[8] = {
	0xFFFFFFFC, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
};

__device__ __constant__ uint32_t SM2_B[8] = {
	0x4D940E93, 0xDDBCBD41, 0x15AB8F92, 0xF39789F5,
	0xCF6509A7, 0x4D5A9E4B, 0x9D9F5E34, 0x28E9FA9E
};

__device__ __constant__ uint32_t SM2_N[8] = {
	0x39D54123, 0x53BBF409, 0x21C6052B, 0x7203DF6B,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE
};

__device__ __constant__ uint32_t SM2_GX[8] = {
	0x334C74C7, 0x715A4589, 0xF2660BE1, 0x8FE30BBF,
	0x6A39C994, 0x5F990446, 0x1F198119, 0x32C4AE2C
};

__device__ __constant__ uint32_t SM2_GY[8] = {
	0x2139F0A0, 0x02DF32E5, 0xC62A4740, 0xD0A9877C,
	0x6B692153, 0x59BDCEE3, 0xF4F6779C, 0xBC3736A2
};
