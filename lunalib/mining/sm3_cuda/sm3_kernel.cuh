// sm3_kernel.cuh
// SM3 CUDA kernel header

#include "sm3_kernel_constants.h"
#include "sm3_kernel_utils.cu"

extern "C" __global__ void sm3_hash(const unsigned char* input, unsigned char* output, int input_len);
