// Stand-in for <cuda.h> in the HIP-CPU build, which runs the AMD kernels as plain C++ on
// the CPU to check them without a GPU (see build-amd.sh).
//
// Like ../gpu/cuda.h, plus the device functions HIP-CPU does not have. Each one matches
// what the CUDA and ROCm headers define.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#define __grid_constant__
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
typedef struct alignas(64) CUtensorMap_st {
  uint64_t opaque[16];
} CUtensorMap;

// Byte i of the result is byte ((s >> 4i) & 7) of the 8 bytes {y:x}, x the low four.
static inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
  const uint64_t v = ((uint64_t)y << 32) | x;
  uint32_t r = 0;
  for (int i = 0; i < 4; i++) r |= (uint32_t)((v >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
  return r;
}
static inline uint32_t __umulhi(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * b) >> 32);
}
