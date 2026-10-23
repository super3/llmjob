// Stand-in for <cuda.h> in the AMD build (ROCm's hip-clang).
//
// The AMD build compiles the first part of ../../src/pearl_kernel.cu unchanged (see
// build-amd.sh). That part includes <cuda.h>, <cuda_runtime.h> and <mma.h> and uses a few
// CUDA spellings. HIP already provides the device functions it calls (__byte_perm,
// __umulhi, __clz, __syncthreads), so this only has to supply the HIP runtime and the one
// CUDA type pearl_tensor_map.h names. No AMD code uses the tensor map.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#define __grid_constant__
typedef struct alignas(64) CUtensorMap_st {
  uint64_t opaque[16];
} CUtensorMap;
