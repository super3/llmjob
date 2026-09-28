// The tall fold's TMA descriptor (PEARL_TALL_TMA), with its alignment written
// down here instead of taken from cuda.h.
//
// cuda.h declares CUtensorMap as 16 uint64 words behind
//
//   #if defined(__cplusplus) && (__cplusplus >= 201103L)
//       alignas(128)          // 64 in CUDA 12.8
//
// and MSVC reports __cplusplus as 199711L unless it is given /Zc:__cplusplus,
// which our Windows build is not. nvcc's front end copies the host compiler's
// value, so on Windows the guard is false in the device pass too, the struct is
// only 8-byte aligned, and the tall fold's two descriptors landed at parameter
// offsets 0x98 and 0x118 (read from the Windows CI artifacts) where Linux puts
// them at 0x100 and 0x180 (CUDA 13.3) or 0xc0 and 0x140 (12.8). sm_120's
// parameters start at c[0x0][0x380], a multiple of 128, so an offset's alignment
// is the descriptor's. cp.async.bulk.tensor needs the descriptor 64-byte
// aligned, so every Blackwell card on Windows faulted on its first TMA load with
// "CUDA error during search: misaligned address" in v0.5.7 and v0.5.8 (the
// fold shipped in v0.5.7). Building v0.5.8 on Linux against a cuda.h with that
// guard replaced by `#if 0` reproduced both the offsets and the fault on an
// RTX 5090.
//
// alignas is a keyword, not something a header switches on, so MSVC honours it
// whatever __cplusplus says. Every descriptor the fold reads -- the kernel's
// parameters, the host's copies it encodes into, and the launch -- is this type,
// never a bare CUtensorMap. The CI build checks the resulting offsets
// (.github/workflows/native-core.yml, "Check the tensor-map parameter offsets").

#ifndef PEARL_TENSOR_MAP_H
#define PEARL_TENSOR_MAP_H

#include <cuda.h>  // CUtensorMap (types only; the addon links cudart alone)

// 128, what cuda.h asks for from CUDA 13 on, rather than the hardware's 64: it
// is the most any toolkit has asked, and it costs nothing, since the descriptors
// are the fold's last parameters and the host keeps two of them.
struct alignas(128) PearlTensorMap {
  CUtensorMap map;
};

static_assert(alignof(PearlTensorMap) == 128, "a TMA descriptor must be 128-byte aligned");
static_assert(sizeof(PearlTensorMap) == 128,
              "PearlTensorMap must be exactly one CUtensorMap (128 bytes), with no padding");

#endif  // PEARL_TENSOR_MAP_H
