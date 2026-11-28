// The NEB core on NVIDIA cards: CUDA's names for what neb_host.inc and neb_kernel.h need. All of the
// core is in those two files; the AMD core (amd/neb_hip.cpp) is the same with HIP's names.
#include <cuda_runtime.h>

#define G(x) cuda##x
#define NB_VENDOR "nvidia"
#define NB_PROP cudaDeviceProp
// Turing and newer: the cards the build makes code for.
#define NB_MIN_OK(p) ((p).major * 10 + (p).minor >= 75)
#define NEB_DEV __device__ __forceinline__
#define NEB_HD __host__ __device__ __forceinline__
#define NEB_ATOMIC_INC(p) atomicAdd((p), 1u)
#define NB_ATOMIC_ADD64(p, x) atomicAdd((p), (x))
#define NEB_MULHI(u, n) __umulhi((u), (n))

#include "../neb_host.inc"
