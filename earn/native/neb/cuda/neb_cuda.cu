// The NEB core on NVIDIA cards: CUDA's names for what neb_host.inc and neb_kernel.h need. All of the
// core is in those two files; the AMD core (amd/neb_hip.cpp) is the same with HIP's names.
#include <cuda_runtime.h>

#define G(x) cuda##x
#define NB_VENDOR "nvidia"
#define NB_PROP cudaDeviceProp
// Turing and newer: the cards the build makes code for.
#define NB_MIN_OK(p) ((p).major * 10 + (p).minor >= 75)
// Timed with tools/nebbench.js on an RTX 4090 (28 Nov 2026): 2^22 threads a launch, 20.5 G nonces/s on the
// recorded Kepler model, against 19.8 with 2^20; one nonce a thread between ticket steps (two: 19.0).
#ifndef NB_SCREEN_LOG2
#define NB_SCREEN_LOG2 22
#endif
// The triple build keeps a tile's rows a in shared memory (neb_host.inc, nbk_triple).
#ifndef NB_TRIPLE_NEAR
#define NB_TRIPLE_NEAR 1
#endif
// And each entry's errors on the thread's stack (neb_kernel.h, nebk_triple).
#ifndef NEBK_ERRS_ON_STACK
#define NEBK_ERRS_ON_STACK 1
#endif
#define NEB_DEV __device__ __forceinline__
#define NEB_HD __host__ __device__ __forceinline__
// A block's shared memory whose size the launch gives.
#define NB_SHARED_AT_LAUNCH(type, name) extern __shared__ type name[]
#define NEB_ATOMIC_INC(p) atomicAdd((p), 1u)
#define NB_ATOMIC_ADD64(p, x) atomicAdd((p), (x))
#define NEB_ATOMIC_ADD(p, x) atomicAdd((p), (x))
#define NEB_MULHI(u, n) __umulhi((u), (n))
#define NEB_CTZ(x) ((unsigned int)(__ffs((int)(x)) - 1))
#define NEB_POPC(x) ((unsigned int)__popc(x))
// NEB_MAD: which hash's adds run as multiply-adds by a 1 the compiler cannot see. 1: BLAKE3's; 2: ChaCha8's;
// 3: both. An SM issues about two adder instructions a clock (add, xor, rotate all count) and about two
// multiply-adds beside them (earn/tuning/quantus-benchmark.md, ubench), and these hashes are a third adds.
// Timed on an RTX 4090: BLAKE3's adds this way, 20.5 G nonces/s against 20.0; ChaCha8's too, no change.
#ifndef NEB_MAD
#define NEB_MAD 1
#endif
#if (NEB_MAD & 1) && defined(__CUDA_ARCH__)
#define NEB_ADDB(a, b, one) ((b) * (one) + (a))
#endif
#if (NEB_MAD & 2) && defined(__CUDA_ARCH__)
#define NEB_ADDC(a, b, one) ((b) * (one) + (a))
#endif
// Tried and not taken (RTX 4090, 28 Nov 2026): rotations as multiplies (x * 2^n plus the high word of that
// product, the 2^n from `one`), to move work from the adder to the multiplier: 22.65 G nonces/s against 23.4
// with ChaCha8's rotations by 16 and 8 that way, and slower the more rotations went. The high-word multiply
// costs more than the adder slot it frees. Rotations as the byte-permute instruction: no change.

#include "../neb_host.inc"
