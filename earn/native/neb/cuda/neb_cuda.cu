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
// Under 8% of nonces being candidates, each thread keeps its own: 20.8 G nonces/s against 20.0 on the recorded
// Kepler model (4.1%), where the recorded TESS model (25%) does 11.7 that way against 17.5.
#ifndef NB_OWN_BELOW
#define NB_OWN_BELOW 80
#endif
#define NEB_DEV __device__ __forceinline__
#define NEB_HD __host__ __device__ __forceinline__
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
// -DNEB_PRMT=1: rotations by whole bytes as the byte-permute instruction, to time against shifts.
// Device code only: the host has no such call, and takes the shifts.
#if defined(NEB_PRMT) && NEB_PRMT && defined(__CUDA_ARCH__)
#define NEB_ROT16(x) __byte_perm((x), 0u, 0x1032u)
#define NEB_ROTR8(x) __byte_perm((x), 0u, 0x0321u)
#define NEB_ROTL8(x) __byte_perm((x), 0u, 0x2103u)
#endif

#include "../neb_host.inc"
