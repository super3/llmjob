// The NEB core on AMD cards: HIP's names for what neb_host.inc and neb_kernel.h need. All of the core
// is in those two files; the NVIDIA core (cuda/neb_cuda.cu) is the same with CUDA's names.
#include <hip/hip_runtime.h>

#define G(x) hip##x
#define NB_VENDOR "amd"
#define NB_PROP hipDeviceProp_t
// Every card HIP lists that the build has code for (amd/build.sh's targets).
#define NB_MIN_OK(p) (true)
#define NEB_DEV __device__ inline __attribute__((always_inline))
#define NEB_HD __host__ __device__ inline __attribute__((always_inline))
#define NEB_ATOMIC_INC(p) atomicAdd((p), 1u)

#include "../neb_host.inc"
