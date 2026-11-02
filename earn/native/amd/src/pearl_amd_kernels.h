// Every kernel the AMD core launches, in one place.
//
// The portable kernels -- BLAKE3, the commitment trees, the seeds, the noise, materialise
// and restamp -- are pearl_kernel.cu's own, compiled unchanged: build-amd.sh copies that
// file up to the start of the CUDA folds (pearl_kernel_portable.cu, in the build folder)
// and fails if the marker it cuts at has moved. compat/ stands in for the CUDA headers.
// The folds and the transcript hash are AMD's own (pearl_amd_fold.h).
#pragma once

#include <hip/hip_runtime.h>

// pearl_config.h marks its helpers __host__ __device__ only under nvcc (__CUDACC__), which
// hip-clang does not define. The pragma gives them both sides here without touching the
// header. HIP-CPU runs everything on the host and needs nothing.
#if !defined(__HIP_CPU_RT__)
#pragma clang force_cuda_host_device begin
#endif
#include "pearl_config.h"
#if !defined(__HIP_CPU_RT__)
#pragma clang force_cuda_host_device end
#endif

#include "pearl_kernel_portable.cu"
#include "pearl_amd_fold.h"
