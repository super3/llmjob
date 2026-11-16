// The few runtime calls the CUDA and HIP host files share code through
// (pearl_host_common.inc): gpu* names mapped to cuda* or hip*.
//
// pearl_host_hip.cpp defines PEARL_GPU_HIP before including this; pearl_host.cu
// does not. HIP's CPU build (HIP-CPU) has no <<<>>> launch, so a launch goes
// through PEARL_GPU_LAUNCH, which is hipLaunchKernelGGL there.

#ifndef PEARL_GPU_API_H
#define PEARL_GPU_API_H

#if defined(PEARL_GPU_HIP)
#include <hip/hip_runtime.h>

typedef hipStream_t gpuStream_t;
typedef hipError_t gpuError_t;
#define gpuSuccess hipSuccess
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuGetLastError hipGetLastError
#define gpuFreeHost hipHostFree
#define gpuHostAlloc(p, n) hipHostMalloc((p), (n), hipHostMallocDefault)
#define PEARL_GPU_LAUNCH(kernel, grid, block, stream, ...) \
  hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), 0, stream, __VA_ARGS__)

#else
#include <cuda_runtime.h>

typedef cudaStream_t gpuStream_t;
typedef cudaError_t gpuError_t;
#define gpuSuccess cudaSuccess
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuFreeHost cudaFreeHost
#define gpuHostAlloc(p, n) cudaHostAlloc((p), (n), cudaHostAllocDefault)
#define PEARL_GPU_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#endif

#endif  // PEARL_GPU_API_H
