// mmapeak -- pure mma.m16n8k32.s8 from registers, zero memory traffic.
// Establishes this card's tensor ceiling (T-MAC/s) and the power it draws getting there.
// Per the 4090 tuning log: 346 T-MAC/s @ 156 W on Ada.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ACC 16            // accumulators per thread
#define ITERS 4096

__global__ __launch_bounds__(512) void mmapeak(int *sink, int iters) {
  extern __shared__ char occupancy_ballast[];
  if (threadIdx.x == 1023 && iters < 0) occupancy_ballast[0] = 1;
  int a[4], b[2];
  int c[ACC][4];
#pragma unroll
  for (int i = 0; i < 4; i++) a[i] = 0x01010101 + threadIdx.x;
#pragma unroll
  for (int i = 0; i < 2; i++) b[i] = 0x01010101 + threadIdx.x;
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0;

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < ACC; i++) {
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) s += c[i][j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;   // real store: nothing is dead
}

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 512;
  int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  int blocks = (argc > 2) ? atoi(argv[2]) : p.multiProcessorCount;
  int *sink; cudaMalloc(&sink, (size_t)blocks * threads * 4);
  cudaMemset(sink, 0, (size_t)blocks * threads * 4);
  const size_t SH = 96 * 1024;
  cudaFuncSetAttribute(mmapeak, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SH);
  mmapeak<<<blocks, threads, SH>>>(sink, 8);  // warmup
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { printf("threads=%-5d LAUNCH FAILED: %s\n", threads, cudaGetErrorString(le)); return 1; }
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int iters = (argc > 3) ? atoi(argv[3]) : ITERS;
  mmapeak<<<blocks, threads, SH>>>(sink, iters);
  cudaEventRecord(e1);
  le = cudaGetLastError();
  if (le != cudaSuccess) { printf("threads=%-5d LAUNCH FAILED: %s\n", threads, cudaGetErrorString(le)); return 1; }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  // MACs = warps * ACC * ITERS * (16*8*32)
  double warps = (double)blocks * (threads / 32);
  double macs = warps * ACC * (double)iters * 16.0 * 8.0 * 32.0;
  printf("threads/block=%-5d blocks=%-4d  %.1f T-MAC/s   (%.2f ms)\n",
         threads, blocks, macs / (ms * 1e-3) / 1e12, ms);
  return 0;
}
