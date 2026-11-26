// trhome.cu (goal/r1-rtx30, a measuring probe, nothing ships): where a fold's transcript word can live on sm_86
// without costing clock. One block of 256 threads an SM with 96 KB of shared memory (the rotation fold's form), a
// loop of int8 IMMA on registers that change every step (so the pipe and power look like a fold), and at every
// "chunk" (48 IMMA a warp) the thread's transcript word c & 15 is loaded, turned and XORed with the sums, and
// stored back, in one of several homes:
//   0 none (the word stays in a register: the floor)
//   1 the kernel's frame (local memory, LDL/STL), as B drawn once and the 96 x 512 loop keep it
//   2 global memory, [word][thread] per block, plain loads and stores
//   3 global memory, [word][thread], loads and stores with an L2 evict_last policy (createpolicy)
//   4 shared memory, [thread][word], as the shipped sm_86 loop keeps it
//   5 global memory, [word][thread], .L1::no_allocate stores and evict_last policy, loads .L1::evict_last
// Usage: trhome <home> <seconds>   prints one JSON line: home, chunks a second, TH/s-equivalent.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef HOME
#define HOME 0
#endif

extern "C" __global__ __launch_bounds__(256, 1) void k(uint32_t *gtr, uint32_t *out, uint32_t chunks, uint32_t seed) {
  extern __shared__ __align__(16) uint32_t sm[];
  const uint32_t tid = threadIdx.x;
  uint32_t a0 = seed * 0x9E3779B1u + tid, a1 = a0 * 0x85EBCA6Bu, a2 = a1 ^ 0xC2B2AE35u, a3 = a2 * 3u;
  uint32_t b0 = a0 ^ 0x27D4EB2Fu, b1 = b0 * 0x165667B1u;
  int32_t acc[6][8][4];
#pragma unroll
  for (int i = 0; i < 6; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int q = 0; q < 4; q++) acc[i][j][q] = 0;
  uint32_t *g = gtr + (size_t)blockIdx.x * 4096u + tid;          // [word][thread]
  volatile uint32_t trl[16];
#pragma unroll
  for (int i = 0; i < 16; i++) trl[i] = 0;
  for (uint32_t i = tid; i < 4096u; i += 256u) sm[i] = 0;
  uint32_t treg = 0;
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  __syncthreads();
  for (uint32_t c = 0; c < chunks; c++) {
#pragma unroll
    for (int s = 0; s < 1; s++) {
#pragma unroll
      for (int i = 0; i < 6; i++) {
        // fresh operand bits every step, as gathers make them
        a0 = a0 * 1664525u + 1013904223u;
        a1 ^= a0 >> 7;
#pragma unroll
        for (int j = 0; j < 8; j++)
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                       "{%0,%1,%2,%3};"
                       : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                       : "r"(a0), "r"(a1), "r"(a2 ^ (uint32_t)j), "r"(a3), "r"(b0 + (uint32_t)i), "r"(b1));
      }
    }
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 6; i++)
#pragma unroll
      for (int j = 0; j < 8; j++)
        x ^= (uint32_t)acc[i][j][0] ^ (uint32_t)acc[i][j][1] ^ (uint32_t)acc[i][j][2] ^ (uint32_t)acc[i][j][3];
    const uint32_t wd = c & 15u;
#if HOME == 0
    treg = __funnelshift_l(treg, treg, 13u) ^ x;
#elif HOME == 1
    trl[wd] = __funnelshift_l(trl[wd], trl[wd], 13u) ^ x;
#elif HOME == 2
    {
      volatile uint32_t *p = g + wd * 256u;
      uint32_t w = *p;
      *p = __funnelshift_l(w, w, 13u) ^ x;
    }
#elif HOME == 3
    {
      uint32_t *p = g + wd * 256u, w;
      asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(w) : "l"(p), "l"(pol) : "memory");
      w = __funnelshift_l(w, w, 13u) ^ x;
      asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;" ::"l"(p), "r"(w), "l"(pol) : "memory");
    }
#elif HOME == 4
    {
      volatile uint32_t *p = sm + tid * 16u + wd;
      uint32_t w = *p;
      *p = __funnelshift_l(w, w, 13u) ^ x;
    }
#elif HOME == 5
    {
      uint32_t *p = g + wd * 256u, w;
      asm volatile("ld.global.L1::evict_last.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(w) : "l"(p), "l"(pol) : "memory");
      w = __funnelshift_l(w, w, 13u) ^ x;
      asm volatile("st.global.L1::no_allocate.L2::cache_hint.u32 [%0], %1, %2;" ::"l"(p), "r"(w), "l"(pol) : "memory");
    }
#endif
  }
  uint32_t r = treg;
#pragma unroll
  for (int i = 0; i < 16; i++) r ^= trl[i];
  r ^= sm[tid * 16u];
  out[blockIdx.x * 256u + tid] = r;
}

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 60.0;
  int dev = 0, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const size_t shm = 96u * 1024u;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
  uint32_t *gtr, *out;
  cudaMalloc(&gtr, (size_t)sms * 4096u * 4u);
  cudaMemset(gtr, 0, (size_t)sms * 4096u * 4u);
  cudaMalloc(&out, (size_t)sms * 256u * 4u);
  const uint32_t chunks = 4096;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  k<<<sms, 256, shm>>>(gtr, out, chunks, 1);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));
    return 1;
  }
  double t = 0;
  uint64_t launches = 0;
  cudaEventRecord(e0);
  while (t < secs) {
    for (int i = 0; i < 20; i++) k<<<sms, 256, shm>>>(gtr, out, chunks, (uint32_t)(launches + i));
    launches += 20;
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    t = ms / 1000.0;
  }
  // MACs: a chunk is 48 IMMA a warp of 16 x 8 x 32, 8 warps a block.
  const double macs = (double)launches * sms * chunks * 8.0 * 48.0 * 16 * 8 * 32;
  printf("{\"home\":%d,\"secs\":%.1f,\"ths\":%.3f}\n", HOME, t, macs / t / 1e12);
  return 0;
}
