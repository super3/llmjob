// mmabench.cu: how much of the tensor pipe a stream of nothing but int8 mma (m16n8k32) holds on this card, by the
// order of the mma. A measuring tool for sm_120 (goal-nv50b), not the miner.
//
//   mmabench [secs]
//
// Every warp keeps 48 sums (6 m16 tiles x 8 n8 tiles, as the rotation fold) and folds fixed A and B fragments
// into them, 48 mma a turn. Orders: 0 tile by tile (8 mma in a row share A, as pearl_rot_c.inc writes it), 1 n8
// tile by n8 tile (6 in a row share B), 2 neither (each mma takes another A and another B than the one before).
// One block of 256 threads (two warps a scheduler) or 128 (one) a multiprocessor. Pipe: 16 cycles an mma on a
// scheduler is all of it.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

template <int ORDER>
__global__ void __launch_bounds__(256, 1) bench(uint32_t turns, unsigned long long *cyc, uint32_t *sink) {
  int32_t acc[6][8][4];
  uint32_t af[6][4], bf[8][2];
  const uint32_t t = threadIdx.x;
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int i = 0; i < 4; i++) af[m][i] = (t * 0x9e3779b9u + m * 77u + i) & 0x3f3f3f3fu;
#pragma unroll
  for (int n = 0; n < 8; n++)
#pragma unroll
    for (int i = 0; i < 2; i++) bf[n][i] = (t * 0x85ebca6bu + n * 31u + i) & 0x3f3f3f3fu;
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][n][i] = 0;
  __syncthreads();
  const unsigned long long c0 = clock64();
  for (uint32_t r = 0; r < turns; r++) {
#pragma unroll
    for (int s = 0; s < 48; s++) {
      int m, n;
      if (ORDER == 0) { m = s / 8; n = s % 8; }
      else if (ORDER == 1) { n = s / 6; m = s % 6; }
      else { m = s % 6; n = (s + s / 6) % 8; }
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+r"(acc[m][n][0]), "+r"(acc[m][n][1]), "+r"(acc[m][n][2]), "+r"(acc[m][n][3])
          : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]), "r"(bf[n][0]), "r"(bf[n][1]));
    }
  }
  const unsigned long long c1 = clock64();
  uint32_t x = 0;
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) x ^= (uint32_t)acc[m][n][i];
  if ((t & 31u) == 0) cyc[blockIdx.x * 8u + (t >> 5)] = c1 - c0;
  if (x == 0x12345678u) sink[0] = x;
}

template <int ORDER>
static void run(int threads, double secs, int sms) {
  unsigned long long *dc;
  uint32_t *ds;
  cudaMalloc(&dc, sms * 8 * sizeof(unsigned long long));
  cudaMalloc(&ds, 4);
  cudaMemset(dc, 0, sms * 8 * sizeof(unsigned long long));
  // Shared memory so that one block fits a multiprocessor.
  const int smem = 80 * 1024;
  cudaFuncSetAttribute(bench<ORDER>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  uint32_t turns = 20000;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms = 0, total = 0;
  int n = 0;
  do {
    cudaEventRecord(e0);
    bench<ORDER><<<sms, threads, smem>>>(turns, dc, ds);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
    n++;
  } while (total < secs * 1000.0);
  cudaError_t err = cudaGetLastError();
  unsigned long long h[8 * 256];
  cudaMemcpy(h, dc, sms * 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  const int warps = threads / 32;
  double sum = 0;
  int cnt = 0;
  for (int b = 0; b < sms; b++) {
    unsigned long long mx = 0;
    for (int w = 0; w < warps; w++) mx = h[b * 8 + w] > mx ? h[b * 8 + w] : mx;
    if (!mx) continue;
    // mma a scheduler: warps / 4 warps each 48 x turns.
    sum += 16.0 * (warps / 4.0) * 48.0 * turns / (double)mx;
    cnt++;
  }
  printf("mmabench order %d threads %d pipe %.2fpc launches %d ms %.1f %s\n", ORDER, threads,
         cnt ? 100.0 * sum / cnt : 0.0, n, total / n, err == cudaSuccess ? "ok" : cudaGetErrorString(err));
  cudaFree(dc);
  cudaFree(ds);
}

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 10.0;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (sms > 256) sms = 256;
  for (int th = 256; th >= 128; th -= 128) {
    run<0>(th, secs, sms);
    run<1>(th, secs, sms);
    run<2>(th, secs, sms);
  }
  return 0;
}
