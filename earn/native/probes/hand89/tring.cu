// tring.cu (goal-nvk28, TIMING ONLY): does B streamed through shared memory toggle the tensor unit as B drawn on
// the chip does? 48 m16n8k32 u8 mma a warp a k-step (6 m16 x 8 n8, as the fold), A fixed in registers, B's 16
// fragment words loaded each k-step with four ldmatrix.x4 (no transpose) from shared memory. 256 threads, one
// block a multiprocessor.
//
//   tring <mode> <copies> <secs> [stall]
//
// mode 0: B from a 16 KB shared table, a different 2 KB window each k-step (no L2 traffic); 1: B from a ring that
// cp.async fills from a 64 MB global buffer, two k-steps ahead, <copies> 16-byte copies a lane a k-step (4 = the
// warp's whole 2 KB; 2 = half, the other half from the table); 2: as 1 but the copies go to a junk area and B is
// read from the table (the fill priced on its own). Data: env TR_DATA=0 random (63 + d0 - d1), 1 constant 63.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma(int32_t *d, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldm(uint32_t *r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void cpa(uint32_t s, const void *g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}

template <int MODE, int NC>
__global__ void __launch_bounds__(256, 1) bench(uint32_t ksteps, const uint4 *frag, const uint4 *big, size_t bigN,
                                                unsigned long long *cyc, int32_t *sink) {
  extern __shared__ __align__(16) uint4 sm[];
  // [0, 16 KB): the table. [16 KB, 48 KB): rings, 4 KB a warp (two k-steps of 2 KB). [48 KB, 80 KB): junk.
  const uint32_t t = threadIdx.x, w = t >> 5, l = t & 31u;
  for (uint32_t i = t; i < 1024u; i += 256u) sm[i] = frag[1024u + i];
  for (uint32_t i = t; i < 2048u; i += 256u) sm[1024u + i] = frag[1024u + (i & 1023u)];
  uint32_t af[6][4];
  const uint4 *f = frag + t * 6u;
#pragma unroll
  for (int m = 0; m < 6; m++) {
    const uint4 v = f[m % 4];
    af[m][0] = v.x; af[m][1] = v.y; af[m][2] = v.z; af[m][3] = v.w;
  }
  int32_t acc[6][8][4];
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][n][i] = 0;
  __syncthreads();
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t ring = sb + 16384u + w * 4096u + l * 16u, junk = sb + 49152u + w * 4096u + l * 16u;
  // The global stream: a warp's own 512 B x NC a k-step, all blocks on the same bytes (as blocks on one column
  // block of B' would be).
  const size_t per = (size_t)8u * 32u * NC;   // uint4 a k-step for the block
  size_t gpos = (size_t)w * 32u * NC + l;
  const unsigned long long c0 = clock64();
  for (uint32_t ks = 0; ks < ksteps; ks++) {
    uint32_t bf[8][2];
    if (MODE >= 1) {
      const uint32_t dst = (MODE == 1 ? ring : junk) + (ks & 1u) * 2048u;
#pragma unroll
      for (int i = 0; i < NC; i++) cpa(dst + 512u * ((i + NC * ((ks >> 1) & 1u)) & 3u), big + gpos + 32u * i);
      asm volatile("cp.async.commit_group;" ::: "memory");
      gpos += per;
      if (gpos + per > bigN) gpos = (size_t)w * 32u * NC + l;
      asm volatile("cp.async.wait_group 1;" ::: "memory");
      __syncwarp();
    }
    const uint32_t src = MODE == 1 ? ring + ((ks + 1u) & 1u) * 2048u : sb + ((ks * 2048u) & 16383u) + l * 16u;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      uint32_t r[4];
      ldm(r, src + 512u * p);
      bf[2 * p][0] = r[0]; bf[2 * p][1] = r[1]; bf[2 * p + 1][0] = r[2]; bf[2 * p + 1][1] = r[3];
    }
#pragma unroll
    for (int m = 0; m < 6; m++)
#pragma unroll
      for (int n = 0; n < 8; n++) mma(acc[m][n], af[m], bf[n]);
  }
  const unsigned long long c1 = clock64();
  int32_t s = 0;
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) s ^= acc[m][n][i];
  sink[blockIdx.x * 256 + t] = s;
  if (t == 0) cyc[blockIdx.x] = c1 - c0;
}

static uint8_t val(std::mt19937 &g) { return (uint8_t)(63 + (int)(g() & 63) - (int)(g() & 63)); }

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "tring <mode> <copies> <secs>\n");
    return 2;
  }
  const int mode = atoi(argv[2]), nc = atoi(argv[3]);
  const double secs = atof(argv[4]);
  const int cst = getenv("TR_DATA") && atoi(getenv("TR_DATA")) == 1;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int blocks = prop.multiProcessorCount;
  std::mt19937 g(777);
  const size_t bigBytes = (size_t)64 << 20;
  std::vector<uint8_t> F(2048 * 16), Bg(bigBytes);
  for (auto &b : F) b = cst ? 63 : val(g);
  for (size_t i = 0; i < 1024 * 16; i++) F[i] = val(g);   // A always random
  for (auto &b : Bg) b = cst ? 63 : val(g);
  uint4 *dF, *dB;
  unsigned long long *dC;
  int32_t *dS;
  cudaMalloc(&dF, F.size());
  cudaMalloc(&dB, Bg.size());
  cudaMalloc(&dC, blocks * 8);
  cudaMalloc(&dS, (size_t)blocks * 256 * 4);
  cudaMemcpy(dF, F.data(), F.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bg.data(), Bg.size(), cudaMemcpyHostToDevice);
  const uint32_t ksteps = 20000;
  const size_t bigN = bigBytes / 16;
  void (*k)(uint32_t, const uint4 *, const uint4 *, size_t, unsigned long long *, int32_t *) = nullptr;
  if (mode == 0) k = bench<0, 1>;
  else if (mode == 1 && nc == 2) k = bench<1, 2>;
  else if (mode == 1 && nc == 4) k = bench<1, 4>;
  else if (mode == 2 && nc == 1) k = bench<2, 1>;
  else if (mode == 2 && nc == 2) k = bench<2, 2>;
  else if (mode == 2 && nc == 4) k = bench<2, 4>;
  if (!k) {
    fprintf(stderr, "no such form\n");
    return 2;
  }
  const int smem = 81920;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k<<<blocks, 256, smem>>>(ksteps, dF, dB, bigN, dC, dS);
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  double cycSum = 0;
  long n = 0;
  std::vector<unsigned long long> hc(blocks);
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
    k<<<blocks, 256, smem>>>(ksteps, dF, dB, bigN, dC, dS);
    cudaMemcpy(hc.data(), dC, blocks * 8, cudaMemcpyDeviceToHost);
    for (int b = 0; b < blocks; b++) cycSum += (double)hc[b] / (ksteps * 48.0 * 2.0);
    n += blocks;
  }
  const cudaError_t e = cudaGetLastError();
  printf("%s tring m%d c%d d%d: pipe %.2f%% %.3f cycles an mma a scheduler, %ld launches%s%s\n", argv[1], mode, nc, cst,
         n ? 1600.0 / (cycSum / n) : 0.0, n ? cycSum / n : 0.0, n / blocks, e ? " ERROR " : "", e ? cudaGetErrorString(e) : "");
  return e ? 1 : 0;
}
