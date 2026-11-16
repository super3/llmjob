// tbench.cu (goal-nvk26): what operand toggling costs the tensor unit, in watts at a locked clock. Nothing but
// m16n8k32 u8 mma in the rotation fold's order (6 m16 tiles x 8 n8 tiles, 8 in a row share A), 256 threads a
// block, one block a multiprocessor. The code is the same in every mode; only the fragments' values change, and
// they are fixed in registers for the whole run, so what differs is how alike consecutive operands are.
//
//   tbench <mode> <secs>
//
// Values are drawn as the fold's operands are: 63 + d0 - d1, d0 and d1 uniform in 0..63 (0..126, peak at 63).
// Modes (A side the first digit, B side the second; e.g. 0 0 = all random):
//   B: 0 random; 1 all eight n8 fragments the same (consecutive mma share B; inside one, columns and k vary);
//      2 inside a fragment all eight columns the same (lanes with the same l & 3 hold the same words);
//      3 inside a fragment the k the same (every byte of a column one value); 4 constant 63; 5 random but the
//      two warps of a scheduler hold the same fragments; 6 fragment n+1 = fragment n with one column changed;
//      7 uniform 0..63 (a dense noise factor plus 32, 6 bits); 8 uniform 48..79.
//   A: 0 random; 1 all six m16 fragments the same; 2 inside a fragment rows the same; 3 k the same; 4 constant 63;
//      5 the two warps of a scheduler the same.
// Prints cycles an mma on a scheduler (16 is the whole pipe) and runs as long as asked.
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

__global__ void __launch_bounds__(256, 1) bench(uint32_t turns, const uint32_t *frag, unsigned long long *cyc, int32_t *sink) {
  int32_t acc[6][8][4];
  uint32_t af[6][4], bf[8][2];
  const uint32_t t = threadIdx.x;
  const uint32_t *f = frag + (size_t)t * 40u;
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int i = 0; i < 4; i++) af[m][i] = f[m * 4 + i];
#pragma unroll
  for (int n = 0; n < 8; n++)
#pragma unroll
    for (int i = 0; i < 2; i++) bf[n][i] = f[24 + n * 2 + i];
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
static uint32_t word(std::mt19937 &g) { return val(g) | val(g) << 8 | val(g) << 16 | (uint32_t)val(g) << 24; }

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "tbench <A mode> <B mode> <secs>\n");
    return 2;
  }
  const int am = atoi(argv[1]), bm = atoi(argv[2]);
  const double secs = atof(argv[3]);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int blocks = prop.multiProcessorCount;
  std::mt19937 g(777);
  // frag[thread][40]: A's 6 x 4 words, B's 8 x 2. Draw per warp: lane l, g = l >> 2, q = l & 3.
  // A (row): regs 0,1,2,3 = rows g, g+8, g, g+8 at k 4q..4q+3 (0,1) and 16+4q.. (2,3).
  // B (col): regs 0,1 = column g at k 4q.. and 16+4q..
  std::vector<uint32_t> F((size_t)256 * 40);
  for (int w = 0; w < 8; w++) {
    const int src = (am == 5 || bm == 5) ? (w & 3) : w;   // the two warps of scheduler w & 3 share data in mode 5
    std::mt19937 gw(1000 + src * 17);
    uint32_t A[6][32][4], B[8][32][2];
    for (int m = 0; m < 6; m++)
      for (int l = 0; l < 32; l++)
        for (int i = 0; i < 4; i++) A[m][l][i] = word(gw);
    for (int n = 0; n < 8; n++)
      for (int l = 0; l < 32; l++)
        for (int i = 0; i < 2; i++) B[n][l][i] = word(gw);
    if (am == 5 && bm != 5) {   // only A shared: give B the warp's own draw
      std::mt19937 gb(5000 + w * 31);
      for (int n = 0; n < 8; n++)
        for (int l = 0; l < 32; l++)
          for (int i = 0; i < 2; i++) B[n][l][i] = word(gb);
    }
    if (bm == 5 && am != 5) {
      std::mt19937 ga(9000 + w * 31);
      for (int m = 0; m < 6; m++)
        for (int l = 0; l < 32; l++)
          for (int i = 0; i < 4; i++) A[m][l][i] = word(ga);
    }
    for (int m = 0; m < 6; m++)
      for (int l = 0; l < 32; l++)
        for (int i = 0; i < 4; i++) {
          uint32_t &x = A[m][l][i];
          if (am == 1) x = A[0][l][i];
          if (am == 2) x = A[m][l & 3][i & 2];                       // every row as row 0 (g 0, the first half)
          if (am == 3) x = (A[m][l & ~3][i & 1] & 0xffu) * 0x01010101u;   // one value a row: k 0's byte
          if (am == 4) x = 0x3f3f3f3fu;
        }
    for (int n = 0; n < 8; n++)
      for (int l = 0; l < 32; l++)
        for (int i = 0; i < 2; i++) {
          uint32_t &x = B[n][l][i];
          if (bm == 1) x = B[0][l][i];
          if (bm == 2) x = B[n][l & 3][i];                           // every column as column 0
          if (bm == 3) x = (B[n][l & ~3][0] & 0xffu) * 0x01010101u;     // one value a column
          if (bm == 4) x = 0x3f3f3f3fu;
          if (bm == 7) x = gw() & 0x3f3f3f3fu;                          // uniform 0..63 (a dense factor + 32)
          if (bm == 8) x = (gw() & 0x1f1f1f1fu) + 0x30303030u;           // uniform 48..79 (half the spread)
        }
    if (bm == 6)
      for (int n = 1; n < 8; n++)
        for (int l = 0; l < 32; l++)
          for (int i = 0; i < 2; i++)
            if ((l >> 2) != n - 1) B[n][l][i] = B[n - 1][l][i];
    for (int l = 0; l < 32; l++) {
      uint32_t *f = &F[(size_t)(w * 32 + l) * 40];
      for (int m = 0; m < 6; m++)
        for (int i = 0; i < 4; i++) f[m * 4 + i] = A[m][l][i];
      for (int n = 0; n < 8; n++)
        for (int i = 0; i < 2; i++) f[24 + n * 2 + i] = B[n][l][i];
    }
  }
  uint32_t *dF;
  unsigned long long *dC;
  int32_t *dS;
  cudaMalloc(&dF, F.size() * 4);
  cudaMalloc(&dC, blocks * 8);
  cudaMalloc(&dS, (size_t)blocks * 256 * 4);
  cudaMemcpy(dF, F.data(), F.size() * 4, cudaMemcpyHostToDevice);
  const uint32_t turns = 20000;
  bench<<<blocks, 256>>>(turns, dF, dC, dS);
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  double cycSum = 0;
  long n = 0;
  std::vector<unsigned long long> hc(blocks);
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
    bench<<<blocks, 256>>>(turns, dF, dC, dS);
    cudaMemcpy(hc.data(), dC, blocks * 8, cudaMemcpyDeviceToHost);
    for (int b = 0; b < blocks; b++) cycSum += (double)hc[b] / (turns * 48.0 * 2.0);   // two warps a scheduler
    n += blocks;
  }
  const cudaError_t e = cudaGetLastError();
  printf("tbench A%d B%d: %.3f cycles an mma a scheduler (16 = full pipe), %ld launches%s%s\n", am, bm, n ? cycSum / n : 0.0,
         n / blocks, e ? " ERROR " : "", e ? cudaGetErrorString(e) : "");
  return e ? 1 : 0;
}
