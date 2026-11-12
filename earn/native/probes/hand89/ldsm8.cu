// ldsm8.cu (goal-nv50d): which bytes ldmatrix.m16n16.x2.trans.b8 (sm_120a's LDSM.8.MT1616.2) hands each lane, on
// the card. rfold.py's b8 gathers take it that lanes 0-15 address the 16 rows of matrix 0 and lanes 16-31 those
// of matrix 1, and that register 2 m + i of lane l (g = l >> 2, t = l & 3) holds, in byte j, column g + 8 i of
// address row 4 t + j of matrix m. One line out: "ldsm8 as assumed" or the first places that differ.
//   nvcc -arch=sm_120a -o ldsm8 ldsm8.cu
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// Address row r (lane r's address) at 64 r. Two passes: every byte holds its row (what 0), then its column (1).
__global__ void k(uint32_t *out, uint32_t what) {
  __shared__ __align__(16) uint8_t sm[32 * 64];
  const uint32_t l = threadIdx.x;
  for (uint32_t i = l; i < 32u * 64u; i += 32u) {
    const uint32_t r = i / 64u, c = i % 64u;
    sm[i] = (uint8_t)(what ? c : r);
  }
  __syncwarp();
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(sm) + 64u * l;
  uint32_t r0, r1, r2, r3;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
  out[4 * l] = r0;
  out[4 * l + 1] = r1;
  out[4 * l + 2] = r2;
  out[4 * l + 3] = r3;
}

int main() {
  uint32_t *d, h[2][128];
  cudaMalloc(&d, 512);
  for (uint32_t w = 0; w < 2; w++) {
    k<<<1, 32>>>(d, w);
    if (cudaMemcpy(h[w], d, 512, cudaMemcpyDeviceToHost) != cudaSuccess) {
      printf("ldsm8 FAIL %s\n", cudaGetErrorString(cudaGetLastError()));
      return 1;
    }
  }
  int bad = 0;
  char msg[300];
  int n = 0;
  for (uint32_t l = 0; l < 32; l++)
    for (uint32_t reg = 0; reg < 4; reg++)
      for (uint32_t j = 0; j < 4; j++) {
        const uint32_t row = (h[0][4 * l + reg] >> (8 * j)) & 255, col = (h[1][4 * l + reg] >> (8 * j)) & 255;
        const uint32_t g = l >> 2, t = l & 3, m = reg >> 1, i = reg & 1;
        const uint32_t wr = 16 * m + 4 * t + j, wc = g + 8 * i;
        if (row != wr || col != wc) {
          if (bad < 8 && n < 240) n += snprintf(msg + n, sizeof msg - n, " l%u r%u b%u:%u,%u", l, reg, j, row, col);
          bad++;
        }
      }
  // Lanes 0, 5 and 17 in full, as row,column a byte, for the record.
  if (bad) {
    printf("ldsm8 DIFFERS at %d of 512 bytes:%s\n", bad, msg);
    for (uint32_t l : {0u, 5u, 17u}) {
      n = 0;
      for (uint32_t reg = 0; reg < 4; reg++)
        for (uint32_t j = 0; j < 4; j++)
          n += snprintf(msg + n, sizeof msg - n, " %u,%u", (h[0][4 * l + reg] >> (8 * j)) & 255,
                        (h[1][4 * l + reg] >> (8 * j)) & 255);
      printf("ldsm8 lane %u:%s\n", l, msg);
    }
  } else {
    printf("ldsm8 as assumed: 512 of 512 bytes\n");
  }
  return 0;
}
