// What gfx11's v_wmma_i32_16x16x16_iu8 does with each operand byte, in wave64 (-mwavefrontsize64 -DLAY_W=64) and in
// wave32 (the default) (goal-f-rdna3). For gfx1100 or gfx1101. A lane holds 16 bytes of A and of B in both wave
// sizes, and 4 sums in wave64 or 8 in wave32. One WMMA a wave, one probe byte a wave:
//   mode 0  A has one byte set (lane la, byte ja), B all ones: the sums that byte reaches.
//   mode 1  the same for B.
//   mode 2  A byte (lane 16 qa, byte ja) set, and B byte (lane lb, byte jb) = 1 + 16 (lb / 16) + jb: a sum that A
//           byte reaches reads the B byte that meets it, so it says which B byte has the same k and whose copy.
// The host prints, for each lane group (lane / 16) and register g: the A row and B column its sums take, which lane
// groups' copies of those bytes reach them and how many bytes; then the k pairing as each lane group's sums see it.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <set>
#include <vector>

#ifndef LAY_W
#define LAY_W 32
#endif
#define LAY_G (256 / LAY_W)   // sums a lane
#define LAY_Q (LAY_W / 16)    // lane groups

typedef int32_t l11_v4i __attribute__((ext_vector_type(4)));
typedef int32_t l11_v8i __attribute__((ext_vector_type(8)));

__global__ __launch_bounds__(LAY_W) void lay_kernel(int mode, int32_t *out) {
  const int w = (int)blockIdx.x, lane = (int)threadIdx.x;
  uint32_t a[4] = {0u, 0u, 0u, 0u}, b[4] = {0u, 0u, 0u, 0u};
  const int pl = w / 16, pj = w % 16;   // modes 0 and 1: the probe's lane and byte
  if (mode == 0) {
    if (lane == pl) a[pj / 4] = 1u << (8 * (pj % 4));
    b[0] = b[1] = b[2] = b[3] = 0x01010101u;
  } else if (mode == 1) {
    a[0] = a[1] = a[2] = a[3] = 0x01010101u;
    if (lane == pl) b[pj / 4] = 1u << (8 * (pj % 4));
  } else {
    const int qa = w / 16, ja = w % 16;
    if (lane == 16 * qa) a[ja / 4] = 1u << (8 * (ja % 4));
    for (int j = 0; j < 16; j++) b[j / 4] |= (uint32_t)(1 + 16 * (lane / 16) + j) << (8 * (j % 4));
  }
  l11_v4i av = {(int32_t)a[0], (int32_t)a[1], (int32_t)a[2], (int32_t)a[3]};
  l11_v4i bv = {(int32_t)b[0], (int32_t)b[1], (int32_t)b[2], (int32_t)b[3]};
#if LAY_W == 64
  l11_v4i c = {0, 0, 0, 0};
#else
  l11_v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#if LAY_W == 64
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, av, true, bv, c, false);
#else
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, c, false);
#endif
#else
  (void)av; (void)bv;
#endif
  for (int g = 0; g < LAY_G; g++) out[(w * LAY_W + lane) * LAY_G + g] = c[g];
}

int main() {
  constexpr int W = LAY_W, G = LAY_G, Q = LAY_Q, NP = W * 16;   // NP: probe bytes
  int32_t *d = nullptr;
  if (hipMalloc(&d, (size_t)NP * 256 * 4) != hipSuccess) return 1;
  std::vector<int32_t> h[3];
  for (int mode = 0; mode < 3; mode++) {
    const int waves = mode == 2 ? 16 * Q : NP;
    hipLaunchKernelGGL(lay_kernel, dim3(waves), dim3(W), 0, 0, mode, d);
    h[mode].resize((size_t)waves * 256);
    if (hipDeviceSynchronize() != hipSuccess) { printf("lay11 launch failed\n"); return 1; }
    if (hipMemcpy(h[mode].data(), d, h[mode].size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return 1;
  }
  printf("lay11 wave%d: %d sums a lane, %d lane groups\n", W, G, Q);
  // For every sum (lane, g): the A bytes and B bytes that reach it.
  int bad = 0;
  static int row[64][8], col[64][8];
  for (int m2 = 0; m2 < 2; m2++) {
    for (int q = 0; q < Q; q++)
      for (int g = 0; g < G; g++) {
        std::set<int> rows, bytes;
        int n = 0, other = 0, cnt[4] = {0, 0, 0, 0};
        for (int l = 16 * q; l < 16 * q + 16; l++) {
          std::set<int> mine;
          for (int w = 0; w < NP; w++) {
            const int v = h[m2][((size_t)w * W + l) * G + g];
            if (v == 0) continue;
            if (v != 1) other++;
            const int pl = w / 16, pj = w % 16;
            mine.insert(pl % 16);
            cnt[pl / 16]++;
            bytes.insert(pj);
            n++;
          }
          if (mine.size() != 1) bad++;
          (m2 == 0 ? row : col)[l][g] = mine.empty() ? -1 : *mine.begin();
          if (l == 16 * q + 5) rows = mine;
        }
        printf("lay11 %s group %d g%d: lane %d takes %s", m2 == 0 ? "A" : "B", q, g, 16 * q + 5, m2 == 0 ? "row" : "col");
        for (int r : rows) printf(" %d", r);
        printf("; bytes from lane groups 0-%d: %d %d %d %d; %d different byte places; %d not 1\n", Q - 1, cnt[0] / 16,
               cnt[1] / 16, cnt[2] / 16, cnt[3] / 16, (int)bytes.size(), other);
      }
  }
  printf("lay11 lanes whose sums take more or less than one row or column: %d\n", bad);
  // The guesses a fold could be written for.
  int h1 = 0, h2 = 0, h3 = 0, h4 = 0;
  for (int l = 0; l < W; l++)
    for (int g = 0; g < G; g++) {
      const bool c = col[l][g] == l % 16;
      h1 += c && row[l][g] == Q * g + l / 16;
      h2 += c && row[l][g] == g + G * (l / 16);
      h3 += c && row[l][g] == 2 * g + (l / 16) % 2 + 8 * (l / 32);
      h4 += c && row[l][g] == g + 8 * ((l / 16) & 1) + 4 * (l / 32);
    }
  printf("lay11 C[%dg+l/16][l%%16]: %d; C[g+%d(l/16)][l%%16]: %d; C[2g+(l/16)%%2+8(l/32)][l%%16]: %d; "
         "C[g+8((l/16)&1)+4(l/32)][l%%16]: %d (of 256)\n", Q, h1, G, h2, h3, h4);
  for (int q = 0; q < Q; q++) {
    printf("lay11 rows of lane %2d by g:", 16 * q + 5);
    for (int g = 0; g < G; g++) printf(" %d", row[16 * q + 5][g]);
    printf("; cols:");
    for (int g = 0; g < G; g++) printf(" %d", col[16 * q + 5][g]);
    printf("\n");
  }
  // k pairing: for A byte (group qa, byte ja), the B byte each lane group's sums read, as (group, byte); none: that
  // group's sums don't take this copy of the A byte.
  for (int qa = 0; qa < Q; qa++) {
    int same = 0, lit = 0;
    std::set<int> from[4];
    for (int ja = 0; ja < 16; ja++) {
      const int w = 16 * qa + ja;
      for (int q = 0; q < Q; q++)
        for (int l = 16 * q; l < 16 * q + 16; l++)
          for (int g = 0; g < G; g++) {
            const int v = h[2][((size_t)w * W + l) * G + g];
            if (!v) continue;
            lit++;
            same += (v - 1) % 16 == ja;
            from[q].insert((v - 1) / 16);
          }
    }
    printf("lay11 k: A copy in group %d: %d sums lit, %d of them read B's byte of the same place; B copy read, by the "
           "sums' lane group:", qa, lit, same);
    for (int q = 0; q < Q; q++) {
      printf(" [");
      for (int x : from[q]) printf(" %d", x);
      printf(" ]");
    }
    printf("\n");
  }
  return 0;
}
