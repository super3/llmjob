// What gfx11's wave64 v_wmma_i32_16x16x16_iu8 does with each operand byte (goal-f-rdna3). Build with
// -mwavefrontsize64 for gfx1100 or gfx1101. In wave64 a lane holds 8 bytes of A and of B and 4 sums. One WMMA a
// wave, one probe byte a wave:
//   mode 0  A has one byte set (lane la, byte ja), B all ones: the sums that byte reaches.
//   mode 1  the same for B.
//   mode 2  A byte (lane 16 qa, byte ja) set, and B byte (lane lb, byte jb) = 1 + 8 (lb / 16) + jb: a sum that A
//           byte reaches reads the B byte that meets it, so it says which B byte has the same k.
// The host prints, for each lane group (lane / 16) and register g: the A row and B column its sums take, the lane
// groups those bytes come from and how many bytes; then the k pairing as each lane group's sums see it.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <set>
#include <vector>

typedef int32_t l11_v2i __attribute__((ext_vector_type(2)));
typedef int32_t l11_v4i __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(64) void lay_kernel(int mode, int32_t *out) {
  const int w = (int)blockIdx.x, lane = (int)threadIdx.x;
  uint32_t a[2] = {0u, 0u}, b[2] = {0u, 0u};
  const int pl = w / 8, pj = w % 8;   // modes 0 and 1: the probe's lane and byte
  if (mode == 0) {
    if (lane == pl) a[pj / 4] = 1u << (8 * (pj % 4));
    b[0] = b[1] = 0x01010101u;
  } else if (mode == 1) {
    a[0] = a[1] = 0x01010101u;
    if (lane == pl) b[pj / 4] = 1u << (8 * (pj % 4));
  } else {
    const int qa = w / 8, ja = w % 8;
    if (lane == 16 * qa) a[ja / 4] = 1u << (8 * (ja % 4));
    for (int j = 0; j < 8; j++) b[j / 4] |= (uint32_t)(1 + 8 * (lane / 16) + j) << (8 * (j % 4));
  }
  l11_v2i av = {(int32_t)a[0], (int32_t)a[1]}, bv = {(int32_t)b[0], (int32_t)b[1]};
  l11_v4i c = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, av, true, bv, c, false);
#else
  (void)av; (void)bv;
#endif
  for (int g = 0; g < 4; g++) out[(w * 64 + lane) * 4 + g] = c[g];
}

int main() {
  int32_t *d = nullptr;
  if (hipMalloc(&d, 512 * 64 * 4 * 4) != hipSuccess) return 1;
  std::vector<int32_t> h[3];
  for (int mode = 0; mode < 3; mode++) {
    const int waves = mode == 2 ? 32 : 512;
    hipLaunchKernelGGL(lay_kernel, dim3(waves), dim3(64), 0, 0, mode, d);
    h[mode].resize(waves * 256);
    if (hipDeviceSynchronize() != hipSuccess) { printf("lay11 launch failed\n"); return 1; }
    if (hipMemcpy(h[mode].data(), d, h[mode].size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return 1;
  }
  // For every sum (lane, g): the A bytes and B bytes that reach it.
  int bad = 0;
  int row[64][4], col[64][4];
  for (int m2 = 0; m2 < 2; m2++) {
    for (int q = 0; q < 4; q++)
      for (int g = 0; g < 4; g++) {
        std::set<int> rows, groups, bytes;
        int n = 0, other = 0;
        for (int l = 16 * q; l < 16 * q + 16; l++) {
          std::set<int> mine;
          for (int w = 0; w < 512; w++) {
            const int v = h[m2][(w * 64 + l) * 4 + g];
            if (v == 0) continue;
            if (v != 1) other++;
            const int pl = w / 8, pj = w % 8;
            mine.insert(pl % 16);
            groups.insert(pl / 16);
            bytes.insert(pj);
            n++;
          }
          if (mine.size() != 1) bad++;
          (m2 == 0 ? row : col)[l][g] = mine.empty() ? -1 : *mine.begin();
          if (l == 16 * q + 5) rows = mine;
        }
        printf("lay11 %s lanes %2d-%2d g%d: lane %d takes %s", m2 == 0 ? "A" : "B", 16 * q, 16 * q + 15, g, 16 * q + 5,
               m2 == 0 ? "row" : "col");
        for (int r : rows) printf(" %d", r);
        printf("; from lane groups");
        for (int x : groups) printf(" %d", x);
        printf("; bytes");
        for (int x : bytes) printf(" %d", x);
        printf("; %d probe bytes a lane, %d not 1\n", n / 16, other);
      }
  }
  printf("lay11 lanes whose sums take more or less than one row or column: %d\n", bad);
  // Which lane group's bytes each sum takes, per source group: counts[q][g][source group][A or B].
  for (int m2 = 0; m2 < 2; m2++)
    for (int q = 0; q < 4; q++) {
      printf("lay11 %s source groups of lane group %d by g:", m2 == 0 ? "A" : "B", q);
      for (int g = 0; g < 4; g++) {
        int cnt[4] = {0, 0, 0, 0};
        for (int w = 0; w < 512; w++)
          if (h[m2][(w * 64 + 16 * q + 5) * 4 + g]) cnt[(w / 8) / 16]++;
        printf(" | %d %d %d %d", cnt[0], cnt[1], cnt[2], cnt[3]);
      }
      printf("\n");
    }
  // The guesses the fold could be written for.
  int h1 = 0, h2 = 0, h3 = 0, h4 = 0;
  for (int l = 0; l < 64; l++)
    for (int g = 0; g < 4; g++) {
      const bool c = col[l][g] == l % 16;
      h1 += c && row[l][g] == 4 * g + l / 16;
      h2 += c && row[l][g] == g + 4 * (l / 16);
      h3 += c && row[l][g] == 2 * g + (l / 16) % 2 + 8 * (l / 32);
      h4 += c && row[l][g] == g + 8 * ((l / 16) & 1) + 4 * (l / 32);
    }
  printf("lay11 C[4g+l/16][l%%16]: %d; C[g+4(l/16)][l%%16]: %d; C[2g+(l/16)%%2+8(l/32)][l%%16]: %d; "
         "C[g+8((l/16)&1)+4(l/32)][l%%16]: %d (of 256)\n", h1, h2, h3, h4);
  for (int q = 0; q < 4; q++) {
    printf("lay11 rows of lane %2d by g:", 16 * q + 5);
    for (int g = 0; g < 4; g++) printf(" %d", row[16 * q + 5][g]);
    printf("; cols:");
    for (int g = 0; g < 4; g++) printf(" %d", col[16 * q + 5][g]);
    printf("\n");
  }
  // k pairing: for A byte (group qa, byte ja), the B byte id each lane group's sums read (0: that group's sums of
  // the A row don't take this A byte).
  for (int w = 0; w < 32; w++) {
    printf("lay11 k: A group %d byte %d meets B (group, byte) as lane groups 0-3 see it:", w / 8, w % 8);
    for (int q = 0; q < 4; q++) {
      std::set<int> vals;
      for (int l = 16 * q; l < 16 * q + 16; l++)
        for (int g = 0; g < 4; g++)
          if (h[2][(w * 64 + l) * 4 + g]) vals.insert(h[2][(w * 64 + l) * 4 + g]);
      printf(" [");
      for (int v : vals) printf(" %d,%d", (v - 1) / 8, (v - 1) % 8);
      printf(" ]");
    }
    printf("\n");
  }
  return 0;
}
