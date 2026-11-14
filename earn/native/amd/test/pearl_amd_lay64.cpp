// What gfx12's wave64 v_wmma_i32_16x16x16_iu8 does with each operand byte (goal-rdna4b, PEARL_AMD_W64). Build with
// -mwavefrontsize64 for gfx1200 or gfx1201. One WMMA a wave, one probe byte a wave:
//   mode 0  A' has one byte set (lane la, byte ja), B' all ones: the sums it lights are C row r(la, ja).
//   mode 1  the same for B': C column c(lb, jb).
//   mode 2  A' byte (lane 16 qa, byte ja) set, B' byte (lane lb, byte jb) = 1 + 4 (lb / 16) + jb: the sums of that
//           A' row read the B' byte that meets it, so they say which B' byte has the same k.
// The host prints, for each accumulator (lane, g), the A' lane % 16 and B' lane % 16 whose bytes reach it, checks
// that every byte of a lane reaches the same ones, and checks the fold's k pairing: A' lane group q byte j meets B'
// lane group q byte j.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

typedef int32_t l64_v4i __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(64) void lay_kernel(int mode, int32_t *out) {
  const int w = (int)blockIdx.x, lane = (int)threadIdx.x;
  int32_t a = 0, b = 0;
  if (mode == 0) {
    a = lane == w / 4 ? (int32_t)(1u << (8 * (w % 4))) : 0;
    b = 0x01010101;
  } else if (mode == 1) {
    a = 0x01010101;
    b = lane == w / 4 ? (int32_t)(1u << (8 * (w % 4))) : 0;
  } else {
    const int qa = w / 4, ja = w % 4;
    a = lane == 16 * qa ? (int32_t)(1u << (8 * ja)) : 0;
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) v |= (uint32_t)(1 + 4 * (lane / 16) + j) << (8 * j);
    b = (int32_t)v;
  }
  l64_v4i c = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(true, a, true, b, c, false);
#endif
  for (int g = 0; g < 4; g++) out[(w * 64 + lane) * 4 + g] = c[g];
}

int main() {
  int32_t *d = nullptr;
  if (hipMalloc(&d, 256 * 64 * 4 * 4) != hipSuccess) return 1;
  std::vector<int32_t> h[3];
  for (int mode = 0; mode < 3; mode++) {
    const int waves = mode == 2 ? 16 : 256;
    hipLaunchKernelGGL(lay_kernel, dim3(waves), dim3(64), 0, 0, mode, d);
    h[mode].resize(waves * 256);
    if (hipMemcpy(h[mode].data(), d, h[mode].size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return 1;
  }
  int crow[64][4], ccol[64][4], bad = 0;
  for (int l = 0; l < 64; l++)
    for (int g = 0; g < 4; g++) crow[l][g] = ccol[l][g] = -1;
  for (int mode = 0; mode < 2; mode++)
    for (int w = 0; w < 256; w++)
      for (int l = 0; l < 64; l++)
        for (int g = 0; g < 4; g++) {
          const int v = h[mode][(w * 64 + l) * 4 + g];
          if (v == 0) continue;
          int &slot = mode == 0 ? crow[l][g] : ccol[l][g];
          const int lab = (w / 4) % 16;
          if (v != 1 || (slot >= 0 && slot != lab)) bad++;
          slot = lab;
        }
  printf("lay64 mode 0/1 conflicts: %d\n", bad);
  for (int l = 0; l < 64; l += 4) {
    printf("lay64 C lanes %2d-%2d (row,col) by g:", l, l + 3);
    for (int ll = l; ll < l + 4; ll++) {
      printf(" |");
      for (int g = 0; g < 4; g++) printf(" %d,%d", crow[ll][g], ccol[ll][g]);
    }
    printf("\n");
  }
  int h1 = 0, h2 = 0;
  for (int l = 0; l < 64; l++)
    for (int g = 0; g < 4; g++) {
      h1 += crow[l][g] == 4 * (l / 16) + g && ccol[l][g] == l % 16;
      h2 += crow[l][g] == g + 8 * ((l / 16) & 1) + 4 * (l / 32) && ccol[l][g] == l % 16;
    }
  printf("lay64 C[4(l/16)+g][l%%16]: %d of 256; C[g+8((l/16)&1)+4(l/32)][l%%16]: %d of 256\n", h1, h2);
  int kok = 0, kbad = 0;
  for (int w = 0; w < 16; w++) {
    const int want = 1 + 4 * (w / 4) + w % 4;
    int got = -1;
    for (int l = 0; l < 64 && got < 0; l++)
      for (int g = 0; g < 4; g++)
        if (crow[l][g] == 0 && h[2][(w * 64 + l) * 4 + g]) { got = h[2][(w * 64 + l) * 4 + g]; break; }
    if (got == want) kok++;
    else { kbad++; printf("lay64 k: A lane group %d byte %d meets B byte value %d (want %d)\n", w / 4, w % 4, got, want); }
  }
  printf("lay64 k pairing: %d of 16 as the fold assumes, %d not\n", kok, kbad);
  return 0;
}
