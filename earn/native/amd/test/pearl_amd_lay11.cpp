// Which lanes of gfx11's wave32 v_wmma_i32_16x16x16_iu8 operands reach which sums (g18-rdna3). The fold loads A' and
// B' duplicated in both half-waves, as AMD's layout asks. The CPU emulation (pearl_amd_fold.h, PearlAmdWmma11) says
// more: output row 2g + h (lane half h) reads A row r from lane r + 16h, so lanes 1, 3, ..., 15 and 16, 18, ..., 30
// of A would never be read. If the card agrees, those lanes can hold a constant and the WMMAs see half the A bits
// change. One WMMA a mode; A row r is r + 1 in every byte, B all ones, so a sum is 16 (r + 1).
//   mode 0  A duplicated (the fold's)
//   mode 1  A lanes 16-31 zero;  mode 2  lanes 0-15 zero
//   mode 3  lanes the emulation says aren't read (odd lanes of 0-15, even lanes of 16-31) set to 0x55
//   mode 4  the same lanes zero;  mode 5  B: the same lanes 0x55 (B has no such lanes in the emulation)
// Build: clang++ -x hip --offload-arch=gfx1100 -O3 test/pearl_amd_lay11.cpp -o lay11
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

typedef int32_t l_v4i __attribute__((ext_vector_type(4)));
typedef int32_t l_v8i __attribute__((ext_vector_type(8)));

__global__ __launch_bounds__(32) void lay_kernel(int mode, int32_t *out) {
  const int lane = (int)threadIdx.x, r = lane & 15, h = lane >> 4;
  const bool unread = (r & 1) != h;   // lower odd lanes and upper even lanes
  int32_t av = (int32_t)(0x01010101u * (uint32_t)(r + 1)), bv = 0x01010101;
  if (mode == 1 && h) av = 0;
  if (mode == 2 && !h) av = 0;
  if (mode == 3 && unread) av = 0x55555555;
  if (mode == 4 && unread) av = 0;
  if (mode == 5 && unread) bv = 0x55555555;
  l_v4i a = {av, av, av, av}, b = {bv, bv, bv, bv};
  l_v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#endif
  for (int g = 0; g < 8; g++) out[(2 * g + h) * 16 + r] = c[g];   // C[2g + h][lane % 16]
}

int main() {
  int32_t *d = nullptr;
  if (hipMalloc(&d, 256 * sizeof(int32_t)) != hipSuccess) return 2;
  std::vector<int32_t> c0(256), c(256);
  for (int mode = 0; mode <= 5; mode++) {
    hipLaunchKernelGGL(lay_kernel, dim3(1), dim3(32), 0, 0, mode, d);
    if (hipMemcpy(c.data(), d, 256 * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) return 2;
    if (mode == 0) c0 = c;
    int diff = 0;
    for (int i = 0; i < 256; i++) diff += c[i] != c0[i];
    printf("mode %d: %3d of 256 sums differ from mode 0; rows' column-0 sums:", mode, diff);
    for (int row = 0; row < 16; row++) printf(" %d", c[row * 16]);
    printf("\n");
  }
  return 0;
}
