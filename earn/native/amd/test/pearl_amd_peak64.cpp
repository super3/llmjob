// pearl_amd_peak.cpp's question for a wave64 RDNA 4 fold (goal-rdna4b, 14 November): with the same 128 accumulator
// VGPRs, a wave64 covers 64 x 128 of C (4 x 8 WMMA tiles of 16 x 16, 4 VGPRs each) against wave32's 64 x 64, so a
// WMMA needs a quarter fewer LDS fragment bytes and instructions (12 ds_read_b64 of 64 lanes for 64 WMMAs a slab,
// against 8 ds_read_b128 of 32 lanes for 32). Does the card run v_wmma_i32_16x16x16_iu8 at the same rate in wave64,
// and what do the reads and the readout cost there? Build with -mwavefrontsize64, gfx12 only:
//
//   clang++ -x hip --offload-arch=gfx1201 -mwavefrontsize64 -O3 test/pearl_amd_peak64.cpp -o peak64
//   peak64 [secs=2.5] [only]
//
// Blocks are 256 threads (4 waves64, 2 x 2), so a block's tile is 128 x 256 with a 4 x 8 wave tile, the LDS32 layout
// of pearl_amd_peak's lk32 variants (24,576 bytes: [2][128 x 32] then [2][256 x 32]). A slab is 32 bytes of k, two
// k-steps; each fragment is one ds_read_b64 that serves both (lanes 16q .. 16q + 15 read bytes 8q .. 8q + 7 of the
// row; the low word is k-step 0's, the high word k-step 1's): k is permuted the same way for A and B, so the sums
// are the same. Variants:
//   regs     fragments in registers, two sets in turn
//   lds      the reads, no barrier;  ldsbar  the reads and the slab barrier
//   rdout    regs plus a readout every 4 slabs: each lane XORs its 128 sums into 4 words (2 row blocks x 2 column
//            blocks), then the 8 lanes of a region (lane bits 0, 3 and 5) by two DPP steps and v_permlane64
//   ldsrd    ldsbar plus that readout
//   pin, pinrd  swzbar and swzrd with the barrier pinned after the WMMAs (the fold's PEARL_AMD_SCHEDB)
//   swz, swzbar, swzrd  the same with the 8-byte groups swapped by halves on row bits 2 xor 3
// The clock and MAC/CU/clk as pearl_amd_peak reports them. It checks nothing.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int32_t p64_v4i __attribute__((ext_vector_type(4)));
typedef int32_t p64_v2i __attribute__((ext_vector_type(2)));

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
#define P64_GFX12 1
#else
#define P64_GFX12 0
#endif

namespace {

// -DP64_T=512: blocks of 8 waves64 (4 x 2), a 256 x 256 tile, as the RDNA 4 fold's 512-thread blocks.
#ifndef P64_T
#define P64_T 256
#endif
constexpr int kThreads = P64_T, kWaves = kThreads / 64, kBM = 64 * (kWaves / 2);
constexpr int kLds = 2 * kBM * 32 + 2 * 256 * 32;   // 24,576 (32,768 at 512 threads)
constexpr int kStampWords = 8;
constexpr size_t kPerBlock = (size_t)kLds + (size_t)kThreads * 2 * 12 * 4;

__device__ __forceinline__ void p64_mma(p64_v4i &c, int32_t a, int32_t b) {
#if P64_GFX12
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(true, a, true, b, c, false);
#else
  (void)c; (void)a; (void)b;
#endif
}
__device__ __forceinline__ uint32_t p64_cycles() {
#if P64_GFX12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t p64_realtime() {
#if P64_GFX12
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t p64_hw_id1() {
#if P64_GFX12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t p64_rotl13(uint32_t x) { return (x << 13) | (x >> 19); }
// The 8 lanes of a region in wave64: lane bits 0 and 3 (DPP inside a row of 16) and bit 5 (the other half-wave).
__device__ __forceinline__ uint32_t p64_group_xor(uint32_t x) {
#if P64_GFX12
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, false);
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x128, 0xF, 0xF, false);
  x ^= (uint32_t)__builtin_amdgcn_permlane64((int)x);
#endif
  return x;
}

template <int SRC, int NA, int NB, int BAR, int RD>
__global__ __launch_bounds__(kThreads) void p64_kernel(const int8_t *__restrict__ ops, uint32_t iters,
                                                        uint32_t *__restrict__ out, uint32_t *__restrict__ st) {
  extern __shared__ __attribute__((aligned(16))) int8_t lds[];
  const int t = (int)threadIdx.x, lane = t & 63, wave = t >> 6;
  const int wr = wave >> 1, wc = wave & 1;
  const uint32_t hw0 = p64_hw_id1();
  const uint32_t c0 = p64_cycles();
  const uint32_t r0 = p64_realtime();
  const int8_t *src = ops + (size_t)(blockIdx.x & 63u) * kPerBlock;
  p64_v4i acc[NA][NB];
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++) acc[i][j] = p64_v4i{0, 0, 0, 0};
  int32_t ra[2][NA], rb[2][NB];
  if constexpr (SRC == 0) {
    const int32_t *p = reinterpret_cast<const int32_t *>(src + kLds) + (size_t)t * 2 * (NA + NB);
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int i = 0; i < NA; i++) ra[s][i] = p[s * (NA + NB) + i];
#pragma unroll
      for (int j = 0; j < NB; j++) rb[s][j] = p[s * (NA + NB) + NA + j];
    }
  } else {
    for (int i = 16 * t; i < kLds; i += 16 * kThreads)
      *reinterpret_cast<p64_v4i *>(lds + i) = *reinterpret_cast<const p64_v4i *>(src + i);
    __syncthreads();
  }
  uint32_t keepLo[2][2], keepHi[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int q = 0; q < 2; q++) keepLo[i][q] = keepHi[i][q] = 0u;
  const uint32_t r8 = (uint32_t)((lane & 1) | ((lane >> 2) & 2) | ((lane >> 3) & 4));
  for (uint32_t it = 0; it < iters; it++) {
    int32_t pa[2][NA], pb[2][NB];
    if constexpr (SRC >= 1) {
      const int buf = (int)(it & 1u);
      const int8_t *aBase = lds + buf * kBM * 32 + wr * (16 * NA) * 32;
      const int8_t *bBase = lds + 2 * kBM * 32 + buf * 256 * 32 + wc * (16 * NB) * 32;
      // SRC 2: the 8-byte groups of a row stored swapped in halves by row bits 2 xor 3 (PEARL_AMD_LDS32's swap), so
      // rows r and r + 8 of a half-wave's read fall in different banks.
      const int row = lane & 15;
      const int sw = SRC == 2 ? 2 * (__builtin_popcount((unsigned)row & 12u) & 1) : 0;
      const int off = row * 32 + 8 * ((lane >> 4) ^ sw);
#pragma unroll
      for (int i = 0; i < NA; i++) {
        const p64_v2i q = *reinterpret_cast<const p64_v2i *>(aBase + i * 16 * 32 + off);
        pa[0][i] = q[0]; pa[1][i] = q[1];
      }
#pragma unroll
      for (int j = 0; j < NB; j++) {
        const p64_v2i q = *reinterpret_cast<const p64_v2i *>(bBase + j * 16 * 32 + off);
        pb[0][j] = q[0]; pb[1][j] = q[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int i = 0; i < NA; i++)
#pragma unroll
        for (int j = 0; j < NB; j++)
          p64_mma(acc[i][j], SRC == 0 ? ra[kk][i] : pa[kk][i], SRC == 0 ? rb[kk][j] : pb[kk][j]);
    }
    if constexpr (RD) {
      static_assert(NA == 4 && NB == 8, "the readout is for the 4 x 8 wave tile");
      if ((it & 3u) == 3u) {
        const uint32_t c = it >> 2;
        if ((c & 15u) == 0u) {
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
              keepLo[i][q] = p64_rotl13(keepLo[i][q]);
              keepHi[i][q] = p64_rotl13(keepHi[i][q]);
            }
        }
        const bool mine = (c & 7u) == r8;
#pragma unroll
        for (int rb2 = 0; rb2 < 2; rb2++)
#pragma unroll
          for (int cb = 0; cb < 2; cb++) {
            uint32_t x = 0u;
#pragma unroll
            for (int i = 2 * rb2; i < 2 * rb2 + 2; i++)
#pragma unroll
              for (int j = 4 * cb; j < 4 * cb + 4; j++)
#pragma unroll
                for (int g = 0; g < 4; g++) x ^= (uint32_t)acc[i][j][g];
            const uint32_t v = p64_group_xor(x);
            if (c & 8u) keepHi[rb2][cb] ^= mine ? v : 0u;
            else keepLo[rb2][cb] ^= mine ? v : 0u;
          }
      }
    }
    // BAR 2: the barrier pinned after the slab's WMMAs, as the fold's PEARL_AMD_SCHEDB keeps it (the compiler
    // otherwise moves it above them, right after the reads).
    if constexpr (BAR == 2) {
#if P64_GFX12
#pragma unroll
      for (int i = 0; i < NA; i++)
#pragma unroll
        for (int j = 0; j < NB; j++) asm volatile("" ::"v"(acc[i][j]));
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    if constexpr (BAR) __syncthreads();
    if constexpr (SRC >= 1) asm volatile("" ::: "memory");
  }
  const uint32_t c1 = p64_cycles();
  const uint32_t r1 = p64_realtime();
  const uint32_t hw1 = p64_hw_id1();
  uint32_t x = 0u;
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int q = 0; q < 2; q++) x ^= keepLo[i][q] * 5u ^ keepHi[i][q] * 7u;
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++)
#pragma unroll
      for (int g = 0; g < 4; g++) x ^= (uint32_t)acc[i][j][g] * (uint32_t)(1 + 2 * (i * NB + j + g));
  out[(size_t)blockIdx.x * kThreads + t] = x;
  if (lane == 0) {
    uint32_t *o = st + ((size_t)blockIdx.x * kWaves + (size_t)wave) * kStampWords;
    o[0] = 0x50454b31u;
    o[1] = c1 - c0;
    o[2] = r0;
    o[3] = r1;
    o[4] = hw0;
    o[5] = hw1;
    o[6] = blockIdx.x;
    o[7] = (uint32_t)wave;
  }
}

typedef void (*Kernel)(const int8_t *, uint32_t, uint32_t *, uint32_t *);
struct Cfg {
  const char *name;
  Kernel k;
  int src, na, nb;
  int bpw;   // 256-thread blocks a WGP (4 waves64 each): 4 = 4 waves64 a SIMD, the thread count of the wave32 fold's 8
};
#define K(S, A, B, BR, R) reinterpret_cast<Kernel>(&p64_kernel<S, A, B, BR, R>)
const Cfg kCfgs[] = {
    {"w64regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 4},
    {"w64regs4x8", K(0, 4, 8, 0, 0), 0, 4, 8, 2},
    {"w64regs4x8", K(0, 4, 8, 0, 0), 0, 4, 8, 3},
    {"w64regs4x8", K(0, 4, 8, 0, 0), 0, 4, 8, 4},
    {"w64lds4x8", K(1, 4, 8, 0, 0), 1, 4, 8, 4},
    {"w64ldsbar4x8", K(1, 4, 8, 1, 0), 1, 4, 8, 4},
    {"w64rdout4x8", K(0, 4, 8, 0, 1), 0, 4, 8, 4},
    {"w64ldsrd4x8", K(1, 4, 8, 1, 1), 1, 4, 8, 4},
    {"w64ldsrd4x8", K(1, 4, 8, 1, 1), 1, 4, 8, 3},
    {"w64swz4x8", K(2, 4, 8, 0, 0), 2, 4, 8, 4},
    {"w64swzbar4x8", K(2, 4, 8, 1, 0), 2, 4, 8, 4},
    {"w64swzrd4x8", K(2, 4, 8, 1, 1), 2, 4, 8, 4},
    {"w64pin4x8", K(2, 4, 8, 2, 0), 2, 4, 8, 4},
    {"w64pinrd4x8", K(2, 4, 8, 2, 1), 2, 4, 8, 4},
    {"w64pinrd4x8", K(2, 4, 8, 2, 1), 2, 4, 8, 3},
};
#undef K

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return (uint32_t)g_rng;
}
bool ok(hipError_t e, const char *what) {
  if (e == hipSuccess) return true;
  fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
  return false;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
void launch(const Cfg &c, int blocks, size_t lds, const int8_t *ops, uint32_t iters, uint32_t *out, uint32_t *st) {
  void *args[] = {&ops, &iters, &out, &st};
  (void)hipLaunchKernel(reinterpret_cast<const void *>(c.k), dim3(blocks), dim3(kThreads), args, lds, 0);
}

}  // namespace

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 2.5;
  const char *only = argc > 2 && argv[2][0] ? argv[2] : nullptr;
  hipDeviceProp_t prop;
  if (!ok(hipGetDeviceProperties(&prop, 0), "device")) return 1;
  int wallKhz = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0) != hipSuccess) wallKhz = 0;
  const int wgps = prop.multiProcessorCount, cus = 2 * wgps;
  printf("device: %s, %s, %d WGPs (%d CUs), warp size %d, REALTIME %d kHz\n", prop.name, prop.gcnArchName, wgps, cus,
         prop.warpSize, wallKhz);
  if (wallKhz <= 0) wallKhz = 100000;
  std::vector<int8_t> host(64 * kPerBlock);
  for (auto &b : host) {
    const int x = 48 + (int)(next_rand() & 63u) - 32 - ((int)(next_rand() & 63u) - 32);
    b = (int8_t)std::max(-128, std::min(127, x));
  }
  int8_t *dOps = nullptr;
  uint32_t *dOut = nullptr, *dSt = nullptr;
  const int maxBlocks = wgps * 8;
  if (!ok(hipMalloc(&dOps, host.size()), "malloc ops") ||
      !ok(hipMalloc(&dOut, (size_t)maxBlocks * kThreads * 4), "malloc out") ||
      !ok(hipMalloc(&dSt, (size_t)maxBlocks * kWaves * kStampWords * 4), "malloc stamps") ||
      !ok(hipMemcpy(dOps, host.data(), host.size(), hipMemcpyHostToDevice), "copy ops"))
    return 1;
  for (const Cfg &c : kCfgs) {
    if (c.bpw * 256 % kThreads) continue;
    if (only && !strstr(c.name, only)) continue;
    hipFuncAttributes fa;
    const int vgprs = hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(c.k)) == hipSuccess ? fa.numRegs : -1;
    size_t lds = 0;
    int fit = 0;
    for (size_t l = 65536; l >= 1024; l -= 512) {
      int n = 0;
      if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, reinterpret_cast<const void *>(c.k), kThreads, l) !=
          hipSuccess)
        continue;
      if (n <= c.bpw * 256 / kThreads) { lds = l; fit = n; }
      if (n > c.bpw * 256 / kThreads) break;
    }
    if (c.src >= 1 && lds < (size_t)kLds) lds = kLds;
    if (lds == 0) lds = 65536;
    const int blocks = wgps * c.bpw * 256 / kThreads;   // bpw counts 256-thread blocks
    uint32_t iters = 2000;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    float ms = 0.f;
    for (int tries = 0; tries < 6; tries++) {
      (void)hipEventRecord(e0);
      launch(c, blocks, lds, dOps, iters, dOut, dSt);
      (void)hipEventRecord(e1);
      if (!ok(hipEventSynchronize(e1), c.name)) return 1;
      (void)hipEventElapsedTime(&ms, e0, e1);
      if (ms > 30.f) break;
      iters = (uint32_t)std::min(4.0e8, iters * std::max(2.0, 40.0 / std::max(0.05f, ms)));
    }
    if (!ok(hipGetLastError(), c.name)) return 1;
    std::vector<double> rates;
    double total = 0.0, timed = 0.0, macs = 0.0;
    const double perLaunch = (double)blocks * kWaves * iters * 2.0 * c.na * c.nb * 4096.0;
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      (void)hipEventRecord(e0);
      launch(c, blocks, lds, dOps, iters, dOut, dSt);
      (void)hipEventRecord(e1);
      if (!ok(hipEventSynchronize(e1), c.name)) return 1;
      (void)hipEventElapsedTime(&ms, e0, e1);
      total += ms / 1000.0;
      if (total > secs / 3) {
        timed += ms / 1000.0;
        macs += perLaunch;
        rates.push_back(perLaunch / (ms / 1000.0) / 1e12);
      }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > secs) break;
    }
    std::vector<uint32_t> s((size_t)blocks * kWaves * kStampWords);
    if (!ok(hipMemcpy(s.data(), dSt, s.size() * 4, hipMemcpyDeviceToHost), "stamps")) return 1;
    std::vector<double> mhz;
    for (size_t i = 0; i < (size_t)blocks * kWaves; i++) {
      const uint32_t *r = &s[i * kStampWords];
      if (r[0] != 0x50454b31u || r[4] != r[5]) continue;
      const double ticks = (double)(uint32_t)(r[3] - r[2]);
      if (ticks > 0) mhz.push_back((double)r[1] / (ticks / ((double)wallKhz * 1e3)) / 1e6);
    }
    const double tmac = timed > 0 ? macs / timed / 1e12 : 0.0;
    const double clk = median(mhz);
    printf("peak %-12s T%d waves64/SIMD %d, %d VGPRs, LDS %zu (fit %d), %u iters: %.2f TMAC/s (median launch %.2f), "
           "%.0f MHz (median of %zu waves), %.1f MAC/CU/clk\n",
           c.name, kThreads, c.bpw, vgprs, lds, fit, iters, tmac, median(rates), clk, mhz.size(),
           clk > 0 ? tmac * 1e12 / (cus * clk * 1e6) : 0.0);
    fflush(stdout);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
  }
  return 0;
}
