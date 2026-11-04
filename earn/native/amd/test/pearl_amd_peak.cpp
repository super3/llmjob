// WMMA peak probe for RDNA 3 and RDNA 4: what the card's int8 matrix instruction does with little or nothing else
// in the way, and the clock it runs at while doing it. It answers the first question about a fold's headroom: is the
// fold short of the matrix units' own rate at the clock the card holds, or is the card already at it?
//
//   pearl_amd_peak [secs=2.5] [only]
//
// Every kernel is v_wmma_i32_16x16x16_iu8 in wave32 (gfx12's form on RDNA 4, gfx11's on RDNA 3), 256 threads a
// block in the fold's shape (8 waves, 2 x 4), and a loop whose step is one of the fold's slabs: two k-steps of 16,
// each NA x NB WMMAs (the fold's wave tile is 4 x 4). What feeds them is the variant:
//   regs   fragments in registers, two sets that take turns, so the matrix units see changing bits as they do in
//          the fold. Nothing else runs in the loop.
//   valuN  regs, plus N VALU instructions a slab (v_xor_b32 in volatile asm, which the compiler keeps): does VALU
//          work take time from the WMMAs, as the fold's readout and staging addresses do?
//   lds    the fold's ratio of LDS fragment reads: 4 A and 4 B fragments a k-step, from the fold's 36,864 bytes of
//          LDS at the fold's addresses, filled once. No barrier, no global loads, no stores.
//   ldsbar lds, plus the fold's s_barrier at the end of each slab.
//   bar    regs, plus the barrier.
//   valuiN regs, plus N VALU a slab on 8 registers in turn: no instruction waits on the one before it.
//   rdout  regs, plus the fold's readout at every chunk end (4 slabs at rank 128): RDNA 4's, with the wrap rotate.
//          Its words are kept and stored, so none of it can be dropped.
//   rdlocal  rdout without the cross-lane steps (each lane keeps its own XOR), as PEARL_AMD_READOUT_DROP times.
//   ldsrd  ldsbar plus the readout: the fold with no staging.
// The data the fragments hold: miner (the fold test's miner operands: 48 plus noise, [-15, 111]), zero, or rand
// (uniform int8). On a card at its power limit the bits move the clock.
//
// Occupancy is set the fold's way, with LDS: a block asks for enough that only `bpw` blocks fit on a WGP (2 waves a
// SIMD each), and the grid is exactly that many blocks a WGP, so every block runs at once and ends together.
// HW_ID1 from each block's wave 0 says how many blocks each WGP really got.
//
// The clock: each wave reads SHADER_CYCLES and REALTIME (a constant 100 MHz clock on RDNA) at its start and end;
// cycles over ticks is the clock the wave ran at. gfx11's counter is 20 bits, so there it is summed every slab.
// MAC/CU/clk is the rate over the CUs (two a WGP: HIP counts WGPs) and that clock. AMD's figures put RDNA 4 at
// 1,024 int8 MACs a CU a clock with this instruction (no sparsity) and RDNA 3 at 256.
//
// It checks nothing. It is for timing; the fold test checks the folds.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int32_t peak_v8i __attribute__((ext_vector_type(8)));
typedef int32_t peak_v4i __attribute__((ext_vector_type(4)));
typedef int32_t peak_v2i __attribute__((ext_vector_type(2)));

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
#define PEAK_GFX12 1
#else
#define PEAK_GFX12 0
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#define PEAK_GFX11 1
#else
#define PEAK_GFX11 0
#endif

// gfx12's fragment is 8 bytes a lane, gfx11's 16 (A and B duplicated in both half-waves). The host sizes buffers
// for 16.
#if PEAK_GFX11
typedef peak_v4i PeakFrag;
#else
typedef peak_v2i PeakFrag;
#endif

namespace {

constexpr int kThreads = 256, kWaves = 8, kS = 48;                 // the fold's LDS row: 32 bytes of k + 16
constexpr int kFoldLds = 2 * 128 * kS + 2 * 256 * kS;             // 36,864: sA[2][128 x 48], sB[2][256 x 48]
constexpr int kStampWords = 8;
// One block's operands: the fold's LDS bytes, then up to 4 x 4 fragments of 16 bytes in two sets for each thread.
constexpr size_t kPerBlock = (size_t)kFoldLds + (size_t)kThreads * 2 * 8 * 16;

__device__ __forceinline__ void peak_mma(peak_v8i &c, PeakFrag a, PeakFrag b) {
#if PEAK_GFX12
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#elif PEAK_GFX11
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
  (void)c; (void)a; (void)b;
#endif
}
__device__ __forceinline__ uint32_t peak_cycles() {
#if PEAK_GFX12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);              // SHADER_CYCLES_LO, 32 bits
#elif PEAK_GFX11
  return (uint32_t)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;   // SHADER_CYCLES, 20 bits
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t peak_realtime() {
#if PEAK_GFX12 || PEAK_GFX11
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);          // MSG_RTN_GET_REALTIME
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t peak_hw_id1() {
#if PEAK_GFX12 || PEAK_GFX11
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);              // HW_ID1
#else
  return 0u;
#endif
}

__device__ __forceinline__ uint32_t peak_rotl13(uint32_t x) { return (x << 13) | (x >> 19); }
// The fold's cross-lane XOR over the eight lanes that share a region (lane bits 0, 3 and 4), RDNA's DPP form.
__device__ __forceinline__ uint32_t peak_group_xor(uint32_t x) {
#if PEAK_GFX12 || PEAK_GFX11
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, false);
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x128, 0xF, 0xF, false);
  x ^= __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, false, false);
#endif
  return x;
}

// SRC 0: fragments in registers; 1: from LDS at the fold's addresses. VALU: filler instructions a slab, in a chain
// (VM 0) or on 8 registers in turn (VM 1). VM 2: the fold's readout every 4 slabs; VM 3: the same without its
// cross-lane steps. BAR: the fold's slab barrier.
template <int SRC, int NA, int NB, int VALU, int BAR, int VM>
__global__ __launch_bounds__(kThreads) void peak_kernel(const int8_t *__restrict__ ops, uint32_t iters,
                                                         uint32_t *__restrict__ out, uint32_t *__restrict__ st) {
  extern __shared__ __attribute__((aligned(16))) int8_t lds[];
  const int t = (int)threadIdx.x, lane = t & 31, wave = t >> 5;
  const int wr = wave / 4, wc = wave % 4;
  const uint32_t hw0 = peak_hw_id1();
  const uint32_t c0 = peak_cycles();
  const uint32_t r0 = peak_realtime();
  // Each block reads one of 64 copies of the operands, so blocks don't all hold the same bits.
  const int8_t *src = ops + (size_t)(blockIdx.x & 63u) * kPerBlock;
  peak_v8i acc[NA][NB];
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++) acc[i][j] = peak_v8i{0, 0, 0, 0, 0, 0, 0, 0};
  PeakFrag ra[2][NA], rb[2][NB];
  if constexpr (SRC == 0) {
    const PeakFrag *p = reinterpret_cast<const PeakFrag *>(src + kFoldLds) + (size_t)t * 2 * (NA + NB);
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int i = 0; i < NA; i++) ra[s][i] = p[s * (NA + NB) + i];
#pragma unroll
      for (int j = 0; j < NB; j++) rb[s][j] = p[s * (NA + NB) + NA + j];
    }
  } else {
    for (int i = 16 * t; i < kFoldLds; i += 16 * kThreads)
      *reinterpret_cast<peak_v4i *>(lds + i) = *reinterpret_cast<const peak_v4i *>(src + i);
    __syncthreads();
  }
  uint32_t vx = (uint32_t)t * 2654435761u, vy = (uint32_t)lane + 1u;
  uint32_t vi[8];
#pragma unroll
  for (int i = 0; i < 8; i++) vi[i] = vx + (uint32_t)i;
  // The readout's state, as the fold keeps it: words r and r + 8 of each region a lane shares (2 x 2 regions).
  uint32_t keepLo[2][2], keepHi[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int q = 0; q < 2; q++) keepLo[i][q] = keepHi[i][q] = 0u;
  const uint32_t r8 = (uint32_t)((lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4));
#if PEAK_GFX11
  uint32_t cyc = 0u, last = c0;
#endif
  for (uint32_t it = 0; it < iters; it++) {
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      PeakFrag a[NA], b[NB];
      if constexpr (SRC == 0) {
#pragma unroll
        for (int i = 0; i < NA; i++) a[i] = ra[kk][i];
#pragma unroll
        for (int j = 0; j < NB; j++) b[j] = rb[kk][j];
      } else {
        const int buf = (int)(it & 1u);
        const int8_t *aBase = lds + buf * 128 * kS + wr * 64 * kS;
        const int8_t *bBase = lds + 2 * 128 * kS + buf * 256 * kS + wc * 64 * kS;
#if PEAK_GFX11
        const int off = (lane & 15) * kS + kk * 16;
#else
        const int off = (lane & 15) * kS + kk * 16 + 8 * (lane >> 4);
#endif
#pragma unroll
        for (int i = 0; i < NA; i++) a[i] = *reinterpret_cast<const PeakFrag *>(aBase + i * 16 * kS + off);
#pragma unroll
        for (int j = 0; j < NB; j++) b[j] = *reinterpret_cast<const PeakFrag *>(bBase + j * 16 * kS + off);
      }
#pragma unroll
      for (int i = 0; i < NA; i++)
#pragma unroll
        for (int j = 0; j < NB; j++) peak_mma(acc[i][j], a[i], b[j]);
#if PEAK_GFX12 || PEAK_GFX11
      if constexpr (VM == 0) {
#pragma unroll
        for (int v = 0; v < VALU / 2; v++) asm volatile("v_xor_b32 %0, %1, %0" : "+v"(vx) : "v"(vy));
      } else if constexpr (VM == 1) {
#pragma unroll
        for (int v = 0; v < VALU / 2; v++) asm volatile("v_xor_b32 %0, %1, %0" : "+v"(vi[v % 8]) : "v"(vy));
      }
#endif
    }
    if constexpr (VM >= 2) {
      static_assert(NA == 4 && NB == 4, "the readout is the fold's: a 4 x 4 wave tile");
      // A chunk ends every 4 slabs (rank 128, 32 bytes a slab): the fold's readout of RDNA 4 (PEARL_AMD_KWRAP).
      if ((it & 3u) == 3u) {
        const uint32_t c = it >> 2;
        if ((c & 15u) == 0u) {
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
              keepLo[i][q] = peak_rotl13(keepLo[i][q]);
              keepHi[i][q] = peak_rotl13(keepHi[i][q]);
            }
        }
        const bool mine = (c & 7u) == r8;
        uint32_t v[2][2];
#pragma unroll
        for (int rb = 0; rb < 2; rb++) {
          uint32_t x[2] = {0u, 0u};
#pragma unroll
          for (int i = 2 * rb; i < 2 * rb + 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
              for (int g = 0; g < 8; g++) x[g >> 2] ^= (uint32_t)acc[i][j][g];
#pragma unroll
          for (int q = 0; q < 2; q++) v[rb][q] = VM == 2 ? peak_group_xor(x[q]) : x[q];
        }
        if constexpr (VM == 2) {
          if (c & 8u) {
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
              for (int q = 0; q < 2; q++) keepHi[i][q] ^= mine ? v[i][q] : 0u;
          } else {
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
              for (int q = 0; q < 2; q++) keepLo[i][q] ^= mine ? v[i][q] : 0u;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int q = 0; q < 2; q++) keepLo[i][q] ^= v[i][q];
        }
      }
    }
    if constexpr (BAR) __syncthreads();
    // The LDS reads can't be hoisted out of the loop: as far as the compiler knows, memory changed.
    if constexpr (SRC == 1) asm volatile("" ::: "memory");
#if PEAK_GFX11
    const uint32_t now = peak_cycles();
    cyc += (now - last) & 0xFFFFFu;
    last = now;
#endif
  }
  const uint32_t c1 = peak_cycles();
  const uint32_t r1 = peak_realtime();
  const uint32_t hw1 = peak_hw_id1();
  uint32_t x = vx;
#pragma unroll
  for (int i = 0; i < 8; i++) x ^= vi[i] * (uint32_t)(3 + 2 * i);
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int q = 0; q < 2; q++) x ^= keepLo[i][q] * 5u ^ keepHi[i][q] * 7u;
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++)
#pragma unroll
      for (int g = 0; g < 8; g++) x ^= (uint32_t)acc[i][j][g] * (uint32_t)(1 + 2 * (i * NB + j + g));
  out[(size_t)blockIdx.x * kThreads + t] = x;
  if (lane == 0) {
#if PEAK_GFX11
    const uint32_t cycles = cyc + ((c1 - last) & 0xFFFFFu);
#else
    const uint32_t cycles = c1 - c0;
#endif
    uint32_t *o = st + ((size_t)blockIdx.x * kWaves + (size_t)wave) * kStampWords;
    o[0] = 0x50454b31u;   // "PEK1": a written record
    o[1] = cycles;
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
  int bpw;            // blocks a WGP: 2 waves a SIMD each
  const char *data;   // miner, zero or rand
};

#define K(S, A, B, V, R) reinterpret_cast<Kernel>(&peak_kernel<S, A, B, V, R, 0>)
#define KV(S, A, B, V, R, M) reinterpret_cast<Kernel>(&peak_kernel<S, A, B, V, R, M>)
const Cfg kCfgs[] = {
    // The fold's shape: a 4 x 4 wave tile, 6 waves a SIMD, miner operands.
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 3, "miner"},
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 3, "zero"},
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 3, "rand"},
    // Occupancy: 2, 4 and 8 waves a SIMD.
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 1, "miner"},
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 2, "miner"},
    {"regs4x4", K(0, 4, 4, 0, 0), 0, 4, 4, 4, "miner"},
    // Fewer accumulators a wave: how many WMMAs must be in flight.
    {"regs1x1", K(0, 1, 1, 0, 0), 0, 1, 1, 3, "miner"},
    {"regs1x2", K(0, 1, 2, 0, 0), 0, 1, 2, 3, "miner"},
    {"regs2x2", K(0, 2, 2, 0, 0), 0, 2, 2, 3, "miner"},
    {"regs2x4", K(0, 2, 4, 0, 0), 0, 2, 4, 3, "miner"},
    // VALU beside the WMMAs: 16, 64 and 256 a slab of 32 WMMAs.
    {"valu16", K(0, 4, 4, 16, 0), 0, 4, 4, 3, "miner"},
    {"valu64", K(0, 4, 4, 64, 0), 0, 4, 4, 3, "miner"},
    {"valu256", K(0, 4, 4, 256, 0), 0, 4, 4, 3, "miner"},
    // The fold's LDS fragment reads, then its barrier.
    {"lds4x4", K(1, 4, 4, 0, 0), 1, 4, 4, 3, "miner"},
    {"ldsbar4x4", K(1, 4, 4, 0, 1), 1, 4, 4, 3, "miner"},
    {"bar4x4", K(0, 4, 4, 0, 1), 0, 4, 4, 3, "miner"},
    // VALU with no chain: 4, 16 and 64 a slab.
    {"valui4", KV(0, 4, 4, 4, 0, 1), 0, 4, 4, 3, "miner"},
    {"valui16", KV(0, 4, 4, 16, 0, 1), 0, 4, 4, 3, "miner"},
    {"valui64", KV(0, 4, 4, 64, 0, 1), 0, 4, 4, 3, "miner"},
    {"valu4", K(0, 4, 4, 4, 0), 0, 4, 4, 3, "miner"},
    // The fold's readout, then without its cross-lane steps; then on the LDS-fed loop with the barrier.
    {"rdout", KV(0, 4, 4, 0, 0, 2), 0, 4, 4, 3, "miner"},
    {"rdlocal", KV(0, 4, 4, 0, 0, 3), 0, 4, 4, 3, "miner"},
    {"ldsrd", KV(1, 4, 4, 0, 1, 2), 1, 4, 4, 3, "miner"},
};
#undef K
#undef KV

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return (uint32_t)g_rng;
}

void fill(std::vector<int8_t> &v, const char *data) {
  if (!strcmp(data, "zero")) { std::fill(v.begin(), v.end(), (int8_t)0); return; }
  for (auto &b : v) {
    if (!strcmp(data, "rand")) { b = (int8_t)(next_rand() & 255u); continue; }
    // The fold test's miner operands, roughly: 48 plus the difference of two draws in [-32, 32).
    const int x = 48 + (int)(next_rand() & 63u) - 32 - ((int)(next_rand() & 63u) - 32);
    b = (int8_t)std::max(-128, std::min(127, x));
  }
}

bool ok(hipError_t e, const char *what) {
  if (e == hipSuccess) return true;
  fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
  return false;
}

void launch(const Cfg &c, int blocks, size_t lds, const int8_t *ops, uint32_t iters, uint32_t *out, uint32_t *st) {
  void *args[] = {&ops, &iters, &out, &st};
  (void)hipLaunchKernel(reinterpret_cast<const void *>(c.k), dim3(blocks), dim3(kThreads), args, lds, 0);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 2.5;
  const char *only = argc > 2 && argv[2][0] ? argv[2] : nullptr;
  hipDeviceProp_t prop;
  if (!ok(hipGetDeviceProperties(&prop, 0), "device")) return 1;
  int wallKhz = 0, ldsWgp = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0) != hipSuccess) wallKhz = 0;
  if (hipDeviceGetAttribute(&ldsWgp, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 0) != hipSuccess) ldsWgp = 0;
  const int wgps = prop.multiProcessorCount, cus = 2 * wgps;
  printf("device: %s, %s, %d WGPs (%d CUs), %d MHz top clock (HIP), REALTIME %d kHz, LDS a WGP %d bytes\n", prop.name,
         prop.gcnArchName, wgps, cus, prop.clockRate / 1000, wallKhz, ldsWgp);
  if (wallKhz <= 0) wallKhz = 100000;

  std::vector<int8_t> host(64 * kPerBlock);
  int8_t *dOps = nullptr;
  uint32_t *dOut = nullptr, *dSt = nullptr;
  const int maxBlocks = wgps * 8;
  if (!ok(hipMalloc(&dOps, host.size()), "malloc ops") ||
      !ok(hipMalloc(&dOut, (size_t)maxBlocks * kThreads * 4), "malloc out") ||
      !ok(hipMalloc(&dSt, (size_t)maxBlocks * kWaves * kStampWords * 4), "malloc stamps"))
    return 1;
  std::string loaded;

  for (const Cfg &c : kCfgs) {
    if (only && !strstr(c.name, only)) continue;
    if (loaded != c.data) {
      fill(host, c.data);
      if (!ok(hipMemcpy(dOps, host.data(), host.size(), hipMemcpyHostToDevice), "copy ops")) return 1;
      loaded = c.data;
    }
    // LDS a block, so that `bpw` blocks fit on a WGP and no more: the most that still lets bpw fit, and at least
    // the fold's 36,864 bytes when the kernel reads it.
    hipFuncAttributes fa;
    const int vgprs = hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(c.k)) == hipSuccess ? fa.numRegs : -1;
    size_t lds = 0;
    int fit = 0;
    for (size_t l = 65536; l >= 1024; l -= 512) {
      int n = 0;
      if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, reinterpret_cast<const void *>(c.k), kThreads, l) !=
          hipSuccess)
        continue;
      if (n <= c.bpw) { lds = l; fit = n; }
      if (n > c.bpw) break;
    }
    if (c.src == 1 && lds < (size_t)kFoldLds) lds = kFoldLds;
    if (lds == 0) lds = 65536;
    const int blocks = wgps * c.bpw;
    // Iterations for a launch of about 40 ms, from a short one.
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
    // Run for `secs`; time all but the first third, launch by launch.
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
    // The last launch's stamps: each wave's clock, and how many blocks each WGP held.
    std::vector<uint32_t> s((size_t)blocks * kWaves * kStampWords);
    if (!ok(hipMemcpy(s.data(), dSt, s.size() * 4, hipMemcpyDeviceToHost), "stamps")) return 1;
    std::vector<double> mhz;
    std::vector<int> perWgp(64 * 4 * 16, 0);
    int bad = 0, moved = 0;
    for (int b = 0; b < blocks; b++)
      for (int w = 0; w < kWaves; w++) {
        const uint32_t *r = &s[((size_t)b * kWaves + w) * kStampWords];
        if (r[0] != 0x50454b31u || r[6] != (uint32_t)b) { bad++; continue; }
        if (r[4] != r[5]) { moved++; continue; }
        const double ticks = (double)(uint32_t)(r[3] - r[2]);
        if (ticks > 0) mhz.push_back((double)r[1] / (ticks / ((double)wallKhz * 1e3)) / 1e6);
        if (w == 0) {
          const uint32_t h = r[4];
          const int wgp = (int)((h >> 10) & 15u), sa = (int)((h >> 16) & 1u), se = (int)((h >> 18) & 7u);
          perWgp[(se * 2 + sa) * 16 + wgp]++;
        }
      }
    int lo = 1 << 30, hi = 0, used = 0;
    for (int n : perWgp)
      if (n) { lo = std::min(lo, n); hi = std::max(hi, n); used++; }
    const double tmac = timed > 0 ? macs / timed / 1e12 : 0.0;
    const double clk = median(mhz);
    printf("peak %-9s %-5s waves/SIMD %d, %d VGPRs, LDS %zu (fit %d), %u iters: %.2f TMAC/s (median launch %.2f, %zu "
           "launches, %.1f ms each), %.0f MHz (median of %zu waves; %.0f-%.0f), %.1f MAC/CU/clk; %d WGPs held %d-%d "
           "blocks%s\n",
           c.name, c.data, 2 * c.bpw, vgprs, lds, fit, iters, tmac, median(rates), rates.size(), ms, clk, mhz.size(),
           mhz.empty() ? 0.0 : *std::min_element(mhz.begin(), mhz.end()),
           mhz.empty() ? 0.0 : *std::max_element(mhz.begin(), mhz.end()), clk > 0 ? tmac * 1e12 / (cus * clk * 1e6) : 0.0,
           used, used ? lo : 0, hi, (bad || moved) ? (" (" + std::to_string(bad) + " bad, " + std::to_string(moved) +
                                                      " moved records)").c_str() : "");
    fflush(stdout);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
  }
  (void)hipFree(dOps);
  (void)hipFree(dOut);
  (void)hipFree(dSt);
  return 0;
}
