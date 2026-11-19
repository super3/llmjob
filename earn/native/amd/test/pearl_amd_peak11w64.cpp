// pearl_amd_peak's question for a wave64 RDNA 3 fold (g18-rdna3, 18 November). On gfx11 a wave64
// v_wmma_i32_16x16x16_iu8 takes A and B as 16 bytes a lane, as wave32 does, and C as 4 VGPRs, not 8. So in the same
// 128 accumulator VGPRs a wave64 covers 64 x 128 of C (4 x 8 tiles) against wave32's 64 x 64, and a k-step reads 12
// fragments for 32 WMMAs where wave32 reads 8 for 16: a quarter fewer LDS read instructions and unique bytes a WMMA,
// but each read delivers 1,024 bytes to registers (64 lanes x 16), not 512. On the RX 7900 XTX the fold's LDS reads
// cost about 4% of clock (pearl_amd_peak: lds4x4 against regs4x4), so which of those counts matters. Build with
// -mwavefrontsize64 for gfx1100 or gfx1101:
//
//   clang++ -x hip --offload-arch=gfx1100 -mwavefrontsize64 -O3 test/pearl_amd_peak11w64.cpp -o peak11w64
//   peak11w64 [secs=3] [only]
//
// Blocks are 256 threads (4 waves64, 2 x 2), a 128 x 256 tile, the fold's LDS ([2][128 x 48] then [2][256 x 48]).
// Each lane of a fragment read takes row lane % 16 at the k-step's 16 bytes, the same address in all four groups of
// 16 lanes (the bits are timing-only: the layout of gfx11's wave64 operands isn't read here). Variants:
//   regs    fragments in registers, two sets in turn;  lds  the reads, no barrier;  ldsbar  the reads and the slab
//   barrier. Data: the fold test's miner operands, roughly (48 plus noise). It checks nothing.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef int32_t w_v4i __attribute__((ext_vector_type(4)));
typedef int32_t w_v8i __attribute__((ext_vector_type(8)));

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#define W_GFX11 1
#else
#define W_GFX11 0
#endif

namespace {
#ifndef W11_W64
#define W11_W64 1
#endif
// -DW11_W64=0 (built without -mwavefrontsize64): the same kernels in wave32, the fold's 8 waves of 64 x 64, so the
// read variants compare in one form against the other.
constexpr int kW = W11_W64 ? 64 : 32, kThreads = 256, kS = 48, NA = 4, NB = W11_W64 ? 8 : 4, kWC = W11_W64 ? 2 : 4;
constexpr int kLds = 2 * 128 * kS + 2 * 256 * kS;   // 36,864
constexpr size_t kPerBlock = (size_t)kLds + (size_t)kThreads * 2 * (NA + NB) * 16;

#if W11_W64
typedef w_v4i w_acc;
#else
typedef w_v8i w_acc;
#endif
__device__ __forceinline__ void w_mma(w_acc &c, w_v4i a, w_v4i b) {
#if W_GFX11 && W11_W64
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c, false);
#elif W_GFX11
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
  (void)c; (void)a; (void)b;
#endif
}
__device__ __forceinline__ uint32_t w_cycles() {
#if W_GFX11
  return (uint32_t)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t w_realtime() {
#if W_GFX11
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#else
  return 0u;
#endif
}

template <int SRC, int BAR>
__global__ __launch_bounds__(kThreads) void w_kernel(const int8_t *__restrict__ ops, uint32_t iters,
                                                     uint32_t *__restrict__ out, uint32_t *__restrict__ st) {
  extern __shared__ __attribute__((aligned(16))) int8_t lds[];
  const int t = (int)threadIdx.x, lane = t % kW, wave = t / kW;
  const int wr = wave / kWC, wc = wave % kWC;
  const uint32_t r0 = w_realtime();
  uint32_t last = w_cycles(), cyc = 0u;
  const int8_t *src = ops + (size_t)(blockIdx.x & 63u) * kPerBlock;
  w_acc acc[NA][NB];
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++) acc[i][j] = w_acc{};
  w_v4i ra[2][NA], rb[2][NB];
  // SRC 4 and 5 (wave32): A's lanes that gfx11's WMMA never reads (pearl_amd_lay11.cpp: odd lanes of 0-15, even
  // lanes of 16-31) held at zero. 4: the reads, A's under an exec mask of the read lanes only, into fragments kept
  // across the loop; 5: registers, those lanes zero.
  const bool unread = ((lane & 15) & 1) != ((lane >> 4) & 1);
  w_v4i aK[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) aK[i] = w_v4i{0, 0, 0, 0};
  if constexpr (SRC == 0 || SRC == 5) {
    const w_v4i *p = reinterpret_cast<const w_v4i *>(src + kLds) + (size_t)t * 2 * (NA + NB);
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int i = 0; i < NA; i++) ra[s][i] = p[s * (NA + NB) + i];
#pragma unroll
      for (int j = 0; j < NB; j++) rb[s][j] = p[s * (NA + NB) + NA + j];
    }
    if constexpr (SRC == 5) {
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int i = 0; i < NA; i++) if (unread) ra[s][i] = w_v4i{0, 0, 0, 0};
    }
  } else {
    for (int i = 16 * t; i < kLds; i += 16 * kThreads)
      *reinterpret_cast<w_v4i *>(lds + i) = *reinterpret_cast<const w_v4i *>(src + i);
    __syncthreads();
  }
  for (uint32_t it = 0; it < iters; it++) {
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      w_v4i a[NA], b[NB];
      if constexpr (SRC == 0 || SRC == 5) {
#pragma unroll
        for (int i = 0; i < NA; i++) a[i] = ra[kk][i];
#pragma unroll
        for (int j = 0; j < NB; j++) b[j] = rb[kk][j];
      } else {
        const int buf = (int)(it & 1u);
        const int8_t *aBase = lds + buf * 128 * kS + wr * 64 * kS;
        const int8_t *bBase = lds + 2 * 128 * kS + buf * 256 * kS + wc * (16 * NB) * kS;
        // SRC 1: every group of 16 lanes at the same address (the fold's); 2: each group at its own 16 bytes of the
        // row (distinct addresses, more bank reads, the same bytes delivered); 3: groups past the first all at
        // lane 0's address (one broadcast).
        const int g = lane >> 4;
        const int off = SRC == 3 && g ? kk * 16 : (lane & 15) * kS + (SRC == 2 ? 16 * ((kk + g) % 3) : kk * 16);
        // SRC 6: A's unread lanes take their read neighbour's row (lanes 0-15 row r & ~1, lanes 16-31 row r | 1), so
        // each pair of lanes holds one row; B as the fold reads it. No instruction more than the fold's.
        const int offA = SRC == 6 ? (g ? ((lane & 15) | 1) : ((lane & 15) & ~1)) * kS + kk * 16 : off;
        if constexpr (SRC == 6) {
#pragma unroll
          for (int i = 0; i < NA; i++) a[i] = *reinterpret_cast<const w_v4i *>(aBase + i * 16 * kS + offA);
        } else if constexpr (SRC == 4) {
          if (!unread) {
#pragma unroll
            for (int i = 0; i < NA; i++) aK[i] = *reinterpret_cast<const w_v4i *>(aBase + i * 16 * kS + off);
          }
#pragma unroll
          for (int i = 0; i < NA; i++) a[i] = aK[i];
        } else {
#pragma unroll
          for (int i = 0; i < NA; i++) a[i] = *reinterpret_cast<const w_v4i *>(aBase + i * 16 * kS + off);
        }
#pragma unroll
        for (int j = 0; j < NB; j++) b[j] = *reinterpret_cast<const w_v4i *>(bBase + j * 16 * kS + off);
      }
#pragma unroll
      for (int i = 0; i < NA; i++)
#pragma unroll
        for (int j = 0; j < NB; j++) w_mma(acc[i][j], a[i], b[j]);
    }
    if constexpr (BAR) {
#if W_GFX11
#pragma unroll
      for (int i = 0; i < NA; i++)
#pragma unroll
        for (int j = 0; j < NB; j++) asm volatile("" ::"v"(acc[i][j]));
      __builtin_amdgcn_sched_barrier(0);
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#endif
    }
    const uint32_t c = w_cycles();
    cyc += (c - last) & 0xFFFFFu;
    last = c;
  }
  const uint32_t r1 = w_realtime();
  uint32_t x = 0u;
#pragma unroll
  for (int i = 0; i < NA; i++)
#pragma unroll
    for (int j = 0; j < NB; j++) x ^= (uint32_t)(acc[i][j][0] ^ acc[i][j][1] ^ acc[i][j][2] ^ acc[i][j][3]);
  out[blockIdx.x * kThreads + t] = x;
  if (lane == 0) {
    st[(blockIdx.x * (kThreads / kW) + wave) * 2] = cyc;
    st[(blockIdx.x * (kThreads / kW) + wave) * 2 + 1] = r1 - r0;
  }
}

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t next_rand() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }

struct V { const char *name; const void *k; int src; };
}  // namespace

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 3.0;
  const char *only = argc > 2 && argv[2][0] ? argv[2] : nullptr;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { fprintf(stderr, "no device\n"); return 2; }
  const int wgps = prop.multiProcessorCount;
  int wallKhz = 100000;
  (void)hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0);
  printf("device: %s, %s, %d WGPs, wave size %d\n", prop.name, prop.gcnArchName, wgps, prop.warpSize);
  const V vs[] = {{"regs", reinterpret_cast<const void *>(&w_kernel<0, 0>), 0},
                  {"lds", reinterpret_cast<const void *>(&w_kernel<1, 0>), 1},
                  {"ldsbar", reinterpret_cast<const void *>(&w_kernel<1, 1>), 1},
                  {"ldsdist", reinterpret_cast<const void *>(&w_kernel<2, 0>), 1},
                  {"ldsone", reinterpret_cast<const void *>(&w_kernel<3, 0>), 1},
                  {"ldsah", reinterpret_cast<const void *>(&w_kernel<4, 0>), 1},
                  {"ldsahbar", reinterpret_cast<const void *>(&w_kernel<4, 1>), 1},
                  {"regz", reinterpret_cast<const void *>(&w_kernel<5, 0>), 0},
                  {"ldsp", reinterpret_cast<const void *>(&w_kernel<6, 0>), 1},
                  {"ldspbar", reinterpret_cast<const void *>(&w_kernel<6, 1>), 1}};
  std::vector<int8_t> host(64 * kPerBlock);
  for (auto &b : host) {
    const int x = 48 + (int)(next_rand() & 63u) - 32 - ((int)(next_rand() & 63u) - 32);
    b = (int8_t)std::max(-128, std::min(127, x));
  }
  int8_t *ops = nullptr;
  uint32_t *out = nullptr, *st = nullptr;
  // The grid is exactly blocks a WGP x WGPs. The register variant asks for LDS so that only that many fit; the LDS
  // variants take the fold's 36,864 bytes (3 fit a WGP, as the fold's blocks do), so registers set the rest.
  for (int bpw : {1, 2, 3, 4}) {
    for (const V &v : vs) {
      if (only && !strstr(v.name, only)) continue;
      if (bpw != 3) continue;   // the LDS variants as the fold runs: 3 blocks a WGP
      const size_t ldsAsk = v.src ? (size_t)kLds : (size_t)(65536 / bpw - 256);
      const int blocks = bpw * wgps;
      if (!ops) {
        if (hipMalloc(&ops, host.size()) != hipSuccess || hipMalloc(&out, (size_t)4 * wgps * kThreads * 4) != hipSuccess ||
            hipMalloc(&st, (size_t)4 * wgps * 8 * 2 * 4) != hipSuccess) { fprintf(stderr, "malloc\n"); return 2; }
        (void)hipMemcpy(ops, host.data(), host.size(), hipMemcpyHostToDevice);
      }
      if (hipFuncSetAttribute(v.k, hipFuncAttributeMaxDynamicSharedMemorySize, (int)ldsAsk) != hipSuccess) { /* fine */ }
      hipFuncAttributes fa;
      (void)hipFuncGetAttributes(&fa, v.k);
      uint32_t iters = 2000;
      double el = 0;
      for (int rep = 0; rep < 8; rep++) {
        void *args[] = {&ops, &iters, &out, &st};
        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        (void)hipLaunchKernel(v.k, dim3(blocks), dim3(kThreads), args, ldsAsk, 0);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        el = ms / 1e3;
        if (el > 0.03) break;
        iters *= 2;
      }
      const int launches = std::max(1, (int)(secs / el));
      std::vector<double> rates, clocks;
      std::vector<uint32_t> hs((size_t)blocks * (kThreads / kW) * 2);
      for (int l = 0; l < launches; l++) {
        void *args[] = {&ops, &iters, &out, &st};
        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        (void)hipLaunchKernel(v.k, dim3(blocks), dim3(kThreads), args, ldsAsk, 0);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        const double macs = (double)blocks * 128.0 * 256.0 * 32.0 * iters;
        rates.push_back(macs / (ms / 1e3) / 1e12);
        if (l == launches / 2) {
          (void)hipMemcpy(hs.data(), st, hs.size() * 4, hipMemcpyDeviceToHost);
          for (size_t w = 0; w < (size_t)blocks * (kThreads / kW); w++)
            if (hs[2 * w + 1]) clocks.push_back((double)hs[2 * w] / ((double)hs[2 * w + 1] / (wallKhz * 1e3)) / 1e6);
        }
      }
      std::sort(rates.begin(), rates.end());
      std::sort(clocks.begin(), clocks.end());
      const double r = rates[rates.size() / 2], mhz = clocks.empty() ? 0 : clocks[clocks.size() / 2];
      printf("w%d %-7s blocks/WGP %d, %d VGPRs, LDS %zu: %.2f TMAC/s (median of %d launches), %.0f MHz, %.1f MAC/CU/clk\n",
             kW, v.name, bpw, (int)fa.numRegs, ldsAsk, r, launches, mhz, mhz > 0 ? r * 1e12 / (2.0 * wgps) / (mhz * 1e6) : 0.0);
      fflush(stdout);
    }
  }
  return 0;
}
