// What one instruction beside the WMMAs costs on RDNA 4 (goal-rdna4f). Timing only; it checks nothing.
//
//   pearl_amd_cost [secs=2] [name filter]
//
// Every kernel runs the fold's slab from registers: 32 v_wmma_i32_16x16x16_iu8 (two k-steps over a 4 x 4 wave tile,
// serpentine), 256 threads a block, 4 blocks a WGP (8 waves a SIMD, as the fold runs), one set of fragments. Each
// variant adds N instructions of one kind to every slab, at one place:
//   PLACE 0  all N before the slab's first WMMA        PLACE 1  all N after its last WMMA
//   PLACE 2  spread: after every 32 / N WMMAs           (N > 32: N / 32 after each WMMA)
// Kinds (the instructions are inline asm, so the compiler neither drops nor moves them; a sched_barrier after every
// WMMA keeps the order; loads go to registers no WMMA reads, so only their issue and write-back are priced, not
// their latency):
//   1 ds_load_b128   2 ds_load_b64   3 buffer_load_b128 (an L2-resident 64 KB a block)   4 ds_store_b128
//   5 v_xor_b32 on 8 registers in turn (independent)      6 v_xor_b32 chain (each needs the one before)
//   7 s_add_u32 (SALU)   8 v_xor3_b32 of two accumulators the WMMA just before wrote (the readout's input)
//   9 v_xor3_b32 of two accumulators written 8 WMMAs before   10 the chain of 6 with s_delay_alu VALU_DEP_1
//   11 v_xor3_b32 on 8 registers in turn (independent, no accumulator)
// BAR 1 adds the fold's slab barrier (s_barrier_signal / s_barrier_wait) after the WMMAs.
// MAC/CU/clk is the rate over the CUs and the clock the waves read (SHADER_CYCLES over REALTIME). 1,024 is AMD's.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef int32_t c_v8i __attribute__((ext_vector_type(8)));
typedef int32_t c_v4i __attribute__((ext_vector_type(4)));
typedef int32_t c_v2i __attribute__((ext_vector_type(2)));

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
#define C_G12 1
#else
#define C_G12 0
#endif

namespace {

constexpr int kThreads = 256, kWaves = 8, kStampWords = 8, kRegion = 65536;

__device__ __forceinline__ uint32_t c_cycles() {
#if C_G12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);   // SHADER_CYCLES_LO
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t c_realtime() {
#if C_G12
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t c_hw_id1() {
#if C_G12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);
#else
  return 0u;
#endif
}

// One instruction of kind K, number j of the slab, w the WMMA it follows (for 8 and 9).
template <int K>
__device__ __forceinline__ void c_op(int j, int w, c_v4i (&sk)[4], uint32_t (&vi)[8], uint32_t &vx, uint32_t vy,
                                     uint32_t &sx, uint32_t lA, uint32_t gOff, c_v4i rs, c_v8i (&acc)[16]) {
#if C_G12
  if constexpr (K == 1) {
    asm volatile("ds_load_b128 %0, %1 offset:%2" : "+v"(sk[j & 3]) : "v"(lA), "i"(512 * (j & 3)));
  } else if constexpr (K == 2) {
    asm volatile("ds_load_b64 %0, %1 offset:%2" : "+v"(*reinterpret_cast<c_v2i *>(&sk[j & 3])) : "v"(lA), "i"(512 * (j & 3)));
  } else if constexpr (K == 3) {
    asm volatile("buffer_load_b128 %0, %1, %2, null offen offset:%3" : "+v"(sk[j & 3]) : "v"(gOff), "s"(rs), "i"(512 * (j & 7)));
  } else if constexpr (K == 4) {
    asm volatile("ds_store_b128 %0, %1 offset:%2" ::"v"(lA), "v"(sk[j & 3]), "i"(512 * (j & 3)));
  } else if constexpr (K == 5) {
    asm volatile("v_xor_b32 %0, %1, %0" : "+v"(vi[j & 7]) : "v"(vy));
  } else if constexpr (K == 6) {
    asm volatile("v_xor_b32 %0, %1, %0" : "+v"(vx) : "v"(vy));
  } else if constexpr (K == 7) {
    asm volatile("s_add_co_u32 %0, %0, 3" : "+s"(sx));
  } else if constexpr (K == 8 || K == 9) {
    // The serpentine order's w-th WMMA writes tile ser(w); 8 reads that one, 9 the one 8 WMMAs before.
    const int ww = K == 8 ? w : (w + 24) % 32;
    const int i = (ww % 16) / 4, jj = ((ww % 16) / 4) & 1 ? 3 - (ww % 4) : (ww % 4);
    const int g = (2 * j) & 7;
    asm volatile("v_xor3_b32 %0, %1, %2, %0" : "+v"(vi[j & 7]) : "v"(acc[i * 4 + jj][g]), "v"(acc[i * 4 + jj][g + 1]));
  } else if constexpr (K == 10) {
    asm volatile("s_delay_alu instid0(VALU_DEP_1)\n\tv_xor_b32 %0, %1, %0" : "+v"(vx) : "v"(vy));
  } else if constexpr (K == 11) {
    asm volatile("v_xor3_b32 %0, %1, %2, %0" : "+v"(vi[j & 7]) : "v"(vy), "v"(vx));
  } else if constexpr (K == 12) {
    asm volatile("s_nop 0");
  } else if constexpr (K == 13) {
    asm volatile("s_wait_dscnt 0x0");
  } else if constexpr (K == 14) {
    // 14: one LDS read and a wait for it right away (the data unused): the latency in the run.
    asm volatile("ds_load_b128 %0, %1 offset:%2\n\ts_wait_dscnt 0x0" : "+v"(sk[j & 3]) : "v"(lA), "i"(512 * (j & 3)));
  }
#else
  (void)j; (void)w; (void)sk; (void)vi; (void)vx; (void)vy; (void)sx; (void)lA; (void)gOff; (void)rs; (void)acc;
#endif
}

template <int K, int N, int PLACE, int BAR>
__global__ __launch_bounds__(kThreads) void cost_kernel(const int8_t *__restrict__ ops, int8_t *__restrict__ gbuf,
                                                         uint32_t iters, uint32_t *__restrict__ out,
                                                         uint32_t *__restrict__ st) {
  extern __shared__ __attribute__((aligned(16))) int8_t lds[];
  const int t = (int)threadIdx.x, lane = t & 31, wave = t >> 5;
  const uint32_t hw0 = c_hw_id1();
  const uint32_t c0 = c_cycles();
  const uint32_t r0 = c_realtime();
  const int8_t *src = ops + (size_t)(blockIdx.x & 63u) * 4096u;
  c_v2i fa[2][4], fb[2][4];
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      fa[h][i] = *reinterpret_cast<const c_v2i *>(src + ((t * 16 + h * 8 + i) & 511) * 8);
      fb[h][i] = *reinterpret_cast<const c_v2i *>(src + ((t * 16 + h * 8 + 4 + i) & 511) * 8);
    }
  for (int i = 16 * t; i < 16384; i += 16 * kThreads)
    *reinterpret_cast<c_v4i *>(lds + i) = *reinterpret_cast<const c_v4i *>(src + (i & 4095));
  __syncthreads();
  c_v8i acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = c_v8i{0, 0, 0, 0, 0, 0, 0, 0};
  c_v4i sk[4];
#pragma unroll
  for (int i = 0; i < 4; i++) sk[i] = c_v4i{t, t + 1, t + 2, t + 3};
  uint32_t vi[8];
#pragma unroll
  for (int i = 0; i < 8; i++) vi[i] = (uint32_t)t * 2654435761u + (uint32_t)i;
  uint32_t vx = (uint32_t)t, vy = (uint32_t)lane + 1u, sx = blockIdx.x;
  uint32_t lA = (uint32_t)(wave * 2048 + lane * 16);
  uint32_t gOff = (uint32_t)(lane * 16);
  asm volatile("" : "+v"(lA), "+v"(gOff));
  c_v4i rs = {0, 0, 0, 0};
#if C_G12
  {
    const uint64_t p = (uint64_t)(gbuf + (size_t)(blockIdx.x & 63u) * kRegion);
    rs = c_v4i{__builtin_amdgcn_readfirstlane((int)(uint32_t)p), __builtin_amdgcn_readfirstlane((int)(uint32_t)(p >> 32)) & 0xffff,
               kRegion, 0x31004000};
  }
#endif
  constexpr int kEvery = N >= 32 ? 1 : (N > 0 ? 32 / N : 64);
  constexpr int kPer = N >= 32 ? N / 32 : 1;
  for (uint32_t it = 0; it < iters; it++) {
    int j = 0;
    if constexpr (PLACE == 0) {
#pragma unroll
      for (int q = 0; q < N; q++) c_op<K>(q, 31, sk, vi, vx, vy, sx, lA, gOff, rs, acc);
#if C_G12
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
#pragma unroll
    for (int w = 0; w < 32; w++) {
      // Serpentine over the 4 x 4 tiles: row i = (w % 16) / 4, columns forward on even rows, back on odd.
      const int h = w / 16, i = (w % 16) / 4, jj = (i & 1) ? 3 - (w % 4) : (w % 4);
#if C_G12
      acc[i * 4 + jj] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, fa[h][i], true, fb[h][jj], acc[i * 4 + jj], false);
      __builtin_amdgcn_sched_barrier(0);
#endif
      if constexpr (PLACE == 2) {
        if ((w % kEvery) == kEvery - 1) {
#pragma unroll
          for (int q = 0; q < kPer; q++) {
            if (j < N) c_op<K>(j, w, sk, vi, vx, vy, sx, lA, gOff, rs, acc);
            j++;
          }
#if C_G12
          __builtin_amdgcn_sched_barrier(0);
#endif
        }
      }
    }
    if constexpr (PLACE == 1) {
#pragma unroll
      for (int q = 0; q < N; q++) c_op<K>(q, 31, sk, vi, vx, vy, sx, lA, gOff, rs, acc);
#if C_G12
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    if constexpr (BAR) {
#if C_G12
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
  }
#if C_G12
  asm volatile("s_wait_dscnt 0x0\n\ts_wait_loadcnt 0x0" ::: "memory");
#endif
  const uint32_t c1 = c_cycles();
  const uint32_t r1 = c_realtime();
  const uint32_t hw1 = c_hw_id1();
  uint32_t x = vx ^ sx;
#pragma unroll
  for (int i = 0; i < 8; i++) x ^= vi[i] * (uint32_t)(3 + 2 * i);
#pragma unroll
  for (int i = 0; i < 4; i++) x ^= (uint32_t)(sk[i][0] ^ sk[i][1] ^ sk[i][2] ^ sk[i][3]);
#pragma unroll
  for (int i = 0; i < 16; i++)
#pragma unroll
    for (int g = 0; g < 8; g++) x ^= (uint32_t)acc[i][g] * (uint32_t)(1 + 2 * (i + g));
  out[(size_t)blockIdx.x * kThreads + t] = x;
  if (lane == 0) {
    uint32_t *o = st + ((size_t)blockIdx.x * kWaves + (size_t)wave) * kStampWords;
    o[0] = 0x434f5354u;
    o[1] = c1 - c0;
    o[2] = r0;
    o[3] = r1;
    o[4] = hw0;
    o[5] = hw1;
    o[6] = blockIdx.x;
    o[7] = (uint32_t)wave;
  }
}

typedef void (*Kernel)(const int8_t *, int8_t *, uint32_t, uint32_t *, uint32_t *);
struct Cfg {
  const char *name;
  Kernel k;
};
#define C(NAME, K, N, P, B) {NAME, reinterpret_cast<Kernel>(&cost_kernel<K, N, P, B>)}
const Cfg kCfgs[] = {
    C("base", 0, 0, 0, 0),
    C("ld128x8@0", 1, 8, 0, 0), C("ld128x8@1", 1, 8, 1, 0), C("ld128x8@2", 1, 8, 2, 0), C("ld128x16@2", 1, 16, 2, 0),
    C("ld128x4@2", 1, 4, 2, 0),
    C("ld64x8@2", 2, 8, 2, 0), C("ld64x16@2", 2, 16, 2, 0),
    C("buf128x2@2", 3, 2, 2, 0), C("buf128x8@2", 3, 8, 2, 0), C("buf128x8@0", 3, 8, 0, 0),
    C("st128x2@2", 4, 2, 2, 0), C("st128x8@2", 4, 8, 2, 0), C("st128x8@1", 4, 8, 1, 0),
    C("vxor16@0", 5, 16, 0, 0), C("vxor16@1", 5, 16, 1, 0), C("vxor16@2", 5, 16, 2, 0), C("vxor4@2", 5, 4, 2, 0),
    C("vxor64@1", 5, 64, 1, 0),
    C("vchain16@1", 6, 16, 1, 0), C("vchain16@2", 6, 16, 2, 0),
    C("vchaind16@1", 10, 16, 1, 0), C("vchaind16@2", 10, 16, 2, 0),
    C("salu16@2", 7, 16, 2, 0), C("salu64@2", 7, 64, 2, 0),
    C("xacc16@1", 8, 16, 1, 0), C("xacc16@2", 8, 16, 2, 0), C("xacc32@2", 8, 32, 2, 0),
    C("xold16@2", 9, 16, 2, 0), C("xold32@2", 9, 32, 2, 0),
    C("x3i16@1", 11, 16, 1, 0), C("x3i16@2", 11, 16, 2, 0),
    C("bar", 0, 0, 0, 1),
    C("nop8@2", 12, 8, 2, 0), C("nop32@2", 12, 32, 2, 0), C("nop8@1", 12, 8, 1, 0),
    C("wait8@2", 13, 8, 2, 0), C("wait1@2", 13, 1, 2, 0), C("wait2@2", 13, 2, 2, 0),
    C("ldw1@0", 14, 1, 0, 0), C("ldw1@2", 14, 1, 2, 0), C("ldw4@2", 14, 4, 2, 0),
    C("ld128x8b@0", 1, 8, 0, 1),
};
#undef C

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

}  // namespace

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 2.0;
  const char *only = argc > 2 && argv[2][0] ? argv[2] : nullptr;
  const int bpw = getenv("COST_BPW") ? atoi(getenv("COST_BPW")) : 4;
  hipDeviceProp_t prop;
  if (!ok(hipGetDeviceProperties(&prop, 0), "device")) return 1;
  int wallKhz = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0) != hipSuccess || wallKhz <= 0) wallKhz = 100000;
  const int wgps = prop.multiProcessorCount, cus = 2 * wgps, blocks = wgps * bpw;
  printf("device: %s, %s, %d WGPs, %d blocks a WGP\n", prop.name, prop.gcnArchName, wgps, bpw);
  std::vector<int8_t> host(64 * 4096);
  uint64_t g = 0x9E3779B97F4A7C15ull;
  for (auto &b : host) {
    g ^= g << 13; g ^= g >> 7; g ^= g << 17;
    b = (int8_t)std::max(-128, std::min(127, 48 + (int)(g & 63u) - 32 - ((int)((g >> 8) & 63u) - 32)));
  }
  int8_t *dOps = nullptr, *dG = nullptr;
  uint32_t *dOut = nullptr, *dSt = nullptr;
  if (!ok(hipMalloc(&dOps, host.size()), "malloc") || !ok(hipMalloc(&dG, (size_t)64 * kRegion), "malloc") ||
      !ok(hipMalloc(&dOut, (size_t)blocks * kThreads * 4), "malloc") ||
      !ok(hipMalloc(&dSt, (size_t)blocks * kWaves * kStampWords * 4), "malloc") ||
      !ok(hipMemcpy(dOps, host.data(), host.size(), hipMemcpyHostToDevice), "copy") ||
      !ok(hipMemset(dG, 1, (size_t)64 * kRegion), "memset"))
    return 1;
  for (const Cfg &c : kCfgs) {
    if (only && !strstr(c.name, only)) continue;
    hipFuncAttributes fa;
    const int vgprs = hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(c.k)) == hipSuccess ? fa.numRegs : -1;
    const int scratch = hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(c.k)) == hipSuccess ? (int)fa.localSizeBytes : -1;
    // LDS a block so that bpw blocks fit on a WGP and no more (at least the 16 KB the kernel uses).
    size_t lds = 16384;
    int fit = 0;
    for (size_t l = 65536; l >= 16384; l -= 1024) {
      int n = 0;
      if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, reinterpret_cast<const void *>(c.k), kThreads, l) != hipSuccess) continue;
      if (n <= bpw) { lds = l; fit = n; }
      if (n > bpw) break;
    }
    uint32_t iters = 2000;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    float ms = 0.f;
    auto launch = [&]() {
      void *args[] = {&dOps, &dG, &iters, &dOut, &dSt};
      (void)hipLaunchKernel(reinterpret_cast<const void *>(c.k), dim3(blocks), dim3(kThreads), args, lds, 0);
    };
    for (int tries = 0; tries < 6; tries++) {
      (void)hipEventRecord(e0);
      launch();
      (void)hipEventRecord(e1);
      if (!ok(hipEventSynchronize(e1), c.name)) return 1;
      (void)hipEventElapsedTime(&ms, e0, e1);
      if (ms > 30.f) break;
      iters = (uint32_t)std::min(4.0e8, iters * std::max(2.0, 40.0 / std::max(0.05f, ms)));
    }
    double timed = 0.0, total = 0.0, macs = 0.0;
    const double perLaunch = (double)blocks * kWaves * iters * 32.0 * 4096.0;
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      (void)hipEventRecord(e0);
      launch();
      (void)hipEventRecord(e1);
      if (!ok(hipEventSynchronize(e1), c.name)) return 1;
      (void)hipEventElapsedTime(&ms, e0, e1);
      total += ms / 1000.0;
      if (total > secs / 3) { timed += ms / 1000.0; macs += perLaunch; }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > secs) break;
    }
    std::vector<uint32_t> s((size_t)blocks * kWaves * kStampWords);
    if (!ok(hipMemcpy(s.data(), dSt, s.size() * 4, hipMemcpyDeviceToHost), "stamps")) return 1;
    std::vector<double> mhz, cyc;
    for (size_t i = 0; i < (size_t)blocks * kWaves; i++) {
      const uint32_t *r = &s[i * kStampWords];
      if (r[0] != 0x434f5354u || r[4] != r[5]) continue;
      const double ticks = (double)(uint32_t)(r[3] - r[2]);
      if (ticks > 0) mhz.push_back((double)r[1] / (ticks / ((double)wallKhz * 1e3)) / 1e6);
      cyc.push_back((double)r[1]);
    }
    const double tmac = timed > 0 ? macs / timed / 1e12 : 0.0, clk = median(mhz);
    // Cycles a slab a wave, from the waves' own counters: what each added instruction cost the SIMD.
    const double cps = median(cyc) / (double)iters;
    printf("cost %-12s %d VGPRs %d scratch, fit %d: %.2f TMAC/s, %.0f MHz, %.1f MAC/CU/clk, %.1f cycles a slab a wave\n",
           c.name, vgprs, scratch, fit, tmac, clk, clk > 0 ? tmac * 1e12 / (cus * clk * 1e6) : 0.0, cps);
    fflush(stdout);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
  }
  return 0;
}
