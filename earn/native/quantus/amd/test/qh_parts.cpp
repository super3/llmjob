// The Quantus kernel's parts timed one at a time on a card, for finding where a nonce's cycles go. Timing only.
//
//   qh_parts [device]     built by the box job with the kernel's own flags (QH_CFLAGS), e.g.
//                         clang++ -x hip --offload-arch=gfx1100 -O3 -I.. $QH_CFLAGS qh_parts.cpp -lamdhip64
//
// Each kernel runs one part of qh_kernel.h (a field multiply, a square, an S-box, a multiply-add by the diagonal,
// the external layer, the internal layer, a partial round, a full round) over and over, on 8 independent values a
// thread (the layers: on one state of 12), from many waves a SIMD, as the miner runs. It prints SIMD-cycles per
// part: the card's cycles times its SIMDs over the waves' part count, so a part whose instructions all issue
// back to back costs its instruction count (a v_mad_u64_u32 four). The clock is read by one wave (SHADER_CYCLES
// against REALTIME) right after each kernel. gfx11 and gfx12.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../qh_kernel.h"

#define CHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)
#define CHAINS 8

__global__ void __launch_bounds__(32) k_clk(u32 steps, u32 *out) {
#if defined(__GFX11__) || defined(__GFX12__)
  const u32 r0 = (u32)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#if defined(__GFX11__)
  u32 last = (u32)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;
#else
  u32 last = (u32)__builtin_amdgcn_s_getreg(0xF81D);
#endif
  u32 sum = 0, x = threadIdx.x + 1;
  for (u32 s = 0; s < steps; s++) {
    for (int i = 0; i < 256; i++) { x = x * 1664525u + 1013904223u; asm volatile("" : "+v"(x)); }
#if defined(__GFX11__)
    u32 c = (u32)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;
    sum += (c - last) & 0xFFFFFu;
#else
    u32 c = (u32)__builtin_amdgcn_s_getreg(0xF81D);
    sum += c - last;
#endif
    last = c;
  }
  const u32 r1 = (u32)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  if (threadIdx.x == 0) { out[0] = r1 - r0; out[1] = sum; out[2] = x; }
#else
  if (threadIdx.x == 0) { out[0] = 0; out[1] = 0; out[2] = steps; }
#endif
}

__device__ __forceinline__ u64 qp_next(u64 &t) {
  t = t * 6364136223846793005ull + 1442695040888963407ull;
  return t;
}

// Eight values a thread, each through OP every pass.
#define K8(NAME, BODY)                                                                     \
  __global__ void __launch_bounds__(256) NAME(u64 *out, u64 seed, u32 iters) {           \
    u64 t = seed ^ ((u64)blockIdx.x * 256 + threadIdx.x);                                 \
    u64 x[CHAINS], y[CHAINS];                                                             \
    for (int i = 0; i < CHAINS; i++) { x[i] = qp_next(t) >> 1; y[i] = qp_next(t) >> 1; }  \
    for (u32 it = 0; it < iters; it++) {                                                  \
      _Pragma("unroll") for (int i = 0; i < CHAINS; i++) { BODY; }                        \
    }                                                                                     \
    u64 r = 0;                                                                            \
    for (int i = 0; i < CHAINS; i++) r ^= x[i];                                           \
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;                                       \
  }

K8(k_mul, x[i] = qh_mul(x[i], y[i]))
K8(k_sqr, x[i] = qh_sqr(x[i]))
K8(k_sbox, x[i] = qh_sbox(x[i]))
K8(k_muladd, { QhWide w; w.lo = y[i]; w.h = (u32)y[i] & 7u; x[i] = qh_muladd(x[i], qh_tab.diag[i], w); })

// One state of 12 a thread, through a layer every pass.
#define K12(NAME, BODY)                                                                    \
  __global__ void __launch_bounds__(256) NAME(u64 *out, u64 seed, u32 iters) {           \
    u64 t = seed ^ ((u64)blockIdx.x * 256 + threadIdx.x);                                 \
    u64 s[12];                                                                            \
    for (int i = 0; i < 12; i++) s[i] = qp_next(t) >> 1;                                  \
    for (u32 it = 0; it < iters; it++) { BODY; }                                          \
    u64 r = 0;                                                                            \
    for (int i = 0; i < 12; i++) r ^= s[i];                                               \
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;                                       \
  }

K12(k_ext, qh_external(s, qh_tab.full[it & 7]))
K12(k_int, qh_internal(s, qh_tab.diag, qh_tab.part0[it & 15]))
K12(k_partial, { s[0] = qh_sbox(s[0]); qh_internal(s, qh_tab.diag, qh_tab.part0[it & 15]); })
__device__ __forceinline__ void qp_round(u64 *s, const u64 *c) {
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = qh_sbox(s[i]);
  qh_external(s, c);
}
K12(k_round, qp_round(s, qh_tab.full[it & 7]))

typedef void (*kfn)(u64 *, u64, u32);
struct K { const char *name; kfn f; u32 iters; int per; };  // per: parts a thread a pass

int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHK(hipSetDevice(dev));
  hipDeviceProp_t p;
  CHK(hipGetDeviceProperties(&p, dev));
  int wgps = p.multiProcessorCount;  // HIP counts WGPs on RDNA 3 and 4 (4 SIMD32s each)
  int simds = 4 * wgps;
  printf("device %d: %s %s, %d WGPs, %d SIMDs\n", dev, p.name, p.gcnArchName, wgps, simds);
  // Constants: any values do for timing.
  static QhTables h;
  u64 t = 0x9e3779b97f4a7c15ull;
  u64 *hw = (u64 *)&h;
  for (size_t i = 0; i < sizeof(h) / 8; i++) { t = t * 6364136223846793005ull + 1442695040888963407ull; hw[i] = t >> 1; }
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(qh_tab), &h, sizeof h));
  int blocks = wgps * 64, threads = 256;
  u64 *out;
  u32 *cko;
  CHK(hipMalloc(&out, (size_t)blocks * threads * 8));
  CHK(hipMalloc(&cko, 16));
  int wallKhz = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || wallKhz <= 0) wallKhz = 100000;
  (void)hipGetLastError();
  auto clk_mhz = [&]() {
    hipLaunchKernelGGL(k_clk, dim3(1), dim3(32), 0, 0, 64u, cko);
    CHK(hipDeviceSynchronize());
    u32 hh[3];
    CHK(hipMemcpy(hh, cko, 12, hipMemcpyDeviceToHost));
    double secs = (double)hh[0] / ((double)wallKhz * 1e3);
    return secs > 0 ? (double)hh[1] / secs / 1e6 : 0.0;
  };
  K ks[] = {
    {"warm-up (mul)", k_mul, 1000, CHAINS}, {"field multiply", k_mul, 1000, CHAINS}, {"square", k_sqr, 1000, CHAINS},
    {"S-box x^7", k_sbox, 250, CHAINS}, {"diagonal multiply-add", k_muladd, 1000, CHAINS},
    {"external layer", k_ext, 500, 1}, {"internal layer", k_int, 400, 1}, {"partial round", k_partial, 300, 1},
    {"full round", k_round, 100, 1},
  };
  hipEvent_t e0, e1;
  CHK(hipEventCreate(&e0));
  CHK(hipEventCreate(&e1));
  for (auto &k : ks) {
    double best = 1e30, mhz = 0;
    for (int rep = 0; rep < 4; rep++) {
      CHK(hipEventRecord(e0));
      hipLaunchKernelGGL(k.f, dim3(blocks), dim3(threads), 0, 0, out, 12345ull + rep, k.iters);
      CHK(hipEventRecord(e1));
      CHK(hipEventSynchronize(e1));
      float ms = 0;
      CHK(hipEventElapsedTime(&ms, e0, e1));
      double m = clk_mhz();
      if (rep > 0 && ms < best) { best = ms; mhz = m; }
    }
    double waves = (double)blocks * threads / 32.0;
    double parts = waves * k.iters * k.per;                  // wave-parts
    double cyc = best * 1e-3 * mhz * 1e6 * simds / parts;    // SIMD-cycles a wave-part
    printf("%-22s %8.3f ms  %5.0f MHz  %8.1f SIMD-cycles a part\n", k.name, best, mhz, cyc);
    fflush(stdout);
  }
  return 0;
}
