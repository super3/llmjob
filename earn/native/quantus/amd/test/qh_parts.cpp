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
// QH_ARITH 4's blocks own six VGPRs: keep the compiler off them and zero the two they read as zero, as qh_search does.
#if QH_ARITH == 4
#define QP_INIT qh_x_init()
#else
#define QP_INIT (void)0
#endif

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
  __global__ void __launch_bounds__(256) QH_ATTR NAME(u64 *out, u64 seed, u32 iters) {   \
    QP_INIT;                                                                              \
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
K8(k_sbox_e, x[i] = qh_sbox_e(x[i], qh_tab.efold[1][i]))
K8(k_muladd, { QhWide w; w.lo = y[i]; w.h = (u32)y[i] & 7u; x[i] = qh_muladd(x[i], qh_tab.diag[i], w); })

// One state of 12 a thread, through a layer every pass.
#define K12(NAME, BODY)                                                                    \
  __global__ void __launch_bounds__(256) QH_ATTR NAME(u64 *out, u64 seed, u32 iters) {   \
    QP_INIT;                                                                              \
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
// The forms the QH_ARITH 4 kernel runs: constants folded into the S-boxes (QH_EFOLD, QH_PFOLD), and the sum alone.
K12(k_ext_nc, qh_external<false>(s, nullptr))
K12(k_partial_pf, { s[0] = qh_sbox_e(s[0], qh_tab.pe[it & 15]); qh_internal<false>(s, qh_tab.diag, 0); })
__device__ __forceinline__ void qp_round_e(u64 *s, const u64 *e) {
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = qh_sbox_e(s[i], e[i]);
  qh_external<false>(s, nullptr);
}
K12(k_round_e, qp_round_e(s, qh_tab.efold[1 + (it & 3)]))
K12(k_sum, {
  QhWide sum = qh_w(s[0]);
  _Pragma("unroll") for (int i = 1; i < 12; i++) sum = qh_wadd(sum, s[i]);
  s[it % 12] = qh_reduce96(sum.lo, sum.h);
})

// Single instructions, 8 independent chains a thread, 4 a chain a pass, in the build's wave size: what one costs a
// 32 lanes in wave32 and in wave64. QP_SM is a lane-mask SGPR of the wave size.
#if defined(QH_W64) && QH_W64
#define QP_SM "s[0:1]"
#else
#define QP_SM "s0"
#endif
#define KI(NAME, ASM)                                                                      \
  __global__ void __launch_bounds__(256) NAME(u64 *out, u64 seed, u32 iters) {           \
    u32 a[8], b[8];                                                                       \
    for (int i = 0; i < 8; i++) { a[i] = (u32)seed * (threadIdx.x + 3 + i); b[i] = (u32)seed ^ (i * 0x9e3779b9u); } \
    asm volatile("s_mov_b32 s0, 0\n\ts_mov_b32 s1, 0" ::: "s0", "s1");                    \
    for (u32 it = 0; it < iters; it++) {                                                  \
      _Pragma("unroll") for (int r = 0; r < 4; r++) {                                     \
        _Pragma("unroll") for (int i = 0; i < 8; i++) asm volatile(ASM : "+v"(a[i]) : "v"(b[i]) : "s0", "s1"); \
      }                                                                                   \
    }                                                                                     \
    u32 x = 0;                                                                            \
    for (int i = 0; i < 8; i++) x ^= a[i];                                                \
    out[blockIdx.x * blockDim.x + threadIdx.x] = x;                                       \
  }
KI(k_i_add, "v_add_nc_u32 %0, %0, %1")
KI(k_i_add3, "v_add3_u32 %0, %0, %1, %0")
KI(k_i_carry, "v_add_co_u32 %0, null, %0, %1")
KI(k_i_carryin, "v_add_co_ci_u32_e64 %0, null, %0, %1, " QP_SM)
KI(k_i_cnd, "v_cndmask_b32_e64 %0, %0, %1, " QP_SM)
KI(k_i_shl, "v_lshlrev_b32 %0, 1, %0")

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
    {"S-box + e", k_sbox_e, 250, CHAINS}, {"external layer, no constants", k_ext_nc, 500, 1},
    {"partial round, folded", k_partial_pf, 300, 1}, {"full round, folded", k_round_e, 100, 1},
    {"sum of 12 + reduce96", k_sum, 1000, 1},
    {"v_add_nc_u32", k_i_add, 2000, 32}, {"v_add3_u32", k_i_add3, 2000, 32}, {"v_add_co_u32", k_i_carry, 2000, 32},
    {"v_add_co_ci_u32 (in)", k_i_carryin, 2000, 32}, {"v_cndmask_b32", k_i_cnd, 2000, 32},
    {"v_lshlrev_b32", k_i_shl, 2000, 32},
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
