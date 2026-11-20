// What a carry costs on RDNA 3 and 4, for pricing the Quantus kernel. Timing only.
//
//   qh_carry [device]      built by the box job: hipcc -x hip --offload-arch=<gfx> qh_carry.cpp
//
// qh_peak's carry rows (v_add_co_u32 at 0.56x an add on an RX 7900 XT) write the same carry register in all 8
// chains. Here each of the 8 chains has its own SGPR (s0..s7), or none (null), so only the instruction's own
// cost is left. Every kernel runs 8 independent chains, 32 instructions a loop pass, from 16 waves a SIMD.
// Prints lane-ops a WGP a clock (full rate: 128) and the ratio to v_add_nc_u32; the clock is read by one wave
// (SHADER_CYCLES against REALTIME) right after each kernel. gfx11 and gfx12 only.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef uint32_t u32;
typedef uint64_t u64;

#define ITERS 1024
#define CHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

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

// One asm statement holds all 8 chains' instructions (8 x PER), so the order is fixed. A: 8 VGPR accumulators,
// B: 8 VGPR sources; S: SGPRs s0..s7 (clobbered).
#define CLOB "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "vcc_lo"
#define KERN(NAME, BODY)                                                                        \
  __global__ void __launch_bounds__(256) NAME(u32 *out, u32 seed) {                           \
    u32 a0 = seed * (threadIdx.x + 3), a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; \
    u32 b0 = seed ^ 0x9e3779b9u, b1 = b0 + 11, b2 = b0 + 22, b3 = b0 + 33, b4 = b0 + 44, b5 = b0 + 55, b6 = b0 + 66, b7 = b0 + 77; \
    asm volatile("s_mov_b32 s0, 0\n\ts_mov_b32 s1, 0\n\ts_mov_b32 s2, 0\n\ts_mov_b32 s3, 0\n\t"                  \
                 "s_mov_b32 s4, 0\n\ts_mov_b32 s5, 0\n\ts_mov_b32 s6, 0\n\ts_mov_b32 s7, 0" ::: CLOB);              \
    for (int it = 0; it < ITERS; it++) {                                                      \
      _Pragma("unroll") for (int rep = 0; rep < 4; rep++) {                                   \
        asm volatile(BODY : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7) \
                     : "v"(b0), "v"(b1), "v"(b2), "v"(b3), "v"(b4), "v"(b5), "v"(b6), "v"(b7) : CLOB);   \
      }                                                                                       \
    }                                                                                         \
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;        \
  }

// I8(fmt): the instruction for chains 0..7; %N is chain N's accumulator, %(N+8) its source, sN its SGPR.
#define I8(F0, F1, F2, F3, F4, F5, F6, F7) F0 "\n\t" F1 "\n\t" F2 "\n\t" F3 "\n\t" F4 "\n\t" F5 "\n\t" F6 "\n\t" F7 "\n\t"
#define CH(OP, TAIL) I8(OP " %0, " TAIL(0, 8), OP " %1, " TAIL(1, 9), OP " %2, " TAIL(2, 10), OP " %3, " TAIL(3, 11), \
                        OP " %4, " TAIL(4, 12), OP " %5, " TAIL(5, 13), OP " %6, " TAIL(6, 14), OP " %7, " TAIL(7, 15))
#define S_(x) #x
#define T_ADD(a, b) "%" S_(a) ", %" S_(b)
#define T_NULL(a, b) "null, %" S_(a) ", %" S_(b)
#define T_SN(a, b) "s" S_(a) ", %" S_(a) ", %" S_(b)
#define T_CISN(a, b) "s" S_(a) ", %" S_(a) ", %" S_(b) ", s" S_(a)
#define T_CINULL(a, b) "null, %" S_(a) ", %" S_(b) ", s" S_(a)
#define T_CND(a, b) "%" S_(a) ", %" S_(b) ", s" S_(a)
#define T_VCC(a, b) "vcc_lo, %" S_(a) ", %" S_(b)

KERN(k_add, CH("v_add_nc_u32", T_ADD) CH("v_add_nc_u32", T_ADD) CH("v_add_nc_u32", T_ADD) CH("v_add_nc_u32", T_ADD))
KERN(k_addco_null, CH("v_add_co_u32", T_NULL) CH("v_add_co_u32", T_NULL) CH("v_add_co_u32", T_NULL) CH("v_add_co_u32", T_NULL))
KERN(k_addco_sn, CH("v_add_co_u32", T_SN) CH("v_add_co_u32", T_SN) CH("v_add_co_u32", T_SN) CH("v_add_co_u32", T_SN))
KERN(k_addcoci_sn, CH("v_add_co_ci_u32_e64", T_CISN) CH("v_add_co_ci_u32_e64", T_CISN) CH("v_add_co_ci_u32_e64", T_CISN) CH("v_add_co_ci_u32_e64", T_CISN))
KERN(k_addcoci_null, CH("v_add_co_ci_u32_e64", T_CINULL) CH("v_add_co_ci_u32_e64", T_CINULL) CH("v_add_co_ci_u32_e64", T_CINULL) CH("v_add_co_ci_u32_e64", T_CINULL))
KERN(k_subco_sn, CH("v_sub_co_u32", T_SN) CH("v_sub_co_u32", T_SN) CH("v_sub_co_u32", T_SN) CH("v_sub_co_u32", T_SN))
KERN(k_cmp_sn, CH("v_cmp_gt_u32_e64", T_SN) CH("v_cmp_gt_u32_e64", T_SN) CH("v_cmp_gt_u32_e64", T_SN) CH("v_cmp_gt_u32_e64", T_SN))
KERN(k_cnd_sn, CH("v_cndmask_b32_e64", T_CND) CH("v_cndmask_b32_e64", T_CND) CH("v_cndmask_b32_e64", T_CND) CH("v_cndmask_b32_e64", T_CND))
// Half the instructions add with a carry out (own SGPR), half plain adds: do the two kinds pair up?
KERN(k_mix_co_add, CH("v_add_co_u32", T_SN) CH("v_add_nc_u32", T_ADD) CH("v_add_co_u32", T_SN) CH("v_add_nc_u32", T_ADD))
// Carry out into VCC from every chain (the compiler's usual choice), and plain VOP2 (e32) with VCC in and out.
KERN(k_addco_vcc, CH("v_add_co_u32", T_VCC) CH("v_add_co_u32", T_VCC) CH("v_add_co_u32", T_VCC) CH("v_add_co_u32", T_VCC))
// Dual issue: VOPD pairs of adds on fixed registers v40-v55 (clobbered), laid out for VOPD's rules: the two
// destinations of a pair have different parity, and each pair's two sources sit in different banks (reg mod 4).
#define DUAL(x, y, a, b) "v_dual_add_nc_u32 v" #x ", v" #x ", v" #a " :: v_dual_add_nc_u32 v" #y ", v" #y ", v" #b
#define D4 DUAL(40, 41, 48, 49) "\n\t" DUAL(42, 43, 50, 51) "\n\t" DUAL(44, 45, 52, 53) "\n\t" DUAL(46, 47, 54, 55) "\n\t"
#define VCLOB "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55"
__global__ void __launch_bounds__(256) k_dual_add(u32 *out, u32 seed) {
  asm volatile("v_mov_b32 v40, %0\n\tv_mov_b32 v41, %0\n\tv_mov_b32 v42, %0\n\tv_mov_b32 v43, %0\n\t"
               "v_mov_b32 v44, %0\n\tv_mov_b32 v45, %0\n\tv_mov_b32 v46, %0\n\tv_mov_b32 v47, %0\n\t"
               "v_mov_b32 v48, %0\n\tv_mov_b32 v49, %0\n\tv_mov_b32 v50, %0\n\tv_mov_b32 v51, %0\n\t"
               "v_mov_b32 v52, %0\n\tv_mov_b32 v53, %0\n\tv_mov_b32 v54, %0\n\tv_mov_b32 v55, %0"
               :: "v"(seed + threadIdx.x) : VCLOB);
  for (int it = 0; it < ITERS; it++) {
    _Pragma("unroll") for (int rep = 0; rep < 4; rep++) { asm volatile(D4 D4 D4 D4 D4 D4 D4 D4 ::: VCLOB); }
  }
  u32 s;
  asm volatile("v_xor_b32 %0, v40, v47" : "=v"(s) :: "v40", "v47");
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

typedef void (*kfn)(u32 *, u32);
struct K { const char *name; kfn f; double per; };  // per: lane-ops in one asm statement, per thread

int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHK(hipSetDevice(dev));
  hipDeviceProp_t p;
  CHK(hipGetDeviceProperties(&p, dev));
  int wgps = p.multiProcessorCount;
  printf("device %d: %s %s, %d WGPs (HIP's count)\n", dev, p.name, p.gcnArchName, wgps);
  int blocks = wgps * 32, threads = 256;
  u32 *out, *cko;
  CHK(hipMalloc(&out, (size_t)blocks * threads * 4));
  CHK(hipMalloc(&cko, 16));
  int wallKhz = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || wallKhz <= 0) wallKhz = 100000;
  (void)hipGetLastError();
  auto clk_mhz = [&]() {
    hipLaunchKernelGGL(k_clk, dim3(1), dim3(32), 0, 0, 64u, cko);
    CHK(hipDeviceSynchronize());
    u32 h[3];
    CHK(hipMemcpy(h, cko, 12, hipMemcpyDeviceToHost));
    double secs = (double)h[0] / ((double)wallKhz * 1e3);
    return secs > 0 ? (double)h[1] / secs / 1e6 : 0.0;
  };
  K ks[] = {
    {"warm-up (add)", k_add, 32}, {"v_add_nc_u32", k_add, 32},
    {"v_add_co_u32, null", k_addco_null, 32}, {"v_add_co_u32, own SGPR", k_addco_sn, 32},
    {"v_add_co_u32, all VCC", k_addco_vcc, 32},
    {"v_add_co_ci_u32, own SGPR", k_addcoci_sn, 32}, {"v_add_co_ci_u32, in only", k_addcoci_null, 32},
    {"v_sub_co_u32, own SGPR", k_subco_sn, 32}, {"v_cmp_gt_u32, own SGPR", k_cmp_sn, 32},
    {"v_cndmask_b32, own SGPR", k_cnd_sn, 32}, {"add_co + add, half each", k_mix_co_add, 32},
    {"VOPD add pairs", k_dual_add, 64}, {"v_add_nc_u32 again", k_add, 32},
  };
  hipEvent_t e0, e1;
  CHK(hipEventCreate(&e0));
  CHK(hipEventCreate(&e1));
  double base = 0;
  for (auto &k : ks) {
    double best = 1e30, mhz = 0;
    for (int rep = 0; rep < 4; rep++) {
      CHK(hipEventRecord(e0));
      hipLaunchKernelGGL(k.f, dim3(blocks), dim3(threads), 0, 0, out, 12345u + rep);
      CHK(hipEventRecord(e1));
      CHK(hipEventSynchronize(e1));
      float ms = 0;
      CHK(hipEventElapsedTime(&ms, e0, e1));
      double m = clk_mhz();
      if (rep > 0 && ms < best) { best = ms; mhz = m; }
    }
    double lane = (double)blocks * threads * ITERS * 4 * k.per;
    double perclk = lane / (best * 1e-3 * mhz * 1e6) / wgps;
    if (base == 0 && k.f == k_add && k.name[0] == 'v') base = perclk;
    printf("%-28s %8.3f ms  %5.0f MHz  %6.1f lane-ops/WGP/clk  %5.2fx add\n", k.name, best, mhz, perclk, base ? perclk / base : 0);
    fflush(stdout);
  }
  return 0;
}
