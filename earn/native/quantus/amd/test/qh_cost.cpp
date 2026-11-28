// What single instructions and small mixes cost on RDNA 3 and 4, in the build's wave size, for pricing new field
// arithmetic for the Quantus kernel. Timing only.
//
//   qh_cost [device]     built by the box job, e.g.
//                        clang++ -x hip --offload-arch=gfx1100 -O3 -mwavefrontsize64 -DQH_W64=1 qh_cost.cpp -lamdhip64
//                        -DQC_DOT2=1, -DQC_DOT4=1, -DQC_MAD16=1 add v_dot2_u32_u16, v_dot4_u32_u8 and v_mad_u32_u16
//                        (each its own build: not every target has them)
//
// Every kernel runs 8 independent chains a thread, 4 passes a loop trip, from 256-thread blocks that fill every
// SIMD. It prints SIMD-cycles per instruction a 32 lanes (a wave64 instruction is two of those): the card's cycles
// times its SIMDs over the waves' instruction count. The clock is read by one wave (SHADER_CYCLES against REALTIME)
// right after each kernel. gfx11 and gfx12 only.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef uint32_t u32;
typedef uint64_t u64;

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

// A lane mask of the wave size: SM for the chains' own use, SN and SO for scalar-only work.
#if defined(QH_W64) && QH_W64
#define SM "s[0:1]"
#define SN "s[2:3]"
#define SO "s[4:5]"
#define SB "_b64"
#define VCC "vcc"
#else
#define SM "s0"
#define SN "s2"
#define SO "s4"
#define SB "_b32"
#define VCC "vcc_lo"
#endif
#define SCLOB "s0", "s1", "s2", "s3", "s4", "s5", "vcc", "scc"

// KI: one asm statement a chain. In it %0, %1 and %2 are three 32-bit values of the chain (in and out), %3 a 64-bit
// one (in and out) and %4 a 32-bit input.
#define KI(NAME, ASM)                                                                      \
  __global__ void __launch_bounds__(256) NAME(u64 *out, u64 seed, u32 iters) {           \
    u32 a[8], c[8], e[8], b[8];                                                           \
    u64 w[8];                                                                             \
    for (int i = 0; i < 8; i++) {                                                         \
      a[i] = (u32)seed * (threadIdx.x + 3 + i);                                           \
      c[i] = a[i] ^ 0x5bd1e995u;                                                          \
      e[i] = a[i] + 0x27d4eb2fu;                                                          \
      b[i] = (u32)seed ^ (i * 0x9e3779b9u);                                               \
      w[i] = seed * (threadIdx.x + 7 + i);                                                \
    }                                                                                     \
    asm volatile("s_mov_b32 s0, 0\n\ts_mov_b32 s1, 0\n\ts_mov_b32 s2, 0\n\ts_mov_b32 s3, 0\n\ts_mov_b32 s4, 0\n\t" \
                 "s_mov_b32 s5, 0" ::: SCLOB);                                            \
    for (u32 it = 0; it < iters; it++) {                                                  \
      _Pragma("unroll") for (int r = 0; r < 4; r++) {                                     \
        _Pragma("unroll") for (int i = 0; i < 8; i++)                                     \
          asm volatile(ASM : "+v"(a[i]), "+v"(c[i]), "+v"(e[i]), "+v"(w[i]) : "v"(b[i]) : SCLOB); \
      }                                                                                   \
    }                                                                                     \
    u64 x = 0;                                                                            \
    for (int i = 0; i < 8; i++) x ^= a[i] ^ c[i] ^ e[i] ^ w[i];                           \
    out[blockIdx.x * blockDim.x + threadIdx.x] = x;                                       \
  }
#define D "\n\ts_delay_alu instid0(VALU_DEP_1)"

KI(k_add, "v_add_nc_u32 %0, %0, %4")
KI(k_add3, "v_add3_u32 %0, %0, %4, %0")
KI(k_sub, "v_sub_nc_u32 %0, %0, %4")
KI(k_shl, "v_lshlrev_b32 %0, 1, %0")
KI(k_shr, "v_lshrrev_b32 %0, 1, %0")
KI(k_and, "v_and_b32 %0, %0, %4")
KI(k_xor, "v_xor_b32 %0, %0, %4")
KI(k_not, "v_not_b32 %0, %0")
KI(k_or3, "v_or3_b32 %0, %0, %4, %0")
KI(k_bfe, "v_bfe_u32 %0, %0, 5, 11")
KI(k_lshl_add, "v_lshl_add_u32 %0, %0, 3, %4")
KI(k_lshl_or, "v_lshl_or_b32 %0, %0, 16, %4")
KI(k_alignbit, "v_alignbit_b32 %0, %0, %4, 27")
KI(k_and_or, "v_and_or_b32 %0, %0, %4, %0")
KI(k_xad, "v_xad_u32 %0, %0, %4, %0")
KI(k_cnd, "v_cndmask_b32_e64 %0, %0, %4, " SM)
KI(k_cmp_s, "v_cmp_lt_u32_e64 " SM ", %0, %4")
KI(k_cmp_vcc, "v_cmp_lt_u32 " VCC ", %0, %4")
KI(k_carry, "v_add_co_u32 %0, null, %0, %4")
KI(k_carry_s, "v_add_co_u32 %0, " SM ", %0, %4")
KI(k_carryin, "v_add_co_ci_u32_e64 %0, null, %0, %4, " SM)
KI(k_mad, "v_mad_u64_u32 %3, null, %0, %4, %3")
KI(k_mad_s, "v_mad_u64_u32 %3, " SM ", %0, %4, %3")
KI(k_mul24, "v_mul_u32_u24 %0, %0, %4")
KI(k_mulhi24, "v_mul_hi_u32_u24 %0, %0, %4")
KI(k_mad24, "v_mad_u32_u24 %0, %0, %4, %0")
KI(k_mullo, "v_mul_lo_u32 %0, %0, %4")
KI(k_mulhi, "v_mul_hi_u32 %0, %0, %4")
#if defined(QC_DOT2) && QC_DOT2
KI(k_dot2, "v_dot2_u32_u16 %0, %0, %4, %0")
#endif
#if defined(QC_DOT4) && QC_DOT4
KI(k_dot4, "v_dot4_u32_u8 %0, %0, %4, %0")
#endif
#if defined(QC_MAD16) && QC_MAD16
KI(k_mad16, "v_mad_u32_u16 %0, %0, %4, %0")
#endif
// Scalar instructions beside vector ones: an add with 1, 2 and 4 scalar xors after it.
#define SX "\n\ts_xor" SB " " SN ", " SN ", " SO
KI(k_add_s1, "v_add_nc_u32 %0, %0, %4" SX)
KI(k_add_s2, "v_add_nc_u32 %0, %0, %4" SX SX)
KI(k_add_s4, "v_add_nc_u32 %0, %0, %4" SX SX SX SX)
// Mixes on separate values of a chain, so no instruction reads the one before it.
KI(k_carry_add, "v_add_co_u32 %0, null, %0, %4\n\tv_add_nc_u32 %1, %1, %4")
KI(k_carry_cnd, "v_add_co_u32 %0, null, %0, %4\n\tv_cndmask_b32_e64 %1, %1, %4, " SM)
KI(k_carry_2add, "v_add_co_u32 %0, null, %0, %4\n\tv_add_nc_u32 %1, %1, %4\n\tv_xor_b32 %2, %2, %4")
KI(k_2carry_add, "v_add_co_u32 %0, null, %0, %4\n\tv_add_co_u32 %1, null, %1, %4\n\tv_xor_b32 %2, %2, %4")
KI(k_cmp_add, "v_cmp_lt_u32_e64 " SM ", %0, %4\n\tv_add_nc_u32 %1, %1, %4")
KI(k_cmp_cnd, "v_cmp_lt_u32_e64 " SM ", %0, %4\n\tv_cndmask_b32_e64 %1, %1, %4, " SM)
KI(k_mad_add, "v_mad_u64_u32 %3, null, %0, %4, %3\n\tv_add_nc_u32 %1, %1, %4")
KI(k_mad_carry, "v_mad_u64_u32 %3, null, %0, %4, %3\n\tv_add_co_u32 %1, null, %1, %4")
KI(k_mad_2carry, "v_mad_u64_u32 %3, null, %0, %4, %3\n\tv_add_co_u32 %1, null, %1, %4\n\tv_add_co_u32 %2, null, %2, %4")
KI(k_mad_cnd, "v_mad_u64_u32 %3, null, %0, %4, %3\n\tv_cndmask_b32_e64 %1, %1, %4, " SM)
KI(k_mad_2add, "v_mad_u64_u32 %3, null, %0, %4, %3\n\tv_add_nc_u32 %1, %1, %4\n\tv_xor_b32 %2, %2, %4")
// The same instruction on a value it has just written (a dependent pair), with and without the delay hint.
KI(k_dep_add, "v_add_nc_u32 %0, %0, %4\n\tv_add_nc_u32 %0, %0, %4")
KI(k_dep_add_d, "v_add_nc_u32 %0, %0, %4" D "\n\tv_add_nc_u32 %0, %0, %4" D)

typedef void (*kfn)(u64 *, u64, u32);
struct K { const char *name; kfn f; u32 iters; int per; };  // per: instructions in the asm statement

int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHK(hipSetDevice(dev));
  hipDeviceProp_t p;
  CHK(hipGetDeviceProperties(&p, dev));
  int wgps = p.multiProcessorCount;  // HIP counts WGPs on RDNA 3 and 4 (4 SIMD32s each)
  int simds = 4 * wgps;
  printf("device %d: %s %s, %d WGPs, %d SIMDs\n", dev, p.name, p.gcnArchName, wgps, simds);
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
    {"warm-up", k_add, 1500, 1}, {"add_nc", k_add, 1500, 1}, {"add3", k_add3, 1500, 1}, {"sub_nc", k_sub, 1500, 1},
    {"lshl", k_shl, 1500, 1}, {"lshr", k_shr, 1500, 1}, {"and", k_and, 1500, 1}, {"xor", k_xor, 1500, 1},
    {"not", k_not, 1500, 1}, {"or3", k_or3, 1500, 1}, {"bfe_u32", k_bfe, 1500, 1}, {"lshl_add", k_lshl_add, 1500, 1},
    {"lshl_or", k_lshl_or, 1500, 1}, {"alignbit", k_alignbit, 1500, 1}, {"and_or", k_and_or, 1500, 1},
    {"xad", k_xad, 1500, 1}, {"cndmask (SGPR)", k_cnd, 1500, 1}, {"cmp_lt -> SGPR", k_cmp_s, 1500, 1},
    {"cmp_lt -> VCC", k_cmp_vcc, 1500, 1}, {"add_co, null", k_carry, 1500, 1}, {"add_co -> SGPR", k_carry_s, 1500, 1},
    {"add_co_ci (in)", k_carryin, 1500, 1}, {"mad_u64_u32", k_mad, 800, 1}, {"mad_u64_u32 -> SGPR", k_mad_s, 800, 1},
    {"mul_u32_u24", k_mul24, 1500, 1}, {"mul_hi_u32_u24", k_mulhi24, 1500, 1}, {"mad_u32_u24", k_mad24, 1500, 1},
    {"mul_lo_u32", k_mullo, 800, 1}, {"mul_hi_u32", k_mulhi, 800, 1},
#if defined(QC_DOT2) && QC_DOT2
    {"dot2_u32_u16", k_dot2, 1500, 1},
#endif
#if defined(QC_DOT4) && QC_DOT4
    {"dot4_u32_u8", k_dot4, 1500, 1},
#endif
#if defined(QC_MAD16) && QC_MAD16
    {"mad_u32_u16", k_mad16, 1500, 1},
#endif
    {"add + 1 s_xor (per add)", k_add_s1, 1500, 1}, {"add + 2 s_xor (per add)", k_add_s2, 1500, 1},
    {"add + 4 s_xor (per add)", k_add_s4, 1500, 1},
    {"carry + add (pair)", k_carry_add, 1000, 1}, {"carry + cndmask (pair)", k_carry_cnd, 1000, 1},
    {"carry + 2 plain (three)", k_carry_2add, 1000, 1}, {"2 carry + plain (three)", k_2carry_add, 1000, 1},
    {"cmp + add (pair)", k_cmp_add, 1000, 1},
    {"cmp + cndmask (pair)", k_cmp_cnd, 1000, 1}, {"mad + add (pair)", k_mad_add, 800, 1},
    {"mad + carry (pair)", k_mad_carry, 800, 1}, {"mad + 2 carry (three)", k_mad_2carry, 800, 1},
    {"mad + cndmask (pair)", k_mad_cnd, 800, 1}, {"mad + 2 plain (three)", k_mad_2add, 800, 1},
    {"dependent add pair", k_dep_add, 1000, 1}, {"dependent add pair + delay", k_dep_add_d, 1000, 1},
    {"add_nc again", k_add, 1500, 1},
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
    double stmts = waves * k.iters * 32.0;                   // asm statements a 32 lanes
    double cyc = best * 1e-3 * mhz * 1e6 * simds / stmts;    // SIMD-cycles a statement
    printf("%-28s %8.3f ms  %5.0f MHz  %7.2f SIMD-cycles\n", k.name, best, mhz, cyc);
    fflush(stdout);
  }
  return 0;
}
