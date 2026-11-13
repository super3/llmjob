// Integer instruction rates on an AMD card, for pricing the Quantus kernel. Timing only.
//
//   qh_peak [device]
//
// Each kernel runs 8 independent chains of one instruction (or a fixed mix), 32 instructions an
// iteration, from enough waves to fill every SIMD. It prints, per kernel, the instructions a CU
// retires a clock (full rate on RDNA is 64: two SIMD32s) and the ratio to v_add_nc_u32. The
// clock is the shader clock the card ran at, read from s_memtime against s_memrealtime (100 MHz)
// in one wave.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef uint32_t u32;
typedef uint64_t u64;

#define ITERS 1024
#define CHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

#define R8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

// One kernel per instruction kind. ACC is the per-chain register, IN a second operand.
#define KERNEL32(NAME, ASM)                                                                    \
  __global__ void __launch_bounds__(256) NAME(u32 *out, u32 seed, u64 *clk) {               \
    u32 a[8], b[8];                                                                          \
    for (int i = 0; i < 8; i++) { a[i] = seed * (threadIdx.x + 3 + i); b[i] = seed ^ (i * 0x9e3779b9u); } \
    u64 t0 = 0, r0 = 0;                                                                      \
    if (threadIdx.x == 0 && blockIdx.x == 0) { t0 = __builtin_amdgcn_s_memtime(); r0 = __builtin_amdgcn_s_memrealtime(); } \
    for (int it = 0; it < ITERS; it++) {                                                     \
      _Pragma("unroll") for (int rep = 0; rep < 4; rep++) {                                  \
        _Pragma("unroll") for (int i = 0; i < 8; i++) { asm volatile(ASM : "+v"(a[i]) : "v"(b[i]) : "vcc", "s0"); } \
      }                                                                                      \
    }                                                                                        \
    if (threadIdx.x == 0 && blockIdx.x == 0) { clk[0] = __builtin_amdgcn_s_memtime() - t0; clk[1] = __builtin_amdgcn_s_memrealtime() - r0; } \
    u32 s = 0;                                                                               \
    for (int i = 0; i < 8; i++) s ^= a[i];                                                   \
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;                                          \
  }

KERNEL32(k_add_nc, "v_add_nc_u32 %0, %0, %1")
KERNEL32(k_mul_lo, "v_mul_lo_u32 %0, %0, %1")
KERNEL32(k_mul_hi, "v_mul_hi_u32 %0, %0, %1")
KERNEL32(k_mul_u24, "v_mul_u32_u24 %0, %0, %1")
KERNEL32(k_mul_hi_u24, "v_mul_hi_u32_u24 %0, %0, %1")
KERNEL32(k_mad_u24, "v_mad_u32_u24 %0, %1, %1, %0")
KERNEL32(k_mad_u16, "v_mad_u32_u16 %0, %1, %1, %0")
KERNEL32(k_alignbit, "v_alignbit_b32 %0, %0, %1, 7")
KERNEL32(k_cndmask, "v_cndmask_b32 %0, %0, %1, vcc_lo")
KERNEL32(k_add_co, "v_add_co_u32 %0, vcc_lo, %0, %1")
KERNEL32(k_addc_co, "v_add_co_ci_u32 %0, vcc_lo, %0, %1, vcc_lo")
KERNEL32(k_sub_co3, "v_sub_co_u32 %0, s0, %0, %1")
KERNEL32(k_dot2_u16, "v_dot2_u32_u16 %0, %1, %1, %0")
KERNEL32(k_dot4_u8, "v_dot4_u32_u8 %0, %1, %1, %0")
KERNEL32(k_bfe, "v_bfe_u32 %0, %0, %1, 24")
KERNEL32(k_lshl_or, "v_lshl_or_b32 %0, %0, 8, %1")
KERNEL32(k_xad, "v_xad_u32 %0, %0, %1, %0")
KERNEL32(k_add3, "v_add3_u32 %0, %0, %1, %0")
// Mixes: a 32-bit multiply followed by N full-rate adds on another chain. If the multiply holds the
// SIMD for its whole quarter-rate time, a group costs 4 + N clocks; if adds fill its gaps, about 4.
KERNEL32(k_mulhi_add1, "v_mul_hi_u32 %0, %0, %1\n\tv_add_nc_u32 %1, %1, 1")
KERNEL32(k_mulhi_add3, "v_mul_hi_u32 %0, %0, %1\n\tv_add_nc_u32 %1, %1, 1\n\tv_add_nc_u32 %1, %1, 3\n\tv_add_nc_u32 %1, %1, 5")
KERNEL32(k_mul24_pair, "v_mul_u32_u24 %0, %0, %1\n\tv_mul_hi_u32_u24 %1, %0, %1")

// 64-bit accumulator kinds.
#define KERNEL64(NAME, ASM)                                                                    \
  __global__ void __launch_bounds__(256) NAME(u32 *out, u32 seed, u64 *clk) {               \
    u64 a[8]; u32 b[8];                                                                      \
    for (int i = 0; i < 8; i++) { a[i] = (u64)seed * (threadIdx.x + 3 + i); b[i] = seed ^ (i * 0x9e3779b9u); } \
    u64 t0 = 0, r0 = 0;                                                                      \
    if (threadIdx.x == 0 && blockIdx.x == 0) { t0 = __builtin_amdgcn_s_memtime(); r0 = __builtin_amdgcn_s_memrealtime(); } \
    for (int it = 0; it < ITERS; it++) {                                                     \
      _Pragma("unroll") for (int rep = 0; rep < 4; rep++) {                                  \
        _Pragma("unroll") for (int i = 0; i < 8; i++) { asm volatile(ASM : "+v"(a[i]) : "v"(b[i]) : "vcc", "s0"); } \
      }                                                                                      \
    }                                                                                        \
    if (threadIdx.x == 0 && blockIdx.x == 0) { clk[0] = __builtin_amdgcn_s_memtime() - t0; clk[1] = __builtin_amdgcn_s_memrealtime() - r0; } \
    u32 s = 0;                                                                               \
    for (int i = 0; i < 8; i++) s ^= (u32)a[i] ^ (u32)(a[i] >> 32);                         \
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;                                          \
  }

KERNEL64(k_mad_u64, "v_mad_u64_u32 %0, s0, %1, %1, %0")
KERNEL64(k_mad_u64_vcc, "v_mad_u64_u32 %0, vcc_lo, %1, %1, %0")
KERNEL64(k_lshl_b64, "v_lshlrev_b64 %0, 1, %0")
KERNEL64(k_mad_u64_add1, "v_mad_u64_u32 %0, s0, %1, %1, %0\n\tv_add_nc_u32 %1, %1, 1")
KERNEL64(k_mad_u64_add2, "v_mad_u64_u32 %0, s0, %1, %1, %0\n\tv_add_nc_u32 %1, %1, 1\n\tv_add_nc_u32 %1, %1, 3")
KERNEL64(k_mad_u64_add4, "v_mad_u64_u32 %0, s0, %1, %1, %0\n\tv_add_nc_u32 %1, %1, 1\n\tv_add_nc_u32 %1, %1, 3\n\tv_add_nc_u32 %1, %1, 5\n\tv_add_nc_u32 %1, %1, 7")

// The mixes above write a register the multiply reads, so they can't overlap by construction. These keep
// the multiply's accumulator and sources apart from the other instructions' register, so only the issue
// rate keeps them apart.
#define KERNELMIX(NAME, ASM)                                                                   \
  __global__ void __launch_bounds__(256) NAME(u32 *out, u32 seed, u64 *clk) {               \
    u64 a[8]; u32 b[8], c[8];                                                                \
    for (int i = 0; i < 8; i++) { a[i] = (u64)seed * (threadIdx.x + 3 + i); b[i] = seed ^ (i * 0x9e3779b9u); c[i] = seed + i; } \
    u64 t0 = 0, r0 = 0;                                                                      \
    if (threadIdx.x == 0 && blockIdx.x == 0) { t0 = __builtin_amdgcn_s_memtime(); r0 = __builtin_amdgcn_s_memrealtime(); } \
    for (int it = 0; it < ITERS; it++) {                                                     \
      _Pragma("unroll") for (int rep = 0; rep < 4; rep++) {                                  \
        _Pragma("unroll") for (int i = 0; i < 8; i++) { asm volatile(ASM : "+v"(a[i]), "+v"(c[i]) : "v"(b[i]) : "vcc", "s0"); } \
      }                                                                                      \
    }                                                                                        \
    if (threadIdx.x == 0 && blockIdx.x == 0) { clk[0] = __builtin_amdgcn_s_memtime() - t0; clk[1] = __builtin_amdgcn_s_memrealtime() - r0; } \
    u32 s = 0;                                                                               \
    for (int i = 0; i < 8; i++) s ^= (u32)a[i] ^ (u32)(a[i] >> 32) ^ c[i];                 \
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;                                          \
  }

KERNELMIX(k_x_mad1, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_add_nc_u32 %1, %1, 1")
KERNELMIX(k_x_mad2, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_add_nc_u32 %1, %1, 1\n\tv_xor_b32 %1, %1, 3")
KERNELMIX(k_x_mad3, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_add_nc_u32 %1, %1, 1\n\tv_xor_b32 %1, %1, 3\n\tv_add_nc_u32 %1, %1, 5")
KERNELMIX(k_x_mad4, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_add_nc_u32 %1, %1, 1\n\tv_xor_b32 %1, %1, 3\n\tv_add_nc_u32 %1, %1, 5\n\tv_xor_b32 %1, %1, 7")
KERNELMIX(k_x_madco3, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_add_co_u32 %1, vcc_lo, %1, 1\n\tv_add_co_ci_u32_e32 %1, vcc_lo, %1, %2, vcc_lo\n\tv_add_co_ci_u32_e32 %1, vcc_lo, 0, %1, vcc_lo")
KERNELMIX(k_x_mul24_2, "v_mad_u64_u32 %0, null, %2, %2, %0\n\tv_mul_u32_u24 %1, %1, %2\n\tv_mul_hi_u32_u24 %1, %1, %2")
KERNELMIX(k_x_madonly, "v_mad_u64_u32 %0, null, %2, %2, %0")
// Does the operand kind change a multiply's rate? An SGPR or a literal as one source, a zero addend.
KERNELMIX(k_x_mad_sgpr, "s_mov_b32 s0, 0x9e3779b9\n\tv_mad_u64_u32 %0, null, %2, s0, %0")
KERNELMIX(k_x_mad_lit, "v_mad_u64_u32 %0, null, %2, 0x9e3779b9, %0")
KERNELMIX(k_x_mad_zero, "v_mad_u64_u32 %0, null, %2, %2, 0")
KERNELMIX(k_x_mulhi_sgpr, "s_mov_b32 s0, 0x9e3779b9\n\tv_mul_hi_u32 %1, %1, s0")


typedef void (*kfn)(u32 *, u32, u64 *);
struct K { const char *name; kfn f; int per; };  // per: instructions in one asm statement

int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHK(hipSetDevice(dev));
  hipDeviceProp_t p;
  CHK(hipGetDeviceProperties(&p, dev));
  int cus = p.multiProcessorCount;
  printf("device %d: %s %s, %d CUs, pci %02x:%02x.0\n", dev, p.name, p.gcnArchName, cus, p.pciBusID, p.pciDeviceID);
  int blocks = cus * 16, threads = 256;  // 16 waves a SIMD32 asked for; the registers decide how many run
  u32 *out; u64 *clk;
  CHK(hipMalloc(&out, (size_t)blocks * threads * 4));
  CHK(hipMalloc(&clk, 16));
  K ks[] = {
    {"v_add_nc_u32", k_add_nc, 1}, {"v_mul_lo_u32", k_mul_lo, 1}, {"v_mul_hi_u32", k_mul_hi, 1},
    {"v_mul_u32_u24", k_mul_u24, 1}, {"v_mul_hi_u32_u24", k_mul_hi_u24, 1}, {"v_mad_u32_u24", k_mad_u24, 1},
    {"v_mad_u32_u16", k_mad_u16, 1}, {"v_alignbit_b32", k_alignbit, 1}, {"v_cndmask_b32", k_cndmask, 1},
    {"v_add_co_u32 (vcc)", k_add_co, 1}, {"v_add_co_ci_u32 (vcc)", k_addc_co, 1}, {"v_sub_co_u32 (sgpr)", k_sub_co3, 1},
    {"v_dot2_u32_u16", k_dot2_u16, 1}, {"v_dot4_u32_u8", k_dot4_u8, 1}, {"v_bfe_u32", k_bfe, 1},
    {"v_lshl_or_b32", k_lshl_or, 1}, {"v_xad_u32", k_xad, 1}, {"v_add3_u32", k_add3, 1},
    {"mul_hi + 1 add", k_mulhi_add1, 2}, {"mul_hi + 3 add", k_mulhi_add3, 4}, {"mul_u24 + mul_hi_u24", k_mul24_pair, 2},
    {"v_mad_u64_u32 (sgpr)", k_mad_u64, 1}, {"v_mad_u64_u32 (vcc)", k_mad_u64_vcc, 1}, {"v_lshlrev_b64", k_lshl_b64, 1},
    {"mad_u64 + 1 add", k_mad_u64_add1, 2}, {"mad_u64 + 2 add", k_mad_u64_add2, 3},
    {"mad_u64 + 4 add", k_mad_u64_add4, 5},
    {"mad + 1 indep add", k_x_mad1, 2}, {"mad + 2 indep", k_x_mad2, 3}, {"mad + 3 indep", k_x_mad3, 4},
    {"mad + 4 indep", k_x_mad4, 5}, {"mad + 3 indep carry", k_x_madco3, 4}, {"mad + mul24 pair", k_x_mul24_2, 3},
    {"mad alone (mix shape)", k_x_madonly, 1}, {"mad, SGPR source", k_x_mad_sgpr, 1},
    {"mad, literal source", k_x_mad_lit, 1}, {"mad, zero addend", k_x_mad_zero, 1}, {"mul_hi, SGPR source", k_x_mulhi_sgpr, 1},
  };
  double base = 0;
  hipEvent_t e0, e1;
  CHK(hipEventCreate(&e0));
  CHK(hipEventCreate(&e1));
  for (auto &k : ks) {
    double best = 1e30, mhz = 0;
    for (int rep = 0; rep < 4; rep++) {
      CHK(hipEventRecord(e0));
      hipLaunchKernelGGL(k.f, dim3(blocks), dim3(threads), 0, 0, out, 12345u + rep, clk);
      CHK(hipEventRecord(e1));
      CHK(hipEventSynchronize(e1));
      float ms = 0;
      CHK(hipEventElapsedTime(&ms, e0, e1));
      u64 c[2];
      CHK(hipMemcpy(c, clk, 16, hipMemcpyDeviceToHost));
      if (rep > 0 && ms < best) { best = ms; mhz = c[1] ? (double)c[0] / ((double)c[1] / 100.0) : 0; }
    }
    double instr = (double)blocks * threads * ITERS * 32 * k.per;  // lane-instructions
    double clocks = best * 1e-3 * mhz * 1e6;
    double perclk = instr / 32 / clocks / cus * 32;                   // lane-instructions a CU a clock
    if (base == 0) base = perclk;
    printf("%-24s %8.3f ms  %6.0f MHz  %6.1f lane-ops/CU/clk  %5.2fx add\n", k.name, best, mhz, perclk, perclk / base);
    fflush(stdout);
  }
  return 0;
}
