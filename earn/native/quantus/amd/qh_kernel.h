// The Quantus (QPoW) nonce kernel for AMD GPUs (HIP, RDNA 2 to 4).
//
// Per nonce it runs the two Poseidon2 permutations of earn/native/quantus/README.md (blocks 3 and 4
// from the midstate) and reports the counter when the hash's top 64 bits are at or below the target's.
// The field is Goldilocks, p = 2^64 - 2^32 + 1. Values are kept in [0, 2^64), not reduced below p,
// except the one compared at the end.
//
// The round constants and the diagonal are qp-poseidon 3.1.0's (src/shared/quantus/poseidon2.js has
// them with their licence). The way the work is cut (the midstate, only the counter word moving, the
// constants folded into the layer sums) follows quantus-miner's CUDA kernel
// (crates/engine-cuda/src/kernels/mining.cu, Apache-2.0); no code is copied from it.
//
// On RDNA a 32 x 32 multiply (v_mad_u64_u32) takes four issue cycles and nothing overlaps it, so a nonce
// costs about 4 x (its 5,416 multiplies) + 1 x (every other VALU instruction) clocks; the knobs below cut the
// second term. Build knobs (-D...), default first, and what each did on an RX 6600 (quantus-benchmark.md):
//   QH_ARITH 2|1|3    carries as inline-asm lane masks | clang carry builtins: 52.0 | 45.2 MH/s (plain C++: 35.5);
//                     gfx11 and gfx12 default to 3: carries made by asm carry instructions and added in as values
//                     (see qh_m1). In wave64: RX 7900 XT 299 MH/s against 1's 244, RX 9070 XT 270 against 216
//                     (quantus-benchmark.md, 20 Nov)
//   QH_RED 1|0        a 7-instruction reduction that drops carries rarer than 2^-31 | the exact 8: +2.9%
//   QH_SQR 1|0        squares from three multiplies | four: +2.7%
//   QH_WADD 1|0       wide adds in asm, round constants read from SGPRs | carry builtins: +1.4%
//   QH_MULADD 1|0     the internal layer's sum rides in the multiplies | added after (QH_ARITH 1): +2.5%
//   QH_BLOCK, QH_WPE  block size, and the fewest waves a SIMD the compiler leaves room for
//   QH_NPT, QH_PIPE   nonces a thread (1: with the builtins' 7-step reduction 2 was level on an RX 7900 XTX, and
//                     0.3% slower on RDNA 2), and the partial rounds pipelined (off: slower on both)
//   QH_R96, QH_LAST1  a 3-instruction reduce96; the last row's external layer computes element 0 only (gfx11, 12)
//   QH_W64            the asm's carries as SGPR pairs, for a wave64 build (-mwavefrontsize64); build.sh builds gfx11
//                     and gfx12 wave64 (on QH_ARITH 3: RX 7900 XT +5.3%, RX 9070 XT +6.4%). RDNA 2 is wave32.

#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define QH_EPS 0xffffffffull
#define QH_P 0xffffffff00000001ull

// The schedule. A full row is "S-box all 12 (unless the row says not to), the external layer, then
// add the row's constants": the constants of the next round ride in this one's layer sums.
//   rows 1-3    block 3, initial rounds 1-3. Round 0 (row 0) is not a row of the loop: along a line of nonces
//               only two of its S-boxes change, so qh_first_round does it (src/shared/quantus/qpow.js)
//   rows 4-7    block 3, terminal rounds 0-3; row 7 adds block 4's terminator and pad (1, 1, 0...)
//   row  8      block 4's first layer, no S-box, plus its initial round 0 constants
//   rows 9-12   block 4, initial rounds 0-3
//   rows 13-16  block 4, terminal rounds 0-3; only element 0 of row 16 is used
// The partial rounds sit after rows 3 and 12: S-box on element 0, the internal layer, and element 0
// gets the next round's constant; after the last one all 12 get terminal round 0's.
#define QH_ROWS 17
#define QH_NOSBOX_ROW 8

struct QhTables {
  u64 full[QH_ROWS][12];
  u64 part0[22];   // element 0's constant after partial round r
  u64 term0[12];   // terminal round 0's constants (element 0's is in part0[21])
  u64 diag[12];
};

// The host fills this once (qh_host.cpp builds it from poseidon2's constants).
__constant__ QhTables qh_tab;

struct QhArgs {
  u64 k[12];       // the line's constants C (qpow.lineWork), each below p
  u64 x0, x4;      // elements 0 and 4 before the first round's S-boxes, at this launch's first point; below p
  u64 target;      // a hit is top64 <= target
  u32 base;        // this launch's first point on the line (what a hit is numbered from)
  u32 n;           // points in this launch, at most QH_LAUNCH_N
  u32 cap;         // hit slots
};

// What one step along a line adds to element 0 and takes from element 4.
#define QH_SLOPE 35u

// The arithmetic each target defaults to. On gfx11 and gfx12 the compiler hides VALU latency with s_delay_alu
// hints it can't place after inline asm, so there the builtins (QH_ARITH 1) beat the asm path (+11% on an RX
// 7900 XTX, where on RDNA 2 the asm is 15% ahead), until each asm instruction carries its own hint (QH_DLY) and
// carries are added in as values (QH_ARITH 3, gfx11 and gfx12).
#ifndef QH_ARITH
#if defined(__GFX11__) || defined(__GFX12__)
#define QH_ARITH 3
#else
#define QH_ARITH 2
#endif
#endif
// QH_ARITH 4 is gfx11 and gfx12 only (s_delay_alu); other targets keep their default.
#if QH_ARITH == 4 && defined(__HIP_DEVICE_COMPILE__) && !(defined(__GFX11__) || defined(__GFX12__))
#undef QH_ARITH
#define QH_ARITH 2
#endif
#ifndef QH_RED
#define QH_RED 1
#endif
// Squares: three multiplies, except QH_ARITH 3 in wave64, where four (a plain multiply) are faster: RX 7900 XT
// 298.5-299.2 MH/s against 293.9-295.6 (and -2.2% in wave32).
#ifndef QH_SQR
#if QH_ARITH == 3 && defined(QH_W64) && QH_W64
#define QH_SQR 0
#else
#define QH_SQR 1
#endif
#endif
#ifndef QH_WADD
#define QH_WADD 1
#endif
#ifndef QH_MULADD
#define QH_MULADD 1
#endif

// ── Field helpers ────────────────────────────────────────────────────────────────────────────────
// Every carry stays in a lane mask (v_add_co_u32 / v_add_co_ci_u32): plain C++ made the compiler recompute
// carries with 64-bit compares (32 VALU a multiply, not 27). QH_ARITH 1 uses clang's carry builtins and
// inline asm only for the multiply-add whose carry C++ can't express; QH_ARITH 2 does every carry step of a
// multiply and reduction in inline asm, each carry a lane mask in an SGPR (wave32), which saves the copies
// and the separate add of the last carry.

__device__ __forceinline__ u32 qh_lo(u64 x) { return (u32)x; }
__device__ __forceinline__ u32 qh_hi(u64 x) { return (u32)(x >> 32); }
__device__ __forceinline__ u64 qh_pack(u32 lo, u32 hi) { return ((u64)hi << 32) | lo; }

// QH_DLY 1 (gfx11 and gfx12 only): every inline-asm VALU instruction is followed by s_delay_alu
// instid0(VALU_DEP_1). The compiler puts s_delay_alu between dependent VALU instructions it can see, but not
// after inline asm; without one, an instruction that reads a result still in flight holds the whole SIMD
// (qh_peak on an RX 7900 XT: one dependent add chain a wave runs at 0.24x even with every SIMD full of waves).
// With it, the next instruction of that wave waits and the SIMD issues other waves' instructions meanwhile.
#ifndef QH_DLY
#define QH_DLY (QH_ARITH >= 3)
#endif
// QH_DLY 2: only after the multiply-adds (QH_DM); the other asm instructions get none.
#if QH_DLY && (defined(__GFX11__) || defined(__GFX12__))
#define QH_DM "\n\ts_delay_alu instid0(VALU_DEP_1)"
#else
#define QH_DM ""
#endif
#if QH_DLY == 1 && (defined(__GFX11__) || defined(__GFX12__))
#define QH_D QH_DM
#else
#define QH_D ""
#endif

// A lane mask: one SGPR in wave32, a pair in wave64 (QH_W64 1, built with -mwavefrontsize64). Every asm helper's
// carry is one, so QH_ARITH 1 and 3 build for wave64 too (QH_R6 does not).
#ifndef QH_W64
#define QH_W64 0
#endif
#if QH_W64 && defined(__HIP_DEVICE_COMPILE__)
typedef u64 qh_lm;
#else
typedef u32 qh_lm;
#endif

typedef unsigned int qh_cy;  // a carry, 0 or 1

__device__ __forceinline__ u32 qh_addc(u32 a, u32 b, qh_cy ci, qh_cy &co) { return __builtin_addc(a, b, ci, &co); }
__device__ __forceinline__ u32 qh_subc(u32 a, u32 b, qh_cy bi, qh_cy &bo) { return __builtin_subc(a, b, bi, &bo); }

// QH_PURE 1 (QH_ARITH 1 only): the three helpers below that QH_ARITH 1 uses in inline asm (the multiply-add's
// carry out, and adds of a carry) as builtins too, their carries 0 or 1 instead of lane masks, so the compiler
// schedules every instruction.
#ifndef QH_PURE
#define QH_PURE 0
#endif

// a * b + c as 65 bits: the low 64, and the carry as a lane mask in an SGPR (v_mad_u64_u32's carry out).
__device__ __forceinline__ u64 qh_madc(u32 a, u32 b, u64 c, qh_lm &k) {
  u64 d;
#if QH_ARITH == 1 && QH_PURE
  k = __builtin_add_overflow((u64)a * b, c, &d) ? 1u : 0u;
#else
  asm("v_mad_u64_u32 %0, %1, %2, %3, %4" QH_DM : "=v"(d), "=s"(k) : "v"(a), "v"(b), "v"(c));
#endif
  return d;
}

// The same with an SGPR operand (a uniform constant, e.g. the diagonal), so it needs no copy to a VGPR.
__device__ __forceinline__ u64 qh_madc_s(u32 a, u32 b, u64 c, qh_lm &k) {
  u64 d;
  asm("v_mad_u64_u32 %0, %1, %2, %3, %4" QH_DM : "=v"(d), "=s"(k) : "v"(a), "s"(b), "v"(c));
  return d;
}

// x + (the lane mask k ? 1 : 0)
__device__ __forceinline__ u32 qh_addk(u32 x, qh_lm k) {
#if QH_ARITH == 1 && QH_PURE
  return x + (u32)k;
#else
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, %1, 0, %2" QH_D : "=v"(r) : "v"(x), "s"(k));
  return r;
#endif
}

// Carry steps in inline asm, each carry a lane mask in an SGPR (QH_ARITH 2).
__device__ __forceinline__ u32 qh_a_add(u32 a, u32 b, qh_lm &co) {
  u32 r;
  asm("v_add_co_u32 %0, %1, %2, %3" QH_D : "=v"(r), "=s"(co) : "v"(a), "v"(b));
  return r;
}
__device__ __forceinline__ u32 qh_a_sub(u32 a, u32 b, qh_lm &bo) {
  u32 r;
  asm("v_sub_co_u32 %0, %1, %2, %3" QH_D : "=v"(r), "=s"(bo) : "v"(a), "v"(b));
  return r;
}
// a - b - bi, borrow out
__device__ __forceinline__ u32 qh_a_subb(u32 a, u32 b, qh_lm bi, qh_lm &bo) {
  u32 r;
  asm("v_sub_co_ci_u32_e64 %0, %1, %2, %3, %4" QH_D : "=v"(r), "=s"(bo) : "v"(a), "v"(b), "s"(bi));
  return r;
}
// a - bi, no borrow out
__device__ __forceinline__ u32 qh_a_subb0(u32 a, qh_lm bi) {
  u32 r;
  asm("v_sub_co_ci_u32_e64 %0, null, %1, 0, %2" QH_D : "=v"(r) : "v"(a), "s"(bi));
  return r;
}
// ci ? 0 : 0xffffffff
__device__ __forceinline__ u32 qh_a_mask(qh_lm ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, -1, 0, %1" QH_D : "=v"(r) : "s"(ci));
  return r;
}
// a + b + ci, carry out; the same with no carry out; ci as 0 or 1
__device__ __forceinline__ u32 qh_addck(u32 a, u32 b, qh_lm ci, qh_lm &co) {
#if QH_ARITH == 1 && QH_PURE
  qh_cy c;
  u32 r = __builtin_addc(a, b, (qh_cy)ci, &c);
  co = c;
  return r;
#else
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, %1, %2, %3, %4" QH_D : "=v"(r), "=s"(co) : "v"(a), "v"(b), "s"(ci));
  return r;
#endif
}
__device__ __forceinline__ u32 qh_addck0(u32 a, u32 b, qh_lm ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, %1, %2, %3" QH_D : "=v"(r) : "v"(a), "v"(b), "s"(ci));
  return r;
}
__device__ __forceinline__ u32 qh_bit(qh_lm ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, 0, 0, %1" QH_D : "=v"(r) : "s"(ci));
  return r;
}

// QH_ARITH 3 (gfx11 and gfx12): as few carry instructions as can be. On an RX 7900 XT every carry instruction
// (v_add_co_u32, v_add_co_ci_u32, v_sub_co_ci_u32, ...) takes two cycles where a plain one (v_add_nc_u32,
// v_add3_u32, v_cndmask_b32, shifts) takes one, and a plain instruction can use a carry one's second cycle
// (qh_carry: carries alone 0.56x an add, half carries and half adds 0.97x). QH_ARITH 1 and 2 spend about 19,000
// carry instructions a nonce to 6,000 plain ones, so carries set the pace. QH_ARITH 3 keeps a carry
// instruction only where a carry or borrow is made, and adds carries in as values (v_cndmask) with plain adds.
// A lane mask as a value: m ? 1 : 0, and m ? -1 : 0.
__device__ __forceinline__ u32 qh_m1(qh_lm m) {
  u32 r;
  asm("v_cndmask_b32_e64 %0, 0, 1, %1" QH_D : "=v"(r) : "s"(m));
  return r;
}
__device__ __forceinline__ u32 qh_mn(qh_lm m) {
  u32 r;
  asm("v_cndmask_b32_e64 %0, 0, -1, %1" QH_D : "=v"(r) : "s"(m));
  return r;
}
// The high word of a wide add: 0 + c, h + c, h + h' + c, for the carry lane mask c.
#ifndef QH_HC
#define QH_HC 0
#endif
__device__ __forceinline__ u32 qh_hk0(qh_lm c) {
#if QH_ARITH >= 3 && !QH_HC
  return qh_m1(c);
#else
  return qh_bit(c);
#endif
}
__device__ __forceinline__ u32 qh_hk(u32 h, qh_lm c) {
#if QH_ARITH >= 3 && !QH_HC
  return h + qh_m1(c);
#else
  return qh_addk(h, c);
#endif
}
__device__ __forceinline__ u32 qh_hk2(u32 h, u32 h2, qh_lm c) {
#if QH_ARITH >= 3 && !QH_HC
  return h + h2 + qh_m1(c);
#else
  return qh_addck0(h, h2, c);
#endif
}

// QH_ARITH 3's reduction: (w1:w0) + 2^64 (w3 + k : w2) (mod p), as QH_R6 computes it: three carry instructions
// make the carry of w1 + w2 and the two borrows of the low word, and the high word takes them as values
// (v_cndmask) in plain adds: 3 carry + 5 plain instructions, where QH_RED 1 is 7 carry. QH_RB 1 takes the last
// borrow with a carry instruction instead (4 carry + 3 plain): +1.4% in wave32 on an RX 7900 XT, the default there.
// In wave64, where a v_cndmask costs less (0.7 cycles a 32 lanes on an RX 9070 XT, a carry instruction 2.0),
// QH_RB 0: +0.2-0.4% on the 7900 XT and RX 9060 XT, level on the 9070 XT.
#ifndef QH_C2C
#define QH_C2C 0
#endif
#ifndef QH_RB
#if QH_W64
#define QH_RB 0
#else
#define QH_RB 1
#endif
#endif
__device__ __forceinline__ u64 qh_red3(u32 w0, u32 w1, u32 w2, u32 w3, qh_lm k) {
  qh_lm c1, b1, b2;
  u32 h1 = qh_a_add(w1, w2, c1);
  u32 l1 = qh_a_subb(w0, w2, c1, b1);
  u32 l2 = qh_a_subb(l1, w3, k, b2);
#if QH_RB
  u32 h = qh_a_subb0(h1 + qh_m1(c1) + qh_mn(b1), b2);
#else
  u32 h = h1 + qh_m1(c1) + qh_mn(b1) + qh_mn(b2);
#endif
  return qh_pack(l2, h);
}

// QH_ARITH 2: (w1:w0) + 2^64 * (w3 + k : w2) (mod p) with k, a lane mask, the multiply's last carry: it
// rides in the first subtraction as its borrow in, so it costs nothing.
//   QH_RED 0: the exact reduction (qh_red4's, below), 8 VALU.
//   QH_RED 1: x = (w0 - w2 - w3) + (w1 + w2) * 2^32, with the carry of w1 + w2 as + 2^32 - 1. 7 VALU. It
//             drops the borrow or carry out of the last three steps, which happens only when a word is
//             within 2 of 0 or 2^32 (odds about 2^-31 a multiply, so about 1 nonce in 10^6 hashes wrong;
//             the host rechecks every hit).
__device__ __forceinline__ u64 qh_red4k(u32 w0, u32 w1, u32 w2, u32 w3, qh_lm k) {
#if QH_RED == 1
  qh_lm b1, b2, c1;
  u32 l1 = qh_a_sub(w0, w2, b1);
  u32 l2 = qh_a_subb(l1, w3, k, b2);
  u32 h1 = qh_a_add(w1, w2, c1);
  u32 h2 = qh_a_subb0(h1, b1);
  u32 h3 = qh_a_subb0(h2, b2);
  u32 h4 = qh_addk(h3, c1);
  u32 l3 = qh_a_subb0(l2, c1);
  return qh_pack(l3, h4);
#else
  qh_lm b, c, C, d;
  u32 ylo = qh_a_subb(~w2, w3, k, b);
  u32 yhi = qh_a_subb0(w2, b);
  u32 slo = qh_a_add(w0, ylo, c);
  u32 shi = qh_addck(w1, yhi, c, C);
  u32 m = qh_a_mask(C);
  u32 rlo = qh_a_sub(slo, m, d);
  u32 rhi = qh_a_subb0(shi, d);
  return qh_pack(rlo, rhi);
#endif
}

// x = (w1:w0) + 2^64 * (w3:w2) (mod p), to [0, 2^64). x = lo64 + w2 * EPS - w3 = lo64 + y - EPS with
// y = (w2 + 1) * EPS - w3 = (w2 : ~w2) - w3, in [0, 2^64 - 2^32]. If lo64 + y carries, the 2^64 is EPS and
// cancels the -EPS; if not, subtract EPS. That subtraction can wrap only when w2 = 0 and lo64 < w3 (odds
// about 2^-64); the host checks every hit anyway. 8 VALU (QH_RED 0; QH_RED 1 takes qh_red4k's 7 steps).
__device__ __forceinline__ u64 qh_red4(u32 w0, u32 w1, u32 w2, u32 w3) {
#if QH_RED == 1
  // QH_RED 1 with carry builtins: qh_red4k's 7-step reduction.
  qh_cy b1, b2, c1, z;
  u32 l1 = qh_subc(w0, w2, 0, b1);
  u32 l2 = qh_subc(l1, w3, 0, b2);
  u32 h1 = qh_addc(w1, w2, 0, c1);
  u32 h2 = qh_subc(h1, 0, b1, z);
  u32 h3 = qh_subc(h2, 0, b2, z);
  u32 h4 = qh_addc(h3, 0, c1, z);
  u32 l3 = qh_subc(l2, 0, c1, z);
  return qh_pack(l3, h4);
#else
  qh_cy b, c, C, d, z;
  u32 ylo = qh_subc(~w2, w3, 0, b);
  u32 yhi = qh_subc(w2, 0, b, z);
  u32 slo = qh_addc(w0, ylo, 0, c);
  u32 shi = qh_addc(w1, yhi, c, C);
  u32 m = C ? 0u : 0xffffffffu;
  u32 rlo = qh_subc(slo, m, 0, d);
  u32 rhi = qh_subc(shi, 0, d, z);
  return qh_pack(rlo, rhi);
#endif
}

// QH_R6 1 (QH_ARITH 1, gfx11 and gfx12): (w1:w0) + 2^64 (w3 + k : w2) (mod p), k a lane mask (the multiply's last
// carry, which belongs in w3), in 6 VALU in one asm block that carries its own s_delay_alu:
//   h = w1 + w2 (c1); l = w0 - w2 - c1 (b1); l = l - w3 - k (b2); h += c1; h -= b1; h -= b2.
// c1 is 2^64 = 2^32 - 1: +1 on the high word and -1 on the low, the latter folded into the first subtraction's
// borrow in. QH_RED 1's reduction takes 7 with the separate add of k. Like it, this drops the carry or borrow
// out of the high word's last steps (odds about 2^-31).
#ifndef QH_R6
#define QH_R6 0
#endif
#if QH_R6 && (defined(__GFX11__) || defined(__GFX12__))
#define QH_R6_ON 1
#else
#define QH_R6_ON 0
#endif
__device__ __forceinline__ u64 qh_red6k(u32 w0, u32 w1, u32 w2, u32 w3, u32 k) {
  u32 l, h, c1, b1, b2;
  asm("v_add_co_u32 %1, %2, %5, %6\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_sub_co_ci_u32_e64 %0, %3, %7, %6, %2\n\t"
      "v_add_co_ci_u32_e64 %1, null, %1, 0, %2\n\t"
      "s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)\n\t"
      "v_sub_co_ci_u32_e64 %0, %4, %0, %8, %9\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %1, 0, %3\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %1, 0, %4\n\t"
      "s_delay_alu instid0(VALU_DEP_1)"
      : "=&v"(l), "=&v"(h), "=&s"(c1), "=&s"(b1), "=&s"(b2)
      : "v"(w1), "v"(w2), "v"(w0), "v"(w3), "s"(k));
  return qh_pack(l, h);
}

// x = lo64 + 2^64 * h (mod p) for a small h: y = (h + 1) * EPS = (h : ~h) >= EPS, so it is exact. 6 VALU.
// QH_R96 1: x = lo64 + h * 2^32 - h in 3 VALU (a plain add of h to the high word, then h taken from the low word
// with its borrow). It drops the case where that passes 2^64, about h * 2^-32 (h is at most about 30 here), as
// QH_RED 1 drops its rare carries; the host rechecks every hit.
#ifndef QH_R96
#if defined(__GFX11__) || defined(__GFX12__)
#define QH_R96 1
#else
#define QH_R96 0
#endif
#endif
__device__ __forceinline__ u64 qh_reduce96(u64 lo64, u32 h) {
#if QH_R96
  qh_cy b, z;
  u32 hi1 = qh_hi(lo64) + h;
  u32 rlo = qh_subc(qh_lo(lo64), h, 0, b);
  u32 rhi = qh_subc(hi1, 0, b, z);
  return qh_pack(rlo, rhi);
#elif QH_ARITH == 2 && QH_WADD
  qh_lm c, C, d;
  u32 slo = qh_a_add(qh_lo(lo64), ~h, c);
  u32 shi = qh_addck(qh_hi(lo64), h, c, C);
  u32 m = qh_a_mask(C);
  u32 rlo = qh_a_sub(slo, m, d);
  u32 rhi = qh_a_subb0(shi, d);
#else
  qh_cy c, C, d, z;
  u32 slo = qh_addc(qh_lo(lo64), ~h, 0, c);
  u32 shi = qh_addc(qh_hi(lo64), h, c, C);
  u32 m = C ? 0u : 0xffffffffu;
  u32 rlo = qh_subc(slo, m, 0, d);
  u32 rhi = qh_subc(shi, 0, d, z);
#endif
  return qh_pack(rlo, rhi);
}

// QH_ARITH 4 (gfx11 and gfx12): a whole multiply and its reduction in one asm block, in eight VGPRs the compiler
// may not use (QH_FX to QH_FX + 7; the kernel's amdgpu_num_vgpr(QH_FX) keeps it below them). Two of them stay 0 for
// the whole kernel, so the multiply-adds whose addend is (a high word, 0) read it in place: the compiler's way copies
// the high word next to a zero register first, two v_mov_b32 a multiply. QH_SALU 1 (gfx11) also folds the reduction's
// three high-word corrections (+c1 - b1 - b2) into two lane masks with scalar instructions, which run beside the
// vector ones: 2 v_cndmask + 1 add where QH_ARITH 3 takes 3 + 2.
#ifndef QH_FX
#define QH_FX 72
#endif
#if QH_FX == 72
#define QH_X0 "72"
#define QH_X1 "73"
#define QH_X2 "74"
#define QH_X3 "75"
#define QH_X4 "76"
#define QH_X5 "77"
#define QH_X6 "78"
#define QH_X7 "79"
#elif QH_FX == 64
#define QH_X0 "64"
#define QH_X1 "65"
#define QH_X2 "66"
#define QH_X3 "67"
#define QH_X4 "68"
#define QH_X5 "69"
#define QH_X6 "70"
#define QH_X7 "71"
#elif QH_FX == 56
#define QH_X0 "56"
#define QH_X1 "57"
#define QH_X2 "58"
#define QH_X3 "59"
#define QH_X4 "60"
#define QH_X5 "61"
#define QH_X6 "62"
#define QH_X7 "63"
#else
#error "QH_FX: 56, 64 or 72"
#endif
#define QH_XCLOB "v" QH_X0, "v" QH_X1, "v" QH_X2, "v" QH_X3, "v" QH_X4, "v" QH_X5, "v" QH_X6, "v" QH_X7
#if QH_W64
#define QH_SB "_b64"
#else
#define QH_SB "_b32"
#endif
#ifndef QH_SALU
#if defined(__GFX11__)
#define QH_SALU 1
#else
#define QH_SALU 0
#endif
#endif
// QH_HZ 1: s_waitcnt_depctr sa_sdst(0) between a scalar write of a mask and a scalar read of it. On gfx11 a scalar
// write to an SGPR a vector instruction lately read as a lane mask, then a scalar read of it, can read the old value
// (LLVM's VALUMaskWriteHazard), and the registers come from the compiler, so the block can't rule that out.
#ifndef QH_HZ
#define QH_HZ 1
#endif
#if QH_HZ && defined(__GFX12__)
#define QH_SAW "\n\ts_wait_alu 0xfffe"
#elif QH_HZ
#define QH_SAW "\n\ts_waitcnt_depctr 0xfffe"
#else
#define QH_SAW ""
#endif

// Zero the two registers the blocks read as zero high words. Every block names all eight as clobbered, so the
// compiler keeps the blocks in order after this one.
__device__ __forceinline__ void qh_x_init() {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("v_mov_b32 v" QH_X2 ", 0\n\tv_mov_b32 v" QH_X5 ", 0" ::: QH_XCLOB);
#endif
}

// The reduction's tail inside a block: w0 = X0, w1 = X3, w2 = X6, w3 = X7, k (%2) the multiply's last carry; l in
// %0, h in %1. Operands %3-%8 are scratch masks; X1 and X4 (A's and C's high words, read by then) are scratch VGPRs.
#if QH_SALU == 1
// T2 = c1 ^ b1 ^ b2 (the correction is odd), T1 = the correction is negative; h += T2 - 2 T1.
#define QH_RTAIL                                                   \
  "v_add_co_u32 %1, %3, v" QH_X3 ", v" QH_X6 "\n\t"                \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_sub_co_ci_u32_e64 %0, %4, v" QH_X0 ", v" QH_X6 ", %3\n\t"     \
  "s_xor" QH_SB " %6, %3, %4\n\t"                                  \
  "s_and_not1" QH_SB " %7, %4, %3\n\t"                             \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_sub_co_ci_u32_e64 %0, %5, %0, v" QH_X7 ", %2" QH_SAW "\n\t"   \
  "s_and_not1" QH_SB " %8, %5, %6\n\t"                             \
  "s_xor" QH_SB " %6, %6, %5" QH_SAW "\n\t"                        \
  "s_xor" QH_SB " %7, %7, %8\n\t"                                  \
  "v_cndmask_b32_e64 v" QH_X1 ", 0, 1, %6\n\t"                     \
  "v_cndmask_b32_e64 v" QH_X4 ", 0, -2, %7\n\t"                    \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_add3_u32 %1, %1, v" QH_X1 ", v" QH_X4 "\n\t"                  \
  "s_delay_alu instid0(VALU_DEP_1)"
#else
#define QH_RTAIL                                                   \
  "v_add_co_u32 %1, %3, v" QH_X3 ", v" QH_X6 "\n\t"                \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_sub_co_ci_u32_e64 %0, %4, v" QH_X0 ", v" QH_X6 ", %3\n\t"     \
  "v_cndmask_b32_e64 v" QH_X1 ", 0, 1, %3\n\t"                     \
  "s_delay_alu instid0(VALU_DEP_2)\n\t"                            \
  "v_sub_co_ci_u32_e64 %0, %5, %0, v" QH_X7 ", %2\n\t"             \
  "v_cndmask_b32_e64 v" QH_X4 ", 0, -1, %4\n\t"                    \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_add3_u32 %1, %1, v" QH_X1 ", v" QH_X4 "\n\t"                  \
  "v_cndmask_b32_e64 v" QH_X1 ", 0, -1, %5\n\t"                    \
  "s_delay_alu instid0(VALU_DEP_1)\n\t"                            \
  "v_add_nc_u32 %1, %1, v" QH_X1 "\n\t"                            \
  "s_delay_alu instid0(VALU_DEP_1)"
#endif

__device__ __forceinline__ u64 qh_mul_x(u32 a0, u32 a1, u32 b0, u32 b1) {
  u32 l, h;
  qh_lm k, m3, m4, m5, m6, m7, m8;
#if defined(__HIP_DEVICE_COMPILE__)
  // l and h are written only after the last read of a and b, so they may take their registers.
  asm("v_mad_u64_u32 v[" QH_X0 ":" QH_X1 "], null, %9, %11, 0\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X6 ":" QH_X7 "], null, %9, %12, v[" QH_X1 ":" QH_X2 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X3 ":" QH_X4 "], %2, %10, %11, v[" QH_X6 ":" QH_X7 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X6 ":" QH_X7 "], null, %10, %12, v[" QH_X4 ":" QH_X5 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t" QH_RTAIL
      : "=v"(l), "=v"(h), "=&s"(k), "=&s"(m3), "=&s"(m4), "=&s"(m5), "=&s"(m6), "=&s"(m7), "=&s"(m8)
      : "v"(a0), "v"(a1), "v"(b0), "v"(b1)
      : QH_XCLOB);
#else
  l = h = 0;
#endif
  return qh_pack(l, h);
}

// diag * s + a for a uniform diag (in SGPRs) and a 96-bit a: a's low 64 bits ride in the first multiply-add, a's
// high word and that multiply-add's carry go into C's high word with one carry instruction (it wraps only when C's
// high word is within a.h + 1 of 2^32, odds below 2^-28, as QH_MULADD 2 drops), and D reads it in place.
__device__ __forceinline__ u64 qh_muladd_x(u64 s, u64 diag, u64 alo, u32 ah) {
  u32 l, h;
  qh_lm k, kA, m3, m4, m5, m6, m7, m8;
#if defined(__HIP_DEVICE_COMPILE__)
  asm("v_mad_u64_u32 v[" QH_X0 ":" QH_X1 "], %9, %10, %12, %14\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X6 ":" QH_X7 "], null, %10, %13, v[" QH_X1 ":" QH_X2 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X3 ":" QH_X4 "], %2, %11, %12, v[" QH_X6 ":" QH_X7 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_add_co_ci_u32_e64 v" QH_X4 ", null, v" QH_X4 ", %15, %9\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t"
      "v_mad_u64_u32 v[" QH_X6 ":" QH_X7 "], null, %11, %13, v[" QH_X4 ":" QH_X5 "]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)\n\t" QH_RTAIL
      : "=v"(l), "=v"(h), "=&s"(k), "=&s"(m3), "=&s"(m4), "=&s"(m5), "=&s"(m6), "=&s"(m7), "=&s"(m8),
        "=&s"(kA)
      : "v"(qh_lo(s)), "v"(qh_hi(s)), "s"(qh_lo(diag)), "s"(qh_hi(diag)), "v"(alo), "v"(ah)
      : QH_XCLOB);
#else
  l = h = 0;
#endif
  return qh_pack(l, h);
}

// a * b as four words: A = a0 b0, X = a0 b1 + A's high word, C = a1 b0 + X (65 bits), D = a1 b1 + C's high
// word, then D's high word gets C's carry. Four v_mad_u64_u32 and three VALU.
struct QhP4 {
  u32 w0, w1, w2, w3;
};

__device__ __forceinline__ QhP4 qh_mul4(u64 a, u64 b) {
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  qh_lm k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  QhP4 r;
  r.w0 = qh_lo(A);
  r.w1 = qh_lo(C);
  r.w2 = qh_lo(D);
  r.w3 = qh_addk(qh_hi(D), k);
  return r;
}

__device__ __forceinline__ u64 qh_mul(u64 a, u64 b) {
#if QH_ARITH == 4
  return qh_mul_x(qh_lo(a), qh_hi(a), qh_lo(b), qh_hi(b));
#elif QH_ARITH == 2
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  qh_lm k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  return qh_red4k(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#elif QH_ARITH == 3
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  qh_lm k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  return qh_red3(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#elif QH_R6_ON
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  qh_lm k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  return qh_red6k(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#else
  QhP4 p = qh_mul4(a, b);
  return qh_red4(p.w0, p.w1, p.w2, p.w3);
#endif
}

// QH_SQR 1 (QH_ARITH 2): a^2 from three multiplies, a0^2, a0 a1 and a1^2, the middle one doubled with
// adds: 3 v_mad_u64_u32 and 5 VALU before the reduction, against 4 and 2.
__device__ __forceinline__ u64 qh_sqr(u64 a) {
#if QH_ARITH == 4
  return qh_mul_x(qh_lo(a), qh_hi(a), qh_lo(a), qh_hi(a));
#elif QH_ARITH == 3 && QH_SQR
  // 2M by shifts: its low word, its high word, and the bit that leaves at 2^128 (weight 2^96 here).
  u32 a0 = qh_lo(a), a1 = qh_hi(a);
  u64 A = (u64)a0 * a0, M = (u64)a0 * a1, D = (u64)a1 * a1;
  u32 ml = qh_lo(M), mh = qh_hi(M);
  u32 tl = ml << 1, th = (mh << 1) | (ml >> 31), top = mh >> 31;
  qh_lm c2, c4;
  u32 w1 = qh_a_add(qh_hi(A), tl, c2);
  u32 w2 = qh_addck(qh_lo(D), th, c2, c4);
  return qh_red3(qh_lo(A), w1, w2, qh_hi(D) + top, c4);
#elif (QH_ARITH == 2 || QH_R6_ON) && QH_SQR
  u32 a0 = qh_lo(a), a1 = qh_hi(a);
  u64 A = (u64)a0 * a0, M = (u64)a0 * a1, D = (u64)a1 * a1;
  qh_lm c1, c2, c3, c4;
  u32 t = qh_a_add(qh_lo(M), qh_lo(M), c1);
  u32 w1 = qh_a_add(qh_hi(A), t, c2);
  u32 u = qh_addck(qh_hi(M), qh_hi(M), c1, c3);
  u32 w2 = qh_addck(qh_lo(D), u, c2, c4);
  u32 w3 = qh_addk(qh_hi(D), c3);
#if QH_R6_ON
  return qh_red6k(qh_lo(A), w1, w2, w3, c4);
#else
  return qh_red4k(qh_lo(A), w1, w2, w3, c4);
#endif
#elif QH_SQR
  // The same with carry builtins.
  u32 a0 = qh_lo(a), a1 = qh_hi(a);
  u64 A = (u64)a0 * a0, M = (u64)a0 * a1, D = (u64)a1 * a1;
  qh_cy c1, c2, c3, c4, z;
  u32 t = qh_addc(qh_lo(M), qh_lo(M), 0, c1);
  u32 w1 = qh_addc(qh_hi(A), t, 0, c2);
  u32 u = qh_addc(qh_hi(M), qh_hi(M), c1, c3);
  u32 w2 = qh_addc(qh_lo(D), u, c2, c4);
  u32 w3 = qh_addc(qh_addc(qh_hi(D), 0, c3, z), 0, c4, z);
  return qh_red4(qh_lo(A), w1, w2, w3);
#else
  return qh_mul(a, a);
#endif
}

__device__ __forceinline__ u64 qh_sbox(u64 x) {
  u64 x2 = qh_sqr(x);
  u64 x3 = qh_mul(x2, x);
  u64 x4 = qh_sqr(x2);
  return qh_mul(x4, x3);
}

// A 96-bit accumulator: lo64 + 2^64 * h.
struct QhWide {
  u64 lo;
  u32 h;
};

__device__ __forceinline__ QhWide qh_w(u64 x) { QhWide w; w.lo = x; w.h = 0; return w; }

// QH_WADD 1 (QH_ARITH 2): the three steps in inline asm with mask carries, and a known-zero high word costs
// no register: 3 VALU an add.
__device__ __forceinline__ QhWide qh_wadd(QhWide w, u64 x) {
  QhWide r;
#if (QH_ARITH == 2 && QH_WADD) || QH_ARITH >= 3
  qh_lm c1, c2;
  u32 l = qh_a_add(qh_lo(w.lo), qh_lo(x), c1);
  u32 h = qh_addck(qh_hi(w.lo), qh_hi(x), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = (__builtin_constant_p(w.h) && w.h == 0) ? qh_hk0(c2) : qh_hk(w.h, c2);
#else
  qh_cy c1, c2, z;
  u32 l = qh_addc(qh_lo(w.lo), qh_lo(x), 0, c1);
  u32 h = qh_addc(qh_hi(w.lo), qh_hi(x), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = qh_addc(w.h, 0, c2, z);
#endif
  return r;
}

// w + c for a uniform c (a round constant, a kernel argument): c's words are read from SGPRs, so they need no
// copy to VGPRs, and the compiler keeps loading the constants a row at a time.
__device__ __forceinline__ QhWide qh_waddc(QhWide w, u64 c) {
  QhWide r;
#if (QH_ARITH == 2 && QH_WADD) || QH_ARITH >= 3
  qh_lm c1, c2;
  u32 l, h;
  asm("v_add_co_u32 %0, %1, %2, %3" QH_D : "=v"(l), "=s"(c1) : "v"(qh_lo(w.lo)), "s"(qh_lo(c)));
  asm("v_add_co_ci_u32_e64 %0, %1, %2, %3, %4" QH_D : "=v"(h), "=s"(c2) : "v"(qh_hi(w.lo)), "s"(qh_hi(c)), "s"(c1));
  r.lo = qh_pack(l, h);
  r.h = (__builtin_constant_p(w.h) && w.h == 0) ? qh_hk0(c2) : qh_hk(w.h, c2);
#else
  r = qh_wadd(w, c);
#endif
  return r;
}

__device__ __forceinline__ QhWide qh_wadd(QhWide w, QhWide x) {
  QhWide r;
#if (QH_ARITH == 2 && QH_WADD) || QH_ARITH >= 3
  qh_lm c1, c2;
  u32 l = qh_a_add(qh_lo(w.lo), qh_lo(x.lo), c1);
  u32 h = qh_addck(qh_hi(w.lo), qh_hi(x.lo), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = qh_hk2(w.h, x.h, c2);
#else
  qh_cy c1, c2, z;
  u32 l = qh_addc(qh_lo(w.lo), qh_lo(x.lo), 0, c1);
  u32 h = qh_addc(qh_hi(w.lo), qh_hi(x.lo), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = qh_addc(w.h, x.h, c2, z);
#endif
  return r;
}

// diag * s + a (a 96-bit sum), reduced.
//   QH_ARITH 2: a's low 64 bits ride in the first multiply-add, its high word in the third word, and the
//               multiplies' last carry in the reduction: 4 v_mad_u64_u32 and 4 + 7 VALU.
//   QH_ARITH 1, QH_MULADD 1: the same with carry builtins, 4 and 5 + 8. QH_MULADD 0: the product, then a
//               128-bit add, 4 and 3 + 4 + 8.
//   QH_ARITH 1, QH_MULADD 2: a's high word and the first multiply-add's carry ride in the fourth multiply's
//               addend (C's high word + a.h + kA, one VALU in place of QH_MULADD 1's copy), so D's words need no
//               adds after it: 2 VALU fewer. C.hi + a.h wraps only when C.hi is within a.h (at most 12) of
//               2^32, odds below 2^-28 a multiply; that carry is dropped, as QH_RED 1 drops its rare ones.
__device__ __forceinline__ u64 qh_muladd(u64 s, u64 diag, QhWide a) {
#if QH_ARITH == 4
  return qh_muladd_x(s, diag, a.lo, a.h);
#elif QH_ARITH == 3
  // QH_MULADD 3's shape, the first carry added in as a value.
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  qh_lm kA, k;
  u64 A = qh_madc_s(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc_s(a1, b0, X, k);
#if QH_C2C
  u32 c2 = qh_addck0(qh_hi(C), a.h, kA);
#else
  u32 c2 = qh_hi(C) + a.h + qh_m1(kA);
#endif
  u64 D = (u64)a1 * b1 + c2;
  return qh_red3(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#elif QH_ARITH == 2
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  qh_lm kA, k, c;
  u64 A = qh_madc_s(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc_s(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  u32 w2 = qh_addck(qh_lo(D), a.h, kA, c);
  u32 w3 = qh_addk(qh_hi(D), c);
  return qh_red4k(qh_lo(A), qh_lo(C), w2, w3, k);
#elif QH_MULADD >= 2
  // QH_MULADD 3: the diagonal's low word straight from its SGPR in the asm multiply-adds (qh_madc_s), where the
  // "v" operand of qh_madc makes the compiler copy it to a VGPR each time.
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  qh_lm kA, k;
#if QH_MULADD == 3 && !QH_PURE
  u64 A = qh_madc_s(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc_s(a1, b0, X, k);
#else
  u64 A = qh_madc(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc(a1, b0, X, k);
#endif
#if QH_PURE
  u32 c2 = qh_hi(C) + a.h + (u32)kA;
#else
  u32 c2 = qh_addck0(qh_hi(C), a.h, kA);
#endif
  u64 D = (u64)a1 * b1 + c2;
#if QH_R6_ON
  return qh_red6k(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#else
  return qh_red4(qh_lo(A), qh_lo(C), qh_lo(D), qh_addk(qh_hi(D), k));
#endif
#elif QH_MULADD
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  qh_lm kA, k, c;
  u64 A = qh_madc(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  u32 w2 = qh_addck(qh_lo(D), a.h, kA, c);
  u32 w3 = qh_addk(qh_addk(qh_hi(D), c), k);
  return qh_red4(qh_lo(A), qh_lo(C), w2, w3);
#else
  QhP4 p = qh_mul4(s, diag);
  qh_cy c1, c2, c3, z;
  u32 w0 = qh_addc(p.w0, qh_lo(a.lo), 0, c1);
  u32 w1 = qh_addc(p.w1, qh_hi(a.lo), c1, c2);
  u32 w2 = qh_addc(p.w2, a.h, c2, c3);
  u32 w3 = qh_addc(p.w3, 0, c3, z);
  return qh_red4(w0, w1, w2, w3);
#endif
}

// 2x as a wide value, by shifts (QH_DBL): three plain instructions where an add of x takes two carry ones.
#ifndef QH_DBL
#define QH_DBL 0
#endif
__device__ __forceinline__ QhWide qh_w2(u64 x) {
  u32 l = qh_lo(x), h = qh_hi(x);
  QhWide w;
  w.lo = qh_pack(l << 1, (h << 1) | (l >> 31));
  w.h = h >> 31;
  return w;
}

// The external layer (M_4 on each group of four, then each element plus its column's sum across the
// three groups), plus the constants c, reduced to [0, 2^64).
__device__ __forceinline__ void qh_external(u64 *s, const u64 *c) {
  QhWide y[12];
#pragma unroll
  for (int g = 0; g < 12; g += 4) {
    u64 x0 = s[g], x1 = s[g + 1], x2 = s[g + 2], x3 = s[g + 3];
    QhWide t01 = qh_wadd(qh_w(x0), x1);
    QhWide t23 = qh_wadd(qh_w(x2), x3);
    QhWide t0123 = qh_wadd(t01, t23);
    QhWide t01123 = qh_wadd(t0123, x1);
    QhWide t01233 = qh_wadd(t0123, x3);
#if QH_DBL
    y[g + 3] = qh_wadd(t01233, qh_w2(x0));       // 3x0 + x1 + x2 + 2x3
    y[g + 1] = qh_wadd(t01123, qh_w2(x2));       // x0 + 2x1 + 3x2 + x3
#else
    y[g + 3] = qh_wadd(qh_wadd(t01233, x0), x0);  // 3x0 + x1 + x2 + 2x3
    y[g + 1] = qh_wadd(qh_wadd(t01123, x2), x2);  // x0 + 2x1 + 3x2 + x3
#endif
    y[g + 0] = qh_wadd(t01123, t01);              // 2x0 + 3x1 + x2 + x3
    y[g + 2] = qh_wadd(t01233, t23);              // x0 + x1 + 2x2 + 3x3
  }
#pragma unroll
  for (int k = 0; k < 4; k++) {
    QhWide sum = qh_wadd(qh_wadd(y[k], y[k + 4]), y[k + 8]);
#pragma unroll
    for (int g = 0; g < 12; g += 4) {
      QhWide w = qh_waddc(qh_wadd(y[g + k], sum), c[g + k]);
      s[g + k] = qh_reduce96(w.lo, w.h);
    }
  }
}

// Element 0 of the external layer, plus c0: 2 y0 + y4 + y8 + c0, with y_g = 2 x_g + 3 x_{g+1} + x_{g+2} + x_{g+3}.
// The hash reads only element 0 of the last row (QH_LAST1), so this replaces that row's whole layer.
__device__ __forceinline__ u64 qh_external0(const u64 *s, u64 c0) {
  QhWide y[3];
#pragma unroll
  for (int g = 0; g < 3; g++) {
    u64 x0 = s[4 * g], x1 = s[4 * g + 1], x2 = s[4 * g + 2], x3 = s[4 * g + 3];
    QhWide t01 = qh_wadd(qh_w(x0), x1);
    QhWide t = qh_wadd(qh_wadd(t01, t01), x1);
    y[g] = qh_wadd(qh_wadd(t, x2), x3);
  }
  QhWide r = qh_wadd(qh_wadd(qh_wadd(y[0], y[0]), y[1]), y[2]);
  r = qh_waddc(r, c0);
  return qh_reduce96(r.lo, r.h);
}

// The internal layer: s[i] = sum + diag[i] * s[i], and element 0 also gets c0.
__device__ __forceinline__ void qh_internal(u64 *s, const u64 *diag, u64 c0) {
  QhWide sum = qh_w(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) sum = qh_wadd(sum, s[i]);
  QhWide sum0 = qh_waddc(sum, c0);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = qh_muladd(s[i], diag[i], i == 0 ? sum0 : sum);
}

__device__ __forceinline__ u32 qh_bswap32(u32 v) { return __builtin_bswap32(v); }

// QH_NPT: nonces a thread, hashed side by side so the compiler can interleave them (instruction-level
// parallelism; RDNA 3 needs it more than RDNA 2). QH_PIPE: each partial round's element 0 first, and the next
// round's S-box on it beside this round's other eleven products, which don't depend on it.
#ifndef QH_NPT
#if defined(__GFX11__) || defined(__GFX12__)
#define QH_NPT QH_NPT_GFX11
#else
#define QH_NPT 1
#endif
#endif
#ifndef QH_NPT_GFX11
#define QH_NPT_GFX11 1
#endif
// The nonces a thread this target's code object hashes: the host reads it to size its launches, since a
// build's targets can differ.
__constant__ u32 qh_npt = QH_NPT;
#ifndef QH_PIPE
#define QH_PIPE 0
#endif
// QH_LAST1: the last row's external layer computes element 0 only, the one the hash reads (qh_external0).
#ifndef QH_LAST1
#if defined(__GFX11__) || defined(__GFX12__)
#define QH_LAST1 1
#else
#define QH_LAST1 0
#endif
#endif

// One partial round of the internal layer for s, its S-boxed element 0 in x: s[0] gets the new element 0 with
// the next round's constant c0, and (QH_PIPE) x the next round's S-box of it when `next`.
__device__ __forceinline__ void qh_partial_pipe(u64 *s, u64 &x, u64 c0, bool next) {
  QhWide sum = qh_w(x);
#pragma unroll
  for (int i = 1; i < 12; i++) sum = qh_wadd(sum, s[i]);
  s[0] = qh_muladd(x, qh_tab.diag[0], qh_waddc(sum, c0));
  if (next) x = qh_sbox(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) s[i] = qh_muladd(s[i], qh_tab.diag[i], sum);
}

// The two permutations from the state of N nonces after block 3's first round (qh_first_round); element 0
// of each result ends in s[n][0], in [0, 2^64).
template <int N>
__device__ __forceinline__ void qh_hash(u64 (&s)[N][12]) {
  int row = 1;  // row 0 is qh_first_round's
#pragma unroll 1
  for (int seg = 0; seg < 3; seg++) {
    int rows = seg == 1 ? 9 : 4;
#pragma unroll 1
    for (int r = seg ? 0 : 1; r < rows; r++, row++) {
      if (row != QH_NOSBOX_ROW) {
#pragma unroll
        for (int i = 0; i < 12; i++)
#pragma unroll
          for (int n = 0; n < N; n++) s[n][i] = qh_sbox(s[n][i]);
      }
#pragma unroll
      for (int n = 0; n < N; n++) {
#if QH_LAST1
        if (row == QH_ROWS - 1)
          s[n][0] = qh_external0(s[n], qh_tab.full[row][0]);
        else
#endif
          qh_external(s[n], qh_tab.full[row]);
      }
    }
    if (seg == 2) break;
#if QH_PIPE
    u64 x[N];
#pragma unroll
    for (int n = 0; n < N; n++) x[n] = qh_sbox(s[n][0]);
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
#pragma unroll
      for (int n = 0; n < N; n++) qh_partial_pipe(s[n], x[n], qh_tab.part0[r], r < 21);
    }
#else
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
#pragma unroll
      for (int n = 0; n < N; n++) {
        s[n][0] = qh_sbox(s[n][0]);
        qh_internal(s[n], qh_tab.diag, qh_tab.part0[r]);
      }
    }
#endif
#pragma unroll
    for (int n = 0; n < N; n++)
#pragma unroll
      for (int i = 1; i < 12; i++) {
        QhWide w = qh_waddc(qh_w(s[n][i]), qh_tab.term0[i]);
        s[n][i] = qh_reduce96(w.lo, w.h);
      }
  }
}

// The state after block 3's first round for the launch's point i, with the second round's constants.
// Along a line only elements 0 and 4 change before that round's S-boxes: x0 = X0 + 35 t, x4 = X4 - 35 t.
// With a = x0^7 and b = x4^7 the round's linear layer gives
//   s[4g + k] = C[4g + k] + M[k] * (a * (g == 0 ? 2 : 1) + b * (g == 1 ? 2 : 1)),   M = (2, 1, 1, 3),
// C being the host's constants for the other ten S-boxes (qpow.js has the algebra). d = 35 i fits a u32
// because a launch is at most 2^22 points; x4 takes p - d.
__device__ __forceinline__ void qh_first_round(const QhArgs &a, u32 i, u64 *s) {
  const u64 d = (u64)(QH_SLOPE * i);
  QhWide w0 = qh_waddc(qh_w(d), a.x0);
  QhWide w4 = qh_waddc(qh_w(QH_P - d), a.x4);
  const u64 sa = qh_sbox(qh_reduce96(w0.lo, w0.h));
  const u64 sb = qh_sbox(qh_reduce96(w4.lo, w4.h));
  QhWide v[3];
  v[2] = qh_wadd(qh_w(sa), sb);
  v[0] = qh_wadd(v[2], sa);
  v[1] = qh_wadd(v[2], sb);
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const QhWide v2 = qh_wadd(v[g], v[g]);
    const QhWide v3 = qh_wadd(v2, v[g]);
    QhWide w;
    w = qh_waddc(v2, a.k[4 * g]);
    s[4 * g] = qh_reduce96(w.lo, w.h);
    w = qh_waddc(v[g], a.k[4 * g + 1]);
    s[4 * g + 1] = qh_reduce96(w.lo, w.h);
    w = qh_waddc(v[g], a.k[4 * g + 2]);
    s[4 * g + 2] = qh_reduce96(w.lo, w.h);
    w = qh_waddc(v3, a.k[4 * g + 3]);
    s[4 * g + 3] = qh_reduce96(w.lo, w.h);
  }
}

// The block size, and (QH_WPE) the fewest waves a SIMD the compiler must leave room for: a lower number lets it
// use more registers.
#ifndef QH_BLOCK
#define QH_BLOCK 256
#endif
#ifdef QH_WPE
#define QH_ATTR __attribute__((amdgpu_waves_per_eu(QH_WPE)))
#elif QH_ARITH == 4
// The compiler's registers stop below the blocks' eight (QH_FX up).
#define QH_ATTR __attribute__((amdgpu_num_vgpr(QH_FX)))
#else
#define QH_ATTR
#endif

extern "C" __global__ void __launch_bounds__(QH_BLOCK) QH_ATTR qh_search(QhArgs a, u64 *hits) {
  u32 i0 = (blockIdx.x * blockDim.x + threadIdx.x) * QH_NPT;
  if (i0 >= a.n) return;
#if QH_ARITH == 4
  qh_x_init();
#endif
  u64 s[QH_NPT][12];
#pragma unroll
  for (int n = 0; n < QH_NPT; n++) qh_first_round(a, i0 + n, s[n]);
  qh_hash<QH_NPT>(s);
#pragma unroll
  for (int n = 0; n < QH_NPT; n++) {
    u64 r = s[n][0];
    if (r >= QH_P) r -= QH_P;
    u64 top = ((u64)qh_bswap32(qh_lo(r)) << 32) | qh_bswap32(qh_hi(r));
    // With QH_NPT > 1 a thread's last nonces can lie past the launch's end.
    if (top <= a.target && (QH_NPT == 1 || i0 + n < a.n)) {
      unsigned long long slot = atomicAdd((unsigned long long *)hits, 1ull);
      if (slot < a.cap) hits[1 + slot] = (u64)a.base + i0 + n;
    }
  }
}
