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
//   QH_ARITH 2|1|0    carries as inline-asm lane masks | clang carry builtins | plain C++: 52.0 | 45.2 | 35.5 MH/s
//   QH_RED 1|0        a 7-instruction reduction that drops carries rarer than 2^-31 | the exact 8: +2.9%
//   QH_SQR 1|0        squares from three multiplies | four: +2.7%
//   QH_WADD 1|0       wide adds in asm, round constants read from SGPRs | carry builtins: +1.4%
//   QH_MULADD 1|0     the internal layer's sum rides in the multiplies | added after (QH_ARITH 1): +2.5%
//   QH_BLOCK, QH_WPE, QH_NPT, QH_PUNROLL  block size, waves per SIMD, nonces a thread, partial-round unroll
// The asm passes each carry as one SGPR, so it is for wave32, HIP's default on gfx10 and later.

#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define QH_EPS 0xffffffffull
#define QH_P 0xffffffff00000001ull

// The schedule. A full row is "S-box all 12 (unless the row says not to), the external layer, then
// add the row's constants": the constants of the next round ride in this one's layer sums.
//   rows 0-3    block 3, initial rounds 0-3 (the host folded the first layer and round 0's constants in)
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
  u64 k[12];       // the state after block 3's first layer and round-0 constants, counter word 7 = 0
  u64 target;      // a hit is top64 <= target
  u32 hi;          // counter >> 32 for this launch
  u32 lo0;         // first counter & 0xffffffff
  u32 n;           // counters in this launch
  u32 cap;         // hit slots
};

// ── Field helpers ────────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ u32 qh_lo(u64 x) { return (u32)x; }
__device__ __forceinline__ u32 qh_hi(u64 x) { return (u32)(x >> 32); }
__device__ __forceinline__ u64 qh_pack(u32 lo, u32 hi) { return ((u64)hi << 32) | lo; }

#ifndef QH_ARITH
#define QH_ARITH 2
#endif

#if QH_ARITH == 0
// ── QH_ARITH 0: plain C++ (the first version) ──────────────────────────────────────────────────────
// x = lo64 + 2^64 * hi64 (mod p), to [0, 2^64). With r2, r3 the words of hi64:
// x = lo64 + r2 * EPS - r3 = lo64 + y - EPS where y = (r2 + 1) * EPS - r3 = (r2 : ~r2) - r3, which is in
// [0, 2^64 - 2^32]. If lo64 + y carries, the 2^64 is EPS and cancels the -EPS; if not, subtract EPS.
// That last subtraction can wrap only when r2 = 0 and lo64 < r3 (odds about 2^-64); the host checks
// every hit anyway.
__device__ __forceinline__ u64 qh_reduce128(u64 lo64, u64 hi64) {
  u32 r2 = qh_lo(hi64), r3 = qh_hi(hi64);
  u64 y = qh_pack(~r2, r2) - r3;
  u64 s = lo64 + y;
  bool c = s < y;
  return s - (c ? 0ull : QH_EPS);
}

// x = lo64 + 2^64 * h (mod p) for a small h (a sum of a few 64-bit values), to [0, 2^64). Exact: with
// r3 = 0, y = (h + 1) * EPS >= EPS, so s >= EPS when there is no carry.
__device__ __forceinline__ u64 qh_reduce96(u64 lo64, u32 h) {
  u64 y = qh_pack(~h, h);
  u64 s = lo64 + y;
  bool c = s < y;
  return s - (c ? 0ull : QH_EPS);
}

// a * b as 128 bits: four 32 x 32 -> 64 multiplies (v_mad_u64_u32).
__device__ __forceinline__ void qh_mul128(u64 a, u64 b, u64 &lo, u64 &hi) {
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 t0 = (u64)a0 * b0;
  u64 t1 = (u64)a0 * b1 + (t0 >> 32);
  u64 t2 = (u64)a1 * b0 + (u32)t1;
  hi = (u64)a1 * b1 + (t1 >> 32) + (t2 >> 32);
  lo = qh_pack((u32)t0, (u32)t2);
}

__device__ __forceinline__ u64 qh_mul(u64 a, u64 b) {
  u64 lo, hi;
  qh_mul128(a, b, lo, hi);
  return qh_reduce128(lo, hi);
}

// QH_SQR=1 squares with three 32 x 32 multiplies and a shift for the doubled middle one, in place of four.
#ifndef QH_SQR
#define QH_SQR 0
#endif

__device__ __forceinline__ u64 qh_sqr(u64 a) {
#if QH_SQR
  u32 a0 = qh_lo(a), a1 = qh_hi(a);
  u64 ll = (u64)a0 * a0, lh = (u64)a0 * a1, hh = (u64)a1 * a1;
  // a^2 = ll + lh * 2^33 + hh * 2^64
  u64 lo = ll + ((u64)((u32)lh << 1) << 32);
  u64 c = lo < ll ? 1ull : 0ull;
  u64 hi = hh + (lh >> 31) + c;
  return qh_reduce128(lo, hi);
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

__device__ __forceinline__ QhWide qh_wadd(QhWide w, u64 x) {
  QhWide r;
  r.lo = w.lo + x;
  r.h = w.h + (r.lo < x ? 1u : 0u);
  return r;
}

__device__ __forceinline__ QhWide qh_wadd(QhWide w, QhWide x) {
  QhWide r;
  r.lo = w.lo + x.lo;
  r.h = w.h + x.h + (r.lo < x.lo ? 1u : 0u);
  return r;
}

__device__ __forceinline__ u64 qh_muladd(u64 s, u64 diag, QhWide a) {
  u64 lo, hi;
  qh_mul128(s, diag, lo, hi);
  u64 l2 = lo + a.lo;
  u64 h2 = hi + a.h + (l2 < lo ? 1u : 0u);
  return qh_reduce128(l2, h2);
}

#else
// ── QH_ARITH 1 and 2 (2 is the default): carries kept in carry flags ────────────────────────────────
// Written so the compiler keeps every carry in a lane mask (v_add_co_u32 / v_add_co_ci_u32) instead of
// recomputing it with 64-bit compares, as the plain C++ above made it do (32 VALU a multiply, not 27).
// QH_ARITH 1 uses clang's carry builtins and inline asm only for the multiply-add whose carry C++ can't
// express; QH_ARITH 2 does every carry step of a multiply and reduction in inline asm, each carry a lane mask
// in an SGPR (wave32), which saves the copies and the separate add of the last carry.

typedef unsigned int qh_cy;  // a carry, 0 or 1

__device__ __forceinline__ u32 qh_addc(u32 a, u32 b, qh_cy ci, qh_cy &co) { return __builtin_addc(a, b, ci, &co); }
__device__ __forceinline__ u32 qh_subc(u32 a, u32 b, qh_cy bi, qh_cy &bo) { return __builtin_subc(a, b, bi, &bo); }

// a * b + c as 65 bits: the low 64, and the carry as a lane mask in an SGPR (v_mad_u64_u32's carry out).
__device__ __forceinline__ u64 qh_madc(u32 a, u32 b, u64 c, u32 &k) {
  u64 d;
  asm("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(d), "=s"(k) : "v"(a), "v"(b), "v"(c));
  return d;
}

// x + (the lane mask k ? 1 : 0)
__device__ __forceinline__ u32 qh_addk(u32 x, u32 k) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, %1, 0, %2" : "=v"(r) : "v"(x), "s"(k));
  return r;
}

// The same with an SGPR operand (a uniform constant, e.g. the diagonal), so it needs no copy to a VGPR.
__device__ __forceinline__ u64 qh_madc_s(u32 a, u32 b, u64 c, u32 &k) {
  u64 d;
  asm("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(d), "=s"(k) : "v"(a), "s"(b), "v"(c));
  return d;
}

// Carry steps in inline asm, each carry a lane mask in an SGPR (QH_ARITH 2).
__device__ __forceinline__ u32 qh_a_add(u32 a, u32 b, u32 &co) {
  u32 r;
  asm("v_add_co_u32 %0, %1, %2, %3" : "=v"(r), "=s"(co) : "v"(a), "v"(b));
  return r;
}
__device__ __forceinline__ u32 qh_a_sub(u32 a, u32 b, u32 &bo) {
  u32 r;
  asm("v_sub_co_u32 %0, %1, %2, %3" : "=v"(r), "=s"(bo) : "v"(a), "v"(b));
  return r;
}
// a - b - bi, borrow out
__device__ __forceinline__ u32 qh_a_subb(u32 a, u32 b, u32 bi, u32 &bo) {
  u32 r;
  asm("v_sub_co_ci_u32_e64 %0, %1, %2, %3, %4" : "=v"(r), "=s"(bo) : "v"(a), "v"(b), "s"(bi));
  return r;
}
// a - bi, no borrow out
__device__ __forceinline__ u32 qh_a_subb0(u32 a, u32 bi) {
  u32 r;
  asm("v_sub_co_ci_u32_e64 %0, null, %1, 0, %2" : "=v"(r) : "v"(a), "s"(bi));
  return r;
}
// ci ? 0 : 0xffffffff
__device__ __forceinline__ u32 qh_a_mask(u32 ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, -1, 0, %1" : "=v"(r) : "s"(ci));
  return r;
}
__device__ __forceinline__ u32 qh_addck(u32 a, u32 b, u32 ci, u32 &co) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, %1, %2, %3, %4" : "=v"(r), "=s"(co) : "v"(a), "v"(b), "s"(ci));
  return r;
}

#ifndef QH_RED
#define QH_RED 1
#endif
#ifndef QH_WADD
#define QH_WADD 1
#endif

// QH_ARITH 2: (w1:w0) + 2^64 * (w3 + k : w2) (mod p) with k, a lane mask, the multiply's last carry: it
// rides in the first subtraction as its borrow in, so it costs nothing.
//   QH_RED 0: the exact reduction below (qh_red4's), 8 VALU.
//   QH_RED 1: x = (w0 - w2 - w3) + (w1 + w2) * 2^32, with the carry of w1 + w2 as + 2^32 - 1. 7 VALU. It
//             drops the borrow or carry out of the last three steps, which happens only when a word is
//             within 2 of 0 or 2^32 (odds about 2^-31 a multiply, so about 1 nonce in 10^6 hashes wrong;
//             the host rechecks every hit).
__device__ __forceinline__ u64 qh_red4k(u32 w0, u32 w1, u32 w2, u32 w3, u32 k) {
#if QH_RED == 1
  u32 b1, b2, c1;
  u32 l1 = qh_a_sub(w0, w2, b1);
  u32 l2 = qh_a_subb(l1, w3, k, b2);
  u32 h1 = qh_a_add(w1, w2, c1);
  u32 h2 = qh_a_subb0(h1, b1);
  u32 h3 = qh_a_subb0(h2, b2);
  u32 h4 = qh_addk(h3, c1);
  u32 l3 = qh_a_subb0(l2, c1);
  return qh_pack(l3, h4);
#else
  u32 b, c, C, d;
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
// about 2^-64); the host checks every hit anyway. 8 VALU.
__device__ __forceinline__ u64 qh_red4(u32 w0, u32 w1, u32 w2, u32 w3) {
  qh_cy b, c, C, d, z;
  u32 ylo = qh_subc(~w2, w3, 0, b);
  u32 yhi = qh_subc(w2, 0, b, z);
  u32 slo = qh_addc(w0, ylo, 0, c);
  u32 shi = qh_addc(w1, yhi, c, C);
  u32 m = C ? 0u : 0xffffffffu;
  u32 rlo = qh_subc(slo, m, 0, d);
  u32 rhi = qh_subc(shi, 0, d, z);
  return qh_pack(rlo, rhi);
}

__device__ __forceinline__ u64 qh_reduce128(u64 lo64, u64 hi64) {
  return qh_red4(qh_lo(lo64), qh_hi(lo64), qh_lo(hi64), qh_hi(hi64));
}

// x = lo64 + 2^64 * h (mod p) for a small h: y = (h + 1) * EPS = (h : ~h) >= EPS, so it is exact. 6 VALU.
__device__ __forceinline__ u64 qh_reduce96(u64 lo64, u32 h) {
#if QH_ARITH == 2 && QH_WADD
  u32 c, C, d;
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

// a * b as four words: A = a0 b0, X = a0 b1 + A's high word, C = a1 b0 + X (65 bits), D = a1 b1 + C's high
// word, then D's high word gets C's carry. Four v_mad_u64_u32 and three VALU.
struct QhP4 {
  u32 w0, w1, w2, w3;
};

__device__ __forceinline__ QhP4 qh_mul4(u64 a, u64 b) {
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  u32 k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  QhP4 r;
  r.w0 = qh_lo(A);
  r.w1 = qh_lo(C);
  r.w2 = qh_lo(D);
  r.w3 = qh_addk(qh_hi(D), k);
  return r;
}

__device__ __forceinline__ void qh_mul128(u64 a, u64 b, u64 &lo, u64 &hi) {
  QhP4 p = qh_mul4(a, b);
  lo = qh_pack(p.w0, p.w1);
  hi = qh_pack(p.w2, p.w3);
}

__device__ __forceinline__ u64 qh_mul(u64 a, u64 b) {
#if QH_ARITH == 2
  u32 a0 = qh_lo(a), a1 = qh_hi(a), b0 = qh_lo(b), b1 = qh_hi(b);
  u64 A = (u64)a0 * b0;
  u64 X = (u64)a0 * b1 + (A >> 32);
  u32 k;
  u64 C = qh_madc(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  return qh_red4k(qh_lo(A), qh_lo(C), qh_lo(D), qh_hi(D), k);
#else
  QhP4 p = qh_mul4(a, b);
  return qh_red4(p.w0, p.w1, p.w2, p.w3);
#endif
}

#ifndef QH_SQR
#define QH_SQR 1
#endif

// QH_SQR 1 (QH_ARITH 2): a^2 from three multiplies, a0^2, a0 a1 and a1^2, the middle one doubled with
// adds: 3 v_mad_u64_u32 and 5 VALU before the reduction, against 4 and 2.
__device__ __forceinline__ u64 qh_sqr(u64 a) {
#if QH_ARITH == 2 && QH_SQR
  u32 a0 = qh_lo(a), a1 = qh_hi(a);
  u64 A = (u64)a0 * a0, M = (u64)a0 * a1, D = (u64)a1 * a1;
  u32 c1, c2, c3, c4;
  u32 t = qh_a_add(qh_lo(M), qh_lo(M), c1);
  u32 w1 = qh_a_add(qh_hi(A), t, c2);
  u32 u = qh_addck(qh_hi(M), qh_hi(M), c1, c3);
  u32 w2 = qh_addck(qh_lo(D), u, c2, c4);
  u32 w3 = qh_addk(qh_hi(D), c3);
  return qh_red4k(qh_lo(A), w1, w2, w3, c4);
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

// A 96-bit accumulator: lo64 + 2^64 * h. Each add is three VALU with the carries in flags.
struct QhWide {
  u64 lo;
  u32 h;
};

__device__ __forceinline__ QhWide qh_w(u64 x) { QhWide w; w.lo = x; w.h = 0; return w; }

// a + b + the lane mask ci, no carry out; and the mask ci as 0 or 1.
__device__ __forceinline__ u32 qh_addck0(u32 a, u32 b, u32 ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "s"(ci));
  return r;
}
__device__ __forceinline__ u32 qh_bit(u32 ci) {
  u32 r;
  asm("v_add_co_ci_u32_e64 %0, null, 0, 0, %1" : "=v"(r) : "s"(ci));
  return r;
}

// QH_WADD 1 (QH_ARITH 2): the three steps in inline asm with mask carries, and a known-zero high word costs
// no register: 3 VALU an add.
__device__ __forceinline__ QhWide qh_wadd(QhWide w, u64 x) {
  QhWide r;
#if QH_ARITH == 2 && QH_WADD
  u32 c1, c2;
  u32 l = qh_a_add(qh_lo(w.lo), qh_lo(x), c1);
  u32 h = qh_addck(qh_hi(w.lo), qh_hi(x), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = (__builtin_constant_p(w.h) && w.h == 0) ? qh_bit(c2) : qh_addk(w.h, c2);
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
#if QH_ARITH == 2 && QH_WADD
  u32 c1, c2, l, h;
  asm("v_add_co_u32 %0, %1, %2, %3" : "=v"(l), "=s"(c1) : "v"(qh_lo(w.lo)), "s"(qh_lo(c)));
  asm("v_add_co_ci_u32_e64 %0, %1, %2, %3, %4" : "=v"(h), "=s"(c2) : "v"(qh_hi(w.lo)), "s"(qh_hi(c)), "s"(c1));
  r.lo = qh_pack(l, h);
  r.h = (__builtin_constant_p(w.h) && w.h == 0) ? qh_bit(c2) : qh_addk(w.h, c2);
#else
  r = qh_wadd(w, c);
#endif
  return r;
}

__device__ __forceinline__ QhWide qh_wadd(QhWide w, QhWide x) {
  QhWide r;
#if QH_ARITH == 2 && QH_WADD
  u32 c1, c2;
  u32 l = qh_a_add(qh_lo(w.lo), qh_lo(x.lo), c1);
  u32 h = qh_addck(qh_hi(w.lo), qh_hi(x.lo), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = qh_addck0(w.h, x.h, c2);
#else
  qh_cy c1, c2, z;
  u32 l = qh_addc(qh_lo(w.lo), qh_lo(x.lo), 0, c1);
  u32 h = qh_addc(qh_hi(w.lo), qh_hi(x.lo), c1, c2);
  r.lo = qh_pack(l, h);
  r.h = qh_addc(w.h, x.h, c2, z);
#endif
  return r;
}

#ifndef QH_MULADD
#define QH_MULADD 1
#endif

// diag * s + a (a 96-bit sum), reduced.
//   QH_MULADD 1: a's low 64 bits ride in the first multiply-add, its high word in the third word: 4
//                v_mad_u64_u32 and 5 + 8 VALU.
//   QH_MULADD 0: the product, then a 128-bit add: 4 v_mad_u64_u32 and 3 + 4 + 8 VALU.
__device__ __forceinline__ u64 qh_muladd(u64 s, u64 diag, QhWide a) {
#if QH_ARITH == 2
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  u32 kA, k, c;
  u64 A = qh_madc_s(a0, b0, a.lo, kA);
  u64 X = (u64)a0 * b1 + (A >> 32);
  u64 C = qh_madc_s(a1, b0, X, k);
  u64 D = (u64)a1 * b1 + (C >> 32);
  u32 w2 = qh_addck(qh_lo(D), a.h, kA, c);
  u32 w3 = qh_addk(qh_hi(D), c);
  return qh_red4k(qh_lo(A), qh_lo(C), w2, w3, k);
#elif QH_MULADD
  u32 a0 = qh_lo(s), a1 = qh_hi(s), b0 = qh_lo(diag), b1 = qh_hi(diag);
  u32 kA, k, c;
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
#endif

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
    y[g + 3] = qh_wadd(qh_wadd(t01233, x0), x0);  // 3x0 + x1 + x2 + 2x3
    y[g + 1] = qh_wadd(qh_wadd(t01123, x2), x2);  // x0 + 2x1 + 3x2 + x3
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

// The two permutations from the folded state of N nonces at once (QH_NPT; independent, so the compiler can
// interleave them); returns element 0 of each result, in [0, 2^64).
#ifndef QH_NPT
#define QH_NPT 1
#endif
// The partial rounds' loop unrolled this many times (22 must divide by it), so the compiler can start the next
// round's S-box while this round's other eleven products are still going.
#ifndef QH_PUNROLL
#define QH_PUNROLL 1
#endif

template <int N>
__device__ __forceinline__ void qh_hash(u64 (&s)[N][12]) {
  int row = 0;
#pragma unroll 1
  for (int seg = 0; seg < 3; seg++) {
    int rows = seg == 1 ? 9 : 4;
#pragma unroll 1
    for (int r = 0; r < rows; r++, row++) {
      if (row != QH_NOSBOX_ROW) {
#pragma unroll
        for (int i = 0; i < 12; i++)
#pragma unroll
          for (int n = 0; n < N; n++) s[n][i] = qh_sbox(s[n][i]);
      }
#pragma unroll
      for (int n = 0; n < N; n++) qh_external(s[n], qh_tab.full[row]);
    }
    if (seg == 2) break;
#pragma unroll QH_PUNROLL
    for (int r = 0; r < 22; r++) {
#pragma unroll
      for (int n = 0; n < N; n++) {
        s[n][0] = qh_sbox(s[n][0]);
        qh_internal(s[n], qh_tab.diag, qh_tab.part0[r]);
      }
    }
#pragma unroll
    for (int n = 0; n < N; n++)
#pragma unroll
      for (int i = 1; i < 12; i++) {
        QhWide w = qh_waddc(qh_w(s[n][i]), qh_tab.term0[i]);
        s[n][i] = qh_reduce96(w.lo, w.h);
      }
  }
}

// Element j's coefficient on counter word 7 after the first external layer.
__device__ __forceinline__ u32 qh_coef7(int j) {
  const int m = j & 3;
  const u32 base = m == 2 ? 3u : m == 3 ? 2u : 1u;
  return (j >= 4 && j < 8) ? 2u * base : base;
}

// The block size, and (QH_WPE) the fewest waves a SIMD the compiler must leave room for: a lower number lets it
// use more registers, for more instruction-level parallelism.
#ifndef QH_BLOCK
#define QH_BLOCK 256
#endif
#ifdef QH_WPE
#define QH_ATTR __attribute__((amdgpu_waves_per_eu(QH_WPE)))
#else
#define QH_ATTR
#endif

extern "C" __global__ void __launch_bounds__(QH_BLOCK) QH_ATTR qh_search(QhArgs a, u64 *hits) {
  u32 i0 = (blockIdx.x * blockDim.x + threadIdx.x) * QH_NPT;
  if (i0 >= a.n) return;
  u64 s[QH_NPT][12];
#pragma unroll
  for (int n = 0; n < QH_NPT; n++) {
    u64 x = qh_bswap32(a.lo0 + i0 + n);
#pragma unroll
    for (int j = 0; j < 12; j++) {
      u64 add = x * qh_coef7(j);
      QhWide w = qh_waddc(qh_w(add), a.k[j]);
      s[n][j] = qh_reduce96(w.lo, w.h);
    }
  }
  qh_hash<QH_NPT>(s);
#pragma unroll
  for (int n = 0; n < QH_NPT; n++) {
    u64 r = s[n][0];
    if (r >= QH_P) r -= QH_P;
    u64 top = ((u64)qh_bswap32(qh_lo(r)) << 32) | qh_bswap32(qh_hi(r));
    // A thread's last nonces can lie past the launch's end when n is not a multiple of QH_NPT.
    if (top <= a.target && i0 + n < a.n) {
      unsigned long long slot = atomicAdd((unsigned long long *)hits, 1ull);
      if (slot < a.cap) hits[1 + slot] = qh_pack(a.lo0 + i0 + n, a.hi);
    }
  }
}
