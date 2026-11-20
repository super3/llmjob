// Goldilocks field arithmetic for the Quantus CUDA core (p = 2^64 - 2^32 + 1).
//
// A field element lives in a u64 and may be any value below 2^64 that is
// congruent to it (not always canonical); qc_canon() makes it canonical. The
// kernel (qc_kernel.cuh) is built on these.
//
// The tricks are the usual ones for this prime: 2^64 = 2^32 - 1 (EPS) and
// 2^96 = -1 (mod p), so a 128-bit product folds with a multiply by EPS and a
// subtraction. Sums of several elements ride in 96-bit accumulators (Q96) and
// are folded once. quantus-miner's CUDA kernel (Apache-2.0) does the same; this
// file is our own code.
//
// Exactness: with QC_EXACT 1 every routine is exact. With QC_EXACT 0 (the
// default) the 128-bit fold skips one correction: when the final subtraction
// borrows (about once in 2^33 multiplies), the result is off by EPS. A nonce
// whose hash took such a slip comes out wrong, so it is a missed or a false
// hit; the host checks every hit on the CPU, and verify-core.js's exact checks
// would see a slip about once in 1,500 runs. qc_add_small and q96_fold are
// always exact.
#pragma once
#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define QC_P 0xFFFFFFFF00000001ull
#define QC_EPS32 0xFFFFFFFFu

#ifndef QC_EXACT
#define QC_EXACT 0
#endif
// Build h: below sm_120 the external layers run on sums of 32-bit halves folded
// on the ALU (QC_EXTV 1, QC_HFOLD 1 in qc_kernel.cuh) and squares double their
// middle product with one 64-bit add (QC_SQR 2): +3-4% on an RTX 4090. For
// sm_120 ptxas turns the half sums into IADD.64s and that build ran 5% slower
// on an RTX 5080, so RTX 50 keeps build g.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define QC_BUILD_H 0
#else
#define QC_BUILD_H 1
#endif
// The RTX 50 (sm_120) path, QC_B120 in qc_kernel.cuh. ptxas for sm_120 puts every
// add on the ALU (no IMAD.X or IMAD.MOV) and makes 64-bit C adds IADD.64, which
// take two ALU slots and a MOV for each zero-extended word. So on sm_120 every
// sum is written as 32-bit adds with explicit carries, the partial rounds' sum
// rides in the products' addends, and folds take two ALU ops.
#ifndef QC_B120
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define QC_B120 1
#else
#define QC_B120 0
#endif
#endif
#ifndef QC_SQR
#define QC_SQR (QC_BUILD_H ? 2 : 1)
#endif
#ifndef QC_RED
#define QC_RED 2
#endif
#ifndef QC_FOLD
#define QC_FOLD 2
#endif

// QC_IMX (sm_120): adds that end a carry chain run as IMAD.X / IMAD with a
// multiplier of one that ptxas cannot see (QC_ONE, in constant memory), so
// they go to the multiplier pipe instead of the ALU. On an RTX 5070 a plain
// IMAD runs beside IADD3 (3.7 a clock a SM for the pair) while an IMAD.WIDE
// takes an ALU issue slot as well (WIDE + 2 IADD3: 2.0 a clock in all), so the
// ALU side binds and the IMAD side has room. Bits: 1 the reduction's carry
// adds, 2 the Q96 accumulators' top word, 4 the two-op fold's add, 8 the
// borrow that ends the fold and the reduction (as IMAD.X of 2^32 - 1: ptxas's
// carry after a subtract is "no borrow"), 16 a carry turned into a word (the
// products' middle carry, a Q96 sum's top word) as IMAD.X of QC_ZERO.
#ifndef QC_IMX
#define QC_IMX 0
#endif
__constant__ uint32_t QC_ONE = 1;
__constant__ uint32_t QC_ZERO = 0;

__device__ __forceinline__ u32 qc_lo(u64 x) { return (u32)x; }
__device__ __forceinline__ u32 qc_hi(u64 x) { return (u32)(x >> 32); }
__device__ __forceinline__ u64 qc_pack(u32 lo, u32 hi) {
  u64 r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "r"(lo), "r"(hi));
  return r;
}

// A 96-bit accumulator: l0 + l1 * 2^32 + h * 2^64, h small.
struct Q96 {
  u32 l0, l1, h;
};

__device__ __forceinline__ Q96 q96(u64 x) {
  Q96 w;
  w.l0 = qc_lo(x);
  w.l1 = qc_hi(x);
  w.h = 0;
  return w;
}

__device__ __forceinline__ void q96_add(Q96 &w, u64 x) {
#if QC_IMX & 2
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "madc.lo.u32 %2, %2, %5, 0;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(qc_lo(x)), "r"(qc_hi(x)), "r"(QC_ONE));
#else
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "addc.u32 %2, %2, 0;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(qc_lo(x)), "r"(qc_hi(x)));
#endif
}

__device__ __forceinline__ void q96_addq(Q96 &w, const Q96 &x) {
#if QC_IMX & 2
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "madc.lo.u32 %2, %2, %6, %5;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(x.l0), "r"(x.l1), "r"(x.h), "r"(QC_ONE));
#else
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "addc.u32 %2, %2, %5;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(x.l0), "r"(x.l1), "r"(x.h));
#endif
}

__device__ __forceinline__ Q96 q96_sum(u64 a, u64 b) {
  Q96 w;
#if QC_IMX & 16
  asm("add.cc.u32 %0, %3, %5;\n\t"
      "addc.cc.u32 %1, %4, %6;\n\t"
      "madc.lo.u32 %2, %7, %8, 0;"
      : "=r"(w.l0), "=r"(w.l1), "=r"(w.h)
      : "r"(qc_lo(a)), "r"(qc_hi(a)), "r"(qc_lo(b)), "r"(qc_hi(b)), "r"(QC_ONE), "r"(QC_ZERO));
#else
  asm("add.cc.u32 %0, %3, %5;\n\t"
      "addc.cc.u32 %1, %4, %6;\n\t"
      "addc.u32 %2, 0, 0;"
      : "=r"(w.l0), "=r"(w.l1), "=r"(w.h)
      : "r"(qc_lo(a)), "r"(qc_hi(a)), "r"(qc_lo(b)), "r"(qc_hi(b)));
#endif
  return w;
}

__device__ __forceinline__ Q96 q96_sumq(const Q96 &a, const Q96 &b) {
  Q96 w = a;
  q96_addq(w, b);
  return w;
}

// A Q96 with h below 2^24 to below 2^64: l + h * EPS, plus EPS if that carries.
// The carried value is below h * EPS < 2^56, so the second add cannot carry.
__device__ __forceinline__ u64 q96_fold(const Q96 &w) {
#if QC_FOLD == 3
  // l + h * EPS in one IMAD.WIDE.U32. It wraps past 2^64 (and then comes out
  // EPS low) only when l is within h * EPS of 2^64: about h in 2^32.
  return qc_pack(w.l0, w.l1) + (u64)w.h * 0xFFFFFFFFull;
#else
  u32 o0, o1;
#endif
#if QC_FOLD == 3
#elif QC_FOLD == 2
  // h * EPS = (h - (h != 0)) * 2^32 + (-h mod 2^32), with no multiply: on Ada
  // and Ampere the multiplier pipe is the busy one, the ALU has room.
  asm("{\n\t"
      ".reg .u32 e0, e1, c;\n\t"
      "sub.cc.u32 e0, 0, %2;\n\t"
      "subc.u32 e1, %2, 0;\n\t"
      "add.cc.u32 %0, %3, e0;\n\t"
      "addc.cc.u32 %1, %4, e1;\n\t"
      "addc.u32 c, 0, 0;\n\t"
      "neg.s32 c, c;\n\t"
      "add.cc.u32 %0, %0, c;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(w.h), "r"(w.l0), "r"(w.l1));
#else
  asm("{\n\t"
      ".reg .u32 c;\n\t"
      "mad.lo.cc.u32 %0, %2, %5, %3;\n\t"
      "madc.hi.cc.u32 %1, %2, %5, %4;\n\t"
      "addc.u32 c, 0, 0;\n\t"
      "mad.lo.cc.u32 %0, c, %5, %0;\n\t"
      "madc.hi.u32 %1, c, %5, %1;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(w.h), "r"(w.l0), "r"(w.l1), "r"(QC_EPS32));
#endif
#if QC_FOLD != 3
  return qc_pack(o0, o1);
#endif
}

// A Q96 to below 2^64 in two ALU ops: lo = l0 - h, hi = l1 + h - borrow, which
// is l + h * EPS. It wraps past 2^64 when l1 + h passes 2^32, about h in 2^32
// folds, and the result is then EPS low: a wrong hash about once in 10^6 nonces,
// as build h's fold below sm_120 (the host checks every hit).
__device__ __forceinline__ u64 q96_fold2(const Q96 &w) {
  u32 o0, o1;
#if (QC_IMX & 4) && (QC_IMX & 8)
  asm("{.reg .u32 t;\n\t"
      "mad.lo.u32 t, %4, %5, %3;\n\t"
      "sub.cc.u32 %0, %2, %4;\n\t"
      "madc.lo.u32 %1, t, %5, 0xFFFFFFFF;}"
      : "=r"(o0), "=r"(o1)
      : "r"(w.l0), "r"(w.l1), "r"(w.h), "r"(QC_ONE));
#elif QC_IMX & 4
  asm("{.reg .u32 t;\n\t"
      "mad.lo.u32 t, %4, %5, %3;\n\t"
      "sub.cc.u32 %0, %2, %4;\n\t"
      "subc.u32 %1, t, 0;}"
      : "=r"(o0), "=r"(o1)
      : "r"(w.l0), "r"(w.l1), "r"(w.h), "r"(QC_ONE));
#else
  asm("{.reg .u32 t;\n\t"
      "add.u32 t, %3, %4;\n\t"
      "sub.cc.u32 %0, %2, %4;\n\t"
      "subc.u32 %1, t, 0;}"
      : "=r"(o0), "=r"(o1)
      : "r"(w.l0), "r"(w.l1), "r"(w.h));
#endif
  return qc_pack(o0, o1);
}

// a + b mod p, below 2^64, for any a below 2^64 and b at most p (a nonce
// word's small multiple, or a canonical round constant). A carry out of 2^64
// adds EPS, and that cannot carry again: a + b - 2^64 + EPS <= 2^64 - 1 when
// b <= p. test/quantusCudaConstants.test.js checks the bound.
__device__ __forceinline__ u64 qc_add_small(u64 a, u64 b) {
  u32 o0, o1;
  asm("{\n\t"
      ".reg .u32 c;\n\t"
      "add.cc.u32 %0, %2, %4;\n\t"
      "addc.cc.u32 %1, %3, %5;\n\t"
      "addc.u32 c, 0, 0;\n\t"
      "mad.lo.cc.u32 %0, c, %6, %0;\n\t"
      "madc.hi.u32 %1, c, %6, %1;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(qc_lo(a)), "r"(qc_hi(a)), "r"(qc_lo(b)), "r"(qc_hi(b)), "r"(QC_EPS32));
  return qc_pack(o0, o1);
}

// 64 x 64 -> 128 bits, as four 32-bit words r0 (low) .. r3. mul.lo.u64 and
// mul.hi.u64 of the same operands in one block: ptxas then shares the partial
// products (four IMAD.WIDE in all); split across C expressions it often did not.
__device__ __forceinline__ void qc_mul128(u64 a, u64 b, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  asm("{.reg .u64 L, H;\n\t"
      "mul.lo.u64 L, %4, %5;\n\t"
      "mul.hi.u64 H, %4, %5;\n\t"
      "mov.b64 {%0, %1}, L;\n\t"
      "mov.b64 {%2, %3}, H;}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(a), "l"(b));
}

// a * b + c -> 128 bits, for c = clo + chi * 2^32 with clo and chi below 2^32,
// each passed zero-extended. clo rides in the low product's addend and chi in
// the middle one's (a 32 x 32 product plus a 32-bit value cannot carry), so
// adding c costs nothing; a caller adding the same c to many products builds
// the two zero-extended pairs once. The total is below 2^128.
__device__ __forceinline__ void qc_mul128_add(u64 a, u64 b, u64 clo, u64 chi, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  asm("{.reg .u32 a0, a1, b0, b1, m0, m1, l1, cm;\n\t"
      ".reg .u64 M, T, L, X, H;\n\t"
      "mov.b64 {a0, a1}, %4;\n\t"
      "mov.b64 {b0, b1}, %5;\n\t"
      "mad.wide.u32 M, a1, b0, %7;\n\t"
      "mul.wide.u32 T, a0, b1;\n\t"
      "add.cc.u64 M, M, T;\n\t"
#if QC_IMX & 16
      "madc.lo.u32 cm, %8, %9, 0;\n\t"
#else
      "addc.u32 cm, 0, 0;\n\t"
#endif
      "mad.wide.u32 L, a0, b0, %6;\n\t"
      "mov.b64 {%0, l1}, L;\n\t"
      "mov.b64 {m0, m1}, M;\n\t"
      "add.cc.u32 %1, l1, m0;\n\t"
      "mov.b64 X, {m1, cm};\n\t"
      "mul.wide.u32 H, a1, b1;\n\t"
      "addc.u64 H, H, X;\n\t"
      "mov.b64 {%2, %3}, H;}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(a), "l"(b), "l"(clo), "l"(chi), "r"(QC_ONE), "r"(QC_ZERO));
}

// a^2 with three partial products: a0^2 + 2 a0 a1 2^32 + a1^2 2^64. Mode M is
// QC_SQR's: 0 and 3 as a full multiply (four products), 1 and 2 three.
template <int M>
__device__ __forceinline__ void qc_sqr128m(u64 a, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  if (M == 0) {
    qc_mul128(a, a, r0, r1, r2, r3);
  } else if (M == 3) {
    qc_mul128_add(a, a, 0, 0, r0, r1, r2, r3);
  } else if (M == 2) {
  // 2 a0 a1 as one 64-bit add with a carry out, then the high product takes
  // it in its addend.
  asm("{.reg .u32 a0, a1, l1, m0, m1, cm;\n\t"
      ".reg .u64 LL, M, X, H;\n\t"
      "mov.b64 {a0, a1}, %4;\n\t"
      "mul.wide.u32 M, a0, a1;\n\t"
      "add.cc.u64 M, M, M;\n\t"
      "addc.u32 cm, 0, 0;\n\t"
      "mul.wide.u32 LL, a0, a0;\n\t"
      "mov.b64 {%0, l1}, LL;\n\t"
      "mov.b64 {m0, m1}, M;\n\t"
      "add.cc.u32 %1, l1, m0;\n\t"
      "mov.b64 X, {m1, cm};\n\t"
      "mul.wide.u32 H, a1, a1;\n\t"
      "addc.u64 H, H, X;\n\t"
      "mov.b64 {%2, %3}, H;}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(a));
  } else {
  asm("{.reg .u32 a0, a1, l1, m0, m1, h0, h1;\n\t"
      ".reg .u64 LL, LH, HH;\n\t"
      "mov.b64 {a0, a1}, %4;\n\t"
      "mul.wide.u32 LL, a0, a0;\n\t"
      "mul.wide.u32 LH, a0, a1;\n\t"
      "mul.wide.u32 HH, a1, a1;\n\t"
      "mov.b64 {%0, l1}, LL;\n\t"
      "mov.b64 {m0, m1}, LH;\n\t"
      "mov.b64 {h0, h1}, HH;\n\t"
      "add.cc.u32 %1, l1, m0;\n\t"
      "addc.cc.u32 %2, h0, m1;\n\t"
      "addc.u32 %3, h1, 0;\n\t"
      "add.cc.u32 %1, %1, m0;\n\t"
      "addc.cc.u32 %2, %2, m1;\n\t"
      "addc.u32 %3, %3, 0;}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(a));
  }
}
__device__ __forceinline__ void qc_sqr128(u64 a, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  qc_sqr128m<QC_SQR>(a, r0, r1, r2, r3);
}

// (r3 r2 r1 r0) mod p, below 2^64.
//   x = (r1:r0) + r2 * 2^64 + r3 * 2^96 = (r1:r0) + r2 * EPS - r3   (mod p)
// A = (r1:r0) + r2 * EPS carries c into 2^64 = EPS, which is added as
// c * 2^32 - c (no overflow: A - 2^64 < 2^64 - 2^33 when c is set). Then
// r3 + c comes off. If that borrows, the wrap added 2^64 = EPS too many: the
// exact form takes EPS off again (the wrapped value is at least 2^64 - 2^32,
// so no second borrow); the fast form leaves it (see the top of this file).
// As QC_RED 2 with no multiply: r2 * EPS = (r2 - (r2 != 0)) * 2^32 + (-r2 mod
// 2^32), QC_FOLD 2's identity, added to (r1:r0) on the ALU. Eight ALU ops for
// one IMAD.WIDE and three. On sm_120 the multiplier pipe binds (an RTX 5070's
// ALU runs 4 register-operand IADD3 a clock a SM, its IMAD.WIDE 1), so some
// reductions can move there (QC_RED 5 for all, QC_REDP 5 for the partial
// rounds' eleven products only).
__device__ __forceinline__ u64 qc_reduce_alu(u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 o0, o1;
  asm("{\n\t"
      ".reg .u32 e0, e1, c;\n\t"
      "sub.cc.u32 e0, 0, %4;\n\t"
      "subc.u32 e1, %4, 0;\n\t"
      "add.cc.u32 %0, %2, e0;\n\t"
      "addc.cc.u32 %1, %3, e1;\n\t"
      "addc.u32 c, %5, 0;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
      "subc.u32 %1, %1, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return qc_pack(o0, o1);
}

__device__ __forceinline__ u64 qc_reduce(u32 r0, u32 r1, u32 r2, u32 r3) {
#if QC_RED == 5
  return qc_reduce_alu(r0, r1, r2, r3);
#endif
  u32 o0, o1;
#if QC_EXACT
  asm("{\n\t"
      ".reg .u32 c, b;\n\t"
      "mad.lo.cc.u32 %0, %4, %6, %2;\n\t"
      "madc.hi.cc.u32 %1, %4, %6, %3;\n\t"
      "addc.u32 c, %5, 0;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
      "subc.cc.u32 %1, %1, 0;\n\t"
      "subc.u32 b, 0, 0;\n\t"
      "sub.cc.u32 %0, %0, b;\n\t"
      "subc.u32 %1, %1, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(QC_EPS32));
#elif QC_RED == 1
  asm("{\n\t"
      ".reg .u32 c;\n\t"
      "mad.lo.cc.u32 %0, %4, %6, %2;\n\t"
      "madc.hi.cc.u32 %1, %4, %6, %3;\n\t"
      "addc.u32 c, %5, 0;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
      "subc.u32 %1, %1, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(QC_EPS32));
#elif QC_RED == 3
  // As QC_RED 2, with the wide add's carry c taken twice: addc (no .cc) puts it
  // on the high word, then subc takes r3 and the carry off the low word.
  asm("{\n\t"
      ".reg .u32 hc;\n\t"
      ".reg .u64 L, E;\n\t"
      "mov.b64 L, {%2, %3};\n\t"
      "mul.wide.u32 E, %4, %6;\n\t"
      "add.cc.u64 E, E, L;\n\t"
      "mov.b64 {%0, %1}, E;\n\t"
      "addc.u32 hc, %1, 0;\n\t"
      "subc.cc.u32 %0, %0, %5;\n\t"
      "subc.u32 %1, hc, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(QC_EPS32));
#elif QC_IMX & 1
  // As below, with the two adds that take the wide add's carry as IMAD.X.
  asm("{\n\t"
      ".reg .u32 c;\n\t"
      ".reg .u64 L, E;\n\t"
      "mov.b64 L, {%2, %3};\n\t"
      "mul.wide.u32 E, %4, %6;\n\t"
      "add.cc.u64 E, E, L;\n\t"
      "mov.b64 {%0, %1}, E;\n\t"
      "madc.lo.u32 c, %5, %7, 0;\n\t"
      "madc.lo.u32 %1, %1, %7, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
#if QC_IMX & 8
      "madc.lo.u32 %1, %1, %7, 0xFFFFFFFF;\n\t"
#else
      "subc.u32 %1, %1, 0;\n\t"
#endif
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(QC_EPS32), "r"(QC_ONE));
#else
  // r2 * EPS + (r1:r0) as one mul.wide.u32 into a 64-bit add: ptxas makes it a
  // single IMAD.WIDE.U32 with a carry out.
  asm("{\n\t"
      ".reg .u32 c;\n\t"
      ".reg .u64 L, E;\n\t"
      "mov.b64 L, {%2, %3};\n\t"
      "mul.wide.u32 E, %4, %6;\n\t"
      "add.cc.u64 E, E, L;\n\t"
      "mov.b64 {%0, %1}, E;\n\t"
      "addc.u32 c, %5, 0;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
      "subc.u32 %1, %1, 0;\n\t"
      "}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(QC_EPS32));
#endif
  return qc_pack(o0, o1);
}

#ifndef QC_MUL
#define QC_MUL 0
#endif

__device__ __forceinline__ u64 qc_mul(u64 a, u64 b) {
  u32 r0, r1, r2, r3;
#if QC_MUL == 1
  qc_mul128_add(a, b, 0, 0, r0, r1, r2, r3);
#else
  qc_mul128(a, b, r0, r1, r2, r3);
#endif
  return qc_reduce(r0, r1, r2, r3);
}

// a * b + c mod p for c below 2^64.
__device__ __forceinline__ u64 qc_mul_add64(u64 a, u64 b, u64 c) {
  u32 r0, r1, r2, r3;
  qc_mul128_add(a, b, (u64)qc_lo(c), (u64)qc_hi(c), r0, r1, r2, r3);
  return qc_reduce(r0, r1, r2, r3);
}

// The same with the addend's halves already as 64-bit pairs. QC_REDP 5 reduces
// on the ALU (the partial rounds' products use this).
#ifndef QC_REDP
#define QC_REDP 0
#endif
__device__ __forceinline__ u64 qc_mul_add_pair(u64 a, u64 b, u64 clo, u64 chi) {
  u32 r0, r1, r2, r3;
  qc_mul128_add(a, b, clo, chi, r0, r1, r2, r3);
#if QC_REDP == 5
  return qc_reduce_alu(r0, r1, r2, r3);
#else
  return qc_reduce(r0, r1, r2, r3);
#endif
}

template <int M = QC_SQR>
__device__ __forceinline__ u64 qc_sqr(u64 a) {
  u32 r0, r1, r2, r3;
  qc_sqr128m<M>(a, r0, r1, r2, r3);
  return qc_reduce(r0, r1, r2, r3);
}

// (r3 r2 r1 r0) += w, for a 128-bit value that cannot pass 2^128.
__device__ __forceinline__ void qc_add128q(u32 &r0, u32 &r1, u32 &r2, u32 &r3, const Q96 &w) {
  asm("add.cc.u32 %0, %0, %4;\n\t"
      "addc.cc.u32 %1, %1, %5;\n\t"
      "addc.cc.u32 %2, %2, %6;\n\t"
      "addc.u32 %3, %3, 0;"
      : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)
      : "r"(w.l0), "r"(w.l1), "r"(w.h));
}

// a * d + w mod p, for d below 2^64 - 2^9 and w a Q96 below 2^72, so the
// 128-bit sum cannot overflow. The diagonal constants are all below 0xf4 << 56.
__device__ __forceinline__ u64 qc_mul_add(u64 a, u64 d, const Q96 &w) {
  u32 r0, r1, r2, r3;
  qc_mul128(a, d, r0, r1, r2, r3);
  qc_add128q(r0, r1, r2, r3, w);
  return qc_reduce(r0, r1, r2, r3);
}

// x^7 = x^3 * x^4, with x^3 and x^4 both from x^2. M picks the squares' form.
template <int M = QC_SQR>
__device__ __forceinline__ u64 qc_sbox(u64 x) {
  u64 x2 = qc_sqr<M>(x);
  u64 x3 = qc_mul(x2, x);
  u64 x4 = qc_sqr<M>(x2);
  return qc_mul(x3, x4);
}

__device__ __forceinline__ u64 qc_canon(u64 x) { return x >= QC_P ? x - QC_P : x; }
