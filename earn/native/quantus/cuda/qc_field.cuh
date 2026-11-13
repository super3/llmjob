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
#ifndef QC_SQR
#define QC_SQR 0
#endif
#ifndef QC_RED
#define QC_RED 2
#endif

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
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "addc.u32 %2, %2, 0;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(qc_lo(x)), "r"(qc_hi(x)));
}

__device__ __forceinline__ void q96_addq(Q96 &w, const Q96 &x) {
  asm("add.cc.u32 %0, %0, %3;\n\t"
      "addc.cc.u32 %1, %1, %4;\n\t"
      "addc.u32 %2, %2, %5;"
      : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
      : "r"(x.l0), "r"(x.l1), "r"(x.h));
}

__device__ __forceinline__ Q96 q96_sum(u64 a, u64 b) {
  Q96 w;
  asm("add.cc.u32 %0, %3, %5;\n\t"
      "addc.cc.u32 %1, %4, %6;\n\t"
      "addc.u32 %2, 0, 0;"
      : "=r"(w.l0), "=r"(w.l1), "=r"(w.h)
      : "r"(qc_lo(a)), "r"(qc_hi(a)), "r"(qc_lo(b)), "r"(qc_hi(b)));
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
  u32 o0, o1;
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
  return qc_pack(o0, o1);
}

// a + b for b below 2^40 (a constant or a small multiple of a nonce word).
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

// a^2 with three partial products: a0^2 + 2 a0 a1 2^32 + a1^2 2^64.
__device__ __forceinline__ void qc_sqr128(u64 a, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
#if QC_SQR == 0
  qc_mul128(a, a, r0, r1, r2, r3);
#else
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
#endif
}

// (r3 r2 r1 r0) mod p, below 2^64.
//   x = (r1:r0) + r2 * 2^64 + r3 * 2^96 = (r1:r0) + r2 * EPS - r3   (mod p)
// A = (r1:r0) + r2 * EPS carries c into 2^64 = EPS, which is added as
// c * 2^32 - c (no overflow: A - 2^64 < 2^64 - 2^33 when c is set). Then
// r3 + c comes off. If that borrows, the wrap added 2^64 = EPS too many: the
// exact form takes EPS off again (the wrapped value is at least 2^64 - 2^32,
// so no second borrow); the fast form leaves it (see the top of this file).
__device__ __forceinline__ u64 qc_reduce(u32 r0, u32 r1, u32 r2, u32 r3) {
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

__device__ __forceinline__ u64 qc_mul(u64 a, u64 b) {
  u32 r0, r1, r2, r3;
  qc_mul128(a, b, r0, r1, r2, r3);
  return qc_reduce(r0, r1, r2, r3);
}

__device__ __forceinline__ u64 qc_sqr(u64 a) {
  u32 r0, r1, r2, r3;
  qc_sqr128(a, r0, r1, r2, r3);
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

// x^7 = x^3 * x^4, with x^3 and x^4 both from x^2.
__device__ __forceinline__ u64 qc_sbox(u64 x) {
  u64 x2 = qc_sqr(x);
  u64 x3 = qc_mul(x2, x);
  u64 x4 = qc_sqr(x2);
  return qc_mul(x3, x4);
}

__device__ __forceinline__ u64 qc_canon(u64 x) { return x >= QC_P ? x - QC_P : x; }
