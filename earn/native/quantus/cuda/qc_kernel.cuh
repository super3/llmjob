// The Quantus scan kernel: for each counter, the two Poseidon2 permutations of
// blocks 3 and 4 from the midstate, and a hit when the hash's top 64 bits are at
// or below the target's. earn/native/quantus/README.md has the loop this is;
// src/shared/quantus/qpow.js (nonceTop64) is the reference it is checked against.
//
// The host folds everything that is fixed for a launch into twelve constants
// (qc_host.cu, qc_prestate): the midstate, nonce bytes 32..59 (the tail and the
// counter's high word, which a launch never changes), the permutation's first
// linear layer and its first round constants. What is left per nonce is the
// counter's low word w7 = bswap32(counter): the first linear layer is linear,
// so w7 enters as k[j] * w7 on element j. quantus-miner's CUDA kernel
// (Apache-2.0) does the same with two words ("mining_prestate_low64").
//
// Rounds, with each layer's round constants added in the layer before it:
//   4 full rounds    x^7 on all 12, then the external layer E
//   22 partial       x^7 on element 0, then s[i] = sum(s) + D[i] * s[i]
//   4 full rounds
// Block 4 adds 1 to elements 0 and 1, folded into the first permutation's last
// layer. The second permutation needs only element 0 after its last layer.
#pragma once
#include "qc_field.cuh"

#ifndef QC_THREADS
#define QC_THREADS 256
#endif
#ifndef QC_MINB
#define QC_MINB 4
#endif
#ifndef QC_UNROLL_F
#define QC_UNROLL_F 1
#endif
// "#pragma unroll MACRO" is not expanded by GCC's preprocessor (Linux builds), so the count goes through _Pragma.
#define QC_STR(x) #x
#define QC_UNROLL(n) _Pragma(QC_STR(unroll n))
#ifndef QC_UNROLL_P
#define QC_UNROLL_P 1
#endif

// Round constants, from qp-poseidon (the same as src/shared/quantus/poseidon2.js).
// QC_RC_F1[r] is what the layer after the first permutation half's full round r
// adds: rounds 1..3 of the initial constants, then the first partial round's
// constant on element 0.
__constant__ u64 QC_RC_F1[4][12] = {
    {0xada1f8dc7676ed25ull, 0x2711aa8b5509d516ull, 0x4ae6acd0c9c92897ull, 0x56eb3d6b5256d67aull,
     0x1f7a9d55923bf51eull, 0x3600427d397a7f68ull, 0xe5076df75b72c3d0ull, 0xfcd59aa12c6090adull,
     0xcd895e8c68b57a9eull, 0x41df7ef9d730ae3eull, 0xee3e2b889abe977dull, 0xd29bb7edbeb9c405ull},
    {0x7d5c08eef608e382ull, 0x89ae889caaf0802cull, 0xb35a8e976d2af617ull, 0xdb14234eafaf5173ull,
     0x78f04462d48b1c98ull, 0x265293b0e47ce88aull, 0x999a649b69b9d32full, 0x64b0a186698e01d3ull,
     0xee0b22d0dfae8bb8ull, 0x4fd53e50ca04a7eeull, 0x5762bfe181f25047ull, 0xf51593e2beb5e3bdull},
    {0x1e5e2b5760e32477ull, 0x622462a1f9aaaeedull, 0xaa284b3ecdb222aeull, 0x63c8e72f542bf3fcull,
     0x3ba588cacb43b5e0ull, 0x23eda6f3c99150ddull, 0xaad3bea4baac9a5aull, 0xe9da8d699b94184aull,
     0xcdb13f4cd93e024cull, 0x902cbd0956f655e3ull, 0x5b4e40ffc759532full, 0xde795c20a2357af7ull},
    {0x97f7798a784ad863ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

// The first permutation's initial layer constants (perm 2 uses them; perm 1's
// are in the host's prestate).
__constant__ u64 QC_RC_I0[12] = {
    0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull, 0xe4f2fe29e03234acull,
    0xd5e0ba2f541b6449ull, 0xec33b868f3cc46c1ull, 0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull,
    0xe3f30d509a1436bbull, 0xd9a64f068dca7c29ull, 0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull};

// Partial rounds 1..21's constants (round 0's is in QC_RC_F1[3]).
__constant__ u64 QC_RC_P[21] = {
    0xd1d2bf082f60d4f0ull, 0x69a377a79f9ad206ull, 0xa9d06906a3858e24ull, 0x295275001eede5b5ull,
    0x5874e441117bd746ull, 0x8a084bbba8ed86ccull, 0x3defd7645cde6425ull, 0x3998cfe6871cc137ull,
    0x3e52ef8bca48314aull, 0x964a209f85dc9eccull, 0x3fcc9ee82cc4577eull, 0x8e79b4a5d0096d6dull,
    0x8492362ad2392556ull, 0xee72f470262574d6ull, 0x1e0e18496da2444aull, 0x0f3a74bf215eaac6ull,
    0x1b061b76a1c0ded3ull, 0x192c42d86803d7a6ull, 0xf6d49ff997ae0260ull, 0x3ec372e7a0fa3786ull,
    0x5538cdf4f23445d3ull};

// The terminal constants: [0] is added by the last partial round, [1..3] by
// the layers after terminal full rounds 0..2, and [4] by the layer after the
// first permutation's last round: block 4's terminator and pad (1, 1).
__constant__ u64 QC_RC_T[5][12] = {
    {0x7b72c539e0ea4c6eull, 0x144573dae2ce9976ull, 0x802028b68f35fc88ull, 0x6d36c5022c4fe7c2ull,
     0xa205d0ffa9b9def3ull, 0xf6e7e38b1ea6ba2full, 0x34f7909ae5258d64ull, 0xb0464d9d77b97fcaull,
     0x64ddb9d5de7e00a6ull, 0x0ed0d75c27975d97ull, 0x1cbb36f11127338bull, 0x6673e505cfd0b6baull},
    {0x605f902830872e01ull, 0x3fd5eb927e95fe4full, 0xe81025b5a24c69cdull, 0xf7d0ce75de23f74eull,
     0xf39942b6a8585089ull, 0x6d808a08f7b71df6ull, 0xf8806b6588f49a8bull, 0x57df2d8c2a32107aull,
     0x16e7c2074d654a2dull, 0x213de241fcf33835ull, 0xb0f2b8905a0976f6ull, 0xd8e3cf2bbd355417ull},
    {0xe498691679d9330full, 0x763b45d2a3821b28ull, 0x0908bf65eb0a1f0dull, 0x7691eb2d194b24f4ull,
     0x0e43551233ae13b2ull, 0x93c393dbfc2fe76full, 0x98f607485d48cdeaull, 0xe3d95f30309819c0ull,
     0x1ef581a93eaf6acfull, 0x0b24c1b7a030fca4ull, 0x624370be5670b327ull, 0x5f1e28615a11e486ull},
    {0xfe04051f909e042bull, 0x7257e5b147fd3803ull, 0xe6ae134bb82f2e78ull, 0x5711fd5cf4784511ull,
     0xf83a42660c08c0bcull, 0x2cd8c96d9a3ce855ull, 0x7d2ffb1bb0e17271ull, 0x85ae1528caea3811ull,
     0x52a345d5c7adb0b8ull, 0x504c4c51f3faee94ull, 0xbce34a649cfccaf9ull, 0xe0a3389266fb6dc9ull},
    {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

// The internal layer's diagonal: s[i] = sum + D[i] * s[i]. Immediates.
#define QC_D0 0xc3b6c08e23ba9300ull
#define QC_D1 0xd84b5de94a324fb6ull
#define QC_D2 0x0d0c371c5b35b84full
#define QC_D3 0x7964f570e7188037ull
#define QC_D4 0x5daf18bbd996604bull
#define QC_D5 0x6743bc47b9595257ull
#define QC_D6 0x5528b9362c59bb70ull
#define QC_D7 0xac45e25b7127b68bull
#define QC_D8 0xa2077d7dfbb606b5ull
#define QC_D9 0xf3faac6faee378aeull
#define QC_D10 0x0c6388b51545e883ull
#define QC_D11 0xd27dbb6944917b60ull

// The 4x4 matrix [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]] on one group, in Q96
// (Plonky3's chain: 10 adds).
__device__ __forceinline__ void qc_mat4(u64 x0, u64 x1, u64 x2, u64 x3, Q96 &y0, Q96 &y1, Q96 &y2, Q96 &y3) {
  Q96 t01 = q96_sum(x0, x1);
  Q96 t23 = q96_sum(x2, x3);
  Q96 t0123 = q96_sumq(t01, t23);
  Q96 t01123 = t0123;
  q96_add(t01123, x1);
  Q96 t01233 = t0123;
  q96_add(t01233, x3);
  y3 = t01233;
  q96_add(y3, x0);
  q96_add(y3, x0);
  y1 = t01123;
  q96_add(y1, x2);
  q96_add(y1, x2);
  y0 = q96_sumq(t01123, t01);
  y2 = q96_sumq(t01233, t23);
}

#ifndef QC_EXTV
#define QC_EXTV QC_BUILD_H
#endif
#ifndef QC_PSIG
#define QC_PSIG 0
#endif
#ifndef QC_HFIX
#define QC_HFIX 0
#endif

// A sum of field elements kept as two sums of 32-bit halves: the value is
// L + H * 2^32, with L and H plain u64s. Adding is a 64-bit add on each, which
// ptxas does with three-input IADD3s and no carry chain across them.
struct QH {
  u64 L, H;
};
__device__ __forceinline__ QH qh(u64 x) {
  QH r;
  r.L = qc_lo(x);
  r.H = qc_hi(x);
  return r;
}
__device__ __forceinline__ QH qh_add(const QH &a, const QH &b) {
  QH r;
  r.L = a.L + b.L;
  r.H = a.H + b.H;
  return r;
}
__device__ __forceinline__ QH qh_add3(const QH &a, const QH &b, const QH &c) {
  QH r;
  r.L = a.L + b.L + c.L;
  r.H = a.H + b.H + c.H;
  return r;
}
// L + H * 2^32 mod p, below 2^64, for L and H below 2^40: (H >> 32) * EPS + L
// is one IMAD.WIDE.U32 (below 2^41), then H's low word goes onto the high word.
// That add wraps past 2^64 (the result is then EPS low) about once in 2^32 /
// (t >> 32) folds, about 2^-27 here, unless QC_HFIX adds EPS back.
#ifndef QC_HFOLD
#define QC_HFOLD QC_BUILD_H
#endif
__device__ __forceinline__ u64 qh_fold(const QH &a) {
#if QC_HFOLD == 1
  // The same on the ALU: lo = L.lo - H.hi, hi = L.hi + H.lo + H.hi - borrow.
  u32 o0, o1;
  asm("{.reg .u32 t;\n\t"
      "add.u32 t, %3, %4;\n\t"
      "add.u32 t, t, %5;\n\t"
      "sub.cc.u32 %0, %2, %5;\n\t"
      "subc.u32 %1, t, 0;}"
      : "=r"(o0), "=r"(o1)
      : "r"(qc_lo(a.L)), "r"(qc_hi(a.L)), "r"(qc_lo(a.H)), "r"(qc_hi(a.H)));
  return qc_pack(o0, o1);
#endif
  u64 t = a.L + (a.H >> 32) * 0xFFFFFFFFull;
  u64 v = t + (a.H << 32);
#if QC_HFIX
  if (v < t) v += 0xFFFFFFFFull;
#endif
  return v;
}

// M4 on one group, as half sums (Plonky3's chain).
__device__ __forceinline__ void qh_mat4(const QH &x0, const QH &x1, const QH &x2, const QH &x3, QH &y0, QH &y1,
                                        QH &y2, QH &y3) {
  QH t01 = qh_add(x0, x1);
  QH t23 = qh_add(x2, x3);
  QH t0123 = qh_add(t01, t23);
  QH t01123 = qh_add(t0123, x1);
  QH t01233 = qh_add(t0123, x3);
  y3 = qh_add3(t01233, x0, x0);
  y1 = qh_add3(t01123, x2, x2);
  y0 = qh_add(t01123, t01);
  y2 = qh_add(t01233, t23);
}

// s = E(s) + rc. E: M4 on each group of four, then each element gets the sum of
// its column across the three groups.
__device__ __forceinline__ void qc_ext(u64 s[12], const u64 *rc) {
#if QC_EXTV == 1
  QH y[12];
#pragma unroll
  for (int g = 0; g < 12; g += 4) qh_mat4(qh(s[g]), qh(s[g + 1]), qh(s[g + 2]), qh(s[g + 3]), y[g], y[g + 1], y[g + 2], y[g + 3]);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    QH col = qh_add3(y[k], y[k + 4], y[k + 8]);
#pragma unroll
    for (int g = 0; g < 12; g += 4) s[g + k] = qh_fold(qh_add3(y[g + k], col, qh(rc[g + k])));
  }
#else
  Q96 y[12];
#pragma unroll
  for (int g = 0; g < 12; g += 4) qc_mat4(s[g], s[g + 1], s[g + 2], s[g + 3], y[g], y[g + 1], y[g + 2], y[g + 3]);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    Q96 col = q96_sumq(y[k], y[k + 4]);
    q96_addq(col, y[k + 8]);
#pragma unroll
    for (int g = 0; g < 12; g += 4) {
      Q96 w = q96_sumq(y[g + k], col);
      q96_add(w, rc[g + k]);
      s[g + k] = q96_fold(w);
    }
  }
#endif
}

// Element 0 of E(s): 2 * m0 + m4 + m8, m the first row of M4 on each group.
__device__ __forceinline__ u64 qc_ext0(const u64 s[12]) {
#if QC_EXTV == 1
  QH m[3];
#pragma unroll
  for (int g = 0; g < 3; g++) {
    QH a = qh_add(qh(s[4 * g]), qh(s[4 * g + 1]));  // m = 2 x0 + 3 x1 + x2 + x3
    m[g] = qh_add3(a, a, qh_add3(qh(s[4 * g + 1]), qh(s[4 * g + 2]), qh(s[4 * g + 3])));
  }
  return qh_fold(qh_add3(qh_add(m[0], m[0]), m[1], m[2]));
#else
  Q96 w = q96_sum(s[0], s[0]);
  q96_add(w, s[1]);
  q96_add(w, s[1]);
  q96_add(w, s[1]);
  q96_add(w, s[2]);
  q96_add(w, s[3]);
  Q96 m = w;
  q96_addq(w, m);  // 2 * m0
#pragma unroll
  for (int g = 4; g < 12; g += 4) {
    q96_add(w, s[g]);
    q96_add(w, s[g]);
    q96_add(w, s[g + 1]);
    q96_add(w, s[g + 1]);
    q96_add(w, s[g + 1]);
    q96_add(w, s[g + 2]);
    q96_add(w, s[g + 3]);
  }
  return q96_fold(w);
#endif
}

__device__ __forceinline__ void qc_full(u64 s[12], const u64 *rc) {
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = qc_sbox(s[i]);
  qc_ext(s, rc);
}

#ifndef QC_SUM
#define QC_SUM 2
#endif

// x + s[1] + ... + s[11] as a Q96. QC_SUM 2 sums the low and the high halves
// separately as u64s, which ptxas turns into three-input IADD3s.
__device__ __forceinline__ Q96 qc_sum12(const u64 s[12], u64 x) {
#if QC_SUM == 2
  u64 L = (u64)qc_lo(x), H = (u64)qc_hi(x);
#pragma unroll
  for (int i = 1; i < 12; i++) {
    L += qc_lo(s[i]);
    H += qc_hi(s[i]);
  }
  Q96 w;
  w.l0 = qc_lo(L);
  asm("add.cc.u32 %0, %2, %3;\n\t"
      "addc.u32 %1, %4, 0;"
      : "=r"(w.l1), "=r"(w.h)
      : "r"(qc_hi(L)), "r"(qc_lo(H)), "r"(qc_hi(H)));
  return w;
#else
  Q96 sum = q96_sum(x, s[1]);
#pragma unroll
  for (int i = 2; i < 12; i++) q96_add(sum, s[i]);
  return sum;
#endif
}

// One partial round's linear layer. x is this round's S-box output for element
// 0; s[1..11] are updated in place and the new element 0 (with the next
// round's constant rc0 added) is returned first, so the next S-box can start
// while the other eleven products are still being made.
__device__ __forceinline__ u64 qc_partial(u64 s[12], u64 x, u64 rc0, u32 z) {
#if QC_PSIG
  // The sum folded once to 64 bits, then carried in each product's addend.
  QH h = qh(x);
#pragma unroll
  for (int i = 1; i < 12; i++) {
    h.L += qc_lo(s[i]);
    h.H += qc_hi(s[i]);
  }
  const u64 sig = qh_fold(h);
  const u64 sig0 = qh_fold(qh_add(h, qh(rc0)));
#if QC_PSIG == 2
  // The addend halves as register pairs (half, z): with z a launch parameter
  // ptxas cannot drop the zero word, so each pair stays one IMAD.WIDE addend.
  const u64 l0 = qc_pack(qc_lo(sig0), z), h0 = qc_pack(qc_hi(sig0), z);
  const u64 l = qc_pack(qc_lo(sig), z), hh = qc_pack(qc_hi(sig), z);
  const u64 out0 = qc_mul_add_pair(x, QC_D0, l0, h0);
  s[1] = qc_mul_add_pair(s[1], QC_D1, l, hh);
  s[2] = qc_mul_add_pair(s[2], QC_D2, l, hh);
  s[3] = qc_mul_add_pair(s[3], QC_D3, l, hh);
  s[4] = qc_mul_add_pair(s[4], QC_D4, l, hh);
  s[5] = qc_mul_add_pair(s[5], QC_D5, l, hh);
  s[6] = qc_mul_add_pair(s[6], QC_D6, l, hh);
  s[7] = qc_mul_add_pair(s[7], QC_D7, l, hh);
  s[8] = qc_mul_add_pair(s[8], QC_D8, l, hh);
  s[9] = qc_mul_add_pair(s[9], QC_D9, l, hh);
  s[10] = qc_mul_add_pair(s[10], QC_D10, l, hh);
  s[11] = qc_mul_add_pair(s[11], QC_D11, l, hh);
#else
  (void)z;
  const u64 out0 = qc_mul_add64(x, QC_D0, sig0);
  s[1] = qc_mul_add64(s[1], QC_D1, sig);
  s[2] = qc_mul_add64(s[2], QC_D2, sig);
  s[3] = qc_mul_add64(s[3], QC_D3, sig);
  s[4] = qc_mul_add64(s[4], QC_D4, sig);
  s[5] = qc_mul_add64(s[5], QC_D5, sig);
  s[6] = qc_mul_add64(s[6], QC_D6, sig);
  s[7] = qc_mul_add64(s[7], QC_D7, sig);
  s[8] = qc_mul_add64(s[8], QC_D8, sig);
  s[9] = qc_mul_add64(s[9], QC_D9, sig);
  s[10] = qc_mul_add64(s[10], QC_D10, sig);
  s[11] = qc_mul_add64(s[11], QC_D11, sig);
#endif
  return out0;
#else
  Q96 sum = qc_sum12(s, x);
  Q96 sum0 = sum;
  q96_add(sum0, rc0);
  u64 out0 = qc_mul_add(x, QC_D0, sum0);
  s[1] = qc_mul_add(s[1], QC_D1, sum);
  s[2] = qc_mul_add(s[2], QC_D2, sum);
  s[3] = qc_mul_add(s[3], QC_D3, sum);
  s[4] = qc_mul_add(s[4], QC_D4, sum);
  s[5] = qc_mul_add(s[5], QC_D5, sum);
  s[6] = qc_mul_add(s[6], QC_D6, sum);
  s[7] = qc_mul_add(s[7], QC_D7, sum);
  s[8] = qc_mul_add(s[8], QC_D8, sum);
  s[9] = qc_mul_add(s[9], QC_D9, sum);
  s[10] = qc_mul_add(s[10], QC_D10, sum);
  s[11] = qc_mul_add(s[11], QC_D11, sum);
  return out0;
#endif
}

// The last partial round: every element also gets the first terminal constant.
__device__ __forceinline__ void qc_partial_last(u64 s[12], u64 x) {
  Q96 sum = qc_sum12(s, x);
  const u64 *rc = QC_RC_T[0];
  Q96 w;
  w = sum; q96_add(w, rc[0]); s[0] = qc_mul_add(x, QC_D0, w);
  w = sum; q96_add(w, rc[1]); s[1] = qc_mul_add(s[1], QC_D1, w);
  w = sum; q96_add(w, rc[2]); s[2] = qc_mul_add(s[2], QC_D2, w);
  w = sum; q96_add(w, rc[3]); s[3] = qc_mul_add(s[3], QC_D3, w);
  w = sum; q96_add(w, rc[4]); s[4] = qc_mul_add(s[4], QC_D4, w);
  w = sum; q96_add(w, rc[5]); s[5] = qc_mul_add(s[5], QC_D5, w);
  w = sum; q96_add(w, rc[6]); s[6] = qc_mul_add(s[6], QC_D6, w);
  w = sum; q96_add(w, rc[7]); s[7] = qc_mul_add(s[7], QC_D7, w);
  w = sum; q96_add(w, rc[8]); s[8] = qc_mul_add(s[8], QC_D8, w);
  w = sum; q96_add(w, rc[9]); s[9] = qc_mul_add(s[9], QC_D9, w);
  w = sum; q96_add(w, rc[10]); s[10] = qc_mul_add(s[10], QC_D10, w);
  w = sum; q96_add(w, rc[11]); s[11] = qc_mul_add(s[11], QC_D11, w);
}

// Both permutations from the state after the first one's initial layer.
// Returns element 0 after the second one (not canonical).
__device__ __forceinline__ u64 qc_two_perms(u64 s[12], u32 z) {
  u64 out0 = 0;
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    if (pass) qc_ext(s, QC_RC_I0);
    QC_UNROLL(QC_UNROLL_F)
    for (int r = 0; r < 4; r++) qc_full(s, QC_RC_F1[r]);
    u64 x = qc_sbox(s[0]);
    QC_UNROLL(QC_UNROLL_P)
    for (int r = 0; r < 21; r++) x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
    qc_partial_last(s, x);
    QC_UNROLL(QC_UNROLL_F)
    for (int r = 1; r < 4; r++) qc_full(s, QC_RC_T[r]);
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = qc_sbox(s[i]);
    if (pass) {
      out0 = qc_ext0(s);
    } else {
      qc_ext(s, QC_RC_T[4]);
    }
  }
  return out0;
}

struct QcLaunch {
  u64 c[12];      // the prestate (see the top of this file), each below p
  u64 target;     // targetHigh
  u64 base;       // the first counter; base + count - 1 keeps base's high word
  u32 count;      // counters in this launch
  u32 cap;        // hit slots in hits[]
  u32 zero;       // 0, which ptxas cannot see (QC_PSIG 2)
};

// w7 = bswap32(counter) enters element j as k[j] * w7 (the first linear layer's
// columns for element 7: M4's last column [1,1,3,2] in groups 0 and 2, twice it
// in group 1, where element 7 is).
__global__ void __launch_bounds__(QC_THREADS, QC_MINB)
qc_scan_kernel(const QcLaunch L, unsigned long long *__restrict__ nhits, u64 *__restrict__ hits) {
  const u32 stride = gridDim.x * blockDim.x;
  const u32 base_lo = qc_lo(L.base);
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < L.count; i += stride) {
    const u32 w7 = __byte_perm(base_lo + i, 0, 0x0123);
    const u64 w1 = w7, w2 = (u64)w7 << 1, w3 = w1 + w2, w4 = (u64)w7 << 2, w6 = w2 + w4;
    u64 s[12];
    s[0] = qc_add_small(L.c[0], w1);
    s[1] = qc_add_small(L.c[1], w1);
    s[2] = qc_add_small(L.c[2], w3);
    s[3] = qc_add_small(L.c[3], w2);
    s[4] = qc_add_small(L.c[4], w2);
    s[5] = qc_add_small(L.c[5], w2);
    s[6] = qc_add_small(L.c[6], w6);
    s[7] = qc_add_small(L.c[7], w4);
    s[8] = qc_add_small(L.c[8], w1);
    s[9] = qc_add_small(L.c[9], w1);
    s[10] = qc_add_small(L.c[10], w3);
    s[11] = qc_add_small(L.c[11], w2);
    const u64 h = qc_canon(qc_two_perms(s, L.zero));
    // The hash's first 8 bytes are h's little-endian bytes, read big-endian.
    const u64 top = qc_pack(__byte_perm(qc_hi(h), 0, 0x0123), __byte_perm(qc_lo(h), 0, 0x0123));
    if (top <= L.target) {
      const unsigned long long slot = atomicAdd(nhits, 1ull);
      if (slot < L.cap) hits[slot] = L.base + i;
    }
  }
}
