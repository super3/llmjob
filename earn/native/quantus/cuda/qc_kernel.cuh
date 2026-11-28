// The Quantus scan kernel: for each point of a line of nonces, the two Poseidon2
// permutations of blocks 3 and 4 from the midstate, and a hit when the hash's
// top 64 bits are at or below the target's. earn/native/quantus/README.md has
// the loop this is; src/shared/quantus/qpow.js (indexTop64) is the reference
// it is checked against, and says how a line is laid out.
//
// Along a line only elements 0 and 4 change after the first permutation's first
// linear layer: x0 = X0 + 35 t and x4 = X4 - 35 t. The host (qpow.lineWork)
// has already worked out the other ten first-round S-boxes and the first
// round's linear layer of them, with the second round's constants: the twelve
// constants C. So per nonce the kernel does two S-boxes, a = x0^7 and b = x4^7,
// and the twelve sums
//   s[4g + k] = C[4g + k] + M[k] * (a * (g == 0 ? 2 : 1) + b * (g == 1 ? 2 : 1)),   M = (2, 1, 1, 3)
// (qc_first_round), then carries on from the second round's S-boxes. That is 10
// of a nonce's 236 S-boxes and most of one of its 17 linear layers saved
// against starting at the first layer (core API 1 did, with a counter in the
// nonce's last word).
//
// Rounds, with each layer's round constants added in the layer before it:
//   4 full rounds    x^7 on all 12, then the external layer E (the first of
//                    the first permutation's is qc_first_round)
//   22 partial       x^7 on element 0, then s[i] = sum(s) + D[i] * s[i]
//   4 full rounds
// Block 4 adds 1 to elements 0 and 1, folded into the first permutation's last
// layer. The second permutation needs only element 0 after its last layer.
#pragma once
#include "qc_field.cuh"

// The block shape. The host reads it back from the kernel built for the card (cudaFuncGetAttributes'
// maxThreadsPerBlock), so it may differ by architecture. QC_THREADS stays the host's default.
#ifndef QC_THREADS
#define QC_THREADS 256
#endif
#ifndef QC_THREADS_K
// RTX 50: one block of 768 an SM (the same 768 threads and 80 registers as 3 blocks of 256). On an RTX 5070 2 blocks
// of 384 ran 1.4-1.7% faster than 3 of 256, and one of 768 another 1.5% as compiled and 1.4% with the patched tails
// (21 Nov).
#define QC_THREADS_K (QC_B120 ? 768 : QC_THREADS)
#endif
#ifndef QC_MINB
// 3 blocks of 256 an SM (80 registers): build h ran 1.3% faster than at 4 on
// an RTX 4090 (at 64 registers it spilled inside the full-round loop), and
// build g 1.1-1.6% faster on an RTX 5080.
#define QC_MINB (QC_B120 ? 1 : 3)
#endif
#ifndef QC_UNROLL_F
#define QC_UNROLL_F 1
#endif
// "#pragma unroll MACRO" is not expanded by GCC's preprocessor (Linux builds), so the count goes through _Pragma.
#define QC_STR(x) #x
#define QC_UNROLL(n) _Pragma(QC_STR(unroll n))
#ifndef QC_UNROLL_P
// sm_120: the 21 partial rounds unrolled in full, so the state's registers never have to line up again for a
// loop's next trip (three a trip saved 14 MOVs a round; all 21 ran 0.9% faster than three on an RTX 5070 as
// compiled and with the patched tails, 21 Nov).
#define QC_UNROLL_P (QC_B120 ? 21 : 1)
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
// are in the host's line constants).
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

// RTX 30 (sm_86) only: the partial rounds' diagonal multiplies as a0 * D + a1 * D' with D' = D * 2^32 mod p
// (QC_DPRIME, qc_dmul below). The partial round's 25 instructions fewer a round made an RTX 3070 3.7% faster
// (20 Nov). The constants' 32-bit words are all below 2^32 - 2^26, so a0 * d + (a sum below 13 * 2^32) cannot pass
// 2^64.
// RTX 40 (sm_89) uses it too: on an RTX 4070 Ti it beat QC_DIAG (below) by 0.9% and 1.1% in two rounds
// (quantus-benchmark.md, 20 Nov).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890 && !defined(QC_DPRIME)
#define QC_DPRIME 1
#endif
// RTX 20 (sm_75) uses it too: with the compact kernel it ran 7% more a clock than the compact kernel alone on an
// RTX 2060 SUPER (quantus-benchmark.md, 20 Nov).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750 && !defined(QC_DPRIME)
#define QC_DPRIME 1
#endif
// RTX 50 (sm_120) too, through its own sum path (qc_partial, QC_PSIG 3): +0.75% with CUDA 12.8 and +0.8% with
// 13.3 on an RTX 5070, where qc_dmuluv (QC_DIAG) lost 4.1% with 12.8 and tied with 13.3 (20 Nov).
#if QC_B120 && !defined(QC_DPRIME)
#define QC_DPRIME 1
#endif
#ifndef QC_DPRIME
#define QC_DPRIME QC_SM86
#endif
#define QC_DP0 0xe771538d3c493f72ull
#define QC_DP1 0x227dad9f27b4a216ull
#define QC_DP2 0x6841ef6af2f3c8e4ull
#define QC_DP3 0x607d75a7869b0a8full
#define QC_DP4 0x37457906a250e744ull
#define QC_DP5 0x209d0e9e98bc43b8ull
#define QC_DP6 0x818274a5aad746caull
#define QC_DP7 0x1d6d98e653ba1da4ull
#define QC_DP8 0x9dbd84325df88282ull
#define QC_DP9 0xa2de251d0c055390ull
#define QC_DP10 0x21a97137f39c774bull
#define QC_DP11 0x170f36c92d824496ull

// a * D + (L + H * 2^32) mod p, below 2^64, for L and H below 13 * 2^32 and Dp = D * 2^32 mod p.
//   a * D = a0 * D + a1 * D * 2^32 = a0 * D + a1 * Dp   (mod p)
// so the sum is X + Y * 2^32 with X = a0 d0 + L + a1 dp0 and Y = a0 d1 + H + a1 dp1, each two IMAD.WIDE with
// no glue between them (the first product's register pair is the second one's addend). With cx, cy their carries
// past 2^64, z1 = X1 + Y0 (carry k) and e = Y1 + cx + k:
//   X + Y 2^32 = (z1:X0) + e 2^64 + cy 2^96 = (z1:X0) + e EPS - cy   (mod p)
// and the tail is the one qc_reduce has, with cy as r3. e passes 2^32 only when Y1 is within 2 of 2^32 (about
// 2^-31 a multiply; the result is then 1 too high), and the tail's own wrap is as rare as qc_reduce's.
// CX and CY say whether X and Y can pass 2^64 for this D: they cannot when the two constants' words sum to below
// 2^32 - 13 (L, H < 13 * 2^32), and then that carry is left out (9 of the 12 Y chains, 6 of the X chains).
#define QC_DM_HEAD                        \
  "{.reg .u32 a0, a1, x0, x1, y0, y1, z1, e, cy, c;\n\t" \
  ".reg .u64 P, Q, X, Y, Z, E;\n\t"       \
  "mov.b64 {a0, a1}, %2;\n\t"             \
  "mul.wide.u32 Q, a0, %4;\n\t"           \
  "add.u64 Q, Q, %8;\n\t"                 \
  "mul.wide.u32 Y, a1, %6;\n\t"
#define QC_DM_Y_CY "add.cc.u64 Y, Y, Q;\n\t" QC_CARRY_DM("cy", "%10")
#define QC_DM_Y "add.u64 Y, Y, Q;\n\t"
#define QC_DM_MID                         \
  "mov.b64 {y0, y1}, Y;\n\t"              \
  "mul.wide.u32 P, a0, %3;\n\t"           \
  "add.u64 P, P, %7;\n\t"                 \
  "mul.wide.u32 X, a1, %5;\n\t"
#define QC_DM_X_CX "add.cc.u64 X, X, P;\n\t" "addc.u32 e, y1, 0;\n\t"
#define QC_DM_X "add.u64 X, X, P;\n\t" "mov.u32 e, y1;\n\t"
#define QC_DM_E                           \
  "mov.b64 {x0, x1}, X;\n\t"              \
  "add.cc.u32 z1, x1, y0;\n\t"            \
  "addc.u32 e, e, 0;\n\t"                 \
  "mov.b64 Z, {x0, z1};\n\t"              \
  "mul.wide.u32 E, e, %9;\n\t"            \
  "add.cc.u64 E, E, Z;\n\t"               \
  "mov.b64 {%0, %1}, E;\n\t"
#if QC_RED == 3
// QC_RED 3's tail (see qc_reduce), with cy as r3: right only once carryfix.py has inverted a carry-in.
#define QC_DM_T_CY "addc.u32 c, %1, 0;\n\t" "subc.cc.u32 %0, %0, cy;\n\t" "subc.u32 %1, c, 0;}"
#else
#define QC_DM_T_CY "addc.u32 c, cy, 0;\n\t" "addc.u32 %1, %1, 0;\n\t" "sub.cc.u32 %0, %0, c;\n\t" "subc.u32 %1, %1, 0;}"
#endif
#ifndef QC_DM_T0
#define QC_DM_T0 1
#endif
#if QC_RED == 3 && QC_DM_T0
// The same with r3 = 0 (no cy): lo = E0 + ~0 + c as compiled, E0 - c once carryfix.py has inverted the carry-in.
// It drops a SEL from 9 of the 12 multiplies (+0.6 to 1.1% on an RTX 3070, 20 Nov).
#define QC_DM_T "addc.u32 c, %1, 0;\n\t" "subc.cc.u32 %0, %0, 0;\n\t" "subc.u32 %1, c, 0;}"
#else
#define QC_DM_T "addc.u32 c, 0, 0;\n\t" "addc.u32 %1, %1, 0;\n\t" "sub.cc.u32 %0, %0, c;\n\t" "subc.u32 %1, %1, 0;}"
#endif
#define QC_DM_IO                                                                                               \
  : "=&r"(o0), "=&r"(o1)                                                                                       \
  : "l"(a), "r"(qc_lo(D)), "r"(qc_hi(D)), "r"(qc_lo(Dp)), "r"(qc_hi(Dp)), "l"(L), "l"(H), "r"(QC_EPS32) QC_KZ_ARG

template <bool CX, bool CY>
__device__ __forceinline__ u64 qc_dmul(u64 a, u64 D, u64 Dp, u64 L, u64 H) {
  u32 o0, o1;
  if (CX && CY) asm(QC_DM_HEAD QC_DM_Y_CY QC_DM_MID QC_DM_X_CX QC_DM_E QC_DM_T_CY QC_DM_IO);
  else if (CY) asm(QC_DM_HEAD QC_DM_Y_CY QC_DM_MID QC_DM_X QC_DM_E QC_DM_T_CY QC_DM_IO);
  else if (CX) asm(QC_DM_HEAD QC_DM_Y QC_DM_MID QC_DM_X_CX QC_DM_E QC_DM_T QC_DM_IO);
  else asm(QC_DM_HEAD QC_DM_Y QC_DM_MID QC_DM_X QC_DM_E QC_DM_T QC_DM_IO);
  return qc_pack(o0, o1);
}
// Which chains can carry, for D0..D11 (worked out from QC_D* and QC_DP*: d0 + dp0 and d1 + dp1 against 2^32 - 13).
#define QC_DM0 qc_dmul<false, true>
#define QC_DM1 qc_dmul<false, false>
#define QC_DM2 qc_dmul<true, false>
#define QC_DM3 qc_dmul<true, false>
#define QC_DM4 qc_dmul<true, false>
#define QC_DM5 qc_dmul<true, false>
#define QC_DM6 qc_dmul<false, false>
#define QC_DM7 qc_dmul<false, false>
#define QC_DM8 qc_dmul<true, true>
#define QC_DM9 qc_dmul<false, true>
#define QC_DM10 qc_dmul<true, false>
#define QC_DM11 qc_dmul<false, false>

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
// sm_120 (QC_B120): the partial rounds' sum folded once and carried in each
// product's addend (QC_PSIG 3), and the Q96 folds in two ALU ops (QC_F2).
#ifndef QC_PSIG
#define QC_PSIG (QC_B120 ? 3 : 0)
#endif
#ifndef QC_F2
#define QC_F2 QC_B120
#endif
#if QC_F2
#define QC_FOLDQ q96_fold2
#else
#define QC_FOLDQ q96_fold
#endif
// QC_DIAG 1: the partial rounds' products with qc_dmuluv (four IMAD.WIDE each, the sum folded once and carried in
// their addends, no reduction multiply) instead of qc_mul_add (five, and a 128-bit add). +3.8-4.2% over build h on
// an RTX 4070 Ti, but QC_DPRIME, which keeps the sum unfolded, beat it by about 1% there and on an RTX 3070, so
// it is off everywhere (quantus-benchmark.md, 20 Nov). QC_DPRIME takes precedence when both are set.
#ifndef QC_DIAG
#define QC_DIAG 0
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
#if QC_EXTV == 2
  // The same nine adds with no doubling: ptxas made each "x + x" of a zero-extended half a shift, a mask and a
  // shifted add (six instructions where a three-input add takes two). With s = x0 + x1 + x2 + x3,
  // y_i = s + (x_i + x_(i+1)) + x_(i+1). Not the default: it moves the shifts' IMAD.SHL onto the ALU, which is
  // the pipe that binds on sm_86, and an RTX 3070 ran 0.6% slower with it (20 Nov).
  QH t01 = qh_add(x0, x1), t12 = qh_add(x1, x2), t23 = qh_add(x2, x3), t30 = qh_add(x3, x0);
  QH s = qh_add(t01, t23);
  y0 = qh_add3(s, t01, x1);
  y1 = qh_add3(s, t12, x2);
  y2 = qh_add3(s, t23, x3);
  y3 = qh_add3(s, t30, x0);
#else
  QH t01 = qh_add(x0, x1);
  QH t23 = qh_add(x2, x3);
  QH t0123 = qh_add(t01, t23);
  QH t01123 = qh_add(t0123, x1);
  QH t01233 = qh_add(t0123, x3);
  y3 = qh_add3(t01233, x0, x0);
  y1 = qh_add3(t01123, x2, x2);
  y0 = qh_add(t01123, t01);
  y2 = qh_add(t01233, t23);
#endif
}

// QC_M4 2: the same chain with each half of the doubled terms (2 x0 in y3, 2 x2 in y1) made by one IMAD.WIDE,
// t + x * two. ptxas turns x + x into a 64-bit shift with masks (about 36 instructions an external layer); two is
// 2 from the launch's zero, so it cannot see the multiply is a shift. That also moves work from the ALU, the pipe
// that binds on Ada, to the multiplier. Experiment, off by default.
#ifndef QC_M4
#define QC_M4 0
#endif
__device__ __forceinline__ QH qh_add2x(const QH &t, u64 x, u32 two) {
  QH r;
  r.L = (u64)qc_lo(x) * two + t.L;
  r.H = (u64)qc_hi(x) * two + t.H;
  return r;
}
__device__ __forceinline__ void qh_mat4w(u64 s0, u64 s1, u64 s2, u64 s3, QH &y0, QH &y1, QH &y2, QH &y3, u32 two) {
  const QH x1 = qh(s1), x3 = qh(s3);
  QH t01 = qh_add(qh(s0), x1);
  QH t23 = qh_add(qh(s2), x3);
  QH t0123 = qh_add(t01, t23);
  QH t01123 = qh_add(t0123, x1);
  QH t01233 = qh_add(t0123, x3);
  y3 = qh_add2x(t01233, s0, two);
  y1 = qh_add2x(t01123, s2, two);
  y0 = qh_add(t01123, t01);
  y2 = qh_add(t01233, t23);
}

// s = E(s) + rc. E: M4 on each group of four, then each element gets the sum of
// its column across the three groups. two is 2 (see QC_M4).
__device__ __forceinline__ void qc_ext(u64 s[12], const u64 *rc, u32 two = 2) {
#if QC_EXTV >= 1
  QH y[12];
#if QC_M4 == 2
#pragma unroll
  for (int g = 0; g < 12; g += 4) qh_mat4w(s[g], s[g + 1], s[g + 2], s[g + 3], y[g], y[g + 1], y[g + 2], y[g + 3], two);
#else
  (void)two;
#pragma unroll
  for (int g = 0; g < 12; g += 4) qh_mat4(qh(s[g]), qh(s[g + 1]), qh(s[g + 2]), qh(s[g + 3]), y[g], y[g + 1], y[g + 2], y[g + 3]);
#endif
#pragma unroll
  for (int k = 0; k < 4; k++) {
    QH col = qh_add3(y[k], y[k + 4], y[k + 8]);
#pragma unroll
    for (int g = 0; g < 12; g += 4) s[g + k] = qh_fold(qh_add3(y[g + k], col, qh(rc[g + k])));
  }
#else
  (void)two;
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
      s[g + k] = QC_FOLDQ(w);
    }
  }
#endif
}

// Element 0 of E(s): 2 * m0 + m4 + m8, m the first row of M4 on each group.
__device__ __forceinline__ u64 qc_ext0(const u64 s[12]) {
#if QC_EXTV >= 1
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
  return QC_FOLDQ(w);
#endif
}

// The full rounds' squares (QC_SQR's forms). On sm_120 the best form depends on
// the compiler: built with CUDA 12.8, four products (QC_SQR 0) ran 1.6% faster
// than three on an RTX 5070; built with 13.3, three (QC_SQR 1) ran 3.0% faster
// than four (20 Nov), so the two cores took different forms.
// With one block of 768 an SM three products won with CUDA 12.8 as well: 2.4% as compiled and 3.1% with the patched
// tails on an RTX 5070 (21 Nov), so both RTX 50 cores take QC_SQR 1 now.
#ifndef QC_SQRF
#define QC_SQRF (QC_B120 ? 1 : QC_SQR)
#endif

__device__ __forceinline__ void qc_full(u64 s[12], const u64 *rc, u32 two = 2) {
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = qc_sbox<QC_SQRF>(s[i]);
  qc_ext(s, rc, two);
}

#ifndef QC_SUM
#define QC_SUM (QC_B120 ? 1 : 2)
#endif

// x + s[1] + ... + s[11] as a Q96. QC_SUM 2 sums the low and the high halves
// separately as u64s, which ptxas turns into three-input IADD3s below sm_120
// (on sm_120, IADD.64s with a MOV a term, so it takes the add-with-carry chain).
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
#elif QC_SUM == 3
  // x last: x is the S-box output the round waits on, while s[1..11] are ready
  // earlier, so only one add stands between x and the fold.
  Q96 sum = q96_sum(s[1], s[2]);
#pragma unroll
  for (int i = 3; i < 12; i++) q96_add(sum, s[i]);
  q96_add(sum, x);
  return sum;
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
#if QC_PSIG == 3 || QC_PSIG == 4
  // sm_120: the sum as a Q96 add-with-carry chain, folded twice in two ALU ops
  // each (with and without rc0), then each fold's halves ride in the low and the
  // middle product's addends as (half, z) pairs. z is a launch-time zero ptxas
  // cannot see, so it keeps each pair whole (one MOV a pair a round).
#if QC_PSIG == 4
  // Element 0 without waiting for x to enter the sum: with S = s[1] + ... +
  // s[11], out0 = D0 x + (x + S + rc0) = (D0 + 1) x + (S + rc0). Only the
  // product stands between this round's S-box and the next one.
  Q96 S = q96_sum(s[1], s[2]);
#pragma unroll
  for (int i = 3; i < 12; i++) q96_add(S, s[i]);
  Q96 w = S, w0 = S;
  q96_add(w0, rc0);
  q96_add(w, x);
  const u64 d0 = QC_D0 + 1;
#else
  Q96 w = qc_sum12(s, x);
  Q96 w0 = w;
  q96_add(w0, rc0);
  const u64 d0 = QC_D0;
#endif
  const u64 sig = q96_fold2(w), sig0 = q96_fold2(w0);
  const u64 l0 = qc_pack(qc_lo(sig0), z), h0 = qc_pack(qc_hi(sig0), z);
  const u64 l = qc_pack(qc_lo(sig), z), hh = qc_pack(qc_hi(sig), z);
#if QC_DPRIME
  // The shared D' multiply (q3-rtx30's qc_dmul) on this path's folded sums: their halves are below 2^32, inside
  // its chains' bounds. Element 0 keeps D0 (QC_DP0 is D0's D'), so QC_PSIG 4 does not apply.
#if QC_PSIG == 4
#error "QC_DPRIME on the sm_120 path needs QC_PSIG 3: its element-0 constants are D0's"
#endif
  (void)d0;
  const u64 out0 = QC_DM0(x, QC_D0, QC_DP0, l0, h0);
  s[1] = QC_DM1(s[1], QC_D1, QC_DP1, l, hh);
  s[2] = QC_DM2(s[2], QC_D2, QC_DP2, l, hh);
  s[3] = QC_DM3(s[3], QC_D3, QC_DP3, l, hh);
  s[4] = QC_DM4(s[4], QC_D4, QC_DP4, l, hh);
  s[5] = QC_DM5(s[5], QC_D5, QC_DP5, l, hh);
  s[6] = QC_DM6(s[6], QC_D6, QC_DP6, l, hh);
  s[7] = QC_DM7(s[7], QC_D7, QC_DP7, l, hh);
  s[8] = QC_DM8(s[8], QC_D8, QC_DP8, l, hh);
  s[9] = QC_DM9(s[9], QC_D9, QC_DP9, l, hh);
  s[10] = QC_DM10(s[10], QC_D10, QC_DP10, l, hh);
  s[11] = QC_DM11(s[11], QC_D11, QC_DP11, l, hh);
#else
  const u64 out0 = qc_mul_add_pair(x, d0, l0, h0);
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
#endif
  return out0;
#elif QC_DPRIME
  // The sum as two sums of 32-bit halves (each below 12 * 2^32), which go straight into qc_dmul's first products.
  (void)z;
  u64 L = qc_lo(x), H = qc_hi(x);
#pragma unroll
  for (int i = 1; i < 12; i++) {
    L += qc_lo(s[i]);
    H += qc_hi(s[i]);
  }
  const u64 out0 = QC_DM0(x, QC_D0, QC_DP0, L + qc_lo(rc0), H + qc_hi(rc0));
  s[1] = QC_DM1(s[1], QC_D1, QC_DP1, L, H);
  s[2] = QC_DM2(s[2], QC_D2, QC_DP2, L, H);
  s[3] = QC_DM3(s[3], QC_D3, QC_DP3, L, H);
  s[4] = QC_DM4(s[4], QC_D4, QC_DP4, L, H);
  s[5] = QC_DM5(s[5], QC_D5, QC_DP5, L, H);
  s[6] = QC_DM6(s[6], QC_D6, QC_DP6, L, H);
  s[7] = QC_DM7(s[7], QC_D7, QC_DP7, L, H);
  s[8] = QC_DM8(s[8], QC_D8, QC_DP8, L, H);
  s[9] = QC_DM9(s[9], QC_D9, QC_DP9, L, H);
  s[10] = QC_DM10(s[10], QC_D10, QC_DP10, L, H);
  s[11] = QC_DM11(s[11], QC_D11, QC_DP11, L, H);
  return out0;
#elif QC_DIAG == 1
  // The sum of all twelve folded to 64 bits once (twice for element 0, which also takes the next round's constant),
  // its halves zero-extended once for the twelve products.
  QH h = qh(x);
#pragma unroll
  for (int i = 1; i < 12; i++) {
    h.L += qc_lo(s[i]);
    h.H += qc_hi(s[i]);
  }
  const u64 S = qh_fold(h);
  const u64 S0 = qh_fold(qh_add(h, qh(rc0)));
  const u64 l = qc_lo(S), hh = qc_hi(S);
  (void)z;
  const u64 out0 = qc_dmuluv<QC_D0>(x, qc_lo(S0), qc_hi(S0));
  s[1] = qc_dmuluv<QC_D1>(s[1], l, hh);
  s[2] = qc_dmuluv<QC_D2>(s[2], l, hh);
  s[3] = qc_dmuluv<QC_D3>(s[3], l, hh);
  s[4] = qc_dmuluv<QC_D4>(s[4], l, hh);
  s[5] = qc_dmuluv<QC_D5>(s[5], l, hh);
  s[6] = qc_dmuluv<QC_D6>(s[6], l, hh);
  s[7] = qc_dmuluv<QC_D7>(s[7], l, hh);
  s[8] = qc_dmuluv<QC_D8>(s[8], l, hh);
  s[9] = qc_dmuluv<QC_D9>(s[9], l, hh);
  s[10] = qc_dmuluv<QC_D10>(s[10], l, hh);
  s[11] = qc_dmuluv<QC_D11>(s[11], l, hh);
  return out0;
#elif QC_PSIG
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
  const u64 *rc = QC_RC_T[0];
#if QC_DPRIME && !QC_B120
  u64 L = qc_lo(x), H = qc_hi(x);
#pragma unroll
  for (int i = 1; i < 12; i++) {
    L += qc_lo(s[i]);
    H += qc_hi(s[i]);
  }
  s[0] = QC_DM0(x, QC_D0, QC_DP0, L + qc_lo(rc[0]), H + qc_hi(rc[0]));
  s[1] = QC_DM1(s[1], QC_D1, QC_DP1, L + qc_lo(rc[1]), H + qc_hi(rc[1]));
  s[2] = QC_DM2(s[2], QC_D2, QC_DP2, L + qc_lo(rc[2]), H + qc_hi(rc[2]));
  s[3] = QC_DM3(s[3], QC_D3, QC_DP3, L + qc_lo(rc[3]), H + qc_hi(rc[3]));
  s[4] = QC_DM4(s[4], QC_D4, QC_DP4, L + qc_lo(rc[4]), H + qc_hi(rc[4]));
  s[5] = QC_DM5(s[5], QC_D5, QC_DP5, L + qc_lo(rc[5]), H + qc_hi(rc[5]));
  s[6] = QC_DM6(s[6], QC_D6, QC_DP6, L + qc_lo(rc[6]), H + qc_hi(rc[6]));
  s[7] = QC_DM7(s[7], QC_D7, QC_DP7, L + qc_lo(rc[7]), H + qc_hi(rc[7]));
  s[8] = QC_DM8(s[8], QC_D8, QC_DP8, L + qc_lo(rc[8]), H + qc_hi(rc[8]));
  s[9] = QC_DM9(s[9], QC_D9, QC_DP9, L + qc_lo(rc[9]), H + qc_hi(rc[9]));
  s[10] = QC_DM10(s[10], QC_D10, QC_DP10, L + qc_lo(rc[10]), H + qc_hi(rc[10]));
  s[11] = QC_DM11(s[11], QC_D11, QC_DP11, L + qc_lo(rc[11]), H + qc_hi(rc[11]));
  return;
#endif
  Q96 sum = qc_sum12(s, x);
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

// RTX 20 (sm_75) runs the compact kernel (QC_COMPACT 2, below): build h's 3,300
// instructions (53 KB) overflow Turing's instruction cache, and the compact one
// (1,288, 21 KB) ran 7-11% faster a clock on an RTX 2060 SUPER and an RTX 2060
// with the same pipes and the same per-nonce work. Other architectures keep
// build h until they race it.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
#define QC_SM75 1
#else
#define QC_SM75 0
#endif
#ifndef QC_COMPACT
#define QC_COMPACT (QC_SM75 ? 2 : 0)
#endif

// The layer constants in the order a pass uses them: [0] before the second
// permutation, [1..4] after the first half's full rounds, [5..7] after terminal
// rounds 0..2, [8] after the first permutation's last round.
__constant__ u64 QC_RC_SEQ[9][12] = {
    {0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull, 0xe4f2fe29e03234acull,
     0xd5e0ba2f541b6449ull, 0xec33b868f3cc46c1ull, 0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull,
     0xe3f30d509a1436bbull, 0xd9a64f068dca7c29ull, 0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull},
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

// Partial rounds 1..21's constants on element 0, then the first terminal
// round's on element 0 (what the last partial round adds there).
__constant__ u64 QC_RC_P22[22] = {
    0xd1d2bf082f60d4f0ull, 0x69a377a79f9ad206ull, 0xa9d06906a3858e24ull, 0x295275001eede5b5ull,
    0x5874e441117bd746ull, 0x8a084bbba8ed86ccull, 0x3defd7645cde6425ull, 0x3998cfe6871cc137ull,
    0x3e52ef8bca48314aull, 0x964a209f85dc9eccull, 0x3fcc9ee82cc4577eull, 0x8e79b4a5d0096d6dull,
    0x8492362ad2392556ull, 0xee72f470262574d6ull, 0x1e0e18496da2444aull, 0x0f3a74bf215eaac6ull,
    0x1b061b76a1c0ded3ull, 0x192c42d86803d7a6ull, 0xf6d49ff997ae0260ull, 0x3ec372e7a0fa3786ull,
    0x5538cdf4f23445d3ull, 0x7b72c539e0ea4c6eull};

#ifndef QC_HAND89
#define QC_HAND89 0
#endif
// RTX 20 (sm_75): the same function, called from the compact kernel, for sm75/ (QC_HAND75).
#ifndef QC_HAND75
#define QC_HAND75 0
#endif
#if QC_HAND89 || QC_HAND75
// RTX 40 (sm_89) hand-written partial rounds (sm89/hand.py): the 22 partial rounds of one permutation half in a
// function ptxas may not inline, so hand.py can find it in the cubin and replace its body. As compiled it is the
// same code as the inline path, and the kernel is right either way.
struct QcState {
  u64 v[12];
};
__device__ __noinline__ QcState qc_hand_loop(QcState st, u32 z) {
  u64 *s = st.v;
  u64 x = qc_sbox(s[0]);
  // The first round peeled and two rounds a trip, only to make the function big enough for hand.py's code (its
  // room is this code's size).
  x = qc_sbox(qc_partial(s, x, QC_RC_P[0], z));
#pragma unroll 2
  for (int r = 1; r < 21; r++) x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
  qc_partial_last(s, x);
  return st;
}
#endif

#if QC_COMPACT
// Both permutations with one copy of each kind of round in the code: 16 steps
// of (linear layer, S-boxes), the partial rounds inside step 4 of each pass.
// Step k runs layer QC_RC_SEQ[k mod 9]. The first permutation's first round
// (step 0, and step 1's layer) is qc_first_round's, so the loop starts at step
// 1's S-boxes; step 8 (the first permutation's last layer) has no S-boxes.
// About 1,450 instructions against 3,300 for qc_two_perms.
__device__ __forceinline__ u64 qc_two_perms_compact(u64 s[12], u32 z) {
  const u32 two = z + 2;  // 2, which ptxas cannot see (QC_M4 2)
#pragma unroll 1
  for (int k = 1; k < 17; k++) {
    const int j = k >= 9 ? k - 9 : k;
    if (k > 1) qc_ext(s, QC_RC_SEQ[j], two);
    if (j == 4) {
#if QC_HAND75
      QcState st;
#pragma unroll
      for (int i = 0; i < 12; i++) st.v[i] = s[i];
      st = qc_hand_loop(st, z);
#pragma unroll
      for (int i = 0; i < 12; i++) s[i] = st.v[i];
#elif QC_COMPACT >= 2
      // All 22 partial rounds in one loop; the last one's constants for
      // elements 1..11 are added after it.
      u64 x = s[0];
      QC_UNROLL(QC_UNROLL_P)
      for (int r = 0; r < 22; r++) x = qc_partial(s, qc_sbox(x), QC_RC_P22[r], z);
      s[0] = x;
#pragma unroll
      for (int i = 1; i < 12; i++) s[i] = qc_add_small(s[i], QC_RC_T[0][i]);
#else
      u64 x = qc_sbox(s[0]);
      QC_UNROLL(QC_UNROLL_P)
      for (int r = 0; r < 21; r++) x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
      qc_partial_last(s, x);
#endif
    }
    if (j != 8) {
#pragma unroll
      for (int i = 0; i < 12; i++) s[i] = qc_sbox<QC_SQRF>(s[i]);
    }
  }
  return qc_ext0(s);
}
#endif

// Both permutations from the state after the first one's first round
// (qc_first_round). Returns element 0 after the second one (not canonical).
__device__ __forceinline__ u64 qc_two_perms(u64 s[12], u32 z) {
#if QC_COMPACT
  return qc_two_perms_compact(s, z);
#else
  u64 out0 = 0;
  const u32 two = z + 2;  // 2, which ptxas cannot see (QC_M4 2)
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    if (pass) qc_ext(s, QC_RC_I0, two);
    // The first permutation's round 0 is already done.
    QC_UNROLL(QC_UNROLL_F)
    for (int r = pass ? 0 : 1; r < 4; r++) qc_full(s, QC_RC_F1[r], two);
#if QC_HAND89
    {
      QcState st;
#pragma unroll
      for (int i = 0; i < 12; i++) st.v[i] = s[i];
      st = qc_hand_loop(st, z);
#pragma unroll
      for (int i = 0; i < 12; i++) s[i] = st.v[i];
    }
#else
    u64 x = qc_sbox(s[0]);
    QC_UNROLL(QC_UNROLL_P)
    for (int r = 0; r < 21; r++) x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
    qc_partial_last(s, x);
#endif
    QC_UNROLL(QC_UNROLL_F)
    for (int r = 1; r < 4; r++) qc_full(s, QC_RC_T[r], two);
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = qc_sbox<QC_SQRF>(s[i]);
    if (pass) {
      out0 = qc_ext0(s);
    } else {
      qc_ext(s, QC_RC_T[4], two);
    }
  }
  return out0;
#endif
}


#ifndef QC_NPT
#define QC_NPT 1
#endif
// QC_DESYNC: see qc_scan_kernel. RTX 30 (sm_86) runs 8: its patched kernel ran 1.4% faster with it on an RTX 3070
// (quantus-benchmark.md, 21 Nov). The other architectures keep 0, so their code is unchanged.
#ifndef QC_DESYNC
#define QC_DESYNC (QC_SM86 ? 8 : 0)
#endif
#if QC_NPT == 2
// Two nonces a thread, each step of both permutations run for both states back to back, so ptxas can interleave
// their independent multiplies and adds. Off: on an RTX 4070 Ti (sm_89, QC_DPRIME, clock held at 2,400 MHz) it ran
// 2.6% slower at 2 blocks of 256 (128 registers, 64 bytes of spill), 4.8% at 1 block (222 registers) and 7.6% at
// 3 blocks of 128 (quantus-benchmark.md, 20 Nov). Kept for the other architectures to race.
__device__ __forceinline__ void qc_two_perms2(u64 a[12], u64 b[12], u32 z, u64 &oa, u64 &ob) {
  const u32 two = z + 2;
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    if (pass) { qc_ext(a, QC_RC_I0, two); qc_ext(b, QC_RC_I0, two); }
#pragma unroll 1
    for (int r = pass ? 0 : 1; r < 4; r++) { qc_full(a, QC_RC_F1[r], two); qc_full(b, QC_RC_F1[r], two); }
    u64 xa = qc_sbox(a[0]), xb = qc_sbox(b[0]);
#pragma unroll 1
    for (int r = 0; r < 21; r++) {
      xa = qc_sbox(qc_partial(a, xa, QC_RC_P[r], z));
      xb = qc_sbox(qc_partial(b, xb, QC_RC_P[r], z));
    }
    qc_partial_last(a, xa);
    qc_partial_last(b, xb);
#pragma unroll 1
    for (int r = 1; r < 4; r++) { qc_full(a, QC_RC_T[r], two); qc_full(b, QC_RC_T[r], two); }
#pragma unroll
    for (int i = 0; i < 12; i++) { a[i] = qc_sbox(a[i]); b[i] = qc_sbox(b[i]); }
    if (pass) {
      oa = qc_ext0(a);
      ob = qc_ext0(b);
    } else {
      qc_ext(a, QC_RC_T[4], two);
      qc_ext(b, QC_RC_T[4], two);
    }
  }
}
#endif

struct QcLaunch {
  u64 c[12];      // the line's constants C (see the top of this file), each below p
  u64 x0, x4;     // elements 0 and 4 before the first round's S-boxes, at this launch's first point; below p
  u64 target;     // targetHigh
  u64 base;       // this launch's first point on the line (what a hit is numbered from)
  u32 count;      // points in this launch, at most 2^26
  u32 cap;        // hit slots in hits[]
  u32 zero;       // 0, which ptxas cannot see (QC_PSIG 2)
};

// What one step along a line adds to element 0 and takes from element 4.
#define QC_SLOPE 35u

// The state after the first permutation's first round for the launch's point i
// (see the top of this file). d = 35 i fits a u32 because a launch is at most
// 2^26 points; x4 takes p - d, which is at most p, qc_add_small's bound. The
// sums are of 32-bit halves (QH): the largest is a constant plus six S-box
// outputs, far inside qh_fold's 2^40.
__device__ __forceinline__ void qc_first_round(const QcLaunch &L, u32 i, u64 s[12]) {
  const u32 d = QC_SLOPE * i;
  const QH a = qh(qc_sbox<QC_SQRF>(qc_add_small(L.x0, (u64)d)));
  const QH b = qh(qc_sbox<QC_SQRF>(qc_add_small(L.x4, QC_P - (u64)d)));
  QH v[3];
  v[2] = qh_add(a, b);
  v[0] = qh_add(v[2], a);
  v[1] = qh_add(v[2], b);
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const QH v2 = qh_add(v[g], v[g]);
    s[4 * g] = qh_fold(qh_add(v2, qh(L.c[4 * g])));
    s[4 * g + 1] = qh_fold(qh_add(v[g], qh(L.c[4 * g + 1])));
    s[4 * g + 2] = qh_fold(qh_add(v[g], qh(L.c[4 * g + 2])));
    s[4 * g + 3] = qh_fold(qh_add3(v2, v[g], qh(L.c[4 * g + 3])));
  }
}

__global__ void __launch_bounds__(QC_THREADS_K, QC_MINB)
qc_scan_kernel(const QcLaunch L, unsigned long long *__restrict__ nhits, u64 *__restrict__ hits) {
  const u32 stride = gridDim.x * blockDim.x;
#if QC_DESYNC
  // Warps 4-7 of each block (one of each SM sub-partition's pair) first run QC_DESYNC S-boxes on junk, so the warps
  // that share a scheduler start out of phase. The junk is kept alive by a compare no input meets. On sm_86 it made
  // the patched kernel 1.4% faster, but partbench saw no gain from the phase offset itself: ptxas allocates the
  // kernel's registers differently with the extra loop (196 fewer IMAD.MOV and 19 fewer spill loads and stores a
  // nonce), which is the likelier cause. It costs 8 S-boxes a thread a launch, about 0.005% of the work.
  if ((threadIdx.x >> 7) & 1) {
    u64 t = L.x0 ^ threadIdx.x;
    for (int k = 0; k < QC_DESYNC; k++) t = qc_sbox(t);
    if (t == 0x123456789abcdefull) atomicAdd(nhits, 0ull);
  }
#endif
#if QC_NPT == 2
  for (u32 i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); i < L.count; i += 2 * stride) {
    u64 st[2][12];
#pragma unroll
    for (int n = 0; n < 2; n++) qc_first_round(L, i + n, st[n]);
    u64 hh[2];
    qc_two_perms2(st[0], st[1], L.zero, hh[0], hh[1]);
#pragma unroll
    for (int n = 0; n < 2; n++) {
      const u64 h = qc_canon(hh[n]);
      const u64 top = qc_pack(__byte_perm(qc_hi(h), 0, 0x0123), __byte_perm(qc_lo(h), 0, 0x0123));
      if (i + n < L.count && top <= L.target) {
        const unsigned long long slot = atomicAdd(nhits, 1ull);
        if (slot < L.cap) hits[slot] = L.base + i + n;
      }
    }
  }
#else
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < L.count; i += stride) {
    u64 s[12];
    qc_first_round(L, i, s);
    const u64 h = qc_canon(qc_two_perms(s, L.zero));
    // The hash's first 8 bytes are h's little-endian bytes, read big-endian.
    const u64 top = qc_pack(__byte_perm(qc_hi(h), 0, 0x0123), __byte_perm(qc_lo(h), 0, 0x0123));
    if (top <= L.target) {
      const unsigned long long slot = atomicAdd(nhits, 1ull);
      if (slot < L.cap) hits[slot] = L.base + i;
    }
  }
#endif
}
