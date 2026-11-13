// Poseidon2 over the Goldilocks field, as Quantus's proof of work uses it, for
// the Intel (SYCL) Quantus core. The same code runs on the host (the midstate
// fold in quantus_sycl.cpp, qtc_selftest.cpp) and in the kernel.
//
// Width 12, 4 + 4 full rounds, 22 partial rounds, S-box x^7, Plonky3's linear
// layers. The constants are qp-poseidon-core 3.1.0's, the same as
// earn/src/shared/quantus/poseidon2.js (which carries the crate's MIT notice).
//
// Every value is a u64 below 2^64 that stands for its residue mod p: the
// arithmetic is exact, but a value is not always below p. canonical() makes it
// so, and only the hash's first word needs that.
//
// The round constant of each round is added in the linear layer before it, as
// quantus-miner's CUDA kernel does: a full round is then S-boxes and one linear
// layer, and the constant rides in the linear layer's wide sums.
#pragma once
#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__)
#define QTC_INLINE inline __attribute__((always_inline))
#else
#define QTC_INLINE inline
#endif

// How the round loops are written. 1: every round unrolled (the constants become
// immediates, the kernel is large); 0: loops over rounds reading the constants
// from tables (a small kernel). The partial rounds and the full rounds can be
// set apart.
#ifndef QTC_UNROLL_FULL
#define QTC_UNROLL_FULL 0
#endif
#ifndef QTC_UNROLL_PARTIAL
#define QTC_UNROLL_PARTIAL 0
#endif

#define QTC_PRAGMA(x) _Pragma(#x)
#if QTC_UNROLL_FULL
#define QTC_FULL_LOOP QTC_PRAGMA(unroll)
#else
#define QTC_FULL_LOOP QTC_PRAGMA(unroll 1)
#endif
#if QTC_UNROLL_PARTIAL
#define QTC_PARTIAL_LOOP QTC_PRAGMA(unroll)
#else
#define QTC_PARTIAL_LOOP QTC_PRAGMA(unroll 1)
#endif

namespace qtc {

constexpr uint64_t P = 0xffffffff00000001ull;
// 2^64 mod p.
constexpr uint64_t EPS = 0xffffffffull;
constexpr int WIDTH = 12;

// The internal layer is s[i] = sum(s) + s[i] * DIAG[i].
constexpr uint64_t DIAG[12] = {
    0xc3b6c08e23ba9300ull, 0xd84b5de94a324fb6ull, 0x0d0c371c5b35b84full, 0x7964f570e7188037ull,
    0x5daf18bbd996604bull, 0x6743bc47b9595257ull, 0x5528b9362c59bb70ull, 0xac45e25b7127b68bull,
    0xa2077d7dfbb606b5ull, 0xf3faac6faee378aeull, 0x0c6388b51545e883ull, 0xd27dbb6944917b60ull,
};

// The first full round's constants, added in the first linear layer (the host
// folds that layer into the scan's prestate).
constexpr uint64_t RC_FIRST[12] = {
    0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull, 0xe4f2fe29e03234acull,
    0xd5e0ba2f541b6449ull, 0xec33b868f3cc46c1ull, 0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull,
    0xe3f30d509a1436bbull, 0xd9a64f068dca7c29ull, 0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull,
};

// The constant added in the linear layer after full round r (r = 0..7): the next
// round's. After round 3 it is the first partial round's (element 0 only); after
// round 7, nothing.
constexpr uint64_t RC_FULL[8][12] = {
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
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

// The constant added to element 0 after partial round r (r = 0..20): the next
// partial round's. After partial round 21 the first terminal full round's
// constants are added to every element (RC_LAST_PARTIAL).
constexpr uint64_t RC_PARTIAL[21] = {
    0xd1d2bf082f60d4f0ull, 0x69a377a79f9ad206ull, 0xa9d06906a3858e24ull, 0x295275001eede5b5ull,
    0x5874e441117bd746ull, 0x8a084bbba8ed86ccull, 0x3defd7645cde6425ull, 0x3998cfe6871cc137ull,
    0x3e52ef8bca48314aull, 0x964a209f85dc9eccull, 0x3fcc9ee82cc4577eull, 0x8e79b4a5d0096d6dull,
    0x8492362ad2392556ull, 0xee72f470262574d6ull, 0x1e0e18496da2444aull, 0x0f3a74bf215eaac6ull,
    0x1b061b76a1c0ded3ull, 0x192c42d86803d7a6ull, 0xf6d49ff997ae0260ull, 0x3ec372e7a0fa3786ull,
    0x5538cdf4f23445d3ull,
};
constexpr uint64_t RC_LAST_PARTIAL[12] = {
    0x7b72c539e0ea4c6eull, 0x144573dae2ce9976ull, 0x802028b68f35fc88ull, 0x6d36c5022c4fe7c2ull,
    0xa205d0ffa9b9def3ull, 0xf6e7e38b1ea6ba2full, 0x34f7909ae5258d64ull, 0xb0464d9d77b97fcaull,
    0x64ddb9d5de7e00a6ull, 0x0ed0d75c27975d97ull, 0x1cbb36f11127338bull, 0x6673e505cfd0b6baull,
};

// ── Field arithmetic ────────────────────────────────────────────────────────

// A sum of a few u64s, kept as lo + 2^64 * hi (hi small), reduced once.
struct Wide {
  uint64_t lo;
  uint32_t hi;
};

QTC_INLINE Wide wide(uint64_t x) { return Wide{x, 0u}; }

QTC_INLINE void wadd(Wide &w, uint64_t x) {
  w.lo += x;
  w.hi += (uint32_t)(w.lo < x);
}

QTC_INLINE void waddw(Wide &w, const Wide &x) {
  w.lo += x.lo;
  w.hi += x.hi + (uint32_t)(w.lo < x.lo);
}

// ── The fold (QTC_ARITH 2) ──────────────────────────────────────────────────
//
// Every reduction below ends here: x = c0 - N + S 2^32 (mod p), for a 32-bit
// c0 and N, S below 2^62. With 2^64 = 2^32 - 1 (mod p):
//
//   S 2^32 = Slo 2^32 + Shi 2^64 = (Slo + Shi) 2^32 - Shi
//   A = Slo + Shi = Alo + a 2^32 (a is 0 or 1; when it is 1, Alo < Shi)
//   x = (c0 - N - Shi - a) + (Alo + a) 2^32 = B + hiw 2^32
//
// hiw is below 2^32. B is a small negative or a 32-bit number, so V = B +
// hiw 2^32 is below 2^64, or negative by a little: then V + p, which is V's
// 64-bit wrap less 2^32 - 1. Only 32-bit adds and one compare: no 64-bit
// overflow test, which costs the GPU three compares.
QTC_INLINE uint64_t fold(uint32_t c0, uint64_t N, uint64_t S) {
  const uint32_t Slo = (uint32_t)S, Shi = (uint32_t)(S >> 32);
  const uint64_t A = (uint64_t)Slo + Shi;
  const uint32_t a = (uint32_t)(A >> 32);
  const uint32_t hiw = (uint32_t)A + a;
  const uint64_t B = (uint64_t)c0 - N - (uint64_t)(Shi + a);
  const uint32_t vhi = hiw + (uint32_t)(B >> 32);
  uint64_t V = ((uint64_t)vhi << 32) | (uint32_t)B;
  // B's high word is 0 or -k; vhi passes hiw only when hiw - k went below 0.
  if (vhi > hiw) V -= EPS;
  return V;
}

#ifndef QTC_ARITH
#define QTC_ARITH 2
#endif

#if QTC_ARITH == 2
// lo + 2^64 hi = (lo_lo) + (lo_hi + hi) 2^32 - ... : the fold with N = 0.
QTC_INLINE uint64_t wred(const Wide &w) {
  return fold((uint32_t)w.lo, 0, (w.lo >> 32) + ((uint64_t)w.hi << 32));
}
#else
// lo + 2^64 hi = lo + hi (2^32 - 1) mod p. hi (2^32 - 1) is below 2^64 - 2^32,
// so after one carry the sum cannot carry again.
QTC_INLINE uint64_t wred(const Wide &w) {
  const uint64_t t = ((uint64_t)w.hi << 32) - (uint64_t)w.hi;
  uint64_t r = w.lo + t;
  if (r < t) r += EPS;
  return r;
}
#endif

QTC_INLINE uint64_t add(uint64_t a, uint64_t b) {
  uint64_t s = a + b;
  const uint64_t c = s < a ? EPS : 0;
  s += c;
  const uint64_t c2 = s < c ? EPS : 0;
  return s + c2;
}

QTC_INLINE uint64_t canonical(uint64_t a) { return a >= P ? a - P : a; }

// x = lo + 2^64 hi, reduced with 2^64 = 2^32 - 1 and 2^96 = -1 (mod p).
// Plonky2's reduce128: exact for every 128-bit x.
QTC_INLINE uint64_t reduce128(uint64_t lo, uint64_t hi) {
  const uint64_t hh = hi >> 32;
  const uint64_t hl = hi & EPS;
  uint64_t t0 = lo - hh;
  if (lo < hh) t0 -= EPS;
  const uint64_t t1 = (hl << 32) - hl;
  uint64_t r = t0 + t1;
  if (r < t1) r += EPS;
  return r;
}

// The 128-bit product a b, from four 32 x 32 products.
QTC_INLINE void mul128(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) {
  const uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
  lo = (mid << 32) | (uint32_t)p00;
  hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// a^2 from three 32 x 32 products.
QTC_INLINE void sqr128(uint64_t a, uint64_t &lo, uint64_t &hi) {
  const uint64_t a0 = (uint32_t)a, a1 = a >> 32;
  const uint64_t p00 = a0 * a0, p01 = a0 * a1, p11 = a1 * a1;
  const uint64_t mid = (p00 >> 32) + 2 * (uint64_t)(uint32_t)p01;
  lo = (mid << 32) | (uint32_t)p00;
  hi = p11 + 2 * (p01 >> 32) + (mid >> 32);
}

#if QTC_ARITH == 2
// a b from its four 32 x 32 products pIJ = aI bJ (lIJ + hIJ 2^32):
//   a b = l00 + Y 2^32 + X 2^64 + h11 2^96,  Y = h00 + l01 + l10,  X = h01 + h10 + l11
//       = (l00 - X - h11) + (X + Y) 2^32     (2^64 = 2^32 - 1, 2^96 = -1 mod p)
QTC_INLINE uint64_t mul(uint64_t a, uint64_t b) {
  const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32), b0 = (uint32_t)b, b1 = (uint32_t)(b >> 32);
  const uint64_t p00 = (uint64_t)a0 * b0, p01 = (uint64_t)a0 * b1, p10 = (uint64_t)a1 * b0, p11 = (uint64_t)a1 * b1;
  const uint64_t Y = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
  const uint64_t X = (p01 >> 32) + (p10 >> 32) + (uint32_t)p11;
  return fold((uint32_t)p00, X + (p11 >> 32), X + Y);
}

QTC_INLINE uint64_t sqr(uint64_t a) {
  const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
  const uint64_t p00 = (uint64_t)a0 * a0, p01 = (uint64_t)a0 * a1, p11 = (uint64_t)a1 * a1;
  const uint64_t Y = (p00 >> 32) + 2 * (uint64_t)(uint32_t)p01;
  const uint64_t X = 2 * (p01 >> 32) + (uint32_t)p11;
  return fold((uint32_t)p00, X + (p11 >> 32), X + Y);
}

// a d + w (w below 2^69): w's low word joins l00 (its carry joins Y), its
// second word joins Y and its top joins X (w.hi 2^64).
QTC_INLINE uint64_t mul_add(uint64_t a, uint64_t d, const Wide &w) {
  const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32), b0 = (uint32_t)d, b1 = (uint32_t)(d >> 32);
  const uint64_t p00 = (uint64_t)a0 * b0, p01 = (uint64_t)a0 * b1, p10 = (uint64_t)a1 * b0, p11 = (uint64_t)a1 * b1;
  const uint64_t c = (uint64_t)(uint32_t)p00 + (uint32_t)w.lo;
  const uint64_t Y = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10 + (w.lo >> 32) + (c >> 32);
  const uint64_t X = (p01 >> 32) + (p10 >> 32) + (uint32_t)p11 + w.hi;
  return fold((uint32_t)c, X + (p11 >> 32), X + Y);
}
#else
QTC_INLINE uint64_t mul(uint64_t a, uint64_t b) {
  uint64_t lo, hi;
  mul128(a, b, lo, hi);
  return reduce128(lo, hi);
}

QTC_INLINE uint64_t sqr(uint64_t a) {
  uint64_t lo, hi;
  sqr128(a, lo, hi);
  return reduce128(lo, hi);
}

// a d + w, for a constant d below 2^64 - 2^59 (every DIAG is) and w below 2^69:
// the 128-bit sum cannot overflow.
QTC_INLINE uint64_t mul_add(uint64_t a, uint64_t d, const Wide &w) {
  uint64_t lo, hi;
  mul128(a, d, lo, hi);
  lo += w.lo;
  hi += (uint64_t)w.hi + (uint64_t)(lo < w.lo);
  return reduce128(lo, hi);
}
#endif

// x^7, three multiplies deep.
QTC_INLINE uint64_t sbox(uint64_t x) {
  const uint64_t x2 = sqr(x);
  const uint64_t x3 = mul(x2, x);
  const uint64_t x4 = sqr(x2);
  return mul(x3, x4);
}

// ── Lazy values (QTC_LAZY 1) ────────────────────────────────────────────────
//
// c - N + S 2^32 (mod p), each part a u64 below 2^62: a product before its
// fold, or a sum of them. The linear layers are linear in the three parts, so
// an S-box's last product goes through them unfolded and each output is folded
// once, where QTC_LAZY 0 folds the S-box output and then the layer's sum again.
// Sums of 32-bit halves in u64s carry for free, where a 64-bit sum needs a
// compare for its carry.
//
// QTC_LAZY 1 carries the S-box outputs this way; it spilled 4.9 KB of
// registers on an A750 (three u64s a value, 12 values twice). QTC_LAZY 2
// (default) folds the S-box outputs and keeps only the sums lazy (Hs below
// and the internal layer's sum); QTC_LAZY 0 is the first form (Wide sums).
#ifndef QTC_LAZY
#define QTC_LAZY 2
#endif

struct Lz {
  uint64_t c, N, S;
};

QTC_INLINE Lz lz(uint64_t v) { return Lz{(uint32_t)v, 0, v >> 32}; }
QTC_INLINE Lz operator+(const Lz &a, const Lz &b) { return Lz{a.c + b.c, a.N + b.N, a.S + b.S}; }
QTC_INLINE Lz lz_rc(const Lz &a, uint64_t rc) { return Lz{a.c + (uint32_t)rc, a.N, a.S + (rc >> 32)}; }
QTC_INLINE uint64_t fold_lz(const Lz &z) { return fold((uint32_t)z.c, z.N, z.S + (z.c >> 32)); }

// a b, unfolded: c = l00 (below 2^32), N = X + h11 (below 2^34), S = X + Y (below 2^35).
QTC_INLINE Lz mul_lz(uint64_t a, uint64_t b) {
  const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32), b0 = (uint32_t)b, b1 = (uint32_t)(b >> 32);
  const uint64_t p00 = (uint64_t)a0 * b0, p01 = (uint64_t)a0 * b1, p10 = (uint64_t)a1 * b0, p11 = (uint64_t)a1 * b1;
  const uint64_t Y = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
  const uint64_t X = (p01 >> 32) + (p10 >> 32) + (uint32_t)p11;
  return Lz{(uint32_t)p00, X + (p11 >> 32), X + Y};
}

// a d + w for a lazy w (parts below 2^40): the internal layer's step.
QTC_INLINE uint64_t mul_add_lz(uint64_t a, uint64_t d, const Lz &w) {
  const Lz p = mul_lz(a, d);
  return fold_lz(Lz{p.c + w.c, p.N + w.N, p.S + w.S});
}

// x^7 with its last product unfolded.
QTC_INLINE Lz sbox_lz(uint64_t x) {
  const uint64_t x2 = sqr(x);
  const uint64_t x3 = mul(x2, x);
  const uint64_t x4 = sqr(x2);
  return mul_lz(x3, x4);
}

// The external layer on lazy values, rc added: out = M_E in + rc. Weights add
// to 28 at most, so one layer takes parts below 2^35 to below 2^40, and two
// layers in a row stay below 2^46.
QTC_INLINE void ext_lz(const Lz in[12], Lz out[12], const uint64_t rc[12]) {
  Lz y[12];
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const int o = 4 * g;
    const Lz x0 = in[o], x1 = in[o + 1], x2 = in[o + 2], x3 = in[o + 3];
    const Lz t01 = x0 + x1, t23 = x2 + x3, t0123 = t01 + t23;
    const Lz t01123 = t0123 + x1, t01233 = t0123 + x3;
    y[o] = t01123 + t01;
    y[o + 1] = t01123 + x2 + x2;
    y[o + 2] = t01233 + t23;
    y[o + 3] = t01233 + x0 + x0;
  }
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const Lz sum = y[k] + y[k + 4] + y[k + 8];
#pragma unroll
    for (int j = k; j < 12; j += 4) out[j] = lz_rc(y[j] + sum, rc[j]);
  }
}

// Its first output alone: 2 y0 + y4 + y8.
QTC_INLINE Lz ext_lz_first(const Lz in[12]) {
  Lz w = Lz{0, 0, 0};
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const int o = 4 * g;
    const Lz y = in[o] + in[o] + in[o + 1] + in[o + 1] + in[o + 1] + in[o + 2] + in[o + 3];
    w = w + y;
    if (g == 0) w = w + y;
  }
  return w;
}

// ── Half sums (QTC_LAZY 2) ──────────────────────────────────────────────────
//
// L + H 2^32 with L the sum of 32-bit low halves and H of high halves, each in
// a u64: adding a u64 is two adds of a 32-bit value into a u64, which carry
// for free (no compare). Folded with fold(L_lo, 0, H + L_hi).
struct Hs {
  uint64_t L, H;
};

QTC_INLINE Hs hs(uint64_t v) { return Hs{(uint32_t)v, v >> 32}; }
QTC_INLINE Hs operator+(const Hs &a, const Hs &b) { return Hs{a.L + b.L, a.H + b.H}; }
QTC_INLINE Hs operator+(const Hs &a, uint64_t v) { return Hs{a.L + (uint32_t)v, a.H + (v >> 32)}; }
QTC_INLINE uint64_t fold_hs(const Hs &z) { return fold((uint32_t)z.L, 0, z.H + (z.L >> 32)); }

// M_4 on four half sums.
QTC_INLINE void m4_hs(const Hs x[4], Hs y[4]) {
  const Hs t01 = x[0] + x[1], t23 = x[2] + x[3], t0123 = t01 + t23;
  const Hs t01123 = t0123 + x[1], t01233 = t0123 + x[3];
  y[0] = t01123 + t01;
  y[1] = t01123 + x[2] + x[2];
  y[2] = t01233 + t23;
  y[3] = t01233 + x[0] + x[0];
}

// The external layer as M_4 (x_g + T) for each group g, T the sum of the three
// groups: the column sums go in first, so no group's M_4 output waits for the
// others (fewer live values: the 12 Wide outputs of QTC_LAZY 0 spilled). The
// inputs are u64 or half sums; the outputs half sums (rc added), unfolded.
template <typename In>
QTC_INLINE void ext_hs(const In s[12], Hs out[12], const uint64_t rc[12]) {
  Hs T[4];
#pragma unroll
  for (int k = 0; k < 4; k++) T[k] = Hs{0, 0} + s[k] + s[k + 4] + s[k + 8];
#pragma unroll
  for (int g = 0; g < 3; g++) {
    Hs z[4], y[4];
#pragma unroll
    for (int k = 0; k < 4; k++) z[k] = T[k] + s[4 * g + k];
    m4_hs(z, y);
#pragma unroll
    for (int k = 0; k < 4; k++) out[4 * g + k] = y[k] + rc[4 * g + k];
  }
}

// Its first output alone: M_4 (x_0 + T) row 0 = 2 z0 + 3 z1 + z2 + z3.
QTC_INLINE uint64_t ext_hs_first(const uint64_t s[12]) {
  Hs z[4];
#pragma unroll
  for (int k = 0; k < 4; k++) z[k] = Hs{0, 0} + s[k] + s[k] + s[k + 4] + s[k + 8];
  return fold_hs(z[0] + z[0] + z[1] + z[1] + z[1] + z[2] + z[3]);
}

// ── Linear layers ───────────────────────────────────────────────────────────

// M_4 = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]] on each group of four, then
// every element gets the sum of its column over the three groups, then rc.
QTC_INLINE void ext_layer(uint64_t s[12], const uint64_t rc[12]) {
#if QTC_LAZY == 2
  Hs h[12];
  ext_hs(s, h, rc);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = fold_hs(h[i]);
  return;
#elif QTC_LAZY == 1
  Lz in[12], out[12];
#pragma unroll
  for (int i = 0; i < 12; i++) in[i] = lz(s[i]);
  ext_lz(in, out, rc);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = fold_lz(out[i]);
  return;
#endif
  Wide y[12];
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const int o = 4 * g;
    const uint64_t x0 = s[o], x1 = s[o + 1], x2 = s[o + 2], x3 = s[o + 3];
    Wide t01 = wide(x0);
    wadd(t01, x1);
    Wide t23 = wide(x2);
    wadd(t23, x3);
    Wide t0123 = t01;
    waddw(t0123, t23);
    Wide t01123 = t0123;
    wadd(t01123, x1);
    Wide t01233 = t0123;
    wadd(t01233, x3);
    y[o] = t01123;
    waddw(y[o], t01);
    y[o + 1] = t01123;
    wadd(y[o + 1], x2);
    wadd(y[o + 1], x2);
    y[o + 2] = t01233;
    waddw(y[o + 2], t23);
    y[o + 3] = t01233;
    wadd(y[o + 3], x0);
    wadd(y[o + 3], x0);
  }
#pragma unroll
  for (int k = 0; k < 4; k++) {
    Wide sum = y[k];
    waddw(sum, y[k + 4]);
    waddw(sum, y[k + 8]);
#pragma unroll
    for (int j = k; j < 12; j += 4) {
      Wide w = y[j];
      waddw(w, sum);
      wadd(w, rc[j]);
      s[j] = wred(w);
    }
  }
}

// The same layer when only its first output is needed (the last layer of the
// hash): 2 y0 + y4 + y8.
QTC_INLINE uint64_t ext_layer_first(const uint64_t s[12]) {
  Wide w = wide(0);
#pragma unroll
  for (int g = 0; g < 3; g++) {
    const int o = 4 * g;
    // y_o = 2 x0 + 3 x1 + x2 + x3, twice for group 0.
    const int times = g == 0 ? 2 : 1;
#pragma unroll
    for (int t = 0; t < times; t++) {
      wadd(w, s[o]);
      wadd(w, s[o]);
      wadd(w, s[o + 1]);
      wadd(w, s[o + 1]);
      wadd(w, s[o + 1]);
      wadd(w, s[o + 2]);
      wadd(w, s[o + 3]);
    }
  }
  return wred(w);
}

// s[i] = sum(s) + s[i] DIAG[i], then rc0 into element 0.
QTC_INLINE void int_layer(uint64_t s[12], uint64_t rc0) {
#if QTC_LAZY
  Lz lsum = lz(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) lsum = lsum + lz(s[i]);
  s[0] = mul_add_lz(s[0], DIAG[0], lz_rc(lsum, rc0));
#pragma unroll
  for (int i = 1; i < 12; i++) s[i] = mul_add_lz(s[i], DIAG[i], lsum);
  return;
#endif
  Wide sum = wide(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) wadd(sum, s[i]);
  Wide s0 = sum;
  wadd(s0, rc0);
  s[0] = mul_add(s[0], DIAG[0], s0);
#pragma unroll
  for (int i = 1; i < 12; i++) s[i] = mul_add(s[i], DIAG[i], sum);
}

// The last partial round: the first terminal round's constants into every element.
QTC_INLINE void int_layer_last(uint64_t s[12]) {
#if QTC_LAZY
  Lz lsum = lz(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) lsum = lsum + lz(s[i]);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = mul_add_lz(s[i], DIAG[i], lz_rc(lsum, RC_LAST_PARTIAL[i]));
  return;
#endif
  Wide sum = wide(s[0]);
#pragma unroll
  for (int i = 1; i < 12; i++) wadd(sum, s[i]);
#pragma unroll
  for (int i = 0; i < 12; i++) {
    Wide w = sum;
    wadd(w, RC_LAST_PARTIAL[i]);
    s[i] = mul_add(s[i], DIAG[i], w);
  }
}

// ── The permutation ─────────────────────────────────────────────────────────

QTC_INLINE void full_round(uint64_t s[12], const uint64_t rc[12]) {
#if QTC_LAZY == 1
  Lz t[12], u[12];
#pragma unroll
  for (int i = 0; i < 12; i++) t[i] = sbox_lz(s[i]);
  ext_lz(t, u, rc);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = fold_lz(u[i]);
#else
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
  ext_layer(s, rc);
#endif
}

// Every round but the last: s holds M s_in + RC_FIRST on entry.
QTC_INLINE void permute_rounds(uint64_t s[12]) {
  QTC_FULL_LOOP
  for (int r = 0; r < 4; r++) full_round(s, RC_FULL[r]);
  QTC_PARTIAL_LOOP
  for (int r = 0; r < 21; r++) {
    s[0] = sbox(s[0]);
    int_layer(s, RC_PARTIAL[r]);
  }
  s[0] = sbox(s[0]);
  int_layer_last(s);
  QTC_FULL_LOOP
  for (int r = 4; r < 7; r++) full_round(s, RC_FULL[r]);
}

// The last round when only the permutation's element 0 is needed.
QTC_INLINE uint64_t last_round_first(uint64_t s[12]) {
#if QTC_LAZY == 1
  Lz t[12];
#pragma unroll
  for (int i = 0; i < 12; i++) t[i] = sbox_lz(s[i]);
  return fold_lz(ext_lz_first(t));
#else
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
#if QTC_LAZY == 2
  return ext_hs_first(s);
#else
  return ext_layer_first(s);
#endif
#endif
}

// Everything after the first linear layer. When first_only, the last linear
// layer gives only element 0, which is returned (s is then not the output).
template <bool first_only>
QTC_INLINE uint64_t permute_rest(uint64_t s[12]) {
  permute_rounds(s);
  if (first_only) return last_round_first(s);
  full_round(s, RC_FULL[7]);
  return s[0];
}

// The sponge's step between the two permutations of a nonce: the terminator
// and the pad (1, 1) into elements 0 and 1, as the first permutation's last
// linear layer's constant.
constexpr uint64_t RC_PAD[12] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// The first permutation's last round with the pad, then the second
// permutation's first linear layer: two linear layers in a row, folded once
// (QTC_LAZY 1 and 2).
QTC_INLINE void between(uint64_t s[12]) {
#if QTC_LAZY == 1
  Lz t[12], u[12], v[12];
#pragma unroll
  for (int i = 0; i < 12; i++) t[i] = sbox_lz(s[i]);
  ext_lz(t, u, RC_PAD);
  ext_lz(u, v, RC_FIRST);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = fold_lz(v[i]);
#else
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
#if QTC_LAZY == 2
  Hs u[12], v[12];
  ext_hs(s, u, RC_PAD);
  ext_hs(u, v, RC_FIRST);
#pragma unroll
  for (int i = 0; i < 12; i++) s[i] = fold_hs(v[i]);
#else
  ext_layer(s, RC_PAD);
  ext_layer(s, RC_FIRST);
#endif
#endif
}

// QTC_SHARE 1: one copy of the rounds' code for both permutations of a nonce
// (a loop of two passes), so the kernel is half the size. It lost on an A750:
// 11.7 MH/s against 85.2, the card at 99 W with its EUs mostly idle. Kept to
// try again, off.
#ifndef QTC_SHARE
#define QTC_SHARE 0
#endif

// The whole permutation (host side: the midstate and the self-test).
QTC_INLINE void permute(uint64_t s[12]) {
  ext_layer(s, RC_FIRST);
  permute_rest<false>(s);
}

// ── One nonce ───────────────────────────────────────────────────────────────

QTC_INLINE uint32_t bswap32(uint32_t v) {
  return (v << 24) | ((v & 0xff00u) << 8) | ((v >> 8) & 0xff00u) | (v >> 24);
}

QTC_INLINE uint64_t bswap64(uint64_t v) {
  return ((uint64_t)bswap32((uint32_t)v) << 32) | bswap32((uint32_t)(v >> 32));
}

// The scan's prestate: the midstate with nonce bytes 32..55 (the tail, six
// little-endian words) added, through the first linear layer and RC_FIRST. The
// counter's two words are added after the layer (it is linear): counter_part.
QTC_INLINE void prestate(const uint64_t mid[12], const uint32_t tail[6], uint64_t pre[12]) {
#pragma unroll
  for (int i = 0; i < 12; i++) pre[i] = mid[i];
#pragma unroll
  for (int i = 0; i < 6; i++) pre[i] = add(pre[i], tail[i]);
  ext_layer(pre, RC_FIRST);
#pragma unroll
  for (int i = 0; i < 12; i++) pre[i] = canonical(pre[i]);
}

// The first linear layer of words 6 and 7 alone ([0, 0, w6, w7] in group 1):
// M_4 gives c = (w6 + w7, 3 w6 + w7, 2 w6 + 3 w7, w6 + 2 w7) in group 1 and zero
// elsewhere; the column sums then add c to every group, so group 1 holds 2c.
QTC_INLINE void add_counter(uint64_t s[12], const uint64_t pre[12], uint64_t counter) {
  const uint64_t w6 = bswap32((uint32_t)(counter >> 32));
  const uint64_t w7 = bswap32((uint32_t)counter);
  const uint64_t c[4] = {w6 + w7, 3 * w6 + w7, 2 * w6 + 3 * w7, w6 + 2 * w7};
#pragma unroll
  for (int k = 0; k < 4; k++) {
    s[k] = add(pre[k], c[k]);
    s[k + 4] = add(pre[k + 4], 2 * c[k]);
    s[k + 8] = add(pre[k + 8], c[k]);
  }
}

// The hash's top 64 bits for one counter: qpow.nonceTop64 in
// earn/src/shared/quantus/qpow.js.
QTC_INLINE uint64_t nonce_top(const uint64_t pre[12], uint64_t counter) {
  uint64_t s[12];
  add_counter(s, pre, counter);
  uint64_t s0 = 0;
#if QTC_SHARE
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    permute_rounds(s);
    if (pass == 0) between(s);
    else s0 = last_round_first(s);
  }
#else
  permute_rounds(s);
  between(s);
  permute_rounds(s);
  s0 = last_round_first(s);
#endif
  return bswap64(canonical(s0));
}

}  // namespace qtc
