// cfold_h.cuh -- the copy fold as a probe: hand89/fold_h.cuh's staged fold with the draw taken out. A' and B' are
// ready in global memory, already noised, in the stage's own byte layout, and every warp copies its part of a later
// k-step's stage while it folds this one: one 16-byte global load and one 16-byte shared store a lane a unit. The
// stage, the ldmatrix.trans fragment loads, the barrier a k-step and the readout are the staged draw fold's.
// Branch goal/nv-kernel, 11 November 2026 (seventh round).
//
// Why it exists: to price a copy unit by where its source sits (the SM's cache, the L2, DRAM) in a loop that is
// otherwise the same, and to price what an order written by hand buys a copy fold. The release fold copies the
// same bytes with LDGSTS (3.5 units a warp a k-step on a 192 x 256 block). This file is the reference: the chunk
// loop is in a function of its own that is never inlined, so hand89/cfold.py can replace that function's code in
// the cubin with a loop written by hand, and hand89/cprobe.cu checks either against a CPU fold.
//
// Layouts (cprobe.cu has the writer of each):
//   B' in global memory, a column block after the other: [block][k-step][k-pair j 0..15][group G 0..31] of 16
//     bytes, 8,192 bytes a k-step. Byte 2c of a word is column 8G + c at the pair's first k (k-step * 32 + 2j),
//     byte 2c + 1 at its second. CP_PADK k-steps of zeros behind a block's last.
//   A' the same with 4 CP_MT groups of 8 rows: [row group][k-step][j][G] of 16 bytes, 256 * 4 CP_MT a k-step.
//   Stage, in shared memory: a ring of four k-steps each of B and A. K-pair j's word of group G is at
//     j * CP_SJ + 16 G of its buffer; CP_SJ is 16 bytes more than the groups take, so the eight rows of an
//     ldmatrix matrix (one group, eight k-pairs) fall in 32 different banks. Stage X is in buffer X & 3.
//   Copy units, a warp a k-step: two of B (k-pairs W and W + 8, groups 0-15 and 16-31), one of A (the same
//     k-pairs, groups 0-15), and in warps 0 to 3 a second of A (groups 16 up: a quarter warp takes one k-pair
//     and its groups from 16, so a quarter's eight lanes read one 128-byte line of the record and no line is
//     asked for twice). That is 3.5 units a warp for a 160 x 256 block (with five tiles a warp half of the
//     second A unit's lanes repeat others) and for a 192 x 256 block (none repeat).
//   Regions and transcripts: as fold_h.cuh (2 CP_MT row offsets by 16 column offsets a tile, 64 bytes each).
//
// K-step s stores stage s + 2: nothing reads that buffer before the barrier at the k-step's end. B's fragments
// and A's first tile of k-step s + 1 are loaded at the end of k-step s, behind the barrier that ended k-step
// s - 1, by which every warp had stored its part of stage s + 1.

#ifndef CP_FN
#error "cfold_h.cuh: define CP_FN before including it"
#endif
#ifndef CP_MT
#define CP_MT 5
#endif
#if CP_MT != 5 && CP_MT != 6
#error "CP_MT is 5 m16 tiles a warp (160 x 256) or 6 (192 x 256)"
#endif
// Instructions of ballast in the loop function (never run): room and registers for the loop written by hand.
#ifndef CP_BALLAST
#define CP_BALLAST 3
#endif
#ifndef CP_BQ
#define CP_BQ 61
#endif

#define CP_GA (4u * CP_MT)                          // A's groups of 8 rows
#define CP_SJB (512u + 16u)                         // B's stage bytes a k-pair
#define CP_SJA (16u * CP_GA + 16u)                  // A's
#define CP_STB (16u * CP_SJB)                       // one buffer of B's stage
#define CP_STA (16u * CP_SJA)                       // one of A's
#define CP_RING 4u
#define CP_STGB 0u
#define CP_STGA (CP_RING * CP_STB)
#define CP_TR (CP_STGA + CP_RING * CP_STA)          // transcripts, [region][16 words]
#define CP_CNT (CP_TR + CP_MT * 2048u)              // the probe's stamps
#define CP_PTR (CP_CNT + 192u)                      // the tile's two source pointers, 8 bytes each
#define CP_SCR (CP_CNT + 512u)                       // a buffer of B and one of A that nothing reads (cfold.py cp=2)
#define CP_SMEM (CP_SCR + CP_STB + CP_STA)
#define CP_THREADS 256u
#define CP_NI (8u * CP_MT)                          // mma a k-step
#define CP_RECB 8192u                               // B' bytes a k-step
#define CP_RECA (256u * CP_GA)                      // A' bytes a k-step
#define CP_PADK 8u                                  // k-steps of zeros behind an operand's last

#ifndef CP_HELPERS
#define CP_HELPERS
template <unsigned V>
struct CpC {
  static constexpr unsigned v = V;
};
#endif

#define CP_CAT2(a, b) a##b
#define CP_CAT(a, b) CP_CAT2(a, b)
#define CP_LOOPFN CP_CAT(CP_FN, _loop)
#define CP_PREFN CP_CAT(CP_FN, _pre)
#define CP_POSTFN CP_CAT(CP_FN, _post)

// What the kernel and its loop function both begin with: the lane's constants and the small pieces.
#define CP_COMMON                                                                                              \
  constexpr uint32_t MT = CP_MT, NB = 8u, NI = CP_NI;                                                          \
  constexpr uint32_t RO = 2u * MT;            /* valid row offsets a tile */                                   \
  constexpr uint32_t RPS = 4u * RO;           /* regions a column slot */                                      \
  extern __shared__ __align__(128) uint32_t cp_smem[];                                                         \
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(cp_smem);                                          \
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;                                            \
  const uint32_t hL = lane >> 4, l16 = lane & 15u;                                                             \
  /* ldmatrix rows, as fold_h.cuh: lane bits 0-2 are the k-pair of a half; for A bit 3 is the group of the  */ \
  /* m16 tile's two and bit 4 the k half; for B bit 3 is the k half and bit 4 the n8 tile of the pair.      */ \
  uint32_t fragBL = sbase + CP_STGB + (((lane >> 3) & 1u) * 8u + (lane & 7u)) * CP_SJB + (lane >> 4) * 16u     \
                    + (warp & 3u) * 128u;                                                                      \
  uint32_t aFL = sbase + CP_STGA + ((lane >> 4) * 8u + (lane & 7u)) * CP_SJA + ((lane >> 3) & 1u) * 16u        \
                 + (warp >> 2) * (MT * 32u);                                                                   \
  /* The copy units' words in a buffer and in a k-step's record. */                                            \
  const uint32_t jU = warp + hL * 8u;                                                                          \
  const uint32_t g2 = 16u + (lane & 7u) % (CP_GA - 16u);                                                       \
  const uint32_t j2 = 4u * (warp & 3u) + (lane >> 3);                                                          \
  uint32_t dstBL = sbase + CP_STGB + jU * CP_SJB + 16u * l16;                                                  \
  uint32_t dstA1 = sbase + CP_STGA + jU * CP_SJA + 16u * l16;                                                  \
  uint32_t dstA2 = sbase + CP_STGA + j2 * CP_SJA + 16u * g2;                                                   \
  const uint32_t wB = jU * 32u + l16, wA1 = jU * CP_GA + l16, wA2 = j2 * CP_GA + g2;                           \
  uint32_t trL = sbase + CP_TR + ((warp & 3u) * RPS + (lane & 3u) * RO + ((lane >> 4) & 1u)) * 64u;            \
  asm volatile("" : "+r"(fragBL), "+r"(aFL), "+r"(dstBL), "+r"(dstA1), "+r"(dstA2), "+r"(trL));                \
  const uint32_t sCnt = sbase + CP_CNT;                                                                        \
  auto ld4 = [](uint32_t r[4], uint32_t a) {                                                                   \
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"                                                       \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");                        \
  };                                                                                                           \
  auto st4 = [](uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {                              \
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3)          \
                 : "memory");                                                                                  \
  };                                                                                                           \
  auto ldg4 = [](uint32_t r[4], const uint4 *p) {                                                              \
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"                                                       \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p) : "memory");                        \
  };                                                                                                           \
  auto ldmt = [](uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {                         \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"                         \
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");                                \
  };                                                                                                           \
  auto xor3 = [](uint32_t a, uint32_t b, uint32_t c) -> uint32_t {                                             \
    uint32_t d;                                                                                                \
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));                                  \
    return d;                                                                                                  \
  };                                                                                                           \
  /* One warp's copy units of stage ts (0 .. 3 of the ring: its buffer) from k-step record ks. */              \
  auto copy_stage = [&](const uint4 *pA_, const uint4 *pB_, uint32_t ks, uint32_t buf) {                       \
    uint32_t q[4];                                                                                             \
    ldg4(q, pB_ + (size_t)ks * (CP_RECB / 16u) + wB);                                                          \
    st4(dstBL + buf * CP_STB, q[0], q[1], q[2], q[3]);                                                         \
    ldg4(q, pB_ + (size_t)ks * (CP_RECB / 16u) + wB + 16u);                                                    \
    st4(dstBL + buf * CP_STB + 256u, q[0], q[1], q[2], q[3]);                                                  \
    ldg4(q, pA_ + (size_t)ks * (CP_RECA / 16u) + wA1);                                                         \
    st4(dstA1 + buf * CP_STA, q[0], q[1], q[2], q[3]);                                                         \
    if (warp < 4u) {                                                                                           \
      ldg4(q, pA_ + (size_t)ks * (CP_RECA / 16u) + wA2);                                                       \
      st4(dstA2 + buf * CP_STA, q[0], q[1], q[2], q[3]);                                                       \
    }                                                                                                          \
  };

// The tile's fold: zero sums, the first fragments, the chunk loop, the last readouts.
static __device__ __noinline__ void CP_LOOPFN(uint32_t chunks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  CP_COMMON
  // The tile's two source pointers, left in shared memory by the tile's prologue (so the kernel keeps nothing
  // in registers through this call, and the loop written by hand may use them all).
  const uint4 *pA, *pB;
  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pA) : "r"(sbase + CP_PTR) : "memory");
  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pB) : "r"(sbase + CP_PTR + 8u) : "memory");
#if CP_BALLAST
  // Never run (chunks has no top bit): CP_BQ quads alive at once, so the function writes every register it may,
  // and room for a loop longer than the one ptxas makes below.
  if (chunks >> 31) {
#pragma unroll
    for (uint32_t rep = 0; rep < (uint32_t)CP_BALLAST; rep++) {
      uint32_t z[CP_BQ][4], x = 0u;
#pragma unroll
      for (uint32_t i = 0; i < (uint32_t)CP_BQ; i++) ldg4(z[i], pA + i * 7u + rep);
      __syncthreads();   // every load on this side of it, every use on the other: all the quads are alive here
#pragma unroll
      for (uint32_t i = 0; i < (uint32_t)CP_BQ; i++) x = xor3(x, z[i][0] + z[i][1], z[i][2] ^ z[i][3]);
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(sCnt + 128u), "r"(x) : "memory");
    }
  }
#endif
  int32_t acc[MT][NB][4];
  uint32_t af[2][4];        // A's fragment of the m16 tile being folded, and the next one's
  uint32_t bf[NB][2];       // B's eight fragments of the k-step
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  auto fold_tile = [&](uint32_t t) -> uint32_t {
    uint32_t x = (uint32_t)acc[t][0][0] ^ (uint32_t)acc[t][0][1];
    x = xor3(x, (uint32_t)acc[t][0][2], (uint32_t)acc[t][0][3]);
#pragma unroll
    for (uint32_t nb = 1; nb < NB; nb++) {
      x = xor3(x, (uint32_t)acc[t][nb][0], (uint32_t)acc[t][nb][1]);
      x = xor3(x, (uint32_t)acc[t][nb][2], (uint32_t)acc[t][nb][3]);
    }
    return x;
  };
  // Rotated once for every later chunk of the same word, then XORed into the region's word.
  auto fold_end = [&](uint32_t x, uint32_t tr, uint32_t q) {
    const uint32_t later = (chunks >> 4) - 1u - (q >> 4);
    x = __funnelshift_l(x, x, (13u * later) & 31u);
    asm volatile("red.shared.xor.b32 [%0], %1;" ::"r"(tr + (q & 15u) * 4u), "r"(x) : "memory");
  };
  // The first k-step's fragments, from stage 0 (the kernel copied stages 0 to 2 before the call).
  ldmt(af[0][0], af[0][1], af[0][2], af[0][3], aFL);
#pragma unroll
  for (uint32_t p = 0; p < NB / 2u; p++)
    ldmt(bf[2u * p][0], bf[2u * p][1], bf[2u * p + 1u][0], bf[2u * p + 1u][1], fragBL + p * 32u);
  uint32_t c = 0u;
  auto slot = [&](auto cW, auto cS, auto cM, auto cN) {
    constexpr uint32_t W = decltype(cW)::v, s = decltype(cS)::v, mb = decltype(cM)::v, nb = decltype(cN)::v;
    constexpr uint32_t wr = W >> 2;
    constexpr uint32_t i = mb * NB + nb;
    constexpr uint32_t as = (mb + (MT & 1u) * s) & 1u, as0n = ((MT & 1u) * (s + 1u)) & 1u;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
        : "r"(af[as][0]), "r"(af[as][1]), "r"(af[as][2]), "r"(af[as][3]), "r"(bf[nb][0]),
          "r"(bf[nb][1]));
    // The copy of stage s + 2, a unit behind each of four mma.
    if constexpr (i == 2u) {
      uint32_t q[4];
      ldg4(q, pB + (size_t)(4u * c + s + 2u) * (CP_RECB / 16u) + wB);
      st4(dstBL + ((s + 2u) & 3u) * CP_STB, q[0], q[1], q[2], q[3]);
    }
    if constexpr (i == 10u) {
      uint32_t q[4];
      ldg4(q, pB + (size_t)(4u * c + s + 2u) * (CP_RECB / 16u) + wB + 16u);
      st4(dstBL + ((s + 2u) & 3u) * CP_STB + 256u, q[0], q[1], q[2], q[3]);
    }
    if constexpr (i == 18u) {
      uint32_t q[4];
      ldg4(q, pA + (size_t)(4u * c + s + 2u) * (CP_RECA / 16u) + wA1);
      st4(dstA1 + ((s + 2u) & 3u) * CP_STA, q[0], q[1], q[2], q[3]);
    }
    if constexpr (i == 26u && W < 4u) {
      uint32_t q[4];
      ldg4(q, pA + (size_t)(4u * c + s + 2u) * (CP_RECA / 16u) + wA2);
      st4(dstA2 + ((s + 2u) & 3u) * CP_STA, q[0], q[1], q[2], q[3]);
    }
    // A's next m16 tile of this k-step, into the other fragment set, eight mma ahead.
    if constexpr (nb == 0u && mb + 1u < MT)
      ldmt(af[as ^ 1u][0], af[as ^ 1u][1], af[as ^ 1u][2], af[as ^ 1u][3], aFL + (s & 3u) * CP_STA + (mb + 1u) * 32u);
    // The next k-step's fragments: A's first tile, then B's pairs, each behind the last mma that reads the
    // pair it replaces.
    if constexpr (i == NI - 8u)
      ldmt(af[as0n][0], af[as0n][1], af[as0n][2], af[as0n][3], aFL + ((s + 1u) & 3u) * CP_STA);
#define CP_BLOAD(PB)                                                                       \
    if constexpr (i == NI - 7u + 2u * PB)                                                  \
      ldmt(bf[2u * PB][0], bf[2u * PB][1], bf[2u * PB + 1u][0], bf[2u * PB + 1u][1],       \
           fragBL + ((s + 1u) & 3u) * CP_STB + PB * 32u);
    CP_BLOAD(0u)
    CP_BLOAD(1u)
    CP_BLOAD(2u)
    CP_BLOAD(3u)
#undef CP_BLOAD
    // Every warp meets here, at each k-step's end.
    if constexpr (i == NI - 1u) __syncthreads();
    // The readout: each m16 tile's chain, in the chunk's last k-step, a few mma behind the tile's last.
    if constexpr (s == 3u && i == NI - 1u) {
#pragma unroll
      for (uint32_t t = 0; t < MT; t++) fold_end(fold_tile(t), trL + 2u * ((wr * MT + t) >> 1) * 64u, c);
    }
  };
#define CP_SLOT(W, S, M, N) slot(CpC<W>{}, CpC<S>{}, CpC<M>{}, CpC<N>{});
#if CP_MT > 5
#define CP_TILE5(W, S) CP_TILE(W, S, 5)
#else
#define CP_TILE5(W, S)
#endif
#define CP_TILE(W, S, M)                                                                   \
  CP_SLOT(W, S, M, 0) CP_SLOT(W, S, M, 1) CP_SLOT(W, S, M, 2) CP_SLOT(W, S, M, 3)          \
  CP_SLOT(W, S, M, 4) CP_SLOT(W, S, M, 5) CP_SLOT(W, S, M, 6) CP_SLOT(W, S, M, 7)
#define CP_KSTEP(W, S)                                                                     \
  CP_TILE(W, S, 0) CP_TILE(W, S, 1) CP_TILE(W, S, 2) CP_TILE(W, S, 3) CP_TILE(W, S, 4) CP_TILE5(W, S)
#define CP_CHUNK(W) CP_KSTEP(W, 0) CP_KSTEP(W, 1) CP_KSTEP(W, 2) CP_KSTEP(W, 3)
#define CP_LOOP(W)                                                                         \
  case (W) >> 2:                                                                           \
    for (; c < chunks; c++) {                                                              \
      CP_CHUNK(W)                                                                          \
    }                                                                                      \
    break;
  switch (warp >> 2) {
    CP_LOOP(0)
    CP_LOOP(4)
    default:
      break;
  }
#undef CP_SLOT
#undef CP_LOOP
#undef CP_TILE
#undef CP_TILE5
#undef CP_KSTEP
#undef CP_CHUNK
#endif
}

// The tile's prologue: its operands' addresses (left in shared memory for the loop), and stages 0 to 3 (the
// whole ring: the loop needs 0 to 2, and a timing-only loop that copies nothing then folds real bytes throughout).
static __device__ __noinline__ void CP_PREFN(const uint4 *__restrict__ Ap, const uint4 *__restrict__ Bp,
                                             uint32_t chunks, uint32_t rg, uint32_t cb, uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  CP_COMMON
  const uint32_t ksteps_all = 4u * chunks + CP_PADK;
  const uint32_t rbg = v % rg, cbg = (v / rg) % cb;
  if (lane == 0u) {
    const long long pt0 = clock64();
    asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + warp * 24u), "l"(pt0) : "memory");
  }
  const uint4 *const pA = Ap + (size_t)rbg * ksteps_all * (CP_RECA / 16u);
  const uint4 *const pB = Bp + (size_t)cbg * ksteps_all * (CP_RECB / 16u);
  if (threadIdx.x == 0u) {
    asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + CP_PTR), "l"(pA) : "memory");
    asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + CP_PTR + 8u), "l"(pB) : "memory");
  }
#pragma unroll
  for (uint32_t ts = 0; ts < 4u; ts++) copy_stage(pA, pB, ts, ts);
  __syncthreads();
  if (lane == 0u) {
    const long long pt1 = clock64();
    asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 8u + warp * 24u), "l"(pt1) : "memory");
  }
#endif
}

// The tile's end: the regions out, the cycle counts.
static __device__ __noinline__ void CP_POSTFN(uint32_t v, unsigned long long *cyc, uint32_t *dump,
                                              uint32_t dump_tiles) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  CP_COMMON
  if (lane == 0u) {
    const long long pt2 = clock64();
    asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 16u + warp * 24u), "l"(pt2) : "memory");
  }
  // Hand-off: the column slot's two warps meet, so every word is in shared.
  asm volatile("bar.sync %0, %1;" ::"r"(1u + (warp & 3u)), "r"(64u) : "memory");
  if (warp < 4u) {
    // The probe: the slot's 8 MT regions go out as they are, then are zeroed.
#pragma unroll
    for (uint32_t i = 0; i < MT; i++) {
      const uint32_t o = warp * (RPS * 64u) + i * 512u + lane * 16u;
      uint32_t w[4];
      ld4(w, sbase + CP_TR + o);
      if (v < dump_tiles) {
        uint32_t *const d = dump + (size_t)v * (RPS * 64u) + o / 4u;
        d[0] = w[0];
        d[1] = w[1];
        d[2] = w[2];
        d[3] = w[3];
      }
      st4(sbase + CP_TR + o, 0u, 0u, 0u, 0u);
    }
  }
  if (lane == 0u) {
    // Cycles: the tile's prologue, its chunk loop, and its end, summed a warp; and the tiles.
    const long long pt3 = clock64();
    long long pt0, pt1, pt2;
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt0) : "r"(sCnt + warp * 24u) : "memory");
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt1) : "r"(sCnt + 8u + warp * 24u) : "memory");
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt2) : "r"(sCnt + 16u + warp * 24u) : "memory");
    atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 0u, (unsigned long long)(pt1 - pt0));
    atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 1u, (unsigned long long)(pt2 - pt1));
    atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 2u, (unsigned long long)(pt3 - pt2));
    atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 3u, 1ull);
  }
  __syncthreads();
#endif
}

extern "C" __global__ __launch_bounds__(CP_THREADS, 1) void CP_FN(
    const uint4 *__restrict__ Ap, const uint4 *__restrict__ Bp, uint32_t chunks, uint32_t rg, uint32_t cb,
    uint32_t tiles, unsigned long long *cyc, uint32_t *dump, uint32_t dump_tiles) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (blockDim.x != CP_THREADS) return;
  // Three calls a tile, and nothing but the tile's number kept between them.
  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    CP_PREFN(Ap, Bp, chunks, rg, cb, v);
    CP_LOOPFN(chunks);
    CP_POSTFN(v, cyc, dump, dump_tiles);
  }
#endif
}
