// The draw fold: eight warps over a tile of 160 rows by 256 columns, with both operands drawn
// on the SM instead of copied from memory. Branch goal/nv-kernel, 11 November 2026. This is
// the third design; the first two (tables by row group, a counter and a wait a k-step) are in
// the branch's history and probes/README.md says why they lost.
//
// A' and B' are never stored. A tile keeps, in shared memory, the dense noise factors of its
// rows and columns, and a few k-steps of the noised operands (the stage). ldmatrix.trans
// reads the stage; every warp draws a part of a later k-step's stage while it folds.
//
// What shapes it:
//   1. No table read has a bank conflict. The tables are laid out by rank index: one 16-byte
//      word holds sixteen rows at one rank index, and a rank index's words are side by side.
//      A lane reads the word at its own place, so the eight lanes of a quarter warp read 32
//      different banks whatever rank index each half of the warp is at.
//   2. Every warp draws the same: a k-step, one unit of B (two k-pairs for all 256 columns)
//      and one of A (two k-pairs for all 160 rows; six lanes of each half repeat others).
//   3. K-step s draws the stage of k-step s + 2. B's stage has two buffers and A's three.
//   4. The warps meet at a barrier at every k-step's end (PD_BAR 3, the default). That is
//      what makes the fold correct: checked against a CPU fold, 2,560 of 2,560 regions.
//
// It is correct and it is slow: 72 to 73% of the tensor pipe on an RTX 4090, the same as the
// second design. The design was meant to run with no warp waiting for another (PD_BAR 0 with
// the check below, or 2 with a wait a chunk). Those builds are faster and wrong: ptxas keeps
// memory instructions in the source's order around every store, but it puts the draw's stores
// anywhere in the k-step and moves the mma themselves, so nothing in the source holds a store
// late enough for a warp a few mma behind. probes/README.md, "The draw fold", has the figures.
//
// Included by probes/drawprobe.cu (cycle stamps and a transcript dump in place of the hash).
// Before including, define PD_FN (the kernel's name) and PD_PROBE (1 for the probe).
//
// Layouts the host shares (the probe has a writer of each):
//   table B  [rank index p][w 0..15] of 16 bytes, 32,768 bytes a tile: word (p, w) holds the
//            dense factor plus 32 of columns 8w + 0..7 (bytes 0-7) and 128 + 8w + 0..7.
//   table A  [p][w 0..9] of 16 bytes, 20,480 bytes a tile: rows 8w + 0..7 and 80 + 8w + 0..7.
//   offsets  [k-step][B, A][k-pair j 0..15] of four 32-bit words: the pair's first k's plus
//            and minus rank index, then its second k's, each times the table's bytes a rank
//            index (256 for B, 160 for A). Three k-steps more than the tile has, at the end.
//   stage    16 bytes for a group of 8 rows (or columns) and a k-pair: byte 2c is row c at
//            the pair's first k, byte 2c + 1 at its second. Group G of k-pair j is at
//            j * PD_SJ + 16 G in its buffer. PD_SJ is 16 bytes more than the groups take, so
//            the eight rows of an ldmatrix matrix (one group, eight k-pairs) fall in 32
//            different banks, and so do a quarter warp's stores (one k-pair, eight groups).
//   regions  a tile has 2 PD_MT valid row offsets by 16 valid column offsets. Column slot wc
//            (64 columns) holds 8 PD_MT of them: region 2 PD_MT tq + ro is column offset tq
//            (0..3) of the slot and row offset ro of the tile, 64 bytes of transcript each.

#ifndef PD_FN
#error "pearl_fold_draw.cuh: define PD_FN and PD_PROBE before including it"
#endif
#ifndef PD_MT
#define PD_MT 5
#endif
#if PD_MT != 5
#error "PD_MT is 5 m16 tiles a warp (with 6 the three A buffers do not fit in shared memory)"
#endif

// Shared memory, bytes from the block's base.
#define PD_WA (2u * PD_MT)                          // A's table words a rank index
#define PD_SA (16u * PD_WA)                         // A's table bytes a rank index
#define PD_SB 256u                                  // B's
#define PD_TABB 0u
#define PD_TABA 32768u
#define PD_SJB (512u + 16u)                         // B's stage bytes a k-pair
#define PD_SJA (2u * PD_SA + 16u)                   // A's
#define PD_STB (16u * PD_SJB)                       // one buffer of B's stage
#define PD_STA (16u * PD_SJA)                       // one of A's
#define PD_STGB (PD_TABA + 128u * PD_SA)            // B's two buffers
#define PD_STGA (PD_STGB + 2u * PD_STB)             // A's three
#define PD_TR (PD_STGA + 3u * PD_STA)               // transcripts, [region][16 words]
#define PD_CNT (PD_TR + PD_MT * 2048u)              // the check's four counters, the probe's stamps
#define PD_PIN (PD_CNT + 256u)                      // 256 bytes the pin stores go to
#define PD_SMEM (PD_PIN + 256u)
#define PD_THREADS 256u
#define PD_NI (8u * PD_MT)                          // mma a k-step

// Where the pieces go in a k-step, as the index of the mma they follow (0 .. PD_NI - 1).
// ptxas keeps memory instructions in the source's order around every store, and moves whole
// runs: these are the order, and only roughly the place.
// The draw of a k-step, in order: the pin and the check's load; A's unit; B's unit; the next
// k-step's fragment loads; the add. A unit is five steps, one mma apart: two table reads;
// the first k's subtracts and the other two reads; the second k's subtracts; the byte
// selects; the two stores. Warps 4 to 7 start PD_STAG mma later than warps 0 to 3, so the two
// warps of a scheduler are not both busy with it at once.
#ifndef PD_PIN_AT
#define PD_PIN_AT 10u
#endif
#ifndef PD_STAG
#define PD_STAG 8u
#endif
// 1: the pin's value passes through the sum of the mma it follows. Meant to keep ptxas from
// moving the pin (and with it the draw) ahead of that mma; it moves the mma instead.
#ifndef PD_PINDEP
#define PD_PINDEP 0
#endif
// A tile's readout is one chain of 16 XOR3 (ptxas weaves a chain between the mma, two or so
// to an mma; it clumps XOR3 that do not wait for each other). It starts this many mma after
// the tile's last mma of the chunk: a sum read soon after its mma is dear (measured).
#ifndef PD_RLAG
#define PD_RLAG 12u
#endif
// Warps 4 to 7 start theirs this many mma later still.
#ifndef PD_RSTAG
#define PD_RSTAG 0u
#endif
// Timing only: pieces off. PD_NO_DRAW folds what the prologue drew, over and over;
// PD_NO_READOUT reads out at the tile's end only; PD_NO_CHECK drops the add, the pin and
// the check's load.
#ifndef PD_NO_DRAW
#define PD_NO_DRAW 0
#endif
#ifndef PD_NO_READOUT
#define PD_NO_READOUT 0
#endif
#ifndef PD_NO_CHECK
#define PD_NO_CHECK (PD_BAR == 3)   // the barrier a k-step needs no check
#endif
// What holds the warps together, once a chunk. 0: nothing. 1: a barrier at each chunk's end
// (measured dear: a warp is put to sleep and woken). 2: a warp begins a chunk only when every
// warp has made the last add of the chunk before; one load and a test when they have.
// 3: a barrier at each k-step's end (correct by construction, with no check).
#ifndef PD_BAR
#define PD_BAR 3
#endif

// The fill of A and B the draw is written for: the table's bytes are the dense factor plus 32
// (0 .. 63), so fill + plus - minus never borrows between the bytes of a word when the fill is
// 63 or more, and 63 is the largest an int7 operand may hold. A' and B' are then 0 .. 126.
#ifndef PD_FILL
#define PD_FILL 63
#endif
#if PD_FILL != 63 && PD_FILL != 64
#error "PD_FILL is 63 or 64: under 63 a byte's subtract can borrow from the next"
#endif
#define PD_FILL4 ((uint32_t)PD_FILL * 0x01010101u)

#ifndef PD_HELPERS
#define PD_HELPERS
// constexpr, and callable from device code.
#ifdef __CUDACC__
#define PD_CE __host__ __device__ constexpr
#else
#define PD_CE constexpr
#endif
// A constant as a type: how the chunk loop's copies get the warp, the k-step and the mma.
template <unsigned V>
struct PdC {
  static constexpr unsigned v = V;
};
// m16 tile t's readout: whether it is made a chunk late (in the next chunk's first k-step, for
// want of room in the last), and the mma it follows there or in the chunk's last k-step. The
// chain wants about eight mma of room before the k-step's end (or before the tile's next mma).
PD_CE bool pd_tile_late(unsigned t, unsigned lag) { return 8u * t + 7u + lag + 8u > 8u * (unsigned)PD_MT; }
PD_CE unsigned pd_tile_first(unsigned t, unsigned lag) {
  const unsigned at = 8u * t + 7u + lag;
  return !pd_tile_late(t, lag) ? at : (at > 8u * (unsigned)PD_MT ? at - 8u * (unsigned)PD_MT : 0u);
}
#endif

extern "C" __global__ __launch_bounds__(PD_THREADS, 1) void PD_FN(
    const uint4 *__restrict__ tabA, const uint4 *__restrict__ tabB, const uint4 *__restrict__ offs,
    const int8_t *__restrict__ stamp, uint32_t stamp_bytes, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t tiles
#if PD_PROBE
    , unsigned long long *cyc, uint32_t *dump, uint32_t dump_tiles
#else
    , const PearlTranscriptTest test, const PearlHitList hits
#endif
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr uint32_t MT = PD_MT, NB = 8u, NI = PD_NI;
  constexpr uint32_t RO = 2u * MT;            // valid row offsets a tile
  constexpr uint32_t RPS = 4u * RO;           // regions a column slot
  if (blockDim.x != PD_THREADS) return;
  extern __shared__ __align__(128) uint32_t pd_smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(pd_smem);
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;

  const uint32_t row_groups = (rows_valid + RO - 1u) / RO;
  (void)col_groups;

  // ---- lane constants: the registers a thread keeps through the chunk loop ----
  // A lane's half of the warp (hL) is its k-pair of a unit; its place in the half (0 .. 15)
  // is its table word. A has ten words: places 10 to 15 repeat words 2 to 7, which keeps a
  // quarter warp's eight reads in different banks (words 8 and 9 sit where 0 and 1 do).
  const uint32_t hL = lane >> 4, l16 = lane & 15u;
  const uint32_t wA = l16 < PD_WA ? l16 : l16 - 8u;
  uint32_t tabBL = sbase + PD_TABB + 16u * l16;
  uint32_t tabAL = sbase + PD_TABA + 16u * wA;
  // The draw's stores: the lane's first group of its k-pair, less the unit's and the buffer's
  // part. A unit's k-pairs are W and W + 8.
  uint32_t dstBL = sbase + PD_STGB + (warp + hL * 8u) * PD_SJB + 16u * l16;
  uint32_t aD[3], aF[3];
  // ldmatrix rows. Lane bits 0-2 are the k-pair of a half. For A bit 3 is the group of the
  // m16 tile's two and bit 4 the k half; for B bit 3 is the k half and bit 4 the n8 tile of
  // the pair (the mma wants a tile's two k halves in adjacent registers for B, and its two
  // row groups for A).
  uint32_t fragBL = sbase + PD_STGB + (((lane >> 3) & 1u) * 8u + (lane & 7u)) * PD_SJB + (lane >> 4) * 16u
                    + (warp & 3u) * 128u;
#pragma unroll
  for (uint32_t x = 0; x < 3u; x++) {
    aD[x] = sbase + PD_STGA + x * PD_STA + (warp + hL * 8u) * PD_SJA + 16u * wA;
    aF[x] = sbase + PD_STGA + x * PD_STA + ((lane >> 4) * 8u + (lane & 7u)) * PD_SJA + ((lane >> 3) & 1u) * 16u
            + (warp >> 2) * (MT * 32u);
  }
  // trL: the lane's region words, less the warp's and the row group's part: its column
  // offset (lane & 3) and which of a 32-row group's two row offsets it holds (lane bit 4).
  uint32_t trL = sbase + PD_TR + ((warp & 3u) * RPS + (lane & 3u) * RO + ((lane >> 4) & 1u)) * 64u;
  asm volatile("" : "+r"(tabBL), "+r"(tabAL), "+r"(dstBL), "+r"(fragBL), "+r"(trL));
  const uint32_t sCnt = sbase + PD_CNT;

  // ---- small pieces ----
  auto ld4 = [](uint32_t r[4], uint32_t a) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto st4 = [](uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
                 : "memory");
  };
  auto ldg4 = [](uint32_t r[4], const uint4 *p) {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
  };
  auto ldmt = [](uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
  };
  auto xor3 = [](uint32_t a, uint32_t b, uint32_t c) -> uint32_t {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
  };

  // A unit's steps. A lane draws sixteen rows (its table word) of its k-pair: for each k,
  // fill + the word at the plus rank index - the word at the minus one (the table's bytes
  // are the dense factor plus 32, so no byte borrows: PD_FILL), then the two k's bytes
  // interleaved, eight rows to a 16-byte stage row.
  uint32_t dp[4], dm[4], dx[4], dy[4], dw[8];
  auto u_sub = [&](uint32_t out[4]) {
#pragma unroll
    for (int q = 0; q < 4; q++) out[q] = dp[q] + PD_FILL4 - dm[q];
  };
  auto u_sel = [&]() {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      dw[2 * q] = __byte_perm(dx[q], dy[q], 0x5140u);
      dw[2 * q + 1] = __byte_perm(dx[q], dy[q], 0x7362u);
    }
  };
  // The whole unit at once (the prologue's): tab is the lane's table word at rank index 0,
  // o its four offsets, dst its first group's stage row, second the bytes to its second.
  auto u_all = [&](uint32_t tab, const uint4 *po, uint32_t dst, uint32_t second) {
    uint32_t o[4];
    ldg4(o, po);
    ld4(dp, tab + o[0]);
    ld4(dm, tab + o[1]);
    u_sub(dx);
    ld4(dp, tab + o[2]);
    ld4(dm, tab + o[3]);
    u_sub(dy);
    u_sel();
    st4(dst, dw[0], dw[1], dw[2], dw[3]);
    st4(dst + second, dw[4], dw[5], dw[6], dw[7]);
  };

  int32_t acc[MT][NB][4];
  uint32_t af[2][4];        // A's fragment of the m16 tile being folded, and the next one's
  uint32_t bf[NB][2];       // B's eight fragments of the k-step

  // A tile's readout: the lane's 32 sums of the tile XORed together, as one chain.
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
  // Its end: rotated once for every later chunk of the same word (the reference rotates
  // the word left 13 before each chunk's XOR), then XORed into the region's word by every
  // lane that holds a part of it. tr is the lane's word of the group: trL plus the warp's
  // and the group's part.
  auto fold_end = [&](uint32_t x, uint32_t tr, uint32_t q) {
    const uint32_t later = (chunks >> 4) - 1u - (q >> 4);
    x = __funnelshift_l(x, x, (13u * later) & 31u);
    asm volatile("red.shared.xor.b32 [%0], %1;" ::"r"(tr + (q & 15u) * 4u), "r"(x) : "memory");
  };

#define PD_TILE(W, S, M)                                                                   \
  PD_SLOT(W, S, M, 0) PD_SLOT(W, S, M, 1) PD_SLOT(W, S, M, 2) PD_SLOT(W, S, M, 3)          \
  PD_SLOT(W, S, M, 4) PD_SLOT(W, S, M, 5) PD_SLOT(W, S, M, 6) PD_SLOT(W, S, M, 7)
#define PD_KSTEP(W, S)                                                                     \
  PD_TILE(W, S, 0) PD_TILE(W, S, 1) PD_TILE(W, S, 2) PD_TILE(W, S, 3) PD_TILE(W, S, 4)
#define PD_CHUNK(W) PD_KSTEP(W, 0) PD_KSTEP(W, 1) PD_KSTEP(W, 2) PD_KSTEP(W, 3)

  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    const uint32_t rbg = v % row_groups, cbg = v / row_groups;
#if PD_PROBE
    // The probe's stamps live in shared (sCnt + 64 + 24 * warp), not in registers.
    if (lane == 0u) {
      const long long pt0 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 64u + warp * 24u), "l"(pt0) : "memory");
    }
#endif
    // ---- the tile's tables, from the host's: B's 2048 words, then A's 128 PD_WA ----
    {
      const uint4 *const tb = tabB + (size_t)(col_off + cbg) * 2048u;
      const uint4 *const ta = tabA + (size_t)rbg * (128u * PD_WA);
      for (uint32_t i = threadIdx.x; i < 2048u + 128u * PD_WA; i += PD_THREADS) {
        const uint4 w = i < 2048u ? tb[i] : ta[i - 2048u];
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(sbase + PD_TABB + i * 16u), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w)
                     : "memory");
      }
      if (threadIdx.x < 4u) asm volatile("st.shared.u32 [%0], %1;" ::"r"(sCnt + 4u * threadIdx.x), "r"(0u) : "memory");
    }
    __syncthreads();
    // ---- stages 0 and 1, every unit of them: warp W draws k-pairs W and W + 8 ----
    // A's buffers go round in threes: aF[x] and aD[x] are the buffer of k-step x of the
    // chunk (x 0 .. 2; k-step 3 has k-step 0's), turned by one at each chunk's end.
#pragma unroll
    for (uint32_t x = 0; x < 3u; x++) {
      aD[x] = sbase + PD_STGA + x * PD_STA + (warp + hL * 8u) * PD_SJA + 16u * wA;
      aF[x] = sbase + PD_STGA + x * PD_STA + ((lane >> 4) * 8u + (lane & 7u)) * PD_SJA + ((lane >> 3) & 1u) * 16u
              + (warp >> 2) * (MT * 32u);
    }
    const uint4 *pOffL = offs + 8u * hL + warp;        // k-step 0, B's, this warp's unit, this lane's k-pair
#pragma unroll
    for (uint32_t ts = 0; ts < 2u; ts++) {
      u_all(tabBL, pOffL + ts * 32u, dstBL + ts * PD_STB, 256u);
      u_all(tabAL, pOffL + ts * 32u + 16u, aD[ts], PD_SA);
    }
    __syncthreads();
    // The salt stamp: A's row 0 holds it in place of the fill for its first bytes, so those
    // bytes of A' are the stamp plus the noise (saturated, as the materialise does).
    if (rbg == 0u && threadIdx.x < stamp_bytes) {
      unsigned char *const b = reinterpret_cast<unsigned char *>(pd_smem) + PD_STGA
                               + (threadIdx.x >> 1) * PD_SJA + (threadIdx.x & 1u);
      int s = (int)(signed char)*b - PD_FILL + (int)stamp[threadIdx.x];
      s = s < -128 ? -128 : (s > 127 ? 127 : s);
      *b = (unsigned char)s;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
    // The first k-step's fragments: A's first m16 tile and B's eight. A's four matrices are
    // (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15, k 16-31); B's
    // of a pair of n8 tiles are (tile 0, k 0-15), (tile 0, k 16-31), (tile 1, k 0-15),
    // (tile 1, k 16-31).
    ldmt(af[0][0], af[0][1], af[0][2], af[0][3], aF[0]);
#pragma unroll
    for (uint32_t p = 0; p < NB / 2u; p++)
      ldmt(bf[2u * p][0], bf[2u * p][1], bf[2u * p + 1u][0], bf[2u * p + 1u][1],
           fragBL + p * 32u);
    // The offsets of the stage k-step 0 draws (k-step 2's); the loop loads each k-step's in
    // the one before.
    uint32_t oB[4], oA[4];
#if !PD_NO_DRAW
    ldg4(oB, pOffL + 64u);
    ldg4(oA, pOffL + 64u + 16u);
#endif
#if PD_PROBE
    if (lane == 0u) {
      const long long pt1 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 72u + warp * 24u), "l"(pt1) : "memory");
    }
#endif

    // One mma of a k-step and what follows it. The warp, the k-step and the mma come as types
    // (PdC<n>), so each is a constant expression in its copy and every test below is settled
    // by the front end: a slot is straight-line code.
    uint32_t bad = 0u;
    uint32_t c = 0u;
    const uint32_t zero = chunks >> 16;                 // zero for every k the fold takes; ptxas cannot know
    const uint32_t pinL = tabBL + (PD_PIN - PD_TABB) + zero;
    auto slot = [&](auto cW, auto cS, auto cM, auto cN) {
      constexpr uint32_t W = decltype(cW)::v, s = decltype(cS)::v, mb = decltype(cM)::v, nb = decltype(cN)::v;
      constexpr uint32_t wr = W >> 2;   // the copy is warps 0-3's (W 0) or warps 4-7's (W 4)
      // B's buffers alternate, and a chunk is four k-steps: bCur holds this k-step's stage
      // (read in the k-step before) and takes k-step s + 2's.
      constexpr uint32_t bCur = (s & 1u) * PD_STB, bNxt = PD_STB - bCur;
      // A's: this k-step's, the next one's, and the one drawn into.
      constexpr uint32_t aCur = s % 3u, aNxt = (s + 1u) % 3u, aDrw = (s + 2u) % 3u;
      constexpr uint32_t i = mb * NB + nb;
      // This warp's places: the pin, then A's unit and B's.
      constexpr uint32_t pp = PD_PIN_AT + (W < 4u ? 0u : PD_STAG);
      constexpr uint32_t au = pp + 1u, bu = pp + 6u;
      constexpr uint32_t lag = PD_RLAG + (W < 4u ? 0u : PD_RSTAG);
      // A's two fragment sets alternate from tile to tile, and on through the k-steps (a
      // warp has an odd number of tiles).
      constexpr uint32_t as = (mb + (MT & 1u) * s) & 1u, as0n = ((MT & 1u) * (s + 1u)) & 1u;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9}, {%0,%1,%2,%3};\n"
          : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
          : "r"(af[as][0]), "r"(af[as][1]), "r"(af[as][2]), "r"(af[as][3]), "r"(bf[nb][0]),
            "r"(bf[nb][1]));
      // A's next m16 tile of this k-step, into the other fragment set, eight mma ahead.
      if constexpr (nb == 0u && mb + 1u < MT)
        ldmt(af[as ^ 1u][0], af[as ^ 1u][1], af[as ^ 1u][2], af[as ^ 1u][3],
             aF[aCur] + (mb + 1u) * 32u);
      // ---- the check, before this k-step's first store into a stage: have all eight warps
      // made the add of the k-step before (see below)? Then none still reads the buffers
      // this k-step draws into, and all have drawn what this warp reads next. The counter
      // of a k-step of the chunk is 256 a chunk when they have. No branch: a short count
      // sets bad's top bit, and the tile's end looks at it. The pin, a store ptxas cannot
      // tell from the counters, keeps it from moving the load up to this warp's own add.
      if constexpr (!PD_NO_CHECK && i == pp) {
        uint32_t seen, pv = c;
        if constexpr (PD_PINDEP)
          asm("lop3.b32 %0, %1, %2, %3, 0xEC;" : "=r"(pv) : "r"(acc[mb][nb][0]), "r"(c), "r"(zero));
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(pinL), "r"(pv) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(seen) : "r"(sCnt + 4u * ((s + 3u) & 3u)) : "memory");
        bad |= seen - (s == 0u ? c : c + 1u) * PD_THREADS;
      }
      // ---- the draw: this warp's unit of A and its unit of B, of k-step s + 2's stage ----
#if !PD_NO_DRAW
#define PD_UNIT(U, TAB, O, DST, SECOND)                                                    \
      if constexpr (i == (U)) {                                                            \
        ld4(dp, (TAB) + (O)[0]);                                                           \
        ld4(dm, (TAB) + (O)[1]);                                                           \
      }                                                                                    \
      if constexpr (i == (U) + 1u) {                                                       \
        u_sub(dx);                                                                         \
        ld4(dp, (TAB) + (O)[2]);                                                           \
        ld4(dm, (TAB) + (O)[3]);                                                           \
      }                                                                                    \
      if constexpr (i == (U) + 2u) u_sub(dy);                                              \
      if constexpr (i == (U) + 3u) u_sel();                                                \
      if constexpr (i == (U) + 4u) {                                                       \
        st4((DST), dw[0], dw[1], dw[2], dw[3]);                                            \
        st4((DST) + (SECOND), dw[4], dw[5], dw[6], dw[7]);                                 \
      }
      PD_UNIT(au, tabAL, oA, aD[aDrw], PD_SA)
      PD_UNIT(bu, tabBL, oB, dstBL + bCur, 256u)
#undef PD_UNIT
      // The next k-step's offsets, once this k-step's unit has read with its own.
      if constexpr (i == au + 2u) ldg4(oA, pOffL + (s + 3u) * 32u + 16u);
      if constexpr (i == bu + 2u) ldg4(oB, pOffL + (s + 3u) * 32u);
#endif
      // ---- the next k-step's fragments: A's first tile into the set the last tile does not
      // use, then B's pairs, each after the last mma that reads the pair it replaces. ptxas
      // moves them up when it has registers, but never past this k-step's stores, which it
      // cannot tell from them.
      if constexpr (i == NI - 8u)
        ldmt(af[as0n][0], af[as0n][1], af[as0n][2], af[as0n][3], aF[aNxt]);
#define PD_BLOAD(PB)                                                                       \
      if constexpr (i == NI - 7u + 2u * PB)                                                \
        ldmt(bf[2u * PB][0], bf[2u * PB][1], bf[2u * PB + 1u][0], bf[2u * PB + 1u][1],     \
             fragBL + bNxt + PB * 32u);
      PD_BLOAD(0u)
      PD_BLOAD(1u)
      PD_BLOAD(2u)
      PD_BLOAD(3u)
#undef PD_BLOAD
      // ---- the add, behind those loads: this warp has read all it will of A's stage s and
      // of stage s + 1's B and first A tile, and has drawn its part of stage s + 2. One
      // ATOMS for the warp.
      if constexpr ((!PD_NO_CHECK || PD_BAR == 2) && i == NI - 1u)
        atomicAdd(reinterpret_cast<unsigned *>(pd_smem) + PD_CNT / 4u + s, 1u);
      // PD_BAR 3: every warp meets here, at each k-step's end. Nothing is then read or drawn
      // out of turn whatever ptxas does with the order: the loads above cannot sink past
      // the barrier and the next k-step's stores cannot rise past it.
      if constexpr (PD_BAR == 3 && i == NI - 1u) __syncthreads();
      // ---- the readout: each m16 tile's chain, PD_RLAG mma after the tile's last mma of the
      // chunk's last k-step. One with no room there is made early in the next chunk's first
      // k-step, for the chunk before (zero sums in a tile's first chunk, so it XORs nothing
      // there), and once more after the loop.
#if !PD_NO_READOUT
#define PD_RT(T)                                                                           \
      if constexpr (s == (pd_tile_late(T, lag) ? 0u : 3u) && i == pd_tile_first(T, lag))   \
        fold_end(fold_tile(T), trL + 2u * ((wr * MT + T) >> 1) * 64u,                      \
                 pd_tile_late(T, lag) ? c - 1u : c);
      PD_RT(0u) PD_RT(1u) PD_RT(2u) PD_RT(3u) PD_RT(4u)
#undef PD_RT
#endif
    };
#define PD_SLOT(W, S, M, N) slot(PdC<W>{}, PdC<S>{}, PdC<M>{}, PdC<N>{});
#define PD_LOOP(W)                                                                         \
  case (W) >> 2:                                                                           \
    for (; c < chunks; c++) {                                                              \
      if (PD_BAR == 2) {                                                                   \
        /* Every warp has made the last add of chunk c - 1: the counter is 256 c. (A   */  \
        /* volatile load: ptxas lifts a plain one out of the loop.)                    */  \
        for (;;) {                                                                         \
          uint32_t seen;                                                                   \
          asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(seen) : "r"(sCnt + 12u) : "memory"); \
          if (!__any_sync(0xffffffffu, (int)(seen - c * PD_THREADS) < 0)) break;           \
        }                                                                                  \
      }                                                                                    \
      PD_CHUNK(W)                                                                          \
      pOffL += 128; /* four k-steps of B's and A's offsets, 16 k-pairs each */             \
      { const uint32_t t = aF[0]; aF[0] = aF[1]; aF[1] = aF[2]; aF[2] = t; }               \
      { const uint32_t t = aD[0]; aD[0] = aD[1]; aD[1] = aD[2]; aD[2] = t; }               \
      if (PD_BAR == 1) __syncthreads();                                                    \
    }                                                                                      \
    break;
    switch (warp >> 2) {
      PD_LOOP(0)
      PD_LOOP(4)
      default:
        break;
    }
#undef PD_SLOT
#undef PD_LOOP
#if PD_PROBE
    if (lane == 0u) {
      const long long pt2 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 80u + warp * 24u), "l"(pt2) : "memory");
    }
#endif
    // The last chunk's tiles that the loop reads out a chunk late (all five without the
    // readout in the loop).
#pragma unroll
    for (uint32_t t = 0; t < MT; t++)
      if (PD_NO_READOUT || (warp < 4u ? pd_tile_late(t, PD_RLAG) : pd_tile_late(t, PD_RLAG + PD_RSTAG)))
        fold_end(fold_tile(t), trL + 2u * (((warp >> 2) * MT + t) >> 1) * 64u, chunks - 1u);

    // Hand-off: the column slot's two warps meet, so every word is in shared.
    asm volatile("bar.sync %0, %1;" ::"r"(1u + (warp & 3u)), "r"(64u) : "memory");
#if PD_PROBE
    if (warp < 4u) {
      // The probe: the slot's 8 MT regions go out as they are, then are zeroed.
#pragma unroll
      for (uint32_t i = 0; i < MT; i++) {
        const uint32_t o = warp * (RPS * 64u) + i * 512u + lane * 16u;
        uint32_t w[4];
        ld4(w, sbase + PD_TR + o);
        if (v < dump_tiles) {
          uint32_t *const d = dump + (size_t)v * (RPS * 64u) + o / 4u;
          d[0] = w[0];
          d[1] = w[1];
          d[2] = w[2];
          d[3] = w[3];
        }
        st4(sbase + PD_TR + o, 0u, 0u, 0u, 0u);
      }
    }
    if (lane == 0u) {
      // Cycles: the tile's prologue, its chunk loop, and its end, summed a warp; the tiles;
      // and the tiles whose check failed in this warp.
      const long long pt3 = clock64();
      long long pt0, pt1, pt2;
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt0) : "r"(sCnt + 64u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt1) : "r"(sCnt + 72u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt2) : "r"(sCnt + 80u + warp * 24u) : "memory");
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 0u, (unsigned long long)(pt1 - pt0));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 1u, (unsigned long long)(pt2 - pt1));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 2u, (unsigned long long)(pt3 - pt2));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 3u, 1ull);
      if ((int)bad < 0) atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 4u, 1ull);
    }
#else
#error "the miner's hash is not written yet"
#endif
    __syncthreads();
  }
#undef PD_TILE
#undef PD_KSTEP
#undef PD_CHUNK
#endif
}
