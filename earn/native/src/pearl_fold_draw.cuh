// The draw fold: eight warps over a tile of 32 PD_MT rows by 256 columns, with both operands
// drawn on the SM instead of copied from memory. Branch goal/nv-kernel, 11 November 2026.
//
// A' and B' are never stored. A tile keeps, in shared memory, the dense noise factors of its
// rows and columns (transposed: one 16-byte word holds sixteen rows at one rank index), and
// two k-step stages of the noised operands. Each warp draws the next k-step's stage while it
// folds this one, in half units: eight rows by one k-pair, four 8-byte table reads, four
// subtracts, four byte selects and a 16-byte store. The stage is laid out so ldmatrix.trans
// hands every lane its k-packed fragment word.
//
// Included by pearl_kernel.cu (the miner's kernel) and by probes/drawprobe.cu (the same body
// with cycle stamps and a transcript dump in place of the hash). Before including, define
//   PD_FN      the kernel's name
//   PD_PROBE   1 for the probe's build, 0 for the miner's
// and, for the miner's build, everything the tall fold's hash uses (pearl_kernel.cu has it).
//
// PD_MT is the m16 tiles a warp folds: 5 (80x64 a warp, 160 sums a thread, a 160x256 tile) or
// 6 (96x64, 192 sums, the tall fold's 192x256). With 6 a thread has 39 registers beside its
// sums and fragments, ptxas fills them by lifting loads and then spills; measured on an RTX
// 4090, the spills cost more than everything else in the stream (probes/README.md, "The draw
// fold"). With 5 it has 71.
//
// The chunk loop is written out once a warp (eight copies), so everything that depends on the
// warp is a constant in its copy: a table read's address is its offset plus a constant, with
// no add.
//
// Layouts the host shares (see the probe for a writer of each):
//   table    [quad q][e][rank index p] of 16 bytes: a quad is 32 rows (or columns); word (q, e, p)
//            holds the dense factor plus 32 of rows 32q + 8e + 0..7 (bytes 0-7) and
//            32q + 8e + 16 + 0..7 (bytes 8-15) at rank index p.
//   offsets  [k-step][B, A][lane 0..31] of four 32-bit words: for the lane's k-pair, its first
//            k's plus and minus rank index, then its second k's, each times 16, plus 2048 e.
//            B's lane L draws e = L >> 4, k-pair L & 15; A's lane L draws e = (L >> 3) & 1,
//            k-pair 8 (L >> 4) + (L & 7). Three k-steps more than the tile has, at the end
//            (the loop loads one and touches two it never draws with).
//   stage    16 bytes for a group of 8 rows and a k-pair: byte 2c is row c at the pair's
//            first k and byte 2c + 1 at its second. A's group G, k-pair j is at
//            PD_AOFF + 256 G + 16 j. B's is at 512 (G >> 1) + 256 (j >> 3) + 128 (G & 1)
//            + 16 (j & 7): half-major inside a pair of groups, so that one lane register
//            addresses both operands' ldmatrix rows (the mma wants a tile's two k halves in
//            adjacent registers for B, and its two row groups for A).
//   regions  a tile has 2 PD_MT valid row offsets by 16 valid column offsets. Column slot wc
//            (64 columns) holds 8 PD_MT of them: region 2 PD_MT tq + ro is column offset tq
//            (0..3) of the slot and row offset ro of the tile, 64 bytes of transcript each.

#ifndef PD_FN
#error "pearl_fold_draw.cuh: define PD_FN and PD_PROBE before including it"
#endif
#ifndef PD_MT
#define PD_MT 5
#endif
#if PD_MT != 5 && PD_MT != 6
#error "PD_MT is 5 or 6 m16 tiles a warp"
#endif

// Shared memory, bytes from the block's base.
#define PD_TABB 0u                                  // B's table: 8 quads
#define PD_TABA 32768u                              // A's table: PD_MT quads
#define PD_STG (32768u + PD_MT * 4096u)             // two stages, PD_STAGE bytes each
#define PD_AOFF 8192u                               // A's groups in a stage, after B's
#define PD_STAGE (PD_AOFF + PD_MT * 1024u)
#define PD_TR (PD_STG + 2u * PD_STAGE)              // transcripts, [region][16 words]
#define PD_CNT (PD_TR + PD_MT * 2048u)              // the stage counter, then the probe's stamps
#define PD_SCR (PD_CNT + 256u)                      // a 16-byte row a lane (512 bytes), for the holds
#define PD_SMEM (PD_SCR + 512u)
#define PD_THREADS 256u
#define PD_NI (8u * PD_MT)                          // mma a k-step

// The pieces' places in a k-step, as the index of the mma they follow (0 .. PD_NI - 1).
#ifndef PD_WAIT_AT
#define PD_WAIT_AT (PD_NI - 9u)   // the wait for the next stage: before the last m16 tile
#endif
#ifndef PD_ADD_AT
#define PD_ADD_AT (PD_NI - 15u)   // the counter add: after the draw's last store and A's last load
#endif
// A half unit's four steps: half unit h starts after mma PD_HS h, and its subtracts, byte
// selects and store follow that many mma later.
#ifndef PD_HS
#define PD_HS 6u
#endif
#ifndef PD_HA
#define PD_HA 2u
#endif
#ifndef PD_HP
#define PD_HP 4u
#endif
#ifndef PD_HST
#define PD_HST 5u
#endif
// The mma after which the next k-step's offsets are loaded: past this k-step's last use
// of the registers (half unit 3's reads).
#ifndef PD_OFFS_AT
#define PD_OFFS_AT (3u * PD_HS + 2u)
#endif
// A readout run starts this many mma after its tiles' last, and its four parts and its end
// (the rotate and the shared XOR) follow one mma apart.
#ifndef PD_RLAG
#define PD_RLAG 9u
#endif
// 1: two of the warps touch the next k-step's offsets a k-step early, so the loads that use
// them hit the L1 (every warp of a block reads the same 1 KB a k-step).
#ifndef PD_PREFETCH
#define PD_PREFETCH 1
#endif
// Timing only: pieces off. PD_NO_DRAW folds the stage the prologue drew, over and over;
// PD_NO_READOUT reads out at the tile's end only; PD_NO_SYNC drops the add and the wait.
#ifndef PD_NO_DRAW
#define PD_NO_DRAW 0
#endif
#ifndef PD_NO_READOUT
#define PD_NO_READOUT 0
#endif
#ifndef PD_NO_SYNC
#define PD_NO_SYNC 0
#endif
// Timing only: every warp runs warp 0's copy of the chunk loop (wrong sums): what eight
// copies of the loop cost against one.
#ifndef PD_T_ONEVER
#define PD_T_ONEVER 0
#endif
// Nanoseconds a warp sleeps between two tests of a stage that is not whole yet (0: it spins).
#ifndef PD_NAP
#define PD_NAP 0
#endif
// Timing only (wrong sums). 1: the loop loads no offsets; every k-step draws with the tile's
// first ones (prices the global loads). 2: as 1, and the table reads go to addresses with
// the least bank conflict a warp can have (prices the conflicts of the real ones).
#ifndef PD_T_OFFS
#define PD_T_OFFS 0
#endif
// Probe only: count the waits that found the next stage not whole (adds code to the spin's path).
#ifndef PD_SPINCOUNT
#define PD_SPINCOUNT 0
#endif
// 1: a k-step's first subtracts wait for mma PD_CP of the m16 tile before them (see the
// chunk loop).
#ifndef PD_CPIN
#define PD_CPIN 1
#endif
#ifndef PD_CP
#define PD_CP 2u
#endif
// (Tried and dropped, PD_MT 6: holding each of the next k-step's B loads behind a 4-byte store
// of a sum, so they reuse the registers they replace. The four stores cost 5.4% of the pipe:
// each waits for its sum.)

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
// Warp-row wr's readout run r (0 .. 2, in the order its tiles finish): its m16 tiles (tB is
// PD_MT when it has one) and the 32-row group of the tile they belong to. A group is two m16
// tiles; with PD_MT 5 the tile's middle group is split between the two warps of a column
// slot, a tile each.
struct PdRun {
  unsigned tA, tB, grp;
};
PD_CE PdRun pd_run_of(unsigned wr, unsigned r) {
  const unsigned first = wr * PD_MT;   // the warp's first m16 tile, of the tile's 2 PD_MT
  unsigned t = 0;
  PdRun out = {0u, (unsigned)PD_MT, 0u};
  for (unsigned i = 0; i <= r; i++) {
    const bool single = ((first + t) & 1u) != 0u || t + 1u >= (unsigned)PD_MT;
    out.tA = t;
    out.tB = single ? (unsigned)PD_MT : t + 1u;
    out.grp = (first + t) >> 1;
    t += single ? 1u : 2u;
  }
  return out;
}
// Whether run r is made a chunk late (in the next chunk's first k-step), and the mma its
// first part follows there or in the chunk's last k-step.
PD_CE bool pd_run_late(unsigned wr, unsigned r, unsigned lag) {
  const PdRun rn = pd_run_of(wr, r);
  const unsigned done = (rn.tB < (unsigned)PD_MT ? rn.tB : rn.tA) * 8u + 7u;
  return done + lag + 4u >= 8u * (unsigned)PD_MT;
}
PD_CE unsigned pd_run_first(unsigned wr, unsigned r, unsigned lag) {
  const PdRun rn = pd_run_of(wr, r);
  const unsigned done = (rn.tB < (unsigned)PD_MT ? rn.tB : rn.tA) * 8u + 7u;
  return pd_run_late(wr, r, lag) ? done + lag - 8u * (unsigned)PD_MT : done + lag;
}
// What the readout does after mma i of k-step s: part 0 .. 3 of run r, its end (4), or
// nothing (r is 9).
struct PdStep {
  unsigned r, part;
  bool late;
};
PD_CE PdStep pd_readout_at(unsigned wr, unsigned s, unsigned i, unsigned lag) {
  for (unsigned r = 0; r < 3u; r++) {
    const bool late = pd_run_late(wr, r, lag);
    const unsigned i0 = pd_run_first(wr, r, lag);
    if (s == (late ? 0u : 3u) && i >= i0 && i <= i0 + 4u) return PdStep{r, i - i0, late};
  }
  return PdStep{9u, 0u, false};
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
  // fragL: this lane's 16-byte row in a 512-byte block of a stage (two groups by two k
  // halves), less everything the warp and the k-step fix. ldmatrix.trans reads it and the
  // draw writes it. For A bit 3 of the lane is the group and bit 4 the half; for B bit 3 is
  // the half and bit 4 the group (see "stage" above).
  uint32_t fragL = sbase + PD_STG + ((lane >> 3) & 1u) * 256u + (lane >> 4) * 128u + (lane & 7u) * 16u;
  // trL: the lane's region words, less the warp's and the row group's part: its column
  // offset (lane & 3) and which of a 32-row group's two row offsets it holds (lane bit 4).
  uint32_t trL = sbase + PD_TR + ((lane & 3u) * RO + ((lane >> 4) & 1u)) * 64u;
  // dstL: fragL again, for the draw's stores. A second register on purpose: with one, ptxas
  // sees that a store and a fragment load never overlap and lifts every ldmatrix of the
  // k-step to its top, into registers of their own. With two it must keep a load after the
  // stores before it. (Plus chunks >> 16, which is zero for every k the fold takes: an
  // empty asm alone leaves ptxas a copy it folds away.)
  uint32_t dstL = fragL + (chunks >> 16);
  asm volatile("" : "+r"(fragL), "+r"(trL), "+r"(dstL));
  const uint32_t zero = chunks >> 16;
  (void)zero;
  const uint32_t sCnt = sbase + PD_CNT;

  // ---- small pieces ----
  auto ld4 = [](uint32_t r[4], uint32_t a) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto ld2 = [](uint32_t r[2], uint32_t a) {
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
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

  // A whole unit (the prologue's): sixteen rows of one k-pair. o holds its four table
  // offsets; x and y the sixteen noised bytes of each k. The table's bytes are the dense
  // factor plus 32, so fill + plus - minus has no borrow between bytes (PD_FILL) and is
  // the noised operand. tab is the table's address less the offset.
  struct Unit {
    uint32_t o[4], x[4], y[4];
  };
  auto u_half = [&](uint32_t out[4], uint32_t tab, uint32_t op, uint32_t om) {
    uint32_t p[4], q[4];
    ld4(p, tab + op);
    ld4(q, tab + om);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = p[i] + PD_FILL4 - q[i];
  };
  auto u_store = [&](const Unit &u, uint32_t dst) {
    st4(dst, __byte_perm(u.x[0], u.y[0], 0x5140u), __byte_perm(u.x[0], u.y[0], 0x7362u),
        __byte_perm(u.x[1], u.y[1], 0x5140u), __byte_perm(u.x[1], u.y[1], 0x7362u));
    st4(dst + 512u, __byte_perm(u.x[2], u.y[2], 0x5140u), __byte_perm(u.x[2], u.y[2], 0x7362u),
        __byte_perm(u.x[3], u.y[3], 0x5140u), __byte_perm(u.x[3], u.y[3], 0x7362u));
  };
  // Half a unit, as the chunk loop draws it: rows 8 hi .. 8 hi + 7 of the sixteen, from
  // four 8-byte reads.
  struct Half {
    uint32_t p0[2], m0[2], p1[2], m1[2];
  };

  int32_t acc[MT][NB][4];
  uint32_t af[2][4];        // A's fragment of the m16 tile being folded, and the next one's
  uint32_t bf[NB][2];       // B's eight fragments of the k-step

  // A readout run: the sums a lane holds of one 32-row group, XORed together. A group is two
  // m16 tiles; with PD_MT 5 the tile's middle group is split between the two warps of a
  // column slot, a tile each. One part is two n8 tiles: 16 sums of a pair of m16 tiles in
  // 8 xor3, or 8 sums of a single one in 4.
  auto fold_part = [&](uint32_t x, uint32_t tA, uint32_t tB, uint32_t np) -> uint32_t {
    const int32_t *p0 = acc[tA][2u * np], *p2 = acc[tA][2u * np + 1u];
    if (tB >= MT) {
      const uint32_t ta = xor3((uint32_t)p0[0], (uint32_t)p0[1], (uint32_t)p0[2]);
      const uint32_t tb = xor3((uint32_t)p0[3], (uint32_t)p2[0], (uint32_t)p2[1]);
      const uint32_t tc = np == 0u ? ((uint32_t)p2[2] ^ (uint32_t)p2[3])
                                   : xor3((uint32_t)p2[2], (uint32_t)p2[3], x);
      return xor3(ta, tb, tc);
    }
    const int32_t *p1 = acc[tB][2u * np], *p3 = acc[tB][2u * np + 1u];
    const uint32_t ta = xor3((uint32_t)p0[0], (uint32_t)p0[1], (uint32_t)p0[2]);
    const uint32_t tb = xor3((uint32_t)p0[3], (uint32_t)p1[0], (uint32_t)p1[1]);
    const uint32_t tc = xor3((uint32_t)p1[2], (uint32_t)p1[3], (uint32_t)p2[0]);
    const uint32_t td = xor3((uint32_t)p2[1], (uint32_t)p2[2], (uint32_t)p2[3]);
    const uint32_t te = xor3(ta, tb, tc);
    const uint32_t rest = np == 0u ? (te ^ td) : xor3(te, td, x);
    const uint32_t t3 = xor3((uint32_t)p3[0], (uint32_t)p3[1], (uint32_t)p3[2]);
    return xor3(t3, (uint32_t)p3[3], rest);
  };
  // A run's end: rotated once for every later chunk of the same word (the reference rotates
  // the word left 13 before each chunk's XOR), then XORed into the region's word by every
  // lane that holds a part of it. tr is the lane's word of the group: trL plus the warp's
  // and the group's part.
  auto fold_end = [&](uint32_t x, uint32_t tr, uint32_t q) {
    const uint32_t later = (chunks >> 4) - 1u - (q >> 4);
    x = __funnelshift_l(x, x, (13u * later) & 31u);
    asm volatile("red.shared.xor.b32 [%0], %1;" ::"r"(tr + (q & 15u) * 4u), "r"(x) : "memory");
  };
  // The fences' branches (see PD_SLOT below): one a label, never taken (the host never asks
  // for this many tiles), so each label has a second way in and ptxas keeps its block apart.
  // Each tests its own value: with one value for all, ptxas keeps the first branch only.
  // A fence costs register moves where the blocks join (measured: 176 moves a chunk at one
  // fence in four), so none is set; kept for experiments.
#ifndef PD_FENCE_MASK
#define PD_FENCE_MASK 0ull
#endif
#define PD_FENCED(M, N) (((PD_FENCE_MASK) >> ((M) * 8 + (N))) & 1ull)
#define PD_SLOT(W, S, M, N)                                                                \
  if (PD_FENCED(M, N))                                                                     \
    asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %0, %1;\n @p bra PDL_" #W "_" #S "_" #M "_" #N ";\n}" \
                 ::"r"(tiles), "r"(0x7ffff000u + W * 192u + S * 48u + M * 8u + N));
#define PD_TILE(W, S, M)                                                                   \
  PD_SLOT(W, S, M, 0) PD_SLOT(W, S, M, 1) PD_SLOT(W, S, M, 2) PD_SLOT(W, S, M, 3)          \
  PD_SLOT(W, S, M, 4) PD_SLOT(W, S, M, 5) PD_SLOT(W, S, M, 6) PD_SLOT(W, S, M, 7)
#if PD_MT == 6
#define PD_KSTEP(W, S)                                                                     \
  PD_TILE(W, S, 0) PD_TILE(W, S, 1) PD_TILE(W, S, 2) PD_TILE(W, S, 3) PD_TILE(W, S, 4) PD_TILE(W, S, 5)
#else
#define PD_KSTEP(W, S)                                                                     \
  PD_TILE(W, S, 0) PD_TILE(W, S, 1) PD_TILE(W, S, 2) PD_TILE(W, S, 3) PD_TILE(W, S, 4)
#endif
#define PD_CHUNK(W) PD_KSTEP(W, 0) PD_KSTEP(W, 1) PD_KSTEP(W, 2) PD_KSTEP(W, 3)
#define PD_WARPS PD_CHUNK(0) PD_CHUNK(1) PD_CHUNK(2) PD_CHUNK(3) PD_CHUNK(4) PD_CHUNK(5) PD_CHUNK(6) PD_CHUNK(7)
#if PD_FENCE_MASK
  PD_WARPS
#endif
#undef PD_SLOT

  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    const uint32_t rbg = v % row_groups, cbg = v / row_groups;
#if PD_PROBE
    // The probe's stamps live in shared (sCnt + 16 + 24 * warp), not in registers.
    if (lane == 0u) {
      const long long pt0 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 16u + warp * 24u), "l"(pt0) : "memory");
    }
#endif
    // ---- the tile's tables, from the host's: B's 8 quads, then A's MT ----
    {
      const uint4 *const tb = tabB + (size_t)(col_off + cbg * 16u) * 128u;
      const uint4 *const ta = tabA + (size_t)(rbg * RO) * 128u;
      for (uint32_t i = threadIdx.x; i < 2048u + MT * 256u; i += PD_THREADS) {
        const uint4 w = i < 2048u ? tb[i] : ta[i - 2048u];
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(sbase + PD_TABB + i * 16u), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w)
                     : "memory");
      }
      if (threadIdx.x == 0u) asm volatile("st.shared.u32 [%0], %1;" ::"r"(sCnt), "r"(0u) : "memory");
    }
    __syncthreads();
    // ---- stage 0, every unit of it: B's eight quads (a warp each) and A's MT ----
    {
      Unit u;
      ldg4(u.o, offs + lane);
      u_half(u.x, sbase + PD_TABB + warp * 4096u, u.o[0], u.o[1]);
      u_half(u.y, sbase + PD_TABB + warp * 4096u, u.o[2], u.o[3]);
      u_store(u, fragL + warp * 1024u);
      if (warp < MT) {
        ldg4(u.o, offs + 32u + lane);
        u_half(u.x, sbase + PD_TABA + warp * 4096u, u.o[0], u.o[1]);
        u_half(u.y, sbase + PD_TABA + warp * 4096u, u.o[2], u.o[3]);
        u_store(u, fragL + PD_AOFF + warp * 1024u);
      }
    }
    __syncthreads();
    // The salt stamp: A's row 0 holds it in place of the fill for its first bytes, so those
    // bytes of A' are the stamp plus the noise (saturated, as the materialise does).
    if (rbg == 0u && threadIdx.x < stamp_bytes) {
      unsigned char *const b = reinterpret_cast<unsigned char *>(pd_smem) + PD_STG + PD_AOFF
                               + (threadIdx.x >> 1) * 16u + (threadIdx.x & 1u);
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
    ldmt(af[0][0], af[0][1], af[0][2], af[0][3], fragL + PD_AOFF + (warp >> 2) * (MT * 512u));
#pragma unroll
    for (uint32_t p = 0; p < NB / 2u; p++)
      ldmt(bf[2u * p][0], bf[2u * p][1], bf[2u * p + 1u][0], bf[2u * p + 1u][1],
           fragL + (warp & 3u) * 2048u + p * 512u);
    const uint4 *pOff = offs + lane;                  // this lane's offsets, k-step 0, B's
#if PD_PROBE
    if (lane == 0u) {
      const long long pt1 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 24u + warp * 24u), "l"(pt1) : "memory");
    }
#endif

    // One mma of a k-step and what follows it. The warp, the k-step and the mma come as types
    // (PdC<n>), so each is a constant expression in its copy and every test below is settled
    // by the front end: a slot is straight-line code. (As plain arguments the optimiser folded
    // them, and took minutes over the 1,280 copies.)
    Unit ub, ua;
#if !PD_NO_DRAW
    // The first k-step's offsets (the loop loads each k-step's in the one before).
    ldg4(ub.o, pOff + 64u);
    ldg4(ua.o, pOff + 64u + 32u);
#endif
    Half hf;
    uint32_t xy[4], ww[4], rx = 0u;
    uint32_t c = 0u;
    auto slot = [&](auto cW, auto cS, auto cM, auto cN) {
      constexpr uint32_t W = decltype(cW)::v, s = decltype(cS)::v, mb = decltype(cM)::v, nb = decltype(cN)::v;
      constexpr uint32_t wr = W >> 2, wc = W & 3u;
      // Stage of this k-step and of the next: buffers alternate, and a chunk is four.
      constexpr uint32_t cur = (s & 1u) * PD_STAGE, nxt = PD_STAGE - cur;
      constexpr uint32_t i = mb * NB + nb;
      // A's two fragment sets alternate from tile to tile, and on through the k-steps when
      // a warp has an odd number of tiles.
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
             fragL + PD_AOFF + wr * (MT * 512u) + cur + (mb + 1u) * 512u);
      // ---- the draw: the next k-step's stage, this warp's unit of B (its own quad) and,
      // in the k-steps it has one, a unit of A ----
      {
        // Warps 0 to 3 draw their quad of B and a quad of A: four half units. Warps 4 to 7
        // draw their quad of B and half a quad of A: three. So every scheduler (warps w and
        // w + 4) draws seven half units every k-step, and none runs ahead of the others into
        // the wait. With PD_MT 6 that is A's twelve halves exactly; with 5, quad 4's two
        // halves are each drawn twice.
        constexpr uint32_t nh = PD_NO_DRAW ? 0u : (W < 4u ? 4u : 3u);
        constexpr uint32_t qa = W < 4u ? W : 4u + (MT == 6u ? (W - 4u) >> 1 : 0u);
        // The next k-step's offsets, loaded now, once this k-step's half units have read
        // theirs: they are in registers when the next block starts. (Loaded in the k-step
        // that uses them, ptxas rates the global load at about 15 instructions, issues the
        // k-step's first mma meanwhile, and the first half unit's reads and subtracts then
        // come out as one run with nothing between them.)
        if constexpr (!PD_NO_DRAW && !PD_T_OFFS && i == PD_OFFS_AT) {
          ldg4(ub.o, pOff + (s + 2u) * 64u);
          ldg4(ua.o, pOff + (s + 2u) * 64u + 32u);
        }
        // And the k-step after that one's, touched: warp 0 B's 512 bytes, warp 1 A's.
        if constexpr (PD_PREFETCH && !PD_NO_DRAW && i == 1u && W < 2u) {
          uint32_t touched;
          asm volatile("ld.global.u32 %0, [%1];" : "=r"(touched) : "l"(pOff + (s + 3u) * 64u + W * 32u));
        }
        // Half unit h: B's rows 0-7 and 8-15 of its sixteen, then A's. Its table reads go
        // out after mma PD_HS h; its subtracts, byte selects and store PD_HA, PD_HP and
        // PD_HST mma later.
        constexpr uint32_t h = i / PD_HS, st = i % PD_HS;
        if constexpr (h < nh) {
          constexpr bool isA = h >= 2u;
          constexpr uint32_t hi = !isA || W < 4u ? (h & 1u) : (W & 1u);     // rows 0-7 or 8-15 of the sixteen
          constexpr uint32_t tab = (isA ? PD_TABA + qa * 4096u : PD_TABB + W * 4096u) + 8u * hi;
          constexpr uint32_t dst = nxt + (isA ? PD_AOFF + qa * 1024u : W * 1024u) + 512u * hi;
          const uint32_t *const o = isA ? ua.o : ub.o;
          if constexpr (st == 0u) {
#if PD_T_OFFS == 2
            // Timing only: every lane reads its own two words of the table (lane L and
            // lane L + 16 share a bank, no others): the least bank conflict a warp's
            // 8-byte reads can have.
            ld2(hf.p0, sbase + tab + (dstL & 0xf0u) / 2u);
            ld2(hf.m0, sbase + tab + (dstL & 0xf0u) / 2u + 512u);
            ld2(hf.p1, sbase + tab + (dstL & 0xf0u) / 2u + 1024u);
            ld2(hf.m1, sbase + tab + (dstL & 0xf0u) / 2u + 1536u);
#else
            ld2(hf.p0, sbase + tab + o[0]);
            ld2(hf.m0, sbase + tab + o[1]);
            ld2(hf.p1, sbase + tab + o[2]);
            ld2(hf.m1, sbase + tab + o[3]);
#endif
          }
          if constexpr (st == PD_HA) {
            // The fill's constant. For the k-step's first half unit (PD_CPIN) it is made to
            // wait for mma PD_CP of the last m16 tile: that sum ANDed with a zero ptxas cannot
            // see, ORed with the constant, in one lop3. Its reads go out at the top of the
            // block, and ptxas would issue the subtracts right behind them, before any mma.
            uint32_t fc = PD_FILL4;
            if constexpr (PD_CPIN && h == 0u)
              asm("lop3.b32 %0, %1, %3, %2, 0xEC;" : "=r"(fc) : "r"(acc[MT - 1u][PD_CP][0]), "r"(zero), "n"(PD_FILL4));
            xy[0] = hf.p0[0] + fc - hf.m0[0];
            xy[1] = hf.p0[1] + fc - hf.m0[1];
            xy[2] = hf.p1[0] + fc - hf.m1[0];
            xy[3] = hf.p1[1] + fc - hf.m1[1];
          }
          if constexpr (st == PD_HP) {
            ww[0] = __byte_perm(xy[0], xy[2], 0x5140u);
            ww[1] = __byte_perm(xy[0], xy[2], 0x7362u);
            ww[2] = __byte_perm(xy[1], xy[3], 0x5140u);
            ww[3] = __byte_perm(xy[1], xy[3], 0x7362u);
          }
          if constexpr (st == PD_HST) st4(dstL + dst, ww[0], ww[1], ww[2], ww[3]);
        }
      }
      // ---- the readout: this warp's three runs of the chunk. A run starts PD_RLAG mma after
      // its tiles' last mma of the chunk's last k-step. One that would pass the k-step's end
      // is made at the same place of the next chunk's first k-step, for the chunk before
      // (zero sums in a tile's first chunk, so it XORs nothing there), and once more after
      // the loop.
      {
        constexpr PdStep ra = PD_NO_READOUT ? PdStep{9u, 0u, false} : pd_readout_at(wr, s, i, PD_RLAG);
        if constexpr (ra.r < 3u) {
          constexpr PdRun rn = pd_run_of(wr, ra.r);
          if constexpr (ra.part < 4u) rx = fold_part(ra.part == 0u ? 0u : rx, rn.tA, rn.tB, ra.part);
          if constexpr (ra.part == 4u)
            fold_end(rx, trL + (wc * RPS + 2u * rn.grp) * 64u, ra.late ? c - 1u : c);
        }
      }
      // After the draw's last store and A's last load of this stage: this warp has drawn
      // its part of the next stage and loaded all of this one.
      if constexpr (!PD_NO_SYNC && i == PD_ADD_AT)
        atomicAdd(reinterpret_cast<unsigned *>(pd_smem) + PD_CNT / 4u, 1u);
      // Every warp has added: the next stage is whole, and this one's buffer is free to draw
      // into from the next k-step on. The counter is 256 a k-step of the tile. (A volatile
      // load: ptxas lifted a plain one out of the loop, and the wait never ended.)
      if constexpr (!PD_NO_SYNC && i == PD_WAIT_AT) {
        for (;;) {
          uint32_t seen;
          asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(seen) : "r"(sCnt) : "memory");
          if (!__any_sync(0xffffffffu, (int)(seen - (c << 10)) < (int)((s + 1u) * PD_THREADS))) break;
#if PD_NAP
          // Not whole yet: leave the scheduler to the other warp for a while (a spinning
          // warp's tests take issue slots from it).
          asm volatile("nanosleep.u32 %0;" ::"r"((uint32_t)PD_NAP) : "memory");
#endif
#if PD_PROBE && PD_SPINCOUNT
          atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 4u, 1ull);
#endif
        }
      }
      // The next k-step's fragments, behind the wait: A's first tile, then B's pairs, each
      // after the last mma that reads the pair it replaces or right behind the wait,
      // whichever is later.
      if constexpr (i == PD_WAIT_AT + 1u)
        ldmt(af[as0n][0], af[as0n][1], af[as0n][2], af[as0n][3], fragL + PD_AOFF + wr * (MT * 512u) + nxt);
#define PD_BLOAD(PB)                                                                         if constexpr (i == (NI - 7u + 2u * PB > PD_WAIT_AT + 1u ? NI - 7u + 2u * PB : PD_WAIT_AT + 1u))     ldmt(bf[2u * PB][0], bf[2u * PB][1], bf[2u * PB + 1u][0], bf[2u * PB + 1u][1],                  fragL + wc * 2048u + nxt + PB * 512u);
      PD_BLOAD(0u)
      PD_BLOAD(1u)
      PD_BLOAD(2u)
      PD_BLOAD(3u)
#undef PD_BLOAD
    };
    // A fence is a label: ptxas orders instructions inside a basic block only, and a label
    // that something can branch to ends one (see the branches above).
#define PD_SLOT(W, S, M, N)                                                                \
  slot(PdC<W>{}, PdC<S>{}, PdC<M>{}, PdC<N>{});                                            \
  if (PD_FENCED(M, N)) asm volatile("PDL_" #W "_" #S "_" #M "_" #N ":" ::: "memory");
#define PD_LOOP(W)                                                                         \
  case W:                                                                                  \
    for (; c < chunks; c++) {                                                              \
      PD_CHUNK(W)                                                                          \
      pOff += 256; /* four k-steps of B's and A's offsets, 32 lanes each */                \
    }                                                                                      \
    break;
    switch (PD_T_ONEVER ? 0u : warp) {
      PD_LOOP(0)
      PD_LOOP(1)
      PD_LOOP(2)
      PD_LOOP(3)
      PD_LOOP(4)
      PD_LOOP(5)
      PD_LOOP(6)
      PD_LOOP(7)
      default:
        break;
    }
#undef PD_SLOT
#undef PD_LOOP
#if PD_PROBE
    if (lane == 0u) {
      const long long pt2 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 32u + warp * 24u), "l"(pt2) : "memory");
    }
#endif
    // The last chunk's runs that the loop makes a chunk late (all three without the readout
    // in the loop).
#pragma unroll
    for (uint32_t r = 0; r < 3u; r++) {
#pragma unroll
      for (uint32_t wr = 0; wr < 2u; wr++) {
        const PdRun rn = pd_run_of(wr, r);
        if ((warp >> 2) == wr && (PD_NO_READOUT || pd_run_late(wr, r, PD_RLAG))) {
          uint32_t x = 0u;
#pragma unroll
          for (uint32_t np = 0; np < 4u; np++) x = fold_part(x, rn.tA, rn.tB, np);
          fold_end(x, trL + ((warp & 3u) * RPS + 2u * rn.grp) * 64u, chunks - 1u);
        }
      }
    }

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
      // Cycles: the tile's prologue, its chunk loop, and its end, summed a warp.
      const long long pt3 = clock64();
      long long pt0, pt1, pt2;
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt0) : "r"(sCnt + 16u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt1) : "r"(sCnt + 24u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt2) : "r"(sCnt + 32u + warp * 24u) : "memory");
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 0u, (unsigned long long)(pt1 - pt0));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 1u, (unsigned long long)(pt2 - pt1));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 2u, (unsigned long long)(pt3 - pt2));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 3u, 1ull);
    }
#else
#error "the miner's hash is not written yet"
#endif
    __syncthreads();
  }
#undef PD_FENCED
#undef PD_TILE
#undef PD_KSTEP
#undef PD_CHUNK
#undef PD_WARPS
#endif
}
