// The tall fold, kept here so pearl_kernel.cu can compile it twice: as pearl_tile_fold_tall,
// and as pearl_tile_fold_tall_pinb, Ampere's flip (PEARL_AMPERE_FLIP), whose B copies take
// the persisting L2 slice instead of A's. pearl_kernel.cu includes it once for each, with
// PEARL_TALL_FN the kernel's name and PEARL_TALL_PIN_B 0 or 1. The second kernel has a body
// only on sm_86 (PEARL_TALL_PIN_B_ARCH). No include guard: it is included twice on purpose.
#if !defined(PEARL_TALL_FN) || !defined(PEARL_TALL_PIN_B)
#error "pearl_fold_tall.cuh: define PEARL_TALL_FN and PEARL_TALL_PIN_B before including it"
#endif
extern "C" __global__ __launch_bounds__(PEARL_TALL_THREADS) void PEARL_TALL_FN(
    const int8_t *__restrict__ Aprime, const int8_t *__restrict__ Bprime,
    uint32_t m, uint32_t n, uint32_t k_arg, uint32_t rank_arg, uint32_t chunks_arg,
    uint32_t col_off, uint32_t rows_valid, uint32_t col_groups, uint32_t tiles,
    const PearlTranscriptTest test, const PearlHitList hits,
    // TMA descriptors for the k-blocked A' and B', read only by Blackwell's build
    // (PEARL_TALL_TMA); Ada's ignores them. Last, so no other parameter moves.
    // PearlTensorMap, not CUtensorMap: cuda.h's alignment is lost under MSVC,
    // which put these at 0x98 and faulted every TMA load (pearl_tensor_map.h).
    const PEARL_GRID_CONSTANT PearlTensorMap tmA, const PEARL_GRID_CONSTANT PearlTensorMap tmB) {
#if PEARL_FOLD_TALL && defined(__CUDA_ARCH__) && PEARL_TALL_BODY_ARCH(__CUDA_ARCH__) \
    && (!PEARL_TALL_PIN_B || PEARL_TALL_PIN_B_ARCH(__CUDA_ARCH__))
  constexpr uint32_t k = PEARL_FOLD_K;
  constexpr uint32_t rank = PEARL_FOLD_RANK;
  constexpr uint32_t chunks = PEARL_FOLD_CHUNKS;
  if (k_arg != k || rank_arg != rank || chunks_arg != chunks) return;
  if (blockDim.x != PEARL_TALL_THREADS) return;
#if PEARL_TALL_TMA_BODY
  (void)m;
  (void)n;
#endif

  constexpr uint32_t MT = PEARL_TALL_ROW_TILES;       // m16 tiles a warp: 96 rows
  constexpr uint32_t NB = 8u;                         // n8 tiles a warp: 64 columns
  constexpr uint32_t RPL = MT / 2u;                   // regions a lane folds into
  constexpr uint32_t WARP_REGIONS = RPL * 8u;         // regions a warp holds: 24
  constexpr uint32_t BM = PEARL_TALL_BM, BN = PEARL_TALL_BN;
  constexpr uint32_t SK = PEARL_TALL_STAGE_K;         // k a stage, and bytes a staged row
  constexpr uint32_t NST = PEARL_TALL_STAGES;
  constexpr uint32_t SPC = rank / SK;                 // stages a chunk
  constexpr uint32_t KS = SK / 32u;                   // k-steps a stage
  constexpr uint32_t STAGE = PEARL_TALL_STAGE_BYTES;
  static_assert(SPC == 2u && KS == 2u && NST == SPC + 1u,
                "the ring issues a chunk ahead, into the buffer the stage before read");
  static_assert(STAGE % 128u == 0u, "stage buffers keep rows 64-byte aligned (lane bases)");
  static_assert(PEARL_ROWS_MASK == 0x1Bu && PEARL_COLS_MASK == 0x39u,
                "the readout folds the regions the m16n8k32 accumulators hold");
  static_assert((chunks <= PEARL_JACKPOT_BUCKETS
                 || (PEARL_TALL_RED_ON && !PEARL_TALL_PREGUARD_ON
                     && chunks % PEARL_JACKPOT_BUCKETS == 0u))
                    && PEARL_JACKPOT_BUCKETS == 16u,
                "one transcript word a chunk, one 64-byte block a region (past 16 chunks, "
                "the shared-XOR readout folds chunk c into word c % 16)");
#if PEARL_TALL_TMA_BODY
  static_assert(PEARL_TALL_SMEM_TMA <= 101376u, "Ada gives a block 99 KB of shared");
#if PEARL_TALL_TMA_ROLES
  // A buffer's rows, then its FULL and EMPTY barriers (PEARL_TALL_TMA_ROLES). 512 bytes
  // apart keeps every buffer on SWIZZLE_64B's 512-byte period (see the lane bases).
  constexpr uint32_t STRIDE = PEARL_TALL_STAGE_STRIDE_TMA;
  static_assert(STRIDE % 512u == 0u && STRIDE >= STAGE + 16u,
                "stage buffers stay 512-byte aligned, with their barriers behind them");
#endif
#else
  // A buffer's rows, then its FULL and EMPTY barriers (see the ring's roles, below).
  constexpr uint32_t STRIDE = PEARL_TALL_STAGE_STRIDE;
  static_assert(STRIDE % 128u == 0u && STRIDE >= STAGE + 16u,
                "stage buffers stay 128-byte aligned, with their barriers behind them");
  static_assert(PEARL_TALL_SMEM <= 101376u, "Ada and Ampere give a block 99 KB of shared");
#endif

  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  // Column slot fastest, as in the eight-warp fold: warps wc and 4 + wc share scheduler
  // wc, stage column slot wc's B between them, and hand their transcripts to warp wc.
  const uint32_t wr = warp >> 2;
  const uint32_t wc = warp & 3u;

  // Tiles are row groups of 12 valid row offsets (192 rows) by column groups of 16 valid
  // column offsets (256 columns). The last row group runs past m (PEARL_TALL_A_ROWS).
  const uint32_t row_groups = (rows_valid + PEARL_TALL_ROW_OFFSETS - 1u) / PEARL_TALL_ROW_OFFSETS;
  const uint32_t col_block_groups = col_groups / PEARL_TALL_COL_OFFSETS;
#if PEARL_TALL_CLUSTER_BODY
  // The cluster build (PEARL_TALL_CLUSTER): the pair walks tiles of (row-group PAIR,
  // column group), both CTAs over the same cluster-level index, and this CTA's row group
  // is twice the pair plus its rank. So the walk's rows are the pairs, and in the odd
  // last pair rank 1's row group is past the last one (its A box is zero-filled and its
  // hasher skips every row; see hash_region). The host passes `tiles` as pair tiles.
  constexpr uint32_t CLUSTER = PEARL_TALL_CLUSTER_SIZE;
  static_assert(CLUSTER == 2u, "the pair walk, the mask and the rank arithmetic are for two CTAs");
  // The rank is not held: pearl_cluster_ctarank() is read where it is needed.
  const uint32_t cluster_idx = blockIdx.x / CLUSTER;
  const uint32_t walk_rows = (row_groups + CLUSTER - 1u) / CLUSTER;
#else
  const uint32_t walk_rows = row_groups;
#endif
  // The eight-warp fold's band walk (PEARL_BLOCK_GROUP, PEARL_FOLD_SERPENTINE), in bands
  // PEARL_TALL_BAND row groups deep. 683 row groups do not fill whole bands, so every
  // band but the last takes the shift and mask and the last one divides.
#if PEARL_TALL_BAND_RT
  // Ampere: the depth the host picked from the L2 (PEARL_AMPERE_BAND_L2_SHARE), a power of
  // two, PEARL_BD_BAND_WORD words past the slot's hit counter.
  const uint32_t band_depth = hits.count[PEARL_BD_BAND_WORD];
  const uint32_t band_shift = (uint32_t)__ffs(band_depth) - 1u;
#else
  static_assert((PEARL_TALL_BAND & (PEARL_TALL_BAND - 1u)) == 0u, "band depth: a power of two");
  constexpr uint32_t band_shift = pearl_popcount_ce(PEARL_TALL_BAND - 1u);
  constexpr uint32_t band_depth = PEARL_TALL_BAND;
#endif
  auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {
    const uint32_t band_blocks = band_depth * col_block_groups;
    uint32_t band, in_band;
    if ((col_block_groups & (col_block_groups - 1u)) == 0u) {
      band = v >> (band_shift + __popc(col_block_groups - 1u));
      in_band = v & (band_blocks - 1u);
    } else {
      band = v / band_blocks;
      in_band = v % band_blocks;
    }
    const uint32_t band_first = band * band_depth;
    if (band_first + band_depth <= walk_rows) {
      rbg_ = band_first + (in_band & (band_depth - 1u));
      cbg_ = in_band >> band_shift;
    } else {
      const uint32_t band_rows = walk_rows - band_first;
      rbg_ = band_first + in_band % band_rows;
      cbg_ = in_band / band_rows;
    }
#if PEARL_FOLD_SERPENTINE
    if (band & 1u) cbg_ = col_block_groups - 1u - cbg_;
#endif
#if PEARL_TALL_CLUSTER_BODY
    rbg_ = rbg_ * CLUSTER + pearl_cluster_ctarank();   // the pair's row group for this CTA
#endif
  };

  // Shared: the three stage buffers (B's 256 columns, then A's 192 rows, 64 bytes each),
  // their FULL and EMPTY barriers, then the transcripts, [region][16 words].
#if PEARL_TALL_TMA_BODY
  // SWIZZLE_64B XORs address bits [4, 6) with bits [7, 9) of the shared address, which
  // is the q ^ ((r >> 1) & 3) the lane bases expect only for a box that starts on a
  // 512-byte boundary. Every stage and operand offset is a multiple of 512 (of 1024
  // without PEARL_TALL_TMA_ROLES, whose barrier pad makes the stride 29184 = 57 x 512),
  // so an aligned base is enough; the prologue traps rather than fold a mis-swizzled
  // tile. The 512-byte layout was verified on an RTX 5090 (verify-hits 359/359).
  // (Its own name: a redeclaration of smem_u32 may not change its alignment.)
  extern __shared__ __align__(1024) uint32_t pearl_tall_smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(pearl_tall_smem);
#if PEARL_TALL_TMA_ROLES
  // Each buffer's FULL and EMPTY in the pad behind its rows.
  const uint32_t sTr = sbase + NST * STRIDE;
#else
  // FULL[3] and EMPTY[3] after the three buffers.
  const uint32_t barFull = sbase + NST * STAGE;
  const uint32_t barEmpty = barFull + 8u * NST;
  const uint32_t sTr = sbase + NST * STAGE + 64u;
#endif
#else
  // Each buffer's FULL and EMPTY in the 128 bytes behind its rows.
  extern __shared__ __align__(128) uint32_t smem_u32[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_u32);
  const uint32_t sTr = sbase + NST * STRIDE;
#endif

#if !PEARL_TALL_TMA_BODY
  // Group staging. Row slot wr's 96 rows are staged by its four warps (tA 0..127), three
  // 16-byte copies a thread a stage, slots 32 rows apart; column slot wc's 64 columns by
  // its two (tB 0..63), four copies, slots 16 columns apart. Whole swizzle bands a slot,
  // so the swizzle is fixed per thread and a slot is a constant offset.
  constexpr uint32_t ASLOTS = 3u, BSLOTS = 4u;
  constexpr uint32_t aStep = 32u, bStep = 16u;
  static_assert(ASLOTS * aStep == BM / 2u && BSLOTS * bStep == 64u && aStep % 8u == 0u
                    && bStep % 8u == 0u, "a group splits evenly, in whole swizzle bands");
  const uint32_t tA = wc * 32u + lane;
  const uint32_t tB = wr * 32u + lane;
  const uint32_t aq = tA & 3u, bq = tB & 3u;
  const uint32_t arow = wr * (BM / 2u) + (tA >> 2);
  const uint32_t bcol = wc * 64u + (tB >> 2);
  uint32_t adst0 = (BN + arow) * SK + 16u * (aq ^ ((arow >> 1) & 3u));
  uint32_t bdst0 = bcol * SK + 16u * (bq ^ ((bcol >> 1) & 3u));
  // Held opaque, as in the eight-warp fold, so ptxas does not rebuild them per copy group.
  asm volatile("" : "+r"(adst0), "+r"(bdst0));
  // In per-tile order (PEARL_TALL_TILE_ORDER) a tile's 192 rows of A' and 256 columns of
  // B' are each one block of k / 64 slabs [rows][64]: a stage of the tile is one slab,
  // one contiguous run, and the next stage the next slab. So a thread's source is one
  // offset an operand for the tile, held as a pointer; a slot is a constant offset from
  // it, and a stage a constant stride, 12 KB of A' and 16 KB of B'. B's pointer is
  // centred on its four slots, which puts all of them in reach of one register's
  // immediate offset. The last row group's rows past m are the zeroed rows
  // PEARL_TALL_A_ROWS pads A' with; the fold hashes no region on them.
  // (k-blocked, [k / 64][rows][64]: the same runs, a stage a k-block on, m * 64 bytes of
  // A' and n * 64 of B'; the rows past m are the next k-block's first rows, or past the
  // last k-block the padding.)
  auto tile_srcs = [&](uint32_t v, uint32_t &bsrc_, uint32_t &asrc_) {
    uint32_t rbg_, cbg_;
    tile_coords(v, rbg_, cbg_);
#ifdef PEARL_ABLATE_DRAM
    // Diagnostic only when defined: every tile copies from row groups 0 and 1 of A' and
    // the first two column groups of B' (768 KB and 1 MB at the mainnet k), which stay
    // in any RTX 30 card's L2, so DRAM serves next to nothing. The same copies and mma,
    // two more instructions a tile. Prices the fold's DRAM traffic (its watts, at the
    // power limit); the output is meaningless. The hash still names the real regions.
    // Measured on Salad (29 October, bench 4 rounds of 45 s, interleaved with base):
    //   RTX 3090, GDDR6X    memory 17% -> 0%, 1605 -> 1665 MHz, 126.9 -> 131.3 TH/s (+3.3%)
    //   RTX 3060 Ti, GDDR6  memory 34% -> 0%, 1778 -> 1845 MHz,  65.4 ->  68.1 TH/s (+4.1%)
    // at the same power, with work per clock unchanged (TH/s per GHz -0.2% and +0.4%;
    // PEARL_TALL_CYC 102,923 -> 102,896 cycles a tile on the 3060 Ti). So the DRAM traffic
    // costs RTX 30 about 3-4% of clock, and nothing per clock: the ceiling for a fold that
    // keeps its operands on chip, as PeakMiner's and SRBMiner's 0-1% memory use suggests
    // theirs now do.
#ifdef PEARL_E4_GENLOAD
    // E4's data path: A from the team's ring (row group 0 of it, the ring's base), B from the
    // n = 256 PEARL_E4_C columns E4 keeps in the L2.
    rbg_ = 0u;
    cbg_ &= PEARL_E4_C - 1u;
#else
    rbg_ &= 1u;
    cbg_ &= 1u;
#endif
    const uint32_t coff = 0u;
#else
    const uint32_t coff = col_off;
#endif
#if PEARL_TALL_TILE_ORDER
    bsrc_ = (coff + cbg_ * PEARL_TALL_COL_OFFSETS) * PEARL_COLS_COUNT * k + bcol * SK + bq * 16u;
    asrc_ = rbg_ * BM * k + arow * SK + aq * 16u;
#else
    bsrc_ = ((coff + cbg_ * PEARL_TALL_COL_OFFSETS) * PEARL_COLS_COUNT + bcol) * SK + bq * 16u;
    asrc_ = (rbg_ * BM + arow) * SK + aq * 16u;
#endif
  };
#if PEARL_TALL_TILE_ORDER
  (void)m;
  (void)n;
  constexpr uint32_t aKB = BM * SK, bKB = BN * SK;     // bytes a stage of the tile
#else
  const uint32_t aKB = m * SK, bKB = n * SK;           // bytes a k-block
#endif
  constexpr uint32_t BMID = BSLOTS / 2u * bStep * SK;   // B's pointer, from its slot 0
  // Stage sg of the tile (0 .. 2 * chunks - 1: its slab, or its k-block), slot p, into
  // the buffer at ib.
#if PEARL_AMPERE_PERSIST_A && !PEARL_TALL_PIN_B && defined(__CUDA_ARCH__) && PEARL_AMPERE_PERSIST_ARCH(__CUDA_ARCH__)
  // Ampere: A's copies evict_last, so they land in the persisting L2 slice the host set
  // aside for the band's A' (PEARL_AMPERE_PERSIST_A).
  uint64_t polA;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(polA));
#define PEARL_TALL_ISSUE_A(ib, sg, p)                                                  \
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"          \
               ::"r"((ib) + adst0 + (p) * aStep * SK),                                  \
               "l"(aP + (sg) * aKB + (p) * aStep * SK), "l"(polA));
#else
#define PEARL_TALL_ISSUE_A(ib, sg, p)                                                  \
  pearl_cp_async16((ib) + adst0 + (p) * aStep * SK, aP + (sg) * aKB + (p) * aStep * SK);
#endif
#if PEARL_TALL_PIN_B && defined(__CUDA_ARCH__) && PEARL_TALL_PIN_B_ARCH(__CUDA_ARCH__)
  // The flip (pearl_tile_fold_tall_pinb, PEARL_AMPERE_FLIP): B's copies evict_last instead,
  // so they land in the slice the host set aside for the batch's B', and A' streams.
  uint64_t polB;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(polB));
#define PEARL_TALL_ISSUE_B(ib, sg, p)                                            \
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"    \
               ::"r"((ib) + bdst0 + (p) * bStep * SK),                           \
               "l"(bP + (sg) * bKB + (int)((p) * bStep * SK) - (int)BMID), "l"(polB));
#else
#define PEARL_TALL_ISSUE_B(ib, sg, p)                                                  \
  pearl_cp_async16((ib) + bdst0 + (p) * bStep * SK,                                    \
                   bP + (sg) * bKB + (int)((p) * bStep * SK) - (int)BMID);
#endif
#else
  // TMA staging (PEARL_TALL_TMA). The producer is lane 0 of warp 4: row slot 1, so it
  // never hashes, and the next tile's third stage does not wait for a hash.
  constexpr uint32_t PRODUCER = 4u * 32u;
  static_assert(PRODUCER / 32u >= 4u, "the producer must not be in a hashing warp (wr 0)");
  static_assert(PEARL_TALL_TMA_FILL_PT <= KS * NB, "a fill point is after an n8 column");
  static_assert(!PEARL_TALL_TMA_PRODUCER2 || (PRODUCER + 32u * (SPC - 1u)) / 32u < 8u,
                "stage 1's producer (PEARL_TALL_TMA_PRODUCER2) is another non-hashing warp");
  static_assert(STAGE % 1024u == 0u && (BN * SK) % 1024u == 0u,
                "every box must start 1024-byte aligned for the swizzle");
  static_assert(BM <= 256u && BN <= 256u, "a TMA box is at most 256 rows");
  (void)Aprime;
  (void)Bprime;
  // A tile's boxes start at its first B column and its first A row: the sources the
  // cp.async walk starts from, as coordinates. Held for the tile (block-uniform).
  uint32_t boxB = 0u, boxA = 0u;
  auto tile_box = [&](uint32_t v, uint32_t &bcol_, uint32_t &arow_) {
    uint32_t rbg_, cbg_;
    tile_coords(v, rbg_, cbg_);
    bcol_ = (col_off + cbg_ * PEARL_TALL_COL_OFFSETS) * PEARL_COLS_COUNT;
    arow_ = rbg_ * BM;
  };
  // Fill buffer bi with k bytes [kofs, kofs + SK) of the tile at (boxB, boxA): its
  // k-block kofs / SK, whole (the host blocks A' and B' by exactly one stage). Both
  // boxes count their full size toward FULL's transaction, the rows past m that TMA
  // zero-fills included.
#if PEARL_TALL_TMA_ROLES
  // (Here bi is the buffer's shared address, a ring role, and FULL sits behind it.)
  auto tma_fill = [&](uint32_t bi, uint32_t kofs) {
    const uint32_t full = bi + STAGE, dst = bi;
#else
  auto tma_fill = [&](uint32_t bi, uint32_t kofs) {
    const uint32_t full = barFull + 8u * bi, dst = sbase + bi * STAGE;
#endif
    pearl_mbar_expect_tx(full, STAGE);
#if PEARL_TALL_CLUSTER_BODY
    // B once for the pair: rank 0's box lands in both CTAs' buffers at this offset and
    // completes on both FULLs at this offset (PEARL_TALL_CLUSTER). The peer's FULL may
    // see those bytes before its own producer's expect_tx above, which takes its
    // tx-count below zero for a while; the mbarrier allows that, and the phase cannot
    // complete before the expect_tx's arrival, so it is how CUTLASS's multicast
    // pipelines run too. Each CTA's own A box follows, as before.
    if (pearl_cluster_ctarank() == 0u)
      pearl_tma_3d_multicast(dst, &tmB, 0u, boxB, kofs / SK, full,
                             (uint16_t)((1u << CLUSTER) - 1u));
#else
    pearl_tma_3d(dst, &tmB, 0u, boxB, kofs / SK, full);
#endif
    pearl_tma_3d(dst + BN * SK, &tmA, 0u, boxA, kofs / SK, full);
  };
#endif

  // Lane bases for buffer 0, k-step 0. The swizzle and the half select touch only bits 4
  // and 5, and every other term is a multiple of 64, so k-step t is (base + stage) ^ 32t
  // plus a constant fragment offset -- the eight-warp fold's lane bases, at 64-byte rows.
  const uint32_t f = (lane >> 1) & 3u;   // (row >> 1) & 3 of every row this lane addresses
  const uint32_t alrow = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
  const uint32_t blcol = (lane & 7u) + ((lane >> 4) & 1u) * 8u;
#if PEARL_TALL_TMA_BODY && !PEARL_TALL_TMA_ROLES
  uint32_t aLane0 = sbase + (BN + wr * (BM / 2u) + alrow) * SK + 16u * (((lane >> 4) & 1u) ^ f);
  uint32_t bLane0 = sbase + (wc * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);
#else
  // Less sbase: the ring's role registers carry it.
  uint32_t aLane0 = (BN + wr * (BM / 2u) + alrow) * SK + 16u * (((lane >> 4) & 1u) ^ f);
  uint32_t bLane0 = (wc * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);
#endif
  asm volatile("" : "+r"(aLane0), "+r"(bLane0));

  int32_t acc[MT][NB][4];
  // The readout, as the eight-warp fold's: the lane's 64 accumulators of m16 tiles 2rl
  // and 2rl + 1 in 32 xor3, three shuffles to the lanes that share the region, and the
  // lane that keeps chunk c (c & 3 == (lane >> 2) & 3) stores the word. Region Lc of the
  // column slot is 24 wr + 6 (lane & 3) + 2 rl + lane bit 4: the warp, the column offset
  // (lane & 3 of 0, 2, 4, 6) and the row offset (0 or 4 in each 32 rows).
  auto fold_pair = [&](uint32_t x, uint32_t rl, uint32_t np) -> uint32_t {
    const int32_t *p0 = acc[2u * rl][2u * np], *p1 = acc[2u * rl + 1u][2u * np];
    const int32_t *p2 = acc[2u * rl][2u * np + 1u], *p3 = acc[2u * rl + 1u][2u * np + 1u];
    const uint32_t ta = pearl_xor3((uint32_t)p0[0], (uint32_t)p0[1], (uint32_t)p0[2]);
    const uint32_t tb = pearl_xor3((uint32_t)p0[3], (uint32_t)p1[0], (uint32_t)p1[1]);
    const uint32_t tc = pearl_xor3((uint32_t)p1[2], (uint32_t)p1[3], (uint32_t)p2[0]);
    const uint32_t td = pearl_xor3((uint32_t)p2[1], (uint32_t)p2[2], (uint32_t)p2[3]);
    const uint32_t te = pearl_xor3(ta, tb, tc);
    const uint32_t rest = np == 0u ? (te ^ td) : pearl_xor3(te, td, x);
    const uint32_t t3 = pearl_xor3((uint32_t)p3[0], (uint32_t)p3[1], (uint32_t)p3[2]);
    return pearl_xor3(t3, (uint32_t)p3[3], rest);
  };
  const uint32_t trBase =
      sTr + (wc * 2u * WARP_REGIONS + wr * WARP_REGIONS + (lane & 3u) * (2u * RPL)
             + ((lane >> 4) & 1u)) * 64u;
  auto readout = [&](uint32_t rl, uint32_t c) {
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);
#if PEARL_TALL_RED_ON
#if PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS
    // k past 2048 (PEARL_FOLD_K): word c % 16 is rotl13(word) ^ chunk c's XOR in the
    // reference, once a chunk. The rotation distributes over XOR, so chunk c's part goes in
    // already rotated once for every later chunk of its word, and the words still start at
    // zero and take XORs only.
    {
      const uint32_t later = PEARL_FOLD_CHUNKS / PEARL_JACKPOT_BUCKETS - 1u - (c >> 4);
      x = __funnelshift_l(x, x, (13u * later) & 31u);
      c &= PEARL_JACKPOT_BUCKETS - 1u;
    }
#endif
    // Every lane XORs its part in. The predicate is always true (lane is never 99), but
    // inside the asm: without one, or with any predicate ptxas can see, it spills 220 bytes.
    asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 99;\n @p red.shared.xor.b32 [%0], %1;\n}"
                 ::"r"(trBase + rl * 128u + c * 4u), "r"(x), "r"(lane) : "memory");
    return;
#endif
    const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4);
    const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8);
    const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12);
    x = pearl_xor3(x, s4, s8) ^ s12;
    if (((c ^ (lane >> 2)) & 3u) == 0u)
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(trBase + rl * 128u + c * 4u), "r"(x)
                   : "memory");
  };

  if (threadIdx.x < NST) {
#if PEARL_TALL_TMA_BODY
    if ((sbase & 1023u) != 0u) __trap();
    // FULL's one arrival is the producer's expect_tx; the boxes' bytes complete it.
    // EMPTY's are the eight warps', or every thread's (PEARL_TALL_TMA_EMPTY_ALL), from
    // both CTAs of the pair in the cluster build (PEARL_TALL_CLUSTER).
#if PEARL_TALL_TMA_ROLES
    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, 1u);
    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, PEARL_TALL_TMA_EMPTY_ARRIVALS);
#else
    pearl_mbar_init(barFull + 8u * threadIdx.x, 1u);
    pearl_mbar_init(barEmpty + 8u * threadIdx.x, PEARL_TALL_TMA_EMPTY_ARRIVALS);
#endif
    // Makes the initialised barriers visible to the async proxy (the TMA unit) too.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    // FULL at +STAGE, EMPTY at +STAGE + 8: every thread arrives on both.
    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, PEARL_TALL_THREADS);
    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, PEARL_TALL_THREADS);
#endif
  }
#if PEARL_TALL_RED_ON
  // The transcripts start at zero: the readout XORs into them (PEARL_TALL_RED_READOUT).
  for (uint32_t o = threadIdx.x * 16u; o < BM * BN / 4u; o += PEARL_TALL_THREADS * 16u)
    asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(sTr + o), "r"(0u) : "memory");
#endif
#if defined(PEARL_E4_GENLOAD) && !PEARL_TALL_TMA_BODY
  // E4's draw tables (see pearl_e4_draw): this block's dense rows, biased to 0 .. 63, and the
  // sixteen perm slices every block draws with (each block writes the same values). A cheap
  // hash: the lookups only have to land where random pairs would.
  if (gridDim.x > PEARL_E4_MAX_BLOCKS) __trap();
  for (uint32_t i = threadIdx.x; i < PEARL_E4_RROWS * 32u; i += PEARL_TALL_THREADS)
    pearl_e4_dense[blockIdx.x * PEARL_E4_RROWS * 32u + i] =
        pearl_e4_mix(blockIdx.x * 0x10000u + i) & 0x3F3F3F3Fu;
  for (uint32_t i = threadIdx.x; i < 16u * 32u; i += PEARL_TALL_THREADS) {
    uint32_t lx = 0u, ly = 0u, sx = 0u, sy = 0u;
#pragma unroll
    for (uint32_t jj = 0; jj < 4u; jj++) {
      const uint32_t h = pearl_e4_mix(0x9E3779B9u + i * 4u + jj);
      const uint32_t p0 = h & 127u, p1 = p0 ^ (1u + ((h >> 8) & 126u));
      // Selector nibbles: bytes 0, 1 of the first PRMT (low half), 2, 3 of the second (high).
      const uint32_t sh = jj < 2u ? 4u * jj : 16u + 4u * jj, hi = (jj & 1u) ? 4u : 0u;
      lx |= (p0 >> 2) << (8u * jj);
      ly |= (p1 >> 2) << (8u * jj);
      sx |= (hi + (p0 & 3u)) << sh;
      sy |= (hi + (p1 & 3u)) << sh;
    }
    pearl_e4_perm[i] = make_uint4(lx, ly, sx, sy);
  }
#endif
  // Publishes the initialised barriers: the fold's only block-wide barrier (the
  // cluster build adds the two cluster barriers, here and before the return).
  __syncthreads();
#if defined(PEARL_E4_GENLOAD) && defined(PEARL_E4_NOGEN) && !PEARL_TALL_TMA_BODY
  // The data path without the draw still needs real-looking operands in the ring: an
  // all-zero A' toggles nothing in the tensor pipe, and on a 3090 ran 150 MHz faster at
  // less power than the same fold on drawn values (29 October). So the ring's three slabs
  // are drawn once, here, by the draw itself.
#pragma unroll 1
  for (uint32_t g = 0; g < 3u; g++)
#pragma unroll 1
    for (uint32_t j = 0; j < PEARL_E4_WORDS; j++) pearl_e4_draw(j, g, lane, warp);
#endif
#if PEARL_TALL_CLUSTER_BODY
  // And across the pair: the peer's barriers exist before this CTA arrives on them or
  // multicasts into its shared memory. Both CTAs take the same early returns below
  // (the geometry ones above took neither here), so neither waits on a CTA that left.
  pearl_cluster_sync();
  if (cluster_idx >= tiles) return;   // whole pair; the host never does this
  const uint32_t tile_stride = gridDim.x / CLUSTER;
  const uint32_t tile_first = cluster_idx;
#else
  if ((uint32_t)blockIdx.x >= tiles) return;   // whole block; the host never does this
  const uint32_t tile_stride = gridDim.x;
  const uint32_t tile_first = blockIdx.x;
#endif
#if PEARL_TALL_TMA_BODY
  tile_box(tile_first, boxB, boxA);
  // The first tile's chunk 0: stages 0 and 1, into buffers 0 and 1, which nothing has
  // read yet.
  if (threadIdx.x == PRODUCER) {
#if PEARL_TALL_TMA_ROLES
#pragma unroll
    for (uint32_t s = 0; s < SPC; s++) tma_fill(sbase + s * STRIDE, s * SK);
#else
#pragma unroll
    for (uint32_t s = 0; s < SPC; s++) tma_fill(s, s * SK);
#endif
  }

#if PEARL_TALL_TMA_ROLES
  // The ring's roles (PEARL_TALL_TMA_ROLES), Ada's: each a buffer's shared address and
  // the parity of its next FULL phase. A is read by the chunk's stage 0, B by its stage
  // 1, and C is refilled by stage 0; at the kernel's start buffers 0, 1 and 2, phase 0.
  uint32_t soA = sbase, soB = sbase + STRIDE, soC = sbase + 2u * STRIDE;
  uint32_t pA = 0u, pB = 0u, pC = 0u;
#else
  // The stage being read: its buffer fb, and the parity of that buffer's use.
  uint32_t fb = 0u, fpar = 0u;
#endif
#else
  uint32_t bsrc0, asrc0;
  tile_srcs(tile_first, bsrc0, asrc0);
  const int8_t *aP = PEARL_E4_ABASE + asrc0, *bP = Bprime + bsrc0 + BMID;
  // The first tile's chunk 0: stages 0 and 1, into buffers 0 and 1.
#pragma unroll
  for (uint32_t s = 0; s < SPC; s++) {
    const uint32_t ib = sbase + s * STRIDE;
#pragma unroll
    for (uint32_t p = 0; p < ASLOTS; p++) PEARL_TALL_ISSUE_A(ib, s, p)
#pragma unroll
    for (uint32_t p = 0; p < BSLOTS; p++) PEARL_TALL_ISSUE_B(ib, s, p)
    pearl_mbar_arrive_copies(ib + STAGE);
  }

  // The ring's roles, each a buffer (with sbase) and the parity of its next FULL phase:
  // A is read by the chunk's stage 0, B by its stage 1, and C is refilled by stage 0. At
  // the kernel's start they are buffers 0, 1 and 2, all at phase 0.
  uint32_t soA = sbase, soB = sbase + STRIDE, soC = sbase + 2u * STRIDE;
  uint32_t pA = 0u, pB = 0u, pC = 0u;
#endif
#if PEARL_TALL_CTR_ON
  // Tiles from the slot's counter (PEARL_TALL_CTR): the next tile is the word below,
  // written in chunk PEARL_TALL_CTR_CHUNK and read from the last chunk on.
  static_assert(PEARL_TALL_CTR_CHUNK >= 1u && PEARL_TALL_CTR_CHUNK + 1u < chunks,
                "written after the tile's first chunk, and read in a later one");
  static_assert(PEARL_TALL_CTR_THREAD / 32u >= 4u && PEARL_TALL_CTR_THREAD < PEARL_TALL_THREADS,
                "the counter is read by a warp that does not hash (wr 1)");
#ifdef PEARL_TALL_CYC
#error "PEARL_TALL_CYC counts the grid stride's tiles: build it without PEARL_TALL_CTR"
#endif
  __shared__ uint32_t sTallNext;
  volatile uint32_t *const nextTile = &sTallNext;
  (void)tile_stride;
#define PEARL_TALL_NEXT (*nextTile)
#else
#define PEARL_TALL_NEXT (v + tile_stride)
#endif
  // (In the cluster build v, tile_stride, has_next and so every stage_next and cn below
  // are cluster-level, never from blockIdx.x: the pair must agree on all of them.)
  for (uint32_t v = tile_first; v < tiles; v = PEARL_TALL_NEXT) {
#if !PEARL_TALL_CTR_ON
    const bool has_next = v + tile_stride < tiles;
#if defined(PEARL_TALL_CYC) && !PEARL_TALL_TMA_BODY
    if (threadIdx.x == 0u) {
      const long long c = clock64();
      const uint32_t t1 = sbase + STAGE + 16u;
      if (v == tile_first + tile_stride) {
        asm volatile("st.shared.u64 [%0], %1;" ::"r"(t1), "l"(c) : "memory");
      } else if (!has_next && v > tile_first + tile_stride) {
        long long c1;
        asm volatile("ld.shared.u64 %0, [%1];" : "=l"(c1) : "r"(t1) : "memory");
        atomicAdd(&pearl_tall_cyc[0], (unsigned long long)(c - c1));
        atomicAdd(&pearl_tall_cyc[1], (unsigned long long)((v - tile_first) / tile_stride - 1u));
      }
    }
#endif
#endif  // !PEARL_TALL_CTR_ON
#if PEARL_TALL_TMA_BODY
    // The next tile's boxes, for the last chunk's fills (its first two stages). Worked
    // out here rather than in that chunk: two registers across the chunk loop, where
    // the tile walk's divides in the loop were seventy instructions of its body.
    uint32_t nextB = 0u, nextA = 0u;
    if (has_next) tile_box(v + tile_stride, nextB, nextA);
#endif
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;

    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
      // The last chunk stages the next tile's chunk 0, from the next tile's sources.
      const bool last = chunk + 1u == chunks;
#if PEARL_TALL_CTR_ON
      // The counter's word, read in the last chunk only (see PEARL_TALL_CTR).
      const bool stage_next = !last || PEARL_TALL_NEXT < tiles;
#else
      const bool stage_next = !last || has_next;
#endif
#if PEARL_TALL_TMA_BODY
      if (last) {
        boxB = nextB;
        boxA = nextA;
      }
      const uint32_t cn = last ? 0u : chunk + 1u;
#pragma unroll
      for (uint32_t s = 0; s < SPC; s++) {
#if PEARL_TALL_TMA_ROLES
        // Stage 0 reads A and refills C, which the stage before it read; stage 1 reads B
        // and refills A. A refill waits for its buffer's previous use to be released: C's
        // is the phase before its next FULL (pC ^ 1), A's the one stage 0 just read (pA).
        // At the kernel's first stage that is the phase before phase 0, which counts as
        // done. Every address is a role plus a constant.
        const uint32_t so = s == 0u ? soA : soB;
        const uint32_t bi = s == 0u ? soC : soA;
        const uint32_t rpar = s == 0u ? pA : pB;
        const uint32_t epar = s == 0u ? (pC ^ 1u) : pA;
        const uint32_t emptyBi = bi + STAGE + 8u, fullSo = so + STAGE, emptySo = so + STAGE + 8u;
#else
        const uint32_t so = fb * STAGE;
        // This stage refills buffer (fb + 2) % 3, which the stage before it read. Stage
        // g = 3 * turn + fb refills it for use (g + 2) / 3, so it waits for that buffer's
        // previous use to be released: parity fpar ^ 1 when fb is 0, fpar otherwise. At
        // the kernel's first stage that is the phase before phase 0, which counts as done.
        const uint32_t bi = fb == 0u ? 2u : fb - 1u;
        const uint32_t epar = fb == 0u ? fpar ^ 1u : fpar;
        // The ring addresses by the names the roles give them (macros: as locals, ptxas
        // 13.3 schedules this build differently).
#define emptyBi (barEmpty + 8u * bi)
#define fullSo (barFull + 8u * fb)
#define rpar fpar
#define emptySo (barEmpty + 8u * fb)
#endif
        // The producer refills bi at the top of the stage, before its warp reads
        // anything, so the boxes have two stages of mma to land under (where the ring on
        // perf/sm120-throughput issued its fills, measured). Its EMPTY wait -- all eight
        // warps have released bi -- is the WAR guard, and the only EMPTY wait but the
        // hand-off's below. (PEARL_TALL_TMA_FILL_PT moves it into the stage body, and
        // PEARL_TALL_TMA_PRODUCER2 gives stage 1's refills to lane 0 of warp 5.)
#ifndef PEARL_ABLATE_RING
#if PEARL_TALL_CLUSTER_BODY
        // Both CTAs have released bi: the phase counts the peer's arrivals too, and the
        // acquire is at cluster scope to pair with them (PEARL_TALL_CLUSTER).
#define PEARL_TALL_REFILL_WAIT pearl_mbar_wait_cluster(emptyBi, epar);
#else
#define PEARL_TALL_REFILL_WAIT pearl_mbar_wait(emptyBi, epar);
#endif
#else
#define PEARL_TALL_REFILL_WAIT
#endif
#define PEARL_TALL_REFILL(prod)                                                        \
  if (threadIdx.x == (prod) && stage_next) {                                           \
    PEARL_TALL_REFILL_WAIT                                                             \
    tma_fill(bi, cn * rank + s * SK);                                                  \
  }
#if PEARL_TALL_TMA_PRODUCER2
#define PEARL_TALL_PRODUCER_S (PRODUCER + 32u * s)
#else
#define PEARL_TALL_PRODUCER_S PRODUCER
#endif
#if PEARL_TALL_TMA_FILL_PT == 0
        PEARL_TALL_REFILL(PEARL_TALL_PRODUCER_S)
#endif
#ifndef PEARL_ABLATE_RING
        // Diagnostic only when defined: no ring waits, so warps read stages that may not
        // have landed and refill ones still being read. Prices the synchronisation; the
        // output is meaningless.
        pearl_mbar_wait(fullSo, rpar);
#endif
#if PEARL_TALL_FRAG_PIPE
        {
          // The streamed stage body (PEARL_TALL_FRAG_PIPE): the same mma, from the same
          // fragments, into the same acc[mb][nb], in the same order -- column-major, m
          // inner, each column's six sharing one B -- with every fragment loaded ahead of
          // its first use. The stage's first B pair and six A behind the FULL wait; B
          // pair p + 1 as pair p starts; each A reloaded for the next k32 step in place,
          // right after its last mma of this one.
          const uint32_t aS = aLane0 + so, bS = bLane0 + so;
          uint32_t ag[MT][4];
          uint32_t bp[2][4];
          pearl_ldmatrix_x4(bp[0][0], bp[0][1], bp[0][2], bp[0][3], bS);
#pragma unroll
          for (uint32_t mb = 0; mb < MT; mb++)
            pearl_ldmatrix_x4(ag[mb][0], ag[mb][1], ag[mb][2], ag[mb][3], aS + mb * 16u * SK);
#pragma unroll
          for (uint32_t t = 0; t < KS; t++) {
            const uint32_t kt = t * 32u;
            const bool step_next = t + 1u < KS;
#pragma unroll
            for (uint32_t p = 0; p < NB / 2u; p++) {
              const uint32_t cur = p & 1u, nxt = cur ^ 1u;
              if (p + 1u < NB / 2u)
                pearl_ldmatrix_x4(bp[nxt][0], bp[nxt][1], bp[nxt][2], bp[nxt][3],
                                  (bS ^ kt) + (p + 1u) * 16u * SK);
              else if (step_next)
                pearl_ldmatrix_x4(bp[nxt][0], bp[nxt][1], bp[nxt][2], bp[nxt][3],
                                  bS ^ (kt + 32u));
#pragma unroll
              for (uint32_t j = 0; j < 2u; j++) {
                const uint32_t nb = 2u * p + j;
#pragma unroll
                for (uint32_t mb = 0; mb < MT; mb++) {
                  pearl_mma_m16n8k32(acc[mb][nb][0], acc[mb][nb][1], acc[mb][nb][2],
                                     acc[mb][nb][3], ag[mb][0], ag[mb][1], ag[mb][2], ag[mb][3],
                                     bp[cur][2u * j], bp[cur][2u * j + 1u]);
                  if (step_next && nb + 1u == NB)
                    pearl_ldmatrix_x4(ag[mb][0], ag[mb][1], ag[mb][2], ag[mb][3],
                                      (aS ^ (kt + 32u)) + mb * 16u * SK);
                }
                // Keeps ptxas from re-sorting the column-major order into A-major runs
                // (see PEARL_TALL_MMA_FENCE_MASK).
                if ((PEARL_TALL_MMA_FENCE_MASK >> nb) & 1u)
                  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#if PEARL_TALL_TMA_FILL_PT
                if (t * NB + nb + 1u == PEARL_TALL_TMA_FILL_PT) {
                  PEARL_TALL_REFILL(PEARL_TALL_PRODUCER_S)
                }
#endif
              }
            }
          }
          // And one after the stage (see PEARL_TALL_STAGE_FENCE).
          if ((PEARL_TALL_STAGE_FENCE >> s) & 1u)
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
#else
        static_assert(PEARL_TALL_TMA_FILL_PT == 0, "a fill point needs the streamed body");
        // Each k32 step loads its six A fragments, then each B pair right before the
        // twelve mma that read it.
#pragma unroll
        for (uint32_t t = 0; t < KS; t++) {
          const uint32_t kt = t * 32u;
          uint32_t af[MT][4];
#pragma unroll
          for (uint32_t mb = 0; mb < MT; mb++)
            pearl_ldmatrix_x4(af[mb][0], af[mb][1], af[mb][2], af[mb][3],
                              ((aLane0 + so) ^ kt) + mb * 16u * SK);
#pragma unroll
          for (uint32_t nb = 0; nb < NB; nb += 2) {
            uint32_t b0, b1, b2, b3;
            pearl_ldmatrix_x4(b0, b1, b2, b3, ((bLane0 + so) ^ kt) + nb * 8u * SK);
#pragma unroll
            for (uint32_t mb = 0; mb < MT; mb++)
              pearl_mma_m16n8k32(acc[mb][nb][0], acc[mb][nb][1], acc[mb][nb][2], acc[mb][nb][3],
                                 af[mb][0], af[mb][1], af[mb][2], af[mb][3], b0, b1);
            if ((PEARL_TALL_MMA_FENCE_MASK >> nb) & 1u)
              asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
            for (uint32_t mb = 0; mb < MT; mb++)
              pearl_mma_m16n8k32(acc[mb][nb + 1][0], acc[mb][nb + 1][1], acc[mb][nb + 1][2],
                                 acc[mb][nb + 1][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3],
                                 b2, b3);
            if ((PEARL_TALL_MMA_FENCE_MASK >> (nb + 1u)) & 1u)
              asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
          }
        }
#endif  // PEARL_TALL_FRAG_PIPE
#if PEARL_TALL_CLUSTER_BODY
        // The cluster build: the same arrival, made twice -- on the peer's EMPTY for this
        // buffer and on its own -- both releases at cluster scope. Rank 0's refill of the
        // buffer lands in both CTAs, so each producer's wait needs both CTAs' reads done.
        // Nothing orders one CTA's two arrivals against the other CTA's, so a late local
        // arrival for one use can be counted toward the phase of the next: the peer may
        // read the next use, and arrive here for it, once its own barrier has this CTA's
        // remote arrivals. That is harmless, by counting. A phase needs every reader of
        // both CTAs (PEARL_TALL_TMA_EMPTY_ARRIVALS). A CTA with a reader still on a use
        // has made fewer than its share of arrivals for it, remote and local alike, and
        // none for a later use; so neither CTA's phase for that use can complete, neither
        // producer refills the buffer, no CTA reads the next use, and no arrival for the
        // next use exists yet. That rests on two things: every refill of a buffer, in
        // either CTA, sits behind an EMPTY wait, and a CTA reads a use only after its own
        // producer's expect_tx and refill. The remote arrive is issued first by
        // preference; the order is not what makes the ring safe.
#define PEARL_TALL_EMPTY_ARRIVE(bar)                       \
  pearl_mbar_arrive_remote(pearl_mapa((bar), pearl_cluster_ctarank() ^ 1u)); \
  pearl_mbar_arrive_cluster(bar);
#else
#define PEARL_TALL_EMPTY_ARRIVE(bar) pearl_mbar_arrive(bar);
#endif
#if PEARL_TALL_TMA_EMPTY_ALL
        // This thread is done reading the stage: its own arrival, a release, which the
        // producer's EMPTY wait acquires before it refills the buffer. PTX orders every
        // ldmatrix this thread made of the stage before the arrival, so no TMA write can
        // overtake a read (PEARL_TALL_TMA_EMPTY_ALL).
        PEARL_TALL_EMPTY_ARRIVE(emptySo)
#else
        // This warp is done reading the stage: one arrival for the whole warp, after a
        // __syncwarp that orders every lane's ldmatrix before it.
        __syncwarp();
        if (lane == 0u) {
          PEARL_TALL_EMPTY_ARRIVE(emptySo)
        }
#endif
#undef PEARL_TALL_EMPTY_ARRIVE
#undef PEARL_TALL_REFILL_WAIT
#undef PEARL_TALL_REFILL
#undef PEARL_TALL_PRODUCER_S
#if PEARL_TALL_TMA_ROLES
      }
#if PEARL_TALL_PREGUARD_ON
      // The readout's XOR folds, into registers, before the chunk-0 guard below
      // (PEARL_TALL_PREGUARD): only its three shared XORs at the chunk's end write the
      // transcripts the guard protects, so only they need to wait for it.
      uint32_t xg[RPL];
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) {
        uint32_t x = 0u;
#pragma unroll
        for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);
        xg[rl] = x;
      }
#endif
#ifndef PEARL_ABLATE_RING
      // The hand-off's ordering (see below): the tile's stage 0 read role A, at parity pA.
      // (In the cluster build that phase also counts the peer's arrivals, so this waits
      // for the peer's stage 0 reads too: stricter than it needs, and still correct.)
      if (chunk == 0u && wr != 0u) pearl_mbar_wait(soA + STAGE + 8u, pA);
#endif
      // The next chunk starts on C: the roles rotate, and A and B have each been used once.
      {
        const uint32_t tso = soC, tpar = pC;
        soC = soB;
        pC = pB ^ 1u;
        soB = soA;
        pB = pA ^ 1u;
        soA = tso;
        pA = tpar;
      }
#else
#undef emptyBi
#undef fullSo
#undef rpar
#undef emptySo
        if (++fb == NST) {
          fb = 0u;
          fpar ^= 1u;
        }
      }
#endif
#if !PEARL_TALL_TMA_ROLES && !defined(PEARL_ABLATE_RING)
      // The hand-off's ordering. Under TMA only the producer waits on EMPTY, so
      // nothing else would stop warp wr = 1 writing its chunk 0 words over the ones
      // its column slot's hasher may still be reading. It waits for the tile's stage 0
      // to be released by all eight warps, which the hasher does only after hashing.
      // That stage is two back: buffer (fb + 1) % 3, in the use whose parity is fpar
      // unless the ring wrapped since (fb 0 or 1). (In the cluster build the phase also
      // counts the peer's arrivals: stricter than it needs, and still correct.)
      if (chunk == 0u && wr != 0u)
        pearl_mbar_wait(barEmpty + 8u * (fb == 2u ? 0u : fb + 1u), fb == 2u ? fpar : fpar ^ 1u);
#endif
#else
#if PEARL_TALL_CTR_ON
      if (last && stage_next) {
#else
      if (last && has_next) {
#endif
        tile_srcs(PEARL_TALL_NEXT, bsrc0, asrc0);
        aP = PEARL_E4_ABASE + asrc0;
        bP = Bprime + bsrc0 + BMID;
      }
      // The first stage the chunk's refills carry, within its tile. Opaque, so it is made
      // once here rather than again in every copy group (0.35% in the bench).
#ifdef PEARL_E4_GENLOAD
      // From the ring: chunk c's slab is slot c % 3.
      uint32_t sg0 = ((last ? 0u : chunk + 1u) % 3u) * SPC;
#elif defined(PEARL_ABLATE_DRAM) && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS
      // The ablation past 2048 of k (PEARL_FOLD_K): the copies wrap every 16 chunks, so the
      // operands they read stay 2048 deep, 1.75 MB, in the L2 at any k. That is what a fold
      // that draws its operands on chip reads (nothing from DRAM), at the large k PeakMiner
      // and SRBMiner mine with (32768 and 65536; their shares, 29 October).
      uint32_t sg0 = ((last ? 0u : chunk + 1u) & 15u) * SPC;
#else
      uint32_t sg0 = (last ? 0u : chunk + 1u) * SPC;
#endif
      asm volatile("" : "+r"(sg0));
#pragma unroll
      for (uint32_t s = 0; s < SPC; s++) {
        // Stage 0 reads A and refills C, which the stage before it read; stage 1 reads B
        // and refills A. A refill waits for its buffer's previous use to be released: C's
        // is the phase before its next FULL (pC ^ 1), A's the one stage 0 just read (pA).
        // At the kernel's first stage that is the phase before phase 0, which counts as
        // done.
        const uint32_t so = s == 0u ? soA : soB;
        const uint32_t ib = s == 0u ? soC : soA;
        const uint32_t rpar = s == 0u ? pA : pB;
        const uint32_t epar = s == 0u ? (pC ^ 1u) : pA;
        const uint32_t sg = sg0 + s;   // the stage the refill carries
#ifndef PEARL_ABLATE_RING
        // Diagnostic only when defined: no ring waits, so warps read stages that may not
        // have landed and refill ones still being read. Prices the synchronisation; the
        // output is meaningless.
        pearl_mbar_wait(so + STAGE, rpar);
#endif
#pragma unroll
        for (uint32_t t = 0; t < KS; t++) {
          const uint32_t kt = t * 32u;
          uint32_t bfr[NB / 2u][4];
#pragma unroll
          for (uint32_t nb2 = 0; nb2 < NB / 2u; nb2++)
            pearl_ldmatrix_x4(bfr[nb2][0], bfr[nb2][1], bfr[nb2][2], bfr[nb2][3],
                              ((bLane0 + so) ^ kt) + nb2 * 16u * SK);
#pragma unroll
          for (uint32_t mb = 0; mb < MT; mb++) {
            uint32_t a0, a1, a2, a3;
            pearl_ldmatrix_x4(a0, a1, a2, a3, ((aLane0 + so) ^ kt) + mb * 16u * SK);
#pragma unroll
            for (uint32_t nb = 0; nb < NB; nb++)
              pearl_mma_m16n8k32(acc[mb][nb][0], acc[mb][nb][1], acc[mb][nb][2], acc[mb][nb][3],
                                 a0, a1, a2, a3, bfr[nb / 2u][(nb & 1u) * 2u],
                                 bfr[nb / 2u][(nb & 1u) * 2u + 1u]);
            // The next chunk's copies for this stage (see PEARL_TALL_APT): A after m16
            // tile 3 of k-step 0, behind the EMPTY wait; B after tile 2 of k-step 1, then
            // the arrival that counts them.
            const uint32_t pt = t * MT + mb;
            static_assert(PEARL_TALL_APT < PEARL_TALL_BPT && PEARL_TALL_BPT < KS * MT,
                          "A's copies, with the EMPTY wait, go out before B's");
            if (pt == PEARL_TALL_APT && stage_next) {
#ifndef PEARL_ABLATE_RING
              pearl_mbar_wait(ib + STAGE + 8u, epar);
#endif
#if defined(PEARL_E4_GENLOAD) && defined(PEARL_E4_FLAGS)
              // E4's acquire of the ring slab these copies read (PEARL_E4_FLAGS): the load
              // and its ordering, without the wait (the values are not the real ones).
              if (s == 0u) {
                if (lane == 0u) {
                  uint32_t f;
                  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f)
                               : "l"(pearl_e4_flag + (blockIdx.x / PEARL_E4_C) * 3u + sg0 / SPC) : "memory");
                  asm volatile("" ::"r"(f));
                }
                __syncwarp();
              }
#endif
#pragma unroll
              for (uint32_t p = 0; p < ASLOTS; p++) PEARL_TALL_ISSUE_A(ib, sg, p)
            }
            if (pt == PEARL_TALL_BPT && stage_next) {
#pragma unroll
              for (uint32_t p = 0; p < BSLOTS; p++) PEARL_TALL_ISSUE_B(ib, sg, p)
              pearl_mbar_arrive_copies(ib + STAGE);
            }
#if defined(PEARL_E4_GENLOAD) && !defined(PEARL_E4_NOGEN)
            // E4's draw, two chunks ahead: this thread's words of the chunk spread over the
            // chunk's four k-step ends, where the step's B fragments are dead.
            if (mb + 1u == MT) {
#pragma unroll
              for (uint32_t j = 0; j < PEARL_E4_WORDS; j++)
                if (j * 4u / PEARL_E4_WORDS == s * KS + t) pearl_e4_draw(j, chunk + 2u, lane, warp);
#ifdef PEARL_E4_FLAGS
              // After this warp's last word of the chunk: its release on the slab's flag.
              if ((PEARL_E4_WORDS - 1u) * 4u / PEARL_E4_WORDS == s * KS + t) {
                __syncwarp();
                if (lane == 0u)
                  asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(
                                   pearl_e4_flag + (blockIdx.x / PEARL_E4_C) * 3u + (chunk + 2u) % 3u)
                               : "memory");
              }
#endif
            }
#endif
          }
        }
#if PEARL_TALL_CTR_ON
        // The next tile, before this stage's release, which every thread acquires at the
        // stage-1 refill (PEARL_TALL_CTR).
        if (s == 0u && chunk == PEARL_TALL_CTR_CHUNK && threadIdx.x == PEARL_TALL_CTR_THREAD)
          *nextTile = gridDim.x + atomicAdd(hits.count + PEARL_BD_CTR_SLOTS, 1u);
#endif
#if defined(PEARL_ABLATE_RING) && defined(PEARL_ABLATE_ARRIVE)
        // Diagnostic only: with the ring's waits gone, its EMPTY arrivals (a MEMBAR.ALL.CTA
        // and an arrive a stage) go too. Prices them; the output is meaningless.
#else
        // This thread is done reading the stage: a release (see the ring above).
        pearl_mbar_arrive(so + STAGE + 8u);
#endif
      }
      // The next chunk starts on C: the roles rotate, and A and B have each been used once.
      {
        const uint32_t tso = soC, tpar = pC;
        soC = soB;
        pC = pB ^ 1u;
        soB = soA;
        pB = pA ^ 1u;
        soA = tso;
        pA = tpar;
      }
#endif
#if PEARL_TALL_PREGUARD_ON
      // The folds were made before the guard; every lane XORs its part in (see readout).
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++)
        asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 99;\n @p red.shared.xor.b32 [%0], %1;\n}"
                     ::"r"(trBase + rl * 128u + chunk * 4u), "r"(xg[rl]), "r"(lane) : "memory");
#elif defined(PEARL_ABLATE_READOUT)
      // Diagnostic only: the readout at the tile's last chunk alone, so the accumulators
      // stay live (ptxas deletes mma whose results nothing reads). Prices the per-chunk
      // readout. Output is meaningless.
      if (chunk + 1u == chunks) {
#pragma unroll
        for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunk);
      }
#else
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunk);
#endif
    }

#if PEARL_TALL_UNFUSED_ON
    // GA100 (PEARL_TALL_UNFUSED): no hasher and no hand-off. Each warp copies the 24
    // regions it reads out (lanes L, L ^ 4, L ^ 8, L ^ 12 hold one region, a quarter each)
    // to this slot's buffer, indexed by the region's batch-local number as a hit would be,
    // and zeroes them for the next tile's XORs. Only this warp writes these words, so the
    // __syncwarp pair orders its lanes' XORs, reads and zeroing; pearl_tall_hash80 hashes
    // the buffer after the launch. The last row group's rows past m are zeroed, not stored.
    {
      uint32_t uv = v, urbg, ucbg;
      asm volatile("" : "+r"(uv));
      tile_coords(uv, urbg, ucbg);
      uint4 *const trg = *reinterpret_cast<uint4 *const *>(&tmA);
      const uint32_t uq = (lane >> 2) & 3u;
      const uint32_t urow = (urbg * 2u + wr) * (2u * RPL) + ((lane >> 4) & 1u);
      const uint32_t ucol = (ucbg * 4u + wc) * 4u + (lane & 3u);
      __syncwarp();
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) {
        const uint32_t sa = trBase + rl * 128u + uq * 16u;
        uint4 w;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w) : "r"(sa) : "memory");
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(sa), "r"(0u) : "memory");
        const uint32_t row_idx = urow + 2u * rl;
        if (row_idx < rows_valid) trg[((size_t)ucol * rows_valid + row_idx) * 4u + uq] = w;
      }
      __syncwarp();
    }
#else
    // Hand-off: the column slot's two warps (one scheduler) meet, so every chunk-15 word
    // is in shared, and the first hashes the slot's 48 regions -- all 32 lanes, then 16.
    pearl_bar_sync(1u + wc, 64u);
#if PEARL_TALL_HASH_PAIRS_ON
    // The same test, made warp-uniform by a vote: the shuffles of the paired hash below
    // sit inside it, and behind a plain wr == 0 branch ptxas fences every ring wait of the
    // tile loop with convergence barriers to reach them (-1.2% a tile, g40-042).
    if (__any_sync(0xffffffffu, wr == 0u)) {
#else
    if (wr == 0u) {
#endif
      // The tile's coordinates again, from an opaque copy of v (not held through the tile).
      uint32_t hv = v;
      asm volatile("" : "+r"(hv));
      uint32_t hrbg, hcbg;
      tile_coords(hv, hrbg, hcbg);
      auto hash_region = [&](uint32_t Lc) {
        const uint32_t owr = Lc / WARP_REGIONS, rem = Lc % WARP_REGIONS;
        const uint32_t ocb = rem / (2u * RPL), oreg = rem % (2u * RPL);
        const uint32_t row_idx = (hrbg * 2u + owr) * (2u * RPL) + oreg;
#if !PEARL_TALL_RED_ON
        if (row_idx >= rows_valid) return;   // the last row group's rows past m
#endif
        uint32_t tm[16];
        const uint32_t ra = sTr + (wc * 2u * WARP_REGIONS + Lc) * 64u;
#pragma unroll
        for (uint32_t q = 0; q < 4; q++)
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(tm[4 * q]), "=r"(tm[4 * q + 1]), "=r"(tm[4 * q + 2]),
                         "=r"(tm[4 * q + 3])
                       : "r"(ra + 16u * q)
                       : "memory");
#if PEARL_TALL_RED_ON
        // Read: zero it for the next tile's XORs, the rows past m included.
#pragma unroll
        for (uint32_t q = 0; q < 4; q++)
          asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(ra + 16u * q), "r"(0u)
                       : "memory");
        if (row_idx >= rows_valid) return;   // the last row group's rows past m
#endif
#ifdef PEARL_ABLATE_TRANSCRIPT_HASH
        return;   // diagnostic only: prices the hashing; no hit is ever reported
#endif
        const uint32_t region = ((hcbg * 4u + wc) * 4u + ocb) * rows_valid + row_idx;
        if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;
        uint32_t h[8];
        pearl_transcript_hash_again(test.key, tm, h);
        if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;
        const uint32_t slot = atomicAdd(hits.count, 1u);
        if (slot >= PEARL_MAX_HITS) return;
        hits.index[slot] = region;
#pragma unroll
        for (int i = 0; i < 8; i++) hits.hash[slot * 8u + i] = h[i];
#pragma unroll
        for (int i = 0; i < 16; i++) hits.transcript[slot * 16u + i] = tm[i];
      };
#if !PEARL_TALL_HASH_PAIRS_ON
      hash_region(lane);
      if (lane < 2u * WARP_REGIONS - 32u) hash_region(32u + lane);
#else
      static_assert(2u * WARP_REGIONS == 48u, "1.5 regions a lane");
      // Lane L hashes region L whole and, with lane L ^ 1, half of region 32 + L / 2. Both
      // transcripts are read, hashed, and zeroed for the next tile's XORs only after the
      // whole warp is done with them; the rare hit reads its transcript back from shared,
      // so neither stays live through the compressions (holding them spills 8 bytes).
      const uint32_t Lp = 32u + (lane >> 1), hh = lane & 1u;
      const uint32_t ra = sTr + (wc * 2u * WARP_REGIONS + lane) * 64u;
      const uint32_t rp = sTr + (wc * 2u * WARP_REGIONS + Lp) * 64u;
      uint32_t tm[16], tp[16];
#pragma unroll
      for (uint32_t q = 0; q < 4; q++)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(tm[4 * q]), "=r"(tm[4 * q + 1]), "=r"(tm[4 * q + 2]), "=r"(tm[4 * q + 3])
                     : "r"(ra + 16u * q)
                     : "memory");
      const uint32_t w1 = pearl_hp_msw(test.key, tm, test.hash_big_endian);
#pragma unroll
      for (uint32_t q = 0; q < 4; q++)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(tp[4 * q]), "=r"(tp[4 * q + 1]), "=r"(tp[4 * q + 2]), "=r"(tp[4 * q + 3])
                     : "r"(rp + 16u * q)
                     : "memory");
      const uint32_t w2 = pearl_hp_msw_pair(test.key, tp, hh, test.hash_big_endian);
      auto record = [&](uint32_t Lc) {
        const uint32_t owr = Lc / WARP_REGIONS, rem = Lc % WARP_REGIONS;
        const uint32_t ocb = rem / (2u * RPL);
        const uint32_t row_idx = (hrbg * 2u + owr) * (2u * RPL) + rem % (2u * RPL);
        if (row_idx >= rows_valid) return;   // the last row group's rows past m
        uint32_t t[16];
        const uint32_t rr = sTr + (wc * 2u * WARP_REGIONS + Lc) * 64u;
#pragma unroll
        for (uint32_t q = 0; q < 4; q++)
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(t[4 * q]), "=r"(t[4 * q + 1]), "=r"(t[4 * q + 2]), "=r"(t[4 * q + 3])
                       : "r"(rr + 16u * q)
                       : "memory");
        uint32_t h[8];
        pearl_transcript_hash_again(test.key, t, h);
        if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;
        const uint32_t region = ((hcbg * 4u + wc) * 4u + ocb) * rows_valid + row_idx;
        const uint32_t slot = atomicAdd(hits.count, 1u);
        if (slot >= PEARL_MAX_HITS) return;
        hits.index[slot] = region;
#pragma unroll
        for (int i = 0; i < 8; i++) hits.hash[slot * 8u + i] = h[i];
#pragma unroll
        for (int i = 0; i < 16; i++) hits.transcript[slot * 16u + i] = t[i];
      };
      if (wr == 0u) {
#ifndef PEARL_ABLATE_TRANSCRIPT_HASH
        if (w1 <= test.target_w[0]) record(lane);
        if (hh == (test.hash_big_endian ? 0u : 1u) && w2 <= test.target_w[0]) record(Lp);
#endif
      }
      __syncwarp();   // every lane's reads (hit re-reads too) of both regions before any zeroing
#pragma unroll
      for (uint32_t q = 0; q < 4; q++) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(ra + 16u * q), "r"(0u) : "memory");
        if (hh == 0u)
          asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(rp + 16u * q), "r"(0u) : "memory");
      }
#endif
    }
#if defined(PEARL_E4_GENLOAD) && defined(PEARL_E4_HASH)
    else {
      // E4's dense rows (PEARL_E4_HASH): the block's PEARL_E4_RROWS rows of a later row
      // group, rank / 32 = 4 BLAKE3 a row, hashed by the 128 lanes of the warps that do not
      // hash transcripts, while they would otherwise wait for the hasher. Their accumulators
      // are dead here, as the hasher's are.
      for (uint32_t hi = (warp - 4u) * 32u + lane; hi < PEARL_E4_RROWS * 4u; hi += 128u) {
        uint8_t lab[32];
#pragma unroll
        for (int i = 0; i < 32; i++) lab[i] = 0;
        uint8_t d[32];
        pearl_random_hash(test.key, lab, v * (PEARL_E4_RROWS * 4u) + hi, 0, d);
        uint32_t *const dst = pearl_e4_dense + (blockIdx.x * PEARL_E4_RROWS + (hi >> 2)) * 32u + (hi & 3u) * 8u;
#pragma unroll
        for (int i = 0; i < 8; i++)
          dst[i] = ((uint32_t)d[4 * i] | ((uint32_t)d[4 * i + 1] << 8) | ((uint32_t)d[4 * i + 2] << 16)
                    | ((uint32_t)d[4 * i + 3] << 24)) & 0x3F3F3F3Fu;
      }
    }
#endif
#endif  // PEARL_TALL_UNFUSED_ON
  }
#if PEARL_TALL_CLUSTER_BODY
  // No CTA leaves while its peer may still arrive on its EMPTY barriers: the peer's last
  // stage's remote arrivals are ordered before its arrive here. (Every multicast into this
  // CTA landed before it read that stage, so none is in flight.)
  pearl_cluster_sync();
#endif
#undef PEARL_TALL_ISSUE_A
#undef PEARL_TALL_ISSUE_B
#undef PEARL_TALL_NEXT
#else
  (void)Aprime; (void)Bprime; (void)m; (void)n; (void)k_arg; (void)rank_arg; (void)chunks_arg;
  (void)col_off; (void)rows_valid; (void)col_groups; (void)tiles; (void)test; (void)hits;
  (void)tmA; (void)tmB;
#endif
}
