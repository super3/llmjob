// Turing's B-direct fold (PEARL_TURING_BDIRECT): the sm_75 body of
// pearl_tile_fold_wmma. pearl_kernel.cu includes this only in the sm_75 pass
// that builds it (PEARL_TURING_BD_BODY); see PEARL_TURING_BDIRECT in
// pearl_config.h for why it exists and what it measured.
//
// The tile is the 128x256 one, eight 64x64 warp tiles: warp w is row slot
// w >> 2 and column slot w & 3, so each scheduler (w & 3) holds one warp of
// each row slot, and those two warps read the same 64 columns of B.
//
// B never goes through shared. The host writes B' fragment-ordered
// (pearl_materialize16_frag):
//   [n / 16][k / 32][lane 32][b0 b1 b2 b3]   16 bytes a lane, 512 a block-step,
// which is what one ldmatrix.x4 would hand each lane. A warp loads its four B
// pairs of a k-step with four LDG.128 (ld.global.nc), one k-step ahead, into
// registers. The second warp of a scheduler reads the same lines, which should
// mostly hit L1 (not measured).
//
// A' stays row-major and goes through two 16 KB shared stages: chunk c lives in
// stage c & 1. A thread loads its four A slots of chunk c + 1 (ld.global.cg, L2
// only, so L1 is left to B) at k-step 0 of chunk c and stores them at k-step 2
// into the other stage. That stage was last read in chunk c - 1, before the
// barrier that ended it, so a chunk needs one barrier and no read barrier, and
// nothing is stored in the seam.
//
// Transcripts go to 8 KB of shared after the A stages, one word a region a
// chunk. Held in registers like the other folds' they spilled.
//
// The chunk loop is rotated: an iteration is k-steps 2-3 of chunk c, its
// barrier and readout, then k-steps 0-1 of chunk c + 1. The last chunk is
// peeled, so nothing in the loop is conditional. Unrotated, ptxas sank the next
// chunk's first B load down to the barrier, because the mma that read it were
// across a branch.
//
// The seam. ptxas places a barrier just before the next shared access after it,
// not where the source puts it, so with the barrier after k-step 3 it sat at the
// end of the readout, and the next chunk's first mma waited on the barrier, then
// on its ldmatrix. So the barrier goes in k-step 3, after its ldmatrix (the
// last reads of chunk c's stage) and before its mma, and k-step 0 of chunk c + 1
// loads its A there, behind the barrier, into its own registers (afn): those
// loads are the shared access that pins the barrier, and their latency hides
// under k-step 3's 128 mma. The readout's words are stored after that k-step's
// ldmatrix, so the loads do not queue behind stores that wait on the shuffles.
// 2080 Ti, bench, three rounds each: at the 260 W cap 85.1-86.1 -> 86.1-86.7
// TH/s (+1.0 to 1.2%), and on a card held at 1095 MHz 60.6 -> 62.8 (+3.6%);
// 253 registers.
//
// PEARL_BD_GROUPS 2 (default): each row slot is a group of four warps. It
// stages its own 64 rows of A, waits on its own 128-thread named barrier, and
// hashes its own 64 regions, so the two row slots' chunk seams need not line
// up. Nothing else is shared between them. The barrier id is 1 + the row slot,
// a register, so ptxas reserves all 16 barriers; the fold runs one block an SM
// anyway (238 registers). PEARL_BD_GROUPS 1: the whole block stages A and
// meets at one __syncthreads.
//
// Persistent (PEARL_BD_PERSIST, see pearl_config.h): one block an SM, tiles
// handed out by a counter. With PEARL_BD_PERSIST=0 the host launches one block
// per tile; a smaller grid still works then (each block walks tiles a grid
// apart and restages chunk 0 of each).
//
// The ablation macros price here what they price in the other folds, where
// they apply: PEARL_ABLATE_BARRIER (the barrier a chunk), _STAGING (A's copies
// in the chunk loop), _LDMATRIX (A's), _TRANSCRIPT, _READOUT and
// _TRANSCRIPT_HASH. PEARL_ABLATE_BLOAD is this fold's own: B is loaded once a
// tile instead of every k-step, so the mma stay and the B traffic goes. The
// other folds' feature switches below are not built here, so asking for one is
// an error rather than a build that ignores it.

#if PEARL_FOLD_GROUP_STAGE || PEARL_FOLD_LANE_BASES || PEARL_FOLD_FAST_COORDS \
    || PEARL_FOLD_SERPENTINE || PEARL_STAGE_AT_BARRIER || !PEARL_STAGE_REGS
#error "PEARL_FOLD_GROUP_STAGE, _LANE_BASES, _FAST_COORDS, _SERPENTINE, PEARL_STAGE_AT_BARRIER and PEARL_STAGE_REGS=0 are not in the B-direct fold: build with PEARL_TURING_BDIRECT=0"
#endif

#if PEARL_BD_GROUPS == 2
#define PEARL_BD_BAR() pearl_bar_sync(1u + wr, 128u)
#else
#define PEARL_BD_BAR() __syncthreads()
#endif

__device__ __forceinline__ void pearl_st_shared_v4(uint32_t a, int4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

// A B fragment, ld.global.nc. Volatile and ordered against memory, as it was
// measured.
__device__ __forceinline__ int4 pearl_bd_ldg(const int8_t *p) {
  int4 v;
  asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// An A staging load. Each A byte is read by one thread, so it goes L2-only
// (ld.global.cg) and leaves L1 to the B fragments.
__device__ __forceinline__ int4 pearl_bd_lda(const int8_t *p) {
  return __ldcg(reinterpret_cast<const int4 *>(p));
}

__device__ __forceinline__ void pearl_fold_bd(const int8_t *__restrict__ Aprime,
                                              const int8_t *__restrict__ Bprime, uint32_t col_off,
                                              uint32_t rows_valid, uint32_t tiles,
                                              const PearlTranscriptTest &test,
                                              const PearlHitList &hits) {
  constexpr uint32_t k = PEARL_FOLD_K, rank = PEARL_FOLD_RANK, chunks = PEARL_FOLD_CHUNKS;
  constexpr uint32_t MB = 4u, NB = 8u, KSTEPS = rank / 32u;
  constexpr uint32_t ROWS = 128u;
  constexpr uint32_t ABUF = ROWS * 128u;             // one A stage: 16 KB
  constexpr uint32_t FRAG_BLOCK = (k / 32u) * 512u;  // one 16-column block of B': 32 KB
  // The host sizes and launches this fold from the wide geometry's names and
  // PEARL_BD_A_STAGES (pearl_host.cu); this is the layout those names describe.
  static_assert(PEARL_FOLD_THREADS == 256u && PEARL_WARP_ROWS == 2 && PEARL_WMMA_ROW_TILES == 4
                    && PEARL_WMMA_COL_BLK == 4 && PEARL_WMMA_ROWS == 16 && PEARL_ROWS_COUNT == 16,
                "the B-direct fold is eight 64x64 warp tiles over 128x256");
  static_assert(rank == PEARL_SB_STRIDE && KSTEPS == 4u, "a staged A row is one 128-byte chunk");
  static_assert(chunks == PEARL_JACKPOT_BUCKETS, "one transcript word a region a chunk");
  static_assert(PEARL_BD_A_STAGES == 2u && ROWS == PEARL_WARP_ROWS * PEARL_WMMA_ROW_TILES * 16u,
                "two A stages of the tile's rows");
  static_assert(PEARL_ROWS_MASK == 0x1Bu && PEARL_COLS_MASK == 0x39u,
                "the readout folds the regions the m16n8k32 accumulators hold");
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const uint32_t wr = warp >> 2, wc = warp & 3u;
  constexpr uint32_t regions_per_warp = 4u, warp_regions = 16u;
  const uint32_t row_blocks = rows_valid / regions_per_warp;
  const uint32_t row_block_groups = row_blocks / 2u;
  const uint32_t col_block_groups = tiles / row_block_groups;
  // The other folds' general walk (see PEARL_BLOCK_GROUP): bands of row groups,
  // a band's rows first, the band as deep as this card's L2 suits: the host
  // leaves it PEARL_BD_BAND_WORD words past the hit counter (PEARL_BD_L2_SHARE).
  const uint32_t band_depth = __ldg(hits.count + PEARL_BD_BAND_WORD);
  const uint32_t band_blocks = band_depth * col_block_groups;
  auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {
    const uint32_t band = v / band_blocks;
    const uint32_t in_band = v % band_blocks;
    const uint32_t band_first = band * band_depth;
    const uint32_t band_rows = row_block_groups - band_first < band_depth
                                   ? row_block_groups - band_first
                                   : band_depth;
    rbg_ = band_first + in_band % band_rows;
    cbg_ = in_band / band_rows;
  };

  extern __shared__ __align__(128) uint32_t smem_u32[];
  const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_u32);
  uint32_t *sT = smem_u32 + PEARL_BD_A_STAGES * ABUF / 4u;  // the transcripts, after the A stages
  static_assert(PEARL_BD_A_STAGES * ABUF + (PEARL_FOLD_THREADS / 32u) * warp_regions
                        * PEARL_JACKPOT_BUCKETS * 4u == 40960u,
                "40 KB of shared: the host's smem for this fold");

  // A staging: eight threads a row, sixteen bytes each, rows SSTEP apart. Unit q
  // of row r is stored at q ^ (r & 7), as in the other folds; SSTEP is a
  // multiple of eight, so the XOR is the same for all of a thread's slots.
#if PEARL_BD_GROUPS == 2
  constexpr uint32_t STHREADS = 128u;
  const uint32_t sid = threadIdx.x & 127u;
  const uint32_t srow0 = wr * 64u + sid / 8u;
#else
  constexpr uint32_t STHREADS = 256u;
  const uint32_t sid = threadIdx.x;
  const uint32_t srow0 = sid / 8u;
#endif
  constexpr uint32_t SSTEP = STHREADS / 8u;
  constexpr uint32_t ASLOTS = (PEARL_BD_GROUPS == 2 ? 64u : 128u) / SSTEP;
  static_assert(ASLOTS == 4u, "four A slots a thread");
  const uint32_t sq = sid & 7u;
  uint32_t adst0 = sA + srow0 * 128u + ((sq ^ (srow0 & 7u)) << 4);

  // Each lane's ldmatrix address for A, k-step 0, stage 0 (see
  // PEARL_FOLD_LANE_BASES): a k-step is this plus the stage, XORed with the k
  // offset. Held opaque with the staging destination, or ptxas rebuilds both
  // from the lane id after every barrier.
  const uint32_t alrow = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
  const uint32_t albyte = ((lane >> 4) & 1u) * 16u;
  const uint32_t swz = (lane & 7u) << 4;
  uint32_t aLane0 = sA + (wr * 64u + alrow) * 128u + ((albyte ^ swz) & 0x10u) + (swz & 0x60u);
  asm volatile("" : "+r"(aLane0), "+r"(adst0));

  int32_t acc[MB][NB][4];
  constexpr uint32_t RPL = MB / 2u;

  // The other folds' readout (see pearl_tile_fold_wmma): B pair np of region rl
  // into x, eight xor3 for sixteen values.
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
  // Region rl of this lane for chunk c: fold it, gather the other three lanes'
  // shares, and lanes 0-3 and 16-19 write the region's word. Word c of region L
  // sits at L * 16 + (c ^ ((L >> 1) & 15)), so the eight writers hit eight
  // banks.
  auto readout_x = [&](uint32_t rl) -> uint32_t {
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);
#ifndef PEARL_ABLATE_TRANSCRIPT
    // Diagnostic only when defined: skip the shuffles. Output is meaningless.
    const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4);
    const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8);
    const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12);
    x = pearl_xor3(x, s4, s8) ^ s12;
#endif
    return x;
  };
  auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {
    if (((lane >> 2) & 3u) == 0u) {
      const uint32_t L = warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl
                         + ((lane >> 4) & 1u);
      sT[L * 16u + (c ^ ((L >> 1) & 15u))] = x;
    }
  };
  auto readout = [&](uint32_t rl, uint32_t c) { readout_st(rl, c, readout_x(rl)); };

  auto a_src = [&](uint32_t rbg_) {
    return Aprime + (size_t)(rbg_ * ROWS + srow0) * k + sq * 16u;
  };
  auto b_src = [&](uint32_t cbg_) {
    return Bprime + (size_t)(col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;
  };
  // Chunk 0 of A into stage 0, and k-step 0 of B into registers. Stage 0 was
  // last read in the previous tile's chunk 14, before that chunk's barrier.
  auto tile_prologue = [&](const int8_t *asrc_, const int8_t *bsrc_, int4 (&bf_)[2][4]) {
    {
      int4 ra[ASLOTS];
#pragma unroll
      for (uint32_t p = 0; p < ASLOTS; p++) ra[p] = pearl_bd_lda(asrc_ + p * SSTEP * k);
#pragma unroll
      for (uint32_t p = 0; p < ASLOTS; p++) pearl_st_shared_v4(adst0 + p * SSTEP * 128u, ra[p]);
    }
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++)
      bf_[0][j] = __ldg(reinterpret_cast<const int4 *>(bsrc_ + j * FRAG_BLOCK));
#ifdef PEARL_ABLATE_BLOAD
    // Diagnostic only: k-step 1's B too, once a tile. The k-steps below load no
    // B and alternate between these two, so both stay live all tile, as the
    // two buffers do when they load. Output is meaningless.
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++)
      bf_[1][j] = __ldg(reinterpret_cast<const int4 *>(bsrc_ + j * FRAG_BLOCK + 512u));
#endif
  };

#if PEARL_BD_PERSIST
  // Persistent (PEARL_BD_PERSIST): block b starts on tile b and takes every
  // later tile from the slot's counter, PEARL_BD_CTR_SLOTS words past its hit
  // counter, which the host zeroes before the launch. Thread 0 takes the tile
  // after next at the start of each tile; the tile's hand-off __syncthreads
  // publishes it. The first tile's chunk 0 is loaded here, every later one's
  // under the last chunk of the tile before it.
  if (blockIdx.x >= tiles) return;
  uint32_t *const tile_ctr = hits.count + PEARL_BD_CTR_SLOTS;
  __shared__ uint32_t sNext[2];
  if (threadIdx.x == 0u) sNext[0] = gridDim.x + atomicAdd(tile_ctr, 1u);
  int4 bf[2][4];
  {
    uint32_t rbg0, cbg0;
    tile_coords(blockIdx.x, rbg0, cbg0);
    tile_prologue(a_src(rbg0), b_src(cbg0), bf);
  }
  __syncthreads();
  uint32_t it = 0u;
  for (uint32_t v = blockIdx.x; v < tiles; it++) {
    const uint32_t vnext = sNext[it & 1u];
    if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = gridDim.x + atomicAdd(tile_ctr, 1u);
    uint32_t rbg, cbg;
    tile_coords(v, rbg, cbg);
    const int8_t *asrc = a_src(rbg);
    const int8_t *bsrc = b_src(cbg);
    // The next tile's sources, as 32-bit offsets from this tile's (A' and B'
    // are each under 2 GB); the last tile reloads its own, which nothing reads.
    uint32_t rbgn, cbgn;
    tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn);
    const int32_t adn = (int32_t)(rbgn * ROWS * k) - (int32_t)(rbg * ROWS * k);
    const int32_t bdn = (int32_t)(cbgn * 16u * FRAG_BLOCK) - (int32_t)(cbg * 16u * FRAG_BLOCK);
#pragma unroll
    for (uint32_t mb = 0; mb < MB; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
#else
  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    uint32_t rbg, cbg;
    tile_coords(v, rbg, cbg);
    const int8_t *asrc = a_src(rbg);
    const int8_t *bsrc = b_src(cbg);
    int4 bf[2][4];
    tile_prologue(asrc, bsrc, bf);
#pragma unroll
    for (uint32_t mb = 0; mb < MB; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
    PEARL_BD_BAR();
#endif

    int4 ra[ASLOTS];
    // K-step 0 of the next chunk's A, loaded at the seam (see the top).
    uint32_t afn[MB][4];
    // K-step t of chunk ch: load the next k-step's B (and at t = 0 the next
    // chunk's A), ldmatrix this k-step's A from stage ch & 1, 32 mma, and at
    // t = 2 store the next chunk's A into the other stage. seam: k-step 3 of a
    // chunk with another after it, which meets the barrier after its ldmatrix
    // and then loads afn. pre: k-step 0 after a seam, whose A is afn.
    auto kstep = [&](uint32_t ch, uint32_t t, bool stage_next, bool seam, bool pre) {
      const uint32_t buf = (ch & 1u) * ABUF;
      const uint32_t s = ch * KSTEPS + t;
      // Both loads wrap rather than test: past the last chunk they re-read
      // chunk 0, in bounds, and nothing uses it. A predicate here is a block
      // boundary to ptxas.
#ifndef PEARL_ABLATE_STAGING
      // Diagnostic only when defined: no A copies in the chunk loop (chunk 0's
      // stay), as PEARL_ABLATE_STAGING does in the other folds. The B fragment
      // loads stay: they feed the mma, like the other folds' ldmatrix. Output
      // is meaningless.
      if (t == 0u) {
#if PEARL_BD_PERSIST
        // Under the last chunk, the next tile's chunk 0 (a select, not a branch).
        const int8_t *an = asrc + (ch + 1u < chunks ? (int32_t)((ch + 1u) * rank) : adn);
#else
        const int8_t *an = asrc + ((ch + 1u) % chunks) * rank;
#endif
#pragma unroll
        for (uint32_t p = 0; p < ASLOTS; p++) ra[p] = pearl_bd_lda(an + p * SSTEP * k);
      }
#endif
#ifndef PEARL_ABLATE_BLOAD
      {
        const uint32_t s1 = (s + 1u) % (chunks * KSTEPS);
#if PEARL_BD_PERSIST
        // Past the last k-step, the next tile's k-step 0.
        const int8_t *bs = bsrc + (s + 1u < chunks * KSTEPS ? 0 : bdn);
#else
        const int8_t *bs = bsrc;
#endif
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++)
          bf[(t + 1u) & 1u][j] = pearl_bd_ldg(bs + j * FRAG_BLOCK + s1 * 512u);
      }
#else
      // Diagnostic only when defined: no B loads in the chunk loop (see the
      // tile's start). The memory clobber keeps the order the loads imposed.
      (void)s;
      asm volatile("" ::: "memory");
#endif
      uint32_t af[MB][4];
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        const uint32_t ap = ((aLane0 + buf) ^ (t * 32u)) + mb * 16u * 128u;
        if (pre) {
#pragma unroll
          for (uint32_t i = 0; i < 4u; i++) af[mb][i] = afn[mb][i];
        } else {
#ifdef PEARL_ABLATE_LDMATRIX
          // Diagnostic only, and a poor bound: see the other folds'.
          af[mb][0] = 0x01010101u ^ ap; af[mb][1] = 0x01010101u ^ mb;
          af[mb][2] = 0x01010101u ^ 1u; af[mb][3] = 0x01010101u ^ 2u;
#else
          pearl_ldmatrix_x4(af[mb][0], af[mb][1], af[mb][2], af[mb][3], ap);
#endif
        }
      }
      if (seam) {
        // Publishes chunk + 1's A, and certifies that every warp is done
        // reading chunk's stage before k-step 2 of chunk + 1 stores into it.
#ifndef PEARL_ABLATE_BARRIER
        PEARL_BD_BAR();
#endif
#pragma unroll
        for (uint32_t mb = 0; mb < MB; mb++) {
          const uint32_t ap = (aLane0 + (ABUF - buf)) + mb * 16u * 128u;
#ifdef PEARL_ABLATE_LDMATRIX
          afn[mb][0] = 0x01010101u ^ ap; afn[mb][1] = 0x01010101u ^ mb;
          afn[mb][2] = 0x01010101u ^ 1u; afn[mb][3] = 0x01010101u ^ 2u;
#else
          pearl_ldmatrix_x4(afn[mb][0], afn[mb][1], afn[mb][2], afn[mb][3], ap);
#endif
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++) {
        const uint32_t b0 = (uint32_t)bf[t & 1u][j].x, b1 = (uint32_t)bf[t & 1u][j].y;
        const uint32_t b2 = (uint32_t)bf[t & 1u][j].z, b3 = (uint32_t)bf[t & 1u][j].w;
#pragma unroll
        for (uint32_t mb = 0; mb < MB; mb++)
          pearl_mma_m16n8k32(acc[mb][2u * j][0], acc[mb][2u * j][1], acc[mb][2u * j][2],
                             acc[mb][2u * j][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3], b0, b1);
#pragma unroll
        for (uint32_t mb = 0; mb < MB; mb++)
          pearl_mma_m16n8k32(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1],
                             acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], af[mb][0], af[mb][1],
                             af[mb][2], af[mb][3], b2, b3);
      }
#ifndef PEARL_ABLATE_STAGING
      if (t == 2u && stage_next) {
#pragma unroll
        for (uint32_t p = 0; p < ASLOTS; p++)
          pearl_st_shared_v4((adst0 + (ABUF - buf)) + p * SSTEP * 128u, ra[p]);
      }
#else
      (void)stage_next;
      (void)ra;
#endif
    };
    kstep(0u, 0u, true, false, false);
    kstep(0u, 1u, true, false, false);
    for (uint32_t chunk = 0; chunk + 1u < chunks; chunk++) {
      kstep(chunk, 2u, true, false, false);
      kstep(chunk, 3u, true, true, false);  // the barrier, then chunk + 1's afn
#ifndef PEARL_ABLATE_READOUT
      uint32_t rx[RPL];
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_x(rl);
#endif
      kstep(chunk + 1u, 0u, true, false, true);
#ifndef PEARL_ABLATE_READOUT
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]);
#endif
      kstep(chunk + 1u, 1u, true, false, false);
    }
    // Persistent: k-step 2 of the last chunk stores the next tile's chunk 0 into
    // stage 0, which chunk 14 last read before its barrier.
    kstep(chunks - 1u, 2u, PEARL_BD_PERSIST != 0, false, false);
    kstep(chunks - 1u, 3u, false, false, false);
    // The last chunk's readout; with PEARL_ABLATE_READOUT the tile's only one,
    // so the accumulators stay live.
#pragma unroll
    for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunks - 1u);

    // Hand-off: every word of the group's regions is written. Hashing thread L
    // takes region L of the group's 64 (the block's 128), the same arithmetic
    // its warp used to write it. The next tile writes these words again only
    // after its first barrier, which the hashers reach after reading them.
    // Persistent: the whole block meets here, which also publishes the next
    // tile's chunk 0 and the tile after it (sNext).
#if PEARL_BD_PERSIST
    __syncthreads();
#else
    PEARL_BD_BAR();
#endif
#if PEARL_BD_GROUPS == 2 && PEARL_BD_PERSIST
    // The first group hashes on schedulers 0-1 and the second on 2-3, so each
    // scheduler keeps one warp on the next tile's chunk 0.
    const bool hasher = wr ? sid >= 64u : sid < 64u;
    const uint32_t L = wr * 64u + (sid & 63u);
#elif PEARL_BD_GROUPS == 2
    const bool hasher = sid < 64u;
    const uint32_t L = wr * 64u + sid;
#else
    const bool hasher = threadIdx.x < 128u;
    const uint32_t L = threadIdx.x;
#endif
    if (hasher) {
      uint32_t tm[16];
#pragma unroll
      for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[L * 16u + (c ^ ((L >> 1) & 15u))];
      const uint32_t ow = L / warp_regions;
      const uint32_t ocb = (L % warp_regions) / regions_per_warp;
      const uint32_t oreg = L % regions_per_warp;
      const uint32_t orb = rbg * 2u + ow / 4u;
      const uint32_t ocgb = cbg * 4u + ow % 4u;
      const uint32_t region = (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;
      [&]() {
#ifdef PEARL_ABLATE_TRANSCRIPT_HASH
        // Diagnostic only: stop after the hand-off, which prices the hashing.
        // No hit is ever reported -- never ship this.
        return;
#endif
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
      }();
    }
#if PEARL_BD_PERSIST
    v = vnext;
#endif
  }
}

#undef PEARL_BD_BAR
