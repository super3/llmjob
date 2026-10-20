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
// tile instead of every k-step, so the mma stay and the B traffic goes. So are
// _BSHARE (the second warp of a scheduler reads other B lines than the first),
// _BHIT (every B load after the first two an L1 hit) and _AHIT (every A load an
// L2 hit). On an RTX 2060 (bench, against the same build): -11%, +12%, +3%: the
// two warps do share B in L1, and B's L2 latency is the largest cost left. The
// other folds' feature switches below are not built here, so asking for one is
// an error rather than a build that ignores it.
//
// A-direct (PEARL_BD_A_DIRECT, see pearl_config.h): A' is fragment-ordered too
// (pearl_materialize16_afrag), [m / 16][k / 32][lane 32][a0 a1 a2 a3], and a
// warp loads its four A fragments of a k-step with four more LDG.128, one
// k-step ahead, beside its B. Lane L's sixteen bytes of 16-row block R and
// k-step s are
//   a0 = A'[16R + L / 4][32s + 4 (L % 4) ..+3]      rows 0-7,  k 0-15
//   a1 = A'[16R + 8 + L / 4][32s + 4 (L % 4) ..+3]  rows 8-15, k 0-15
//   a2 = A'[16R + L / 4][32s + 16 + 4 (L % 4) ..+3] rows 0-7,  k 16-31
//   a3 = A'[16R + 8 + L / 4][32s + 16 + 4 (L % 4) ..+3]
// which is what the ldmatrix.x4 below hands lane L from the staged rows: its
// lanes 0-7, 8-15, 16-23 and 24-31 address rows 0-7 at k 0-15, rows 8-15 at
// k 0-15, rows 0-7 at k 16-31 and rows 8-15 at k 16-31 of the block, and
// matrix i gives lane L bytes 4 (L % 4)..+3 of its row L / 4 in register i.
// The swizzle moves only where a row's 16-byte units sit in shared, and the
// load undoes it. So the mma get the same a0-a3 either way
// (earn/test/turingAdirect.test.js checks it byte for byte).
//
// The four warps of a row group read the same A lines, so all but the first
// should hit L1. Nothing of A is in shared, so there is no barrier a chunk:
// the group meets once a tile (PEARL_BD_A_REALIGN), and shared holds only the
// transcripts.

#if PEARL_FOLD_GROUP_STAGE || PEARL_FOLD_LANE_BASES || PEARL_FOLD_FAST_COORDS \
    || PEARL_FOLD_SERPENTINE || PEARL_STAGE_AT_BARRIER || !PEARL_STAGE_REGS
#error "PEARL_FOLD_GROUP_STAGE, _LANE_BASES, _FAST_COORDS, _SERPENTINE, PEARL_STAGE_AT_BARRIER and PEARL_STAGE_REGS=0 are not in the B-direct fold: build with PEARL_TURING_BDIRECT=0"
#endif
#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER) || defined(PEARL_ABLATE_STAGING) \
                          || defined(PEARL_ABLATE_LDMATRIX) || defined(PEARL_ABLATE_AHIT))
#error "PEARL_ABLATE_BARRIER, _STAGING, _LDMATRIX and _AHIT price A's staging, which A-direct does not have (PEARL_BD_A_REALIGN=0 drops its one meeting a tile)"
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

// A B fragment (and with A-direct an A one), ld.global.nc. Volatile and ordered
// against memory, as it was measured.
__device__ __forceinline__ int4 pearl_bd_ldg(const int8_t *p) {
  int4 v;
  asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// b, unchanged, but only once x is computed: a byte permute that takes all four
// bytes from b (selector 0x3210), so an mma that reads the result waits for x.
// A-direct uses it to order the next k-step's mma after the readout's xors.
__device__ __forceinline__ uint32_t pearl_bd_after(uint32_t b, uint32_t x) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, 0x3210;" : "=r"(r) : "r"(b), "r"(x));
  return r;
}

#if PEARL_BD_KORDER
// One of pearl_mma_m16n8k32's four m8n8k16 on Turing: c0, c1 += a x b, a one A
// word (a0-a3), b one B word (b0-b3). One instruction reads its operands before
// it writes, so it needs no early clobber.
__device__ __forceinline__ void pearl_bd_imma(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+r"(c0), "+r"(c1)
               : "r"(a), "r"(b));
}
#endif

// An A staging load. Each A byte is read by one thread, so it goes L2-only
// (ld.global.cg) and leaves L1 to the B fragments.
__device__ __forceinline__ int4 pearl_bd_lda(const int8_t *p) {
  return __ldcg(reinterpret_cast<const int4 *>(p));
}

#if PEARL_BD_HASH_SPREAD
// The spread hash (PEARL_BD_HASH_SPREAD): a warp hashes its own 16 regions of the
// last tile while it folds the next, half a round at each chunk. Lanes 2r and 2r + 1
// take region r, and lane h holds two of the state's four columns, slots 2h (f) and
// 2h + 1 (s) of each row, in hs[] as {a_f, a_s, b_f, b_s, c_f, c_s, d_f, d_s}. A step
// is two G, one on each slot, then a fixed exchange T: a_f and a_s swap, b_f and d_s
// come from the other lane, and c_f and c_s come from the other lane's c_s and c_f.
// Then slot k holds row a's word 1 - k, b's 2 - k, c's 3 - k and d's -k (mod 4),
// which is diagonal 1 - k; and since T undoes itself, the step after holds columns
// again. So the steps alternate columns and diagonals, as a round's two halves do,
// with the same code.
//
// The message words a step's G read, for step q and lane h (word 2q + h): bytes 0-3
// are the byte offsets (word << 2) of slot f's x and y and slot s's x and y in the
// 64-byte transcript, BLAKE3's permutation applied q / 2 times. Step 14 runs (the
// chunk loop has fifteen) but its result is never read; it reads step 0's words.
// earn/test/turingHashSpread.test.js rebuilds this table and runs the schedule
// against BLAKE3. Thread 0 copies it to shared once (not __constant__, which the host
// pass would have to declare as well).
#define PEARL_BD_HS_MSG                                                                   \
  {0x0c080400u, 0x1c181410u, 0x24202c28u, 0x34303c38u, 0x280c1808u, 0x3410001cu,          \
   0x2c041430u, 0x3824203cu, 0x3028100cu, 0x381c0834u, 0x14180024u, 0x3c2c0420u,          \
   0x24301c28u, 0x3c340c38u, 0x0010082cu, 0x20141804u, 0x2c243430u, 0x2038283cu,          \
   0x081c0c14u, 0x04001018u, 0x142c3824u, 0x043c3020u, 0x0c342800u, 0x18081c10u,          \
   0x00143c2cu, 0x18202404u, 0x28383008u, 0x100c341cu, 0x0c080400u, 0x1c181410u}

// One BLAKE3 G, as g() in pearl_kernel.cu.
__device__ __forceinline__ void pearl_bd_g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                           uint32_t mx, uint32_t my) {
  a = a + b + mx;
  d = rotr32(d ^ a, 16);
  c = c + d;
  b = rotr32(b ^ c, 12);
  a = a + b + my;
  d = rotr32(d ^ a, 8);
  c = c + d;
  b = rotr32(b ^ c, 7);
}

// The hand-off with the hash spread: every thread arrives once a tile, but not at the
// same place in the code (see the tile loop), so it is the barrier that allows that,
// not __syncthreads.
__device__ __forceinline__ void pearl_bd_hand_off() {
  asm volatile("barrier.sync 1, %0;" ::"n"(PEARL_FOLD_THREADS) : "memory");
}
#endif

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
#if PEARL_BD_A_DIRECT
  // A' is fragment-ordered only if the host was built with A-direct too; it
  // says so in the band word (PEARL_BD_A_DIRECT_TAG).
  const uint32_t band_word = __ldg(hits.count + PEARL_BD_BAND_WORD);
  if (!(band_word & PEARL_BD_A_DIRECT_TAG)) __trap();
  const uint32_t band_depth = band_word & ~PEARL_BD_A_DIRECT_TAG;
#if PEARL_BD_KORDER
  // B' comes in pairs of 16-row blocks (see kstep), so a warp's four blocks
  // must start on a pair: col_off steps by whole column spans (pearl_host.cu).
  if (col_off & 1u) __trap();
#endif
#else
  const uint32_t band_depth = __ldg(hits.count + PEARL_BD_BAND_WORD);
#endif
#if PEARL_BD_WALK == 1
  // Column bands instead (PEARL_BD_WALK 1): band_depth column groups of B stay
  // in L2 while every row group's A streams past them, a band's columns first.
  const uint32_t band_tiles = band_depth * row_block_groups;
  auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {
    const uint32_t band = v / band_tiles;
    const uint32_t in_band = v % band_tiles;
    const uint32_t band_first = band * band_depth;
    const uint32_t band_cols = col_block_groups - band_first < band_depth
                                   ? col_block_groups - band_first
                                   : band_depth;
    cbg_ = band_first + in_band % band_cols;
    rbg_ = in_band / band_cols;
  };
#else
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
#endif

  extern __shared__ __align__(128) uint32_t smem_u32[];
#if PEARL_BD_A_DIRECT
  // Only the transcripts: PEARL_BD_T_BUFS buffers of 8 KB, a tile in each.
  constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;
  uint32_t *sT = smem_u32;
  static_assert(PEARL_BD_SMEM_A_STAGES == 0
                    && PEARL_BD_T_BUFS * TWORDS * 4u == (PEARL_BD_A_REALIGN ? 8192u : 16384u),
                "8 KB of shared, or 16 with two transcript buffers: the host's smem for this fold");
#if PEARL_BD_GROUPS == 2 && !PEARL_BD_HASH_SPREAD
  const uint32_t sid = threadIdx.x & 127u;  // a thread's place in its row group, for the hash
#endif
#else
  const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_u32);
  uint32_t *sT = smem_u32 + PEARL_BD_A_STAGES * ABUF / 4u;  // the transcripts, after the A stages
  static_assert(PEARL_BD_A_STAGES * ABUF + (PEARL_FOLD_THREADS / 32u) * warp_regions
                        * PEARL_JACKPOT_BUCKETS * 4u == 40960u,
                "40 KB of shared: the host's smem for this fold");
  static_assert(PEARL_BD_SMEM_A_STAGES == PEARL_BD_A_STAGES && PEARL_BD_T_BUFS == 1,
                "the host sizes this fold's shared from these");

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
#endif  // PEARL_BD_A_DIRECT

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
#if PEARL_BD_T_BUFS == 2
  // Two transcript buffers (A-direct with PEARL_BD_A_REALIGN 0): this tile's
  // words go to buffer tile & 1, so the next tile's never land on words a hasher
  // has yet to read, and the tile after it writes here only past the next
  // hand-off, which every hasher reaches after its reads.
  uint32_t tbuf = 0u;
#else
  constexpr uint32_t tbuf = 0u;
#endif
  auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {
    if (((lane >> 2) & 3u) == 0u) {
      const uint32_t L = warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl
                         + ((lane >> 4) & 1u);
      sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))] = x;
    }
  };
  auto readout = [&](uint32_t rl, uint32_t c) { readout_st(rl, c, readout_x(rl)); };

#if !PEARL_BD_HASH_SPREAD
  // After the hand-off, every word of the group's regions is written. Hashing
  // thread L takes region L of the group's 64 (the block's 128), the same
  // arithmetic its warp used to write it, for the tile at row group rbg and
  // column group cbg.
  auto hash_tile = [&](uint32_t rbg, uint32_t cbg) {
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
      for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))];
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
  };
#endif

#if PEARL_BD_A_DIRECT
  // This lane's sixteen bytes of k-step 0 of the warp's first 16-row block of A'
  // (row group rbg_) and of its first 16-column block of B' (column group cbg_),
  // as byte offsets. A 16-row block is FRAG_BLOCK bytes in either operand, so a
  // row group's 128 rows are the same 128 * k bytes as row-major, and mb (or j)
  // blocks and s k-steps on are + mb * FRAG_BLOCK + s * 512. 32 bits: the host
  // refuses this fold for operands of 4 GB or more.
  static_assert(FRAG_BLOCK == 16u * k && ROWS * k == 8u * FRAG_BLOCK,
                "a 16-row block of A' is 16 rows of k bytes, a row group eight of them");
  (void)ABUF;  // no A stages
  auto a_off = [&](uint32_t rbg_) -> uint32_t {
    return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u;
  };
  auto b_off = [&](uint32_t cbg_) -> uint32_t {
#ifdef PEARL_ABLATE_BSHARE
    // Diagnostic only: as in the staged form. Output is meaningless.
    cbg_ = wr ? cbg_ ^ 1u : cbg_;
#endif
    return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;
  };
  // The readout's shuffles (readout_x's second half).
  auto readout_gather = [&](uint32_t x) -> uint32_t {
#ifndef PEARL_ABLATE_TRANSCRIPT
    const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4);
    const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8);
    const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12);
    x = pearl_xor3(x, s4, s8) ^ s12;
#endif
    return x;
  };

  // af[b][mb]: A's fragments, as bf[b][j] are B's, for this k-step (b = t & 1)
  // and the next. aoff and boff: this tile's offsets; aoffn and boffn: those
  // the tile's last k-step loads from, the next tile's (or, not persistent,
  // its own k-step 0 again: the wrap the staged form does).
  int4 af[2][MB], bf[2][4];
  uint32_t aoff, boff, aoffn = 0u, boffn = 0u;
  auto tile_prologue = [&]() {
#pragma unroll
    for (uint32_t mb = 0; mb < MB; mb++) af[0][mb] = pearl_bd_ldg(Aprime + aoff + mb * FRAG_BLOCK);
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) bf[0][j] = pearl_bd_ldg(Bprime + boff + j * FRAG_BLOCK);
#ifdef PEARL_ABLATE_BLOAD
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++)
      bf[1][j] = pearl_bd_ldg(Bprime + boff + j * FRAG_BLOCK + 512u);
#endif
  };

  // K-step t of chunk ch: the next k-step's four A and four B fragments into
  // the other buffers, then this k-step's 32 mma (128 IMMA). last: the tile's
  // last k-step, which loads from aoffn and boffn.
  //
  // ro: the k-step after a chunk's last, which overwrites the accumulators the
  // chunk's readout reads. The readout folds each B pair's accumulators just
  // before that pair's mma, and the pair's B goes through pearl_bd_after, so
  // the mma wait for the fold. Left to itself ptxas ran this k-step's mma ahead
  // of the readout's xors, held about 66 old accumulator words beside the new
  // ones, and spilled (255 registers, 60 bytes, reloaded every chunk).
  uint32_t rxs[RPL];
  auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last) {
    const uint32_t s = ch * KSTEPS + t;
#if PEARL_BD_KORDER
    // K-lo first (PEARL_BD_KORDER): a k-lo pass of every accumulator pair's
    // k 0-15 IMMA, then a k-hi pass of its k 16-31 IMMA, in the same order, so
    // the two IMMA on a pair are 64 apart. Each pair still gets k-lo, k-hi, then
    // the next k-step's k-lo, as in pearl_mma_m16n8k32.
    //
    // The host draws A' and B' in pairs of 16-row blocks (pearl_materialize16_frag
    // and _afrag): block 2p's slot holds both blocks' k-lo words and block 2p + 1's
    // their k-hi words, a lane's in the order {2p rows 0-7, 2p rows 8-15, 2p + 1
    // rows 0-7, 2p + 1 rows 8-15}. So quads 0 and 2 of a k-step (af[][0], af[][2]
    // and B's) hold its k-lo words and quads 1 and 3 its k-hi ones: m-block mb's
    // a0, a1 (or a2, a3) are words (mb & 1) * 2 + 0, 1 of quad (mb / 2) * 2 (or
    // + 1), and n-block pair j's b0, b2 (or b1, b3) the same of B's quads. ptxas
    // schedules a k-step one loaded quad at a time; with quads that mix k-lo and
    // k-hi words it undid the order (193 of a chunk loop's 512 IMMA within 32 of
    // the other IMMA on their pair, against 214 without the switch; 90 now).
    //
    // The next k-step's k-lo quads load at the top and its k-hi quads between
    // the passes, each a k-step before the pass that reads it, into registers
    // last read a pass earlier.
    const uint32_t cur = t & 1u, nxt = (t + 1u) & 1u;
    const uint32_t ao = last ? aoffn : aoff + (s + 1u) * 512u;
#ifdef PEARL_ABLATE_BHIT
    // Diagnostic only: as in the staged form. Output is meaningless.
    const uint32_t bo = (last ? boffn : boff) + ((s + 1u) & 1u) * 512u;
#else
    const uint32_t bo = last ? boffn : boff + (s + 1u) * 512u;
#endif
    // The next k-step's k-lo quads (hi 0) or k-hi quads (hi 1).
    auto load_half = [&](uint32_t hi) {
#pragma unroll
      for (uint32_t q = hi; q < 4u; q += 2u) af[nxt][q] = pearl_bd_ldg(Aprime + ao + q * FRAG_BLOCK);
#ifndef PEARL_ABLATE_BLOAD
#pragma unroll
      for (uint32_t q = hi; q < 4u; q += 2u) bf[nxt][q] = pearl_bd_ldg(Bprime + bo + q * FRAG_BLOCK);
#else
      // Diagnostic only: as in the staged form. Output is meaningless.
      (void)bo;
      asm volatile("" ::: "memory");
#endif
    };
    // A word u (0 rows 0-7, 1 rows 8-15) of m-block mb, and B word odd (0 the
    // even n-block, 1 the odd) of n-block pair j, from half hi of this k-step.
    auto a_word = [&](uint32_t mb, uint32_t hi, uint32_t u) -> uint32_t {
      const int4 v = af[cur][(mb / 2u) * 2u + hi];
      return (uint32_t)(mb & 1u ? (u ? v.w : v.z) : (u ? v.y : v.x));
    };
    auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t {
      const int4 v = bf[cur][(j / 2u) * 2u + hi];
      return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x));
    };
    load_half(0u);
    // The k-lo pass. With ro, n-block pair j's IMMA on m-blocks 2rl and 2rl + 1
    // wait for the fold of their region (rxs[rl]); the k-hi IMMA follow the k-lo
    // ones on the same accumulators.
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
      uint32_t b0[RPL], b2[RPL];
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) {
        b0[rl] = b_word(j, 0u, 0u);
        b2[rl] = b_word(j, 0u, 1u);
      }
      if (ro) {
#pragma unroll
        for (uint32_t rl = 0; rl < RPL; rl++) rxs[rl] = fold_pair(rxs[rl], rl, j);
#pragma unroll
        for (uint32_t rl = 0; rl < RPL; rl++) {
          b0[rl] = pearl_bd_after(b0[rl], rxs[rl]);
          b2[rl] = pearl_bd_after(b2[rl], rxs[rl]);
        }
      }
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 0u, 0u), b0[mb / 2u]);
        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 0u, 1u), b0[mb / 2u]);
      }
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 0u, 0u),
                      b2[mb / 2u]);
        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 0u, 1u),
                      b2[mb / 2u]);
      }
    }
    load_half(1u);
    // The k-hi pass.
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
      const uint32_t b1 = b_word(j, 1u, 0u), b3 = b_word(j, 1u, 1u);
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 1u, 0u), b1);
        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 1u, 1u), b1);
      }
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 1u, 0u), b3);
        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 1u, 1u), b3);
      }
    }
#else
    {
      const uint32_t ao = last ? aoffn : aoff + (s + 1u) * 512u;
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++)
        af[(t + 1u) & 1u][mb] = pearl_bd_ldg(Aprime + ao + mb * FRAG_BLOCK);
    }
#ifndef PEARL_ABLATE_BLOAD
    {
#ifdef PEARL_ABLATE_BHIT
      // Diagnostic only: as in the staged form. Output is meaningless.
      const uint32_t bo = (last ? boffn : boff) + ((s + 1u) & 1u) * 512u;
#else
      const uint32_t bo = last ? boffn : boff + (s + 1u) * 512u;
#endif
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++)
        bf[(t + 1u) & 1u][j] = pearl_bd_ldg(Bprime + bo + j * FRAG_BLOCK);
    }
#else
    // Diagnostic only: as in the staged form. Output is meaningless.
    asm volatile("" ::: "memory");
#endif
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
      uint32_t b0 = (uint32_t)bf[t & 1u][j].x, b1 = (uint32_t)bf[t & 1u][j].y;
      uint32_t b2 = (uint32_t)bf[t & 1u][j].z, b3 = (uint32_t)bf[t & 1u][j].w;
      if (ro) {
#pragma unroll
        for (uint32_t rl = 0; rl < RPL; rl++) rxs[rl] = fold_pair(rxs[rl], rl, j);
        b0 = pearl_bd_after(b0, rxs[0]);
        b1 = pearl_bd_after(b1, rxs[1]);
        b2 = pearl_bd_after(b2, rxs[0]);
        b3 = pearl_bd_after(b3, rxs[1]);
      }
      // a0-a3 are the four words of af as they come (see the top of this file).
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        const int4 a = af[t & 1u][mb];
        pearl_mma_m16n8k32(acc[mb][2u * j][0], acc[mb][2u * j][1], acc[mb][2u * j][2],
                           acc[mb][2u * j][3], (uint32_t)a.x, (uint32_t)a.y, (uint32_t)a.z,
                           (uint32_t)a.w, b0, b1);
      }
#pragma unroll
      for (uint32_t mb = 0; mb < MB; mb++) {
        const int4 a = af[t & 1u][mb];
        pearl_mma_m16n8k32(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1],
                           acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], (uint32_t)a.x,
                           (uint32_t)a.y, (uint32_t)a.z, (uint32_t)a.w, b2, b3);
      }
    }
#endif  // PEARL_BD_KORDER
  };
#ifdef PEARL_ABLATE_READOUT
  constexpr bool RO = false;
#else
  constexpr bool RO = true;
#endif

#if PEARL_BD_HASH_SPREAD
  // PEARL_BD_HS_MSG, which thread 0 writes before the fold's first __syncthreads.
  __shared__ uint2 sHsMsg[15];
  // The spread hash's state (see PEARL_BD_HS_MSG): this lane's two columns of the
  // state of region hs_L, the word the target test reads, and the region's 64
  // bytes in the buffer being hashed (hs_base, with bits 6-7, region L & 3, in
  // hs_k instead, so a byte permute can build each word's address).
  uint32_t hs[8], hs_w = 0u, hs_base = 0u;
  const uint32_t hs_L = warp * warp_regions + (lane >> 1);
  const uint32_t hs_k = (((hs_L >> 1) & 15u) << 2 | (hs_L & 3u) << 6) * 0x01010101u;
  // The warps that hashed at the hand-off before (one of each scheduler's two, and
  // two of each row group's four) pass it two k-steps early, and so run two k-steps
  // behind the others. Taken through a vote so that ptxas knows it is the same for
  // the whole warp: as a plain test of the warp number, ptxas took the branches on
  // it as divergent and guarded every shuffle and store in the chunk loop.
  const bool hs_late = __any_sync(0xffffffffu, ((warp & 3u) < 2u) != (wr != 0u));
  // A new hash, of the regions in buffer buf (words), in the column order.
  auto hs_init = [&](uint32_t buf) {
    const bool h1 = (lane & 1u) != 0u;
    hs[0] = h1 ? test.key[2] : test.key[0];
    hs[1] = h1 ? test.key[3] : test.key[1];
    hs[2] = h1 ? test.key[6] : test.key[4];
    hs[3] = h1 ? test.key[7] : test.key[5];
    hs[4] = h1 ? 0x3C6EF372u : 0x6A09E667u;  // BLAKE3_IV[0..3]
    hs[5] = h1 ? 0xA54FF53Au : 0xBB67AE85u;
    hs[6] = h1 ? 64u : 0u;  // counter 0, 0; block length 64; flags
    hs[7] = h1 ? (uint32_t)PEARL_TRANSCRIPT_FLAGS : 0u;
    hs_base = buf * 4u + (hs_L >> 2) * 256u;
  };
  // Step q: keep the test's word (out16[0] or out16[7] once fourteen steps are
  // done, which is what step 14 sees), then two G and T.
  auto hs_step = [&](uint32_t q) {
    hs_w = test.hash_big_endian ? hs[0] ^ hs[4] : hs[3] ^ hs[7];
    const uint2 e = sHsMsg[q];
    const uint32_t t = ((lane & 1u) ? e.y : e.x) ^ hs_k;
    const char *tb = reinterpret_cast<const char *>(sT);
    const uint32_t m0 = *reinterpret_cast<const uint32_t *>(tb + __byte_perm(t, hs_base, 0x7650u));
    const uint32_t m1 = *reinterpret_cast<const uint32_t *>(tb + __byte_perm(t, hs_base, 0x7651u));
    const uint32_t m2 = *reinterpret_cast<const uint32_t *>(tb + __byte_perm(t, hs_base, 0x7652u));
    const uint32_t m3 = *reinterpret_cast<const uint32_t *>(tb + __byte_perm(t, hs_base, 0x7653u));
    pearl_bd_g(hs[0], hs[2], hs[4], hs[6], m0, m1);
    pearl_bd_g(hs[1], hs[3], hs[5], hs[7], m2, m3);
    const uint32_t a = hs[0];
    hs[0] = hs[1];
    hs[1] = a;
    hs[2] = __shfl_xor_sync(0xffffffffu, hs[2], 1);
    const uint32_t c = __shfl_xor_sync(0xffffffffu, hs[5], 1);
    hs[5] = __shfl_xor_sync(0xffffffffu, hs[4], 1);
    hs[4] = c;
    hs[7] = __shfl_xor_sync(0xffffffffu, hs[7], 1);
  };
  // The test, once fifteen steps are done: the lane that holds the word (lane 0 of
  // the pair for out16[0], 1 for out16[7]) goes on as hash_tile does, with the
  // region's sixteen words, for the tile sVw[warp].
  __shared__ uint32_t sVw[PEARL_FOLD_THREADS / 32u];
  auto hs_test = [&]() {
    const bool mine = test.hash_big_endian ? (lane & 1u) == 0u : (lane & 1u) != 0u;
    const uint32_t msw = test.hash_big_endian ? pearl_bswap32(hs_w) : hs_w;
    if (!mine || msw > test.target_w[0]) return;
    uint32_t tm[16];
    const uint32_t *src = sT + hs_base / 4u + (hs_L & 3u) * 16u;
#pragma unroll
    for (uint32_t c = 0; c < 16u; c++) tm[c] = src[c ^ ((hs_L >> 1) & 15u)];
    uint32_t h[8];
    pearl_transcript_hash_again(test.key, tm, h);
    if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;
    uint32_t rbg, cbg;
    tile_coords(sVw[warp], rbg, cbg);
    const uint32_t ocb = (hs_L % warp_regions) / regions_per_warp;
    const uint32_t oreg = hs_L % regions_per_warp;
    const uint32_t orb = rbg * 2u + warp / 4u;
    const uint32_t ocgb = cbg * 4u + warp % 4u;
    const uint32_t region = (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;
    const uint32_t slot = atomicAdd(hits.count, 1u);
    if (slot >= PEARL_MAX_HITS) return;
    hits.index[slot] = region;
#pragma unroll
    for (int i = 0; i < 8; i++) hits.hash[slot * 8u + i] = h[i];
#pragma unroll
    for (int i = 0; i < 16; i++) hits.transcript[slot * 16u + i] = tm[i];
  };
#endif

  // One tile: its 64 k-steps and its readouts, up to the last chunk's. The chunk
  // loop is rotated as in the staged form: an iteration is k-steps 2-3 of chunk
  // c, k-step 0 of chunk c + 1 with c's readout folded in, c's stores, then
  // k-step 1. next_srcs sets aoffn and boffn before the last k-step.
  auto tile = [&](auto next_srcs) {
#pragma unroll
    for (uint32_t mb = 0; mb < MB; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
    kstep(0u, 0u, false, false);
    kstep(0u, 1u, false, false);
#if PEARL_BD_A_REALIGN
    // Chunk 0, peeled: the row group meets once a tile, before the readout's
    // stores of chunk 0. A warp that does not hash has done five k-steps by then
    // while its scheduler's other warp hashed the last tile, so the groups set
    // off level and the two warps of a scheduler read the same B for the rest
    // of the tile. The meeting also keeps these stores off words the group's
    // hashers have yet to read: each hasher reads its words before it hashes,
    // and reaches this barrier only after.
    {
      kstep(0u, 2u, false, false);
      kstep(0u, 3u, false, false);
      kstep(1u, 0u, RO, false);
#ifndef PEARL_ABLATE_READOUT
      uint32_t rx[RPL];
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rxs[rl]);
#endif
      PEARL_BD_BAR();
#ifndef PEARL_ABLATE_READOUT
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, 0u, rx[rl]);
#endif
      kstep(1u, 1u, false, false);
    }
    constexpr uint32_t first_chunk = 1u;
#else
    constexpr uint32_t first_chunk = 0u;
#endif
    for (uint32_t chunk = first_chunk; chunk + 1u < chunks; chunk++) {
      kstep(chunk, 2u, false, false);
      kstep(chunk, 3u, false, false);
#if PEARL_BD_HASH_SPREAD && !defined(PEARL_ABLATE_TRANSCRIPT_HASH)
      // ptxas puts the step in the last k-step of the pass wherever it goes here.
      // From here, the pass's mma wait 11 times on an mma 1-7 back (10 without the
      // step, 14 with the hash at the hand-off); from after the readout's stores,
      // 25 times.
      hs_step(chunk);
#endif
      kstep(chunk + 1u, 0u, RO, false);
#ifndef PEARL_ABLATE_READOUT
      uint32_t rx[RPL];
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rxs[rl]);
#pragma unroll
      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]);
#endif
      kstep(chunk + 1u, 1u, false, false);
    }
    next_srcs();
#if PEARL_BD_HASH_SPREAD
    // The late warps' hand-off, two k-steps before their tile ends. ptxas moves
    // register-only work across a barrier, and ran these two k-steps' mma (and the
    // loads of the second's operands) ahead of it, which put the late warps' hand-off
    // back at the end of the tile. So their B and the next loads' offsets go through
    // a shared load that cannot cross it. The other warps' hand-off is after these
    // k-steps anyway; for them the load only comes first.
    if (hs_late) pearl_bd_hand_off();
    {
      const uint32_t g = sVw[warp];
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++)
        bf[0][j] = make_int4((int)pearl_bd_after((uint32_t)bf[0][j].x, g),
                             (int)pearl_bd_after((uint32_t)bf[0][j].y, g),
                             (int)pearl_bd_after((uint32_t)bf[0][j].z, g),
                             (int)pearl_bd_after((uint32_t)bf[0][j].w, g));
      aoff = pearl_bd_after(aoff, g);
      boff = pearl_bd_after(boff, g);
    }
#endif
    kstep(chunks - 1u, 2u, false, false);
    kstep(chunks - 1u, 3u, false, true);
    // The last chunk's readout; with PEARL_ABLATE_READOUT the tile's only one,
    // so the accumulators stay live.
#pragma unroll
    for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunks - 1u);
  };

#if PEARL_BD_PERSIST
  // Persistent, as the staged form: block b starts on tile b and takes every
  // later one from the slot's counter; thread 0 asks for the tile after next
  // at the start of each tile and publishes it at the hand-off.
  if (blockIdx.x >= tiles) return;
  uint32_t *const tile_ctr = hits.count + PEARL_BD_CTR_SLOTS;
  __shared__ uint32_t sNext[2];
  if (threadIdx.x == 0u) sNext[0] = gridDim.x + atomicAdd(tile_ctr, 1u);
#if PEARL_BD_HASH_SPREAD
  if (threadIdx.x == 0u) {
    const uint32_t msg[30] = PEARL_BD_HS_MSG;
#pragma unroll
    for (uint32_t i = 0; i < 15u; i++) sHsMsg[i] = make_uint2(msg[2u * i], msg[2u * i + 1u]);
  }
#endif
  {
    uint32_t rbg0, cbg0;
    tile_coords(blockIdx.x, rbg0, cbg0);
    aoff = a_off(rbg0);
    boff = b_off(cbg0);
    tile_prologue();
  }
  __syncthreads();
#if PEARL_BD_HASH_SPREAD
  // The first tile's chunks hash nothing that is tested; this keeps that defined.
  hs_init(TWORDS);
#endif
  uint32_t it = 0u;
  for (uint32_t v = blockIdx.x; v < tiles; it++) {
#if PEARL_BD_T_BUFS == 2
    tbuf = (it & 1u) * TWORDS;
#endif
    uint32_t vgrab = 0u;
    if (threadIdx.x == 0u) vgrab = gridDim.x + atomicAdd(tile_ctr, 1u);
    // The next tile is read in this tile's tail (next_srcs), before its
    // hand-off: that slot is written again only at the end of the next tile,
    // which thread 0 reaches only past this hand-off. Neither it nor this tile's
    // coordinates (the hash works them out again) are held through the tile: a
    // build that held them, with 64-bit offsets, spilled 24 bytes.
    uint32_t vnext = 0u;
#if PEARL_BD_HASH_SPREAD
    // With the hash spread, a warp may pass the hand-off before its tail (hs_late),
    // so the slot is read, and thread 0 writes the other, before that: each warp
    // reads slot it & 1 between hand-offs it - 1 and it, and thread 0 writes slot
    // (it + 1) & 1 there too, which every warp read before hand-off it - 1.
    tile([&]() {
      vnext = sNext[it & 1u];
      uint32_t rbgn, cbgn;
      tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn);
      aoffn = a_off(rbgn);
      boffn = b_off(cbgn);
      if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab;
    });
    if (!hs_late) pearl_bd_hand_off();
    // This warp's words of this tile are all written (the tail's readout), and
    // the hash of the last tile's, run through this tile's chunks, is done. Test
    // it, then start this tile's. Only this warp reads or writes either buffer's
    // words of its regions, so __syncwarp orders it all: this tile's stores before
    // the next tile's loads, and the test's loads before the next tile's stores
    // land on the same words.
    __syncwarp();
#ifndef PEARL_ABLATE_TRANSCRIPT_HASH
    if (it != 0u) hs_test();
#endif
    __syncwarp();
    if (lane == 0u) sVw[warp] = v;
    hs_init(tbuf);
#else
    tile([&]() {
      vnext = sNext[it & 1u];
      uint32_t rbgn, cbgn;
      tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn);
      aoffn = a_off(rbgn);
      boffn = b_off(cbgn);
    });
    if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab;
    __syncthreads();
    {
      uint32_t rbg, cbg;
      tile_coords(v, rbg, cbg);
      hash_tile(rbg, cbg);
    }
#endif
    // The next tile's k-step 0 is already in af[0] and bf[0].
    aoff = aoffn;
    boff = boffn;
    v = vnext;
  }
#if PEARL_BD_HASH_SPREAD && !defined(PEARL_ABLATE_TRANSCRIPT_HASH)
  // The last tile's regions, with nothing left to hide their hash under.
  __syncwarp();
#pragma unroll 1
  for (uint32_t q = 0; q < chunks - 1u; q++) hs_step(q);
  hs_test();
#endif
#else
  // A block a tile, or a smaller grid walking tiles a grid apart. Nothing of A
  // is in shared, so a tile starts without a barrier.
  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
#if PEARL_BD_T_BUFS == 2
    tbuf = ((v - blockIdx.x) / gridDim.x & 1u) * TWORDS;
#endif
    {
      uint32_t rbg, cbg;
      tile_coords(v, rbg, cbg);
      aoff = a_off(rbg);
      boff = b_off(cbg);
    }
    tile_prologue();
    tile([&]() {
      aoffn = aoff;
      boffn = boff;
    });
    PEARL_BD_BAR();
    {
      uint32_t rbg, cbg;
      tile_coords(v, rbg, cbg);
      hash_tile(rbg, cbg);
    }
  }
#endif
#else
  auto a_src = [&](uint32_t rbg_) {
    return Aprime + (size_t)(rbg_ * ROWS + srow0) * k + sq * 16u;
  };
  auto b_src = [&](uint32_t cbg_) {
#ifdef PEARL_ABLATE_BSHARE
    // Diagnostic only: row slot 1 reads another column group's B, so the two
    // warps of a scheduler never share a line in L1. Output is meaningless.
    cbg_ = wr ? cbg_ ^ 1u : cbg_;
#endif
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
  // after next at the start of each tile and writes it to shared just before
  // the tile's hand-off __syncthreads, which publishes it. The first tile's
  // chunk 0 is loaded here, every later one's under the last chunk of the tile
  // before it.
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
    // The tile after next: asked for now, written to shared only before the
    // hand-off, so the atomic's round trip hides under the tile. (Written at
    // once it held warp 0 for the round trip every tile: -2.6% on a 2070.)
    uint32_t vgrab = 0u;
    if (threadIdx.x == 0u) vgrab = gridDim.x + atomicAdd(tile_ctr, 1u);
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
#ifdef PEARL_ABLATE_AHIT
        // Diagnostic only: always chunk 1 of this tile's A, so the staging
        // loads all hit L2. Output is meaningless.
        (void)an;
#pragma unroll
        for (uint32_t p = 0; p < ASLOTS; p++) ra[p] = pearl_bd_lda(asrc + rank + p * SSTEP * k);
#else
#pragma unroll
        for (uint32_t p = 0; p < ASLOTS; p++) ra[p] = pearl_bd_lda(an + p * SSTEP * k);
#endif
      }
#endif
#ifndef PEARL_ABLATE_BLOAD
      {
#ifdef PEARL_ABLATE_BHIT
        // Diagnostic only: B for k-steps 0 and 1 over and over, so every B load
        // after the first two hits L1. Output is meaningless.
        const uint32_t s1 = (s + 1u) & 1u;
#else
        const uint32_t s1 = (s + 1u) % (chunks * KSTEPS);
#endif
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

    // Hand-off: every word of the group's regions is written (hash_tile). The
    // next tile writes these words again only after its first barrier, which
    // the hashers reach after reading them. Persistent: the whole block meets
    // here, which also publishes the next tile's chunk 0 and the tile after it
    // (sNext).
#if PEARL_BD_PERSIST
    if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab;
    __syncthreads();
#else
    PEARL_BD_BAR();
#endif
    hash_tile(rbg, cbg);
#if PEARL_BD_PERSIST
    v = vnext;
#endif
  }
#endif  // PEARL_BD_A_DIRECT
}

#undef PEARL_BD_BAR
