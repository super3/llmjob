// PearlHash mining profile — the C mirror of src/shared/miner/pearlhash.js.
//
// These offsets and widths MUST match buildConfig52() on the JS side byte for
// byte: config52 is hashed with the 76-byte header to derive job_key, so a
// single mismatched field silently changes every downstream hash and the miner
// produces work no pool will ever accept. The round-trip test in
// test/nativeConfig.test.js pins the two together.
//
// Algorithm reference: pearl-research-labs/pearl (ISC), zk-pow crate. This is an
// independent implementation from that specification.

#ifndef PEARL_CONFIG_H
#define PEARL_CONFIG_H

#include <stdint.h>

// The helpers below are called from BOTH sides: pearl_host.cu/pearl_kernel.cu
// run them on the device, and pearl_core.cc (compiled by the plain C++ host
// compiler, which has never heard of __device__) runs them on the host. Marking
// them for both under nvcc and leaving them bare otherwise is what lets one
// header serve both — without this the kernels fail to compile with "calling a
// __host__ function from a __global__ function is not allowed".
#ifdef __CUDACC__
#define PEARL_HD __host__ __device__
#else
#define PEARL_HD
#endif

#define PEARL_CONFIG_BYTES 52
#define PEARL_HEADER_BYTES 76
#define PEARL_HASH_BYTES 32
#define PEARL_JACKPOT_BUCKETS 16
#define PEARL_ROTL_BITS 13

// Regions searched per launch. One CUDA block each, so this is also the grid
// width — big enough to fill every SM on a large card, small enough that a job
// switch is picked up promptly.
// Regions per launch. Also the width of the partials pass, which runs one thread
// per (chunk, row): at m = 4096 that is 65536 threads against the 196608 a 4090
// holds resident, so a third of the machine. Sized to fill it instead.
#define PEARL_BATCH_REGIONS 16384

// The difficulty adjustment factor: tile_size * dot_product_length.
//
// The protocol scales the jackpot bound "in proportion to the work one attempt
// costs", so a reported hashrate is NOT attempts per second — it is
// multiply-accumulates per second, which is why competing miners quote hundreds
// of TH/s on a card that could never perform 1e14 BLAKE3 hashes. One attempt
// over a 4x8 tile with k = 2048 costs 32 * 2048 = 65536 MACs, and counts for
// exactly that much difficulty.
//
// Reporting attempts as though they were hashes under-reported this miner by
// four and a half orders of magnitude.
#define PEARL_DAF(profile) ((double)(PEARL_ROWS_COUNT * PEARL_COLS_COUNT) * (double)(profile).k)

// The mandated mainnet profile. rank=128 is the post-softfork value; mining any
// other rank produces work the network does not credit.
// The tile is 16 CONSECUTIVE ROWS by 16 CONSECUTIVE COLUMNS.
//
// Tile size is free. It cancels out of the share rate exactly:
//   shares/s = regions/s * bound/2^256
//            = (MACs/s / (tile*k)) * (target * tile * (k/rank) * 128) / 2^256
// leaves MACs/s * 128 / (rank * difficulty). So the tile is chosen purely for
// what it costs to READ OUT, and 16x16 is the shape that costs nothing.
//
// A 16x16 tile is exactly one int8 wmma accumulator fragment. The XOR over the
// tile is then the XOR over every lane's registers followed by one warp
// reduction -- no shared memory, no barriers, and no need to know which element
// sits in which register, because XOR does not care about order.
//
// The previous 4x16 tile was a QUARTER of a fragment, so the fold had to spill
// each accumulator to shared memory and gather four rows back out, twice per
// fragment per chunk. Measured on a 4090: deleting the readout entirely took
// the kernel from 57 to 185 TH/s, so that gather was two thirds of all runtime.
//
// h*w = 256 is the largest the sanity checks allow, and both dimensions are
// divisible by TILE_H = 2.
//
// The pattern is self-describing in config52 and the miner picks it, so this is
// as legal as the strided 4x8 set MiningConfiguration carries as a DEFAULT.
#define PEARL_ROWS_COUNT 16
#define PEARL_COLS_COUNT 16
static constexpr uint32_t PEARL_ROWS_PATTERN[PEARL_ROWS_COUNT] = {0, 1, 2,  3,  4,  5,  6,  7,
                                                                  8, 9, 10, 11, 12, 13, 14, 15};
static constexpr uint32_t PEARL_COLS_PATTERN[PEARL_COLS_COUNT] = {0, 1, 2,  3,  4,  5,  6,  7,
                                                                  8, 9, 10, 11, 12, 13, 14, 15};

// The six-byte periodic encoding of each pattern: (factor-1, length-1) per
// dimension. Precomputed rather than derived at runtime — the derivation is
// exercised on the JS side, and the values are asserted equal by
// test/nativeConfig.test.js so the two cannot drift.
// A contiguous run is a single (stride 1, length N) dimension: factor byte 0,
// length byte N-1.
static const uint8_t PEARL_ROWS_PATTERN_BYTES[6] = {0, 15, 0, 0, 0, 0};
static const uint8_t PEARL_COLS_PATTERN_BYTES[6] = {0, 15, 0, 0, 0, 0};

typedef struct PearlProfile {
  // Hashed into config52 — protocol-mandated.
  uint32_t k;          // common dimension; the sanity checks require k >= 16*rank
  uint16_t rank;       // 128, the rank-penalty floor
  uint16_t mma_type;   // 0 = Int7xInt7ToInt32
  // NOT hashed: the miner's own choice of workload dimensions. They size the
  // operands and bound the tile offset, and never enter job_key.
  uint32_t m;
  uint32_t n;
  // Which seed derivation binds the operand roots.
  //   0 = cert-v3 salted: hash_a' = blake3(hash_a ‖ pad32(m), key=SEED_SALT_A)
  //   1 = legacy: the raw roots
  // Salted is the default because it is the ONLY thing that commits m and n —
  // they are the miner's own choice and deliberately absent from config52.
  //
  // Not confirmed against the live network. Both derivations give a perfectly
  // self-consistent miner and differ only in whether a pool accepts the share,
  // so this is the first flag to flip if everything else verifies.
  uint32_t seed_derivation;
  // How many column offsets one launch covers.
  //
  // A batch used to be a single column offset, i.e. m regions. That made the
  // search launch-bound rather than compute-bound: measured on a 4090, a batch
  // cost a flat 134-213us whether it carried 1024 regions or 8192, because
  // three kernel launches and a synchronising copy dominated whatever work was
  // inside them. Widening the batch amortises that fixed cost, and it also
  // gives the partials kernel far better arithmetic intensity, since each A row
  // it reads is now used against col_batch*8 columns instead of 8.
  //
  // Costs col_batch * chunks * m * cols * 4 bytes of partial table.
  uint32_t col_batch;
  // 0 = read the jackpot hash little-endian, as the reference does; 1 = big.
  // A diagnostic for the share rejections, not a protocol choice.
  uint32_t hash_big_endian;
} PearlProfile;

// A tile offset is only VALID if it has the pattern's own bits clear.
//
// The verifier rebuilds the pattern from the row indices in a submitted proof
// and then checks the offset with PeriodicPattern::offset_is_valid, which
// reduces the offset modulo each (stride, length) dimension in turn and
// requires it to stay below the stride. For these patterns that is exactly
//
//     (offset & mask) == 0
//
// where the mask is the OR of the pattern's own values -- rows {0,1,2,3} are
// the subsets of bits {0,1}, and columns {0..15} the subsets of bits {0..3}, so
// a valid offset is a multiple of 4 down and of 16 across. Verified against a
// transcription of offset_is_valid in the JS tests rather than taken on trust.
//
// This matters twice over. A share at an invalid offset is unverifiable and
// gets rejected with "offset N is not valid for pattern", so 31 of every 32
// regions searched were unsubmittable (63 of 64 with the contiguous tile). And because the pattern bits are clear
// in a valid offset, a tile row is a bitwise OR rather than an addition, and
// valid tiles PARTITION the grid instead of overlapping.
// DERIVED, not written down. These were hand-maintained constants, and when
// the tile went from 4 rows to 16 the rows mask stayed at 3. Nothing failed:
// the fold was right, the hash was right, the Merkle proof verified against its
// own root -- but the proof described the wrong ROWS, because the row offset was
// expanded against a stale mask. The pool answered "Failed to extract strip",
// which is the verifier asking for bytes [row*k, row*k+k) that the submitted
// leaves do not cover. A constant that must agree with a table should be
// computed from that table.
PEARL_HD constexpr uint32_t pearl_pattern_mask(const uint32_t *p, uint32_t n) {
  uint32_t m = 0u;
  for (uint32_t i = 0; i < n; i++) m |= p[i];
  return m;
}
PEARL_HD constexpr uint32_t pearl_popcount_ce(uint32_t x) {
  uint32_t c = 0u;
  for (; x; x >>= 1) c += (x & 1u);
  return c;
}

#define PEARL_ROWS_MASK (pearl_pattern_mask(PEARL_ROWS_PATTERN, PEARL_ROWS_COUNT))
#define PEARL_COLS_MASK (pearl_pattern_mask(PEARL_COLS_PATTERN, PEARL_COLS_COUNT))

// offset_is_valid((off & mask) == 0) only PARTITIONS the grid when the pattern
// is exactly the set of subsets of its own mask bits. Otherwise tiles overlap
// or leave gaps, and the search silently repeats or skips work.
static_assert(PEARL_ROWS_COUNT == (1u << pearl_popcount_ce(PEARL_ROWS_MASK)),
              "rows pattern must be every subset of its mask bits");
static_assert(PEARL_COLS_COUNT == (1u << pearl_popcount_ce(PEARL_COLS_MASK)),
              "cols pattern must be every subset of its mask bits");

// How many rows of A one thread carries.
//
// The partials kernel is 86% of a batch and runs at about an eighth of the
// card's __dp4a peak, because it issues one 16-byte load of B for every four
// multiply-accumulate instructions. Carrying several rows against the same
// eight B columns multiplies that ratio directly: at two rows it is eight
// __dp4a per load, at four it is sixteen.
//
// The cost is registers -- each row holds a whole k-slice, so this trades
// occupancy for arithmetic intensity.
#define PEARL_ROWS_PER_THREAD 2

// How many 16-byte groups of an A row slice a thread can hold in registers.
// 8 covers rank 128, the mandated profile. A rank needing more falls back to
// re-reading the slice per column group, which is correct but slower.
#define PEARL_MAX_A_QUADS 16

// How many regions share one warp in the fold. The producer collapses each
// row's columns, so a region needs only PEARL_ROWS_COUNT lanes; giving it a
// whole warp left 28 of 32 idle.
#define PEARL_REGIONS_PER_WARP (32 / PEARL_ROWS_COUNT)

// How many hits one batch can report. The search returns on the first one, so
// this only has to be large enough that a pathologically easy target does not
// silently lose hits it would never have submitted anyway.
#define PEARL_MAX_HITS 64

// Rows of A one warp covers in the tensor-core partials kernel. The WMMA int8
// shape is 16x16x16, and valid row offsets are multiples of PEARL_ROWS_COUNT,
// so a 16-row block is exactly four consecutive row offsets.
#define PEARL_WMMA_ROWS 16

// How many 16-wide k-steps of A a warp holds in registers at once. 8 covers
// rank 128. Each fragment is only two registers a lane, so holding the whole
// rank-slice costs about sixteen -- cheap next to re-reading A once per column
// group, which is where the tensor-core version's time went at first.
#define PEARL_MAX_K_FRAGS 16

// How many 16-row blocks one warp covers. Each B fragment loaded serves all of
// them, so this divides B traffic directly.
//
// With one block, B was about 34 GB a batch: every one of the m/16 row-block
// warps re-reads the same sixteen columns for every column group. That fits in
// L2, but it still wants roughly 2.9 TB/s of L2 bandwidth, which is where the
// tensor-core kernel was actually stuck.
// Measured on a 4090 at m=n=32768: four row blocks 65.7 TH/s, eight 60.8.
// Eight halves B traffic again but needs twice the fragments and accumulators
// in registers, and the register pressure costs more than the traffic saves.
// Row blocks and column groups one warp carries at once.
//
// The tile accumulator is cumulative over the whole of k, so it cannot be
// reused between column groups -- a warp needs ROW_TILES*COL_BLK accumulators
// live for the entire k loop. That is the whole tradeoff: more of them means
// A and B are each read fewer times, at the cost of registers.
//
// With one column group per warp, A was re-read once per group: about 33 GB a
// batch, which put the fused kernel at 4.7 TH/s.
// Bytes of shared memory one staged B column occupies. The chunk is rank
// elements of int8, and the stride is padded past 128 so that the 16 columns of
// a fragment do not all land on the same shared-memory banks. wmma requires a
// 16-byte-aligned leading dimension for integer types, which 144 satisfies.
// How many int4 of each operand one thread stages per chunk, which is also how
// many staging addresses it precomputes. ceil(max(sb_cols, sa_rows) * quads /
// threads): at the mandated geometry 128 columns (and 128 rows) x 8 int4 over
// 256 threads = 4. A geometry needing more still works -- the surplus falls
// through to a tail loop -- it just pays the arithmetic per chunk again.
#define PEARL_STAGE_SLOTS 4

// How many row-block groups a wave of blocks walks before moving across.
//
// The blocks resident at any moment decide what has to be in L2. Numbering them
// so that consecutive blocks walk DOWN the rows means the ~256 blocks resident
// on a 4090 span every row of A and a single column group of B: A is re-read
// from memory once per column group, which for the mandated geometry is 256
// times, or 16 GB a sweep against a 64 MB operand.
//
// Grouping them into a square instead -- 32 row groups by however many column
// groups the wave covers -- means a wave touches 32*128 rows and 8*128
// columns, a quarter of a megabyte an operand, which stays in L2. The work is
// identical; only the order changes.
#define PEARL_BLOCK_GROUP 32

// Transcript registers per lane: a warp's regions times buckets, over 32 lanes.
#define PEARL_JACKPOT_REGS \
  ((PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT) * PEARL_WMMA_COL_BLK \
    * PEARL_JACKPOT_BUCKETS + 31u) / 32u)

#define PEARL_SB_STRIDE 144

// How many of a block's warps sit along the ROW dimension. The rest go across
// the columns, so a block covers
//   PEARL_WARP_ROWS * PEARL_WMMA_ROW_TILES * 16   rows
//   (warps/PEARL_WARP_ROWS) * PEARL_WMMA_COL_BLK * 16 columns.
//
// This decides arithmetic intensity, which is what the kernel is actually
// limited by. A sweep reads A once per column-block and B once per row-block:
//   traffic = m*k*(n/bN) + n*k*(m/bM)
// so for a fixed number of warps the tile wants to be SQUARE. Laying all eight
// warps along the rows made a 256x64 block, which re-read A 512 times a sweep
// for 43 GB; 4x2 makes it 128x128 and 34 GB for exactly the same registers,
// shared memory and occupancy.
#define PEARL_WARP_ROWS 4

#define PEARL_WMMA_ROW_TILES 2
#define PEARL_WMMA_COL_BLK 4

#define PEARL_SEED_SALTED 0u
#define PEARL_SEED_LEGACY 1u

// blake3("pearl/cert-v3/noise-seed/A") and .../B. Hardcoded in the reference so
// consensus does not depend on runtime string hashing; both are re-derived from
// their strings in the JS tests.
static const uint8_t PEARL_SEED_SALT_A[32] = {
    0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6,
    0x92, 0xfc, 0xf0, 0x76, 0xf8, 0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23,
    0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b};
static const uint8_t PEARL_SEED_SALT_B[32] = {
    0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf,
    0x71, 0x8b, 0x3f, 0x4d, 0x4f, 0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8,
    0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9};

// k = 16 * rank is the smallest common dimension the protocol allows at the
// mandated rank, and k/rank = 16 chunks is exactly the transcript lane count, so
// each chunk lands in its own lane and the rotation never wraps.
static const PearlProfile PEARL_MAINNET_PROFILE = {2048u, 128u, 0u,
                                                   65536u, 65536u,
                                                   PEARL_SEED_SALTED, 2048u, 0u};

// Serialize the 52-byte mining configuration, matching the reference's
// MiningConfiguration::to_bytes byte for byte:
//
//   common_dim u32 (4) | rank u16 (2) | mma_type u16 (2)
//   rows_pattern   (6) | cols_pattern (6) | MoE trailer (32)
//
// Note m and n are absent: they are the miner's choice, not protocol. An earlier
// version packed them here along with a hash_tile and two pattern COUNTS, none of
// which the protocol carries — which changed job_key and so every hash after it,
// silently. `out` must have room for PEARL_CONFIG_BYTES.
// Host only: it reads file-scope constant arrays, which device code cannot see,
// and the only caller is pearl_host_set_job on the host side.
static inline void pearl_write_config52(const PearlProfile *p, uint8_t *out) {
  for (int i = 0; i < PEARL_CONFIG_BYTES; i++) out[i] = 0;
  out[0] = (uint8_t)(p->k); out[1] = (uint8_t)(p->k >> 8);
  out[2] = (uint8_t)(p->k >> 16); out[3] = (uint8_t)(p->k >> 24);
  out[4] = (uint8_t)(p->rank); out[5] = (uint8_t)(p->rank >> 8);
  out[6] = (uint8_t)(p->mma_type); out[7] = (uint8_t)(p->mma_type >> 8);
  for (int i = 0; i < 6; i++) out[8 + i] = PEARL_ROWS_PATTERN_BYTES[i];
  for (int i = 0; i < 6; i++) out[14 + i] = PEARL_COLS_PATTERN_BYTES[i];
  // Bytes 20..51 are the MoE trailer, zero for a standard job.
}

// rotl on a 32-bit lane — the transcript fold's mixing step. Mirrors rotl13().
// The i-th offset with (offset & mask) == 0: deposit the bits of i into the
// positions the mask leaves free. This enumerates exactly the VALID offsets,
// so every region the search visits is one a pool will accept a proof for.
PEARL_HD static inline uint32_t pearl_expand_offset(uint32_t i, uint32_t mask) {
  uint32_t out = 0u;
  uint32_t bit = 1u;
  while (i) {
    if (!(mask & bit)) {
      if (i & 1u) out |= bit;
      i >>= 1;
    }
    bit <<= 1;
  }
  return out;
}

PEARL_HD static inline uint32_t pearl_rotl13(uint32_t x) {
  return (x << PEARL_ROTL_BITS) | (x >> (32 - PEARL_ROTL_BITS));
}

// Compare a 32-byte little-endian jackpot hash against a 32-byte BIG-endian
// target. Both endiannesses are load-bearing and opposite: the hash is read
// least-significant-byte-first, the pool's target most-significant-first.
// Returns non-zero when the hash is a share.
// Does the jackpot hash meet the bound?
//
// The reference reads the hash LITTLE-endian --
// U256::from_little_endian(hash_jackpot) -- and the pool sends its target as a
// big-endian hex string, so the default walks the hash from its last byte and
// the target from its first.
//
// hash_big_endian exists because that pairing is not producing accepted shares.
// A hash 36x inside the computed bound was still rejected, which is what it
// would look like if the pool read the hash the other way round: its value
// would be effectively random with respect to ours, so no margin would ever
// help. Selectable so the two can be told apart against a live pool, which is
// the only place the question can be settled.
PEARL_HD static inline int pearl_meets_target_mode(const uint8_t *hash_le,
                                                   const uint8_t *target_be,
                                                   int hash_big_endian) {
  for (int i = 0; i < PEARL_HASH_BYTES; i++) {
    uint8_t h = hash_big_endian ? hash_le[i] : hash_le[PEARL_HASH_BYTES - 1 - i];
    uint8_t t = target_be[i];
    if (h < t) return 1;
    if (h > t) return 0;
  }
  return 1;  // exactly equal counts as a share
}

PEARL_HD static inline int pearl_meets_target(const uint8_t *hash_le, const uint8_t *target_be) {
  return pearl_meets_target_mode(hash_le, target_be, 0);
}

#endif  // PEARL_CONFIG_H
