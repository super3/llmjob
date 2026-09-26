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
// The tile is rows {0,1,2,3} + 8j (j < 4) by columns {0,1} + 8i (i < 8): sixteen
// rows spread over 32, and sixteen columns spread over 64.
//
// Tile size is free. It cancels out of the share rate exactly:
//   shares/s = regions/s * bound/2^256
//            = (MACs/s / (tile*k)) * (target * tile * (k/rank) * 128) / 2^256
// leaves MACs/s * 128 / (rank * difficulty). So the tile is chosen purely for
// what it costs to READ OUT, and so is its shape.
//
// This shape is the one the tensor cores already hand over. An m16n8k32
// accumulator gives lane L the elements at row g (c0, c1) and row g + 8 (c2,
// c3), columns 2t and 2t + 1, where g = L >> 2 and t = L & 3. Across the fold's
// 32x64 warp tile -- two m16 tiles down, eight n8 tiles across -- lane L holds
// rows g + {0, 8, 16, 24} by columns 2t + {0, 1} + 8i. That is a quarter of
// exactly one region: the one at row offset g & 4 and column offset 2t. Lanes
// L, L^4, L^8 and L^12 hold the other three quarters. So a region's XOR is the
// lane's own 64 accumulators folded in registers plus one shuffle round trip
// among those four lanes, and every lane ends up holding its own region's
// value. A 64x64 warp tile gives each lane two whole regions the same way, and
// 96x64 three.
//
// The contiguous 16x16 tile this replaced was one wmma fragment. Folding it
// took a whole-warp REDUX per region per chunk: eight a warp, each landing in a
// uniform register that then had to be moved back into a vector one. Measured
// on a 4090 at 450 W against that: fold 263.8 -> 265.1 TH/s (bench, three
// interleaved rounds), full miner loop 262.8 -> 264.5 (two rounds), all of it
// from 7% fewer instructions; see the fold's readout for what else was tried.
//
// h*w = 256 is the largest the sanity checks allow, and both counts are
// divisible by TILE_H = 2.
//
// The pattern is self-describing: the verifier rebuilds it from the row indices
// a share carries, and config52 carries its encoding. So this is as legal as
// the strided {0,8,64,72} x {0,1,8,9,32,33,40,41} MiningConfiguration carries
// as a DEFAULT -- which is itself neither contiguous nor square.
#define PEARL_ROWS_COUNT 16
#define PEARL_COLS_COUNT 16
static constexpr uint32_t PEARL_ROWS_PATTERN[PEARL_ROWS_COUNT] = {0,  1,  2,  3,  8,  9,  10, 11,
                                                                  16, 17, 18, 19, 24, 25, 26, 27};
static constexpr uint32_t PEARL_COLS_PATTERN[PEARL_COLS_COUNT] = {0,  1,  8,  9,  16, 17, 24, 25,
                                                                  32, 33, 40, 41, 48, 49, 56, 57};

// The six-byte periodic encoding of each pattern: (factor-1, length-1) per
// dimension, where factor is the stride divided by the running product of the
// dimensions before it. Precomputed rather than derived at runtime -- the
// derivation is exercised on the JS side, and the values are asserted equal by
// test/nativeConfig.test.js so the two cannot drift.
//   rows: (stride 1, length 4) then (stride 8, length 4): factors 1 and 8/4 = 2
//   cols: (stride 1, length 2) then (stride 8, length 8): factors 1 and 8/2 = 4
// The unused third dimension pads as factor 1, length 1, i.e. two zero bytes.
static const uint8_t PEARL_ROWS_PATTERN_BYTES[6] = {0, 3, 1, 3, 0, 0};
static const uint8_t PEARL_COLS_PATTERN_BYTES[6] = {0, 1, 3, 7, 0, 0};

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
// where the mask is the OR of the pattern's own values. Rows {0,1,2,3} + 8j are
// the subsets of bits {0,1,3,4} (mask 0x1B) and columns {0,1} + 8i the subsets
// of bits {0,3,4,5} (mask 0x39), so a valid row offset is 0, 4, 32, 36, 64, ...
// and a valid column offset 0, 2, 4, 6, 64, 66, ... Verified against a
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

// The smallest aligned run of rows (columns) that whole tiles cover exactly:
// the power of two just above the mask. 32 rows hold the two row offsets 0 and
// 4; 64 columns hold the four column offsets 0, 2, 4 and 6. A run of valid
// offset INDICES that starts on a multiple of span/count is therefore a
// contiguous block of the operand, beginning at index * count -- which is what
// lets the fold stage a CTA tile as one block of rows and one of columns.
PEARL_HD constexpr uint32_t pearl_pattern_span(uint32_t mask) {
  uint32_t s = 1u;
  while (s <= mask) s <<= 1;
  return s;
}
#define PEARL_ROWS_SPAN (pearl_pattern_span(PEARL_ROWS_MASK))
#define PEARL_COLS_SPAN (pearl_pattern_span(PEARL_COLS_MASK))

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

// How many leading bytes of A a same-job redraw rewrites (pearl_restamp_operand):
// six bits of salt each, so 11 bytes carry a full 64-bit salt.
#define PEARL_STAMP_BYTES 11

// What the fold needs to hash its own transcripts and test them. Passed to the
// kernel BY VALUE, so the key and target sit in the constant bank as operands
// rather than being loaded at the end of every block. The host already holds
// both: a_seed is read back once per draw, and the target is the job's.
typedef struct {
  uint32_t key[8];       // a_seed as the eight little-endian words BLAKE3 keys with
  uint32_t target_w[8];  // the target as big-endian words, most significant first
  int hash_big_endian;   // PearlProfile.hash_big_endian
} PearlTranscriptTest;

// Where the fold reports hits. Written only on a hit, appended with an atomic.
typedef struct {
  uint32_t *count;       // hits this batch; may exceed PEARL_MAX_HITS
  uint32_t *index;       // [PEARL_MAX_HITS] batch-local region index
  uint32_t *hash;        // [PEARL_MAX_HITS][8] jackpot hash words, bytes in order
  uint32_t *transcript;  // [PEARL_MAX_HITS][16] the transcript that hashed to it
} PearlHitList;

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
// There is no staging-slot count any more. A thread's staging addresses turned
// out to be exactly linear in the slot index -- q is constant for the thread and
// the column advances by a fixed step -- so the fold walks base + stride instead
// of precomputing a table, and there is no surplus to fall through to a tail
// loop. That tail loop was an integer divide, an offset expansion and a 64-bit
// multiply PER COPY, which is cheap when all 512 threads stage and ruinous when
// only a few do.

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
//
// The right depth depends on how many blocks are RESIDENT, which is the SM count:
// 128 on a 4090, 170 on a 5090. Measured on Blackwell, deeper bands are steadily
// worse -- 1 -> 160.1, 2 -> 160.1, 4 -> 157.9, 8 -> 154.6, 16 -> 150.7, 32 -> 148.8
// TH/s -- and an alternating A/B against the Ada default over three rounds puts
// BLOCK_GROUP=1 at +3.4/+4.3/+4.9%. Ada measured flat at 16/32/64, so 32 stays its
// default and only sm_120 moves; this is gated rather than changed outright so no
// Ampere or Ada rig regresses.
//
// Only ever used inside pearl_tile_fold_wmma, so __CUDA_ARCH__ is always defined
// where it is read and the host never sees a differing value.
#ifndef PEARL_BLOCK_GROUP
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define PEARL_BLOCK_GROUP 1
#else
#define PEARL_BLOCK_GROUP 32
#endif
#endif

// Walk the bands as a serpentine: odd bands take their column groups in
// reverse, so each band starts on the B columns the one before it ended on.
//
// One launch sweeps 64 MB of B per band (2048 column groups of 16 columns, k
// bytes each), and a band 32 row groups deep adds 8 MB of A: 72 MB, all of a
// 4090's L2. Walking every band the same way is a cyclic sweep bigger than the
// cache, which LRU misses end to end, so B came from DRAM once per band --
// 2.08 GB read per launch (Nsight Compute). Reversing odd bands: 0.60 GB, and
// on the persistent fold 260.8 -> 263.7 TH/s in the full miner loop (bench
// 263.0 -> 265.1), the clock rising 2340 -> 2377 MHz at the same 450 W.
//
// A shallower band gets there the other way -- at 8 deep the band's A is 2 MB
// and B stays put (264.9 in the full loop, 0.61 GB) -- but only while B and the
// band's A fit the L2, so only on the largest Ada parts; a smaller L2 would
// then re-read B four times as often as at 32. The serpentine helps whatever
// the cache size, so it is what ships and the depth stays.
//
// Ada only: it is what measured. Blackwell's one-deep bands re-sweep B every
// row group, so it may gain there too, but that has not been measured.
#ifndef PEARL_FOLD_SERPENTINE
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define PEARL_FOLD_SERPENTINE 1
#else
#define PEARL_FOLD_SERPENTINE 0
#endif
#endif

// Transcript registers per lane: a warp's regions times buckets, over 32 lanes.
#define PEARL_JACKPOT_REGS \
  ((PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT) * PEARL_WMMA_COL_BLK \
    * PEARL_JACKPOT_BUCKETS + 31u) / 32u)

// One staged row or column is exactly one chunk of k: 128 bytes, UNPADDED.
//
// The stride used to be padded to 144 so that the eight rows one ldmatrix
// reads landed on different banks. Padding costs shared memory, and shared is
// now the scarce thing: two full-chunk stages of a 128x256 tile are 96 KB,
// which fits the 99 KB Ada allows only without padding. Bank conflicts are
// prevented by a swizzle instead: 16-byte unit q of row r is stored at unit
// q XOR (r mod 8), so the eight rows of an ldmatrix read touch eight distinct
// units -- all 32 banks -- at zero bytes of padding. XOR is an involution, so
// the staging store and the fragment load apply the same transform.
#define PEARL_SB_STRIDE 128

// The geometry the fold kernel is COMPILED for, rather than passed at runtime.
//
// rank is the consensus floor and k = 16*rank is the smallest the sanity checks
// allow, so neither ever varies in practice -- and carrying them as kernel
// arguments cost 4.4% of throughput. quads, ksteps and the chunk count were all
// runtime with them, so the k-loop was unrolled over a bound ptxas could not
// see, and the kernel sat pinned at its 128-register cap; the same loop with
// these as constants needs 120 and schedules better. The host refuses any other
// rank or k rather than silently folding the wrong shape.
#define PEARL_FOLD_RANK 128u
#define PEARL_FOLD_K 2048u
#define PEARL_FOLD_CHUNKS (PEARL_FOLD_K / PEARL_FOLD_RANK)

// Threads per fold block, frozen for the same reason: it makes the staging trip
// counts compile-time. Sixteen warps in a 4x4 grid over the 128x256 tile, which
// is what PEARL_WARP_ROWS and PEARL_WMMA_COL_BLK already assume.
#ifndef PEARL_FOLD_THREADS
#define PEARL_FOLD_THREADS 512u
#endif
// Full-chunk stages in the double buffer.
#ifndef PEARL_STAGE_BUFS
#define PEARL_STAGE_BUFS 2
#endif

// Stage the fold per warp GROUP rather than across the whole block (see the
// fold's staging notes): a warp reads one row group of A and one column group
// of B, and each group is staged by exactly the warps that read it, with every
// copy slot in bounds at compile time and issued in the middle of a k-step.
//
// Ada only, like PEARL_BLOCK_GROUP is Blackwell only: this is what measured.
// Ampere (sm_86) has Ada's SM layout and very likely gains too, but has not
// been measured, and neither has Blackwell, so both keep the block-wide walk.
// Device side only -- the host sizes and launches the fold identically either
// way.
#ifndef PEARL_FOLD_GROUP_STAGE
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define PEARL_FOLD_GROUP_STAGE 1
#else
#define PEARL_FOLD_GROUP_STAGE 0
#endif
#endif

// Whether the fold is PERSISTENT: one block per resident slot, each walking
// tiles and staging the next tile's chunk 0 under the last chunk of this one.
//
// Ada only, because only Ada has been measured: +3.1% on a 4090. Blackwell is
// the reason to hold back. It is power-capped hard, the staging ALU is where its
// power goes, and the persistent walk adds ~17% non-tensor instructions to its
// chunk (a 64-bit source rebuild and a wrap per copy); the one persistent fold
// ever run there regressed. Ampere is merely unmeasured. Both keep one block
// per tile, and with the next-tile staging compiled out their chunk loop is the
// one they ran before.
//
// The host must launch the matching grid, and it decides from the fold binary
// it actually loaded (cudaFuncAttributes::binaryVersion == 89). Either kind of
// mismatch stays correct -- a persistent build launched one block per tile runs
// each block once, and a non-persistent one given fewer blocks restages each
// later tile's chunk 0 -- it is only slower.
//
// A -DPEARL_FOLD_PERSISTENT=0/1 override binds BOTH sides, so a build can run
// another arch's launch shape on this card (how the Ampere/Blackwell path was
// checked on a 4090).
#ifdef PEARL_FOLD_PERSISTENT
#define PEARL_FOLD_PERSISTENT_FORCED 1
#endif
#ifndef PEARL_FOLD_PERSISTENT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define PEARL_FOLD_PERSISTENT 1
#else
#define PEARL_FOLD_PERSISTENT 0
#endif
#endif

// Keep each lane's ldmatrix base address live across the whole kernel, and
// drop the fold's per-warp `active` test (see the tile loop and the k-loop).
//
// At the 128-register cap ptxas could not keep the lane bases, so it rebuilt
// row * 128 + swizzle from the lane id at the top of every chunk -- between the
// barrier and the first ldmatrix, where all sixteen warps wait on the same
// instructions and each one costs about 0.12% of the rate. With this the first
// ldmatrix is two instructions after the barrier and a warp runs 221
// instructions a chunk (five and 222 once PEARL_FOLD_FAST_COORDS moves the
// stage-base multiplies past the barrier).
//
// The two halves only work together: the XOR addressing alone measured -1.2%
// (the bases still did not fit) and the compile-time `active` alone +1.0%.
// Both (4090 at 450 W, interleaved against v0.5.5):
//   bench  264.9 / 263.7 / 263.7 -> 269.8 / 266.9 / 269.4 TH/s
//   full miner loop  261.9 / 262.8 -> 268.4 / 268.4 TH/s (+2.3%)
//
// Ada only, like the other fold switches: it is what measured, and the
// register budget it depends on is the Ada fold's. Device side only.
#ifndef PEARL_FOLD_LANE_BASES
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define PEARL_FOLD_LANE_BASES 1
#else
#define PEARL_FOLD_LANE_BASES 0
#endif
#endif

// Tile coordinates by shift and mask when every band is whole and the
// column-group count is a power of two, which the mainnet geometry is. The
// general path's two integer divides are about 60 instructions, and they run
// at the tile seam, where the block waits on them: once for the next tile's
// sources in the last chunk, once more for the hashers. On top of
// PEARL_FOLD_LANE_BASES (bench, TH/s, 4090, interleaved, six rounds):
//   divides            272.1 / 269.3 / 269.6 / 269.5 / 269.5 / 269.4
//   shift and mask     273.7 / 273.2 / 272.8 / 273.0 / 272.8 / 272.9   (+1.2%)
// and against v0.5.5 with both: bench 264.0 -> 273.2 (+3.5%), full miner loop
// 262.6 / 262.8 -> 272.0 / 271.8 (+3.5%).
//
// The shift is recomputed on each call rather than held for the kernel: the
// held version cost ptxas the lane bases and measured 265.0 against 273.0.
// Ada only, with PEARL_FOLD_LANE_BASES. Device side only.
#ifndef PEARL_FOLD_FAST_COORDS
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define PEARL_FOLD_FAST_COORDS 1
#else
#define PEARL_FOLD_FAST_COORDS 0
#endif
#endif

// Transcript words a lane carries: a warp's regions times buckets, spread over
// its 32 lanes. 8 regions x 16 buckets / 32 = 4 at the mandated geometry.
#define PEARL_JACKPOT_REGS \
  ((PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT) * PEARL_WMMA_COL_BLK \
    * PEARL_JACKPOT_BUCKETS + 31u) / 32u)

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
#ifndef PEARL_WARP_ROWS
#define PEARL_WARP_ROWS 4
#endif

#ifndef PEARL_WMMA_ROW_TILES
#define PEARL_WMMA_ROW_TILES 2
#endif
#ifndef PEARL_WMMA_COL_BLK
#define PEARL_WMMA_COL_BLK 4
#endif

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
                                                   131072u, 131072u,
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
  // A contiguous low mask leaves every bit from popcount(mask) up free, so
  // depositing i there is a shift. Every caller passes a constant mask, so the
  // test folds away at compile time.
  //
  // Neither shipped pattern is contiguous any more, and the fold no longer
  // calls this per tile: it only ever expands the start of a span-aligned run
  // (see PEARL_COLS_SPAN), which is index * count. The general loop measured
  // 0.4% behind the shift when the persistent fold did run it once per tile in
  // every thread (246.6 -> 247.7 TH/s, bench, 4090); here it runs on the host,
  // once per share, and in the gather kernels nothing launches.
  if (mask != 0xFFFFFFFFu && (mask & (mask + 1u)) == 0u) return i << pearl_popcount_ce(mask);
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
