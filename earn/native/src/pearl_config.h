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
// The tile index sets. These are the reference implementation's own defaults —
// a 4 x 8 tile whose indices are NOT a simple stride, which is what an earlier
// guess assumed.
#define PEARL_ROWS_COUNT 4
#define PEARL_COLS_COUNT 8
static const uint32_t PEARL_ROWS_PATTERN[PEARL_ROWS_COUNT] = {0, 8, 64, 72};
static const uint32_t PEARL_COLS_PATTERN[PEARL_COLS_COUNT] = {0, 1, 8, 9, 32, 33, 40, 41};

// The six-byte periodic encoding of each pattern: (factor-1, length-1) per
// dimension. Precomputed rather than derived at runtime — the derivation is
// exercised on the JS side, and the values are asserted equal by
// test/nativeConfig.test.js so the two cannot drift.
static const uint8_t PEARL_ROWS_PATTERN_BYTES[6] = {7, 1, 3, 1, 0, 0};
static const uint8_t PEARL_COLS_PATTERN_BYTES[6] = {0, 1, 3, 1, 1, 1};

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
} PearlProfile;

// How many 16-byte groups of an A row slice a thread can hold in registers.
// 8 covers rank 128, the mandated profile. A rank needing more falls back to
// re-reading the slice per column group, which is correct but slower.
#define PEARL_MAX_A_QUADS 8

// How many regions share one warp in the fold. The producer collapses each
// row's columns, so a region needs only PEARL_ROWS_COUNT lanes; giving it a
// whole warp left 28 of 32 idle.
#define PEARL_REGIONS_PER_WARP (32 / PEARL_ROWS_COUNT)

// How many hits one batch can report. The search returns on the first one, so
// this only has to be large enough that a pathologically easy target does not
// silently lose hits it would never have submitted anyway.
#define PEARL_MAX_HITS 64

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
                                                   6144u, 6144u,
                                                   PEARL_SEED_SALTED, 32u};

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
PEARL_HD static inline uint32_t pearl_rotl13(uint32_t x) {
  return (x << PEARL_ROTL_BITS) | (x >> (32 - PEARL_ROTL_BITS));
}

// Compare a 32-byte little-endian jackpot hash against a 32-byte BIG-endian
// target. Both endiannesses are load-bearing and opposite: the hash is read
// least-significant-byte-first, the pool's target most-significant-first.
// Returns non-zero when the hash is a share.
PEARL_HD static inline int pearl_meets_target(const uint8_t *hash_le, const uint8_t *target_be) {
  for (int i = 0; i < PEARL_HASH_BYTES; i++) {
    uint8_t h = hash_le[PEARL_HASH_BYTES - 1 - i];  // walk hash high→low
    uint8_t t = target_be[i];                       // target is already high→low
    if (h < t) return 1;
    if (h > t) return 0;
  }
  return 1;  // exactly equal counts as a share
}

#endif  // PEARL_CONFIG_H
