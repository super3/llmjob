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
#define PEARL_BATCH_REGIONS 4096

// The mandated mainnet profile. rank=128 is the post-softfork value; mining any
// other rank produces work the network does not credit.
typedef struct PearlProfile {
  uint32_t m;          // offset 0
  uint32_t n;          // offset 4
  uint32_t k;          // offset 8
  uint16_t rank;       // offset 12
  uint16_t hash_tile;  // offset 14
  uint16_t rows_count; // offset 16
  uint16_t cols_count; // offset 18
  // bytes 20..51 reserved zero
} PearlProfile;

static const PearlProfile PEARL_MAINNET_PROFILE = {
    131072u, 131072u, 4096u, 128u, 16u, 2u, 64u};

// Serialize a profile into the 52-byte config block, little-endian, matching
// buildConfig52(). `out` must have room for PEARL_CONFIG_BYTES.
PEARL_HD static inline void pearl_write_config52(const PearlProfile *p, uint8_t *out) {
  for (int i = 0; i < PEARL_CONFIG_BYTES; i++) out[i] = 0;
  out[0] = (uint8_t)(p->m); out[1] = (uint8_t)(p->m >> 8);
  out[2] = (uint8_t)(p->m >> 16); out[3] = (uint8_t)(p->m >> 24);
  out[4] = (uint8_t)(p->n); out[5] = (uint8_t)(p->n >> 8);
  out[6] = (uint8_t)(p->n >> 16); out[7] = (uint8_t)(p->n >> 24);
  out[8] = (uint8_t)(p->k); out[9] = (uint8_t)(p->k >> 8);
  out[10] = (uint8_t)(p->k >> 16); out[11] = (uint8_t)(p->k >> 24);
  out[12] = (uint8_t)(p->rank); out[13] = (uint8_t)(p->rank >> 8);
  out[14] = (uint8_t)(p->hash_tile); out[15] = (uint8_t)(p->hash_tile >> 8);
  out[16] = (uint8_t)(p->rows_count); out[17] = (uint8_t)(p->rows_count >> 8);
  out[18] = (uint8_t)(p->cols_count); out[19] = (uint8_t)(p->cols_count >> 8);
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
