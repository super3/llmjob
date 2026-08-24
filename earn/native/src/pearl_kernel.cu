// PearlHash CUDA core — noise generation, the int8 GEMM, and the per-tile
// jackpot transcript fold.
//
// Independent implementation from the ISC-licensed specification in
// pearl-research-labs/pearl (zk-pow crate) and the protocol notes captured from
// the live pool. It is deliberately NOT derived from any dev-fee-licensed miner:
// this file exists so LLMJob ships a core we own outright, with no fee to
// retain, no redistribution restriction, and a binary we can code-sign so
// Windows Defender stops eating it.
//
// STATUS: this is the reference-correct scalar/dp4a path. It is structured so
// the tensor-core (mma.sync int8) mainloop drops into pearl_gemm_fold() without
// touching the surrounding pipeline — that specialisation is what takes a card
// from tens of TH/s to hundreds, and is the next piece of work. Every kernel
// here is written to be bit-exact with the JS reference so the two can be
// cross-checked before any performance work begins; a fast core that disagrees
// with the spec mines nothing.
//
// Pipeline per job:
//   1. job_key = blake3(header76 ‖ config52)                       [host]
//   2. hash_a/hash_b = keyed blake3 over the padded operands       [pearl_hash_operands]
//   3. b_seed, a_seed derived from those                           [host]
//   4. E_A = E_AL·E_AR, E_B = E_BL·E_BR   (low-rank noise, rank r) [pearl_gen_noise]
//   5. C accumulated in rank chunks; per chunk fold the sub-tile   [pearl_gemm_fold]
//        jackpot[tid] = rotl13(jackpot[tid]) ^ xor(tile), tid = chunk % 16
//   6. jackpot_hash = blake3(transcript64, key=a_seed); share iff <= target

#include <cuda_runtime.h>
#include <stdint.h>

#include "pearl_config.h"

namespace {

// ---------------------------------------------------------------------------
// BLAKE3 compression, device side.
//
// Only the pieces the PoW needs: a keyed compression over short, fixed-size
// inputs (the operand blocks and the 64-byte transcript). The full streaming
// tree hasher is not required here — inputs are always <= one chunk — which
// keeps this small enough to audit against the spec.
// ---------------------------------------------------------------------------

__constant__ uint32_t BLAKE3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u,
                                      0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu,
                                      0x1F83D9ABu, 0x5BE0CD19u};

__constant__ uint8_t BLAKE3_MSG_PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13,
                                            1, 11, 12, 5, 9, 14, 15, 8};

#define CHUNK_START (1u << 0)
#define CHUNK_END (1u << 1)
#define PARENT (1u << 2)
#define ROOT (1u << 3)
#define KEYED_HASH (1u << 4)

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void g(uint32_t *s, int a, int b, int c, int d,
                                  uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

__device__ void blake3_compress(const uint32_t cv[8], const uint32_t block[16],
                                uint64_t counter, uint32_t block_len,
                                uint32_t flags, uint32_t out[16]) {
  uint32_t s[16];
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 8; i++) s[i] = cv[i];
  s[8] = BLAKE3_IV[0]; s[9] = BLAKE3_IV[1];
  s[10] = BLAKE3_IV[2]; s[11] = BLAKE3_IV[3];
  s[12] = (uint32_t)counter;
  s[13] = (uint32_t)(counter >> 32);
  s[14] = block_len;
  s[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = block[i];

  for (int round = 0; round < 7; round++) {
    g(s, 0, 4, 8, 12, m[0], m[1]);
    g(s, 1, 5, 9, 13, m[2], m[3]);
    g(s, 2, 6, 10, 14, m[4], m[5]);
    g(s, 3, 7, 11, 15, m[6], m[7]);
    g(s, 0, 5, 10, 15, m[8], m[9]);
    g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7, 8, 13, m[12], m[13]);
    g(s, 3, 4, 9, 14, m[14], m[15]);
    if (round < 6) {
      uint32_t t[16];
#pragma unroll
      for (int i = 0; i < 16; i++) t[i] = m[BLAKE3_MSG_PERM[i]];
#pragma unroll
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i] = s[i] ^ s[i + 8];
    out[i + 8] = s[i + 8] ^ cv[i];
  }
}

// Keyed BLAKE3 over an input of at most one 1024-byte chunk, producing 32 bytes.
// This covers every hash the PoW takes: the padded operand blocks and the
// 64-byte transcript.
__device__ void blake3_keyed(const uint32_t key[8], const uint8_t *input,
                             uint32_t len, uint8_t out[32]) {
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];

  uint32_t block[16];
  uint32_t offset = 0;
  uint32_t flags_start = CHUNK_START;

  while (offset + 64 < len) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
      uint32_t j = offset + i * 4;
      block[i] = (uint32_t)input[j] | ((uint32_t)input[j + 1] << 8) |
                 ((uint32_t)input[j + 2] << 16) | ((uint32_t)input[j + 3] << 24);
    }
    uint32_t out16[16];
    blake3_compress(cv, block, 0, 64, KEYED_HASH | flags_start, out16);
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = out16[i];
    flags_start = 0;
    offset += 64;
  }

  // Final (possibly short) block, zero-padded.
  uint8_t tail[64];
  uint32_t rem = len - offset;
#pragma unroll
  for (int i = 0; i < 64; i++) tail[i] = (i < rem) ? input[offset + i] : 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    block[i] = (uint32_t)tail[i * 4] | ((uint32_t)tail[i * 4 + 1] << 8) |
               ((uint32_t)tail[i * 4 + 2] << 16) |
               ((uint32_t)tail[i * 4 + 3] << 24);
  }
  uint32_t out16[16];
  blake3_compress(cv, block, 0, rem, KEYED_HASH | flags_start | CHUNK_END | ROOT,
                  out16);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i * 4 + 0] = (uint8_t)(out16[i]);
    out[i * 4 + 1] = (uint8_t)(out16[i] >> 8);
    out[i * 4 + 2] = (uint8_t)(out16[i] >> 16);
    out[i * 4 + 3] = (uint8_t)(out16[i] >> 24);
  }
}

// A keyed-BLAKE3 stream used as the noise RNG: successive 32-byte blocks keyed
// by the seed and indexed by a counter, expanded to int8 draws.
__device__ __forceinline__ int8_t noise_draw(const uint32_t key[8],
                                             uint64_t index) {
  uint8_t buf[8];
#pragma unroll
  for (int i = 0; i < 8; i++) buf[i] = (uint8_t)(index >> (i * 8));
  uint8_t h[32];
  blake3_keyed(key, buf, 8, h);
  // int7 range, matching the reference's Int7xInt7ToInt32 MMA type: values in
  // [-63, 63] so products stay inside int32 without saturation.
  return (int8_t)((int32_t)(h[0] % 127) - 63);
}

}  // namespace


// ---------------------------------------------------------------------------
// BLAKE3 over inputs LARGER than one chunk.
//
// The operand commitments hash all of A and all of Bt — 64 KiB on the test
// profile, 512 MiB at mainnet — and BLAKE3 is a Merkle tree over 1024-byte
// chunks, not one long chain. Hashing them as a single chunk (which is what the
// first version of this file did) silently produces the wrong digest for any
// input over 1024 bytes, and therefore the wrong seeds, and therefore a
// transcript no pool will ever accept.
//
// Chunk counts here are always powers of two (m*k with power-of-two m and k), so
// the tree is perfectly balanced and the reduction is a clean pairwise fold. The
// host asserts that rather than assuming it.
// ---------------------------------------------------------------------------

// The chaining value of one complete 1024-byte chunk at index `counter`.
__device__ void blake3_chunk_cv(const uint32_t key[8], const uint8_t *in,
                                uint32_t len, uint64_t counter,
                                uint32_t base_flags, uint32_t out_cv[8]) {
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];

  uint32_t block[16];
  uint32_t offset = 0;
  uint32_t start = CHUNK_START;
  uint32_t out16[16];

  while (offset + 64 < len) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
      uint32_t j = offset + i * 4;
      block[i] = (uint32_t)in[j] | ((uint32_t)in[j + 1] << 8) |
                 ((uint32_t)in[j + 2] << 16) | ((uint32_t)in[j + 3] << 24);
    }
    blake3_compress(cv, block, counter, 64, base_flags | start, out16);
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = out16[i];
    start = 0;
    offset += 64;
  }

  uint8_t tail[64];
  uint32_t rem = len - offset;
#pragma unroll
  for (int i = 0; i < 64; i++) tail[i] = (i < rem) ? in[offset + i] : 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    block[i] = (uint32_t)tail[i * 4] | ((uint32_t)tail[i * 4 + 1] << 8) |
               ((uint32_t)tail[i * 4 + 2] << 16) |
               ((uint32_t)tail[i * 4 + 3] << 24);
  }
  blake3_compress(cv, block, counter, rem, base_flags | start | CHUNK_END, out16);
#pragma unroll
  for (int i = 0; i < 8; i++) out_cv[i] = out16[i];
}

// One thread per 1024-byte chunk: the leaf layer of the tree.
extern "C" __global__ void pearl_blake3_chunk_cvs(const uint32_t *key,
                                                  const uint8_t *data,
                                                  uint64_t chunks,
                                                  uint32_t *cvs_out) {
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunks) return;
  uint32_t key_l[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key_l[i] = key[i];
  uint32_t cv[8];
  blake3_chunk_cv(key_l, data + idx * 1024, 1024, idx, KEYED_HASH, cv);
#pragma unroll
  for (int i = 0; i < 8; i++) cvs_out[idx * 8 + i] = cv[i];
}

// One pairwise parent layer. When `is_root` the single remaining compression
// carries ROOT and its first 8 words ARE the digest.
extern "C" __global__ void pearl_blake3_parent_layer(const uint32_t *key,
                                                     const uint32_t *in_cvs,
                                                     uint64_t pairs,
                                                     uint32_t is_root,
                                                     uint32_t *out_cvs) {
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pairs) return;
  uint32_t key_l[8], block[16], out16[16];
#pragma unroll
  for (int i = 0; i < 8; i++) key_l[i] = key[i];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    block[i] = in_cvs[idx * 16 + i];
    block[i + 8] = in_cvs[idx * 16 + 8 + i];
  }
  uint32_t flags = PARENT | KEYED_HASH | (is_root ? ROOT : 0u);
  blake3_compress(key_l, block, 0, 64, flags, out16);
#pragma unroll
  for (int i = 0; i < 8; i++) out_cvs[idx * 8 + i] = out16[i];
}

// Unkeyed BLAKE3 over a short (<= 64 byte) input — used for b_seed and a_seed,
// which hash a 64-byte concatenation.
extern "C" __global__ void pearl_blake3_short(const uint8_t *in, uint32_t len,
                                              uint8_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];
  uint8_t tail[64];
#pragma unroll
  for (int i = 0; i < 64; i++) tail[i] = (i < len) ? in[i] : 0;
  uint32_t block[16], out16[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    block[i] = (uint32_t)tail[i * 4] | ((uint32_t)tail[i * 4 + 1] << 8) |
               ((uint32_t)tail[i * 4 + 2] << 16) |
               ((uint32_t)tail[i * 4 + 3] << 24);
  }
  blake3_compress(cv, block, 0, len, CHUNK_START | CHUNK_END | ROOT, out16);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i * 4 + 0] = (uint8_t)(out16[i]);
    out[i * 4 + 1] = (uint8_t)(out16[i] >> 8);
    out[i * 4 + 2] = (uint8_t)(out16[i] >> 16);
    out[i * 4 + 3] = (uint8_t)(out16[i] >> 24);
  }
}


// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Generate one low-rank noise factor: E[rows, rank] drawn from keyed BLAKE3.
// E_A = E_AL·E_AR and E_B = E_BL·E_BR are each built from two of these, which is
// what makes the noise cheap to regenerate on the fly instead of storing a full
// dense matrix.
extern "C" __global__ void pearl_gen_noise(const uint32_t *seed, int8_t *out,
                                           uint32_t rows, uint32_t rank,
                                           uint64_t index_base) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = rows * rank;
  if (idx >= total) return;
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key[i] = seed[i];
  out[idx] = noise_draw(key, index_base + idx);
}

// Keyed hash of a padded operand block — hash_a / hash_b. One block per launch;
// the operands are padded to 1024 bytes as the spec requires.
extern "C" __global__ void pearl_hash_operands(const uint32_t *job_key,
                                               const uint8_t *padded,
                                               uint32_t len, uint8_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key[i] = job_key[i];
  blake3_keyed(key, padded, len, out);
}


// Materialise a noised operand once per job:
//   A'[r,kk] = A[r,kk] + Σ_j E_AL[r,j]·E_AR[j,kk]
// One thread per (r,kk). This is the O(rank²) work the fold used to redo for
// every region — hoisting it here turns it from a per-attempt cost into a
// per-job cost, which is the single biggest lever in the whole pipeline: the
// fold then reads a precomputed int32 and does a plain dot product.
//
// The result is int32, not int8: A is int7 and each of the rank noise products
// is up to 63·63, so a rank-128 sum reaches ~5·10⁵ — far outside int8 but well
// inside int32. Storing int8 here would silently truncate every value.
extern "C" __global__ void pearl_materialize(const int8_t *__restrict__ base,
                                             const int8_t *__restrict__ EL,
                                             const int8_t *__restrict__ ER,
                                             int32_t *__restrict__ out,
                                             uint32_t rows, uint32_t k,
                                             uint32_t rank) {
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (uint64_t)rows * k) return;
  const uint32_t r = (uint32_t)(idx / k);
  const uint32_t kk = (uint32_t)(idx % k);
  int32_t v = base[idx];
  for (uint32_t j = 0; j < rank; j++)
    v += (int32_t)EL[(size_t)r * rank + j] * (int32_t)ER[(size_t)j * k + kk];
  out[idx] = v;
}

// The heart of the PoW: accumulate C in `rank`-sized chunks and fold the
// mandated sub-tile of each chunk into the 16-lane jackpot transcript.
//
//   jackpot[tid] = rotl13(jackpot[tid]) ^ xor(tile),  tid = chunk % 16
//
// ONE BLOCK PER REGION. The first version ran <<<1, threads>>> and searched one
// region per launch, which used a single SM of the 128 on a 4090 and never
// finished a batch at the mainnet profile — 90 s of 100% utilisation and not one
// completed attempt. Regions are independent, so they are the natural axis to
// parallelise over: blockIdx.x IS the region offset from region_base, and each
// block writes its own transcript to jackpot_out[blockIdx.x].
//
// THE OPERANDS ARE RECONSTRUCTED ONCE, NOT PER CELL. The noised values are
//   A'[r,kk] = A[r,kk] + Σ_j E_AL[r,j]·E_AR[j,kk]
//   B'[c,kk] = B[c,kk] + Σ_j E_BL[c,j]·E_BR[j,kk]
// and the naive loop recomputed A'[r,kk] once for every column sharing that row
// — 64 times over, for a rank-length dot product each time. Hoisting both into
// shared memory turns (rows·cols·rank·2rank) into ((rows+cols)·rank² + cells·rank).
//
// This is still the scalar path. The tensor-core mainloop replaces only the
// accumulation below; the transcript semantics are what the parity vectors pin.
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *__restrict__ Aprime,   // [m, k] noised, materialised
    const int32_t *__restrict__ Bprime,   // [n, k] noised, materialised
    const uint32_t *__restrict__ rows_pattern,
    const uint32_t *__restrict__ cols_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    uint32_t chunks, uint64_t region_base, uint32_t *__restrict__ jackpot_out) {
  // ONE WARP PER REGION, ONE LANE PER TILE CELL.
  //
  // The tile is rows_count * cols_count = 4 * 8 = 32 cells, which is exactly a
  // warp. That makes the whole per-chunk reduction a shuffle: no __syncthreads,
  // no shared memory, no barrier at all.
  //
  // The previous block-per-region version staged operands into shared memory and
  // ran a log-depth reduction, costing ~9 barriers per chunk and ~144 per region
  // to protect just 4096 multiply-accumulates. Measured on a 4090 it sat at
  // 530 GH/s — 0.66% of the card's scalar int32 peak — and was flat across every
  // m/n from 512 to 8192, which is the signature of synchronisation overhead
  // rather than a memory bound.
  //
  // Lanes in a warp share only 4 distinct A rows and 8 distinct B columns, so the
  // loads coalesce and hit cache without needing an explicit staging pass.
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps_per_block = blockDim.x >> 5;
  const uint64_t region = region_base + (uint64_t)blockIdx.x * warps_per_block + warp;

  const uint32_t row_off = (uint32_t)(region % m);
  const uint32_t col_off = (uint32_t)((region / m) % n);

  // Cell assignment. Guarded so a profile whose tile is not exactly 32 cells
  // parks the surplus lanes rather than reading out of bounds.
  const uint32_t tile_cells = rows_count * cols_count;
  const bool active = lane < tile_cells;
  const uint32_t ri = active ? lane / cols_count : 0u;
  const uint32_t ci = active ? lane % cols_count : 0u;
  const uint32_t r = (rows_pattern[ri] + row_off) % m;
  const uint32_t c = (cols_pattern[ci] + col_off) % n;
  const int32_t *__restrict__ arow = Aprime + (size_t)r * k;
  const int32_t *__restrict__ bcol = Bprime + (size_t)c * k;

  uint32_t jackpot[PEARL_JACKPOT_BUCKETS];
#pragma unroll
  for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) jackpot[i] = 0u;

  for (uint32_t chunk = 0; chunk < chunks; chunk++) {
    const uint32_t k0 = chunk * rank;
    int32_t acc = 0;
    if (active) {
      const uint32_t lim = (k0 + rank <= k) ? rank : (k - k0);
#pragma unroll 4
      for (uint32_t t = 0; t < lim; t++) acc += arow[k0 + t] * bcol[k0 + t];
    }
    // Warp-wide XOR. Inactive lanes contribute zero, which is the XOR identity.
    uint32_t x = (uint32_t)acc;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, s);
    if (lane == 0) {
      const uint32_t l = chunk % PEARL_JACKPOT_BUCKETS;
      jackpot[l] = pearl_rotl13(jackpot[l]) ^ x;
    }
  }

  if (lane == 0) {
    uint32_t *out = jackpot_out
        + ((size_t)blockIdx.x * warps_per_block + warp) * PEARL_JACKPOT_BUCKETS;
#pragma unroll
    for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) out[i] = jackpot[i];
  }
}

// Hash and target-test a whole batch of transcripts, one thread per region.
// Sequentialising this over 4096 regions with a device-to-host copy each time
// was most of what made the old search slow even before the single-block fold.
extern "C" __global__ void pearl_finalize_many(const uint32_t *a_seed,
                                               const uint32_t *jackpots,
                                               uint32_t count,
                                               const uint8_t *target_be,
                                               uint8_t *hashes_out,
                                               int *flags_out) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const uint32_t *j = jackpots + (size_t)idx * PEARL_JACKPOT_BUCKETS;
  uint8_t transcript[PEARL_JACKPOT_BUCKETS * 4];
#pragma unroll
  for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) {
    transcript[i * 4 + 0] = (uint8_t)(j[i]);
    transcript[i * 4 + 1] = (uint8_t)(j[i] >> 8);
    transcript[i * 4 + 2] = (uint8_t)(j[i] >> 16);
    transcript[i * 4 + 3] = (uint8_t)(j[i] >> 24);
  }
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key[i] = a_seed[i];
  uint8_t h[PEARL_HASH_BYTES];
  blake3_keyed(key, transcript, sizeof(transcript), h);
#pragma unroll
  for (int i = 0; i < PEARL_HASH_BYTES; i++) hashes_out[(size_t)idx * PEARL_HASH_BYTES + i] = h[i];
  flags_out[idx] = pearl_meets_target(h, target_be);
}

// Hash the 64-byte transcript under a_seed and test it against the target. The
// host re-checks every reported hit in JS before submitting, so a bug here can
// waste work but can never push a bad share to the pool.
extern "C" __global__ void pearl_finalize(const uint32_t *a_seed,
                                          const uint32_t *jackpot,
                                          const uint8_t *target_be,
                                          uint8_t *hash_out, int *is_share) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint8_t transcript[PEARL_JACKPOT_BUCKETS * 4];
#pragma unroll
  for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) {
    transcript[i * 4 + 0] = (uint8_t)(jackpot[i]);
    transcript[i * 4 + 1] = (uint8_t)(jackpot[i] >> 8);
    transcript[i * 4 + 2] = (uint8_t)(jackpot[i] >> 16);
    transcript[i * 4 + 3] = (uint8_t)(jackpot[i] >> 24);
  }
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key[i] = a_seed[i];
  blake3_keyed(key, transcript, sizeof(transcript), hash_out);
  *is_share = pearl_meets_target(hash_out, target_be);
}
