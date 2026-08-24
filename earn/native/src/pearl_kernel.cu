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
//   4. E_A = E_AL·E_AR, E_B = E_BL·E_BR   (E_AR/E_BL are sparse ±1 selectors,
//      so this is two lookups per element, not a rank-length dot product)
//                                                    [pearl_gen_dense/_perm]
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

// The BLAKE3 message permutation is applied below with LITERAL indices rather
// than read from an array. It used to live in __constant__ memory and be
// applied as m[BLAKE3_MSG_PERM[i]], which is a runtime index into a local
// array — so nvcc placed the whole 16-word message block in LOCAL memory and
// every one of the ~112 accesses per compression became a memory round trip.
// The permutation, for reference:
//   {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}

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

  // Fully unrolled, so every index into s and m is a compile-time constant and
  // both stay in registers. A rolled loop here costs far more than the seven
  // copies of the round: the arrays spill, and the hash is the fixed per-region
  // cost that dominates the search.
#pragma unroll
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
      // Literal indices only — see the note by the permutation above.
      const uint32_t p0 = m[2], p1 = m[6], p2 = m[3], p3 = m[10];
      const uint32_t p4 = m[7], p5 = m[0], p6 = m[4], p7 = m[13];
      const uint32_t p8 = m[1], p9 = m[11], p10 = m[12], p11 = m[5];
      const uint32_t p12 = m[9], p13 = m[14], p14 = m[15], p15 = m[8];
      m[0] = p0; m[1] = p1; m[2] = p2; m[3] = p3;
      m[4] = p4; m[5] = p5; m[6] = p6; m[7] = p7;
      m[8] = p8; m[9] = p9; m[10] = p10; m[11] = p11;
      m[12] = p12; m[13] = p13; m[14] = p14; m[15] = p15;
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
// BLAKE3 over at most one 1024-byte chunk, parameterised by its starting
// chaining value and base flags. Keyed mode seeds the CV from the key and sets
// KEYED_HASH; unkeyed mode starts from the IV with no extra flag. They are
// DIFFERENT FUNCTIONS — hashing data with a zero KEY does not give the unkeyed
// digest. That is exactly the bug that had the device deriving a different
// job_key from the oracle while both stayed perfectly self-consistent.
__device__ void blake3_one_chunk(const uint32_t cv_init[8], uint32_t base_flags,
                                 const uint8_t *input, uint32_t len,
                                 uint8_t out[32]) {
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = cv_init[i];

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
    blake3_compress(cv, block, 0, 64, base_flags | flags_start, out16);
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
  blake3_compress(cv, block, 0, rem, base_flags | flags_start | CHUNK_END | ROOT,
                  out16);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i * 4 + 0] = (uint8_t)(out16[i]);
    out[i * 4 + 1] = (uint8_t)(out16[i] >> 8);
    out[i * 4 + 2] = (uint8_t)(out16[i] >> 16);
    out[i * 4 + 3] = (uint8_t)(out16[i] >> 24);
  }
}

// Keyed BLAKE3 over at most one chunk: the padded operand blocks, the 64-byte
// transcript, and the cert-v3 root binding.
__device__ __forceinline__ void blake3_keyed(const uint32_t key[8],
                                             const uint8_t *input, uint32_t len,
                                             uint8_t out[32]) {
  blake3_one_chunk(key, KEYED_HASH, input, len, out);
}

// The noise RNG, ported from the reference's get_random_hash(). The message is
// 64 bytes laid out as
//
//   [ 8 int32 slots | 32-byte seed LABEL ]
//
// with slot `prepend` holding (1 + index), and the commitment-derived seed used
// as the BLAKE3 KEY. Note which value plays which role: the reference names its
// parameters the other way round (`seed` is the label, `key` is the commitment
// hash), and transposing them is silent and fatal.
//
// The +1 on the index exists so that entry 0 of the dense and sparse streams
// cannot coincide; the two streams are otherwise separated only by which slot
// the index lands in (0 for dense, 1 for sparse).
__device__ __forceinline__ void pearl_random_hash(const uint32_t key[8],
                                                  const uint8_t label[32],
                                                  uint32_t index, int prepend,
                                                  uint8_t out[32]) {
  uint8_t msg[64];
#pragma unroll
  for (int i = 0; i < 32; i++) msg[i] = 0;
  const uint32_t v = index + 1u;
  msg[prepend * 4 + 0] = (uint8_t)(v);
  msg[prepend * 4 + 1] = (uint8_t)(v >> 8);
  msg[prepend * 4 + 2] = (uint8_t)(v >> 16);
  msg[prepend * 4 + 3] = (uint8_t)(v >> 24);
#pragma unroll
  for (int i = 0; i < 32; i++) msg[32 + i] = label[i];
  blake3_keyed(key, msg, 64, out);
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
// Unkeyed BLAKE3 over at most one chunk. This is what job_key and both seed
// links use: the reference passes None as the key, NOT a zero key.
//
// The version this replaces handled only 64 bytes, which silently truncated
// job_key's 128-byte input (header76 ‖ config52) to its first block — and hashed
// it keyed with zeros besides.
extern "C" __global__ void pearl_blake3_unkeyed(const uint8_t *in, uint32_t len,
                                                uint8_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  blake3_one_chunk(BLAKE3_IV, 0, in, len, out);
}

// cert-v3 root binding: blake3(root ‖ dim_le32 ‖ 28 zeros, key=salt).
//
// This is what commits m and n. They are the miner's own choice and are
// deliberately absent from config52, so without this nothing anywhere in the
// chain pins the dimensions the work was actually done at.
extern "C" __global__ void pearl_bind_root(const uint8_t *salt,
                                           const uint8_t *root, uint32_t dim,
                                           uint8_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint8_t msg[64];
#pragma unroll
  for (int i = 0; i < 64; i++) msg[i] = 0;
#pragma unroll
  for (int i = 0; i < 32; i++) msg[i] = root[i];
  msg[32] = (uint8_t)(dim);
  msg[33] = (uint8_t)(dim >> 8);
  msg[34] = (uint8_t)(dim >> 16);
  msg[35] = (uint8_t)(dim >> 24);

  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    key[i] = (uint32_t)salt[i * 4] | ((uint32_t)salt[i * 4 + 1] << 8) |
             ((uint32_t)salt[i * 4 + 2] << 16) |
             ((uint32_t)salt[i * 4 + 3] << 24);
  }
  blake3_keyed(key, msg, 64, out);
}


// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// The noise is a product of two factors, and they are NOT both dense. This is
// the single most important structural fact in the whole core:
//
//   E_AL, E_BR   dense,  (rows x rank), values in [-32, 32)
//   E_AR, E_BL   SPARSE, (k x rank), each row exactly one +1 and one -1
//
// So the noise for one element is a DIFFERENCE OF TWO LOOKUPS,
//
//   noise[r][kk] = dense[r][p0(kk)] - dense[r][p1(kk)]
//
// not a rank-length dot product. An earlier version of this file treated both
// factors as dense and reconstructed at full rank — internally consistent,
// `rank` times too expensive, and producing a transcript no pool would accept.
//
// Both are int7-ranged (a difference of two [-32, 32) draws is at most 63), so
// adding the noise to an int7 operand still lands inside int8. That is exactly
// what the configuration's Int7xInt7ToInt32 name is telling us.

// Dense factor: `rank` values per requested row, (byte & 63) - 32.
// One thread per 32-byte digest. The byte stream is GLOBAL and indexed by
// row*rank, so a row's draw is tied to its absolute index — which is what makes
// the tile offset change the noise instead of every offset seeing one draw.
extern "C" __global__ void pearl_gen_dense(const uint32_t *seed,
                                           const uint8_t *label,
                                           const uint32_t *row_indices,
                                           int8_t *out, uint32_t num_rows,
                                           uint32_t rank) {
  const uint32_t blocks_per_row = rank >> 5;  // rank is a multiple of 32
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rows * blocks_per_row) return;
  const uint32_t ri = idx / blocks_per_row;
  const uint32_t blk = idx % blocks_per_row;

  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) key[i] = seed[i];
  uint8_t lab[32];
#pragma unroll
  for (int i = 0; i < 32; i++) lab[i] = label[i];

  uint8_t h[32];
  // A null row_indices means "the whole operand", where a row's index IS its
  // position. The parameter exists for the verifier's path, which only ever
  // wants the handful of rows in one tile.
  const uint32_t row = row_indices ? row_indices[ri] : ri;
  pearl_random_hash(key, lab, ((row * rank) >> 5) + blk, 0, h);

  int8_t *dst = out + (size_t)ri * rank + (blk << 5);
#pragma unroll
  for (int i = 0; i < 32; i++) dst[i] = (int8_t)((int32_t)(h[i] & 63) - 32);
}

// Sparse factor: k rows, each a (+1 at p0, -1 at p1) pair, written as two u32.
// One thread per digest, which covers eight rows.
//
// p1 = p0 ^ (1 + mulhi(rank-1, u)) is always distinct from p0 and always inside
// [0, rank): mulhi(rank-1, u) <= rank-2, so the xor operand is in [1, rank-1]
// and rank is a power of two.
//
// This factor does NOT depend on the tile offset — only on the job seeds — so
// the host generates it once per commitment and every attempt reuses it.
extern "C" __global__ void pearl_gen_perm(const uint32_t *seed,
                                          const uint8_t *label,
                                          uint32_t *out, uint32_t k,
                                          uint32_t rank) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (k + 7u) / 8u) return;

  uint32_t key[8];
#pragma unroll
  for (int i2 = 0; i2 < 8; i2++) key[i2] = seed[i2];
  uint8_t lab[32];
#pragma unroll
  for (int i2 = 0; i2 < 32; i2++) lab[i2] = label[i2];

  uint8_t h[32];
  pearl_random_hash(key, lab, i, 1, h);

  const uint32_t mask = rank - 1u;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t row = i * 8u + (uint32_t)j;
    if (row >= k) break;
    const uint32_t u = (uint32_t)h[j * 4] | ((uint32_t)h[j * 4 + 1] << 8) |
                       ((uint32_t)h[j * 4 + 2] << 16) |
                       ((uint32_t)h[j * 4 + 3] << 24);
    const uint32_t p0 = u & mask;
    out[row * 2u] = p0;
    out[row * 2u + 1u] = p0 ^ (1u + __umulhi(rank - 1u, u));
  }
}

// Synthesise the miner's own operands. m and n are the miner's choice of
// workload and are not protocol, so the CONTENTS here are arbitrary — but the
// RANGE is not. Values must be int7 ([-63, 63]), because the noise adds another
// int7 and the sum has to stay inside int8 for the Int7xInt7ToInt32 MMA.
//
// One hash per 32 output bytes rather than one per byte.
extern "C" __global__ void pearl_gen_operand(const uint32_t *key,
                                             const uint8_t *label, int8_t *out,
                                             uint64_t total) {
  const uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base = blk * 32u;
  if (base >= total) return;

  uint32_t kbuf[8];
#pragma unroll
  for (int i = 0; i < 8; i++) kbuf[i] = key[i];
  uint8_t lab[32];
#pragma unroll
  for (int i = 0; i < 32; i++) lab[i] = label[i];

  uint8_t h[32];
  pearl_random_hash(kbuf, lab, (uint32_t)blk, 0, h);
#pragma unroll
  for (int i = 0; i < 32; i++) {
    if (base + (uint64_t)i < total)
      out[base + i] = (int8_t)((int32_t)(h[i] % 127) - 63);
  }
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


// Materialise a noised operand once per commitment:
//   A'[r,kk] = sat_i8( A[r,kk] + dense[r][p0(kk)] - dense[r][p1(kk)] )
// One thread per (r,kk), two lookups and a subtract each.
//
// The output is int8, matching the reference, which converts its noised operand
// back down before the main GEMM:
//
//   pearl::convert_type_out(tCrApEA, tCrApEA_int8);   // pearl_noisingA_kernel.h
//
// That is a cutlass::NumericArrayConverter<int8_t, int32_t, 4>, i.e. a
// saturating cvt.pack.sat.s8.s32. It clamps rather than wrapping, but with int7
// operands and int7 noise the sum is already inside int8, so the clamp is a
// guard rail rather than the main effect.
//
// int8 output is also what makes the fold's __dp4a path (and, later, the int8
// tensor cores) usable at all — an int32 operand rules both out.
extern "C" __global__ void pearl_materialize(const int8_t *__restrict__ base,
                                             const int8_t *__restrict__ dense,
                                             const uint32_t *__restrict__ perm,
                                             int8_t *__restrict__ out,
                                             uint32_t rows, uint32_t k,
                                             uint32_t rank) {
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (uint64_t)rows * k) return;
  const uint32_t r = (uint32_t)(idx / k);
  const uint32_t kk = (uint32_t)(idx % k);
  const int8_t *__restrict__ row = dense + (size_t)r * rank;
  const int32_t v = (int32_t)base[idx] + (int32_t)row[perm[kk * 2u]] -
                    (int32_t)row[perm[kk * 2u + 1u]];
  out[idx] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
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
// Partial dot products for one batch: D[chunk][r][c].
//
// THE REDUNDANCY THIS REMOVES. A region folds 32 cells, each a dot product of
// A'[rows_pattern[i] + row_off] against B'[cols_pattern[j] + col_off]. Across a
// batch, row_off sweeps every row — so row R is reached by FOUR different
// regions, once per rows_pattern element, and the fold recomputed its dot
// product every time. Exactly 4x more arithmetic than the batch actually needs.
//
// Computing each distinct partial once into D and having the fold gather from it
// removes that. It also changes the shape of the work: D is a dense
// [m x cols_count] GEMM with a k-reduction, which is what tensor cores want,
// where the per-warp tile fold never was.
//
// A batch shares one col_off (regions are launched m at a time from a multiple
// of m), so D only needs the 8 columns that batch touches: chunks*m*8 int32,
// 2 MiB at the mainnet geometry, comfortably L2-resident.
extern "C" __global__ void pearl_partials(const int8_t *__restrict__ Aprime,
                                          const int8_t *__restrict__ Bprime,
                                          const uint32_t *__restrict__ cols_pattern,
                                          uint32_t cols_count, uint32_t m,
                                          uint32_t n, uint32_t k, uint32_t rank,
                                          uint32_t chunks, uint32_t col_off,
                                          uint32_t col_groups,
                                          int32_t *__restrict__ D) {
  // One thread per (column group, chunk, row), producing ALL of that row's
  // columns for its group.
  //
  // A thread per (chunk, row, col) reads the row's k-slice once per column —
  // eight times over, and the A side is the streaming operand: 33 MiB a batch
  // read eight times is 268 MiB. Holding the row in registers and accumulating
  // eight columns against it reads it once.
  //
  // The B side is the opposite: only eight distinct column slices exist per
  // batch and every thread wants them, so they stay resident in cache.
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t total = (uint64_t)chunks * m;
  if (idx >= total) return;

  const uint32_t r = (uint32_t)(idx % m);
  const uint32_t chunk = (uint32_t)(idx / m);
  const uint32_t k0 = chunk * rank;
  const uint32_t lim = (k0 + rank <= k) ? rank : (k - k0);

  const int8_t *__restrict__ ar = Aprime + (size_t)r * k + k0;

  // Hold this row's k-slice in REGISTERS and reuse it across every column
  // group. It used to be re-read once per group, which at 64 groups over a
  // 12.6 MiB operand is about 805 MiB of traffic per batch on a card with
  // roughly 1 TB/s — where the time went once the fold stopped dominating.
  //
  // The loops below are bounded by the compile-time PEARL_MAX_A_QUADS and
  // break on the runtime count, rather than being bounded by the runtime count
  // directly. That matters: a runtime bound makes `av` dynamically indexed, and
  // nvcc then spills the whole array to local memory, which is exactly the
  // traffic this is trying to avoid.
  const uint32_t quads = lim >> 4;
  const bool vectorised = ((lim & 15u) == 0u) && quads <= PEARL_MAX_A_QUADS;

  int4 av[PEARL_MAX_A_QUADS];
  if (vectorised) {
    const int4 *a4 = reinterpret_cast<const int4 *>(ar);
#pragma unroll
    for (uint32_t q = 0; q < PEARL_MAX_A_QUADS; q++) {
      if (q >= quads) break;
      av[q] = a4[q];
    }
  }

  for (uint32_t cg = 0; cg < col_groups; cg++) {
    // Only eight distinct column slices exist per group and every thread in the
    // block wants the same ones, so they stay resident in cache.
    const int8_t *bc[PEARL_COLS_COUNT];
#pragma unroll
    for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) {
      const uint32_t cc = (cols_pattern[c] + col_off + cg) % n;
      bc[c] = Bprime + (size_t)cc * k + k0;
    }

    int32_t acc[PEARL_COLS_COUNT];
#pragma unroll
    for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) acc[c] = 0;

    if (vectorised) {
      // One 16-byte load of B per column feeds 16 multiply-accumulates against
      // the A slice already in registers, issued as __dp4a (four int8 MACs
      // each). The int8 operands are what make this possible at all.
#pragma unroll
      for (uint32_t q = 0; q < PEARL_MAX_A_QUADS; q++) {
        if (q >= quads) break;
        const int32_t *aw = reinterpret_cast<const int32_t *>(&av[q]);
#pragma unroll
        for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) {
          const int4 bv = reinterpret_cast<const int4 *>(bc[c])[q];
          const int32_t *bw = reinterpret_cast<const int32_t *>(&bv);
          int32_t s = acc[c];
#pragma unroll
          for (int w = 0; w < 4; w++) s = __dp4a(aw[w], bw[w], s);
          acc[c] = s;
        }
      }
    } else {
      for (uint32_t t = 0; t < lim; t++) {
        const int32_t a = (int32_t)ar[t];
#pragma unroll
        for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) acc[c] += a * bc[c][t];
      }
    }

    // XOR the eight columns together HERE rather than storing them. The
    // transcript folds a tile by XOR, and XOR is associative and commutative:
    //
    //   tile_xor = XOR over ri of ( XOR over ci of C[r_ri][c_ci] )
    //
    // Every column of a row is in the tile, so the inner XOR depends only on
    // the row and can be collapsed at the producer.
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) x ^= (uint32_t)acc[c];
    D[((size_t)cg * chunks + chunk) * m + r] = (int32_t)x;
  }
}

// The fold is now a gather. Every product it needs is already in D, so a region
// costs 32 loads and a warp reduction per chunk instead of 32 dot products.
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *__restrict__ D,
    const uint32_t *__restrict__ rows_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t chunks, uint64_t region_base,
    uint32_t *__restrict__ jackpot_out) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps_per_block = blockDim.x >> 5;

  // PEARL_REGIONS_PER_WARP regions share a warp, each using rows_count lanes.
  //
  // The producer already XORed each row's columns together, so a region needs
  // only rows_count values combined — four at the mandated tile. Giving each
  // region a whole warp left 28 of 32 lanes idle, and measured per-stage timing
  // put this kernel at 42% of the batch, the largest single share. Packing
  // eight regions per warp fills it.
  (void)cols_count;
  const uint32_t sub = lane / rows_count;   // which region within the warp
  const uint32_t ri = lane % rows_count;    // which row of that region's tile
  const bool active = sub < PEARL_REGIONS_PER_WARP;

  const uint64_t slot =
      ((uint64_t)blockIdx.x * warps_per_block + warp) * PEARL_REGIONS_PER_WARP
      + (active ? sub : 0u);
  const uint32_t cg = (uint32_t)(slot / m);
  const uint32_t row_off = (uint32_t)(slot % m);
  const uint32_t r = (rows_pattern[ri] + row_off) % m;

  uint32_t jackpot[PEARL_JACKPOT_BUCKETS];
#pragma unroll
  for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) jackpot[i] = 0u;

  for (uint32_t chunk = 0; chunk < chunks; chunk++) {
    // Lanes sharing a row read cols_count contiguous ints — one transaction.
    const int32_t v =
        active ? D[((size_t)cg * chunks + chunk) * m + r] : 0;
    uint32_t x = (uint32_t)v;
    // Reduce only within each region's own lanes, not across the whole warp.
#pragma unroll
    for (uint32_t sft = 1; sft < PEARL_ROWS_COUNT; sft <<= 1) {
      x ^= __shfl_xor_sync(0xffffffffu, x, sft);
    }
    if (ri == 0) {
      const uint32_t l = chunk % PEARL_JACKPOT_BUCKETS;
      jackpot[l] = pearl_rotl13(jackpot[l]) ^ x;
    }
  }

  if (ri == 0 && active) {
    uint32_t *out = jackpot_out + (size_t)slot * PEARL_JACKPOT_BUCKETS;
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
                                               uint32_t *hit_count,
                                               uint32_t *hit_index) {
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
  // Write ONLY on a hit. Storing every region's 32-byte hash unconditionally
  // was 12.6 MiB a batch at the mainnet geometry, spent almost entirely on
  // hashes that miss, and it forced the host to read a per-region flag array
  // back across the bus to find the one that did not.
  if (!pearl_meets_target(h, target_be)) return;
  const uint32_t slot = atomicAdd(hit_count, 1u);
  if (slot >= PEARL_MAX_HITS) return;
  hit_index[slot] = idx;
#pragma unroll
  for (int i = 0; i < PEARL_HASH_BYTES; i++) {
    hashes_out[(size_t)slot * PEARL_HASH_BYTES + i] = h[i];
  }
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
