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

#include <mma.h>

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
                                             uint64_t total, uint64_t salt) {
  const uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base = blk * 32u;
  if (base >= total) return;

  uint32_t kbuf[8];
#pragma unroll
  for (int i = 0; i < 8; i++) kbuf[i] = key[i];

  // The SALT is what gives the search somewhere to go.
  //
  // The Pearl header has no nonce -- it is version, prev_block, merkle_root,
  // timestamp, nbits and nothing else, which is why the type is called
  // IncompleteBlockHeader. The miner's free variable is its own choice of A and
  // B: different operands commit to different roots, which seed different
  // noise, which is a completely fresh space of regions to fold.
  //
  // Without it the operands were a function of job_key alone, so one pool job
  // offered exactly m*n distinct regions -- 151M, which this miner exhausts in
  // 0.133 seconds. Everything after that re-mined identical regions at full
  // reported hashrate. Expected time to a share was about 20 hours rather than
  // the four minutes the rate implies.
  //
  // Salt 0 leaves the message byte-identical to the unsalted version, so the
  // frozen parity vectors still describe it.
  uint8_t msg[64];
#pragma unroll
  for (int i = 0; i < 32; i++) msg[i] = 0;
  const uint32_t v = (uint32_t)blk + 1u;
  msg[0] = (uint8_t)(v);
  msg[1] = (uint8_t)(v >> 8);
  msg[2] = (uint8_t)(v >> 16);
  msg[3] = (uint8_t)(v >> 24);
#pragma unroll
  for (int i = 0; i < 8; i++) msg[8 + i] = (uint8_t)(salt >> (i * 8));
#pragma unroll
  for (int i = 0; i < 32; i++) msg[32 + i] = label[i];

  uint8_t h[32];
  blake3_keyed(kbuf, msg, 64, h);
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
  const uint32_t row_blocks = m / PEARL_ROWS_PER_THREAD;
  const uint64_t total = (uint64_t)chunks * row_blocks;
  if (idx >= total) return;

  const uint32_t rb = (uint32_t)(idx % row_blocks);
  const uint32_t chunk = (uint32_t)(idx / row_blocks);
  const uint32_t r0 = rb * PEARL_ROWS_PER_THREAD;
  const uint32_t k0 = chunk * rank;
  const uint32_t lim = (k0 + rank <= k) ? rank : (k - k0);

  // Hold each row's k-slice in REGISTERS and reuse it across every column
  // group. Re-reading it per group is about 805 MB a batch at 64 groups over a
  // 12.6 MiB operand, on a card with roughly 1 TB/s.
  //
  // The loops are bounded by the compile-time PEARL_MAX_A_QUADS and break on
  // the runtime count, rather than being bounded by the runtime count directly.
  // A runtime bound makes `av` dynamically indexed, and nvcc then spills the
  // whole array to local memory — exactly the traffic this avoids.
  const uint32_t quads = lim >> 4;
  const bool vectorised = ((lim & 15u) == 0u) && quads <= PEARL_MAX_A_QUADS;

  int4 av[PEARL_ROWS_PER_THREAD][PEARL_MAX_A_QUADS];
  if (vectorised) {
#pragma unroll
    for (uint32_t rr = 0; rr < PEARL_ROWS_PER_THREAD; rr++) {
      const int4 *a4 =
          reinterpret_cast<const int4 *>(Aprime + (size_t)(r0 + rr) * k + k0);
#pragma unroll
      for (uint32_t q = 0; q < PEARL_MAX_A_QUADS; q++) {
        if (q >= quads) break;
        av[rr][q] = a4[q];
      }
    }
  }

  for (uint32_t cg = 0; cg < col_groups; cg++) {
    // col_off is a valid-offset INDEX; expanding it gives an offset with the
    // column pattern's bits clear, so the tile column is a bitwise OR.
    const uint32_t coff = pearl_expand_offset(col_off + cg, PEARL_COLS_MASK);
    // The tile's columns are CONTIGUOUS, so they need one base pointer and a
    // stride rather than an array of sixteen. That array cost thirty-two
    // registers a thread, which at sixteen columns was enough to halve
    // occupancy and with it the achieved throughput.
    (void)cols_pattern;
    const int8_t *__restrict__ bbase = Bprime + (size_t)coff * k + k0;

    int32_t acc[PEARL_ROWS_PER_THREAD][PEARL_COLS_COUNT];
#pragma unroll
    for (uint32_t rr = 0; rr < PEARL_ROWS_PER_THREAD; rr++) {
#pragma unroll
      for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) acc[rr][c] = 0;
    }

    if (vectorised) {
      // One 16-byte load of B now feeds PEARL_ROWS_PER_THREAD * 4 __dp4a
      // instead of 4. Every thread in the warp wants the same B, so the load
      // broadcasts; what it buys is arithmetic per instruction issued.
#pragma unroll
      for (uint32_t q = 0; q < PEARL_MAX_A_QUADS; q++) {
        if (q >= quads) break;
#pragma unroll
        for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) {
          const int4 bv = reinterpret_cast<const int4 *>(bbase + (size_t)c * k)[q];
          const int32_t *bw = reinterpret_cast<const int32_t *>(&bv);
#pragma unroll
          for (uint32_t rr = 0; rr < PEARL_ROWS_PER_THREAD; rr++) {
            const int32_t *aw = reinterpret_cast<const int32_t *>(&av[rr][q]);
            int32_t s = acc[rr][c];
#pragma unroll
            for (int w = 0; w < 4; w++) s = __dp4a(aw[w], bw[w], s);
            acc[rr][c] = s;
          }
        }
      }
    } else {
      for (uint32_t t = 0; t < lim; t++) {
#pragma unroll
        for (uint32_t rr = 0; rr < PEARL_ROWS_PER_THREAD; rr++) {
          const int32_t a = (int32_t)Aprime[(size_t)(r0 + rr) * k + k0 + t];
#pragma unroll
          for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) acc[rr][c] += a * bbase[(size_t)c * k + t];
        }
      }
    }

    // XOR the eight columns together HERE rather than storing them. The
    // transcript folds a tile by XOR, and XOR is associative and commutative:
    //
    //   tile_xor = XOR over ri of ( XOR over ci of C[r_ri][c_ci] )
    //
    // Every column of a row is in the tile, so the inner XOR depends only on
    // the row and can be collapsed at the producer.
#pragma unroll
    for (uint32_t rr = 0; rr < PEARL_ROWS_PER_THREAD; rr++) {
      uint32_t x = 0u;
#pragma unroll
      for (uint32_t c = 0; c < PEARL_COLS_COUNT; c++) x ^= (uint32_t)acc[rr][c];
      D[((size_t)cg * chunks + chunk) * m + (r0 + rr)] = (int32_t)x;
    }
  }
}

// The same partials, on the int8 TENSOR CORES.
//
// The dp4a kernel tops out around an eighth of that instruction's own peak, and
// dp4a's peak is itself about half what the int8 tensor cores can do. Since
// valid tiles partition the grid there is no reuse left to exploit, so the
// reported hashrate IS the multiply-accumulate rate -- and closing the gap to a
// competitive miner means going to the tensor cores.
//
// The contiguous tile is what makes this clean. WMMA's int8 shape is 16x16x16,
// and:
//
//   - the tile's sixteen columns are consecutive, so they are exactly one B
//     fragment rather than sixteen scattered rows;
//   - valid row offsets are multiples of four, so a sixteen-row block is
//     precisely four consecutive row offsets and nothing is wasted;
//   - A is row-major [m][k] and B is [n][k], which for the product means A is
//     row_major with leading dimension k and B is COL_MAJOR with the same
//     leading dimension -- no staging or transpose needed.
//
// One warp computes a 16x16 block of C for one chunk and one column group.
// The tile fold, on the int8 tensor cores, with the accumulator kept ACROSS
// chunks -- which is what the protocol actually specifies.
//
// From the reference miner (zk-pow/src/ffi/mine.rs):
//
//   let mut jackpot_tile = vec![vec![0; tile_w]; tile_h];   // OUTSIDE the loop
//   for ll in (rank..=k).step_by(rank) {
//       ... jackpot_tile[u][v] += a_noised[..][l] * b_noised_t[..][l];
//       let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
//       jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
//   }
//
// jackpot_tile is declared outside the chunk loop and never reset, so the value
// XORed at chunk c is the dot product over ALL of k up to that point. This code
// used to reset per chunk, which computes a different function entirely -- and
// the only symptom is that no pool ever accepts a share.
//
// That also kills the two-stage split. A cumulative tile cannot be decomposed
// into reusable per-chunk partials (XOR of running sums is not a running XOR),
// so the fold fuses into the GEMM and the partial table disappears. Nothing is
// lost by that: valid tiles partition the grid, so there was no sharing between
// regions to exploit in the first place.
//
// One warp covers PEARL_WMMA_ROW_TILES 16-row blocks against one column group.
// A 16-row block is four consecutive row offsets, so a warp carries
// 4*PEARL_WMMA_ROW_TILES regions and emits a transcript for each.
// XOR of three words in one instruction. 0x96 is the lop3 truth table for
// a ^ b ^ c. ptxas often finds this itself, but the fold's reduction tree is
// hot enough to be worth stating outright.
__device__ __forceinline__ uint32_t pearl_xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// XOR across the whole warp. Ampere and later do this as a single REDUX.SYNC
// instead of the five-step shuffle butterfly it replaces -- and the fold runs
// one of these per region per chunk, which is often enough to matter.
__device__ __forceinline__ uint32_t pearl_warp_xor(uint32_t x) {
#if __CUDA_ARCH__ >= 800
  return __reduce_xor_sync(0xffffffffu, x);
#else
#pragma unroll
  for (uint32_t sft = 16; sft > 0; sft >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, sft);
  return x;
#endif
}

extern "C" __global__ void pearl_tile_fold_wmma(
    const int8_t *__restrict__ Aprime, const int8_t *__restrict__ Bprime,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank, uint32_t chunks,
    uint32_t col_off, uint32_t rows_valid, uint32_t col_groups,
    uint32_t *__restrict__ jackpot_out) {
  using namespace nvcuda;

  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps_per_block = blockDim.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;

  const uint32_t regions_per_warp = PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT);
  const uint32_t row_blocks = rows_valid / regions_per_warp;

  const uint64_t slot = (uint64_t)blockIdx.x * warps_per_block + warp;
  // Each warp now carries PEARL_WMMA_COL_BLK column groups, so there are that
  // many times fewer of them. Counting the old way let surplus warps compute a
  // column-group block past the end and write outside the transcript buffer.
  //
  // A surplus warp may NOT simply return: staging B is a block-wide cooperative
  // load, and a __syncthreads() some warps never reach hangs the launch.
  const bool active = slot < (uint64_t)row_blocks * (col_groups / PEARL_WMMA_COL_BLK);

  const uint32_t rb = (uint32_t)(slot % row_blocks);
  const uint32_t cgb = (uint32_t)(slot / row_blocks);        // column-group BLOCK
  const uint32_t r0idx = rb * regions_per_warp;              // first row-offset index
  const uint32_t kfrags = rank / 16;

  // The column-group block for the WHOLE CTA, taken from warp 0's slot rather
  // than each warp's own. Every warp in the block shares it -- consecutive
  // slots differ only in the row block -- and the host guarantees that by
  // refusing to stage unless row_blocks divides evenly by the warp count. It
  // has to come from blockIdx, because a surplus warp's own cgb is past the end.
  const uint32_t cg0 = (uint32_t)(((uint64_t)blockIdx.x * warps_per_block) / row_blocks)
                       * PEARL_WMMA_COL_BLK;

  // Shared: this warp's transcripts, then the staged B columns shared by the
  // whole block. The per-warp 16x16 unpack buffer is gone with the gather that
  // needed it.
  extern __shared__ uint32_t smem_u32[];
  // Stride by the FULL per-warp transcript block: regions times column groups.
  // Omitting the column-group factor made neighbouring warps share storage.
  uint32_t *jp = smem_u32 + (size_t)warp * regions_per_warp * PEARL_WMMA_COL_BLK
                                * PEARL_JACKPOT_BUCKETS;

  // B for this block, one chunk at a time. Every warp in the block wants the
  // same PEARL_WMMA_COL_BLK*16 columns, so reading them once here instead of
  // once per warp divides B's share of the traffic by the warp count. B is the
  // dominant term: a warp covers only PEARL_WMMA_ROW_TILES*16 rows, so B gets
  // re-read far more often than A does.
  int8_t *sB = (int8_t *)(smem_u32 + (size_t)warps_per_block * regions_per_warp
                                         * PEARL_WMMA_COL_BLK * PEARL_JACKPOT_BUCKETS);
  const uint32_t sb_cols = PEARL_WMMA_COL_BLK * 16;

  // A is staged too, and NOT to save traffic -- the CTA's warps each hold their
  // own rows, so nothing is shared. It is staged because of HOW wmma reads it.
  // load_matrix_sync from global has each lane fetch 16 bytes from a different
  // matrix row, k bytes apart: 32 scattered sectors per fragment. Pinning that
  // load to an L1-resident address took the kernel from 72 to 141 TH/s, so the
  // scatter, not the byte count, was costing about half of all runtime.
  //
  // Staging turns it into one coalesced bulk copy -- eight consecutive threads
  // per row, 128 contiguous bytes at a time -- and the wmma load then comes out
  // of shared memory.
  //
  // The CTA's rows are contiguous: warp w covers rows (rb0+w)*32 .. +31, and
  // consecutive slots differ in the row block first.
  const uint32_t sa_rows = warps_per_block * regions_per_warp * PEARL_ROWS_COUNT;
  int8_t *sA = sB + (size_t)sb_cols * PEARL_SB_STRIDE;
  const uint32_t rb0 = (uint32_t)(((uint64_t)blockIdx.x * warps_per_block) % row_blocks);
  const uint32_t row_base = rb0 * regions_per_warp * PEARL_ROWS_COUNT;

  if (active) {
    for (uint32_t i = lane; i < regions_per_warp * PEARL_WMMA_COL_BLK * PEARL_JACKPOT_BUCKETS; i += 32) jp[i] = 0u;
  }
  __syncwarp();

  // The accumulators live across the whole of k, not one chunk -- one per
  // (row block, column group) pair this warp carries.
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> cF[PEARL_WMMA_ROW_TILES][PEARL_WMMA_COL_BLK];
#pragma unroll
  for (uint32_t ti = 0; ti < PEARL_WMMA_ROW_TILES; ti++) {
#pragma unroll
    for (uint32_t cb = 0; cb < PEARL_WMMA_COL_BLK; cb++) wmma::fill_fragment(cF[ti][cb], 0);
  }

  for (uint32_t chunk = 0; chunk < chunks; chunk++) {
    const uint32_t k0 = chunk * rank;

    // Cooperative block-wide load of this chunk's B columns, as int4 so each
    // thread moves 16 bytes at a time. The two barriers are both required: the
    // first publishes the new chunk, the second keeps a fast warp from
    // overwriting it while a slow one is still reading the old one.
    __syncthreads();
    {
      const uint32_t quads = rank / 16;                 // int4 per row or column
      const uint32_t btotal = sb_cols * quads;
      for (uint32_t i = threadIdx.x; i < btotal; i += blockDim.x) {
        const uint32_t col = i / quads;
        const uint32_t q = i % quads;
        const uint32_t c = pearl_expand_offset(col_off + cg0 + (col >> 4), PEARL_COLS_MASK)
                           + (col & 15u);
        const int4 *src = (const int4 *)(Bprime + (size_t)c * k + k0 + q * 16);
        *(int4 *)(sB + (size_t)col * PEARL_SB_STRIDE + q * 16) = *src;
      }
      const uint32_t atotal = sa_rows * quads;
      for (uint32_t i = threadIdx.x; i < atotal; i += blockDim.x) {
        const uint32_t row = i / quads;
        const uint32_t q = i % quads;
        const int4 *src = (const int4 *)(Aprime + (size_t)(row_base + row) * k + k0 + q * 16);
        *(int4 *)(sA + (size_t)row * PEARL_SB_STRIDE + q * 16) = *src;
      }
    }
    __syncthreads();

    if (!active) continue;
    for (uint32_t t = 0; t < kfrags; t++) {
      // Load each A fragment ONCE and drive it against every column group this
      // warp carries. That is the whole point of the blocking.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> aF[PEARL_WMMA_ROW_TILES];
#pragma unroll
      for (uint32_t ti = 0; ti < PEARL_WMMA_ROW_TILES; ti++) {
        wmma::load_matrix_sync(
            aF[ti],
            sA + (size_t)(warp * regions_per_warp * PEARL_ROWS_COUNT + ti * PEARL_WMMA_ROWS)
                     * PEARL_SB_STRIDE + t * 16,
            PEARL_SB_STRIDE);
      }
#pragma unroll
      for (uint32_t cb = 0; cb < PEARL_WMMA_COL_BLK; cb++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> bF;
        wmma::load_matrix_sync(bF, sB + (size_t)(cb * 16) * PEARL_SB_STRIDE + t * 16,
                               PEARL_SB_STRIDE);
#pragma unroll
        for (uint32_t ti = 0; ti < PEARL_WMMA_ROW_TILES; ti++) {
          wmma::mma_sync(cF[ti][cb], aF[ti], bF, cF[ti][cb]);
        }
      }
    }

    // Chunk boundary: XOR the RUNNING tile and fold it into each region's lane.
    //
    // The region IS the fragment. A 16x16 tile is exactly what one int8 wmma
    // accumulator holds across the warp, so the XOR over the tile is the XOR
    // over every lane's registers and then one warp reduction. Which matrix
    // element lives in which register is unspecified -- and irrelevant, because
    // XOR does not care about order, only that each element appears once.
    //
    // What this replaces: a store_matrix_sync to shared and a strided gather
    // back, twice per fragment per chunk, because a 4x16 region was a quarter
    // of a fragment. That gather was two thirds of the kernel's runtime.
    const uint32_t bucket = chunk % PEARL_JACKPOT_BUCKETS;
#pragma unroll
    for (uint32_t ti = 0; ti < PEARL_WMMA_ROW_TILES; ti++) {
#pragma unroll
      for (uint32_t cb = 0; cb < PEARL_WMMA_COL_BLK; cb++) {
        // Fold this lane's slice of the fragment three words at a time, then
        // across the warp. Every lane holds part of the tile, so the warp
        // reduction spans all 32.
        uint32_t x = 0u;
#pragma unroll
        for (int i = 0; i + 2 < cF[ti][cb].num_elements; i += 3) {
          x ^= pearl_xor3((uint32_t)cF[ti][cb].x[i], (uint32_t)cF[ti][cb].x[i + 1],
                          (uint32_t)cF[ti][cb].x[i + 2]);
        }
#pragma unroll
        for (int i = (cF[ti][cb].num_elements / 3) * 3; i < cF[ti][cb].num_elements; i++) {
          x ^= (uint32_t)cF[ti][cb].x[i];
        }
        x = pearl_warp_xor(x);
        if (lane == 0) {
          uint32_t *slotjp = jp + (cb * regions_per_warp + ti) * PEARL_JACKPOT_BUCKETS;
          slotjp[bucket] = pearl_rotl13(slotjp[bucket]) ^ x;
        }
      }
    }
  }

  // Emit one transcript per region.
  if (!active) return;
  __syncwarp();
  const uint32_t total_jp = regions_per_warp * PEARL_WMMA_COL_BLK * PEARL_JACKPOT_BUCKETS;
  for (uint32_t i = lane; i < total_jp; i += 32) {
    const uint32_t cb = i / (regions_per_warp * PEARL_JACKPOT_BUCKETS);
    const uint32_t rest = i % (regions_per_warp * PEARL_JACKPOT_BUCKETS);
    const uint32_t reg = rest / PEARL_JACKPOT_BUCKETS;
    const uint32_t b = rest % PEARL_JACKPOT_BUCKETS;
    const uint64_t region = (uint64_t)(cg0 + cb) * rows_valid + r0idx + reg;
    jackpot_out[region * PEARL_JACKPOT_BUCKETS + b] = jp[i];
  }
}

// The fold is now a gather. Every product it needs is already in D, so a region
// costs 32 loads and a warp reduction per chunk instead of 32 dot products.
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *__restrict__ D,
    const uint32_t *__restrict__ rows_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t rows_valid, uint32_t chunks,
    uint64_t region_base, uint32_t *__restrict__ jackpot_out) {
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
  // rows_valid decomposes the region index; m stays the STRIDE of the partial
  // table, which is indexed by the actual row. Conflating the two is silent:
  // the fold reads the wrong partials and every hash differs.
  const uint32_t cg = (uint32_t)(slot / rows_valid);
  const uint32_t row_off =
      pearl_expand_offset((uint32_t)(slot % rows_valid), PEARL_ROWS_MASK);
  const uint32_t r = row_off | rows_pattern[ri];

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
                                               uint32_t *hit_index,
                                               int hash_big_endian) {
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
  if (!pearl_meets_target_mode(h, target_be, hash_big_endian)) return;
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
