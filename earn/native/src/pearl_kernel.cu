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

// The heart of the PoW: accumulate C in `rank`-sized chunks and fold the
// mandated sub-tile of each chunk into the 16-lane jackpot transcript.
//
//   jackpot[tid] = rotl13(jackpot[tid]) ^ xor(tile),  tid = chunk % 16
//
// One block per candidate nonce; each thread owns a slice of the tile and the
// lanes are reduced in shared memory. This is the scalar/dp4a formulation —
// correct and auditable. The tensor-core specialisation replaces only the inner
// accumulation below (mma.sync m16n8k32 int8 with the fold applied in the
// mainloop epilogue), leaving the transcript semantics identical.
extern "C" __global__ void pearl_gemm_fold(
    const int8_t *__restrict__ A,     // [m, k] row-major, quantised
    const int8_t *__restrict__ B,     // [n, k] row-major (Bᵀ), quantised
    const int8_t *__restrict__ E_AL,  // [m, rank]
    const int8_t *__restrict__ E_AR,  // [rank, k]
    const int8_t *__restrict__ E_BL,  // [n, rank]
    const int8_t *__restrict__ E_BR,  // [rank, k]
    const uint32_t *__restrict__ rows_pattern,
    const uint32_t *__restrict__ cols_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    uint32_t chunks, uint64_t region, uint32_t *__restrict__ jackpot_out) {
  extern __shared__ uint32_t smem[];  // [blockDim.x] partial XORs

  const uint32_t tid = threadIdx.x;
  const uint32_t tile_cells = rows_count * cols_count;

  // WHERE THE SEARCH ACTUALLY HAPPENS. The tile pattern is fixed by the
  // profile, so what varies between attempts is WHERE in the output that tile
  // is read from. Without this offset every launch recomputes an identical
  // transcript and the miner is not searching at all — it grinds the GPU at
  // full utilisation producing one value over and over. That is exactly what
  // the first on-device run did: 65536 "attempts", one distinct result.
  const uint32_t row_off = (uint32_t)(region % m);
  const uint32_t col_off = (uint32_t)((region / m) % n);

  uint32_t jackpot[PEARL_JACKPOT_BUCKETS];
#pragma unroll
  for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) jackpot[i] = 0u;

  for (uint32_t chunk = 0; chunk < chunks; chunk++) {
    const uint32_t k0 = chunk * rank;
    uint32_t partial = 0u;

    // Each thread walks a strided share of the tile's cells.
    for (uint32_t cell = tid; cell < tile_cells; cell += blockDim.x) {
      const uint32_t r = (rows_pattern[cell / cols_count] + row_off) % m;
      const uint32_t c = (cols_pattern[cell % cols_count] + col_off) % n;

      int32_t acc = 0;
      for (uint32_t kk = k0; kk < k0 + rank && kk < k; kk++) {
        // Noised operands, reconstructed on the fly:
        //   A'[r,kk] = A[r,kk] + Σ_j E_AL[r,j]·E_AR[j,kk]
        //   B'[c,kk] = B[c,kk] + Σ_j E_BL[c,j]·E_BR[j,kk]
        int32_t a = A[(size_t)r * k + kk];
        int32_t b = B[(size_t)c * k + kk];
        for (uint32_t j = 0; j < rank; j++) {
          a += (int32_t)E_AL[(size_t)r * rank + j] * (int32_t)E_AR[(size_t)j * k + kk];
          b += (int32_t)E_BL[(size_t)c * rank + j] * (int32_t)E_BR[(size_t)j * k + kk];
        }
        acc += a * b;
      }
      partial ^= (uint32_t)acc;
    }

    // Reduce the tile XOR across the block.
    smem[tid] = partial;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) smem[tid] ^= smem[tid + s];
      __syncthreads();
    }

    if (tid == 0) {
      const uint32_t lane = chunk % PEARL_JACKPOT_BUCKETS;
      jackpot[lane] = pearl_rotl13(jackpot[lane]) ^ smem[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < PEARL_JACKPOT_BUCKETS; i++) jackpot_out[i] = jackpot[i];
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
