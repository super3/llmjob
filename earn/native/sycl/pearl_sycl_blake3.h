// BLAKE3 for the SYCL core, one copy for the host and the device.
//
// The CUDA core has two copies, one per side (pearl_kernel.cu and the hb3_* host
// functions in pearl_host.cu). Here the functions are plain C++ with no
// pointers to globals, so the same code compiles into SYCL kernels and into the
// host driver. It covers only what Pearl needs: full 1024-byte chunks (the
// operand commitments), parent nodes, and inputs of at most one chunk (job_key,
// the seed links, the noise RNG and the 64-byte transcript).
//
// Written from the BLAKE3 specification. Checked end to end by verify-hits.js,
// which recomputes every reported hit with the JS BLAKE3 in
// earn/src/shared/miner/blake3.js (itself checked against the official vectors).

#ifndef PEARL_SYCL_BLAKE3_H
#define PEARL_SYCL_BLAKE3_H

#include <stdint.h>

namespace pearl_b3 {

constexpr uint32_t kChunkStart = 1u, kChunkEnd = 2u, kParent = 4u, kRoot = 8u, kKeyed = 16u;

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void g(uint32_t *s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x;
  s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y;
  s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 7);
}

// The BLAKE3 IV, as literals: device code cannot read a host global.
inline uint32_t iv(int i) {
  switch (i) {
    case 0: return 0x6A09E667u;
    case 1: return 0xBB67AE85u;
    case 2: return 0x3C6EF372u;
    case 3: return 0xA54FF53Au;
    case 4: return 0x510E527Fu;
    case 5: return 0x9B05688Cu;
    case 6: return 0x1F83D9ABu;
    default: return 0x5BE0CD19u;
  }
}

// One compression. out[0..7] is the next chaining value (and, under ROOT, the
// first 32 bytes of output); out[8..15] the rest of the output block.
inline void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter,
                     uint32_t len, uint32_t flags, uint32_t out[16]) {
  uint32_t s[16], m[16];
#pragma unroll
  for (int i = 0; i < 8; i++) s[i] = cv[i];
  s[8] = iv(0); s[9] = iv(1); s[10] = iv(2); s[11] = iv(3);
  s[12] = (uint32_t)counter;
  s[13] = (uint32_t)(counter >> 32);
  s[14] = len;
  s[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = block[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    g(s, 0, 4, 8, 12, m[0], m[1]);
    g(s, 1, 5, 9, 13, m[2], m[3]);
    g(s, 2, 6, 10, 14, m[4], m[5]);
    g(s, 3, 7, 11, 15, m[6], m[7]);
    g(s, 0, 5, 10, 15, m[8], m[9]);
    g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7, 8, 13, m[12], m[13]);
    g(s, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      // The message permutation {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8}, with
      // literal indices so the block stays in registers.
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

inline uint32_t load_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline void store_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// The keyed chaining value of one whole 1024-byte chunk at index `counter`: a
// leaf of an operand commitment. `chunk` must be 4-byte aligned.
inline void chunk_cv(const uint32_t key[8], const uint8_t *chunk, uint64_t counter,
                     uint32_t out[8]) {
  uint32_t cv[8], block[16], o[16];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  const uint32_t *w = reinterpret_cast<const uint32_t *>(chunk);
  for (int b = 0; b < 16; b++) {
#pragma unroll
    for (int i = 0; i < 16; i++) block[i] = w[b * 16 + i];
    const uint32_t flags =
        kKeyed | (b == 0 ? kChunkStart : 0u) | (b == 15 ? kChunkEnd : 0u);
    compress(cv, block, counter, 64, flags, o);
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = o[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// A parent node: the two children's chaining values as one 64-byte block.
inline void parent_cv(const uint32_t key[8], const uint32_t left[8], const uint32_t right[8],
                      bool root, uint32_t out[8]) {
  uint32_t block[16], o[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { block[i] = left[i]; block[8 + i] = right[i]; }
  compress(key, block, 0, 64, kParent | kKeyed | (root ? kRoot : 0u), o);
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = o[i];
}

// BLAKE3 of exactly 64 bytes given as 16 little-endian words: one compression,
// the root of a one-block chunk. Keyed when `keyed`, else unkeyed (the IV as the
// key). This is the noise RNG, the transcript hash and both seed links.
inline void hash64(const uint32_t key[8], bool keyed, const uint32_t msg[16], uint32_t out[8]) {
  uint32_t cv[8], o[16];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = keyed ? key[i] : iv(i);
  compress(cv, msg, 0, 64, kChunkStart | kChunkEnd | kRoot | (keyed ? kKeyed : 0u), o);
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = o[i];
}

// BLAKE3 of at most one chunk (len <= 1024), keyed or not. Host side: job_key
// hashes the 128-byte header76 || config52.
inline void hash_small(const uint32_t *key, const uint8_t *in, uint32_t len, uint8_t out[32]) {
  uint32_t cv[8], block[16], o[16];
  for (int i = 0; i < 8; i++) cv[i] = key ? key[i] : iv(i);
  const uint32_t base = key ? kKeyed : 0u;
  uint32_t off = 0, start = kChunkStart;
  while (off + 64 < len) {
    for (int i = 0; i < 16; i++) block[i] = load_le32(in + off + 4 * i);
    compress(cv, block, 0, 64, base | start, o);
    for (int i = 0; i < 8; i++) cv[i] = o[i];
    start = 0;
    off += 64;
  }
  uint8_t tail[64];
  const uint32_t rem = len - off;
  for (uint32_t i = 0; i < 64; i++) tail[i] = i < rem ? in[off + i] : 0;
  for (int i = 0; i < 16; i++) block[i] = load_le32(tail + 4 * i);
  compress(cv, block, 0, rem, base | start | kChunkEnd | kRoot, o);
  for (int i = 0; i < 8; i++) store_le32(out + 4 * i, o[i]);
}

// Does the jackpot hash meet the bound? The hash is read little-endian (as the
// reference does) unless hash_big_endian, and target_w holds the target as
// big-endian words, most significant first. Equal counts as a share. The same
// rule as pearl_hash_meets_words in pearl_kernel.cu.
inline bool meets(const uint32_t h[8], const uint32_t target_w[8], int hash_big_endian) {
  for (int i = 0; i < 8; i++) {
    uint32_t hw;
    if (hash_big_endian) {
      const uint32_t x = h[i];
      hw = (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
    } else {
      hw = h[7 - i];
    }
    if (hw != target_w[i]) return hw < target_w[i];
  }
  return true;
}

}  // namespace pearl_b3

#endif  // PEARL_SYCL_BLAKE3_H
