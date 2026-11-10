// Volta's fold (pearl_tile_fold_hmma, see PEARL_HMMA_BODY): the index arithmetic, the
// int8-to-fp16 conversion and the readout, as functions both the kernel and a plain C++
// program can run. probes/hmma-check.cc runs them on the CPU against a plain fold.
//
// A V100's tensor cores multiply fp16 only (no int8 mma before Turing). The operands are
// int8, which fp16 holds exactly (any integer up to 2048 does), and each product is at
// most 128 * 128 = 2^14, exact in the fp32 accumulator. A sum of integer products stays
// exact in fp32 while every partial sum is an integer below 2^24: the tensor core aligns
// the addends to the largest one and drops bits below its last place, and below 2^24 the
// last place is at most 1. So the fold is exact while |accumulator| < 2^24, and the
// readout below reads it exactly inside a window of 2^23. How far that is from what the
// operands give: see PEARL_VOLTA_HMMA in pearl_config.h.
//
// The CTA tile is the dp4a fold's, 128 rows of A' by 128 columns of B' (8 row offsets by
// 8 column offsets, 64 regions), and the tile walk, the k-blocked operands (32-byte
// blocks), the region index and the hasher are the dp4a fold's too (pearl_fold_dp4a.h).
// What differs is inside the block: 4 warps in a 2x2 grid, each a 64x64 warp tile, on
// mma.sync m8n8k4 (Volta's HMMA.884). One such mma is four independent 8x8x4 products,
// one per quad pair: lanes 4q .. 4q + 3 and 4q + 16 .. 4q + 19 are quad pair q. Each lane
// brings one row of its pair's A (4 halfs of k) and one column of its B, row and column
// (lane & 3) + 4 * ((lane >> 4) & 1), and the pair's 8x8 result is spread over its 8
// lanes, 8 floats each. Quad pair q of a warp covers a 32x32 quarter of the warp tile:
// rows 32 (q & 1) .. + 31 and columns 32 (q >> 1) .. + 31, as 4x4 mma tiles of 8x8.
//
// The tile's rows and columns are PERMUTED as they are staged into shared, so that shared
// rows 16g .. 16g + 15 are row group g (one row offset, the dp4a fold's group g) and
// likewise for columns. A 16x16 block of the warp tile is then exactly one region: four
// 8x8 mma tiles of one quad pair. So a region's word is the XOR of the 32 values a lane
// holds of those four, over the pair's 8 lanes, whatever order the hardware puts the
// values in within a pair.
//
// In shared, a stage is 32 halfs of k per row (two 16-deep steps), rows
// PEARL_HMMA_LDM halfs apart: A' row-major (row = slot), B' column-major (column = slot),
// both with k contiguous. Thread t stages slot t of both: the 32 bytes of k of its row of
// A' and of its column of B', converted to 32 halfs.

#ifndef PEARL_FOLD_HMMA_H
#define PEARL_FOLD_HMMA_H

#include <stdint.h>

#include "pearl_config.h"
#include "pearl_fold_dp4a.h"

static_assert(PEARL_HMMA_THREADS == 128u, "4 warps of 64x64");
static_assert(PEARL_HMMA_STAGE_K == PEARL_DP4A_STAGE_K,
              "a stage is one 32-byte k-block of the dp4a fold's operand layout");
static_assert(PEARL_HMMA_LDM % 8u == 0u && PEARL_HMMA_LDM >= PEARL_HMMA_STAGE_K,
              "wmma wants a row stride of whole 16-byte units, and a row holds a stage");

// The tile row staged at shared row p (0..127): row group g = p >> 4 is span g >> 1, row
// offset 4 * (g & 1); element e = p & 15 is pattern row (e & 3) + 8 * (e >> 2).
PEARL_HD inline uint32_t pearl_hmma_row_of(uint32_t p) {
  const uint32_t g = p >> 4, e = p & 15u;
  return 32u * (g >> 1) + 4u * (g & 1u) + (e & 3u) + 8u * (e >> 2);
}
// The tile column staged at shared column p: column group g = p >> 4 is span g >> 2,
// column offset 2 * (g & 3); element e = p & 15 is pattern column (e & 1) + 8 * (e >> 1).
PEARL_HD inline uint32_t pearl_hmma_col_of(uint32_t p) {
  const uint32_t g = p >> 4, e = p & 15u;
  return 64u * (g >> 2) + 2u * (g & 3u) + (e & 1u) + 8u * (e >> 1);
}
// Byte offsets into the k-blocked A' and B' of thread tid's 32 bytes at stage 0 of tile
// (R, C); each stage is pearl_dp4a_stage_step(rows) further on.
PEARL_HD inline uint64_t pearl_hmma_a_src(uint32_t tid, uint32_t R) {
  return ((uint64_t)R * PEARL_DP4A_BM + pearl_hmma_row_of(tid)) * PEARL_HMMA_STAGE_K;
}
PEARL_HD inline uint64_t pearl_hmma_b_src(uint32_t tid, uint32_t C, uint32_t col_off) {
  const uint64_t col0 = 16ull * (col_off + C * PEARL_DP4A_COL_OFFSETS);
  return (col0 + pearl_hmma_col_of(tid)) * PEARL_HMMA_STAGE_K;
}

// Four int8 (one word, byte b = element b) as four fp16, two to a word, element 0 in the
// low half. Biased to unsigned (x + 128), the byte becomes the low mantissa bits of
// 0x6400 = 1024, so each half is 1152 + x exactly; one fp16 subtract of 1152 leaves x.
// On the device: one LOP3, two PRMT and two HSUB2. The CPU builds the same bits.
PEARL_HD inline uint32_t pearl_hmma_biased(uint32_t w, uint32_t hi) {
  const uint32_t u = w ^ 0x80808080u;
#if defined(__CUDA_ARCH__)
  return __byte_perm(u, 0x64646464u, hi ? 0x5342u : 0x5140u);
#else
  const uint32_t b0 = (u >> (16u * hi)) & 0xFFu, b1 = (u >> (16u * hi + 8u)) & 0xFFu;
  return b0 | 0x6400u | (b1 << 16) | 0x64000000u;
#endif
}
#define PEARL_HMMA_BIAS2 0x64806480u   // 1152 in both halves

// The fp16 bits of a small integer (|x| <= 2048), for the CPU side: what the subtract of
// 1152 gives.
inline uint16_t pearl_hmma_half_bits(int32_t x) {
  if (x == 0) return 0;
  uint32_t s = x < 0 ? 0x8000u : 0u, a = (uint32_t)(x < 0 ? -x : x), e = 0;
  while ((a >> e) > 1u) e++;
  return (uint16_t)(s | ((e + 15u) << 10) | ((a << (10u - e)) & 0x3FFu));
}

// The readout. The accumulators do not start at 0 but at a bias B (pearl_hmma_bias), so
// one that has summed the integer n holds B + n. While B + n is in [2^23, 2^24) an
// fp32's last place is 1, so it is exact and its bits are 0x4B000000 + (B + n - 2^23):
// n is the bits less U = 0x4B000000 + B - 2^23 (pearl_hmma_unbias). One integer subtract
// an accumulator, where F2I runs at a quarter of the rate. When B is 2^23, U is
// 0x4B000000, whose low 23 bits are zero: for 0 <= n < 2^23 the bits are n | 0x4B000000,
// and the XOR of an even number of them is the XOR of the n. A lane XORs 32 into each of
// its words, so then the readout is the XORs alone.
//
// The window [2^23 - B, 2^24 - B) must hold every accumulator at every readout. The mean
// product (`center`: the constant fill squared, 0 under the hashed fill) puts chunk c's
// near center * 128 * (c + 1). For center 2048 to 3072 (fills 46 to 55; 48 ships) B is
// 2^23 and the window [0, 2^23): at fill 48 chunk 0's 295,000 is 14 sd above 0 and chunk
// 15's 4.72 million 43 sd below 2^23. Otherwise B centres a window of 2^22 either side on
// the middle chunk's, center * 1088: under the hashed fill, 0 +- 2^22.
PEARL_HD inline uint32_t pearl_hmma_bias(int32_t center) {
  if (center >= 2048 && center <= 3072) return 8388608u;
  return 12582912u - (uint32_t)center * 1088u;
}
PEARL_HD inline uint32_t pearl_hmma_unbias(uint32_t bias) { return 0x4B000000u + bias - 8388608u; }

// The row (of A') and column (of B') within its quad pair's 8x8 mma tile that lane L
// brings to an m8n8k4 mma, and its quad pair.
PEARL_HD inline uint32_t pearl_hmma_r8(uint32_t lane) { return (lane & 3u) | (((lane >> 4) & 1u) << 2); }
PEARL_HD inline uint32_t pearl_hmma_qp(uint32_t lane) { return (lane >> 2) & 3u; }
// The shared row of A' (column of B') lane L of warp w reads for mma tile row i (column
// j), 0..3: warp quarter, quad pair quarter, then the 8x8 tile.
//
// Within an odd quad pair's A tile, rows 0-3 and 4-7 swap (PEARL_HMMA_ASWZ). A 16-byte
// shared load is served 8 lanes at a time: lanes 0-7 are rows 0-3 of pair 0 and of pair
// 1, 32 rows apart. Shared rows 80 bytes apart repeat their banks every 8 rows, so
// those two sets of 4 rows hit the same 4 bank groups, a 2-way conflict on every load of
// A'. Four rows further on they land on the other 4 groups. Rows within an 8x8 tile can
// go in any order: the readout XORs whole tiles. B' needs nothing: pairs 0 and 1 read
// the same columns, so their lanes share addresses.
PEARL_HD inline uint32_t pearl_hmma_a_row(uint32_t warp, uint32_t lane, uint32_t i) {
  const uint32_t odd = pearl_hmma_qp(lane) & 1u;
  return 64u * (warp >> 1) + 32u * odd + 8u * i
         + (pearl_hmma_r8(lane) ^ (PEARL_HMMA_ASWZ ? 4u * odd : 0u));
}
PEARL_HD inline uint32_t pearl_hmma_b_col(uint32_t warp, uint32_t lane, uint32_t j) {
  return 64u * (warp & 1u) + 32u * (pearl_hmma_qp(lane) >> 1) + 8u * j + pearl_hmma_r8(lane);
}
// The readout's reduction over a quad pair. Each lane starts with 4 words, region (a, b)
// of its pair's 2x2 at index 2a + b (a = i >> 1, b = j >> 1 of the mma tiles). The
// partner across lane bit 4 takes a, across bit 1 b, and bit 0 adds the last pair, so
// lane L ends with region (a, b) = (bit 4, bit 1) of its pair; the lane with bit 0 clear
// writes it. The region (row group * 8 + column group) of (a, b) of quad pair q of warp w:
PEARL_HD inline uint32_t pearl_hmma_region(uint32_t warp, uint32_t lane) {
  const uint32_t q = pearl_hmma_qp(lane), a = (lane >> 4) & 1u, b = (lane >> 1) & 1u;
  return (4u * (warp >> 1) + 2u * (q & 1u) + a) * 8u + 4u * (warp & 1u) + 2u * (q >> 1) + b;
}

#endif  // PEARL_FOLD_HMMA_H
