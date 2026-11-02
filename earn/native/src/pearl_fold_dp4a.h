// Pascal's fold (pearl_tile_fold_dp4a, see PEARL_DP4A_ARCH): the index arithmetic and
// the int8 inner loop, as functions both the kernel and a plain C++ program can run.
//
// The kernel is in pearl_kernel.cu. probes/dp4a-check.cc runs these same functions on
// the CPU, thread by thread and phase by phase, and probes/dp4a-check.js compares what
// comes out against the JS reference (src/shared/miner/reference.js). No GPU is needed
// for that check, so a mistake in the layout below shows up before a card runs it.
//
// The CTA tile is 128 rows of A' by 128 columns of B': 8 valid row offsets by 8 valid
// column offsets, so 64 regions. 256 threads in a 16x16 grid each hold an 8x8 block of
// int32 accumulators. The tile's rows and columns are PERMUTED as they are staged into
// shared memory, so that a thread's 8 rows and 8 columns all belong to one region:
//
//   - Rows. A 32-row span holds two row offsets (0 and 4: rows {0,1,2,3} + 8j and
//     {4,5,6,7} + 8j). So the tile has 8 row GROUPS of 16 rows, one a row offset:
//     group g = 2 * span + class. Thread row ty owns group ty >> 1, elements
//     8 * (ty & 1) .. + 7 of its 16.
//   - Columns. A 64-column span holds four column offsets (0, 2, 4, 6: columns
//     {0,1} + 8i, ...). So 8 column groups of 16, group g = 4 * span + class, and
//     thread column tx owns group tx >> 1, elements 8 * (tx & 1) .. + 7.
//
// So the four threads (ty, tx), (ty ^ 1, tx), (ty, tx ^ 1), (ty ^ 1, tx ^ 1) hold one
// region's 256 accumulators between them, 64 each. A thread folds its own 64, and two
// shuffles combine the four.
//
// In shared, a stage is 8 k-quads (32 bytes of k), each a row of PEARL_DP4A_SSTRIDE
// words: word p of k-quad q is 4 bytes of k of the row (column) staged at slot p.
// Thread ty reads its rows as two 16-byte loads, slots 4ty .. 4ty+3 and 64+4ty ..
// 64+4ty+3, and tx its columns the same way. The 8 tx of a warp then read 8 different
// 16-byte units covering all 32 banks, and the 4 ty of a warp 4 units, each one read by
// 8 lanes at once. So slot p belongs to thread (p & 63) >> 2, as its element
// (p & 3) + 4 * (p >> 6) of eight.
//
// In global memory A' and B' are k-BLOCKED for this fold, in 32-byte blocks of k:
// element (row, kk) sits at ((kk / 32) * rows + row) * 32 + kk % 32, which is what
// pearl_materialize16_kblocked writes with kb_log2 = 5. One stage of a tile, 32 bytes of
// k from 128 consecutive rows, is then one contiguous 4 KB run of whole 128-byte lines.
// Row-major, the same stage was one 32-byte sector out of each of 128 lines 2 KB apart,
// and the other three sectors of every line were asked for in the next three stages,
// thousands of cycles later. The raw A and B (commitments, proofs) stay row-major.

#ifndef PEARL_FOLD_DP4A_H
#define PEARL_FOLD_DP4A_H

#include <stdint.h>

#include "pearl_config.h"

// The readout folds the regions of this tile pattern; another pattern needs another
// permutation below.
static_assert(PEARL_ROWS_COUNT == 16 && PEARL_COLS_COUNT == 16, "a 16x16 tile pattern");
static_assert(PEARL_ROWS_MASK == 0x1Bu && PEARL_COLS_MASK == 0x39u,
              "the dp4a fold's groups are rows {0,1,2,3}+8j by cols {0,1}+8i");
static_assert(PEARL_DP4A_THREADS == 256u && PEARL_DP4A_BM == 128u && PEARL_DP4A_BN == 128u,
              "16x16 threads of 8x8 accumulators");
static_assert(PEARL_DP4A_STAGE_K == 32u, "a stage is 8 k-quads: two 16-byte copies a row");
static_assert(PEARL_DP4A_SSTRIDE % 4u == 0u && PEARL_DP4A_SSTRIDE >= PEARL_DP4A_BM,
              "a k-quad row keeps 16-byte loads aligned and holds every slot");

#define PEARL_DP4A_KQ (PEARL_DP4A_STAGE_K / 4u)                     // 8 k-quads a stage
#define PEARL_DP4A_STAGE_WORDS (PEARL_DP4A_KQ * PEARL_DP4A_SSTRIDE)  // one operand
#define PEARL_DP4A_REGIONS 64u                                       // regions a tile
// A region's 16 transcript words in shared, padded to 17 so the 8 regions a warp writes
// at once, and the 32 a warp of hashers reads, land on different banks.
#define PEARL_DP4A_TR_STRIDE 17u

// Thread (tx, ty) of the 16x16 grid. A warp is 8 tx by 4 ty: lane bits 0-2 are tx & 7
// and bits 3-4 ty & 3, so a region's four threads are lanes L, L ^ 1, L ^ 8 and L ^ 9.
PEARL_HD inline uint32_t pearl_dp4a_tx(uint32_t tid) {
  return (tid & 7u) | (((tid >> 5) & 1u) << 3);
}
PEARL_HD inline uint32_t pearl_dp4a_ty(uint32_t tid) {
  return ((tid >> 3) & 3u) | ((tid >> 6) << 2);
}
// The region (0..63) a thread's accumulators belong to: row group * 8 + column group.
PEARL_HD inline uint32_t pearl_dp4a_region(uint32_t tid) {
  return (pearl_dp4a_ty(tid) >> 1) * 8u + (pearl_dp4a_tx(tid) >> 1);
}
// Whether this thread writes its region's word: one of the four, lane bits 0 and 3 clear.
PEARL_HD inline bool pearl_dp4a_writer(uint32_t tid) { return (tid & 9u) == 0u; }

// Slot p of a stage (0..127): the group it is staged for, and its element in the group.
PEARL_HD inline uint32_t pearl_dp4a_slot_elem(uint32_t p, uint32_t *group) {
  const uint32_t t = (p & 63u) >> 2;   // the thread row (or column) that reads slot p
  *group = t >> 1;
  return ((t & 1u) << 3) | (p & 3u) | ((p >> 6) << 2);
}
// The tile row staged at slot p. Group g is span g >> 1, row offset 4 * (g & 1);
// element e is pattern row (e & 3) + 8 * (e >> 2) (PEARL_ROWS_PATTERN).
PEARL_HD inline uint32_t pearl_dp4a_row_of(uint32_t p) {
  uint32_t g;
  const uint32_t e = pearl_dp4a_slot_elem(p, &g);
  return 32u * (g >> 1) + 4u * (g & 1u) + (e & 3u) + 8u * (e >> 2);
}
// The tile column staged at slot p. Group g is span g >> 2, column offset 2 * (g & 3);
// element e is pattern column (e & 1) + 8 * (e >> 1) (PEARL_COLS_PATTERN).
PEARL_HD inline uint32_t pearl_dp4a_col_of(uint32_t p) {
  uint32_t g;
  const uint32_t e = pearl_dp4a_slot_elem(p, &g);
  return 64u * (g >> 2) + 2u * (g & 3u) + (e & 1u) + 8u * (e >> 1);
}

// Which tile a block folds: bands of `band` row groups, walked down first, so the
// blocks resident together share a few row groups of A' and column groups of B'. The
// last band may be short. A bijection on 0 .. rowGroups * colGroups - 1.
PEARL_HD inline void pearl_dp4a_tile(uint32_t v, uint32_t rowGroups, uint32_t colGroups,
                                     uint32_t band, uint32_t *R, uint32_t *C) {
  const uint32_t bandTiles = band * colGroups;
  const uint32_t b = v / bandTiles, in = v % bandTiles;
  const uint32_t first = b * band;
  const uint32_t rows = rowGroups - first < band ? rowGroups - first : band;
  *R = first + in % rows;
  *C = in / rows;
}

// The batch-local region index of region `reg` of tile (R, C), as every fold reports
// it: column offset index (from col_off) times rows_valid, plus row offset index.
// Row group g of tile R is row offset index 8R + g; column group g of tile C is
// column offset index col_off + 8C + g.
PEARL_HD inline uint32_t pearl_dp4a_region_index(uint32_t R, uint32_t C, uint32_t reg,
                                                 uint32_t rows_valid) {
  return (C * PEARL_DP4A_COL_OFFSETS + (reg & 7u)) * rows_valid
         + R * PEARL_DP4A_ROW_OFFSETS + (reg >> 3);
}

// Where a staging thread's 16 bytes come from and go: slot tid >> 1, k bytes
// 16 * (tid & 1) .. + 15 of the stage. Byte offsets into the k-blocked A' and B' (see
// the top) at stage 0 of tile (R, C); stage s is pearl_dp4a_stage_step(rows) * s bytes
// further on. The tile's first row is 128 R, its first column the expansion of column
// offset index col_off + 8C, a multiple of four: 16 * (col_off + 8C).
static_assert((PEARL_DP4A_STAGE_K & (PEARL_DP4A_STAGE_K - 1u)) == 0u,
              "a stage is one power-of-two k-block of the operands");
PEARL_HD inline uint64_t pearl_dp4a_a_src(uint32_t tid, uint32_t R) {
  return ((uint64_t)R * PEARL_DP4A_BM + pearl_dp4a_row_of(tid >> 1)) * PEARL_DP4A_STAGE_K
         + 16u * (tid & 1u);
}
PEARL_HD inline uint64_t pearl_dp4a_b_src(uint32_t tid, uint32_t C, uint32_t col_off) {
  const uint64_t col0 = 16ull * (col_off + C * PEARL_DP4A_COL_OFFSETS);
  return (col0 + pearl_dp4a_col_of(tid >> 1)) * PEARL_DP4A_STAGE_K + 16u * (tid & 1u);
}
// From one stage of an operand to the next: one k-block of all `rows` rows.
PEARL_HD inline uint64_t pearl_dp4a_stage_step(uint32_t rows) {
  return (uint64_t)rows * PEARL_DP4A_STAGE_K;
}
// Where element (row, kk) of an operand of `rows` rows sits in that layout. The CPU
// check lays its operands out with this; the device draw is pearl_materialize16_kblocked.
PEARL_HD inline uint64_t pearl_dp4a_kblocked(uint32_t row, uint32_t kk, uint32_t rows) {
  return ((uint64_t)(kk / PEARL_DP4A_STAGE_K) * rows + row) * PEARL_DP4A_STAGE_K
         + kk % PEARL_DP4A_STAGE_K;
}
// Word i (0..3) of that copy: k-quad 4 * (tid & 1) + i, slot tid >> 1.
PEARL_HD inline uint32_t pearl_dp4a_dst(uint32_t tid, uint32_t i) {
  return (4u * (tid & 1u) + i) * PEARL_DP4A_SSTRIDE + (tid >> 1);
}

// Four signed int8 products summed into c, as __dp4a does. The device uses the
// instruction; the CPU check uses this, byte b of each word against byte b.
PEARL_HD inline int32_t pearl_dp4a_dot(uint32_t a, uint32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && PEARL_DP4A_BODY
  return __dp4a((int)a, (int)b, c);
#else
  int32_t s = c;
  for (int i = 0; i < 4; i++)
    s += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
  return s;
#endif
}

// Four consecutive words of a k-quad row, as one 16-byte load on the device.
struct PearlDp4aQuad {
  uint32_t x, y, z, w;
};
PEARL_HD inline PearlDp4aQuad pearl_dp4a_ld4(const uint32_t *p) {
#if defined(__CUDA_ARCH__)
  const uint4 v = *reinterpret_cast<const uint4 *>(p);
  return {v.x, v.y, v.z, v.w};
#else
  return {p[0], p[1], p[2], p[3]};
#endif
}

// One stage of the GEMM for thread (tx, ty): for each of the 8 k-quads, its 8 rows of A'
// and 8 columns of B' (two 16-byte loads each), and 64 dp4a. Accumulator (i, j) is the
// row at slot 4ty + i (i < 4) or 64 + 4ty + i - 4, by the column at the same slot of tx.
PEARL_HD inline void pearl_dp4a_mac_stage(const uint32_t *sA, const uint32_t *sB, uint32_t tx,
                                          uint32_t ty, int32_t acc[8][8]) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (uint32_t q = 0; q < PEARL_DP4A_KQ; q++) {
    const uint32_t *ra = sA + q * PEARL_DP4A_SSTRIDE + 4u * ty;
    const uint32_t *rb = sB + q * PEARL_DP4A_SSTRIDE + 4u * tx;
    const PearlDp4aQuad a0 = pearl_dp4a_ld4(ra), a1 = pearl_dp4a_ld4(ra + 64);
    const PearlDp4aQuad b0 = pearl_dp4a_ld4(rb), b1 = pearl_dp4a_ld4(rb + 64);
    const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (uint32_t i = 0; i < 8; i++) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
      for (uint32_t j = 0; j < 8; j++) acc[i][j] = pearl_dp4a_dot(a[i], b[j], acc[i][j]);
    }
  }
}

// The XOR of a thread's 64 accumulators: its quarter of its region's tile.
PEARL_HD inline uint32_t pearl_dp4a_xor64(const int32_t acc[8][8]) {
  uint32_t x = 0u;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (uint32_t i = 0; i < 8; i++) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (uint32_t j = 0; j < 8; j++) x ^= (uint32_t)acc[i][j];
  }
  return x;
}

#endif  // PEARL_FOLD_DP4A_H
