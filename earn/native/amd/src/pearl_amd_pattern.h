// The hash-region patterns of the AMD folds. A region is h rows by w columns of C: a row list
// plus a valid row offset, by a column list plus a valid column offset (pearl_config.h, "A tile
// offset is only VALID ..."). A pool rebuilds the pattern from the row indices a share carries,
// and config52 bytes 8-19 carry its encoding, so a fold may use any legal pattern, as long as
// the host and the JS describe its hits with the same one.
//
//   16x16  rows {0-3, 8-11, 16-19, 24-27}, columns {0, 1, 8, 9, ..., 56, 57}: pearl_config.h's,
//          which every NVIDIA fold and every AMD fold uses by default.
//   32x8   rows {0-7, 16-23, 32-39, 48-55}, columns {0, 1, 16, 17, 32, 33, 48, 49}: the RDNA 4
//          fold's two-lane regions (PEARL_AMD_TWOLANE below). In gfx12's WMMA layout lane l of a
//          wave holds rows 16 * mt + 8 * (l / 16) + g and columns 16 * nt + l % 16, so lanes l and
//          l ^ 1 hold all 256 cells of one 32x8 region between them. A chunk's word then needs one
//          lane move and no selects, where a 16x16 region needs three lane moves and two selects.
//
// Both are 256 cells, so a region is one hash for 256 * k MACs either way, and a share's credit
// (the penalized factor, 524,288 at k 2048) is the same.
#pragma once

#include <stdint.h>

#include "pearl_config.h"

// PEARL_AMD_TWOLANE=1: the RDNA 4 fold (wmma12) reads out 32x8 regions; RDNA 3, RDNA 2 and CDNA
// keep 16x16. Round 4 item 2. A timing-only probe of the readout it removes (PEARL_AMD_READOUT_DROP)
// beat the default core by +1.48 to +1.62% on an RX 9070 XT and +1.61 to +1.64% on an RX 9060 XT
// (benchmark.md, 1 November), so it is being built in two stages:
//   Stage A (this build): the fold and its fold test only. The host still describes every hit as
//     16x16, so it refuses the wmma12 fold when this is on ("two-lane folds are fold-test only in
//     this build"), and only pearl_amd_foldtest runs it. test/sweep.sh races its fold alone.
//   Stage B, only if Stage A's race passes: the host and the JS learn the pattern per core.
// Off by default: with it off, every fold compiles to the code it did before it existed.
#ifndef PEARL_AMD_TWOLANE
#define PEARL_AMD_TWOLANE 0
#endif
// PEARL_AMD_TWOLANE_STORE=1, with PEARL_AMD_TWOLANE=1: the second arm of Stage A's race. The
// even lane of each pair stores word c to the transcript at the end of chunk c, instead of the
// pair keeping its 16 words in registers and storing them at the end of the tile. It writes 16
// lines a wave every chunk where the register form writes them once a tile, so it is raced, not
// assumed: the register form stays unless this one is at least 0.3% faster on both cards.
#ifndef PEARL_AMD_TWOLANE_STORE
#define PEARL_AMD_TWOLANE_STORE 0
#endif
#if PEARL_AMD_TWOLANE != 0 && PEARL_AMD_TWOLANE != 1
#error "PEARL_AMD_TWOLANE is 0 (off) or 1 (two-lane 32x8 regions in the RDNA 4 fold)"
#endif
#if PEARL_AMD_TWOLANE_STORE && !PEARL_AMD_TWOLANE
#error "PEARL_AMD_TWOLANE_STORE is a form of PEARL_AMD_TWOLANE: set -DPEARL_AMD_TWOLANE=1 as well"
#endif

// The 32x8 pattern. Its six-byte encodings, (factor - 1, length - 1) per dimension as in
// pearl_config.h:
//   rows: (stride 1, length 8) then (stride 16, length 4): factors 1 and 16/8 = 2
//   cols: (stride 1, length 2) then (stride 16, length 4): factors 1 and 16/2 = 8
// The static_asserts below decode them back into the lists, and test/amdTwoLane.test.js checks
// them against the JS derivation (pearlhash.js's patternFromList and patternToBytes).
#define PEARL_AMD_32X8_ROWS_COUNT 32
#define PEARL_AMD_32X8_COLS_COUNT 8
static constexpr uint32_t PEARL_AMD_32X8_ROWS[PEARL_AMD_32X8_ROWS_COUNT] = {
    0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
    32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55};
static constexpr uint32_t PEARL_AMD_32X8_COLS[PEARL_AMD_32X8_COLS_COUNT] = {0, 1, 16, 17, 32, 33, 48, 49};
static constexpr uint8_t PEARL_AMD_32X8_ROWS_BYTES[6] = {0, 7, 1, 3, 0, 0};
static constexpr uint8_t PEARL_AMD_32X8_COLS_BYTES[6] = {0, 1, 7, 3, 0, 0};

// Element e of the list a six-byte encoding describes, as PeriodicPattern::to_list builds it:
// the first dimension varies fastest.
constexpr uint32_t pearl_amd_pattern_elem(const uint8_t *bytes, uint32_t e) {
  uint32_t out = 0u, minStride = 1u;
  for (int d = 0; d < 3; d++) {
    const uint32_t stride = (bytes[2 * d] + 1u) * minStride, length = bytes[2 * d + 1] + 1u;
    out += (e % length) * stride;
    e /= length;
    minStride = stride * length;
  }
  return out;
}
constexpr uint32_t pearl_amd_pattern_size(const uint8_t *bytes) {
  return (bytes[1] + 1u) * (bytes[3] + 1u) * (bytes[5] + 1u);
}
// A list a pattern can be: strictly increasing, starting at 0, and every subset of its own mask
// bits (so valid offsets partition the grid), and exactly what its bytes decode to.
constexpr bool pearl_amd_pattern_ok(const uint32_t *list, uint32_t n, const uint8_t *bytes) {
  if (n == 0u || list[0] != 0u || pearl_amd_pattern_size(bytes) != n) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (i > 0u && list[i] <= list[i - 1]) return false;
    if (pearl_amd_pattern_elem(bytes, i) != list[i]) return false;
  }
  return n == (1u << pearl_popcount_ce(pearl_pattern_mask(list, n)));
}

static_assert(pearl_amd_pattern_ok(PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_ROWS_COUNT, PEARL_AMD_32X8_ROWS_BYTES),
              "32x8 rows: every subset of the mask, in order, and what the bytes decode to");
static_assert(pearl_amd_pattern_ok(PEARL_AMD_32X8_COLS, PEARL_AMD_32X8_COLS_COUNT, PEARL_AMD_32X8_COLS_BYTES),
              "32x8 columns: every subset of the mask, in order, and what the bytes decode to");
static_assert(pearl_pattern_mask(PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_ROWS_COUNT) == 0x37u &&
                  pearl_pattern_mask(PEARL_AMD_32X8_COLS, PEARL_AMD_32X8_COLS_COUNT) == 0x31u,
              "32x8 masks: rows 0x37, columns 0x31");
static_assert(PEARL_AMD_32X8_ROWS_COUNT % 2 == 0 && PEARL_AMD_32X8_COLS_COUNT % 2 == 0,
              "the sanity checks want h and w even");
static_assert(PEARL_AMD_32X8_ROWS_COUNT * PEARL_AMD_32X8_COLS_COUNT == 256 &&
                  PEARL_AMD_32X8_ROWS_COUNT * PEARL_AMD_32X8_COLS_COUNT == PEARL_ROWS_COUNT * PEARL_COLS_COUNT,
              "256 cells, as 16x16: the same hashes per MAC and the same credit");
static_assert(pearl_pattern_span(pearl_pattern_mask(PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_ROWS_COUNT)) == 64u &&
                  pearl_pattern_span(pearl_pattern_mask(PEARL_AMD_32X8_COLS, PEARL_AMD_32X8_COLS_COUNT)) == 64u,
              "32x8 spans 64 rows and 64 columns: a wave's 64x64 tile holds 16 whole regions");
// pearl_config.h's 16x16 bytes are a plain const array, which a static_assert can't read; its
// lists and bytes are checked against the JS in test/nativeConfig.test.js.

// A pattern as one value, for the fold test now and the host in Stage B. Host side only, as
// pearl_config.h's byte arrays are.
struct AmdPattern {
  const char *name;
  uint32_t h, w;                         // rows and columns of a region
  const uint32_t *rows, *cols;           // the lists
  uint32_t rowsMask, colsMask;           // the OR of each list: a valid offset has these bits clear
  uint32_t rowsSpan, colsSpan;           // m and a batch's columns are whole numbers of these
  const uint8_t *rowsBytes, *colsBytes;  // config52 bytes 8-13 and 14-19
};
static const AmdPattern kAmdPattern16x16 = {
    "16x16", PEARL_ROWS_COUNT, PEARL_COLS_COUNT, PEARL_ROWS_PATTERN, PEARL_COLS_PATTERN,
    PEARL_ROWS_MASK, PEARL_COLS_MASK, PEARL_ROWS_SPAN, PEARL_COLS_SPAN,
    PEARL_ROWS_PATTERN_BYTES, PEARL_COLS_PATTERN_BYTES};
static const AmdPattern kAmdPattern32x8 = {
    "32x8", PEARL_AMD_32X8_ROWS_COUNT, PEARL_AMD_32X8_COLS_COUNT, PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_COLS,
    pearl_pattern_mask(PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_ROWS_COUNT),
    pearl_pattern_mask(PEARL_AMD_32X8_COLS, PEARL_AMD_32X8_COLS_COUNT),
    pearl_pattern_span(pearl_pattern_mask(PEARL_AMD_32X8_ROWS, PEARL_AMD_32X8_ROWS_COUNT)),
    pearl_pattern_span(pearl_pattern_mask(PEARL_AMD_32X8_COLS, PEARL_AMD_32X8_COLS_COUNT)),
    PEARL_AMD_32X8_ROWS_BYTES, PEARL_AMD_32X8_COLS_BYTES};
