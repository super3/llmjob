// Volta's fp16 fold, run on the CPU. No GPU and no CUDA toolkit needed.
//
//   c++ -O2 -std=c++17 -I../src hmma-check.cc -o hmma-check && ./hmma-check
//
// It does what launches of pearl_tile_fold_hmma do, block by block, warp by warp and
// lane by lane, with the kernel's own functions (src/pearl_fold_hmma.h): which tile a
// block takes, where each thread's 32 bytes come from and which shared row they land in,
// the int8-to-fp16 bits, which rows and columns each lane brings to the mma, the
// readout's biased fp32 bits, the shuffles over a quad pair, which lane writes which
// region's word, and the region index a hasher reports. A quad pair's 8x8 result is
// handed to its 8 lanes in one fixed order: the readout XORs a region's whole four
// tiles over the pair, so the hardware's own order cannot change the word, and the check
// holds for any. (What it cannot check is the hardware's operand layout itself, which
// rows and columns a lane's registers stand for: verify-hits.js on a V100 does.)
//
// It also checks what the kernel cannot: that every shared row a stage reads was
// written for that stage, that every partial sum stays below 2^24 (where fp32 is exact),
// that every readout lands inside the bias's window, that every transcript word is
// written exactly once, and that the batch's region indices come out once each.
//
// Then it folds every region the plain way, from the region index alone, as the dp4a
// check does, and both must agree. Four runs: the constant fill twice (48 plus noise, as the
// miner draws), the hashed fill (int7 plus noise) and every int8 value (uniform bytes,
// center 0). Prints a line a run and exits 1 on any mismatch or failed check.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "pearl_fold_hmma.h"

static void die(const char *msg) {
  fprintf(stderr, "hmma-check: %s\n", msg);
  exit(1);
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next32() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}
// Uniform in [lo, hi).
static int32_t uni(int32_t lo, int32_t hi) { return lo + (int32_t)(next32() % (uint32_t)(hi - lo)); }
static int8_t sat8(int32_t v) { return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); }

// One operand of `rows` rows, k wide: base plus noise, the noise a difference of two
// entries of a row of 128 uniform [-32, 32) values, picked per k (noise.js's structure).
// fill: 0..63 the constant fill, -1 hashed int7, -2 uniform bytes and no noise.
static std::vector<int8_t> draw(uint32_t rows, uint32_t k, int fill) {
  std::vector<int8_t> v((size_t)rows * k);
  std::vector<uint32_t> p0(k), p1(k);
  for (uint32_t kk = 0; kk < k; kk++) {
    p0[kk] = next32() % PEARL_FOLD_RANK;
    p1[kk] = next32() % PEARL_FOLD_RANK;
  }
  std::vector<int32_t> dense(PEARL_FOLD_RANK);
  for (uint32_t r = 0; r < rows; r++) {
    for (auto &d : dense) d = uni(-32, 32);
    for (uint32_t kk = 0; kk < k; kk++) {
      int32_t base = fill >= 0 ? fill : uni(-64, 64);
      if (fill == -2) {
        v[(size_t)r * k + kk] = (int8_t)uni(-128, 128);
        continue;
      }
      v[(size_t)r * k + kk] = sat8(base + dense[p0[kk]] - dense[p1[kk]]);
    }
  }
  return v;
}

struct Run {
  int mismatches = 0;
  uint32_t regions = 0;
  int64_t maxAbs = 0;          // the largest |bias + partial sum| seen
  int64_t margin = 1 << 24;    // the least distance of bias + n from the window's ends
};

static Run check(uint32_t m, uint32_t n, uint32_t col_off, uint32_t col_groups, int fill,
                 int32_t center) {
  const uint32_t bias = pearl_hmma_bias(center), U = pearl_hmma_unbias(bias);
  const uint32_t k = PEARL_FOLD_K, rank = PEARL_FOLD_RANK, LDM = PEARL_HMMA_LDM;
  const uint32_t rows_valid = m / PEARL_ROWS_COUNT, cols_valid = n / PEARL_COLS_COUNT;
  if (rows_valid % PEARL_DP4A_ROW_OFFSETS || col_groups % PEARL_DP4A_COL_OFFSETS
      || col_off % PEARL_DP4A_COL_OFFSETS || col_off + col_groups > cols_valid)
    die("geometry is not whole 128x128 tiles inside the operands");
  const std::vector<int8_t> A = draw(m, k, fill), B = draw(n, k, fill);
  std::vector<int8_t> Ak((size_t)m * k), Bk((size_t)n * k);
  for (uint32_t r = 0; r < m; r++)
    for (uint32_t kk = 0; kk < k; kk++) Ak[pearl_dp4a_kblocked(r, kk, m)] = A[(size_t)r * k + kk];
  for (uint32_t c = 0; c < n; c++)
    for (uint32_t kk = 0; kk < k; kk++) Bk[pearl_dp4a_kblocked(c, kk, n)] = B[(size_t)c * k + kk];

  const uint32_t rowGroups = rows_valid / PEARL_DP4A_ROW_OFFSETS;
  const uint32_t colGroups = col_groups / PEARL_DP4A_COL_OFFSETS;
  const uint32_t tiles = rowGroups * colGroups;
  const uint32_t stages = k / PEARL_HMMA_STAGE_K, per_chunk = rank / PEARL_HMMA_STAGE_K;
  const uint32_t T = PEARL_HMMA_THREADS, batch = col_groups * rows_valid;

  Run run;
  std::vector<std::vector<uint32_t>> out(batch);
  std::vector<int> seen(batch, 0), tileSeen(tiles, 0);
  // Shared: the staged values (as integers; the fp16 bits are checked on the way in)
  // and which stage wrote each row.
  std::vector<int32_t> sA(2 * 128 * LDM), sB(2 * 128 * LDM);
  std::vector<int> wA(2 * 128), wB(2 * 128);
  std::vector<uint32_t> sT(PEARL_DP4A_REGIONS * PEARL_DP4A_TR_STRIDE);
  std::vector<int> wT(PEARL_DP4A_REGIONS * PEARL_DP4A_TR_STRIDE);
  // acc[warp][quad pair][i][j][cell of 8x8], exact integers, and the copies held in
  // registers.
  std::vector<int64_t> acc(4 * 4 * 16 * 64);
  std::vector<uint8_t> ra((size_t)T * 32), rb((size_t)T * 32);

  for (uint32_t v = 0; v < tiles; v++) {
    uint32_t R, C;
    pearl_dp4a_tile(v, rowGroups, colGroups, PEARL_DP4A_BAND, &R, &C);
    if (R >= rowGroups || C >= colGroups || tileSeen[(size_t)R * colGroups + C]++)
      die("the tile walk is not a bijection");
    for (auto &w : wA) w = -1;
    for (auto &w : wB) w = -1;
    for (auto &w : wT) w = -1;
    for (auto &a : acc) a = 0;

    auto copy = [&](uint32_t tid, uint32_t s) {
      const uint64_t oa = pearl_hmma_a_src(tid, R) + s * pearl_dp4a_stage_step(m);
      const uint64_t ob = pearl_hmma_b_src(tid, C, col_off) + s * pearl_dp4a_stage_step(n);
      if (oa + 32 > Ak.size() || ob + 32 > Bk.size()) die("a copy reads past an operand");
      if (oa % 16 || ob % 16) die("a copy is not 16-byte aligned");
      memcpy(&ra[tid * 32], &Ak[oa], 32);
      memcpy(&rb[tid * 32], &Bk[ob], 32);
    };
    // The kernel's put(): each word to two fp16 words, then minus 1152.
    auto put = [&](std::vector<int32_t> &sh, std::vector<int> &wr, const uint8_t *src,
                   uint32_t buf, uint32_t tid, uint32_t s) {
      if (wr[buf * 128 + tid] == (int)s) die("two copies of one stage land on one row");
      wr[buf * 128 + tid] = (int)s;
      for (uint32_t i = 0; i < 8; i++) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++) w |= (uint32_t)src[4 * i + b] << (8 * b);
        for (uint32_t j = 0; j < 2; j++) {
          const uint32_t bits = pearl_hmma_biased(w, j);
          for (uint32_t h = 0; h < 2; h++) {
            const uint32_t hb = (bits >> (16 * h)) & 0xFFFFu;
            // A normal fp16 in [1024, 2048): exponent 25, so the value is 1024 + mantissa.
            if ((hb >> 10) != 25u) die("a biased half is not in [1024, 2048)");
            const int32_t x = (int32_t)(1024u + (hb & 0x3FFu)) - 1152;
            const int32_t want = (int8_t)src[4 * i + 2 * j + h];
            if (x != want) die("the fp16 bits are not the int8 value");
            if (pearl_hmma_half_bits(x) != pearl_hmma_half_bits(want)) die("half bits");
            sh[(buf * 128 + tid) * LDM + 4 * i + 2 * j + h] = x;
          }
        }
      }
    };

    for (uint32_t tid = 0; tid < T; tid++) copy(tid, 0);
    for (uint32_t tid = 0; tid < T; tid++) {
      put(sA, wA, &ra[tid * 32], 0, tid, 0);
      put(sB, wB, &rb[tid * 32], 0, tid, 0);
    }
    for (uint32_t s = 0; s < stages; s++) {
      const bool more = s + 1 < stages;
      const uint32_t buf = s & 1u;
      if (more)
        for (uint32_t tid = 0; tid < T; tid++) copy(tid, s + 1);
      // The mma: per warp, per lane, each of its 4x4 tiles of 8x8. Lane L brings row
      // pearl_hmma_a_row and column pearl_hmma_b_col; the quad pair's 8x8 result cell
      // (r, c) is row r8 = r of the pair's lanes by column r8 = c.
      for (uint32_t warp = 0; warp < 4; warp++) {
        for (uint32_t kk = 0; kk < PEARL_HMMA_STAGE_K; kk += 16) {
          for (uint32_t lane = 0; lane < 32; lane++) {
            const uint32_t q = pearl_hmma_qp(lane), r = pearl_hmma_r8(lane);
            for (uint32_t i = 0; i < 4; i++) {
              const uint32_t ar = pearl_hmma_a_row(warp, lane, i);
              if (wA[buf * 128 + ar] != (int)s) die("a stage reads a row of A' it did not write");
              for (uint32_t j = 0; j < 4; j++) {
                for (uint32_t pl = 0; pl < 32; pl++) {   // the pair's lanes, for the columns
                  if (pearl_hmma_qp(pl) != q) continue;
                  const uint32_t bc = pearl_hmma_b_col(warp, pl, j);
                  if (wB[buf * 128 + bc] != (int)s) die("a stage reads a row of B' it did not write");
                  int64_t sum = 0;
                  for (uint32_t t = 0; t < 16; t++)
                    sum += (int64_t)sA[(buf * 128 + ar) * LDM + kk + t]
                           * sB[(buf * 128 + bc) * LDM + kk + t];
                  int64_t &a = acc[((((warp * 4 + q) * 4 + i) * 4 + j) * 64) + r * 8 + pearl_hmma_r8(pl)];
                  a += sum;
                  // What the fp32 accumulator holds; exact as an integer under 2^24.
                  const int64_t held = (int64_t)bias + a, aa = held < 0 ? -held : held;
                  if (aa > run.maxAbs) run.maxAbs = aa;
                  if (aa >= (1 << 24)) die("an accumulator reached 2^24, past fp32's exact range");
                }
              }
            }
          }
        }
      }
      if (s % per_chunk == per_chunk - 1) {
        const uint32_t c = s / per_chunk;
        for (uint32_t warp = 0; warp < 4; warp++) {
          // Each lane's 4 words: lane-in-pair li holds cells li * 8 .. + 7 of each tile.
          uint32_t v[32][4] = {};
          for (uint32_t lane = 0; lane < 32; lane++) {
            const uint32_t q = pearl_hmma_qp(lane), li = pearl_hmma_r8(lane);
            for (uint32_t i = 0; i < 4; i++)
              for (uint32_t j = 0; j < 4; j++)
                for (uint32_t e = 0; e < 8; e++) {
                  const int64_t nv = acc[((((warp * 4 + q) * 4 + i) * 4 + j) * 64) + li * 8 + e];
                  const int64_t held = (int64_t)bias + nv;
                  const int64_t lo = held - (1 << 23), hi = (1 << 24) - 1 - held;
                  if (lo < run.margin) run.margin = lo;
                  if (hi < run.margin) run.margin = hi;
                  if (lo < 0 || hi < 0) die("an accumulator is outside the readout's window");
                  const float t = (float)held;   // exact: an integer under 2^24
                  uint32_t bits;
                  memcpy(&bits, &t, 4);
                  if ((uint32_t)(int32_t)nv != bits - U) die("the bits less U are not the integer");
                  // As the kernel: the bits alone when U's low 23 bits are zero.
                  v[lane][2 * (i >> 1) + (j >> 1)] ^= bias == 8388608u ? bits : bits - U;
                }
          }
          // The kernel's three shuffles, all lanes in step.
          uint32_t w0[32], w1[32], x[32], y[32];
          for (uint32_t L = 0; L < 32; L++) {
            const bool upA = (L & 16u) != 0, pA = ((L ^ 16u) & 16u) != 0;
            w0[L] = (upA ? v[L][2] : v[L][0]) ^ (pA ? v[L ^ 16u][0] : v[L ^ 16u][2]);
            w1[L] = (upA ? v[L][3] : v[L][1]) ^ (pA ? v[L ^ 16u][1] : v[L ^ 16u][3]);
          }
          for (uint32_t L = 0; L < 32; L++) {
            const bool upB = (L & 2u) != 0, pB = ((L ^ 2u) & 2u) != 0;
            x[L] = (upB ? w1[L] : w0[L]) ^ (pB ? w0[L ^ 2u] : w1[L ^ 2u]);
          }
          for (uint32_t L = 0; L < 32; L++) y[L] = x[L] ^ x[L ^ 1u];
          for (uint32_t lane = 0; lane < 32; lane += 2) {
            const uint32_t q = pearl_hmma_qp(lane), a = (lane >> 4) & 1u, b = (lane >> 1) & 1u;
            const uint32_t region = pearl_hmma_region(warp, lane);
            // What the lane should hold: the XOR of the region's four 8x8 tiles, and
            // those tiles' rows and columns must all be the region's groups.
            uint32_t want = 0;
            for (uint32_t i = 2 * a; i < 2 * a + 2; i++)
              for (uint32_t j = 2 * b; j < 2 * b + 2; j++) {
                for (uint32_t e = 0; e < 64; e++)
                  want ^= (uint32_t)acc[((((warp * 4 + q) * 4 + i) * 4 + j) * 64) + e];
                for (uint32_t pl = 0; pl < 32; pl++) {
                  if (pearl_hmma_qp(pl) != q) continue;
                  if (pearl_hmma_a_row(warp, pl, i) >> 4 != region >> 3
                      || pearl_hmma_b_col(warp, pl, j) >> 4 != (region & 7u))
                    die("a region's tiles are not its row and column groups");
                }
              }
            if (y[lane] != want) die("the shuffles do not leave the lane its region");
            const uint32_t t = region * PEARL_DP4A_TR_STRIDE + c;
            if (wT[t] != -1) die("two lanes write one transcript word");
            sT[t] = y[lane];
            wT[t] = (int)c;
          }
        }
      }
      if (more)
        for (uint32_t tid = 0; tid < T; tid++) {
          put(sA, wA, &ra[tid * 32], (s + 1) & 1u, tid, s + 1);
          put(sB, wB, &rb[tid * 32], (s + 1) & 1u, tid, s + 1);
        }
    }

    for (uint32_t tid = 0; tid < PEARL_DP4A_REGIONS; tid++) {
      std::vector<uint32_t> tm(16);
      for (uint32_t w = 0; w < 16; w++) {
        if (wT[tid * PEARL_DP4A_TR_STRIDE + w] != (int)w) die("a transcript word was never written");
        tm[w] = sT[tid * PEARL_DP4A_TR_STRIDE + w];
      }
      const uint32_t idx = pearl_dp4a_region_index(R, C, tid, rows_valid);
      if (idx >= batch || seen[idx]++) die("region indices are not one per region of the batch");
      out[idx] = tm;
    }
  }

  // The plain fold, region by region, from the index alone.
  for (uint32_t idx = 0; idx < batch; idx++) {
    const uint32_t rowOff = pearl_expand_offset(idx % rows_valid, PEARL_ROWS_MASK);
    const uint32_t colOff = pearl_expand_offset(col_off + idx / rows_valid, PEARL_COLS_MASK);
    int32_t tile[16][16] = {};
    uint32_t jp[PEARL_JACKPOT_BUCKETS] = {};
    for (uint32_t c = 0; c < k / rank; c++) {
      uint32_t tx = 0;
      for (int i = 0; i < 16; i++) {
        const int8_t *ar = &A[(size_t)(rowOff | PEARL_ROWS_PATTERN[i]) * k];
        for (int j = 0; j < 16; j++) {
          const int8_t *bc = &B[(size_t)(colOff | PEARL_COLS_PATTERN[j]) * k];
          int32_t a = tile[i][j];
          for (uint32_t kk = c * rank; kk < (c + 1) * rank; kk++)
            a = (int32_t)((uint32_t)a + (uint32_t)((int32_t)ar[kk] * (int32_t)bc[kk]));
          tile[i][j] = a;
          tx ^= (uint32_t)a;
        }
      }
      const uint32_t lane = c % PEARL_JACKPOT_BUCKETS;
      jp[lane] = pearl_rotl13(jp[lane]) ^ tx;
    }
    if (memcmp(jp, out[idx].data(), sizeof jp) != 0) {
      if (run.mismatches < 5) fprintf(stderr, "hmma-check: region %u differs from the plain fold\n", idx);
      run.mismatches++;
    }
  }
  run.regions = batch;
  return run;
}

int main() {
  // 4 row groups by 2 column groups from column offset 16 (a band shorter than
  // PEARL_DP4A_BAND), then 10 row groups by 1 from offset 8 (a whole band and a short one).
  struct Case { const char *name; int fill; int32_t center; uint32_t m, n, off, groups; };
  const Case cases[] = {
      {"constant fill 48", PEARL_OPERAND_FILL, PEARL_OPERAND_FILL * PEARL_OPERAND_FILL,
       4 * 128 * 2, 64 * 16 * 2, 16, 16},
      {"constant fill 48", PEARL_OPERAND_FILL, PEARL_OPERAND_FILL * PEARL_OPERAND_FILL,
       10 * 128 * 2, 32 * 16 * 2, 8, 8},
      {"hashed fill", -1, 0, 4 * 128 * 2, 32 * 16 * 2, 16, 16},
      {"uniform bytes", -2, 0, 2 * 128 * 2, 32 * 16 * 2, 0, 8},
  };
  int bad = 0;
  for (const Case &c : cases) {
    const Run r = check(c.m, c.n, c.off, c.groups, c.fill, c.center);
    printf("%s, m %u n %u: %u regions, %d mismatches; largest accumulator %lld (2^24 is %d), "
           "nearest a window end by %lld\n",
           c.name, c.m, c.n, r.regions, r.mismatches, (long long)r.maxAbs, 1 << 24,
           (long long)r.margin);
    bad += r.mismatches;
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 1 : 0;
}
