// Pascal's dp4a fold, run on the CPU. No GPU and no CUDA toolkit needed.
//
//   c++ -O2 -std=c++17 -I../src dp4a-check.cc -o dp4a-check
//   ./dp4a-check <A'.bin> <B'.bin> <m> <n> <col_off> <col_groups>
//
// Normally started by dp4a-check.js, which also writes the operands and compares a
// sample of what this prints against the JS reference (src/shared/miner/reference.js).
//
// It does what one launch of pearl_tile_fold_dp4a does, block by block and thread by
// thread, in the kernel's own phases and with the kernel's own functions
// (src/pearl_fold_dp4a.h). It first lays the operands out k-blocked, as the device draw
// does for this fold, and reads them only through that layout: which tile a block
// takes, where each thread's two 16-byte
// copies come from and land in the two stage buffers, the 64 dp4a a k-quad (the same
// pearl_dp4a_mac_stage, with __dp4a written out byte by byte), the readout's XOR, the
// two shuffles, which thread writes the word, and the region index a hasher reports.
// It also checks what the kernel cannot: that every word a stage reads was written
// for that stage, that every transcript word is written exactly once, and that the
// batch's region indices come out exactly once each.
//
// Then it folds every region again the plain way, as reference.js does: the tile's 16
// rows and 16 columns from the region index (pearl_expand_offset, as the host decodes a
// hit), one int32 accumulator a cell over the whole of k, the XOR read out every rank
// bytes. Both must agree on every region.
//
// Prints one line a region, "<batch index> <16 transcript words in hex>", and a last
// line "checked N regions, M mismatches". Exits 1 on any mismatch or failed check.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "pearl_fold_dp4a.h"

static void die(const char *msg) {
  fprintf(stderr, "dp4a-check: %s\n", msg);
  exit(1);
}

static std::vector<int8_t> slurp(const char *path, size_t want) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open an operand file");
  std::vector<int8_t> v(want);
  if (fread(v.data(), 1, want, f) != want) die("operand file is shorter than m*k or n*k");
  fclose(f);
  return v;
}

// A thread's 16-byte copy as the four words the device's int4 load gives it.
static void load16(const std::vector<int8_t> &op, uint64_t off, uint32_t w[4]) {
  if (off + 16u > op.size()) die("a copy reads past the operand");
  for (int i = 0; i < 4; i++) {
    w[i] = 0;
    for (int b = 0; b < 4; b++) w[i] |= (uint32_t)(uint8_t)op[off + 4 * i + b] << (8 * b);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) die("usage: dp4a-check <A'.bin> <B'.bin> <m> <n> <col_off> <col_groups>");
  const uint32_t k = PEARL_FOLD_K, rank = PEARL_FOLD_RANK;
  const uint32_t m = (uint32_t)strtoul(argv[3], nullptr, 10);
  const uint32_t n = (uint32_t)strtoul(argv[4], nullptr, 10);
  const uint32_t col_off = (uint32_t)strtoul(argv[5], nullptr, 10);
  const uint32_t col_groups = (uint32_t)strtoul(argv[6], nullptr, 10);
  const uint32_t rows_valid = m / PEARL_ROWS_COUNT, cols_valid = n / PEARL_COLS_COUNT;
  // What pearl_submit_dp4a insists on.
  if (rows_valid % PEARL_DP4A_ROW_OFFSETS || col_groups % PEARL_DP4A_COL_OFFSETS
      || col_off % PEARL_DP4A_COL_OFFSETS || col_off + col_groups > cols_valid)
    die("geometry is not whole 128x128 tiles inside the operands");
  const std::vector<int8_t> A = slurp(argv[1], (size_t)m * k);
  const std::vector<int8_t> B = slurp(argv[2], (size_t)n * k);
  // The noised operands as the fold reads them: k-blocked (pearl_dp4a_kblocked, the
  // layout pearl_materialize16_kblocked writes with 32-byte blocks). The plain fold
  // below reads the row-major originals.
  std::vector<int8_t> Ak((size_t)m * k), Bk((size_t)n * k);
  for (uint32_t r = 0; r < m; r++)
    for (uint32_t kk = 0; kk < k; kk++) Ak[pearl_dp4a_kblocked(r, kk, m)] = A[(size_t)r * k + kk];
  for (uint32_t c = 0; c < n; c++)
    for (uint32_t kk = 0; kk < k; kk++) Bk[pearl_dp4a_kblocked(c, kk, n)] = B[(size_t)c * k + kk];

  const uint32_t rowGroups = rows_valid / PEARL_DP4A_ROW_OFFSETS;
  const uint32_t colGroups = col_groups / PEARL_DP4A_COL_OFFSETS;
  const uint32_t tiles = rowGroups * colGroups;
  const uint32_t stages = k / PEARL_DP4A_STAGE_K, per_chunk = rank / PEARL_DP4A_STAGE_K;
  const uint32_t T = PEARL_DP4A_THREADS;
  const uint32_t batch = col_groups * rows_valid;

  std::vector<std::vector<uint32_t>> out(batch);   // the kernel's transcripts, by index
  std::vector<int> seen(batch, 0);
  std::vector<int> tileSeen((size_t)rowGroups * colGroups, 0);
  int bad = 0;

  // The block's shared memory, and which stage last wrote each word (-1: none yet).
  std::vector<uint32_t> sA(2 * PEARL_DP4A_STAGE_WORDS), sB(2 * PEARL_DP4A_STAGE_WORDS);
  std::vector<int> wA(2 * PEARL_DP4A_STAGE_WORDS), wB(2 * PEARL_DP4A_STAGE_WORDS);
  std::vector<uint32_t> sT(PEARL_DP4A_REGIONS * PEARL_DP4A_TR_STRIDE);
  std::vector<int> wT(PEARL_DP4A_REGIONS * PEARL_DP4A_TR_STRIDE);
  // Per thread: the accumulators and the copies held in registers between phases.
  std::vector<int32_t> acc((size_t)T * 64);
  std::vector<uint32_t> ra((size_t)T * 4), rb((size_t)T * 4);
  std::vector<uint32_t> x(T), x1(T);

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
      load16(Ak, pearl_dp4a_a_src(tid, R) + s * pearl_dp4a_stage_step(m), &ra[tid * 4]);
      load16(Bk, pearl_dp4a_b_src(tid, C, col_off) + s * pearl_dp4a_stage_step(n), &rb[tid * 4]);
    };
    auto store = [&](uint32_t tid, uint32_t buf, uint32_t s) {
      for (uint32_t i = 0; i < 4; i++) {
        const uint32_t d = buf * PEARL_DP4A_STAGE_WORDS + pearl_dp4a_dst(tid, i);
        if (wA[d] == (int)s || wB[d] == (int)s) die("two copies of one stage land on one word");
        sA[d] = ra[tid * 4 + i];
        sB[d] = rb[tid * 4 + i];
        wA[d] = wB[d] = (int)s;
      }
    };

    // Stage 0, then the barrier.
    for (uint32_t tid = 0; tid < T; tid++) copy(tid, 0);
    for (uint32_t tid = 0; tid < T; tid++) store(tid, 0, 0);

    for (uint32_t s = 0; s < stages; s++) {
      const bool more = s + 1 < stages;
      const uint32_t buf = s & 1u;
      if (more)
        for (uint32_t tid = 0; tid < T; tid++) copy(tid, s + 1);
      for (uint32_t tid = 0; tid < T; tid++) {
        const uint32_t tx = pearl_dp4a_tx(tid), ty = pearl_dp4a_ty(tid);
        // Every word this thread's two 16-byte loads a k-quad read must hold this stage.
        for (uint32_t q = 0; q < PEARL_DP4A_KQ; q++) {
          for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t h = 0; h < 2; h++) {
              const uint32_t base = buf * PEARL_DP4A_STAGE_WORDS + q * PEARL_DP4A_SSTRIDE + 64 * h;
              if (wA[base + 4 * ty + i] != (int)s || wB[base + 4 * tx + i] != (int)s)
                die("a stage reads a word it did not write");
            }
          }
        }
        int32_t a8[8][8];
        memcpy(a8, &acc[(size_t)tid * 64], sizeof a8);
        pearl_dp4a_mac_stage(&sA[buf * PEARL_DP4A_STAGE_WORDS], &sB[buf * PEARL_DP4A_STAGE_WORDS],
                             tx, ty, a8);
        memcpy(&acc[(size_t)tid * 64], a8, sizeof a8);
      }
      if (s % per_chunk == per_chunk - 1) {
        const uint32_t chunk = s / per_chunk;
        for (uint32_t tid = 0; tid < T; tid++) {
          int32_t a8[8][8];
          memcpy(a8, &acc[(size_t)tid * 64], sizeof a8);
          x[tid] = pearl_dp4a_xor64(a8);
        }
        // __shfl_xor_sync(.., 1) then (.., 8): lane ^ 1 and lane ^ 8 are tid ^ 1 and tid ^ 8.
        for (uint32_t tid = 0; tid < T; tid++) x1[tid] = x[tid] ^ x[tid ^ 1u];
        for (uint32_t tid = 0; tid < T; tid++) {
          const uint32_t x2 = x1[tid] ^ x1[tid ^ 8u];
          if (!pearl_dp4a_writer(tid)) continue;
          const uint32_t t = pearl_dp4a_region(tid) * PEARL_DP4A_TR_STRIDE + chunk;
          if (wT[t] != -1) die("two threads write one transcript word");
          sT[t] = x2;
          wT[t] = (int)chunk;
        }
      }
      if (more)
        for (uint32_t tid = 0; tid < T; tid++) store(tid, (s + 1) & 1u, s + 1);
    }

    // The hashers: threads 0-63, a region each.
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
      if (bad < 5) fprintf(stderr, "dp4a-check: region %u differs from the plain fold\n", idx);
      bad++;
    }
  }

  for (uint32_t idx = 0; idx < batch; idx++) {
    printf("%u", idx);
    for (uint32_t w = 0; w < 16; w++) printf(" %08x", out[idx][w]);
    printf("\n");
  }
  printf("checked %u regions, %d mismatches\n", batch, bad);
  return bad ? 1 : 0;
}
