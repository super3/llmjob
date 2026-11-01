// Fold check: run every AMD fold this build has over random operands and compare each
// region's 16 transcript words with a plain CPU computation.
//
//   pearl_amd_foldtest [m=256] [cols=512] [seed=1] [reps=1] [check=1] [fold|all] [k] [int8|miner]
//
// Built for HIP-CPU (build-amd.sh cpu), it runs all five folds through the emulated matrix
// instructions. Built for a real card (build-amd.sh gpu), it runs the folds that card has
// and is the first thing to run on new hardware: it shows whether the hardware's lane
// layout is the one the folds were written against, before anything else depends on it.
// It also times `reps` launches of each fold and prints the rate in TMAC/s. check=0 skips
// the CPU computation, which takes minutes at sizes big enough to time a GPU with. `fold`
// runs only the fold of that name (the reference fold is slow to time at big sizes); "all"
// runs every one. `k` is the k to fold. By default it is the k the card mines: 4096 where
// the card's code object has the wrap readout (RDNA 4, PEARL_AMD_CAP_KWRAP), else 2048, or
// the build's PEARL_FOLD_K where that is larger. The device line names it; k 0 asks for it.
//
// Operands are random over all of int8, wider than the miner's int7 operands make, so a
// sign or byte-order mistake cannot hide. B' has twice `cols` rows and the batch starts at
// row `cols`, so a fold that ignores its column start fails too.
//
// `miner` folds what the miner folds instead: every byte PEARL_OPERAND_FILL (48) plus the
// protocol's noise, so A' and B' sit in [-15, 111] (pearl_config.h, PEARL_OPERAND_CONST). The
// bits the matrix units see move a power-capped card's clock (3-7% on a 4090, pearl_config.h),
// so time on these when the clock matters: sweep.sh's SWEEP_OPERANDS=miner. The salt stamp,
// 11 bytes of A, is left out.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pearl_amd_kernels.h"

namespace {

struct Fold {
  const char *name;
  uint32_t cap;
  bool tall;  // a 128x256 tile (the matrix folds) rather than 32x64 (the reference)
};

uint64_t g_rng = 1;
uint32_t next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return (uint32_t)g_rng;
}

// `rows` rows of k bytes as the miner fills them: PEARL_OPERAND_FILL plus the noise, in the
// shape noise.js ports from the reference. A row's noise at kk is e[p0] - e[p1]: e is the row's
// `rank` draws in [-32, 32), and p0 != p1 is one pair per kk, the same for every row.
void miner_operands(std::vector<int8_t> &x, uint32_t rows, uint32_t k, uint32_t rank) {
  std::vector<uint32_t> p0(k), p1(k);
  for (uint32_t kk = 0; kk < k; kk++) {
    const uint32_t u = next_rand();
    p0[kk] = u & (rank - 1u);
    p1[kk] = p0[kk] ^ (1u + (uint32_t)(((uint64_t)(rank - 1u) * u) >> 32));
  }
  std::vector<int32_t> e(rank);
  for (uint32_t r = 0; r < rows; r++) {
    for (auto &v : e) v = (int32_t)(next_rand() & 63u) - 32;
    for (uint32_t kk = 0; kk < k; kk++) {
      const int32_t v = PEARL_OPERAND_FILL + e[p0[kk]] - e[p1[kk]];
      x[(size_t)r * k + kk] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
    }
  }
}

bool ok(hipError_t e, const char *what) {
  if (e == hipSuccess) return true;
  fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
  return false;
}

void launch(const Fold &f, dim3 grid, const int8_t *a, const int8_t *b, uint32_t k,
            uint32_t rank, uint32_t col0, uint32_t rowsValid, uint32_t *tr) {
  const dim3 block(PEARL_AMD_THREADS);
  if (f.cap == PEARL_AMD_CAP_REF)
    hipLaunchKernelGGL(pearl_amd_fold_ref, grid, block, 0, 0, a, b, k, rank, col0, rowsValid, tr);
  else if (f.cap == PEARL_AMD_CAP_MFMA16)
    hipLaunchKernelGGL(pearl_amd_fold_mfma16, grid, block, 0, 0, a, b, k, rank, col0, rowsValid, tr);
  else if (f.cap == PEARL_AMD_CAP_MFMA8)
    hipLaunchKernelGGL(pearl_amd_fold_mfma8, grid, block, 0, 0, a, b, k, rank, col0, rowsValid, tr);
  else if (f.cap == PEARL_AMD_CAP_WMMA11)
    hipLaunchKernelGGL(pearl_amd_fold_wmma11, grid, block, 0, 0, a, b, k, rank, col0, rowsValid, tr);
  else
    hipLaunchKernelGGL(pearl_amd_fold_wmma12, grid, block, 0, 0, a, b, k, rank, col0, rowsValid, tr);
}

}  // namespace

int main(int argc, char **argv) {
  const uint32_t m = argc > 1 ? (uint32_t)atoi(argv[1]) : 256u;
  const uint32_t cols = argc > 2 ? (uint32_t)atoi(argv[2]) : 512u;
  g_rng = argc > 3 ? (uint64_t)atoll(argv[3]) | 1u : 1u;
  const int reps = argc > 4 ? atoi(argv[4]) : 1;
  const bool check = argc > 5 ? atoi(argv[5]) != 0 : true;
  const char *only = argc > 6 && argv[6][0] && strcmp(argv[6], "all") != 0 ? argv[6] : nullptr;
  const uint32_t kArg = argc > 7 ? (uint32_t)atoi(argv[7]) : 0u;
  const bool miner = argc > 8 && strcmp(argv[8], "miner") == 0;
  const uint32_t rank = PEARL_FOLD_RANK;
  if (m == 0 || m % PEARL_AMD_BM || cols == 0 || cols % PEARL_AMD_BN) {
    fprintf(stderr, "m must be a multiple of %u and cols of %u\n", PEARL_AMD_BM, PEARL_AMD_BN);
    return 2;
  }
  if (argc > 8 && !miner && strcmp(argv[8], "int8") != 0) {
    fprintf(stderr, "operands are int8 or miner, not %s\n", argv[8]);
    return 2;
  }
  const uint32_t n = 2 * cols, col0 = cols;
  const uint32_t rowsValid = m / PEARL_ROWS_COUNT, colIdx = cols / PEARL_COLS_COUNT;
  const uint32_t regions = rowsValid * colIdx;

  hipDeviceProp_t prop;
  if (!ok(hipGetDeviceProperties(&prop, 0), "no device")) return 2;
  // What the code object for this card has, first: it decides the k.
  uint32_t *dCaps = nullptr;
  if (!ok(hipMalloc(&dCaps, 8), "hipMalloc caps")) return 2;
  hipLaunchKernelGGL(pearl_amd_caps, dim3(1), dim3(1), 0, 0, dCaps);
  uint32_t caps[2] = {0, 0};
  if (!ok(hipDeviceSynchronize(), "caps") || !ok(hipMemcpy(caps, dCaps, 8, hipMemcpyDeviceToHost), "caps")) return 2;
  const bool wrap = (caps[0] & PEARL_AMD_CAP_KWRAP) != 0;
  const uint32_t k = kArg ? kArg : (wrap && PEARL_FOLD_K < 4096u ? 4096u : (uint32_t)PEARL_FOLD_K);
  const uint32_t chunks = k / rank;
  // The host's rule (pearl_host_submit): at most 16 chunks without the wrap readout, and past 16
  // a whole number of 16 with it.
  if (k % rank || chunks == 0 || (chunks > 16u && (!wrap || chunks % 16u))) {
    fprintf(stderr, "k %u: it must be a multiple of %u, at most %u without the wrap readout, and past that a "
            "multiple of %u (this card's code object %s the wrap readout)\n", k, rank, 16u * rank, 16u * rank,
            wrap ? "has" : "does not have");
    return 2;
  }
#if defined(__HIP_CPU_RT__)
  printf("device: %s (HIP-CPU: every fold runs on the emulated instructions); k %u%s\n", prop.name, k,
         wrap ? ", wrap readout" : "");
#else
  printf("device: %s, %s, %d CUs; k %u%s\n", prop.name, prop.gcnArchName, prop.multiProcessorCount, k,
         wrap ? ", wrap readout" : "");
#endif
  if (miner) printf("operands: the miner's, fill %d plus the noise\n", (int)PEARL_OPERAND_FILL);

  std::vector<int8_t> A((size_t)m * k), B((size_t)n * k);
  if (miner) {
    miner_operands(A, m, k, rank);
    miner_operands(B, n, k, rank);
  } else {
    for (auto &v : A) v = (int8_t)next_rand();
    for (auto &v : B) v = (int8_t)next_rand();
  }

  // The reference: every element of the batch's C, after every chunk, folded per region.
  std::vector<uint32_t> want((size_t)regions * 16, 0u);
  if (check) {
    std::vector<int32_t> run((size_t)m * cols, 0);
    for (uint32_t c = 0; c < chunks; c++) {
      for (uint32_t r = 0; r < m; r++)
        for (uint32_t j = 0; j < cols; j++) {
          const int8_t *a = &A[(size_t)r * k + (size_t)c * rank];
          const int8_t *b = &B[(size_t)(col0 + j) * k + (size_t)c * rank];
          int32_t s = 0;
          for (uint32_t q = 0; q < rank; q++) s += (int32_t)a[q] * (int32_t)b[q];
          run[(size_t)r * cols + j] += s;
        }
      for (uint32_t ci = 0; ci < colIdx; ci++)
        for (uint32_t ri = 0; ri < rowsValid; ri++) {
          const uint32_t ro = pearl_expand_offset(ri, PEARL_ROWS_MASK);
          const uint32_t co = pearl_expand_offset(ci, PEARL_COLS_MASK);
          uint32_t x = 0u;
          for (uint32_t a = 0; a < PEARL_ROWS_COUNT; a++)
            for (uint32_t b = 0; b < PEARL_COLS_COUNT; b++)
              x ^= (uint32_t)run[(size_t)(ro | PEARL_ROWS_PATTERN[a]) * cols + (co | PEARL_COLS_PATTERN[b])];
          uint32_t &w = want[((size_t)ci * rowsValid + ri) * 16 + (c % 16)];
          w = pearl_rotl13(w) ^ x;
        }
    }
  }

  int8_t *dA = nullptr, *dB = nullptr;
  uint32_t *dTr = nullptr;
  if (!ok(hipMalloc(&dA, A.size()), "hipMalloc A") || !ok(hipMalloc(&dB, B.size()), "hipMalloc B")
      || !ok(hipMalloc(&dTr, want.size() * 4), "hipMalloc tr"))
    return 2;
  // The matrix folds read the operands in per-tile order when built with PEARL_AMD_TILED, as
  // the host materialises them then (pearl_materialize16_tiled's addresses); ref reads
  // row-major, so it gets its own copies.
  auto tiled = [&](const std::vector<int8_t> &src, uint32_t rows, uint32_t br) {
    std::vector<int8_t> out(src.size());
    const uint32_t kb = PEARL_AMD_SLAB;
    for (uint32_t r = 0; r < rows; r++)
      for (uint32_t kk = 0; kk < k; kk++)
        out[(size_t)(r / br) * br * k + ((size_t)(kk / kb) * br + r % br) * kb + kk % kb] = src[(size_t)r * k + kk];
    return out;
  };
  int8_t *dAt = dA, *dBt = dB;
  if (PEARL_AMD_TILED) {
    if (!ok(hipMalloc(&dAt, A.size()), "hipMalloc A tiled") || !ok(hipMalloc(&dBt, B.size()), "hipMalloc B tiled"))
      return 2;
    const std::vector<int8_t> At = tiled(A, m, PEARL_AMD_BM), Bt = tiled(B, n, PEARL_AMD_BN);
    hipMemcpy(dAt, At.data(), At.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBt, Bt.data(), Bt.size(), hipMemcpyHostToDevice);
  }
  hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice);
  hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice);
  printf("folds in this build for this device: 0x%x, wave %u\n", caps[0], caps[1]);

  const Fold folds[] = {{"ref", PEARL_AMD_CAP_REF, false},
                        {"mfma16", PEARL_AMD_CAP_MFMA16, true},
                        {"mfma8", PEARL_AMD_CAP_MFMA8, true},
                        {"wmma11", PEARL_AMD_CAP_WMMA11, true},
                        {"wmma12", PEARL_AMD_CAP_WMMA12, true}};
  int failed = 0, ran = 0;
  std::vector<uint32_t> got(want.size());
  for (const Fold &f : folds) {
    if (!(caps[0] & f.cap) || (only && strcmp(only, f.name) != 0)) continue;
    ran++;
    const dim3 grid = f.tall ? dim3(cols / PEARL_AMD_BN, m / PEARL_AMD_BM)
                             : dim3(cols / PEARL_AMD_REF_BN, m / PEARL_AMD_REF_BM);
    hipMemset(dTr, 0xA5, want.size() * 4);  // a word the fold forgets cannot read as zero
    // One launch to check (and to load the code), then `reps` more to time.
    const int8_t *fa = f.tall ? dAt : dA, *fb = f.tall ? dBt : dB;
    launch(f, grid, fa, fb, k, rank, col0, rowsValid, dTr);
    hipError_t e = hipDeviceSynchronize();
    if (!ok(e, f.name) || !ok(hipGetLastError(), f.name)) { failed++; continue; }
    hipMemcpy(got.data(), dTr, got.size() * 4, hipMemcpyDeviceToHost);
    const int n = reps > 0 ? reps : 1;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) launch(f, grid, fa, fb, k, rank, col0, rowsValid, dTr);
    e = hipDeviceSynchronize();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok(e, f.name) || !ok(hipGetLastError(), f.name)) { failed++; continue; }
    const double tmacs = (double)m * cols * k * n / secs / 1e12;
    size_t bad = 0, first = 0;
    for (size_t i = 0; check && i < got.size(); i++)
      if (got[i] != want[i] && bad++ == 0) first = i;
    if (!check) {
      printf("%-7s timed only: %u regions, %d launches, %.4f s, %.3f TMAC/s\n", f.name, regions, n,
             secs, tmacs);
    } else if (bad) {
      failed++;
      printf("%-7s FAIL: %zu of %zu words differ; first: region %zu word %zu, got %08x want %08x\n",
             f.name, bad, got.size(), first / 16, first % 16, got[first], want[first]);
    } else {
      printf("%-7s ok: %u regions, all 16 words match; %d launches, %.4f s, %.3f TMAC/s\n", f.name,
             regions, n, secs, tmacs);
    }
  }
  if (dAt != dA) hipFree(dAt);
  if (dBt != dB) hipFree(dBt);
  hipFree(dA); hipFree(dB); hipFree(dTr); hipFree(dCaps);
  if (!ran) { printf("no fold ran\n"); return 1; }
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
