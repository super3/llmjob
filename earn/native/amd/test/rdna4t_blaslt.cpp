// goal-rdna4t: AMD's own int8 GEMM (hipBLASLt, Tensile's gfx12 assembly kernels) on this card, for comparison with
// the fold's TMAC/s. Timing only; it checks nothing.
//
//   rdna4t_blaslt SECS M:N:K [M:N:K ...]
//
// A is M x K and B is N x K, both with K contiguous (the fold's operand order: Tensile's Alik_Bljk, "TN"); D is
// M x N int32. Operands are the miner's kind of int8 (48 plus small noise). For each shape it asks the heuristic for
// up to 24 algorithms, times each for SECS, and prints each one's TMAC/s and kernel name, then the best.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#ifdef RDNA4T_KNAME
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    auto e_ = (x);                                                                               \
    if ((int)e_ != 0) { printf("blaslt FAIL %s = %d (line %d)\n", #x, (int)e_, __LINE__); return 1; } \
  } while (0)

__global__ void fill(int8_t *p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = (int8_t)(48 + (int)(x % 31) - 15);
  }
}

static int run(double secs, long M, long N, long K) {
  int8_t *A = nullptr, *B = nullptr;
  int32_t *D = nullptr;
  void *ws = nullptr;
  const size_t wsBytes = 256u << 20;
  CK(hipMalloc(&A, (size_t)M * K));
  CK(hipMalloc(&B, (size_t)N * K));
  CK(hipMalloc(&D, (size_t)M * N * 4));
  CK(hipMalloc(&ws, wsBytes));
  hipLaunchKernelGGL(fill, dim3(1024), dim3(256), 0, 0, A, (size_t)M * K, 1u);
  hipLaunchKernelGGL(fill, dim3(1024), dim3(256), 0, 0, B, (size_t)N * K, 2u);
  CK(hipDeviceSynchronize());

  hipblasLtHandle_t h;
  CK(hipblasLtCreate(&h));
  hipblasLtMatmulDesc_t op;
  CK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
  hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
  CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  hipblasLtMatrixLayout_t la, lb, lc;
  CK(hipblasLtMatrixLayoutCreate(&la, HIP_R_8I, K, M, K));
  CK(hipblasLtMatrixLayoutCreate(&lb, HIP_R_8I, K, N, K));
  CK(hipblasLtMatrixLayoutCreate(&lc, HIP_R_32I, M, N, M));
  hipblasLtMatmulPreference_t pref;
  CK(hipblasLtMatmulPreferenceCreate(&pref));
  uint64_t wsb = wsBytes;
  CK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
  hipblasLtMatmulHeuristicResult_t res[24];
  int got = 0;
  CK(hipblasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 24, res, &got));
  printf("blaslt shape %ld:%ld:%ld algos %d\n", M, N, K, got);
  fflush(stdout);
  const int32_t alpha = 1, beta = 0;
  const double macs = (double)M * N * K;
  double best = 0;
  int bi = -1;
  for (int a = 0; a < got; a++) {
    if (res[a].state != HIPBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < 3 && ok; w++)
      ok = hipblasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, lc, D, lc, &res[a].algo, ws, wsBytes, 0) ==
           HIPBLAS_STATUS_SUCCESS;
    if (!ok || hipDeviceSynchronize() != hipSuccess) { printf("blaslt algo %d failed to run\n", a); continue; }
    // calibrate: how many launches fill SECS
    auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    double el = 0;
    while (el < secs) {
      for (int r = 0; r < 4; r++, n++)
        hipblasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, lc, D, lc, &res[a].algo, ws, wsBytes, 0);
      (void)hipDeviceSynchronize();
      el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    const double tmac = macs * n / el / 1e12;
    std::string kn = "?";
#ifdef RDNA4T_KNAME
    kn = hipblaslt_ext::getKernelNameFromAlgo(h, res[a].algo);
#endif
    printf("blaslt algo %d %.2f TMAC/s (%d launches, %.3f ms each) ws %zu %s\n", a, tmac, n, el / n * 1e3,
           (size_t)res[a].workspaceSize, kn.c_str());
    fflush(stdout);
    if (tmac > best) { best = tmac; bi = a; }
  }
  printf("blaslt best %ld:%ld:%ld algo %d %.2f TMAC/s\n", M, N, K, bi, best);
  fflush(stdout);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(la);
  hipblasLtMatrixLayoutDestroy(lb);
  hipblasLtMatrixLayoutDestroy(lc);
  hipblasLtMatmulDescDestroy(op);
  hipblasLtDestroy(h);
  hipFree(A); hipFree(B); hipFree(D); hipFree(ws);
  return 0;
}

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 2.0;
  for (int i = 2; i < argc; i++) {
    long M, N, K;
    if (sscanf(argv[i], "%ld:%ld:%ld", &M, &N, &K) != 3) continue;
    run(secs, M, N, K);
  }
  return 0;
}
