// qcdrv: runs the scan kernel from a cubin file (a rebalance.py output) through the driver API, beside the same
// kernel compiled into this program, and either checks that the two find exactly the same hits or times the cubin.
//
//   qcdrv <file.cubin> check [rounds]      random prestates, 1 in 16 nonces hit, 2^22 nonces a round: hit lists must match
//   qcdrv <file.cubin> time <seconds>      back-to-back launches of 2^26 nonces; prints "mean <MH/s>"
//   qcdrv - time <seconds>                 the same for the compiled-in kernel
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <vector>

#include "qc_kernel.cuh"

#define CK(x)                                                              \
  do {                                                                     \
    CUresult r_ = (x);                                                     \
    if (r_ != CUDA_SUCCESS) {                                              \
      const char *s_ = 0;                                                  \
      cuGetErrorString(r_, &s_);                                           \
      printf("%s failed: %s\n", #x, s_ ? s_ : "?");                        \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static unsigned long long rng = 0x9e3779b97f4a7c15ull;
static unsigned long long rnd() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: qcdrv <file.cubin|-> check|time [n]\n");
    return 2;
  }
  bool use_file = strcmp(argv[1], "-") != 0;
  bool check = strcmp(argv[2], "check") == 0;
  double n = argc > 3 ? atof(argv[3]) : (check ? 20 : 20);
  cudaSetDevice(0);
  cudaFree(0);
  CUfunction f = 0;
  if (use_file) {
    CUmodule mod;
    CK(cuModuleLoad(&mod, argv[1]));
    CK(cuModuleGetFunction(&f, mod, "_Z14qc_scan_kernel8QcLaunchPyS0_"));
  }
  int sms = 0, per = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (use_file) CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per, f, QC_THREADS, 0));
  else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, qc_scan_kernel, QC_THREADS, 0);
  int grid = sms * per;
  unsigned long long *d_n;
  u64 *d_h;
  const unsigned cap = 1 << 20;
  cudaMalloc(&d_n, 8);
  cudaMalloc(&d_h, cap * 8);
  QcLaunch L;
  memset(&L, 0, sizeof L);
  L.cap = cap;
  auto launch = [&](bool file) {
    void *args[] = {&L, &d_n, &d_h};
    unsigned blocks = (L.count + QC_THREADS - 1) / QC_THREADS;
    unsigned g = blocks < (unsigned)grid ? blocks : (unsigned)grid;
    if (file) CK(cuLaunchKernel(f, g, 1, 1, QC_THREADS, 1, 1, 0, 0, args, 0));
    else qc_scan_kernel<<<g, QC_THREADS>>>(L, d_n, d_h);
  };
  auto hits = [&](bool file) {
    cudaMemset(d_n, 0, 8);
    launch(file);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("kernel: %s\n", cudaGetErrorString(e));
      exit(1);
    }
    unsigned long long k = 0;
    cudaMemcpy(&k, d_n, 8, cudaMemcpyDeviceToHost);
    std::vector<u64> v(k < cap ? k : cap);
    if (!v.empty()) cudaMemcpy(v.data(), d_h, v.size() * 8, cudaMemcpyDeviceToHost);
    std::sort(v.begin(), v.end());
    return v;
  };
  if (check) {
    int bad = 0;
    size_t total = 0;
    for (int r = 0; r < (int)n; r++) {
      for (int i = 0; i < 12; i++) L.c[i] = rnd() % 0xFFFFFFFF00000001ull;
      L.target = 1ull << 60;
      L.base = (rnd() & 0xFFFFFFFF00000000ull) | (rnd() & 0x3FFFFFFFull);
      L.count = 1u << 22;
      std::vector<u64> a = hits(false), b = hits(true);
      total += a.size();
      if (a != b) {
        bad++;
        printf("round %d: %zu hits compiled-in, %zu from the cubin\n", r, a.size(), b.size());
      }
    }
    printf("%s: %d of %d rounds differ (%zu hits compared)\n", bad ? "FAIL" : "PASS", bad, (int)n, total);
    return bad ? 1 : 0;
  }
  for (int i = 0; i < 12; i++) L.c[i] = rnd() % 0xFFFFFFFF00000001ull;
  L.target = 0;
  L.count = 1u << 26;
  double t0 = now_s(), done = 0;
  L.base = 1ull << 40;
  while (now_s() - t0 < n) {
    launch(use_file);
    cudaDeviceSynchronize();
    done += L.count;
    L.base += L.count;
  }
  double t = now_s() - t0;
  printf("mean %.2f MH/s over %.1f s (%s, grid %d)\n", done / t / 1e6, t, use_file ? argv[1] : "compiled-in", grid);
  return 0;
}
