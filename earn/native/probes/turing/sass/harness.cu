// Runs kernels.cu's kernels (from the nvcc cubin) and hand.py's (from the
// CuAssembler cubin) on one card and prints [sass] lines.
//
// usage: harness kernels.cubin hand.cubin [tiles per block] [rounds]
//
// Every kernel runs one block an SM, 256 threads, 40 KB of dynamic shared.
// Rate is per clock: each warp reads clock64 around its loop, a block's span
// is its last warp's end minus its first warp's start, and IMMA/clk/SM is the
// block's IMMA (tiles x 64 k-steps x 128 x 8 warps) over its span, averaged
// over the blocks. The clock a launch ran at is the mean span over its wall
// time. Rounds are interleaved, so a power-capped card's clock drift hits
// every kernel alike.
//
// Correctness: each hand kernel's accumulators must equal sass_ptxas's,
// word for word, on the real loop (b=L2), and sass_ptxas's must equal a CPU
// reference on four blocks.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      printf("[sass] FAIL %s: %s\n", #x, cudaGetErrorString(e_));                             \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)
#define CD(x)                                                                                 \
  do {                                                                                        \
    CUresult r_ = (x);                                                                        \
    if (r_ != CUDA_SUCCESS) {                                                                 \
      const char *s_ = nullptr;                                                               \
      cuGetErrorString(r_, &s_);                                                              \
      printf("[sass] FAIL %s: %s\n", #x, s_ ? s_ : "?");                                      \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

static const uint32_t K = 2048, KS = K / 32, SMEM = 40960;

struct Kern {
  const char *name;  // short name in the output
  const char *sym;   // kernel symbol
  int hand;          // from the CuAssembler cubin
  int ceiling;       // a denominator: no B, run once a round
  CUfunction f;
};

static uint32_t lcg(uint64_t &s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 33);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s kernels.cubin hand.cubin [tiles] [rounds]\n", argv[0]);
    return 2;
  }
  const uint32_t tiles = argc > 3 ? (uint32_t)atoi(argv[3]) : 600u;
  const int rounds = argc > 4 ? atoi(argv[4]) : 7;
  CK(cudaFree(0));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const uint32_t sms = (uint32_t)p.multiProcessorCount;
  // The fold's band of column groups (pearl_bd_band_for, column walk): 8,
  // halved while 8 x 512 KB is more than 75% of the L2, never below 2.
  uint32_t ncg = 8;
  while (ncg > 2 && (uint64_t)ncg * 256u * K * 100u > (uint64_t)p.l2CacheSize * 75u) ncg /= 2;
  printf("[sass] device %s sm_%d%d SMs %u L2 %.1f MB band %u groups (%u KB of B) tiles %u rounds %d\n",
         p.name, p.major, p.minor, sms, p.l2CacheSize / 1048576.0, ncg, ncg * 512, tiles, rounds);

  CUmodule mk, mh;
  CD(cuModuleLoad(&mk, argv[1]));
  CD(cuModuleLoad(&mh, argv[2]));
  std::vector<Kern> ks = {
      {"ceiling", "sass_ceiling", 0, 1, nullptr}, {"hceil", "sass_hceil", 1, 1, nullptr},
      {"ptxas", "sass_ptxas", 0, 0, nullptr},     {"hand1", "sass_hand1", 1, 0, nullptr},
      {"hand2", "sass_hand2", 1, 0, nullptr},     {"hand3", "sass_hand3", 1, 0, nullptr}};
  for (auto &k : ks) CD(cuModuleGetFunction(&k.f, k.hand ? mh : mk, k.sym));

  // Operands. A: each block's 128 rows of 256 bytes (two chunks). B: ncg
  // column groups of 256 columns x k, written fragment-ordered as
  // pearl_materialize16_frag writes B'. Values in [-32, 31], so 600 tiles of
  // sums stay far inside int32.
  uint64_t seed = 0x5eed;
  std::vector<int8_t> A((size_t)sms * 128 * 256), Bl((size_t)ncg * 256 * K), Bf(Bl.size());
  for (auto &x : A) x = (int8_t)((int)(lcg(seed) & 63) - 32);
  for (auto &x : Bl) x = (int8_t)((int)(lcg(seed) & 63) - 32);
  for (uint32_t col = 0; col < ncg * 256u; col++)
    for (uint32_t k = 0; k < K; k++) {
      const uint32_t blk = col / 16, r = col % 16, s = k / 32, kk = k % 32;
      const uint32_t lane = 4 * (r & 7) + ((kk >> 2) & 3), reg = ((r >> 3) & 1) * 2 + ((kk >> 4) & 1);
      Bf[((size_t)blk * KS + s) * 512 + lane * 16 + reg * 4 + (kk & 3)] = Bl[(size_t)col * K + k];
    }
  int8_t *dA, *dB;
  CK(cudaMalloc(&dA, A.size()));
  CK(cudaMalloc(&dB, Bf.size()));
  CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, Bf.data(), Bf.size(), cudaMemcpyHostToDevice));
  const size_t outWords = (size_t)sms * 128 * 256;
  int32_t *dOut;
  long long *dCyc;
  CK(cudaMalloc(&dOut, outWords * 4));
  CK(cudaMalloc(&dCyc, (size_t)sms * 16 * 8));
  std::vector<int32_t> out(outWords), ref(outWords);
  std::vector<long long> cyc((size_t)sms * 16);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // One launch: returns IMMA/clk/SM and the clock in MHz; out and cyc hold
  // its results.
  auto launch = [&](Kern &k, uint32_t kmask, double &mhz, double &ms) -> double {
    CK(cudaMemset(dOut, 0, outWords * 4));
    CK(cudaMemset(dCyc, 0, (size_t)sms * 16 * 8));
    uint32_t t = tiles, n = ncg, m = kmask;
    void *args[] = {&dA, &dB, &dOut, &dCyc, &t, &n, &m};
    CK(cudaEventRecord(e0));
    CD(cuLaunchKernel(k.f, sms, 1, 1, 256, 1, 1, SMEM, 0, args, nullptr));
    CK(cudaEventRecord(e1));
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    float fms;
    CK(cudaEventElapsedTime(&fms, e0, e1));
    CK(cudaMemcpy(out.data(), dOut, outWords * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cyc.data(), dCyc, cyc.size() * 8, cudaMemcpyDeviceToHost));
    double rate = 0, span_sum = 0;
    for (uint32_t b = 0; b < sms; b++) {
      long long lo = cyc[b * 16], hi = cyc[b * 16 + 1];
      for (int w = 1; w < 8; w++) {
        lo = std::min(lo, cyc[b * 16 + w * 2]);
        hi = std::max(hi, cyc[b * 16 + w * 2 + 1]);
      }
      const double span = (double)(hi - lo);
      rate += (double)tiles * 65536.0 / span;
      span_sum += span;
    }
    ms = fms;
    mhz = span_sum / sms / (fms * 1e3);
    return rate / sms;
  };

  // Warm up every kernel once.
  for (auto &k : ks) {
    double mhz, ms;
    launch(k, 63, mhz, ms);
  }

  // The CPU reference for sass_ptxas on four blocks.
  auto reference = [&](uint32_t b) {
    std::vector<int32_t> bs((size_t)256 * K, 0);  // the block's B, summed over its tiles
    for (uint32_t i = 0; i < tiles; i++) {
      const uint32_t cg = (b + i) % ncg;
      for (size_t x = 0; x < (size_t)256 * K; x++) bs[x] += Bl[(size_t)cg * 256 * K + x];
    }
    for (uint32_t tid = 0; tid < 256; tid++) {
      const uint32_t w = tid >> 5, lane = tid & 31, wr = w >> 2, wc = w & 3, g = lane >> 2, tq = lane & 3;
      for (uint32_t i = 0; i < 128; i++) {
        const uint32_t mb = i / 32, nb = (i / 4) % 8, c = i % 4;
        const uint32_t row = wr * 64 + mb * 16 + g + 8 * (c >> 1), col = wc * 64 + nb * 8 + 2 * tq + (c & 1);
        int64_t sum = 0;
        for (uint32_t k = 0; k < K; k++) {
          const uint32_t s = k / 32, st = (s >> 2) & 1, x = (s & 3) * 32 + k % 32;
          sum += (int64_t)A[((size_t)b * 128 + row) * 256 + st * 128 + x] * bs[(size_t)col * K + k];
        }
        ref[((size_t)b * 128 + i) * 256 + tid] = (int32_t)sum;
      }
    }
  };

  // Correctness, on the real loop.
  {
    double mhz, ms;
    launch(ks[2], 63, mhz, ms);
    std::vector<int32_t> base = out;
    const uint32_t check[4] = {0, 1, sms / 2, sms - 1};
    size_t bad = 0;
    for (uint32_t b : check) {
      reference(b);
      for (size_t x = (size_t)b * 32768; x < (size_t)(b + 1) * 32768; x++) bad += base[x] != ref[x];
    }
    printf("[sass] check ptxas vs CPU (blocks 0 1 %u %u): %s (%zu of %u words differ)\n", sms / 2,
           sms - 1, bad ? "FAIL" : "ok", bad, 4 * 32768);
    for (auto &k : ks) {
      if (!k.hand || k.ceiling) continue;
      launch(k, 63, mhz, ms);
      size_t d = 0, first = (size_t)-1;
      for (size_t x = 0; x < outWords; x++)
        if (out[x] != base[x]) {
          if (first == (size_t)-1) first = x;
          d++;
        }
      k.ceiling = d ? -1 : 0;  // -1 marks a kernel that failed its check
      if (d)
        printf("[sass] check %s vs ptxas: FAIL, %zu of %zu words differ, first at block %zu word %zu (%d vs %d)\n",
               k.name, d, outWords, first / 32768, first % 32768, out[first], base[first]);
      else
        printf("[sass] check %s vs ptxas: ok, all %zu words equal\n", k.name, outWords);
    }
  }

  // Timed rounds. b=L2 is the real loop (B from the L2-resident band);
  // b=L1 keeps B to four k-steps a tile, 32 KB an SM (kmask 3; the hand
  // kernels stop advancing B), which prices B's L2 latency. Ceilings load
  // no B.
  struct Cfg {
    int k;
    uint32_t kmask;
    std::vector<double> r, mhz;
  };
  std::vector<Cfg> cfgs;
  for (int i = 0; i < (int)ks.size(); i++) {
    cfgs.push_back({i, 63, {}, {}});
    if (ks[i].ceiling != 1) cfgs.push_back({i, 3, {}, {}});
  }
  for (int r = 0; r < rounds; r++)
    for (auto &c : cfgs) {
      double mhz, ms;
      const double v = launch(ks[c.k], c.kmask, mhz, ms);
      c.r.push_back(v);
      c.mhz.push_back(mhz);
      printf("[sass] run %s b=%s round %d: %.4f IMMA/clk/SM, %.0f MHz, %.2f ms\n", ks[c.k].name,
             c.kmask == 63 ? "L2" : "L1", r + 1, v, mhz, ms);
    }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  double ceil = 0;
  for (auto &c : cfgs)
    if (ks[c.k].ceiling == 1) ceil = std::max(ceil, med(c.r));
  printf("[sass] ceiling %.4f IMMA/clk/SM (the higher of ceiling and hceil; peak is 1.0, 1024 MAC/clk/SM)\n", ceil);
  double best = 0;
  const char *bestName = "none";
  for (auto &c : cfgs) {
    std::vector<double> v = c.r;
    std::sort(v.begin(), v.end());
    const double m = med(c.r);
    printf("[sass] %s b=%s: median %.4f IMMA/clk/SM (%.4f-%.4f) = %.1f%% of ceiling, %.0f MHz%s\n",
           ks[c.k].name, c.kmask == 63 ? "L2" : "L1", m, v.front(), v.back(), 100.0 * m / ceil,
           med(c.mhz), ks[c.k].ceiling == -1 ? ", WRONG RESULTS" : "");
    if (ks[c.k].hand && ks[c.k].ceiling == 0 && c.kmask == 63 && m > best) {
      best = m;
      bestName = ks[c.k].name;
    }
  }
  double pt = 0;
  for (auto &c : cfgs)
    if (!strcmp(ks[c.k].name, "ptxas") && c.kmask == 63) pt = med(c.r);
  printf("[sass] verdict: best correct hand loop %s %.1f%% of ceiling per clock (ptxas %.1f%%): %s\n", bestName,
         100.0 * best / ceil, 100.0 * pt / ceil,
         best / ceil > 0.93 ? "above 93%, worth finishing into the miner" : "not above 93%, stop");
  return 0;
}
