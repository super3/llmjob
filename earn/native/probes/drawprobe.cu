// drawprobe -- the draw fold's own body (src/pearl_fold_draw.cuh), run alone on made-up tables,
// timed in SM cycles and checked against a CPU fold. Our own code, for the nv-kernel work.
//
//   drawprobe <name> <k> <tiles a block> [seconds] [tiles to check]
//
// One block an SM. Each warp stamps the SM's cycle counter around a tile's prologue, its chunk
// loop and its end. Pipe use is 16 cycles an mma, two warps a scheduler, over the cycles the
// chunk loop took; "tile" is the same over the whole tile, prologue and end included. "bad"
// counts the tiles a warp's check failed on (a warp got further ahead of another than the
// stage allows). With tiles to check, the first tiles' transcripts are compared with a fold
// done on the CPU from the same tables.
//
// Build: nvcc -arch=sm_89 -O3 -std=c++17 [-DPD_...] -o drawprobe drawprobe.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define PD_PROBE 1
#define PD_FN pearl_draw_probe
#include "../src/pearl_fold_draw.cuh"

static uint32_t rng_state = 0x12345678u;
static uint32_t rng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
static inline uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "drawprobe name k tiles-a-block [seconds] [tiles to check]\n");
    return 2;
  }
  const char *name = argv[1];
  const uint32_t k = (uint32_t)atoi(argv[2]);
  const uint32_t tpb = (uint32_t)atoi(argv[3]);
  const double secs = argc > 4 ? atof(argv[4]) : 0.0;
  const uint32_t check = argc > 5 ? (uint32_t)atoi(argv[5]) : 0u;
  if (k < 2048u || (k & (k - 1u)) || tpb < 1u) {
    fprintf(stderr, "k is a power of two from 2048; at least one tile a block\n");
    return 2;
  }
  const uint32_t chunks = k / 128u, ksteps = k / 32u;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const uint32_t blocks = (uint32_t)prop.multiProcessorCount;
  // Tiles: RG row groups by CB column block groups, RG * CB = blocks * tpb.
  const uint32_t RG = 8u, tiles = blocks * tpb, CB = (tiles + RG - 1u) / RG;
  const uint32_t MT = PD_MT, BM = 32u * MT, RO = 2u * MT, RPS = 4u * RO, REG = 4u * RPS;
  const uint32_t ga = RG * RO, gb = CB * 16u;        // table words a rank index
  const uint32_t stampBytes = 11u;

  // Plain inputs: the dense factors plus 32 (0 .. 63), a row of 128 for every row of A and
  // column of B, and every k's plus and minus rank index (distinct, as the protocol's are).
  const uint32_t rowsA = ga * 16u, colsB = gb * 16u, kpad = k + 96u;
  std::vector<uint8_t> DA((size_t)rowsA * 128u), DB((size_t)colsB * 128u);
  for (auto &b : DA) b = (uint8_t)(rng() >> 9) & 63u;
  for (auto &b : DB) b = (uint8_t)(rng() >> 9) & 63u;
  std::vector<uint8_t> pm[2][2];   // [B, A][plus, minus][k]
  for (int op = 0; op < 2; op++) {
    pm[op][0].resize(kpad);
    pm[op][1].resize(kpad);
    for (uint32_t kk = 0; kk < kpad; kk++) {
      const uint32_t u = rng() >> 7, p0 = u & 127u, p1 = p0 ^ (1u + ((u >> 8) % 127u));
      pm[op][0][kk] = (uint8_t)p0;
      pm[op][1][kk] = (uint8_t)p1;
    }
  }
  // The kernel's tables, a tile's after the other: [tile][rank index p][w], 16 bytes a word.
  // B's word w holds columns 8w + 0..7 and 128 + 8w + 0..7 of the tile; A's rows 8w + 0..7
  // and 8 (WA + w) + 0..7, WA words a rank index.
  const uint32_t WA = PD_WA;
  std::vector<uint8_t> hTabA((size_t)RG * 128u * WA * 16u), hTabB((size_t)CB * 128u * 256u);
  for (uint32_t t = 0; t < CB; t++)
    for (uint32_t p = 0; p < 128u; p++)
      for (uint32_t w = 0; w < 16u; w++)
        for (uint32_t b = 0; b < 16u; b++) {
          const uint32_t col = t * 256u + (b < 8u ? 8u * w + b : 128u + 8u * w + b - 8u);
          hTabB[(((size_t)t * 128u + p) * 16u + w) * 16u + b] = DB[(size_t)col * 128u + p];
        }
  for (uint32_t t = 0; t < RG; t++)
    for (uint32_t p = 0; p < 128u; p++)
      for (uint32_t w = 0; w < WA; w++)
        for (uint32_t b = 0; b < 16u; b++) {
          const uint32_t row = t * BM + (b < 8u ? 8u * w + b : 8u * (WA + w) + b - 8u);
          hTabA[(((size_t)t * 128u + p) * WA + w) * 16u + b] = DA[(size_t)row * 128u + p];
        }
  // The kernel's offsets: [k-step][B, A][k-pair j] of four: the pair's first k's plus and minus
  // rank index, then its second k's, each times the table's bytes a rank index. Three k-steps
  // of padding at the end (the tile's last k-steps draw stages nothing reads).
  std::vector<uint32_t> hOffs((size_t)(ksteps + 3u) * 32u * 4u);
  for (uint32_t ts = 0; ts < ksteps + 3u; ts++)
    for (uint32_t op = 0; op < 2u; op++)
      for (uint32_t j = 0; j < 16u; j++) {
        uint32_t *o = &hOffs[(((size_t)ts * 2u + op) * 16u + j) * 4u];
        const uint32_t k0 = ts * 32u + 2u * j, S = op ? PD_SA : PD_SB;
        o[0] = pm[op][0][k0] * S;
        o[1] = pm[op][1][k0] * S;
        o[2] = pm[op][0][k0 + 1u] * S;
        o[3] = pm[op][1][k0 + 1u] * S;
      }
  int8_t hStamp[16];
  for (int i = 0; i < 16; i++) hStamp[i] = (int8_t)((rng() >> 11) & 63u);

  uint4 *dTabA, *dTabB, *dOffs;
  int8_t *dStamp;
  unsigned long long *dCyc;
  uint32_t *dDump;
  const size_t cycN = (size_t)blocks * 8u * 5u;
  const uint32_t dumpTiles = check < tiles ? check : tiles;
  cudaMalloc(&dTabA, hTabA.size());
  cudaMalloc(&dTabB, hTabB.size());
  cudaMalloc(&dOffs, hOffs.size() * 4u);
  cudaMalloc(&dStamp, 16);
  cudaMalloc(&dCyc, cycN * 8u);
  cudaMalloc(&dDump, (size_t)(dumpTiles ? dumpTiles : 1u) * REG * 16u * 4u);
  cudaMemcpy(dTabA, hTabA.data(), hTabA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dTabB, hTabB.data(), hTabB.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dOffs, hOffs.data(), hOffs.size() * 4u, cudaMemcpyHostToDevice);
  cudaMemcpy(dStamp, hStamp, 16, cudaMemcpyHostToDevice);
  cudaMemset(dCyc, 0, cycN * 8u);
  cudaFuncSetAttribute(pearl_draw_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_SMEM);

  // PD_CUBIN=<file>: the kernel is launched from that cubin and not from the program. It is a build of this same
  // source with the same switches, its loop reordered by probes/drawtools/spread.py. The driver's own calls are
  // looked up by name, so the probe needs no driver header or library to build.
  typedef int (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                          void **, void **);
  void *cubin_fn = nullptr;
  launch_t cubin_launch = nullptr;
  const char *cb = getenv("PD_CUBIN");
  if (cb && *cb) {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
#define PD_SYM(n) (lib ? (void *)GetProcAddress(lib, n) : nullptr)
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
#define PD_SYM(n) (lib ? dlsym(lib, n) : nullptr)
#endif
    typedef int (*load_t)(void **, const char *);
    typedef int (*get_t)(void **, void *, const char *);
    typedef int (*attr_t)(void *, int, int);
    load_t load = (load_t)PD_SYM("cuModuleLoad");
    get_t get = (get_t)PD_SYM("cuModuleGetFunction");
    attr_t attr = (attr_t)PD_SYM("cuFuncSetAttribute");
    cubin_launch = (launch_t)PD_SYM("cuLaunchKernel");
#undef PD_SYM
    void *mod = nullptr;
    int rc = -1;
    if (!load || !get || !attr || !cubin_launch || (rc = load(&mod, cb)) ||
        (rc = get(&cubin_fn, mod, "pearl_draw_probe")) ||
        (rc = attr(cubin_fn, 8 /* the most dynamic shared memory */, (int)PD_SMEM))) {
      fprintf(stderr, "%s: PD_CUBIN %s: driver error %d\n", name, cb, rc);
      return 1;
    }
  }
  uint32_t a_stamp = stampBytes, a_chunks = chunks, a_coloff = 0u, a_rows = RG * RO, a_cg = CB * 16u,
           a_tiles = tiles, a_dt = dumpTiles;
  void *cubin_args[13] = {&dTabA, &dTabB, &dOffs, &dStamp, &a_stamp, &a_chunks, &a_coloff, &a_rows, &a_cg,
                          &a_tiles, &dCyc, &dDump, &a_dt};

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaEventRecord(e0);
    cudaError_t err;
    if (cubin_fn) {
      err = cubin_launch(cubin_fn, blocks, 1, 1, PD_THREADS, 1, 1, PD_SMEM, nullptr, cubin_args, nullptr)
                ? cudaErrorLaunchFailure
                : cudaSuccess;
    } else {
      pearl_draw_probe<<<blocks, PD_THREADS, PD_SMEM>>>(dTabA, dTabB, dOffs, dStamp, stampBytes, chunks,
                                                       0u, RG * RO, CB * 16u, tiles, dCyc, dDump,
                                                       dumpTiles);
      err = cudaGetLastError();
    }
    cudaEventRecord(e1);
    if (err == cudaSuccess) err = cudaEventSynchronize(e1);
    if (err != cudaSuccess) {
      fprintf(stderr, "%s: %s\n", name, cudaGetErrorString(err));
      return 1;
    }
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
    launches++;
  } while (total < secs * 1000.0);

  std::vector<unsigned long long> cyc(cycN);
  cudaMemcpy(cyc.data(), dCyc, cycN * 8u, cudaMemcpyDeviceToHost);
  double useSum = 0, useMin = 1e9, useMax = 0, tileSum = 0;
  unsigned long long spins = 0, tilesDone = 0;
  int nsched = 0;
  for (uint32_t b = 0; b < blocks; b++) {
    for (uint32_t s = 0; s < 4u; s++) {
      const unsigned long long *w0 = &cyc[((size_t)b * 8u + s) * 5u], *w1 = &cyc[((size_t)b * 8u + s + 4u) * 5u];
      if (!w0[3] || !w1[3]) continue;
      const double imma = (double)(w0[3] + w1[3]) * chunks * 4.0 * PD_NI;
      // The slower warp's cycles: with nothing holding the two together one can run ahead,
      // and the scheduler's pipe is busy until the other is done.
      const double loop = (double)(w0[1] > w1[1] ? w0[1] : w1[1]);
      const double all0 = (double)(w0[0] + w0[1] + w0[2]), all1 = (double)(w1[0] + w1[1] + w1[2]);
      const double all = all0 > all1 ? all0 : all1;
      const double use = 16.0 * imma / loop;
      useSum += use;
      tileSum += 16.0 * imma / all;
      if (use < useMin) useMin = use;
      if (use > useMax) useMax = use;
      spins += w0[4] + w1[4];
      tilesDone += w0[3] + w1[3];
      nsched++;
    }
  }
  const double use = nsched ? useSum / nsched : 0, tuse = nsched ? tileSum / nsched : 0;
  const double ksteps_run = (double)tilesDone * ksteps;
  (void)ksteps_run;
  printf("%-22s k %u pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA tile %6.2f%% bad %llu of %llu %.0f ms x%d",
         name, k, use * 100, useMin * 100, useMax * 100, use > 0 ? 16.0 / use : 0.0, tuse * 100,
         PD_T_STAMP ? 0ull : spins, tilesDone, ms, launches);

  if (dumpTiles) {
    // The CPU's fold of the same tiles, from the same tables.
    std::vector<uint32_t> dump((size_t)dumpTiles * REG * 16u);
    cudaMemcpy(dump.data(), dDump, dump.size() * 4u, cudaMemcpyDeviceToHost);
    uint32_t bad = 0, total_regions = 0, badro[16] = {0};
    std::vector<int8_t> Ap((size_t)BM * 128), Bp((size_t)256 * 128);
    std::vector<int32_t> C((size_t)BM * 256);
    std::vector<uint32_t> tr((size_t)REG * 16);
    for (uint32_t v = 0; v < dumpTiles; v++) {
      const uint32_t rbg = v % RG, cbg = v / RG;
      std::fill(C.begin(), C.end(), 0);
      std::fill(tr.begin(), tr.end(), 0u);
      for (uint32_t ch = 0; ch < chunks; ch++) {
        for (uint32_t kk = 0; kk < 128u; kk++) {
          const uint32_t gk = ch * 128u + kk;
          const uint32_t bp0 = pm[0][0][gk], bp1 = pm[0][1][gk], ap0 = pm[1][0][gk], ap1 = pm[1][1][gk];
          for (uint32_t r = 0; r < BM; r++) {
            const uint32_t gr = rbg * BM + r;
            const uint8_t *T = &DA[(size_t)gr * 128u];
            int val = PD_FILL + (int)T[ap0] - (int)T[ap1];
            if (gr == 0u && gk < stampBytes) {
              val = val - PD_FILL + (int)hStamp[gk];
              val = val < -128 ? -128 : (val > 127 ? 127 : val);
            }
            Ap[(size_t)r * 128u + kk] = (int8_t)val;
          }
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const uint32_t gc = cbg * 256u + cidx;
            const uint8_t *T = &DB[(size_t)gc * 128u];
            Bp[(size_t)cidx * 128u + kk] = (int8_t)(PD_FILL + (int)T[bp0] - (int)T[bp1]);
          }
        }
        for (uint32_t r = 0; r < BM; r++)
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const int8_t *a = &Ap[(size_t)r * 128u], *bb = &Bp[(size_t)cidx * 128u];
            int32_t sum = 0;
            for (uint32_t kk = 0; kk < 128u; kk++) sum += (int32_t)a[kk] * (int32_t)bb[kk];
            C[(size_t)r * 256u + cidx] += sum;
          }
        for (uint32_t wc = 0; wc < 4u; wc++)
          for (uint32_t Lc = 0; Lc < RPS; Lc++) {
            // Region Lc of the slot: column offset tq, row offset ro (32-row group ro / 2, half ro & 1).
            const uint32_t tq = Lc / RO, ro = Lc % RO, grp = ro / 2u, rh = ro & 1u;
            uint32_t x = 0;
            for (uint32_t q = 0; q < 4u; q++)
              for (uint32_t jj = 0; jj < 4u; jj++)
                for (uint32_t e = 0; e < 2u; e++)
                  for (uint32_t ii = 0; ii < 8u; ii++) {
                    const uint32_t r = grp * 32u + 4u * rh + q + 8u * jj;
                    const uint32_t cidx = wc * 64u + 2u * tq + e + 8u * ii;
                    x ^= (uint32_t)C[(size_t)r * 256u + cidx];
                  }
            uint32_t &w = tr[(size_t)(wc * RPS + Lc) * 16u + (ch & 15u)];
            w = rotl13(w) ^ x;
          }
      }
      for (uint32_t reg = 0; reg < REG; reg++) {
        total_regions++;
        if (memcmp(&tr[(size_t)reg * 16u], &dump[(size_t)v * REG * 16u + reg * 16u], 64) != 0) {
          if (bad < 4u)
            fprintf(stderr, "tile %u region %u: cpu %08x %08x.. gpu %08x %08x..\n", v, reg, tr[reg * 16u],
                    tr[reg * 16u + 1u], dump[(size_t)v * REG * 16u + reg * 16u],
                    dump[(size_t)v * REG * 16u + reg * 16u + 1u]);
          bad++;
          badro[(reg % RPS) % RO]++;
        }
      }
    }
    printf(" check %u/%u", total_regions - bad, total_regions);
    if (bad) {
      // Which row offsets of the tile the bad regions are on.
      printf(" bad by row offset:");
      for (uint32_t ro = 0; ro < RO; ro++) printf(" %u", badro[ro]);
    }
  }
#if PD_T_STAMP
  {
    // The cycles a warp spent at a k-step's barrier, a loop copy (warps 0-3, warps 4-7), a k-step.
    double w[2] = {0, 0}, n = 0;
    for (uint32_t b = 0; b < blocks; b++)
      for (uint32_t wp = 0; wp < 8u; wp++) {
        w[wp >> 2] += (double)cyc[((size_t)b * 8u + wp) * 5u + 4u];
        n += (double)cyc[((size_t)b * 8u + wp) * 5u + 3u] * ksteps;
      }
    printf(" | at the barrier, cycles a k-step: warps 0-3 %.1f, warps 4-7 %.1f", w[0] / (n / 2.0), w[1] / (n / 2.0));
  }
#endif
  printf("\n");
  return 0;
}
