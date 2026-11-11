// cprobe -- the harness of the copy fold (hand89/cfold_h.cuh): it noises A and B on the CPU, writes A' and B' in
// the stage's own byte layout, runs the kernel (PD_CUBIN=<file> runs the cubin's, which is how a loop written by
// hand89/cfold.py is launched), prints how much of the tensor pipe the chunk loop holds, and checks the first
// tiles' transcripts against a CPU fold of the same plain inputs. Our own code, for the nv-kernel work.
//
//   cprobe <name> <k> <tiles a block> [seconds] [tiles to check] [row groups] [column blocks]
//
// Row groups and column blocks set how much operand memory the copies walk and how many blocks share a byte:
// tile v folds row group v % rg against column block (v / rg) % cb, so with 128 blocks, 8 row groups and 16
// column blocks (the defaults, and the release's band) every block has its own pair, 16 blocks read each byte
// of A' and 8 each byte of B'. A' is rg * (k / 32 + 8) * 1024 CP_MT bytes and B' cb * (k / 32 + 8) * 8192 (printed).
//
// Build: nvcc -arch=sm_89 -O3 -std=c++17 -o cprobe cprobe.cu   (and -cubin for the file cfold.py patches)
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

#define CP_FN pearl_copy_probe
#include "cfold_h.cuh"

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
    fprintf(stderr, "cprobe name k tiles-a-block [seconds] [tiles to check] [row groups] [column blocks]\n");
    return 2;
  }
  const char *name = argv[1];
  const uint32_t k = (uint32_t)atoi(argv[2]);
  const uint32_t tpb = (uint32_t)atoi(argv[3]);
  const double secs = argc > 4 ? atof(argv[4]) : 0.0;
  const uint32_t check = argc > 5 ? (uint32_t)atoi(argv[5]) : 0u;
  const uint32_t RG = argc > 6 ? (uint32_t)atoi(argv[6]) : 8u;
  const uint32_t CB = argc > 7 ? (uint32_t)atoi(argv[7]) : 16u;
  if (k < 2048u || (k & (k - 1u)) || tpb < 1u || RG < 1u || CB < 1u) {
    fprintf(stderr, "k is a power of two from 2048; at least one tile a block, one row group, one column block\n");
    return 2;
  }
  const uint32_t chunks = k / 128u, ksteps = k / 32u, ksAll = ksteps + CP_PADK;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const uint32_t blocks = (uint32_t)prop.multiProcessorCount;
  const uint32_t tiles = blocks * tpb;
  const uint32_t MT = CP_MT, BM = 32u * MT, RO = 2u * MT, RPS = 4u * RO, REG = 4u * RPS, GA = CP_GA;
  const int FILL = 63;

  // Plain inputs: the dense factors plus 32 (0 .. 63), a row of 128 for every row of A and column of B, and
  // every k's plus and minus rank index (distinct, as the protocol's are). pm[B, A][plus, minus][k].
  const uint32_t rowsA = RG * BM, colsB = CB * 256u;
  std::vector<uint8_t> DA((size_t)rowsA * 128u), DB((size_t)colsB * 128u);
  for (auto &b : DA) b = (uint8_t)(rng() >> 9) & 63u;
  for (auto &b : DB) b = (uint8_t)(rng() >> 9) & 63u;
  std::vector<uint8_t> pm[2][2];
  for (int op = 0; op < 2; op++) {
    pm[op][0].resize(k);
    pm[op][1].resize(k);
    for (uint32_t kk = 0; kk < k; kk++) {
      const uint32_t u = rng() >> 7, p0 = u & 127u, p1 = p0 ^ (1u + ((u >> 8) % 127u));
      pm[op][0][kk] = (uint8_t)p0;
      pm[op][1][kk] = (uint8_t)p1;
    }
  }
  // A' and B', noised here, in the stage's layout: [group of rows or block][k-step][k-pair j][group G] of 16
  // bytes, byte 2c row (or column) 8G + c at k-step * 32 + 2j, byte 2c + 1 at the next k.
  const size_t recA = CP_RECA, recB = CP_RECB;
  std::vector<uint8_t> hA((size_t)RG * ksAll * recA, 0), hB((size_t)CB * ksAll * recB, 0);
  for (uint32_t t = 0; t < RG; t++)
    for (uint32_t G = 0; G < GA; G++)
      for (uint32_t c = 0; c < 8u; c++) {
        const uint8_t *T = &DA[(size_t)(t * BM + 8u * G + c) * 128u];
        uint8_t *base = &hA[(size_t)t * ksAll * recA + (size_t)G * 16u + 2u * c];
        for (uint32_t kk = 0; kk < k; kk++)
          base[(size_t)(kk >> 5) * recA + (size_t)((kk & 31u) >> 1) * (16u * GA) + (kk & 1u)] =
              (uint8_t)(FILL + (int)T[pm[1][0][kk]] - (int)T[pm[1][1][kk]]);
      }
  for (uint32_t t = 0; t < CB; t++)
    for (uint32_t G = 0; G < 32u; G++)
      for (uint32_t c = 0; c < 8u; c++) {
        const uint8_t *T = &DB[(size_t)(t * 256u + 8u * G + c) * 128u];
        uint8_t *base = &hB[(size_t)t * ksAll * recB + (size_t)G * 16u + 2u * c];
        for (uint32_t kk = 0; kk < k; kk++)
          base[(size_t)(kk >> 5) * recB + (size_t)((kk & 31u) >> 1) * 512u + (kk & 1u)] =
              (uint8_t)(FILL + (int)T[pm[0][0][kk]] - (int)T[pm[0][1][kk]]);
      }

  uint4 *dA, *dB;
  unsigned long long *dCyc;
  uint32_t *dDump;
  const size_t cycN = (size_t)blocks * 8u * 4u;
  uint32_t dumpTiles = check < tiles ? check : tiles;
  // The hand-written loop moves its source pointers with 32-bit adds: no table may cross a multiple of 4 GiB.
  auto alloc_clear = [&](uint4 **p, size_t bytes) {
    for (int tries = 0; tries < 8; tries++) {
      if (cudaMalloc(p, bytes) != cudaSuccess) return false;
      const unsigned long long lo = (unsigned long long)(uintptr_t)*p & 0xffffffffull;
      if (lo >= 0x100000ull && lo + bytes + 0x100000ull < 0x100000000ull) return true;
    }
    return false;
  };
  if (!alloc_clear(&dA, hA.size() + 65536u) || !alloc_clear(&dB, hB.size() + 65536u)) {
    fprintf(stderr, "%s: no operand table clear of a 4 GiB line\n", name);
    return 1;
  }
  cudaMalloc(&dCyc, cycN * 8u);
  cudaMalloc(&dDump, (size_t)(dumpTiles ? dumpTiles : 1u) * REG * 16u * 4u);
  cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice);
  cudaMemset(dCyc, 0, cycN * 8u);
  const double mbA = hA.size() / 1048576.0, mbB = hB.size() / 1048576.0;
  // The CPU check needs neither table again.
  std::vector<uint8_t>().swap(hA);
  std::vector<uint8_t>().swap(hB);
  cudaFuncSetAttribute(pearl_copy_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)CP_SMEM);

  // PD_CUBIN=<file>: the kernel is launched from that cubin and not from the program. The driver's own calls
  // are looked up by name, so the probe needs no driver header or library to build.
  typedef int (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                          void **, void **);
  void *cubin_fn = nullptr;
  launch_t cubin_launch = nullptr;
  const char *cbn = getenv("PD_CUBIN");
  if (cbn && *cbn) {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
#define CP_SYM(n) (lib ? (void *)GetProcAddress(lib, n) : nullptr)
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
#define CP_SYM(n) (lib ? dlsym(lib, n) : nullptr)
#endif
    typedef int (*load_t)(void **, const char *);
    typedef int (*get_t)(void **, void *, const char *);
    typedef int (*attr_t)(void *, int, int);
    load_t load = (load_t)CP_SYM("cuModuleLoad");
    get_t get = (get_t)CP_SYM("cuModuleGetFunction");
    attr_t attr = (attr_t)CP_SYM("cuFuncSetAttribute");
    cubin_launch = (launch_t)CP_SYM("cuLaunchKernel");
#undef CP_SYM
    void *mod = nullptr;
    int rc = -1;
    if (!load || !get || !attr || !cubin_launch || (rc = load(&mod, cbn)) ||
        (rc = get(&cubin_fn, mod, "pearl_copy_probe")) ||
        (rc = attr(cubin_fn, 8 /* the most dynamic shared memory */, (int)CP_SMEM))) {
      fprintf(stderr, "%s: PD_CUBIN %s: driver error %d\n", name, cbn, rc);
      return 1;
    }
  }
  uint32_t a_chunks = chunks, a_rg = RG, a_cb = CB, a_tiles = tiles;
  void *cubin_args[9] = {&dA, &dB, &a_chunks, &a_rg, &a_cb, &a_tiles, &dCyc, &dDump, &dumpTiles};

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaEventRecord(e0);
    cudaError_t err;
    if (cubin_fn) {
      err = cubin_launch(cubin_fn, blocks, 1, 1, CP_THREADS, 1, 1, CP_SMEM, nullptr, cubin_args, nullptr)
                ? cudaErrorLaunchFailure
                : cudaSuccess;
    } else {
      pearl_copy_probe<<<blocks, CP_THREADS, CP_SMEM>>>(dA, dB, chunks, RG, CB, tiles, dCyc, dDump, dumpTiles);
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

  // Pipe use, a scheduler: its two warps' mma against the slower one's loop cycles (16 cycles an mma is all
  // of the pipe), as hprobe counts it.
  std::vector<unsigned long long> cyc(cycN);
  cudaMemcpy(cyc.data(), dCyc, cycN * 8u, cudaMemcpyDeviceToHost);
  double useSum = 0, useMin = 1e9, useMax = 0, tileSum = 0, g[2] = {0, 0}, gn[2] = {0, 0};
  unsigned long long tilesDone = 0;
  int nsched = 0;
  for (uint32_t b = 0; b < blocks; b++)
    for (uint32_t s = 0; s < 4u; s++) {
      const unsigned long long *w0 = &cyc[((size_t)b * 8u + s) * 4u], *w1 = &cyc[((size_t)b * 8u + s + 4u) * 4u];
      if (!w0[3] || !w1[3]) continue;
      const double imma = (double)(w0[3] + w1[3]) * chunks * 4.0 * CP_NI;
      const double loop = (double)(w0[1] > w1[1] ? w0[1] : w1[1]);
      const double all0 = (double)(w0[0] + w0[1] + w0[2]), all1 = (double)(w1[0] + w1[1] + w1[2]);
      const double use = 16.0 * imma / loop;
      useSum += use;
      tileSum += 16.0 * imma / (all0 > all1 ? all0 : all1);
      if (use < useMin) useMin = use;
      if (use > useMax) useMax = use;
      tilesDone += w0[3] + w1[3];
      g[0] += (double)w0[1];
      g[1] += (double)w1[1];
      gn[0] += (double)w0[3];
      gn[1] += (double)w1[3];
      nsched++;
    }
  const double use = nsched ? useSum / nsched : 0, tuse = nsched ? tileSum / nsched : 0;
  printf("%-22s k %u mt %u rg %u cb %u A' %.1f MB B' %.1f MB pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA tile %6.2f%% "
         "tiles %llu %.0f ms x%d",
         name, k, MT, RG, CB, mbA, mbB, use * 100, useMin * 100, useMax * 100, use > 0 ? 16.0 / use : 0.0,
         tuse * 100, tilesDone / 8ull, ms, launches);
  if (getenv("PD_GROUPS"))
    printf(" | loop cycles a tile: warps 0-3 %.1f, warps 4-7 %.1f", gn[0] > 0 ? g[0] / gn[0] : 0.0,
           gn[1] > 0 ? g[1] / gn[1] : 0.0);

  if (dumpTiles) {
    // The CPU's fold of the same tiles from the plain inputs.
    std::vector<uint32_t> dump((size_t)dumpTiles * REG * 16u);
    cudaMemcpy(dump.data(), dDump, dump.size() * 4u, cudaMemcpyDeviceToHost);
    uint32_t bad = 0, total_regions = 0, bad_tiles = 0, first_bad = 0xffffffffu, badro[16] = {0};
    std::vector<int8_t> Apc((size_t)BM * 128), Bpc((size_t)256 * 128);
    std::vector<int32_t> C((size_t)BM * 256);
    std::vector<uint32_t> tr((size_t)REG * 16);
    for (uint32_t v = 0; v < dumpTiles; v++) {
      const uint32_t rbg = v % RG, cbg = (v / RG) % CB;
      std::fill(C.begin(), C.end(), 0);
      std::fill(tr.begin(), tr.end(), 0u);
      for (uint32_t ch = 0; ch < chunks; ch++) {
        for (uint32_t kk = 0; kk < 128u; kk++) {
          const uint32_t gk = ch * 128u + kk;
          const uint32_t bp0 = pm[0][0][gk], bp1 = pm[0][1][gk], ap0 = pm[1][0][gk], ap1 = pm[1][1][gk];
          for (uint32_t r = 0; r < BM; r++) {
            const uint8_t *T = &DA[(size_t)(rbg * BM + r) * 128u];
            Apc[(size_t)r * 128u + kk] = (int8_t)(FILL + (int)T[ap0] - (int)T[ap1]);
          }
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const uint8_t *T = &DB[(size_t)(cbg * 256u + cidx) * 128u];
            Bpc[(size_t)cidx * 128u + kk] = (int8_t)(FILL + (int)T[bp0] - (int)T[bp1]);
          }
        }
        for (uint32_t r = 0; r < BM; r++)
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const int8_t *a = &Apc[(size_t)r * 128u], *bb = &Bpc[(size_t)cidx * 128u];
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
      uint32_t bad_here = 0;
      for (uint32_t reg = 0; reg < REG; reg++) {
        total_regions++;
        if (memcmp(&tr[(size_t)reg * 16u], &dump[(size_t)v * REG * 16u + reg * 16u], 64) != 0) {
          if (bad < 4u)
            fprintf(stderr, "tile %u region %u: cpu %08x %08x.. gpu %08x %08x..\n", v, reg, tr[reg * 16u],
                    tr[reg * 16u + 1u], dump[(size_t)v * REG * 16u + reg * 16u],
                    dump[(size_t)v * REG * 16u + reg * 16u + 1u]);
          bad++;
          bad_here++;
          badro[(reg % RPS) % RO]++;
        }
      }
      if (bad_here) {
        bad_tiles++;
        if (first_bad == 0xffffffffu) first_bad = v;
      }
    }
    printf(" | check %u/%u", total_regions - bad, total_regions);
    if (bad) {
      printf(" (%u tiles with a bad region, the first tile %u; by row offset:", bad_tiles, first_bad);
      for (uint32_t ro = 0; ro < RO; ro++) printf(" %u", badro[ro]);
      printf(")");
    }
  }
  printf("\n");
  return 0;
}
