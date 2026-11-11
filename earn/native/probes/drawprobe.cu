// drawprobe -- the draw fold's own body (src/pearl_fold_draw.cuh), run alone on made-up tables,
// timed in SM cycles and checked against a CPU fold. Our own code, for the nv-kernel work.
//
//   drawprobe <name> <k> <tiles a block> [seconds] [tiles to check]
//
// One block an SM. Each warp stamps the SM's cycle counter around a tile's prologue, its chunk
// loop and its end. Pipe use is 16 cycles an mma, two warps a scheduler, over the cycles the
// chunk loop took; "tile" is the same over the whole tile, prologue and end included. "spins"
// counts the times a warp found the next stage not drawn yet. With tiles to check, the first
// tiles' transcripts are compared with a fold done on the CPU from the same tables.
//
// Build: nvcc -arch=sm_89 -O3 -std=c++17 [-DPD_...] -o drawprobe drawprobe.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

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
  const uint32_t ga = RG * 12u, gb = CB * 16u;       // groups of sixteen rows / columns
  const uint32_t stampBytes = 11u;

  // Plain inputs: the dense factors plus 32 (0 .. 63), a row of 128 for every row of A and
  // column of B, and every k's plus and minus rank index (distinct, as the protocol's are).
  const uint32_t rowsA = ga * 16u, colsB = gb * 16u, kpad = k + 32u;
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
  // The kernel's tables: word (quad q, e, p) holds rows 32q + 8e + 0..7 and 32q + 8e + 16 + 0..7.
  std::vector<uint8_t> hTabA((size_t)ga * 2048u), hTabB((size_t)gb * 2048u);
  for (int op = 0; op < 2; op++) {
    std::vector<uint8_t> &T = op ? hTabA : hTabB;
    const std::vector<uint8_t> &D = op ? DA : DB;
    const uint32_t words = (op ? ga : gb) * 128u;
    for (uint32_t w = 0; w < words; w++) {
      const uint32_t P = w / 128u, pi = w % 128u, q = P / 2u, e = P & 1u;
      for (uint32_t i = 0; i < 16u; i++) {
        const uint32_t row = 32u * q + 8u * e + (i < 8u ? i : 16u + i - 8u);
        T[(size_t)w * 16u + i] = D[(size_t)row * 128u + pi];
      }
    }
  }
  // The kernel's offsets: [k-step][B, A][lane] of four, each a rank index times 16 plus 2048 e.
  // B's lane L draws e = L >> 4, k-pair L & 15; A's draws e = (L >> 3) & 1, k-pair
  // 8 (L >> 4) + (L & 7). One k-step of padding at the end (the tile's last k-step draws a
  // stage nothing reads).
  std::vector<uint32_t> hOffs((size_t)(ksteps + 1u) * 64u * 4u);
  for (uint32_t ts = 0; ts <= ksteps; ts++)
    for (uint32_t op = 0; op < 2u; op++)
      for (uint32_t L = 0; L < 32u; L++) {
        const uint32_t e = op ? (L >> 3) & 1u : L >> 4, kp = op ? 8u * (L >> 4) + (L & 7u) : L & 15u;
        uint32_t *o = &hOffs[(((size_t)ts * 2u + op) * 32u + L) * 4u];
        const uint32_t k0 = ts * 32u + 2u * kp;
        o[0] = pm[op][0][k0] * 16u + e * 2048u;
        o[1] = pm[op][1][k0] * 16u + e * 2048u;
        o[2] = pm[op][0][k0 + 1u] * 16u + e * 2048u;
        o[3] = pm[op][1][k0 + 1u] * 16u + e * 2048u;
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
  cudaMalloc(&dDump, (size_t)(dumpTiles ? dumpTiles : 1u) * 3072u * 4u);
  cudaMemcpy(dTabA, hTabA.data(), hTabA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dTabB, hTabB.data(), hTabB.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dOffs, hOffs.data(), hOffs.size() * 4u, cudaMemcpyHostToDevice);
  cudaMemcpy(dStamp, hStamp, 16, cudaMemcpyHostToDevice);
  cudaMemset(dCyc, 0, cycN * 8u);
  cudaFuncSetAttribute(pearl_draw_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_SMEM);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaEventRecord(e0);
    pearl_draw_probe<<<blocks, PD_THREADS, PD_SMEM>>>(dTabA, dTabB, dOffs, dStamp, stampBytes, chunks,
                                                     0u, RG * 12u, CB * 16u, tiles, dCyc, dDump,
                                                     dumpTiles);
    cudaError_t err = cudaGetLastError();
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
      const double imma = (double)(w0[3] + w1[3]) * chunks * 192.0;
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
  printf("%-22s k %u pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA tile %6.2f%% spins %.4f/k-step %.0f ms x%d",
         name, k, use * 100, useMin * 100, useMax * 100, use > 0 ? 16.0 / use : 0.0, tuse * 100,
         ksteps_run > 0 ? (double)spins / ksteps_run : 0.0, ms, launches);

  if (dumpTiles) {
    // The CPU's fold of the same tiles, from the same tables.
    std::vector<uint32_t> dump((size_t)dumpTiles * 3072u);
    cudaMemcpy(dump.data(), dDump, dump.size() * 4u, cudaMemcpyDeviceToHost);
    uint32_t bad = 0, total_regions = 0;
    std::vector<int8_t> Ap((size_t)192 * 128), Bp((size_t)256 * 128);
    std::vector<int32_t> C((size_t)192 * 256);
    std::vector<uint32_t> tr((size_t)192 * 16);
    for (uint32_t v = 0; v < dumpTiles; v++) {
      const uint32_t rbg = v % RG, cbg = v / RG;
      std::fill(C.begin(), C.end(), 0);
      std::fill(tr.begin(), tr.end(), 0u);
      for (uint32_t ch = 0; ch < chunks; ch++) {
        for (uint32_t kk = 0; kk < 128u; kk++) {
          const uint32_t gk = ch * 128u + kk;
          const uint32_t bp0 = pm[0][0][gk], bp1 = pm[0][1][gk], ap0 = pm[1][0][gk], ap1 = pm[1][1][gk];
          for (uint32_t r = 0; r < 192u; r++) {
            const uint32_t gr = rbg * 192u + r;
            const uint8_t *T = &DA[(size_t)gr * 128u];
            int val = 64 + (int)T[ap0] - (int)T[ap1];
            if (gr == 0u && gk < stampBytes) {
              val = val - 64 + (int)hStamp[gk];
              val = val < -128 ? -128 : (val > 127 ? 127 : val);
            }
            Ap[(size_t)r * 128u + kk] = (int8_t)val;
          }
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const uint32_t gc = cbg * 256u + cidx;
            const uint8_t *T = &DB[(size_t)gc * 128u];
            Bp[(size_t)cidx * 128u + kk] = (int8_t)(64 + (int)T[bp0] - (int)T[bp1]);
          }
        }
        for (uint32_t r = 0; r < 192u; r++)
          for (uint32_t cidx = 0; cidx < 256u; cidx++) {
            const int8_t *a = &Ap[(size_t)r * 128u], *bb = &Bp[(size_t)cidx * 128u];
            int32_t sum = 0;
            for (uint32_t kk = 0; kk < 128u; kk++) sum += (int32_t)a[kk] * (int32_t)bb[kk];
            C[(size_t)r * 256u + cidx] += sum;
          }
        for (uint32_t wc = 0; wc < 4u; wc++)
          for (uint32_t Lc = 0; Lc < 48u; Lc++) {
            const uint32_t wr = Lc / 24u, rem = Lc % 24u, tq = rem / 6u, rl = (rem % 6u) / 2u, rh = rem & 1u;
            uint32_t x = 0;
            for (uint32_t q = 0; q < 4u; q++)
              for (uint32_t jj = 0; jj < 4u; jj++)
                for (uint32_t e = 0; e < 2u; e++)
                  for (uint32_t ii = 0; ii < 8u; ii++) {
                    const uint32_t r = wr * 96u + rl * 32u + 4u * rh + q + 8u * jj;
                    const uint32_t cidx = wc * 64u + 2u * tq + e + 8u * ii;
                    x ^= (uint32_t)C[(size_t)r * 256u + cidx];
                  }
            uint32_t &w = tr[(size_t)(wc * 48u + Lc) * 16u + (ch & 15u)];
            w = rotl13(w) ^ x;
          }
      }
      for (uint32_t reg = 0; reg < 192u; reg++) {
        total_regions++;
        if (memcmp(&tr[(size_t)reg * 16u], &dump[(size_t)v * 3072u + reg * 16u], 64) != 0) {
          if (bad < 4u)
            fprintf(stderr, "tile %u region %u: cpu %08x %08x.. gpu %08x %08x..\n", v, reg, tr[reg * 16u],
                    tr[reg * 16u + 1u], dump[(size_t)v * 3072u + reg * 16u],
                    dump[(size_t)v * 3072u + reg * 16u + 1u]);
          bad++;
        }
      }
    }
    printf(" check %u/%u", total_regions - bad, total_regions);
  }
  printf("\n");
  return 0;
}
