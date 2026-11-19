// powcap.cu (goal-nv50w): which kind of work trips a capped card's "SW Power Cap", and at what board power.
//
// On a 575 W RTX 5090 our fold sits at 539 W under "SW Power Cap", a timing-only variant of it at 510 W at the same
// clock, and SRBMiner at 571 W. So the board total is not what binds; some part of it is. Each mode here is one
// kind of work alone, at full occupancy, for a few seconds, while the program reads NVML every 100 ms: board power,
// SM clock and the clock-event reasons. A mode that is held by the cap at a low board power loads the part that
// binds. Plain observation; nothing is set on the card.
//
//   nvcc -arch=sm_120 -O3 -o powcap powcap.cu -ldl && ./powcap <seconds a mode> [mode ...]
//
// Modes: lds (LDS.128, no bank conflicts), ldsm (ldmatrix x4), ffma (FP32 FMA chains), idle, and IMMA (mma m16n8k32
// s8) in wimma's variants: imma / immas / immam / immax random operands, A held / a snake / B held / both changing;
// immaz zero operands; immah / immahs / immahm bytes 0-127 (as ours) A held / snake / B held; mix4 / mix8 / mix16 one
// LDS.128 every 4 / 8 / 16 IMMA feeding the operands (bytes 0-127), mixs8 / mixm8 mix8 as a snake / B held;
// ldgl2 / ldgdram 16-byte global loads from a 32 MB buffer (in a 5090's L2) / a 1 GB one (from DRAM); fst / fgl2 /
// fgdr / fnoa (goal-rtx50c) the fold's A side from the stage / an L2 stream / DRAM / nowhere (see wfold). Run
// one mode a process: a fault ends the CUDA context. A line a mode: work a second, mean board watts, mean SM MHz, and the share of samples with each reason bit
// (b2 = SW power cap, b3 = HW slowdown, b5 = SW thermal, b6 = HW thermal, b7 = HW power brake, b8 = display clock).
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define THREADS 256
#define ITERS 4096

__device__ __forceinline__ void mma8(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// mode: 0 lds, 1 ldsm, 4 ffma (the IMMA modes are wimma, below)
__global__ void __launch_bounds__(THREADS) work(int mode, int, uint32_t seed, unsigned long long *sink) {
  __shared__ __align__(16) uint32_t sm[THREADS * 4 * 4];
  const int t = threadIdx.x;
  uint32_t x = seed ^ (t * 2654435761u) ^ (blockIdx.x * 40503u);
  for (int i = t; i < THREADS * 16; i += THREADS) {
    x = x * 1664525u + 1013904223u;
    sm[i] = x;
  }
  __syncthreads();
  unsigned long long acc = 0;
  if (mode == 0) {
    uint4 r = make_uint4(0, 0, 0, 0);
    const uint4 *p = reinterpret_cast<const uint4 *>(sm);
    int idx = t;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        uint4 v = p[(idx + j * 32) & (THREADS * 4 - 1)];
        r.x ^= v.x; r.y += v.y; r.z ^= v.z; r.w += v.w;
      }
      idx += r.x & 32;
    }
    acc = r.x + r.y + r.z + r.w;
  } else if (mode == 1) {
    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    // Each lane names one 16-byte row; every address stays inside the 16 KB array.
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + (t & 31) * 16;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        uint32_t a0, a1, a2, a3;
        uint32_t ad = base + ((j * 1024 + (r0 & 512)) & 0x3C00);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(ad & ~15u));
        r0 ^= a0; r1 += a1; r2 ^= a2; r3 += a3;
      }
    }
    acc = r0 + r1 + r2 + r3;
  } else if (mode == 4) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)(x >> (i + 1)) * 1e-9f;
    const float m = 0.9999f, c = 1e-7f;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) f[j] = fmaf(f[j], m, c);
    }
    for (int i = 0; i < 16; i++) acc += (unsigned)__float_as_uint(f[i]);
  }
  if (acc == 0x123456789ull) sink[0] = acc;  // never true; keeps the work live
}

// IMMA alone, 6 A tiles by 4 B tiles a step, in order ORD: 0 B inner (A held for 4 IMMA, as our fold's loop holds
// it for 8), 1 the same but every other row of B backwards (a snake: one operand changes at every IMMA), 2 A inner
// (B held for 6), 3 both change at every IMMA. `mask` keeps the operand bits (0 zeroes them; 0x7f7f7f7f gives
// bytes 0-127, as our operands are 0-126). MIX > 0: one LDS.128 every MIX IMMA, whose data XORs into the operands.
// KIND (with MIX): where the fed-in 16 bytes come from. 0 LDS.128 (shared memory); 1 LDG.128 of a 16 KB buffer
// (stays in the SM's L1); 2 LDG.128 of a 32 MB buffer with .cg (the L2, as the fold's record stream partly is);
// 3 a read-modify-write of a word of a 16-word local array (LDL/STL, as the fold's transcript word); 4 the same
// in shared memory (LDS/STS).
template <int ORD, int MIX, int KIND = 0>
__global__ void __launch_bounds__(THREADS) wimma(uint32_t mask, uint32_t seed, unsigned long long *sink,
                                                 const uint4 *__restrict__ g = nullptr) {
  __shared__ __align__(16) uint32_t sm[THREADS * 16];
  const int t = threadIdx.x;
  uint32_t loc[16];
  for (int i = 0; i < 16; i++) loc[i] = seed + i * t;
  uint32_t gi = (blockIdx.x * THREADS + t) * 7u;
  uint32_t x = seed ^ (t * 2654435761u) ^ (blockIdx.x * 40503u);
  for (int i = t; i < THREADS * 16; i += THREADS) {
    x = x * 1664525u + 1013904223u;
    sm[i] = x & mask;
  }
  __syncthreads();
  unsigned a[6][4], b[4][2];
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 4; j++) { x = x * 1664525u + 1013904223u; a[i][j] = x & mask; }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) { x = x * 1664525u + 1013904223u; b[i][j] = x & mask; }
  int d[6][4][4];
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++) d[i][j][k] = 0;
  const uint4 *p = reinterpret_cast<const uint4 *>(sm);
  int idx = t;
#pragma unroll 1
  for (int it = 0; it < ITERS / 4; it++) {
#pragma unroll
    for (int s = 0; s < 24; s++) {
      int i, j;
      if (ORD == 0) { i = s / 4; j = s % 4; }
      else if (ORD == 1) { i = s / 4; j = (i & 1) ? 3 - s % 4 : s % 4; }
      else if (ORD == 2) { i = s % 6; j = s / 6; }
      else { i = s % 6; j = (s + s / 12) % 4; }  // both change each step, every (i, j) once
      mma8(d[i][j], a[i], b[j]);
      if (MIX > 0 && s % MIX == MIX - 1) {
        uint4 v;
        if (KIND == 0) {
          v = p[(idx + s * 32) & (THREADS * 4 - 1)];
        } else if (KIND == 1 || KIND == 2) {
          // The next address hangs on the data, so the loads cannot be hoisted or merged.
          const uint32_t words = KIND == 1 ? 1024u : (32u << 20) / 16u;
          v = KIND == 1 ? __ldg(g + (gi & (words - 1))) : __ldcg(g + (gi & (words - 1)));
          gi += 32u * 7u + (v.x & 32u);
        } else if (KIND == 3) {
          const int w = (idx >> 5) & 15;
          loc[w] ^= a[i][0];
          v = make_uint4(loc[(w + 5) & 15], 0u, 0u, 0u);
        } else {
          uint32_t *q = sm + t * 16 + ((idx >> 5) & 15);
          *q ^= a[i][0];
          v = make_uint4(sm[t * 16 + ((idx >> 5) + 5 & 15)], 0u, 0u, 0u);
        }
        a[i][0] ^= v.x & mask; a[i][1] ^= v.y & mask; b[j][0] ^= v.z & mask; b[j][1] ^= v.w & mask;
      }
    }
    idx += 32;
  }
  unsigned long long acc = 0;
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++) acc += (unsigned)d[i][j][k];
  if (acc == 0x123456789ull) sink[0] = acc;
}

// wfold (goal-rtx50c): the fold's A side priced under the cap, the lead's "A from global memory" against the stage.
// One block an SM, eight warps, as the 96 x 512 fold: a warp's k-step is 48 IMMA (6 A tiles, each held for 8: two
// sets of 4 B tiles), its 8 B words from 4 LDS.128 of a 16 KB table (the same in every mode), operand bytes 0-127.
// The modes differ only in where a k-step's six A fragments come from:
//   AM 0 (fst): the stage. Warps 0-5 each load one tile from a 6 KB table (the gather), add the fill, store it to the
//        k-step's slot (STS.128); a barrier of all eight warps; every warp loads the six tiles (6 LDS.128). Two slots.
//   AM 1 (fgl2): 6 LDG.128 a k-step from an A' stream of 6 MB (2048 k-steps of 3 KB) that every SM reads at once, so
//        it stays in the L2 and the SM's eight warps share each line through the L1; loaded a k-step ahead.
//   AM 2 (fgdr): the same, each SM its own 6 MB of a 1 GB buffer (from DRAM, as tiles of different row groups).
//   AM 3 (fnoa): no A loads: the fragments change by ALU steps alone. What the A side costs at all.
//   AM 4 (fstn): fst's loads and stores into a ring of four slots with no barrier (the slot read is two k-steps
//        behind the one written; the data race changes no step). fst's one barrier a k-step starves the pipe in this
//        kernel, which the fold's group barriers do not, so fstn is the stage's traffic alone at a full pipe.
// Every four k-steps a readout XORs the sums together, as the fold's does.
template <int AM>
__global__ void __launch_bounds__(THREADS, 1) wfold(const uint4 *__restrict__ g, uint32_t seed, unsigned long long *sink) {
  __shared__ __align__(16) uint4 btab[1024];   // 16 KB
  __shared__ __align__(16) uint4 atab[384];    // 6 KB
  __shared__ __align__(16) uint4 stg[4][192];  // four 3 KB slots (fst uses two)
  const uint32_t t = threadIdx.x, w = t >> 5, lane = t & 31u;
  const uint32_t H = 0x7f7f7f7fu;
  uint32_t x = seed ^ (t * 2654435761u) ^ (blockIdx.x * 40503u);
  for (uint32_t i = t; i < 1024u; i += THREADS) {
    uint4 v; x = x * 1664525u + 1013904223u; v.x = x; x = x * 1664525u + 1013904223u; v.y = x;
    x = x * 1664525u + 1013904223u; v.z = x; x = x * 1664525u + 1013904223u; v.w = x;
    btab[i] = v;
    if (i < 384u) atab[i] = make_uint4(v.y, v.z, v.w, v.x);
  }
  __syncthreads();
  int d[6][4][4];
#pragma unroll
  for (int i = 0; i < 6; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int k = 0; k < 4; k++) d[i][j][k] = 0;
  unsigned a[6][4], an[6][4];
  const size_t base = AM == 2 ? (size_t)(blockIdx.x % 160u) * (6u << 20) / 16u : 0;
  auto ldA = [&](uint32_t s, unsigned (&r)[6][4]) {
#pragma unroll
    for (int m = 0; m < 6; m++) {
      const uint4 v = __ldg(g + base + (size_t)(s & 2047u) * 192u + m * 32u + lane);
      r[m][0] = v.x; r[m][1] = v.y; r[m][2] = v.z; r[m][3] = v.w;
    }
  };
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int k = 0; k < 4; k++) { x = x * 1664525u + 1013904223u; a[m][k] = an[m][k] = x; }
  if (AM == 1 || AM == 2) ldA(seed, an);
  uint32_t rd = 0;
#pragma unroll 1
  for (uint32_t s = 0; s < ITERS; s++) {
    unsigned b[2][4][2];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 v = btab[((s * 37u + q * 256u) & 1023u & ~31u) | lane];
      b[q >> 1][(q & 1) * 2][0] = v.x & H; b[q >> 1][(q & 1) * 2][1] = v.y & H;
      b[q >> 1][(q & 1) * 2 + 1][0] = v.z & H; b[q >> 1][(q & 1) * 2 + 1][1] = v.w & H;
    }
    if (AM == 0 || AM == 4) {
      if (w < 6u) {
        uint4 v = atab[((s * 64u + w * 32u) % 384u) + lane];
        // Bytes 0-127 as in the other modes (a first run used 63 + 0-63, 6 bits that toggle less: fstn then read
        // 3019 MHz at 232 W on a 460 W 5090 against fnoa's 2913 MHz at 342 W, a flattered price).
        v.x = (v.x ^ s) & H; v.y = (v.y ^ s) & H; v.z = (v.z ^ s) & H; v.w = (v.w ^ s) & H;
        stg[AM == 4 ? s & 3u : s & 1u][w * 32u + lane] = v;
      }
      if (AM == 0) __syncthreads();
#pragma unroll
      for (int m = 0; m < 6; m++) {
        const uint4 v = stg[AM == 4 ? (s + 2u) & 3u : s & 1u][m * 32u + lane];
        a[m][0] = v.x; a[m][1] = v.y; a[m][2] = v.z; a[m][3] = v.w;
      }
    } else if (AM == 3) {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int k = 0; k < 4; k++) a[m][k] = (a[m][k] * 1664525u + 1013904223u) & H;
    } else {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int k = 0; k < 4; k++) a[m][k] = an[m][k] & H;
      ldA(seed + s + 1u, an);
    }
#pragma unroll
    for (int p = 0; p < 2; p++)
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) {
          const unsigned aa[4] = {a[m][0], a[m][1], a[m][2], a[m][3]}, bb[2] = {b[p][n][0], b[p][n][1]};
          mma8(d[m][n], aa, bb);
        }
    if ((s & 3u) == 3u) {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) rd ^= (uint32_t)d[m][n][0] ^ (uint32_t)d[m][n][1] ^ (uint32_t)d[m][n][2] ^ (uint32_t)d[m][n][3];
    }
  }
  if (rd == 0x12345678u) sink[0] = rd;
}

// wfoldc (goal-rtx50c): wfold in clusters of two blocks (two SMs) that share A through distributed shared memory,
// the lead's next design priced before a hand loop. fstnc (AM 4) is fstn launched the same way; fdsms (5) gathers
// half the tiles a block and stores each in both blocks' stages; fdsml (6) stores locally and loads the peer's half
// remotely. No barriers on the stage (timing only, as fstn): the cluster's sync costs come on top.
template <int AM>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1) wfoldc(const uint4 *__restrict__ g, uint32_t seed, unsigned long long *sink) {
  __shared__ __align__(16) uint4 btab[1024];   // 16 KB
  __shared__ __align__(16) uint4 atab[384];    // 6 KB
  __shared__ __align__(16) uint4 stg[4][192];  // four 3 KB slots (fst uses two)
  const uint32_t t = threadIdx.x, w = t >> 5, lane = t & 31u;
  const uint32_t H = 0x7f7f7f7fu;
  uint32_t x = seed ^ (t * 2654435761u) ^ (blockIdx.x * 40503u);
  for (uint32_t i = t; i < 1024u; i += THREADS) {
    uint4 v; x = x * 1664525u + 1013904223u; v.x = x; x = x * 1664525u + 1013904223u; v.y = x;
    x = x * 1664525u + 1013904223u; v.z = x; x = x * 1664525u + 1013904223u; v.w = x;
    btab[i] = v;
    if (i < 384u) atab[i] = make_uint4(v.y, v.z, v.w, v.x);
  }
  __syncthreads();
  int d[6][4][4];
#pragma unroll
  for (int i = 0; i < 6; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int k = 0; k < 4; k++) d[i][j][k] = 0;
  unsigned a[6][4], an[6][4];
  const size_t base = AM == 2 ? (size_t)(blockIdx.x % 160u) * (6u << 20) / 16u : 0;
  auto ldA = [&](uint32_t s, unsigned (&r)[6][4]) {
#pragma unroll
    for (int m = 0; m < 6; m++) {
      const uint4 v = __ldg(g + base + (size_t)(s & 2047u) * 192u + m * 32u + lane);
      r[m][0] = v.x; r[m][1] = v.y; r[m][2] = v.z; r[m][3] = v.w;
    }
  };
#pragma unroll
  for (int m = 0; m < 6; m++)
#pragma unroll
    for (int k = 0; k < 4; k++) { x = x * 1664525u + 1013904223u; a[m][k] = an[m][k] = x; }
  if (AM == 1 || AM == 2) ldA(seed, an);
  uint32_t rd = 0;
#pragma unroll 1
  for (uint32_t s = 0; s < ITERS; s++) {
    unsigned b[2][4][2];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 v = btab[((s * 37u + q * 256u) & 1023u & ~31u) | lane];
      b[q >> 1][(q & 1) * 2][0] = v.x & H; b[q >> 1][(q & 1) * 2][1] = v.y & H;
      b[q >> 1][(q & 1) * 2 + 1][0] = v.z & H; b[q >> 1][(q & 1) * 2 + 1][1] = v.w & H;
    }
    if (AM == 4) {
      // fstnc: fstn exactly, launched in clusters of two (the control for the cluster modes).
      if (w < 6u) {
        uint4 v = atab[((s * 64u + w * 32u) % 384u) + lane];
        v.x = (v.x ^ s) & H; v.y = (v.y ^ s) & H; v.z = (v.z ^ s) & H; v.w = (v.w ^ s) & H;
        stg[s & 3u][w * 32u + lane] = v;
      }
#pragma unroll
      for (int m = 0; m < 6; m++) {
        const uint4 v = stg[(s + 2u) & 3u][m * 32u + lane];
        a[m][0] = v.x; a[m][1] = v.y; a[m][2] = v.z; a[m][3] = v.w;
      }
    } else if (AM == 5 || AM == 6) {
      // The cluster's two blocks share A: each gathers three of the six tiles (warps 0-2).
      // fdsms (5): the gatherer stores its tile in its own stage and in the peer's (st.shared::cluster); every
      // warp then loads all six tiles locally. fdsml (6): it stores only locally, and every warp loads its own
      // block's three tiles locally and the peer's three with ld.shared::cluster.
      const uint32_t rank = blockIdx.x & 1u, peer = rank ^ 1u;
      if (w < 3u) {
        uint4 v = atab[((s * 64u + w * 32u) % 384u) + lane];
        v.x = (v.x ^ s) & H; v.y = (v.y ^ s) & H; v.z = (v.z ^ s) & H; v.w = (v.w ^ s) & H;
        const uint32_t tile = 3u * rank + w;
        const uint32_t loc = (uint32_t)__cvta_generic_to_shared(&stg[s & 3u][tile * 32u + lane]);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(loc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
        if (AM == 5) {
          uint32_t rem;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rem) : "r"(loc), "r"(peer));
          asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(rem), "r"(v.x), "r"(v.y), "r"(v.z),
                       "r"(v.w) : "memory");
        }
      }
#pragma unroll
      for (int m = 0; m < 6; m++) {
        const uint32_t loc = (uint32_t)__cvta_generic_to_shared(&stg[(s + 2u) & 3u][m * 32u + lane]);
        uint4 v;
        if (AM == 6 && (uint32_t)m / 3u != rank) {
          uint32_t rem;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rem) : "r"(loc), "r"(peer));
          asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "r"(rem) : "memory");
        } else {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "r"(loc) : "memory");
        }
        a[m][0] = v.x; a[m][1] = v.y; a[m][2] = v.z; a[m][3] = v.w;
      }
    } else if (AM == 3) {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int k = 0; k < 4; k++) a[m][k] = (a[m][k] * 1664525u + 1013904223u) & H;
    } else {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int k = 0; k < 4; k++) a[m][k] = an[m][k] & H;
      ldA(seed + s + 1u, an);
    }
#pragma unroll
    for (int p = 0; p < 2; p++)
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) {
          const unsigned aa[4] = {a[m][0], a[m][1], a[m][2], a[m][3]}, bb[2] = {b[p][n][0], b[p][n][1]};
          mma8(d[m][n], aa, bb);
        }
    if ((s & 3u) == 3u) {
#pragma unroll
      for (int m = 0; m < 6; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) rd ^= (uint32_t)d[m][n][0] ^ (uint32_t)d[m][n][1] ^ (uint32_t)d[m][n][2] ^ (uint32_t)d[m][n][3];
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (rd == 0x12345678u) sink[0] = rd;
}


typedef void *nd_t;
static int (*n_init)(void), (*n_idx)(unsigned, nd_t *), (*n_pw)(nd_t, unsigned *), (*n_clk)(nd_t, int, unsigned *),
    (*n_rs)(nd_t, unsigned long long *), (*n_ut)(nd_t, unsigned *);
static nd_t dev;

// One launch of mode's kernel. The wimma variants: full random operands in orders 0-3, zero operands, bytes 0-127
// in orders 0, 1, 2, and LDS mixes (bytes 0-127) in orders 0, 1, 2.
// Loads from global memory: a buffer of `words` uint4 (32 MB: it stays in a 5090's 96 MB L2; 1 GB: from DRAM),
// read in coalesced rows, the data XORed so it is not dead. Where an L2 byte's energy sits against the cap.
static uint4 *gbuf;
__global__ void __launch_bounds__(THREADS) wldg(const uint4 *__restrict__ g, uint32_t words, uint32_t seed,
                                                unsigned long long *sink) {
  uint32_t i = (blockIdx.x * THREADS + threadIdx.x + seed * 4096u) % words;
  const uint32_t step = gridDim.x * THREADS;
  uint4 r = make_uint4(0, 0, 0, 0);
#pragma unroll 1
  for (int it = 0; it < ITERS / 4; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      uint4 v = __ldcg(g + i);
      r.x ^= v.x; r.y += v.y; r.z ^= v.z; r.w += v.w;
      i += step;
      if (i >= words) i -= words;
    }
  }
  if (r.x + r.y + r.z + r.w == 0x12345678u) sink[0] = r.x;
}

// Random-looking words for the buffer, so its loads toggle as drawn data does.
__global__ void ginit(uint4 *g, uint32_t words) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
    uint32_t x = i * 2654435761u + 12345u;
    uint4 v;
    v.x = x = x * 1664525u + 1013904223u; v.y = x = x * 1664525u + 1013904223u;
    v.z = x = x * 1664525u + 1013904223u; v.w = x * 1664525u + 1013904223u;
    g[i] = v;
  }
}

static void launch(int mode, int blocks, cudaStream_t st, uint32_t seed, unsigned long long *sink) {
  if (mode == 30) { wldg<<<blocks, THREADS, 0, st>>>(gbuf, (32u << 20) / 16, seed, sink); return; }
  if (mode == 31) { wldg<<<blocks, THREADS, 0, st>>>(gbuf, (1024u << 20) / 16, seed, sink); return; }
  // wfold: one block an SM (blocks / 4).
  if (mode >= 40 && mode <= 44) {
    const int b1 = blocks / 4;
    if (mode == 40) wfold<0><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 41) wfold<1><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 42) wfold<2><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 43) wfold<3><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 44) wfold<4><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    return;
  }
  if (mode >= 45 && mode <= 47) {
    const int b1 = (blocks / 4) & ~1;
    if (mode == 45) wfoldc<4><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 46) wfoldc<5><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    if (mode == 47) wfoldc<6><<<b1, THREADS, 0, st>>>(gbuf, seed, sink);
    return;
  }
  const uint32_t F = 0xffffffffu, H = 0x7f7f7f7fu;
  switch (mode) {
    case 10: wimma<0, 0><<<blocks, THREADS, 0, st>>>(F, seed, sink); break;
    case 11: wimma<1, 0><<<blocks, THREADS, 0, st>>>(F, seed, sink); break;
    case 12: wimma<2, 0><<<blocks, THREADS, 0, st>>>(F, seed, sink); break;
    case 13: wimma<3, 0><<<blocks, THREADS, 0, st>>>(F, seed, sink); break;
    case 14: wimma<0, 0><<<blocks, THREADS, 0, st>>>(0, seed, sink); break;
    case 15: wimma<0, 0><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 16: wimma<1, 0><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 17: wimma<2, 0><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 18: wimma<0, 4><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 19: wimma<0, 8><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 20: wimma<0, 16><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 21: wimma<1, 8><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 22: wimma<2, 8><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 23: wimma<0, 8, 1><<<blocks, THREADS, 0, st>>>(H, seed, sink, gbuf); break;
    case 24: wimma<0, 8, 2><<<blocks, THREADS, 0, st>>>(H, seed, sink, gbuf); break;
    case 25: wimma<0, 8, 3><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    case 26: wimma<0, 8, 4><<<blocks, THREADS, 0, st>>>(H, seed, sink); break;
    default: work<<<blocks, THREADS, 0, st>>>(mode, 0, seed, sink); break;
  }
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 30;
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!h) { printf("powcap no NVML\n"); return 1; }
  n_init = (int (*)(void))dlsym(h, "nvmlInit_v2");
  n_idx = (int (*)(unsigned, nd_t *))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  n_pw = (int (*)(nd_t, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
  n_clk = (int (*)(nd_t, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
  n_rs = (int (*)(nd_t, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
  n_ut = (int (*)(nd_t, unsigned *))dlsym(h, "nvmlDeviceGetUtilizationRates");  // {gpu %, memory controller %}
  if (!n_rs) n_rs = (int (*)(nd_t, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
  if (!n_init || n_init() != 0 || n_idx(0, &dev) != 0) { printf("powcap NVML init failed\n"); return 1; }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  unsigned long long *sink;
  cudaMalloc(&sink, 8);
  if (cudaMalloc(&gbuf, 1024u << 20) == cudaSuccess) ginit<<<4096, 256>>>(gbuf, (1024u << 20) / 16);
  const char *defs[] = {"idle", "imma", "immaz", "immah", "lds", "ldsm", "ffma", "mix8"};
  int nm = argc > 2 ? argc - 2 : (int)(sizeof defs / sizeof defs[0]);
  cudaStream_t st;
  cudaStreamCreate(&st);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int m = 0; m < nm; m++) {
    const char *name = argc > 2 ? argv[m + 2] : defs[m];
    // mode: 0 lds, 1 ldsm, 4 ffma (work); 10 + k: wimma's variant k (see launch below); -1 idle.
    int mode = -1;
    static const char *im[] = {"imma", "immas", "immam", "immax", "immaz", "immah", "immahs", "immahm",
                               "mix4", "mix8", "mix16", "mixs8", "mixm8", "mixl1", "mixl2", "mixldl", "mixsts"};
    if (!strcmp(name, "lds")) mode = 0;
    else if (!strcmp(name, "ldsm")) mode = 1;
    else if (!strcmp(name, "ffma")) mode = 4;
    else if (!strcmp(name, "ldgl2")) mode = 30;
    else if (!strcmp(name, "ldgdram")) mode = 31;
    else if (!strcmp(name, "fst")) mode = 40;
    else if (!strcmp(name, "fgl2")) mode = 41;
    else if (!strcmp(name, "fgdr")) mode = 42;
    else if (!strcmp(name, "fnoa")) mode = 43;
    else if (!strcmp(name, "fstn")) mode = 44;
    else if (!strcmp(name, "fstnc")) mode = 45;
    else if (!strcmp(name, "fdsms")) mode = 46;
    else if (!strcmp(name, "fdsml")) mode = 47;
    else {
      for (int k = 0; k < (int)(sizeof im / sizeof im[0]); k++)
        if (!strcmp(name, im[k])) mode = 10 + k;
      if (mode < 0 && strcmp(name, "idle")) { printf("powcap unknown mode %s\n", name); continue; }
    }
    const int blocks = sms * 4;
    double t0 = now_s(), warm = 10.0, wsum = 0, csum = 0, usum = 0;
    int ns = 0, launches = 0, counted = 0;
    int bits[32] = {0};
    float kms = 0;
    double next = t0;
    while (now_s() - t0 < secs + warm) {
      if (mode >= 0) {
        // A few launches queued, then one sample of NVML: the card is never idle between them.
        cudaEventRecord(e0, st);
        for (int q = 0; q < 8; q++) launch(mode, blocks, st, 12345u + launches * 7u + q, sink);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        launches += 8;
        if (now_s() - t0 >= warm) { kms += ms; counted += 8; }
      } else {
        usleep(50000);
      }
      double tn = now_s();
      if (tn - t0 >= warm && tn >= next) {
        next = tn + 0.1;
        unsigned mw = 0, mhz = 0;
        unsigned long long r = 0;
        if (n_pw(dev, &mw) == 0 && n_clk(dev, 1, &mhz) == 0) {
          wsum += mw / 1000.0; csum += mhz; ns++;
          unsigned ut[2] = {0, 0};
          if (n_ut && n_ut(dev, ut) == 0) usum += ut[1];
          if (n_rs && n_rs(dev, &r) == 0)
            for (int b = 0; b < 32; b++) if (r >> b & 1) bits[b]++;
        }
      }
    }
    cudaError_t err = cudaGetLastError();
    // Work a second: kernel launches a second, as each mode's own unit (a launch is blocks x THREADS x ITERS steps).
    printf("powcap %s launches/s=%.2f W=%.1f MHz=%.0f mem%%=%.1f n=%d", name,
           counted && kms > 0 ? counted / (kms / 1000.0) : 0.0, ns ? wsum / ns : 0.0, ns ? csum / ns : 0.0,
           ns ? usum / ns : 0.0, ns);
    for (int b = 0; b < 32; b++) if (bits[b]) printf(" b%d=%.0f", b, 100.0 * bits[b] / ns);
    printf("%s%s\n", err ? " err=" : "", err ? cudaGetErrorString(err) : "");
    fflush(stdout);
  }
  return 0;
}
