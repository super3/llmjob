// gprobe -- the harness of the gather fold (hand89/gfold_h.cuh): it makes the tables and the offsets, runs the
// kernel (PD_CUBIN=<file> runs the cubin's, which is how a loop written by hand89/gfold.py is launched), prints
// how much of the tensor pipe the chunk loop holds, and checks every thread's transcript against a CPU fold of
// the same tiles. Our own code, for the nv-kernel work.
//
//   gprobe <name> <k> <tiles a block> [seconds] [tiles to check] [order] [fake] [rot]
//
// order: how the k of a chunk are put into its 128 slots (any order gives the same sums at the chunk's end).
//   0 the definition's; 1 searched for the fewest bank collisions in A's and B's gathers; 2 searched for B's
//   alone; 3 for A's alone; 4 to 6 searched for both with B's rows weighing more or the deepest collisions
//   costing most (see octet_cost). The search is hand89's port of the maths review's (S/algo/order.js).
// fake: TIMING ONLY, the results are wrong on purpose and the check will say so. Bit 0: A's offsets are made up
//   so that no matrix shares a bank (as a stage's reads would have it); bit 1: B's. For what the operands'
//   values cost in watts: 4, both row bands read the first band's A (the two warps of a scheduler then fold
//   the same operands); 8, all four column slots read the first slot's B; 16, the plus and the minus rank
//   index are the same made-up one for A and B (every operand byte is then the fill); 32, the tables hold
//   bytes spread as noised operands are (for gfold.py direct=1, a loop with no draw).
//
// rot: 1 writes the offsets for the rotation fold (hand89/rfold.py; build with -DGP_STAGE=24576). Every warp
//   takes the four k-steps of a chunk in an order of its own (any order inside a chunk gives the same sums):
//   column slot j of row band 0 takes k-step (j + i) & 3 at its i-th turn, of row band 1 k-step (j + (i ^ 1)) & 3.
//   So at a chunk's first turn the four warps of a band fold four different k-steps: each makes that k-step's A
//   fragments itself and stores them, and the other three load them at a later turn. A warp's records are in
//   its own order; a record of a later turn holds, in place of A's first offset, where that k-step's stored
//   fragments are. The kernel as ptxas compiles it cannot use these offsets: rot is for the hand-written loop.
//   rot 2 is the same in records of 8 bytes, two offsets to a word (rfold.py's co, its default now); rot 1
//   writes the 16-byte records, for rfold.py co=0.
// Build: nvcc -arch=sm_89 -O3 -std=c++17 -o gprobe gprobe.cu   (and -cubin for the file gfold.py patches)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
// GP_CFILLA, GP_CFILLB (goal-nvk17): the fill the CPU check gives A and B, for hand loops (rfold.py fa, fb, xa, xb)
// whose operands are another fill plus the noise: -64 (fa=0xc0 read as S8), 0 (fa=0x80 xa=0x80), 64 (fa=0x40).
#ifndef GP_CFILLA
#define GP_CFILLA GP_FILL
#endif
#ifndef GP_CFILLB
#define GP_CFILLB GP_FILL
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define GP_FN pearl_gather_probe
#include "gfold_h.cuh"

static uint32_t rng_state = 0x12345678u;
static uint32_t rng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
static inline uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }

// The order of a chunk's k: member[o][x] is the k (0 .. 127 of the chunk) in place x of octet o. An octet is
// the eight slots one ldmatrix matrix reads: slots 4t + 2oo + {0, 1} of half-k-step o >> 1, oo = o & 1. The
// search lowers the rows of a matrix that share their banks with an earlier row (entries with the same rank
// index mod 8), over the streams in `mask` (bit 0 A's plus index, 1 A's minus, 2 B's plus, 3 B's minus).
// g_wt: what a colliding row of each stream costs the search (A's two, B's two); g_sq: a bank shared by n rows
// costs (n - 1) squared and not n - 1, so that the search breaks up the deepest collisions first (a matrix is
// served in as many passes as its fullest bank has rows).
static int g_wt[4] = {1, 1, 1, 1}, g_sq = 0;
static int octet_cost(const uint8_t *cnt, unsigned mask) {
  int c = 0;
  for (unsigned st = 0; st < 4u; st++)
    if (mask >> st & 1u)
      for (unsigned r = 0; r < 8u; r++)
        if (cnt[8u * st + r] > 1) {
          const int x = cnt[8u * st + r] - 1;
          c += g_wt[st] * (g_sq ? x * x : x);
        }
  return c;
}
static void order_search(const uint8_t res[128][4], unsigned mask, int sweeps, uint8_t member[16][8]) {
  uint8_t cnt[16][32];
  memset(cnt, 0, sizeof cnt);
  auto bump = [&](unsigned o, unsigned i, int d) {
    for (unsigned st = 0; st < 4u; st++) cnt[o][8u * st + res[i][st]] = (uint8_t)(cnt[o][8u * st + res[i][st]] + d);
  };
  for (unsigned o = 0; o < 16u; o++)
    for (unsigned x = 0; x < 8u; x++) {
      member[o][x] = (uint8_t)(8u * o + x);
      bump(o, member[o][x], 1);
    }
  for (int sw = 0; sw < sweeps; sw++) {
    int moved = 0;
    for (unsigned o1 = 0; o1 < 16u; o1++)
      for (unsigned x = 0; x < 8u; x++) {
        int best = 0, bo = -1, by = -1;
        const unsigned i1 = member[o1][x];
        for (unsigned o2 = 0; o2 < 16u; o2++) {
          if (o2 == o1) continue;
          const int before = octet_cost(cnt[o1], mask) + octet_cost(cnt[o2], mask);
          for (unsigned y = 0; y < 8u; y++) {
            const unsigned i2 = member[o2][y];
            bump(o1, i1, -1); bump(o1, i2, 1); bump(o2, i2, -1); bump(o2, i1, 1);
            const int gain = before - octet_cost(cnt[o1], mask) - octet_cost(cnt[o2], mask);
            bump(o1, i1, 1); bump(o1, i2, -1); bump(o2, i2, 1); bump(o2, i1, -1);
            if (gain > best) { best = gain; bo = (int)o2; by = (int)y; }
          }
        }
        if (bo >= 0) {
          const unsigned i2 = member[bo][by];
          bump(o1, i1, -1); bump(o1, i2, 1); bump((unsigned)bo, i2, -1); bump((unsigned)bo, i1, 1);
          member[o1][x] = (uint8_t)i2;
          member[bo][by] = (uint8_t)i1;
          moved++;
        }
      }
    if (!moved) break;
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "gprobe name k tiles-a-block [seconds] [tiles to check] [order] [fake]\n");
    return 2;
  }
  const char *name = argv[1];
  const uint32_t k = (uint32_t)atoi(argv[2]);
  const uint32_t tpb = (uint32_t)atoi(argv[3]);
  const double secs = argc > 4 ? atof(argv[4]) : 0.0;
  const uint32_t check = argc > 5 ? (uint32_t)atoi(argv[5]) : 0u;
  const uint32_t order_mode = argc > 6 ? (uint32_t)atoi(argv[6]) : 0u;
  const uint32_t fake = argc > 7 ? (uint32_t)atoi(argv[7]) : 0u;
  // rot 3 (goal-nv50d): rot 2's records with the tables in plain row order (byte c of an entry is row c of the
  // group), for rfold.py's b8 gathers (ldmatrix's byte transpose, sm_120).
  const bool tab8 = argc > 8 && atoi(argv[8]) == 3;
  // rot 4 (goal-nvk19): records the eight warps share, for rfold.py's sr. One stream of 8-byte records, k-step by
  // k-step: lane l's {16 pa0 | 16 pa1 << 16, 16 pb0 | 16 pb1 << 16}, no warp's base in them (the loop keeps those
  // in uniform registers), then four k-steps of zeros (the loop reads up to three k-steps past the last chunk).
  const bool shared = argc > 8 && atoi(argv[8]) == 4;
  const uint32_t rot = argc > 8 ? (tab8 || shared ? 2u : (uint32_t)atoi(argv[8])) : 0u;
  if (k < 2048u || (k & (k - 1u)) || tpb < 1u || order_mode > 11u) {
    fprintf(stderr, "k is a power of two from 2048; at least one tile a block; order 0 to 6\n");
    return 2;
  }
  const uint32_t chunks = k / 128u, ksteps = k / 32u;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const uint32_t blocks = (uint32_t)prop.multiProcessorCount;
  // At most 1024 column blocks of tables (34 MB): a long timing run folds its tiles on them again.
  const uint32_t RG = 8u, tiles = blocks * tpb, CBall = (tiles + RG - 1u) / RG, CBmax = GP_DIRECT ? 16u : 1024u, CB = CBall < CBmax ? CBall : CBmax;
  const uint32_t MT = GP_MT, BM = 32u * MT, NA = GP_NA;

  // Plain inputs: the dense factors plus 32 (0 .. 63), a row of 128 for every row of A and column of B, and
  // every k's plus and minus rank index (distinct, as the protocol's are). pm[A, B][plus, minus][k].
  const uint32_t rowsA = RG * BM, colsB = CB * 256u;
  std::vector<uint8_t> DA((size_t)rowsA * 128u), DB((size_t)colsB * 128u);
  for (auto &b : DA) b = (uint8_t)(rng() >> 9) & 63u;
  for (auto &b : DB) b = (uint8_t)(rng() >> 9) & 63u;
  if (fake & 32u) {
    // TIMING ONLY: the tables hold bytes spread as noised operands are (fill + plus - minus, 0 .. 126), for a
    // loop that takes its fragments from the table as they stand (gfold.py direct=1).
    for (auto &b : DA) b = (uint8_t)(63u + ((rng() >> 9) & 63u) - ((rng() >> 17) & 63u));
    for (auto &b : DB) b = (uint8_t)(63u + ((rng() >> 9) & 63u) - ((rng() >> 17) & 63u));
  }
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
  // PD_INPUT=<file>: the plain inputs come from a file and not from the generator above, so that a job's own
  // noise can be folded (hand89/hostcheck.js writes it from the protocol's reference code): 16 bytes "GPIN",
  // rows of A, columns of B, k; then the rows' dense factors plus 32 (128 bytes each), the columns', and the
  // four index lists (A plus, A minus, B plus, B minus; k bytes each).
  if (const char *inp = getenv("PD_INPUT")) {
    FILE *f = fopen(inp, "rb");
    uint32_t hdr[4] = {0, 0, 0, 0};
    bool ok = f && fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x4e495047u && hdr[1] == rowsA && hdr[2] == colsB && hdr[3] == k &&
              fread(DA.data(), 1, DA.size(), f) == DA.size() && fread(DB.data(), 1, DB.size(), f) == DB.size();
    for (int op = 0; ok && op < 2; op++)
      for (int sg = 0; ok && sg < 2; sg++) ok = fread(pm[op][sg].data(), 1, k, f) == k;
    if (f) fclose(f);
    if (!ok) {
      fprintf(stderr, "%s: PD_INPUT %s does not hold %u rows, %u columns, k %u\n", name, inp, rowsA, colsB, k);
      return 1;
    }
  }
#if GP_DIRECT
  // The direct fold's operands: A' and B' as the mma's fragments hold them (gfold_h.cuh, GP_DIRECT), noised here
  // on the CPU from the same plain inputs the check uses. One k-step of zeros at each table's end.
  (void)order_mode;
  double coll[2] = {0, 0}, worst[2] = {0, 0}, order_ms = 0;
  const double mats = 1.0;
  std::vector<uint32_t> hOffs(16u, 0u);
  const size_t aRec = (size_t)64u * MT * 16u, bRec = 8192u;       // bytes a k-step
  std::vector<uint8_t> hTabA((size_t)RG * (ksteps + 1u) * aRec, 0), hTabB((size_t)CB * (ksteps + 1u) * bRec, 0);
  for (uint32_t t = 0; t < RG; t++)
    for (uint32_t ks = 0; ks < ksteps; ks++)
      for (uint32_t x = 0; x < 2u * MT; x++)                       // row band and m16 tile: the block's tile x
        for (uint32_t l = 0; l < 32u; l++)
          for (uint32_t wi = 0; wi < 4u; wi++) {
            const uint32_t row = t * BM + 16u * x + (l >> 2) + (wi & 1u) * 8u;
            const uint32_t k0 = ks * 32u + (wi >> 1) * 16u + 4u * (l & 3u);
            uint8_t *w = &hTabA[((size_t)t * (ksteps + 1u) + ks) * aRec + ((size_t)x * 32u + l) * 16u + wi * 4u];
            const uint8_t *T = &DA[(size_t)row * 128u];
            for (uint32_t i = 0; i < 4u; i++)
              w[i] = (uint8_t)(GP_FILL + (int)T[pm[0][0][k0 + i]] - (int)T[pm[0][1][k0 + i]]);
          }
  for (uint32_t t = 0; t < CB; t++)
    for (uint32_t ks = 0; ks < ksteps; ks++)
      for (uint32_t x = 0; x < 16u; x++)                           // column slot and pair
        for (uint32_t l = 0; l < 32u; l++)
          for (uint32_t wi = 0; wi < 4u; wi++) {
            const uint32_t col = t * 256u + 16u * x + (l >> 2) + (wi >> 1) * 8u;
            const uint32_t k0 = ks * 32u + (wi & 1u) * 16u + 4u * (l & 3u);
            uint8_t *w = &hTabB[((size_t)t * (ksteps + 1u) + ks) * bRec + ((size_t)x * 32u + l) * 16u + wi * 4u];
            const uint8_t *T = &DB[(size_t)col * 128u];
            for (uint32_t i = 0; i < 4u; i++)
              w[i] = (uint8_t)(GP_FILL + (int)T[pm[1][0][k0 + i]] - (int)T[pm[1][1][k0 + i]]);
          }
#else
  // The order of k in each chunk: slot -> k of the chunk.
  std::vector<uint8_t> order((size_t)chunks * 128u);
  double order_ms = 0;
  {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t ch = 0; ch < chunks; ch++) {
      uint8_t *o = &order[(size_t)ch * 128u];
      for (uint32_t i = 0; i < 128u; i++) o[i] = (uint8_t)i;
      if (!order_mode) continue;
      if (order_mode >= 10u) {
        // TIMING ONLY in effect (more bank collisions, correct sums): the chunk's k sorted by A's plus then minus
        // rank index (10) or B's (11), so that k next to each other in a fragment share table entries. Does the
        // tensor core's power follow how alike the k inside one IMMA are? (goal-nvk12)
        const uint32_t op = order_mode - 10u;
        std::stable_sort(o, o + 128, [&](uint8_t x, uint8_t y) {
          const uint32_t kx = ch * 128u + x, ky = ch * 128u + y;
          return pm[op][0][kx] * 256u + pm[op][1][kx] < pm[op][0][ky] * 256u + pm[op][1][ky];
        });
        continue;
      }
      uint8_t res[128][4], member[16][8];
      for (uint32_t i = 0; i < 128u; i++) {
        const uint32_t kk = ch * 128u + i;
        res[i][0] = pm[0][0][kk] & 7u;
        res[i][1] = pm[0][1][kk] & 7u;
        res[i][2] = pm[1][0][kk] & 7u;
        res[i][3] = pm[1][1][kk] & 7u;
      }
      // 4 to 6 are for the rotation fold, where a warp gathers B eight times a k-step and A three: B's rows
      // weigh 3 (4), or the cost is squared with B's rows at 1 (5), 2 (6) or 3 (9). 7 and 8: squared, A's
      // rows at 2 or 3 and B's at 1.
      g_wt[2] = g_wt[3] = order_mode == 4u || order_mode == 9u ? 3 : order_mode == 6u ? 2 : 1;
      g_wt[0] = g_wt[1] = order_mode == 7u ? 2 : order_mode == 8u ? 3 : 1;
      g_sq = order_mode >= 5u;
      order_search(res, order_mode == 2u ? 12u : order_mode == 3u ? 3u : 15u, 8, member);
      for (uint32_t oc = 0; oc < 16u; oc++)
        for (uint32_t x = 0; x < 8u; x++)
          o[16u * (oc >> 1) + 4u * (x >> 1) + 2u * (oc & 1u) + (x & 1u)] = member[oc][x];
    }
    order_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / chunks;
  }
  // The kernel's offsets, and what its gathers will meet in the banks: the rows of a matrix (8 lanes) whose
  // entry shares its four banks with an earlier row's, a matrix, for A and for B.
  // One table a warp: the words carry the base of the warp's groups in shared memory.
  const size_t warpWords = (size_t)(ksteps + 1u) * 32u * 4u;
  std::vector<uint32_t> hOffs(8u * warpWords, 0u);
  std::vector<uint32_t> hShared(shared ? (size_t)(ksteps + 4u) * 64u : 0u, 0u);
  if (shared && (fake & 12u)) {
    fprintf(stderr, "%s: rot 4 has no fake 4 or 8 (the bases are the loop's)\n", name);
    return 2;
  }
  double coll[2] = {0, 0}, worst[2] = {0, 0};
  for (uint32_t ks = 0; ks <= ksteps; ks++) {
    const uint32_t ch = ks / 4u, s = ks % 4u;
    uint32_t p4[32][4];
    for (uint32_t l = 0; l < 32u; l++) {
      const uint32_t sign = l >> 4, slot = 4u * ((l & 7u) >> 1) + 2u * ((l >> 3) & 1u) + (l & 1u);
      for (uint32_t h = 0; h < 2u; h++) {
        uint32_t pa = 0, pb = 0;
        if (ks < ksteps) {
          const uint32_t kk = ch * 128u + order[(size_t)ch * 128u + 32u * s + 16u * h + slot];
          pa = pm[0][sign][kk];
          pb = pm[1][sign][kk];
          const uint32_t made = (l & 7u) + 8u * ((ks * 5u + 3u * (l >> 3) + h) & 15u);
          const uint32_t same = (l & 7u) + 8u * ((ks * 5u + 3u * ((l >> 3) & 1u) + h) & 15u);
          if (fake & 1u) pa = made;
          if (fake & 2u) pb = made;
          if (fake & 16u) pa = pb = same;
        }
        p4[l][h] = pa;
        p4[l][2u + h] = pb;
        if (shared && ks < ksteps) {
          hShared[((size_t)ks * 32u + l) * 2u] |= (16u * pa) << (16u * h);
          hShared[((size_t)ks * 32u + l) * 2u + 1u] |= (16u * pb) << (16u * h);
        }
        for (uint32_t w = 0; w < 8u; w++) {
          // rot: this k-step is warp w's turn `turn` of the chunk (the last record, of the k-step that never
          // comes, is a first turn).
          const uint32_t j = w & 3u, band = w >> 2;
          const uint32_t turn = !rot || ks == ksteps ? s : band ? (((s + 4u - j) & 3u) ^ 1u) : ((s + 4u - j) & 3u);
          uint32_t *o = &hOffs[w * warpWords + ((size_t)(ch * 4u + turn) * 32u + l) * 4u];
          o[h] = 16u * pa + ((fake & 4u) ? 0u : 2048u * MT * (w >> 2));
          o[2u + h] = 16u * pb + GP_TABB + ((fake & 8u) ? 0u : 8192u * (w & 3u));
          if (rot && turn && ks < ksteps) o[h] = h ? 0u : GP_STG + band * (2048u * MT) + s * (512u * MT) + 16u * l;
        }
      }
    }
    if (ks == ksteps) break;
    for (uint32_t q = 0; q < 4u; q++)
      for (uint32_t u = 0; u < 4u; u++) {
        uint32_t cnt[8] = {0}, extra = 0, most = 0;
        for (uint32_t j = 0; j < 8u; j++) cnt[p4[8u * u + j][q] & 7u]++;
        for (uint32_t r = 0; r < 8u; r++) {
          if (cnt[r] > 1u) extra += cnt[r] - 1u;
          if (cnt[r] > most) most = cnt[r];
        }
        coll[q >> 1] += extra;
        worst[q >> 1] += most;
      }
  }
  // rot 2: the same offsets in half the bytes (rfold.py co=1). A record is two words where it was four. A
  // first turn's {x, y, z, w} is {x | y << 16, z | w << 16}; a later turn's (x the stage address, y nothing)
  // is {x, z | w << 16}. Every offset is a shared-memory address under 65536. The records come from the L2:
  // 256 bytes a warp a k-step, not 512.
  if (rot == 2u) {
    const size_t recs = hOffs.size() / 4u;
    std::vector<uint32_t> packed(recs * 2u);
    for (size_t i = 0; i < recs; i++) {
      const uint32_t *o = &hOffs[i * 4u];
      const uint32_t turn = (uint32_t)((i / 32u) % (size_t)(ksteps + 1u)) & 3u;
      if (((turn ? 0u : o[0] | o[1]) | o[2] | o[3]) >> 16) {
        fprintf(stderr, "%s: an offset does not fit 16 bits\n", name);
        return 1;
      }
      packed[2u * i] = turn ? o[0] : o[0] | (o[1] << 16);
      packed[2u * i + 1u] = o[2] | (o[3] << 16);
    }
    hOffs.swap(packed);
  }
  if (shared) hOffs.swap(hShared);
  const double mats = (double)ksteps * 8.0;
  // The tables: [row group][A group 0 .. NA-1][p] and [column block][B group 0 .. 15][p], 16 bytes an entry.
  std::vector<uint8_t> hTabA((size_t)RG * NA * 2048u), hTabB((size_t)CB * 16u * 2048u);
  for (uint32_t t = 0; t < RG; t++)
    for (uint32_t g = 0; g < NA; g++)
      for (uint32_t p = 0; p < 128u; p++)
        for (uint32_t e = 0; e < 8u; e++) {
          const uint32_t row = t * BM + 16u * g + e;
          uint8_t *w = &hTabA[(((size_t)t * NA + g) * 128u + p) * 16u];
          w[tab8 ? e : 2u * e] = DA[(size_t)row * 128u + p];
          w[tab8 ? 8u + e : 2u * e + 1u] = DA[(size_t)(row + 8u) * 128u + p];
        }
  for (uint32_t t = 0; t < CB; t++)
    for (uint32_t g = 0; g < 16u; g++)
      for (uint32_t p = 0; p < 128u; p++)
        for (uint32_t e = 0; e < 8u; e++) {
          const uint32_t col = t * 256u + 16u * g + e;
          uint8_t *w = &hTabB[(((size_t)t * 16u + g) * 128u + p) * 16u];
          w[tab8 ? e : 2u * e] = DB[(size_t)col * 128u + p];
          w[tab8 ? 8u + e : 2u * e + 1u] = DB[(size_t)(col + 8u) * 128u + p];
        }

#endif
  uint4 *dTabA, *dTabB, *dOffs;
  unsigned long long *dCyc;
  uint32_t *dDump;
  const size_t cycN = (size_t)blocks * 8u * 4u;
  uint32_t dumpTiles = check < tiles ? check : tiles;
  // The hand-written loops move their pointers with 32-bit adds (the direct fold's B pointer starts a chunk
  // early): no table may cross a multiple of 4 GiB or lie in the first megabyte above one.
  auto alloc_clear = [&](uint4 **p, size_t bytes) {
    for (int tries = 0; tries < 8; tries++) {
      cudaMalloc(p, bytes);
      const unsigned long long lo = (unsigned long long)(uintptr_t)*p & 0xffffffffull;
      if (lo >= 0x100000ull && lo + bytes < 0x100000000ull) return true;
    }
    return false;
  };
  // (64 KiB more than the tables: the direct fold's look-ahead loads run a few k-steps past a tile's end.)
  if (!alloc_clear(&dTabA, hTabA.size() + 65536u) || !alloc_clear(&dTabB, hTabB.size() + 65536u)) {
    fprintf(stderr, "%s: no table clear of a 4 GiB line\n", name);
    return 1;
  }
  // The hand-written loop moves its offsets pointer with a 32-bit add: the table must not cross 4 GiB.
  for (int tries = 0;; tries++) {
    cudaMalloc(&dOffs, hOffs.size() * 4u);
    if (((uintptr_t)dOffs & 0xffffffffull) + hOffs.size() * 4u < 0x100000000ull) break;
    if (tries == 8) {
      fprintf(stderr, "%s: no offsets table clear of a 4 GiB line\n", name);
      return 1;
    }
  }
  // PD_HASH=<file> (a build with -DGP_HASH=1): the file's 64 bytes are the hash's key and the bound, each 8
  // little-endian words. The card then hashes every thread's transcript of every tile and lists the hits;
  // PD_HITFILE=<file> gets the count and the (tile, thread) pairs.
  const uint32_t hitRoom = 1u << 20;
  uint32_t *dHits = nullptr;
  if (const char *hf = getenv("PD_HASH")) {
    uint32_t kb[18];
    FILE *f = fopen(hf, "rb");
    const bool ok = f && fread(kb, 4, 16, f) == 16;
    if (f) fclose(f);
    if (!ok || !GP_HASH) {
      fprintf(stderr, "%s: PD_HASH needs a 64-byte file and a build with -DGP_HASH=1", name);
      return 1;
    }
    kb[16] = 0u;
    kb[17] = hitRoom;
    cudaMalloc(&dHits, (18u + 2u * (size_t)hitRoom) * 4u);
    cudaMemcpy(dHits, kb, sizeof kb, cudaMemcpyHostToDevice);
  }
  cudaMalloc(&dCyc, cycN * 8u);
  cudaMalloc(&dDump, (size_t)(dumpTiles ? dumpTiles : 1u) * GP_THREADS * 16u * 4u);
  cudaMemcpy(dTabA, hTabA.data(), hTabA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dTabB, hTabB.data(), hTabB.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dOffs, hOffs.data(), hOffs.size() * 4u, cudaMemcpyHostToDevice);
  cudaMemset(dCyc, 0, cycN * 8u);
  cudaFuncSetAttribute(pearl_gather_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GP_SMEM);

  // PD_CUBIN=<file>: the kernel is launched from that cubin and not from the program. The driver's own calls
  // are looked up by name, so the probe needs no driver header or library to build.
  typedef int (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                          void **, void **);
  void *cubin_fn = nullptr;
  launch_t cubin_launch = nullptr;
  const char *cb = getenv("PD_CUBIN");
  if (cb && *cb) {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
#define GP_SYM(n) (lib ? (void *)GetProcAddress(lib, n) : nullptr)
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
#define GP_SYM(n) (lib ? dlsym(lib, n) : nullptr)
#endif
    typedef int (*load_t)(void **, const char *);
    typedef int (*get_t)(void **, void *, const char *);
    typedef int (*attr_t)(void *, int, int);
    load_t load = (load_t)GP_SYM("cuModuleLoad");
    get_t get = (get_t)GP_SYM("cuModuleGetFunction");
    attr_t attr = (attr_t)GP_SYM("cuFuncSetAttribute");
    cubin_launch = (launch_t)GP_SYM("cuLaunchKernel");
#undef GP_SYM
    void *mod = nullptr;
    int rc = -1;
    if (!load || !get || !attr || !cubin_launch || (rc = load(&mod, cb)) ||
        (rc = get(&cubin_fn, mod, "pearl_gather_probe")) ||
        (rc = attr(cubin_fn, 8 /* the most dynamic shared memory */, (int)GP_SMEM))) {
      fprintf(stderr, "%s: PD_CUBIN %s: driver error %d\n", name, cb, rc);
      return 1;
    }
  }
  uint32_t a_chunks = chunks, a_rg = RG, a_tiles = tiles, a_cb = CB;
  void *cubin_args[11] = {&dTabA, &dTabB, &dOffs, &a_chunks, &a_rg, &a_tiles, &dCyc, &dDump, &dumpTiles, &a_cb, &dHits};

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaEventRecord(e0);
    cudaError_t err;
    if (cubin_fn) {
      err = cubin_launch(cubin_fn, blocks, 1, 1, GP_THREADS, 1, 1, GP_SMEM, nullptr, cubin_args, nullptr)
                ? cudaErrorLaunchFailure
                : cudaSuccess;
    } else {
      pearl_gather_probe<<<blocks, GP_THREADS, GP_SMEM>>>(dTabA, dTabB, dOffs, chunks, RG, tiles, dCyc, dDump,
                                                         dumpTiles, CB, dHits);
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
  double useSum = 0, useMin = 1e9, useMax = 0, tileSum = 0;
  unsigned long long tilesDone = 0;
  int nsched = 0;
  for (uint32_t b = 0; b < blocks; b++)
    for (uint32_t s = 0; s < 4u; s++) {
      const unsigned long long *w0 = &cyc[((size_t)b * 8u + s) * 4u], *w1 = &cyc[((size_t)b * 8u + s + 4u) * 4u];
      if (!w0[3] || !w1[3]) continue;
      const double imma = (double)(w0[3] + w1[3]) * chunks * 4.0 * GP_NI;
      const double loop = (double)(w0[1] > w1[1] ? w0[1] : w1[1]);
      const double all0 = (double)(w0[0] + w0[1] + w0[2]), all1 = (double)(w1[0] + w1[1] + w1[2]);
      const double use = 16.0 * imma / loop;
      useSum += use;
      tileSum += 16.0 * imma / (all0 > all1 ? all0 : all1);
      if (use < useMin) useMin = use;
      if (use > useMax) useMax = use;
      tilesDone += w0[3] + w1[3];
      nsched++;
    }
  const double use = nsched ? useSum / nsched : 0, tuse = nsched ? tileSum / nsched : 0;
  printf("%-22s k %u order %u%s%s pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA tile %6.2f%% tiles %llu %.0f ms x%d | "
         "rows of 8 that collide A %.2f B %.2f, largest A %.2f B %.2f, search %.2f ms/chunk",
         name, k, order_mode, rot ? " rot" : "", fake ? " FAKE(timing only)" : "", use * 100, useMin * 100, useMax * 100,
         use > 0 ? 16.0 / use : 0.0, tuse * 100, tilesDone / 8ull, ms, launches, coll[0] / mats, coll[1] / mats,
         worst[0] / mats, worst[1] / mats, order_ms);

  if (dHits) {
    // The card's hits over every launch made (a timed run folds the same tiles again and lists them again).
    uint32_t head[18];
    cudaMemcpy(head, dHits, sizeof head, cudaMemcpyDeviceToHost);
    const uint32_t kept = head[16] < hitRoom ? head[16] : hitRoom;
    printf(" | the card's hits %u in %d launch%s", head[16], launches, launches == 1 ? "" : "es");
    if (const char *of = getenv("PD_HITFILE")) {
      std::vector<uint32_t> list(2u * (size_t)kept + 2u);
      list[0] = head[16];
      list[1] = (uint32_t)launches;
      if (kept) cudaMemcpy(list.data() + 2, dHits + 18, 2u * (size_t)kept * 4u, cudaMemcpyDeviceToHost);
      FILE *f = fopen(of, "wb");
      if (f) {
        fwrite(list.data(), 4, list.size(), f);
        fclose(f);
      }
    }
  }
  if (getenv("PD_WARPS")) {
    // Each warp's loop cycles a tile (the mean over the blocks), and how far apart a block's eight warps end
    // a tile's loop: they start it together, so that is what they drift in a tile.
    double sum[8] = {0}, spread = 0, worst = 0;
    uint32_t nb = 0;
    for (uint32_t b = 0; b < blocks; b++) {
      double lo = 1e30, hi = 0;
      bool ok = true;
      for (uint32_t w = 0; w < 8u; w++) {
        const unsigned long long *c = &cyc[((size_t)b * 8u + w) * 4u];
        if (!c[3]) { ok = false; break; }
        const double per = (double)c[1] / (double)c[3];
        sum[w] += per;
        if (per < lo) lo = per;
        if (per > hi) hi = per;
      }
      if (!ok) continue;
      nb++;
      spread += hi - lo;
      if (hi - lo > worst) worst = hi - lo;
    }
    printf(" | loop cycles a tile by warp:");
    for (uint32_t w = 0; w < 8u; w++) printf(" %.0f", nb ? sum[w] / nb : 0.0);
    printf("; slowest less fastest warp of a block, mean %.0f worst %.0f", nb ? spread / nb : 0.0, worst);
  }
  if (dumpTiles) {
    // The CPU's fold of the same tiles from the plain inputs, in the definition's order of k.
    std::vector<uint32_t> dump((size_t)dumpTiles * GP_THREADS * 16u);
    cudaMemcpy(dump.data(), dDump, dump.size() * 4u, cudaMemcpyDeviceToHost);
    // PD_DUMPFILE=<file>: the card's transcripts of the checked tiles, [tile][thread][16 words], as they are.
    if (const char *df = getenv("PD_DUMPFILE")) {
      FILE *f = fopen(df, "wb");
      if (f) {
        fwrite(dump.data(), 4, dump.size(), f);
        fclose(f);
      }
    }
    uint32_t bad = 0, total_tr = 0, bad_tiles = 0, first_bad = 0xffffffffu;
    std::vector<int8_t> Ap((size_t)BM * 128), Bp((size_t)256 * 128);
    std::vector<int32_t> C((size_t)BM * 256);
    std::vector<uint32_t> tr((size_t)GP_THREADS * 16);
    for (uint32_t v = 0; v < dumpTiles; v++) {
      const uint32_t rbg = v % RG, cbg = v / RG;
      std::fill(C.begin(), C.end(), 0);
      std::fill(tr.begin(), tr.end(), 0u);
      for (uint32_t ch = 0; ch < chunks; ch++) {
        for (uint32_t kk = 0; kk < 128u; kk++) {
          const uint32_t gk = ch * 128u + kk;
          const uint32_t ap0 = pm[0][0][gk], ap1 = pm[0][1][gk], bp0 = pm[1][0][gk], bp1 = pm[1][1][gk];
          for (uint32_t r = 0; r < BM; r++) {
            const uint8_t *T = &DA[(size_t)(rbg * BM + r) * 128u];
            Ap[(size_t)r * 128u + kk] = (int8_t)(GP_CFILLA + (int)T[ap0] - (int)T[ap1]);
          }
          for (uint32_t c = 0; c < 256u; c++) {
            const uint8_t *T = &DB[(size_t)(cbg * 256u + c) * 128u];
            Bp[(size_t)c * 128u + kk] = (int8_t)(GP_CFILLB + (int)T[bp0] - (int)T[bp1]);
          }
        }
        for (uint32_t r = 0; r < BM; r++)
          for (uint32_t c = 0; c < 256u; c++) {
            const int8_t *a = &Ap[(size_t)r * 128u], *bb = &Bp[(size_t)c * 128u];
            int32_t sum = 0;
            for (uint32_t kk = 0; kk < 128u; kk++) sum += (int32_t)a[kk] * (int32_t)bb[kk];
            C[(size_t)r * 256u + c] += sum;
          }
        for (uint32_t t = 0; t < GP_THREADS; t++) {
          const uint32_t w = t >> 5, l = t & 31u, g = l >> 2, q = l & 3u;
          uint32_t x = 0;
          for (uint32_t mb = 0; mb < MT; mb++)
            for (uint32_t nb = 0; nb < 8u; nb++) {
              const uint32_t r = 16u * (MT * (w >> 2) + mb) + g, c = 64u * (w & 3u) + 8u * nb + 2u * q;
              x ^= (uint32_t)C[(size_t)r * 256u + c] ^ (uint32_t)C[(size_t)r * 256u + c + 1u]
                   ^ (uint32_t)C[(size_t)(r + 8u) * 256u + c] ^ (uint32_t)C[(size_t)(r + 8u) * 256u + c + 1u];
            }
          uint32_t &wd = tr[(size_t)t * 16u + (ch & 15u)];
          wd = rotl13(wd) ^ x;
        }
      }
      uint32_t bad_here = 0;
      for (uint32_t t = 0; t < GP_THREADS; t++) {
        total_tr++;
        if (memcmp(&tr[(size_t)t * 16u], &dump[((size_t)v * GP_THREADS + t) * 16u], 64) != 0) {
          if (bad < 4u)
            fprintf(stderr, "tile %u warp %u lane %u: cpu %08x %08x.. gpu %08x %08x..\n", v, t >> 5, t & 31u,
                    tr[t * 16u], tr[t * 16u + 1u], dump[((size_t)v * GP_THREADS + t) * 16u],
                    dump[((size_t)v * GP_THREADS + t) * 16u + 1u]);
          bad++;
          bad_here++;
        }
      }
      if (bad_here) {
        bad_tiles++;
        if (first_bad == 0xffffffffu) first_bad = v;
      }
    }
    printf(" | check %u/%u", total_tr - bad, total_tr);
    if (bad) printf(" (%u tiles with a bad transcript, the first tile %u)", bad_tiles, first_bad);
  }
  printf("\n");
  return 0;
}
