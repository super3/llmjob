// The host side of a NEB core (neb_host.h), written once for CUDA and HIP. The including file has
// already included the vendor's runtime and defined
//   G(x)          the vendor's name of runtime call or constant x: cuda##x or hip##x
//   NB_VENDOR     "nvidia" or "amd"
//   NB_PROP       the type of a card's properties
//   NB_MIN_OK(p)  whether a card with these device properties can run the core
//   NEB_DEV, NEB_HD, NEB_ATOMIC_INC  for neb_kernel.h
//   NB_SHARED_AT_LAUNCH(type, name)  declares a block's shared array whose size the launch gives
//   NB_ATOMIC_ADD64(p, x)  adds x to the 64-bit count at p, atomically
//   NEB_ATOMIC_ADD(p, x)   the same for a 32-bit count
// CUDA and HIP share the kernel syntax (__global__, blockIdx, the <<< >>> launch), so the kernels are here too.
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "neb_host.h"
#include "neb_kernel.h"
extern "C" {
#include "ref/nebhash2.h"
}

// Each of these can be set when building (-DNB_..._LOG2=n), to time one against another on a card.
#ifndef NB_THREADS_LOG2
#define NB_THREADS_LOG2 8
#endif
#ifndef NB_SCREEN_LOG2
#define NB_SCREEN_LOG2 20
#endif
// Nonces a thread of nbk_scan screens between two ticket steps (2^n). More means fewer barriers and a larger list
// in each block's shared memory: 40 bytes a thread a nonce.
#ifndef NB_ROUND_LOG2
#define NB_ROUND_LOG2 0
#endif
#define NB_ROUND (1u << NB_ROUND_LOG2)
// There are two scan kernels, and a model takes the one that suits its share of candidates: below NB_LEAN_BELOW
// candidates in a thousand nonces nbk_lean, from there up nbk_scan. 0 is never nbk_lean. Timed on the recorded
// models (4.1% and 25% candidates): an RTX 4090 screens 23.2 G nonces/s with nbk_lean against 20.7 on the first
// and 16.9 against 17.4 on the second; an RX 7900 XTX 11.8 against 10.1, and 8.5 against 9.0.
#ifndef NB_LEAN_BELOW
#define NB_LEAN_BELOW 150
#endif
// Threads of a launch of nbk_lean (2^n), each with NB_CHUNK / that many nonces; and the most candidates a block's
// list holds.
#ifndef NB_LEAN_LOG2
#define NB_LEAN_LOG2 18
#endif
#define NB_LEAN (1u << NB_LEAN_LOG2)
#define NB_CAP_MOST 4096u
// Sparse rows (neb_kernel.h, nebk_third): on a model that scans with nbk_lean, the live samples whose gradients have
// at most T entries (of KEP1's 26,000), T picked a model by nb_sparse_pick, are sparse rows, and an attempt with
// three live rows, one of them sparse, is evaluated from the pair table's digest of the other two and the sparse
// row's entries. NB_SPARSE_MAX bounds T (0: no sparse rows); NB_SPARSE_FORCE, when set, is T.
#ifndef NB_SPARSE_MAX
#define NB_SPARSE_MAX 8192u
#endif
// What nb_sparse_pick takes an attempt with a sparse third row to cost, in nonces screened: NB_SP_C0 / 100 for
// the record, its ChaCha8 again and the ticket, and NB_SP_C1 / 10,000 an entry it walks.
#ifndef NB_SP_C0
#define NB_SP_C0 130u
#endif
#ifndef NB_SP_C1
#define NB_SP_C1 65u
#endif
// A sparse row's description must fit a block's shared memory (words), and the gathered rows and buckets the card.
#define NB_META_MAX 11264u
#define NB_GT_MAX (3ull << 30)
#define NB_REC_MAX (512ull << 20)
// nbk_lean keeps the live bytes of the dataset's samples in each block's shared memory, a byte a sample, so a
// dataset of more samples than this scans with nbk_scan. The main network's have 13,404 and 7,089.
#define NB_LEAN_SAMPLES 16384u
// Threads a block.
#define NB_THREADS (1u << NB_THREADS_LOG2)
// Nonces one scan launch takes (2^n): about a millisecond on an RTX 4090. A scan queues all its launches and
// waits once, at the end.
#ifndef NB_CHUNK_LOG2
#define NB_CHUNK_LOG2 24
#endif
#define NB_CHUNK (1u << NB_CHUNK_LOG2)
// Threads of a launch of nbk_scan: each tries NB_CHUNK / NB_SCREEN nonces, so starting a thread costs little a nonce.
// Many more threads than a card runs at once: with only as many as fit (2^17 on an RTX 4090) the last blocks of
// a launch ran on half-empty SMs, and the screen was 11% slower than one thread a nonce.
#define NB_SCREEN (1u << NB_SCREEN_LOG2)
// Table entries one build launch takes: short enough for Windows' 2-second limit on a kernel on a slow card.
#define NB_BUILD_CHUNK (1u << 17)
#define NB_EVAL_CHUNK 4096u
// Card memory left free when a pair table is sized, and when the triple buffer is: that one leaves room for the
// next models' pair tables (450 MB for a Kepler model, and they grow and shrink) and the candidate lists.
#define NB_RESERVE (384ull << 20)
#define NB_TRIPLE_RESERVE (1280ull << 20)
// Rows a triple table can cover: its entries must number below 2^31 (neb_kernel.h, NEBK_TRIPLE).
#define NB_TRIPLE_ROWS 2000u
// Slabs one launch of the triple build takes: a launch must stay short of Windows' 2-second limit on a slow card.
#define NB_SLABS 8u

// ── kernels ─────────────────────────────────────────────────────────────────

#define NB_GID (blockIdx.x * blockDim.x + threadIdx.x)

// A block of the scan. Its threads go through their nonces in step: each screens NB_ROUND nonces and lists the
// candidates in the block's shared memory, then the block's threads share the candidates out and do their
// tickets, while the rest wait. So the tickets cost only the threads that have one, and no candidate leaves the
// SM. One counter and one list for the whole card were the limit on a model with many candidates: an RTX 4090
// took about 4 billion atomic adds a second and no more, and wrote and read back every candidate through its
// memory. `total` gets the block's count of candidates.
//
// Tried and lost: the live bytes copied into the block's shared memory, as nbk_lean has them. A block here
// has a few nonces a thread, and the copy cost more than the shorter look-ups saved: 16.9 G nonces/s against
// 17.8 on an RTX 4090 on the recorded TESS model.
__global__ void nbk_scan(NebScan S, const unsigned short *live, const unsigned char *alive, const nu32 *pairs,
                         const nu32 *triples, nu32 *hits, nu64 *total) {
  __shared__ nu32 n;
  __shared__ NebCand list[NB_THREADS * NB_ROUND];
  const nu32 gid = NB_GID, mask = NB_THREADS * NB_ROUND - 1u;
  if (threadIdx.x == 0) n = 0;
  nu32 seen = 0;
  for (nu32 it = 0; it < S.iters; it += NB_ROUND) {
    __syncthreads();  // every ticket of the last round is done: its list can be written over
    if (gid < S.threads) {
      for (nu32 r = 0; r < NB_ROUND && it + r < S.iters; r++) {
        NebCand c;
        if (nebk_screen(S.start + gid + (it + r) * S.threads, &S, live, alive, &c)) list[NEB_ATOMIC_INC(&n) & mask] = c;
      }
    }
    __syncthreads();
    const nu32 now = n;
    for (nu32 at = seen + threadIdx.x; at - seen < now - seen; at += NB_THREADS) nebk_finish(&S, list + (at & mask), pairs, triples, hits);
    seen = now;
  }
  if (threadIdx.x == 0 && seen) NB_ATOMIC_ADD64(total, (nu64)seen);
}

// The scan for a model with few candidates: one list a block, in the card's memory. Each thread screens all its
// nonces without waiting for anyone and lists the few that pass (the nonce and its key); then the block's threads
// share the list out, find each candidate's rows (from ChaCha8 of the key again) and do the tickets.
//
// What this is for: a warp's 32 threads step together, so whatever one thread does for a candidate, the other 31
// wait for. With a list a thread, a warp looked up a candidate's rows whenever any of its threads had one (three
// nonces in four on a model with 4% candidates) and did as many tickets as its luckiest thread had: an RTX 4090
// screened 24.9 G nonces/s on a model with 0.36% candidates and 20.5 on one with 4.1%. Shared out, the rows and
// tickets are done by full warps, for 400 instructions more a candidate. With many candidates every warp has one
// every nonce and nbk_scan's rounds are the better way.
//
// A list holds S.cap candidates, sized by the host from the model's share of them. A thread that finds it full
// does its candidate itself, there and then.
//
// The block copies the live bytes (one a sample, S.n of them) into shared memory first. A look-up there is one
// instruction; in the card's memory it is three, two of them for the 64-bit address, and a nonce has 16.
// An attempt with a sparse third row, into its row's bucket. A full bucket drops it: the attempt is then not tried,
// and not counted.
__device__ static void nb_bucket(const NebScan &S, nu32 nonce, const nu32 *key, const nu32 *u, const unsigned short *live,
                                 NebRec *recs, nu32 *bcount) {
  NebRec r;
  const nu32 k = nebk_third_rec(nonce, key, u, &S, live, &r);
  if (k == ~0u) return;
  const nu32 at = NEB_ATOMIC_INC(bcount + k);
  if (at < S.bcap) recs[(nu64)k * S.bcap + at] = r;
}

__global__ void nbk_lean(NebScan S, const unsigned short *live, const unsigned char *alive, const nu32 *pairs,
                         const nu32 *triples, NebCand *cand, NebRec *recs, nu32 *bcount, nu32 *hits, nu64 *total) {
  // Two lists: the candidates the tables answer alone, and the attempts with a sparse third row, which go to
  // their row's bucket for nbk_third.
  __shared__ nu32 n, n2, sum;
  NB_SHARED_AT_LAUNCH(unsigned char, bytes);
  if (threadIdx.x == 0) {
    n = 0;
    n2 = 0;
    sum = 0;
  }
  for (nu32 i = threadIdx.x; i < S.n; i += NB_THREADS) bytes[i] = alive[i];
  __syncthreads();
  const nu32 gid = NB_GID;
  NebCand *list = cand + (nu64)blockIdx.x * (S.cap + S.cap2), *list2 = list + S.cap;
  nu32 mine = 0;
  if (gid < S.threads) {
    for (nu32 it = 0; it < S.iters; it++) {
      const nu32 nonce = S.start + gid + it * S.threads;
      nu32 key[8], u[16];
      nebk_key(S.pre, S.tail, nonce, S.one, key);
      nebk_chacha8(key, S.one, u);
      const nu32 bsum = nebk_count(u, &S, bytes);
      if (!nebk_pass(bsum, &S)) continue;
      NebCand c;
      c.nonce = nonce;
      c.entry = 0;
      for (int i = 0; i < 8; i++) c.key[i] = key[i];
      if (bsum >= NEBK_SPARSE && nebk_sparse_kind(bsum) == 2u) {
        const nu32 at = NEB_ATOMIC_INC(&n2);
        if (at < S.cap2) list2[at] = c;
        else nb_bucket(S, nonce, key, u, live, recs, bcount);
        continue;
      }
      const nu32 at = NEB_ATOMIC_INC(&n);
      if (at < S.cap) {
        list[at] = c;
      } else if (nebk_cand(nonce, key, u, &S, live, &c)) {
        nebk_finish(&S, &c, pairs, triples, hits);
        mine++;
      }
    }
  }
  __syncthreads();
  const nu32 have = n < S.cap ? n : S.cap;
  for (nu32 at = threadIdx.x; at < have; at += NB_THREADS) {
    NebCand c = list[at];
    nu32 u[16];
    nebk_chacha8(c.key, S.one, u);
    if (!nebk_cand(c.nonce, c.key, u, &S, live, &c)) continue;
    nebk_finish(&S, &c, pairs, triples, hits);
    mine++;
  }
  const nu32 have2 = n2 < S.cap2 ? n2 : S.cap2;
  for (nu32 at = threadIdx.x; at < have2; at += NB_THREADS) {
    const NebCand c = list2[at];
    nu32 u[16];
    nebk_chacha8(c.key, S.one, u);
    nb_bucket(S, c.nonce, c.key, u, live, recs, bcount);
  }
  if (mine) NEB_ATOMIC_ADD(&sum, mine);
  __syncthreads();
  if (threadIdx.x == 0 && sum) NB_ATOMIC_ADD64(total, (nu64)sum);
}

// The buckets nbk_lean filled: S.bcap / NB_THREADS blocks a sparse row, block b of a row taking its records
// b * NB_THREADS on, a thread a record. The blocks of a row run one after another, so its gathered rows are read
// from the card's memory about once. The host clears the counts after.
__global__ void nbk_third(NebScan S, NebRows R, const nu32 *pairs, const signed char *d5, const nu32 *meta,
                          const nu32 *moff, const unsigned char *gt, const NebRec *recs, const nu32 *bcount, nu32 *hits,
                          nu64 *total) {
  NB_SHARED_AT_LAUNCH(uint4, M4);
  nu32 *const M = (nu32 *)M4;
  const nu32 per = S.bcap / NB_THREADS;
  const nu32 k = blockIdx.x / per, first = (blockIdx.x % per) * NB_THREADS;
  nu32 have = bcount[k];
  if (have > S.bcap) have = S.bcap;
  if (first >= have) return;
  const nu32 *src = meta + moff[k], words = moff[k + 1u] - moff[k];
  for (nu32 i = threadIdx.x; i < words; i += NB_THREADS) M[i] = src[i];
  __syncthreads();
  const nu32 at = first + threadIdx.x;
  if (at < have) {
    const NebRec r = recs[(nu64)k * S.bcap + at];
    nebk_third(&S, &R, &r, M, gt, pairs, d5, hits);
  }
  if (threadIdx.x == 0) NB_ATOMIC_ADD64(total, (nu64)((have - first) < NB_THREADS ? have - first : NB_THREADS));
}

// The gathered rows: a thread a row x of a sparse row k (k = gid / m1).
__global__ void nbk_gather(NebRows R, nu32 count, const nu32 *meta, const nu32 *moff, const short *conv,
                           const signed char *x5, unsigned char *gt) {
  const nu32 gid = NB_GID;
  if (gid >= count) return;
  const nu32 k = gid / R.m1, x = gid % R.m1;
  nebk_gather(x, &R, meta + moff[k], conv, x5, gt);
}

__global__ void nbk_pair(NebRows R, nu32 first, nu32 count, nu32 pairs, const short *conv, const nu32 *tag,
                         const signed char *d5, const signed char *x5, const nu32 *zero5, nu32 *table) {
  const nu32 gid = NB_GID;
  if (gid < count) nebk_pair(first + gid, pairs, &R, conv, tag, d5, x5, zero5, table);
}

__global__ void nbk_out1(NebRows R, nu32 count, const signed char *d5, const signed char *x5, nu32 *out1) {
  const nu32 gid = NB_GID;
  if (gid < count) nebk_out1(gid, &R, d5, x5, out1);
}

__global__ void nbk_out2(NebRows R, nu32 count, nu32 c, const signed char *d5, const signed char *x5, nu32 *out2) {
  const nu32 gid = NB_GID;
  if (gid < count) nebk_out2(gid, c, &R, d5, x5, out2);
}

__global__ void nbk_g2(NebRows R, nu32 first, nu32 count, const nu32 *mask, const nu32 *off2, const signed char *d5,
                       const signed char *x5, nu32 *g2) {
  const nu32 gid = NB_GID;
  if (gid < count) nebk_g2(first + gid, &R, mask, off2, d5, x5, g2);
}

// A slab's entries, a tile a block: NB_TILE rows a by NB_TILE rows b, each warp one b and each of its threads
// one a. Block t is the tile (ta, tb) = nebk_unpair(t): tiles with ta <= tb hold every a <= b.
//
// Why tiles: a build is as fast as the rows reach the SM, not as the card adds. With the entries taken in order
// every thread read a row a of its own from the card's memory, 3.6 KB an entry. In a tile the block's warps all
// read the same 32 rows a.
//
// NB_TRIPLE_NEAR (a vendor file sets it): the direct entries are the large part of a row (2.8 KB), so the block
// takes them NB_CONV_PART at a time. It copies that part of its 32 rows a into shared memory, once, and every
// warp sums from there, with its own row b and the slab's row c from the card's memory. Timed on an RTX 4090,
// a 22 GB table of Kepler rows: 88 entries a part 2.7 s, 176 3.0 s, 352 3.9 s, 704 5.6 s, 40 2.8 s, 16 3.2 s.
// On an RX 7900 XTX the build with the copy crashed inside the vendor's runtime, and going through the direct
// entries in steps without it changed nothing there, so AMD reads the rows where they are.
#define NB_TILE 32u
#ifndef NB_TRIPLE_NEAR
#define NB_TRIPLE_NEAR 0
#endif
// Direct entries a step: a multiple of 8.
#ifndef NB_CONV_PART
#define NB_CONV_PART 88u
#endif
// A launch takes several slabs of the same number of tiles: `each` blocks a slab, the first slab row `first`. A
// card runs one of these blocks an SM at a time, so with a slab a launch the SMs that finished their blocks
// first stood idle until the last was done, 1,600 times a table. `out2` holds each slab's table, `per` words
// apart; `table` is the whole triple table.
__global__ void nbk_triple(NebRows R, nu32 first, nu32 each, nu32 per, const short *conv, const nu32 *tag,
                           const nu32 *mask, const signed char *d5, const signed char *x5, const nu32 *zero5,
                           const nu32 *out1, const nu32 *out2, const nu32 *off2, const nu32 *g2, nu32 *table) {
  const nu32 at = blockIdx.x / each, c = first + at;
  out2 += (nu64)at * per;
  nu32 *slab = table + (nu64)nebk_entry3(0u, 0u, c - 1u) * 8u;
  nu32 ta, tb;
  nebk_unpair(blockIdx.x % each, &ta, &tb);
  const nu32 lane = threadIdx.x % NB_TILE, a = ta * NB_TILE + lane + 1u, b = tb * NB_TILE + threadIdx.x / NB_TILE + 1u;
  const bool on = a <= b && b <= c;
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if NB_TRIPLE_NEAR
  // The tile's 32 rows a, then each of its 32 rows b summed with the slab's row c: this part of them.
  __shared__ short part[2u * NB_TILE * NB_CONV_PART];
  __shared__ nu32 tags[NB_CONV_PART];
  const nu32 warp = threadIdx.x / NB_TILE;
  short *mine = part + lane * NB_CONV_PART, *sum = part + (NB_TILE + warp) * NB_CONV_PART;
  for (nu32 first = 0; first < R.nc; first += NB_CONV_PART) {
    const nu32 count = R.nc - first < NB_CONV_PART ? R.nc - first : NB_CONV_PART;
    // Row a of this thread's lane: one entry in NB_TILE of it, so the block's warps copy it between them. Row b
    // of this thread's warp plus row c: one entry in NB_TILE, so the warp's threads make it between them.
    if (a <= c) {
      const short *from = conv + (nu64)a * R.nc + first;
      for (nu32 k = warp; k < count; k += NB_TILE) mine[k] = from[k];
    }
    if (b <= c) {
      const short *fb = conv + (nu64)b * R.nc + first, *fc = conv + (nu64)c * R.nc + first;
      for (nu32 k = lane; k < count; k += NB_TILE) sum[k] = (short)(fb[k] + fc[k]);
    }
    if (threadIdx.x < count) tags[threadIdx.x] = tag[first + threadIdx.x];
    __syncthreads();
    if (on) nebk_triple_conv2(mine, sum, tags, count, R.nclip > first ? R.nclip - first : 0u, D);
    __syncthreads();
  }
#else
  if (on) nebk_triple_conv(conv + (nu64)a * R.nc, conv + (nu64)b * R.nc, conv + (nu64)c * R.nc, tag, R.nc, R.nclip, D);
#endif
  if (on) nebk_triple(a, b, c, &R, mask, d5, x5, zero5, out1, out2, off2, g2, D, slab);
}

__global__ void nbk_full(NebRows R, nu32 count, const nu32 *keys, const unsigned short *live, const short *conv,
                         const nu32 *tag, const signed char *d5, const signed char *x5, nu32 *out) {
  const nu32 gid = NB_GID;
  if (gid < count) nebk_full(keys + 8u * gid, &R, live, conv, tag, d5, x5, out + 16u * gid);
}

// ── the core ────────────────────────────────────────────────────────────────

struct NbSlot {
  int arch = 0;
  uint32_t n = 0;
  std::vector<uint8_t> values, labels;
  bool ready = false;
  NebRows R;
  uint32_t max_live = 0;
  uint32_t t_rows = 0;       // rows the triple table covers now
  uint64_t entries = 0;
  uint32_t live_rows = 0;
  uint32_t dense = 0;        // rows 1 .. dense are dense; the sparse rows follow (neb_kernel.h, nebk_third)
  // The sparse rows (nebk_third): their descriptions (d_meta, word moff[k] on for sparse row k), the gathered rows,
  // and a bucket of records each, bcap records long, done every `flush` launches of nbk_lean.
  uint32_t nsparse = 0, meta_max = 0, bcap = 0, flush = 0;
  double third = 0;          // the share of nonces with a sparse third row
  nu32 *d_meta = NULL, *d_moff = NULL, *d_bcount = NULL;
  unsigned char *d_gt = NULL;
  NebRec *d_recs = NULL;
  std::vector<nu32> mask;    // each row's outputs of the factored layer with an error, a bit each
  std::vector<unsigned short> live;   // each sample's row, 0 for a dead one
  uint32_t alive_rows = 0;   // the t_rows the card's live bytes (d_alive) were written for: see nebk_pass
  nu32 *d_mask = NULL;       // the same on the card, for the triple build
  unsigned short *d_live = NULL;
  unsigned char *d_alive = NULL;
  short *d_conv = NULL;
  nu32 *d_tag = NULL;
  signed char *d_d5 = NULL, *d_x5 = NULL;
  nu32 *d_zero5 = NULL, *d_table = NULL;
};

struct nb_core {
  int index;
  std::mutex mu;
  NbSlot slot[NB_SLOTS];
  nu64 *d_total;
  nu32 *d_hits, *d_keys, *d_out;
  NebCand *d_cand;       // nbk_lean: every block's lists of candidates
  uint64_t cand_bytes;
  // The triple table's buffer (and, after the table, what its build needs): one for the card, kept from one
  // model to the next. A slot's table is good for one
  // block (its next block is on a new model, and a block of the other slot comes between), so the slots take
  // turns in it, and each can have as large a table as the card holds. The slot whose table it holds has t_rows > 0.
  nu32 *d_tri;
  uint64_t tri_bytes;
};

typedef G(Error_t) nb_err;

static int nb_fail(char *err, int errlen, const char *what, nb_err e) {
  if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s: %s", what, G(GetErrorString)(e));
  G(GetLastError)();
  return -1;
}

static int nb_say(char *err, int errlen, const char *what) {
  if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s", what);
  return -1;
}

#define NB_TRY(call, what)                                      \
  do {                                                          \
    const nb_err e_ = (call);                                   \
    if (e_ != G(Success)) return nb_fail(err, errlen, what, e_); \
  } while (0)

// Set bits of a word, on the host.
static int NB_POPCOUNT(nu32 x) {
  int n = 0;
  for (; x; x &= x - 1u) n++;
  return n;
}

static double nb_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static nu32 nb_blocks(nu32 threads) { return (threads + NB_THREADS - 1) / NB_THREADS; }

extern "C" int nb_device_count(void) {
  int n = 0;
  if (G(GetDeviceCount)(&n) != G(Success)) {
    G(GetLastError)();
    return 0;
  }
  return n;
}

extern "C" int nb_device_info(int index, char *name, int len) {
  NB_PROP p;
  if (G(GetDeviceProperties)(&p, index) != G(Success)) {
    G(GetLastError)();
    return -1;
  }
  if (name && len > 0) snprintf(name, (size_t)len, "%s", p.name);
  return NB_MIN_OK(p) ? 0 : -1;
}

extern "C" const char *nb_vendor(void) { return NB_VENDOR; }

// Frees a slot's model. The card's triple buffer stays, for the next model.
static void nb_free_model(NbSlot *s) {
  G(Free)(s->d_live); G(Free)(s->d_alive); G(Free)(s->d_conv); G(Free)(s->d_tag);
  G(Free)(s->d_d5); G(Free)(s->d_x5); G(Free)(s->d_zero5); G(Free)(s->d_table);
  G(Free)(s->d_mask); G(Free)(s->d_meta); G(Free)(s->d_moff); G(Free)(s->d_bcount); G(Free)(s->d_gt); G(Free)(s->d_recs);
  s->d_mask = NULL; s->d_meta = NULL; s->d_moff = NULL; s->d_bcount = NULL; s->d_gt = NULL; s->d_recs = NULL;
  s->nsparse = 0;
  s->d_live = NULL; s->d_alive = NULL; s->d_conv = NULL; s->d_tag = NULL;
  s->d_d5 = NULL; s->d_x5 = NULL; s->d_zero5 = NULL; s->d_table = NULL;
  s->ready = false;
  s->t_rows = 0;
}

extern "C" void nb_close(nb_core *c) {
  if (!c) return;
  {
    std::lock_guard<std::mutex> lock(c->mu);
    if (G(SetDevice)(c->index) == G(Success)) {
      for (int i = 0; i < NB_SLOTS; i++) nb_free_model(&c->slot[i]);
      G(Free)(c->d_total); G(Free)(c->d_hits); G(Free)(c->d_keys); G(Free)(c->d_out); G(Free)(c->d_cand); G(Free)(c->d_tri);
    }
    G(GetLastError)();
  }
  delete c;
}

static int nb_open_buffers(nb_core *c, char *err, int errlen) {
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  NB_TRY(G(Malloc)((void **)&c->d_total, 8), "memory for the candidate count");
  NB_TRY(G(Malloc)((void **)&c->d_hits, ((size_t)NB_MAX_HITS + 1) * 4), "memory for the hits");
  NB_TRY(G(Malloc)((void **)&c->d_keys, (size_t)NB_EVAL_CHUNK * 32), "memory for the keys");
  NB_TRY(G(Malloc)((void **)&c->d_out, (size_t)NB_EVAL_CHUNK * 64), "memory for the tickets");
  return 0;
}

extern "C" nb_core *nb_open(int index, char *err, int errlen) {
  nb_core *c = new nb_core();
  c->index = index;
  c->d_hits = c->d_keys = c->d_out = NULL;
  c->d_total = NULL;
  c->d_cand = NULL;
  c->cand_bytes = 0;
  c->d_tri = NULL;
  c->tri_bytes = 0;
  if (nb_open_buffers(c, err, errlen) != 0) {
    nb_close(c);
    return NULL;
  }
  return c;
}

extern "C" int nb_set_dataset(nb_core *c, int slot, int arch, uint32_t n, const uint8_t *values,
                              const uint8_t *labels, char *err, int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  const neb2_arch *A = neb2_arch_get(arch);
  // KEP1 only: a sample's direct gradients fit int16 there (583 * 8 * 7 = 32,648), and the main network has no other.
  if (!A || arch != NEB2_KEP1) return nb_say(err, errlen, "this core runs the KEP1 network only");
  if (n < 1 || n > 65534u) return nb_say(err, errlen, "a dataset of 1 to 65,534 samples");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (G(SetDevice)(c->index) == G(Success)) nb_free_model(s);
  s->arch = arch;
  s->n = n;
  s->values.assign(values, values + (size_t)n * A->nin);
  s->labels.assign(labels, labels + n);
  return 0;
}

// One live sample's row on the host.
struct NbRow {
  std::vector<int16_t> conv;
  std::vector<int8_t> d5, x5;
  uint32_t entries = 0;   // entries of its gradient that are not zero
};

template <typename T>
static int nb_upload(T **dst, const void *src, size_t bytes, char *err, int errlen, const char *what) {
  NB_TRY(G(Malloc)((void **)dst, bytes ? bytes : 1), what);
  if (bytes) NB_TRY(G(Memcpy)(*dst, src, bytes, G(MemcpyHostToDevice)), what);
  return 0;
}

// Entries of a triple table over `rows` rows.
static uint64_t nb_triple_entries(uint64_t rows) { return rows * (rows + 1) * (rows + 2) / 6; }

// The most rows a triple table of at most `bytes` covers.
static uint32_t nb_triple_rows(uint64_t bytes, uint32_t limit) {
  uint32_t lo = 0, hi = limit;
  while (lo < hi) {
    const uint32_t mid = (lo + hi + 1) / 2;
    if (nb_triple_entries(mid) * 32 <= bytes) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// A live row's shape as a sparse row (neb_kernel.h, nebk_third): chunks of 8 slots a run, and what it costs.
struct NbShape {
  uint32_t cl[8] = {0}, nl[8] = {0}, jl[8] = {0};   // chunks: direct entries that can clip, that cannot; inputs
  uint32_t pc = 0, pj = 0, ne = 0;
  uint32_t words = 0;      // its description, in words
  uint32_t stride = 0;     // bytes of a gathered row
  double work = 0;         // slots walked an attempt, a clipped one counted as 1.5
};

// Whether a sparse row's value v at direct entry e (take: its place in a row) can make three rows' sum pass the clip.
static bool nb_can_clip(int v, uint32_t e, const std::vector<uint32_t> &take, const std::vector<int> &most) {
  return (v < 0 ? -v : v) + 2 * most[take[e]] > NEBK_CLIP;
}

static void nb_shape(const NbRow &r, const std::vector<uint32_t> &take, const std::vector<int> &most, uint32_t nc,
                     uint32_t cin, NbShape *sh) {
  uint32_t a[8] = {0}, b[8] = {0}, in[8] = {0};
  for (uint32_t e = 0; e < nc; e++) {
    const int v = r.conv[take[e]];
    if (!v) continue;
    if (nb_can_clip(v, e, take, most)) a[e & 7]++;
    else b[e & 7]++;
  }
  for (int8_t d : r.d5) sh->ne += d != 0;
  for (uint32_t j = 0; j < cin; j++) in[j & 7] += r.x5[j] != 0;
  uint32_t clipped = 0;
  for (int L = 0; L < 8; L++) {
    sh->cl[L] = (a[L] + 7) / 8;
    sh->nl[L] = (b[L] + 7) / 8;
    sh->jl[L] = sh->ne ? (in[L] + 7) / 8 : 0;
    clipped += 8 * sh->cl[L];
    sh->pc += 8 * (sh->cl[L] + sh->nl[L]);
    sh->pj += 8 * sh->jl[L];
  }
  sh->words = NEBK_M_HEAD + 3 * sh->pc + ((sh->ne + 3) & ~3u) + 2 * sh->pj;
  sh->stride = (2 * sh->pc + sh->pj + 15) & ~15u;
  sh->work = sh->pc + 0.5 * clipped + (double)sh->pj * sh->ne;
}

// Appends row r's description (neb_kernel.h, before nebk_third) to meta; its gathered rows start `at` bytes into gt.
static void nb_describe(const NbRow &r, const NbShape &sh, const std::vector<uint32_t> &take, const std::vector<int> &most,
                        uint32_t nc, uint32_t cin, uint32_t cout, const std::vector<nu32> &tag, uint64_t at,
                        std::vector<nu32> *meta) {
  const size_t m0 = meta->size();
  meta->resize(m0 + sh.words, 0);
  nu32 *M = meta->data() + m0;
  for (int L = 0; L < 8; L++) {
    M[L] = sh.cl[L];
    M[8 + L] = sh.nl[L];
    M[16 + L] = sh.jl[L];
  }
  M[NEBK_M_NE] = sh.ne;
  M[NEBK_M_STRIDE] = sh.stride;
  M[NEBK_M_AT] = (nu32)at;
  M[NEBK_M_AT + 1] = (nu32)(at >> 32);
  M[NEBK_M_PC] = sh.pc;
  M[NEBK_M_PJ] = sh.pj;
  nu32 *V = M + NEBK_M_HEAD, *T = V + sh.pc, *C = T + sh.pc;
  uint32_t slot = 0;
  for (int pass = 0; pass < 2; pass++)
    for (uint32_t L = 0; L < 8; L++) {
      const uint32_t end = slot + 8 * (pass ? sh.nl[L] : sh.cl[L]);
      for (uint32_t e = L; e < nc; e += 8) {
        const int v = r.conv[take[e]];
        if (!v || nb_can_clip(v, e, take, most) != (pass == 0)) continue;
        V[slot] = (nu32)v;
        T[slot] = tag[e];
        C[slot] = e;
        slot++;
      }
      // The run's padding: s's value 0 at an entry of the same lane adds nothing.
      for (; slot < end; slot++) {
        V[slot] = 0;
        T[slot] = tag[L];
        C[slot] = L;
      }
    }
  nu32 *E = M + nebk_m_e(sh.pc);
  uint32_t ne = 0;
  for (uint32_t o = 0; o < cout; o++)
    if (r.d5[o] != 0) E[ne++] = ((nu32)(int32_t)r.d5[o] << 16) | o;
  nu32 *J = E + ((sh.ne + 3) & ~3u), *XS = J + sh.pj;
  uint32_t fs = 0;
  for (uint32_t L = 0; L < 8; L++) {
    const uint32_t end = fs + 8 * sh.jl[L];
    for (uint32_t j = L; j < cin && fs < end; j += 8) {
      if (!r.x5[j]) continue;
      J[fs] = j;
      XS[fs] = (nu32)(int32_t)r.x5[j];
      fs++;
    }
    for (; fs < end; fs++) {
      J[fs] = L;
      XS[fs] = 0;
    }
  }
}

// The most entries a sparse row may have on this model: the T that nb_sparse_pick takes to give the highest rate,
// by what NB_SP_C0 and NB_SP_C1 say an attempt with a sparse third row costs. 0: no sparse rows. Only models that
// scan with nbk_lean have them. *third gets the share of nonces whose attempt has a sparse third row.
static nu32 nb_sparse_pick(const std::vector<std::unique_ptr<NbRow>> &rows, const std::vector<NbShape> &shape, uint32_t n,
                           uint32_t nlive, uint32_t m1, uint64_t budget, double *third) {
  *third = 0;
  if (NB_SPARSE_MAX == 0 || n > NB_LEAN_SAMPLES || nlive < 3) return 0;
  const double p = (double)nlive / n, q = 1 - p;
  const double base = pow(q, 16) + 16 * p * pow(q, 15) + 120 * p * p * pow(q, 14);
  if (base * 1000 >= NB_LEAN_BELOW) return 0;
  // Attempts with exactly 3 live rows.
  const double three = 560 * p * p * p * pow(q, 13);
  std::vector<uint32_t> order;
  for (uint32_t i = 0; i < n; i++)
    if (rows[i]) order.push_back(i);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return rows[a]->entries < rows[b]->entries; });
  double best = base, work = 0, pick_share = 0;
  uint64_t gt = 0;
  nu32 pick = 0;
  for (uint32_t k = 0; k < order.size(); k++) {
    const NbRow &r = *rows[order[k]];
    const NbShape &sh = shape[order[k]];
    if (r.entries > NB_SPARSE_MAX || sh.words > NB_META_MAX) break;
    gt += (uint64_t)m1 * sh.stride;
    if (gt > budget || gt > NB_GT_MAX) break;
    work += sh.work;
    if (k + 1 < order.size() && rows[order[k + 1]]->entries == r.entries) continue;
    const double kept = (double)(k + 1) / nlive;
    const double extra = three * (1 - pow(1 - kept, 3));
    const double rate = (base + extra) / (1 + extra * (NB_SP_C0 / 100.0 + NB_SP_C1 / 10000.0 * work / (k + 1)));
#ifdef NB_SPARSE_FORCE
    const bool take = r.entries <= NB_SPARSE_FORCE;
#else
    const bool take = rate > best;
#endif
    if (take) {
      best = rate;
      pick = r.entries;
      pick_share = extra;
    }
  }
  *third = pick_share;
  return pick;
}

extern "C" int nb_set_model(nb_core *c, int slot, const int8_t *weights, uint32_t nweights, nb_model_info *info,
                            char *err, int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (!s->n) return nb_say(err, errlen, "the slot has no dataset");
  const neb2_arch *A = neb2_arch_get(s->arch);
  if (nweights != (uint32_t)A->nweights) return nb_say(err, errlen, "a model of the wrong size for this network");
  for (uint32_t i = 0; i < nweights; i++)
    if (weights[i] < -8 || weights[i] > 7) return nb_say(err, errlen, "a weight outside -8..7");

  // The factored layer is the largest linear one; every other layer's entries are kept directly.
  int lf = -1;
  for (int i = 0; i < A->nlayers; i++)
    if (A->l[i].kind == NEB2_LINEAR && (lf < 0 || A->l[i].nw > A->l[lf].nw)) lf = i;
  const neb2_layer &F = A->l[lf];
  if (F.cin % 8 != 0) return nb_say(err, errlen, "a factored layer whose width is not a multiple of 8");
  // The kernels keep a bit an output of the factored layer in one word.
  if (F.cout > 32) return nb_say(err, errlen, "a factored layer of more than 32 outputs");
  if (A->l[A->nlayers - 1].kind != NEB2_LINEAR) return nb_say(err, errlen, "a network whose last layer is not linear");
  // Where the factored layer's errors and inputs sit in neb2_sample_grad's factors, and the last layer's errors.
  int fac_at = 0, fac_last = 0, at = 0;
  for (int i = 0; i < A->nlayers; i++) {
    if (A->l[i].kind != NEB2_LINEAR) continue;
    if (i == lf) fac_at = at;
    if (i == A->nlayers - 1) fac_last = at;
    at += A->l[i].cout + A->l[i].cin;
  }
  std::vector<uint32_t> src;     // model index of each direct entry
  std::vector<nu32> tag;
  for (int i = 0; i < A->nlayers; i++) {
    if (i == lf) continue;
    // The kernels take entry e of this list as lane e mod 8, which holds while every layer before it is a multiple of 8.
    if (A->l[i].nw % 8 != 0) return nb_say(err, errlen, "a layer whose size is not a multiple of 8");
    for (int f = 0; f < A->l[i].nw; f++) {
      src.push_back((uint32_t)(A->l[i].w_off + f));
      tag.push_back((nu32)f + 7919u * (nu32)(i + 1));
    }
  }
  const uint32_t nc = (uint32_t)src.size();

  // Every sample once, on the host's threads, with the chain's reference.
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<neb2_model> m(new neb2_model());
  memset(m.get(), 0, sizeof(neb2_model));
  m->arch = s->arch;
  memcpy(m->w, weights, nweights);
  std::vector<std::unique_ptr<NbRow>> rows(s->n);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  if (nt > 32) nt = 32;
  std::vector<int> bad(nt, 0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
      std::vector<int8_t> fac(2 * NEB2_MAX_WEIGHTS);
      for (uint32_t i = t; i < s->n; i += nt) {
        if (neb2_sample_grad(m.get(), s->values.data() + (size_t)i * A->nin, s->labels[i], g.data(), fac.data()) != 0) {
          bad[t] = 1;
          return;
        }
        bool live = false;
        for (int o = 0; o < A->l[A->nlayers - 1].cout; o++) live = live || fac[fac_last + o] != 0;
        if (!live) continue;
        std::unique_ptr<NbRow> r(new NbRow());
        r->conv.resize(nc);
        for (uint32_t e = 0; e < nc; e++) {
          const int32_t v = g[src[e]];
          if (v < -32768 || v > 32767) bad[t] = 2;
          r->conv[e] = (int16_t)v;
        }
        r->d5.assign(fac.begin() + fac_at, fac.begin() + fac_at + F.cout);
        r->x5.assign(fac.begin() + fac_at + F.cout, fac.begin() + fac_at + F.cout + F.cin);
        // The factored layer's gradient is its errors times its inputs, so its entries are counted from those.
        uint32_t errs = 0, ins = 0;
        for (int8_t d : r->d5) errs += d != 0;
        for (int8_t x : r->x5) ins += x != 0;
        r->entries = errs * ins;
        for (int16_t v : r->conv) r->entries += v != 0;
        rows[i] = std::move(r);
      }
    });
  for (auto &t : th) t.join();
  for (int b : bad) {
    if (b == 1) return nb_say(err, errlen, "the reference could not evaluate a sample");
    if (b == 2) return nb_say(err, errlen, "a sample's gradient does not fit 16 bits");
  }
  // The direct entries in a new order, each kept in its lane (its place mod 8): first the ones where the sum of
  // three rows can pass the clip (three times the largest value any row has there is over 256), so that the
  // triple build clips only those. take[e] is the entry now at place e.
  uint32_t nlive = 0;
  for (uint32_t i = 0; i < s->n; i++) nlive += rows[i] ? 1 : 0;
  std::vector<uint32_t> take(nc);
  std::vector<int> most(nc, 0);
  uint32_t nclip = 0;
  {
    for (uint32_t i = 0; i < s->n; i++) {
      if (!rows[i]) continue;
      for (uint32_t e = 0; e < nc; e++) {
        const int v = rows[i]->conv[e] < 0 ? -rows[i]->conv[e] : rows[i]->conv[e];
        if (v > most[e]) most[e] = v;
      }
    }
    for (uint32_t lane = 0; lane < 8; lane++) {
      uint32_t at = lane, clips = 0;
      for (int pass = 0; pass < 2; pass++)
        for (uint32_t e = lane; e < nc; e += 8) {
          const bool can = 3 * most[e] > NEBK_CLIP;
          if (can != (pass == 0)) continue;
          take[at] = e;
          at += 8;
          clips += can ? 8 : 0;
        }
      if (clips > nclip) nclip = clips;
    }
    std::vector<nu32> moved(nc);
    for (uint32_t e = 0; e < nc; e++) moved[e] = tag[take[e]];
    tag.swap(moved);
  }
  // Which live samples are sparse rows (neb_kernel.h, nebk_third): the ones with at most sparse_max gradient entries.
  // Each live sample's shape as one, for nb_sparse_pick and the descriptions.
  std::vector<NbShape> shape(s->n);
  for (uint32_t i = 0; i < s->n; i++)
    if (rows[i]) nb_shape(*rows[i], take, most, nc, (uint32_t)F.cin, &shape[i]);
  const uint32_t m1 = nlive + 1;
  // The pair table is over every live row (and row 0).
  const uint64_t pair_entries = (uint64_t)m1 * (m1 + 1) / 2;
  size_t free_b = 0, total_b = 0;
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  NB_TRY(G(MemGetInfo)(&free_b, &total_b), "reading the card's free memory");
  double third = 0;
  nu32 sparse_max = nb_sparse_pick(rows, shape, s->n, nlive, m1, total_b / 8, &third);
  // Row 0 is the zero row; live samples take rows 1 and up in dataset order, the dense rows first and then the
  // sparse ones. (A triple table covers the first rows, and the host's check of a core knows which by that order.
  // Numbering the rows with the fewest outputs of the factored layer in error first made a Kepler table cheaper to
  // build, by about half a second in 150.)
  std::vector<unsigned short> live(s->n, 0);
  uint32_t dense = 0;
  for (uint32_t i = 0; i < s->n; i++)
    if (rows[i] && (!sparse_max || rows[i]->entries > sparse_max)) live[i] = (unsigned short)(++dense);
  uint32_t numbered = dense;
  for (uint32_t i = 0; i < s->n; i++)
    if (rows[i] && sparse_max && rows[i]->entries <= sparse_max) live[i] = (unsigned short)(++numbered);
  uint32_t nsparse = nlive - dense;
  // The sparse rows' descriptions, and where each one's gathered rows go.
  std::vector<nu32> meta, moff(1, 0);
  uint64_t gt_bytes = 0;
  uint32_t meta_max = 0;
  for (uint32_t k = 0; k < nsparse; k++) {
    uint32_t i = 0;
    while (live[i] != dense + 1 + k) i++;
    const uint32_t at = (uint32_t)meta.size();
    nb_describe(*rows[i], shape[i], take, most, nc, (uint32_t)F.cin, (uint32_t)F.cout, tag, gt_bytes, &meta);
    gt_bytes += (uint64_t)m1 * shape[i].stride;
    moff.push_back((nu32)meta.size());
    if (meta.size() - at > meta_max) meta_max = (uint32_t)(meta.size() - at);
  }
  // The buckets: about NB_CHUNK * third / nsparse records a launch each; `flush` launches fill one to 4 in 5.
  uint32_t bcap = 0, flush = 0;
  if (nsparse) {
    const double each = (double)NB_CHUNK * third / nsparse;
    flush = (uint32_t)((double)NB_REC_MAX / ((double)nsparse * sizeof(NebRec) * (each * 1.25 + 4 * sqrt(each) + NB_THREADS)));
    if (flush > 32) flush = 32;
    if (flush < 1) flush = 1;
    const double want = flush * each * 1.25 + 4 * sqrt(flush * each) + NB_THREADS;
    bcap = (uint32_t)((want + NB_THREADS - 1) / NB_THREADS) * NB_THREADS;
  }
  const uint64_t rec_bytes = (uint64_t)nsparse * bcap * sizeof(NebRec);
  std::vector<unsigned char> alive(s->n, 0);
  for (uint32_t i = 0; i < s->n; i++) alive[i] = !live[i] ? 0 : (live[i] > dense ? NEBK_SPARSE : 1);
  std::vector<int16_t> conv((size_t)m1 * nc, 0);
  std::vector<int8_t> d5((size_t)m1 * F.cout, 0), x5((size_t)m1 * F.cin, 0);
  for (uint32_t i = 0; i < s->n; i++) {
    if (!rows[i]) continue;
    const uint32_t r = live[i];
    for (uint32_t e = 0; e < nc; e++) conv[(size_t)r * nc + e] = rows[i]->conv[take[e]];
    memcpy(d5.data() + (size_t)r * F.cout, rows[i]->d5.data(), (size_t)F.cout);
    memcpy(x5.data() + (size_t)r * F.cin, rows[i]->x5.data(), (size_t)F.cin);
    rows[i].reset();
  }
  std::vector<nu32> mask(m1, 0);
  for (uint32_t r = 0; r < m1; r++)
    for (int o = 0; o < F.cout; o++)
      if (d5[(size_t)r * F.cout + o] != 0) mask[r] |= 1u << o;
  const nu32 base = 7919u * (nu32)(lf + 1);
  std::vector<nu32> zero5((size_t)F.cout * 8, 0);
  for (int o = 0; o < F.cout; o++)
    for (int j = 0; j < F.cin; j++) zero5[(size_t)o * 8 + (j & 7)] += (nu32)(o * F.cin + j) + base;
  const double cpu_ms = nb_ms(t0);

  // The card: the rows, then the pair table.
  const auto t1 = std::chrono::steady_clock::now();
  nb_free_model(s);
  NB_TRY(G(MemGetInfo)(&free_b, &total_b), "reading the card's free memory");
  const auto fits = [&](uint64_t extra) {
    return pair_entries < 0xffffffffull && free_b > NB_RESERVE && pair_entries * 32 + extra <= free_b - NB_RESERVE;
  };
  if (!fits(gt_bytes + rec_bytes) && c->d_tri) {
    // The pair table comes first: it is every attempt a card evaluates without the triple table, and that
    // table's buffer has taken the room. The buffer goes (whichever slot's table it held is gone with it) and
    // the next triple build takes a smaller one. Seen on an RTX 4090: a Kepler model 40 rows larger than the
    // last found 22 GB held and mined with no pair table, at a fifth of its rate.
    G(Free)(c->d_tri);
    c->d_tri = NULL;
    c->tri_bytes = 0;
    for (int i = 0; i < NB_SLOTS; i++) c->slot[i].t_rows = 0;
    NB_TRY(G(MemGetInfo)(&free_b, &total_b), "reading the card's free memory");
  }
  if (nsparse && !fits(gt_bytes + rec_bytes)) {
    // No room for the sparse rows' parts: every row is dense (the numbering stays a numbering).
    nsparse = 0;
    sparse_max = 0;
    third = 0;
    dense = nlive;
    for (uint32_t i = 0; i < s->n; i++) alive[i] = live[i] ? 1 : 0;
  }
  const bool pairs = fits(nsparse ? gt_bytes + rec_bytes : 0);
  if (!pairs && nsparse) {
    nsparse = 0;
    sparse_max = 0;
    third = 0;
    dense = nlive;
    for (uint32_t i = 0; i < s->n; i++) alive[i] = live[i] ? 1 : 0;
  }
  const uint64_t entries = pairs ? pair_entries : m1;
  NebRows R;
  R.m1 = m1; R.nc = nc; R.cin = (nu32)F.cin; R.cout = (nu32)F.cout; R.base = base; R.n = s->n; R.nclip = nclip;
  if (nb_upload(&s->d_live, live.data(), live.size() * 2, err, errlen, "memory for the live list") != 0 ||
      nb_upload(&s->d_alive, alive.data(), alive.size(), err, errlen, "memory for the live list") != 0 ||
      nb_upload(&s->d_conv, conv.data(), conv.size() * 2, err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_tag, tag.data(), tag.size() * 4, err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_d5, d5.data(), d5.size(), err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_x5, x5.data(), x5.size(), err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_zero5, zero5.data(), zero5.size() * 4, err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_mask, mask.data(), mask.size() * 4, err, errlen, "memory for the rows") != 0) {
    nb_free_model(s);
    return -1;
  }
  if (nsparse) {
    if (nb_upload(&s->d_meta, meta.data(), meta.size() * 4, err, errlen, "memory for the sparse rows") != 0 ||
        nb_upload(&s->d_moff, moff.data(), moff.size() * 4, err, errlen, "memory for the sparse rows") != 0) {
      nb_free_model(s);
      return -1;
    }
    nb_err e = G(Malloc)((void **)&s->d_gt, (size_t)gt_bytes);
    if (e == G(Success)) e = G(Malloc)((void **)&s->d_recs, (size_t)rec_bytes);
    if (e == G(Success)) e = G(Malloc)((void **)&s->d_bcount, (size_t)nsparse * 4);
    if (e == G(Success)) e = G(Memset)(s->d_bcount, 0, (size_t)nsparse * 4);
    if (e != G(Success)) {
      nb_free_model(s);
      return nb_fail(err, errlen, "memory for the sparse rows", e);
    }
    const nu32 count = nsparse * m1;
    nbk_gather<<<nb_blocks(count), NB_THREADS>>>(R, count, s->d_meta, s->d_moff, s->d_conv, s->d_x5, s->d_gt);
    e = G(GetLastError)();
    if (e != G(Success)) {
      nb_free_model(s);
      return nb_fail(err, errlen, "gathering the sparse rows", e);
    }
  }
  nb_err e = G(Malloc)((void **)&s->d_table, (size_t)entries * 32);
  if (e != G(Success)) {
    nb_free_model(s);
    return nb_fail(err, errlen, "memory for the model's table", e);
  }
  for (uint64_t done = 0; done < entries; done += NB_BUILD_CHUNK) {
    const nu32 part = (nu32)(entries - done < NB_BUILD_CHUNK ? entries - done : NB_BUILD_CHUNK);
    nbk_pair<<<nb_blocks(part), NB_THREADS>>>(R, (nu32)done, part, pairs ? 1u : 0u, s->d_conv, s->d_tag, s->d_d5,
                                              s->d_x5, s->d_zero5, s->d_table);
    e = G(GetLastError)();
    if (e == G(Success) && (done / NB_BUILD_CHUNK) % 8 == 7) e = G(DeviceSynchronize)();
    if (e != G(Success)) {
      nb_free_model(s);
      return nb_fail(err, errlen, "building the model's table", e);
    }
  }
  e = G(DeviceSynchronize)();
  if (e != G(Success)) {
    nb_free_model(s);
    return nb_fail(err, errlen, "building the model's table", e);
  }
  s->R = R;
  s->mask.swap(mask);
  s->live.swap(live);
  s->alive_rows = 0;
  s->max_live = pairs ? 2 : 1;
  s->entries = entries;
  s->live_rows = nlive;
  s->dense = dense;
  s->nsparse = nsparse;
  s->meta_max = meta_max;
  s->bcap = bcap;
  s->flush = flush;
  s->third = third;
  s->ready = true;
  if (info) {
    memset(info, 0, sizeof *info);
    info->samples = s->n;
    info->live = nlive;
    info->sparse = nsparse;
    info->sparse_max = sparse_max;
    info->max_live = s->max_live;
    info->entries = entries;
    info->table_bytes = entries * 32;
    info->cpu_ms = cpu_ms;
    info->gpu_ms = nb_ms(t1);
    if (G(MemGetInfo)(&free_b, &total_b) == G(Success)) {
      info->free_bytes = free_b;
      info->total_bytes = total_b;
      info->triple_bytes = c->tri_bytes;
    }
    G(GetLastError)();
  }
  return 0;
}

extern "C" int nb_build_triples(nb_core *c, int slot, uint32_t rows, uint64_t max_bytes, nb_triple_info *info,
                                char *err, int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (!s->ready) return nb_say(err, errlen, "the slot has no model");
  memset(info, 0, sizeof *info);
  if (s->max_live < 2) return 0;  // no pair table: the card has no room for triples either
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  // Whether the buffer holds this slot's table now: asking for no table then gives its memory back to the card.
  const bool mine = s->t_rows > 0;
  s->t_rows = 0;
  const NebRows &R = s->R;
  if (rows > s->dense) rows = s->dense;
  if (rows > NB_TRIPLE_ROWS) rows = NB_TRIPLE_ROWS;
  rows = nb_triple_rows(max_bytes, rows);
  if (!rows) {
    bool used = false;
    for (int i = 0; i < NB_SLOTS; i++) used = used || c->slot[i].t_rows > 0;
    if (mine || !used) {
      G(Free)(c->d_tri);
      c->d_tri = NULL;
      c->tri_bytes = 0;
    }
    return 0;
  }
  // The buffer is this slot's from here: the table another slot had in it is gone.
  for (int i = 0; i < NB_SLOTS; i++) c->slot[i].t_rows = 0;
  // What the build needs beside the table, for the first r rows: the pairs' shared outputs (g2, 32 bytes for each
  // output both rows of a pair have an error at), where each pair's start in it (off2), out1 for those rows and
  // out2 for each slab of a launch. They go in the same buffer, after the table: one piece of the card's memory
  // for all of it, kept from one model to the next. Kept apart, a TESS model that came after a Kepler table of
  // 22 GB found no room for them beside the buffer and built a table of 644 of its 1,474 rows.
  std::vector<uint64_t> shared(rows + 1, 0);   // shared[r]: outputs in g2 for the first r rows
  for (uint32_t b0 = 0; b0 < rows; b0++) {
    uint64_t n = 0;
    const nu32 mb = s->mask[b0 + 1];
    for (uint32_t a0 = 0; a0 <= b0; a0++) n += (uint64_t)NB_POPCOUNT(s->mask[a0 + 1] & mb);
    shared[b0 + 1] = shared[b0] + n;
  }
  // Words (of 4 bytes) of each, for r rows.
  const auto words_off2 = [](uint64_t r) { return r * (r + 1) / 2 + 1; };
  const auto words_g2 = [&](uint64_t r) { return shared[r] * 8 + 8; };
  const auto words_out = [&](uint64_t r) { return (r + 1) * R.cout * 8; };
  const auto need = [&](uint32_t r) {
    return (nb_triple_entries(r) * 8 + words_off2(r) + words_g2(r) + (1 + NB_SLABS) * words_out(r)) * 4;
  };
  // The buffer from the last table does if it is large enough, or nearly: a table of 95% of the rows asked for
  // in the buffer the card has beats waiting for a new one. Otherwise a new one, the rows cut back until it fits
  // the card.
  if (c->d_tri && need(rows) > c->tri_bytes) {
    uint32_t fit = rows;
    while (fit && need(fit) > c->tri_bytes) fit--;
    if ((uint64_t)fit * 100 >= (uint64_t)rows * 95) rows = fit;
  }
  if (need(rows) > c->tri_bytes) {
    G(Free)(c->d_tri);
    c->d_tri = NULL;
    c->tri_bytes = 0;
    size_t free_b = 0, total_b = 0;
    NB_TRY(G(MemGetInfo)(&free_b, &total_b), "reading the card's free memory");
    const uint64_t room = free_b > NB_TRIPLE_RESERVE ? free_b - NB_TRIPLE_RESERVE : 0;
    while (rows && need(rows) > room) rows--;
    if (!rows) return 0;
    NB_TRY(G(Malloc)((void **)&c->d_tri, (size_t)need(rows)), "memory for the triple table");
    c->tri_bytes = need(rows);
  }
  const uint64_t npairs = (uint64_t)rows * (rows + 1) / 2;
  const nu32 per = (nu32)words_out(rows);
  nu32 *const d_off2 = c->d_tri + nb_triple_entries(rows) * 8, *const d_g2 = d_off2 + words_off2(rows);
  nu32 *const d_out1 = d_g2 + words_g2(rows), *const d_out2 = d_out1 + per;
  {
    std::vector<nu32> off2(npairs + 1);
    uint64_t at = 0, e = 0;
    for (uint32_t b0 = 0; b0 < rows; b0++) {
      const nu32 mb = s->mask[b0 + 1];
      for (uint32_t a0 = 0; a0 <= b0; a0++) {
        off2[e++] = (nu32)at;
        at += (uint64_t)NB_POPCOUNT(s->mask[a0 + 1] & mb);
      }
    }
    off2[npairs] = (nu32)at;
    NB_TRY(G(Memcpy)(d_off2, off2.data(), off2.size() * 4, G(MemcpyHostToDevice)), "building the triple table");
  }
  // The time reported is the build's, from here: a card's first table also waits for its buffer (12 s for 22 GB
  // on an RX 7900 XTX through WSL), once, and the engine plans the next build from this figure.
  NB_TRY(G(DeviceSynchronize)(), "building the triple table");
  const auto t0 = std::chrono::steady_clock::now();
  nbk_out1<<<nb_blocks((rows + 1) * R.cout), NB_THREADS>>>(R, (rows + 1) * R.cout, s->d_d5, s->d_x5, d_out1);
  NB_TRY(G(GetLastError)(), "building the triple table");
  for (uint64_t done = 0; done < npairs; done += NB_BUILD_CHUNK) {
    const nu32 part = (nu32)(npairs - done < NB_BUILD_CHUNK ? npairs - done : NB_BUILD_CHUNK);
    nbk_g2<<<nb_blocks(part), NB_THREADS>>>(R, (nu32)done, part, s->d_mask, d_off2, s->d_d5, s->d_x5, d_g2);
    NB_TRY(G(GetLastError)(), "building the triple table");
  }
  NB_TRY(G(DeviceSynchronize)(), "building the triple table");
  // Slab `row` is the entries whose highest row it is. A launch takes up to NB_SLABS slabs that have the same
  // number of tiles.
  nu32 launches = 0;
  for (nu32 row = 1; row <= rows;) {
    const nu32 tiles = (row + NB_TILE - 1u) / NB_TILE;
    nu32 n = 0;
    for (; n < NB_SLABS && row + n <= rows && (row + n + NB_TILE - 1u) / NB_TILE == tiles; n++)
      nbk_out2<<<nb_blocks((row + n + 1) * R.cout), NB_THREADS>>>(R, (row + n + 1) * R.cout, row + n, s->d_d5, s->d_x5,
                                                                 d_out2 + (size_t)n * per);
    const nu32 each = tiles * (tiles + 1u) / 2u;
    nbk_triple<<<each * n, NB_TILE * NB_TILE>>>(R, row, each, per, s->d_conv, s->d_tag, s->d_mask, s->d_d5, s->d_x5,
                                                s->d_zero5, d_out1, d_out2, d_off2, d_g2, c->d_tri);
    NB_TRY(G(GetLastError)(), "building the triple table");
    if (++launches % 8 == 0) NB_TRY(G(DeviceSynchronize)(), "building the triple table");
    row += n;
  }
  NB_TRY(G(DeviceSynchronize)(), "building the triple table");
  s->t_rows = rows;
  info->rows = rows;
  info->entries = nb_triple_entries(rows);
  info->table_bytes = info->entries * 32;
  info->ms = nb_ms(t0);
  return 0;
}

static nu32 nb_le32(const uint8_t *p) {
  return (nu32)p[0] | ((nu32)p[1] << 8) | ((nu32)p[2] << 16) | ((nu32)p[3] << 24);
}

extern "C" int nb_scan(nb_core *c, const nb_work *w, uint32_t *hits, uint32_t *stored, uint64_t *found,
                       uint64_t *evaluated, char *err, int errlen) {
  *stored = 0;
  *found = 0;
  *evaluated = 0;
  if (w->slot < 0 || w->slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  if (w->count < 1 || w->count > 4294967296ull - w->start) return nb_say(err, errlen, "start + count passes 2^32");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[w->slot];
  if (!s->ready) return nb_say(err, errlen, "the slot has no model");
  NebScan S;
  // BLAKE3's state after the header's first block (CHUNK_START, 64 bytes), then the nonce-free part of the last.
  nu32 blk[16], cv[8];
  const nu32 iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
  for (int i = 0; i < 16; i++) blk[i] = nb_le32(w->header + 4 * i);
  nebk_b3(iv, blk, 64u, 1u, cv);
  for (int i = 0; i < 3; i++) S.tail[i] = nb_le32(w->header + 64 + 4 * i);
  nebk_pre(cv, S.tail, S.pre);
  for (int i = 0; i < 8; i++) S.bound[i] = nb_le32(w->target + 4 * i);
  S.n = s->n;
  S.max_live = s->max_live;
  S.t_rows = s->t_rows;
  S.one = 1;
  S.max_hits = w->max_hits > NB_MAX_HITS ? NB_MAX_HITS : w->max_hits;
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  NB_TRY(G(Memset)(c->d_hits, 0, 4), "clearing the hits");
  NB_TRY(G(Memset)(c->d_total, 0, 8), "clearing the candidate count");
  // The live bytes say which live samples the triple table covers, so they follow the table.
  if (s->alive_rows != s->t_rows) {
    std::vector<unsigned char> alive(s->n);
    for (uint32_t i = 0; i < s->n; i++)
      alive[i] = !s->live[i] ? 0 : (s->live[i] > s->dense ? NEBK_SPARSE : (s->t_rows && s->live[i] > s->t_rows ? NEBK_OUT : 1));
    NB_TRY(G(Memcpy)(s->d_alive, alive.data(), alive.size(), G(MemcpyHostToDevice)), "the live samples");
    s->alive_rows = s->t_rows;
  }
  // Which scan kernel: by the share of nonces that will be candidates on this model. `plain` is the share the
  // tables answer: at most max_live live rows, or 3 dense ones the triple table covers. `extra` has three live
  // rows, a sparse one among them (nbk_third).
  double plain = 0;
  {
    const double p = (double)s->live_rows / (double)s->n, q = 1 - p, pd = (double)s->dense / (double)s->n;
    const double cover = s->t_rows && s->dense ? pow((double)s->t_rows / (double)s->dense, 3) : 0;
    double choose = 1;
    for (nu32 k = 0; k <= 2u; k++) {
      if (k <= s->max_live) plain += choose * pow(p, k) * pow(q, 16 - k);
      choose = choose * (16 - k) / (k + 1);
    }
    plain += cover * choose * pow(pd, 3) * pow(q, 13);
  }
  const double extra = s->third;
  S.dense = s->dense;
  // A model with sparse rows was given them because it scans with nbk_lean, and no other kernel knows them.
  const bool lean = s->nsparse > 0 || ((plain + extra) * 1000 < NB_LEAN_BELOW && s->n <= NB_LEAN_SAMPLES);
  const nu32 wide = lean ? NB_LEAN : NB_SCREEN;
  S.cap = 0;
  S.cap2 = 0;
  S.blocks = 0;
  S.bcap = s->bcap;
  if (lean) {
    // A block's threads try NB_THREADS * (NB_CHUNK / NB_LEAN) nonces; each of its lists holds half as many again
    // as the model's share of them, and a thread that finds one full does its candidate itself.
    const nu32 each = NB_THREADS * (NB_CHUNK / NB_LEAN);
    const auto room = [&](double share, nu32 least) {
      nu32 cap = least;
      while (cap < NB_CAP_MOST && cap < each && (double)cap < (double)each * share * 1.5 + 64) cap *= 2;
      return cap;
    };
    S.cap = room(plain, NB_THREADS);
    S.cap2 = s->nsparse ? room(extra, 64u) : 0u;
    const uint64_t need = (uint64_t)nb_blocks(NB_LEAN) * (S.cap + S.cap2) * sizeof(NebCand);
    if (need > c->cand_bytes) {
      G(Free)(c->d_cand);
      c->d_cand = NULL;
      c->cand_bytes = 0;
      NB_TRY(G(Malloc)((void **)&c->d_cand, (size_t)need), "memory for the candidate lists");
      c->cand_bytes = need;
    }
  }
  // The sparse rows' buckets, done every s->flush launches and at the end.
  const auto third = [&]() {
    nbk_third<<<s->nsparse * (s->bcap / NB_THREADS), NB_THREADS, ((s->meta_max + 3u) & ~3u) * 4u>>>(
        S, s->R, s->d_table, s->d_d5, s->d_meta, s->d_moff, s->d_gt, s->d_recs, s->d_bcount, c->d_hits, c->d_total);
    G(MemsetAsync)(s->d_bcount, 0, (size_t)s->nsparse * 4);
  };
  nu32 since = 0;
  for (uint64_t done = 0; done < w->count; done += NB_CHUNK) {
    const nu32 part = (nu32)(w->count - done < NB_CHUNK ? w->count - done : NB_CHUNK);
    const nu32 iters = part / wide, rest = part - iters * wide;
    for (int pass = 0; pass < 2; pass++) {
      // Whole rounds of `wide` threads, then one thread a nonce for what is left.
      S.start = w->start + (nu32)done + (pass ? iters * wide : 0u);
      S.threads = pass ? rest : wide;
      S.iters = pass ? 1u : iters;
      if (!S.threads || !S.iters) continue;
      S.blocks = nb_blocks(S.threads);
      if (lean) nbk_lean<<<S.blocks, NB_THREADS, (s->n + 15u) & ~15u>>>(S, s->d_live, s->d_alive, s->d_table, c->d_tri, c->d_cand, s->d_recs, s->d_bcount, c->d_hits, c->d_total);
      else nbk_scan<<<nb_blocks(S.threads), NB_THREADS>>>(S, s->d_live, s->d_alive, s->d_table, c->d_tri, c->d_hits, c->d_total);
    }
    if (s->nsparse && ++since == s->flush) {
      third();
      since = 0;
    }
    NB_TRY(G(GetLastError)(), "the scan kernels");
  }
  if (s->nsparse && since) third();
  NB_TRY(G(GetLastError)(), "the scan kernels");
  // One wait for the whole scan: the count, then the hits.
  nu64 total = 0;
  NB_TRY(G(Memcpy)(&total, c->d_total, 8, G(MemcpyDeviceToHost)), "the scan");
  *evaluated = total;
  nu32 n = 0;
  NB_TRY(G(Memcpy)(&n, c->d_hits, 4, G(MemcpyDeviceToHost)), "the ticket kernel");
  *found = n;
  const nu32 keep = n < S.max_hits ? n : S.max_hits;
  if (keep) NB_TRY(G(Memcpy)(hits, c->d_hits + 1, (size_t)keep * 4, G(MemcpyDeviceToHost)), "reading the hits");
  *stored = keep;
  return 0;
}

extern "C" int nb_eval(nb_core *c, int slot, const uint8_t *keys, uint32_t nkeys, uint8_t *out, char *err,
                       int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (!s->ready) return nb_say(err, errlen, "the slot has no model");
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  for (uint32_t done = 0; done < nkeys; done += NB_EVAL_CHUNK) {
    const nu32 part = nkeys - done < NB_EVAL_CHUNK ? nkeys - done : NB_EVAL_CHUNK;
    // Keys and tickets are bytes in little-endian word order, which is the host's.
    NB_TRY(G(Memcpy)(c->d_keys, keys + (size_t)done * 32, (size_t)part * 32, G(MemcpyHostToDevice)), "sending the keys");
    nbk_full<<<(part + 63) / 64, 64>>>(s->R, part, c->d_keys, s->d_live, s->d_conv, s->d_tag, s->d_d5, s->d_x5,
                                       c->d_out);
    NB_TRY(G(GetLastError)(), "the full kernel");
    NB_TRY(G(Memcpy)(out + (size_t)done * 64, c->d_out, (size_t)part * 64, G(MemcpyDeviceToHost)), "the full kernel");
  }
  return 0;
}
