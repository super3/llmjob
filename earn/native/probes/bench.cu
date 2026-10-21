// bench -- drive pearl_host_search directly and report the miner's own hashrate
// convention, with no node, no pool and nothing else on the card.
//
// Reported TH/s IS the MAC rate: hashrate = attempts/s * tile_size * k  (at rank 128,
// where the rank-penalty term reduces to 1). tile_size = rows*cols = 16*16 = 256.
//
// TRAP (tuning log s6): a harness that silently measures a stale binary invalidates
// everything. This one is compiled from source every run and prints its own build
// stamp and the geometry it was built with, so a reading can always be attributed.
//
// OPERAND FILL. It benches the mainnet fill (PEARL_OPERAND_CONST, A = B = 48 plus the
// salt stamp); PEARL_OPERAND_FILL_CODE=0 benches the hashed fill from the same binary.
// The fill alone moves the rate about 3.5%. A bench.exe from v0.5.7 or older knows only
// the hashed fill, so an A/B against one compares the fills as well as the code unless
// this side runs with PEARL_OPERAND_FILL_CODE=0.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "pearl_config.h"

// Must match pearl_host.cu's layout exactly: the host writes these through the
// pointer, and the old {leaves, path, root} version was 48 bytes short of it.
struct PearlProofSide {
  std::vector<uint32_t> leaf_indices; std::vector<uint8_t> leaves; std::vector<uint8_t> siblings;
  uint8_t root[PEARL_HASH_BYTES]; uint64_t total_leaves;
};
struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES], a_seed[PEARL_HASH_BYTES], b_seed[PEARL_HASH_BYTES];
  uint64_t nonce, salt; std::vector<uint8_t> proof;
  PearlProofSide proof_a, proof_bt; bool found;
};
extern "C" void *pearl_host_create(const PearlProfile *, char *, size_t);
extern "C" void pearl_host_destroy(void *);
extern "C" void pearl_host_set_job(void *, const uint8_t *, const uint8_t *);
extern "C" bool pearl_host_search(void *, uint64_t, uint32_t, PearlSearchResult *, uint64_t *, char *, size_t);
extern "C" bool pearl_host_next_hit(void *, PearlSearchResult *);
extern "C" void pearl_host_reseed(void *, uint64_t);
extern "C" const char *pearl_host_fold_name(void *);

static void fnv(uint64_t &d, const void *p, size_t n) {
  const uint8_t *q = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; i++) d = (d ^ q[i]) * 1099511628211ull;
}

// PEARL_HITLIST=<batches>: every hit this build finds, to compare with another build's
// (probes/turing/compilers.sh, stage "complete"). The fold reports a hit only after it
// recomputes the whole hash from the transcript, so a fold whose filter misses (a split
// hash that reads a stale state, say) drops hits and reports no wrong one: verify-hits.js
// passes it, and the bench reads the lost work as speed. Here the target has
// PEARL_HITLIST_BITS leading zero bits, by default as many as leave 16 to 32 hits a
// batch on average, well under PEARL_MAX_HITS, so a batch's list is every hit in it.
// From nonce 0 under salt 0, re-drawn at the next salt after every m*n regions as the
// miner does, it prints a line a batch (its salt, nonce, hit count and a digest of its
// hits' nonces and jackpot hashes, in region order) and then the totals. Two builds
// that compute the same hashes print the same lines. A batch whose list is full
// (PEARL_MAX_HITS) may have lost hits, and which ones changes from run to run: it is
// counted as full, and only its count goes into the digest.
static int hitlist(void *h, const PearlProfile &prof, const uint8_t *header, uint64_t regions,
                   unsigned long batches) {
  unsigned bits = 0;
  while ((regions >> (bits + 1)) >= 16u) bits++;
  if (getenv("PEARL_HITLIST_BITS")) bits = (unsigned)strtoul(getenv("PEARL_HITLIST_BITS"), nullptr, 10);
  if (bits < 1 || bits > 255) { printf("hitlist FAIL: %u target bits\n", bits); return 1; }
  // The target as the host reads it: a big-endian number, below 2^(256 - bits).
  uint8_t target[PEARL_HASH_BYTES];
  memset(target, 0xFF, sizeof target);
  memset(target, 0x00, bits / 8);
  target[bits / 8] = (uint8_t)(0xFFu >> (bits % 8));
  pearl_host_set_job(h, header, target);
  const uint64_t span = (uint64_t)(prof.m / PEARL_ROWS_COUNT) * (prof.n / PEARL_COLS_COUNT);
  uint64_t nonce = 0, salt = 0, hits = 0, digest = 1469598103934665603ull;
  unsigned long full = 0;
  char err[512];
  for (unsigned long b = 0; b < batches; b++) {
    PearlSearchResult res;
    uint64_t att = 0;
    err[0] = 0;
    const bool found = pearl_host_search(h, nonce, prof.col_batch, &res, &att, err, sizeof err);
    if (err[0]) { printf("hitlist FAIL: batch %lu: %s\n", b, err); return 1; }
    if (att == 0) { printf("hitlist FAIL: batch %lu: no attempts reported\n", b); return 1; }
    uint64_t d = 1469598103934665603ull, n = 0;
    if (found) {
      fnv(d, &res.nonce, sizeof res.nonce);
      fnv(d, res.jackpot_hash, PEARL_HASH_BYTES);
      n++;
      PearlSearchResult more;
      while (pearl_host_next_hit(h, &more)) {
        fnv(d, &more.nonce, sizeof more.nonce);
        fnv(d, more.jackpot_hash, PEARL_HASH_BYTES);
        n++;
      }
    }
    const bool isfull = n >= PEARL_MAX_HITS;
    if (isfull) { full++; d = n; }
    printf("HL %lu salt=%llu nonce=%llu hits=%llu%s d=%016llx\n", b, (unsigned long long)salt,
           (unsigned long long)nonce, (unsigned long long)n, isfull ? " full" : "",
           (unsigned long long)d);
    fnv(digest, &d, sizeof d);
    hits += n;
    nonce += att;
    if (nonce >= span) {
      pearl_host_reseed(h, ++salt);
      nonce = 0;
    }
  }
  printf("hitlist bits=%u regions=%llu batches=%lu hits=%llu full=%lu salts=%llu digest=%016llx\n",
         bits, (unsigned long long)regions, batches, (unsigned long long)hits, full,
         (unsigned long long)(salt + 1), (unsigned long long)digest);
  return 0;
}

int main(int argc, char **argv) {
  double seconds = argc > 1 ? atof(argv[1]) : 20.0;
  PearlProfile prof{};
  prof.k = 2048; prof.rank = 128; prof.mma_type = 0;
  prof.m = 131072; prof.n = 131072;
  prof.seed_derivation = PEARL_SEED_SALTED;
  prof.col_batch = 2048; prof.hash_big_endian = 0;
  // What ships, unless PEARL_OPERAND_FILL_CODE says otherwise (see OPERAND FILL above).
  prof.operand_fill = PEARL_MAINNET_PROFILE.operand_fill;
  if (getenv("PEARL_OPERAND_FILL_CODE"))
    prof.operand_fill = (uint32_t)strtoul(getenv("PEARL_OPERAND_FILL_CODE"), nullptr, 10);
  if (argc > 2) prof.m = prof.n = (uint32_t)strtoul(argv[2], nullptr, 10);
  if (argc > 3) prof.col_batch = (uint32_t)strtoul(argv[3], nullptr, 10);

  char err[512] = {0};
  void *h = pearl_host_create(&prof, err, sizeof err);
  if (!h) { printf("create failed: %s\n", err); return 1; }
  uint8_t header[PEARL_HEADER_BYTES]; memset(header, 0xA5, sizeof header);
  // verify mode: an easy target so hits actually occur, then print them. Two builds
  // that stage identical bytes must find identical nonces and jackpot hashes.
  const bool verify = getenv("PEARL_VERIFY") != nullptr;
  uint8_t target[PEARL_HASH_BYTES];
  // 0xFF everywhere is the largest possible target, so every transcript clears it
  // and hits come out immediately; 0x00 is the hardest, used for timing runs so no
  // hit-handling work is measured.
  memset(target, verify ? 0xFF : 0x00, sizeof target);
  pearl_host_set_job(h, header, target);

  // The fold the LOADED binary runs, as the host resolved it. The macros after it are
  // this file's host pass, which describe only the sixteen-warp fold's defaults.
  printf("build %s %s | m=n=%u col_batch=%u fill=%u | fold: %s | host-pass BLOCK_GROUP=%d "
         "WARP_ROWS=%d ROW_TILES=%d COL_BLK=%d STAGE_BUFS=%d SB_STRIDE=%d\n",
         __DATE__, __TIME__, prof.m, prof.col_batch, prof.operand_fill, pearl_host_fold_name(h),
         PEARL_BLOCK_GROUP, PEARL_WARP_ROWS, PEARL_WMMA_ROW_TILES, PEARL_WMMA_COL_BLK,
         PEARL_STAGE_BUFS, PEARL_SB_STRIDE);

  PearlSearchResult res; uint64_t attempts = 0, total = 0; int hits = 0;
  uint64_t nonce = 0;
  // warm up: first call allocates and draws operands
  err[0] = 0;
  pearl_host_search(h, nonce, prof.col_batch, &res, &attempts, err, sizeof err);
  if (err[0]) { printf("warmup failed: %s\n", err); return 1; }
  nonce += attempts;
  if (getenv("PEARL_HITLIST")) {
    const int rc = hitlist(h, prof, header, attempts, strtoul(getenv("PEARL_HITLIST"), nullptr, 10));
    pearl_host_destroy(h);
    return rc;
  }

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  double elapsed = 0; int iters = 0;
  cudaEventRecord(e0);
  while (elapsed < seconds * 1000.0) {
    // NB: the return value is `found`, not success -- false is the normal
    // "no hit in this batch" path. Only a non-empty err is an actual failure.
    err[0] = 0;
    pearl_host_search(h, nonce, prof.col_batch, &res, &attempts, err, sizeof err);
    if (err[0]) { printf("search failed: %s\n", err); return 1; }
    if (attempts == 0) { printf("no attempts reported -- aborting\n"); return 1; }
    if (verify && res.found) {
      printf("HIT nonce=%llu salt=%llu jackpot=", (unsigned long long)res.nonce,
             (unsigned long long)res.salt);
      for (int i = 0; i < 8; i++) printf("%02x", res.jackpot_hash[i]);
      printf("\n");
      if (++hits >= 6) break;
    }
    nonce += attempts; total += attempts; iters++;
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime((float *)&elapsed, e0, e1);
    float f; cudaEventElapsedTime(&f, e0, e1); elapsed = f;
  }
  const double DAF = (double)PEARL_ROWS_COUNT * PEARL_COLS_COUNT * prof.k;
  double thps = (double)total * DAF / (elapsed * 1e-3) / 1e12;
  printf("  attempts=%llu  batches=%d  %.3f s  ->  %.1f TH/s\n",
         (unsigned long long)total, iters, elapsed / 1000.0, thps);
  pearl_host_destroy(h);
  return 0;
}
