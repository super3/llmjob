// bench -- drive pearl_host_search directly and report the miner's own hashrate
// convention, with no node, no pool and nothing else on the card.
//
// Reported TH/s IS the MAC rate: hashrate = attempts/s * tile_size * k  (at rank 128,
// where the rank-penalty term reduces to 1). tile_size = rows*cols = 16*16 = 256.
//
// TRAP (tuning log s6): a harness that silently measures a stale binary invalidates
// everything. This one is compiled from source every run and prints its own build
// stamp and the geometry it was built with, so a reading can always be attributed.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "pearl_config.h"

struct PearlProofSide { std::vector<uint8_t> leaves; std::vector<uint8_t> path; uint8_t root[PEARL_HASH_BYTES]; uint64_t total_leaves; };
struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES], a_seed[PEARL_HASH_BYTES], b_seed[PEARL_HASH_BYTES];
  uint64_t nonce, salt; std::vector<uint8_t> proof;
  PearlProofSide proof_a, proof_bt; bool found;
};
extern "C" void *pearl_host_create(const PearlProfile *, char *, size_t);
extern "C" void pearl_host_destroy(void *);
extern "C" void pearl_host_set_job(void *, const uint8_t *, const uint8_t *);
extern "C" bool pearl_host_search(void *, uint64_t, uint32_t, PearlSearchResult *, uint64_t *, char *, size_t);

int main(int argc, char **argv) {
  double seconds = argc > 1 ? atof(argv[1]) : 20.0;
  PearlProfile prof{};
  prof.k = 2048; prof.rank = 128; prof.mma_type = 0;
  prof.m = 131072; prof.n = 131072;
  prof.seed_derivation = PEARL_SEED_SALTED;
  prof.col_batch = 2048; prof.hash_big_endian = 0;
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

  printf("build %s %s | m=n=%u col_batch=%u | BLOCK_GROUP=%d WARP_ROWS=%d "
         "ROW_TILES=%d COL_BLK=%d STAGE_BUFS=%d SB_STRIDE=%d\n",
         __DATE__, __TIME__, prof.m, prof.col_batch, PEARL_BLOCK_GROUP, PEARL_WARP_ROWS,
         PEARL_WMMA_ROW_TILES, PEARL_WMMA_COL_BLK, PEARL_STAGE_BUFS, PEARL_SB_STRIDE);

  PearlSearchResult res; uint64_t attempts = 0, total = 0; int hits = 0;
  uint64_t nonce = 0;
  // warm up: first call allocates and draws operands
  err[0] = 0;
  pearl_host_search(h, nonce, prof.col_batch, &res, &attempts, err, sizeof err);
  if (err[0]) { printf("warmup failed: %s\n", err); return 1; }
  nonce += attempts;

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
