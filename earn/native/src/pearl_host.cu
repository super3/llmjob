// Host-side driver for the Pearl CUDA core: owns device memory, runs the kernel
// pipeline, and implements the four extern "C" entry points pearl_core.cc calls.
// Without this the addon compiles and fails to link on four undefined symbols.
//
// MEMORY BUDGET, because it is the design constraint that shapes everything
// here. The mandated profile is m=n=131072, k=4096, which makes a full int8 A
// (m×k) and Bᵀ (n×k) 512 MiB each — 1 GiB resident, before noise. That fits a
// 24 GB card comfortably and would not fit an 8 GB one alongside a co-running
// LLM, so `pearl_host_create` checks free VRAM up front and fails with a
// readable message rather than dying inside a kernel launch. The app already
// treats a failed core as "engine unavailable" and says so.
//
// WHY A AND B ARE RESIDENT AT ALL. The commitments hash the WHOLE operands
// (hash_a over pad1024(A), hash_b over pad1024(Bᵀ)), so they must exist somewhere
// once per job. They are generated on-device from job_key rather than uploaded,
// which keeps the PCIe bus out of it entirely: generation and hashing are both
// GPU-side and happen once per job, after which the search reads them.
//
// WHAT THE SEARCH VARIES. Not a header nonce — re-deriving job_key per attempt
// would mean re-hashing 1 GiB per attempt. Per job the commitments are computed
// once, and the search then walks output sub-regions: each region index selects
// the offset of the mandated row/column tile, giving a distinct transcript and so
// a distinct jackpot hash. That index is what travels back as `nonce` and goes
// into the share, and it is why the miner reports progress as "regions".
//
// STATUS: never compiled — the development box has no CUDA toolkit and no MSVC.
// Reviewed source with the build wired up, checked against the JS reference's
// known-answer vectors for semantics (test/minerReference.test.js), not against
// a running GPU. Treat every performance claim as absent rather than optimistic.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <set>
#include <vector>

#include <cstdlib>
#include <cstdio>

#include "pearl_config.h"

// Declared in pearl_kernel.cu.
extern "C" __global__ void pearl_gen_dense(const uint32_t *seed,
                                           const uint8_t *label,
                                           const uint32_t *row_indices,
                                           int8_t *out, uint32_t num_rows,
                                           uint32_t rank);
extern "C" __global__ void pearl_gen_perm(const uint32_t *seed,
                                          const uint8_t *label, uint32_t *out,
                                          uint32_t k, uint32_t rank);
extern "C" __global__ void pearl_gen_operand(const uint32_t *key,
                                             const uint8_t *label, int8_t *out,
                                             uint64_t total, uint64_t salt);
extern "C" __global__ void pearl_hash_operands(const uint32_t *job_key,
                                               const uint8_t *padded,
                                               uint32_t len, uint8_t *out);
extern "C" __global__ void pearl_materialize(const int8_t *base,
                                             const int8_t *dense,
                                             const uint32_t *perm, int8_t *out,
                                             uint32_t rows, uint32_t k,
                                             uint32_t rank);
extern "C" __global__ void pearl_tile_fold_wmma(
    const int8_t *Aprime, const int8_t *Bprime, uint32_t m, uint32_t n,
    uint32_t k, uint32_t rank, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t *jackpot_out);
extern "C" __global__ void pearl_partials(const int8_t *Aprime, const int8_t *Bprime,
                                          const uint32_t *cols_pattern,
                                          uint32_t cols_count, uint32_t m, uint32_t n,
                                          uint32_t k, uint32_t rank, uint32_t chunks,
                                          uint32_t col_off, uint32_t col_groups,
                                          int32_t *D);
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *D, const uint32_t *rows_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t rows_valid, uint32_t chunks,
    uint64_t region_base, uint32_t *jackpot_out);
extern "C" __global__ void pearl_finalize_many(const uint32_t *a_seed,
                                               const uint32_t *jackpots,
                                               uint32_t count,
                                               const uint8_t *target_be,
                                               uint8_t *hashes_out,
                                               uint32_t *hit_count,
                                               uint32_t *hit_index,
                                               int hash_big_endian);
extern "C" __global__ void pearl_blake3_chunk_cvs(const uint32_t *key,
                                                  const uint8_t *data,
                                                  uint64_t chunks,
                                                  uint32_t *cvs_out);
extern "C" __global__ void pearl_blake3_parent_layer(const uint32_t *key,
                                                     const uint32_t *in_cvs,
                                                     uint64_t pairs,
                                                     uint32_t is_root,
                                                     uint32_t *out_cvs);
extern "C" __global__ void pearl_blake3_unkeyed(const uint8_t *in, uint32_t len,
                                                uint8_t *out);
extern "C" __global__ void pearl_bind_root(const uint8_t *salt,
                                           const uint8_t *root, uint32_t dim,
                                           uint8_t *out);
extern "C" __global__ void pearl_finalize(const uint32_t *a_seed,
                                          const uint32_t *jackpot,
                                          const uint8_t *target_be,
                                          uint8_t *hash_out, int *is_share);

// Mirrors the struct pearl_core.cc declares. Kept in this one header-free form
// deliberately: the two files must agree on the layout, and a shared header that
// pulled in <napi.h> would drag Node headers into nvcc.
// One side of a share proof, carried WITH the hit.
//
// A single shared snapshot on the context was not enough: the search keeps
// running after a hit, and a later hit overwrote the buffer before the host had
// read the earlier one. The same leaf range would verify for one region and
// fail for the next. The proof has to travel with the result it belongs to.
struct PearlProofSide {
  std::vector<uint32_t> leaf_indices;
  std::vector<uint8_t> leaves;    // leaf_indices.size() * 1024
  std::vector<uint8_t> siblings;  // n * 32
  uint8_t root[PEARL_HASH_BYTES];
  uint64_t total_leaves;
};

struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES];
  uint8_t a_seed[PEARL_HASH_BYTES];
  uint8_t b_seed[PEARL_HASH_BYTES];
  uint64_t nonce;
  // Which operand draw this came from. The proof a pool needs is over the
  // operand data, so a share is unprovable without it.
  uint64_t salt;
  std::vector<uint8_t> proof;
  // The share proof for this hit, captured while the operands still belong
  // to it.
  PearlProofSide proof_a;
  PearlProofSide proof_bt;
  bool found;
};

namespace {

struct Ctx {
  PearlProfile profile;

  // Operands, generated once per job and then read by every region.
  int8_t *dA = nullptr;   // [m, k]
  int8_t *dB = nullptr;   // [n, k]  (Bᵀ, row-major)

  // The noised operands, computed once per commitment. int8, matching the
  // reference's saturating convert-down: operand and noise are both int7, so the
  // sum fits, and an int8 operand is what lets the fold use __dp4a at all.
  int8_t *dAp = nullptr; // [m, k]
  int8_t *dBp = nullptr; // [n, k]

  // The noise factors. Only two of the four are dense — E_AR and E_BL are sparse
  // +-1 selectors, stored as one (p0, p1) index pair per k, and they depend only
  // on the seeds rather than on the tile offset.
  int8_t *dEAL = nullptr;    // dense  [m, rank]
  int8_t *dEBR = nullptr;    // dense  [n, rank]
  uint32_t *dPermA = nullptr; // sparse [k, 2]
  uint32_t *dPermB = nullptr; // sparse [k, 2]

  // The two 32-byte seed labels, "A_tensor" and "B_tensor", zero-padded. They go
  // into the RNG message while the commitment seed is the key.
  uint8_t *dLabelA = nullptr;
  uint8_t *dLabelB = nullptr;

  // The cert-v3 domain-separation salts, and the bound roots they produce.
  uint8_t *dSaltA = nullptr;
  uint8_t *dSaltB = nullptr;
  uint8_t *dBoundA = nullptr;
  uint8_t *dBoundB = nullptr;

  uint32_t *dRows = nullptr;
  uint32_t *dCols = nullptr;
  uint32_t *dJackpot = nullptr;   // [batch][16]
  uint8_t *dHashes = nullptr;     // [PEARL_MAX_HITS][32] — hits only
  uint32_t *dHitCount = nullptr;  // one counter per batch
  uint32_t *dHitIndex = nullptr;  // [PEARL_MAX_HITS]
  uint32_t colBatch = 1;          // valid column offsets per launch
  uint32_t rowsValid = 1;         // count of valid row offsets
  uint32_t colsValid = 1;         // count of valid column offsets
  std::vector<uint32_t> hHitIndex;
  uint32_t batch = 0;
  uint32_t *dJobKey = nullptr;
  uint32_t *dASeed = nullptr;
  uint32_t *dBSeed = nullptr;
  uint32_t *dCvs = nullptr;      // BLAKE3 tree scratch (leaf CVs, reduced in place)

  // The commitment TREE, kept rather than reduced away.
  //
  // A share has to be proved against the operand data the tile touched, and
  // that proof needs sibling digests from every level. The device already
  // computes all of them to get the root; discarding them forced the host to
  // rebuild the whole tree in JS, which at m=32768, k=4096 took 21 seconds a
  // share -- long enough that the job could rotate underneath it.
  //
  // Layers are stored end to end: level 0 (the leaves) first, then each parent
  // level. About 2*leaves nodes in total, 8 MiB at the mainnet geometry.
  uint32_t *dTreeA = nullptr;
  uint32_t *dTreeB = nullptr;
  std::vector<uint64_t> layerOffA;  // node offset of each level
  std::vector<uint64_t> layerOffB;

  // The proof for the most recent hit, captured AT HIT TIME.
  //
  // The operands are re-drawn every time the region space is exhausted, which
  // at this geometry is every few tens of milliseconds. Fetching the proof
  // afterwards therefore reads a DIFFERENT draw's tree than the one the hit was
  // found under -- sometimes a level that no longer exists, sometimes a proof
  // that simply does not verify. Both were observed before this existed.
  std::vector<uint8_t> snapLeavesA, snapLeavesB;
  std::vector<uint8_t> snapSibsA, snapSibsB;
  std::vector<uint32_t> snapLeafIdxA, snapLeafIdxB;
  bool snapValid = false;
  uint8_t *dSeedBuf = nullptr;   // 64-byte concat for b_seed / a_seed
  uint8_t *dHashA = nullptr;
  uint8_t *dHashB = nullptr;
  uint64_t cvCapacity = 0;
  uint8_t *dTarget = nullptr;
  uint8_t *dHash = nullptr;
  int *dIsShare = nullptr;

  uint8_t aSeed[PEARL_HASH_BYTES] = {0};
  uint8_t bSeed[PEARL_HASH_BYTES] = {0};
  bool haveJob = false;
  // The job is kept so the operands can be re-drawn under a new salt without
  // the caller having to hand the header back.
  uint8_t header[PEARL_HEADER_BYTES] = {0};
  uint8_t target[PEARL_HASH_BYTES] = {0};
  uint64_t salt = 0;
};

// The row/column patterns the tile folds over. Derived from the counts plus the
// fixed stride, exactly as the reference does, so the host and the config block
// cannot disagree about what they mean.
void build_patterns(const PearlProfile &, std::vector<uint32_t> *rows,
                    std::vector<uint32_t> *cols) {
  // Straight from the reference's defaults. This used to DERIVE the indices from
  // counts plus an assumed stride, which produced a 2x64 tile of the wrong
  // indices entirely.
  rows->assign(PEARL_ROWS_PATTERN, PEARL_ROWS_PATTERN + PEARL_ROWS_COUNT);
  cols->assign(PEARL_COLS_PATTERN, PEARL_COLS_PATTERN + PEARL_COLS_COUNT);
}

// Keyed BLAKE3 over a whole operand: hash each 1024-byte chunk into a leaf CV,
// then fold pairs until one root remains. The final fold carries ROOT and its
// output IS the digest.
//
// Chunk counts are powers of two here (m*k with power-of-two m and k), so the
// tree is balanced and the fold is exact. A non-power-of-two operand would need
// BLAKE3's left-heavy layout, which this deliberately does not pretend to do.
void operand_commitment(Ctx *ctx, const uint8_t *data, size_t len,
                        uint8_t *out32, uint32_t *tree,
                        std::vector<uint64_t> *offsets) {
  const uint64_t chunks = len / 1024;
  const int threads = 256;
  if (offsets) offsets->clear();
  if (chunks <= 1) {
    // One chunk or less: the chunk's own final compression carries ROOT.
    pearl_hash_operands<<<1, 1>>>(ctx->dJobKey, data, (uint32_t)len, out32);
    return;
  }

  // Each level is written to its own place rather than reduced in place, so the
  // sibling digests a share proof needs are still there afterwards.
  uint32_t *dst = tree ? tree : ctx->dCvs;
  uint64_t base = 0;
  if (offsets) offsets->push_back(0);
  pearl_blake3_chunk_cvs<<<(unsigned)((chunks + threads - 1) / threads), threads>>>(
      ctx->dJobKey, data, chunks, dst);

  uint64_t count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2;
    const uint32_t isRoot = (pairs == 1) ? 1u : 0u;
    const uint64_t next = tree ? base + count : 0;
    pearl_blake3_parent_layer<<<(unsigned)((pairs + threads - 1) / threads), threads>>>(
        ctx->dJobKey, dst + base * 8, pairs, isRoot, dst + next * 8);
    if (offsets) offsets->push_back(next);
    base = next;
    count = pairs;
  }
  cudaMemcpy(out32, dst + base * 8, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
}

// Which 1024-byte chunks hold these matrix rows. A row of k bytes can straddle
// a boundary, so this is a range per row.
void leafIndicesForRows(const uint32_t *rows, uint32_t nrows, uint32_t k,
                        std::vector<uint32_t> *out) {
  std::set<uint32_t> s;
  for (uint32_t i = 0; i < nrows; i++) {
    const uint64_t first = (uint64_t)rows[i] * k / 1024;
    const uint64_t last = ((uint64_t)(rows[i] + 1) * k - 1) / 1024;
    for (uint64_t j = first; j <= last; j++) s.insert((uint32_t)j);
  }
  out->assign(s.begin(), s.end());
}

// Capture one side's proof while the tree still belongs to the hit.
//
// The sibling order must match the verifier's exactly: level by level, visiting
// the live set in ascending index order, emitting a sibling only when it is not
// itself live.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs) {
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves =
      (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

  leafIndicesForRows(rows, nrows, k, leafIdx);

  leaves->resize(leafIdx->size() * 1024);
  for (size_t i = 0; i < leafIdx->size(); i++) {
    cudaMemcpy(leaves->data() + i * 1024,
               operand + (uint64_t)(*leafIdx)[i] * 1024, 1024,
               cudaMemcpyDeviceToHost);
  }

  sibs->clear();
  std::vector<uint32_t> current = *leafIdx;
  uint64_t levelLen = totalLeaves;
  uint32_t level = 0;
  while (levelLen > 1 && !current.empty() && level + 1 < offs.size()) {
    const std::set<uint32_t> live(current.begin(), current.end());
    for (uint32_t i : current) {
      uint32_t want;
      if (i % 2 == 1) {
        if (live.count(i - 1)) continue;
        want = i - 1;
      } else {
        if (live.count(i + 1) || (uint64_t)i + 1 >= levelLen) continue;
        want = i + 1;
      }
      uint8_t node[PEARL_HASH_BYTES];
      cudaMemcpy(node, tree + (offs[level] + want) * 8, PEARL_HASH_BYTES,
                 cudaMemcpyDeviceToHost);
      sibs->insert(sibs->end(), node, node + PEARL_HASH_BYTES);
    }
    std::set<uint32_t> next;
    for (uint32_t i : current) next.insert(i / 2);
    current.assign(next.begin(), next.end());
    levelLen = (levelLen + 1) / 2;
    level++;
  }
}

bool fail(char *err, size_t err_len, const char *msg) {
  if (err && err_len) snprintf(err, err_len, "%s", msg);
  return false;
}

#define CUDA_OK(expr, msg)                                   \
  do {                                                       \
    cudaError_t _e = (expr);                                 \
    if (_e != cudaSuccess) {                                 \
      if (err && err_len)                                    \
        snprintf(err, err_len, "%s: %s", msg,                \
                 cudaGetErrorString(_e));                    \
      return nullptr;                                        \
    }                                                        \
  } while (0)

}  // namespace

extern "C" void *pearl_host_create(const PearlProfile *profile, char *err,
                                   size_t err_len) {
  if (!profile) { fail(err, err_len, "no profile supplied"); return nullptr; }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    fail(err, err_len, "no CUDA device found — is an NVIDIA driver installed?");
    return nullptr;
  }

  // The commitment folds the chunk tree as a balanced pairwise reduction, which
  // is only BLAKE3's tree when the chunk count is a power of two. Refuse
  // anything else rather than computing a wrong root in silence — that produces
  // wrong seeds and a share no pool accepts, with no symptom to follow.
  {
    const uint64_t aChunks = (uint64_t)profile->m * profile->k / 1024u;
    const uint64_t bChunks = (uint64_t)profile->n * profile->k / 1024u;
    const bool aOk = aChunks && (aChunks & (aChunks - 1)) == 0;
    const bool bOk = bChunks && (bChunks & (bChunks - 1)) == 0;
    if (!aOk || !bOk) {
      if (err && err_len) {
        snprintf(err, err_len,
                 "m*k/1024 and n*k/1024 must each be a power of two (got %llu "
                 "and %llu): the commitment tree fold assumes it",
                 (unsigned long long)aChunks, (unsigned long long)bChunks);
      }
      return nullptr;
    }
  }

  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const size_t noiseBytes = (size_t)profile->m * rank + (size_t)profile->n * rank
                            + 2 * k * 2 * sizeof(uint32_t) + 64;
  // The materialised operands are int8, the same size as the sources. They were
  // int32 while the noise was (wrongly) reconstructed at full rank, which cost
  // 2 GiB at mainnet on top of the 1 GiB of sources.
  const size_t primeBytes = aBytes + bBytes;
  // What a batch really costs: one transcript per REGION, and regions are
  // row OFFSETS by column offsets, not rows by columns. Two stale terms lived
  // here and together overstated it by about 25x:
  //
  //   - profile->m instead of m/PEARL_ROWS_COUNT, which is the number of valid
  //     row offsets and therefore the batch's real height;
  //   - a 32-byte hash and a flag PER REGION, from when finalize wrote every
  //     region's hash and the host read back a flag array. It writes only on a
  //     hit now, into a fixed PEARL_MAX_HITS list.
  //
  // The consequence was not cosmetic: this check refused geometries the miner
  // runs fine on, which is what kept the search pinned to the smaller operand
  // draw and paid the redraw cost four times more often than it had to.
  const size_t colBatch = profile->col_batch ? profile->col_batch : 1u;
  const size_t rowsValid = profile->m / PEARL_ROWS_COUNT;
  const size_t colsValid = profile->n / PEARL_COLS_COUNT;
  const size_t batchCols = colBatch > colsValid ? colsValid : colBatch;
  const size_t batchBytes = batchCols * rowsValid * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)
                            + (size_t)PEARL_MAX_HITS * (PEARL_HASH_BYTES + sizeof(uint32_t));
  const size_t need = aBytes + bBytes + primeBytes + noiseBytes
                      + batchBytes + (1u << 20);

  // Check the budget BEFORE allocating, so an 8 GB card gets a sentence it can
  // act on instead of an out-of-memory abort three kernels deep.
  size_t freeMem = 0, totalMem = 0;
  if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess && freeMem < need) {
    if (err && err_len) {
      snprintf(err, err_len,
               "not enough free VRAM for the rank-%u profile: need ~%zu MiB, "
               "%zu MiB free of %zu MiB",
               (unsigned)profile->rank, need >> 20, freeMem >> 20, totalMem >> 20);
    }
    return nullptr;
  }

  Ctx *ctx = new Ctx();
  ctx->profile = *profile;

  CUDA_OK(cudaMalloc(&ctx->dA, aBytes), "allocating A");
  CUDA_OK(cudaMalloc(&ctx->dB, bBytes), "allocating B");
  CUDA_OK(cudaMalloc(&ctx->dAp, aBytes), "allocating the noised A");
  CUDA_OK(cudaMalloc(&ctx->dBp, bBytes), "allocating the noised B");
  CUDA_OK(cudaMalloc(&ctx->dEAL, (size_t)profile->m * rank), "allocating E_AL");
  CUDA_OK(cudaMalloc(&ctx->dEBR, (size_t)profile->n * rank), "allocating E_BR");
  CUDA_OK(cudaMalloc(&ctx->dPermA, k * 2 * sizeof(uint32_t)), "allocating E_AR");
  CUDA_OK(cudaMalloc(&ctx->dPermB, k * 2 * sizeof(uint32_t)), "allocating E_BL");

  // The seed labels are fixed ASCII, so they are uploaded once here rather than
  // rebuilt per job.
  CUDA_OK(cudaMalloc(&ctx->dLabelA, 32), "allocating the A label");
  CUDA_OK(cudaMalloc(&ctx->dLabelB, 32), "allocating the B label");
  {
    uint8_t lab[32];
    memset(lab, 0, 32);
    memcpy(lab, "A_tensor", 8);
    cudaMemcpy(ctx->dLabelA, lab, 32, cudaMemcpyHostToDevice);
    memset(lab, 0, 32);
    memcpy(lab, "B_tensor", 8);
    cudaMemcpy(ctx->dLabelB, lab, 32, cudaMemcpyHostToDevice);
  }

  CUDA_OK(cudaMalloc(&ctx->dSaltA, 32), "allocating the A salt");
  CUDA_OK(cudaMalloc(&ctx->dSaltB, 32), "allocating the B salt");
  CUDA_OK(cudaMalloc(&ctx->dBoundA, PEARL_HASH_BYTES), "allocating the bound A root");
  CUDA_OK(cudaMalloc(&ctx->dBoundB, PEARL_HASH_BYTES), "allocating the bound B root");
  cudaMemcpy(ctx->dSaltA, PEARL_SEED_SALT_A, 32, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->dSaltB, PEARL_SEED_SALT_B, 32, cudaMemcpyHostToDevice);
  // A batch is col_batch column offsets by m row offsets. Widening it past a
  // single column offset is what took the search from launch-bound to
  // compute-bound: the fixed per-batch cost (three launches and a synchronising
  // copy) was flat at 134-213us regardless of the work inside it.
  //
  // Only offsets with the pattern's bits clear are valid, so there are
  // m/PEARL_ROWS_COUNT of them down the rows and n/PEARL_COLS_COUNT across the
  // columns. Searching the other 31/32 produced hashes no pool would take.
  ctx->rowsValid = profile->m / PEARL_ROWS_COUNT;
  ctx->colsValid = profile->n / PEARL_COLS_COUNT;
  ctx->colBatch = profile->col_batch ? profile->col_batch : 1u;
  if (ctx->colBatch > ctx->colsValid) ctx->colBatch = ctx->colsValid;
  ctx->batch = ctx->colBatch * ctx->rowsValid;
  ctx->hHitIndex.resize(PEARL_MAX_HITS);
  // No partial-product table any more. The fold was two passes once -- compute
  // every partial dot product into a table, then gather it -- and the fused
  // tile fold replaced both, because a CUMULATIVE accumulator cannot be
  // decomposed into reusable partials. The allocation outlived its only
  // consumer and was still sized from col_batch, so raising col_batch to 2048
  // silently reserved GIGABYTES that nothing ever read, and the pre-flight VRAM
  // check refused geometries the miner would have run fine.
  CUDA_OK(cudaMalloc(&ctx->dJackpot,
                     (size_t)ctx->batch * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
          "allocating the transcripts");
  CUDA_OK(cudaMalloc(&ctx->dHashes, (size_t)PEARL_MAX_HITS * PEARL_HASH_BYTES),
          "allocating the batch hashes");
  CUDA_OK(cudaMalloc(&ctx->dHitCount, sizeof(uint32_t)), "allocating the hit counter");
  CUDA_OK(cudaMalloc(&ctx->dHitIndex, (size_t)PEARL_MAX_HITS * sizeof(uint32_t)),
          "allocating the hit list");
  CUDA_OK(cudaMalloc(&ctx->dJobKey, 8 * sizeof(uint32_t)), "allocating job_key");
  CUDA_OK(cudaMalloc(&ctx->dASeed, 8 * sizeof(uint32_t)), "allocating a_seed");
  CUDA_OK(cudaMalloc(&ctx->dBSeed, 8 * sizeof(uint32_t)), "allocating b_seed");
  // Leaf CVs for the larger of the two operands: 8 words per 1024-byte chunk.
  ctx->cvCapacity = (aBytes > bBytes ? aBytes : bBytes) / 1024;
  if (ctx->cvCapacity < 1) ctx->cvCapacity = 1;
  CUDA_OK(cudaMalloc(&ctx->dCvs, ctx->cvCapacity * 8 * sizeof(uint32_t)),
          "allocating the BLAKE3 tree scratch");
  // The kept commitment trees. Levels sum to just under 2*leaves nodes.
  {
    const uint64_t aLeaves = (uint64_t)profile->m * profile->k / 1024;
    const uint64_t bLeaves = (uint64_t)profile->n * profile->k / 1024;
    CUDA_OK(cudaMalloc(&ctx->dTreeA, 2 * aLeaves * 8 * sizeof(uint32_t)),
            "allocating the A commitment tree");
    CUDA_OK(cudaMalloc(&ctx->dTreeB, 2 * bLeaves * 8 * sizeof(uint32_t)),
            "allocating the B commitment tree");
  }
  CUDA_OK(cudaMalloc(&ctx->dSeedBuf, 64), "allocating the seed buffer");
  CUDA_OK(cudaMalloc(&ctx->dHashA, PEARL_HASH_BYTES), "allocating hash_a");
  CUDA_OK(cudaMalloc(&ctx->dHashB, PEARL_HASH_BYTES), "allocating hash_b");
  CUDA_OK(cudaMalloc(&ctx->dTarget, PEARL_HASH_BYTES), "allocating the target");
  CUDA_OK(cudaMalloc(&ctx->dHash, PEARL_HASH_BYTES), "allocating the hash");
  CUDA_OK(cudaMalloc(&ctx->dIsShare, sizeof(int)), "allocating the share flag");

  std::vector<uint32_t> rows, cols;
  build_patterns(*profile, &rows, &cols);
  CUDA_OK(cudaMalloc(&ctx->dRows, rows.size() * sizeof(uint32_t)), "allocating rows");
  CUDA_OK(cudaMalloc(&ctx->dCols, cols.size() * sizeof(uint32_t)), "allocating cols");
  cudaMemcpy(ctx->dRows, rows.data(), rows.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->dCols, cols.data(), cols.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  return ctx;
}

extern "C" void pearl_host_destroy(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  cudaFree(ctx->dA); cudaFree(ctx->dB);
  cudaFree(ctx->dAp); cudaFree(ctx->dBp);
  cudaFree(ctx->dEAL); cudaFree(ctx->dEBR);
  cudaFree(ctx->dPermA); cudaFree(ctx->dPermB);
  cudaFree(ctx->dLabelA); cudaFree(ctx->dLabelB);
  cudaFree(ctx->dSaltA); cudaFree(ctx->dSaltB);
  cudaFree(ctx->dBoundA); cudaFree(ctx->dBoundB);
  cudaFree(ctx->dRows); cudaFree(ctx->dCols);

  cudaFree(ctx->dHashes); cudaFree(ctx->dHitCount); cudaFree(ctx->dHitIndex);
  cudaFree(ctx->dTreeA); cudaFree(ctx->dTreeB);
  cudaFree(ctx->dCvs); cudaFree(ctx->dSeedBuf);
  cudaFree(ctx->dHashA); cudaFree(ctx->dHashB);
  cudaFree(ctx->dJackpot); cudaFree(ctx->dJobKey);
  cudaFree(ctx->dASeed); cudaFree(ctx->dBSeed);
  cudaFree(ctx->dTarget); cudaFree(ctx->dHash); cudaFree(ctx->dIsShare);
  delete ctx;
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt);

extern "C" void pearl_host_set_job(void *handle, const uint8_t *header,
                                   const uint8_t *target) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  pearl_host_reseed(handle, 0);
}

// Re-draw the operands under a new salt and rebuild everything downstream of
// them: the commitments, the seeds, the noise, and the noised operands.
//
// This is the outer loop of the search. One salt yields m*n regions and nothing
// more, because the region index is just (row offset, column offset) -- so the
// miner must periodically pick new operands or it re-mines what it has already
// tried, at full reported hashrate and with no chance of a share.
extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  ctx->salt = salt;
  const uint8_t *header = ctx->header;
  const uint8_t *target = ctx->target;

  // job_key = blake3(header76 ‖ config52). Computed on-device so there is one
  // BLAKE3 implementation in the binary rather than two that can disagree.
  uint8_t seedInput[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(seedInput, header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, seedInput + PEARL_HEADER_BYTES);

  uint8_t *dSeedInput = nullptr;
  cudaMalloc(&dSeedInput, sizeof(seedInput));
  cudaMemcpy(dSeedInput, seedInput, sizeof(seedInput), cudaMemcpyHostToDevice);

  // job_key = blake3(header76 ‖ config52), UNKEYED.
  //
  // This used to hash it KEYED with an all-zero key, under the belief that a
  // zero key is the same as no key. It is not: keyed mode seeds the chaining
  // value from the key and sets KEYED_HASH, so the two produce different
  // digests. The device and the oracle therefore derived different job keys —
  // and, both being internally consistent, nothing anywhere said so.
  pearl_blake3_unkeyed<<<1, 1>>>(dSeedInput, sizeof(seedInput),
                                 reinterpret_cast<uint8_t *>(ctx->dJobKey));
  cudaFree(dSeedInput);

  const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
  const size_t bLen = (size_t)ctx->profile.n * ctx->profile.k;

  // Operands and the noise factors, regenerated for this job's key.
  const uint32_t rank = ctx->profile.rank;
  const uint32_t k = ctx->profile.k;
  const int threads = 256;
  auto blocks = [&](size_t n) { return (unsigned)((n + threads - 1) / threads); };

  // The operands are the miner's own workload, so their contents are our choice
  // — but their RANGE is not. They must be int7: the noise adds another int7 and
  // the sum has to stay inside int8 for the Int7xInt7ToInt32 MMA.
  //
  // Keyed by job_key rather than by a commitment seed, so these streams cannot
  // collide with the noise streams even though they share the labels.
  pearl_gen_operand<<<blocks(aLen / 32 + 1), threads>>>(
      ctx->dJobKey, ctx->dLabelA, ctx->dA, aLen, salt);
  pearl_gen_operand<<<blocks(bLen / 32 + 1), threads>>>(
      ctx->dJobKey, ctx->dLabelB, ctx->dB, bLen, salt);

  // hash_a and hash_b: keyed BLAKE3 over the WHOLE operands. These are Merkle
  // trees over 1024-byte chunks, not one long chain — hashing them as a single
  // chunk (which this did until the device run showed a_seed == b_seed) gives the
  // wrong digest for anything over 1024 bytes and so the wrong seeds.
  cudaDeviceSynchronize();
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen,
                     ctx->dHashA, ctx->dTreeA, &ctx->layerOffA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen,
                     ctx->dHashB, ctx->dTreeB, &ctx->layerOffB);

  // Bind the roots before they enter the chain. Under cert-v3 each root is
  // re-hashed with its dimension under a domain-separation salt, which is what
  // commits m and n; legacy passes the raw roots straight through.
  if (ctx->profile.seed_derivation == PEARL_SEED_LEGACY) {
    cudaMemcpy(ctx->dBoundA, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
    cudaMemcpy(ctx->dBoundB, ctx->dHashB, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  } else {
    pearl_bind_root<<<1, 1>>>(ctx->dSaltA, ctx->dHashA, ctx->profile.m, ctx->dBoundA);
    pearl_bind_root<<<1, 1>>>(ctx->dSaltB, ctx->dHashB, ctx->profile.n, ctx->dBoundB);
    cudaDeviceSynchronize();
  }

  // b_seed = blake3(job_key ‖ hash_b), then a_seed = blake3(b_seed ‖ hash_a).
  // The order is NOT symmetric: b_seed is derived first and feeds a_seed.
  cudaMemcpy(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundB, PEARL_HASH_BYTES,
             cudaMemcpyDeviceToDevice);
  pearl_blake3_unkeyed<<<1, 1>>>(ctx->dSeedBuf, 64,
                               reinterpret_cast<uint8_t *>(ctx->dBSeed));

  cudaMemcpy(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundA, PEARL_HASH_BYTES,
             cudaMemcpyDeviceToDevice);
  pearl_blake3_unkeyed<<<1, 1>>>(ctx->dSeedBuf, 64,
                               reinterpret_cast<uint8_t *>(ctx->dASeed));
  cudaDeviceSynchronize();

  // The A side is keyed by a_seed and the B side by b_seed. Obvious as written,
  // but the reference destructures its tuple as
  //   let (b_noise_seed, a_noise_seed) = commitment_hash;
  // i.e. b first, so it is easy to end up with these swapped — silently, and
  // with no symptom other than shares that are never accepted.
  //
  // row_indices is null here because the whole operand is noised, so a row's
  // index IS its position. The parameter exists for the verifier's path, which
  // only ever wants the handful of rows in one tile.
  pearl_gen_dense<<<blocks((size_t)ctx->profile.m * (rank / 32)), threads>>>(
      ctx->dASeed, ctx->dLabelA, nullptr, ctx->dEAL, ctx->profile.m, rank);
  pearl_gen_dense<<<blocks((size_t)ctx->profile.n * (rank / 32)), threads>>>(
      ctx->dBSeed, ctx->dLabelB, nullptr, ctx->dEBR, ctx->profile.n, rank);
  pearl_gen_perm<<<blocks((k + 7) / 8), threads>>>(ctx->dASeed, ctx->dLabelA,
                                                   ctx->dPermA, k, rank);
  pearl_gen_perm<<<blocks((k + 7) / 8), threads>>>(ctx->dBSeed, ctx->dLabelB,
                                                   ctx->dPermB, k, rank);

  // Fold the noise into the operands ONCE. Each element costs two lookups and a
  // subtract, because E_AR and E_BL are sparse +-1 selectors rather than dense
  // factors — the version that reconstructed at full rank did rank times this
  // much work and computed the wrong thing.
  pearl_materialize<<<blocks(aLen), threads>>>(ctx->dA, ctx->dEAL, ctx->dPermA,
                                               ctx->dAp, ctx->profile.m, k, rank);
  pearl_materialize<<<blocks(bLen), threads>>>(ctx->dB, ctx->dEBR, ctx->dPermB,
                                               ctx->dBp, ctx->profile.n, k, rank);

  cudaMemcpy(ctx->dTarget, target, PEARL_HASH_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx->bSeed, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  ctx->haveJob = true;
}

extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base,
                                  uint32_t batch, PearlSearchResult *out,
                                  uint64_t *attempts, char *err,
                                  size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !ctx->haveJob || !out) return false;

  const uint32_t k = ctx->profile.k;
  const uint32_t rank = ctx->profile.rank;
  const uint32_t chunks = (k + rank - 1) / rank;
  // Clamped to m so one launch shares a single col_off: D is built for exactly
  // the columns that batch touches. nonce_base stays a multiple of m because the
  // caller advances by the attempt count we report back.
  // The caller's batch hint is advisory; the real width is the context's, since
  // the partial table was allocated for exactly that many column groups.
  (void)batch;
  const uint32_t regions = ctx->batch;
  const uint32_t col_groups = ctx->colBatch;
  // Block size for the search kernels. Tunable because occupancy against
  // register pressure is not something to guess at.
  static const int threads = []() {
    const char *e = getenv("PEARL_BLOCK");
    // 512 by default: sixteen warps make the CTA tile 128x256, which raises
    // the MACs bought per staged byte from 64 to 85 -- and the two-stage
    // pipeline hides the staging that a single 512-thread block used to expose.
    const int v = e ? atoi(e) : 512;
    return (v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) ? v : 256;
  }();
  const int warps_per_block = threads / 32;
  // A valid-offset INDEX; the kernel expands it into an actual offset.
  const uint32_t col_off =
      (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);

  // One fused launch: the tile fold keeps its accumulator across chunks, so
  // there are no reusable partials to stage and no second pass.
  const uint32_t warpsPerBlock = threads / 32;
  const uint32_t regionsPerWarp = PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT);
  // A block is a 2D grid of warps: PEARL_WARP_ROWS down, the rest across. Both
  // dimensions have to tile exactly, because a warp that falls outside cannot
  // return -- staging is a block-wide cooperative load and a __syncthreads()
  // some warps skip hangs the launch.
  const uint64_t rowBlocks = ctx->rowsValid / regionsPerWarp;
  const uint32_t warpCols = warpsPerBlock / PEARL_WARP_ROWS;
  const uint64_t colBlocks = col_groups / PEARL_WMMA_COL_BLK;
  // The staging walks base + stride rather than a table of addresses, which is
  // only the same sequence when quads divides the staging thread count and the
  // column step lands on a whole number of column groups.
  const uint32_t quadsPerRow = rank / 16u;
  const uint32_t stageThreads = (uint32_t)threads;
  if (quadsPerRow == 0 || stageThreads % quadsPerRow != 0
      || (stageThreads / quadsPerRow) % PEARL_COLS_COUNT != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "staging stride is not uniform: %u threads, %u quads a row, %u columns a group",
               stageThreads, quadsPerRow, (unsigned)PEARL_COLS_COUNT);
    return false;
  }
  if (warpsPerBlock % PEARL_WARP_ROWS != 0 || rowBlocks % PEARL_WARP_ROWS != 0
      || warpCols == 0 || colBlocks % warpCols != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "warp grid %ux%u does not tile %llu row blocks by %llu column blocks",
               PEARL_WARP_ROWS, warpCols, (unsigned long long)rowBlocks,
               (unsigned long long)colBlocks);
    return false;
  }
  const unsigned blocks =
      (unsigned)((rowBlocks / PEARL_WARP_ROWS) * (colBlocks / warpCols));
  // Two full-chunk stages; the transcripts live in registers and global now.
  const size_t smem = (size_t)PEARL_STAGE_BUFS
                      * ((size_t)warpCols * PEARL_WMMA_COL_BLK * 16
                         + (size_t)PEARL_WARP_ROWS * regionsPerWarp * PEARL_ROWS_COUNT)
                      * PEARL_SB_STRIDE;
  // The fold writes each transcript slot exactly once only when every chunk
  // has its own bucket. A geometry with more chunks than buckets would fold
  // into whatever the buffer already held; refuse it rather than mine garbage.
  if (chunks > PEARL_JACKPOT_BUCKETS) {
    if (err && err_len)
      snprintf(err, err_len, "%u chunks exceed %u transcript buckets", chunks,
               (unsigned)PEARL_JACKPOT_BUCKETS);
    return false;
  }
  // Staging both operands puts this past the 48 KB a block gets by default.
  // Ada allows 99 KB per block, but only when asked; without this the launch
  // fails with an invalid-configuration error rather than running slowly.
  static bool smemOptedIn = false;
  if (!smemOptedIn) {
    cudaFuncSetAttribute(reinterpret_cast<const void *>(pearl_tile_fold_wmma),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    smemOptedIn = true;
  }
  pearl_tile_fold_wmma<<<blocks, threads, smem>>>(
      ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,
      col_off, ctx->rowsValid, col_groups, ctx->dJackpot);

  // Hash every transcript and test it against the bound. finalize writes only
  // on a hit and appends to a compact list, so the readback below is four bytes
  // rather than one flag per region.
  cudaMemsetAsync(ctx->dHitCount, 0, sizeof(uint32_t));
  pearl_finalize_many<<<(regions + 255) / 256, 256>>>(
      ctx->dASeed, ctx->dJackpot, regions, ctx->dTarget, ctx->dHashes,
      ctx->dHitCount, ctx->dHitIndex, (int)ctx->profile.hash_big_endian);

  uint32_t hits = 0;
  cudaMemcpy(&hits, ctx->dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  if (attempts) *attempts = regions;

  if (hits > 0) {
    const uint32_t n_hits = hits < PEARL_MAX_HITS ? hits : PEARL_MAX_HITS;
    cudaMemcpy(ctx->hHitIndex.data(), ctx->dHitIndex, (size_t)n_hits * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // The kernel appends with an atomic, so the list is in an arbitrary order.
    // Take the LOWEST region index, which is what a sequential scan would have
    // returned — otherwise which share gets submitted varies run to run.
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_hits; i++) {
      if (ctx->hHitIndex[i] < ctx->hHitIndex[best]) best = i;
    }
    cudaMemcpy(out->jackpot_hash, ctx->dHashes + (size_t)best * PEARL_HASH_BYTES,
               PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    memcpy(out->a_seed, ctx->aSeed, PEARL_HASH_BYTES);
    memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
    out->nonce = nonce_base + ctx->hHitIndex[best];
    out->salt = ctx->salt;

    // Capture the proof NOW, while the operands and tree still belong to this
    // hit. A few tens of milliseconds later they will have been re-drawn.
    {
      // The GLOBAL region index, not the batch-local one. Both give the same
      // row offset, because nonce_base is a multiple of rowsValid -- but the
      // COLUMN offset is (region / rowsValid) % colsValid, and the local index
      // drops the batch base entirely. The columns in the snapshot then belong
      // to a different tile than the row indices the proof declares, which the
      // pool reports as "Failed to extract strip".
      const uint64_t region = out->nonce;
      const uint32_t rowIdx = (uint32_t)(region % ctx->rowsValid);
      const uint32_t colIdx = (uint32_t)((region / ctx->rowsValid) % ctx->colsValid);
      const uint32_t rowOff = pearl_expand_offset(rowIdx, PEARL_ROWS_MASK);
      const uint32_t colOff = pearl_expand_offset(colIdx, PEARL_COLS_MASK);

      uint32_t rows[PEARL_ROWS_COUNT], cols[PEARL_COLS_COUNT];
      for (int i = 0; i < PEARL_ROWS_COUNT; i++) rows[i] = rowOff | PEARL_ROWS_PATTERN[i];
      for (int i = 0; i < PEARL_COLS_COUNT; i++) cols[i] = colOff | PEARL_COLS_PATTERN[i];

      snapshotProof(ctx, true, rows, PEARL_ROWS_COUNT, &out->proof_a.leaf_indices,
                    &out->proof_a.leaves, &out->proof_a.siblings);
      snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices,
                    &out->proof_bt.leaves, &out->proof_bt.siblings);
      cudaMemcpy(out->proof_a.root, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      cudaMemcpy(out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
      out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;
    }
    out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
    cudaMemcpy(out->proof.data(),
               ctx->dJackpot + (size_t)ctx->hHitIndex[best] * PEARL_JACKPOT_BUCKETS,
               PEARL_JACKPOT_BUCKETS * 4, cudaMemcpyDeviceToHost);
    out->found = true;
    return true;
  }
  out->found = false;
  return false;
}

// ---------------------------------------------------------------------------
// Share-proof accessors.
//
// A submitted share carries the 1024-byte operand chunks its tile touched plus
// the sibling digests that authenticate them against the committed root. Both
// already exist on the device -- the chunks in the operand, the digests in the
// commitment tree -- so the host copies out the handful it needs instead of
// rebuilding the tree, which is thousands of times more work.
// ---------------------------------------------------------------------------

// Copy whole 1024-byte leaf chunks out of an operand.// Copy whole 1024-byte leaf chunks out of an operand.
extern "C" bool pearl_host_leaf_chunks(void *handle, int isA,
                                       const uint32_t *leaf_indices,
                                       uint32_t count, uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !leaf_indices || !out) return false;
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  const uint64_t bytes = isA ? (uint64_t)ctx->profile.m * ctx->profile.k
                             : (uint64_t)ctx->profile.n * ctx->profile.k;
  for (uint32_t i = 0; i < count; i++) {
    const uint64_t off = (uint64_t)leaf_indices[i] * 1024;
    if (off + 1024 > bytes) return false;
    cudaMemcpy(out + (size_t)i * 1024, src + off, 1024, cudaMemcpyDeviceToHost);
  }
  return true;
}

// Copy 32-byte nodes out of one level of a commitment tree.
extern "C" bool pearl_host_tree_nodes(void *handle, int isA, uint32_t level,
                                      const uint32_t *indices, uint32_t count,
                                      uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !indices || !out) return false;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  if (!tree || level >= offs.size()) return false;
  const uint64_t base = offs[level];
  for (uint32_t i = 0; i < count; i++) {
    cudaMemcpy(out + (size_t)i * PEARL_HASH_BYTES,
               tree + (base + indices[i]) * 8, PEARL_HASH_BYTES,
               cudaMemcpyDeviceToHost);
  }
  return true;
}

// How many levels the tree has, so the host knows where to stop walking.
extern "C" uint32_t pearl_host_tree_levels(void *handle, int isA) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return 0;
  return (uint32_t)(isA ? ctx->layerOffA.size() : ctx->layerOffB.size());
}
