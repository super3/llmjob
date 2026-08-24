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

#include <vector>

#include "pearl_config.h"

// Declared in pearl_kernel.cu.
extern "C" __global__ void pearl_gen_noise(const uint32_t *seed, int8_t *out,
                                           uint32_t rows, uint32_t rank,
                                           uint64_t index_base);
extern "C" __global__ void pearl_hash_operands(const uint32_t *job_key,
                                               const uint8_t *padded,
                                               uint32_t len, uint8_t *out);
extern "C" __global__ void pearl_materialize(const int8_t *base, const int8_t *EL,
                                             const int8_t *ER, int32_t *out,
                                             uint32_t rows, uint32_t k,
                                             uint32_t rank);
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *Aprime, const int32_t *Bprime, const uint32_t *rows_pattern,
    const uint32_t *cols_pattern, uint32_t rows_count, uint32_t cols_count,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank, uint32_t chunks,
    uint64_t region_base, uint32_t *jackpot_out);
extern "C" __global__ void pearl_finalize_many(const uint32_t *a_seed,
                                               const uint32_t *jackpots,
                                               uint32_t count,
                                               const uint8_t *target_be,
                                               uint8_t *hashes_out,
                                               int *flags_out);
extern "C" __global__ void pearl_blake3_chunk_cvs(const uint32_t *key,
                                                  const uint8_t *data,
                                                  uint64_t chunks,
                                                  uint32_t *cvs_out);
extern "C" __global__ void pearl_blake3_parent_layer(const uint32_t *key,
                                                     const uint32_t *in_cvs,
                                                     uint64_t pairs,
                                                     uint32_t is_root,
                                                     uint32_t *out_cvs);
extern "C" __global__ void pearl_blake3_short(const uint8_t *in, uint32_t len,
                                              uint8_t *out);
extern "C" __global__ void pearl_finalize(const uint32_t *a_seed,
                                          const uint32_t *jackpot,
                                          const uint8_t *target_be,
                                          uint8_t *hash_out, int *is_share);

// Mirrors the struct pearl_core.cc declares. Kept in this one header-free form
// deliberately: the two files must agree on the layout, and a shared header that
// pulled in <napi.h> would drag Node headers into nvcc.
struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES];
  uint8_t a_seed[PEARL_HASH_BYTES];
  uint8_t b_seed[PEARL_HASH_BYTES];
  uint64_t nonce;
  std::vector<uint8_t> proof;
  bool found;
};

namespace {

struct Ctx {
  PearlProfile profile;

  // Operands, generated once per job and then read by every region.
  int8_t *dA = nullptr;   // [m, k]
  int8_t *dB = nullptr;   // [n, k]  (Bᵀ, row-major)

  // Low-rank noise factors. Small next to the operands.
  // The noised operands, computed once per job. int32 because a rank-128 sum of
  // int7 products reaches ~5e5 — int8 here would truncate every value.
  int32_t *dAp = nullptr; // [m, k]
  int32_t *dBp = nullptr; // [n, k]

  int8_t *dEAL = nullptr; // [m, rank]
  int8_t *dEAR = nullptr; // [rank, k]
  int8_t *dEBL = nullptr; // [n, rank]
  int8_t *dEBR = nullptr; // [rank, k]

  uint32_t *dRows = nullptr;
  uint32_t *dCols = nullptr;
  uint32_t *dJackpot = nullptr;   // [batch][16]
  uint8_t *dHashes = nullptr;     // [batch][32]
  int *dFlags = nullptr;          // [batch]
  std::vector<int> hFlags;
  uint32_t batch = 0;
  uint32_t *dJobKey = nullptr;
  uint32_t *dASeed = nullptr;
  uint32_t *dBSeed = nullptr;
  uint32_t *dCvs = nullptr;      // BLAKE3 tree scratch (leaf CVs, reduced in place)
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
                        uint8_t *out32) {
  const uint64_t chunks = len / 1024;
  const int threads = 256;
  if (chunks <= 1) {
    // One chunk or less: the chunk's own final compression carries ROOT.
    pearl_hash_operands<<<1, 1>>>(ctx->dJobKey, data, (uint32_t)len, out32);
    return;
  }
  pearl_blake3_chunk_cvs<<<(unsigned)((chunks + threads - 1) / threads), threads>>>(
      ctx->dJobKey, data, chunks, ctx->dCvs);
  uint64_t count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2;
    const uint32_t isRoot = (pairs == 1) ? 1u : 0u;
    pearl_blake3_parent_layer<<<(unsigned)((pairs + threads - 1) / threads), threads>>>(
        ctx->dJobKey, ctx->dCvs, pairs, isRoot, ctx->dCvs);
    count = pairs;
  }
  cudaMemcpy(out32, ctx->dCvs, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
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

  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const size_t noiseBytes =
      (size_t)profile->m * rank + rank * k + (size_t)profile->n * rank + rank * k;
  // The materialised operands are int32 and dominate the budget: at mainnet
  // that is 2 GiB on top of the 1 GiB of int8 sources. Checked up front so an
  // undersized card gets a sentence rather than an abort inside a kernel.
  const size_t primeBytes = (aBytes + bBytes) * sizeof(int32_t);
  const size_t need = aBytes + bBytes + primeBytes + noiseBytes + (1u << 20);

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
  CUDA_OK(cudaMalloc(&ctx->dAp, aBytes * sizeof(int32_t)), "allocating the noised A");
  CUDA_OK(cudaMalloc(&ctx->dBp, bBytes * sizeof(int32_t)), "allocating the noised B");
  CUDA_OK(cudaMalloc(&ctx->dEAL, (size_t)profile->m * rank), "allocating E_AL");
  CUDA_OK(cudaMalloc(&ctx->dEAR, rank * k), "allocating E_AR");
  CUDA_OK(cudaMalloc(&ctx->dEBL, (size_t)profile->n * rank), "allocating E_BL");
  CUDA_OK(cudaMalloc(&ctx->dEBR, rank * k), "allocating E_BR");
  // One transcript, hash and verdict per region in a batch, so a batch is two
  // kernel launches and ONE copy back rather than 4096 round trips.
  ctx->batch = PEARL_BATCH_REGIONS;
  ctx->hFlags.resize(ctx->batch);
  CUDA_OK(cudaMalloc(&ctx->dJackpot,
                     (size_t)ctx->batch * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
          "allocating the transcripts");
  CUDA_OK(cudaMalloc(&ctx->dHashes, (size_t)ctx->batch * PEARL_HASH_BYTES),
          "allocating the batch hashes");
  CUDA_OK(cudaMalloc(&ctx->dFlags, (size_t)ctx->batch * sizeof(int)),
          "allocating the batch verdicts");
  CUDA_OK(cudaMalloc(&ctx->dJobKey, 8 * sizeof(uint32_t)), "allocating job_key");
  CUDA_OK(cudaMalloc(&ctx->dASeed, 8 * sizeof(uint32_t)), "allocating a_seed");
  CUDA_OK(cudaMalloc(&ctx->dBSeed, 8 * sizeof(uint32_t)), "allocating b_seed");
  // Leaf CVs for the larger of the two operands: 8 words per 1024-byte chunk.
  ctx->cvCapacity = (aBytes > bBytes ? aBytes : bBytes) / 1024;
  if (ctx->cvCapacity < 1) ctx->cvCapacity = 1;
  CUDA_OK(cudaMalloc(&ctx->dCvs, ctx->cvCapacity * 8 * sizeof(uint32_t)),
          "allocating the BLAKE3 tree scratch");
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
  cudaFree(ctx->dEAL); cudaFree(ctx->dEAR);
  cudaFree(ctx->dEBL); cudaFree(ctx->dEBR);
  cudaFree(ctx->dRows); cudaFree(ctx->dCols);
  cudaFree(ctx->dHashes); cudaFree(ctx->dFlags);
  cudaFree(ctx->dCvs); cudaFree(ctx->dSeedBuf);
  cudaFree(ctx->dHashA); cudaFree(ctx->dHashB);
  cudaFree(ctx->dJackpot); cudaFree(ctx->dJobKey);
  cudaFree(ctx->dASeed); cudaFree(ctx->dBSeed);
  cudaFree(ctx->dTarget); cudaFree(ctx->dHash); cudaFree(ctx->dIsShare);
  delete ctx;
}

extern "C" void pearl_host_set_job(void *handle, const uint8_t *header,
                                   const uint8_t *target) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;

  // job_key = blake3(header76 ‖ config52). Computed on-device so there is one
  // BLAKE3 implementation in the binary rather than two that can disagree.
  uint8_t seedInput[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(seedInput, header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, seedInput + PEARL_HEADER_BYTES);

  uint8_t *dSeedInput = nullptr;
  cudaMalloc(&dSeedInput, sizeof(seedInput));
  cudaMemcpy(dSeedInput, seedInput, sizeof(seedInput), cudaMemcpyHostToDevice);

  // An all-zero key gives the unkeyed derivation the spec uses for job_key.
  uint32_t zeroKey[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  cudaMemcpy(ctx->dJobKey, zeroKey, sizeof(zeroKey), cudaMemcpyHostToDevice);
  pearl_hash_operands<<<1, 1>>>(ctx->dJobKey, dSeedInput, sizeof(seedInput),
                                reinterpret_cast<uint8_t *>(ctx->dJobKey));
  cudaFree(dSeedInput);

  const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
  const size_t bLen = (size_t)ctx->profile.n * ctx->profile.k;

  // Operands and the noise factors, regenerated for this job's key.
  const uint32_t rank = ctx->profile.rank;
  const uint32_t k = ctx->profile.k;
  const int threads = 256;
  auto blocks = [&](size_t n) { return (unsigned)((n + threads - 1) / threads); };

  pearl_gen_noise<<<blocks((size_t)ctx->profile.m * k), threads>>>(
      ctx->dJobKey, ctx->dA, ctx->profile.m, k, 0);
  pearl_gen_noise<<<blocks((size_t)ctx->profile.n * k), threads>>>(
      ctx->dJobKey, ctx->dB, ctx->profile.n, k, 1ull << 40);

  // hash_a and hash_b: keyed BLAKE3 over the WHOLE operands. These are Merkle
  // trees over 1024-byte chunks, not one long chain — hashing them as a single
  // chunk (which this did until the device run showed a_seed == b_seed) gives the
  // wrong digest for anything over 1024 bytes and so the wrong seeds.
  cudaDeviceSynchronize();
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen, ctx->dHashA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen, ctx->dHashB);

  // b_seed = blake3(job_key ‖ hash_b), then a_seed = blake3(b_seed ‖ hash_a).
  // The order is NOT symmetric: b_seed is derived first and feeds a_seed.
  cudaMemcpy(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dHashB, PEARL_HASH_BYTES,
             cudaMemcpyDeviceToDevice);
  pearl_blake3_short<<<1, 1>>>(ctx->dSeedBuf, 64,
                               reinterpret_cast<uint8_t *>(ctx->dBSeed));

  cudaMemcpy(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dHashA, PEARL_HASH_BYTES,
             cudaMemcpyDeviceToDevice);
  pearl_blake3_short<<<1, 1>>>(ctx->dSeedBuf, 64,
                               reinterpret_cast<uint8_t *>(ctx->dASeed));
  cudaDeviceSynchronize();

  pearl_gen_noise<<<blocks((size_t)ctx->profile.m * rank), threads>>>(
      ctx->dBSeed, ctx->dEAL, ctx->profile.m, rank, 0);
  pearl_gen_noise<<<blocks((size_t)rank * k), threads>>>(
      ctx->dBSeed, ctx->dEAR, rank, k, 1ull << 20);
  pearl_gen_noise<<<blocks((size_t)ctx->profile.n * rank), threads>>>(
      ctx->dASeed, ctx->dEBL, ctx->profile.n, rank, 0);
  pearl_gen_noise<<<blocks((size_t)rank * k), threads>>>(
      ctx->dASeed, ctx->dEBR, rank, k, 1ull << 20);

  // Fold the noise into the operands ONCE, now that the factors exist. Every
  // region afterwards reads a finished value instead of rebuilding a rank-length
  // dot product per element per attempt.
  pearl_materialize<<<blocks(aLen), threads>>>(ctx->dA, ctx->dEAL, ctx->dEAR,
                                               ctx->dAp, ctx->profile.m, k, rank);
  pearl_materialize<<<blocks(bLen), threads>>>(ctx->dB, ctx->dEBL, ctx->dEBR,
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
  const uint32_t regions = batch < ctx->batch ? batch : ctx->batch;
  const int threads = 256;  // 8 warps, so 8 regions in flight per block

  // One warp per region now, so the grid is regions/warps-per-block and there is
  // no shared memory at all — the per-chunk reduction is a shuffle.
  const int warps_per_block = threads / 32;

  pearl_gemm_fold<<<(regions + warps_per_block - 1) / warps_per_block, threads>>>(
      ctx->dAp, ctx->dBp,
      ctx->dRows, ctx->dCols, PEARL_ROWS_COUNT, PEARL_COLS_COUNT,
      ctx->profile.m, ctx->profile.n, k, rank, chunks, nonce_base, ctx->dJackpot);

  pearl_finalize_many<<<(regions + 255) / 256, 256>>>(
      ctx->dASeed, ctx->dJackpot, regions, ctx->dTarget, ctx->dHashes, ctx->dFlags);

  cudaMemcpy(ctx->hFlags.data(), ctx->dFlags, (size_t)regions * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  if (attempts) *attempts = regions;

  for (uint32_t i = 0; i < regions; i++) {
    if (!ctx->hFlags[i]) continue;
    cudaMemcpy(out->jackpot_hash, ctx->dHashes + (size_t)i * PEARL_HASH_BYTES,
               PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    memcpy(out->a_seed, ctx->aSeed, PEARL_HASH_BYTES);
    memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
    out->nonce = nonce_base + i;
    out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
    cudaMemcpy(out->proof.data(),
               ctx->dJackpot + (size_t)i * PEARL_JACKPOT_BUCKETS,
               PEARL_JACKPOT_BUCKETS * 4, cudaMemcpyDeviceToHost);
    out->found = true;
    return true;
  }
  out->found = false;
  return false;
}
