// The RTX 20 wide-tile fold's host side. Included by pearl_host.cu, inside its anonymous namespace, after
// pearl_host_rot.inc, whose driver calls, file checks and tile it reuses. Nothing here runs unless the card is
// compute 7.5 (RTX 20, GTX 16 have no tensor cores and never get here), the miner let the core choose
// (cardK), PEARL_FOLD_T96 is not 0 and the fold's cubin, pearl_t96_sm75.cubin, is beside the addon (or
// PEARL_T96_CUBIN names a file). The release fold and everything it does are untouched.
//
// What it is: the release's B-direct fold with a warp tile of 96 x 64 (192 sums a lane) in place of 64 x 64,
// so a block is 192 rows by 256 columns. ptxas cannot fit 192 sums and their operands in 255 registers, so
// the k loop is written by hand (probes/turing/t96gen.py) and the kernel is a cubin loaded through the
// driver (src/pearl_t96.cu), as the rotation fold's is. It mines the profile's k.
//
// What is different from the release fold, and so what this file does itself:
//   - The tile: one thread's sums, rows {0, 8, .. 88} by columns {0, 1, 8, 9, .. 56, 57}, 12 by 16: the
//     rotation fold's (kRotRows, kRotCols). So its own config52 and its own count of regions a salt. A tile
//     of 256 regions of 192 k MACs is 192 of the binding's attempts (256 k MACs each), as there.
//   - A' and B' are drawn in the fold's own fragment order (t96_aux_mat in the cubin): a load is four
//     fragments of one k16 half.
//   - m 131072 is not a multiple of 192: the last 128 rows are not mined.
//   - The card is not taken on trust. When a context is made, the same tiles go through the hand-written
//     kernel and through the same kernel as ptxas compiled it (t96_ref_fold), and every transcript word must
//     be equal. If not, or if anything else here fails, the context mines on the release fold.
// What is the release's, reused as it is: the operands and their trees, the seed chain, pearl_gen_dense and
// pearl_gen_perm, the restamp, the pipeline, the hit list and the proof capture.

#define PEARL_T96_FILE "pearl_t96_sm75.cubin"
#define PEARL_T96_K 2048u
#define PEARL_T96_SMEM (12u * 2048u)

struct T96State {
  CUmodule mod = nullptr;
  CUfunction fFold = nullptr, fRef = nullptr, fMat = nullptr;
  RotLaunch launch = nullptr;
  RotUnload unload = nullptr;
  uint32_t rgCount = 0, cgCount = 0, band = 1;   // row groups of 192, column groups of 256, column groups a band
  uint32_t rowsValid = 0;                        // valid row offsets (the share's nonce space)
  uint64_t tiles = 0;                            // tiles a salt
  uint32_t tilesPerBatch = 0, sms = 1;
};

// The cubins this core loads from beside itself: the hashes src/pearl_t96_sm75.ref lists, the files that have
// mined on an RTX 20 card with every hit verified (test/t96FoldShip.test.js keeps the two lists the same). A
// file PEARL_T96_CUBIN names is a developer's own and is not held to the list.
const char *const kT96Cubins[] = {
    "",  // none yet: the first build has not mined on a card
};

bool t96_off() {
  const char *v = getenv("PEARL_FOLD_T96");
  return v && v[0] == '0' && v[1] == 0;
}

bool t96_named() {
  const char *e = getenv("PEARL_T96_CUBIN");
  return e && *e;
}

// Where the cubin is: the file PEARL_T96_CUBIN names, or PEARL_T96_FILE beside this module.
bool t96_cubin_path(char *buf, size_t len) {
  if (t96_named()) {
    snprintf(buf, len, "%s", getenv("PEARL_T96_CUBIN"));
    FILE *f = fopen(buf, "rb");
    if (f) fclose(f);
    return f != nullptr;
  }
  return rot_cubin_path_for(reinterpret_cast<const void *>(&t96_off), buf, len, PEARL_T96_FILE);
}

bool t96_listed(const std::vector<char> &image) {
  if (t96_named()) return true;
  std::vector<uint8_t> copy(image.begin(), image.end() - 1);
  char hex[65] = {0};
  if (!rot_fold_hash(copy.data(), copy.size(), hex)) return false;
  for (const char *h : kT96Cubins)
    if (strcmp(h, hex) == 0) return true;
  return false;
}

// Whether the card is one the fold is for at all: compute 7.5 with tensor cores. A GTX 16 card is compute 7.5
// too and has none; the release runs the dp4a fold there (Ctx::foldDp4a), and t96_create refuses any context
// that is not on the B-direct fold.
bool t96_arch_ok(int device) {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
      || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return major == 7 && minor == 5;
}

// Whether this card and profile can run the fold (the switch aside). `why` gets the reason when not.
bool t96_card_ok(const PearlProfile *p, int device, char *why, size_t why_len) {
  auto no = [&](const char *msg) {
    if (why && why_len) snprintf(why, why_len, "%s", msg);
    return false;
  };
  if (!t96_arch_ok(device)) return no("it is for RTX 20 cards");
  if (p->rank != 128u || p->k != PEARL_T96_K || p->m < 192u || p->m % 32u != 0u || p->n < 256u || p->n % 256u != 0u)
    return no("it does not mine this profile");
  char path[1024];
  if (!t96_cubin_path(path, sizeof path))
    return no(t96_named() ? "the file PEARL_T96_CUBIN names cannot be read" : "its file is not beside the Pearl core");
  std::vector<char> image;
  if (!rot_read_file(path, &image)) return no("its file is not a kernel file");
  if (!t96_listed(image)) return no("its file is not the one this core was built with");
  return true;
}

// The fold's rows pattern in a config52 (PeriodicPattern::to_bytes: one dimension of stride 8, length 12).
void t96_config_rows(uint8_t *config52) { memcpy(config52 + 8, kRotRowsBytes, 6); }

// Tile `tile` of a salt: its row group and column group. Bands of `band` column groups: a band's B stays in
// the L2 while every row group's A streams past it, a band's columns first (the release's walk,
// PEARL_BD_WALK 1). The kernel's t96_coords is the same arithmetic.
void t96_coords(const T96State *r, uint64_t tile, uint32_t *rbg, uint32_t *cbg) {
  const uint64_t bandTiles = (uint64_t)r->band * r->rgCount;
  const uint32_t bnd = (uint32_t)(tile / bandTiles), inb = (uint32_t)(tile % bandTiles);
  const uint32_t first = bnd * r->band;
  const uint32_t colsIn = r->cgCount - first < r->band ? r->cgCount - first : r->band;
  *cbg = first + inb % colsIn;
  *rbg = inb / colsIn;
}

// A hit's region: the share's nonce and the tile's rows and columns, from the batch's base and the fold's
// index (256 a tile, then the thread). The thread's place in the block is pearl_t96.cu's contract.
void t96_region(const Ctx *ctx, uint64_t nonce_base, uint32_t index, uint64_t *nonce, uint32_t *rows,
                uint32_t *cols) {
  const T96State *r = ctx->t96;
  const uint64_t tile = nonce_base / PEARL_ROT_UNITS + index / 256u;
  const uint32_t tid = index % 256u, warp = tid >> 5, lane = tid & 31u;
  uint32_t rbg = 0, cbg = 0;
  t96_coords(r, tile, &rbg, &cbg);
  const uint32_t Q = 2u * rbg + (warp >> 2), g = lane >> 2;
  const uint32_t J = 4u * cbg + (warp & 3u), q = lane & 3u;
  const uint32_t rowIdx = 8u * Q + g, colIdx = 4u * J + q;
  *nonce = (uint64_t)rowIdx + (uint64_t)r->rowsValid * colIdx;
  for (int j = 0; j < PEARL_ROT_ROWS; j++) rows[j] = 96u * Q + g + kRotRows[j];
  for (int j = 0; j < PEARL_ROT_COLS; j++) cols[j] = 64u * J + 2u * q + kRotCols[j];
}

// One operand noised into the fold's fragment order: pearl_materialize16_afrag's arguments, on the default
// stream as the release's draws are.
void t96_materialize(Ctx *ctx, const int8_t *src, const int8_t *dense, const uint32_t *perm, int8_t *dst,
                     uint32_t rows, uint32_t kLog2, uint32_t rank, uint64_t readVecs, uint32_t fillWord) {
  T96State *r = ctx->t96;
  void *args[9] = {&src, &dense, &perm, &dst, &rows, &kLog2, &rank, &readVecs, &fillWord};
  const size_t vecs = ((size_t)rows << kLog2) / 16u;
  r->launch(r->fMat, (unsigned)((vecs + 255u) / 256u), 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
}

bool t96_launch_fold(T96State *r, CUfunction fn, CUstream st, const int8_t *ap, const int8_t *bp, uint32_t *ctr,
                     uint32_t *dump, uint32_t tile0, uint32_t tiles, PearlTranscriptTest test,
                     PearlHitList hits, int *code) {
  uint32_t rg = r->rgCount, cg = r->cgCount, band = r->band;
  void *args[11] = {&ap, &bp, &ctr, &dump, &rg, &cg, &band, &tile0, &tiles, &test, &hits};
  const unsigned blocks = tiles < r->sms ? tiles : r->sms;
  const CUresult rc = r->launch(fn, blocks, 1, 1, 256, 1, 1, PEARL_T96_SMEM, st, args, nullptr);
  if (code) *code = (int)rc;
  return rc == CUDA_SUCCESS;
}

bool t96_submit(Ctx *ctx, int slot, cudaStream_t st, uint64_t nonce_base, uint64_t *regions_out, char *err,
                size_t err_len) {
  T96State *r = ctx->t96;
  const uint64_t first = nonce_base / PEARL_ROT_UNITS;
  PearlHitList hitList = pearl_slot_hits(ctx, slot);
  cudaMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);
  if (first >= r->tiles) {
    // What the salt's tiles leave of the binding's count of attempts: one empty batch.
    const uint64_t span = ((uint64_t)ctx->profile.m / PEARL_ROWS_COUNT) * ((uint64_t)ctx->profile.n / PEARL_COLS_COUNT);
    if (!pearl_queue_batch(ctx, slot, st, nonce_base, 0u, nullptr, err, err_len)) return false;
    if (regions_out) *regions_out = span > nonce_base ? span - nonce_base : 1u;
    return true;
  }
  const uint64_t left = r->tiles - first;
  const uint32_t tiles = left < r->tilesPerBatch ? (uint32_t)left : r->tilesPerBatch;
  // The blocks take their tiles from the slot's counter, the word the release's persistent fold uses.
  uint32_t *ctr = hitList.count + PEARL_BD_CTR_SLOTS;
  cudaMemsetAsync(ctr, 0, sizeof(uint32_t), st);
  int code = 0;
  if (!t96_launch_fold(r, r->fFold, (CUstream)st, ctx->dAp, ctx->dBp, ctr, nullptr, (uint32_t)first, tiles,
                       pearl_slot_test(ctx), hitList, &code)) {
    if (err && err_len) snprintf(err, err_len, "wide-tile fold launch failed: driver error %d", code);
    return false;
  }
  return pearl_queue_batch(ctx, slot, st, nonce_base, tiles * PEARL_ROT_UNITS, regions_out, err, err_len);
}

void t96_destroy(Ctx *ctx) {
  T96State *r = ctx->t96;
  if (!r) return;
  if (r->mod && r->unload) r->unload(r->mod);
  delete r;
  ctx->t96 = nullptr;
}

// The startup check: `tiles` tiles of made-up operand bytes through the hand-written kernel and through the
// compiled one, every region's 16 transcript words compared. The bytes go where a salt's first tiles read
// (row group 0, the first band's column groups); the job's draw overwrites them. False with `why` set when the
// two differ, the words are all zero, or a launch fails.
bool t96_startup_check(Ctx *ctx, char *why, size_t why_len) {
  T96State *r = ctx->t96;
  auto no = [&](const char *msg, int code) {
    if (why && why_len) snprintf(why, why_len, "%s (%d)", msg, code);
    return false;
  };
  const uint32_t tiles = 2u * r->sms < r->tiles ? 2u * r->sms : (uint32_t)r->tiles;
  // Every row group and column group those tiles touch.
  uint32_t rgTop = 0, cgTop = 0;
  for (uint32_t v = 0; v < tiles; v++) {
    uint32_t rbg = 0, cbg = 0;
    t96_coords(r, v, &rbg, &cbg);
    if (rbg > rgTop) rgTop = rbg;
    if (cbg > cgTop) cgTop = cbg;
  }
  const size_t aBytes = (size_t)(rgTop + 1u) * 192u * PEARL_T96_K, bBytes = (size_t)(cgTop + 1u) * 256u * PEARL_T96_K;
  std::vector<int8_t> bytes(aBytes > bBytes ? aBytes : bBytes);
  uint32_t x = 0x9E3779B9u;
  auto fill = [&](size_t n) {
    for (size_t i = 0; i < n; i++) {
      x = x * 1664525u + 1013904223u;
      bytes[i] = (int8_t)(x >> 24);
    }
  };
  fill(aBytes);
  if (cudaMemcpy(ctx->dAp, bytes.data(), aBytes, cudaMemcpyHostToDevice) != cudaSuccess) return no("copying A", 0);
  fill(bBytes);
  if (cudaMemcpy(ctx->dBp, bytes.data(), bBytes, cudaMemcpyHostToDevice) != cudaSuccess) return no("copying B", 0);
  const size_t words = (size_t)tiles * 256u * 16u;
  uint32_t *dDump = nullptr, *dCtr = nullptr;
  if (cudaMalloc(&dDump, 2u * words * 4u) != cudaSuccess || cudaMalloc(&dCtr, 8) != cudaSuccess) {
    (void)cudaGetLastError();
    cudaFree(dDump);
    return no("allocating the check's buffers", 0);
  }
  PearlTranscriptTest test;
  memset(&test, 0, sizeof test);
  PearlHitList hits;
  memset(&hits, 0, sizeof hits);
  hits.count = dCtr + 1;
  bool ok = true;
  int code = 0;
  cudaMemset(dDump, 0, 2u * words * 4u);
  cudaMemset(dCtr, 0, 8);
  ok = t96_launch_fold(r, r->fFold, nullptr, ctx->dAp, ctx->dBp, dCtr, dDump, 0u, tiles, test, hits, &code);
  if (ok) ok = cudaDeviceSynchronize() == cudaSuccess;
  if (ok) {
    cudaMemset(dCtr, 0, 8);
    ok = t96_launch_fold(r, r->fRef, nullptr, ctx->dAp, ctx->dBp, dCtr, dDump + words, 0u, tiles, test, hits, &code);
  }
  if (ok) ok = cudaDeviceSynchronize() == cudaSuccess;
  std::vector<uint32_t> got(2u * words);
  if (ok) ok = cudaMemcpy(got.data(), dDump, 2u * words * 4u, cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(dDump);
  cudaFree(dCtr);
  if (!ok) {
    (void)cudaGetLastError();
    return no("a kernel of the check did not run", code);
  }
  size_t wrong = 0, zero = 0;
  for (size_t i = 0; i < words; i++) {
    if (got[i] != got[words + i]) wrong++;
    if (got[i] == 0u) zero++;
  }
  if (wrong) return no("the hand-written loop and the compiled one disagree on this card, words wrong", (int)wrong);
  if (zero > words / 16u) return no("the check's transcripts are empty", (int)zero);
  return true;
}

// Makes the context a wide-tile one. False, with ctx->t96 null and `why` set, when it cannot: the context then
// mines on the release fold as it was made.
bool t96_create(Ctx *ctx, char *why, size_t why_len) {
  const PearlProfile &p = ctx->profile;
  auto no = [&](const char *msg, int code) {
    if (why && why_len && !why[0]) snprintf(why, why_len, "%s (%d)", msg, code);
    t96_destroy(ctx);
    return false;
  };
  if (!ctx->foldBDirect || ctx->foldDp4a) return no("the card is not on the release's RTX 20 fold", 0);
  T96State *r = new T96State();
  ctx->t96 = r;
  char path[1024];
  std::vector<char> image;
  if (!t96_cubin_path(path, sizeof path) || !rot_read_file(path, &image) || !t96_listed(image))
    return no("cannot read the fold's cubin", 0);
  RotLoadData load = nullptr;
  RotGetFunction getfn = nullptr;
  RotSetAttr setattr = nullptr;
  if (!rot_sym("cuModuleLoadData", &load) || !rot_sym("cuModuleGetFunction", &getfn)
      || !rot_sym("cuFuncSetAttribute", &setattr) || !rot_sym("cuLaunchKernel", &r->launch)
      || !rot_sym("cuModuleUnload", &r->unload))
    return no("the driver's module calls are not there", 0);
  cudaDeviceSynchronize();
  CUresult rc = load(&r->mod, image.data());
  if (rc != CUDA_SUCCESS) return no("the driver refused the cubin", (int)rc);
  if ((rc = getfn(&r->fFold, r->mod, "pearl_t96_fold")) != CUDA_SUCCESS
      || (rc = getfn(&r->fRef, r->mod, "t96_ref_fold")) != CUDA_SUCCESS
      || (rc = getfn(&r->fMat, r->mod, "t96_aux_mat")) != CUDA_SUCCESS)
    return no("a kernel is missing from the cubin", (int)rc);
  if ((rc = setattr(r->fFold, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)PEARL_T96_SMEM)) != CUDA_SUCCESS
      || (rc = setattr(r->fRef, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)PEARL_T96_SMEM)) != CUDA_SUCCESS)
    return no("the card refused the fold's shared memory", (int)rc);
  r->rgCount = p.m / 192u;
  r->cgCount = p.n / 256u;
  r->band = ctx->bdBand ? ctx->bdBand : 4u;
  uint32_t v = 0;
  if (pearl_env_u32("PEARL_T96_BAND", &v) && v >= 1u && v <= 64u) r->band = v;
  if (r->band > r->cgCount) r->band = r->cgCount;
  r->tiles = (uint64_t)r->rgCount * r->cgCount;
  r->rowsValid = (p.m / 96u) * 8u;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);
  r->sms = sms > 0 ? (uint32_t)sms : 1u;
  // A batch as long as the release's: its attempts, in tiles.
  uint32_t tpb = ctx->batch / PEARL_ROT_UNITS;
  if (pearl_env_u32("PEARL_T96_TILES", &v) && v >= 1u && v <= (1u << 20)) tpb = v;
  if (tpb < r->sms) tpb = r->sms;
  r->tilesPerBatch = tpb;
  char cw[160] = {0};
  if (!t96_startup_check(ctx, cw, sizeof cw)) {
    if (why && why_len) snprintf(why, why_len, "its startup check failed: %s", cw);
    return no("", 0);
  }
  ctx->batch = r->tilesPerBatch * PEARL_ROT_UNITS;
  fprintf(stderr,
          "[pearl] GPU %d: wide-tile fold, k %u, tile 12x16, blocks of 192 x 256, %u row groups by %u column "
          "groups a salt, band %u, %u tiles a batch; startup check passed\n",
          ctx->device, p.k, r->rgCount, r->cgCount, r->band, r->tilesPerBatch);
  return true;
}
