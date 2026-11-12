// The rotation fold's host side. Included by pearl_host.cu, inside its anonymous namespace, after the
// restamp helpers it uses. OFF by default: nothing here runs unless PEARL_FOLD_ROT=1 is in the environment, the
// card is compute 8.9 (RTX 40), 8.6 (RTX 30) or 12.0 (RTX 50) and the card's cubin (pearl_rot_sm89.cubin,
// pearl_rot_sm86.cubin or pearl_rot_sm120.cubin, beside the addon, or the file PEARL_ROT_CUBIN names) is there.
// The first two hold the same hand-written loop: ptxas gives sm_86 the same code as sm_89 for this kernel, byte
// for byte. The sm_120 file has the same hand loop, encoded for sm_120 (probes/hand89/rfold120.py). The release fold and everything it does are untouched.
//
// What is different from the release fold, and so what this file does itself:
//   - The kernel is a cubin whose chunk loop was written by hand (src/pearl_rot.cu, probes/hand89/rfold.py). It
//     is loaded and launched through the driver (cuModuleLoadData, cuLaunchKernel), on the context and the
//     streams the runtime already uses.
//   - It mines k 65536 with its own tile: rows {0, 8, .. 88} by columns {0, 1, 8, 9, .. 56, 57}, 12 by 16, one
//     thread's sums. So its own config52, its own count of regions a salt, 192 k MACs a region. The binding
//     (pearl_core.cc) counts a salt as (m / 16) (n / 16) attempts of 256 k MACs and is not changed for this:
//     a tile is 256 regions of 192 k MACs, which is 192 of the binding's attempts exactly, so a batch
//     reports 192 a tile, and what the salt's tiles leave of the binding's count is one empty batch.
//   - It draws the noise on the chip. The host keeps no noised operands (they would be 24 GB at this k): it
//     gives the fold the dense factors as tables (rot_aux_tab) and, for every salt, a table of offsets
//     (rot_aux_offs) from the index lists and an order of k inside each chunk. Any order gives the same sums;
//     a searched one keeps the fold's gathers out of each other's memory banks (probes/README.md).
//   - The operands are filled with 63, not 48: the fold's subtract needs fill >= 63 so no byte borrows.
//   - Row group 0 is not mined: the salt stamp is in row 0 and the fold takes every row as the fill.
// What is the release's, reused as it is: the operands' storage and trees (operand_commitment), the seed chain,
// pearl_gen_dense and pearl_gen_perm, the host-hashed restamp (host_record, pearl_restamp_commit), the pipeline
// (pearl_queue_batch, pearl_host_collect), the hit list and the proof capture (snapshotProof).
//
// The order search runs on one helper thread a card, which sleeps unless it has a salt to search. While a salt
// mines, the helper searches the next salt's order; the tables and offsets of that salt are then built on the
// side stream, so a salt change costs the fold only the restamp's commit, as in the release. A job's first
// salt has nothing prepared: it starts in the definition's order at once, and the searched order replaces it
// when the helper has it.

#define PEARL_ROT_K 65536u
#define PEARL_ROT_FILL 63
#define PEARL_ROT_MT 6u
#define PEARL_ROT_SMEM (2048u * 12u + 2048u * 16u + 256u * 64u + 256u + 24576u)
#define PEARL_ROT_ROWS 12
#define PEARL_ROT_COLS 16
#define PEARL_ROT_UNITS 192u   // the binding's attempts (256 k MACs each) in one tile of 256 regions of 192 k

const uint32_t kRotRows[PEARL_ROT_ROWS] = {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88};
const uint32_t kRotCols[PEARL_ROT_COLS] = {0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57};
// PeriodicPattern::to_bytes of the rows: one dimension of stride 8, length 12.
const uint8_t kRotRowsBytes[6] = {7, 11, 0, 0, 0, 0};

typedef CUresult(CUDAAPI *RotLoadData)(CUmodule *, const void *);
typedef CUresult(CUDAAPI *RotUnload)(CUmodule);
typedef CUresult(CUDAAPI *RotGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult(CUDAAPI *RotSetAttr)(CUfunction, CUfunction_attribute, int);
typedef CUresult(CUDAAPI *RotLaunch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, CUstream, void **, void **);

struct RotState {
  CUmodule mod = nullptr;
  CUfunction fFold = nullptr, fTab = nullptr, fOffs = nullptr;
  RotLaunch launch = nullptr;
  RotUnload unload = nullptr;
  uint32_t k = 0, chunks = 0, ksteps = 0;
  uint32_t rg0 = 1, rgCount = 0, cbCount = 0;      // row groups mined (from rg0) and column blocks
  uint32_t rowsValid = 0, colsValid = 0;           // valid row and column offsets (the share's nonce space)
  uint64_t tiles = 0;                              // tiles a salt
  uint32_t tilesPerBatch = 0, sms = 1;
  // Three sets of A's tables and of the offsets, used in turn: a set is written while batches of the salt
  // before last can no longer be queued.
  uint4 *dTabA[3] = {nullptr, nullptr, nullptr};
  uint4 *dTabB = nullptr;
  uint32_t *dOffs[3] = {nullptr, nullptr, nullptr};
  uint8_t *dOrder[3] = {nullptr, nullptr, nullptr};
  std::vector<void *> spare;                       // allocations that crossed a 4 GiB line, freed at the end
  int8_t *dDenseNext = nullptr;
  uint32_t *dSeedNext = nullptr, *dPermNext = nullptr;
  const uint32_t *dPermCur = nullptr;              // the running salt's A index list on the device
  int tabRing = 0, offsRing = 0, tabCur = 0, offsCur = 0;
  size_t offsWords = 0;
  cudaEvent_t prepEv = nullptr;
  int waitPrep = 0;                                // batches still to be ordered behind prepEv
  bool curNeedsOrder = false;                      // the running salt is in the definition's order
  std::vector<uint32_t> permA, permB;              // host copies, [k][2]
  std::vector<uint8_t> orderHost[3];
  uint32_t seedNextHost[8] = {0};
  bool hostPermOk = false;
  uint64_t stride = 1, lastSalt = 0;
  bool haveLast = false;
  // The next salt, ready: its record, and which sets hold its tables and offsets.
  struct Prep {
    bool valid = false;
    uint64_t salt = 0;
    PearlRestampRecord rec;
    int tab = 0, offs = 0;
    std::vector<uint32_t> permA;
  } prep;
  bool nextAsked = false;
  uint64_t nextSalt = 0;
  PearlRestampRecord nextRec;
  // The helper thread.
  std::thread th;
  std::mutex mu;
  std::condition_variable cv;
  bool quit = false;
  std::atomic<uint64_t> gen{0};
  struct Req {
    uint64_t gen = 0, salt = 0;
    bool hash = false;
    uint32_t seed[8] = {0};
    std::vector<uint32_t> permA, permB;
  } req;
  bool reqValid = false;
  struct Done {
    uint64_t gen = 0, salt = 0;
    std::vector<uint32_t> permA;
    std::vector<uint8_t> order;
    double ms = 0;
  } done;
  bool doneValid = false;
  // Counts for the one line the context prints when it ends.
  uint64_t nAhead = 0, nSync = 0, nSearch = 0, nJobs = 0;
  double searchMs = 0, syncMs = 0, jobMs = 0;
};

bool rot_env() {
  const char *v = getenv("PEARL_FOLD_ROT");
  return v && v[0] == '1' && v[1] == 0;
}

// The cubin a card of compute `major`.`minor` loads from beside this module: one built for its architecture.
const char *rot_file_for(int major, int minor) {
  return major == 12 ? PEARL_ROT_FILE_SM120 : minor == 6 ? PEARL_ROT_FILE_SM86 : PEARL_ROT_FILE;
}

// Where the fold's cubin is: the file PEARL_ROT_CUBIN names, or `name` beside this module (pearl_rot_path.h
// has the lookup and its reasons). False when no file is there; `buf` then holds the first place looked in.
bool rot_cubin_path(char *buf, size_t len, const char *name) {
  return rot_cubin_path_for(reinterpret_cast<const void *>(&rot_env), buf, len, name);
}

bool rot_read_file(const char *path, std::vector<char> *out) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  char tmp[65536];
  size_t n;
  out->clear();
  while ((n = fread(tmp, 1, sizeof tmp, f)) > 0) out->insert(out->end(), tmp, tmp + n);
  fclose(f);
  out->push_back(0);
  // A whole ELF file: anything else is not a cubin, and the driver is not asked to read it (a cubin cut
  // short crashed the driver, and the miner with it; pearl_rot_path.h, rot_elf_ok).
  return rot_elf_ok(reinterpret_cast<const uint8_t *>(out->data()), out->size() - 1, nullptr, nullptr);
}

// The cubins this core loads from beside itself: the hashes src/pearl_rot_sm89.ref, src/pearl_rot_sm86.ref and
// src/pearl_rot_sm120.ref list, the files that have
// mined on a card with every hit verified (test/rotFoldShip.test.js keeps the lists the same). A file
// beside the core with any other hash is an old or damaged copy and is not loaded. A file PEARL_ROT_CUBIN
// names is a developer's own and is not held to the list: that is how a new build gets its first run on a card.
const char *const kRotCubins[] = {
    "d18246d5a53a3ac9cc2862c07d23b80c63900cb12ebd4a3ca17b9583d15cfc1a",  // CUDA 12.6
    "c7219d7dbaac0035d555a1c10e8ab8eb1752a65536f13aae589d374509f29f9d",  // CUDA 12.8
    "cb50068c5f8d01e7f98b241049930ccf8be2aa3dedfe89123ad5bab7ed8e2dec",  // sm_86, CUDA 12.8
    "57b0fdd0f25fd2b649d4655a93377e3e98d3f9f6fb4a896167604b4d99962aa4",  // sm_120, CUDA 13.3
};

bool rot_named() {
  const char *e = getenv("PEARL_ROT_CUBIN");
  return e && *e;
}

// Whether the image rot_read_file read is one this core may load.
bool rot_listed(const std::vector<char> &image) {
  if (rot_named()) return true;
  std::vector<uint8_t> copy(image.begin(), image.end() - 1);
  char hex[65] = {0};
  if (!rot_fold_hash(copy.data(), copy.size(), hex)) return false;
  for (const char *h : kRotCubins)
    if (strcmp(h, hex) == 0) return true;
  return false;
}

size_t rot_needed_bytes(const PearlProfile *p);

// Whether this card and profile can run the rotation fold at all (the switch aside). When they cannot, `why`
// gets the reason in plain words, for the one line the miner logs.
bool rot_card_ok(const PearlProfile *p, int device, char *why, size_t why_len) {
  auto no = [&](const char *fmt, long long a, long long b) {
    if (why && why_len) snprintf(why, why_len, fmt, a, b);
    return false;
  };
  auto nof = [&](const char *fmt, const char *file) {
    if (why && why_len) snprintf(why, why_len, fmt, file);
    return false;
  };
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
      || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    (void)cudaGetLastError();
    return no("the card's compute capability could not be read", 0, 0);
  }
  if (!(major == 8 && (minor == 9 || minor == 6)) && !(major == 12 && minor == 0))
    return no("it is for RTX 30, 40 and 50 cards, and this card is compute %lld.%lld", major, minor);
  const char *file = rot_file_for(major, minor);
  if (p->rank != 128u || p->operand_fill != PEARL_OPERAND_CONST || p->m < 384u || p->n < 256u
      || p->n % 256u != 0u)
    return no("it does not mine this profile", 0, 0);
  char path[1024];
  if (!rot_cubin_path(path, sizeof path, file))
    return nof(rot_named() ? "the file PEARL_ROT_CUBIN names cannot be read" : "its file, %s, is not beside the Pearl core",
               file);
  std::vector<char> image;
  if (!rot_read_file(path, &image)) return nof("its file, %s, is not a kernel file", file);
  if (!rot_listed(image)) return nof("its file, %s, is not the one this core was built with", file);
  // Its tables and its two trees at k 65536 take more of the card than the release fold does.
  PearlProfile q = *p;
  q.k = PEARL_ROT_K;
  const size_t need = rot_needed_bytes(&q);
  size_t freeMem = 0, totalMem = 0;
  if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
    (void)cudaGetLastError();
  } else if (freeMem < need) {
    return no("it needs %lld MiB of card memory and %lld MiB are free", (long long)(need >> 20),
              (long long)(freeMem >> 20));
  }
  return true;
}

// Whether pearl_host_create makes this context a rotation-fold one.
bool rot_wanted(const PearlProfile *p, int device) {
  return rot_env() && p->k == PEARL_ROT_K && rot_card_ok(p, device, nullptr, 0);
}

size_t rot_needed_bytes(const PearlProfile *p) {
  const size_t aLeaves = (size_t)p->m * p->k / 1024, bLeaves = (size_t)p->n * p->k / 1024;
  const size_t tree = 2 * (aLeaves + bLeaves) * 32 + (aLeaves > bLeaves ? aLeaves : bLeaves) * 32;
  const size_t tabs = (size_t)p->m * 128 * 5 + (size_t)p->n * 128 * 2;
  const size_t offs = (size_t)3 * 8 * (p->k / 32 + 1) * 256;
  return tree + tabs + offs + ((size_t)16 << 20);
}

template <class F>
bool rot_sym(const char *name, F *out) {
  void *p = nullptr;
  cudaDriverEntryPointQueryResult q;
  if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess
      || q != cudaDriverEntryPointSuccess || !p) {
    (void)cudaGetLastError();
    return false;
  }
  *out = reinterpret_cast<F>(p);
  return true;
}

// E_AR or E_BL on the host: pearl_gen_perm's [k][2], from the seed's words. One keyed BLAKE3 of 64 bytes for
// eight k (noise.js, generatePermutationMatrix).
void rot_host_perm(const uint32_t seed[8], const char *label, uint32_t k, std::vector<uint32_t> *out) {
  out->resize((size_t)k * 2);
  uint8_t msg[64];
  for (uint32_t i = 0; i < (k + 7u) / 8u; i++) {
    memset(msg, 0, sizeof msg);
    const uint32_t v = 1u + i;
    msg[4] = (uint8_t)v; msg[5] = (uint8_t)(v >> 8); msg[6] = (uint8_t)(v >> 16); msg[7] = (uint8_t)(v >> 24);
    memcpy(msg + 32, label, strlen(label));
    uint32_t h[8];
    hb3_hash64(seed, msg, h);
    for (uint32_t j = 0; j < 8u; j++) {
      const uint32_t row = i * 8u + j;
      if (row >= k) break;
      const uint32_t u = h[j], p0 = u & 127u;
      const uint32_t p1 = p0 ^ (1u + (uint32_t)(((uint64_t)127u * u) >> 32));
      (*out)[2 * (size_t)row] = p0;
      (*out)[2 * (size_t)row + 1] = p1;
    }
  }
}

// The order of one chunk's 128 k (probes/hand89/gprobe.cu's search 6, the one the fold was measured with):
// member[o][x] is the k in place x of octet o, an octet being the eight slots one ldmatrix matrix reads. A row
// of a matrix that shares its banks with another (the same rank index mod 8) costs; a bank shared by n rows
// costs (n - 1) squared, B's rows twice A's. Greedy swaps, the best partner for each place, until a sweep
// moves nothing (at most 8 sweeps). The same choices as the probe's, with a swap's gain worked out from the
// four counters it changes instead of every counter of both octets.
void rot_order_chunk(const uint32_t *pa, const uint32_t *pb, uint8_t *o /* [128] */) {
  static const int wt[4] = {1, 1, 2, 2};
  uint8_t res[128][4], member[16][8], cnt[16][4][8];
  // cost[st][c] = wt (c - 1)^2 for c > 1; dec[st][c] = cost[c] - cost[c - 1]; inc[st][c] = cost[c] - cost[c + 1]
  int dec[4][10], inc[4][10];
  for (int st = 0; st < 4; st++)
    for (int c = 0; c < 10; c++) {
      auto f = [&](int x) { return x > 1 ? wt[st] * (x - 1) * (x - 1) : 0; };
      dec[st][c] = c > 0 ? f(c) - f(c - 1) : 0;
      inc[st][c] = f(c) - f(c + 1);
    }
  for (uint32_t i = 0; i < 128u; i++) {
    res[i][0] = (uint8_t)(pa[2 * i] & 7u);
    res[i][1] = (uint8_t)(pa[2 * i + 1] & 7u);
    res[i][2] = (uint8_t)(pb[2 * i] & 7u);
    res[i][3] = (uint8_t)(pb[2 * i + 1] & 7u);
  }
  memset(cnt, 0, sizeof cnt);
  for (uint32_t oc = 0; oc < 16u; oc++)
    for (uint32_t x = 0; x < 8u; x++) {
      member[oc][x] = (uint8_t)(8u * oc + x);
      for (int st = 0; st < 4; st++) cnt[oc][st][res[8u * oc + x][st]]++;
    }
  for (int sw = 0; sw < 8; sw++) {
    int moved = 0;
    for (uint32_t o1 = 0; o1 < 16u; o1++)
      for (uint32_t x = 0; x < 8u; x++) {
        int best = 0, bo = -1, by = -1;
        const uint32_t i1 = member[o1][x];
        for (uint32_t o2 = 0; o2 < 16u; o2++) {
          if (o2 == o1) continue;
          for (uint32_t y = 0; y < 8u; y++) {
            const uint32_t i2 = member[o2][y];
            int gain = 0;
            for (int st = 0; st < 4; st++) {
              const uint32_t r1 = res[i1][st], r2 = res[i2][st];
              if (r1 == r2) continue;
              gain += dec[st][cnt[o1][st][r1]] + inc[st][cnt[o1][st][r2]] + dec[st][cnt[o2][st][r2]]
                      + inc[st][cnt[o2][st][r1]];
            }
            if (gain > best) {
              best = gain;
              bo = (int)o2;
              by = (int)y;
            }
          }
        }
        if (bo >= 0) {
          const uint32_t i2 = member[bo][by];
          for (int st = 0; st < 4; st++) {
            cnt[o1][st][res[i1][st]]--;
            cnt[o1][st][res[i2][st]]++;
            cnt[bo][st][res[i2][st]]--;
            cnt[bo][st][res[i1][st]]++;
          }
          member[o1][x] = (uint8_t)i2;
          member[bo][by] = (uint8_t)i1;
          moved++;
        }
      }
    if (!moved) break;
  }
  for (uint32_t oc = 0; oc < 16u; oc++)
    for (uint32_t x = 0; x < 8u; x++)
      o[16u * (oc >> 1) + 4u * (x >> 1) + 2u * (oc & 1u) + (x & 1u)] = member[oc][x];
}

// The helper: sleeps until asked, then searches one salt's order (and hashes its A index list first, when
// the salt is one the device has not drawn). Pure CPU: it touches nothing of CUDA. A newer request, or a new
// job (gen), makes it drop what it is doing.
void rot_helper(RotState *r) {
  for (;;) {
    RotState::Req q;
    {
      std::unique_lock<std::mutex> lk(r->mu);
      r->cv.wait(lk, [&] { return r->quit || r->reqValid; });
      if (r->quit) return;
      q = std::move(r->req);
      r->reqValid = false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (q.hash) rot_host_perm(q.seed, "A_tensor", r->k, &q.permA);
    RotState::Done d;
    d.gen = q.gen;
    d.salt = q.salt;
    d.order.resize(r->k);
    bool dropped = false;
    for (uint32_t ch = 0; ch < r->chunks && !dropped; ch++) {
      rot_order_chunk(&q.permA[(size_t)ch * 256], &q.permB[(size_t)ch * 256], &d.order[(size_t)ch * 128]);
      if ((ch & 15u) == 15u && r->gen.load() != q.gen) dropped = true;
    }
    if (dropped) continue;
    d.permA = std::move(q.permA);
    d.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(r->mu);
    if (d.gen == r->gen.load()) {
      r->done = std::move(d);
      r->doneValid = true;
    }
  }
}

void rot_ask(RotState *r, uint64_t salt, const uint32_t *seed, const std::vector<uint32_t> *permA) {
  std::lock_guard<std::mutex> lk(r->mu);
  r->req.gen = r->gen.load();
  r->req.salt = salt;
  r->req.hash = seed != nullptr;
  if (seed) memcpy(r->req.seed, seed, sizeof r->req.seed);
  if (permA) r->req.permA = *permA;
  r->req.permB = r->permB;
  r->reqValid = true;
  r->cv.notify_one();
}

// rot_aux_tab on `st`: a dense factor of `rows` rows as group tables.
void rot_launch_tab(RotState *r, cudaStream_t st, const int8_t *dense, uint4 *out, uint32_t rows) {
  uint32_t groups = rows / 16u;
  void *args[3] = {&dense, &out, &groups};
  r->launch(r->fTab, (groups * 128u + 255u) / 256u, 1, 1, 256, 1, 1, 0, (CUstream)st, args, nullptr);
}

// The offsets of one salt into the next set, on `st`, from an order on the host. Returns the set.
int rot_launch_offs(Ctx *ctx, cudaStream_t st, const uint32_t *dPermA, const uint8_t *order) {
  RotState *r = ctx->rot;
  const int o = r->offsRing = (r->offsRing + 1) % 3;
  r->orderHost[o].assign(order, order + r->k);
  cudaMemcpyAsync(r->dOrder[o], r->orderHost[o].data(), r->k, cudaMemcpyHostToDevice, st);
  const uint32_t *dPermB = ctx->dPermB;
  const uint8_t *dOrder = r->dOrder[o];
  uint32_t *dOffs = r->dOffs[o];
  uint32_t ksteps = r->ksteps;
  void *args[5] = {&dPermA, &dPermB, &dOrder, &dOffs, &ksteps};
  r->launch(r->fOffs, ((ksteps + 1u) * 32u + 255u) / 256u, 1, 1, 256, 1, 1, 0, (CUstream)st, args, nullptr);
  return o;
}

// Ask the helper for the salt after the running one, when the host can hash its seed itself.
void rot_ask_next(Ctx *ctx) {
  RotState *r = ctx->rot;
  r->nextAsked = false;
  if (!ctx->hostSeeds || !r->hostPermOk) return;
  r->nextSalt = ctx->salt + r->stride;
  host_record(ctx, r->nextSalt, true, &r->nextRec);
  r->nextAsked = true;
  rot_ask(r, r->nextSalt, r->nextRec.a_seed, nullptr);
}

// Take what the helper has finished, if anything: the running salt's searched order (its offsets are built
// again on the side stream and the next batches use them), or the next salt's (its tables and offsets are
// built on the side stream, ready for the salt change).
void rot_apply_done(Ctx *ctx) {
  RotState *r = ctx->rot;
  RotState::Done d;
  {
    std::lock_guard<std::mutex> lk(r->mu);
    if (!r->doneValid) return;
    d = std::move(r->done);
    r->doneValid = false;
  }
  if (d.gen != r->gen.load()) return;
  r->searchMs += d.ms;
  r->nSearch++;
  if (r->curNeedsOrder && d.salt == ctx->salt) {
    const int o = rot_launch_offs(ctx, ctx->side, r->dPermCur, d.order.data());
    cudaEventRecord(r->prepEv, ctx->side);
    cudaStreamQuery(ctx->side);
    r->offsCur = o;
    r->waitPrep = Ctx::kSlots;
    r->curNeedsOrder = false;
    rot_ask_next(ctx);
    return;
  }
  if (r->nextAsked && d.salt == r->nextSalt && !r->prep.valid) {
    const uint32_t m = ctx->profile.m, rank = ctx->profile.rank;
    memcpy(r->seedNextHost, r->nextRec.a_seed, sizeof r->seedNextHost);
    cudaMemcpyAsync(r->dSeedNext, r->seedNextHost, 32, cudaMemcpyHostToDevice, ctx->side);
    pearl_gen_dense<<<draw_blocks((size_t)m * (rank / 32)), kDrawThreads, 0, ctx->side>>>(
        r->dSeedNext, ctx->dLabelA, nullptr, r->dDenseNext, m, rank);
    const int t = r->tabRing = (r->tabRing + 1) % 3;
    rot_launch_tab(r, ctx->side, r->dDenseNext, r->dTabA[t], m);
    r->prep.permA = std::move(d.permA);
    cudaMemcpyAsync(r->dPermNext, r->prep.permA.data(), (size_t)r->k * 8, cudaMemcpyHostToDevice, ctx->side);
    const int o = rot_launch_offs(ctx, ctx->side, r->dPermNext, d.order.data());
    cudaEventRecord(r->prepEv, ctx->side);
    cudaStreamQuery(ctx->side);
    r->prep.valid = true;
    r->prep.salt = d.salt;
    r->prep.rec = r->nextRec;
    r->prep.tab = t;
    r->prep.offs = o;
    r->nextAsked = false;
  }
}

// A job's full draw: the release's (full_draw) with this fold's config, fill and no noised operands. Both
// operands, both trees, both seeds, B's dense factor and index list and B's tables.
void rot_full_draw(Ctx *ctx, uint64_t salt) {
  RotState *r = ctx->rot;
  uint8_t seedInput[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(seedInput, ctx->header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, seedInput + PEARL_HEADER_BYTES);
  memcpy(seedInput + PEARL_HEADER_BYTES + 8, kRotRowsBytes, 6);
  cudaMemcpy(ctx->dSeedInput, seedInput, sizeof(seedInput), cudaMemcpyHostToDevice);
  pearl_blake3_unkeyed<<<1, 1>>>(ctx->dSeedInput, sizeof(seedInput),
                                 reinterpret_cast<uint8_t *>(ctx->dJobKey));
  const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
  const size_t bLen = (size_t)ctx->profile.n * ctx->profile.k;
  cudaMemset(ctx->dA, PEARL_ROT_FILL, 2048);
  cudaMemset(ctx->dB, PEARL_ROT_FILL, 2048);
  int8_t stamp[PEARL_STAMP_BYTES];
  for (int i = 0; i < PEARL_STAMP_BYTES; i++) stamp[i] = pearl_stamp_byte(salt, i);
  cudaMemcpy(ctx->dA, stamp, sizeof(stamp), cudaMemcpyHostToDevice);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen, ctx->dHashA, ctx->dTreeA,
                     &ctx->layerOffA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen, ctx->dHashB, ctx->dTreeB,
                     &ctx->layerOffB);
  derive_seeds(ctx, true);
  const uint32_t n = ctx->profile.n, rank = ctx->profile.rank, k = ctx->profile.k;
  pearl_gen_dense<<<draw_blocks((size_t)n * (rank / 32)), kDrawThreads>>>(ctx->dBSeed, ctx->dLabelB, nullptr,
                                                                         ctx->dEBR, n, rank);
  pearl_gen_perm<<<draw_blocks((k + 7) / 8), kDrawThreads>>>(ctx->dBSeed, ctx->dLabelB, ctx->dPermB, k, rank);
  rot_launch_tab(r, nullptr, ctx->dEBR, r->dTabB, n);
  cudaMemcpy(ctx->dTarget, ctx->target, PEARL_HASH_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->bSeed, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
  r->permB.resize((size_t)k * 2);
  cudaMemcpy(r->permB.data(), ctx->dPermB, (size_t)k * 8, cudaMemcpyDeviceToHost);
}

// A's side for the salt whose a_seed the device holds, with nothing prepared: dense factor, tables, index
// list, and offsets in the definition's order, all on the default stream, so the batches that follow wait for
// them. Then the helper is asked for this salt's order.
void rot_draw_a_now(Ctx *ctx) {
  RotState *r = ctx->rot;
  const uint32_t m = ctx->profile.m, rank = ctx->profile.rank, k = ctx->profile.k;
  pearl_gen_dense<<<draw_blocks((size_t)m * (rank / 32)), kDrawThreads>>>(ctx->dASeed, ctx->dLabelA, nullptr,
                                                                         ctx->dEAL, m, rank);
  pearl_gen_perm<<<draw_blocks((k + 7) / 8), kDrawThreads>>>(ctx->dASeed, ctx->dLabelA, ctx->dPermA, k, rank);
  const int t = r->tabRing = (r->tabRing + 1) % 3;
  rot_launch_tab(r, nullptr, ctx->dEAL, r->dTabA[t], m);
  r->permA.resize((size_t)k * 2);
  cudaMemcpy(r->permA.data(), ctx->dPermA, (size_t)k * 8, cudaMemcpyDeviceToHost);
  std::vector<uint8_t> ident(k);
  for (uint32_t i = 0; i < k; i++) ident[i] = (uint8_t)(i & 127u);
  const int o = rot_launch_offs(ctx, nullptr, ctx->dPermA, ident.data());
  r->dPermCur = ctx->dPermA;
  r->tabCur = t;
  r->offsCur = o;
  r->waitPrep = 0;
  r->curNeedsOrder = true;
  // The host's own index list must be the device's before a salt is ever prepared from it.
  {
    uint32_t seed[8];
    hb3_words(ctx->aSeed, seed, 8);
    std::vector<uint32_t> mine;
    rot_host_perm(seed, "A_tensor", k, &mine);
    r->hostPermOk = mine == r->permA;
    if (!r->hostPermOk)
      fprintf(stderr, "pearl: the host's index list disagrees with the device's; the rotation fold "
                      "will prepare no salt ahead\n");
  }
  rot_ask(r, ctx->salt, nullptr, &r->permA);
}

// pearl_host_reseed for a rotation-fold context.
void rot_reseed(Ctx *ctx, uint64_t salt) {
  RotState *r = ctx->rot;
  rot_apply_done(ctx);
  if (ctx->baseDrawn && r->haveLast && salt > r->lastSalt) r->stride = salt - r->lastSalt;
  const bool ahead = ctx->baseDrawn && ctx->hostSeeds && r->prep.valid && r->prep.salt == salt;
  ctx->salt = salt;
  const auto t0 = std::chrono::steady_clock::now();
  if (ahead) {
    // The salt was prepared while the last one mined: only the commit queues behind the batches.
    pearl_restamp_commit<<<1, 256>>>(r->prep.rec, ctx->dA, ctx->dTreeA, ctx->dHashA, ctx->dBoundA,
                                     ctx->dASeed);
    hb3_bytes(r->prep.rec.a_seed, ctx->aSeed);
    keep_record(ctx, salt, r->prep.rec);
    r->tabCur = r->prep.tab;
    r->offsCur = r->prep.offs;
    r->permA = std::move(r->prep.permA);
    r->dPermCur = r->dPermNext;
    r->prep.valid = false;
    r->curNeedsOrder = false;
    r->waitPrep = Ctx::kSlots;
    ctx->afterRedraw = 2;
    r->nAhead++;
    rot_ask_next(ctx);
  } else {
    r->gen++;   // whatever the helper is doing is for a salt that will not come
    r->prep.valid = false;
    r->nextAsked = false;
    const bool job = !ctx->baseDrawn;
    if (job) {
      rot_full_draw(ctx, salt);
      cudaMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      setup_host_seeds(ctx);
    } else if (ctx->hostSeeds) {
      PearlRestampRecord rec;
      host_record(ctx, salt, true, &rec);
      pearl_restamp_commit<<<1, 256>>>(rec, ctx->dA, ctx->dTreeA, ctx->dHashA, ctx->dBoundA, ctx->dASeed);
      hb3_bytes(rec.a_seed, ctx->aSeed);
      keep_record(ctx, salt, rec);
    } else {
      const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
      pearl_restamp_operand<<<1, 1>>>(ctx->dJobKey, ctx->dA, salt, aChunks, ctx->dTreeA, ctx->dHashA);
      derive_seeds(ctx, false);
      cudaMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      PearlRestampRecord rec;
      device_record(ctx, &rec);
      keep_record(ctx, salt, rec);
    }
    rot_draw_a_now(ctx);
    ctx->afterRedraw = 2;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (job) {
      r->nJobs++;
      r->jobMs += ms;
    } else {
      r->nSync++;
      r->syncMs += ms;
    }
  }
  r->lastSalt = salt;
  r->haveLast = true;
  ctx->baseDrawn = true;
  ctx->haveJob = true;
}

// One batch: the tiles from nonce_base / 192 on (a tile is 192 of the binding's attempts, see the head of this
// file), at most tilesPerBatch and no further than the salt's last. Past the last tile: an empty batch that
// takes the binding's count to the salt's end.
bool rot_submit(Ctx *ctx, int slot, cudaStream_t st, uint64_t nonce_base, uint64_t *regions_out, char *err,
                size_t err_len) {
  RotState *r = ctx->rot;
  rot_apply_done(ctx);
  const uint64_t first = nonce_base / PEARL_ROT_UNITS;
  if (first >= r->tiles) {
    const uint64_t span = ((uint64_t)ctx->profile.m / PEARL_ROWS_COUNT) * ((uint64_t)ctx->profile.n / PEARL_COLS_COUNT);
    cudaMemsetAsync(pearl_slot_hits(ctx, slot).count, 0, sizeof(uint32_t), st);
    if (!pearl_queue_batch(ctx, slot, st, nonce_base, 0u, nullptr, err, err_len)) return false;
    if (regions_out) *regions_out = span > nonce_base ? span - nonce_base : 1u;
    return true;
  }
  const uint64_t left = r->tiles - first;
  uint32_t tiles = left < r->tilesPerBatch ? (uint32_t)left : r->tilesPerBatch;
  uint32_t tile0 = (uint32_t)first;
  if (r->waitPrep > 0) {
    cudaStreamWaitEvent(st, r->prepEv, 0);
    r->waitPrep--;
  }
  PearlTranscriptTest test = pearl_slot_test(ctx);
  PearlHitList hitList = pearl_slot_hits(ctx, slot);
  cudaMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);
  const uint4 *tabA = r->dTabA[r->tabCur], *tabB = r->dTabB;
  const uint32_t *offs = r->dOffs[r->offsCur];
  uint32_t chunks = r->chunks, rgCount = r->rgCount, rg0 = r->rg0;
  void *args[10] = {&tabA, &tabB, &offs, &chunks, &rgCount, &rg0, &tile0, &tiles, &test, &hitList};
  const unsigned blocks = tiles < r->sms ? tiles : r->sms;
  const CUresult rc = r->launch(r->fFold, blocks, 1, 1, 256, 1, 1, PEARL_ROT_SMEM, (CUstream)st, args, nullptr);
  if (rc != CUDA_SUCCESS) {
    if (err && err_len) snprintf(err, err_len, "rotation fold launch failed: driver error %d", (int)rc);
    return false;
  }
  return pearl_queue_batch(ctx, slot, st, nonce_base, tiles * PEARL_ROT_UNITS, regions_out, err, err_len);
}

// A hit's region, from its batch's base and the kernel's index (256 x tile of the batch + thread): as the
// share's nonce (row offset index + valid row offsets x column offset index, over valid offsets in rising
// order) and its rows and columns.
void rot_region(const Ctx *ctx, uint64_t nonce_base, uint32_t index, uint64_t *nonce, uint32_t *rows,
                uint32_t *cols) {
  const RotState *r = ctx->rot;
  const uint64_t tile = nonce_base / PEARL_ROT_UNITS + index / 256u;
  const uint32_t tid = index % 256u, warp = tid >> 5, lane = tid & 31u;
  const uint32_t rbg = r->rg0 + (uint32_t)(tile % r->rgCount), cbg = (uint32_t)(tile / r->rgCount);
  const uint32_t Q = 2u * rbg + (warp >> 2), g = lane >> 2;
  const uint32_t J = 4u * cbg + (warp & 3u), q = lane & 3u;
  const uint32_t rowIdx = 8u * Q + g, colIdx = 4u * J + q;
  *nonce = (uint64_t)rowIdx + (uint64_t)r->rowsValid * colIdx;
  for (int j = 0; j < PEARL_ROT_ROWS; j++) rows[j] = 96u * Q + g + kRotRows[j];
  for (int j = 0; j < PEARL_ROT_COLS; j++) cols[j] = 64u * J + 2u * q + kRotCols[j];
}

void rot_destroy(Ctx *ctx) {
  RotState *r = ctx->rot;
  if (!r) return;
  if (r->th.joinable()) {
    {
      std::lock_guard<std::mutex> lk(r->mu);
      r->quit = true;
      r->cv.notify_one();
    }
    r->th.join();
  }
  fprintf(stderr,
          "[pearl] rotation fold: %llu jobs drawn (%.0f ms each), %llu salts prepared ahead, %llu drawn "
          "on the spot (%.0f ms each), %llu order searches (%.0f ms each on the helper thread)\n",
          (unsigned long long)r->nJobs, r->nJobs ? r->jobMs / r->nJobs : 0.0, (unsigned long long)r->nAhead,
          (unsigned long long)r->nSync, r->nSync ? r->syncMs / r->nSync : 0.0,
          (unsigned long long)r->nSearch, r->nSearch ? r->searchMs / r->nSearch : 0.0);
  for (int i = 0; i < 3; i++) {
    cudaFree(r->dTabA[i]);
    cudaFree(r->dOffs[i]);
    cudaFree(r->dOrder[i]);
  }
  for (void *p : r->spare) cudaFree(p);
  cudaFree(r->dTabB);
  cudaFree(r->dDenseNext);
  cudaFree(r->dSeedNext);
  cudaFree(r->dPermNext);
  if (r->prepEv) cudaEventDestroy(r->prepEv);
  if (r->mod && r->unload) r->unload(r->mod);
  delete r;
  ctx->rot = nullptr;
}

// Make `ctx` a rotation-fold context: load the cubin, allocate the tables, start the helper.
bool rot_create(Ctx *ctx, char *err, size_t err_len) {
  const PearlProfile &p = ctx->profile;
  RotState *r = new RotState();
  ctx->rot = r;
  auto bad = [&](const char *msg, int code) {
    if (err && err_len) snprintf(err, err_len, "rotation fold: %s (%d)", msg, code);
    return false;
  };
  char path[1024];
  std::vector<char> image;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ctx->device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ctx->device);
  if (!rot_cubin_path(path, sizeof path, rot_file_for(major, minor)) || !rot_read_file(path, &image) || !rot_listed(image))
    return bad("cannot read the fold's cubin", 0);
  RotLoadData load = nullptr;
  RotGetFunction getfn = nullptr;
  RotSetAttr setattr = nullptr;
  if (!rot_sym("cuModuleLoadData", &load) || !rot_sym("cuModuleGetFunction", &getfn)
      || !rot_sym("cuFuncSetAttribute", &setattr) || !rot_sym("cuLaunchKernel", &r->launch)
      || !rot_sym("cuModuleUnload", &r->unload))
    return bad("the driver's module calls are not there", 0);
  // The runtime's context must be current on this thread before the driver is asked for anything.
  cudaDeviceSynchronize();
  CUresult rc = load(&r->mod, image.data());
  if (rc != CUDA_SUCCESS) return bad("the driver refused the cubin", (int)rc);
  if ((rc = getfn(&r->fFold, r->mod, "pearl_rot_fold")) != CUDA_SUCCESS
      || (rc = getfn(&r->fTab, r->mod, "rot_aux_tab")) != CUDA_SUCCESS
      || (rc = getfn(&r->fOffs, r->mod, "rot_aux_offs")) != CUDA_SUCCESS)
    return bad("a kernel is missing from the cubin", (int)rc);
  if ((rc = setattr(r->fFold, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)PEARL_ROT_SMEM))
      != CUDA_SUCCESS)
    return bad("the card refused the fold's shared memory", (int)rc);
  r->k = p.k;
  r->chunks = p.k / 128u;
  r->ksteps = p.k / 32u;
  r->rg0 = 1u;
  r->rgCount = p.m / 192u - 1u;
  r->cbCount = p.n / 256u;
  r->rowsValid = (p.m / 96u) * 8u;
  r->colsValid = (p.n / 64u) * 4u;
  r->tiles = (uint64_t)r->rgCount * r->cbCount;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);
  r->sms = sms > 0 ? (uint32_t)sms : 1u;
  uint32_t tpb = 32u;
  if (pearl_bd_sweep_env("PEARL_ROT_TILES", &tpb) && (tpb < 1u || tpb > 4096u)) tpb = 32u;
  r->tilesPerBatch = tpb * r->sms;
  ctx->batch = r->tilesPerBatch * PEARL_ROT_UNITS;
  const size_t tabA = (size_t)p.m * 128u, tabB = (size_t)p.n * 128u;
  // Eight warps, a record of two words a lane a k-step (rot_aux_offs), and one k-step more than a tile has.
  r->offsWords = (size_t)8 * (r->ksteps + 1u) * 64u;
  for (int i = 0; i < 3; i++) {
    if (cudaMalloc(&r->dTabA[i], tabA) != cudaSuccess) return bad("allocating A's tables", i);
    if (cudaMalloc(&r->dOrder[i], r->k) != cudaSuccess) return bad("allocating an order", i);
    // The hand-written loop moves its offsets pointer with a 32-bit add: the table must not cross 4 GiB.
    for (int tries = 0;; tries++) {
      uint32_t *q = nullptr;
      if (cudaMalloc(&q, r->offsWords * 4u) != cudaSuccess) return bad("allocating the offsets", i);
      if (((uintptr_t)q & 0xffffffffull) + r->offsWords * 4u < 0x100000000ull) {
        r->dOffs[i] = q;
        break;
      }
      r->spare.push_back(q);
      if (tries == 8) return bad("no offsets table clear of a 4 GiB line", i);
    }
  }
  if (cudaMalloc(&r->dTabB, tabB) != cudaSuccess || cudaMalloc(&r->dDenseNext, (size_t)p.m * p.rank) != cudaSuccess
      || cudaMalloc(&r->dSeedNext, 32) != cudaSuccess || cudaMalloc(&r->dPermNext, (size_t)r->k * 8) != cudaSuccess)
    return bad("allocating B's tables", 0);
  if (cudaEventCreateWithFlags(&r->prepEv, cudaEventDisableTiming) != cudaSuccess)
    return bad("creating the event", 0);
  r->th = std::thread(rot_helper, r);
  fprintf(stderr,
          "[pearl] GPU %d: rotation fold (PEARL_FOLD_ROT=1), k %u, tile 12x16, %u row groups by %u column "
          "blocks a salt, %u tiles a batch\n",
          ctx->device, r->k, r->rgCount, r->cbCount, r->tilesPerBatch);
  return true;
}
