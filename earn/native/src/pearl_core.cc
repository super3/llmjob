// N-API binding: exposes the CUDA core to the JS host as
//
//   createCore(profile, { deviceIndex, saltBase, saltStride, cardK, checkHits }) -> {
//     setJob({ header: Buffer(76), target: BigInt, jobId: string }),
//     stop(),
//     on('hit'|'hashrate'|'error', cb),
//     device: { index, name },       // the card it actually opened
//     k, kNote,                      // the k it mines, and why not a larger one
//     untested                       // check every hit of this card before sending it
//   }
//
// `deviceIndex` pins a card; omit it and the core ranks the cards and picks one.
// Either way `device` reports what it took, because the host has no other way to
// know — and when it assumed, it was wrong.
//
// `cardK` lets the host mine a larger k on this card than the profile asks: one
// its fold is faster at (pearl_host_card_k; on AMD, RDNA 4 mines k 4096 or 8192). A share
// says its own k, so the pool needs nothing from us. `k` is what the card mines,
// and the host builds that card's job_key, share bound and proofs from it.
//
// `checkHits` says the host recomputes every hit before it sends it. The AMD host
// then also opens a card whose fold has not run on its kind of card in a release,
// which it otherwise refuses (pearl_host_hip.cpp, resolve_fold), and `untested` is
// true for that card: a fold wrong on the card makes every share wrong. No NVIDIA
// card is untested.
//
// `saltBase`/`saltStride` are this core's slice of the search space. A rig mines
// with one core per card, and without a slice each of them would search the same
// operands and find the same shares.
//
// The search runs on its own thread so the Electron main thread is never
// blocked, and results are marshalled back with a thread-safe function. The JS
// host re-verifies every hit before submitting (see pearlMiner._onHit), so this
// layer is free to be optimistic.
//
// The CUDA builds also export two functions that have nothing to do with a core:
//
//   getClockOffset(cardIndex)      -> { ok, mhz, min, max } or { ok: false, error, code }
//   setClockOffset(cardIndex, mhz) -> { ok: true }          or { ok: false, error, code }
//
// They read and set a card's core clock offset through NVML, for the host's
// offset trial (see "The core clock offset" below, and main/gpuClockOffset.js).
// The AMD and Intel builds have neither: PEARL_NVML_OFFSETS is defined only
// for CUDA (binding.gyp).
//
// Independent implementation against the ISC-licensed pearl-research-labs
// specification. No dev fee, no fee address, nothing to disclose — see
// pearl_kernel.cu for why that matters.

#include <napi.h>

#ifdef PEARL_NVML_OFFSETS
#ifdef _WIN32
// Before <windows.h>: its min and max macros would break std::max below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pearl_config.h"

// Declared in pearl_host.cu — the host-side driver that owns device memory and
// runs the kernel pipeline for one job.
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

extern "C" {
// Choose the GPU to mine on and make it this thread's device. `requested` >= 0
// pins a specific index (an operator's PEARL_GPU_INDEX); anything negative lets
// the core rank the cards itself. Writes the chosen card's name into `name` and
// returns its index, or -1 with a message in `err`. Must be called BEFORE
// pearl_host_create, which allocates on whatever device is current.
int pearl_host_select_device(const PearlProfile *profile, int requested, char *name,
                            size_t name_len, char *err, size_t err_len);
// The k to mine on `device` (the one select_device chose) when the host lets the
// core choose: the profile's, or a larger one the card's fold is faster at. When
// the card could mine a larger k and won't, `note` says why, for the log.
uint32_t pearl_host_card_k(const PearlProfile *profile, int device, char *note, size_t note_len);
// Point the calling thread at the card a context lives on — the current device
// is per-thread, and the search runs on its own.
void pearl_host_bind_thread(void *ctx);
// Allocate device state for a profile. Returns null on failure (no CUDA device,
// insufficient memory) with a message in `err`.
void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len);
// The same, for an opener that checks every hit before sending it (check_hits). Sets
// `untested` when that let a card open whose fold has not run on its kind of card.
void *pearl_host_create_checked(const PearlProfile *profile, bool check_hits, bool *untested,
                                char *err, size_t err_len);
void pearl_host_destroy(void *ctx);
// Load a job. header is 76 bytes, target is 32 big-endian bytes.
void pearl_host_set_job(void *ctx, const uint8_t *header, const uint8_t *target);
// Load a job and draw its operands under a given salt — one card's slice of the
// search space. See saltBase/saltStride below.
void pearl_host_set_job_salted(void *ctx, const uint8_t *header, const uint8_t *target,
                               uint64_t salt);
// Re-draw the operands under a new salt. One salt is worth m*n regions.
void pearl_host_reseed(void *ctx, uint64_t salt);
// Share-proof accessors: the operand chunks a tile touched, and the sibling
// digests that authenticate them. Both already exist on the device.
// Search one batch of nonces. Returns true and fills `out` when a share is
// found; returns false when the batch is exhausted with no hit. `attempts`
// receives the number of nonces tried, for the hashrate figure.
bool pearl_host_search(void *ctx, uint64_t nonce_base, uint32_t batch,
                       PearlSearchResult *out, uint64_t *attempts, char *err,
                       size_t err_len);
// The same search as a pipeline: submit queues a batch without waiting (at most two at
// once) and reports how many regions it covers; collect waits for the oldest queued
// batch and reports it as pearl_host_search would; pending counts the queued ones.
bool pearl_host_submit(void *ctx, uint64_t nonce_base, uint32_t batch, uint64_t *regions,
                       char *err, size_t err_len);
bool pearl_host_collect(void *ctx, PearlSearchResult *out, uint64_t *attempts, char *err,
                        size_t err_len);
int pearl_host_pending(void *ctx);
// The collected batch's other hits, one a call, after the one collect returned.
bool pearl_host_next_hit(void *ctx, PearlSearchResult *out);
// Which fold this card runs, in words ("unresolved" without a context).
const char *pearl_host_fold_name(void *ctx);
// RTX 30's launch shapes (pearl_host.cu): how many (0: nothing to choose), a shape's name,
// and switching to one with no batch queued.
int pearl_host_shape_count(void *ctx);
void pearl_host_shape_name(void *ctx, int i, char *buf, size_t len);
bool pearl_host_set_shape(void *ctx, int i);
}

namespace {

class PearlCore : public Napi::ObjectWrap<PearlCore> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  PearlCore(const Napi::CallbackInfo &info);
  ~PearlCore();

 private:
  Napi::Value SetJob(const Napi::CallbackInfo &info);
  Napi::Value Stop(const Napi::CallbackInfo &info);
  Napi::Value On(const Napi::CallbackInfo &info);
  Napi::Value GetDevice(const Napi::CallbackInfo &info);
  Napi::Value GetK(const Napi::CallbackInfo &info);
  Napi::Value GetKNote(const Napi::CallbackInfo &info);
  Napi::Value GetUntested(const Napi::CallbackInfo &info);
  Napi::Value GetShapes(const Napi::CallbackInfo &info);
  Napi::Value GetShape(const Napi::CallbackInfo &info);

  void SearchLoop();
  void EmitHit(const PearlSearchResult &r, const std::string &job_id);
  void EmitHashrate(double th_per_sec);
  void EmitError(const std::string &msg);
  void EmitShape(const std::string &json);
  std::string ShapeName(int i) const;

  PearlProfile profile_ = PEARL_MAINNET_PROFILE;
  // Hashrate averaging window. The emitted value is a RATE, so it needs both a
  // work accumulator and a start time.
  std::chrono::steady_clock::time_point win_start_{};
  double win_work_ = 0.0;
  void *ctx_ = nullptr;
  // The card this core actually opened. Exposed to JS as `core.device` so the
  // host names the GPU it is mining on rather than the one it guessed at — the
  // two were different on a multi-card rig, which is the whole of issue #226.
  int device_index_ = -1;
  std::string device_name_;
  // Why this card mines the profile's k when it could mine a larger one (cardK):
  // empty unless there is something to say. JS reads it as core.kNote.
  std::string k_note_;
  // Whether this card opened only because the host checks its hits (checkHits): its
  // fold has not run on its kind of card. JS reads it as core.untested.
  bool untested_ = false;
  // This core's slice of the search space, for a rig mining on several cards at
  // once: it searches salts salt_base_, salt_base_ + salt_stride_, and so on.
  // One card per core, so the host gives card i base i and stride N. Defaults
  // (0, 1) are the single-card walk 0, 1, 2, 3 that shipped before.
  uint64_t salt_base_ = 0;
  uint64_t salt_stride_ = 1;
  // The startup trial of RTX 30's launch shapes (shapeTrialSecs): about how long it runs,
  // 0 for none. And the shape this core runs now, which a cached pick (shape) sets at create.
  uint32_t shape_trial_secs_ = 0;
  std::atomic<int> shape_now_{0};
  std::thread worker_;
  std::atomic<bool> running_{false};
  // Set by the search thread when a search fails, before it clears running_ and
  // exits. A failed core never searches again: its CUDA context is usually dead
  // (a fault such as "misaligned address" is sticky), and the host drops the core
  // when it hears the error. setJob on it does nothing.
  std::atomic<bool> failed_{false};
  std::mutex job_mu_;
  std::string job_id_;
  bool have_job_ = false;
  // A job SetJob has taken and the search thread has not drawn yet. The draw runs on
  // the search thread, between batches, so the device is only ever driven from one
  // thread and no queued batch reads operands a new job is rewriting underneath it.
  bool job_dirty_ = false;
  uint8_t job_header_[PEARL_HEADER_BYTES] = {0};
  uint8_t job_target_[PEARL_HASH_BYTES] = {0};

  Napi::ThreadSafeFunction on_hit_;
  Napi::ThreadSafeFunction on_hashrate_;
  Napi::ThreadSafeFunction on_error_;
  Napi::ThreadSafeFunction on_shape_;
};

PearlCore::PearlCore(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<PearlCore>(info) {
  Napi::Env env = info.Env();
  PearlProfile profile = PEARL_MAINNET_PROFILE;
  if (info.Length() > 0 && info[0].IsObject()) {
    Napi::Object o = info[0].As<Napi::Object>();
    auto u32 = [&](const char *k, uint32_t d) -> uint32_t {
      return o.Has(k) ? o.Get(k).As<Napi::Number>().Uint32Value() : d;
    };
    // k/rank/mmaType are protocol and go into config52; m/n are the miner's own
    // workload dimensions and are deliberately NOT hashed.
    profile.k = u32("k", profile.k);
    profile.rank = (uint16_t)u32("rank", profile.rank);
    profile.mma_type = (uint16_t)u32("mmaType", profile.mma_type);
    profile.m = u32("m", profile.m);
    profile.n = u32("n", profile.n);
    // 0 = cert-v3 salted, 1 = legacy. Exposed so the derivation can be
    // flipped from JS for a diagnostic run rather than needing a rebuild —
    // it is the one thing here that only a live pool can confirm.
    // seedDerivationCode, NOT seedDerivation: the latter is a STRING in JS and
    // Uint32Value() renders any non-numeric text as 0. That is the value for
    // salted, so a caller asking for legacy would have been silently ignored.
    profile.seed_derivation = u32("seedDerivationCode", profile.seed_derivation);
    // Column offsets per launch. Exposed so the batch width can be swept from
    // JS without a rebuild -- it trades VRAM for amortised launch overhead and
    // the sweet spot is card-dependent.
    profile.col_batch = u32("colBatch", profile.col_batch);
    // Diagnostic: which way round the jackpot hash is read when compared to the
    // target. The reference says little-endian; shares are being rejected
    // regardless of margin, which is what the other choice would look like.
    profile.hash_big_endian = u32("hashBigEndian", profile.hash_big_endian);
    // 0 = hashed operands, 1 = the constant fill (PEARL_OPERAND_CONST). A number,
    // for the same reason as seedDerivationCode. Settable from JS so a pool that
    // refuses the constant fill can be answered without a rebuild, and so the
    // frozen parity vectors, captured with the hashed fill, can be reproduced.
    profile.operand_fill = u32("operandFillCode", profile.operand_fill);
  }

  // Which card to mine on. An explicit index (PEARL_GPU_INDEX, passed down by
  // main/pearlCore.js) pins it; otherwise the core ranks the cards itself. This
  // has to happen BEFORE pearl_host_create, which allocates on whatever device
  // is current — and until now nothing set that at all, so it was the runtime's
  // device 0, a card nobody had chosen and the UI did not name.
  int requested = -1;
  bool card_k = false;
  bool check_hits = false;
  if (info.Length() > 1 && info[1].IsObject()) {
    Napi::Object o = info[1].As<Napi::Object>();
    Napi::Value v = o.Get("deviceIndex");
    if (v.IsNumber()) requested = v.As<Napi::Number>().Int32Value();
    Napi::Value ck = o.Get("cardK");
    card_k = ck.IsBoolean() && ck.As<Napi::Boolean>().Value();
    Napi::Value ch = o.Get("checkHits");
    check_hits = ch.IsBoolean() && ch.As<Napi::Boolean>().Value();
    Napi::Value b = o.Get("saltBase");
    if (b.IsNumber()) salt_base_ = (uint64_t)b.As<Napi::Number>().Int64Value();
    Napi::Value st = o.Get("saltStride");
    if (st.IsNumber()) {
      const int64_t v2 = st.As<Napi::Number>().Int64Value();
      // A zero stride would search one salt for ever, which is a stall rather
      // than an error the host could see. Keep the single-card default instead.
      if (v2 > 0) salt_stride_ = (uint64_t)v2;
    }
    Napi::Value tr = o.Get("shapeTrialSecs");
    if (tr.IsNumber() && tr.As<Napi::Number>().DoubleValue() > 0)
      shape_trial_secs_ = tr.As<Napi::Number>().Uint32Value();
  }
  // A cached pick by name (shape), applied once the context exists, below.
  std::string want_shape;
  if (info.Length() > 1 && info[1].IsObject()) {
    Napi::Value sh = info[1].As<Napi::Object>().Get("shape");
    if (sh.IsString()) want_shape = sh.As<Napi::String>().Utf8Value();
  }

  char err[256] = {0};
  char device_name[256] = {0};
  device_index_ = pearl_host_select_device(&profile, requested, device_name,
                                           sizeof(device_name), err, sizeof(err));
  if (device_index_ < 0) {
    Napi::Error::New(env, err[0] ? err : "no usable CUDA device")
        .ThrowAsJavaScriptException();
    return;
  }
  device_name_ = device_name;

  // When the miner lets it (cardK), the host may mine a larger k on this card than
  // the profile asks: one its fold is faster at. A share says its own k, so the
  // pool needs nothing from us. Decided before the allocations, which grow with k,
  // and before profile_ is set: the hashrate counts PEARL_DAF(profile_) MACs an
  // attempt, so profile_ must hold the k the card really mines.
  if (card_k) {
    char note[192] = {0};
    const uint32_t k = pearl_host_card_k(&profile, device_index_, note, sizeof(note));
    if (k) profile.k = k;
    k_note_ = note;
  }
  profile_ = profile;

  ctx_ = pearl_host_create_checked(&profile, check_hits, &untested_, err, sizeof(err));
  if (!ctx_) {
    // A missing or too-old CUDA device is the common case. Throwing here surfaces
    // in the host as a clean 'error' rather than a crash, and the host then
    // reports that the engine is unavailable on this machine.
    Napi::Error::New(env, err[0] ? err : "failed to initialise the Pearl CUDA core")
        .ThrowAsJavaScriptException();
    return;
  }
  // The app's cached pick for this card. A name this build doesn't have (the shapes
  // changed, or the card did) keeps the rule's shape.
  for (int i = 0; !want_shape.empty() && i < pearl_host_shape_count(ctx_); i++)
    if (ShapeName(i) == want_shape && pearl_host_set_shape(ctx_, i)) shape_now_ = i;
}

std::string PearlCore::ShapeName(int i) const {
  char buf[64] = {0};
  pearl_host_shape_name(ctx_, i, buf, sizeof buf);
  return buf;
}

PearlCore::~PearlCore() {
  running_ = false;
  if (worker_.joinable()) worker_.join();
  if (ctx_) pearl_host_destroy(ctx_);
}

Napi::Value PearlCore::SetJob(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  // A failed core stays failed, and a stopped one has no device context left.
  // The pool keeps sending jobs until the host has handled the core's 'error',
  // so this is reached; ignoring the job is the answer, not a throw into the
  // host's job loop.
  if (failed_ || !ctx_) return env.Undefined();
  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "setJob expects a job object").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  Napi::Object job = info[0].As<Napi::Object>();
  auto header = job.Get("header").As<Napi::Buffer<uint8_t>>();
  if (header.Length() != PEARL_HEADER_BYTES) {
    Napi::TypeError::New(env, "header must be 76 bytes").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  // The target arrives as a BigInt; render it to 32 big-endian bytes, which is
  // what the device comparison expects. ToWords takes (sign_bit, word_count,
  // words) — little-endian 64-bit words, so word 0 is the LEAST significant and
  // the byte loop below reverses that into big-endian.
  uint8_t target[PEARL_HASH_BYTES] = {0};
  int sign_bit = 0;
  uint64_t words[4] = {0, 0, 0, 0};
  size_t word_count = 4;
  job.Get("target").As<Napi::BigInt>().ToWords(&sign_bit, &word_count, words);
  for (size_t w = 0; w < 4; w++) {
    for (int b = 0; b < 8; b++) {
      size_t idx = PEARL_HASH_BYTES - 1 - (w * 8 + b);
      if (idx < PEARL_HASH_BYTES) target[idx] = (uint8_t)(words[w] >> (b * 8));
    }
  }

  {
    std::lock_guard<std::mutex> lock(job_mu_);
    job_id_ = job.Get("jobId").As<Napi::String>().Utf8Value();
    memcpy(job_header_, header.Data(), PEARL_HEADER_BYTES);
    memcpy(job_target_, target, PEARL_HASH_BYTES);
    job_dirty_ = true;
    have_job_ = true;
  }

  if (!running_.exchange(true)) {
    // A search thread that ended on its own -- a failed search -- has returned or
    // is about to, but it is still joinable, and assigning a new std::thread over
    // a joinable one calls std::terminate. That is what closed the app on Windows
    // a few seconds after a CUDA error: the next pool job landed here. Join it.
    if (worker_.joinable()) worker_.join();
    // failed_ is set before running_ is cleared, so after the join it is certain;
    // a job that raced the failure past the check above stops here.
    if (failed_) {
      running_ = false;
      return env.Undefined();
    }
    worker_ = std::thread([this] { SearchLoop(); });
  }
  return env.Undefined();
}

// Release a thread-safe function once and only once.
//
// Release() does not clear the handle, so `if (f) f.Release()` releases again on
// the next call -- and napi asserts rather than returning an error, taking the
// whole process with it. Reached by stopping twice, or by stopping and then
// letting the destructor run.
static void release_tsfn(Napi::ThreadSafeFunction &f) {
  if (!f) return;
  f.Release();
  f = Napi::ThreadSafeFunction();
}

Napi::Value PearlCore::Stop(const Napi::CallbackInfo &info) {
  running_ = false;
  if (worker_.joinable()) worker_.join();
  release_tsfn(on_hit_);
  release_tsfn(on_hashrate_);
  release_tsfn(on_error_);
  release_tsfn(on_shape_);
  // Free the device context HERE, not only in the destructor.
  //
  // The destructor runs when V8 finalises the wrapper, which is whenever it next
  // feels like collecting -- so a stopped miner went on holding its VRAM. That is
  // invisible while mining is the only thing on the card, and fatal the moment
  // something else wants the memory: a node switching from mining to a large LLM
  // measured 2,648 MiB still held 20 s after stop(), read 29,959 MiB free where
  // 30,720 was needed, and silently served a smaller model instead.
  //
  // stop() is documented as "release the GPU"; this makes it true.
  if (ctx_) {
    pearl_host_destroy(ctx_);
    ctx_ = nullptr;
  }
  return info.Env().Undefined();
}

Napi::Value PearlCore::On(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  std::string ev = info[0].As<Napi::String>().Utf8Value();
  Napi::Function cb = info[1].As<Napi::Function>();
  auto tsfn = Napi::ThreadSafeFunction::New(env, cb, "pearl-core-" + ev, 0, 1);
  if (ev == "hit") on_hit_ = tsfn;
  else if (ev == "hashrate") on_hashrate_ = tsfn;
  else if (ev == "error") on_error_ = tsfn;
  else if (ev == "shape") on_shape_ = tsfn;
  else tsfn.Release();
  return env.Undefined();
}

// The search thread. Batches nonces so a job switch is picked up promptly —
// the pool replaces jobs every few seconds and grinding a stale one earns
// nothing.
//
// It keeps two batches queued on the device (pearl_host_submit), and reads the older
// one's result (pearl_host_collect) while the newer runs, so the fold never waits for
// the host between launches. The synchronous loop this replaced left the GPU idle for
// ~0.1 ms at every launch and at every redraw, while the host saw the batch finish,
// read its hit count and queued the next one: about 1% of the rate at four launches a
// salt, measured with the GPU's own event timers.
void PearlCore::SearchLoop() {
  // The current CUDA device is per-thread, and every allocation this loop reads
  // belongs to the card the constructor chose. Inherit it here or launch on
  // device 0 against another card's pointers.
  pearl_host_bind_thread(ctx_);
  win_start_ = std::chrono::steady_clock::now();
  win_work_ = 0.0;
  // A hint only: pearl_host_submit searches the context's own width (col_batch
  // column offsets by every valid row offset) and reports the count in
  // `regions`, which is what the nonce advances by.
  const uint32_t BATCH = PEARL_BATCH_REGIONS;
  // Batches queued at once. Two is enough: one runs while the next waits behind it.
  const size_t DEPTH = 2;
  uint64_t nonce = 0;
  // A region is (row offset, column offset), so one choice of operands offers
  // exactly m*n of them. Past that the search repeats itself, so the operands
  // are re-drawn under a fresh salt and the region index starts over.
  // Valid offsets only: m/rows_count down, n/cols_count across.
  const uint64_t span = ((uint64_t)profile_.m / PEARL_ROWS_COUNT)
                        * ((uint64_t)profile_.n / PEARL_COLS_COUNT);
  uint64_t salt = salt_base_;
  // The job each queued batch belongs to, oldest first: a hit is reported under the
  // job its batch searched, not the one current when it is read.
  std::deque<std::string> queued;
  std::string job_id;
  bool drawn = false;

  // The startup trial of RTX 30's launch shapes (shapeTrialSecs). Each shape gets turns of
  // at least LEG seconds, ended on a whole salt, one turn a round, in an order that reverses
  // each round, as probes/race.cu races builds: what drifts (clock, heat) drifts for all.
  // The first WARM seconds of a turn are not counted, as the race's are not: the card's
  // clock takes a moment to settle at a shape's power. Without it, on a 3090 Ti where the
  // race had the band walk 3.2% behind the flip, the trial read it 0.25% behind.
  // A turn starts and ends with nothing queued, so every batch it counts ran in its shape,
  // and its rate is that work over the turn's time. Hits found during the trial are sent
  // as usual: every shape searches the same regions. At the end the core keeps the shape
  // that beat the rule's (shape 0) by TRIAL_MIN_GAIN on average and in at least two
  // thirds of the rounds, else the rule's, and says so in a 'shape' event.
  const double LEG = 4.0;
  const double WARM = 1.0;
  const double TRIAL_MIN_GAIN = 0.003;
  const int nShapes = shape_trial_secs_ > 0 ? pearl_host_shape_count(ctx_) : 0;
  bool trial = nShapes > 1;
  const int rounds = trial ? std::max(3, (int)(shape_trial_secs_ / ((LEG + WARM) * nShapes))) : 0;
  std::vector<std::vector<double>> legRates(trial ? nShapes : 0);
  int round = 0, turn = 0;
  auto turnShape = [&](int r, int t) { return r % 2 == 0 ? t : nShapes - 1 - t; };
  std::chrono::steady_clock::time_point legStart = std::chrono::steady_clock::now();
  double legWork = 0.0;
  bool legWarm = true;
  if (trial && pearl_host_set_shape(ctx_, turnShape(0, 0))) shape_now_ = turnShape(0, 0);

  // Read the oldest queued batch: count its work toward the hashrate and report a hit.
  auto collect = [&](bool report) -> bool {
    PearlSearchResult r;
    uint64_t attempts = 0;
    char err[256] = {0};
    const bool found = pearl_host_collect(ctx_, &r, &attempts, err, sizeof(err));
    const std::string jid = queued.front();
    queued.pop_front();
    // A CUDA fault mid-search used to vanish here: the loop simply produced no
    // hits and no hashrate, which looks exactly like bad luck. Surface it and
    // stop, rather than spinning on a dead device for ever.
    //
    // failed_ first: setJob reads it to refuse a respawn, and must see it set by
    // the time running_ reads false.
    if (err[0]) {
      failed_ = true;
      if (report) EmitError(err);
      running_ = false;
      return false;
    }
    if (!report) return true;
    // attempts * DAF = multiply-accumulates, which is the unit the network and
    // every other miner reports in. Dividing raw attempts by 1e12 treated one
    // attempt as one hash and under-reported by 65536x at the mainnet profile.
    //
    // And then DIVIDE BY TIME. This emitted work-per-batch for a while, which is
    // not a rate at all: it read as a constant to the last digit no matter how
    // fast the card ran, and any number taken from it was meaningless. Averaged
    // over a short window so a single slow batch does not spike it. Work counts
    // when its batch completes, so the window holds only finished work.
    if (attempts > 0) {
      const auto now = std::chrono::steady_clock::now();
      win_work_ += (double)attempts * PEARL_DAF(profile_);
      legWork += (double)attempts * PEARL_DAF(profile_);
      const double win = std::chrono::duration<double>(now - win_start_).count();
      if (win >= 0.5) {
        EmitHashrate(win_work_ / win / 1e12);
        win_work_ = 0.0;
        win_start_ = now;
      }
    }
    if (found) {
      EmitHit(r, jid);
      // A batch can hold more than one hit; each is its own share.
      PearlSearchResult more;
      while (pearl_host_next_hit(ctx_, &more)) EmitHit(more, jid);
    }
    return true;
  };

  while (running_) {
    // A new job: finish what is queued under the old one, then draw the new one's
    // operands here, on this thread.
    bool fresh = false;
    uint8_t header[PEARL_HEADER_BYTES], target[PEARL_HASH_BYTES];
    std::string fresh_id;
    {
      std::lock_guard<std::mutex> lock(job_mu_);
      if (job_dirty_) {
        memcpy(header, job_header_, sizeof header);
        memcpy(target, job_target_, sizeof target);
        fresh_id = job_id_;
        job_dirty_ = false;
        fresh = true;
      }
    }
    if (fresh) {
      while (!queued.empty())
        if (!collect(true)) return;
      // Every job after the first is drawn at the next salt this core owns, not
      // back at salt_base_. A pool re-sends the job it is on after a reconnect,
      // and the dev fee's login switch can hand over the same header, so going
      // back to salt_base_ would search that salt again and resubmit every share
      // it had already found, which the pool rejects as duplicates. Salts only
      // grow, in steps of the stride, so the cards' spaces stay disjoint.
      if (drawn) salt += salt_stride_;
      pearl_host_set_job_salted(ctx_, header, target, salt);
      job_id = fresh_id;
      drawn = true;
      nonce = 0;
      // A trial turn starts again after a new job's draw: the draw is not the shape's.
      legStart = std::chrono::steady_clock::now();
      legWork = 0.0;
      legWarm = true;
    }
    if (!drawn) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    uint64_t regions = 0;
    char err[256] = {0};
    if (!pearl_host_submit(ctx_, nonce, BATCH, &regions, err, sizeof(err))) {
      // Failed like a collect error: set before running_ clears, so setJob will
      // not respawn a search on a context that is probably dead.
      failed_ = true;
      EmitError(err[0] ? err : "the search could not queue a batch");
      running_ = false;
      break;
    }
    queued.push_back(job_id);
    // Advance by what the batch covers. (A hit no longer ends a batch early: the
    // fold searches the whole batch and lists its hits.)
    nonce += regions;
    const double legSecs =
        trial ? std::chrono::duration<double>(std::chrono::steady_clock::now() - legStart).count()
              : 0.0;
    if (nonce >= span && trial && legWarm && legSecs >= WARM) {
      // The end of a turn's warm-up: finish its batches, then start counting.
      while (!queued.empty())
        if (!collect(true)) return;
      legStart = std::chrono::steady_clock::now();
      legWork = 0.0;
      legWarm = false;
    } else if (nonce >= span && trial && !legWarm && legSecs >= LEG) {
      // The end of a trial turn: finish its batches, then time it.
      while (!queued.empty())
        if (!collect(true)) return;
      const auto now = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(now - legStart).count();
      legRates[turnShape(round, turn)].push_back(secs > 0 ? legWork / secs : 0.0);
      if (++turn == nShapes) {
        turn = 0;
        round++;
      }
      int next;
      if (round < rounds) {
        next = turnShape(round, turn);
      } else {
        trial = false;
        next = 0;
        double bestGain = TRIAL_MIN_GAIN;
        std::string rows;
        for (int i = 1; i < nShapes; i++) {
          double sum = 0.0;
          int ahead = 0, n = 0;
          for (size_t r = 0; r < legRates[i].size() && r < legRates[0].size(); r++) {
            if (legRates[0][r] <= 0) continue;
            const double ratio = legRates[i][r] / legRates[0][r];
            sum += ratio - 1.0;
            ahead += ratio > 1.0;
            n++;
          }
          const double gain = n ? sum / n : 0.0;
          if (n && gain > bestGain && ahead * 3 >= n * 2) {
            bestGain = gain;
            next = i;
          }
          char row[160];
          snprintf(row, sizeof row, "%s{\"shape\":\"%s\",\"pct\":%.3f,\"ahead\":%d,\"rounds\":%d}",
                   rows.empty() ? "" : ",", ShapeName(i).c_str(), gain * 100.0, ahead, n);
          rows += row;
        }
        EmitShape("{\"shape\":\"" + ShapeName(next) + "\",\"rule\":\"" + ShapeName(0)
                  + "\",\"trial\":[" + rows + "]}");
      }
      if (pearl_host_set_shape(ctx_, next)) shape_now_ = next;
      legStart = std::chrono::steady_clock::now();
      legWork = 0.0;
      legWarm = true;
    }
    if (nonce >= span) {
      // The next salt THIS core owns, not simply the next one: on a multi-card
      // rig the stride is the card count, so the cards never collide. The redraw
      // queues behind the batches already queued, and the host knows the new
      // a_seed without waiting for it (see pearl_host_reseed).
      salt += salt_stride_;
      pearl_host_reseed(ctx_, salt);
      nonce = 0;
    }
    if (queued.size() >= DEPTH && !collect(true)) return;
  }
  // Stopping: let the queued batches finish, unreported, before the context goes.
  while (!queued.empty())
    if (!collect(false)) break;
}

void PearlCore::EmitHit(const PearlSearchResult &r, const std::string &job_id) {
  if (!on_hit_) return;
  PearlSearchResult *copy = new PearlSearchResult(r);
  std::string *jid = new std::string(job_id);
  on_hit_.BlockingCall(copy, [jid](Napi::Env env, Napi::Function cb,
                                   PearlSearchResult *res) {
    Napi::Object o = Napi::Object::New(env);
    o.Set("jobId", Napi::String::New(env, *jid));
    o.Set("nonce", Napi::Number::New(env, (double)res->nonce));
    // The salt identifies the operand draw, which the host needs to rebuild the
    // Merkle proof the pool verifies the share with.
    o.Set("salt", Napi::Number::New(env, (double)res->salt));
    o.Set("jackpotHash",
          Napi::Buffer<uint8_t>::Copy(env, res->jackpot_hash, PEARL_HASH_BYTES));
    // Seeds travel as hex, which is what the Stratum submit expects.
    char hex[PEARL_HASH_BYTES * 2 + 1];
    auto to_hex = [&](const uint8_t *b) {
      for (int i = 0; i < PEARL_HASH_BYTES; i++) sprintf(hex + i * 2, "%02x", b[i]);
      hex[PEARL_HASH_BYTES * 2] = 0;
      return std::string(hex);
    };
    o.Set("aSeed", Napi::String::New(env, to_hex(res->a_seed)));
    o.Set("bSeed", Napi::String::New(env, to_hex(res->b_seed)));
    o.Set("proof", Napi::Buffer<uint8_t>::Copy(env, res->proof.data(), res->proof.size()));

    // The share proof, carried with this hit rather than read back later --
    // the search does not stop, and a later hit would overwrite a shared one.
    auto side = [&](const PearlProofSide &p) {
      Napi::Object s = Napi::Object::New(env);
      Napi::Array idx = Napi::Array::New(env, p.leaf_indices.size());
      for (size_t i = 0; i < p.leaf_indices.size(); i++) {
        idx.Set((uint32_t)i, Napi::Number::New(env, p.leaf_indices[i]));
      }
      s.Set("leafIndices", idx);
      s.Set("leafData", Napi::Buffer<uint8_t>::Copy(env, p.leaves.data(), p.leaves.size()));
      s.Set("siblings", Napi::Buffer<uint8_t>::Copy(env, p.siblings.data(), p.siblings.size()));
      s.Set("root", Napi::Buffer<uint8_t>::Copy(env, p.root, PEARL_HASH_BYTES));
      s.Set("totalLeaves", Napi::Number::New(env, (double)p.total_leaves));
      return s;
    };
    o.Set("proofA", side(res->proof_a));
    o.Set("proofBt", side(res->proof_bt));
    cb.Call({o});
    delete jid;
    delete res;
  });
}

// Telemetry must never be able to stall the search. Emitting one of these per
// batch -- several hundred a second -- wedged the worker thread inside
// BlockingCall after a few thousand calls: the GPU went to 0% and the miner
// simply stopped, with no error and no hits, looking exactly like bad luck.
//
// Two changes stop that. The caller now windows these to about two a second,
// and this is a NON-blocking call: if the queue is backed up the sample is
// dropped. A missing hashrate sample costs nothing; a blocked miner costs
// everything.
void PearlCore::EmitHashrate(double th) {
  if (!on_hashrate_) return;
  double *v = new double(th);
  napi_status st =
      on_hashrate_.NonBlockingCall(v, [](Napi::Env env, Napi::Function cb, double *d) {
        cb.Call({Napi::Number::New(env, *d)});
        delete d;
      });
  if (st != napi_ok) delete v;  // dropped: free what the callback would have
}

// The trial's outcome, as JSON: the shape kept, the rule's, and each other shape's mean
// change against the rule's in percent, the rounds it led and the rounds counted. The app
// caches the pick for this card.
void PearlCore::EmitShape(const std::string &json) {
  if (!on_shape_) return;
  std::string *m = new std::string(json);
  napi_status st = on_shape_.NonBlockingCall(m, [](Napi::Env env, Napi::Function cb, std::string *s) {
    cb.Call({Napi::String::New(env, *s)});
    delete s;
  });
  if (st != napi_ok) delete m;
}

// Every error names the card: on a multi-card rig "misaligned address" alone
// does not say which GPU stopped.
void PearlCore::EmitError(const std::string &msg) {
  if (!on_error_) return;
  std::string *m = new std::string("GPU " + std::to_string(device_index_) + ": " + msg);
  on_error_.BlockingCall(m, [](Napi::Env env, Napi::Function cb, std::string *s) {
    cb.Call({Napi::Error::New(env, *s).Value()});
    delete s;
  });
}

// { index, name } for the card this core opened — the host shows this rather
// than its own guess, so the name on screen is the card doing the work.
Napi::Value PearlCore::GetDevice(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object out = Napi::Object::New(env);
  out.Set("index", Napi::Number::New(env, device_index_));
  out.Set("name", Napi::String::New(env, device_name_));
  // The fold, for logs and probes (verify-hits.js prints it). The host's readDevice
  // takes only index and name.
  out.Set("fold", Napi::String::New(env, pearl_host_fold_name(ctx_)));
  return out;
}

// The k this card mines (cardK may have raised it), which the host builds this
// card's config52, job_key, share bound and proofs from. A core built before
// this has no `k`, and the host then uses its own profile's.
Napi::Value PearlCore::GetK(const Napi::CallbackInfo &info) {
  return Napi::Number::New(info.Env(), profile_.k);
}

// Why the card mines the profile's k when it could mine a larger one, or "".
Napi::Value PearlCore::GetKNote(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), k_note_);
}

// Whether the host must check this card's hits before sending them (see checkHits).
Napi::Value PearlCore::GetUntested(const Napi::CallbackInfo &info) {
  return Napi::Boolean::New(info.Env(), untested_);
}

// The names of the launch shapes this card can run, the rule's first; [] where there is
// nothing to choose.
Napi::Value PearlCore::GetShapes(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const int n = ctx_ ? pearl_host_shape_count(ctx_) : 0;
  Napi::Array out = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) out.Set((uint32_t)i, Napi::String::New(env, ShapeName(i)));
  return out;
}

// The shape this card runs now, or "" where there is nothing to choose.
Napi::Value PearlCore::GetShape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (!ctx_ || pearl_host_shape_count(ctx_) == 0) return Napi::String::New(env, "");
  return Napi::String::New(env, ShapeName(shape_now_));
}

Napi::Object PearlCore::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "PearlCore",
                  {InstanceMethod("setJob", &PearlCore::SetJob),
                   InstanceMethod("stop", &PearlCore::Stop),
                   InstanceMethod("on", &PearlCore::On),
                   InstanceAccessor("device", &PearlCore::GetDevice, nullptr),
                   InstanceAccessor("k", &PearlCore::GetK, nullptr),
                   InstanceAccessor("kNote", &PearlCore::GetKNote, nullptr),
                   InstanceAccessor("untested", &PearlCore::GetUntested, nullptr),
                   InstanceAccessor("shapes", &PearlCore::GetShapes, nullptr),
                   InstanceAccessor("shape", &PearlCore::GetShape, nullptr)});
  exports.Set("PearlCore", func);
  return exports;
}

Napi::Value CreateCore(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object exports = env.Global().Get("__pearl_exports").As<Napi::Object>();
  Napi::Function ctor = exports.Get("PearlCore").As<Napi::Function>();
  // Both arguments: the profile, and the core options (deviceIndex). Forwarding
  // only the first silently dropped the card choice on the floor.
  return ctor.New({info.Length() > 0 ? info[0] : env.Undefined(),
                   info.Length() > 1 ? info[1] : env.Undefined()});
}

#ifdef PEARL_NVML_OFFSETS
// ---- The core clock offset ---------------------------------------------------
//
// An offset moves a card's whole clock-to-voltage curve: at +100 MHz it runs each
// voltage 100 MHz faster. A card held at a power limit then gets more clock for
// the same watts, which is where the fold's rate comes from (benchmark.md, "core
// clock offset on the RTX 4090"). nvidia-smi has no switch for it, so this calls
// NVML, the library nvidia-smi itself is built on.
//
// NVML is opened at run time and its functions looked up by name. The addon so
// has no link-time dependency on it, and loads on a machine that has none: these
// two functions then answer { ok: false } and everything else works. Nothing here
// throws into JS, whatever it is handed.
//
// The types below are NVIDIA's, from nvml.h (API version 13), declared here
// rather than included: the CUDA 12.8 toolkit the release is built with has an
// nvml.h older than these two calls.
namespace nvml {

typedef struct nvmlDevice_st *Device;  // nvmlDevice_t
typedef int Return;                    // nvmlReturn_t, an enum
// nvmlClockOffset_v1_t. Its two enums (nvmlClockType_t, nvmlPstates_t) are ints.
struct ClockOffset {
  unsigned int version;
  int type;
  int pstate;
  int clockOffsetMHz;
  int minClockOffsetMHz;
  int maxClockOffsetMHz;
};
// nvmlClockOffset_v1, which is NVML_STRUCT_VERSION(ClockOffset, 1): the struct's
// size with the version in the top byte, 0x1000018.
const unsigned int kClockOffsetV1 = (unsigned int)(sizeof(ClockOffset) | (1U << 24U));
const int kClockGraphics = 0;  // NVML_CLOCK_GRAPHICS: the core clock, not the memory's
const int kPstate0 = 0;        // NVML_PSTATE_0: full performance, the state a card mines in
const Return kSuccess = 0;     // NVML_SUCCESS

struct Api {
  Return (*init)() = nullptr;                                 // nvmlInit_v2
  Return (*shutdown)() = nullptr;                             // nvmlShutdown
  Return (*handleByIndex)(unsigned int, Device *) = nullptr;  // nvmlDeviceGetHandleByIndex_v2
  Return (*getOffsets)(Device, ClockOffset *) = nullptr;      // nvmlDeviceGetClockOffsets
  Return (*setOffsets)(Device, ClockOffset *) = nullptr;      // nvmlDeviceSetClockOffsets
  const char *(*errorString)(Return) = nullptr;               // nvmlErrorString
  // Why there is nothing to call: no library, or one too old to have a function.
  std::string missing;
};

void *OpenLibrary() {
#ifdef _WIN32
  // From System32 only, where the driver installs it: never an nvml.dll that
  // happens to sit beside the app or on PATH.
  return (void *)LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
  return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void *Find(void *lib, const char *name) {
#ifdef _WIN32
  return (void *)GetProcAddress((HMODULE)lib, name);
#else
  return dlsym(lib, name);
#endif
}

// Opened once and kept for the life of the process.
Api LoadApi() {
  Api api;
  void *lib = OpenLibrary();
  if (!lib) {
    api.missing = "the NVIDIA management library (NVML) is not installed";
    return api;
  }
  const char *lost = nullptr;
  auto find = [&](const char *name) -> void * {
    void *p = Find(lib, name);
    if (!p && !lost) lost = name;
    return p;
  };
  api.init = reinterpret_cast<Return (*)()>(find("nvmlInit_v2"));
  api.shutdown = reinterpret_cast<Return (*)()>(find("nvmlShutdown"));
  api.handleByIndex =
      reinterpret_cast<Return (*)(unsigned int, Device *)>(find("nvmlDeviceGetHandleByIndex_v2"));
  api.getOffsets =
      reinterpret_cast<Return (*)(Device, ClockOffset *)>(find("nvmlDeviceGetClockOffsets"));
  api.setOffsets =
      reinterpret_cast<Return (*)(Device, ClockOffset *)>(find("nvmlDeviceSetClockOffsets"));
  api.errorString = reinterpret_cast<const char *(*)(Return)>(find("nvmlErrorString"));
  if (lost) {
    api.missing = std::string("this NVIDIA driver's NVML has no ") + lost
                  + ": the driver is too old for clock offsets";
  }
  return api;
}

const Api &GetApi() {
  static const Api api = LoadApi();
  return api;
}

// What a call left behind: nothing, or the words for the log and NVML's own code
// (0 when the failure is not NVML's).
struct Failure {
  std::string error;
  Return code = 0;
};

// A failure of the caller's, not NVML's.
Failure Refused(const char *why) {
  Failure f;
  f.error = why;
  return f;
}

// Run `fn` on card `index`, between nvmlInit and nvmlShutdown. NVML counts its
// inits, so this never ends a session something else in the process holds.
template <typename F>
Failure WithDevice(unsigned int index, const char *what, F fn) {
  const Api &api = GetApi();
  Failure f;
  if (!api.missing.empty()) {
    f.error = api.missing;
    return f;
  }
  auto fail = [&](const char *step, Return r) {
    const char *said = api.errorString(r);
    f.error = std::string(step) + ": " + (said ? said : "unknown NVML error");
    f.code = r;
  };
  Return r = api.init();
  if (r != kSuccess) {
    fail("nvmlInit", r);
    return f;
  }
  Device dev = nullptr;
  r = api.handleByIndex(index, &dev);
  if (r != kSuccess) fail("no such card", r);
  if (f.error.empty()) {
    r = fn(api, dev);
    if (r != kSuccess) fail(what, r);
  }
  api.shutdown();
  return f;
}

ClockOffset Blank() {
  ClockOffset info = {};
  info.version = kClockOffsetV1;
  info.type = kClockGraphics;
  info.pstate = kPstate0;
  return info;
}

Napi::Object Answer(Napi::Env env, const Failure &f) {
  Napi::Object out = Napi::Object::New(env);
  out.Set("ok", Napi::Boolean::New(env, f.error.empty()));
  if (!f.error.empty()) {
    out.Set("error", Napi::String::New(env, f.error));
    out.Set("code", Napi::Number::New(env, f.code));
  }
  return out;
}

// A card index from JS: a whole number from 0 up. -1 for anything else.
int CardIndex(const Napi::CallbackInfo &info) {
  if (info.Length() < 1 || !info[0].IsNumber()) return -1;
  const double d = info[0].As<Napi::Number>().DoubleValue();
  if (!(d >= 0 && d <= 1024) || d != (double)(int)d) return -1;
  return (int)d;
}

// getClockOffset(cardIndex): the card's core clock offset on P-state 0 and the
// range the driver allows, all in MHz.
Napi::Value GetClockOffset(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const int index = CardIndex(info);
  if (index < 0) return Answer(env, Refused("the card index must be a whole number"));
  ClockOffset got = Blank();
  const Failure f = WithDevice((unsigned int)index, "reading the clock offset",
                               [&](const Api &api, Device dev) { return api.getOffsets(dev, &got); });
  Napi::Object out = Answer(env, f);
  if (f.error.empty()) {
    out.Set("mhz", Napi::Number::New(env, got.clockOffsetMHz));
    out.Set("min", Napi::Number::New(env, got.minClockOffsetMHz));
    out.Set("max", Napi::Number::New(env, got.maxClockOffsetMHz));
  }
  return out;
}

// setClockOffset(cardIndex, mhz): set the card's core clock offset on P-state 0.
// The driver refuses a value outside its range, and a caller without root or
// administrator rights (NVML_ERROR_NO_PERMISSION, code 4).
Napi::Value SetClockOffset(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const int index = CardIndex(info);
  if (index < 0) return Answer(env, Refused("the card index must be a whole number"));
  double mhz = 0.0;
  const bool given = info.Length() > 1 && info[1].IsNumber();
  if (given) mhz = info[1].As<Napi::Number>().DoubleValue();
  if (!given || !(mhz >= -100000 && mhz <= 100000) || mhz != (double)(int)mhz) {
    return Answer(env, Refused("the offset must be a whole number of MHz"));
  }
  ClockOffset want = Blank();
  want.clockOffsetMHz = (int)mhz;
  const Failure f = WithDevice((unsigned int)index, "setting the clock offset",
                               [&](const Api &api, Device dev) { return api.setOffsets(dev, &want); });
  return Answer(env, f);
}

}  // namespace nvml
#endif  // PEARL_NVML_OFFSETS

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  PearlCore::Init(env, exports);
  env.Global().Set("__pearl_exports", exports);
  exports.Set("createCore", Napi::Function::New(env, CreateCore));
#ifdef PEARL_NVML_OFFSETS
  exports.Set("getClockOffset", Napi::Function::New(env, nvml::GetClockOffset));
  exports.Set("setClockOffset", Napi::Function::New(env, nvml::SetClockOffset));
#endif
  return exports;
}

}  // namespace

NODE_API_MODULE(pearl_core, InitAll)
