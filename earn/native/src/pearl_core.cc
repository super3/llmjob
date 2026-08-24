// N-API binding: exposes the CUDA core to the JS host as
//
//   createCore(profile) -> {
//     setJob({ header: Buffer(76), target: BigInt, jobId: string }),
//     stop(),
//     on('hit'|'hashrate'|'error', cb)
//   }
//
// The search runs on its own thread so the Electron main thread is never
// blocked, and results are marshalled back with a thread-safe function. The JS
// host re-verifies every hit before submitting (see pearlMiner._onHit), so this
// layer is free to be optimistic.
//
// Independent implementation against the ISC-licensed pearl-research-labs
// specification. No dev fee, no fee address, nothing to disclose — see
// pearl_kernel.cu for why that matters.

#include <napi.h>

#include <atomic>
#include <cstring>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "pearl_config.h"

// Declared in pearl_host.cu — the host-side driver that owns device memory and
// runs the kernel pipeline for one job.
struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES];
  uint8_t a_seed[PEARL_HASH_BYTES];
  uint8_t b_seed[PEARL_HASH_BYTES];
  uint64_t nonce;
  std::vector<uint8_t> proof;
  bool found;
};

extern "C" {
// Allocate device state for a profile. Returns null on failure (no CUDA device,
// insufficient memory) with a message in `err`.
void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len);
void pearl_host_destroy(void *ctx);
// Load a job. header is 76 bytes, target is 32 big-endian bytes.
void pearl_host_set_job(void *ctx, const uint8_t *header, const uint8_t *target);
// Search one batch of nonces. Returns true and fills `out` when a share is
// found; returns false when the batch is exhausted with no hit. `attempts`
// receives the number of nonces tried, for the hashrate figure.
bool pearl_host_search(void *ctx, uint64_t nonce_base, uint32_t batch,
                       PearlSearchResult *out, uint64_t *attempts, char *err,
                       size_t err_len);
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

  void SearchLoop();
  void EmitHit(const PearlSearchResult &r, const std::string &job_id);
  void EmitHashrate(double th_per_sec);
  void EmitError(const std::string &msg);

  PearlProfile profile_ = PEARL_MAINNET_PROFILE;
  // Hashrate averaging window. The emitted value is a RATE, so it needs both a
  // work accumulator and a start time.
  std::chrono::steady_clock::time_point win_start_{};
  double win_work_ = 0.0;
  void *ctx_ = nullptr;
  std::thread worker_;
  std::atomic<bool> running_{false};
  std::mutex job_mu_;
  std::string job_id_;
  bool have_job_ = false;

  Napi::ThreadSafeFunction on_hit_;
  Napi::ThreadSafeFunction on_hashrate_;
  Napi::ThreadSafeFunction on_error_;
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
  }

  profile_ = profile;
  char err[256] = {0};
  ctx_ = pearl_host_create(&profile, err, sizeof(err));
  if (!ctx_) {
    // A missing or too-old CUDA device is the common case. Throwing here surfaces
    // in the host as a clean 'error' rather than a crash, and the host then
    // reports that the engine is unavailable on this machine.
    Napi::Error::New(env, err[0] ? err : "failed to initialise the Pearl CUDA core")
        .ThrowAsJavaScriptException();
  }
}

PearlCore::~PearlCore() {
  running_ = false;
  if (worker_.joinable()) worker_.join();
  if (ctx_) pearl_host_destroy(ctx_);
}

Napi::Value PearlCore::SetJob(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
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
    pearl_host_set_job(ctx_, header.Data(), target);
    have_job_ = true;
  }

  if (!running_.exchange(true)) {
    worker_ = std::thread([this] { SearchLoop(); });
  }
  return env.Undefined();
}

Napi::Value PearlCore::Stop(const Napi::CallbackInfo &info) {
  running_ = false;
  if (worker_.joinable()) worker_.join();
  if (on_hit_) on_hit_.Release();
  if (on_hashrate_) on_hashrate_.Release();
  if (on_error_) on_error_.Release();
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
  else tsfn.Release();
  return env.Undefined();
}

// The search thread. Batches nonces so a job switch is picked up promptly —
// the pool replaces jobs every few seconds and grinding a stale one earns
// nothing.
void PearlCore::SearchLoop() {
  win_start_ = std::chrono::steady_clock::now();
  win_work_ = 0.0;
  // Must match PEARL_BATCH_REGIONS: the fold launches one CUDA block per region
  // and the host sizes its batch scratch to this.
  const uint32_t BATCH = PEARL_BATCH_REGIONS;
  uint64_t nonce = 0;
  while (running_) {
    std::string job_id;
    {
      std::lock_guard<std::mutex> lock(job_mu_);
      if (!have_job_) continue;
      job_id = job_id_;
    }
    PearlSearchResult r;
    uint64_t attempts = 0;
    char err[256] = {0};
    bool found = pearl_host_search(ctx_, nonce, BATCH, &r, &attempts, err, sizeof(err));
    // A CUDA fault mid-search used to vanish here: the loop simply produced no
    // hits and no hashrate, which looks exactly like bad luck. Surface it and
    // stop, rather than spinning on a dead device for ever.
    if (err[0]) {
      EmitError(err);
      running_ = false;
      return;
    }
    // Advance by what was actually consumed, NOT by the batch size. The search
    // returns the moment it finds a share, so a batch that hits at index 0 has
    // tried exactly one region — and skipping ahead a whole batch throws away
    // the other 4095 unexamined.
    //
    // Measured on a 4090 before this fix: with a permissive target every batch
    // hit immediately, so every reported nonce was a multiple of BATCH. BATCH
    // being a multiple of m then pinned row_off at 0 for ever and the entire
    // search collapsed to the 64 distinct column offsets. Against a real target
    // batches rarely hit and attempts == BATCH, so this changes nothing there
    // except that found work is no longer discarded.
    nonce += (attempts > 0 ? attempts : BATCH);
    // attempts * DAF = multiply-accumulates, which is the unit the network and
    // every other miner reports in. Dividing raw attempts by 1e12 treated one
    // attempt as one hash and under-reported by 65536x at the mainnet profile.
    //
    // And then DIVIDE BY TIME. This emitted work-per-batch for a while, which is
    // not a rate at all: it read as a constant to the last digit no matter how
    // fast the card ran, and any number taken from it was meaningless. Averaged
    // over a short window so a single slow batch does not spike it.
    if (attempts > 0) {
      const auto now = std::chrono::steady_clock::now();
      win_work_ += (double)attempts * PEARL_DAF(profile_);
      const double win = std::chrono::duration<double>(now - win_start_).count();
      if (win >= 0.5) {
        EmitHashrate(win_work_ / win / 1e12);
        win_work_ = 0.0;
        win_start_ = now;
      }
    }
    if (found) EmitHit(r, job_id);
  }
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

void PearlCore::EmitError(const std::string &msg) {
  if (!on_error_) return;
  std::string *m = new std::string(msg);
  on_error_.BlockingCall(m, [](Napi::Env env, Napi::Function cb, std::string *s) {
    cb.Call({Napi::Error::New(env, *s).Value()});
    delete s;
  });
}

Napi::Object PearlCore::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "PearlCore",
                  {InstanceMethod("setJob", &PearlCore::SetJob),
                   InstanceMethod("stop", &PearlCore::Stop),
                   InstanceMethod("on", &PearlCore::On)});
  exports.Set("PearlCore", func);
  return exports;
}

Napi::Value CreateCore(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object exports = env.Global().Get("__pearl_exports").As<Napi::Object>();
  Napi::Function ctor = exports.Get("PearlCore").As<Napi::Function>();
  return ctor.New({info.Length() > 0 ? info[0] : env.Undefined()});
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  PearlCore::Init(env, exports);
  env.Global().Set("__pearl_exports", exports);
  exports.Set("createCore", Napi::Function::New(env, CreateCore));
  return exports;
}

}  // namespace

NODE_API_MODULE(pearl_core, InitAll)
