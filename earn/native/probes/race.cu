// race -- two or more builds of the core in one process, on one card, taking turns.
//
//   race <seconds> <leg seconds> <warm seconds> base=libA.so name=libB.so[:K] [name=lib.so[:K] ...]
//
// Why. Separate runs of two builds (bench, hashrate.js) see different clocks, heat and
// power, and on Salad often a different PC: rounds read +-1.5-2.4% apart, and the first
// round of a box ran 2.5-3 points off the rest. Most of the changes left to try are worth
// 0.3-1.5%. Here every build gets turns of a few seconds on the same card, so whatever
// drifts drifts for all of them, and a round compares them seconds apart (research round
// 3, item 7). It is a measuring tool only: nothing in the app runs it.
//
// Each lib.so is one build's pearl_kernel.o and pearl_host.o linked -shared with the
// static CUDA runtime and -Bsymbolic, as compilers.sh's build stage makes them. Each is
// dlopen'd RTLD_LOCAL, so builds with the same symbols don't mix. They share the card's
// one CUDA context, so they never run at the same time: a turn starts on an idle card.
// The first build is the reference. A copy of a file under another name is an A/A
// control (the same path twice would load once).
//
// k. A build mines at k 2048, the mainnet profile's, unless its argument ends in :K. A
// -DPEARL_FOLD_K=4096 build mines only k 4096 (pearl_host_submit refuses any other), so it
// is named name=lib.so:4096. Each build gets its own profile and its own DAF (256 x k), so
// every rate is MACs a second and builds at different k compare fairly. The salt's regions
// (span) do not depend on k, so every build shares them. A salt at k 4096 is twice the
// work, and turns end only on whole salts: on a slow card (RTX 20, where such a salt takes
// 3 s or more) give turns of 8-10 s, so each still counts several.
//
// A round gives every build one turn, in an order that reverses each round. A turn first
// sets the build's own persisting L2 slice again (pearl_host_race_enter: the slice is the
// context's, so the last build to set it would hold for all), then runs the miner's own
// loop, as PearlCore::SearchLoop does: two batches queued, the operands re-drawn under
// the next salt every m*n regions. It warms up for <warm> seconds, to the end of a salt,
// then counts whole salts until <leg> seconds have passed, and stops at the end of a
// salt. Its rate is the counted salts' work over the time from the batch that ended the
// warm-up to the one that ended the last salt: re-draws count as they do in the app, and
// the card never idles inside it. NVML samples SM clock, power, temperature and memory
// controller use every 50 ms of the counted time.
//
// Output, one line each: START, AUDIT a build (pearl_host_l2_audit: the batch width, band
// and persisting slice it got), LEG a turn, then RACE a build: its mean ratio to the
// reference's turn in the same round, as a change in percent, the standard deviation of
// those ratios, a 95% interval of the mean, and in how many rounds it was ahead. AUDIT and
// RACE lines end with the build's k. FAIL and exit 1 on any error.
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "pearl_config.h"

// Must match pearl_host.cu's layout exactly (as bench.cu's copy does).
struct PearlProofSide {
  std::vector<uint32_t> leaf_indices; std::vector<uint8_t> leaves; std::vector<uint8_t> siblings;
  uint8_t root[PEARL_HASH_BYTES]; uint64_t total_leaves;
};
struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES], a_seed[PEARL_HASH_BYTES], b_seed[PEARL_HASH_BYTES];
  uint64_t nonce, salt; std::vector<uint8_t> proof;
  PearlProofSide proof_a, proof_bt; bool found;
};

typedef void *(*CreateFn)(const PearlProfile *, char *, size_t);
typedef void (*SetJobFn)(void *, const uint8_t *, const uint8_t *, uint64_t);
typedef void (*ReseedFn)(void *, uint64_t);
typedef bool (*SubmitFn)(void *, uint64_t, uint32_t, uint64_t *, char *, size_t);
typedef bool (*CollectFn)(void *, PearlSearchResult *, uint64_t *, char *, size_t);
typedef bool (*NextHitFn)(void *, PearlSearchResult *);
typedef const char *(*FoldNameFn)(void *);
typedef void (*AuditFn)(void *, char *, size_t);
typedef void (*EnterFn)(void *);

struct Leg { int round; double th, secs; uint64_t salts; double mhz, watts, tempC, mem; };

struct Build {
  std::string name, path;
  PearlProfile prof{};   // the mainnet profile at this build's k
  double daf = 0;        // PEARL_DAF(prof): MACs a region
  void *lib = nullptr, *ctx = nullptr;
  CreateFn create = nullptr; SetJobFn setJob = nullptr; ReseedFn reseed = nullptr;
  SubmitFn submit = nullptr; CollectFn collect = nullptr; NextHitFn nextHit = nullptr;
  FoldNameFn foldName = nullptr; AuditFn audit = nullptr; EnterFn enter = nullptr;
  uint64_t salt = 0, nonce = 0;
  std::vector<Leg> legs;
};

// NVML through dlopen: the driver ships libnvidia-ml.so.1, and the toolkit nvtest.sh
// installs has no nvml.h. The functions and constants as nvml.h declares them.
typedef int (*NvmlInitFn)();
typedef int (*NvmlHandleFn)(unsigned, void **);
typedef int (*NvmlClockFn)(void *, int, unsigned *);
typedef int (*NvmlPowerFn)(void *, unsigned *);
typedef int (*NvmlTempFn)(void *, int, unsigned *);
struct NvmlUtil { unsigned gpu, memory; };
typedef int (*NvmlUtilFn)(void *, NvmlUtil *);
static void *g_dev = nullptr;
static NvmlClockFn nvClock; static NvmlPowerFn nvPower; static NvmlTempFn nvTemp; static NvmlUtilFn nvUtil;

static bool nvml_open() {
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) return false;
  NvmlInitFn init = (NvmlInitFn)dlsym(h, "nvmlInit_v2");
  NvmlHandleFn handle = (NvmlHandleFn)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  nvClock = (NvmlClockFn)dlsym(h, "nvmlDeviceGetClockInfo");
  nvPower = (NvmlPowerFn)dlsym(h, "nvmlDeviceGetPowerUsage");
  nvTemp = (NvmlTempFn)dlsym(h, "nvmlDeviceGetTemperature");
  nvUtil = (NvmlUtilFn)dlsym(h, "nvmlDeviceGetUtilizationRates");
  if (!init || !handle || !nvClock || !nvPower || !nvTemp || !nvUtil) return false;
  // Index 0, as compilers.sh's nvidia-smi -i 0: the boxes have one card.
  return init() == 0 && handle(0, &g_dev) == 0;
}

// The sampler: while `g_on`, one sample every 50 ms into the sums.
static std::atomic<bool> g_on{false}, g_quit{false};
static std::mutex g_mu;
static double g_sum[4]; static unsigned g_n;
static void sampler() {
  while (!g_quit) {
    if (g_on && g_dev) {
      unsigned mhz = 0, mw = 0, c = 0;
      NvmlUtil u{0, 0};
      const bool ok = nvClock(g_dev, 1 /* NVML_CLOCK_SM */, &mhz) == 0 && nvPower(g_dev, &mw) == 0
                      && nvTemp(g_dev, 0 /* NVML_TEMPERATURE_GPU */, &c) == 0 && nvUtil(g_dev, &u) == 0;
      if (ok) {
        std::lock_guard<std::mutex> l(g_mu);
        g_sum[0] += mhz; g_sum[1] += mw / 1000.0; g_sum[2] += c; g_sum[3] += u.memory; g_n++;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

static void fail(const std::string &m) {
  printf("FAIL %s\n", m.c_str());
  fflush(stdout);
  exit(1);
}

template <typename T> static void sym(Build &b, T &f, const char *name, bool need = true) {
  f = (T)dlsym(b.lib, name);
  if (!f && need) fail(b.name + ": " + b.path + " has no " + name);
}

typedef std::chrono::steady_clock Clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// One turn of build b. span is a salt's regions, stride the salts between this build's.
static Leg run_leg(Build &b, int round, double legSecs, double warmSecs, uint64_t span,
                   uint64_t stride) {
  if (b.enter) b.enter(b.ctx);
  const size_t DEPTH = 2;
  struct Queued { uint64_t regions; bool endsSalt; };
  std::deque<Queued> q;
  char err[512];
  const Clock::time_point t0 = Clock::now();
  Clock::time_point tm0 = t0, tEnd = t0;
  bool stop = false, counting = false;
  uint64_t work = 0, workAtEnd = 0, salts = 0;
  for (;;) {
    while (!stop && q.size() < DEPTH) {
      // A salt starts here: the turn may end, but only after it has counted one.
      if (b.nonce == 0 && counting && secs(t0, Clock::now()) >= legSecs && salts > 0) {
        stop = true;
        break;
      }
      uint64_t regions = 0;
      err[0] = 0;
      if (!b.submit(b.ctx, b.nonce, PEARL_BATCH_REGIONS, &regions, err, sizeof err))
        fail(b.name + ": submit: " + (err[0] ? err : "refused"));
      if (regions == 0) fail(b.name + ": a batch of 0 regions");
      b.nonce += regions;
      bool ends = false;
      if (b.nonce >= span) {
        // Every region of the salt once: the batches must end exactly on the salt.
        if (b.nonce != span) fail(b.name + ": the batches overran the salt");
        b.salt += stride;
        b.reseed(b.ctx, b.salt);
        b.nonce = 0;
        ends = true;
      }
      q.push_back({regions, ends});
    }
    if (q.empty()) break;
    PearlSearchResult r;
    uint64_t attempts = 0;
    err[0] = 0;
    const bool found = b.collect(b.ctx, &r, &attempts, err, sizeof err);
    if (err[0]) fail(b.name + ": collect: " + err);
    if (found) {
      PearlSearchResult more;
      while (b.nextHit(b.ctx, &more)) {}
    }
    const Clock::time_point t = Clock::now();
    const Queued done = q.front();
    q.pop_front();
    if (attempts != done.regions) fail(b.name + ": a batch reported other regions than it queued");
    if (counting) work += done.regions;
    if (done.endsSalt) {
      if (!counting && secs(t0, t) >= warmSecs) {
        counting = true;
        tm0 = t;
        { std::lock_guard<std::mutex> l(g_mu); g_sum[0] = g_sum[1] = g_sum[2] = g_sum[3] = 0; g_n = 0; }
        g_on = true;
      } else if (counting) {
        salts++;
        tEnd = t;
        workAtEnd = work;
      }
    }
  }
  g_on = false;
  Leg leg{round, 0, 0, salts, NAN, NAN, NAN, NAN};
  leg.secs = secs(tm0, tEnd);
  if (salts == 0 || leg.secs <= 0) fail(b.name + ": a turn counted no salt");
  leg.th = (double)workAtEnd * b.daf / leg.secs / 1e12;
  std::lock_guard<std::mutex> l(g_mu);
  if (g_n) {
    leg.mhz = g_sum[0] / g_n; leg.watts = g_sum[1] / g_n; leg.tempC = g_sum[2] / g_n; leg.mem = g_sum[3] / g_n;
  }
  return leg;
}

static double median(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const size_t h = v.size() / 2;
  return v.size() % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
}

// Student's t, two-sided 95%, for n - 1 degrees of freedom.
static double t95(size_t df) {
  static const double t[] = {0, 12.71, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                             2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086};
  return df < sizeof t / sizeof t[0] ? t[df] : (df < 40 ? 2.04 : 1.98);
}

// One argument, name=lib.so or name=lib.so:K, into b: its name, its library, and the
// mainnet profile at its k. K must be a k the protocol allows at the mainnet rank, a power
// of two of at least 16 x rank (2048), as hashrate.js's PEARL_PROFILE_K. Without :K, the
// mainnet profile as it is.
static void parse_build(const char *arg, Build &b) {
  const char *eq = strchr(arg, '=');
  if (!eq || eq == arg) fail(std::string("not name=lib.so[:K]: ") + arg);
  b.name.assign(arg, eq - arg);
  b.path = eq + 1;
  b.prof = PEARL_MAINNET_PROFILE;
  const size_t colon = b.path.rfind(':');
  if (colon != std::string::npos) {
    const std::string ks = b.path.substr(colon + 1);
    const bool digits = !ks.empty() && ks.size() <= 9 && ks.find_first_not_of("0123456789") == std::string::npos;
    const unsigned long k = digits ? strtoul(ks.c_str(), nullptr, 10) : 0;
    if (k < 16ul * b.prof.rank || (k & (k - 1)))
      fail(b.name + ": k must be a power of two of at least " + std::to_string(16 * b.prof.rank) + ", not " + ks);
    b.prof.k = (uint32_t)k;
    b.path.erase(colon);
  }
  if (b.path.empty()) fail(std::string("not name=lib.so[:K]: ") + arg);
  b.daf = PEARL_DAF(b.prof);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: race <seconds> <leg seconds> <warm seconds> base=lib.so name=lib.so[:K] ...\n");
    return 2;
  }
  const double total = atof(argv[1]), legSecs = atof(argv[2]), warmSecs = atof(argv[3]);
  if (!(total > 0) || !(legSecs > 0) || !(warmSecs >= 0)) fail("bad seconds");
  std::vector<Build> builds;
  for (int i = 4; i < argc; i++) {
    Build b;
    parse_build(argv[i], b);
    builds.push_back(b);
  }
  // A salt's regions: every build's m and n are the mainnet profile's, whatever its k.
  const PearlProfile prof = PEARL_MAINNET_PROFILE;
  const uint64_t span = (uint64_t)(prof.m / PEARL_ROWS_COUNT) * (prof.n / PEARL_COLS_COUNT);
  const uint64_t stride = builds.size();
  uint8_t header[PEARL_HEADER_BYTES], target[PEARL_HASH_BYTES];
  memset(header, 0xA5, sizeof header);
  memset(target, 0x00, sizeof target);   // the hardest target: no hits, as hashrate.js

  for (size_t i = 0; i < builds.size(); i++) {
    Build &b = builds[i];
    b.lib = dlopen(b.path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!b.lib) fail(b.name + ": dlopen: " + dlerror());
    for (size_t j = 0; j < i; j++)
      if (builds[j].lib == b.lib) fail(b.name + ": the same library as " + builds[j].name + "; copy the file");
    sym(b, b.create, "pearl_host_create");
    sym(b, b.setJob, "pearl_host_set_job_salted");
    sym(b, b.reseed, "pearl_host_reseed");
    sym(b, b.submit, "pearl_host_submit");
    sym(b, b.collect, "pearl_host_collect");
    sym(b, b.nextHit, "pearl_host_next_hit");
    sym(b, b.foldName, "pearl_host_fold_name");
    sym(b, b.audit, "pearl_host_l2_audit", false);
    sym(b, b.enter, "pearl_host_race_enter", false);
    char err[512] = {0};
    b.ctx = b.create(&b.prof, err, sizeof err);
    if (!b.ctx) fail(b.name + ": create: " + err);
    b.salt = i;
    b.setJob(b.ctx, header, target, b.salt);
  }
  const bool nvml = nvml_open();
  std::thread smp(sampler);
  printf("START builds=%zu seconds=%.0f leg=%.1f warm=%.1f span=%llu nvml=%s\n", builds.size(), total,
         legSecs, warmSecs, (unsigned long long)span, nvml ? "yes" : "no");
  for (Build &b : builds) {
    char a[256] = "-";
    if (b.audit) b.audit(b.ctx, a, sizeof a);
    printf("AUDIT %s fold=[%s] %s k=%u%s\n", b.name.c_str(), b.foldName(b.ctx), a, b.prof.k,
           b.enter ? "" : " no race_enter: its L2 slice is not set again before its turns");
  }
  fflush(stdout);

  // One untimed turn each: the first launches load the modules and settle the clocks.
  for (Build &b : builds) run_leg(b, -1, 1.0, 0.5, span, stride);
  const Clock::time_point start = Clock::now();
  int round = 0;
  while (round < 2 || secs(start, Clock::now()) < total) {
    std::vector<size_t> order(builds.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = round % 2 ? order.size() - 1 - i : i;
    for (size_t i : order) {
      Build &b = builds[i];
      const Leg l = run_leg(b, round, legSecs, warmSecs, span, stride);
      b.legs.push_back(l);
      printf("LEG round=%d %s th=%.3f salts=%llu secs=%.2f mhz=%.0f w=%.1f c=%.0f mem=%.1f\n", round,
             b.name.c_str(), l.th, (unsigned long long)l.salts, l.secs, l.mhz, l.watts, l.tempC, l.mem);
      fflush(stdout);
    }
    round++;
  }
  g_quit = true;
  smp.join();

  const Build &ref = builds[0];
  for (const Build &b : builds) {
    std::vector<double> th, mhz, w, mem, ratio;
    for (const Leg &l : b.legs) {
      th.push_back(l.th); mhz.push_back(l.mhz); w.push_back(l.watts); mem.push_back(l.mem);
    }
    size_t wins = 0;
    for (const Leg &l : b.legs)
      for (const Leg &r : ref.legs)
        if (r.round == l.round) {
          ratio.push_back(l.th / r.th);
          if (l.th > r.th) wins++;
        }
    double mean = 0, sd = 0;
    for (double x : ratio) mean += x;
    mean /= ratio.size();
    for (double x : ratio) sd += (x - mean) * (x - mean);
    sd = ratio.size() > 1 ? sqrt(sd / (ratio.size() - 1)) : 0;
    const double ci = ratio.size() > 1 ? t95(ratio.size() - 1) * sd / sqrt((double)ratio.size()) : 0;
    if (&b == &ref)
      printf("RACE %s reference rounds=%zu th=%.3f mhz=%.0f w=%.1f mem=%.1f k=%u\n", b.name.c_str(),
             b.legs.size(), median(th), median(mhz), median(w), median(mem), b.prof.k);
    else
      printf("RACE %s vs %s rounds=%zu d=%+.3f%% sd=%.3f%% ci95=%.3f%% ahead=%zu/%zu th=%.3f mhz=%.0f "
             "w=%.1f mem=%.1f k=%u\n",
             b.name.c_str(), ref.name.c_str(), ratio.size(), (mean - 1) * 100, sd * 100, ci * 100, wins,
             ratio.size(), median(th), median(mhz), median(w), median(mem), b.prof.k);
  }
  fflush(stdout);
  for (Build &b : builds) {
    typedef void (*DestroyFn)(void *);
    DestroyFn destroy = (DestroyFn)dlsym(b.lib, "pearl_host_destroy");
    if (destroy) destroy(b.ctx);
  }
  return 0;
}
