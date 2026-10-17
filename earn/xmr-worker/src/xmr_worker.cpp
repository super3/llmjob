// llmjob-xmr-worker: LLMJob Earn's Monero CPU miner.
//
// It hashes RandomX (tevador/RandomX, BSD-3-Clause) on the CPU and nothing
// else. It knows nothing about pools or wallets. The app runs it as a child
// process and talks to it over stdin and stdout, one JSON object per line: the
// app sends jobs, and the worker sends back the shares that meet a job's
// target, its hashrate, and anything that went wrong. A crash here, or an
// antivirus quarantine of this file, stops the CPU miner and nothing else.
//
// Commands (app -> worker):
//   {"cmd":"job","id":"<job id>","blob":"<hex>","target":"<16 hex>",
//    "seed":"<64 hex>","variant":"v1","commit":false,"nonce":<start>}
//       target is the 64-bit share target as a big-endian hex number. A hash
//       is a share when its last 8 bytes, read little-endian, are below it.
//       variant "v2" turns on RandomX v2, and commit compares the RandomX
//       commitment against the target instead of the hash. Monero's v17 fork
//       will need both; until then the app sends v1 and no commit.
//       nonce is where the search starts: thread t hashes nonce+t, then every
//       Nth nonce after it (N threads). The same job from the same start finds
//       the same shares, so the app sends a random start with every job line.
//   {"cmd":"pause"}  {"cmd":"resume"}  {"cmd":"quit"}   (EOF on stdin = quit)
//       resume starts the current job over from its start nonce. A job line
//       sent while paused replaces it, and is what the app sends first.
//
// Events (worker -> app):
//   {"type":"hello", ...}      what it is and what it will run with
//   {"type":"selftest", ...}   the known-answer check at start
//   {"type":"dataset", ...}    building / ready, when the seed changes
//   {"type":"share", ...}      a hash under the target, already checked
//   {"type":"hashrate", ...}   every 2 s while mining
//   {"type":"log", ...}  {"type":"error", ...}  {"type":"bench", ...}

#include "randomx.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif
#endif

#ifndef XMR_WORKER_VERSION
#define XMR_WORKER_VERSION "1.0.0"
#endif
#ifndef RANDOMX_COMMIT
#define RANDOMX_COMMIT "unknown"
#endif

// The protocol version the app checks in "hello". Bump it when a command or an
// event changes shape.
static const int kProtocol = 1;

// ── Output ──────────────────────────────────────────────────────────────────

static std::mutex gOutMutex;

static void emit(const std::string& line) {
  std::lock_guard<std::mutex> lk(gOutMutex);
  fwrite(line.data(), 1, line.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

static std::string jsonString(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

static void emitLog(const char* level, const std::string& msg) {
  emit(std::string("{\"type\":\"log\",\"level\":\"") + level + "\",\"msg\":" + jsonString(msg) + "}");
}

static void emitError(const std::string& msg, bool fatal) {
  emit(std::string("{\"type\":\"error\",\"fatal\":") + (fatal ? "true" : "false")
       + ",\"msg\":" + jsonString(msg) + "}");
}

// ── Input: flat JSON objects ────────────────────────────────────────────────
//
// The app is the only writer, and it only ever sends one flat object per line
// with string, number, boolean or null values. This parses exactly that and
// rejects anything else.

struct JVal {
  enum Kind { STR, NUM, BOOL, NUL } kind = NUL;
  std::string s;
  double n = 0;
  bool b = false;
};

static void skipSpace(const std::string& in, size_t& i) {
  while (i < in.size() && (in[i] == ' ' || in[i] == '\t' || in[i] == '\r' || in[i] == '\n')) i++;
}

static bool parseString(const std::string& in, size_t& i, std::string& out) {
  if (i >= in.size() || in[i] != '"') return false;
  i++;
  while (i < in.size()) {
    char c = in[i++];
    if (c == '"') return true;
    if (c != '\\') { out += c; continue; }
    if (i >= in.size()) return false;
    char e = in[i++];
    switch (e) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        if (i + 4 > in.size()) return false;
        unsigned v = 0;
        for (int k = 0; k < 4; k++) {
          char h = in[i++];
          v <<= 4;
          if (h >= '0' && h <= '9') v |= h - '0';
          else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
          else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
          else return false;
        }
        out += v < 0x80 ? static_cast<char>(v) : '?';
        break;
      }
      default: return false;
    }
  }
  return false;
}

static bool parseFlatObject(const std::string& in, std::map<std::string, JVal>& out) {
  size_t i = 0;
  skipSpace(in, i);
  if (i >= in.size() || in[i] != '{') return false;
  i++;
  skipSpace(in, i);
  if (i < in.size() && in[i] == '}') return true;
  while (i < in.size()) {
    std::string key;
    skipSpace(in, i);
    if (!parseString(in, i, key)) return false;
    skipSpace(in, i);
    if (i >= in.size() || in[i] != ':') return false;
    i++;
    skipSpace(in, i);
    JVal v;
    if (i >= in.size()) return false;
    if (in[i] == '"') {
      v.kind = JVal::STR;
      if (!parseString(in, i, v.s)) return false;
    } else if (in.compare(i, 4, "true") == 0) {
      v.kind = JVal::BOOL; v.b = true; i += 4;
    } else if (in.compare(i, 5, "false") == 0) {
      v.kind = JVal::BOOL; v.b = false; i += 5;
    } else if (in.compare(i, 4, "null") == 0) {
      v.kind = JVal::NUL; i += 4;
    } else {
      size_t start = i;
      while (i < in.size() && (isdigit(static_cast<unsigned char>(in[i])) || in[i] == '-' || in[i] == '+'
                               || in[i] == '.' || in[i] == 'e' || in[i] == 'E')) i++;
      if (i == start) return false;
      v.kind = JVal::NUM;
      v.n = atof(in.substr(start, i - start).c_str());
    }
    out[key] = v;
    skipSpace(in, i);
    if (i < in.size() && in[i] == ',') { i++; continue; }
    if (i < in.size() && in[i] == '}') return true;
    return false;
  }
  return false;
}

static int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool fromHex(const std::string& hex, std::vector<uint8_t>& out) {
  if (hex.size() % 2) return false;
  out.resize(hex.size() / 2);
  for (size_t i = 0; i < out.size(); i++) {
    int hi = hexDigit(hex[2 * i]), lo = hexDigit(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<uint8_t>(hi << 4 | lo);
  }
  return true;
}

static std::string toHex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(n * 2, '0');
  for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
  return s;
}

// A big-endian hex number of up to 16 digits, as the app writes the target.
static bool parseHex64(const std::string& hex, uint64_t& out) {
  if (hex.empty() || hex.size() > 16) return false;
  uint64_t v = 0;
  for (char c : hex) {
    int d = hexDigit(c);
    if (d < 0) return false;
    v = v << 4 | static_cast<uint64_t>(d);
  }
  out = v;
  return true;
}

static uint64_t readLe64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = v << 8 | p[i];
  return v;
}

static void writeLe32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v); p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16); p[3] = static_cast<uint8_t>(v >> 24);
}

// ── Priority ────────────────────────────────────────────────────────────────
//
// The CPU miner must never slow the GPU miner, whose share proofs and CUDA host
// threads run on the CPU too, or the desktop. The app keeps a CPU free for the
// GPU miner and picks the priority (src/shared/xmr/worker.js defaultPriority):
//   idle    Linux SCHED_IDLE, which any normal thread preempts; Windows the
//           idle priority class (the default when run by hand)
//   low     nice 19; Windows below-normal
//   normal  unchanged

enum class Priority { Idle, Low, Normal };
static Priority gPriority = Priority::Idle;

static void applyProcessPriority() {
#ifdef _WIN32
  DWORD cls = gPriority == Priority::Idle ? IDLE_PRIORITY_CLASS
            : gPriority == Priority::Low ? BELOW_NORMAL_PRIORITY_CLASS : NORMAL_PRIORITY_CLASS;
  SetPriorityClass(GetCurrentProcess(), cls);
#endif
}

static void applyThreadPriority() {
#ifdef _WIN32
  if (gPriority == Priority::Idle) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
  else if (gPriority == Priority::Low) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
  if (gPriority == Priority::Normal) return;
  if (gPriority == Priority::Idle) {
    sched_param p{};
    p.sched_priority = 0;
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &p) == 0) return;
  }
  // Low, or SCHED_IDLE refused: the lowest nice value for this thread.
  setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 19);
#else
  if (gPriority != Priority::Normal) setpriority(PRIO_PROCESS, 0, gPriority == Priority::Idle ? 19 : 10);
#endif
}

// The CPUs this process may run on: its affinity mask, not the machine's count.
static unsigned usableCpus() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) == 0) {
    int n = CPU_COUNT(&set);
    if (n > 0) return static_cast<unsigned>(n);
  }
#elif defined(_WIN32)
  DWORD active = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  if (active > 0) return active;
#endif
  unsigned n = std::thread::hardware_concurrency();
  return n ? n : 1;
}

// ── RandomX state ───────────────────────────────────────────────────────────

static randomx_flags gFlags = RANDOMX_FLAG_DEFAULT;   // VM flags, without FULL_MEM or V2
static bool gTryLargePages = true;
// Scratchpads in large pages worked. Atomic: every mining thread creates a VM.
static std::atomic<bool> gLargeVm{false};
static randomx_cache* gCache = nullptr;
static randomx_dataset* gDataset = nullptr;
static const char* gPages = "normal";                 // how the dataset is backed
#ifdef __linux__
static bool gThpAsked = false;                        // madvise'd for THP; checked once written
#endif
static unsigned gThreads = 1;
static unsigned gInitThreads = 1;

// Light VMs over the cache, one per variant, for the self-test and for checking
// every share before it goes out. Used under gVerifyMutex.
static std::mutex gVerifyMutex;
static randomx_vm* gLightVm[2] = {nullptr, nullptr};

static randomx_flags cacheFlags() {
  return static_cast<randomx_flags>(gFlags & (RANDOMX_FLAG_JIT | RANDOMX_FLAG_ARGON2));
}

static randomx_vm* lightVm(bool v2) {
  randomx_vm*& vm = gLightVm[v2 ? 1 : 0];
  if (!vm) {
    randomx_flags f = static_cast<randomx_flags>(gFlags | (v2 ? RANDOMX_FLAG_V2 : 0));
    vm = randomx_create_vm(f, gCache, nullptr);
  }
  return vm;
}

static void resetLightVms() {
  for (randomx_vm* vm : gLightVm) if (vm) randomx_vm_set_cache(vm, gCache);
}

static bool allocCache() {
  if (gTryLargePages) gCache = randomx_alloc_cache(static_cast<randomx_flags>(cacheFlags() | RANDOMX_FLAG_LARGE_PAGES));
  if (!gCache) gCache = randomx_alloc_cache(cacheFlags());
  return gCache != nullptr;
}

static bool allocDataset() {
  if (gTryLargePages) {
    gDataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
    if (gDataset) { gPages = "hugetlb"; return true; }
  }
  gDataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
  if (!gDataset) return false;
#ifdef __linux__
  // No huge pages reserved: ask for transparent huge pages instead, before the
  // dataset is first written. Most desktops allow it (the "madvise" setting).
  if (gTryLargePages) {
    const size_t page = 2 * 1024 * 1024;
    uintptr_t base = reinterpret_cast<uintptr_t>(randomx_get_dataset_memory(gDataset));
    size_t size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
    uintptr_t start = (base + page - 1) & ~(page - 1);
    uintptr_t end = (base + size) & ~(page - 1);
    if (end > start && madvise(reinterpret_cast<void*>(start), end - start, MADV_HUGEPAGE) == 0) {
      gPages = "thp";
      gThpAsked = true;
    }
  }
#endif
  return true;
}

#ifdef __linux__
// madvise(MADV_HUGEPAGE) only marks the range, and returns 0 even when the
// kernel goes on to back it with 4 KiB pages: transparent huge pages set to
// "never", turned off for this process, or memory too fragmented. So once the
// dataset is written, count what the process got (AnonHugePages), and call it
// THP only when that covers at least half the dataset. Before Linux 4.14 there
// is no smaps_rollup, and the sysfs setting is the best guess.
static bool thpBacked(size_t datasetBytes) {
  if (FILE* f = fopen("/proc/self/smaps_rollup", "r")) {
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f)) {
      if (sscanf(line, "AnonHugePages: %lld kB", &kb) == 1) break;
    }
    fclose(f);
    if (kb >= 0) return static_cast<unsigned long long>(kb) * 1024 >= datasetBytes / 2;
  }
  char buf[128] = {0};
  if (FILE* f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    buf[n] = 0;
    fclose(f);
  }
  return strstr(buf, "[always]") != nullptr || strstr(buf, "[madvise]") != nullptr;
}
#endif

static randomx_vm* createFullVm(bool v2) {
  randomx_flags f = static_cast<randomx_flags>(gFlags | RANDOMX_FLAG_FULL_MEM | (v2 ? RANDOMX_FLAG_V2 : 0));
  if (gTryLargePages) {
    randomx_vm* vm = randomx_create_vm(static_cast<randomx_flags>(f | RANDOMX_FLAG_LARGE_PAGES), nullptr, gDataset);
    if (vm) { gLargeVm.store(true); return vm; }
  }
  return randomx_create_vm(f, nullptr, gDataset);
}

// ── Self-test ───────────────────────────────────────────────────────────────
//
// Known answers from tevador/RandomX's own tests (src/tests/tests.cpp), hashed
// in light mode with the flags mining will use: the JIT, hardware AES, the
// Argon2 build. A JIT bug or a CPU fault that gives wrong hashes shows up here,
// before any share is sent, instead of as a pool rejecting every share.

struct Vector {
  const char* key;
  const char* input;
  bool hex;      // input is hex bytes, not text
  bool v2;
  bool commit;   // the expected value is the commitment, not the hash
  const char* expect;
};

static const Vector kVectors[] = {
  {"test key 000", "This is a test", false, false, false,
   "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f"},
  {"test key 000", "Lorem ipsum dolor sit amet", false, false, false,
   "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969"},
  {"test key 000", "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua", false, false, false,
   "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8"},
  {"test key 000", "This is a test", false, true, false,
   "22ec6b861b3eb23686b2efbad69513c967ecfce80983df66c9c5b4fbfb4cdb6f"},
  {"test key 000", "This is a test", false, true, true,
   "133be717399046b03ae82ce8ddd9d1ee4d3ea7fca03a50dec09b6848cbb98e18"},
  {"test key 001", "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua", false, false, false,
   "e9ff4503201c0c2cca26d285c93ae883f9b1d30c9eb240b820756f2d5a7905fc"},
  // A 76-byte Monero-style hashing blob.
  {"test key 001",
   "0b0b98bea7e805e0010a2126d287a2a0cc833d312cb786385a7c2f9de69d25537f584a9bc9977b00000000666fd8753bf61a8631f12984e3fd44f4014eca629276817b56f32e9b68bd82f416",
   true, false, false, "c56414121acda1713c2f2a819d8ae38aed7c80c35c2a769298d34f03833cd5f1"},
};

static bool selfTest(std::string& err) {
  std::string key;
  for (const Vector& v : kVectors) {
    if (key != v.key) {
      key = v.key;
      randomx_init_cache(gCache, key.data(), key.size());
      resetLightVms();
    }
    std::vector<uint8_t> input;
    if (v.hex) fromHex(v.input, input);
    else input.assign(v.input, v.input + strlen(v.input));
    randomx_vm* vm = lightVm(v.v2);
    if (!vm) { err = "could not create a RandomX VM"; return false; }
    uint8_t hash[RANDOMX_HASH_SIZE];
    randomx_calculate_hash(vm, input.data(), input.size(), hash);
    if (v.commit) randomx_calculate_commitment(input.data(), input.size(), hash, hash);
    if (toHex(hash, sizeof hash) != v.expect) {
      err = std::string("wrong hash for \"") + (v.hex ? "a Monero blob" : v.input) + "\"" + (v.v2 ? " (v2)" : "")
          + (v.commit ? " (commitment)" : "") + ": got " + toHex(hash, sizeof hash) + ", want " + v.expect;
      return false;
    }
  }
  return true;
}

// ── Mining state ────────────────────────────────────────────────────────────

struct Job {
  std::string id;
  std::vector<uint8_t> blob;
  uint64_t target = 0;
  uint8_t seed[32] = {0};
  bool v2 = false;
  bool commit = false;
  uint32_t nonce = 0;
};

static std::mutex gMutex;
static std::condition_variable gCv;
static Job gJob;
static bool gHaveJob = false;
static bool gPaused = false;
static bool gQuitting = false;
static bool gNeedRebuild = false;
static bool gBuilding = false;             // the dataset thread is building one now
static bool gDatasetReady = false;
static uint8_t gDatasetSeed[32] = {0};
static unsigned gActive = 0;               // miner threads inside a hash loop
static uint64_t gDatasetGen = 0;           // bumps after each rebuild
// Bumps on every change a hashing thread must act on (new job, pause, rebuild,
// quit). Threads hash while it holds the value they started with.
static std::atomic<uint64_t> gEpoch{0};
static std::atomic<bool> gQuitAtomic{false};
static std::atomic<int> gExitCode{0};

struct Counter { std::atomic<uint64_t> n{0}; char pad[56]; };
static std::unique_ptr<Counter[]> gHashes;

static void bump() { gEpoch.fetch_add(1); gCv.notify_all(); }

static bool miningNow() { return gHaveJob && !gPaused && gDatasetReady && !gNeedRebuild && !gQuitting; }

// Check a hash in light mode before it is reported. A share the pool rejects
// costs a ban sooner or later, and the light VM over the cache is the RandomX
// reference for the same input, so a bad JIT or a bit flip in the dataset is
// caught here.
static bool verifyShare(const Job& job, const uint8_t* input, size_t len, const uint8_t* hash) {
  std::lock_guard<std::mutex> lk(gVerifyMutex);
  randomx_vm* vm = lightVm(job.v2);
  if (!vm) return false;
  uint8_t check[RANDOMX_HASH_SIZE];
  randomx_calculate_hash(vm, input, len, check);
  return memcmp(check, hash, RANDOMX_HASH_SIZE) == 0;
}

static void reportShare(const Job& job, const uint8_t* input, size_t len, const uint8_t* hash) {
  if (!verifyShare(job, input, len, hash)) {
    emitError("dropped a share whose hash did not match the light-mode check (job " + job.id + ")", false);
    return;
  }
  std::string line = "{\"type\":\"share\",\"job\":" + jsonString(job.id) + ",\"nonce\":\""
                   + toHex(input + 39, 4) + "\",\"result\":\"" + toHex(hash, RANDOMX_HASH_SIZE) + "\"";
  if (job.commit) {
    uint8_t com[RANDOMX_HASH_SIZE];
    randomx_calculate_commitment(input, len, hash, com);
    line += ",\"commitment\":\"" + toHex(com, RANDOMX_HASH_SIZE) + "\"";
  }
  emit(line + "}");
}

static void minerThread(unsigned tid) {
  applyThreadPriority();
  randomx_vm* vm = nullptr;
  bool vmV2 = false;
  uint64_t vmDatasetGen = 0;
  uint64_t exhaustedEpoch = UINT64_MAX;

  for (;;) {
    Job job;
    uint64_t epoch;
    {
      std::unique_lock<std::mutex> lk(gMutex);
      gCv.wait(lk, [&] { return gQuitting || (miningNow() && gEpoch.load() != exhaustedEpoch); });
      if (gQuitting) break;
      job = gJob;
      epoch = gEpoch.load();
      gActive++;
      if (vm && (vmV2 != job.v2)) { randomx_destroy_vm(vm); vm = nullptr; }
      if (vm && vmDatasetGen != gDatasetGen) randomx_vm_set_dataset(vm, gDataset);
      vmDatasetGen = gDatasetGen;
    }
    if (!vm) {
      vm = createFullVm(job.v2);
      vmV2 = job.v2;
    }
    if (!vm) {
      emitError("could not create a RandomX VM for thread " + std::to_string(tid), true);
      std::lock_guard<std::mutex> lk(gMutex);
      gExitCode.store(4);
      gActive--;
      gCv.notify_all();
      break;
    }

    std::vector<uint8_t> blob = job.blob;
    std::vector<uint8_t> prev = blob;
    const size_t len = blob.size();
    uint8_t hash[RANDOMX_HASH_SIZE];
    uint8_t com[RANDOMX_HASH_SIZE];
    // Thread `tid` takes every gThreads-th nonce from the job's start. The
    // search ends after one pass over the 32-bit nonce, which takes days.
    uint32_t nonce = job.nonce + tid;
    const uint64_t passes = (0x100000000ULL - tid + gThreads - 1) / gThreads;
    uint64_t done = 0;
    Counter& counter = gHashes[tid];

    writeLe32(blob.data() + 39, nonce);
    randomx_calculate_hash_first(vm, blob.data(), len);
    bool exhausted = false;
    while (gEpoch.load(std::memory_order_relaxed) == epoch) {
      memcpy(prev.data() + 39, blob.data() + 39, 4);
      if (++done >= passes) { exhausted = true; break; }
      nonce += gThreads;
      writeLe32(blob.data() + 39, nonce);
      randomx_calculate_hash_next(vm, blob.data(), len, hash);   // the hash of `prev`
      counter.n.fetch_add(1, std::memory_order_relaxed);
      const uint8_t* check = hash;
      if (job.commit) {
        randomx_calculate_commitment(prev.data(), len, hash, com);
        check = com;
      }
      if (readLe64(check + 24) < job.target) reportShare(job, prev.data(), len, hash);
    }
    randomx_calculate_hash_last(vm, hash);
    if (exhausted) {
      emitLog("info", "searched every nonce of job " + job.id + " on thread " + std::to_string(tid));
    }

    std::lock_guard<std::mutex> lk(gMutex);
    if (exhausted) exhaustedEpoch = epoch;
    gActive--;
    gCv.notify_all();
  }
  if (vm) randomx_destroy_vm(vm);
}

// Fill the dataset from the cache on gInitThreads threads, in small chunks so a
// quit does not have to wait for the whole 2 GiB.
static bool initDataset() {
  const unsigned long items = randomx_dataset_item_count();
  const unsigned long chunk = 1UL << 14;
  std::atomic<unsigned long> next{0};
  std::vector<std::thread> ts;
  for (unsigned i = 0; i < gInitThreads; i++) {
    ts.emplace_back([&] {
      applyThreadPriority();
      for (;;) {
        if (gQuitAtomic.load()) return;
        unsigned long start = next.fetch_add(chunk);
        if (start >= items) return;
        randomx_init_dataset(gDataset, gCache, start, std::min(chunk, items - start));
      }
    });
  }
  for (auto& t : ts) t.join();
  return !gQuitAtomic.load();
}

// After a rebuild, hash one input with a full-mode VM and with the light VM and
// compare. A dataset the init got wrong gives every share a wrong hash.
static bool checkDataset(bool v2) {
  randomx_vm* vm = createFullVm(v2);
  if (!vm) return false;
  static const char input[] = "llmjob dataset check";
  uint8_t full[RANDOMX_HASH_SIZE], light[RANDOMX_HASH_SIZE];
  randomx_calculate_hash(vm, input, sizeof input - 1, full);
  randomx_destroy_vm(vm);
  std::lock_guard<std::mutex> lk(gVerifyMutex);
  randomx_vm* lvm = lightVm(v2);
  if (!lvm) return false;
  randomx_calculate_hash(lvm, input, sizeof input - 1, light);
  return memcmp(full, light, sizeof full) == 0;
}

static void datasetThread() {
  // It builds the cache and checks each dataset itself. On Linux priority is
  // per thread, so without this that work ran at normal priority.
  applyThreadPriority();
  for (;;) {
    uint8_t seed[32];
    bool v2;
    {
      std::unique_lock<std::mutex> lk(gMutex);
      gCv.wait(lk, [] { return gQuitting || gNeedRebuild; });
      if (gQuitting) return;
      gDatasetReady = false;
      bump();
      gCv.wait(lk, [] { return gActive == 0 || gQuitting; });
      if (gQuitting) return;
      memcpy(seed, gJob.seed, 32);
      v2 = gJob.v2;
      gNeedRebuild = false;
      gBuilding = true;
    }
    emit("{\"type\":\"dataset\",\"state\":\"building\",\"seed\":\"" + toHex(seed, 32) + "\",\"threads\":"
         + std::to_string(gInitThreads) + ",\"pages\":\"" + gPages + "\"}");
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> vlk(gVerifyMutex);
      randomx_init_cache(gCache, seed, 32);
      resetLightVms();
    }
    if (!initDataset()) return;
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    // "building" says what was asked for; "ready" says what the dataset got.
    const char* pages = gPages;
#ifdef __linux__
    if (gThpAsked) pages = thpBacked(randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE) ? "thp" : "normal";
#endif
    if (!checkDataset(v2)) {
      emitError("the RandomX dataset failed its check after it was built; not mining with it", true);
      std::lock_guard<std::mutex> lk(gMutex);
      gExitCode.store(5);
      gQuitting = true;
      gQuitAtomic.store(true);
      bump();
      return;
    }
    {
      std::lock_guard<std::mutex> lk(gMutex);
      gPages = pages;
      memcpy(gDatasetSeed, seed, 32);
      gDatasetGen++;
      gBuilding = false;
      gDatasetReady = true;
      // A job with yet another seed may have arrived while this one was built.
      if (memcmp(gJob.seed, gDatasetSeed, 32) != 0) gNeedRebuild = true;
      bump();
    }
    emit("{\"type\":\"dataset\",\"state\":\"ready\",\"seed\":\"" + toHex(seed, 32) + "\",\"ms\":"
         + std::to_string(ms) + ",\"pages\":\"" + gPages + "\",\"vmLargePages\":" + (gLargeVm.load() ? "true" : "false") + "}");
  }
}

static void hashrateThread() {
  using clock = std::chrono::steady_clock;
  auto last = clock::now();
  uint64_t lastTotal = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(gMutex);
      if (gCv.wait_for(lk, std::chrono::seconds(2), [] { return gQuitting; })) return;
    }
    uint64_t total = 0;
    for (unsigned i = 0; i < gThreads; i++) total += gHashes[i].n.load(std::memory_order_relaxed);
    auto now = clock::now();
    double secs = std::chrono::duration<double>(now - last).count();
    double hs = secs > 0 ? (total - lastTotal) / secs : 0;
    last = now;
    lastTotal = total;
    char buf[160];
    snprintf(buf, sizeof buf, "{\"type\":\"hashrate\",\"hs\":%.1f,\"total\":%llu,\"threads\":%u}", hs,
             static_cast<unsigned long long>(total), gThreads);
    emit(buf);
  }
}

// ── Commands ────────────────────────────────────────────────────────────────

static std::string str(const std::map<std::string, JVal>& m, const char* k) {
  auto it = m.find(k);
  return it != m.end() && it->second.kind == JVal::STR ? it->second.s : std::string();
}

static bool flag(const std::map<std::string, JVal>& m, const char* k) {
  auto it = m.find(k);
  return it != m.end() && it->second.kind == JVal::BOOL && it->second.b;
}

// A job whose seed the dataset was not built from needs a rebuild. While one
// is being built, the dataset thread checks the latest job's seed when it is
// done, so a job arriving mid-build does not ask again.
static void setJob(Job&& job) {
  std::lock_guard<std::mutex> lk(gMutex);
  gJob = std::move(job);
  gHaveJob = true;
  if (!gBuilding && (!gDatasetReady || memcmp(gJob.seed, gDatasetSeed, 32) != 0)) gNeedRebuild = true;
  bump();
}

static void handleJob(const std::map<std::string, JVal>& m) {
  Job job;
  job.id = str(m, "id");
  std::vector<uint8_t> seed;
  if (job.id.empty()) { emitError("job without an id", false); return; }
  if (!fromHex(str(m, "blob"), job.blob) || job.blob.size() < 43 || job.blob.size() > 512) {
    emitError("job " + job.id + ": the blob is not 43-512 bytes of hex", false);
    return;
  }
  if (!parseHex64(str(m, "target"), job.target) || job.target == 0) {
    emitError("job " + job.id + ": the target is not a 64-bit hex number", false);
    return;
  }
  if (!fromHex(str(m, "seed"), seed) || seed.size() != 32) {
    emitError("job " + job.id + ": the seed is not 32 bytes of hex", false);
    return;
  }
  memcpy(job.seed, seed.data(), 32);
  std::string variant = str(m, "variant");
  if (variant.empty() || variant == "v1") job.v2 = false;
  else if (variant == "v2") job.v2 = true;
  else { emitError("job " + job.id + ": unknown RandomX variant " + variant, false); return; }
  job.commit = flag(m, "commit");
  auto it = m.find("nonce");
  if (it != m.end() && it->second.kind == JVal::NUM && it->second.n >= 0 && it->second.n < 4294967296.0) {
    job.nonce = static_cast<uint32_t>(it->second.n);
  }
  setJob(std::move(job));
}

static void quit() {
  std::lock_guard<std::mutex> lk(gMutex);
  gQuitting = true;
  gQuitAtomic.store(true);
  bump();
}

static void usage() {
  fprintf(stderr,
          "llmjob-xmr-worker %s: LLMJob Earn's RandomX CPU miner (talks JSON lines on stdin/stdout)\n"
          "  --threads N        mining threads (default: usable CPUs - 1)\n"
          "  --init-threads N   threads that build the dataset (default: the mining threads)\n"
          "  --priority P       idle (default), low or normal\n"
          "  --no-huge-pages    do not try huge pages\n"
          "  --no-jit           use the interpreter (slow; for diagnosis)\n"
          "  --self-test        run the known-answer check and exit\n"
          "  --bench SECONDS    mine a fixed job with no pool for SECONDS and print the hashrate\n"
          "  --version          print the version and exit\n",
          XMR_WORKER_VERSION);
}

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stdin), _O_BINARY);
  // Ctrl+C in the CLI's console reaches this process too. The app stops it
  // with "quit" or by closing stdin, after it has said so in its own log.
  SetConsoleCtrlHandler(nullptr, TRUE);
#else
  signal(SIGINT, SIG_IGN);
  signal(SIGPIPE, SIG_IGN);
#ifdef __linux__
  // If the app dies without closing our stdin, go with it.
  prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
#endif

  unsigned cpus = usableCpus();
  gThreads = cpus > 1 ? cpus - 1 : 1;
  unsigned initThreads = 0;
  bool selfTestOnly = false, noJit = false;
  int benchSeconds = 0;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
      return argv[++i];
    };
    if (a == "--threads") gThreads = static_cast<unsigned>(std::max(1, atoi(next("--threads").c_str())));
    else if (a == "--init-threads") initThreads = static_cast<unsigned>(std::max(1, atoi(next("--init-threads").c_str())));
    else if (a == "--priority") {
      std::string p = next("--priority");
      if (p == "idle") gPriority = Priority::Idle;
      else if (p == "low") gPriority = Priority::Low;
      else if (p == "normal") gPriority = Priority::Normal;
      else { fprintf(stderr, "unknown priority %s\n", p.c_str()); return 2; }
    }
    else if (a == "--no-huge-pages") gTryLargePages = false;
    else if (a == "--no-jit") noJit = true;
    else if (a == "--self-test") selfTestOnly = true;
    else if (a == "--bench") benchSeconds = std::max(1, atoi(next("--bench").c_str()));
    else if (a == "--version") { printf("%s\n", XMR_WORKER_VERSION); return 0; }
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
  }
  gInitThreads = initThreads ? initThreads : gThreads;
  applyProcessPriority();

  gFlags = randomx_get_flags();
  if (noJit) gFlags = static_cast<randomx_flags>(gFlags & ~(RANDOMX_FLAG_JIT | RANDOMX_FLAG_SECURE));
  const char* argon = (gFlags & RANDOMX_FLAG_ARGON2_AVX2) ? "avx2" : (gFlags & RANDOMX_FLAG_ARGON2_SSSE3) ? "ssse3" : "ref";
  emit(std::string("{\"type\":\"hello\",\"protocol\":") + std::to_string(kProtocol) + ",\"version\":\""
       + XMR_WORKER_VERSION + "\",\"randomx\":\"" + RANDOMX_COMMIT + "\",\"cpus\":" + std::to_string(cpus)
       + ",\"threads\":" + std::to_string(gThreads) + ",\"jit\":" + ((gFlags & RANDOMX_FLAG_JIT) ? "true" : "false")
       + ",\"hardAes\":" + ((gFlags & RANDOMX_FLAG_HARD_AES) ? "true" : "false") + ",\"argon2\":\"" + argon + "\"}");

  if (!allocCache()) {
    emitError("could not allocate the 256 MiB RandomX cache", true);
    return 4;
  }
  {
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    bool ok = selfTest(err);
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    emit(std::string("{\"type\":\"selftest\",\"ok\":") + (ok ? "true" : "false") + ",\"ms\":" + std::to_string(ms)
         + (ok ? "" : ",\"msg\":" + jsonString(err)) + "}");
    if (!ok) {
      emitError("RandomX self-test failed: " + err, true);
      return 3;
    }
  }
  if (selfTestOnly) return 0;

  if (!allocDataset()) {
    emitError("could not allocate the 2080 MiB RandomX dataset: not enough free memory", true);
    return 4;
  }
  gHashes.reset(new Counter[gThreads]);

  std::vector<std::thread> threads;
  threads.emplace_back(datasetThread);
  threads.emplace_back(hashrateThread);
  for (unsigned i = 0; i < gThreads; i++) threads.emplace_back(minerThread, i);

  if (benchSeconds) {
    // A fixed job nobody will ever submit: a zero seed, a 76-byte zero blob and
    // a target nothing meets. Times the mining only, not the dataset build.
    Job job;
    job.id = "bench";
    job.blob.assign(76, 0);
    job.target = 1;
    setJob(std::move(job));
    {
      std::unique_lock<std::mutex> lk(gMutex);
      gCv.wait(lk, [] { return gDatasetReady || gQuitting; });
    }
    uint64_t before = 0, after = 0;
    for (unsigned i = 0; i < gThreads; i++) before += gHashes[i].n.load();
    auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(benchSeconds));
    for (unsigned i = 0; i < gThreads; i++) after += gHashes[i].n.load();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char buf[200];
    snprintf(buf, sizeof buf, "{\"type\":\"bench\",\"hs\":%.1f,\"seconds\":%.1f,\"threads\":%u,\"pages\":\"%s\"}",
             (after - before) / secs, secs, gThreads, gPages);
    emit(buf);
    quit();
  } else {
    std::string line;
    while (!gQuitAtomic.load() && std::getline(std::cin, line)) {
      if (line.empty() || line == "\r") continue;
      std::map<std::string, JVal> m;
      if (!parseFlatObject(line, m)) { emitError("unreadable command: " + line.substr(0, 120), false); continue; }
      std::string cmd = str(m, "cmd");
      if (cmd == "job") handleJob(m);
      else if (cmd == "pause") { std::lock_guard<std::mutex> lk(gMutex); gPaused = true; bump(); }
      else if (cmd == "resume") { std::lock_guard<std::mutex> lk(gMutex); gPaused = false; bump(); }
      else if (cmd == "quit") break;
      else emitError("unknown command: " + cmd, false);
    }
    quit();
  }

  for (auto& t : threads) t.join();
  for (randomx_vm*& vm : gLightVm) if (vm) { randomx_destroy_vm(vm); vm = nullptr; }
  if (gDataset) randomx_release_dataset(gDataset);
  if (gCache) randomx_release_cache(gCache);
  return gExitCode.load();
}
