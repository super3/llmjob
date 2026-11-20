// The Intel (SYCL) Quantus core: quantus_sycl.node, an N-API addon to the
// contract in earn/native/quantus/README.md (QUANTUS_API 1).
//
//   QUANTUS_API             1
//   devices()               [{ index, name }]: Intel GPUs through Level Zero (else OpenCL)
//   createCore({ device })  a core on that card
//   core.device             { index, name, vendor: 'intel' }
//   core.scan(work)         Promise<{ hits, overflow }>, run on a worker thread
//   core.close()            frees the card
//
// The kernel is qtc::nonce_top (qtc_poseidon2.h) on one counter a work-item:
// two Poseidon2 permutations a nonce, the first linear layer folded into a
// prestate on the host. All arithmetic is exact, so the hits are exactly the
// reference's. A scan runs as launches of about QUANTUS_SYCL_LAUNCH_MS each,
// so a long scan (verify-core's 2^30) never holds the card in one kernel.
//
// Settings, for tuning and tests (no release sets them):
//   QUANTUS_SYCL_DEVICE=cpu    the SYCL CPU device instead of the GPUs
//   QUANTUS_SYCL_WG=N          work-group size (default 128)
//   QUANTUS_SYCL_NPT=N         counters a work-item takes a launch (default 1: on an A750, 1 ran 88.5 MH/s, 4 85.2, 16 79.7)
//   QUANTUS_SYCL_LAUNCH_MS=N   target time of one launch (default 40)
#include <node_api.h>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "qtc_poseidon2.h"

// QTC_ASM8 (default 2): the 8-lane GPU kernels run a second copy of the field
// code, qtc8, whose fold does its correction with inline vISA (QTC_FCORR in
// qtc_poseidon2.h): the two selects run on the float pipe, 3.7% fewer int-pipe
// instructions. On an A750 (13 Nov, 2,200 MHz, 190 W): 105.0-105.1 MH/s
// against 103.6. 1: predicated moves instead of selects; 0: off. The other
// forms and the CPU device keep plain qtc: the asm is written for 8 lanes and
// only Intel's GPU compiler reads it.
#ifndef QTC_ASM8
#define QTC_ASM8 2
#endif
#if QTC_ASM8 && !QTC_FCORR
#undef QTC_NS
#define QTC_NS qtc8
#undef QTC_FCORR
#define QTC_FCORR QTC_ASM8
#include "qtc_poseidon2.h"
#undef QTC_FCORR
#define QTC_FCORR 0
#else
namespace qtc8 = qtc;
#endif

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// The GPU kernel's sub-group size and register mode (128 or 256). Other forms
// are built in too, for tuning: QUANTUS_SYCL_KERNEL=SG,GRF picks one at run
// time (8,128 8,256 16,128 16,256 32,128). On an A750 (13 Nov, all at 190 W):
// 8,128 102.7 MH/s, 16,128 92.6-94.1, 8,256 82.6, 16,256 81.9. Sub-group 8
// gives each lane 128 registers' worth of 32-bit values, twice 16's, and the
// state no longer spills.
#ifndef QTC_SG
#define QTC_SG 8
#endif
#ifndef QTC_GRF
#define QTC_GRF 128
#endif

namespace {

constexpr int API = 1;
// The most hits one scan keeps (8 bytes each on the card). A scan that finds
// more reports overflow even when maxHits is larger.
constexpr uint64_t HIT_CAP = 1ull << 16;
constexpr uint64_t MIN_LAUNCH = 1ull << 14;
constexpr uint64_t MAX_LAUNCH = 1ull << 26;
constexpr uint64_t FIRST_LAUNCH = 1ull << 21;

// What one launch needs; passed by value as the kernel's argument.
struct Params {
  uint64_t pre[12];
  uint64_t target;  // targetHigh
  uint64_t base;    // this launch's first counter
  uint32_t count;   // counters in this launch, at most MAX_LAUNCH
  uint32_t room;    // hits this launch may store
  uint32_t npt;     // counters a work-item
  uint32_t pad;
};

namespace syclex = sycl::ext::oneapi::experimental;

// The scan kernel: one counter a work-item, npt of them a launch. SG is the
// sub-group size; GRF 256 asks for the 256-register mode (half the threads an
// EU, no spills). GRF 0 is the CPU device's form, which has no such mode.
template <int SG, int GRF>
struct ScanKernel {
  Params p;
  uint64_t *hits;
  uint32_t *cnt;

  auto get(syclex::properties_tag) const {
    if constexpr (GRF == 256)
      return syclex::properties{syclex::sub_group_size<SG>, sycl::ext::intel::experimental::grf_size<256>};
    else if constexpr (GRF == 128)
      return syclex::properties{syclex::sub_group_size<SG>, sycl::ext::intel::experimental::grf_size<128>};
    else
      return syclex::properties{syclex::sub_group_size<SG>};
  }

  void operator()(sycl::nd_item<1> it) const {
    const uint32_t gid = (uint32_t)it.get_global_linear_id();
    const uint32_t stride = (uint32_t)it.get_global_range(0);
    for (uint32_t j = 0; j < p.npt; j++) {
      const uint32_t idx = gid + j * stride;
      if (idx >= p.count) break;
      const uint64_t counter = p.base + idx;
      uint64_t top;
      if constexpr (SG == 8 && GRF != 0) top = qtc8::nonce_top(p.pre, counter);
      else top = qtc::nonce_top(p.pre, counter);
      if (top <= p.target) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(*cnt);
        const uint32_t slot = a.fetch_add(1u);
        if (slot < p.room) hits[slot] = counter;
      }
    }
  }
};

// The kernel forms a build carries, as one number: SG * 1000 + GRF.
enum : int { K_CPU = 16000, K_DEFAULT = QTC_SG * 1000 + QTC_GRF };

std::string env_str(const char *name, const char *dflt) {
  const char *v = std::getenv(name);
  return v && *v ? std::string(v) : std::string(dflt);
}

long env_long(const char *name, long dflt, long lo, long hi) {
  const char *v = std::getenv(name);
  if (!v || !*v) return dflt;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (!end || *end || n < lo || n > hi) return dflt;
  return n;
}

bool want_cpu() { return env_str("QUANTUS_SYCL_DEVICE", "gpu") == "cpu"; }

// The cards this core may open, in index order: Intel GPUs through Level Zero
// if there are any (the same card is usually listed by OpenCL too), else
// through OpenCL. QUANTUS_SYCL_DEVICE=cpu: the CPU device, for tests.
std::vector<sycl::device> list_devices() {
  std::vector<sycl::device> l0, ocl, cpu;
  try {
    for (const auto &p : sycl::platform::get_platforms()) {
      const sycl::backend be = p.get_backend();
      for (const auto &d : p.get_devices()) {
        if (d.is_cpu()) {
          cpu.push_back(d);
          continue;
        }
        if (!d.is_gpu() || d.get_info<sycl::info::device::vendor_id>() != 0x8086) continue;
        if (be == sycl::backend::ext_oneapi_level_zero) l0.push_back(d);
        else if (be == sycl::backend::opencl) ocl.push_back(d);
      }
    }
  } catch (const std::exception &) {
  }
  if (want_cpu()) return cpu;
  return !l0.empty() ? l0 : ocl;
}

struct Core {
  int index = 0;
  std::string name;
  sycl::queue *q = nullptr;
  uint64_t *hits = nullptr;
  uint64_t hitCap = 0;
  uint32_t *cnt = nullptr;
  std::atomic<bool> closed{false};
  bool busy = false;       // a scan is in flight (main thread only)
  bool collected = false;  // the JS object is gone (main thread only)
  bool built = false;      // the kernel is compiled for this card (worker thread)
  uint64_t chunk = FIRST_LAUNCH;
  uint32_t wg = 128;
  uint32_t npt = 1;
  int form = K_DEFAULT;
  double launchSecs = 0.04;
  std::mutex asyncMu;
  std::string asyncError;

  void release() {
    if (!q) return;
    try {
      q->wait();
    } catch (...) {
    }
    if (hits) sycl::free(hits, *q);
    if (cnt) sycl::free(cnt, *q);
    hits = nullptr;
    cnt = nullptr;
    delete q;
    q = nullptr;
  }
};

template <int SG, int GRF>
void launch_form(sycl::queue &q, const Params &p, uint64_t *hits, uint32_t *cnt, uint32_t wg) {
  const uint64_t items = (p.count + p.npt - 1) / p.npt;
  const uint64_t global = (items + wg - 1) / wg * wg;
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), ScanKernel<SG, GRF>{p, hits, cnt});
}

template <int SG, int GRF>
sycl::kernel_id form_id() {
  return sycl::get_kernel_id<ScanKernel<SG, GRF>>();
}

// Every form this build carries, by the number K_* names. The CPU device takes
// K_CPU only.
#define QTC_FORMS(X) X(16, 0) X(8, 128) X(8, 256) X(16, 128) X(16, 256) X(32, 128)

void launch(int form, sycl::queue &q, const Params &p, uint64_t *hits, uint32_t *cnt, uint32_t wg) {
  switch (form) {
#define QTC_CASE(SG, GRF) \
  case SG * 1000 + GRF: \
    return launch_form<SG, GRF>(q, p, hits, cnt, wg);
    QTC_FORMS(QTC_CASE)
#undef QTC_CASE
  }
  throw std::runtime_error("no kernel form " + std::to_string(form));
}

sycl::kernel_id kernel_of(int form) {
  switch (form) {
#define QTC_CASE(SG, GRF) \
  case SG * 1000 + GRF: \
    return form_id<SG, GRF>();
    QTC_FORMS(QTC_CASE)
#undef QTC_CASE
  }
  throw std::runtime_error("no kernel form " + std::to_string(form));
}

// The form to run on a device: QUANTUS_SYCL_KERNEL=SG,GRF, else the build's.
int pick_form(const sycl::device &d) {
  if (!d.is_gpu()) return K_CPU;
  const std::string s = env_str("QUANTUS_SYCL_KERNEL", "");
  int sg = 0, grf = 0;
  if (!s.empty() && std::sscanf(s.c_str(), "%d,%d", &sg, &grf) == 2) {
    const int f = sg * 1000 + grf;
#define QTC_CASE(SG, GRF) \
  if (f == SG * 1000 + GRF && GRF != 0) return f;
    QTC_FORMS(QTC_CASE)
#undef QTC_CASE
  }
  return K_DEFAULT;
}

struct Job {
  Core *core = nullptr;
  napi_deferred deferred = nullptr;
  napi_async_work work = nullptr;
  uint64_t mid[12];
  uint32_t tail[6];
  uint64_t start = 0;
  uint64_t count = 0;
  uint64_t target = 0;
  uint64_t maxHits = 0;
  std::vector<uint64_t> hits;
  bool overflow = false;
  std::string error;
};

void check_async(Core *c) {
  std::lock_guard<std::mutex> lk(c->asyncMu);
  if (!c->asyncError.empty()) {
    std::string e;
    e.swap(c->asyncError);
    throw std::runtime_error(e);
  }
}

// On the worker thread: the whole scan, as launches.
void run_scan(Core *c, Job *j) {
  sycl::queue &q = *c->q;
  Params p{};
  qtc::prestate(j->mid, j->tail, p.pre);
  p.target = j->target;
  p.npt = c->npt;
  const uint64_t cap = std::min<uint64_t>(j->maxHits, HIT_CAP);
  if (c->hitCap < std::max<uint64_t>(cap, 1)) {
    if (c->hits) sycl::free(c->hits, q);
    c->hits = nullptr;
    c->hitCap = 0;
    c->hits = sycl::malloc_device<uint64_t>(std::max<uint64_t>(cap, 1), q);
    if (!c->hits) throw std::runtime_error("could not allocate the hit buffer on the card");
    c->hitCap = std::max<uint64_t>(cap, 1);
  }
  // The driver compiles the kernel the first time it is used. Do that here,
  // apart from any launch, so a launch's time is the kernel's: a first launch
  // that took the compile's seconds once set every later launch to the
  // smallest size, and the card ran at a third of its speed.
  if (!c->built) {
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()},
                                                                      {kernel_of(c->form)});
    (void)kb;
    c->built = true;
  }
  uint64_t done = 0, total = 0, stored = 0;
  while (done < j->count) {
    if (c->closed.load()) throw std::runtime_error("the core is closed");
    const uint64_t n = std::min(j->count - done, c->chunk);
    p.base = j->start + done;
    p.count = (uint32_t)n;
    p.room = (uint32_t)(cap - stored);
    const auto t0 = std::chrono::steady_clock::now();
    q.memset(c->cnt, 0, sizeof(uint32_t));
    launch(c->form, q, p, c->hits + stored, c->cnt, c->wg);
    uint32_t h = 0;
    q.memcpy(&h, c->cnt, sizeof(uint32_t));
    q.wait_and_throw();
    check_async(c);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // The next launch's size, from this one's rate: only from a launch of the
    // full size (a scan's last, shorter launch times mostly its overhead),
    // and by at most 4 times either way, so one slow launch cannot pin it.
    if (n == c->chunk && secs > 1e-4) {
      const double want = (double)n / secs * c->launchSecs;
      const double lo = (double)c->chunk / 4, hi = (double)c->chunk * 4;
      uint64_t next = (uint64_t)std::min(hi, std::max(lo, want)) & ~(MIN_LAUNCH - 1);
      c->chunk = std::max(MIN_LAUNCH, std::min(MAX_LAUNCH, next));
    }
    total += h;
    stored += std::min<uint64_t>(h, p.room);
    done += n;
    // cap is maxHits, or HIT_CAP when that is less: past it nothing more is
    // kept, so the scan is over.
    if (total > cap) {
      j->overflow = true;
      break;
    }
  }
  j->hits.resize(stored);
  if (stored) {
    q.memcpy(j->hits.data(), c->hits, stored * sizeof(uint64_t));
    q.wait_and_throw();
  }
}

// ── N-API ──────────────────────────────────────────────────────────────────

napi_value undefined_value(napi_env env) {
  napi_value u;
  napi_get_undefined(env, &u);
  return u;
}

napi_value throw_error(napi_env env, const std::string &msg) {
  napi_throw_error(env, nullptr, msg.c_str());
  return nullptr;
}

napi_value make_error(napi_env env, const std::string &msg) {
  napi_value m, e;
  napi_create_string_utf8(env, msg.c_str(), msg.size(), &m);
  napi_create_error(env, nullptr, m, &e);
  return e;
}

napi_value reject_now(napi_env env, const std::string &msg) {
  napi_value promise;
  napi_deferred d;
  napi_create_promise(env, &d, &promise);
  napi_reject_deferred(env, d, make_error(env, msg));
  return promise;
}

void set_str(napi_env env, napi_value obj, const char *key, const std::string &v) {
  napi_value s;
  napi_create_string_utf8(env, v.c_str(), v.size(), &s);
  napi_set_named_property(env, obj, key, s);
}

void set_num(napi_env env, napi_value obj, const char *key, double v) {
  napi_value n;
  napi_create_double(env, v, &n);
  napi_set_named_property(env, obj, key, n);
}

std::string device_name(const sycl::device &d) {
  try {
    return d.get_info<sycl::info::device::name>();
  } catch (...) {
    return "Intel GPU";
  }
}

napi_value Devices(napi_env env, napi_callback_info) {
  napi_value arr;
  napi_create_array(env, &arr);
  const auto devs = list_devices();
  for (size_t i = 0; i < devs.size(); i++) {
    napi_value o;
    napi_create_object(env, &o);
    set_num(env, o, "index", (double)i);
    set_str(env, o, "name", device_name(devs[i]));
    // Beyond the contract, for telling cards apart on a rig: the PCI address
    // where the driver gives one, and the backend.
    try {
      if (devs[i].has(sycl::aspect::ext_intel_pci_address)) {
        set_str(env, o, "pci", devs[i].get_info<sycl::ext::intel::info::device::pci_address>());
      }
      const sycl::backend be = devs[i].get_backend();
      set_str(env, o, "backend", be == sycl::backend::ext_oneapi_level_zero ? "level_zero"
                                 : be == sycl::backend::opencl              ? "opencl"
                                                                            : "other");
    } catch (...) {
    }
    napi_set_element(env, arr, (uint32_t)i, o);
  }
  return arr;
}

void finalize_core(napi_env, void *data, void *) {
  Core *c = static_cast<Core *>(data);
  c->closed.store(true);
  c->collected = true;
  if (!c->busy) {
    c->release();
    delete c;
  }
}

Core *this_core(napi_env env, napi_callback_info info, size_t *argc, napi_value *argv) {
  napi_value self;
  if (napi_get_cb_info(env, info, argc, argv, &self, nullptr) != napi_ok) return nullptr;
  void *data = nullptr;
  if (napi_unwrap(env, self, &data) != napi_ok || !data) return nullptr;
  return static_cast<Core *>(data);
}

// A Buffer property of exactly `bytes` bytes, or null.
const uint8_t *buffer_prop(napi_env env, napi_value obj, const char *key, size_t bytes) {
  napi_value v;
  bool is = false;
  if (napi_get_named_property(env, obj, key, &v) != napi_ok) return nullptr;
  if (napi_is_buffer(env, v, &is) != napi_ok || !is) return nullptr;
  void *data = nullptr;
  size_t len = 0;
  if (napi_get_buffer_info(env, v, &data, &len) != napi_ok || len != bytes) return nullptr;
  return static_cast<const uint8_t *>(data);
}

bool number_prop(napi_env env, napi_value obj, const char *key, double *out) {
  napi_value v;
  napi_valuetype t;
  if (napi_get_named_property(env, obj, key, &v) != napi_ok) return false;
  if (napi_typeof(env, v, &t) != napi_ok || t != napi_number) return false;
  return napi_get_value_double(env, v, out) == napi_ok;
}

uint64_t be64(const uint8_t *b) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x = (x << 8) | b[i];
  return x;
}

uint64_t le64(const uint8_t *b) {
  uint64_t x = 0;
  for (int i = 7; i >= 0; i--) x = (x << 8) | b[i];
  return x;
}

void execute_scan(napi_env, void *data) {
  Job *j = static_cast<Job *>(data);
  try {
    run_scan(j->core, j);
  } catch (const std::exception &e) {
    j->error = e.what();
  } catch (...) {
    j->error = "the scan failed";
  }
}

void complete_scan(napi_env env, napi_status status, void *data) {
  Job *j = static_cast<Job *>(data);
  Core *c = j->core;
  c->busy = false;
  if (status != napi_ok && j->error.empty()) j->error = "the scan was cancelled";
  if (!j->error.empty()) {
    napi_reject_deferred(env, j->deferred, make_error(env, "Intel Quantus core: " + j->error));
  } else {
    napi_value res, buf, ov;
    void *bytes = nullptr;
    napi_create_buffer(env, j->hits.size() * 8, &bytes, &buf);
    uint8_t *b = static_cast<uint8_t *>(bytes);
    for (size_t i = 0; i < j->hits.size(); i++) {
      for (int k = 0; k < 8; k++) b[8 * i + k] = (uint8_t)(j->hits[i] >> (56 - 8 * k));
    }
    napi_get_boolean(env, j->overflow, &ov);
    napi_create_object(env, &res);
    napi_set_named_property(env, res, "hits", buf);
    napi_set_named_property(env, res, "overflow", ov);
    napi_resolve_deferred(env, j->deferred, res);
  }
  napi_delete_async_work(env, j->work);
  delete j;
  if (c->closed.load()) c->release();
  if (c->collected) delete c;
}

napi_value Scan(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  Core *c = this_core(env, info, &argc, argv);
  if (!c) return throw_error(env, "scan must be called on a core from createCore");
  if (c->closed.load() || !c->q) return reject_now(env, "the core is closed");
  if (c->busy) return reject_now(env, "a scan is already running on this core");
  napi_valuetype t = napi_undefined;
  if (argc < 1 || napi_typeof(env, argv[0], &t) != napi_ok || t != napi_object) {
    return reject_now(env, "scan needs a work object");
  }
  napi_value w = argv[0];
  const uint8_t *mid = buffer_prop(env, w, "midstate", 96);
  if (!mid) return reject_now(env, "midstate must be 96 bytes");
  const uint8_t *tail = buffer_prop(env, w, "tail", 24);
  if (!tail) return reject_now(env, "tail must be 24 bytes");
  const uint8_t *start = buffer_prop(env, w, "start", 8);
  if (!start) return reject_now(env, "start must be 8 bytes");
  const uint8_t *target = buffer_prop(env, w, "targetHigh", 8);
  if (!target) return reject_now(env, "targetHigh must be 8 bytes");
  double count = 0, maxHits = 0;
  if (!number_prop(env, w, "count", &count) || !(count >= 1 && count <= 4294967296.0) || count != (double)(uint64_t)count) {
    return reject_now(env, "count must be 1 to 2^32");
  }
  if (!number_prop(env, w, "maxHits", &maxHits) || !(maxHits >= 0) || maxHits != (double)(uint64_t)maxHits
      || maxHits > 9007199254740991.0) {
    maxHits = 0;
  }
  Job *j = new Job();
  j->core = c;
  for (int i = 0; i < 12; i++) {
    j->mid[i] = le64(mid + 8 * i);
    if (j->mid[i] >= qtc::P) {
      delete j;
      return reject_now(env, "midstate element " + std::to_string(i) + " is not below the field prime");
    }
  }
  for (int i = 0; i < 6; i++) {
    j->tail[i] = (uint32_t)tail[4 * i] | ((uint32_t)tail[4 * i + 1] << 8) | ((uint32_t)tail[4 * i + 2] << 16)
                 | ((uint32_t)tail[4 * i + 3] << 24);
  }
  j->start = be64(start);
  j->count = (uint64_t)count;
  j->target = be64(target);
  j->maxHits = (uint64_t)maxHits;
  if (j->count - 1 > ~j->start) {
    delete j;
    return reject_now(env, "start + count passes 2^64");
  }
  napi_value promise, name;
  napi_create_promise(env, &j->deferred, &promise);
  napi_create_string_utf8(env, "quantus-sycl-scan", NAPI_AUTO_LENGTH, &name);
  napi_create_async_work(env, nullptr, name, execute_scan, complete_scan, j, &j->work);
  c->busy = true;
  napi_queue_async_work(env, j->work);
  return promise;
}

napi_value Close(napi_env env, napi_callback_info info) {
  size_t argc = 0;
  Core *c = this_core(env, info, &argc, nullptr);
  if (!c) return throw_error(env, "close must be called on a core from createCore");
  c->closed.store(true);
  if (!c->busy) c->release();
  return undefined_value(env);
}

napi_value CreateCore(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
  double idx = 0;
  napi_valuetype t = napi_undefined;
  if (argc >= 1 && napi_typeof(env, argv[0], &t) == napi_ok && t == napi_object) {
    double v = 0;
    if (number_prop(env, argv[0], "device", &v)) idx = v;
  }
  const auto devs = list_devices();
  if (devs.empty()) {
    return throw_error(env, want_cpu() ? "no SYCL CPU device (QUANTUS_SYCL_DEVICE=cpu)" : "no Intel GPU found");
  }
  if (!(idx >= 0) || idx != (double)(int)idx || idx >= (double)devs.size()) {
    return throw_error(env, "there is no Intel GPU " + std::to_string((long long)idx) + ": this core sees "
                                + std::to_string(devs.size()));
  }
  Core *c = new Core();
  c->index = (int)idx;
  c->name = device_name(devs[c->index]);
  c->form = pick_form(devs[c->index]);
  c->wg = (uint32_t)env_long("QUANTUS_SYCL_WG", 128, 8, 1024);
  c->npt = (uint32_t)env_long("QUANTUS_SYCL_NPT", 1, 1, 256);
  c->launchSecs = (double)env_long("QUANTUS_SYCL_LAUNCH_MS", 40, 1, 2000) / 1000.0;
  try {
    auto handler = [c](sycl::exception_list el) {
      for (auto &e : el) {
        try {
          std::rethrow_exception(e);
        } catch (const std::exception &x) {
          std::lock_guard<std::mutex> lk(c->asyncMu);
          if (c->asyncError.empty()) c->asyncError = x.what();
        }
      }
    };
    c->q = new sycl::queue(devs[c->index], handler, sycl::property::queue::in_order());
    c->cnt = sycl::malloc_device<uint32_t>(1, *c->q);
    if (!c->cnt) throw std::runtime_error("could not allocate on the card");
    // QUANTUS_SYCL_PREBUILD=1: have the driver compile the kernel now, on this
    // thread, without running it (for ISA dumps). Otherwise the first scan does.
    if (env_long("QUANTUS_SYCL_PREBUILD", 0, 0, 1)) {
      auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(c->q->get_context(), {devs[c->index]},
                                                                        {kernel_of(c->form)});
      (void)kb;
      c->built = true;
    }
  } catch (const std::exception &e) {
    c->release();
    delete c;
    return throw_error(env, std::string("could not open Intel GPU ") + std::to_string((int)idx) + ": " + e.what());
  }
  napi_value obj, dev, fscan, fclose;
  napi_create_object(env, &obj);
  napi_create_object(env, &dev);
  set_num(env, dev, "index", c->index);
  set_str(env, dev, "name", c->name);
  set_str(env, dev, "vendor", "intel");
  napi_set_named_property(env, obj, "device", dev);
  napi_create_function(env, "scan", NAPI_AUTO_LENGTH, Scan, nullptr, &fscan);
  napi_create_function(env, "close", NAPI_AUTO_LENGTH, Close, nullptr, &fclose);
  napi_set_named_property(env, obj, "scan", fscan);
  napi_set_named_property(env, obj, "close", fclose);
  if (napi_wrap(env, obj, c, finalize_core, nullptr, nullptr) != napi_ok) {
    c->release();
    delete c;
    return throw_error(env, "could not make the core object");
  }
  return obj;
}

napi_value Init(napi_env env, napi_value exports) {
  napi_value api, fdev, fcreate;
  napi_create_int32(env, API, &api);
  napi_set_named_property(env, exports, "QUANTUS_API", api);
  napi_create_function(env, "devices", NAPI_AUTO_LENGTH, Devices, nullptr, &fdev);
  napi_set_named_property(env, exports, "devices", fdev);
  napi_create_function(env, "createCore", NAPI_AUTO_LENGTH, CreateCore, nullptr, &fcreate);
  napi_set_named_property(env, exports, "createCore", fcreate);
  return exports;
}

}  // namespace

NAPI_MODULE(quantus_sycl, Init)
