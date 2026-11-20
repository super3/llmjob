// quantus_hip.node: the Quantus core interface (earn/native/quantus/README.md, version 1) over the AMD
// host in qh_host.cpp.
//
//   QUANTUS_API             1
//   devices()               [{ index, name, arch, pci }]
//   createCore({ device })  a core: { device: { index, name, vendor: 'amd', arch, pci }, scan(work), close() }
//
// scan runs on a libuv worker thread (Napi::AsyncWorker), so the event loop keeps running.

#include <napi.h>
#include <string.h>

#include <string>

#include "qh_host.h"

namespace {

const uint64_t P = 0xffffffff00000001ull;

uint64_t be64(const uint8_t *b) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
  return v;
}

uint64_t le64(const uint8_t *b) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
  return v;
}

Napi::Value Rejected(Napi::Env env, const std::string &msg) {
  auto d = Napi::Promise::Deferred::New(env);
  d.Reject(Napi::Error::New(env, msg).Value());
  return d.Promise();
}

class Core;

class ScanWorker : public Napi::AsyncWorker {
 public:
  ScanWorker(Napi::Env env, Core *core, Napi::Object self, const QhWork &work);
  void Execute() override;
  void OnOK() override;
  void OnError(const Napi::Error &e) override;
  Napi::Promise Promise() { return deferred_.Promise(); }

 private:
  void Done();
  Core *core_;
  Napi::ObjectReference self_;
  Napi::Promise::Deferred deferred_;
  QhWork work_;
  std::string hits_;
  uint32_t nhits_ = 0;
  int overflow_ = 0;
};

class Core : public Napi::ObjectWrap<Core> {
 public:
  static Napi::Function Define(Napi::Env env) {
    return DefineClass(env, "QuantusHipCore",
                       {InstanceMethod("scan", &Core::Scan), InstanceMethod("close", &Core::Close)});
  }

  Core(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Core>(info) {
    Napi::Env env = info.Env();
    int index = info.Length() > 0 && info[0].IsNumber() ? info[0].As<Napi::Number>().Int32Value() : 0;
    char err[512] = {0};
    QhDeviceInfo di;
    memset(&di, 0, sizeof(di));
    core_ = qh_open(index, err, sizeof(err));
    if (!core_) {
      Napi::Error::New(env, std::string("Quantus AMD core: ") + err).ThrowAsJavaScriptException();
      return;
    }
    qh_device_info(index, &di, err, sizeof(err));
    Napi::Object dev = Napi::Object::New(env);
    dev.Set("index", index);
    dev.Set("name", std::string(di.name));
    dev.Set("vendor", "amd");
    dev.Set("arch", std::string(di.arch));
    dev.Set("pci", std::string(di.pci));
    info.This().As<Napi::Object>().Set("device", dev);
  }

  ~Core() override {
    if (core_ && !busy_) qh_close(core_);
    core_ = nullptr;
  }

  // Called by the worker when a scan has finished, on the main thread.
  void ScanDone() {
    busy_ = false;
    if (closed_ && core_) {
      qh_close(core_);
      core_ = nullptr;
    }
  }

  QhCore *handle() { return core_; }

 private:
  Napi::Value Scan(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (closed_ || !core_) return Rejected(env, "the core is closed");
    if (busy_) return Rejected(env, "a scan is already running on this core");
    if (info.Length() < 1 || !info[0].IsObject()) return Rejected(env, "scan needs a work object");
    Napi::Object w = info[0].As<Napi::Object>();
    auto buf = [&](const char *k, size_t n, const uint8_t **out) -> bool {
      Napi::Value v = w.Get(k);
      if (!v.IsBuffer()) return false;
      Napi::Buffer<uint8_t> b = v.As<Napi::Buffer<uint8_t>>();
      if (b.Length() != n) return false;
      *out = b.Data();
      return true;
    };
    const uint8_t *mid, *tail, *start, *target;
    if (!buf("midstate", 96, &mid)) return Rejected(env, "midstate must be 96 bytes");
    if (!buf("tail", 24, &tail)) return Rejected(env, "tail must be 24 bytes");
    if (!buf("start", 8, &start)) return Rejected(env, "start must be 8 bytes");
    if (!buf("targetHigh", 8, &target)) return Rejected(env, "targetHigh must be 8 bytes");
    QhWork work;
    memset(&work, 0, sizeof(work));
    for (int i = 0; i < 12; i++) {
      work.mid[i] = le64(mid + 8 * i);
      if (work.mid[i] >= P) return Rejected(env, "midstate element " + std::to_string(i) + " is not below the field prime");
    }
    memcpy(work.tail, tail, 24);
    work.start = be64(start);
    work.target = be64(target);
    Napi::Value cnt = w.Get("count");
    double count = cnt.IsNumber() ? cnt.As<Napi::Number>().DoubleValue() : 0;
    if (!(count >= 1 && count <= 4294967296.0) || count != (double)(uint64_t)count)
      return Rejected(env, "count must be 1 to 2^32");
    work.count = (uint64_t)count;
    if (work.start + (work.count - 1) < work.start) return Rejected(env, "start + count passes 2^64");
    Napi::Value mh = w.Get("maxHits");
    double maxHits = mh.IsNumber() ? mh.As<Napi::Number>().DoubleValue() : 0;
    if (!(maxHits >= 0)) maxHits = 0;
    if (maxHits > 1048576) maxHits = 1048576;
    work.max_hits = (uint32_t)maxHits;
    busy_ = true;
    auto *worker = new ScanWorker(env, this, info.This().As<Napi::Object>(), work);
    Napi::Promise p = worker->Promise();
    worker->Queue();
    return p;
  }

  Napi::Value Close(const Napi::CallbackInfo &info) {
    if (closed_) return info.Env().Undefined();
    closed_ = true;
    if (!core_) return info.Env().Undefined();
    if (busy_) {
      qh_cancel(core_);  // the worker frees it when the scan returns
    } else {
      qh_close(core_);
      core_ = nullptr;
    }
    return info.Env().Undefined();
  }

  QhCore *core_ = nullptr;
  bool busy_ = false;
  bool closed_ = false;
};

ScanWorker::ScanWorker(Napi::Env env, Core *core, Napi::Object self, const QhWork &work)
    : Napi::AsyncWorker(env, "quantus-hip-scan"), core_(core), self_(Napi::Persistent(self)),
      deferred_(Napi::Promise::Deferred::New(env)), work_(work) {}

void ScanWorker::Execute() {
  char err[512] = {0};
  uint64_t *tmp = work_.max_hits ? new uint64_t[work_.max_hits] : nullptr;
  int rc = qh_scan(core_->handle(), &work_, tmp, &nhits_, &overflow_, err, sizeof(err));
  if (rc != 0) {
    delete[] tmp;
    SetError(err);
    return;
  }
  hits_.resize((size_t)nhits_ * 8);
  for (uint32_t i = 0; i < nhits_; i++)
    for (int b = 0; b < 8; b++) hits_[i * 8 + b] = (char)(tmp[i] >> (56 - 8 * b));
  delete[] tmp;
}

void ScanWorker::Done() {
  core_->ScanDone();
  self_.Reset();
}

void ScanWorker::OnOK() {
  Napi::Env env = Env();
  Napi::Object r = Napi::Object::New(env);
  r.Set("hits", Napi::Buffer<uint8_t>::Copy(env, reinterpret_cast<const uint8_t *>(hits_.data()), hits_.size()));
  r.Set("overflow", Napi::Boolean::New(env, overflow_ != 0));
  Done();
  deferred_.Resolve(r);
}

void ScanWorker::OnError(const Napi::Error &e) {
  Done();
  deferred_.Reject(e.Value());
}

Napi::FunctionReference *ctor = nullptr;

Napi::Value Devices(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  char err[512] = {0};
  int n = qh_device_count(err, sizeof(err));
  if (n < 0) {
    Napi::Error::New(env, std::string("Quantus AMD core: ") + err).ThrowAsJavaScriptException();
    return env.Undefined();
  }
  Napi::Array out = Napi::Array::New(env);
  uint32_t k = 0;
  for (int i = 0; i < n; i++) {
    QhDeviceInfo di;
    if (qh_device_info(i, &di, err, sizeof(err)) != 0) continue;
    Napi::Object d = Napi::Object::New(env);
    d.Set("index", i);
    d.Set("name", std::string(di.name));
    d.Set("arch", std::string(di.arch));
    d.Set("pci", std::string(di.pci));
    out.Set(k++, d);
  }
  return out;
}

Napi::Value CreateCore(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  int device = 0;
  if (info.Length() > 0 && info[0].IsObject()) {
    Napi::Value d = info[0].As<Napi::Object>().Get("device");
    if (d.IsNumber()) device = d.As<Napi::Number>().Int32Value();
  }
  return ctor->New({Napi::Number::New(env, device)});
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  ctor = new Napi::FunctionReference(Napi::Persistent(Core::Define(env)));
  exports.Set("QUANTUS_API", Napi::Number::New(env, 1));
  exports.Set("devices", Napi::Function::New(env, Devices, "devices"));
  exports.Set("createCore", Napi::Function::New(env, CreateCore, "createCore"));
  return exports;
}

}  // namespace

NODE_API_MODULE(quantus_hip, Init)
