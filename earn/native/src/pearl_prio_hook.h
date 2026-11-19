// Test hook for issue #312 (mining and the LLM in one process). Not used by the app.
//
// raiseStreamPriority(moduleName, cudartName) rewrites one loaded DLL's import of
// cudaStreamCreateWithFlags so every stream that DLL creates gets the card's highest
// stream priority. Pointed at llama.cpp's ggml-cuda.dll, it puts decode on high-priority
// streams while the miner's streams stay at the default, which is already the lowest.
// streamHookCount() says how many streams went through the hook.
//
// Windows only: it patches the PE import table. Elsewhere both functions are absent.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <napi.h>
#include <atomic>
#include <cstring>

namespace priohook {

typedef int (*WithPriorityFn)(void**, unsigned, int);
typedef int (*RangeFn)(int*, int*);
static WithPriorityFn gWithPriority = nullptr;
static int gPriority = 0;
static std::atomic<int> gCount{0};

static int HookWithFlags(void** stream, unsigned flags) {
  gCount++;
  return gWithPriority(stream, flags, gPriority);
}

// Points `fn`, imported by `mod` from `dll`, at `repl`. Returns whether it found it.
static bool PatchImport(HMODULE mod, const char* dll, const char* fn, void* repl) {
  auto base = reinterpret_cast<unsigned char*>(mod);
  auto dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
  auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (!dir.VirtualAddress) return false;
  for (auto d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; d++) {
    if (_stricmp(reinterpret_cast<char*>(base + d->Name), dll) != 0) continue;
    auto names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
    auto slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
    for (; names->u1.AddressOfData; names++, slots++) {
      if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
      auto byName = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
      if (strcmp(reinterpret_cast<char*>(byName->Name), fn) != 0) continue;
      DWORD old;
      VirtualProtect(&slots->u1.Function, sizeof(void*), PAGE_READWRITE, &old);
      slots->u1.Function = reinterpret_cast<ULONG_PTR>(repl);
      VirtualProtect(&slots->u1.Function, sizeof(void*), old, &old);
      return true;
    }
  }
  return false;
}

// raiseStreamPriority(moduleName = "ggml-cuda.dll", cudartName = "cudart64_12.dll")
//   -> { patched, priority, least, greatest }
static Napi::Value RaiseStreamPriority(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string modName = info.Length() > 0 && info[0].IsString() ? info[0].As<Napi::String>().Utf8Value() : "ggml-cuda.dll";
  std::string rtName = info.Length() > 1 && info[1].IsString() ? info[1].As<Napi::String>().Utf8Value() : "cudart64_12.dll";
  HMODULE mod = GetModuleHandleA(modName.c_str());
  HMODULE rt = GetModuleHandleA(rtName.c_str());
  if (!mod || !rt) {
    Napi::Error::New(env, "raiseStreamPriority: " + modName + " or " + rtName + " is not loaded").ThrowAsJavaScriptException();
    return env.Null();
  }
  gWithPriority = reinterpret_cast<WithPriorityFn>(GetProcAddress(rt, "cudaStreamCreateWithPriority"));
  auto range = reinterpret_cast<RangeFn>(GetProcAddress(rt, "cudaDeviceGetStreamPriorityRange"));
  int least = 0, greatest = 0;
  if (!gWithPriority || !range || range(&least, &greatest) != 0) {
    Napi::Error::New(env, "raiseStreamPriority: no stream priorities in " + rtName).ThrowAsJavaScriptException();
    return env.Null();
  }
  gPriority = greatest;   // numerically lowest = highest priority
  bool ok = PatchImport(mod, rtName.c_str(), "cudaStreamCreateWithFlags", reinterpret_cast<void*>(&HookWithFlags));
  Napi::Object r = Napi::Object::New(env);
  r.Set("patched", ok);
  r.Set("priority", gPriority);
  r.Set("least", least);
  r.Set("greatest", greatest);
  return r;
}

static Napi::Value StreamHookCount(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), gCount.load());
}

// ---- isolateDefaultStreams(moduleName, cudartName, high)
// The miner's fold streams are blocking streams, so work on the legacy default
// stream waits for the fold launch that is running. llama.cpp copies inputs and
// logits on cudaStreamPerThread or stream 0 every token. This sends every such
// call made from `moduleName` to one private non-blocking stream instead, so
// decode never waits on the miner. Test use only.
struct D3 { unsigned x, y, z; };
typedef int (*MemcpyAsyncFn)(void*, const void*, size_t, int, void*);
typedef int (*MemsetAsyncFn)(void*, int, size_t, void*);
typedef int (*Memcpy2DAsyncFn)(void*, size_t, const void*, size_t, size_t, size_t, int, void*);
typedef int (*StreamSyncFn)(void*);
typedef unsigned (*PushCfgFn)(D3, D3, size_t, void*);
typedef int (*LaunchFn)(const void*, D3, D3, void**, size_t, void*);
static MemcpyAsyncFn rMemcpyAsync; static MemsetAsyncFn rMemsetAsync; static Memcpy2DAsyncFn rMemcpy2DAsync;
static StreamSyncFn rStreamSync; static PushCfgFn rPushCfg; static LaunchFn rLaunch;
static void* gPriv = nullptr;
static std::atomic<int> gRedirects{0};
static void* gOwn[64]; static std::atomic<int> gOwnN{0};
static inline void* Fix(void* s) {
  if (s == nullptr || s == reinterpret_cast<void*>(1) || s == reinterpret_cast<void*>(2)) { gRedirects++; return gPriv; }
  return s;
}
static int HMemcpyAsync(void* d, const void* s, size_t n, int k, void* st) { return rMemcpyAsync(d, s, n, k, Fix(st)); }
static int HMemsetAsync(void* p, int v, size_t n, void* st) { return rMemsetAsync(p, v, n, Fix(st)); }
static int HMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, int k, void* st) { return rMemcpy2DAsync(d, dp, s, sp, w, h, k, Fix(st)); }
static int HStreamSync(void* st) { return rStreamSync(Fix(st)); }
static int HMemset(void* p, int v, size_t n) { int e = rMemsetAsync(p, v, n, Fix(nullptr)); return e ? e : rStreamSync(gPriv); }
// Waits for this module's own streams only, not the miner's.
static int HDevSync() {
  int e = rStreamSync(gPriv);
  int n = gOwnN.load();
  if (n > 64) n = 64;
  for (int i = 0; i < n && !e; i++) e = rStreamSync(gOwn[i]);
  return e;
}
static unsigned HPushCfg(D3 g, D3 b, size_t sh, void* st) { return rPushCfg(g, b, sh, Fix(st)); }
static int HLaunch(const void* f, D3 g, D3 b, void** a, size_t sh, void* st) { return rLaunch(f, g, b, a, sh, Fix(st)); }
static int HWithFlagsTracked(void** s, unsigned flags) {
  int e = gWithPriority(s, flags, gPriority);
  if (!e) { int i = gOwnN.fetch_add(1); if (i < 64) gOwn[i] = *s; }
  gCount++;
  return e;
}

static Napi::Value IsolateDefaultStreams(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string modName = info.Length() > 0 && info[0].IsString() ? info[0].As<Napi::String>().Utf8Value() : "ggml-cuda.dll";
  std::string rtName = info.Length() > 1 && info[1].IsString() ? info[1].As<Napi::String>().Utf8Value() : "cudart64_12.dll";
  bool high = info.Length() > 2 && info[2].ToBoolean().Value();
  HMODULE mod = GetModuleHandleA(modName.c_str());
  HMODULE rt = GetModuleHandleA(rtName.c_str());
  if (!mod || !rt) {
    Napi::Error::New(env, "isolateDefaultStreams: " + modName + " or " + rtName + " is not loaded").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto get = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(rt, n)); };
  gWithPriority = reinterpret_cast<WithPriorityFn>(get("cudaStreamCreateWithPriority"));
  auto range = reinterpret_cast<RangeFn>(get("cudaDeviceGetStreamPriorityRange"));
  rMemcpyAsync = reinterpret_cast<MemcpyAsyncFn>(get("cudaMemcpyAsync"));
  rMemsetAsync = reinterpret_cast<MemsetAsyncFn>(get("cudaMemsetAsync"));
  rMemcpy2DAsync = reinterpret_cast<Memcpy2DAsyncFn>(get("cudaMemcpy2DAsync"));
  rStreamSync = reinterpret_cast<StreamSyncFn>(get("cudaStreamSynchronize"));
  rPushCfg = reinterpret_cast<PushCfgFn>(get("__cudaPushCallConfiguration"));
  rLaunch = reinterpret_cast<LaunchFn>(get("cudaLaunchKernel"));
  int least = 0, greatest = 0;
  if (!gWithPriority || !range || !rMemcpyAsync || !rMemsetAsync || !rMemcpy2DAsync || !rStreamSync || !rPushCfg
      || !rLaunch || range(&least, &greatest) != 0) {
    Napi::Error::New(env, "isolateDefaultStreams: missing entry points in " + rtName).ThrowAsJavaScriptException();
    return env.Null();
  }
  gPriority = high ? greatest : least;
  if (!gPriv && gWithPriority(&gPriv, 1u /* cudaStreamNonBlocking */, gPriority) != 0) {
    Napi::Error::New(env, "isolateDefaultStreams: could not create the private stream").ThrowAsJavaScriptException();
    return env.Null();
  }
  const char* r = rtName.c_str();
  int n = 0;
  n += PatchImport(mod, r, "cudaStreamCreateWithFlags", reinterpret_cast<void*>(&HWithFlagsTracked));
  n += PatchImport(mod, r, "cudaMemcpyAsync", reinterpret_cast<void*>(&HMemcpyAsync));
  n += PatchImport(mod, r, "cudaMemsetAsync", reinterpret_cast<void*>(&HMemsetAsync));
  n += PatchImport(mod, r, "cudaMemcpy2DAsync", reinterpret_cast<void*>(&HMemcpy2DAsync));
  n += PatchImport(mod, r, "cudaStreamSynchronize", reinterpret_cast<void*>(&HStreamSync));
  n += PatchImport(mod, r, "cudaMemset", reinterpret_cast<void*>(&HMemset));
  n += PatchImport(mod, r, "cudaDeviceSynchronize", reinterpret_cast<void*>(&HDevSync));
  n += PatchImport(mod, r, "__cudaPushCallConfiguration", reinterpret_cast<void*>(&HPushCfg));
  n += PatchImport(mod, r, "cudaLaunchKernel", reinterpret_cast<void*>(&HLaunch));
  Napi::Object o = Napi::Object::New(env);
  o.Set("patched", n);
  o.Set("priority", gPriority);
  return o;
}

static Napi::Value IsolateStats(const Napi::CallbackInfo& info) {
  Napi::Object o = Napi::Object::New(info.Env());
  o.Set("redirects", gRedirects.load());
  o.Set("ownStreams", gOwnN.load());
  return o;
}

}  // namespace priohook
#endif  // _WIN32
