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

}  // namespace priohook
#endif  // _WIN32
