// N-API binding of the FP8 core (pearl_fp8.cu), built as pearl_core_fp8.node.
//
// A thin, synchronous surface: the JS host (earn/src/shared/miner/fp8Host.js) prepares
// everything the verifier defines in BF16 and BLAKE3, and this hands it to the card.
// Every call throws a JS Error with the CUDA reason on failure.
//
//   open(device)                      -> { name, major, minor }
//   configure(m, n, k, h, maxHits)
//   setCodes(side, Uint8Array)        load A' (0) or B' (1) as E4M3 codes
//   load(side, Int8Array, signal, beta, rewrite)   an operand's values and row numbers
//   patch(side, offset, Int8Array)    overwrite some loaded values
//   setLines(side, factor, Uint8Array) / seedLines(side, factor, key) / getLines(side, factor)
//                                     noise lines as codes, or sampled on the card from a key
//   renoise(side)                     A' or B' from what is loaded
//   noise(side, Int8Array, e, f, signal, beta, rewrite)   load + setLines + renoise
//   tree(key, Uint8Array)             -> Uint8Array, every node of a plane's Merkle tree
//   getCodes(side, row0, rows)        -> Uint8Array
//   search(key, bound, wantTickets)   -> { count, hits: Uint32Array, tickets? }
//   cells(rowA, rowB)                 -> Float32Array, 64 x 128
//   close()

#include <napi.h>

#include <cstdint>
#include <string>
#include <vector>

extern "C" {
const char *fp8_error();
int fp8_open(int device, char *name, int name_len, int *major, int *minor);
int fp8_configure(uint32_t m, uint32_t n, uint32_t k, uint32_t h, uint32_t max_hits);
int fp8_set_codes(int side, const uint8_t *codes);
int fp8_get_codes(int side, uint32_t row0, uint32_t rows, uint8_t *out);
int fp8_load(int side, const int8_t *ints, const float *signal, const float *beta, const uint32_t *rewrite);
int fp8_patch(int side, uint64_t offset, const int8_t *ints, uint32_t len);
int fp8_set_lines(int side, int factor, const uint8_t *lines);
int fp8_seed_lines(int side, int factor, const uint8_t *key);
int fp8_get_lines(int side, int factor, uint8_t *out);
int fp8_renoise(int side);
int fp8_tree(const uint8_t *key, const uint8_t *data, uint32_t chunks, uint8_t *out);
int fp8_search(const uint8_t *key, const uint8_t *bound, uint32_t *hits, uint32_t *count, uint8_t *tickets);
int fp8_cells(uint32_t row_a, uint32_t row_b, float *out);
void fp8_close();
}

namespace {

struct Shape {
  uint32_t m = 0, n = 0, k = 0, h = 0, max_hits = 0;
};
Shape shape;

Napi::Value Throw(Napi::Env env, const std::string &what) {
  Napi::Error::New(env, what).ThrowAsJavaScriptException();
  return env.Undefined();
}

// A typed array argument of exactly `bytes` bytes, or null with a JS error pending.
const uint8_t *Bytes(const Napi::CallbackInfo &info, size_t i, size_t bytes, const char *name) {
  Napi::Env env = info.Env();
  if (info.Length() <= i || !info[i].IsTypedArray()) {
    Throw(env, std::string(name) + " must be a typed array");
    return nullptr;
  }
  Napi::TypedArray ta = info[i].As<Napi::TypedArray>();
  if (ta.ByteLength() != bytes) {
    Throw(env, std::string(name) + " must be " + std::to_string(bytes) + " bytes, got " + std::to_string(ta.ByteLength()));
    return nullptr;
  }
  return static_cast<const uint8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
}

uint32_t U32(const Napi::CallbackInfo &info, size_t i) {
  return info.Length() > i && info[i].IsNumber() ? info[i].As<Napi::Number>().Uint32Value() : 0;
}

uint32_t Rows(uint32_t side) { return side == 0 ? shape.m : shape.n; }

Napi::Value Open(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  char name[256] = "";
  int major = 0, minor = 0;
  const int rc = fp8_open((int)U32(info, 0), name, sizeof(name), &major, &minor);
  if (rc) return Throw(env, fp8_error());
  Napi::Object out = Napi::Object::New(env);
  out.Set("name", name);
  out.Set("major", major);
  out.Set("minor", minor);
  return out;
}

Napi::Value Configure(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Shape s;
  s.m = U32(info, 0); s.n = U32(info, 1); s.k = U32(info, 2); s.h = U32(info, 3); s.max_hits = U32(info, 4);
  if (fp8_configure(s.m, s.n, s.k, s.h, s.max_hits)) return Throw(env, fp8_error());
  shape = s;
  return env.Undefined();
}

Napi::Value SetCodes(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint32_t side = U32(info, 0) & 1;
  const uint8_t *codes = Bytes(info, 1, (size_t)Rows(side) * shape.k, "codes");
  if (!codes) return env.Undefined();
  if (fp8_set_codes((int)side, codes)) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value GetCodes(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint32_t side = U32(info, 0) & 1, row0 = U32(info, 1), rows = U32(info, 2);
  if ((uint64_t)row0 + rows > Rows(side)) return Throw(env, "rows out of range");
  Napi::Uint8Array out = Napi::Uint8Array::New(env, (size_t)rows * shape.k);
  if (fp8_get_codes((int)side, row0, rows, out.Data())) return Throw(env, fp8_error());
  return out;
}

// The loaded-operand arguments shared by load and noise, starting at info[at].
bool LoadFrom(const Napi::CallbackInfo &info, uint32_t side, size_t at) {
  const size_t rows = Rows(side);
  const uint8_t *ints = Bytes(info, at, rows * shape.k, "values");
  if (!ints) return false;
  const uint8_t *signal = Bytes(info, at + 1, rows * (shape.k / 8) * 4, "signal");
  if (!signal) return false;
  const uint8_t *beta = Bytes(info, at + 2, rows * 4, "beta");
  if (!beta) return false;
  const uint8_t *rewrite = Bytes(info, at + 3, rows * 4, "rewrite");
  if (!rewrite) return false;
  const int rc = fp8_load((int)side, reinterpret_cast<const int8_t *>(ints),
                          reinterpret_cast<const float *>(signal), reinterpret_cast<const float *>(beta),
                          reinterpret_cast<const uint32_t *>(rewrite));
  if (rc) Throw(info.Env(), fp8_error());
  return rc == 0;
}

size_t LineBytes(uint32_t side, uint32_t factor) { return (size_t)(factor ? shape.k : Rows(side)) * 32; }

Napi::Value Load(const Napi::CallbackInfo &info) {
  LoadFrom(info, U32(info, 0) & 1, 1);
  return info.Env().Undefined();
}

Napi::Value Patch(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3 || !info[2].IsTypedArray()) return Throw(env, "values must be a typed array");
  Napi::TypedArray ta = info[2].As<Napi::TypedArray>();
  const int8_t *ints = static_cast<const int8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
  if (fp8_patch((int)(U32(info, 0) & 1), U32(info, 1), ints, (uint32_t)ta.ByteLength())) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value SetLines(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint32_t side = U32(info, 0) & 1, factor = U32(info, 1) & 1;
  const uint8_t *lines = Bytes(info, 2, LineBytes(side, factor), "lines");
  if (!lines) return env.Undefined();
  if (fp8_set_lines((int)side, (int)factor, lines)) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value SeedLines(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint8_t *key = Bytes(info, 2, 32, "key");
  if (!key) return env.Undefined();
  if (fp8_seed_lines((int)(U32(info, 0) & 1), (int)(U32(info, 1) & 1), key)) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value GetLines(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint32_t side = U32(info, 0) & 1, factor = U32(info, 1) & 1;
  Napi::Uint8Array out = Napi::Uint8Array::New(env, LineBytes(side, factor));
  if (fp8_get_lines((int)side, (int)factor, out.Data())) return Throw(env, fp8_error());
  return out;
}

Napi::Value Renoise(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (fp8_renoise((int)(U32(info, 0) & 1))) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value Noise(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint32_t side = U32(info, 0) & 1;
  const uint8_t *e = Bytes(info, 2, LineBytes(side, 0), "e");
  if (!e) return env.Undefined();
  const uint8_t *f = Bytes(info, 3, LineBytes(side, 1), "f");
  if (!f) return env.Undefined();
  // info[1] is the values; signal, beta and rewrite follow the lines.
  const size_t rows = Rows(side);
  const uint8_t *ints = Bytes(info, 1, rows * shape.k, "values");
  if (!ints) return env.Undefined();
  const uint8_t *signal = Bytes(info, 4, rows * (shape.k / 8) * 4, "signal");
  if (!signal) return env.Undefined();
  const uint8_t *beta = Bytes(info, 5, rows * 4, "beta");
  if (!beta) return env.Undefined();
  const uint8_t *rewrite = Bytes(info, 6, rows * 4, "rewrite");
  if (!rewrite) return env.Undefined();
  const int rc = fp8_load((int)side, reinterpret_cast<const int8_t *>(ints),
                          reinterpret_cast<const float *>(signal), reinterpret_cast<const float *>(beta),
                          reinterpret_cast<const uint32_t *>(rewrite)) ||
                 fp8_set_lines((int)side, 0, e) || fp8_set_lines((int)side, 1, f) || fp8_renoise((int)side);
  if (rc) return Throw(env, fp8_error());
  return env.Undefined();
}

Napi::Value Tree(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const uint8_t *key = Bytes(info, 0, 32, "key");
  if (!key) return env.Undefined();
  if (info.Length() < 2 || !info[1].IsTypedArray()) return Throw(env, "the plane must be a typed array");
  Napi::TypedArray ta = info[1].As<Napi::TypedArray>();
  const size_t bytes = ta.ByteLength();
  if (bytes % 1024 || bytes / 1024 > 0x40000000u) return Throw(env, "the plane must be whole 1024-byte chunks");
  const uint32_t chunks = (uint32_t)(bytes / 1024);
  Napi::Uint8Array out = Napi::Uint8Array::New(env, chunks < 1 ? 0 : ((size_t)chunks * 2 - 1) * 32);
  const uint8_t *data = static_cast<const uint8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
  if (fp8_tree(key, data, chunks, out.Data())) return Throw(env, fp8_error());
  return out;
}

Napi::Value Search(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (shape.h == 0) return Throw(env, "configure was not called");
  const uint8_t *key = Bytes(info, 0, 32, "key");
  if (!key) return env.Undefined();
  const uint8_t *bound = Bytes(info, 1, 32, "bound");
  if (!bound) return env.Undefined();
  const bool want = info.Length() > 2 && info[2].ToBoolean().Value();
  const size_t tiles = (size_t)(shape.m / shape.h) * (shape.n / (256 / shape.h));
  std::vector<uint32_t> hits((size_t)shape.max_hits * 2);
  std::vector<uint8_t> tickets(want ? tiles * 32 : 0);
  uint32_t count = 0;
  if (fp8_search(key, bound, hits.data(), &count, want ? tickets.data() : nullptr)) return Throw(env, fp8_error());
  const uint32_t got = count < shape.max_hits ? count : shape.max_hits;
  Napi::Uint32Array h = Napi::Uint32Array::New(env, (size_t)got * 2);
  for (size_t i = 0; i < (size_t)got * 2; i++) h[i] = hits[i];
  Napi::Object out = Napi::Object::New(env);
  out.Set("count", count);
  out.Set("hits", h);
  if (want) {
    Napi::Uint8Array t = Napi::Uint8Array::New(env, tickets.size());
    for (size_t i = 0; i < tickets.size(); i++) t[i] = tickets[i];
    out.Set("tickets", t);
  }
  return out;
}

Napi::Value Cells(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Float32Array out = Napi::Float32Array::New(env, 64 * 128);
  if (fp8_cells(U32(info, 0), U32(info, 1), out.Data())) return Throw(env, fp8_error());
  return out;
}

Napi::Value Close(const Napi::CallbackInfo &info) {
  fp8_close();
  shape = Shape();
  return info.Env().Undefined();
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  exports.Set("open", Napi::Function::New(env, Open));
  exports.Set("configure", Napi::Function::New(env, Configure));
  exports.Set("setCodes", Napi::Function::New(env, SetCodes));
  exports.Set("getCodes", Napi::Function::New(env, GetCodes));
  exports.Set("load", Napi::Function::New(env, Load));
  exports.Set("patch", Napi::Function::New(env, Patch));
  exports.Set("setLines", Napi::Function::New(env, SetLines));
  exports.Set("seedLines", Napi::Function::New(env, SeedLines));
  exports.Set("getLines", Napi::Function::New(env, GetLines));
  exports.Set("renoise", Napi::Function::New(env, Renoise));
  exports.Set("noise", Napi::Function::New(env, Noise));
  exports.Set("tree", Napi::Function::New(env, Tree));
  exports.Set("search", Napi::Function::New(env, Search));
  exports.Set("cells", Napi::Function::New(env, Cells));
  exports.Set("close", Napi::Function::New(env, Close));
  return exports;
}

}  // namespace

NODE_API_MODULE(pearl_core_fp8, InitAll)
