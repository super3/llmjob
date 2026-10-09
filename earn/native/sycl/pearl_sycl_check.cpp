// pearl_sycl_check: the SYCL core's fold check and fold bench, without Node.
//
//   pearl_sycl_check [--m M] [--n N] [--col-batch C] [--batches B] [--first F]
//                    [--folds LIST] [--bits T] [--salt S] [--bench SECONDS]
//                    [--no-ref] [--hashed]
//
// Check (the default): draws a job's operands on the device, runs B batches (from
// batch F, default 0) with
// each fold in LIST, and compares EVERY region's transcript with a scalar fold on
// the host, computed from the device's own noised operands. It also checks that
// each fold's hit count is the number of regions whose transcript hash meets
// 2^T. So a fold that loses a hit, not only one that reports a wrong one, fails.
// The draw (seeds, noise, trees, proofs) is checked separately, end to end
// against the JS reference, by verify-hits.js.
//
// Bench (--bench S): runs each fold for about S seconds at the given profile
// (default the mainnet m and n) and prints its rate in TH/s (MACs a second / 1e12,
// the unit the pool and the app use). No host reference.
//
// Folds: dot, xmx16, xmx8, xmx16-emu, xmx8-emu, and auto (the fold the core
// picks for this device) and auto-emu (the reference code of that fold). The
// default is dot,auto,auto-emu. "xmx16" and "xmx8" are the hardware DPAS on a GPU
// that has it in this build, the reference code anywhere else; a fold whose
// sub-group size the device lacks (sub-group 8 on Xe2) fails to launch. On a GPU
// with XMX, auto against auto-emu is the hardware self-check: the same fold with
// the hardware instruction and with the specification's reference code. Prints
// one JSON line per fold and exits non-zero if any check failed.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "../src/pearl_config.h"
#include "pearl_sycl_blake3.h"

extern "C" {
int pearl_host_select_device(const PearlProfile *profile, int requested, char *name, size_t name_len,
                             char *err, size_t err_len);
void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len);
void pearl_host_destroy(void *ctx);
void pearl_host_set_job_salted(void *ctx, const uint8_t *header, const uint8_t *target, uint64_t salt);
const char *pearl_sycl_fold_name(void *ctx);
int pearl_sycl_fold_code(void *ctx);
int pearl_sycl_fold_is_hw(void *ctx);
const char *pearl_sycl_device_name(void *ctx);
uint32_t pearl_sycl_batch_regions(void *ctx);
bool pearl_sycl_run_batch(void *ctx, uint64_t nonce_base, int fold, uint32_t *transcripts,
                          uint32_t *hits, double *ms, char *err, size_t err_len);
bool pearl_sycl_read_operands(void *ctx, int8_t *Ap, int8_t *Bp, uint8_t a_seed[32]);
}

namespace {

const char *kFoldNames[] = {"dot", "xmx16", "xmx8", "xmx16-emu", "xmx8-emu"};

// A fold's code. "auto" is the context's own fold, "auto-emu" the reference code
// of it (none when that is the plain fold: -3, skipped).
int fold_code(const std::string &s, int own) {
  if (s == "auto") return own;
  if (s == "auto-emu") return own == 1 ? 3 : own == 2 ? 4 : -3;
  if (s == "dot") return 0;
  if (s == "xmx16") return 1;
  if (s == "xmx8") return 2;
  if (s == "xmx16-emu") return 3;
  if (s == "xmx8-emu") return 4;
  return -2;
}

// The reference: one region's transcript, the cumulative 16x16 tile XORed at
// every 128-k chunk (16 chunks, one word each), straight from the noised operands.
void ref_transcript(const int8_t *Ap, const int8_t *Bp, uint32_t k, uint32_t rowOff, uint32_t colOff,
                    uint32_t out[16]) {
  int32_t acc[16][16] = {};
  for (int j = 0; j < 16; j++) out[j] = 0;
  for (uint32_t c = 0; c < k / 128u; c++) {
    uint32_t x = 0;
    for (int i = 0; i < 16; i++) {
      const int8_t *a = Ap + (size_t)(rowOff | PEARL_ROWS_PATTERN[i]) * k + c * 128u;
      for (int j = 0; j < 16; j++) {
        const int8_t *b = Bp + (size_t)(colOff | PEARL_COLS_PATTERN[j]) * k + c * 128u;
        int32_t s = acc[i][j];
        for (int t = 0; t < 128; t++) s += (int32_t)a[t] * (int32_t)b[t];
        acc[i][j] = s;
        x ^= (uint32_t)s;
      }
    }
    out[c % 16] = pearl_rotl13(out[c % 16]) ^ x;  // the reference's rule, as foldTranscript
  }
}

}  // namespace

int main(int argc, char **argv) {
  PearlProfile p = PEARL_MAINNET_PROFILE;
  bool bench = false, ref = true;
  double benchSecs = 0;
  uint32_t batches = 4, bits = 248, first = 0;
  uint64_t salt = 0;
  bool mSet = false;
  std::string folds = "dot,auto,auto-emu";
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    const char *v = i + 1 < argc ? argv[i + 1] : nullptr;
    if (a == "--m" && v) { p.m = (uint32_t)atoi(v); mSet = true; i++; }
    else if (a == "--n" && v) { p.n = (uint32_t)atoi(v); i++; }
    else if (a == "--col-batch" && v) { p.col_batch = (uint32_t)atoi(v); i++; }
    else if (a == "--batches" && v) { batches = (uint32_t)atoi(v); i++; }
    else if (a == "--first" && v) { first = (uint32_t)atoi(v); i++; }
    else if (a == "--folds" && v) { folds = v; i++; }
    else if (a == "--bits" && v) { bits = (uint32_t)atoi(v); i++; }
    else if (a == "--salt" && v) { salt = strtoull(v, nullptr, 10); i++; }
    else if (a == "--bench" && v) { bench = true; benchSecs = atof(v); i++; }
    else if (a == "--no-ref") { ref = false; }
    else if (a == "--hashed") { p.operand_fill = PEARL_OPERAND_HASHED; }
    else {
      fprintf(stderr, "usage: %s [--m M] [--n N] [--col-batch C] [--batches B] [--folds LIST] "
                      "[--bits T] [--salt S] [--first B] [--bench SECONDS] [--no-ref] [--hashed]\n", argv[0]);
      return 2;
    }
  }
  // Check default: small, and narrow batches so 4 of them cover every column offset.
  if (!bench && !mSet) { p.m = 1024; p.n = 1024; p.col_batch = 16; }
  if (bench) ref = false;

  char err[512] = {0}, name[256] = {0};
  if (pearl_host_select_device(&p, -1, name, sizeof name, err, sizeof err) < 0) {
    fprintf(stderr, "select: %s\n", err);
    return 2;
  }
  void *ctx = pearl_host_create(&p, err, sizeof err);
  if (!ctx) { fprintf(stderr, "create: %s\n", err); return 2; }
  uint8_t header[PEARL_HEADER_BYTES];
  memset(header, 0x5A, sizeof header);
  uint8_t target[32] = {0};
  if (bits >= 256) memset(target, 0xFF, 32);
  else target[31 - bits / 8] = (uint8_t)(1u << (bits % 8));  // 2^bits, big-endian
  pearl_host_set_job_salted(ctx, header, target, salt);

  const uint32_t regions = pearl_sycl_batch_regions(ctx);
  const uint32_t rowsValid = p.m / 16, colsValid = p.n / 16;
  if (!bench && (uint64_t)(first + batches) * (regions / rowsValid) > colsValid)
    fprintf(stderr, "note: %u batches of %u column offsets wrap past the %u there are; later batches repeat columns\n",
            batches, regions / rowsValid, colsValid);
  fprintf(stderr, "device: %s; m %u n %u; %u regions a batch; default fold %s%s\n",
          pearl_sycl_device_name(ctx), p.m, p.n, regions, pearl_sycl_fold_name(ctx),
          pearl_sycl_fold_is_hw(ctx) ? " (hardware)" : "");

  std::vector<int8_t> Ap, Bp;
  uint8_t aSeed[32];
  std::vector<uint32_t> want;  // reference transcripts, batch after batch
  std::vector<uint8_t> wantHit;
  uint32_t key[8];
  uint32_t target_w[8];
  for (int i = 0; i < 8; i++)
    target_w[i] = ((uint32_t)target[4 * i] << 24) | ((uint32_t)target[4 * i + 1] << 16)
                  | ((uint32_t)target[4 * i + 2] << 8) | target[4 * i + 3];
  if (ref) {
    Ap.resize((size_t)p.m * p.k);
    Bp.resize((size_t)p.n * p.k);
    if (!pearl_sycl_read_operands(ctx, Ap.data(), Bp.data(), aSeed)) {
      fprintf(stderr, "could not read the operands back\n");
      return 2;
    }
    for (int i = 0; i < 8; i++) key[i] = pearl_b3::load_le32(aSeed + 4 * i);
    want.assign((size_t)batches * regions * 16, 0);
    wantHit.assign((size_t)batches * regions, 0);
    const auto t0 = std::chrono::steady_clock::now();
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++)
      th.emplace_back([&, t]() {
        for (size_t r = t; r < (size_t)batches * regions; r += nt) {
          const uint64_t region = (uint64_t)first * regions + r;  // batches run from --first
          const uint32_t rowIdx = (uint32_t)(region % rowsValid);
          const uint32_t colIdx = (uint32_t)((region / rowsValid) % colsValid);
          uint32_t *w = &want[r * 16];
          ref_transcript(Ap.data(), Bp.data(), p.k, pearl_expand_offset(rowIdx, PEARL_ROWS_MASK),
                         pearl_expand_offset(colIdx, PEARL_COLS_MASK), w);
          uint32_t h[8];
          pearl_b3::hash64(key, true, w, h);
          wantHit[r] = pearl_b3::meets(h, target_w, 0) ? 1 : 0;
        }
      });
    for (auto &x : th) x.join();
    fprintf(stderr, "host reference: %zu regions in %.1f s\n", (size_t)batches * regions,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  int failed = 0;
  std::vector<int> done;
  size_t start = 0;
  while (start <= folds.size()) {
    size_t end = folds.find(',', start);
    if (end == std::string::npos) end = folds.size();
    const std::string asked = folds.substr(start, end - start);
    start = end + 1;
    if (asked.empty()) continue;
    const int code = fold_code(asked, pearl_sycl_fold_code(ctx));
    if (code == -3) continue;
    if (code < 0) { fprintf(stderr, "unknown fold %s\n", asked.c_str()); return 2; }
    if (std::find(done.begin(), done.end(), code) != done.end()) continue;
    done.push_back(code);
    const std::string f = kFoldNames[code];
    if (bench) {
      double total = 0, best = 1e30;
      uint32_t n = 0;
      std::string e;
      // The first batch includes any JIT compile; it is run and dropped.
      double ms0 = 0;
      if (!pearl_sycl_run_batch(ctx, 0, code, nullptr, nullptr, &ms0, err, sizeof err)) e = err;
      while (e.empty() && total < benchSecs * 1000.0) {
        double ms = 0;
        const uint64_t nonce = (uint64_t)(n % (colsValid / (regions / rowsValid))) * regions;
        if (!pearl_sycl_run_batch(ctx, nonce, code, nullptr, nullptr, &ms, err, sizeof err)) { e = err; break; }
        total += ms;
        if (ms < best) best = ms;
        n++;
      }
      const double macs = (double)regions * 256.0 * p.k;
      printf("{\"fold\":\"%s\",\"bench\":true,\"batches\":%u,\"regions\":%u,\"meanMs\":%.3f,"
             "\"ths\":%.3f,\"bestThs\":%.3f%s%s%s}\n",
             f.c_str(), n, regions, n ? total / n : 0.0, n ? macs * n / (total / 1000.0) / 1e12 : 0.0,
             n ? macs / (best / 1000.0) / 1e12 : 0.0, e.empty() ? "" : ",\"error\":\"",
             e.c_str(), e.empty() ? "" : "\"");
      if (!e.empty()) failed = 1;
      fflush(stdout);
      continue;
    }
    std::vector<uint32_t> got((size_t)regions * 16);
    size_t bad = 0, compared = 0, hitBad = 0, hits = 0, wantHits = 0;
    double msTotal = 0;
    std::string e;
    for (uint32_t b = 0; b < batches; b++) {
      uint32_t nh = 0;
      double ms = 0;
      if (!pearl_sycl_run_batch(ctx, (uint64_t)(first + b) * regions, code, got.data(), &nh, &ms, err, sizeof err)) {
        e = err;
        break;
      }
      msTotal += ms;
      hits += nh;
      size_t wh = 0;
      if (ref) {
        for (uint32_t r = 0; r < regions; r++) {
          compared++;
          if (memcmp(&got[(size_t)r * 16], &want[((size_t)b * regions + r) * 16], 64) != 0) bad++;
          wh += wantHit[(size_t)b * regions + r];
        }
        if (wh != nh) hitBad++;
        wantHits += wh;
      }
    }
    const bool pass = e.empty() && (!ref || (bad == 0 && hitBad == 0 && compared > 0));
    if (!pass) failed = 1;
    printf("{\"fold\":\"%s\",\"regions\":%zu,\"mismatched\":%zu,\"hits\":%zu,\"expectedHits\":%zu,"
           "\"ms\":%.1f,\"PASS\":%s%s%s%s}\n",
           f.c_str(), compared, bad, hits, wantHits, msTotal, pass ? "true" : "false",
           e.empty() ? "" : ",\"error\":\"", e.c_str(), e.empty() ? "" : "\"");
    fflush(stdout);
  }
  pearl_host_destroy(ctx);
  return failed;
}
