// pearl_sycl_check: the SYCL core's fold check and fold bench, without Node.
//
//   pearl_sycl_check [--m M] [--n N] [--col-batch C] [--batches B] [--first F]
//                    [--folds LIST] [--bits T] [--salt S] [--bench SECONDS]
//                    [--no-ref] [--hashed]
//   pearl_sycl_check --list
//
// --list prints every device the core may open, one line each: the index
// PEARL_GPU_INDEX takes, the name, the backend and the fold it would get.
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
// Folds: dot, xmx16, xmx8, xmx8f, xmx8r, xmx8w, the same with -emu, and auto (the fold the core
// picks for this device) and auto-emu (the reference code of that fold). The
// default is dot,auto,auto-emu. "xmx16" and "xmx8" are the hardware DPAS on a GPU
// that has it in this build, the reference code anywhere else; a fold whose
// sub-group size the device lacks (sub-group 8 on Xe2) fails to launch. On a GPU
// with XMX, auto against auto-emu is the hardware self-check: the same fold with
// the hardware instruction and with the specification's reference code. Prints
// one JSON line per fold and exits non-zero if any check failed.
//
// The card: the one the core picks, or the one PEARL_GPU_INDEX names when it is
// a single number, so each card of a rig can be checked in turn. This tool sends
// no shares and compares every transcript itself, so it opens a GPU the way an
// opener that checks its hits does (pearl_host_create_checked).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <map>
#include <vector>

#include "../src/pearl_config.h"
#include "pearl_sycl_blake3.h"

extern "C" {
int pearl_host_select_device(const PearlProfile *profile, int requested, char *name, size_t name_len,
                             char *err, size_t err_len);
void *pearl_host_create_checked(const PearlProfile *profile, bool check_hits, bool *untested, char *err,
                                size_t err_len);
void pearl_host_destroy(void *ctx);
void pearl_host_set_job_salted(void *ctx, const uint8_t *header, const uint8_t *target, uint64_t salt);
const char *pearl_host_fold_name(void *ctx);
int pearl_sycl_list_devices(char *buf, size_t len);
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

const char *kFoldNames[] = {"dot", "xmx16", "xmx8", "xmx16-emu", "xmx8-emu", "xmx8f", "xmx8f-emu", "xmx8r",
                            "xmx8r-emu"};

// A fold's code. "auto" is the context's own fold, "auto-emu" the reference code
// of it (none when that is the plain fold: -3, skipped).
int fold_code(const std::string &s, int own) {
  if (s == "auto") return own;
  if (s == "auto-emu") return own == 1 ? 3 : own == 2 ? 4 : own == 5 ? 6 : own == 7 ? 8 : own == 9 ? 10 : own == 11 ? 12 : -3;
  if (s == "dot") return 0;
  if (s == "xmx16") return 1;
  if (s == "xmx8") return 2;
  if (s == "xmx16-emu") return 3;
  if (s == "xmx8-emu") return 4;
  if (s == "xmx8f") return 5;
  if (s == "xmx8f-emu") return 6;
  if (s == "xmx8r") return 7;
  if (s == "xmx8r-emu") return 8;
  if (s == "xmx8w") return 11;
  if (s == "xmx8w-emu") return 12;
  // The shared-A' fold, in a core built with -DPEARL_SYCL_XMX8S only (a core without it runs another fold
  // for these codes, and the check then fails).
  if (s == "xmx8s") return 9;
  if (s == "xmx8s-emu") return 10;
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
    else if (a == "--list") {
      // The devices the core may open, by the index PEARL_GPU_INDEX takes.
      char list[4096] = {0};
      const int n = pearl_sycl_list_devices(list, sizeof list);
      printf("%s", list);
      if (n == 0) fprintf(stderr, "no device: no Intel GPU through Level Zero or OpenCL (or, with PEARL_SYCL_DEVICE=cpu, no CPU device)\n");
      return n > 0 ? 0 : 2;
    }
    else {
      fprintf(stderr, "usage: %s [--list] [--m M] [--n N] [--col-batch C] [--batches B] [--folds LIST] "
                      "[--bits T] [--salt S] [--first B] [--bench SECONDS] [--no-ref] [--hashed]\n", argv[0]);
      return 2;
    }
  }
  // Check default: small, and narrow batches so 4 of them cover every column offset.
  if (!bench && !mSet) { p.m = 1024; p.n = 1024; p.col_batch = 16; }
  if (bench) ref = false;

  // PEARL_GPU_INDEX as one number pins the card. A list (the CLI takes one) or
  // nothing leaves the choice to the core.
  int requested = -1;
  if (const char *g = getenv("PEARL_GPU_INDEX")) {
    char *end = nullptr;
    const long v = strtol(g, &end, 10);
    if (end != g && *end == 0 && v >= 0 && v < 1024) requested = (int)v;
  }
  char err[512] = {0}, name[256] = {0};
  if (pearl_host_select_device(&p, requested, name, sizeof name, err, sizeof err) < 0) {
    fprintf(stderr, "select: %s\n", err);
    return 2;
  }
  bool untested = false;
  void *ctx = pearl_host_create_checked(&p, true, &untested, err, sizeof err);
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
  fprintf(stderr, "core.device.fold: %s%s\n", pearl_host_fold_name(ctx),
          untested ? "; an untested kind of card, so the app checks each of its hits before sending it" : "");

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
    // The operands are read back twice and compared. On the Arc rig's card 5 a copy to the host now and then
    // leaves a 64-byte line of the host buffer unwritten (goal-arc24): the reference made from that copy is then
    // wrong, and every fold "fails" on the same few regions while the card's own data is right. Where the two
    // reads differ, a third decides, byte by byte.
    {
      std::vector<int8_t> A2(Ap.size()), B2(Bp.size());
      uint8_t s2[32];
      if (!pearl_sycl_read_operands(ctx, A2.data(), B2.data(), s2)) {
        fprintf(stderr, "could not read the operands back\n");
        return 2;
      }
      size_t diff = 0;
      for (size_t i = 0; i < Ap.size(); i++) diff += Ap[i] != A2[i];
      for (size_t i = 0; i < Bp.size(); i++) diff += Bp[i] != B2[i];
      if (diff) {
        std::vector<int8_t> A3(Ap.size()), B3(Bp.size());
        if (!pearl_sycl_read_operands(ctx, A3.data(), B3.data(), s2)) {
          fprintf(stderr, "could not read the operands back\n");
          return 2;
        }
        size_t unresolved = 0;
        auto vote = [&](std::vector<int8_t> &x, const std::vector<int8_t> &y, const std::vector<int8_t> &z) {
          for (size_t i = 0; i < x.size(); i++) {
            if (x[i] == y[i]) continue;
            if (y[i] == z[i]) x[i] = y[i];
            else if (x[i] != z[i]) unresolved++;
          }
        };
        vote(Ap, A2, A3);
        vote(Bp, B2, B3);
        fprintf(stderr, "note: two reads of the operands from the card differed in %zu bytes; a third read decided "
                        "(%zu bytes undecided). The card's copies to the host lose data; the folds are not at fault.\n",
                diff, unresolved);
        if (unresolved) return 2;
      }
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
    const std::string f = code == 9    ? "xmx8s"
                          : code == 10 ? "xmx8s-emu"
                          : code == 11 ? "xmx8w"
                          : code == 12 ? "xmx8w-emu"
                                       : kFoldNames[code];
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
          if (memcmp(&got[(size_t)r * 16], &want[((size_t)b * regions + r) * 16], 64) != 0) {
            bad++;
            if (getenv("PEARL_CHECK_DIAG") && bad <= 24) {
              // The region, and its reference made again from the same host copy (a second answer that differs
              // from the first means the host's arithmetic or memory, not the copy).
              const uint64_t region = (uint64_t)(first + b) * regions + r;
              const uint32_t rowIdx = (uint32_t)(region % rowsValid), colIdx = (uint32_t)((region / rowsValid) % colsValid);
              uint32_t w2[16];
              ref_transcript(Ap.data(), Bp.data(), p.k, pearl_expand_offset(rowIdx, PEARL_ROWS_MASK),
                             pearl_expand_offset(colIdx, PEARL_COLS_MASK), w2);
              int wj = 0;
              while (wj < 16 && got[(size_t)r * 16 + wj] == want[((size_t)b * regions + r) * 16 + wj]) wj++;
              fprintf(stderr, "diag bad region %u: rowIdx %u (rowOff %u) colIdx %u first wrong word %d; ref again %s got, %s first ref\n",
                      r, rowIdx, pearl_expand_offset(rowIdx, PEARL_ROWS_MASK), colIdx, wj,
                      memcmp(w2, &got[(size_t)r * 16], 64) ? "!=" : "==",
                      memcmp(w2, &want[((size_t)b * regions + r) * 16], 64) ? "!=" : "==");
            }
          }
          wh += wantHit[(size_t)b * regions + r];
        }
        if (wh != nh) hitBad++;
        wantHits += wh;
        // PEARL_CHECK_WORDS=1: for the first batch, regions wrong at each transcript word (word j is chunk j's
        // readout), and the same split by its column window's parity (the set, with the stream) and row parity.
        if (b == 0 && getenv("PEARL_CHECK_WORDS")) {
          size_t wj[16] = {0}, byCol[2] = {0, 0}, byRow[2] = {0, 0};
          for (uint32_t r = 0; r < regions; r++) {
            const uint32_t *g = &got[(size_t)r * 16], *w = &want[(size_t)r * 16];
            bool any = false;
            for (int j = 0; j < 16; j++)
              if (g[j] != w[j]) { wj[j]++; any = true; }
            if (any) { byRow[(r % rowsValid) & 1u]++; byCol[((r / rowsValid) / 4u) & 1u]++; }
          }
          fprintf(stderr, "words wrong by chunk:");
          for (int j = 0; j < 16; j++) fprintf(stderr, " %zu", wj[j]);
          fprintf(stderr, " | set (col window parity) %zu/%zu, row parity %zu/%zu\n", byCol[0], byCol[1], byRow[0], byRow[1]);
          // where a region's words were wanted, by first word (row offset, column offset of the first few)
          std::map<uint32_t, uint32_t> at0;
          for (uint32_t r = 0; r < regions; r++) at0[want[(size_t)r * 16]] = r;
          size_t found = 0, shown = 0;
          for (uint32_t r = 0; r < regions; r++) {
            auto it = at0.find(got[(size_t)r * 16]);
            if (it == at0.end()) continue;
            found++;
            if (it->second != r && shown < 6) {
              shown++;
              fprintf(stderr, "  region %u (row %u col %u) got region %u's word 0 (row %u col %u)\n", r, r % rowsValid,
                      r / rowsValid, it->second, it->second % rowsValid, it->second / rowsValid);
            }
          }
          fprintf(stderr, "  %zu of %u regions' word 0 is some region's word 0\n", found, regions);
        }
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
