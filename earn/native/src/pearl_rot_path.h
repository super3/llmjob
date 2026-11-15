// Where the rotation fold's cubin is on disk: the card's file (pearl_rot_sm89.cubin for RTX 40,
// pearl_rot_sm86.cubin for RTX 30) beside the module that holds the host's code, or the file PEARL_ROT_CUBIN
// names. Plain C++ with no CUDA in it, so that it can be tested alone:
// pearl_host_rot.inc includes it for the host, and probes/rot-path-check.cc runs it on Linux in CI
// (native-core.yml, "Check where the rotation fold's cubin is looked for").
//
// On Linux the module's path is the one the loader was given (dladdr's dli_fname). Every package loads the
// addon by a full path to a real file: the CLI and the HiveOS package from beside the executable (a
// single-file binary cannot load an addon out of itself), the AppImage from resources/native on its mount.
// Two cases that path alone does not cover are tried too. A relative path: dli_fname is then relative to the
// folder the process was in when it loaded the addon, which may no longer be the current one, so the kernel's
// record of the mapping is read (/proc/self/maps has the full path). And an addon that is a symbolic link to
// a file somewhere else: the folder of the link first, then the folder of the file.

#ifndef PEARL_ROT_PATH_H
#define PEARL_ROT_PATH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#endif

#define PEARL_ROT_FILE "pearl_rot_sm89.cubin"
#define PEARL_ROT_FILE_SM86 "pearl_rot_sm86.cubin"
#define PEARL_ROT_FILE_SM120 "pearl_rot_sm120.cubin"
#define PEARL_ROT_FILE_SM120W "pearl_rot_sm120w.cubin"     // RTX 5090s (pearl_host_rot.inc)

// `module` with the cubin's name (`name`) in place of its last component: the file beside the one `module`
// names. Either separator counts. False when the result would not fit.
static inline bool rot_beside(const char *module, char *buf, size_t len, const char *name = PEARL_ROT_FILE) {
  if (!module || !buf || !name) return false;
  size_t at = 0;
  for (size_t i = 0; module[i]; i++)
    if (module[i] == '/' || module[i] == '\\') at = i + 1;
  const size_t n = strlen(name) + 1;
  if (at + n > len) return false;
  memcpy(buf, module, at);
  memcpy(buf + at, name, n);
  return true;
}

static inline bool rot_readable(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  fclose(f);
  return true;
}

// The cubin named `name` for the module that holds the code at `addr`. True when a file is there, with its
// path in `buf`. False when none is: `buf` then holds the first place looked in (empty if the module could
// not be named).
static inline bool rot_cubin_path_for(const void *addr, char *buf, size_t len,
                                      const char *name = PEARL_ROT_FILE) {
  if (!buf || len < 32) return false;
  const char *e = getenv("PEARL_ROT_CUBIN");
  if (e && *e) {
    snprintf(buf, len, "%s", e);
    return rot_readable(buf);
  }
  buf[0] = 0;
#ifdef _WIN32
  HMODULE h = nullptr;
  char mod[1024];
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(addr), &h))
    return false;
  const DWORD n = GetModuleFileNameA(h, mod, (DWORD)sizeof mod);
  if (n == 0 || n >= sizeof mod) return false;
  return rot_beside(mod, buf, len, name) && rot_readable(buf);
#else
  Dl_info info;
  if (!dladdr(const_cast<void *>(addr), &info) || !info.dli_fname || !info.dli_fname[0]) return false;
  char first[1024] = {0}, cand[1024], real[PATH_MAX];
  // 1. Beside the path the loader was given.
  if (rot_beside(info.dli_fname, cand, sizeof cand, name)) {
    snprintf(first, sizeof first, "%s", cand);
    // A relative path is only good while the process is still in the folder it loaded from, and a file of
    // the same name under another folder would be the wrong one: such a path is left to the steps below.
    if (info.dli_fname[0] == '/' && rot_readable(cand)) {
      snprintf(buf, len, "%s", cand);
      return true;
    }
  }
  // 2. Beside the mapping that holds this code, by the kernel's record of it: the full path the file was
  //    opened by, whatever the process's folder is now.
  if (FILE *maps = fopen("/proc/self/maps", "r")) {
    const uintptr_t me = reinterpret_cast<uintptr_t>(addr);
    char line[1280];
    bool found = false;
    while (!found && fgets(line, sizeof line, maps)) {
      unsigned long long lo = 0, hi = 0;
      int at = 0;
      // "lo-hi perms offset dev inode   path"
      if (sscanf(line, "%llx-%llx %*s %*s %*s %*s %n", &lo, &hi, &at) < 2 || at <= 0) continue;
      if (me < lo || me >= hi || line[at] != '/') continue;
      size_t n = strlen(line + at);
      while (n && (line[at + n - 1] == '\n' || line[at + n - 1] == '\r')) line[at + --n] = 0;
      found = rot_beside(line + at, cand, sizeof cand, name) && rot_readable(cand);
    }
    fclose(maps);
    if (found) {
      snprintf(buf, len, "%s", cand);
      return true;
    }
  }
  // 3. Beside the file the loader's path resolves to (a symbolic link's target).
  if (realpath(info.dli_fname, real) && rot_beside(real, cand, sizeof cand, name) && rot_readable(cand)) {
    snprintf(buf, len, "%s", cand);
    return true;
  }
  // 4. The relative path as it is, when nothing above found the file (no /proc, say).
  snprintf(buf, len, "%s", first);
  return first[0] && rot_readable(first);
#endif
}

// ---- Is the file a cubin, and which one?
//
// The driver is handed nothing that has not passed these two. A cubin cut short made cuModuleLoadData crash
// the whole miner (rig01, 11 November: a copy truncated to 8 KB, segmentation fault inside the driver), so the
// file's own structure is checked first. And the file beside the core must be one of the files this core was
// built with, by hash: the loop in it is written by hand, and an old or damaged copy must not run.

static inline uint64_t rot_le(const uint8_t *p, int bytes) {
  uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; i--) v = (v << 8) | p[i];
  return v;
}

// Whether `b` is a 64-bit little-endian ELF file whose section table, sections and segments all lie inside its
// `n` bytes. Gives the place of its string table (.strtab), which the hash leaves out.
static inline bool rot_elf_ok(const uint8_t *b, size_t n, size_t *strtab_off, size_t *strtab_size) {
  if (!b || n < 64 || memcmp(b, "\177ELF", 4) != 0 || b[4] != 2 || b[5] != 1) return false;
  const uint64_t phoff = rot_le(b + 0x20, 8), shoff = rot_le(b + 0x28, 8);
  const uint64_t phentsize = rot_le(b + 0x36, 2), phnum = rot_le(b + 0x38, 2);
  const uint64_t shentsize = rot_le(b + 0x3A, 2), shnum = rot_le(b + 0x3C, 2), strndx = rot_le(b + 0x3E, 2);
  if (shentsize != 64 || shnum == 0 || shoff > n || shnum * 64 > n - shoff || strndx >= shnum) return false;
  if (phnum && (phentsize != 56 || phoff > n || phnum * 56 > n - phoff)) return false;
  for (uint64_t i = 0; i < phnum; i++) {
    const uint8_t *ph = b + phoff + i * 56;
    const uint64_t off = rot_le(ph + 8, 8), filesz = rot_le(ph + 32, 8);
    if (off > n || filesz > n - off) return false;
  }
  const uint8_t *names = nullptr;
  uint64_t names_size = 0;
  for (int pass = 0; pass < 2; pass++) {
    for (uint64_t i = 0; i < shnum; i++) {
      const uint8_t *sh = b + shoff + i * 64;
      const uint64_t name = rot_le(sh, 4), type = rot_le(sh + 4, 4), off = rot_le(sh + 24, 8), size = rot_le(sh + 32, 8);
      if (type != 8 && (off > n || size > n - off)) return false;   // 8: a section with no bytes in the file
      if (pass == 0) {
        if (i == strndx) {
          if (type == 8) return false;
          names = b + off;
          names_size = size;
        }
        continue;
      }
      if (name >= names_size) return false;
      if (names_size - name >= 8 && memcmp(names + name, ".strtab", 8) == 0) {
        if (strtab_off) *strtab_off = (size_t)off;
        if (strtab_size) *strtab_size = (size_t)size;
      }
    }
    if (!names || names_size == 0 || names[names_size - 1] != 0) return false;
  }
  return true;
}

static inline void rot_sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](uint32_t x, int s) { return (x >> s) | (x << (32 - s)); };
  auto block = [&](const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
    }
    h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  };
  size_t i = 0;
  for (; i + 64 <= len; i += 64) block(data + i);
  uint8_t tail[128] = {0};
  const size_t rest = len - i;
  memcpy(tail, data + i, rest);
  tail[rest] = 0x80;
  const size_t total = rest + 9 <= 64 ? 64 : 128;
  const uint64_t bits = (uint64_t)len * 8;
  for (int j = 0; j < 8; j++) tail[total - 1 - j] = (uint8_t)(bits >> (8 * j));
  block(tail);
  if (total == 128) block(tail + 64);
  for (int j = 0; j < 8; j++)
    for (int k = 0; k < 4; k++) out[4 * j + k] = (uint8_t)(h[j] >> (24 - 8 * k));
}

// The cubin's hash as src/pearl_rot_sm89.ref and pearl_rot_sm86.ref list it (probes/hand89/asm89.py, fold_hash): the sha256 of the
// file with its string table zeroed, as 64 hex digits. False when the file is not a whole ELF file. `b` is
// changed: its string table is zeroed.
static inline bool rot_fold_hash(uint8_t *b, size_t n, char hex[65]) {
  size_t off = 0, size = 0;
  if (!rot_elf_ok(b, n, &off, &size)) return false;
  memset(b + off, 0, size);
  uint8_t d[32];
  rot_sha256(b, n, d);
  for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", d[i]);
  return true;
}

#endif  // PEARL_ROT_PATH_H
