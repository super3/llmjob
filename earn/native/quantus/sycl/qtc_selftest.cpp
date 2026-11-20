// Host-only check of qtc_poseidon2.h, no SYCL: the five official nonce hashes
// (quantus-miner pow-core NONCE_HASH_KVS, the same as verifyCore's "vectors")
// from the header and nonce, through the midstate and the line the kernel
// walks (each nonce as its own line's point 0), then points along a line
// against the same nonces as point 0 of the line that starts there. build.sh
// runs it before it builds the addon.
//
//   g++ -O2 -std=c++17 qtc_selftest.cpp -o qtc_selftest && ./qtc_selftest
#include <cstdio>
#include <cstring>
#include <string>

#include "qtc_poseidon2.h"

namespace {

void hex_bytes(const char *hex, unsigned char *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    unsigned v = 0;
    std::sscanf(hex + 2 * i, "%2x", &v);
    out[i] = (unsigned char)v;
  }
}

uint32_t le32(const unsigned char *b) {
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

std::string rep(const char *two, int n) {
  std::string s;
  for (int i = 0; i < n; i++) s += two;
  return s;
}

}  // namespace

int main() {
  const std::string vec[5][3] = {
      {rep("00", 32), rep("00", 64), "8e64e3d8e0f38f88"},
      {rep("01", 32), rep("00", 63) + "01", "e4ef79db80b642a0"},
      {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", rep("00", 56) + "1234567890abcdef",
       "7f6c410ce62e7fc5"},
      {rep("ff", 32), rep("00", 56) + "ffffffffffffffff", "822f70999ee517f6"},
      {rep("07", 32),
       "00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000"
       "001234567890abcdef",
       "eab81cdb33db293a"},
  };
  int bad = 0;
  for (int v = 0; v < 5; v++) {
    unsigned char header[32], nonce[64];
    hex_bytes(vec[v][0].c_str(), header, 32);
    hex_bytes(vec[v][1].c_str(), nonce, 64);
    uint64_t want = 0;
    std::sscanf(vec[v][2].c_str(), "%16llx", (unsigned long long *)&want);

    uint64_t s[12] = {0};
    for (int i = 0; i < 8; i++) s[i] = qtc::add(s[i], le32(header + 4 * i));
    qtc::permute(s);
    for (int i = 0; i < 8; i++) s[i] = qtc::add(s[i], le32(nonce + 4 * i));
    qtc::permute(s);
    uint64_t mid[12];
    for (int i = 0; i < 12; i++) mid[i] = qtc::canonical(s[i]);

    uint32_t base[8];
    for (int i = 0; i < 8; i++) base[i] = le32(nonce + 32 + 4 * i);
    uint64_t line[14];
    qtc::line_work(mid, base, line);
    const uint64_t got = qtc::nonce_top(line, 0);
    std::printf("%s vector %d: %016llx (want %016llx)\n", got == want ? "PASS" : "FAIL", v, (unsigned long long)got,
                (unsigned long long)want);
    if (got != want) bad++;

    // Along the miner's line 123 (qpow.lineBase) under this midstate: point t
    // must be point 0 of the line that starts at base + STEP t.
    const int64_t step[8] = {-4, 17, -11, 3, 4, -17, 11, -3};
    const uint32_t first[8] = {0x8000007bu, 0, 0xffffffffu, 0, 0, 0xffffffffu, 0, 0xffffffffu};
    const uint64_t points[4] = {1, 1000, 1ull << 26, 252645135};
    qtc::line_work(mid, first, line);
    for (int k = 0; k < 4; k++) {
      uint32_t moved[8];
      for (int i = 0; i < 8; i++) moved[i] = (uint32_t)((int64_t)first[i] + step[i] * (int64_t)points[k]);
      uint64_t there[14];
      qtc::line_work(mid, moved, there);
      const uint64_t walked = qtc::nonce_top(line, points[k]), direct = qtc::nonce_top(there, 0);
      if (walked != direct) {
        std::printf("FAIL vector %d, point %llu of a line: %016llx, the nonce itself %016llx\n", v,
                    (unsigned long long)points[k], (unsigned long long)walked, (unsigned long long)direct);
        bad++;
      }
    }
  }
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
