// The C API between the Quantus AMD core's host (qh_host.cpp, built with hip-clang) and its N-API
// wrapper (qh_napi.cc, built as plain C++).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QhCore QhCore;

typedef struct {
  char name[128];
  char arch[32];   // e.g. gfx1032
  char pci[32];    // e.g. 0000:10:00.0
  int cus;
  int clock_khz;
} QhDeviceInfo;

typedef struct {
  uint64_t mid[12];    // the midstate, each below p
  uint8_t tail[24];    // nonce bytes 32..55
  uint64_t start;      // the first counter
  uint64_t count;      // 1 to 2^32
  uint64_t target;     // a hit is top64 <= target
  uint32_t max_hits;
} QhWork;

// Each returns -1 and writes a reason to err on failure.
int qh_device_count(char *err, size_t len);
int qh_device_info(int index, QhDeviceInfo *info, char *err, size_t len);
QhCore *qh_open(int index, char *err, size_t len);
// hits gets up to work->max_hits counters; nhits how many; overflow 1 when the card found more.
int qh_scan(QhCore *core, const QhWork *work, uint64_t *hits, uint32_t *nhits, int *overflow, char *err, size_t len);
// Makes a scan in progress stop early and reject; safe from any thread.
void qh_cancel(QhCore *core);
// Frees the card. Only when no scan is running.
void qh_close(QhCore *core);

#ifdef __cplusplus
}
#endif
