// The CUDA side of the Quantus core, as plain C for the N-API wrapper
// (qc_napi.c). Any thread may call it; a scan holds its core's lock while it
// runs, so two scans on one core take turns.
#ifndef QC_HOST_H
#define QC_HOST_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qc_core qc_core;

typedef struct qc_work {
  uint64_t mid[12];     // the midstate, each below p
  uint32_t tail[6];     // nonce bytes 32..55 as little-endian u32 words
  uint64_t start;       // the first counter
  uint64_t count;       // 1 .. 2^32, and start + count - 1 <= 2^64 - 1
  uint64_t target;      // targetHigh
  uint32_t max_hits;    // at most QC_MAX_SLOTS are kept
} qc_work;

// Most hits one scan keeps.
#define QC_MAX_SLOTS (1u << 20)

// How many CUDA devices there are (0 when there is no driver; err says why).
int qc_device_count(char *err, int errlen);
// Card `index`'s name and compute capability; -1 when it can't be read.
int qc_device_info(int index, char *name, int len, int *major, int *minor);
// Opens card `index`. NULL, with err set, on failure.
qc_core *qc_open(int index, char *err, int errlen);
// One scan. hits must hold min(max_hits, QC_MAX_SLOTS) counters. *found is how
// many counters hit, which can be more than *stored. 0 on success, else -1 and err.
int qc_scan(qc_core *c, const qc_work *w, uint64_t *hits, uint32_t *stored, uint64_t *found, char *err, int errlen);
void qc_close(qc_core *c);

#ifdef __cplusplus
}
#endif
#endif
