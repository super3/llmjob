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

// One line of nonces (src/shared/quantus/qpow.js, lineWork): the twelve constants C, then X0 and X4, each below p.
#define QC_LINE 14

typedef struct qc_work {
  uint64_t line[QC_LINE];
  uint64_t start;       // the first point t on the line
  uint64_t count;       // 1 .. 2^32, and start + count <= 2^32
  uint64_t target;      // targetHigh
  uint32_t max_hits;    // at most QC_MAX_SLOTS are kept
} qc_work;

// Most hits one scan keeps.
#define QC_MAX_SLOTS (1u << 20)

// How many CUDA devices there are (0 when there is no driver; err says why).
int qc_device_count(char *err, int errlen);
// Card `index`'s name and compute capability; -1 when it can't be read.
int qc_device_info(int index, char *name, int len, int *major, int *minor);
// Opens card `index`. NULL, with err set, on failure. qc_open2 with cubin 0 leaves the patched kernel out (the
// compiled one runs); qc_open is qc_open2 with cubin 1, which checks the card's patched kernel, if it has one, and
// runs it when it passes (qc_cubin.inc).
qc_core *qc_open(int index, char *err, int errlen);
qc_core *qc_open2(int index, int cubin, char *err, int errlen);
// Whether card `index` has a patched kernel to try. Makes no context on the card.
int qc_cubin_wanted(int index);
// Runs the patched kernel's check for card `index` in this process, without opening a core: 1 passed, 0 failed (why
// says why), -1 the card has none. *lost is 1 when the check left the card's context dead (the cubin faulted): this
// process can no longer use the card, so run it in a process that ends afterwards.
int qc_cubin_test(int index, int *lost, char *why, int whylen);
// One scan. hits must hold min(max_hits, QC_MAX_SLOTS) points. *found is how
// many points hit, which can be more than *stored. 0 on success, else -1 and err.
int qc_scan(qc_core *c, const qc_work *w, uint64_t *hits, uint32_t *stored, uint64_t *found, char *err, int errlen);
void qc_close(qc_core *c);

#ifdef __cplusplus
}
#endif
#endif
