// The host side of a NEB core, the same for every vendor: neb_host.inc implements it over CUDA
// (cuda/neb_cuda.cu) and HIP (amd/neb_hip.cpp), and neb_napi.c is its N-API face.
#ifndef NEB_HOST_H
#define NEB_HOST_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NB_SLOTS 4            // datasets a core holds; the main network has 2
#define NB_MAX_HITS 4096

typedef struct nb_core nb_core;

typedef struct {
  uint32_t samples;      // samples in the slot's dataset
  uint32_t live;         // samples whose gradient is not zero on this model
  uint32_t max_live;     // live rows an attempt may have to be evaluated: 2 (pair table) or 1
  uint64_t entries;      // table entries
  uint64_t table_bytes;
  double cpu_ms, gpu_ms; // the rows on the host, the table on the card
  uint64_t free_bytes, total_bytes;  // the card's memory after the table
  uint64_t triple_bytes; // the card's triple buffer, kept from the last table, which the next can reuse
} nb_model_info;

typedef struct {
  uint32_t rows;         // rows the triple table covers: 0 when none was built
  uint64_t entries;
  uint64_t table_bytes;
  double ms;
} nb_triple_info;

typedef struct {
  uint8_t header[80];    // the nonce bytes are ignored
  uint8_t target[32];    // a 256-bit little-endian integer
  uint32_t start;        // first nonce
  uint64_t count;        // nonces: start + count never passes 2^32
  uint32_t max_hits;
  int slot;
} nb_work;

int nb_device_count(void);
// 0 and the card's name when card `index` can run the core; -1 when it cannot.
int nb_device_info(int index, char *name, int len);
const char *nb_vendor(void);

nb_core *nb_open(int index, char *err, int errlen);
void nb_close(nb_core *c);

// A slot's dataset: n samples of the network `arch` (1, KEP1), values n * 784 bytes, labels n bytes.
int nb_set_dataset(nb_core *c, int slot, int arch, uint32_t n, const uint8_t *values, const uint8_t *labels,
                   char *err, int errlen);
// A slot's model: the weights, one int8 each. Builds the rows and the table. Seconds, not milliseconds.
int nb_set_model(nb_core *c, int slot, const int8_t *weights, uint32_t nweights, nb_model_info *info, char *err,
                 int errlen);
// The slot's triple table, over its first `rows` live rows or as many as fit max_bytes and the card: attempts
// with 3 live rows that are all among them are then evaluated too. A set_model drops it, and so does building
// another slot's: the card has one buffer for the table, and the slots take turns in it.
int nb_build_triples(nb_core *c, int slot, uint32_t rows, uint64_t max_bytes, nb_triple_info *info, char *err,
                     int errlen);
// Nonces start .. start + count - 1 of the header on the slot's model. hits gets up to max_hits nonces,
// *found how many there were, *evaluated how many attempts were evaluated (the ones with few live rows).
int nb_scan(nb_core *c, const nb_work *w, uint32_t *hits, uint32_t *stored, uint64_t *found, uint64_t *evaluated,
            char *err, int errlen);
// Attempts in full, from their keys (32 bytes each): out gets the digest and the ticket, 64 bytes each.
int nb_eval(nb_core *c, int slot, const uint8_t *keys, uint32_t nkeys, uint8_t *out, char *err, int errlen);

#ifdef __cplusplus
}
#endif
#endif
