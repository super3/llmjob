// The N-API face of a NEB core, the same file for every vendor (the host under it is neb_host.h):
//
//   NEB_API                 1
//   devices()               [{ index, name }]: the cards this core can open
//   createCore({ device })  a core on that card; throws an Error that says why not
//   core.device             { index, name, vendor }
//   core.setDataset(slot, arch, values, labels)   a slot's dataset: labels.length samples of network `arch`
//   core.setModel(slot, weights)   Promise<{ samples, live, maxLive, entries, tableBytes, cpuMs, gpuMs, freeBytes,
//                                  totalBytes, tripleBytes }>; weights is one signed byte a weight
//   core.buildTriples(slot, rows, maxBytes)   Promise<{ rows, entries, tableBytes, ms }>: the triple table over
//                                  the first `rows` live rows, or as many as maxBytes and the card allow
//   core.scan(work)         Promise<{ hits: Buffer, overflow, evaluated }>; work = { slot, header: Buffer(80),
//                           target: Buffer(32), start, count, maxHits }; hits is 4 bytes a nonce, little-endian
//   core.evalKeys(slot, keys)      Promise<Buffer>: digest and ticket (64 bytes) of each 32-byte key
//   core.close()            frees the card; a call after it rejects
//
// The promises run on libuv worker threads. Plain C against node_api.h, so the addon needs nothing
// but Node's headers to build and runs under Node and Electron alike.
#define NAPI_VERSION 8
#include <node_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neb_host.h"

#define NEB_API 1
enum { JOB_MODEL = 1, JOB_SCAN = 2, JOB_EVAL = 3, JOB_TRIPLES = 4 };

typedef struct Core {
  nb_core *h;
  int index;
  char name[256];
  int closed;
  int pending;  // jobs started and not yet settled
} Core;

typedef struct Job {
  int kind;
  napi_async_work work;
  napi_deferred deferred;
  napi_ref self;  // keeps the core object alive until the job settles
  Core *core;
  int slot;
  int8_t *weights;
  uint32_t nweights;
  nb_model_info info;
  nb_triple_info tinfo;
  uint32_t rows;
  uint64_t max_bytes;
  nb_work w;
  uint32_t *hits;
  uint32_t stored;
  uint64_t found, evaluated;
  uint8_t *keys, *out;
  uint32_t nkeys;
  int rc;
  char err[512];
} Job;

static napi_ref core_ctor = NULL;

#define CHECK(call)                               \
  do {                                            \
    if ((call) != napi_ok) return fail_last(env); \
  } while (0)

static napi_value fail_last(napi_env env) {
  bool pending = false;
  napi_is_exception_pending(env, &pending);
  if (!pending) {
    const napi_extended_error_info *info = NULL;
    napi_get_last_error_info(env, &info);
    napi_throw_error(env, NULL, info && info->error_message ? info->error_message : "N-API call failed");
  }
  return NULL;
}

static napi_value make_error(napi_env env, const char *msg) {
  napi_value m, e;
  napi_create_string_utf8(env, msg, NAPI_AUTO_LENGTH, &m);
  napi_create_error(env, NULL, m, &e);
  return e;
}

static napi_value throw_msg(napi_env env, const char *msg) {
  napi_throw_error(env, NULL, msg);
  return NULL;
}

// ── devices() ───────────────────────────────────────────────────────────────

static napi_value js_devices(napi_env env, napi_callback_info info) {
  napi_value arr;
  (void)info;
  CHECK(napi_create_array(env, &arr));
  const int n = nb_device_count();
  uint32_t k = 0;
  for (int i = 0; i < n; i++) {
    char name[256];
    if (nb_device_info(i, name, sizeof name) != 0) continue;
    napi_value o, v;
    CHECK(napi_create_object(env, &o));
    CHECK(napi_create_int32(env, i, &v));
    CHECK(napi_set_named_property(env, o, "index", v));
    CHECK(napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &v));
    CHECK(napi_set_named_property(env, o, "name", v));
    CHECK(napi_set_element(env, arr, k++, o));
  }
  return arr;
}

// ── the core object ─────────────────────────────────────────────────────────

static void core_release(Core *c) {
  if (c->h && c->pending == 0) {
    nb_close(c->h);
    c->h = NULL;
  }
}

static void core_finalize(napi_env env, void *data, void *hint) {
  Core *c = (Core *)data;
  (void)env;
  (void)hint;
  c->closed = 1;
  core_release(c);  // pending is 0 here: a running job holds a reference
  free(c);
}

static napi_value core_new(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1], self;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  int32_t index = 0;
  CHECK(napi_get_value_int32(env, argv[0], &index));
  Core *c = (Core *)calloc(1, sizeof(Core));
  if (!c) return throw_msg(env, "out of memory");
  c->index = index;
  char msg[800];
  if (nb_device_info(index, c->name, sizeof c->name) != 0) {
    free(c);
    snprintf(msg, sizeof msg, "card %d cannot run the NEB core", (int)index);
    return throw_msg(env, msg);
  }
  char err[512] = "";
  c->h = nb_open(index, err, sizeof err);
  if (!c->h) {
    free(c);
    snprintf(msg, sizeof msg, "card %d: %s", (int)index, err);
    return throw_msg(env, msg);
  }
  if (napi_wrap(env, self, c, core_finalize, NULL, NULL) != napi_ok) {
    nb_close(c->h);
    free(c);
    return fail_last(env);
  }
  napi_value dev, v;
  CHECK(napi_create_object(env, &dev));
  CHECK(napi_create_int32(env, index, &v));
  CHECK(napi_set_named_property(env, dev, "index", v));
  CHECK(napi_create_string_utf8(env, c->name, NAPI_AUTO_LENGTH, &v));
  CHECK(napi_set_named_property(env, dev, "name", v));
  CHECK(napi_create_string_utf8(env, nb_vendor(), NAPI_AUTO_LENGTH, &v));
  CHECK(napi_set_named_property(env, dev, "vendor", v));
  CHECK(napi_set_named_property(env, self, "device", dev));
  return self;
}

static Core *unwrap(napi_env env, napi_value self) {
  Core *c = NULL;
  if (napi_unwrap(env, self, (void **)&c) != napi_ok || !c) {
    napi_throw_type_error(env, NULL, "not a NEB core");
    return NULL;
  }
  return c;
}

static napi_value js_close(napi_env env, napi_callback_info info) {
  napi_value self;
  CHECK(napi_get_cb_info(env, info, NULL, NULL, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  c->closed = 1;
  core_release(c);  // or when the last running job settles
  return NULL;
}

// A Buffer argument: its bytes and length, or NULL.
static const uint8_t *buffer_arg(napi_env env, napi_value v, size_t *len) {
  bool is = false;
  void *data = NULL;
  if (!v || napi_is_buffer(env, v, &is) != napi_ok || !is || napi_get_buffer_info(env, v, &data, len) != napi_ok)
    return NULL;
  return (const uint8_t *)data;
}

// A whole number argument from lo to hi, or -1.
static double number_arg(napi_env env, napi_value v, double lo, double hi) {
  double x = 0;
  napi_valuetype t = napi_undefined;
  if (!v || napi_typeof(env, v, &t) != napi_ok || t != napi_number || napi_get_value_double(env, v, &x) != napi_ok)
    return -1;
  if (x != (double)(uint64_t)x || x < lo || x > hi) return -1;
  return x;
}

static napi_value js_set_dataset(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value argv[4] = {NULL, NULL, NULL, NULL}, self;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  if (c->closed || !c->h) return throw_msg(env, "the core is closed");
  if (c->pending) return throw_msg(env, "the core is busy");
  const double slot = number_arg(env, argv[0], 0, NB_SLOTS - 1);
  const double arch = number_arg(env, argv[1], 0, 255);
  size_t nv = 0, nl = 0;
  const uint8_t *values = buffer_arg(env, argv[2], &nv);
  const uint8_t *labels = buffer_arg(env, argv[3], &nl);
  if (slot < 0 || arch < 0 || !values || !labels || nl < 1 || nv % nl != 0)
    return throw_msg(env, "setDataset(slot, arch, values, labels)");
  if (nv / nl != 784) return throw_msg(env, "a dataset whose samples are not 784 values");
  char err[512] = "";
  if (nb_set_dataset(c->h, (int)slot, (int)arch, (uint32_t)nl, values, labels, err, sizeof err) != 0)
    return throw_msg(env, err);
  return NULL;
}

// ── the promises ────────────────────────────────────────────────────────────

static void job_execute(napi_env env, void *data) {
  Job *j = (Job *)data;
  (void)env;
  nb_core *h = j->core->h;
  if (j->kind == JOB_MODEL) j->rc = nb_set_model(h, j->slot, j->weights, j->nweights, &j->info, j->err, sizeof j->err);
  else if (j->kind == JOB_SCAN) j->rc = nb_scan(h, &j->w, j->hits, &j->stored, &j->found, &j->evaluated, j->err, sizeof j->err);
  else if (j->kind == JOB_TRIPLES) j->rc = nb_build_triples(h, j->slot, j->rows, j->max_bytes, &j->tinfo, j->err, sizeof j->err);
  else j->rc = nb_eval(h, j->slot, j->keys, j->nkeys, j->out, j->err, sizeof j->err);
}

static void set_number(napi_env env, napi_value obj, const char *key, double x) {
  napi_value v;
  napi_create_double(env, x, &v);
  napi_set_named_property(env, obj, key, v);
}

static void job_free(napi_env env, Job *j) {
  if (j->work) napi_delete_async_work(env, j->work);
  if (j->self) napi_delete_reference(env, j->self);
  free(j->weights);
  free(j->hits);
  free(j->keys);
  free(j->out);
  free(j);
}

static void job_complete(napi_env env, napi_status status, void *data) {
  Job *j = (Job *)data;
  Core *c = j->core;
  if (status != napi_ok && j->rc == 0) {
    j->rc = -1;
    snprintf(j->err, sizeof j->err, "the call was cancelled");
  }
  if (j->rc != 0) {
    napi_reject_deferred(env, j->deferred, make_error(env, j->err));
  } else if (j->kind == JOB_MODEL) {
    napi_value res;
    napi_create_object(env, &res);
    set_number(env, res, "samples", j->info.samples);
    set_number(env, res, "live", j->info.live);
    set_number(env, res, "maxLive", j->info.max_live);
    set_number(env, res, "entries", (double)j->info.entries);
    set_number(env, res, "tableBytes", (double)j->info.table_bytes);
    set_number(env, res, "cpuMs", j->info.cpu_ms);
    set_number(env, res, "gpuMs", j->info.gpu_ms);
    set_number(env, res, "freeBytes", (double)j->info.free_bytes);
    set_number(env, res, "totalBytes", (double)j->info.total_bytes);
    set_number(env, res, "tripleBytes", (double)j->info.triple_bytes);
    napi_resolve_deferred(env, j->deferred, res);
  } else if (j->kind == JOB_TRIPLES) {
    napi_value res;
    napi_create_object(env, &res);
    set_number(env, res, "rows", j->tinfo.rows);
    set_number(env, res, "entries", (double)j->tinfo.entries);
    set_number(env, res, "tableBytes", (double)j->tinfo.table_bytes);
    set_number(env, res, "ms", j->tinfo.ms);
    napi_resolve_deferred(env, j->deferred, res);
  } else if (j->kind == JOB_SCAN) {
    napi_value res, buf, ov;
    uint8_t *out = NULL;
    napi_create_object(env, &res);
    napi_create_buffer(env, (size_t)j->stored * 4, (void **)&out, &buf);
    for (uint32_t i = 0; i < j->stored; i++) {
      const uint32_t x = j->hits[i];
      out[4 * i] = (uint8_t)x; out[4 * i + 1] = (uint8_t)(x >> 8);
      out[4 * i + 2] = (uint8_t)(x >> 16); out[4 * i + 3] = (uint8_t)(x >> 24);
    }
    napi_get_boolean(env, j->found > j->stored, &ov);
    napi_set_named_property(env, res, "hits", buf);
    napi_set_named_property(env, res, "overflow", ov);
    set_number(env, res, "evaluated", (double)j->evaluated);
    napi_resolve_deferred(env, j->deferred, res);
  } else {
    napi_value buf;
    void *out = NULL;
    napi_create_buffer(env, (size_t)j->nkeys * 64, &out, &buf);
    if (j->nkeys) memcpy(out, j->out, (size_t)j->nkeys * 64);
    napi_resolve_deferred(env, j->deferred, buf);
  }
  c->pending--;
  if (c->closed) core_release(c);
  job_free(env, j);
}

// Queues the job, or rejects its promise with `why`. Returns the promise.
static napi_value job_start(napi_env env, napi_value self, Core *c, Job *j, napi_value promise, const char *why) {
  napi_value name;
  if (!why && (c->closed || !c->h)) why = "the core is closed";
  if (!why && (napi_create_string_utf8(env, "neb_core_job", NAPI_AUTO_LENGTH, &name) != napi_ok ||
               napi_create_reference(env, self, 1, &j->self) != napi_ok ||
               napi_create_async_work(env, NULL, name, job_execute, job_complete, j, &j->work) != napi_ok))
    why = "could not start the call";
  if (!why) {
    j->core = c;
    c->pending++;
    if (napi_queue_async_work(env, j->work) != napi_ok) {
      c->pending--;
      why = "could not queue the call";
    }
  }
  if (why) {
    napi_reject_deferred(env, j->deferred, make_error(env, why));
    job_free(env, j);
  }
  return promise;
}

static Job *job_new(napi_env env, int kind, napi_value *promise) {
  Job *j = (Job *)calloc(1, sizeof(Job));
  if (!j) {
    napi_throw_error(env, NULL, "out of memory");
    return NULL;
  }
  j->kind = kind;
  if (napi_create_promise(env, &j->deferred, promise) != napi_ok) {
    free(j);
    fail_last(env);
    return NULL;
  }
  return j;
}

static napi_value js_set_model(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2] = {NULL, NULL}, self, promise;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  Job *j = job_new(env, JOB_MODEL, &promise);
  if (!j) return NULL;
  const char *why = NULL;
  const double slot = number_arg(env, argv[0], 0, NB_SLOTS - 1);
  size_t n = 0;
  const uint8_t *weights = buffer_arg(env, argv[1], &n);
  if (slot < 0 || !weights || n < 1 || n > 1000000) why = "setModel(slot, weights)";
  if (!why) {
    j->slot = (int)slot;
    j->nweights = (uint32_t)n;
    j->weights = (int8_t *)malloc(n);
    if (!j->weights) why = "out of memory";
    else memcpy(j->weights, weights, n);
  }
  return job_start(env, self, c, j, promise, why);
}

static napi_value js_build_triples(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value argv[3] = {NULL, NULL, NULL}, self, promise;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  Job *j = job_new(env, JOB_TRIPLES, &promise);
  if (!j) return NULL;
  const char *why = NULL;
  const double slot = number_arg(env, argv[0], 0, NB_SLOTS - 1);
  const double rows = number_arg(env, argv[1], 0, 4294967295.0);
  const double max_bytes = number_arg(env, argv[2], 0, 9007199254740991.0);
  if (slot < 0 || rows < 0 || max_bytes < 0) why = "buildTriples(slot, rows, maxBytes)";
  if (!why) {
    j->slot = (int)slot;
    j->rows = (uint32_t)rows;
    j->max_bytes = (uint64_t)max_bytes;
  }
  return job_start(env, self, c, j, promise, why);
}

static const uint8_t *buffer_prop(napi_env env, napi_value obj, const char *key, size_t len) {
  napi_value v;
  size_t n = 0;
  if (napi_get_named_property(env, obj, key, &v) != napi_ok) return NULL;
  const uint8_t *p = buffer_arg(env, v, &n);
  return p && n == len ? p : NULL;
}

static double number_prop(napi_env env, napi_value obj, const char *key, double lo, double hi) {
  napi_value v;
  if (napi_get_named_property(env, obj, key, &v) != napi_ok) return -1;
  return number_arg(env, v, lo, hi);
}

static const char *read_work(napi_env env, napi_value work, nb_work *w) {
  napi_valuetype t = napi_undefined;
  if (!work || napi_typeof(env, work, &t) != napi_ok || t != napi_object) return "work must be an object";
  const uint8_t *header = buffer_prop(env, work, "header", 80);
  if (!header) return "header must be 80 bytes";
  const uint8_t *target = buffer_prop(env, work, "target", 32);
  if (!target) return "target must be 32 bytes";
  memcpy(w->header, header, 80);
  memcpy(w->target, target, 32);
  const double slot = number_prop(env, work, "slot", 0, NB_SLOTS - 1);
  if (slot < 0) return "slot must be a dataset slot";
  const double start = number_prop(env, work, "start", 0, 4294967295.0);
  if (start < 0) return "start must be a 32-bit nonce";
  const double count = number_prop(env, work, "count", 1, 4294967296.0);
  if (count < 0) return "count must be 1 to 2^32";
  if (start + count > 4294967296.0) return "start + count passes 2^32";
  const double max_hits = number_prop(env, work, "maxHits", 0, 4294967295.0);
  if (max_hits < 0) return "maxHits must be a count";
  w->slot = (int)slot;
  w->start = (uint32_t)start;
  w->count = (uint64_t)count;
  w->max_hits = max_hits > NB_MAX_HITS ? NB_MAX_HITS : (uint32_t)max_hits;
  return NULL;
}

static napi_value js_scan(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1] = {NULL}, self, promise;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  Job *j = job_new(env, JOB_SCAN, &promise);
  if (!j) return NULL;
  const char *why = read_work(env, argv[0], &j->w);
  if (!why) {
    j->hits = (uint32_t *)malloc(((size_t)j->w.max_hits + 1) * sizeof(uint32_t));
    if (!j->hits) why = "out of memory";
  }
  return job_start(env, self, c, j, promise, why);
}

static napi_value js_eval_keys(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2] = {NULL, NULL}, self, promise;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  Job *j = job_new(env, JOB_EVAL, &promise);
  if (!j) return NULL;
  const char *why = NULL;
  const double slot = number_arg(env, argv[0], 0, NB_SLOTS - 1);
  size_t n = 0;
  const uint8_t *keys = buffer_arg(env, argv[1], &n);
  if (slot < 0 || !keys || n % 32 != 0 || n > (size_t)32 << 20) why = "evalKeys(slot, keys)";
  if (!why) {
    j->slot = (int)slot;
    j->nkeys = (uint32_t)(n / 32);
    j->keys = (uint8_t *)malloc(n ? n : 1);
    j->out = (uint8_t *)malloc(n ? 2 * n : 1);
    if (!j->keys || !j->out) why = "out of memory";
    else memcpy(j->keys, keys, n);
  }
  return job_start(env, self, c, j, promise, why);
}

// ── createCore({ device }) ──────────────────────────────────────────────────

static napi_value js_create_core(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1] = {NULL}, ctor, idx, inst;
  CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int32_t index = 0;
  napi_valuetype t = napi_undefined;
  if (argc >= 1 && argv[0]) napi_typeof(env, argv[0], &t);
  if (t == napi_object) {
    napi_value d;
    napi_valuetype dt = napi_undefined;
    if (napi_get_named_property(env, argv[0], "device", &d) == napi_ok && napi_typeof(env, d, &dt) == napi_ok &&
        dt != napi_undefined) {
      const double x = number_arg(env, d, 0, 1023);
      if (x < 0) return throw_msg(env, "device must be a card index");
      index = (int32_t)x;
    }
  }
  CHECK(napi_get_reference_value(env, core_ctor, &ctor));
  CHECK(napi_create_int32(env, index, &idx));
  if (napi_new_instance(env, ctor, 1, &idx, &inst) != napi_ok) return fail_last(env);
  return inst;
}

static napi_value init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"setDataset", NULL, js_set_dataset, NULL, NULL, NULL, napi_default_method, NULL},
      {"setModel", NULL, js_set_model, NULL, NULL, NULL, napi_default_method, NULL},
      {"buildTriples", NULL, js_build_triples, NULL, NULL, NULL, napi_default_method, NULL},
      {"scan", NULL, js_scan, NULL, NULL, NULL, napi_default_method, NULL},
      {"evalKeys", NULL, js_eval_keys, NULL, NULL, NULL, napi_default_method, NULL},
      {"close", NULL, js_close, NULL, NULL, NULL, napi_default_method, NULL},
  };
  napi_value ctor, v;
  CHECK(napi_define_class(env, "NebCore", NAPI_AUTO_LENGTH, core_new, NULL, 6, methods, &ctor));
  CHECK(napi_create_reference(env, ctor, 1, &core_ctor));
  CHECK(napi_create_int32(env, NEB_API, &v));
  CHECK(napi_set_named_property(env, exports, "NEB_API", v));
  CHECK(napi_create_function(env, "devices", NAPI_AUTO_LENGTH, js_devices, NULL, &v));
  CHECK(napi_set_named_property(env, exports, "devices", v));
  CHECK(napi_create_function(env, "createCore", NAPI_AUTO_LENGTH, js_create_core, NULL, &v));
  CHECK(napi_set_named_property(env, exports, "createCore", v));
  return exports;
}

#ifndef NODE_GYP_MODULE_NAME
#define NODE_GYP_MODULE_NAME neb_core
#endif
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
