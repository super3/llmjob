// The N-API face of the Quantus CUDA core: the interface in
// earn/native/quantus/README.md (version 1).
//
//   QUANTUS_API             1
//   devices()               [{ index, name }]: CUDA cards of compute capability 7.5 and up
//   createCore({ device })  a core on that card; throws an Error that says why not
//   core.device             { index, name, vendor: 'nvidia' }
//   core.scan(work)         Promise<{ hits: Buffer, overflow: boolean }>, run on a libuv worker thread
//   core.close()            frees the card; a scan after it rejects
//
// Plain C against node_api.h, so the addon needs nothing but Node's headers to
// build and runs under Node and Electron alike.
#define NAPI_VERSION 8
#include <node_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qc_host.h"

#define QC_API 1
#define MIN_CC 75

static const uint64_t FIELD_P = 0xFFFFFFFF00000001ull;

typedef struct Core {
  qc_core *h;
  int index;
  char name[256];
  int closed;
  int pending;  // scans started and not yet settled
} Core;

typedef struct Scan {
  napi_async_work work;
  napi_deferred deferred;
  napi_ref self;  // keeps the core object alive until the scan settles
  Core *core;
  qc_work w;
  uint64_t *hits;
  uint32_t stored;
  uint64_t found;
  int rc;
  char err[512];
} Scan;

static napi_ref core_ctor = NULL;

#define CHECK(call)                                    \
  do {                                                 \
    if ((call) != napi_ok) return fail_last(env);      \
  } while (0)

// Turn a pending N-API failure into a JS exception (if none is pending) and return NULL.
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

static int eligible(int index, char *name, int len) {
  int major = 0, minor = 0;
  if (qc_device_info(index, name, len, &major, &minor) != 0) return 0;
  return major * 10 + minor >= MIN_CC;
}

// ── devices() ───────────────────────────────────────────────────────────────

static napi_value js_devices(napi_env env, napi_callback_info info) {
  napi_value arr;
  CHECK(napi_create_array(env, &arr));
  int n = qc_device_count(NULL, 0);
  uint32_t k = 0;
  for (int i = 0; i < n; i++) {
    char name[256];
    if (!eligible(i, name, sizeof name)) continue;
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
    qc_close(c->h);
    c->h = NULL;
  }
}

static void core_finalize(napi_env env, void *data, void *hint) {
  Core *c = (Core *)data;
  (void)env;
  (void)hint;
  c->closed = 1;
  core_release(c);  // pending is 0 here: a running scan holds a reference
  free(c);
}

// new QuantusCudaCore(index): opens the card. Used by createCore.
static napi_value core_new(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1], self;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  int32_t index = 0;
  CHECK(napi_get_value_int32(env, argv[0], &index));
  Core *c = (Core *)calloc(1, sizeof(Core));
  if (!c) {
    napi_throw_error(env, NULL, "out of memory");
    return NULL;
  }
  c->index = index;
  if (!eligible(index, c->name, sizeof c->name)) {
    free(c);
    char msg[128];
    snprintf(msg, sizeof msg, "no CUDA card %d of compute capability 7.5 or newer", (int)index);
    napi_throw_error(env, NULL, msg);
    return NULL;
  }
  char err[512] = "";
  c->h = qc_open(index, err, sizeof err);
  if (!c->h) {
    free(c);
    char msg[640];
    snprintf(msg, sizeof msg, "CUDA card %d: %s", (int)index, err);
    napi_throw_error(env, NULL, msg);
    return NULL;
  }
  if (napi_wrap(env, self, c, core_finalize, NULL, NULL) != napi_ok) {
    qc_close(c->h);
    free(c);
    return fail_last(env);
  }
  napi_value dev, v;
  CHECK(napi_create_object(env, &dev));
  CHECK(napi_create_int32(env, index, &v));
  CHECK(napi_set_named_property(env, dev, "index", v));
  CHECK(napi_create_string_utf8(env, c->name, NAPI_AUTO_LENGTH, &v));
  CHECK(napi_set_named_property(env, dev, "name", v));
  CHECK(napi_create_string_utf8(env, "nvidia", NAPI_AUTO_LENGTH, &v));
  CHECK(napi_set_named_property(env, dev, "vendor", v));
  CHECK(napi_set_named_property(env, self, "device", dev));
  return self;
}

static Core *unwrap(napi_env env, napi_value self) {
  Core *c = NULL;
  if (napi_unwrap(env, self, (void **)&c) != napi_ok || !c) {
    napi_throw_type_error(env, NULL, "not a Quantus CUDA core");
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
  core_release(c);  // or when the last running scan settles
  return NULL;
}

// ── scan(work) ──────────────────────────────────────────────────────────────

// A Buffer property of exactly `len` bytes, or NULL with *why set.
static const uint8_t *buffer_prop(napi_env env, napi_value obj, const char *key, size_t len, const char **why) {
  napi_value v;
  bool is = false;
  void *data = NULL;
  size_t n = 0;
  if (napi_get_named_property(env, obj, key, &v) != napi_ok || napi_is_buffer(env, v, &is) != napi_ok || !is ||
      napi_get_buffer_info(env, v, &data, &n) != napi_ok || n != len) {
    *why = key;
    return NULL;
  }
  return (const uint8_t *)data;
}

static uint64_t be64(const uint8_t *b) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x = (x << 8) | b[i];
  return x;
}

static uint64_t le64(const uint8_t *b) {
  uint64_t x = 0;
  for (int i = 7; i >= 0; i--) x = (x << 8) | b[i];
  return x;
}

// work -> Scan.w, as src/main/quantusCore.js's readWork checks it. NULL when
// it is fine, else what is wrong.
static const char *read_work(napi_env env, napi_value work, qc_work *w, char *msg, size_t len) {
  napi_valuetype t;
  if (napi_typeof(env, work, &t) != napi_ok || t != napi_object) return "work must be an object";
  const char *why = NULL;
  const uint8_t *mid = buffer_prop(env, work, "midstate", 96, &why);
  if (!mid) return "midstate must be 96 bytes";
  for (int i = 0; i < 12; i++) {
    w->mid[i] = le64(mid + 8 * i);
    if (w->mid[i] >= FIELD_P) {
      snprintf(msg, len, "midstate element %d is not below the field prime", i);
      return msg;
    }
  }
  const uint8_t *tail = buffer_prop(env, work, "tail", 24, &why);
  if (!tail) return "tail must be 24 bytes";
  for (int i = 0; i < 6; i++)
    w->tail[i] = (uint32_t)tail[4 * i] | ((uint32_t)tail[4 * i + 1] << 8) | ((uint32_t)tail[4 * i + 2] << 16) |
                 ((uint32_t)tail[4 * i + 3] << 24);
  const uint8_t *start = buffer_prop(env, work, "start", 8, &why);
  if (!start) return "start must be 8 bytes";
  const uint8_t *target = buffer_prop(env, work, "targetHigh", 8, &why);
  if (!target) return "targetHigh must be 8 bytes";
  w->start = be64(start);
  w->target = be64(target);
  napi_value v;
  double count = 0;
  if (napi_get_named_property(env, work, "count", &v) != napi_ok || napi_get_value_double(env, v, &count) != napi_ok ||
      count != (double)(uint64_t)count || count < 1 || count > 4294967296.0)
    return "count must be 1 to 2^32";
  w->count = (uint64_t)count;
  if (w->count - 1 > UINT64_MAX - w->start) return "start + count passes 2^64";
  double max_hits = 0;
  w->max_hits = 0;
  if (napi_get_named_property(env, work, "maxHits", &v) == napi_ok &&
      napi_get_value_double(env, v, &max_hits) == napi_ok && max_hits >= 0 && max_hits == (double)(uint64_t)max_hits)
    w->max_hits = max_hits > (double)QC_MAX_SLOTS ? QC_MAX_SLOTS : (uint32_t)max_hits;
  return NULL;
}

static void scan_execute(napi_env env, void *data) {
  Scan *s = (Scan *)data;
  (void)env;
  s->rc = qc_scan(s->core->h, &s->w, s->hits, &s->stored, &s->found, s->err, sizeof s->err);
}

static void scan_complete(napi_env env, napi_status status, void *data) {
  Scan *s = (Scan *)data;
  Core *c = s->core;
  if (status != napi_ok && s->rc == 0) {
    s->rc = -1;
    snprintf(s->err, sizeof s->err, "the scan was cancelled");
  }
  if (s->rc != 0) {
    napi_reject_deferred(env, s->deferred, make_error(env, s->err));
  } else {
    napi_value res, buf, ov;
    uint8_t *out = NULL;
    napi_create_object(env, &res);
    napi_create_buffer(env, (size_t)s->stored * 8, (void **)&out, &buf);
    for (uint32_t i = 0; i < s->stored; i++) {
      uint64_t x = s->hits[i];
      for (int b = 7; b >= 0; b--) {
        out[8 * i + b] = (uint8_t)x;
        x >>= 8;
      }
    }
    napi_get_boolean(env, s->found > s->w.max_hits, &ov);
    napi_set_named_property(env, res, "hits", buf);
    napi_set_named_property(env, res, "overflow", ov);
    napi_resolve_deferred(env, s->deferred, res);
  }
  c->pending--;
  if (c->closed) core_release(c);
  napi_delete_async_work(env, s->work);
  napi_delete_reference(env, s->self);
  free(s->hits);
  free(s);
}

static napi_value js_scan(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1], self, promise;
  argv[0] = NULL;
  CHECK(napi_get_cb_info(env, info, &argc, argv, &self, NULL));
  Core *c = unwrap(env, self);
  if (!c) return NULL;
  Scan *s = (Scan *)calloc(1, sizeof(Scan));
  if (!s) {
    napi_throw_error(env, NULL, "out of memory");
    return NULL;
  }
  if (napi_create_promise(env, &s->deferred, &promise) != napi_ok) {
    free(s);
    return fail_last(env);
  }
  char msg[128];
  const char *why = c->closed ? "the core is closed" : NULL;
  if (!why) {
    napi_value work = argv[0];
    if (argc < 1 || !work) napi_get_undefined(env, &work);
    why = read_work(env, work, &s->w, msg, sizeof msg);
  }
  if (!why) {
    uint32_t keep = s->w.max_hits ? s->w.max_hits : 1;
    s->hits = (uint64_t *)malloc((size_t)keep * sizeof(uint64_t));
    if (!s->hits) why = "out of memory";
  }
  napi_value name;
  if (!why && (napi_create_string_utf8(env, "quantus_cuda_scan", NAPI_AUTO_LENGTH, &name) != napi_ok ||
               napi_create_reference(env, self, 1, &s->self) != napi_ok ||
               napi_create_async_work(env, NULL, name, scan_execute, scan_complete, s, &s->work) != napi_ok))
    why = "could not start the scan";
  if (!why) {
    s->core = c;
    c->pending++;
    if (napi_queue_async_work(env, s->work) != napi_ok) {
      c->pending--;
      napi_delete_async_work(env, s->work);
      why = "could not queue the scan";
    }
  }
  if (why) {
    if (s->self) napi_delete_reference(env, s->self);
    napi_reject_deferred(env, s->deferred, make_error(env, why));
    free(s->hits);
    free(s);
  }
  return promise;
}

// ── createCore({ device }) ──────────────────────────────────────────────────

static napi_value js_create_core(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1], ctor, idx, inst;
  argv[0] = NULL;
  CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  int32_t index = 0;
  napi_valuetype t = napi_undefined;
  if (argc >= 1 && argv[0]) napi_typeof(env, argv[0], &t);
  if (t == napi_object) {
    napi_value d;
    napi_valuetype dt = napi_undefined;
    if (napi_get_named_property(env, argv[0], "device", &d) == napi_ok && napi_typeof(env, d, &dt) == napi_ok &&
        dt == napi_number) {
      double x = 0;
      napi_get_value_double(env, d, &x);
      if (x != (double)(int32_t)x || x < 0) {
        napi_throw_error(env, NULL, "device must be a card index");
        return NULL;
      }
      index = (int32_t)x;
    } else if (dt != napi_undefined) {
      napi_throw_error(env, NULL, "device must be a card index");
      return NULL;
    }
  }
  CHECK(napi_get_reference_value(env, core_ctor, &ctor));
  CHECK(napi_create_int32(env, index, &idx));
  if (napi_new_instance(env, ctor, 1, &idx, &inst) != napi_ok) return fail_last(env);
  return inst;
}

static napi_value init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"scan", NULL, js_scan, NULL, NULL, NULL, napi_default_method, NULL},
      {"close", NULL, js_close, NULL, NULL, NULL, napi_default_method, NULL},
  };
  napi_value ctor, v;
  CHECK(napi_define_class(env, "QuantusCudaCore", NAPI_AUTO_LENGTH, core_new, NULL, 2, methods, &ctor));
  CHECK(napi_create_reference(env, ctor, 1, &core_ctor));
  CHECK(napi_create_int32(env, QC_API, &v));
  CHECK(napi_set_named_property(env, exports, "QUANTUS_API", v));
  CHECK(napi_create_function(env, "devices", NAPI_AUTO_LENGTH, js_devices, NULL, &v));
  CHECK(napi_set_named_property(env, exports, "devices", v));
  CHECK(napi_create_function(env, "createCore", NAPI_AUTO_LENGTH, js_create_core, NULL, &v));
  CHECK(napi_set_named_property(env, exports, "createCore", v));
  return exports;
}

#ifndef NODE_GYP_MODULE_NAME
#define NODE_GYP_MODULE_NAME quantus_cuda
#endif
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
