// LD_PRELOAD shim: every stream created through the dynamic libcudart.so.12 (only
// llama.cpp's ggml-cuda uses it; our core links cudart statically) gets the highest
// priority. Build: gcc -shared -fPIC -o prio_shim.so prio_shim.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
typedef int (*prio_fn)(void**, unsigned, int);
typedef int (*range_fn)(int*, int*);
static int raised = 0;
int cudaStreamCreateWithFlags(void** s, unsigned flags) {
  static prio_fn create = 0; static int prio = 0;
  if (!create) {
    // ggml loads libcudart locally, so RTLD_NEXT cannot see it: open it by name.
    void* h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcudart.so.12", RTLD_NOW);
    create = (prio_fn)dlsym(h, "cudaStreamCreateWithPriority");
    range_fn range = (range_fn)dlsym(h, "cudaDeviceGetStreamPriorityRange");
    int lo = 0, hi = 0; if (range) range(&lo, &hi); prio = hi;
  }
  if (!create) return 1;
  raised++;
  fprintf(stderr, "[prio_shim] stream %d at priority %d\n", raised, prio);
  return create(s, flags, prio);
}
