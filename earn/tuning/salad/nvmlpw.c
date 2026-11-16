// nvmlpw.c (goal-nv50w): what NVML says about a card's power while a miner runs, sampled every 50 ms for a few
// seconds. Plain observation, nothing set. It answers why a capped RTX 5090 can show "SW Power Cap" at 539 W of a
// 575 W limit: the board total may not be what binds, so it reads every power field NVML offers in each scope
// (GPU, module, memory), the clock-event reasons as a bit mask, and the instant draw's spread.
//
//   gcc -O2 -o nvmlpw nvmlpw.c -ldl && ./nvmlpw <seconds>
//
// One line a field and scope that the driver answers: "f<id>s<scope> n mean min max", in watts for power fields
// (NVML reports mW), plus the reasons mask's counts. No nvml.h needed: the few types are declared here.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void *dev_t_;
typedef union { double d; unsigned int ui; unsigned long ul; unsigned long long ull; long long sll; } val_t;
typedef struct {
  unsigned int fieldId, scopeId;
  long long timestamp, latencyUsec;
  int valueType, nvmlReturn;
  val_t value;
} fv_t;

static double num(const fv_t *f) {
  switch (f->valueType) {
    case 0: return f->value.d;
    case 1: return f->value.ui;
    case 2: return (double)f->value.ul;
    case 3: return (double)f->value.ull;
    case 4: return (double)f->value.sll;
    default: return (double)f->value.ui;
  }
}

int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 5;
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!h) { printf("nvmlpw no libnvidia-ml.so.1\n"); return 1; }
  int (*init)(void) = dlsym(h, "nvmlInit_v2");
  int (*byidx)(unsigned, dev_t_ *) = dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  int (*fields)(dev_t_, int, fv_t *) = dlsym(h, "nvmlDeviceGetFieldValues");
  int (*reasons)(dev_t_, unsigned long long *) = dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
  if (!reasons) reasons = dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
  int (*usage)(dev_t_, unsigned *) = dlsym(h, "nvmlDeviceGetPowerUsage");
  int (*clk)(dev_t_, int, unsigned *) = dlsym(h, "nvmlDeviceGetClockInfo");
  if (!init || !byidx || !fields || init() != 0) { printf("nvmlpw init failed\n"); return 1; }
  dev_t_ d;
  if (byidx(0, &d) != 0) { printf("nvmlpw no device 0\n"); return 1; }
  // 185 power average, 186 power instant, 190 current limit, 192 requested limit (NVML_FI_DEV_POWER_*).
  static const unsigned ids[] = {185, 186, 190, 192};
  enum { NI = 4, NS = 3, NF = NI * NS };
  fv_t fv[NF];
  double sum[NF] = {0}, mn[NF], mx[NF];
  int cnt[NF] = {0}, ret0[NF];
  for (int i = 0; i < NF; i++) { mn[i] = 1e18; mx[i] = -1e18; ret0[i] = -1; }
  double us = 0, umn = 1e18, umx = -1e18, cs = 0;
  int un = 0;
  unsigned long long rmask_or = 0;
  int rcount[64] = {0}, rn = 0;
  struct timespec t0, t;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (;;) {
    for (int i = 0; i < NF; i++) { memset(&fv[i], 0, sizeof fv[i]); fv[i].fieldId = ids[i / NS]; fv[i].scopeId = i % NS; }
    fields(d, NF, fv);
    for (int i = 0; i < NF; i++) {
      if (ret0[i] < 0) ret0[i] = fv[i].nvmlReturn;
      if (fv[i].nvmlReturn != 0) continue;
      double v = num(&fv[i]) / 1000.0;
      sum[i] += v; cnt[i]++;
      if (v < mn[i]) mn[i] = v;
      if (v > mx[i]) mx[i] = v;
    }
    unsigned mw, mhz;
    if (usage && usage(d, &mw) == 0) {
      double w = mw / 1000.0;
      us += w; un++;
      if (w < umn) umn = w;
      if (w > umx) umx = w;
    }
    if (clk && clk(d, 1, &mhz) == 0) cs += mhz;  // NVML_CLOCK_SM
    unsigned long long r;
    if (reasons && reasons(d, &r) == 0) {
      rmask_or |= r; rn++;
      for (int b = 0; b < 64; b++) if (r >> b & 1) rcount[b]++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t);
    if ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9 >= secs) break;
    struct timespec z = {0, 50 * 1000 * 1000};
    nanosleep(&z, NULL);
  }
  for (int i = 0; i < NF; i++) {
    if (cnt[i]) printf("nvmlpw f%us%d n=%d mean=%.2f min=%.2f max=%.2f\n", ids[i / NS], i % NS, cnt[i], sum[i] / cnt[i], mn[i], mx[i]);
    else printf("nvmlpw f%us%d ret=%d\n", ids[i / NS], i % NS, ret0[i]);
  }
  if (un) printf("nvmlpw usage n=%d mean=%.2f min=%.2f max=%.2f sm=%.0f MHz\n", un, us / un, umn, umx, cs / un);
  printf("nvmlpw reasons n=%d or=0x%llx", rn, rmask_or);
  for (int b = 0; b < 64; b++) if (rcount[b]) printf(" b%d:%d", b, rcount[b]);
  printf("\n");
  return 0;
}
