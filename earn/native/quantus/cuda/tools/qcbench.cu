// qcbench: times our Quantus kernel (qc_host.cu) or quantus-miner's official
// CUDA kernel on one card, the same way: back-to-back scans of a fixed size
// for a number of seconds, then nonces / second. Built with -DOFFICIAL and
// -I <quantus-miner>/crates/engine-cuda/src/kernels it runs the official
// kernel (mining.cu, Apache-2.0, unchanged) with the official miner's launch
// geometry: 256 threads a block, up to 4096 blocks, its 32M-nonce batch.
//
//   qcbench <seconds> [nonces per scan] [device]
//
// Prints one line a second: "t <s> rate <MH/s>", then "mean <MH/s>".
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#ifdef OFFICIAL
#include "mining.cu"
#else
#include "qc_host.h"
#endif

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 10;
  unsigned long long per = argc > 2 ? strtoull(argv[2], 0, 0) : (1ull << 27);
  int dev = argc > 3 ? atoi(argv[3]) : 0;
  cudaSetDevice(dev);
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, dev);
#ifdef OFFICIAL
  if (per > (1ull << 25)) per = 1ull << 25;
  unsigned batch = (unsigned)per;
  unsigned blocks = (batch + 255) / 256;
  if (blocks > 4096) blocks = 4096;
  unsigned total = blocks * 256;
  unsigned npt = (batch + total - 1) / total;
  MiningParams mp;
  memset(&mp, 0, sizeof mp);
  for (int i = 0; i < 24; i++) mp.prestate[i] = 0x9e3779b9u * (i + 1);
  mp.dispatch_config[0] = total;
  mp.dispatch_config[1] = npt;
  mp.dispatch_config[2] = batch;
  unsigned *res;
  cudaMalloc(&res, 64 * 4);
  printf("official mining.cu on %s sm_%d%d: %u blocks x 256, %u nonces a thread, %u a launch\n", p.name, p.major,
         p.minor, blocks, npt, batch);
#else
  char err[256];
  qc_core *c = qc_open(dev, err, sizeof err);
  if (!c) {
    printf("open: %s\n", err);
    return 1;
  }
  qc_work w;
  memset(&w, 0, sizeof w);
  for (int i = 0; i < 12; i++) w.mid[i] = 0x9e3779b97f4a7c15ull * (i + 1) % 0xFFFFFFFF00000001ull;
  w.count = per;
  w.target = 0;
  w.max_hits = 16;
  static uint64_t hits[16];
  printf("ours on %s sm_%d%d: %llu nonces a scan\n", p.name, p.major, p.minor, per);
#endif
  double t0 = now_s(), last = t0, done = 0, mark = 0;
  unsigned long long start = 1ull << 40;
  while (true) {
#ifdef OFFICIAL
    mp.start_nonce[0] = (unsigned)start;
    mp.start_nonce[1] = (unsigned)(start >> 32);
    cudaMemsetAsync(res, 0, 4);
    mining_main<<<blocks, 256>>>(res, mp);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("launch: %s\n", cudaGetErrorString(e));
      return 1;
    }
    done += batch;
    start += batch;
#else
    uint32_t stored;
    uint64_t found;
    w.start = start;
    if (qc_scan(c, &w, hits, &stored, &found, err, sizeof err)) {
      printf("scan: %s\n", err);
      return 1;
    }
    done += (double)per;
    start += per;
#endif
    double t = now_s();
    if (t - last >= 1.0) {
      printf("t %.1f rate %.2f\n", t - t0, (done - mark) / (t - last) / 1e6);
      fflush(stdout);
      mark = done;
      last = t;
    }
    if (t - t0 >= secs) {
      printf("mean %.2f MH/s over %.1f s\n", done / (t - t0) / 1e6, t - t0);
      break;
    }
  }
  return 0;
}
