// qc_fault: a cubin whose scan kernel faults, for testing what the core does when a patched kernel's startup check
// breaks the card's context (qc_cubin.inc, qc_open). Build it for the card and hand it to the core:
//
//   nvcc -cubin -arch=sm_89 -DQC_FAULT=1 -o fault.cubin earn/native/quantus/cuda/tools/qc_fault.cu
//   QUANTUS_CUDA_CUBIN_FILE=fault.cubin QUANTUS_CUDA_LOG=1 node earn/native/quantus/verify-core.js --addon <core>
//
// QC_FAULT=1 traps (an unspecified launch failure), 2 writes to an address no allocation has (an illegal address).
// Both leave an error on the context that only a reset clears. The core should log that the patched kernel is not
// used, reset the card, and pass verify-core on the compiled kernel. Never shipped.
#ifndef QC_FAULT
#define QC_FAULT 1
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

// The same layout as qc_kernel.cuh's QcLaunch, so the kernel's name and parameters match the real one's.
struct QcLaunch {
  u64 c[12];
  u64 x0, x4;
  u64 target;
  u64 base;
  u32 count;
  u32 cap;
  u32 zero;
};

__global__ void qc_scan_kernel(QcLaunch L, unsigned long long *nhits, u64 *hits) {
#if QC_FAULT == 2
  // zero is 0, which the compiler cannot see, so the store stays.
  *(volatile u64 *)(0x10ull + L.zero) = L.base;
#else
  if (L.zero == 0) asm volatile("trap;");
#endif
  if (threadIdx.x == 0 && L.zero) atomicAdd(nhits, 1ull);
  (void)hits;
}
