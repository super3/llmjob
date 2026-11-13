// Integer issue rates of an Intel GPU, for pricing the Quantus kernel (timing
// only, nothing ships from here). Each row runs a long loop of one kind of
// operation on 8 independent 32-bit values a work-item and prints the rate in
// operations a second for the card, and per EU per clock at the clock given.
//
//   qtc_rates [--device N] [--mhz 2000] [--secs 2] [--only ROW]
//
// Rows: add (32-bit add), add3 (three-input add), addc (add with carry out, the
// carry used), mul32 (32 x 32 low half), prod (32 x 32 -> 64: mul + mach + mov),
// mulhi (the high half alone), mix (prod and addc as the field multiply mixes
// them). Read the ISA (IGC_ShaderDumpEnable) to see what each row compiled to:
// the "ops" this prints are source operations, not instructions.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int V = 8;        // independent values a work-item
constexpr int UNROLL = 16;  // steps a loop iteration

template <int MODE>
inline void step(uint32_t (&x)[V], uint32_t k) {
#pragma unroll
  for (int i = 0; i < V; i++) {
    const uint32_t a = x[i], b = x[(i + 1) % V];
    if constexpr (MODE == 0) {  // add
      x[i] = a + b;
    } else if constexpr (MODE == 1) {  // add3
      x[i] = a + b + x[(i + 3) % V];
    } else if constexpr (MODE == 2) {  // addc: the carry out is used
      const uint64_t s = (uint64_t)a + b;
      x[i] = (uint32_t)s + (uint32_t)(s >> 32) * k;
    } else if constexpr (MODE == 3) {  // mul32
      x[i] = a * (b | 1u);
    } else if constexpr (MODE == 4) {  // prod
      const uint64_t p = (uint64_t)a * b;
      x[i] = (uint32_t)p ^ (uint32_t)(p >> 32);
    } else if constexpr (MODE == 5) {  // mulhi
      x[i] = sycl::mul_hi(a, b) ^ k;
    } else if constexpr (MODE == 6) {  // prod with a constant
      const uint64_t p = (uint64_t)a * 0xa2077d7du;
      x[i] = (uint32_t)p ^ (uint32_t)(p >> 32);
    }
  }
}

template <int MODE>
struct Rows;

template <int MODE>
double run(sycl::queue &q, uint32_t *out, size_t global, uint32_t iters) {
  auto t0 = std::chrono::steady_clock::now();
  q.parallel_for<Rows<MODE>>(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
     const uint32_t gid = (uint32_t)it.get_global_linear_id();
     uint32_t x[V];
#pragma unroll
     for (int i = 0; i < V; i++) x[i] = gid * 2654435761u + (uint32_t)i * 40503u + 1u;
     const uint32_t k = iters | 1u;
     for (uint32_t n = 0; n < iters; n++) {
#pragma unroll
       for (int u = 0; u < UNROLL; u++) step<MODE>(x, k);
     }
     uint32_t r = 0;
#pragma unroll
     for (int i = 0; i < V; i++) r ^= x[i];
     out[gid] = r;
   }).wait();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const char *NAMES[] = {"add", "add3", "addc", "mul32", "prod", "mulhi", "prodk"};

template <int MODE>
void row(sycl::queue &q, uint32_t *out, size_t global, double secs, double mhz, size_t eus, const std::string &only) {
  if (!only.empty() && only != NAMES[MODE]) return;
  uint32_t iters = 64;
  run<MODE>(q, out, global, iters);  // compile and warm
  double t = run<MODE>(q, out, global, iters);
  while (t < secs / 4 && iters < (1u << 26)) {
    iters *= 2;
    t = run<MODE>(q, out, global, iters);
  }
  double best = 1e30;
  for (int r = 0; r < 3; r++) best = std::min(best, run<MODE>(q, out, global, iters));
  const double ops = (double)global * iters * UNROLL * V;
  const double rate = ops / best;
  std::printf("%-6s %8.3f Tops/s  %6.2f ops/EU/clk at %.0f MHz (%.3f s, %u iters)\n", NAMES[MODE], rate / 1e12,
              rate / (eus * mhz * 1e6), mhz, best, iters);
}

}  // namespace

int main(int argc, char **argv) {
  int device = 0;
  double mhz = 2000, secs = 2;
  std::string only;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--device")) device = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--mhz")) mhz = std::atof(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--secs")) secs = std::atof(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--only")) only = argv[i + 1];
  }
  std::vector<sycl::device> gpus;
  for (const auto &p : sycl::platform::get_platforms())
    if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
      for (const auto &d : p.get_devices())
        if (d.is_gpu()) gpus.push_back(d);
  if (device >= (int)gpus.size()) {
    std::fprintf(stderr, "no GPU %d\n", device);
    return 1;
  }
  sycl::queue q(gpus[device]);
  const size_t eus = gpus[device].get_info<sycl::info::device::max_compute_units>();
  std::printf("%s, %zu EUs\n", gpus[device].get_info<sycl::info::device::name>().c_str(), eus);
  const size_t global = eus * 8 * 16 * 4;  // 4 full waves of 8 threads x 16 lanes an EU
  uint32_t *out = sycl::malloc_device<uint32_t>(global, q);
  row<0>(q, out, global, secs, mhz, eus, only);
  row<1>(q, out, global, secs, mhz, eus, only);
  row<2>(q, out, global, secs, mhz, eus, only);
  row<3>(q, out, global, secs, mhz, eus, only);
  row<4>(q, out, global, secs, mhz, eus, only);
  row<5>(q, out, global, secs, mhz, eus, only);
  row<6>(q, out, global, secs, mhz, eus, only);
  sycl::free(out, q);
  return 0;
}
