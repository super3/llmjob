// SYCL kernels for the Pearl core on Intel GPUs (and, for testing, the SYCL CPU
// device). Ports of the draw kernels in ../src/pearl_kernel.cu, which they must
// match bit for bit, plus two folds of our own:
//
//   fold_dot  plain int8 dot products in local memory. Needs no XMX, so it runs
//             on any device the build has kernels for. The reference fold.
//   fold_xmx  the XMX matrix engines through DPAS, at sub-group 16 (Xe2, Xe3,
//             Xe-HPC) or 8 (Xe-HPG). With PEARL_SYCL_XMX_HW (an AOT build) it is
//             the hardware instruction on those GPUs; everywhere else it is the
//             reference code from the cl_intel_subgroup_matrix_multiply_accumulate
//             specification, which the CPU device runs. Same source either way.
//
// Both folds compute the same thing as pearl_tile_fold_* in the CUDA core: for
// every region of a batch, the cumulative int32 tile over k, XORed into one
// transcript word at each 128-k boundary, the transcript hashed under a_seed,
// and a hit list appended to when the hash meets the target.

#ifndef PEARL_SYCL_KERNELS_HPP
#define PEARL_SYCL_KERNELS_HPP

#include <sycl/sycl.hpp>
#include <stdint.h>

#if defined(PEARL_SYCL_XMX_HW)
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#endif

#include "../src/pearl_config.h"
#include "pearl_sycl_blake3.h"

// The DPAS vector types, and the SPIR-V builtin
// (SPV_INTEL_subgroup_matrix_multiply_accumulate). The builtin must be declared
// at global scope: inside a namespace its name is mangled and the SPIR-V
// translator no longer recognises it.
typedef int v8i __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_HW)
extern SYCL_EXTERNAL v8i __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int, v8s, v8i, v8i, int);
extern SYCL_EXTERNAL v8i __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int, v8i, v8i, v8i, int);
#endif

namespace pearl_sycl {

// Eight words passed to a kernel by value: a BLAKE3 key or a 32-byte label.
struct Words8 {
  uint32_t w[8];
};

inline Words8 words_of(const uint8_t *b) {
  Words8 o;
  for (int i = 0; i < 8; i++) o.w[i] = pearl_b3::load_le32(b + 4 * i);
  return o;
}

// The same fields as PearlTranscriptTest in pearl_config.h.
struct TranscriptTest {
  uint32_t key[8];       // a_seed, little-endian words
  uint32_t target_w[8];  // target, big-endian words, most significant first
  int hash_big_endian;
};

// One batch's hits, as the fold writes them. Copied to pinned host memory whole
// after the fold, so reading a batch's result takes no further device copy.
struct HitBuf {
  uint32_t count;  // hits this batch; may exceed PEARL_MAX_HITS
  uint32_t pad[3];
  uint32_t index[PEARL_MAX_HITS];            // batch-local region index
  uint32_t hash[PEARL_MAX_HITS * 8];         // jackpot hash words, bytes in order
  uint32_t transcript[PEARL_MAX_HITS * 16];  // the transcript that hashed to it
};

constexpr uint32_t kThreads = 256;

inline size_t round_up(size_t n, size_t m) { return (n + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// The draw: operands, commitment trees, noise, noised operands.
// ---------------------------------------------------------------------------

// The salt stamp over A's first PEARL_STAMP_BYTES bytes (pearl_stamp_byte). A
// kernel, so no host buffer has to outlive an asynchronous copy.
inline void stamp_a(sycl::queue &q, int8_t *dA, uint64_t salt) {
  q.single_task([=]() {
    for (int i = 0; i < PEARL_STAMP_BYTES; i++) dA[i] = pearl_stamp_byte(salt, i);
  });
}

// pearl_gen_operand: uniform int7 bytes keyed by job_key, 32 a hash.
inline void gen_operand(sycl::queue &q, Words8 key, Words8 label, int8_t *out, uint64_t total,
                        uint64_t salt) {
  const uint64_t blocks = (total + 31) / 32;
  q.parallel_for(sycl::nd_range<1>(round_up(blocks, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint64_t blk = it.get_global_id(0);
    const uint64_t base = blk * 32u;
    if (base >= total) return;
    uint32_t msg[16] = {0};
    msg[0] = (uint32_t)blk + 1u;
    msg[2] = (uint32_t)salt;
    msg[3] = (uint32_t)(salt >> 32);
    for (int i = 0; i < 8; i++) msg[8 + i] = label.w[i];
    uint32_t h[8];
    pearl_b3::hash64(key.w, true, msg, h);
    for (int i = 0; i < 32; i++) {
      if (base + (uint64_t)i >= total) break;
      const uint32_t byte = (h[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
      out[base + i] = (int8_t)((int32_t)(byte % 127u) - 63);
    }
  });
}

// Level 0 of a commitment tree: one keyed chunk CV per 1024-byte chunk, at its
// own index. `compact`: the operand is stored as chunk 0 and the one chunk every
// other chunk equals (the constant fill, see Ctx::compact in pearl_host.cu).
inline void chunk_cvs(sycl::queue &q, Words8 key, const uint8_t *data, uint64_t chunks,
                      bool compact, uint32_t *out) {
  q.parallel_for(sycl::nd_range<1>(round_up(chunks, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint64_t i = it.get_global_id(0);
    if (i >= chunks) return;
    const uint8_t *c = compact ? data + (i == 0 ? 0 : 1024) : data + i * 1024;
    uint32_t cv[8];
    pearl_b3::chunk_cv(key.w, c, i, cv);
    for (int j = 0; j < 8; j++) out[i * 8 + j] = cv[j];
  });
}

// One parent level: pairs of children into their parents. `root` on the last.
inline void parent_layer(sycl::queue &q, Words8 key, const uint32_t *in, uint64_t pairs,
                         bool root, uint32_t *out) {
  q.parallel_for(sycl::nd_range<1>(round_up(pairs, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint64_t i = it.get_global_id(0);
    if (i >= pairs) return;
    uint32_t l[8], r[8], o[8];
    for (int j = 0; j < 8; j++) { l[j] = in[i * 16 + j]; r[j] = in[i * 16 + 8 + j]; }
    pearl_b3::parent_cv(key.w, l, r, root, o);
    for (int j = 0; j < 8; j++) out[i * 8 + j] = o[j];
  });
}

// The noise RNG (pearl_random_hash): blake3(msg64, key=seed) with slot
// `prepend` of the first 32 bytes holding index + 1 and the label after them.
inline void random_hash(const uint32_t key[8], const Words8 &label, uint32_t index, int prepend,
                        uint32_t h[8]) {
  uint32_t msg[16] = {0};
  msg[prepend] = index + 1u;
  for (int i = 0; i < 8; i++) msg[8 + i] = label.w[i];
  pearl_b3::hash64(key, true, msg, h);
}

// pearl_gen_dense: E_AL or E_BR, `rank` values a row in [-32, 32), one digest
// per 32 of them, from the global byte stream at row * rank.
inline void gen_dense(sycl::queue &q, Words8 seed, Words8 label, int8_t *out, uint32_t rows,
                      uint32_t rank) {
  const uint32_t per = rank >> 5;
  const uint64_t n = (uint64_t)rows * per;
  q.parallel_for(sycl::nd_range<1>(round_up(n, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint64_t idx = it.get_global_id(0);
    if (idx >= n) return;
    const uint32_t ri = (uint32_t)(idx / per), blk = (uint32_t)(idx % per);
    uint32_t h[8];
    random_hash(seed.w, label, ((ri * rank) >> 5) + blk, 0, h);
    int8_t *dst = out + (size_t)ri * rank + (blk << 5);
    for (int i = 0; i < 32; i++)
      dst[i] = (int8_t)((int32_t)((h[i >> 2] >> ((i & 3) * 8)) & 63u) - 32);
  });
}

// pearl_gen_perm: E_AR or E_BL, a (p0, p1) pair per k, eight per digest.
inline void gen_perm(sycl::queue &q, Words8 seed, Words8 label, uint32_t *out, uint32_t k,
                     uint32_t rank) {
  const uint32_t n = (k + 7u) / 8u;
  q.parallel_for(sycl::nd_range<1>(round_up(n, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint32_t i = (uint32_t)it.get_global_id(0);
    if (i >= n) return;
    uint32_t h[8];
    random_hash(seed.w, label, i, 1, h);
    const uint32_t mask = rank - 1u;
    for (int j = 0; j < 8; j++) {
      const uint32_t row = i * 8u + (uint32_t)j;
      if (row >= k) break;
      const uint32_t u = h[j];
      const uint32_t p0 = u & mask;
      const uint32_t hi = (uint32_t)(((uint64_t)(rank - 1u) * (uint64_t)u) >> 32);
      out[row * 2u] = p0;
      out[row * 2u + 1u] = p0 ^ (1u + hi);
    }
  });
}

// pearl_materialize16: A'[r,kk] = sat_i8(A[r,kk] + dense[r][p0(kk)] - dense[r][p1(kk)]),
// sixteen bytes a work-item, row-major. Only the first read_vecs sixteen-byte
// groups of the operand are read; the rest are fill_word (the constant fill).
inline void materialize16(sycl::queue &q, const int8_t *base, const int8_t *dense,
                          const uint32_t *perm, int8_t *out, uint32_t rows, uint32_t k_log2,
                          uint32_t rank, uint64_t read_vecs, uint32_t fill_word) {
  const uint64_t n = ((uint64_t)rows << k_log2) >> 4;
  q.parallel_for(sycl::nd_range<1>(round_up(n, kThreads), kThreads), [=](sycl::nd_item<1> it) {
    const uint64_t v = it.get_global_id(0);
    if (v >= n) return;
    const uint64_t idx = v << 4;
    const uint32_t r = (uint32_t)(idx >> k_log2);
    const uint32_t kk = (uint32_t)(idx & ((1ull << k_log2) - 1u));
    const int8_t *row = dense + (size_t)r * rank;
    uint32_t bw[4];
    if (v < read_vecs) {
      const uint32_t *b = reinterpret_cast<const uint32_t *>(base) + v * 4;
      for (int w = 0; w < 4; w++) bw[w] = b[w];
    } else {
      for (int w = 0; w < 4; w++) bw[w] = fill_word;
    }
    uint32_t ow[4];
    for (int w = 0; w < 4; w++) {
      uint32_t packed = 0;
      for (int j = 0; j < 4; j++) {
        const uint32_t e = kk + (uint32_t)(w * 4 + j);
        const int32_t a = (int32_t)(int8_t)(bw[w] >> (j * 8));
        int32_t s = a + (int32_t)row[perm[e * 2u]] - (int32_t)row[perm[e * 2u + 1u]];
        s = s < -128 ? -128 : (s > 127 ? 127 : s);
        packed |= ((uint32_t)s & 0xffu) << (j * 8);
      }
      ow[w] = packed;
    }
    uint32_t *o = reinterpret_cast<uint32_t *>(out) + v * 4;
    for (int w = 0; w < 4; w++) o[w] = ow[w];
  });
}

// ---------------------------------------------------------------------------
// The fold.
//
// Both folds work in WINDOWS of 32 rows by 64 columns. A window holds exactly 8
// regions, at row offsets 32*wr + {0, 4} and column offsets 64*wc + {0, 2, 4, 6}
// (the valid offsets interleave; see PEARL_ROWS_SPAN and PEARL_COLS_SPAN), so no
// region crosses a window. Region q of a window (0..7) is row offset 4*(q & 1)
// and column offset 2*(q >> 1).
//
// A batch is colBatch valid column offsets from colStart (a multiple of 4) by
// every valid row offset: rowsValid/2 row windows by colBatch/4 column windows.
// Its batch-local region index is rowIdx + rowsValid * (colIdx - colStart),
// as in the CUDA core, so the host adds the batch's nonce_base.
// ---------------------------------------------------------------------------

struct FoldArgs {
  const int8_t *Ap;   // [m, k] row-major
  const int8_t *Bp;   // [n, k] row-major (B transposed)
  uint32_t k;
  uint32_t rowsValid;
  uint32_t colStart;  // valid column offset index of the batch's first column
  uint32_t colBatch;
  uint32_t band;      // row windows walked together, so their A' stays in the L2
  TranscriptTest test;
  HitBuf *hits;
  uint32_t *allTr;    // debug: every region's transcript, [regions][16], or null
};

// Which window work-group g covers. Bands of `band` row windows: inside a band
// the walk goes down the rows first, so a column window's B' is read by `band`
// groups in a row while it is hot, and the band's A' stays in the L2 while the
// walk crosses every column window.
inline void window_of(uint32_t g, uint32_t rowWins, uint32_t colWins, uint32_t band,
                      uint32_t &wr, uint32_t &wc) {
  const uint32_t per = band * colWins;
  const uint32_t b = g / per;
  const uint32_t r0 = b * band;
  const uint32_t br = rowWins - r0 < band ? rowWins - r0 : band;
  const uint32_t idx = g - b * per;
  wc = idx / br;
  wr = r0 + idx % br;
}

// A region's transcript is complete: hash it, test it, append it on a hit.
inline void region_epilogue(const uint32_t words[16], uint32_t local, const TranscriptTest &t,
                            HitBuf *hits, uint32_t *allTr) {
  if (allTr)
    for (int j = 0; j < 16; j++) allTr[(size_t)local * 16 + j] = words[j];
  uint32_t h[8];
  pearl_b3::hash64(t.key, true, words, h);
  if (!pearl_b3::meets(h, t.target_w, t.hash_big_endian)) return;
  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      cnt(hits->count);
  const uint32_t n = cnt.fetch_add(1u);
  if (n >= PEARL_MAX_HITS) return;
  hits->index[n] = local;
  for (int i = 0; i < 8; i++) hits->hash[n * 8 + i] = h[i];
  for (int j = 0; j < 16; j++) hits->transcript[n * 16 + j] = words[j];
}

inline int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
  return c + (int32_t)(int8_t)a * (int32_t)(int8_t)b
         + (int32_t)(int8_t)(a >> 8) * (int32_t)(int8_t)(b >> 8)
         + (int32_t)(int8_t)(a >> 16) * (int32_t)(int8_t)(b >> 16)
         + (int32_t)(int8_t)(a >> 24) * (int32_t)(int8_t)(b >> 24);
}

// The plain fold. One work-group of 256 a window: 32 work-items a region, each
// holding 4 rows by 2 columns of it (8 accumulators). Each 128-k chunk is
// staged into local memory, 32 rows of A' and 64 of B'. A work-item XORs its 8
// sums into one word a chunk, and since XOR is linear the region's word is the
// XOR of its 32 work-items' words: they are combined once, at the end, not at
// every chunk. k = 2048 and rank = 128 (16 chunks, one transcript word each), as
// the host checks.
constexpr uint32_t kDotStride = 33;  // words a staged row: 32 and one of padding

inline void fold_dot(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> As(32 * kDotStride, h), Bs(64 * kDotStride, h);
    sycl::local_accessor<uint32_t, 1> tp(16 * kThreads, h), trs(8 * 16, h);
    const FoldArgs A = a;
    h.parallel_for(sycl::nd_range<1>((size_t)groups * kThreads, kThreads), [=](sycl::nd_item<1> it) {
      const uint32_t w = (uint32_t)it.get_local_id(0);
      const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
      uint32_t wr, wc;
      window_of((uint32_t)it.get_group(0), rowWins, colWins, A.band, wr, wc);
      const uint32_t k = A.k;
      const int8_t *Ab = A.Ap + (size_t)(32u * wr) * k;
      const int8_t *Bb = A.Bp + (size_t)(16u * A.colStart + 64u * wc) * k;
      const uint32_t q = w >> 5, e = w & 31u;
      const uint32_t rbase = (e >> 3) * 8u + (q & 1u) * 4u;
      const uint32_t cbase = (e & 7u) * 8u + (q >> 1) * 2u;
      int32_t acc[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
      uint32_t t[16];
      for (int i = 0; i < 16; i++) t[i] = 0;
      for (uint32_t c = 0; c < 16u; c++) {
        const uint32_t kc = c * 128u;
        {
          const uint32_t r = w >> 3, p = w & 7u;
          const uint32_t *src = reinterpret_cast<const uint32_t *>(Ab + (size_t)r * k + kc + p * 16u);
          for (int i = 0; i < 4; i++) As[r * kDotStride + p * 4u + i] = src[i];
        }
        for (uint32_t s = 0; s < 2u; s++) {
          const uint32_t idx = w + kThreads * s, r = idx >> 3, p = idx & 7u;
          const uint32_t *src = reinterpret_cast<const uint32_t *>(Bb + (size_t)r * k + kc + p * 16u);
          for (int i = 0; i < 4; i++) Bs[r * kDotStride + p * 4u + i] = src[i];
        }
        sycl::group_barrier(it.get_group());
        for (uint32_t kw = 0; kw < 32u; kw++) {
          uint32_t av[4], bv[2];
          for (int i = 0; i < 4; i++) av[i] = As[(rbase + i) * kDotStride + kw];
          for (int j = 0; j < 2; j++) bv[j] = Bs[(cbase + j) * kDotStride + kw];
          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++) acc[i][j] = dot4(av[i], bv[j], acc[i][j]);
        }
        sycl::group_barrier(it.get_group());
        uint32_t x = 0;
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 2; j++) x ^= (uint32_t)acc[i][j];
        // Chunk c's word lands in t[c]: a shift, so every index stays a constant.
        for (int i = 0; i < 15; i++) t[i] = t[i + 1];
        t[15] = x;
      }
      for (int j = 0; j < 16; j++) tp[j * kThreads + w] = t[j];
      sycl::group_barrier(it.get_group());
      if (e < 16u) {
        uint32_t x = 0;
        for (uint32_t i = 0; i < 32u; i++) x ^= tp[e * kThreads + q * 32u + i];
        trs[q * 16u + e] = x;
      }
      sycl::group_barrier(it.get_group());
      if (e == 0) {
        uint32_t words[16];
        for (int j = 0; j < 16; j++) words[j] = trs[q * 16u + j];
        const uint32_t local = (2u * wr + (q & 1u)) + A.rowsValid * (4u * wc + (q >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// The XMX fold.
//
// DPAS, int8 x int8 -> int32, 8 rows by SG columns by 32 k, in the register
// layout cl_intel_subgroup_matrix_multiply_accumulate fixes:
//   A  SG16: a short per row, lane l holding k = 2l, 2l+1 of each of the 8 rows
//      SG8:  an int per row, lane l holding k = 4l .. 4l+3
//   B  the lane's column, all 32 k packed into 8 dwords (k = 4d .. 4d+3 in d)
//   C  the lane's column, rows 0..7
// ---------------------------------------------------------------------------

using ::v8i;
using ::v8s;

template <int SG> struct AFrag;
template <> struct AFrag<16> { typedef v8s type; };
template <> struct AFrag<8> { typedef v8i type; };


// The extension's reference semantics, with sub-group shuffles: what the CPU
// device runs, and what the hardware DPAS must agree with.
template <int SG>
inline v8i dpas_emu(sycl::sub_group sg, typename AFrag<SG>::type a, v8i b, v8i c) {
  v8i r = c;
  for (int row = 0; row < 8; row++) {
    int acc = r[row];
    if constexpr (SG == 16) {
      for (int l = 0; l < 16; l += 2) {
        const uint32_t lo = (uint16_t)sycl::select_from_group(sg, a[row], l);
        const uint32_t hi = (uint16_t)sycl::select_from_group(sg, a[row], l + 1);
        acc = dot4(lo | (hi << 16), (uint32_t)b[l / 2], acc);
      }
    } else {
      for (int l = 0; l < 8; l++)
        acc = dot4((uint32_t)sycl::select_from_group(sg, a[row], l), (uint32_t)b[l], acc);
    }
    r[row] = acc;
  }
  return r;
}

// HW: use the hardware instruction on the architectures that have it at this
// sub-group size, when the binary carries it (PEARL_SYCL_XMX_HW). Otherwise, and
// on every other device, the reference code.
template <int SG, bool HW>
inline v8i dpas(sycl::sub_group sg, typename AFrag<SG>::type a, v8i b, v8i c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_HW)
  if constexpr (HW) {
    namespace syclex = sycl::ext::oneapi::experimental;
    using arch = syclex::architecture;
    v8i r = c;
    if constexpr (SG == 16) {
      syclex::if_architecture_is<arch::intel_gpu_pvc, arch::intel_gpu_bmg_g21,
                                 arch::intel_gpu_bmg_g31, arch::intel_gpu_lnl_m,
                                 arch::intel_gpu_ptl_h, arch::intel_gpu_ptl_u>([&]() {
        r = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, a, b, c, 0x33);
      }).otherwise([&]() { r = dpas_emu<16>(sg, a, b, c); });
    } else {
      syclex::if_architecture_is<arch::intel_gpu_acm_g10, arch::intel_gpu_acm_g11,
                                 arch::intel_gpu_acm_g12>([&]() {
        r = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, a, b, c, 0x33);
      }).otherwise([&]() { r = dpas_emu<8>(sg, a, b, c); });
    }
    return r;
  }
#endif
  return dpas_emu<SG>(sg, a, b, c);
}

// A window is split by rows over 4/RB sub-groups, each holding RB 8-row blocks by
// all 64 columns: 64/SG column blocks of DPAS accumulators, 8 v8i either way
// (SG16 RB2, SG8 RB1). Per chunk, a lane XORs its accumulators into two words,
// one for each region its column belongs to (rows 0-3 of each block are row
// offset 0, rows 4-7 row offset 4; lane j's column offset is j & 6); lane
// shuffles (lane^1, and lane^8 at SG16) gather a region's columns, and lane q < 8
// writes region q's word. The sub-groups' words are XORed at the end.
template <int SG, int RB, bool HW>
inline void fold_xmx(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  constexpr int S = 4 / RB;       // sub-groups a window
  constexpr int NCB = 64 / SG;    // DPAS column blocks
  constexpr int WG = S * SG;
  typedef typename AFrag<SG>::type afrag;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part(S * 8 * 16, h);
    const FoldArgs A = a;
    h.parallel_for(sycl::nd_range<1>((size_t)groups * WG, WG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
      sycl::sub_group sg = it.get_sub_group();
      const uint32_t lane = (uint32_t)sg.get_local_linear_id();
      const uint32_t sgi = (uint32_t)sg.get_group_linear_id();
      const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
      uint32_t wr, wc;
      window_of((uint32_t)it.get_group(0), rowWins, colWins, A.band, wr, wc);
      const uint32_t k = A.k;
      const int8_t *Ab = A.Ap + (size_t)(32u * wr + sgi * 8u * RB) * k;
      const int8_t *Bb = A.Bp + (size_t)(16u * A.colStart + 64u * wc + lane) * k;
      v8i c[RB][NCB];
      for (int rb = 0; rb < RB; rb++)
        for (int cb = 0; cb < NCB; cb++) c[rb][cb] = 0;
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t k0 = ch * 128u + s * 32u;
          afrag af[RB];
          for (int rb = 0; rb < RB; rb++)
            for (int r = 0; r < 8; r++) {
              const int8_t *p = Ab + (size_t)(8 * rb + r) * k + k0;
              if constexpr (SG == 16)
                af[rb][r] = *reinterpret_cast<const short *>(p + 2u * lane);
              else
                af[rb][r] = *reinterpret_cast<const int *>(p + 4u * lane);
            }
          // B in groups of BG column blocks: four at SG16, two at SG8, where a
          // 32-byte GRF makes each block twice the registers. Keeps both to 128
          // GRF without a spill.
          constexpr int BG = SG == 16 ? 4 : 2;
          for (int g = 0; g < NCB / BG; g++) {
            v8i bf[BG];
            for (int i = 0; i < BG; i++) {
              const uint32_t *p = reinterpret_cast<const uint32_t *>(
                  Bb + (size_t)(SG * (BG * g + i)) * k + k0);
              for (int d = 0; d < 8; d++) bf[i][d] = (int)p[d];
            }
            for (int rb = 0; rb < RB; rb++)
              for (int i = 0; i < BG; i++)
                c[rb][BG * g + i] = dpas<SG, HW>(sg, af[rb], bf[i], c[rb][BG * g + i]);
          }
        }
        uint32_t x0 = 0, x1 = 0;
        for (int rb = 0; rb < RB; rb++)
          for (int cb = 0; cb < NCB; cb++) {
            for (int r = 0; r < 4; r++) x0 ^= (uint32_t)c[rb][cb][r];
            for (int r = 4; r < 8; r++) x1 ^= (uint32_t)c[rb][cb][r];
          }
        x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
        x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
        if constexpr (SG == 16) {
          x0 ^= sycl::permute_group_by_xor(sg, x0, 8u);
          x1 ^= sycl::permute_group_by_xor(sg, x1, 8u);
        }
        if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
      }
      sycl::group_barrier(it.get_group());
      if (sgi == 0 && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        for (int j = 0; j < 16; j++) {
          uint32_t x = 0;
          for (int s = 0; s < S; s++) x ^= part[((uint32_t)s * 8u + qr) * 16u + (uint32_t)j];
          words[j] = x;
        }
        const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
    });
  });
}

}  // namespace pearl_sycl

#endif  // PEARL_SYCL_KERNELS_HPP
