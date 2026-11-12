// SYCL kernels for the Pearl core on Intel GPUs (and, for testing, the SYCL CPU
// device). Ports of the draw kernels in ../src/pearl_kernel.cu, which they must
// match bit for bit, plus two folds of our own:
//
//   fold_dot  plain int8 dot products in local memory. Needs no XMX, so it runs
//             on any device the build has kernels for. The reference fold.
//   fold_xmx  the XMX matrix engines through DPAS, at sub-group 16 (Xe2, Xe3,
//             Xe-HPC) or 8 (Xe-HPG): the hardware instruction on those GPUs, and
//             everywhere else the reference code from the
//             cl_intel_subgroup_matrix_multiply_accumulate specification, which
//             the CPU device runs. Same source either way.
//
// The hardware instruction is that extension's OpenCL built-in,
// intel_sub_group_i8_i8_matrix_mad_k32, called by name: Intel's compiler knows
// the name whether the kernel reaches it as OpenCL C or, as here, as SPIR-V. A
// build reaches it one of two ways:
//
//   PEARL_SYCL_XMX_JIT  (./build.sh --jit, what a release carries) The driver
//             compiles each kernel itself, the first time it is used. A fold
//             with HW set calls the built-in and nothing else, so it compiles
//             only where the built-in exists. Each kernel is a device image of
//             its own (-fsycl-device-code-split=per_kernel), and the host
//             launches a hardware fold only on a device whose architecture has
//             XMX at that sub-group size, so no other device ever sees one.
//   PEARL_SYCL_XMX_HW   (./build.sh, kernels prebuilt by ocloc for named GPUs)
//             The choice is made inside the kernel, per target, with
//             if_architecture_is.
//
// SPIR-V has an instruction for it too (SPV_INTEL_subgroup_matrix_multiply_
// accumulate, which this file used until November 2026), but a driver from
// before 2025 refuses a module that names it: Intel's 24.35, the newest built
// for Ubuntu 22.04, did on an Arc A750, and took the built-in.
//
// Both folds compute the same thing as pearl_tile_fold_* in the CUDA core: for
// every region of a batch, the cumulative int32 tile over k, XORed into one
// transcript word at each 128-k boundary, the transcript hashed under a_seed,
// and a hit list appended to when the hash meets the target.

#ifndef PEARL_SYCL_KERNELS_HPP
#define PEARL_SYCL_KERNELS_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <stdint.h>
#include <utility>

#if defined(PEARL_SYCL_XMX_HW) && defined(PEARL_SYCL_XMX_JIT)
#error "PEARL_SYCL_XMX_HW (prebuilt kernels) and PEARL_SYCL_XMX_JIT (the driver compiles them) are two builds"
#endif
#if defined(PEARL_SYCL_XMX_HW)
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#endif

#include "../src/pearl_config.h"
#include "pearl_sycl_blake3.h"

// The DPAS vector types, and the built-in: 8 rows of A (a word a lane at
// sub-group 16, a dword at 8), the lane's column of B, the accumulator. It must
// be declared at global scope, so that its mangled name is the one OpenCL C
// gives it (short8 or int8, int8, int8), which is how the compiler finds it.
typedef int v8i __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
// A v8i as it lies in an operand: at a dword address, so one load takes it.
typedef int v8iu __attribute__((ext_vector_type(8), aligned(4)));
typedef int v4iu __attribute__((ext_vector_type(4), aligned(4)));  // and half of one
#if defined(__SYCL_DEVICE_ONLY__) && (defined(PEARL_SYCL_XMX_HW) || defined(PEARL_SYCL_XMX_JIT))
extern SYCL_EXTERNAL v8i intel_sub_group_i8_i8_matrix_mad_k32(v8s, v8i, v8i);
extern SYCL_EXTERNAL v8i intel_sub_group_i8_i8_matrix_mad_k32(v8i, v8i, v8i);
#endif
// cl_intel_subgroup_split_matrix_multiply_accumulate (DPASW on Alchemist): a pair of sub-groups shares A, each
// giving half of its 8 rows (4 dwords a lane), and each gets all 8 rows times its own B.
typedef int v4i __attribute__((ext_vector_type(4)));
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
extern SYCL_EXTERNAL v8i intel_sub_group_i8_i8_split_matrix_mad_k32(v4i, v8i, v8i);
#endif
// Four dwords a lane: what a sub-group block read of 128 bytes gives (the ring fold).
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
// SPV_INTEL_split_barrier: a work-group barrier in two halves, arrive and wait. The ring fold's hardware
// kernel uses it in the build the driver compiles (build.sh passes the extension to the translator).
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
extern SYCL_EXTERNAL void __spirv_ControlBarrierArriveINTEL(int, int, int);
extern SYCL_EXTERNAL void __spirv_ControlBarrierWaitINTEL(int, int, int);
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

// Where a noised operand's bytes are stored.
//
//   kRowMajor  [row][k], as the proofs' operands are. What fold_dot and
//              fold_xmx read.
//   kFragA     A' in the order fold_xmxf's DPAS takes it at sub-group 8. Rows
//              in blocks of 16 (a sub-group's), a block `stride` dwords from
//              the next; inside it [k-step of 32][8 rows][lane][row][4 k]: a
//              lane's v8i of a DPAS A operand (its 4 k of each of 8 rows) is 32
//              bytes in a row, and the 8 lanes' are 256 in a row.
//   kFragB     B' the same way. Columns in blocks of 64 (a window's); inside
//              one [k-step of 32][column][32 k]: a lane's v8i of a B operand
//              (its column's 32 k) is 32 bytes in a row, and a DPAS's 8
//              columns are 256 in a row.
//
// A block's own bytes are 16 or 64 rows of k; `stride` may be more, which
// leaves a gap after each block (FoldArgs's aStride and bStride).
//   kFragAT, kFragBT  the same blocks and fragments with the 32 dwords of each
//              half fragment (128 bytes) in the order a sub-group block read
//              hands them out: dword d * 8 + l goes to lane l as its d-th. So
//              one block read of 128 bytes gives each lane its half of the
//              fragment, as the vector loads of kFragA and kFragB do. What the
//              ring fold (fold_xmxr) reads.
enum : uint32_t { kRowMajor = 0, kFragA = 1, kFragB = 2, kFragAT = 3, kFragBT = 4 };
struct Layout {
  uint32_t kind;
  uint32_t stride;  // dwords from one block to the next (kFragA, kFragB, kFragAT, kFragBT)
};

// The dword index of row r's k-word kw (its k = 4 kw .. 4 kw + 3).
inline uint64_t frag_a_word(uint32_t stride, uint32_t r, uint32_t kw) {
  return (uint64_t)(r >> 4) * stride + (uint64_t)(kw >> 3) * 128u + ((r >> 3) & 1u) * 64u
         + (kw & 7u) * 8u + (r & 7u);
}
inline uint64_t frag_b_word(uint32_t stride, uint32_t r, uint32_t kw) {
  return (uint64_t)(r >> 6) * stride + (uint64_t)(kw >> 3) * 512u + (r & 63u) * 8u + (kw & 7u);
}
// An A fragment's lane is kw & 7 and its dword is the row (r & 7): half (r & 7) >> 2, place (r & 3).
inline uint64_t frag_at_word(uint32_t stride, uint32_t r, uint32_t kw) {
  return (uint64_t)(r >> 4) * stride + (uint64_t)(kw >> 3) * 128u + ((r >> 3) & 1u) * 64u
         + ((r & 7u) >> 2) * 32u + (r & 3u) * 8u + (kw & 7u);
}
// A B fragment's lane is the column (r & 7) and its dword is kw & 7: half (kw & 7) >> 2, place (kw & 3).
inline uint64_t frag_bt_word(uint32_t stride, uint32_t r, uint32_t kw) {
  return (uint64_t)(r >> 6) * stride + (uint64_t)(kw >> 3) * 512u + ((r & 63u) >> 3) * 64u
         + ((kw & 7u) >> 2) * 32u + (kw & 3u) * 8u + (r & 7u);
}
inline uint64_t layout_word(const Layout &l, uint32_t r, uint32_t kw, uint32_t k) {
  if (l.kind == kFragA) return frag_a_word(l.stride, r, kw);
  if (l.kind == kFragB) return frag_b_word(l.stride, r, kw);
  if (l.kind == kFragAT) return frag_at_word(l.stride, r, kw);
  if (l.kind == kFragBT) return frag_bt_word(l.stride, r, kw);
  return (uint64_t)r * (k >> 2) + kw;
}

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
// `lay` says where each dword of the result is stored (Layout).
inline void materialize16(sycl::queue &q, const int8_t *base, const int8_t *dense,
                          const uint32_t *perm, int8_t *out, uint32_t rows, uint32_t k_log2,
                          uint32_t rank, uint64_t read_vecs, uint32_t fill_word, Layout lay) {
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
    uint32_t *o = reinterpret_cast<uint32_t *>(out);
    for (int w = 0; w < 4; w++) o[layout_word(lay, r, (kk >> 2) + (uint32_t)w, 1u << k_log2)] = ow[w];
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
  uint32_t aStride;   // fold_xmxf: bytes from one 16-row block of A' to the next
  uint32_t bStride;   // fold_xmxf: bytes from one 64-column block of B' to the next
  uint32_t zero;      // fold_xmxf: 0. The kernel cannot know it, which is the point (see the fold)
  uint32_t wpg;       // fold_xmxr: windows a work-group, all in one column window (divides band and rowsValid / 2)
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
//
// Lane j's A word comes to lane i by an XOR shuffle (j = i ^ s), not by
// select_from_group(sg, x, j): on an Arc A750 (compute runtime 26.40), a
// select_from_group whose index is the same in every lane returned wrong words
// (goal-arc7, 12 Nov: every region of every reference fold wrong on the card,
// right on the CPU device; the XOR form matches the hardware DPAS bit for bit,
// and the reference folds now pass on both).
template <int SG>
inline v8i dpas_emu(sycl::sub_group sg, typename AFrag<SG>::type a, v8i b, v8i c) {
  const uint32_t lane = (uint32_t)sg.get_local_linear_id();
  v8i r = c;
  for (int row = 0; row < 8; row++) {
    int acc = r[row];
    if constexpr (SG == 16) {
      // Lane j holds k = 2 j, 2 j + 1 of the row: the low or high half of B's dword j / 2.
      for (uint32_t s = 0; s < 16u; s++) {
        const uint32_t j = lane ^ s;
        const uint32_t aw = (uint16_t)sycl::permute_group_by_xor(sg, a[row], s);
        const uint32_t bw = ((uint32_t)b[j >> 1] >> (16u * (j & 1u))) & 0xffffu;
        acc = dot4(aw, bw, acc);
      }
    } else {
      for (uint32_t s = 0; s < 8u; s++)
        acc = dot4((uint32_t)sycl::permute_group_by_xor(sg, a[row], s), (uint32_t)b[lane ^ s], acc);
    }
    r[row] = acc;
  }
  return r;
}

// HW: the hardware instruction. In a build whose kernels the driver compiles
// (PEARL_SYCL_XMX_JIT) that is all a HW fold has, and the host launches it only
// where it exists. In a prebuilt one (PEARL_SYCL_XMX_HW) it is chosen here, on
// the architectures that have it at this sub-group size. Otherwise, and on
// every other device, the reference code.
template <int SG, bool HW>
inline v8i dpas(sycl::sub_group sg, typename AFrag<SG>::type a, v8i b, v8i c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return ::intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_HW)
  if constexpr (HW) {
    namespace syclex = sycl::ext::oneapi::experimental;
    using arch = syclex::architecture;
    v8i r = c;
    if constexpr (SG == 16) {
      syclex::if_architecture_is<arch::intel_gpu_pvc, arch::intel_gpu_bmg_g21,
                                 arch::intel_gpu_bmg_g31, arch::intel_gpu_lnl_m,
                                 arch::intel_gpu_ptl_h, arch::intel_gpu_ptl_u>([&]() {
        r = ::intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
      }).otherwise([&]() { r = dpas_emu<16>(sg, a, b, c); });
    } else {
      syclex::if_architecture_is<arch::intel_gpu_acm_g10, arch::intel_gpu_acm_g11,
                                 arch::intel_gpu_acm_g12>([&]() {
        r = ::intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
      }).otherwise([&]() { r = dpas_emu<8>(sg, a, b, c); });
    }
    return r;
  }
#endif
  return dpas_emu<SG>(sg, a, b, c);
}

// The split DPAS (goal-arc7 experiment): hardware only. The reference has no partner's half to read, so the
// HW = false form returns the accumulator unchanged; nothing but the card may run a fold that uses it.
template <bool HW> inline v8i split_dpas(v4i a, v8i b, v8i c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return ::intel_sub_group_i8_i8_split_matrix_mad_k32(a, b, c);
#endif
  (void)a;
  (void)b;
  return c;
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

// ---------------------------------------------------------------------------
// The fragment fold (xmx8f): the XMX fold for Alchemist's sub-group 8, with
// its operands stored the way DPAS takes them (kFragA, kFragB) and loaded one
// fragment a message.
//
// fold_xmx loads a lane's dwords of A' one row at a time and its 32 bytes of B'
// in two halves: 3 load messages a DPAS, and the matrix engines wait for them
// 85% of the time (README.md, "The A750's ceiling"). Here:
//
//   - A sub-group holds 16 rows by 64 columns, 16 accumulators, so a window
//     is 2 sub-groups, not 4, and each B' fragment loaded feeds 2 DPAS. That
//     needs the 256-register mode, which the hardware kernel asks for by a
//     kernel property; an EU then runs 4 threads, not 8, and DPAS alone is as
//     fast as before.
//   - A lane loads a fragment of A', its v8i, with one load, and a fragment
//     of B' with two, 16 bytes each: 2 messages for A' and 16 for B' a k-step
//     of 16 DPAS. The card serves a 128-byte load message about four times as
//     fast as a 256-byte one, so B', four fifths of the bytes, goes in halves
//     (README.md, "Where its time goes now").
//   - The 8 lanes' loads of a fragment are 256 bytes in a row, 4 cache lines.
//   - A k-step sends all 18 loads first and then runs its 16 DPAS as one
//     run. Left alone, the compiler moves each load down to the DPAS that
//     reads it and keeps two fragments of B' in flight; sent together, 16 are.
//     To hold the order, a word of every fragment loaded is ANDed with
//     FoldArgs::zero, which is 0 and which the compiler cannot know, and ORed
//     into A: no DPAS can start before every load of its k-step has landed.
//
// The readout is fold_xmx's: both 8-row blocks XOR into the same two words.
// ---------------------------------------------------------------------------

// f(0) .. f(N - 1), written out when compiling: no loop is left for a compiler
// to keep. The fold's k-step is written with it. Kept as loops, its fragments
// and accumulators stay arrays in memory and it runs at a twentieth of its
// rate: oneAPI's compiler kept them, the open-source one unrolled them.
template <typename F, size_t... I> inline void each_of(F &f, std::index_sequence<I...>) { (f((int)I), ...); }
template <int N, typename F> inline void each(F f) { each_of(f, std::make_index_sequence<N>{}); }

// A lane's 32 bytes of a fragment at p, as two loads of 16.
inline v8i load_halves(const int8_t *p) {
  const v4iu lo = *reinterpret_cast<const v4iu *>(p);
  const v4iu hi = *reinterpret_cast<const v4iu *>(p + 16);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

#if defined(PEARL_A5)
// Timing variants of the fragment fold (goal-arc5), built only with PEARL_SYCL_EXTRA=-DPEARL_A5=N.
#include "pearl_sycl_a5.hpp"
#endif

template <int SG, int RB, bool HW> struct FoldXmxF {
  static_assert(SG == 8, "the fragment layouts are sub-group 8's");
  static constexpr int S = 4 / RB;     // sub-groups a window
  static constexpr int NCB = 64 / SG;  // DPAS column blocks
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> part;

  // The sub-group size, as a kernel property since the kernel has another:
  // the hardware kernel with 16 accumulators needs the 256-register mode. The
  // reference kernel is left alone: the CPU device has no such mode.
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (HW && RB > 1)
      return syclex::properties{syclex::sub_group_size<SG>, sycl::ext::intel::experimental::grf_size<256>};
    else
      return syclex::properties{syclex::sub_group_size<SG>};
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgi = (uint32_t)sg.get_group_linear_id();
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    window_of((uint32_t)it.get_group(0), rowWins, colWins, A.band, wr, wc);
    // The sub-group's rows in their 16-row block of A' and the window's block
    // of B', each at this lane's 32 bytes of a fragment.
    const uint32_t row0 = 32u * wr + sgi * 8u * RB;
    const int8_t *Ab = A.Ap + (size_t)(row0 >> 4) * A.aStride + (size_t)((row0 >> 3) & 1u) * 256u + 32u * lane;
    const int8_t *Bb = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride + 32u * lane;
    v8i c[RB][NCB];
    each<RB>([&](int rb) { each<NCB>([&](int cb) { c[rb][cb] = 0; }); });
    for (uint32_t ch = 0; ch < 16u; ch++) {
      for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t ks = ch * 4u + s;
        v8i af[RB], bf[NCB];
        each<RB>([&](int rb) {
          af[rb] = *reinterpret_cast<const v8iu *>(Ab + (size_t)ks * 512u + (size_t)rb * 256u);
        });
        each<NCB>([&](int cb) { bf[cb] = load_halves(Bb + (size_t)ks * 2048u + (size_t)cb * 256u); });
        // Every DPAS of the k-step waits for every load of it.
        int wait = 0;
        each<RB>([&](int rb) { wait ^= af[rb][0]; });
        each<NCB>([&](int cb) { wait ^= bf[cb][0]; });
        wait &= (int)A.zero;
        each<RB>([&](int rb) { af[rb][0] |= wait; });
        each<NCB>([&](int cb) {
          each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });
        });
      }
      uint32_t x0 = 0, x1 = 0;
      each<RB>([&](int rb) {
        each<NCB>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
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
  }
};

template <int SG, int RB, bool HW>
inline void fold_xmxf(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
#if defined(PEARL_A5)
  fold_xmxf_a5<SG, RB, HW>(q, a, groups);
  return;
#endif
  constexpr int S = 4 / RB;
  constexpr int WG = S * SG;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part(S * 8 * 16, h);
    h.parallel_for(sycl::nd_range<1>((size_t)groups * WG, WG), FoldXmxF<SG, RB, HW>{a, part});
  });
}

// ---------------------------------------------------------------------------
// The ring fold (xmx8r): the fragment fold with B' shared through local
// memory. Added 12 November 2026, from the NVIDIA work: there the fold that
// mines copies each operand tile into shared memory once a block and every
// warp reads it from there, and the rotation fold shares the fragments it
// makes the same way.
//
// In the fragment fold every sub-group loads its window's B' itself, 2,048
// bytes a k-step, and most of it comes from the card's last-level cache, which
// costs the data port time and the card watts. Here a work-group is `wpg`
// windows of one column window (FoldArgs::wpg, 8 on mainnet: 16 sub-groups,
// 256 rows by 64 columns). Each k-step its sub-groups copy the next k-step's
// B' into local memory between them (16 block reads of 128 bytes, one a
// sub-group at 8 windows), and every sub-group reads this k-step's B' from
// local memory. A' is still each sub-group's own, from global memory.
//
//   - A ring of three slots of 2,048 bytes. The k-step's copy goes into the
//     slot after this one's; a slot is written again two k-steps after it was
//     read. A split barrier holds that: each k-step a sub-group waits for
//     the k-step before (every copy into this k-step's slot has landed, and
//     every read of the slot about to be written is done), copies, arrives,
//     and runs its DPAS. So a sub-group may run up to a k-step ahead of the
//     slowest without waiting. What the core runs (VAR 12) loads its copy a
//     k-step early, so the copy's store never waits for memory, and waits
//     with no acquire fence: the release fence before each arrival orders
//     the stores, and local memory has no cache to invalidate.
//   - Operands in the kFragAT and kFragBT layouts, read by sub-group block
//     reads: one message, one address, 128 bytes, the lanes' dwords in place.
//   - The readout is the fragment fold's.
//
// On an A750 (card 5 of the rig, pearl_sycl_check --bench, 20 s, taking turns,
// 12 November): 66.9 to 67.2 TH/s at 2,000 MHz and 189 W, where the fragment
// fold did 62.6 to 62.8 at 1,970 MHz and 189 W (+6.9%). Both are at the card's
// power limit. 16 windows a work-group: 63.5 at 180 W.
//
// HW: the hardware DPAS, block reads and, in the build the driver compiles,
// the split barrier. Otherwise the reference DPAS, plain loads that give each
// lane the same dwords, and a whole work-group barrier where the split one
// arrives: the same results, for the CPU device.
// ---------------------------------------------------------------------------

// IGC's LSC prefetch, called by name: brings a line into the caches with a cache control of its own
// (4: L1 and L3 cached). The ring fold's VAR 256 uses it for A'. A timing build only.
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_RING_PF)
}  // namespace pearl_sycl
extern "C" SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uint4(const __attribute__((opencl_global)) v4u *base,
                                                                    int immElemOff, int cacheOpt);
namespace pearl_sycl {
#endif
template <bool HW> inline void ring_prefetch_l1(const int8_t *p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_RING_PF)
  if constexpr (HW) __builtin_IB_lsc_prefetch_global_uint4((const __attribute__((opencl_global)) v4u *)p, 0, PEARL_RING_PF_CC);
#else
  (void)p;
#endif
}

template <bool HW> inline v4u ring_read_global(const int8_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return __spirv_SubgroupBlockReadINTEL<v4u>((const __attribute__((opencl_global)) uint32_t *)p);
#endif
  const uint32_t *w = reinterpret_cast<const uint32_t *>(p);
  return v4u{w[lane], w[8u + lane], w[16u + lane], w[24u + lane]};
}
template <bool HW> inline v4u ring_read_local(uint32_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return __spirv_SubgroupBlockReadINTEL<v4u>((const __attribute__((opencl_local)) uint32_t *)p);
#endif
  return v4u{p[lane], p[8u + lane], p[16u + lane], p[24u + lane]};
}
// A whole 256-byte fragment in block-read order (kFragBT, kFragAT) as one block read of 8 dwords a lane: the
// same dwords as two 128-byte block reads put side by side, but one value for the compiler.
typedef uint32_t v8u __attribute__((ext_vector_type(8)));
template <bool HW> inline v8i ring_read_global8b(const int8_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return (v8i)__spirv_SubgroupBlockReadINTEL<v8u>((const __attribute__((opencl_global)) uint32_t *)p);
#endif
  const uint32_t *w = reinterpret_cast<const uint32_t *>(p);
  v8i r;
  for (int i = 0; i < 8; i++) r[i] = (int)w[8u * (uint32_t)i + lane];
  return r;
}
template <bool HW> inline v8i ring_read_local8b(uint32_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return (v8i)__spirv_SubgroupBlockReadINTEL<v8u>((const __attribute__((opencl_local)) uint32_t *)p);
#endif
  v8i r;
  for (int i = 0; i < 8; i++) r[i] = (int)p[8u * (uint32_t)i + lane];
  return r;
}
// A lane's 32 bytes of a fragment stored in fragment order (kFragB) in local memory: one 8-dword read.
inline v8i ring_read_local8(uint32_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__)
  return *(const __attribute__((opencl_local)) v8iu *)(p + 8u * lane);
#else
  return *reinterpret_cast<const v8iu *>(p + 8u * lane);
#endif
}
// goal-arc7 timing builds (message-count test): a fragment read as block reads of W dwords a lane, so
// 256 / (32 W) messages of 32 W bytes. W 4 is the fold's form; 2 and 1 send the same bytes in 2 and 4 times
// the messages. Every lane gets the same dwords as ring_read_local twice.
typedef uint32_t v2u __attribute__((ext_vector_type(2)));
template <bool HW, int W, typename P> inline v8i frag_read_w(P *p, uint32_t lane) {
  v8i r;
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    if constexpr (W == 4) {
      const v4u lo = __spirv_SubgroupBlockReadINTEL<v4u>(p), hi = __spirv_SubgroupBlockReadINTEL<v4u>(p + 32);
      return (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
    } else if constexpr (W == 2) {
      const v2u a = __spirv_SubgroupBlockReadINTEL<v2u>(p), b = __spirv_SubgroupBlockReadINTEL<v2u>(p + 16);
      const v2u c = __spirv_SubgroupBlockReadINTEL<v2u>(p + 32), d = __spirv_SubgroupBlockReadINTEL<v2u>(p + 48);
      const v4u lo = __builtin_shufflevector(a, b, 0, 1, 2, 3), hi = __builtin_shufflevector(c, d, 0, 1, 2, 3);
      return (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
    } else {
      each<8>([&](int i) { r[i] = (int)__spirv_SubgroupBlockReadINTEL<uint32_t>(p + 8 * i); });
      return r;
    }
  }
#endif
  for (int i = 0; i < 8; i++) r[i] = (int)p[8u * (uint32_t)i + lane];
  return r;
}
#if defined(__SYCL_DEVICE_ONLY__)
#define PEARL_GPTR(p) ((const __attribute__((opencl_global)) uint32_t *)(p))
#define PEARL_LPTR(p) ((const __attribute__((opencl_local)) uint32_t *)(p))
#else
#define PEARL_GPTR(p) ((const uint32_t *)(p))
#define PEARL_LPTR(p) ((const uint32_t *)(p))
#endif
#ifndef PEARL_RING_BW
#define PEARL_RING_BW 4  // B' read width (dwords a lane a message)
#endif
#ifndef PEARL_RING_AW
#define PEARL_RING_AW 4  // A' read width
#endif
#ifndef PEARL_RING_NB
#define PEARL_RING_NB 8  // B' fragments read a k-step (fewer: the DPAS reuse them; timing only)
#endif
#ifndef PEARL_RING_XR
#define PEARL_RING_XR 0  // extra local-memory fragments read a k-step into the gate (timing only)
#endif
#ifndef PEARL_RING_XW
#define PEARL_RING_XW 4  // their width
#endif

template <bool HW> inline void ring_write_local(uint32_t *p, v4u v, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    __spirv_SubgroupBlockWriteINTEL((__attribute__((opencl_local)) uint32_t *)p, v);
    return;
  }
#endif
  p[lane] = v[0];
  p[8u + lane] = v[1];
  p[16u + lane] = v[2];
  p[24u + lane] = v[3];
}

// VAR: 1 runs a chunk's readout after the next k-step's arrival and loads, not before them; 2 (timing
// only, the probe's r-fold-v8 row) loads A' as one vector a fragment, which wants kFragA addresses;
// 4 loads the copy a k-step early (its bytes wait in registers, so the copy's store never waits for
// memory); 8 waits at the barrier with no acquire fence; 16 keeps each lane's 16 words in registers and
// combines a window's two sub-groups through the ring after the last k-step, so the work-group asks for no
// local memory beyond the ring (see fold_xmxr: more than 32 KB of local memory on an Xe-core costs 10%).
// VAR 512: a 2D work-group, wpg / 2 row windows by 2 column windows. The ring holds both column
// windows' B' (4 KB a slot), and the two windows of a row read the same A' at the same time, so the
// first cache level serves the second. The words go into local memory by XOR atomics, a window's 128
// words shared by its two sub-groups, so the group asks for 16 KB at wpg 8.
// goal-arc6 timing variants: VAR 2048 loads A' a k-step ahead into registers; VAR 4096 (with 4) drops the
// barrier for counters in local memory, a ring of PEARL_RING_SLOTS slots, so a sub-group may run up to
// slots - 2 k-steps ahead of the slowest; VAR 8192 sends a k-step's A' loads before its wait.
#ifndef PEARL_RING_SLOTS
#define PEARL_RING_SLOTS 5
#endif
template <bool HW, int VAR = 0> struct FoldXmxR {
  // VAR 1024: a slot holds two k-steps of one column window's B', and the group meets once every two
  // k-steps. Its words go into local memory by XOR atomics, as VAR 512's.
  static constexpr bool kWide = (VAR & (512 | 1024)) != 0;
  static constexpr bool kCnt = HW && (VAR & 4096) && (VAR & 4);
  static constexpr uint32_t kSlots = kCnt ? PEARL_RING_SLOTS : (VAR & 32768) ? 4u : 3u;
  static constexpr uint32_t kSlotWords = kWide ? 1024u : 512u;  // B' a slot: 64 columns by 32 k, or twice that
  static constexpr uint32_t kBlocks = kWide ? 32u : 16u;       // 128-byte blocks a slot
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> part;  // 2 wpg sub-groups x 8 regions x 16 words
  sycl::local_accessor<uint32_t, 1> ring;  // 3 slots

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (HW)
      return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
    else
      return syclex::properties{syclex::sub_group_size<8>};
  }

  void arrive(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) {
#ifdef PEARL_RING_NOREL
      __spirv_ControlBarrierArriveINTEL(2, 2, 0);  // goal-arc7 timing: no release fence
#else
      __spirv_ControlBarrierArriveINTEL(2, 2, 0x104);  // work-group; work-group memory, release
#endif
      return;
    }
#endif
    sycl::group_barrier(it.get_group());
  }
  void wait(sycl::nd_item<1>) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) {
      if constexpr (VAR & 8) __spirv_ControlBarrierWaitINTEL(2, 2, 0);
      else __spirv_ControlBarrierWaitINTEL(2, 2, 0x102);  // work-group; work-group memory, acquire
    }
#endif
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();  // 0 .. 2 wpg - 1
    const uint32_t nsg = 2u * A.wpg;
    const uint32_t sgi = sgl & 1u, widx = sgl >> 1;
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc, jc = 0u;
    if constexpr (VAR & 512) {
      uint32_t wch;
      jc = widx & 1u;
      window_of((uint32_t)it.get_group(0) * (A.wpg / 2u) + (widx >> 1), rowWins, colWins / 2u, A.band, wr, wch);
      wc = 2u * wch + jc;
    } else {
      window_of((uint32_t)it.get_group(0) * A.wpg + widx, rowWins, colWins, A.band, wr, wc);
    }
    const uint32_t row0 = 32u * wr + sgi * 16u;
    const int8_t *Ab = A.Ap + (size_t)(row0 >> 4) * A.aStride;
    const int8_t *Bb = A.Bp + (size_t)(A.colStart / 4u + wc - jc) * A.bStride;  // the group's first column window
    uint32_t *R = &ring[0];
    // Block blk of a k-step's B' in global memory (blocks 16 and up: the second column window).
    auto bsrc = [&](uint32_t ks, uint32_t blk) {
      if constexpr (VAR & 1024)  // ks is a pair of k-steps here; blocks 16 and up its second
        return Bb + (size_t)(2u * ks + (blk >> 4)) * 2048u + (size_t)(blk & 15u) * 128u;
      return Bb + (size_t)(blk >> 4) * A.bStride + (size_t)ks * 2048u + (size_t)(blk & 15u) * 128u;
    };
    // This sub-group's blocks of a k-step's B' (kBlocks blocks of 128 bytes over nsg sub-groups).
    auto copy = [&](uint32_t ks, uint32_t slot) {
      for (uint32_t blk = sgl; blk < kBlocks; blk += nsg)
        ring_write_local<HW>(R + slot * kSlotWords + blk * 32u, ring_read_global<HW>(bsrc(ks, blk), lane), lane);
    };
    if constexpr (kWide) {
      for (uint32_t i = (uint32_t)it.get_local_linear_id(); i < A.wpg * 128u; i += nsg * 8u) part[i] = 0u;
      sycl::group_barrier(it.get_group());
    }
    // VAR 4: the copy in two halves, a load a k-step early and its store. Only with 8 or more sub-groups
    // a work-group (at most 2 blocks each, which wait in registers); fewer copy in one go.
    const bool early = (VAR & 4) && 2u * nsg >= kBlocks;
    v4u pre[2] = {0, 0};
    v4u pnext = 0;
    (void)pnext;
#ifdef PEARL_RING_PFD
    v4u preQ[PEARL_RING_PFD];
#endif
    const uint32_t mine = sgl < kBlocks ? (kBlocks - sgl + nsg - 1u) / nsg : 0u;  // blocks this sub-group copies
    // goal-arc7 timing builds: PEARL_RING_NOFETCH drops the copy's load (zeros are stored), PEARL_RING_NOPUT its
    // store (the loaded words go into the next k-step's gate instead, so the load stays).
    int xsink = 0;
    // PEARL_RING_XCOPY n (timing only): the copy also loads n 128-byte blocks of the sub-group's own A' (of the
    // k-step it fetches) and stores them into `part`, as a ring of A' would.
#ifndef PEARL_RING_XCOPY
#define PEARL_RING_XCOPY 0
#endif
    v4u xpre[PEARL_RING_XCOPY > 0 ? PEARL_RING_XCOPY : 1];
    auto fetch = [&](uint32_t ks) {
#ifndef PEARL_RING_NOFETCH
      for (uint32_t i = 0; i < 2u; i++)
        if (i < mine) pre[i] = ring_read_global<HW>(bsrc(ks, sgl + i * nsg), lane);
#endif
      each<PEARL_RING_XCOPY>([&](int j) {
        xpre[j] = ring_read_global<HW>(A.Ap + (size_t)((32u * wr + sgi * 16u) >> 4) * A.aStride + (size_t)ks * 512u + (size_t)j * 128u, lane);
      });
    };
    auto put = [&](uint32_t slot) {
      each<PEARL_RING_XCOPY>([&](int j) { ring_write_local<HW>(&part[0] + sgl * 128u + (uint32_t)j * 32u, xpre[j], lane); });
#ifdef PEARL_RING_NOPUT
      for (uint32_t i = 0; i < 2u; i++)
        if (i < mine) xsink ^= (int)(pre[i][0] ^ pre[i][1] ^ pre[i][2] ^ pre[i][3]);
#else
      for (uint32_t i = 0; i < 2u; i++)
        if (i < mine) ring_write_local<HW>(R + slot * kSlotWords + (sgl + i * nsg) * 32u, pre[i], lane);
#endif
    };
    v8i c[2][8];
    each<2>([&](int rb) { each<8>([&](int cb) { c[rb][cb] = 0; }); });
    uint32_t t[16];
    each<16>([&](int j) { t[j] = 0u; });
    // VAR 4096: fill[s] counts the sub-groups that have stored their blocks into slot s, done[s] those that
    // have read it, both over every use of the slot.
    uint32_t *fill = R + kSlots * kSlotWords, *done = fill + kSlots;
    auto bump = [&](uint32_t *p, int dep) {
      if (lane == 0u) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            at(*p);
        at.fetch_add(1u + (uint32_t)dep, sycl::memory_order::release);
      }
    };
    auto until = [&](uint32_t *p, uint32_t need) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                       sycl::access::address_space::local_space>
          at(*p);
      while (at.load() < need) {
      }
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::work_group);
    };
    if constexpr (kCnt) {
      if (sgl == 0u)
        for (uint32_t i = lane; i < 2u * kSlots; i += 8u) fill[i] = 0u;
      sycl::group_barrier(it.get_group());
    }
    // A' of k-step ks into a[2].
    auto load_a = [&](uint32_t ks, v8i *a) {
      each<2>([&](int rb) {
        const int8_t *p = Ab + (size_t)ks * 512u + (size_t)rb * 256u;
        const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
        a[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      });
    };
    v8i ahead[2];
#if defined(PEARL_RING_SPLITMAD)
    // goal-arc7, with VAR 128: this sub-group's half of the window's 4 A' fragments for k-step ks.
    auto load_a4 = [&](uint32_t ks, v4i *a) {
      const int8_t *A32 = A.Ap + (size_t)(2u * wr) * A.aStride + (size_t)sgi * 128u;
      each<4>([&](int rb) {
        a[rb] = (v4i)ring_read_global<HW>(A32 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u + (size_t)(rb & 1) * 256u, lane);
      });
    };
#if defined(PEARL_RING_QA)
    v4i a4n[4];
    load_a4(0u, a4n);
#endif
#elif defined(PEARL_RING_QA)
    // goal-arc7, with VAR 128: the window's 32 rows of A' for k-step ks, 4 fragments.
    auto load_a32 = [&](uint32_t ks, v8i *a) {
      const int8_t *A32 = A.Ap + (size_t)(2u * wr) * A.aStride;
      each<4>([&](int rb) {
        const int8_t *p = A32 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u + (size_t)(rb & 1) * 256u;
        a[rb] = frag_read_w<HW, 4>(PEARL_GPTR(p), lane);
      });
    };
    v8i a32n[4];
    load_a32(0u, a32n);
#endif
    if constexpr (VAR & 2048) load_a(0u, ahead);
    if constexpr (!(VAR & 32768)) {
      copy(0u, 0u);
#if defined(PEARL_RING_FLAT)
      pnext = ring_read_global<HW>(bsrc(1u, sgl), lane);
#elif defined(PEARL_RING_PFD)
      if (early)
        each<PEARL_RING_PFD>([&](int d) {
          fetch(1u + (uint32_t)d);
          preQ[(1u + (uint32_t)d) & (PEARL_RING_PFD - 1u)] = pre[0];
        });
#else
      if (early) fetch(1u);
#endif
      if constexpr (kCnt) bump(&fill[0], 0);
      else arrive(it);
    }
    // A chunk's readout: its cumulative tile XORed into two words a lane, gathered to the lanes that write
    // its 8 regions' words.
    auto readout = [&](uint32_t ch) {
      uint32_t x0 = 0, x1 = 0;
      each<2>([&](int rb) {
        each<8>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
      if constexpr (kWide) {
        if (lane < 8u) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
              at(part[(widx * 8u + lane) * 16u + ch]);
          at.fetch_xor((lane & 1u) ? x1 : x0);
        }
      } else if constexpr (VAR & 16) {
        // Chunk ch's word lands in t[15]: a shift, so every index stays a constant.
        each<15>([&](int j) { t[j] = t[j + 1]; });
        t[15] = (lane & 1u) ? x1 : x0;
      } else {
        // VAR 64, timing only: every word into one place, so the group needs almost no local memory beside
        // the ring. Wrong transcripts on purpose.
        if (lane < 8u) part[(VAR & 64) ? ((sgl * 8u + lane) & 127u) : (sgl * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
      }
    };
    if constexpr (VAR & 32768) {
      // VAR 32768: a software pipeline with no extra registers. A k-step's DPAS read fragments loaded during
      // the k-step before, and each fragment register is loaded again for the next k-step as soon as its
      // last DPAS has been sent: A' block 0 after the 8 DPAS of rows 0-7, each B' fragment after its second
      // DPAS, A' block 1 at the end. So every load has about 8 of the sub-group's DPAS to land. The ring has
      // 4 slots and the copy runs two k-steps ahead: k-step ks waits for the arrivals of ks - 1 (every
      // store into slot ks + 1 has landed, and every read of slot ks + 2, last used two k-steps ago, has been
      // consumed), stores k-step ks + 2's blocks, loads ks + 3's, arrives, then runs.
      // VAR 65536: each reload's address takes (a word of the DPAS that last read the register) AND zero,
      // so no compiler can send it before that DPAS and need a second register for it.
      auto pin = [&](const v8i &v) -> uint32_t {
        if constexpr (VAR & 65536) return (uint32_t)v[0] & A.zero;
        return 0u;
      };
      auto load_b = [&](uint32_t sl, int cb, uint32_t z = 0u) -> v8i {
        uint32_t *p = R + sl * kSlotWords + (uint32_t)cb * 64u + z;
        if constexpr (VAR & 262144) return ring_read_local8(p, lane);
        if constexpr (VAR & 524288) return ring_read_local8b<HW>(p, lane);
        const v4u lo = ring_read_local<HW>(p, lane), hi = ring_read_local<HW>(p + 32, lane);
        return (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      };
      auto load_a1 = [&](uint32_t ks, int rb, uint32_t z = 0u) -> v8i {
        const int8_t *p = Ab + (size_t)ks * 512u + (size_t)rb * 256u + z;
        if constexpr (VAR & 524288) return ring_read_global8b<HW>(p, lane);
        const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
        return (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      };
      copy(0u, 0u);
      copy(1u, 1u);
      if (early) fetch(2u);
      sycl::group_barrier(it.get_group());
      v8i af[2], bf[8];
      each<2>([&](int rb) { af[rb] = load_a1(0u, rb); });
      each<8>([&](int cb) { bf[cb] = load_b(0u, cb); });
      arrive(it);
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t ks = ch * 4u + s;
          wait(it);
          if (early) {
            if (ks + 2u < 64u) put((ks + 2u) % 4u);
            if (ks + 3u < 64u) fetch(ks + 3u);
          } else {
            if (ks + 2u < 64u) copy(ks + 2u, (ks + 2u) % 4u);
          }
          arrive(it);
          const uint32_t kn = ks + 1u < 64u ? ks + 1u : 63u, sn = kn % 4u;
          // VAR 131072: only B' rotates; A' is loaded here, for this k-step, and waited for.
          constexpr bool aRot = !(VAR & 131072);
          if constexpr (!aRot) {
            each<2>([&](int rb) { af[rb] = load_a1(ks, rb); });
            int t = 0;
            each<2>([&](int rb) { t ^= af[rb][0]; });
            t &= (int)A.zero;
            each<2>([&](int rb) { af[rb][0] |= t; });
          }
          each<8>([&](int cb) { c[0][cb] = dpas<8, HW>(sg, af[0], bf[cb], c[0][cb]); });
          if constexpr (aRot) af[0] = load_a1(kn, 0, pin(c[0][7]));
          each<8>([&](int cb) {
            c[1][cb] = dpas<8, HW>(sg, af[1], bf[cb], c[1][cb]);
            bf[cb] = load_b(sn, cb, pin(c[1][cb]));
          });
          if constexpr (aRot) af[1] = load_a1(kn, 1, pin(c[1][7]));
        }
        readout(ch);
      }
    } else
    for (uint32_t ch = 0; ch < 16u; ch++) {
#ifdef PEARL_RING_NOUNROLL
#pragma unroll 1
#endif
      for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t ks = ch * 4u + s;
        uint32_t slot = ks % kSlots, half = 0u;
        v8i af[2];
        // VAR 8192: this k-step's A' loads go out before the wait.
        if constexpr ((VAR & 8192) && !(VAR & 2048)) load_a(ks, af);
        if constexpr (kCnt) {
          const uint32_t j = ks + 1u;
          if (j < 64u) {
            const uint32_t sj = j % kSlots;
            if (j >= kSlots) until(&done[sj], nsg * (j / kSlots));  // every read of k-step j - kSlots is done
            if (early) put(sj);
            else copy(j, sj);
            bump(&fill[sj], 0);
            if (early && j + 1u < 64u) fetch(j + 1u);
          }
          until(&fill[slot], nsg * (ks / kSlots + 1u));
        } else if constexpr (VAR & 1024) {
          slot = (ks >> 1) % 3u;
          half = ks & 1u;
          if (half == 0u) {
            const uint32_t ph = ks >> 1;
            wait(it);
            if (early) {
              if (ph + 1u < 32u) put((ph + 1u) % 3u);
              if (ph + 2u < 32u) fetch(ph + 2u);
            } else {
              if (ph + 1u < 32u) copy(ph + 1u, (ph + 1u) % 3u);
            }
            arrive(it);
          }
        } else {
          // goal-arc7 timing builds (wrong results on purpose): PEARL_RING_NOBAR drops the k-step's meeting,
          // PEARL_RING_NOCOPY the copy, so what is left is the reads and the DPAS.
#ifndef PEARL_RING_BAREVERY
#define PEARL_RING_BAREVERY 1
#endif
#ifndef PEARL_RING_NOBAR
          if (s % PEARL_RING_BAREVERY == 0u) wait(it);
#endif
#ifndef PEARL_RING_NOCOPY
#if defined(PEARL_RING_FLAT)
          // goal-arc7: no branch in the k-step. One block a sub-group (16 sub-groups, wpg 8); past the last
          // k-step it loads k-step 63 again and stores it into a slot nobody reads any more.
          {
            pre[0] = ring_read_global<HW>(bsrc(ks + 2u < 64u ? ks + 2u : 63u, sgl), lane);
            ring_write_local<HW>(R + ((ks + 1u) % 3u) * kSlotWords + sgl * 32u, pnext, lane);
            pnext = pre[0];
          }
          if (false)
#elif defined(PEARL_RING_PFD)
          // goal-arc7: the copy's block is loaded PEARL_RING_PFD k-steps before it is stored (a power of 2;
          // one block a sub-group, wpg 8), so its load has that many k-steps to come from memory.
          if (early) {
            const uint32_t q = (ks + 1u) & (PEARL_RING_PFD - 1u);
            if (ks + 1u < 64u) {
              pre[0] = preQ[q];
              put((ks + 1u) % 3u);
            }
            if (ks + 1u + PEARL_RING_PFD < 64u) {
              fetch(ks + 1u + PEARL_RING_PFD);
              preQ[q] = pre[0];
            }
          } else
#else
          if (early) {
            if (ks + 1u < 64u) put((ks + 1u) % 3u);
            if (ks + 2u < 64u) fetch(ks + 2u);
          } else
#endif
          {
            if (ks + 1u < 64u) copy(ks + 1u, (ks + 1u) % 3u);
          }
#endif
#if !defined(PEARL_RING_NOBAR) && !defined(PEARL_RING_LATEARR)
          if (s % PEARL_RING_BAREVERY == 0u) arrive(it);
#endif
        }
        if constexpr (VAR & 128) {
          // VAR 128: a 32 by 32 tile a sub-group. Both sub-groups of a window read all 32 rows of A' from
          // global memory together (4 fragments), and sub-group sgi its half of B' from the ring (4
          // fragments): 2,048 bytes a k-step, half of them from local memory, in place of 2,560 with 2,048.
          // Tile (rb, cb) lives in c[rb >> 1][(rb & 1) * 4 + cb], so the readout is unchanged.
#if defined(PEARL_RING_SPLITMAD)
          {
            // goal-arc7: the window's two sub-groups share its 32 rows of A' by the split DPAS: sub-group sgi
            // reads half sgi of each A' fragment (rows 4 sgi .. 4 sgi + 3 of each 8-row block, one 128-byte
            // block read in kFragAT order), and its own 32 columns of B' from the ring.
            v4i ah[4];
            v8i bq[4];
#if defined(PEARL_RING_QA)
            each<4>([&](int rb) { ah[rb] = a4n[rb]; });
            load_a4(ks + 1u < 64u ? ks + 1u : 63u, a4n);
#else
            load_a4(ks, ah);
#endif
            each<4>([&](int cb) {
#ifdef PEARL_RING_GB
              // goal-arc7 timing: B' straight from memory (the window's own column block), no ring.
              bq[cb] = frag_read_w<HW, 4>(PEARL_GPTR(Bb + (size_t)ks * 2048u + (sgi * 4u + (uint32_t)cb) * 256u), lane);
#else
              bq[cb] = frag_read_w<HW, 4>(PEARL_LPTR(R + slot * kSlotWords + (sgi * 4u + (uint32_t)cb) * 64u), lane);
#endif
            });
            int t4 = 0;
            each<4>([&](int rb) { t4 ^= ah[rb][0]; });
            each<4>([&](int cb) { t4 ^= bq[cb][0]; });
            t4 &= (int)A.zero;
            each<4>([&](int rb) { ah[rb][0] |= t4; });
            each<4>([&](int cb) {
              each<4>([&](int rb) {
                c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
              });
            });
          }
          continue;
#endif
          v8i af[4], bf[4];
#if defined(PEARL_RING_QA) && !defined(PEARL_RING_SPLITMAD)
          // goal-arc7: A' read a k-step ahead into registers (a32n), so its latency hides behind a k-step.
          each<4>([&](int rb) { af[rb] = a32n[rb]; });
          load_a32(ks + 1u < 64u ? ks + 1u : 63u, a32n);
#else
          const int8_t *A32 = A.Ap + (size_t)(2u * wr) * A.aStride;
          (void)A32;
          each<4>([&](int rb) {
#ifdef PEARL_RING_ASLM
            af[rb] = frag_read_w<HW, 4>(PEARL_LPTR(&part[0] + (uint32_t)(ks & 3u) * 256u + (uint32_t)rb * 64u), lane);
#else
            const int8_t *p = A32 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u + (size_t)(rb & 1) * 256u;
            const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
            af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
#endif
          });
#endif
#if defined(PEARL_RING_QH)
          // goal-arc7: B' in two halves of 2 fragments, so 16 registers of B' are live, not 32.
          each<2>([&](int h) {
            v8i bh[2];
            each<2>([&](int i) {
              uint32_t *p = R + slot * kSlotWords + (sgi * 4u + (uint32_t)(2 * h + i)) * 64u;
              bh[i] = frag_read_w<HW, 4>(PEARL_LPTR(p), lane);
            });
            if (h == 0) {
              int t3 = (bh[0][0] ^ bh[1][0]);
              each<4>([&](int rb) { t3 ^= af[rb][0]; });
              t3 &= (int)A.zero;
              each<4>([&](int rb) { af[rb][0] |= t3; });
            }
            each<2>([&](int i) {
              const int cb = 2 * h + i;
              each<4>([&](int rb) {
                c[rb >> 1][(rb & 1) * 4 + cb] = dpas<8, HW>(sg, af[rb], bh[i], c[rb >> 1][(rb & 1) * 4 + cb]);
              });
            });
          });
          continue;
#endif
          each<4>([&](int cb) {
            uint32_t *p = R + slot * kSlotWords + (sgi * 4u + (uint32_t)cb) * 64u;
            const v4u lo = ring_read_local<HW>(p, lane), hi = ring_read_local<HW>(p + 32, lane);
            bf[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          int t2 = 0;
          each<4>([&](int rb) { t2 ^= af[rb][0]; });
          each<4>([&](int cb) { t2 ^= bf[cb][0]; });
          t2 &= (int)A.zero;
          each<4>([&](int rb) { af[rb][0] |= t2; });
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = dpas<8, HW>(sg, af[rb], bf[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
          continue;
        }
        // VAR 256: prefetch the A' of k-step ks + PEARL_RING_PF into the first cache level (its 512 bytes, 64 a
        // lane, so each of its 4 lines is asked for twice).
        if constexpr (VAR & 256) {
#ifdef PEARL_RING_PF
          if (ks + PEARL_RING_PF < 64u) {
            const int8_t *q = Ab + (size_t)(ks + PEARL_RING_PF) * 512u + 64u * lane;
            ring_prefetch_l1<HW>(q);
          }
#endif
        }
        v8i bf[8];
        if constexpr (VAR & 2048) {
          each<2>([&](int rb) { af[rb] = ahead[rb]; });
          load_a(ks + 1u < 64u ? ks + 1u : 63u, ahead);
        } else if constexpr (!(VAR & 8192)) {
          each<2>([&](int rb) {
            const int8_t *p = Ab + (size_t)ks * 512u + (size_t)rb * 256u;
            if constexpr (VAR & 2) {
              af[rb] = *reinterpret_cast<const v8iu *>(p + 32u * lane);
            } else if constexpr (VAR & 524288) {
              af[rb] = ring_read_global8b<HW>(p, lane);
#ifdef PEARL_RING_ASLM
            } else if constexpr (true) {  // timing only: A' from local memory (the part array), always hot
              af[rb] = frag_read_w<HW, 4>(PEARL_LPTR(&part[0] + (uint32_t)(ks & 3u) * 256u + (uint32_t)rb * 64u + sgi * 128u), lane);
#endif
            } else if constexpr (PEARL_RING_AW != 4) {
              af[rb] = frag_read_w<HW, PEARL_RING_AW>(PEARL_GPTR(p), lane);
            } else {
              const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
              af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            }
          });
        }
        // VAR 16384: B' in two halves of 4 fragments, each waited for as one and then its 8 DPAS, so a
        // sub-group holds 32 registers fewer of B'.
        if constexpr (VAR & 16384) {
          int t = 0;
          each<2>([&](int h) {
            v8i bh[4];
            each<4>([&](int i) {
              uint32_t *p = R + slot * kSlotWords + (jc + half) * 512u + (uint32_t)(4 * h + i) * 64u;
              const v4u lo = ring_read_local<HW>(p, lane), hi = ring_read_local<HW>(p + 32, lane);
              bh[i] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            });
            t = 0;
            if (h == 0) each<2>([&](int rb) { t ^= af[rb][0]; });
            each<4>([&](int i) { t ^= bh[i][0]; });
            t &= (int)A.zero;
            each<2>([&](int rb) { af[rb][0] |= t; });
            if constexpr (kCnt) {
              if (h == 1) bump(&done[slot], t);
            }
            each<4>([&](int i) {
              each<2>([&](int rb) { c[rb][4 * h + i] = dpas<8, HW>(sg, af[rb], bh[i], c[rb][4 * h + i]); });
            });
          });
          continue;
        }
        each<8>([&](int cb) {
          uint32_t *p = R + slot * kSlotWords + (jc + half) * 512u + (uint32_t)cb * 64u;
          // goal-arc7 timing builds: fewer B' fragments read (the DPAS reuse them), or other read widths.
          if constexpr (PEARL_RING_NB < 8) {
            if (cb >= PEARL_RING_NB) {
              bf[cb] = bf[cb % PEARL_RING_NB];
              return;
            }
          }
          // VAR 262144: B' stored in fragment order (kFragB), one 8-dword read a fragment.
          if constexpr (PEARL_RING_BW != 4) {
            bf[cb] = frag_read_w<HW, PEARL_RING_BW>(PEARL_LPTR(p), lane);
          } else if constexpr (VAR & 262144) {
            bf[cb] = ring_read_local8(p, lane);
          } else if constexpr (VAR & 524288) {
            bf[cb] = ring_read_local8b<HW>(p, lane);
          } else {
            const v4u lo = ring_read_local<HW>(p, lane), hi = ring_read_local<HW>(p + 32, lane);
            bf[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          }
        });
        // VAR 1: the chunk before's readout here, after this k-step's arrival and loads, so the other
        // sub-groups never wait at the barrier for it.
        if constexpr (VAR & 1) {
          if (s == 0u && ch > 0u) readout(ch - 1u);
        }
        // Every DPAS of the k-step waits for every load of it (see fold_xmxf).
        // VAR 1048576: the DPAS wait only for A' (the global loads), B' reads from the ring are left to the
        // compiler; VAR 2097152: no forced wait at all.
        int t = xsink;
        // goal-arc7 timing builds: PEARL_RING_XR more fragments a k-step from the slot read a k-step ago (at most
        // 8: more would repeat an address), every dword into the gate so no part of a message is dropped.
        if constexpr (PEARL_RING_XR > 0) {
          uint32_t *xs = R + ((ks + 2u) % kSlots) * kSlotWords;
          each<PEARL_RING_XR>([&](int i) {
            const v8i x = frag_read_w<HW, PEARL_RING_XW>(PEARL_LPTR(xs + (uint32_t)(i % 8) * 64u), lane);
            t ^= x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4] ^ x[5] ^ x[6] ^ x[7];
          });
        }
        if constexpr (!(VAR & 2097152)) {
          each<2>([&](int rb) { t ^= af[rb][0]; });
          if constexpr (!(VAR & 1048576)) each<8>([&](int cb) { t ^= bf[cb][0]; });
          t &= (int)A.zero;
          each<2>([&](int rb) { af[rb][0] |= t; });
        }
        // VAR 4096: this slot's reads have landed (t waits for them): count them done.
        if constexpr (kCnt) bump(&done[slot], t);
        each<8>([&](int cb) {
#if defined(PEARL_RING_LATEARR) && PEARL_RING_LATEARR == 2
          if (cb == 4) arrive(it);
#endif
          each<2>([&](int rb) { c[rb][cb] = dpas<8, HW>(sg, af[rb], bf[cb], c[rb][cb]); });
        });
#if defined(PEARL_RING_LATEARR) && PEARL_RING_LATEARR == 1
        arrive(it);
#endif
      }
      if constexpr (!(VAR & 1)) readout(ch);
    }
    if constexpr (VAR & 1) readout(15u);
    if constexpr (!kCnt) wait(it);
    sycl::group_barrier(it.get_group());
    if constexpr (VAR & 16) {
      // The ring is free now: sub-group 1 of each window leaves its words there (wpg <= 12 fit in it).
      if (sgi == 1u && lane < 8u) each<16>([&](int j) { R[(widx * 8u + lane) * 16u + (uint32_t)j] = t[j]; });
      sycl::group_barrier(it.get_group());
      if (sgi == 0u && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        each<16>([&](int j) { words[j] = t[j] ^ R[(widx * 8u + qr) * 16u + (uint32_t)j]; });
        const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
      return;
    }
    if constexpr (kWide) {
      if (sgi == 0 && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        for (int j = 0; j < 16; j++) words[j] = part[(widx * 8u + qr) * 16u + (uint32_t)j];
        const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
      return;
    }
    if (sgi == 0 && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
      for (int j = 0; j < 16; j++)
        words[j] = (VAR & 64) ? part[(sgl * 8u + qr) & 127u] ^ (uint32_t)j
                              : part[(sgl * 8u + qr) * 16u + (uint32_t)j] ^ part[((sgl + 1u) * 8u + qr) * 16u + (uint32_t)j];
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
  }
};

// `groups` windows, A.wpg to a work-group (the host makes wpg divide the band and the row windows).
template <bool HW, int VAR = 0> inline void fold_xmxr(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  const size_t wg = 16u * a.wpg;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part((VAR & 16)    ? 1u
                                           : (VAR & 64)  ? 128u
                                           : (VAR & (512 | 1024)) ? a.wpg * 8u * 16u
                                                         : 2u * a.wpg * 8u * 16u,
                                           h);
    typedef FoldXmxR<HW, VAR> K;
    const uint32_t ringWords = K::kSlots * K::kSlotWords + (K::kCnt ? 2u * K::kSlots : 0u);
    // VAR 16 leaves 128 words a window in the ring at the end.
    sycl::local_accessor<uint32_t, 1> ring((VAR & 16) && a.wpg * 128u > ringWords ? a.wpg * 128u : ringWords, h);
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / a.wpg) * wg, wg), FoldXmxR<HW, VAR>{a, part, ring});
  });
}

// ---------------------------------------------------------------------------
// The split fold (xmx8w): the ring fold with a 32 by 32 tile a sub-group, whose
// two sub-groups of a window share A' through the split DPAS. Added 12
// November 2026 (goal-arc7).
//
// Alchemist's DPASW (OpenCL's cl_intel_subgroup_split_matrix_multiply_
// accumulate) takes A from a pair of sub-groups, sub-groups 2i and 2i + 1 of a
// work-group: each gives half of the 8 rows (4 dwords a lane), and each gets
// all 8 rows times its own B. So a window (32 rows by 64 columns, two
// sub-groups) is split by columns, not rows: sub-group sgi holds all 32 rows
// by columns 32 sgi .. 32 sgi + 31, 16 accumulators, as before.
//
//   - A' per sub-group and k-step: half of each of the window's 4 fragments,
//     rows 4 sgi .. 4 sgi + 3 of each 8-row block. In the kFragAT layout that
//     half is one 128-byte block read, so 512 bytes, as the ring fold.
//   - B': its 4 fragments from the ring, 1,024 bytes where the ring fold reads
//     2,048. The ring and its split barrier are the ring fold's, but a slot
//     holds four k-steps (8 KB), so the group meets once a chunk: on card 4,
//     87.7 to 88.3 TH/s against 86.1 to 86.6 for two k-steps a slot and 84.2
//     for one, all at 188 W (KPS, PEARL_W_KPS). The ring is 24 KB.
//   - The copy has no branch in the k-step: at 8 windows a work-group (16
//     sub-groups) each sub-group copies one block of 128 bytes, loaded a
//     k-step early. Past the last k-step it loads k-step 63 again and stores
//     it into a slot no one reads any more. The host runs this fold only at 8
//     windows a work-group (FoldArgs::wpg), else the ring fold. In the ring
//     fold IGC put a goto/join around each guarded load and store, which cut
//     the k-step into blocks it would not schedule across: without them the
//     copy costs 7% where it cost 23% (goal-arc7, timing builds).
//   - Accumulator (rb, cb) lives in c[rb >> 1][(rb & 1) * 4 + cb], so the
//     readout is the ring fold's.
//
// On an A750 (pearl_sycl_check --bench, 8 s, taking turns, 12 November): card
// 4, 84.2 TH/s at 1,985 MHz and 188 W with a slot a k-step, against the ring
// fold's 68.2 at 2,000 MHz and 188 W (+23%); 88 with four k-steps a slot. It
// sits at the card's power limit, so watts now bind as much as the clock.
//
// HW, in the build the driver compiles: the split DPAS, block reads and the
// split barrier. Otherwise (the reference, and the prebuilt kernels, which have
// no split DPAS) each sub-group reads all 32 rows of A' and runs the plain
// DPAS: the same results.
// ---------------------------------------------------------------------------
// KPS: k-steps a ring slot (1 or 2); the group meets once a slot. AH: A' loaded a k-step ahead.
// RW by CW windows a work-group (row windows by column windows): 2 RW CW sub-groups, each copying 8 / RW
// blocks of 128 bytes of B' a k-step. With CW 2, a window's A' is read by two pairs, the second from L1.
#if !defined(PEARL_W_NOLO32)
#define PEARL_W_LO32  // goal-arc8: 32-bit address sums (see lo32); the host keeps each operand in one 4 GB window
#endif
template <bool HW, int KPS = 1, bool AH = false, int RW = 8, int CW = 1> struct FoldXmxW {
  static_assert(KPS == 1 || KPS == 2 || KPS == 4, "a slot is 1, 2 or 4 k-steps: the group meets inside a chunk");
  static_assert(RW == 2 || RW == 4 || RW == 8 || (RW == 16 && CW == 1), "each sub-group copies a whole number of blocks");
  // goal-arc8: at RW 16 each of the 32 sub-groups copies half a block (64 bytes) a k-step.
  static constexpr bool kHalf = RW == 16;
  static constexpr uint32_t kNsg = 2u * RW * CW, kBps = kHalf ? 1u : 8u / RW;
  typedef std::conditional_t<kHalf, v2u, v4u> cvec;
  static cvec cp_read(const int8_t *p, uint32_t lane) {
    if constexpr (kHalf) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      if constexpr (HW) return __spirv_SubgroupBlockReadINTEL<v2u>((const __attribute__((opencl_global)) uint32_t *)p);
#endif
      const uint32_t *w = reinterpret_cast<const uint32_t *>(p);
      return v2u{w[lane], w[8u + lane]};
    } else {
      return ring_read_global<HW>(p, lane);
    }
  }
  static void cp_write(uint32_t *p, cvec v, uint32_t lane) {
    if constexpr (kHalf) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      if constexpr (HW) {
        __spirv_SubgroupBlockWriteINTEL((__attribute__((opencl_local)) uint32_t *)p, v);
        return;
      }
#endif
      p[lane] = v[0];
      p[8u + lane] = v[1];
    } else {
      ring_write_local<HW>(p, v, lane);
    }
  }
#if defined(PEARL_W_REGWORDS)
  // goal-arc7: each lane keeps its 16 transcript words in registers, so the work-group needs no local memory
  // beyond the ring (64 KB of it an Xe-core decides how many work-groups fit).
  static constexpr bool kRegWords = true;
#else
  static constexpr bool kRegWords = false;
#endif
#if defined(PEARL_SYCL_XMX_JIT)
  static constexpr bool kSplit = HW;
#else
  static constexpr bool kSplit = false;
#endif
  static constexpr uint32_t kStep = 512u * CW;                  // a k-step's B': CW blocks of 64 columns by 32 k
  static constexpr uint32_t kSlotWords = kStep * (uint32_t)KPS;  // a slot: KPS k-steps
#if defined(PEARL_W_SLOTS)
  // goal-arc8: 2 slots, with the group's arrive after the slot's last local reads (else 3, arrive before them).
  static constexpr uint32_t kSlots = PEARL_W_SLOTS;
#else
  static constexpr uint32_t kSlots = 3u;
#endif
#if !defined(PEARL_W_NOPARTX)
  // goal-arc8: a window's two sub-groups XOR their words into one place (half the local memory). With the
  // readout below as a tree, 90.3 to 90.6 against 89.9 to 90.2 on card 5 (three rounds, 188 W).
  static constexpr bool kPartX = true;
#else
  static constexpr bool kPartX = false;
#endif
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> part;  // 2 wpg sub-groups x 8 regions x 16 words
  sycl::local_accessor<uint32_t, 1> ring;  // 3 slots

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (HW)
      return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
    else
      return syclex::properties{syclex::sub_group_size<8>};
  }
  void arrive(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && !defined(PEARL_W_PLAINBAR)
    if constexpr (HW) {
      __spirv_ControlBarrierArriveINTEL(2, 2, 0x104);  // work-group; work-group memory, release
      return;
    }
#endif
    sycl::group_barrier(it.get_group());
  }
  void wait(sycl::nd_item<1>) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && !defined(PEARL_W_PLAINBAR)
    if constexpr (HW) __spirv_ControlBarrierWaitINTEL(2, 2, 0);  // no acquire fence, as the ring fold's VAR 8
#endif
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
#if defined(PEARL_W_NC) && defined(PEARL_W_NCFIRST)
    // goal-arc8: the copiers are sub-groups 0 .. NC - 1, so the DPAS pairs keep their places in hardware.
    const uint32_t sgr = (uint32_t)sg.get_group_linear_id();
    const uint32_t sgl = sgr >= (uint32_t)(PEARL_W_NC) ? sgr - (uint32_t)(PEARL_W_NC) : kNsg + sgr;
#else
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();  // 0 .. kNsg - 1
#endif
    const uint32_t sgi = sgl & 1u, widx = sgl >> 1;
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    uint32_t wcb;
    {
      uint32_t wrb;
#if defined(PEARL_W_PAIR)
      // goal-arc8: PEARL_W_PAIR work-groups in a row take the same row windows (next column windows).
      {
        const uint32_t g = (uint32_t)it.get_group(0), rw = rowWins / RW, cw = colWins / CW;
        const uint32_t band = A.band / RW > 0u ? A.band / RW : 1u, per = band * cw, b = g / per;
        const uint32_t br = rw - b * band < band ? rw - b * band : band, idx = g - b * per;
        const uint32_t q = idx % (uint32_t)(PEARL_W_PAIR), pp = idx / (uint32_t)(PEARL_W_PAIR);
        wrb = b * band + pp % br;
        wcb = (pp / br) * (uint32_t)(PEARL_W_PAIR) + q;
      }
#else
      window_of((uint32_t)it.get_group(0), rowWins / RW, colWins / CW, A.band / RW > 0u ? A.band / RW : 1u, wrb, wcb);
#endif
      wr = wrb * RW + widx % RW;
      wc = wcb * CW + widx / RW;
    }
    const uint32_t jc = widx / RW;  // the window's column block in the ring
#if defined(PEARL_W_AHOT)
    // goal-arc7 timing (wrong results): A' of only PEARL_W_AHOT row windows, so it stays near.
    const int8_t *Ab = A.Ap + (size_t)(2u * (wr % PEARL_W_AHOT)) * A.aStride;
#else
    const int8_t *Ab = A.Ap + (size_t)(2u * wr) * A.aStride;  // the window's two 16-row blocks
#endif
    // Its copy blocks: block b = sgl + i * kNsg of a k-step is column block b / 16's piece b % 16.
    const int8_t *Bbi[kBps];
    each<kBps>([&](int i) {
      const uint32_t b = sgl + (uint32_t)i * kNsg;
#if defined(PEARL_W_BPAD)
      Bbi[i] = A.Bp + (size_t)(A.colStart / 4u + wcb * CW + b / 16u) * (A.bStride - (PEARL_W_BPAD)) + (size_t)(b % 16u) * 128u;
#else
      if constexpr (kHalf)
        Bbi[i] = A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride + (size_t)(b >> 1) * 128u + (size_t)(b & 1u) * 64u;
      else
        Bbi[i] = A.Bp + (size_t)(A.colStart / 4u + wcb * CW + b / 16u) * A.bStride + (size_t)(b % 16u) * 128u;
#endif
    });
    uint32_t *R = &ring[0];
#if defined(PEARL_W_NC)
    // goal-arc8: PEARL_W_NC copier sub-groups after the 16 that run the DPAS. They fill the ring a chunk at a
    // time (a slot is a chunk, 8 KB of B' in the same order as in memory), two chunks ahead, so the DPAS
    // sub-groups' k-steps hold no copy at all. They meet the group as often as the others do.
    static_assert(KPS == 4 && CW == 1 && RW == 8, "copier sub-groups: 8 by 1 work-group, a chunk a slot");
    static_assert(64 % (PEARL_W_NC) == 0, "copiers share a chunk's 64 blocks evenly");
    if (sgl >= kNsg) {
      constexpr int kPer = 64 / (PEARL_W_NC);
      const uint32_t q = sgl - kNsg;
      const int8_t *Bc = A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride + (size_t)q * 128u;
      v4u hold[kPer];
      auto load_chunk = [&](uint32_t c) {
        each<kPer>([&](int i) { hold[i] = ring_read_global<HW>(Bc + (size_t)c * 8192u + (size_t)i * (128u * (PEARL_W_NC)), lane); });
      };
      auto store_slot = [&](uint32_t slot) {
        each<kPer>([&](int i) { ring_write_local<HW>(R + slot * kSlotWords + (q + (uint32_t)i * (PEARL_W_NC)) * 32u, hold[i], lane); });
      };
#if !defined(PEARL_W_NCIDLE)
      load_chunk(1u);
      store_slot(1u);
      load_chunk(2u);
#endif
      arrive(it);
      for (uint32_t ch = 0; ch < 16u; ch++) {
        wait(it);
#if !defined(PEARL_W_NCIDLE)
        store_slot((ch + 2u) % 3u);
        load_chunk(ch + 3u < 16u ? ch + 3u : 15u);
#endif
        arrive(it);
      }
      wait(it);
      sycl::group_barrier(it.get_group());
      return;
    }
#endif
    // Where k-step k's B' lies in the ring.
    auto at = [&](uint32_t k) { return ((k / (uint32_t)KPS) % kSlots) * kSlotWords + (k % (uint32_t)KPS) * kStep; };
    auto mine = [&](int i) { return kHalf ? R + sgl * 16u : R + (sgl + (uint32_t)i * kNsg) * 32u; };
    each<KPS>([&](int k) {
      each<kBps>([&](int i) {
        cp_write(mine(i) + at((uint32_t)k), cp_read(Bbi[i] + (size_t)k * 2048u, lane), lane);
      });
    });
    cvec next[kBps];
    each<kBps>([&](int i) { next[i] = cp_read(Bbi[i] + (size_t)KPS * 2048u, lane); });
#if defined(PEARL_W_GCOPY)
    // goal-arc8: the copy once a slot: at a chunk's first k-step each sub-group stores the 512 bytes it loaded
    // a chunk ago (k-step sgl / 4 of the next slot, pieces 4 (sgl % 4) .. + 3) and loads the slot after.
    static_assert(KPS == 4 && CW == 1 && RW == 8, "the grouped copy is for the 8 by 1 work-group, a chunk a slot");
    const int8_t *Gsrc = A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride + (size_t)(sgl >> 2) * 2048u
                         + (size_t)(sgl & 3u) * 512u;
    uint32_t *Gdst = R + (sgl >> 2) * kStep + (sgl & 3u) * 128u;
    v4u gn[4];
    each<4>([&](int j) { gn[j] = ring_read_global<HW>(Gsrc + 8192u + (size_t)j * 128u, lane); });
#endif
    // A' of k-step ks: this sub-group's half of each of the window's 4 fragments (the split DPAS), or all of
    // each (otherwise). 8-row block rb of the window is in 16-row block rb >> 1.
#if defined(PEARL_W_LO32)
    // goal-arc8: global addresses as a fixed high word and a 32-bit low word (the host keeps each operand
    // inside one 4 GB-aligned window, so the low word never carries): one 32-bit add an address, where
    // 64-bit sums cost IGC three to five instructions each.
    auto lo32 = [](uint32_t hi, uint32_t lo) {
      return reinterpret_cast<const int8_t *>((uintptr_t)(((uint64_t)hi << 32) | (uint64_t)lo));
    };
    const uint32_t aHi = (uint32_t)((uint64_t)(uintptr_t)A.Ap >> 32);
    const uint32_t aLo = (uint32_t)(uintptr_t)A.Ap + 2u * wr * A.aStride + sgi * 128u;
    const uint32_t bHi = (uint32_t)((uint64_t)(uintptr_t)A.Bp >> 32);
    uint32_t bLo[kBps];
    each<kBps>([&](int i) { bLo[i] = (uint32_t)(uintptr_t)Bbi[i]; });
#if defined(PEARL_W_PTRV)
    // goal-arc8: each global stream as a (low, high) pair whose low word steps on: one add a load.
    v2u apv[4], bpv[kBps];
    each<4>([&](int rb) { apv[rb] = v2u{aLo + (uint32_t)(rb >> 1) * A.aStride + (uint32_t)(rb & 1) * 256u, aHi}; });
    each<kBps>([&](int i) { bpv[i] = v2u{bLo[i] + (uint32_t)(KPS + 1) * 2048u, bHi}; });
    auto vptr = [](v2u v) { return reinterpret_cast<const int8_t *>((uintptr_t)sycl::bit_cast<uint64_t>(v)); };
#endif
#endif
    auto load_ah = [&](uint32_t ks, v4i *ah) {
#if defined(PEARL_W_KMAJ)
      // goal-arc8 timing (wrong results): A' addressed k-step major (a k-step of every row window together),
      // the same bytes and reuse, no power-of-two stride between the streams.
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(A.Ap + (size_t)ks * (size_t)(rowWins * 1024u) + (size_t)wr * 1024u
                                               + (size_t)rb * 256u + sgi * 128u, lane);
      });
      return;
#endif
#if defined(PEARL_W_APAD)
      // goal-arc8 timing (wrong results): the 16-row blocks PEARL_W_APAD bytes closer than aStride.
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(A.Ap + (size_t)(2u * wr + (uint32_t)(rb >> 1)) * (A.aStride - (PEARL_W_APAD))
                                               + (size_t)ks * 512u + (size_t)(rb & 1) * 256u + sgi * 128u, lane);
      });
      return;
#endif
#if defined(PEARL_W_PTRV)
      (void)ks;
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(vptr(apv[rb]), lane);
        apv[rb][0] += 512u;
      });
      return;
#endif
#if defined(PEARL_W_LO32)
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(lo32(aHi, aLo + ks * 512u + (uint32_t)(rb >> 1) * A.aStride + (uint32_t)(rb & 1) * 256u), lane);
      });
      return;
#endif
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(Ab + (size_t)ks * 512u + (size_t)(rb >> 1) * A.aStride + (size_t)(rb & 1) * 256u
                                               + sgi * 128u, lane);
      });
    };
    v4i ahn[4];
    if constexpr (kSplit && AH) load_ah(0u, ahn);
    if constexpr (kPartX) {
      if (sgi == 0u && lane < 8u) each<16>([&](int j) { part[(widx * 8u + lane) * 16u + (uint32_t)j] = 0u; });
    }
    v8i bqn[4];
    (void)bqn;
    v8i c[2][8];
    each<2>([&](int rb) { each<8>([&](int cb) { c[rb][cb] = 0; }); });
    uint32_t tw[16];
    each<16>([&](int j) { tw[j] = 0u; });
#if defined(PEARL_W_AHALF)
    v4i ahp[4];
#endif
#if defined(PEARL_W_NOPUT)
    uint32_t sink = 0u;
#endif
#if defined(PEARL_W_XALU)
    uint32_t xa[4] = {A.zero, A.zero + 1u, A.zero + 2u, A.zero + 3u};
#endif
#if defined(PEARL_W_BHALF)
    v8i bqp[4];
#endif
    arrive(it);
    for (uint32_t ch = 0; ch < 16u; ch++) {
      for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t ks = ch * 4u + s;
#if defined(PEARL_W_ROT)
        // goal-arc8: column window jc takes its slot's k-steps rotated by jc * PEARL_W_ROT (a sum, so any order
        // inside a slot gives the same accumulators), so the windows sharing A' read it at different times and
        // the later one finds it in L1. Sharers reading together all miss.
        const uint32_t kd = ch * 4u + (s / (uint32_t)KPS) * (uint32_t)KPS + (s + jc * (uint32_t)(PEARL_W_ROT)) % (uint32_t)KPS;
#elif defined(PEARL_W_ROTW)
        // goal-arc8: window widx takes the chunk's k-steps rotated by widx * PEARL_W_ROTW.
        const uint32_t kd = ch * 4u + (s + widx * (uint32_t)(PEARL_W_ROTW)) % 4u;
#else
        const uint32_t kd = ks;
#endif
        const uint32_t slot = at(kd);
        // The group meets once a slot: every store into this slot has landed, and every read of the slot the
        // copy writes next is done. The copy stores k-step ks + KPS (loaded a k-step ago) and loads the one after.
#if !defined(PEARL_W_NOBAR)
        if (s % (uint32_t)KPS == 0u) wait(it);
#endif
#if defined(PEARL_W_GCOPY)
#ifndef PEARL_W_GLOAD_S
#define PEARL_W_GLOAD_S 0
#endif
        if (s == 0u)
          each<4>([&](int j) { ring_write_local<HW>(Gdst + ((ch + 1u) % 3u) * kSlotWords + (uint32_t)j * 32u, gn[j], lane); });
        if (s == (uint32_t)(PEARL_W_GLOAD_S)) {
          const uint32_t c2 = ch + 2u < 16u ? ch + 2u : 15u;
#if defined(PEARL_W_LO32)
          each<4>([&](int j) { gn[j] = ring_read_global<HW>(lo32(bHi, (uint32_t)(uintptr_t)Gsrc + c2 * 8192u + (uint32_t)j * 128u), lane); });
#else
          each<4>([&](int j) { gn[j] = ring_read_global<HW>(Gsrc + (size_t)c2 * 8192u + (size_t)j * 128u, lane); });
#endif
        }
#elif defined(PEARL_W_CSAME)
        // goal-arc8: the copy loads at the k-step's start and stores after its DPAS: nothing carried.
        v4u csv[kBps];
        each<kBps>([&](int i) { csv[i] = ring_read_global<HW>(Bbi[i] + (size_t)(ks + (uint32_t)KPS < 64u ? ks + (uint32_t)KPS : 63u) * 2048u, lane); });
#elif !defined(PEARL_W_NOCOPY) && (!defined(PEARL_W_NC) || defined(PEARL_W_NCIDLE))
        auto copy_step = [&]() {
          const uint32_t kn = ks + (uint32_t)KPS + 1u;
          each<kBps>([&](int i) {
#if defined(PEARL_W_NOFETCH)
            // goal-arc8 timing (wrong results): the copy stores without loading.
            const v4u v = next[i] ^ (ks * 0x9E3779B9u);
#else
#if defined(PEARL_W_PTRV)
            // past the last k-step it reads into the next block, for a slot no one reads (timing only: the host
            // leaves no room after the last block).
            const cvec v = cp_read(vptr(bpv[i]), lane);
            bpv[i][0] += 2048u;
#elif defined(PEARL_W_LO32)
#if defined(PEARL_W_NOCLAMP)
            // goal-arc8: past the last k-step the copy reads the next block's first k-steps (the host leaves
            // 16 KB after B'), stored into a slot no one reads: no clamp.
            const cvec v = cp_read(lo32(bHi, bLo[i] + kn * 2048u), lane);
#else
            const cvec v = cp_read(lo32(bHi, bLo[i] + (kn < 64u ? kn : 63u) * 2048u), lane);
#endif
#else
            const cvec v = cp_read(Bbi[i] + (size_t)(kn < 64u ? kn : 63u) * 2048u, lane);
#endif
#endif
#if defined(PEARL_W_NOPUT)
            // goal-arc8 timing (wrong results): the copy loads without storing.
            sink ^= v[0] ^ v[3];
#else
            cp_write(mine(i) + at(ks + (uint32_t)KPS), next[i], lane);
#endif
            next[i] = v;
          });
        };
#if !defined(PEARL_W_CLATE)
        copy_step();
#endif
#endif
#if !defined(PEARL_W_NOBAR) && !defined(PEARL_W_CSAME) && !defined(PEARL_W_CLATE) && !defined(PEARL_W_SLOTS)
        if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
        // B' fragment cb of this sub-group's 32 columns, from the ring.
        v8i bq[4];
        auto read_b_at = [&](uint32_t off, v8i *dst) {
#if defined(PEARL_W_GB)
          // goal-arc8: B' straight from memory (no ring), k-step kd; with PEARL_W_ROTW the windows reading a line
          // come at different k-steps, so all but the first may find it in L1. Use with NOCOPY and NOBAR.
          (void)off;
          each<4>([&](int cb) {
            const int8_t *g = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride + (size_t)kd * 2048u
                              + (size_t)(sgi * 4u + (uint32_t)cb) * 256u;
            const v4u lo = ring_read_global<HW>(g, lane), hi = ring_read_global<HW>(g + 128, lane);
            dst[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          return;
#endif
#if defined(PEARL_W_B8)
          // goal-arc8: a B' fragment as one 256-byte local block read.
          each<4>([&](int cb) { dst[cb] = ring_read_local8b<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
          return;
#endif
          each<4>([&](int cb) {
            uint32_t *p = R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u;
            const v4u lo = ring_read_local<HW>(p, lane), hi = ring_read_local<HW>(p + 32, lane);
            dst[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
        };
        auto read_b = [&]() {
#if defined(PEARL_W_BH)
          // goal-arc7 experiment: with 2 k-steps a slot, the first reads both k-steps' B', the second uses it.
          if constexpr (KPS == 2) {
            if (s % 2u == 0u) {
              read_b_at(slot, bq);
              read_b_at(at(ks + 1u), bqn);
            } else {
              each<4>([&](int cb) { bq[cb] = bqn[cb]; });
            }
            return;
          }
#endif
          read_b_at(slot, bq);
        };
        const int8_t *Ak = Ab + (size_t)kd * 512u;
        if constexpr (kSplit) {
          v4i ah[4];
          if constexpr (AH) {
            each<4>([&](int rb) { ah[rb] = ahn[rb]; });
            load_ah(ks + 1u < 64u ? ks + 1u : 63u, ahn);
          } else {
#if defined(PEARL_W_AHALF)
            // goal-arc8 timing (wrong results): A' loaded every other k-step, XORed so its bits still change.
            if ((s & 1u) == 0u) load_ah(kd, ahp);
            each<4>([&](int rb) { ah[rb] = ahp[rb] ^ (int)(ks * 0x85EBCA6Bu); });
#else
            load_ah(kd, ah);
#if defined(PEARL_W_XONLY)
            each<4>([&](int rb) { ah[rb] ^= (int)(ks * 0x85EBCA6Bu); });
#endif
#endif
          }
#if defined(PEARL_W_BHALF)
          if ((s & 1u) == 0u) read_b_at(slot, bqp);
          each<4>([&](int cb) { bq[cb] = bqp[cb] ^ (int)(ks * 0x9E3779B9u); });
#else
          read_b();
#if defined(PEARL_W_XONLY)
          each<4>([&](int cb) { bq[cb] ^= (int)(ks * 0x9E3779B9u); });
#endif
#endif
#if defined(PEARL_W_SLOTS)
          if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
#if defined(PEARL_W_XALU)
          // goal-arc8 timing: PEARL_W_XALU extra scalar ALU instructions a k-step, four chains.
          each<(PEARL_W_XALU) / 4>([&](int j) { each<4>([&](int m) { xa[m] = (xa[m] ^ (xa[m] << 3)) + ks + (uint32_t)j; }); });
#endif
#if defined(PEARL_W_CLATE)
          // goal-arc8: the copy after this k-step's local reads, so they are not ordered behind its store.
          copy_step();
          if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
#if defined(PEARL_W_AMASK)
          // goal-arc8 timing (wrong results): A' data ANDed with a mask (0: zeros), the loads kept.
          each<4>([&](int rb) { ah[rb] &= (int)(A.zero | (uint32_t)(PEARL_W_AMASK)); });
#endif
#if defined(PEARL_W_BMASK)
          each<4>([&](int cb) { bq[cb] &= (int)(A.zero | (uint32_t)(PEARL_W_BMASK)); });
#endif
#if defined(PEARL_W_AXOR)
          // goal-arc8 timing: with PEARL_W_AHOT, each window's A' made different again (addresses still hot).
          each<4>([&](int rb) { ah[rb] ^= (int)(wr * 0x9E3779B9u + (uint32_t)rb * 0x85EBCA6Bu); });
#endif
          // No forced wait: the DPAS wait only for their own operands. Forcing every DPAS of the k-step to wait
          // for every load of it (see fold_xmxf), PEARL_W_GATE, was 1% slower at the same 188 W.
#if defined(PEARL_W_GATE)
          int t = 0;
          each<4>([&](int rb) { t ^= ah[rb][0]; });
          each<4>([&](int cb) { t ^= bq[cb][0]; });
          t &= (int)A.zero;
          each<4>([&](int rb) { ah[rb][0] |= t; });
#endif
#if defined(PEARL_W_ROWFIRST)
          // goal-arc7 experiment: the DPAS of one A' fragment back to back.
          each<4>([&](int rb) {
            each<4>([&](int cb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
#else
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
#endif
#if defined(PEARL_W_CSAME)
          each<kBps>([&](int i) { ring_write_local<HW>(mine(i) + at(ks + (uint32_t)KPS), csv[i], lane); });
          if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
        } else {
          v8i af[4];
          each<4>([&](int rb) {
            const int8_t *p = Ak + (size_t)(rb >> 1) * A.aStride + (size_t)(rb & 1) * 256u;
            const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
            af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          read_b();
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = dpas<8, HW>(sg, af[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
        }
      }
      // The chunk's readout, the ring fold's.
      uint32_t x0 = 0, x1 = 0;
#if defined(PEARL_W_NORD)
      // goal-arc8 timing (wrong results): the readout reads one accumulator only.
      x0 = (uint32_t)c[0][0][0];
      x1 = (uint32_t)c[0][0][4];
#elif !defined(PEARL_W_NOVRD)
      // goal-arc8: the same XOR as a tree of whole accumulators first, then the 8 dwords (see kPartX).
      v8i t4[4];
      each<4>([&](int q) {
        t4[q] = (c[q >> 1][(q & 1) * 4] ^ c[q >> 1][(q & 1) * 4 + 1]) ^ (c[q >> 1][(q & 1) * 4 + 2] ^ c[q >> 1][(q & 1) * 4 + 3]);
      });
      const v8i t = (t4[0] ^ t4[1]) ^ (t4[2] ^ t4[3]);
      x0 = (uint32_t)((t[0] ^ t[1]) ^ (t[2] ^ t[3]));
      x1 = (uint32_t)((t[4] ^ t[5]) ^ (t[6] ^ t[7]));
#else
      each<2>([&](int rb) {
        each<8>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
#endif
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
      if constexpr (kRegWords) {
        // Chunk ch's word lands in tw[15]: a shift, so every index stays a constant.
        each<15>([&](int j) { tw[j] = tw[j + 1]; });
        tw[15] = (lane & 1u) ? x1 : x0;
      } else {
        if constexpr (kPartX) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
              px(part[(widx * 8u + (lane & 7u)) * 16u + ch]);
          if (lane < 8u) px.fetch_xor((lane & 1u) ? x1 : x0);
        } else {
          if (lane < 8u) part[(sgl * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
        }
      }
    }
    wait(it);
#if defined(PEARL_W_NOPUT)
    if (A.zero && sink == 0x12345678u) part[0] = sink;
#endif
#if defined(PEARL_W_XALU)
    if (A.zero && (xa[0] ^ xa[1] ^ xa[2] ^ xa[3]) == 0x12345678u) part[0] = xa[0];
#endif
    sycl::group_barrier(it.get_group());
    if constexpr (kRegWords) {
      // The ring is free now: sub-group 1 of each window leaves its words there for sub-group 0.
      if (sgi == 1u && lane < 8u) each<16>([&](int j) { R[(widx * 8u + lane) * 16u + (uint32_t)j] = tw[j]; });
      sycl::group_barrier(it.get_group());
      if (sgi == 0 && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        each<16>([&](int j) { words[j] = tw[j] ^ R[(widx * 8u + qr) * 16u + (uint32_t)j]; });
        const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
      return;
    }
    if (sgi == 0 && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
      for (int j = 0; j < 16; j++)
        words[j] = kPartX ? part[(widx * 8u + qr) * 16u + (uint32_t)j]
                          : part[(sgl * 8u + qr) * 16u + (uint32_t)j] ^ part[((sgl + 1u) * 8u + qr) * 16u + (uint32_t)j];
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
  }
};

// `groups` windows, 8 to a work-group (A.wpg must be 8: the copy is one block a sub-group).
#ifndef PEARL_W_KPS
#define PEARL_W_KPS 4
#endif
#ifndef PEARL_W_AH
#define PEARL_W_AH 0
#endif
#ifndef PEARL_W_RW
#define PEARL_W_RW 8
#endif
#ifndef PEARL_W_CW
#define PEARL_W_CW 1
#endif
template <bool HW, int KPS = PEARL_W_KPS, bool AH = (PEARL_W_AH != 0), int RW = PEARL_W_RW, int CW = PEARL_W_CW>
inline void fold_xmxw(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  typedef FoldXmxW<HW, KPS, AH, RW, CW> K;
  q.submit([&](sycl::handler &h) {
#ifndef PEARL_W_SLMPAD
#define PEARL_W_SLMPAD 0  // goal-arc8 timing: extra words of local memory a work-group (fewer work-groups a core)
#endif
    sycl::local_accessor<uint32_t, 1> part((K::kRegWords ? 1u : K::kNsg * (K::kPartX ? 4u : 8u) * 16u) + (PEARL_W_SLMPAD), h);
    sycl::local_accessor<uint32_t, 1> ring(K::kSlots * K::kSlotWords, h);
#if defined(PEARL_W_NC)
    constexpr uint32_t kSgs = K::kNsg + (PEARL_W_NC);  // goal-arc8: and the copier sub-groups
#else
    constexpr uint32_t kSgs = K::kNsg;
#endif
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / (RW * CW)) * kSgs * 8u, kSgs * 8u), K{a, part, ring});
  });
}

// ---------------------------------------------------------------------------
// The shared-A' fold (xmx8s): an experiment, built only with -DPEARL_SYCL_XMX8S
// (PEARL_SYCL_EXTRA). The AMD folds' way of feeding the matrix engines, on
// Alchemist: a work-group copies its A' into local memory once and its
// sub-groups read their fragments from there.
//
//   - A sub-group holds 32 rows by 32 columns (4 by 4 DPAS blocks, 16
//     accumulators), so a k-step reads 4 fragments of A' and 4 of B'.
//   - A work-group is 4 by 8 sub-groups: 4 row windows by 4 column windows,
//     each window two sub-groups split by columns. Its A' (8 blocks of 16
//     rows, 4 KB a k-step) is copied into a ring of 3 stages of 2 k-steps in
//     local memory, 128 bytes a message, and read there as one vector of 8
//     dwords a fragment. So a line of A' comes from memory once for 8
//     sub-groups, where xmx8f reads A' once a sub-group. B' is read from
//     memory as xmx8f reads it (two vectors of 4 dwords a fragment).
//   - Each k-step of a stage sends a sub-group's share of the copy for the
//     stage two ahead; the stage's end stores it to the ring and meets the
//     group. With -DPEARL_SYCL_SPLIT that meeting is a split barrier
//     (SPV_INTEL_split_barrier: arrive after the stores, wait a stage later),
//     which needs a translator extension a release must not ask for; without
//     it, a plain work-group barrier.
//   - Operands in fragment order (kFragA, kFragB), as xmx8f: the draw is the
//     same. The readout is xmx8f's: every block of a window XORs into the
//     same 8 region words; the window's two sub-groups meet in `part`.
//
// Timed first as stand-ins (pearl_sycl_peak, "t-" rows, card 5): 66 to 68
// T-MAC/s at 174 to 177 W against xmx8f's loads at 61.6 and 178 W.
// ---------------------------------------------------------------------------
#if defined(PEARL_SYCL_XMX8S)
#if defined(PEARL_SYCL_SPLIT) && defined(__SYCL_DEVICE_ONLY__) && !defined(PEARL_SYCL_XMX_JIT)  // the JIT build declares them above
}  // namespace pearl_sycl
SYCL_EXTERNAL void __spirv_ControlBarrierArriveINTEL(int, int, int) noexcept;
SYCL_EXTERNAL void __spirv_ControlBarrierWaitINTEL(int, int, int) noexcept;
namespace pearl_sycl {
#endif

#if defined(__SYCL_DEVICE_ONLY__)
#define PEARL_LOCAL __attribute__((opencl_local))
#else
#define PEARL_LOCAL
#endif

template <bool HW> struct FoldXmxS {
  static constexpr uint32_t GA = 4, GB = 8, N = GA * GB;  // sub-groups: 4 row windows by 8 half column windows
  static constexpr uint32_t KS = 2, R = 3, D = 2;          // k-steps a stage, stages in the ring, stages ahead
  static constexpr uint32_t T1 = 8u * 512u, T = KS * T1;   // ring bytes: a k-step's A' (8 blocks), a stage
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> ring;  // R * T bytes
  sycl::local_accessor<uint32_t, 1> part;  // N * 8 * 16 words

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (HW)
      return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
    else
      return syclex::properties{syclex::sub_group_size<8>};
  }

  // The group meets: a sub-group may then store into the ring slot others read two stages ago.
  static void arrive() {
#if defined(PEARL_SYCL_SPLIT) && defined(__SYCL_DEVICE_ONLY__)
    if constexpr (HW) __spirv_ControlBarrierArriveINTEL(2, 2, 0x104);
#endif
  }
  static void wait(sycl::nd_item<1> it) {
#if defined(PEARL_SYCL_SPLIT) && defined(__SYCL_DEVICE_ONLY__)
    if constexpr (HW) { __spirv_ControlBarrierWaitINTEL(2, 2, 0x102); return; }
#endif
    sycl::group_barrier(it.get_group());
  }
  static constexpr bool split() {
#if defined(PEARL_SYCL_SPLIT)
    return HW;
#else
    return false;
#endif
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();
    const uint32_t si = sgl % GA, sj = sgl / GA;  // its row window in the group, its half column window
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t bandS = A.band / GA;
    if (bandS < 1u) bandS = 1u;
    uint32_t swr, swc;
    window_of((uint32_t)it.get_group(0), rowWins / GA, colWins / (GB / 2u), bandS, swr, swc);
    const uint32_t wr0 = GA * swr, wc0 = (GB / 2u) * swc;
    const uint32_t wr = wr0 + si, wc = wc0 + sj / 2u, half = sj % 2u;
    // The group's A': 16-row blocks 2 wr0 .. 2 wr0 + 7. Piece q (0..31) of a k-step: block q / 4, its
    // 128-byte quarter q % 4, a lane's 16 bytes of it.
    const int8_t *Ag = A.Ap + (size_t)(2u * wr0) * A.aStride + 16u * lane;
    auto piece = [&](uint32_t q, uint32_t ks) -> const int8_t * {
      return Ag + (size_t)(q / 4u) * A.aStride + (size_t)ks * 512u + (size_t)(q % 4u) * 128u;
    };
    // This sub-group's B': the window's 64-column block, its half's 4 fragments.
    const int8_t *Bb = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride + (size_t)half * 1024u + 32u * lane;
    PEARL_LOCAL int8_t *rb8 =
        reinterpret_cast<PEARL_LOCAL int8_t *>(ring.template get_multi_ptr<sycl::access::decorated::yes>().get_decorated());
    // A stage holds KS k-steps of 32 pieces; this sub-group copies piece sgl of each.
    auto store_piece = [&](uint32_t slot, uint32_t kk, const v4iu &v) {
      *reinterpret_cast<PEARL_LOCAL v4iu *>(rb8 + (size_t)slot * T + kk * T1 + sgl * 128u + 16u * lane) = v;
    };
    for (uint32_t st = 0; st < D; st++)
      for (uint32_t kk = 0; kk < KS; kk++) store_piece(st, kk, *reinterpret_cast<const v4iu *>(piece(sgl, st * KS + kk)));
    sycl::group_barrier(it.get_group());

    v8i c[4][4];
    each<4>([&](int r) { each<4>([&](int b) { c[r][b] = 0; }); });
    constexpr uint32_t NST = 64u / KS;
    for (uint32_t st = 0; st < NST; st++) {
      const bool more = st + D < NST;  // a stage two ahead to copy
      v4iu held[KS];
      const PEARL_LOCAL int8_t *sl0 = rb8 + (size_t)(st % R) * T;
      each<(int)KS>([&](int kk) {
        const uint32_t ks = st * KS + (uint32_t)kk;
        if (more) held[kk] = *reinterpret_cast<const v4iu *>(piece(sgl, (st + D) * KS + (uint32_t)kk));
        v8i af[4], bf[4];
        each<4>([&](int r) {
          af[r] = *reinterpret_cast<const PEARL_LOCAL v8iu *>(sl0 + (size_t)kk * T1 + (2u * si + (uint32_t)r / 2u) * 512u
                                                  + ((uint32_t)r % 2u) * 256u + 32u * lane);
        });
        each<4>([&](int b) { bf[b] = load_halves(Bb + (size_t)ks * 2048u + (size_t)b * 256u); });
        int w = 0;
        each<4>([&](int r) { w ^= af[r][0]; });
        each<4>([&](int b) { w ^= bf[b][0]; });
        w &= (int)A.zero;
        each<4>([&](int r) { af[r][0] |= w; });
        each<4>([&](int b) { each<4>([&](int r) { c[r][b] = dpas<8, HW>(sg, af[r], bf[b], c[r][b]); }); });
      });
      // A chunk is 4 k-steps, two stages: its readout after the second.
      if (st & 1u) {
        uint32_t x0 = 0, x1 = 0;
        each<4>([&](int r) {
          each<4>([&](int b) {
            const v8i v = c[r][b];
            x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
            x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
          });
        });
        x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
        x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
        if (lane < 8u) part[(sgl * 8u + lane) * 16u + st / 2u] = (lane & 1u) ? x1 : x0;
      }
      if constexpr (split()) {
        if (st > 0u) wait(it);
        if (more) each<(int)KS>([&](int kk) { store_piece((st + D) % R, (uint32_t)kk, held[kk]); });
        arrive();
      } else {
        if (more) each<(int)KS>([&](int kk) { store_piece((st + D) % R, (uint32_t)kk, held[kk]); });
        sycl::group_barrier(it.get_group());
      }
    }
    if constexpr (split()) wait(it);
    sycl::group_barrier(it.get_group());
    // The window's two sub-groups (half 0 and 1 are 4 sub-groups apart); half 0's lanes 0..7 finish it.
    if (half == 0u && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
      for (int j = 0; j < 16; j++)
        words[j] = part[(sgl * 8u + qr) * 16u + (uint32_t)j] ^ part[((sgl + GA) * 8u + qr) * 16u + (uint32_t)j];
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
  }
};

// `groups` is the batch's windows, as for the other folds; a work-group takes 16 of them. The batch must be
// whole groups: rowsValid / 2 and colBatch / 4 multiples of 4.
template <bool HW> inline void fold_xmxs(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  typedef FoldXmxS<HW> K;
  if ((a.rowsValid / 2u) % K::GA || (a.colBatch / 4u) % (K::GB / 2u))
    throw std::runtime_error("xmx8s needs row windows and column windows in fours");
  const uint32_t wgs = groups / (K::GA * K::GB / 2u);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> ring(K::R * K::T / 4u, h), part(K::N * 8u * 16u, h);
    h.parallel_for(sycl::nd_range<1>((size_t)wgs * K::N * 8u, (size_t)K::N * 8u), K{a, ring, part});
  });
}
#endif  // PEARL_SYCL_XMX8S


}  // namespace pearl_sycl

#endif  // PEARL_SYCL_KERNELS_HPP
