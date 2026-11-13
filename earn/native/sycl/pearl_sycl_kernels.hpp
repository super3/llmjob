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
#if defined(PEARL_SYCL_ESIMD) && defined(PEARL_SYCL_XMX_JIT)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#endif

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
// goal-arc12: the 4-row form (dpasw.8x4): each sub-group of the pair gives 2 rows (2 dwords a lane), each gets
// all 4 rows times its own B. The finalizer's dummy-DPAS workaround (Wa_18023229625) applies only to 8x8.
typedef int v2i __attribute__((ext_vector_type(2)));
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
extern SYCL_EXTERNAL v4i intel_sub_group_i8_i8_split_matrix_mad_k32(v2i, v8i, v4i);
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_W_D2T)
extern SYCL_EXTERNAL v2i intel_sub_group_i8_i8_split_matrix_mad_k32(int, v8i, v2i);  // goal-arc27 timing: 8x2
#endif
// goal-arc23 (PEARL_W_U8): the split DPAS with unsigned operands. 1: A' and B' u8; 2: B' only; 3: A' only. With
// PEARL_OPERAND_FILL=63 B' is 0..126 (A' too but for the salt stamp's bytes), so 2 is exact. On NVIDIA unsigned
// operands cost 2% fewer watts; here fill 48, fill 63, 63 with 2 and 63 with 1 all ran 102.8-103.0 at the same
// 175 W and clock (card 5, stream on). Off.
typedef uint32_t v2uu __attribute__((ext_vector_type(2)));
typedef uint32_t v8uu __attribute__((ext_vector_type(8)));
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_W_U8)
extern SYCL_EXTERNAL v4i intel_sub_group_u8_u8_split_matrix_mad_k32(v2uu, v8uu, v4i);
extern SYCL_EXTERNAL v4i intel_sub_group_i8_u8_split_matrix_mad_k32(v2i, v8uu, v4i);
extern SYCL_EXTERNAL v4i intel_sub_group_u8_i8_split_matrix_mad_k32(v2uu, v8i, v4i);
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
#if defined(PEARL_W_SMP)
  uint64_t img;       // goal-arc20 timing: a bindless image's raw handle (see FoldXmxW, PEARL_W_SMP)
#endif
#if defined(PEARL_W_STAG)
  uint32_t *tick;     // goal-arc20 timing: 32 counters, one an Xe-core, zeroed before each launch (PEARL_W_STAG)
#endif
#if defined(PEARL_W_TSC)
  uint64_t *tsc;      // goal-arc23 measurement: 4 words a work-group (start, end, sr0, 0) of one batch
#endif
};

// xmx8w walks a batch in chunks of this many column windows (256 column offsets), each one band walk (FoldXmxW).
#ifndef PEARL_W_CHUNKW
#define PEARL_W_CHUNKW 64
#endif
constexpr uint32_t kChunkWins = PEARL_W_CHUNKW;

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

#if defined(PEARL_W_NOINL)
// goal-arc10 experiment: the epilogue as a function of its own (IGC_FunctionControl decides subroutine or stack call).
__attribute__((noinline)) void region_epilogue_ni(const uint32_t words[16], uint32_t local, const TranscriptTest &t,
                                                  HitBuf *hits, uint32_t *allTr) {
  region_epilogue(words, local, t, hits, allTr);
}
#endif

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
template <bool HW> inline v4i split_dpas4(v2i a, v8i b, v4i c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_W_U8)
  if constexpr (HW && (PEARL_W_U8) == 1) return ::intel_sub_group_u8_u8_split_matrix_mad_k32((v2uu)a, (v8uu)b, c);
  if constexpr (HW && (PEARL_W_U8) == 2) return ::intel_sub_group_i8_u8_split_matrix_mad_k32(a, (v8uu)b, c);
  if constexpr (HW && (PEARL_W_U8) == 3) return ::intel_sub_group_u8_i8_split_matrix_mad_k32((v2uu)a, b, c);
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) return ::intel_sub_group_i8_i8_split_matrix_mad_k32(a, b, c);
#endif
  (void)a;
  (void)b;
  return c;
}
template <bool HW> inline v2i split_dpas2(int a, v8i b, v2i c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_W_D2T)
  if constexpr (HW) return ::intel_sub_group_i8_i8_split_matrix_mad_k32(a, b, c);
#endif
  (void)a;
  (void)b;
  return c;
}
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

// goal-arc10: region_epilogue for a transcript that sits in local memory, reading each message word from there
// when the hash needs it (volatile, so the compiler cannot keep the 16 words in registers). The hash then lives
// in about 20 registers instead of 40, which lets the register allocator keep it out of the registers the fold's
// B' fragments use: Intel's finalizer puts a dummy DPAS before a run of DPAS whose B' registers any ALU
// instruction of the kernel writes (Wa_18023229625).
template <typename P> inline void region_epilogue_lean(P w, uint32_t local, const TranscriptTest &t, HitBuf *hits,
                                                       uint32_t *allTr, uint32_t zero) {
  using namespace pearl_b3;
  if (allTr)
    for (int j = 0; j < 16; j++) allTr[(size_t)local * 16 + j] = w[j];
  uint32_t s[16];
  each<8>([&](int i) { s[i] = t.key[i]; });
  s[8] = iv(0); s[9] = iv(1); s[10] = iv(2); s[11] = iv(3);
  s[12] = 0u; s[13] = 0u; s[14] = 64u;
  s[15] = kChunkStart | kChunkEnd | kRoot | kKeyed;
  // Message word at position i of round r: the permutation applied r times.
  constexpr uint8_t perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  each<7>([&](int r) {
    // Each word's address depends on the state just before its use (ANDed with zero, which the compiler cannot
    // know), so the loads cannot all be issued at the top of the round.
    auto mi = [&](int i, uint32_t dep) {
      int x = i;
      for (int k = 0; k < r; k++) x = perm[x];
      return (uint32_t)w[(uint32_t)x + (dep & zero)];
    };
    auto gg = [&](int a, int b, int c, int d, int i) {
      const uint32_t x = mi(i, s[a]);
      s[a] = s[a] + s[b] + x;
      s[d] = rotr(s[d] ^ s[a], 16);
      s[c] = s[c] + s[d];
      s[b] = rotr(s[b] ^ s[c], 12);
      const uint32_t y = mi(i + 1, s[b]);
      s[a] = s[a] + s[b] + y;
      s[d] = rotr(s[d] ^ s[a], 8);
      s[c] = s[c] + s[d];
      s[b] = rotr(s[b] ^ s[c], 7);
    };
    gg(0, 4, 8, 12, 0);
    gg(1, 5, 9, 13, 2);
    gg(2, 6, 10, 14, 4);
    gg(3, 7, 11, 15, 6);
    gg(0, 5, 10, 15, 8);
    gg(1, 6, 11, 12, 10);
    gg(2, 7, 8, 13, 12);
    gg(3, 4, 9, 14, 14);
  });
  uint32_t h[8];
  each<8>([&](int i) { h[i] = s[i] ^ s[i + 8]; });
  if (!meets(h, t.target_w, t.hash_big_endian)) return;
  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      cnt(hits->count);
  const uint32_t n = cnt.fetch_add(1u);
  if (n >= PEARL_MAX_HITS) return;
  hits->index[n] = local;
  for (int i = 0; i < 8; i++) hits->hash[n * 8 + i] = h[i];
  for (int j = 0; j < 16; j++) hits->transcript[n * 16 + j] = w[j];
}

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
// goal-arc13: a 256-byte fragment from local memory as ONE message, by inline vISA. A transposed block load
// writes memory into registers in order, so a v8i loaded this way holds the same dwords as ring_read_local8b
// (lane j, element e = dword 8 e + j); IGC itself splits a 256-byte block read into two 128-byte messages.
template <bool HW> inline v8i slm_read256(uint32_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    v8i r;
    const uint32_t a = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)p;
#if defined(PEARL_W_D64)
    __asm__ volatile("lsc_load.slm (M1_NM, 1) %0:d64x32t flat[%1]:a32" : "=rw"(r) : "rw"(a));
#elif defined(PEARL_W_BANV)
    __asm__("lsc_load.slm (M1_NM, 1) %0:d32x64t flat[%1]:a32" : "=rw"(r) : "rw"(a));
#else
    __asm__ volatile("lsc_load.slm (M1_NM, 1) %0:d32x64t flat[%1]:a32" : "=rw"(r) : "rw"(a));
#endif
    return r;
  }
#endif
  v8i r;
  for (int i = 0; i < 8; i++) r[i] = (int)p[8u * (uint32_t)i + lane];
  return r;
}
// goal-arc17: prefetches into L1 and L2 (a load with no destination), 256 or 128 bytes, one message each. Used
// only by the PEARL_W_PFA / PFB experiments, which lost (see there).
template <bool HW> inline void g_pf256(const int8_t *p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    const uint64_t a = (uint64_t)(uintptr_t)p;
    __asm__ volatile("lsc_load.ugm.ca.ca (M1_NM, 1) %%null:d32x64t flat[%0]:a64" : : "rw"(a));
  }
#endif
  (void)p;
}
template <bool HW> inline void g_pf128(const int8_t *p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    const uint64_t a = (uint64_t)(uintptr_t)p;
    __asm__ volatile("lsc_load.ugm.ca.ca (M1_NM, 1) %%null:d32x32t flat[%0]:a64" : : "rw"(a));
  }
#endif
  (void)p;
}
// goal-arc13: the same for 256 bytes of global memory, and a 256-byte local store, each one message.
template <bool HW> inline v8i g_read256(const int8_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    v8i r;
    const uint64_t a = (uint64_t)(uintptr_t)p;
#if defined(PEARL_W_AANV)
    // goal-arc16: not volatile, so LLVM may move it (A' is read-only for the whole kernel)
    __asm__("lsc_load.ugm.ca.ca (M1_NM, 1) %0:d32x64t flat[%1]:a64" : "=rw"(r) : "rw"(a));
#else
    __asm__ volatile("lsc_load.ugm.ca.ca (M1_NM, 1) %0:d32x64t flat[%1]:a64" : "=rw"(r) : "rw"(a));
#endif
    return r;
  }
#endif
  const uint32_t *w = reinterpret_cast<const uint32_t *>(p);
  v8i r;
  for (int i = 0; i < 8; i++) r[i] = (int)w[8u * (uint32_t)i + lane];
  return r;
}
template <bool HW> inline void slm_write256(uint32_t *p, v8i v, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    const uint32_t a = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)p;
    __asm__ volatile("lsc_store.slm (M1_NM, 1) flat[%0]:a32 %1:d32x64t" : : "rw"(a), "rw"(v));
    return;
  }
#endif
  for (int i = 0; i < 8; i++) p[8u * (uint32_t)i + lane] = (uint32_t)v[i];
}
// goal-arc16: 512 bytes of local memory in ONE message, as a SIMD8 gather of 64 bytes a lane (d64x8, not
// transposed): lane j reads the 64 bytes at p + 64 j. Register e of the result holds qword e / 2 of lanes
// 4 (e % 2) .. 4 (e % 2) + 3, so the data must be laid out for it (see PEARL_W_BG).
typedef int v16i __attribute__((ext_vector_type(16)));
template <bool HW> inline v16i slm_gather512(uint32_t *p, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  if constexpr (HW) {
    v16i r;
    const uint32_t a = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)p + lane * 64u;
    __asm__ volatile("lsc_load.slm (M1_NM, 8) %0:d64x8 flat[%1]:a32" : "=rw"(r) : "rw"(a));
    return r;
  }
#endif
  v16i r;
  for (int i = 0; i < 16; i++) r[i] = (int)p[16u * lane + (uint32_t)i];
  return r;
}
// goal-arc13 timing of the copy's store form: 128 bytes to local memory by inline vISA, d32x32t (STA 1) or
// d64x16t (STA 2, IGC's own choice).
template <bool HW> inline void slm_write128(uint32_t *p, v4u v, uint32_t lane) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_W_STA)
  if constexpr (HW) {
    const uint32_t a = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)p;
#if PEARL_W_STA == 1
    __asm__ volatile("lsc_store.slm (M1_NM, 1) flat[%0]:a32 %1:d32x32t" : : "rw"(a), "rw"(v));
#else
    __asm__ volatile("lsc_store.slm (M1_NM, 1) flat[%0]:a32 %1:d64x16t" : : "rw"(a), "rw"(v));
#endif
    return;
  }
#endif
  for (int i = 0; i < 4; i++) p[8u * (uint32_t)i + lane] = v[i];
}
// goal-arc21 (PEARL_W_SA): the k-step's messages with their addresses made inside the asm, as one scalar add of
// an immediate to a base the chunk sets. IGC builds each address as an 8-lane value (a broadcast mov for every
// local message, two strided movs for every 64-bit one), and in this loop every instruction costs: about 0.05% a
// chunk each (goal-arc21, PEARL_W_XALU priced on spare30's code). Only for the HW path of the split fold.
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
// The address temporary is an output of the asm (IGC gives it a register), not a .decl in braces: braces kept
// the finalizer from moving the messages among the DPAS (98.5 against 101.7, card 5).
template <int OFF> inline v8i sa_slm_rd256(uint32_t base) {
  v8i r;
  if constexpr (OFF == 0) {
    __asm__ volatile("lsc_load.slm (M1_NM, 1) %0:d32x64t flat[%1]:a32" : "=rw"(r) : "rw"(base));
  } else {
    uint32_t t;
    __asm__ volatile("add (M1_NM, 1) %1(0,0)<1> %2(0,0)<0;1,0> %3\n"
                     "lsc_load.slm (M1_NM, 1) %0:d32x64t flat[%1]:a32"
                     : "=rw"(r), "=rw"(t) : "rw"(base), "i"(OFF));
  }
  return r;
}
template <int OFF> inline void sa_slm_st128(uint32_t base, v4u v) {
  if constexpr (OFF == 0) {
    __asm__ volatile("lsc_store.slm (M1_NM, 1) flat[%0]:a32 %1:d64x16t" : : "rw"(base), "rw"(v));
  } else {
    uint32_t t;
    __asm__ volatile("add (M1_NM, 1) %0(0,0)<1> %1(0,0)<0;1,0> %3\n"
                     "lsc_store.slm (M1_NM, 1) flat[%0]:a32 %2:d64x16t"
                     : "=rw"(t) : "rw"(base), "rw"(v), "i"(OFF));
  }
}
// A 64-bit base plus OFF (a 64-bit add of an immediate).
#define PEARL_SA_G(CC, DT, OUT)                                                     \
  do {                                                                              \
    uint64_t t;                                                                     \
    __asm__ volatile("add (M1_NM, 1) %1(0,0)<1> %2(0,0)<0;1,0> %3\n"                \
                     "lsc_load.ugm." CC " (M1_NM, 1) %0:" DT " flat[%1]:a64"        \
                     : "=rw"(OUT), "=rw"(t) : "rw"(base), "i"(OFF));                \
  } while (0)
template <int OFF> inline v8i sa_g256(uint64_t base) {
  v8i r;
  PEARL_SA_G("ca.ca", "d32x64t", r);
  return r;
}
template <typename F, size_t... I> inline void eachc_of(F &f, std::index_sequence<I...>) {
  (f(std::integral_constant<int, (int)I>{}), ...);
}
template <int N, typename F> inline void eachc(F f) { eachc_of(f, std::make_index_sequence<N>{}); }
#endif
// goal-arc13: 512 bytes each way as two 256-byte messages (d64x64t does not assemble on DG2).
typedef int v16i __attribute__((ext_vector_type(16)));
template <bool HW> inline v16i g_read512(const int8_t *p, uint32_t lane) {
  const v8i lo = g_read256<HW>(p, lane), hi = g_read256<HW>(p + 256, lane);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
template <bool HW> inline void slm_write512(uint32_t *p, v16i v, uint32_t lane) {
  slm_write256<HW>(p, __builtin_shufflevector(v, v, 0, 1, 2, 3, 4, 5, 6, 7), lane);
  slm_write256<HW>(p + 64, __builtin_shufflevector(v, v, 8, 9, 10, 11, 12, 13, 14, 15), lane);
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
#if !defined(PEARL_W_NOD4)
#define PEARL_W_D4  // goal-arc12: 8x4 split DPAS (see below); PEARL_W_NOD4 for the 8x8 form
#endif
#if !defined(PEARL_W_NOLO32)
#define PEARL_W_LO32  // goal-arc8: 32-bit address sums (see lo32); the host keeps each operand in one 4 GB window
#endif
// goal-arc13: A' and B' read as 256-byte messages by inline vISA (PEARL_W_AA, PEARL_W_BA; see slm_read256): IGC
// splits every 256-byte block read into two 128-byte messages. Together 97.9-98.0 against 96.1-96.5 TH/s (card
// 5); BA alone is level. PEARL_W_NOAA, PEARL_W_NOBA for the old forms. AA needs the 32-bit address sums.
#if !defined(PEARL_W_NOAA) && defined(PEARL_W_LO32)
#define PEARL_W_AA
#endif
// BA is off again from e7a3e10a's successor: at 16 row windows HiveOS's 24.35 finalizer gets most regions
// wrong with it under any options (its ISA reloads a B' register by the inline load with no wait on the DPAS
// still reading it), while the new driver passes. Without BA: 98.8-98.9 against 99.5-99.7 (card 5), and 24.35
// passes. PEARL_W_BA turns it on.
// goal-arc13: 16 row windows a work-group (32 sub-groups), each copying 128 bytes of B' at even k-steps only
// (PEARL_W_C16): half the copy's local stores a MAC, and each thread stores as often as the half-copy timing
// build that ran 105.3 at 8 windows. 99.6 against 98.5 for 8 windows (card 5; 16 windows with the old 64-byte
// copy 99.3). PEARL_W_RW=8 for the 8-window form; the host runs this one only where 16 divides the band and the
// row windows.
// goal-arc15: the default is 8 row by 2 column windows a work-group with BA (goal-arc14's opt-in): the two
// column windows read the same A' in step, so the second read is an L1 hit, and A' from L2 is what costs.
// Cool card 4: 101.8-101.9 TH/s against 99.9-100.3 for 16 by 1 (spare23). The host runs CW 1 where the
// column windows are not a multiple of 2 (the check's 4-column batches). PEARL_W_RW=16 for the old default.
#ifndef PEARL_W_RW
#define PEARL_W_RW 8
#endif
#if PEARL_W_RW == 16 && !defined(PEARL_W_NOC16)
#define PEARL_W_C16
#endif
#if PEARL_W_RW == 8 && !defined(PEARL_W_CW)
#define PEARL_W_CW 2
#endif
#if PEARL_W_RW == 8 && !defined(PEARL_W_NOBA) && !defined(PEARL_W_BA)
#define PEARL_W_BA
#endif
// SEQ (goal-arc10): column windows a work-group runs one after the other (1 or 2). With 2, sub-group sgi of each
// window hashes set sgi's words, both at once, so the hash at the end (which holds the work-group's local memory
// and threads while it runs) is paid once for twice the work.
// goal-arc17 timing (unsafe: the copy can overwrite a slot still being read): the group meets every PEARL_W_BARN
// chunks instead of every chunk, to price the meeting itself. Card 5: 2 / 4 chunks 97.2 / 96.9 against 100.3-100.4,
// and one more meeting mid-chunk (PEARL_W_BARX) 100.2-100.3: meeting less loses (the windows drift apart).
#ifndef PEARL_W_BARN
#define PEARL_W_BARN 1
#endif
// goal-arc18: the copy's 128-byte load as one inline-vISA message (PEARL_W_CPCC, its cache control), in work-groups
// of at least PEARL_W_CPCW column windows. IGC's block read sends the same message; the gain is where the volatile
// load lands in the k-step. Card 5, full speed (CW 2): 101.2-101.3 against 100.0-100.3; uc.ca and st.ca the same,
// uc.uc 71.9. Under duty 75 (the CW 1 instance) it lost 0.4% at ca.ca, so CW 1 uses its own cache control
// (goal-arc19, PEARL_W_CPCC1 below). With the inline store too (STA) 100.6-100.8; with SLOTS 2 + NOPARTX 100.2-100.4. PEARL_W_NOCPA: off.
#if !defined(PEARL_W_NOCPA) && !defined(PEARL_W_CPCC)
#define PEARL_W_CPCC "ca.ca"
#endif
#ifndef PEARL_W_CPCW
#define PEARL_W_CPCW 2  // PEARL_W_CPCC applies to work-groups of at least this many column windows
#endif
// goal-arc19: the CW 1 instance (the duty path) sends the inline load too, at its own cache control PEARL_W_CPCC1
// (streaming in L1). Card 5, duty 75: st.ca / uc.ca +0.3%, ca.ca -0.4%. Card 4, duty 75, 8 rounds: 77.27-77.50
// against spare28's 77.08-77.42 (level to +0.1%); full speed unchanged (same CW 2 ISA). PEARL_W_NOCPC1: off.
#if !defined(PEARL_W_NOCPC1) && !defined(PEARL_W_NOCPA) && !defined(PEARL_W_CPCC1)
#define PEARL_W_CPCC1 "st.ca"
#endif
// goal-arc20: the part buffer in PXT 2's layout (word j of region q of window w at (16 w + j) * 8 + q), zeroed and
// read back for the hash as 256-byte block messages (PEARL_W_PXB), with the chunk's atomic by inline vISA
// (PEARL_W_PXA). In the old layout a lane's 16 words lay 64 bytes from the next lane's, so the 16 zeroing stores, the
// 16 reads before the hash and every chunk's atomic put all 8 lanes on one bank; goal-arc18's PXT lost 4-6%
// because its zeroing and reads stayed 16 messages each. Card 5, full speed, 6 rounds: 101.54-101.78 against
// 101.27-101.47 at the same or a lower clock. PEARL_W_NOPXB: the old layout.
#if !defined(PEARL_W_NOPXB) && !defined(PEARL_W_PXT) && !defined(PEARL_W_PARTMIN)
#define PEARL_W_PXT 2
#define PEARL_W_PXB
#define PEARL_W_PXA
#endif
// goal-arc21: the sets of a work-group can run as one stream (PEARL_W_SEQP, see run_set in FoldXmxW), in
// work-groups of at least PEARL_W_SEQPCW column windows (the full-speed instance; under duty 75 the CW 1 instance
// lost 0.5% with it). The instance is compiled in (template switch SP); the host runs it only with
// PEARL_SYCL_SEQP=1, because HiveOS's 24.35 driver gets most regions wrong in it (every form tried: see run_set).
// Card 4, full speed: 103.6-103.8 TH/s against 102.8-103.0. PEARL_W_NOSEQP: not compiled. goal-arc23: the S24
// instance (see kSqM) is the stream in forms 24.35 compiles right, and the host's default on a 24.x driver.
#if !defined(PEARL_W_NOSEQP) && !defined(PEARL_W_SEQP)
#define PEARL_W_SEQP
#endif
#if defined(PEARL_W_PFG) && !defined(PEARL_W_PFGS)
#define PEARL_W_PFGS 0  // the k-steps (even 0, odd 1) at which the A' line prefetch goes out
#endif
#if defined(PEARL_W_PFG) && !defined(PEARL_W_PFGC)
#define PEARL_W_PFGC "ca.ca"
#endif
#ifndef PEARL_W_SEQPCW
#define PEARL_W_SEQPCW 2
#endif
#if defined(PEARL_W_PFN) && !defined(PEARL_W_PFNC)
#define PEARL_W_PFNC "uc.ca"  // the next band's A' prefetch: L1 uncached, L2 cached
#endif
#if defined(PEARL_W_PFT) && !defined(PEARL_W_PFTC)
#define PEARL_W_PFTC "uc.ca"  // the hash tail's A' loads for the next band: L1 uncached, L2 cached
#endif
#if defined(PEARL_W_PFN) && !defined(PEARL_W_PFNB)
#define PEARL_W_PFNB 1  // which band ahead (0: the work-group's own)
#endif
#if defined(PEARL_W_PFN) && !defined(PEARL_W_PFN0)
#define PEARL_W_PFN0 0  // the first work-group of the band (its index in the band) that prefetches
#endif
#if defined(PEARL_W_SA) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
// goal-arc21 (PEARL_W_SA): the copy's 128-byte load with its cache control (PEARL_W_CPCC, PEARL_W_CPCC1).
template <int OFF, int CW> inline v4u sa_g128(uint64_t base) {
  v4u r;
#if defined(PEARL_W_CPCC1)
  if constexpr (CW == 1) {
    PEARL_SA_G(PEARL_W_CPCC1, "d32x32t", r);
    return r;
  }
#endif
#if defined(PEARL_W_CPCC)
  PEARL_SA_G(PEARL_W_CPCC, "d32x32t", r);
#else
  PEARL_SA_G("ca.ca", "d32x32t", r);
#endif
  return r;
}
#endif
template <bool HW, int KPS = 1, bool AH = false, int RW = 8, int CW = 1, int SEQ = 1, bool SP = false, bool S24 = false> struct FoldXmxW {
  static_assert(SEQ == 1 || SEQ == 2 || SEQ == 4 || SEQ == 8 || SEQ == 16, "1 to 16 column windows in turn");
  static_assert(KPS == 1 || KPS == 2 || KPS == 4, "a slot is 1, 2 or 4 k-steps: the group meets inside a chunk");
  static_assert(RW == 2 || RW == 4 || RW == 8 || (RW == 16 && (CW == 1 || CW == 2)), "each sub-group copies a whole number of blocks");
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
#if defined(PEARL_W_CPCC) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      // goal-arc18 (PEARL_W_CPCC, see there): the copy's load as one inline 128-byte message with its own cache
      // control. Tried uncached in L1 to leave room for A'; every L1 option gained the same 1.1%.
      if constexpr (HW && (CW >= (PEARL_W_CPCW))) {
        v4u r;
        const uint64_t a = (uint64_t)(uintptr_t)p;
#if defined(PEARL_W_CPNV)
        __asm__("lsc_load.ugm." PEARL_W_CPCC " (M1_NM, 1) %0:d32x32t flat[%1]:a64" : "=rw"(r) : "rw"(a));
#else
        __asm__ volatile("lsc_load.ugm." PEARL_W_CPCC " (M1_NM, 1) %0:d32x32t flat[%1]:a64" : "=rw"(r) : "rw"(a));
#endif
        return r;
      }
#if defined(PEARL_W_CPCC1)
      if constexpr (HW && CW == 1) {
        v4u r;
        const uint64_t a = (uint64_t)(uintptr_t)p;
        __asm__ volatile("lsc_load.ugm." PEARL_W_CPCC1 " (M1_NM, 1) %0:d32x32t flat[%1]:a64" : "=rw"(r) : "rw"(a));
        return r;
      }
#endif
#endif
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
  // goal-arc19 (PEARL_W_S2CW=n): work-groups of at least n column windows (the HW fold only) run SLOTS 2 +
  // NOPARTX, the CW 1 instance keeps 3 slots and the atomic. 0: off. PASS (emu too), but on spare28's code it
  // loses: 100.4-100.5 against 101.4-101.6 (card 5, full speed). Off.
#ifndef PEARL_W_S2CW
#define PEARL_W_S2CW 0
#endif
  static constexpr bool kS2 = kSplit && (PEARL_W_S2CW) > 0 && CW >= (PEARL_W_S2CW);
#if defined(PEARL_W_SA)
  // goal-arc21: the k-step with addresses made in the asm (see the chunk loop); the default shape only.
  static constexpr bool kSA = kSplit && AH && KPS == 4 && !kHalf && kBps == 1u;
#else
  static constexpr bool kSA = false;
#endif
#if defined(PEARL_W_SEQP)
  static constexpr bool kSeqP = SP && SEQ > 1 && CW >= (PEARL_W_SEQPCW) && !kHalf && kBps == 1u && KPS == 4;
#else
  static constexpr bool kSeqP = false;
#endif
  // goal-arc22 (PEARL_W_SQX, a bisection mask for HiveOS's 24.35 driver): the stream's parts, each on its own.
  // 1: set s of column window jc takes column block base + jc SEQ + s; 2: the ring's slots count on across sets;
  // 4: the stream itself (one fill, the copy runs on into the next set unclamped, one meeting; needs 1 and 2).
#ifndef PEARL_W_SQX
#define PEARL_W_SQX 7
#endif
  static constexpr bool kSqI = kSeqP && ((PEARL_W_SQX) & 1), kSqS = kSeqP && ((PEARL_W_SQX) & 2),
                        kSqF = kSeqP && ((PEARL_W_SQX) & 4);
  // goal-arc23 (template switch S24, the host's PEARL_SYCL_SEQP=2): the stream in the forms HiveOS's 24.35 driver
  // compiles right. Three changes, each needed: the slot without a division (kSqM, goal-arc22's SQM); the copy's
  // address past a set's last k-step as the clamped address plus a separately selected offset (kSqRo; the plain
  // run-on and a multiplied offset get every region wrong); and an instruction that reads the copy's hold after
  // each set's chunk loop (kSqWt), without which 24.35 lets the set's last copy load land after its register is
  // loaded again. On 24.35 (card 5, host): 94.5-94.7 TH/s against 93.6 with the stream off. On the newer driver it
  // loses (99.9 against 101.6 off, 102.9 for S24 false): the host picks it only on a 24.x OpenCL driver.
#if defined(PEARL_W_SQM)
  static constexpr bool kSqM = true;
#else
  static constexpr bool kSqM = S24;
#endif
#if defined(PEARL_W_SQRO)
  static constexpr int kSqRo = (PEARL_W_SQRO);
#else
  static constexpr int kSqRo = S24 ? 1 : 0;
#endif
#if defined(PEARL_W_SQWT)
  static constexpr int kSqWt = (PEARL_W_SQWT + 0) > 0 ? (PEARL_W_SQWT + 0) : 1;
#else
  static constexpr int kSqWt = S24 ? 1 : 0;
#endif
#if defined(PEARL_W_SQB)
  // goal-arc22 (PEARL_W_SQB): the stream with no barrier under a condition on the set: set 0's fill is a prologue
  // call of run_set, then one arrive, the set loop (the chunks' own meetings only), then the wait and the meeting.
  static constexpr bool kSqB = kSqF;
#else
  static constexpr bool kSqB = false;
#endif
#if defined(PEARL_W_SLOTS)
  // goal-arc8: 2 slots, with the group's arrive after the slot's last local reads (else 3, arrive before them).
  static constexpr uint32_t kSlots = PEARL_W_SLOTS;
#else
  static constexpr uint32_t kSlots = kS2 ? 2u : 3u;
#endif
#if !defined(PEARL_W_NOPARTX)
  // goal-arc8: a window's two sub-groups XOR their words into one place (half the local memory). With the
  // readout below as a tree, 90.3 to 90.6 against 89.9 to 90.2 on card 5 (three rounds, 188 W).
  static constexpr bool kPartX = !kS2;
#else
  static constexpr bool kPartX = false;
#endif
#if defined(PEARL_W_PARTMIN)
  // goal-arc18 timing (wrong results; with NOPXA): a part of 64 words, every index masked into it, so the
  // work-group needs only its ring (two slots: 32 KB, two work-groups a core). With SLOTS 2 + NOPXA, SEQ 1, card
  // 5: 98.9-99.2, and 99.0-99.2 padded to one work-group a core; without the hash 102.4-102.8 either way, also
  // with the ring cut to 28 KB (PEARL_W_RINGCUT). A second work-group on the core neither hides the hash nor
  // anything else.
  static constexpr uint32_t kPartWords = 64u;
#else
  static constexpr uint32_t kPartWords = kNsg * (kPartX ? 4u : 8u) * 16u;  // words of `part` a set
#endif
#if defined(PEARL_W_PART1)
  static constexpr bool kPart1 = true;
#else
  static constexpr bool kPart1 = false;
#endif
  // goal-arc18 (PEARL_W_PXT): word j of region q of window w at (j * windows + w) * 8 + q, so a sub-group's 8
  // lanes touch 8 neighbouring words (8 banks) in the readout's atomic; the old place, (w * 8 + q) * 16 + j,
  // put all 8 lanes 64 bytes apart, in one bank. PXT=2: (w * 16 + j) * 8 + q. Both PASS but lose: 95.4 / 96.1
  // against 100.3 (card 5): IGC rebuilds the chunk loop around the new addresses. Off.
  static uint32_t pix(uint32_t w, uint32_t q, uint32_t j) {
#if defined(PEARL_W_PARTMIN)
    return ((w * 8u + q) * 16u + j) & 15u;
#elif defined(PEARL_W_PXT) && PEARL_W_PXT == 2
    return (w * 16u + j) * 8u + q;
#elif defined(PEARL_W_PXT)
    return (j * (kNsg / 2u) + w) * 8u + q;
#else
    return (w * 8u + q) * 16u + j;
#endif
  }
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> part;  // 2 wpg sub-groups x 8 regions x 16 words
  sycl::local_accessor<uint32_t, 1> ring;  // 3 slots
  // goal-arc9 (PEARL_W_ARGP, with PEARL_W_NOLO32): the operands as kernel arguments of their own, addressed as
  // pointer + 32-bit offset. IGC then makes the loop's loads 32-bit surface accesses: 517 instructions in the loop
  // where there were 598 (movs 84 to 10), but 90.8 TH/s against 91.9 (card 5, three rounds), with 8 dummy DPAS
  // in the loop instead of 4. Kept for the record; off. Only in that build: two unused kernel arguments cost
  // xmx8w 2.7% on card 4 (89.6 against 92.2 TH/s).
#if defined(PEARL_W_ARGP)
  const int8_t *Ap0 = nullptr, *Bp0 = nullptr;
#endif

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
#if defined(PEARL_W_NOREL)
      // goal-arc11 timing (unsafe: stores may not be seen): no release fence. 93.0 with 6 dummy DPAS, level.
      __spirv_ControlBarrierArriveINTEL(2, 2, 0);
#else
      __spirv_ControlBarrierArriveINTEL(2, 2, 0x104);  // work-group; work-group memory, release
#endif
      return;
    }
#endif
    sycl::group_barrier(it.get_group());
  }
#if defined(PEARL_W_A2)
  // goal-arc12 (PEARL_W_A2): the copy stores two slots ahead, so the release fence for a slot's stores can wait
  // until the next slot starts (long after they landed), and the arrive itself carries no fence. PASS, but 93.1
  // against 94.4 with the 8x4 DPAS (card 5): the fence was not what the copy costs. Off.
  static constexpr uint32_t kAhead = 2u;
  void arrive_nf(sycl::nd_item<1> it) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) {
      __spirv_ControlBarrierArriveINTEL(2, 2, 0);
      return;
    }
#endif
    sycl::group_barrier(it.get_group());
  }
  void fence_rel() const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) __spirv_MemoryBarrier(2, 0x104);  // work-group scope, release, work-group memory only
#endif
  }
#else
  static constexpr uint32_t kAhead = 1u;
#endif
  void wait(sycl::nd_item<1>) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && !defined(PEARL_W_PLAINBAR)
    if constexpr (HW) __spirv_ControlBarrierWaitINTEL(2, 2, 0);  // no acquire fence, as the ring fold's VAR 8
#endif
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
#if defined(PEARL_W_TSC) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) {
      uint32_t t0 = 0, t1 = 0, sr = 0;
      __asm__ volatile("mov (M1_NM, 1) %0(0,0)<1> %%tsc(0,0)<0;1,0>\nmov (M1_NM, 1) %1(0,0)<1> %%tsc(0,1)<0;1,0>\n"
                       "mov (M1_NM, 1) %2(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(t0), "=rw"(t1), "=rw"(sr));
      if (lane == 0u && A.tsc) {
        const uint64_t ts = ((uint64_t)t1 << 32) | t0;
        sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(A.tsc[(size_t)it.get_group(0) * 4u])
            .fetch_min(ts);
        if (sg.get_group_linear_id() == 0u) A.tsc[(size_t)it.get_group(0) * 4u + 2u] = sr;
      }
    }
#endif
#if defined(PEARL_W_NC) && defined(PEARL_W_NCFIRST)
    // goal-arc8: the copiers are sub-groups 0 .. NC - 1, so the DPAS pairs keep their places in hardware.
    const uint32_t sgr = (uint32_t)sg.get_group_linear_id();
    const uint32_t sgl = sgr >= (uint32_t)(PEARL_W_NC) ? sgr - (uint32_t)(PEARL_W_NC) : kNsg + sgr;
#else
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();  // 0 .. kNsg - 1
#endif
    const uint32_t sgi = sgl & 1u, widx = sgl >> 1;
#if defined(PEARL_W_STAG) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc20 timing: the second work-group to start on an Xe-core in a launch first waits PEARL_W_STAG
    // timestamp ticks, so two work-groups sharing a core are out of step and hash at different times.
    if constexpr (HW) {
      uint32_t sr = 0;
      __asm__ volatile("mov (M1_NM, 1) %0(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(sr));
      const uint32_t core = (sr >> 9) & 31u;
      if (sgl == 0u && lane == 0u)
        part[0] = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>(A.tick[core]).fetch_add(1u);
      sycl::group_barrier(it.get_group());
      const uint32_t tk = part[0];
      sycl::group_barrier(it.get_group());
      if (tk == 1u) {
        uint32_t x = A.zero + lane;
#pragma unroll 1
        for (uint32_t i = 0; i < (uint32_t)(PEARL_W_STAG); i++) x = x * 1664525u + 1013904223u;
        if (x == 0x12345678u && A.zero) A.tick[31] = x;
      }
    }
#endif
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
      // goal-arc25: in the CW 1 instances (the duty path), a batch wider than kChunkWins column windows (256
      // column offsets) is walked as chunks of that many, one after the other, each the usual band walk. One
      // chunk's B' (8 MB at mainnet) is what the L2 holds; a wider band walk would push it out (column batch 512
      // without chunks: 96.9 against 102.9). The host gives a card under duty 1,024 column offsets a batch, so
      // it launches, waits and rests four times less often (pearl_host_create_checked). The CW 2 instances (full
      // speed) keep the plain walk: their ISA is unchanged.
      if constexpr (CW != 1) {
        window_of((uint32_t)it.get_group(0), rowWins / RW, colWins / CW / (uint32_t)SEQ, A.band / RW > 0u ? A.band / RW : 1u, wrb, wcb);
      } else {
        const uint32_t cwAll = colWins / CW / (uint32_t)SEQ, cwMax = kChunkWins / CW / (uint32_t)SEQ;
        const uint32_t cwc = cwAll > cwMax && cwMax > 0u && cwAll % cwMax == 0u ? cwMax : cwAll;
        const uint32_t gc = (rowWins / RW) * cwc, g = (uint32_t)it.get_group(0), c = g / gc;
        window_of(g - c * gc, rowWins / RW, cwc, A.band / RW > 0u ? A.band / RW : 1u, wrb, wcb);
        wcb += c * cwc;
      }
#endif
#if defined(PEARL_W_CORE) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      // goal-arc11 timing (wrong results): the row block from the Xe-core this work-group runs on (sr0 bits 9 to
      // 13, read by inline vISA), inside the same band, so the four work-groups an Xe-core holds share A'. 94.0
      // against 93.0 (card 5). Done for real by a queue a core (a global atomic a work-group) it lost 3%.
      {
        uint32_t sr = 0;
        __asm__ volatile("mov (M1_NM, 1) %0(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(sr));
        const uint32_t core = (sr >> 9) & 31u;
        wrb = (wrb / (uint32_t)(PEARL_W_CORE)) * (uint32_t)(PEARL_W_CORE) + core % (uint32_t)(PEARL_W_CORE);
      }
#endif
      wr = wrb * RW + widx % RW;
      wc = wcb * CW + widx / RW;
    }
    const uint32_t wcbS = wcb;
#if defined(PEARL_W_PFN) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc23 (PEARL_W_PFN=n): the first n work-groups of each band prefetch the next band's A' into L2 (uncached
    // in L1), before their sets start. The band's first column work-groups otherwise take its A' rows from DRAM.
    // After the last band it wraps to band 0, which the next batch reads first. Loses 0.2-0.3%, and changes nothing
    // in the timeline (PEARL_W_TSC): a load with no destination does nothing on this card, whatever its cache
    // control (so the PFA, PFB, PFG and PFS prefetches of earlier goals never fetched either). PEARL_W_PFNR (real
    // loads, data kept): from the band's last 16 work-groups (PFN0 48) the first column runs normally, but the
    // prefetching work-groups wait for their loads and the batch is 1.5% longer. PFNB: bands ahead; PFN0: first
    // prefetching work-group's index in the band. Off.
    if constexpr (HW) {
      const uint32_t bandG = A.band / RW > 0u ? A.band / RW : 1u, rowG = rowWins / RW;
      const uint32_t per = bandG * (colWins / CW / (uint32_t)SEQ), g = (uint32_t)it.get_group(0);
      const uint32_t b = g / per, idx = g - b * per, nb = (rowG + bandG - 1u) / bandG;
      if (idx - (uint32_t)(PEARL_W_PFN0) < (uint32_t)(PEARL_W_PFN)) {
        const uint32_t b1 = b + (uint32_t)(PEARL_W_PFNB) < nb ? b + (uint32_t)(PEARL_W_PFNB) : b + (uint32_t)(PEARL_W_PFNB) - nb;
        const uint32_t rows1 = rowG - b1 * bandG < bandG ? rowG - b1 * bandG : bandG;  // row-WGs in band b1
        const uint64_t bytes = 2ull * rows1 * RW * A.aStride, share = (bytes / (PEARL_W_PFN) + 255u) & ~255ull;
        const uint64_t base = (uint64_t)(uintptr_t)A.Ap + 2ull * b1 * bandG * RW * A.aStride + (idx - (uint32_t)(PEARL_W_PFN0)) * share;
        const uint64_t end = (uint64_t)(uintptr_t)A.Ap + 2ull * b1 * bandG * RW * A.aStride + bytes;
#if defined(PEARL_W_PFNR)
        // test: real loads, their data kept (XORed into a value stored only if the runtime zero says so)
        v8i pfs = 0;
#pragma unroll 1
        for (uint64_t o = base + (uint64_t)sgl * 256u; o < base + share && o < end; o += (uint64_t)kNsg * 256u) {
          v8i r;
          __asm__ volatile("lsc_load.ugm." PEARL_W_PFNC " (M1_NM, 1) %0:d32x64t flat[%1]:a64" : "=rw"(r) : "rw"(o));
          pfs ^= r;
        }
        if (A.zero && pfs[0] == 0x12345678) A.hits->index[0] = (uint32_t)pfs[1];
#else
#pragma unroll 1
        for (uint64_t o = base + (uint64_t)sgl * 256u; o < base + share && o < end; o += (uint64_t)kNsg * 256u)
          __asm__ volatile("lsc_load.ugm." PEARL_W_PFNC " (M1_NM, 1) %%null:d32x64t flat[%0]:a64" : : "rw"(o));
#endif
      }
    }
#endif
#if defined(PEARL_W_PFS2) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc23 (PEARL_W_PFS2=n): the last 64 / n work-groups of band b, as they start, bring n k-steps each of
    // every 16-row block of band b + 1's A' into L2, in k-step order, by atomic loads with no destination (done in
    // L2; nothing waits for them). They start 1-20 us before band b + 1's first column, which would otherwise read
    // its A' from DRAM a k-step ahead. The first column then runs normally, but every work-group is ~4% slower:
    // L2 atomics are dear for the whole card. PFS2 4 / 2: 101.5 / 101.9 against 102.9 (card 5). Off.
    if constexpr (HW) {
      constexpr uint32_t kK = (PEARL_W_PFS2), kW = 64u / kK;
      const uint32_t bandG = A.band / RW > 0u ? A.band / RW : 1u, rowG = rowWins / RW;
      const uint32_t per = bandG * (colWins / CW / (uint32_t)SEQ), g = (uint32_t)it.get_group(0);
      const uint32_t b = g / per, idx = g - b * per, nb = (rowG + bandG - 1u) / bandG;
      if (b + 1u < nb && per >= kW && idx >= per - kW) {
        const uint32_t b1 = b + 1u, rows1 = rowG - b1 * bandG < bandG ? rowG - b1 * bandG : bandG;
        const uint32_t blocks = 2u * rows1 * RW;
        const uint64_t base = (uint64_t)(uintptr_t)A.Ap + 2ull * b1 * bandG * RW * A.aStride
                              + (uint64_t)(idx - (per - kW)) * kK * 512u + lane * 64u;
        each<2 * (int)kK>([&](int m) {
          const uint32_t blk = 2u * sgl + (uint32_t)(m & 1);
          const uint64_t a = base + (uint64_t)(blk < blocks ? blk : 0u) * A.aStride + (uint64_t)(m >> 1) * 512u;
          __asm__ volatile("lsc_atomic_load.ugm (M1, 8) %%null:d32 flat[%0]:a64 %%null %%null" : : "rw"(a));
        });
      }
    }
#endif
#if defined(PEARL_W_PADR) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc22 (PEARL_W_PADR=n): n registers kept live across the whole kernel (set by inline vISA, so IGC cannot
    // recompute them later), to move where the register allocator puts the loop-carried A'.
    uint32_t padr[(PEARL_W_PADR)];
    each<(PEARL_W_PADR)>([&](int i) { __asm__ volatile("mov (M1, 8) %0(0,0)<1> 0x0:ud" : "=rw"(padr[i])); });
#endif
    cvec nextS[kBps];  // the copy's hold, carried from set to set (kSeqP)
    auto run_set = [&](int set, auto proc) {
    constexpr bool kPro = decltype(proc)::value;  // the prologue call (kSqB): set 0's fill, no chunks
    // goal-arc21 (kSeqP, PEARL_W_SEQP): the sets of a work-group run as one stream of 64 SEQ k-steps. The ring's
    // slots count on across sets; set s of column window jc takes column block base + jc SEQ + s, so each
    // sub-group's copy runs from one set's B' straight into the next set's (no clamp: past the last set it reads
    // the next block or the host's tail, into a slot no one reads); only the first set fills the ring, zeroes
    // `part` (every set's) and arrives before its loop; only the last waits and meets the whole group after it.
    // A' is loaded again at each set's start: carried in registers, IGC took the block read's result for a
    // uniform value across the set loop and set 1 read wrong A'; made per-lane to it, it lost 2-6%. Card 5, full
    // speed: 102.8-103.0 against 101.6-101.8 without it (the same build, PEARL_SYCL_SEQP=0). HiveOS's 24.35
    // driver gets most regions wrong here, in this form and in every other tried (A' a k-step on into the next
    // set, the copy with or without the column interleave or a select, the slot as a rotation), while the other
    // driver passes them all: so the host runs it only when asked (PEARL_SYCL_SEQP=1).
    const bool firstSet = !kSqF || set == 0, lastSet = !kSqF || set == SEQ - 1;
#if defined(PEARL_W_PART1)
    // goal-arc17 timing (wrong results): every set XORs into one part buffer, so SEQ 4 to 16 fit 64 KB. With
    // NOEPI, card 5: SEQ 2 / 4 / 8 / 16 101.8-101.9 / 102.2 / 97.0 / 74.7. Longer work-groups gain only 0.3%.
    // goal-arc21, on the set stream (with the hash): SEQ 2 / 4 / 8 102.5-102.6 / 103.0-103.3 / 98.0 against
    // 102.4-102.6. SEQ 4 for real needs 32 KB of `part`: with a two-slot ring to make room (SLOTS 2) it ran
    // 101.3-101.5 (the two slots cost 1.4%); with the words in global memory (each sub-group storing its chunk
    // word instead of the local atomic, the hash reading them back) 88-91: dropped.
    const uint32_t pOff = 0u * (uint32_t)set;
#else
    const uint32_t pOff = (uint32_t)set * kPartWords;
#endif
    if constexpr (SEQ > 1) {
      wcb = wcbS * (uint32_t)SEQ + (uint32_t)set;
      wc = kSqI ? wcbS * (uint32_t)SEQ * CW + (widx / RW) * (uint32_t)SEQ + (uint32_t)set : wcb * CW + widx / RW;
    }
    const uint32_t jc = widx / RW;  // the window's column block in the ring
#if defined(PEARL_W_HROT)
#if defined(PEARL_W_HROTP)
    const uint32_t hrot = (widx & 1u) ? (uint32_t)(PEARL_W_HROT) : 0u;  // odd windows
#elif defined(PEARL_W_HROTJ)
    // goal-arc21 (PEARL_W_HROTJ): the second column window (jc 1) takes each chunk's k-steps rotated, so it reads
    // the A' its row twin (jc 0) read a k-step or more before, from L1, instead of missing beside it.
    const uint32_t hrot = jc * (uint32_t)(PEARL_W_HROT);
#else
    const uint32_t hrot = widx >= (uint32_t)RW / 2u ? (uint32_t)(PEARL_W_HROT) : 0u;
#endif
    auto kdo = [&](uint32_t k) { return (k & ~3u) + ((k + hrot) & 3u); };
#else
    auto kdo = [](uint32_t k) { return k; };
#endif
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
        Bbi[i] = A.Bp + (size_t)(A.colStart / 4u + wcb * CW + (b >> 1) / 16u) * A.bStride + (size_t)((b >> 1) % 16u) * 128u
                 + (size_t)(b & 1u) * 64u;
      else
#if defined(PEARL_W_BHOT)
        // goal-arc23 timing (wrong results): B' of only PEARL_W_BHOT column blocks, so it stays in L2. BHOT 8 / 32
        // 102.6-102.7 / 102.4-102.6 against 102.8-103.0: B' already stays in L2 for the batch (A' does not: AHOT 32
        // 104.4-104.8, +1.7%, with a higher clock at the same watts).
        Bbi[i] = A.Bp + (size_t)((A.colStart / 4u + (kSqI ? wcbS * (uint32_t)SEQ * CW + (b / 16u) * (uint32_t)SEQ + (uint32_t)set
                                                           : wcb * CW + b / 16u)) % (uint32_t)(PEARL_W_BHOT)) * A.bStride
                 + (size_t)(b % 16u) * 128u;
#else
        Bbi[i] = A.Bp + (size_t)(A.colStart / 4u + (kSqI ? wcbS * (uint32_t)SEQ * CW + (b / 16u) * (uint32_t)SEQ + (uint32_t)set
                                                          : wcb * CW + b / 16u)) * A.bStride
                 + (size_t)(b % 16u) * 128u;
#endif
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
    auto at = [&](uint32_t k) {
      // goal-arc21: with the set stream the slot counts on across sets (k + 64 set). Written as a rotation added
      // after the % 3 (which, with the rest of the stream's code off, passed on HiveOS's 24.35 driver where this
      // form did not) the stream gained 0.1% instead of 1.1% (card 5).
      if constexpr (!kSqS) return ((k / (uint32_t)KPS) % kSlots) * kSlotWords + (k % (uint32_t)KPS) * kStep;
      // goal-arc22 (PEARL_W_SQM, and S24): the same slot without a 32-bit division. (k + 64 set) / 4 = k / 4 + 16
      // set, and 16 = 1 (mod 3), so the slot is (k / 4 + set) % 3, of a number below 2^15: x / 3 = (x * 0xAAAB) >> 17
      // there, one 32 by 16-bit multiply. The division by 3 IGC writes for the % (a mul and a mach through the
      // accumulator) is what HiveOS's 24.35 driver gets wrong: with it every region fails. On the newer driver it
      // costs 0.35% (goal-arc23), so only S24 takes it.
      if constexpr (kSqM) {
        static_assert(kSlots == 3u && KPS == 4, "kSqM: three slots of four k-steps");
        const uint32_t x = k / (uint32_t)KPS + (uint32_t)set, q = (x * 0xAAABu) >> 17;
        return (x - 3u * q) * kSlotWords + (k % (uint32_t)KPS) * kStep;
      } else {
        return (((k + 64u * (uint32_t)set) / (uint32_t)KPS) % kSlots) * kSlotWords + (k % (uint32_t)KPS) * kStep;
      }
    };
    auto mine = [&](int i) { return kHalf ? R + sgl * 16u : R + (sgl + (uint32_t)i * kNsg) * 32u; };
#if defined(PEARL_W_NOPRO)
    // goal-arc21 timing (wrong results): no fill of the first slot of set 0 (only later sets fill theirs), to price
    // the start-of-set bubble: card 5, 102.1-102.3 against 101.6-101.7.
    if (set > 0)
#endif
#if defined(PEARL_W_SQFILL)
    // goal-arc22 test (PEARL_W_SQFILL): every set fills its first slot (the same bytes the copy ran on into)
    if (kSqB ? kPro : true)
#else
    if (kSqB ? kPro : firstSet)
#endif
    each<KPS>([&](int k) {
      each<kBps>([&](int i) {
        cp_write(mine(i) + at((uint32_t)k), cp_read(Bbi[i] + (size_t)k * 2048u, lane), lane);
      });
    });
#if defined(PEARL_W_NOCOPY) && defined(PEARL_W_FILLALL)
    // goal-arc12 timing: with no copy, all slots hold real B' (k-steps KPS .. 3 KPS - 1), so the DPAS see real
    // operand bits (their power depends on them) and only the copy's own cost is missing. On card 5 with the 8x4
    // DPAS and A' ahead: 100.5 against 96.4 TH/s. With the ring left as it was (zeros), 101 at a lower 177 W: the
    // operands' bits alone move the rate 4%, so time copy-free builds with this.
    each<(int)kSlots * KPS - KPS>([&](int k0) {
      const uint32_t k = (uint32_t)k0 + (uint32_t)KPS;
      each<kBps>([&](int i) { cp_write(mine(i) + at(k), cp_read(Bbi[i] + (size_t)k * 2048u, lane), lane); });
    });
#endif
    cvec nextL[kBps];
#if defined(PEARL_W_SEQPRL)
    // goal-arc22 (PEARL_W_SEQPRL): the copy's hold is not carried from set to set either; each set loads it again
    // (the same address the last set's last k-step read, an L1 hit), so no load result crosses the set loop.
    cvec(&next)[kBps] = nextL;
    (void)nextS;
#else
    cvec(&next)[kBps] = (kSqF && !kSqB) ? nextS : nextL;
#endif
#if defined(PEARL_W_A2)
    // the second slot too (k-steps KPS .. 2 KPS - 1), then the copy runs two slots ahead
    each<KPS>([&](int k) {
      each<kBps>([&](int i) {
        cp_write(mine(i) + at((uint32_t)(k + KPS)), cp_read(Bbi[i] + (size_t)(k + KPS) * 2048u, lane), lane);
      });
    });
#endif
#if defined(PEARL_W_SEQPRL)
    each<kBps>([&](int i) { next[i] = cp_read(Bbi[i] + (size_t)(kAhead * KPS) * 2048u, lane); });
#else
    if (kSqB || firstSet) each<kBps>([&](int i) { next[i] = cp_read(Bbi[i] + (size_t)(kAhead * KPS) * 2048u, lane); });
#endif
#if defined(PEARL_W_CPH)
    // goal-arc12 timing (wrong results): the copy's global load as one 256-byte read every other k-step, its two
    // halves stored in turn: half the load messages, the same bytes. 96.6 against 96.3 (card 5): message count
    // is not what the copy costs.
    static_assert(kBps == 1 && !kHalf, "PEARL_W_CPH: one 128-byte copy block a sub-group");
    v8i n8 = ring_read_global8b<HW>(Bbi[0] + (size_t)(kAhead * KPS) * 2048u, lane);
#endif
#if defined(PEARL_W_CPD2)
    // goal-arc12: the copy's global load two k-steps before its store (next2 is the one after next). PASS, but
    // 95.4 against 96.3 (card 5): the load's latency is not what the copy costs. Off.
    cvec next2[kBps];
    each<kBps>([&](int i) { next2[i] = cp_read(Bbi[i] + (size_t)(kAhead * KPS + 1u) * 2048u, lane); });
#endif
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
#if defined(PEARL_W_AHOT)
    // goal-arc14 timing (wrong results): AHOT on the 32-bit address path too (goal-arc12's AHOT row read Ab, which
    // this path never uses, so it timed the shipped code).
    const uint32_t aLo = (uint32_t)(uintptr_t)A.Ap + 2u * (wr % (uint32_t)(PEARL_W_AHOT)) * A.aStride + sgi * 128u;
#else
    const uint32_t aLo = (uint32_t)(uintptr_t)A.Ap + 2u * wr * A.aStride + sgi * 128u;
#endif
    const uint32_t bHi = (uint32_t)((uint64_t)(uintptr_t)A.Bp >> 32);
#if defined(PEARL_W_PFG)
    // goal-arc22 (PEARL_W_PFG=d): an L1 prefetch of A' that touches each 64-byte line once: an 8-lane gather of one
    // dword a lane, lane l on line l & 3 of 16-row block l >> 2 of this sub-group's A' (its 512 bytes of a k-step).
    // At even k-steps the twin column windows prefetch k-steps ks + d (jc 0) and ks + d + 1 (jc 1), so each
    // sub-group sends half a message a k-step. Past the set's end it wraps to the set's first k-steps (the next
    // set reads the same rows).
    const uint32_t pfgLo = aLo + sgi * 128u + (lane >> 2) * A.aStride + (lane & 3u) * 64u + jc * 512u;
#endif
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
#if defined(PEARL_W_CA)
    // goal-arc13 (PEARL_W_CA): the copy as one 256-byte load and one 256-byte store (inline vISA) every other
    // k-step: sub-group sgl copies pieces 2 (sgl % 8) and 2 (sgl % 8) + 1 of k-step kp + sgl / 8, so the 16
    // sub-groups cover two k-steps. Half the copy's messages and address code, the same bytes. PASS, but its
    // rate follows register-allocation luck: with BA 97.6 or 95.9 depending on how the slot address is
    // written, with AA and BA 93.8-94.8 against 97.9 (card 5). Off.
    static_assert(KPS == 4 && kBps == 1 && !kHalf, "PEARL_W_CA: 8 row windows, a chunk a slot");
    // goal-arc16: at CW 2 sub-group sgl copies column block (sgl / 8) % CW of k-step kp + sgl / (8 CW).
    const uint32_t caK = sgl / (8u * (uint32_t)CW), caC = (sgl / 8u) % (uint32_t)CW;
    const int8_t *Bca = A.Bp + (size_t)(A.colStart / 4u + wcb * CW + caC) * A.bStride + (size_t)(sgl & 7u) * 256u;
    auto ca_load = [&](uint32_t kp) {
      const uint32_t k = kp + caK;
#if defined(PEARL_W_LO32)
      return g_read256<HW>(lo32(bHi, (uint32_t)(uintptr_t)Bca + (k < 64u ? k : 63u) * 2048u), lane);
#else
      return g_read256<HW>(Bca + (size_t)(k < 64u ? k : 63u) * 2048u, lane);
#endif
    };
    v8i ca = ca_load((uint32_t)KPS);
#endif
#if defined(PEARL_W_C16)
    // goal-arc13 (PEARL_W_C16, RW 16): 32 sub-groups copy 128 bytes each at even k-steps only, two k-steps'
    // B' (piece sgl % 16 of k-step kb + sgl / 16): each thread stores what STE's did, with the full copy.
    // goal-arc15: at CW 2 (64 sub-groups) sub-group sgl copies piece sgl % 16 of column block (sgl / 16) % 2 of
    // k-step kb + sgl / 32.
    static_assert(KPS == 4 && RW == 16, "PEARL_W_C16: 16 row windows, a chunk a slot");
    const uint32_t c16Col = (sgl >> 4) % (uint32_t)CW, c16K = (sgl >> 4) / (uint32_t)CW;
    const uint32_t c16Lo = (uint32_t)(uintptr_t)(A.Bp + (size_t)(A.colStart / 4u + wcb * CW + c16Col) * A.bStride) + (sgl & 15u) * 128u;
    auto c16_load = [&](uint32_t kb) {
      const uint32_t k = kb + c16K;
      return ring_read_global<HW>(lo32(bHi, c16Lo + (k < 64u ? k : 63u) * 2048u), lane);
    };
    v4u c16n = c16_load((uint32_t)KPS);
#endif
#if defined(PEARL_W_C2)
    static_assert(KPS == 4 && kBps == 1 && !kHalf, "PEARL_W_C2: one copy block a sub-group, a chunk a slot");
    cvec c2n[2];
    each<2>([&](int j) { c2n[j] = cp_read(lo32(bHi, bLo[0] + (uint32_t)(KPS + j) * 2048u), lane); });
#endif
#if defined(PEARL_W_CB)
    // goal-arc13 (PEARL_W_CB): the copy's store as one 256-byte message every other k-step (the local store
    // is what the copy costs: no store 103.3 at 2,000 MHz, no load 97.0, against 98.5), its load as IGC's
    // 128-byte block read every k-step. Sub-group sgl copies pieces 2 (sgl % 8), + 1 of k-step kp + sgl / 8.
    static_assert(KPS == 4 && kBps == 1 && !kHalf && CW == 1, "PEARL_W_CB: 8 by 1 work-group, a chunk a slot");
    const uint32_t cbLo = (uint32_t)(uintptr_t)(A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride) + (sgl & 7u) * 256u;
    auto cb_load = [&](uint32_t kp, uint32_t half) {
      const uint32_t k = kp + (sgl >> 3);
      return ring_read_global<HW>(lo32(bHi, cbLo + (k < 64u ? k : 63u) * 2048u + half * 128u), lane);
    };
    v4u cbl = cb_load((uint32_t)KPS, 0u), cbh = cb_load((uint32_t)KPS, 1u);
#endif
#if defined(PEARL_W_GA)
    // goal-arc13 (PEARL_W_GA): the copy once a chunk as one 512-byte load and one 512-byte store (inline vISA):
    // a chunk's B' is 8 KB in a row both in memory and in its ring slot, and sub-group sgl copies its 512-byte
    // piece sgl. Stored at k-step PEARL_W_GA of a chunk into the next chunk's slot, loaded a chunk ahead. PASS,
    // but 92.4 (GA 0) and 94.6 (GA 2) against 97.9 with AA and BA (card 5): the clock rises, so threads wait. Off.
    static_assert(KPS == 4 && kBps == 1 && !kHalf && CW == 1, "PEARL_W_GA: 8 by 1 work-group, a chunk a slot");
    const int8_t *Bga = A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride + (size_t)sgl * 512u;
    auto ga_load = [&](uint32_t c) {
#if defined(PEARL_W_LO32)
      return g_read512<HW>(lo32(bHi, (uint32_t)(uintptr_t)Bga + (c < 16u ? c : 15u) * 8192u), lane);
#else
      return g_read512<HW>(Bga + (size_t)(c < 16u ? c : 15u) * 8192u, lane);
#endif
    };
    v16i ga = ga_load(1u);
#endif
#if defined(PEARL_W_ARGP)
    const uint32_t aOff = 2u * wr * A.aStride + sgi * 128u;
    const uint32_t bOff = (A.colStart / 4u + wcb) * A.bStride + (sgl % 16u) * 128u;
#endif
    auto load_ah = [&](uint32_t ks, v4i *ah) {
#if defined(PEARL_W_ARGP)
      each<4>([&](int rb) {
        ah[rb] = (v4i)ring_read_global<HW>(Ap0 + (size_t)(aOff + ks * 512u + (uint32_t)(rb >> 1) * A.aStride + (uint32_t)(rb & 1) * 256u), lane);
      });
      return;
#endif
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
#if defined(PEARL_W_AA) && defined(PEARL_W_LO32)
      // goal-arc13 (PEARL_W_AA): A' as 2 messages of 256 bytes a k-step for 4 of 128. Sub-group sgi reads both
      // halves of 8-row block sgi of each 16-row block, so the split DPAS pairs block 0's half j (sub-group 0)
      // with block 1's half j (sub-group 1): every row of accumulator (rb, cb) is in row class rb & 1, and the
      // readout sorts by rb & 1 instead of by element (see below). Same regions, same words.
      if constexpr (kSplit) {
#if defined(PEARL_W_AKM)
        // goal-arc27 energy timing (wrong results): the k-step part of A''s address ANDed with PEARL_W_AKM (0: every
        // k-step reads k-step 0's A', an L1 hit, and the DPAS see the same A' bits every k-step). AKM 0: 7.6% fewer
        // joules a MAC (card 5), the price of A' from L2 and DRAM; DRAM alone (AHOT 32) is 2.4%.
        ks &= (uint32_t)(PEARL_W_AKM);
#endif
#if defined(PEARL_W_AKJ)
        // goal-arc27 energy timing (wrong results): only column window PEARL_W_AKJ (0 or 1) reads k-step 0's A' every
        // k-step; the other reads its real A'. Tells whether the twin windows' A' reads share L1 or both go to L2.
        // Both level with def (card 5): they already share L1.
        ks &= (jc == (uint32_t)(PEARL_W_AKJ)) ? 0u : 63u;
#endif
        each<2>([&](int bb) {
          const v8i t = g_read256<HW>(lo32(aHi, aLo + sgi * 128u + ks * 512u + (uint32_t)bb * A.aStride), lane);
          ah[2 * bb] = __builtin_shufflevector(t, t, 0, 1, 2, 3);
          ah[2 * bb + 1] = __builtin_shufflevector(t, t, 4, 5, 6, 7);
        });
        return;
      }
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
#if defined(PEARL_W_AV) && defined(PEARL_W_AA) && defined(PEARL_W_LO32)
    // goal-arc22 (PEARL_W_AV): A' a k-step ahead carried as the two 256-byte loads (v8i), split at use, to see
    // whether IGC then keeps the loop-carried A' where k-step 0's DPAS read it (it copies 16 registers a chunk).
    v8i ahv[2];
    auto load_ahv = [&](uint32_t ks) {
      each<2>([&](int bb) { ahv[bb] = g_read256<HW>(lo32(aHi, aLo + sgi * 128u + ks * 512u + (uint32_t)bb * A.aStride), lane); });
    };
    if constexpr (kSplit && AH) load_ahv(kdo(0u));
#endif
#if defined(PEARL_W_TR)
    // goal-arc20 timing (wrong results): the transposed fold's messages. B' is the split operand (each sub-group
    // reads half of its 4 fragments from the ring: 2 local messages of 256 bytes a k-step), A' the full one (4
    // global messages of 256 bytes: the window's 4 fragments, read by both sub-groups of the pair, as 4 column
    // windows sharing A' rows would). PEARL_W_TR=1: A' a k-step ahead (32 registers); 2: A' in the k-step.
    auto load_a8 = [&](uint32_t ks, v8i *dst) {
      const uint32_t base = aLo - sgi * 128u + ks * 512u;
      each<4>([&](int rb) { dst[rb] = g_read256<HW>(lo32(aHi, base + (uint32_t)(rb >> 1) * A.aStride + (uint32_t)(rb & 1) * 256u), lane); });
    };
    auto load_a8r = [&](uint32_t ks, v8i *dst, int r0, int r1) {
      const uint32_t base = aLo - sgi * 128u + ks * 512u;
      each<4>([&](int rb) {
        if (rb >= r0 && rb < r1) dst[rb] = g_read256<HW>(lo32(aHi, base + (uint32_t)(rb >> 1) * A.aStride + (uint32_t)(rb & 1) * 256u), lane);
      });
    };
    (void)load_a8r;
    v8i an8[4];
    if constexpr (kSplit && (PEARL_W_TR) == 1) load_a8(0u, an8);
    if constexpr (kSplit && (PEARL_W_TR) == 3) load_a8r(0u, an8, 0, 2);
#else
    if constexpr (kSplit && AH) load_ah(kdo(0u), ahn);
#endif
#if defined(PEARL_W_AH2)
    // goal-arc16 (PEARL_W_AH2): A' two k-steps ahead (16 more registers).
    v4i ahn2[4];
    if constexpr (kSplit && AH) load_ah(kdo(1u), ahn2);
#endif
#if defined(PEARL_W_PXA) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    static_assert((PEARL_W_PXT + 0) == 2, "PEARL_W_PXA: PXT 2's layout");
    const uint32_t pxa0 = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)(&part[0] + pOff + widx * 128u) + lane * 4u;
#endif
    if constexpr (kPartX) {
#if defined(PEARL_W_PXB)
      // goal-arc20 (PEARL_W_PXB, PXT 2's layout): a window's 128 words lie together, so zero them as two 256-byte
      // block stores instead of 16 stores whose 8 lanes all hit one bank.
#if defined(PEARL_W_SQZS)
      // goal-arc22 test (PEARL_W_SQZS, with SQMEET): each set zeroes its own words at its start, as without the stream
      constexpr bool kZ1 = false;
      if (sgi == 0u) {
#else
      constexpr bool kZ1 = kSqF;
      if (sgi == 0u && (kSqB ? kPro : firstSet)) {
#endif
        uint32_t *P = &part[0];
        each<(kZ1 && !kPart1 ? SEQ : 1)>([&](int z) {
          const uint32_t zo = kZ1 ? (uint32_t)z * kPartWords : pOff;
#if defined(PEARL_W_SQZ)
          // goal-arc22 test (PEARL_W_SQZ): the stream's zeroing as plain stores
          if constexpr (kZ1) {
            each<16>([&](int j) { P[zo + widx * 128u + (uint32_t)j * 8u + lane] = 0u; });
            return;
          }
#endif
          slm_write256<HW>(P + zo + widx * 128u, (v8i)0, lane);
          slm_write256<HW>(P + zo + widx * 128u + 64u, (v8i)0, lane);
        });
      }
#elif defined(PEARL_W_PXT) || defined(PEARL_W_PARTMIN)
      if (sgi == 0u && lane < 8u) each<16>([&](int j) { part[pOff + pix(widx, lane, (uint32_t)j)] = 0u; });
#else
      if (sgi == 0u && lane < 8u) each<16>([&](int j) { part[pOff + (widx * 8u + lane) * 16u + (uint32_t)j] = 0u; });
#endif
    }
    v8i bqn[4];
    (void)bqn;
#if defined(PEARL_W_SMPAH)
    auto smp_fetch = [&](uint32_t k) {
      namespace sx = sycl::ext::oneapi::experimental;
      const sx::unsampled_image_handle hb{A.img};
      const uint32_t c = (A.colStart / 4u + wc) * 64u + sgi * 32u + 24u + lane;
      const int x = (int)((c & 31u) * 128u + 2u * k), y = (int)((c >> 5) & 2047u);
      const sycl::vec<uint32_t, 4> lo = sx::fetch_image<sycl::vec<uint32_t, 4>>(hb, sycl::vec<int, 2>{x, y});
      const sycl::vec<uint32_t, 4> hi = sx::fetch_image<sycl::vec<uint32_t, 4>>(hb, sycl::vec<int, 2>{x + 1, y});
      return v8i{(int)lo[0], (int)lo[1], (int)lo[2], (int)lo[3], (int)hi[0], (int)hi[1], (int)hi[2], (int)hi[3]};
    };
    v8i smpn = smp_fetch(0u);
#endif
    v8i c[2][8];
    each<2>([&](int rb) { each<8>([&](int cb) { c[rb][cb] = 0; }); });
#if defined(PEARL_W_D4)
    // goal-arc12 (PEARL_W_D4): each 8-row block as two 4-row split DPAS. Half h of accumulator (rb, cb) holds
    // rows {2h, 2h + 1} of sub-group 0's half and the same of sub-group 1's: rows 2h, 2h + 1, 2h + 4, 2h + 5.
    v4i c4[2][8][2];
    each<2>([&](int rb) { each<8>([&](int cb) { c4[rb][cb][0] = 0; c4[rb][cb][1] = 0; }); });
#endif
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
#if defined(PEARL_W_NOLD)
    // goal-arc16 timing (wrong results): bit 1, A' held in registers (loaded once, real bits); bit 2, B' the same
    // (built from A' bits). With NOCOPY and NOBAR: the loop's own ceiling a clock, operand traffic taken out.
    v4i ahc[4];
    v8i bqc[4];
    load_ah(1u, ahc);
    each<4>([&](int cb) { bqc[cb] = (v8i)__builtin_shufflevector(ahc[cb], ahc[cb ^ 1], 0, 1, 2, 3, 4, 5, 6, 7) ^ (int)(0x9E3779B9u * (uint32_t)(cb + 1)); });
#if defined(PEARL_W_PHASE) && defined(PEARL_W_D4)
    // goal-arc18 timing (wrong results; NOLD 3 only): the second half of the sub-groups runs a delay (a dependent
    // ALU chain, PEARL_W_PHASE x 16 steps) before the loop, so its chunk ends (readout) fall between the first
    // half's. Meant to test whether a 64-thread work-group loses because its threads run in step: RW 16 x CW 2
    // skeleton without hash 101.3, PHASE 1 / 4 / 8 99.7 / 99.4 / 99.4. It does not help. (Extra DPAS in a branch
    // there instead ran at a third of the speed.)
    if constexpr (kSplit) {
      uint32_t ph = A.zero + lane;
      const uint32_t nph = sgl >= kNsg / 2u ? (uint32_t)(PEARL_W_PHASE) * 16u : 0u;
#pragma unroll 1
      for (uint32_t i = 0; i < nph; i++) ph = ph * 1664525u + 1013904223u;
      ahc[0][0] ^= (int)(ph & A.zero);
    }
#endif
#endif
#if defined(PEARL_W_BREG)
    // goal-arc27 energy timing (wrong results): B' held in registers, real bits (k-step 0's B' from the ring, read
    // once after a meeting), so the k-step has no B' local reads but the DPAS see real B' values (NOLD 2's B' was
    // hashed bytes, which cost more watts than real ones). With AKM 0 and CPHOT 0: the B' local reads are 7.7% of
    // def's joules a MAC (card 5), the largest message item.
    v8i bqr[4];
    sycl::group_barrier(it.get_group());
    each<4>([&](int cb) { bqr[cb] = slm_read256<HW>(R + at(0u) + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
#endif
    if constexpr (kPro) return;
#if defined(PEARL_W_SQMEET)
    // goal-arc22 test (PEARL_W_SQMEET): the stream, but every set arrives first and waits and meets at its end.
    if (!kSqB) arrive(it);
#else
    if (!kSqB && firstSet) arrive(it);
#endif
#if defined(PEARL_W_CHUNROLL)
#pragma unroll PEARL_W_CHUNROLL
#endif
    for (uint32_t ch = 0; ch < 16u; ch++) {
#if defined(PEARL_W_RDI)
      // goal-arc21 (PEARL_W_RDI): the readout XORed during the chunk's last k-step, a B' fragment behind the DPAS.
      v4i rdP[2] = {0, 0};
#endif
#if defined(PEARL_W_SA) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      // goal-arc21 (PEARL_W_SA): the default k-step (copy, A' a k-step ahead as 256-byte messages, B' as 256-byte
      // local reads, 8x4 split DPAS) with every message's address a chunk base plus an immediate, added inside
      // the asm. Past the last k-step the A' and copy loads read beyond the operand (the host leaves a tail
      // after each); the copy stores those bytes into a slot no one reads.
      if constexpr (kSA) {
        // vz (a runtime 0 that differs by lane to the compiler) keeps each base an 8-lane value set once: as a
        // uniform value IGC broadcasts it into a new register at every asm that reads it.
        const uint32_t vz = lane * A.zero;
        const uint32_t ring0 = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)R + vz;
        const uint32_t rbB = ring0 + 4u * ((ch % kSlots) * kSlotWords + jc * 512u + sgi * 256u);
        const uint32_t stB = ring0 + 4u * (((ch + 1u) % kSlots) * kSlotWords + sgl * 32u);
        const uint32_t aL0 = aLo + vz + sgi * 128u + (4u * ch + 1u) * 512u;
        const uint64_t aB0 = ((uint64_t)aHi << 32) | aL0, aB1 = ((uint64_t)aHi << 32) | (aL0 + A.aStride);
        const uint64_t cB = ((uint64_t)bHi << 32) | (bLo[0] + vz + (4u * ch + 5u) * 2048u);
        eachc<4>([&](auto Sc) {
          constexpr int S = decltype(Sc)::value;
          constexpr int kStepB = 4 * (int)kStep;
          if constexpr (S == 0) wait(it);
          const v4u cv = sa_g128<S * 2048, CW>(cB);
          sa_slm_st128<S * kStepB>(stB, next[0]);
          next[0] = cv;
          if constexpr (S == 3) arrive(it);
          v4i ah[4];
          each<4>([&](int rb) { ah[rb] = ahn[rb]; });
          {
            const v8i t0 = sa_g256<S * 512>(aB0), t1 = sa_g256<S * 512>(aB1);
            ahn[0] = __builtin_shufflevector(t0, t0, 0, 1, 2, 3);
            ahn[1] = __builtin_shufflevector(t0, t0, 4, 5, 6, 7);
            ahn[2] = __builtin_shufflevector(t1, t1, 0, 1, 2, 3);
            ahn[3] = __builtin_shufflevector(t1, t1, 4, 5, 6, 7);
          }
          v8i bq[4];
          bq[0] = sa_slm_rd256<S * kStepB>(rbB);
          bq[1] = sa_slm_rd256<S * kStepB + 256>(rbB);
          bq[2] = sa_slm_rd256<S * kStepB + 512>(rbB);
          bq[3] = sa_slm_rd256<S * kStepB + 768>(rbB);
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              each<2>([&](int h) {
                v4i &acc = c4[rb >> 1][(rb & 1) * 4 + cb][h];
                acc = split_dpas4<HW>(v2i{ah[rb][2 * h], ah[rb][2 * h + 1]}, bq[cb], acc);
              });
            });
          });
        });
      } else
#endif
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
#elif defined(PEARL_W_HROT)
        // goal-arc14: the second half of the windows (PEARL_W_HROTP: the odd ones) takes the chunk's k-steps
        // rotated by PEARL_W_HROT, so RW 16's one work-group a core is not all in step. PASS, but HROT 2 93.7,
        // HROT 1 97.5, odd windows 97.2 against 99.0; with no copy 97.6 against 101.7 (card 5). Windows reading
        // the same B' lines at the same time is worth keeping. Off.
        const uint32_t kd = ch * 4u + (s + hrot) % 4u;
#else
        const uint32_t kd = ks;
#endif
        const uint32_t slot = at(kd);
#if defined(PEARL_W_AEARLY)
        // goal-arc19 (PEARL_W_AEARLY): A''s next load before the group's wait (it does not depend on it). PASS,
        // 95.3-95.7 against 101.3-101.5 (card 5, full speed); with the copy's load there too 94.8-95.2. Off.
        v4i ahe[4];
        each<4>([&](int rb) { ahe[rb] = ahn[rb]; });
        load_ah(ks + 1u < 64u ? kdo(ks + 1u) : 63u, ahn);
#endif
#if (PEARL_W_CPLD + 0) < 0
        // goal-arc19 (PEARL_W_CPLD=-1): the copy's load before the group's wait.
        const uint32_t knE = ks + kAhead * (uint32_t)KPS + 1u;
        cvec cpE[kBps];
        each<kBps>([&](int i) { cpE[i] = cp_read(lo32(bHi, bLo[i] + (knE < 64u ? knE : 63u) * 2048u), lane); });
#endif
        // The group meets once a slot: every store into this slot has landed, and every read of the slot the
        // copy writes next is done. The copy stores k-step ks + KPS (loaded a k-step ago) and loads the one after.
#if !defined(PEARL_W_NOBAR)
#if defined(PEARL_W_BARX)
        // goal-arc17 timing: one more meeting in the middle of a chunk (arrive after k-step 1's copy, wait at 2).
        if (s == 2u) wait(it);
#endif
        if (s % (uint32_t)KPS == 0u && (ch % (uint32_t)(PEARL_W_BARN)) == 0u) {
          wait(it);
#if defined(PEARL_W_A2)
          fence_rel();  // the stores of the slot before last: they landed long ago
#endif
        }
#endif
#if defined(PEARL_W_GCOPY)
        auto cp_at = [&](int) {};
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
        auto cp_at = [&](int) {};
        v4u csv[kBps];
        each<kBps>([&](int i) { csv[i] = ring_read_global<HW>(Bbi[i] + (size_t)(ks + (uint32_t)KPS < 64u ? ks + (uint32_t)KPS : 63u) * 2048u, lane); });
#elif !defined(PEARL_W_NOCOPY) && (!defined(PEARL_W_NC) || defined(PEARL_W_NCIDLE))
#if defined(PEARL_W_CPLD) || defined(PEARL_W_CPST)
        // goal-arc19 (timing; the HW path only, so not for the emu fold): where in the k-step the copy's global
        // load (PEARL_W_CPLD) and local store (PEARL_W_CPST) go: 0 the k-step's start (as shipped), 1 after A''s
        // next load, 2 after the B' reads, 3 between the two halves of the DPAS, 4 after the DPAS; load -1 before
        // the group's wait. With the store moved, the group's arrive moves with it. All PASS, all lose (card 5,
        // full speed, def 101.1-101.5): load 1 / 2 / 3 / 4 / -1 100.3-100.6 / 98.0-98.1 / 98.0-98.1 / 98.7-98.9 /
        // 100.8-100.9; store 2 / 3 / 4 95.6-95.8 / 95.8-96.0 / 91.7-91.8; load 1 + store 2 95.3-95.5. The finalizer
        // still moves the messages across k-steps; the start of the k-step is the best place for both.
#ifndef PEARL_W_CPLD
#define PEARL_W_CPLD 0
#endif
#ifndef PEARL_W_CPST
#define PEARL_W_CPST 0
#endif
        cvec cpn[kBps];
        auto cp_at = [&](int pos) {
#if (PEARL_W_CPLD) < 0
          if (pos == 0) each<kBps>([&](int i) { cpn[i] = cpE[i]; });
#endif
          if (pos == (PEARL_W_CPLD)) {
            const uint32_t kn = ks + kAhead * (uint32_t)KPS + 1u;
            each<kBps>([&](int i) { cpn[i] = cp_read(lo32(bHi, bLo[i] + (kn < 64u ? kn : 63u) * 2048u), lane); });
          }
          if (pos == (PEARL_W_CPST)) {
            each<kBps>([&](int i) { cp_write(mine(i) + at(ks + kAhead * (uint32_t)KPS), next[i], lane); });
#if (PEARL_W_CPST) > 0
            if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
          }
          if (pos == ((PEARL_W_CPLD) > (PEARL_W_CPST) ? (PEARL_W_CPLD) : (PEARL_W_CPST))) each<kBps>([&](int i) { next[i] = cpn[i]; });
        };
#else
        auto cp_at = [&](int) {};
#endif
        auto copy_step = [&]() {
#if defined(PEARL_W_C16) && defined(PEARL_W_C16S)
          // goal-arc14 (PEARL_W_C16S): sub-groups 16-31 copy one k-step later (odd k-steps), so 16 threads store
          // each k-step instead of 32 at even ones.
          if (s % 2u == (sgl >> 4)) {
            const uint32_t kb = ks - (sgl >> 4) + (uint32_t)KPS;
            ring_write_local<HW>(R + at(kb) + (sgl >> 4) * kStep + (sgl & 15u) * 32u, c16n, lane);
            c16n = c16_load(kb + 2u);
          }
          return;
#endif
#if defined(PEARL_W_C16)
          if (s % 2u == 0u) {
            const uint32_t kb = ks + (uint32_t)KPS;
            ring_write_local<HW>(R + at(kb) + c16K * kStep + c16Col * 512u + (sgl & 15u) * 32u, c16n, lane);
            c16n = c16_load(kb + 2u);
          }
          return;
#endif
#if defined(PEARL_W_C2)
          // goal-arc13 (PEARL_W_C2): the copy at even k-steps only, two k-steps' pieces at a time (two 128-byte
          // loads and two 128-byte stores), none at odd ones. PASS, but 94.3-94.4 against 98.4-98.5 (card 5).
          // goal-arc13 prices of the copy on spare23's code (card 5, 98.5 for the real one): no copy (NOCOPY +
          // FILLALL) 104.8; no store (NOPUT + FILLALL) 103.3 at the 2,000 MHz cap, 172 W; no load (NOFETCH) 97.0;
          // half the copy, 128-byte messages at even k-steps (STE, timing) 105.3; stores of 64 bytes (STH) 94.7 at
          // 2,000 MHz; the store by inline vISA (STA 1 / 2) 98.9-99.2; 256-byte stores every other k-step with
          // 128-byte loads (CB) 94.5; CA at k-steps 1, 3 (CAP 1) 95.5. So the local store's bytes cost the energy,
          // and moving them in fewer or bigger messages stalls: only fewer bytes a MAC can help.
          if (s % 2u == 0u) {
            const uint32_t kb = ks + (uint32_t)KPS;  // kb, kb + 1: one slot
            cp_write(mine(0) + at(kb), c2n[0], lane);
            cp_write(mine(0) + at(kb) + kStep, c2n[1], lane);
            each<2>([&](int j) {
              const uint32_t k = kb + 2u + (uint32_t)j;
              c2n[j] = cp_read(lo32(bHi, bLo[0] + (k < 64u ? k : 63u) * 2048u), lane);
            });
          }
          return;
#endif
#if defined(PEARL_W_CB)
          if (s % 2u == 0u) {
            const v8i w = (v8i)__builtin_shufflevector(cbl, cbh, 0, 1, 2, 3, 4, 5, 6, 7);
            slm_write256<HW>(R + at(ks + (uint32_t)KPS) + (sgl >> 3) * kStep + (sgl & 7u) * 64u, w, lane);
            cbl = cb_load(ks + (uint32_t)KPS + 2u, 0u);
          } else {
            cbh = cb_load(ks + (uint32_t)KPS + 1u, 1u);
          }
          return;
#endif
#if defined(PEARL_W_GA)
          if (s == (uint32_t)(PEARL_W_GA)) {
            slm_write512<HW>(R + ((ch + 1u) % kSlots) * kSlotWords + sgl * 128u, ga, lane);
            ga = ga_load(ch + 2u);
          }
          return;
#endif
#if defined(PEARL_W_CA)
#ifndef PEARL_W_CAP
#define PEARL_W_CAP 0  // the k-steps of a chunk (0, 2 or 1, 3) the copy runs at
#endif
          if (s % 2u == (uint32_t)(PEARL_W_CAP)) {
            // k-steps kb and kb + 1 lie in the same slot
            const uint32_t kb = ks + (uint32_t)KPS - (uint32_t)(PEARL_W_CAP);
            slm_write256<HW>(R + at(kb) + caK * kStep + caC * 512u + (sgl & 7u) * 64u, ca, lane);
            ca = ca_load(kb + 2u);
          }
          return;
#endif
#if defined(PEARL_W_CPH)
          {
            const v8i h = n8;
            const v4u half = (s & 1u) ? v4u{(uint32_t)h[4], (uint32_t)h[5], (uint32_t)h[6], (uint32_t)h[7]}
                                      : v4u{(uint32_t)h[0], (uint32_t)h[1], (uint32_t)h[2], (uint32_t)h[3]};
            cp_write(mine(0) + at(ks + kAhead * (uint32_t)KPS), half, lane);
            if (s & 1u) {
              const uint32_t kn = ks + kAhead * (uint32_t)KPS + 1u;
              n8 = ring_read_global8b<HW>(Bbi[0] + (size_t)(kn < 64u ? kn : 63u) * 2048u, lane);
            }
            return;
          }
#endif
#if defined(PEARL_W_CPD2)
          const uint32_t kn = ks + kAhead * (uint32_t)KPS + 2u;
#else
          const uint32_t kn = ks + kAhead * (uint32_t)KPS + 1u;
#endif
#if defined(PEARL_W_CPLD) || defined(PEARL_W_CPST)
          // goal-arc19 (see cp_at): the parts that stay at the k-step's start.
          (void)kn;
          cp_at(0);
          return;
#endif
          each<kBps>([&](int i) {
#if defined(PEARL_W_NOFETCH)
            // goal-arc8 timing (wrong results): the copy stores without loading.
            const v4u v = next[i] ^ (ks * 0x9E3779B9u);
#elif defined(PEARL_W_NOFETCHR)
            // goal-arc27 energy timing (wrong results): the copy stores without loading, the same real B' bytes
            // every k-step (NOFETCH's XOR made the DPAS see random bytes, which costs more watts than real B').
            const v4u v = next[i];
#else
#if defined(PEARL_W_ARGP)
            static_assert(kBps == 1, "PEARL_W_ARGP: one copy block a sub-group");
            const cvec v = cp_read(Bp0 + (size_t)(bOff + (kn < 64u ? kn : 63u) * 2048u), lane);
#elif defined(PEARL_W_PTRV)
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
#if defined(PEARL_W_CPHOT)
            // goal-arc12 timing (wrong results): the same instructions, but the k-step part of the address is ANDed
            // with PEARL_W_CPHOT (0: always the first k-step's block, an L1 hit). 0: 97.6, 3: 96.3, against 96.3;
            // no load at all (NOFETCH) 99.1 at the same clock. So of the copy load's ~2.8%, misses are ~1%, the
            // rest its instructions and scoreboard tokens.
            const cvec v = cp_read(lo32(bHi, bLo[i] + ((kn < 64u ? kn : 63u) & (uint32_t)(PEARL_W_CPHOT)) * 2048u), lane);
#else
#if defined(PEARL_W_SQUC)
            // goal-arc22 test (PEARL_W_SQUC, without the stream): what the copy loads past the set's last k-step (it
            // goes into a slot no one reads): 1 unclamped, 2 clamped at 71 (never fires), 3 wrapped to k-steps 0-4.
            // All three pass on the new driver; on HiveOS's 24.35 all three get every region wrong, while the
            // clamp at 63 (k-step 63's bytes again) passes. That is the stream's other 24.35 failure.
            const uint32_t knc = (PEARL_W_SQUC) == 3 ? (kn < 64u ? kn : kn - 64u) : (PEARL_W_SQUC) == 2 ? (kn < 72u ? kn : 71u) : kn;
            const cvec v = cp_read(lo32(bHi, bLo[i] + knc * 2048u), lane);
#else
#if defined(PEARL_W_SQCL)
            // goal-arc23 diagnostic (PEARL_W_SQCL): the stream with the copy clamped at k-step 63 (use with SQFILL)
            const cvec v = cp_read(lo32(bHi, bLo[i] + (kn < 64u ? kn : 63u) * 2048u), lane);
#else
#if defined(PEARL_W_SQRON)
            constexpr bool kSqRoN = true;  // goal-arc23 diagnostic: the run-on without the stream too
#else
            constexpr bool kSqRoN = false;
#endif
            cvec v;
            if constexpr (kSqRo > 0 && (kSqF || kSqRoN)) {
              // goal-arc23 (kSqRo, S24): the run-on past k-step 63 as the clamped address plus a separate offset (the
              // same addresses as the plain run-on). 1: the offset by a select, which 24.35 compiles right; 2: by a
              // multiply, which it does not (nor the plain form).
              const uint32_t knC = kn < 64u ? kn : 63u;
              const uint32_t knX = kSqRo == 2 ? kn - knC : (kn > 63u ? kn - 63u : 0u);
              v = cp_read(lo32(bHi, bLo[i] + knC * 2048u + knX * 2048u), lane);
            } else {
              v = cp_read(lo32(bHi, bLo[i] + (kSqF ? kn : (kn < 64u ? kn : 63u)) * 2048u), lane);
            }
#endif
#endif
#endif
#endif
#else
            const cvec v = cp_read(Bbi[i] + (size_t)(kn < 64u ? kn : 63u) * 2048u, lane);
#endif
#endif
#if defined(PEARL_W_NOPUT)
            // goal-arc8 timing (wrong results): the copy loads without storing. goal-arc27: FILLALL needs NOCOPY, so
            // here slots 1 and 2 keep stale bytes (zeros) and the DPAS see them: its low watts are not the store's.
            sink ^= v[0] ^ v[3];
#else
#if defined(PEARL_W_STH) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
            // goal-arc13 timing (wrong results): the store writes only the first 64 bytes of each piece.
            __spirv_SubgroupBlockWriteINTEL((__attribute__((opencl_local)) uint32_t *)(mine(i) + at(ks + kAhead * (uint32_t)KPS)),
                                            v2u{next[i][0], next[i][1]});
#elif defined(PEARL_W_STA)
            slm_write128<HW>(mine(i) + at(ks + kAhead * (uint32_t)KPS), next[i], lane);
#else
#if defined(PEARL_W_SQNS)
            // goal-arc23 diagnostic (PEARL_W_SQNS): no store of what the copy loaded past k-step 63. On 24.35 the
            // run-on loads alone (never stored) still get 14,040 of 16,384 regions wrong, mostly as other regions'
            // right words; clamped loads with no store pass. See PEARL_W_SQWT.
            if (ks + kAhead * (uint32_t)KPS < 64u)
#endif
            cp_write(mine(i) + at(ks + kAhead * (uint32_t)KPS), next[i], lane);
#endif
#endif
#if defined(PEARL_W_CPD2)
            next[i] = next2[i];
            next2[i] = v;
#else
            next[i] = v;
#endif
          });
        };
#if defined(PEARL_W_STE)
        // goal-arc13 timing (wrong results; use with FILLALL): the copy only at even k-steps, half its stores and
        // loads in the same 128-byte messages, as a work-group of half the columns would need.
        if (s % 2u == 0u) copy_step();
#elif !defined(PEARL_W_CLATE)
#if defined(PEARL_W_BFIRST)
        // goal-arc16 (PEARL_W_BFIRST): the k-step's B' reads go out before the copy and A''s next load.
        v8i bqf[4];
        each<4>([&](int cb) { bqf[cb] = slm_read256<HW>(R + slot + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
#endif
        copy_step();
#endif
#else
        auto cp_at = [&](int) {};
#endif
#if defined(PEARL_W_PFS) && defined(PEARL_W_LO32)
        // goal-arc21 (PEARL_W_PFS=d): PFA without its loop (CW 2 only: 32 sub-groups, 32 pieces of 256 bytes, one
        // each): sub-group sgl prefetches piece sgl of the work-group's A' of k-step ks + d into L1.
        if constexpr (kNsg == (uint32_t)RW * 4u) {
          const uint32_t kf = (ks + (uint32_t)(PEARL_W_PFS)) & 63u;
          const uint32_t wr0 = wr - widx % (uint32_t)RW;
          const uint32_t lo = (uint32_t)(uintptr_t)A.Ap + (2u * (wr0 + sgl / 4u) + (sgl / 2u) % 2u) * A.aStride + kf * 512u + (sgl % 2u) * 256u;
          g_pf256<HW>(lo32(aHi, lo));
        }
#endif
#if defined(PEARL_W_PFS) && defined(PEARL_W_LO32)
        // goal-arc21 (PEARL_W_PFS=d): PFA without its loop (CW 2 only: 32 sub-groups, 32 pieces of 256 bytes, one
        // each): sub-group sgl prefetches piece sgl of the work-group's A' of k-step ks + d into L1. Card 5, on the
        // set stream: PFS 2 99.1-99.2, PFS 3 100.4-100.5 against 102.2-102.4. Off.
        if constexpr (kNsg == (uint32_t)RW * 4u) {
          const uint32_t kf = (ks + (uint32_t)(PEARL_W_PFS)) & 63u;
          const uint32_t wr0 = wr - widx % (uint32_t)RW;
          const uint32_t lo = (uint32_t)(uintptr_t)A.Ap + (2u * (wr0 + sgl / 4u) + (sgl / 2u) % 2u) * A.aStride + kf * 512u + (sgl % 2u) * 256u;
          g_pf256<HW>(lo32(aHi, lo));
        }
#endif
#if defined(PEARL_W_PFA) && defined(PEARL_W_LO32)
        // goal-arc17 (PEARL_W_PFA=d): the work-group's A' of k-step ks + d prefetched into L1, 256 bytes a message:
        // RW row windows x 2 blocks x 2 halves, spread over the sub-groups (CW 2: one each; CW 1: two each).
        // Card 5: PFA 2 96.5 against 100.4 (full speed), 74.1 against 76.9 (duty 75); PFA 4 FAILS the check
        // (thousands of regions wrong, cause not found). Off.
        {
          const uint32_t kf = ks + (uint32_t)(PEARL_W_PFA) < 64u ? ks + (uint32_t)(PEARL_W_PFA) : 63u;
          const uint32_t wr0 = wr - widx % (uint32_t)RW;
          for (uint32_t u = sgl; u < (uint32_t)RW * 4u; u += kNsg) {
            const uint32_t lo = (uint32_t)(uintptr_t)A.Ap + (2u * (wr0 + u / 4u) + (u / 2u) % 2u) * A.aStride + kf * 512u + (u % 2u) * 256u;
            g_pf256<HW>(lo32(aHi, lo));
          }
        }
#endif
#if defined(PEARL_W_PFB) && defined(PEARL_W_LO32)
        // goal-arc17 (PEARL_W_PFB=d): the copy's own block of k-step ks + KPS + 1 + d prefetched. PFB 4: 99.7
        // against 100.4 (full speed); PFB 4 / 8 76.8-76.9 against 76.9-77.0 (duty 75). Off.
        {
          const uint32_t kf = ks + (uint32_t)KPS + 1u + (uint32_t)(PEARL_W_PFB);
          each<kBps>([&](int i) { g_pf128<HW>(lo32(bHi, bLo[i] + (kf < 64u ? kf : 63u) * 2048u)); });
        }
#endif
#if !defined(PEARL_W_NOBAR) && !defined(PEARL_W_CSAME) && !defined(PEARL_W_CLATE) && !defined(PEARL_W_SLOTS) && !(defined(PEARL_W_CPST) && (PEARL_W_CPST + 0) > 0)
#if defined(PEARL_W_A2)
        if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive_nf(it);
#else
        if (!kS2 && s % (uint32_t)KPS == (uint32_t)KPS - 1u && (ch % (uint32_t)(PEARL_W_BARN)) == (uint32_t)(PEARL_W_BARN) - 1u) arrive(it);
#if defined(PEARL_W_BARX)
        if (s == 1u) arrive(it);
#endif
#endif
#endif
        auto rd1 = [&](uint32_t off, int cb) {
          return slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane);
        };
        (void)rd1;
        // B' fragment cb of this sub-group's 32 columns, from the ring.
        v8i bq[4];
        auto read_b_at = [&](uint32_t off, v8i *dst) {
#if defined(PEARL_W_GB)
          // goal-arc8: B' straight from memory (no ring), k-step kd; with PEARL_W_ROTW the windows reading a line
          // come at different k-steps, so all but the first may find it in L1. Use with NOCOPY and NOBAR.
          (void)off;
#if defined(PEARL_W_GBA) && defined(PEARL_W_LO32)
          // goal-arc13: the same as four 256-byte messages (inline vISA), no ring, no copy, no barrier.
          {
            const uint32_t gl = (uint32_t)(uintptr_t)(A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride) + kd * 2048u + sgi * 1024u;
            each<4>([&](int cb) { dst[cb] = g_read256<HW>(lo32(bHi, gl + (uint32_t)cb * 256u), lane); });
            return;
          }
#endif
          each<4>([&](int cb) {
            const int8_t *g = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride + (size_t)kd * 2048u
                              + (size_t)(sgi * 4u + (uint32_t)cb) * 256u;
            const v4u lo = ring_read_global<HW>(g, lane), hi = ring_read_global<HW>(g + 128, lane);
            dst[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          return;
#endif
#if defined(PEARL_W_BSPLIT) && defined(PEARL_W_LO32)
          // goal-arc16 (PEARL_W_BSPLIT=n): the last n B' fragments read straight from global memory (L1: the
          // work-group's 8 row windows read the same lines), the rest from the ring. Results stay right.
          {
            const uint32_t gl = (uint32_t)(uintptr_t)(A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride) + kd * 2048u + sgi * 1024u;
            each<4>([&](int cb) {
              if (cb >= 4 - (PEARL_W_BSPLIT)) dst[cb] = g_read256<HW>(lo32(bHi, gl + (uint32_t)cb * 256u), lane);
              else dst[cb] = slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane);
            });
            return;
          }
#endif
#if defined(PEARL_W_BGT)
          // goal-arc16 timing (wrong results, the layout is not DPAS's): B' as 2 gathers of 512 bytes a k-step.
          each<2>([&](int pp) {
            const v16i g = slm_gather512<HW>(R + off + jc * 512u + (sgi * 4u + 2u * (uint32_t)pp) * 64u, lane);
            dst[2 * pp] = __builtin_shufflevector(g, g, 0, 1, 2, 3, 4, 5, 6, 7);
            dst[2 * pp + 1] = __builtin_shufflevector(g, g, 8, 9, 10, 11, 12, 13, 14, 15);
          });
          return;
#endif
#if defined(PEARL_W_BHB)
          // goal-arc16 timing (wrong results): each B' fragment's first 128 bytes only, used twice: half the bytes,
          // the same 4 messages.
          each<4>([&](int cb) {
            const v4u h = ring_read_local<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane);
            dst[cb] = (v8i)__builtin_shufflevector(h, h, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          return;
#endif
#if defined(PEARL_W_SMP) && defined(PEARL_W_BA)
          // goal-arc20 timing (wrong results): the last PEARL_W_SMP of the 4 fragments through the sampler, from a
          // bindless image laid out as B' [n, k] row-major is (32 columns of 128 texels an image row, 2,048 rows):
          // lane j fetches the 2 texels of column j's 32 bytes, which are DPAS src1's order. 2 messages of 128
          // bytes for one 256-byte local read. The copy still fills the whole ring.
          {
            namespace sx = sycl::ext::oneapi::experimental;
            const sx::unsampled_image_handle hb{A.img};
#if defined(PEARL_W_SMPAH)
            // the sampler fragment (SMP 1 only) fetched a k-step ahead
            each<3>([&](int cb) { dst[cb] = slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
            dst[3] = smpn;
            smpn = smp_fetch(kd + 1u < 64u ? kd + 1u : 63u);
            return;
#endif
            each<4>([&](int cb) {
              if (cb >= 4 - (PEARL_W_SMP)) {
                const uint32_t c = (A.colStart / 4u + wc) * 64u + sgi * 32u + (uint32_t)cb * 8u + lane;
                const int x = (int)((c & 31u) * 128u + 2u * kd), y = (int)((c >> 5) & 2047u);
                const sycl::vec<uint32_t, 4> lo = sx::fetch_image<sycl::vec<uint32_t, 4>>(hb, sycl::vec<int, 2>{x, y});
                const sycl::vec<uint32_t, 4> hi = sx::fetch_image<sycl::vec<uint32_t, 4>>(hb, sycl::vec<int, 2>{x + 1, y});
                dst[cb] = v8i{(int)lo[0], (int)lo[1], (int)lo[2], (int)lo[3], (int)hi[0], (int)hi[1], (int)hi[2], (int)hi[3]};
              } else {
                dst[cb] = slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane);
              }
            });
            return;
          }
#endif
#if defined(PEARL_W_BA)
          // goal-arc13: a B' fragment as one 256-byte message (inline vISA): 4 local reads a k-step for 8.
          // goal-arc14: keeping the last k-step's B' live so the finalizer cannot reuse its registers (an empty
          // asm using it, or extra unused inputs on the last load) changes nothing in the ISA: IGC drops operands
          // the asm text does not name, and k-step 1's loads still land in k-step 0's registers. "=&rw" drops
          // the load. So BA stays off for 24.35 at RW 16. At RW 8 (the default from goal-arc15) 24.35 passes.
#if defined(PEARL_W_BJIT)
          each<(PEARL_W_BJIT)>([&](int cb) { dst[cb] = slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
          return;
#endif
#if defined(PEARL_W_VZB)
          // goal-arc21 (PEARL_W_VZB): the ring pointer made per-lane to the compiler (plus a runtime 0 times the lane),
          // so each read's address is one 8-lane add instead of a scalar add and a broadcast mov.
          each<4>([&](int cb) { dst[cb] = slm_read256<HW>(R + (lane * A.zero) + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
#else
          each<4>([&](int cb) { dst[cb] = slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane); });
#endif
          return;
#endif
#if defined(PEARL_W_B8)
          // goal-arc8: a B' fragment as one 256-byte local block read (IGC still sends two 128-byte messages).
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
#if defined(PEARL_W_TR)
          v8i a8[4];
          if constexpr ((PEARL_W_TR) == 1) {
            each<4>([&](int rb) { a8[rb] = an8[rb]; });
            load_a8(ks + 1u < 64u ? ks + 1u : 63u, an8);
          } else if constexpr ((PEARL_W_TR) == 3) {
            // fragments 0, 1 a k-step ahead (16 registers), 2, 3 in the k-step
            a8[0] = an8[0];
            a8[1] = an8[1];
            load_a8r(kd, a8, 2, 4);
            load_a8r(ks + 1u < 64u ? ks + 1u : 63u, an8, 0, 2);
          } else {
            load_a8(kd, a8);
          }
          v4i bh[4];
          each<2>([&](int bb) {
            const v8i t = slm_read256<HW>(R + slot + jc * 512u + sgi * 128u + (uint32_t)bb * 256u, lane);
            bh[2 * bb] = __builtin_shufflevector(t, t, 0, 1, 2, 3);
            bh[2 * bb + 1] = __builtin_shufflevector(t, t, 4, 5, 6, 7);
          });
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              each<2>([&](int h) {
                v4i &acc = c4[rb >> 1][(rb & 1) * 4 + cb][h];
                acc = split_dpas4<HW>(v2i{bh[cb][2 * h], bh[cb][2 * h + 1]}, a8[rb], acc);
              });
            });
          });
#else
          v4i ah[4];
#if defined(PEARL_W_NOLD) && ((PEARL_W_NOLD) & 1)
          if constexpr (false) {
#else
          if constexpr (AH) {
#endif
#if defined(PEARL_W_AEARLY)
            each<4>([&](int rb) { ah[rb] = ahe[rb]; });
#elif defined(PEARL_W_AV) && defined(PEARL_W_AA) && defined(PEARL_W_LO32)
            each<2>([&](int bb) {
              ah[2 * bb] = __builtin_shufflevector(ahv[bb], ahv[bb], 0, 1, 2, 3);
              ah[2 * bb + 1] = __builtin_shufflevector(ahv[bb], ahv[bb], 4, 5, 6, 7);
            });
#else
            each<4>([&](int rb) { ah[rb] = ahn[rb]; });
#endif
#if defined(PEARL_W_AROLL) || defined(PEARL_W_AEARLY)
#elif defined(PEARL_W_AV) && defined(PEARL_W_AA) && defined(PEARL_W_LO32)
#if defined(PEARL_W_AHEND)
            if (s < 3u)
#endif
            load_ahv(ks + 1u < 64u ? kdo(ks + 1u) : 63u);
#elif defined(PEARL_W_AHEND)
            if (s < 3u) load_ah(ks + 1u < 64u ? kdo(ks + 1u) : 63u, ahn);
#elif defined(PEARL_W_AH2)
            each<4>([&](int rb) { ahn[rb] = ahn2[rb]; });
            load_ah(ks + 2u < 64u ? kdo(ks + 2u) : 63u, ahn2);
#elif !defined(PEARL_W_AHLATE)
            load_ah(ks + 1u < 64u ? kdo(ks + 1u) : 63u, ahn);
#endif
#if defined(PEARL_W_PFG) && defined(PEARL_W_LO32) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
            if constexpr (HW && CW == 2) {
              if ((s & 1u) == (uint32_t)(PEARL_W_PFGS)) {
                const uint32_t lo = pfgLo + ((ks + (uint32_t)(PEARL_W_PFG)) & 63u) * 512u;
                const uint64_t pa = ((uint64_t)aHi << 32) | lo;
                __asm__ volatile("lsc_load.ugm." PEARL_W_PFGC " (M1, 8) %%null:d32 flat[%0]:a64" : : "rw"(pa));
              }
            }
#endif
          } else {
#if defined(PEARL_W_AHALF)
            // goal-arc8 timing (wrong results): A' loaded every other k-step, XORed so its bits still change.
            if ((s & 1u) == 0u) load_ah(kd, ahp);
            each<4>([&](int rb) { ah[rb] = ahp[rb] ^ (int)(ks * 0x85EBCA6Bu); });
#else
#if !(defined(PEARL_W_NOLD) && ((PEARL_W_NOLD) & 1))
            load_ah(kd, ah);
#endif
#if defined(PEARL_W_XONLY)
            each<4>([&](int rb) { ah[rb] ^= (int)(ks * 0x85EBCA6Bu); });
#endif
#endif
          }
          cp_at(1);
#if defined(PEARL_W_BHALF)
          if ((s & 1u) == 0u) read_b_at(slot, bqp);
          each<4>([&](int cb) { bq[cb] = bqp[cb] ^ (int)(ks * 0x9E3779B9u); });
#else
#if defined(PEARL_W_BREG)
          each<4>([&](int cb) { bq[cb] = bqr[cb]; });
#elif defined(PEARL_W_NOLD) && ((PEARL_W_NOLD) & 2)
          each<4>([&](int cb) { bq[cb] = bqc[cb]; });
#elif defined(PEARL_W_BFIRST)
          each<4>([&](int cb) { bq[cb] = bqf[cb]; });
#elif defined(PEARL_W_BROLL)
          // goal-arc16 (PEARL_W_BROLL): B' kept in bqn across k-steps; each fragment's next k-step is read as soon
          // as its DPAS are issued (see below), into the same registers. The chunk's first k-step reads its own.
          if (s == 0u) each<4>([&](int cb) { bqn[cb] = rd1(slot, cb); });
#elif defined(PEARL_W_BAH)
          // goal-arc16 (PEARL_W_BAH): B' of the chunk's next k-step read a k-step ahead (its slot is ready since
          // the chunk's wait); the chunk's first k-step reads its own. Costs 32 registers.
          // PEARL_W_BAH=n: only fragments 0 .. n - 1 ahead (8 registers each); the rest read in the k-step.
          {
            constexpr int kBah = (PEARL_W_BAH) > 0 && (PEARL_W_BAH) < 4 ? (PEARL_W_BAH) : 4;
            auto rd1 = [&](uint32_t off, int cb) {
              return slm_read256<HW>(R + off + jc * 512u + (sgi * 4u + (uint32_t)cb) * 64u, lane);
            };
            each<4>([&](int cb) {
              if (cb >= kBah || s == 0u) bq[cb] = rd1(slot, cb);
              else bq[cb] = bqn[cb];
            });
            if (s < 3u) each<kBah>([&](int cb) { bqn[cb] = rd1(at(kd + 1u), cb); });
          }
#else
          read_b();
#endif
#if defined(PEARL_W_XONLY)
          each<4>([&](int cb) { bq[cb] ^= (int)(ks * 0x9E3779B9u); });
#endif
#endif
          cp_at(2);
#if defined(PEARL_W_NOLD) && ((PEARL_W_NOLD) & 1)
          each<4>([&](int rb) { ah[rb] = ahc[rb]; });
#endif
#if defined(PEARL_W_XSEND)
          // goal-arc16 timing: PEARL_W_XSEND extra 256-byte local reads a k-step whose data no one uses (the price of
          // a message by itself, no wait on it); PEARL_W_XSENDG the same from global memory (an L1 hit).
          each<(PEARL_W_XSEND)>([&](int j) { (void)slm_read256<HW>(R + slot + (uint32_t)j * 64u, lane); });
#endif
#if defined(PEARL_W_XSEND1) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
          // goal-arc16 timing: PEARL_W_XSEND1 unused 32-byte local reads a k-step (one register each).
          each<(PEARL_W_XSEND1)>([&](int j) {
            uint32_t r1;
            const uint32_t a1 = (uint32_t)(uintptr_t)(__attribute__((opencl_local)) uint32_t *)(R + slot + (uint32_t)j * 64u);
            __asm__ volatile("lsc_load.slm (M1_NM, 1) %0:d32x8t flat[%1]:a32" : "=rw"(r1) : "rw"(a1));
            (void)r1;
          });
#endif
#if defined(PEARL_W_XSENDG)
          each<(PEARL_W_XSENDG)>([&](int j) { (void)g_read256<HW>(lo32(aHi, aLo + (uint32_t)j * 256u), lane); });
#endif
#if defined(PEARL_W_SLOTS)
          if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#else
          if (kS2 && s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
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
#if defined(PEARL_W_SNAKE)
          // goal-arc10 experiment: a snake through each pair of B' fragments, so neighbours share A' or B' and
          // a run's first two DPAS have different B' (the finalizer's dummy DPAS goes only before runs whose
          // first two share it).
          each<2>([&](int cp) {
            each<4>([&](int rb) {
              each<2>([&](int j) {
                const int cb = 2 * cp + ((rb & 1) ? 1 - j : j);
                c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
              });
            });
          });
#elif defined(PEARL_W_ROWFIRST) && !defined(PEARL_W_D4)
          // goal-arc7 experiment: the DPAS of one A' fragment back to back.
          each<4>([&](int rb) {
            each<4>([&](int cb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
#elif defined(PEARL_W_D4) && defined(PEARL_W_AROLL) && defined(PEARL_W_AA)
          // goal-arc16 (PEARL_W_AROLL): DPAS A'-major (row block outer), and the next k-step's A' read into each
          // 16-row block's registers as soon as its DPAS are issued: A' a k-step ahead with no second set.
          each<2>([&](int bb) {
            each<2>([&](int r2) {
              const int rb = 2 * bb + r2;
              each<4>([&](int cb) {
                each<2>([&](int h) {
                  v4i &acc = c4[rb >> 1][(rb & 1) * 4 + cb][h];
                  acc = split_dpas4<HW>(v2i{ahn[rb][2 * h], ahn[rb][2 * h + 1]}, bq[cb], acc);
                });
              });
            });
            const uint32_t kn = ks + 1u < 64u ? ks + 1u : 63u;
            const v8i t = g_read256<HW>(lo32(aHi, aLo + sgi * 128u + kn * 512u + (uint32_t)bb * A.aStride), lane);
            ahn[2 * bb] = __builtin_shufflevector(t, t, 0, 1, 2, 3);
            ahn[2 * bb + 1] = __builtin_shufflevector(t, t, 4, 5, 6, 7);
          });
#elif defined(PEARL_W_D4) && defined(PEARL_W_BROLL)
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              each<2>([&](int h) {
                v4i &acc = c4[rb >> 1][(rb & 1) * 4 + cb][h];
                acc = split_dpas4<HW>(v2i{ah[rb][2 * h], ah[rb][2 * h + 1]}, bqn[cb], acc);
              });
            });
            if (s < 3u) bqn[cb] = rd1(at(kd + 1u), cb);
          });
#elif defined(PEARL_W_D4)
          // goal-arc12: each 8x8 split DPAS as two 8x4 ones (rows {2h, 2h + 1} of each sub-group's half of A').
          // The finalizer adds its dummy DPAS (Wa_18023229625) only before 8x8 runs, so this form never gets one:
          // 94.43 against 93.55 TH/s (card 5, three rounds), with 0 dummies for 3. Every design that changed which
          // registers the hash's ALU code lands in had cost 0.5% a dummy before; now none can.
          each<4>([&](int cb) {
            if (cb == 2) cp_at(3);
            each<4>([&](int rb) {
              each<2>([&](int h) {
                v4i &acc = c4[rb >> 1][(rb & 1) * 4 + cb][h];
#if defined(PEARL_W_D2T)
                // goal-arc27 energy timing: each 8x4 split DPAS as two 8x2 ones (one row of each sub-group's half a
                // DPAS). The check passes, but the CW 2 instance spills and costs 18% more joules a MAC (card 5).
                v2i lo = __builtin_shufflevector(acc, acc, 0, 1), hi = __builtin_shufflevector(acc, acc, 2, 3);
                lo = split_dpas2<HW>(ah[rb][2 * h], bq[cb], lo);
                hi = split_dpas2<HW>(ah[rb][2 * h + 1], bq[cb], hi);
                acc = __builtin_shufflevector(lo, hi, 0, 1, 2, 3);
#elif defined(PEARL_W_BSEL) || defined(PEARL_W_ASEL)
                // goal-arc27 energy timing (wrong results): the DPAS read only fragments cb & BSEL of B' and rb & ASEL,
                // h & ASEL of A' (all loads kept), so src1 / src2 change less often between DPAS, real bits. BSEL 0
                // (one B' fragment for the whole k-step): 0.8% fewer joules a MAC; ASEL 0 changes the loop (worse).
#ifndef PEARL_W_BSEL
#define PEARL_W_BSEL 3
#endif
#ifndef PEARL_W_ASEL
#define PEARL_W_ASEL 3
#endif
                const int ra = rb & (PEARL_W_ASEL), ha = h & (PEARL_W_ASEL);
                acc = split_dpas4<HW>(v2i{ah[ra][2 * ha], ah[ra][2 * ha + 1]}, bq[cb & (PEARL_W_BSEL)], acc);
#else
                acc = split_dpas4<HW>(v2i{ah[rb][2 * h], ah[rb][2 * h + 1]}, bq[cb], acc);
#endif
              });
            });
#if defined(PEARL_W_RDI)
            if (s == 3u && cb >= (PEARL_W_RDI)) {
              const int pc = cb - (PEARL_W_RDI);
              each<2>([&](int j) {
                rdP[j] ^= (c4[0][j * 4 + pc][0] ^ c4[0][j * 4 + pc][1]) ^ (c4[1][j * 4 + pc][0] ^ c4[1][j * 4 + pc][1]);
              });
            }
#endif
#if defined(PEARL_W_BJIT)
            // goal-arc21 (PEARL_W_BJIT=n): only n B' fragments read before the DPAS; fragment cb + n is read once
            // fragment cb's DPAS are issued, so at most n + 1 are live (registers for A' further ahead).
            if (cb + (PEARL_W_BJIT) < 4) bq[cb + (PEARL_W_BJIT)] = rd1(slot, cb + (PEARL_W_BJIT));
#endif
          });
          cp_at(4);
#else
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = split_dpas<HW>(ah[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
#endif
#if defined(PEARL_W_AHLATE)
          if constexpr (AH) load_ah(ks + 1u < 64u ? kdo(ks + 1u) : 63u, ahn);
#endif
#if defined(PEARL_W_CSAME)
          each<kBps>([&](int i) { ring_write_local<HW>(mine(i) + at(ks + (uint32_t)KPS), csv[i], lane); });
          if (s % (uint32_t)KPS == (uint32_t)KPS - 1u) arrive(it);
#endif
#endif  // PEARL_W_TR
        } else {
          v8i af[4];
          each<4>([&](int rb) {
            const int8_t *p = Ak + (size_t)(rb >> 1) * A.aStride + (size_t)(rb & 1) * 256u;
            const v4u lo = ring_read_global<HW>(p, lane), hi = ring_read_global<HW>(p + 128, lane);
            af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          read_b();
#if defined(PEARL_W_BJIT)
          each<4>([&](int cb) { if (cb >= (PEARL_W_BJIT)) bq[cb] = rd1(slot, cb); });
#endif
          each<4>([&](int cb) {
            each<4>([&](int rb) {
              c[rb >> 1][(rb & 1) * 4 + cb] = dpas<8, HW>(sg, af[rb], bq[cb], c[rb >> 1][(rb & 1) * 4 + cb]);
            });
          });
        }
      }
#if defined(PEARL_W_AHEND)
      // goal-arc22 (PEARL_W_AHEND): the next chunk's first A' after the chunk's k-steps (1: before the readout; 2:
      // after it), so no A' is live in k-step 3 for the next chunk.
      auto ahend = [&]() {
        if constexpr (kSplit && AH) {
          const uint32_t kn = ch * 4u + 4u < 64u ? kdo(ch * 4u + 4u) : 63u;
#if defined(PEARL_W_AV) && defined(PEARL_W_AA) && defined(PEARL_W_LO32)
          load_ahv(kn);
#else
          load_ah(kn, ahn);
#endif
        }
      };
      if constexpr ((PEARL_W_AHEND) == 1) ahend();
#endif
      // The chunk's readout, the ring fold's.
      uint32_t x0 = 0, x1 = 0;
#if defined(PEARL_W_D4) && defined(PEARL_W_ASMRD4) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
      // goal-arc12: the D4 readout as a chain of three-input XORs (bfn 0x96) 16 lanes wide in inline vISA: 32
      // instructions a chunk where IGC emits about 60 eight-lane ones. Every lane is active here. PASS, but 93.8
      // against 96.4 (card 5): a 16-lane instruction costs what two 8-lane ones do, and IGC's own tree keeps its
      // temporaries in the acc registers. Off.
      {
        auto x3 = [](v4i a, v4i b, v4i d) {
          v4i r;
          __asm__("bfn.x96 (M1_NM, 16) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0> %3(0,0)<1;1,0>\n"
                  "bfn.x96 (M1_NM, 16) %0(2,0)<1> %1(2,0)<1;1,0> %2(2,0)<1;1,0> %3(2,0)<1;1,0>"
                  : "=rw"(r)
                  : "rw"(a), "rw"(b), "rw"(d));
          return r;
        };
        v4i t = x3(c4[0][0][0], c4[0][0][1], c4[0][1][0]);
        // the other 29 halves two at a time, then the last one
        each<14>([&](int j) {
          const int i0 = 3 + 2 * j, i1 = 4 + 2 * j;  // flat half index: (rb * 8 + cb) * 2 + h
          t = x3(t, c4[i0 >> 4][(i0 >> 1) & 7][i0 & 1], c4[i1 >> 4][(i1 >> 1) & 7][i1 & 1]);
        });
        t ^= c4[1][7][1];
        x0 = (uint32_t)(t[0] ^ t[1]);
        x1 = (uint32_t)(t[2] ^ t[3]);
      }
#elif defined(PEARL_W_D4)
      // rows 0-3 are elements 0, 1 of both halves; rows 4-7 elements 2, 3. The reference code (kSplit false)
      // runs the plain 8x8 DPAS into c, so it reads c the 8x8 way (goal-arc13: reading the unused c4 made
      // xmx8w-emu fail on every region from 2e3d4887 on).
      if constexpr (!kSplit) {
        v8i t4[4];
        each<4>([&](int q) {
          t4[q] = (c[q >> 1][(q & 1) * 4] ^ c[q >> 1][(q & 1) * 4 + 1]) ^ (c[q >> 1][(q & 1) * 4 + 2] ^ c[q >> 1][(q & 1) * 4 + 3]);
        });
        const v8i t = (t4[0] ^ t4[1]) ^ (t4[2] ^ t4[3]);
        x0 = (uint32_t)((t[0] ^ t[1]) ^ (t[2] ^ t[3]));
        x1 = (uint32_t)((t[4] ^ t[5]) ^ (t[6] ^ t[7]));
      } else {
#if defined(PEARL_W_AA)
        // A' read as 256-byte pieces (PEARL_W_AA): accumulators with rb & 1 == 0 hold row class 0 (all four
        // elements), rb & 1 == 1 class 1. c4[rb >> 1][(rb & 1) * 4 + cb][h].
#if defined(PEARL_W_RDI)
        // the fragments the k-step's DPAS loop did not reach (the last PEARL_W_RDI)
        each<(PEARL_W_RDI)>([&](int i) {
          const int pc = 4 - (PEARL_W_RDI) + i;
          each<2>([&](int j) {
            rdP[j] ^= (c4[0][j * 4 + pc][0] ^ c4[0][j * 4 + pc][1]) ^ (c4[1][j * 4 + pc][0] ^ c4[1][j * 4 + pc][1]);
          });
        });
        const v4i t0 = rdP[0], t1 = rdP[1];
#else
        v4i u[2][2];
        each<2>([&](int j) {
          each<2>([&](int q) {
            u[j][q] = (c4[q][j * 4][0] ^ c4[q][j * 4][1]) ^ (c4[q][j * 4 + 1][0] ^ c4[q][j * 4 + 1][1])
                      ^ ((c4[q][j * 4 + 2][0] ^ c4[q][j * 4 + 2][1]) ^ (c4[q][j * 4 + 3][0] ^ c4[q][j * 4 + 3][1]));
          });
        });
        const v4i t0 = u[0][0] ^ u[0][1], t1 = u[1][0] ^ u[1][1];
#endif
        x0 = (uint32_t)((t0[0] ^ t0[1]) ^ (t0[2] ^ t0[3]));
        x1 = (uint32_t)((t1[0] ^ t1[1]) ^ (t1[2] ^ t1[3]));
#else
        v4i q8[8];
        each<8>([&](int q) {
          q8[q] = (c4[q >> 2][(q & 3) * 2][0] ^ c4[q >> 2][(q & 3) * 2][1]) ^ (c4[q >> 2][(q & 3) * 2 + 1][0] ^ c4[q >> 2][(q & 3) * 2 + 1][1]);
        });
#if defined(PEARL_W_RDH)
        // goal-arc13 timing (wrong results): the chunk's readout reads only the first PEARL_W_RDH of the 8 groups
        // of 4 halves; all of them are XORed once after the loop, so no DPAS is dropped. RDH 4 96.65, RDH 1 98.2
        // against 96.5 (card 5, before AA): the readout costs about 2%, not the 6% the 8x8 form's RD2 suggested.
        each<8>([&](int q) { if (q >= (PEARL_W_RDH)) q8[q] = 0; });
#endif
        const v4i t = ((q8[0] ^ q8[1]) ^ (q8[2] ^ q8[3])) ^ ((q8[4] ^ q8[5]) ^ (q8[6] ^ q8[7]));
        x0 = (uint32_t)(t[0] ^ t[1]);
        x1 = (uint32_t)(t[2] ^ t[3]);
#endif
      }
#elif defined(PEARL_W_NORD)
      // goal-arc8 timing (wrong results): the readout reads one accumulator only.
      x0 = (uint32_t)c[0][0][0];
      x1 = (uint32_t)c[0][0][4];
#elif !defined(PEARL_W_NOVRD)
      // goal-arc8: the same XOR as a tree of whole accumulators first, then the 8 dwords (see kPartX).
      v8i t4[4];
      each<4>([&](int q) {
        t4[q] = (c[q >> 1][(q & 1) * 4] ^ c[q >> 1][(q & 1) * 4 + 1]) ^ (c[q >> 1][(q & 1) * 4 + 2] ^ c[q >> 1][(q & 1) * 4 + 3]);
      });
#if defined(PEARL_W_AA)
      if constexpr (kSplit) {
        // goal-arc13 (8x8 DPAS with AA): accumulators with rb & 1 == 0 hold row class 0 in all 8 elements.
        const v8i u0 = t4[0] ^ t4[2], u1 = t4[1] ^ t4[3];
        x0 = (uint32_t)(((u0[0] ^ u0[1]) ^ (u0[2] ^ u0[3])) ^ ((u0[4] ^ u0[5]) ^ (u0[6] ^ u0[7])));
        x1 = (uint32_t)(((u1[0] ^ u1[1]) ^ (u1[2] ^ u1[3])) ^ ((u1[4] ^ u1[5]) ^ (u1[6] ^ u1[7])));
      } else
#endif
      {
      const v8i t = (t4[0] ^ t4[1]) ^ (t4[2] ^ t4[3]);
      x0 = (uint32_t)((t[0] ^ t[1]) ^ (t[2] ^ t[3]));
      x1 = (uint32_t)((t[4] ^ t[5]) ^ (t[6] ^ t[7]));
      }
#if defined(PEARL_W_RD2)
      // goal-arc12 timing: a second tree over the accumulators (sums, so it is not the XOR tree again), ANDed with
      // a runtime zero: the readout's instructions twice, the same results. 84.8 against 93.55 (card 5, 8x8 DPAS,
      // 9 dummy DPAS for 3): the readout's 60 instructions a chunk cost about 6%.
      {
        v8i u4[4];
        each<4>([&](int q) {
          u4[q] = (c[q >> 1][(q & 1) * 4] + c[q >> 1][(q & 1) * 4 + 1]) + (c[q >> 1][(q & 1) * 4 + 2] + c[q >> 1][(q & 1) * 4 + 3]);
        });
        const v8i u = (u4[0] + u4[1]) + (u4[2] + u4[3]);
        x0 ^= (uint32_t)((u[0] + u[1]) + (u[2] + u[3])) & A.zero;
        x1 ^= (uint32_t)((u[4] + u[5]) + (u[6] + u[7])) & A.zero;
      }
#endif
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
#if defined(PEARL_W_PXA) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
        // goal-arc20 (PEARL_W_PXA, PXT 2's layout): the chunk's atomic by inline vISA, its address set once a set
        // and stepped by 32 bytes a chunk, so IGC has no address arithmetic in the loop to rebuild.
        if constexpr (kPartX && HW) {
          const uint32_t pa = pxa0 + ch * 32u, pv = (lane & 1u) ? x1 : x0;
#if defined(PEARL_W_PXST)
          // goal-arc21 timing (wrong results): a plain 8-lane store where the atomic was: the price of the atomic itself.
          __asm__ volatile("lsc_store.slm (M1, 8) flat[%0]:a32 %1:d32" : : "rw"(pa), "rw"(pv));
#else
          __asm__ volatile("lsc_atomic_xor.slm (M1, 8) %%null:d32 flat[%0]:a32 %1 %%null" : : "rw"(pa), "rw"(pv));
#endif
        } else
#endif
        if constexpr (kPartX) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
#if defined(PEARL_W_PXT) || defined(PEARL_W_PARTMIN)
              px(part[pOff + pix(widx, lane & 7u, ch)]);
#else
              px(part[pOff + (widx * 8u + (lane & 7u)) * 16u + ch]);
#endif
#if defined(PEARL_W_NOPXA)
          // goal-arc18 timing (wrong results): no local atomic a chunk; the words only XORed into a register.
          // Card 5: def 100.8-100.9 against 100.2-100.3 (+0.6%); 32-thread skeleton +0.5%, 64-thread +2.4%.
          // Two chunks a 64-bit atomic instead: OUT_OF_RESOURCES. See PXT and NOPARTX for real forms.
          (void)px;
          tw[0] ^= (lane & 1u) ? x1 : x0;
#elif defined(PEARL_W_RDNB)
          // goal-arc11 experiment: no `if` (a sub-group has 8 lanes, so it is always true). The loop is then one
          // block with no goto/join, but has 7 dummy DPAS for 3: 92.6 against 93.55 (card 5).
          px.fetch_xor((lane & 1u) ? x1 : x0);
#else
          if (lane < 8u) px.fetch_xor((lane & 1u) ? x1 : x0);
#endif
        } else {
          if (lane < 8u) part[pOff + (sgl * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
        }
      }
#if defined(PEARL_W_AHEND)
      if constexpr ((PEARL_W_AHEND) == 2) ahend();
#endif
    }
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc23 (kSqWt, S24; PEARL_W_SQWT forces it): an instruction that reads the copy's hold after the chunk
    // loop, so the set's last copy load is waited for here. Without it 24.35 lets that load land after its register
    // is loaded again (an L1 hit, as k-step 63 re-read, lands first; a miss past the block lands last): with the
    // run-on loads unstored, 14,040 of 16,384 regions wrong, with this 0. PEARL_W_SQWT=2 also reads A' a k-step ahead
    // here: that breaks even forms that pass (15,128 wrong), as A' carried out of a loop broke in goal-arc21.
    if constexpr (kSeqP && kSqWt > 0) each<kBps>([&](int i) {
      uint32_t y;
      __asm__ volatile("mov (M1, 8) %0(0,0)<1> %1(0,0)<1;1,0>" : "=rw"(y) : "rw"(next[i]) : "memory");
      if (A.zero && y == 0x12345678u) part[0] = y;
    });
    if constexpr (kSeqP && kSqWt >= 2 && kSplit && AH) each<4>([&](int rb) {
      uint32_t y;
      __asm__ volatile("mov (M1, 8) %0(0,0)<1> %1(0,0)<1;1,0>" : "=rw"(y) : "rw"(ahn[rb]) : "memory");
      if (A.zero && y == 0x12345678u) part[1] = y;
    });
#endif
#if defined(PEARL_W_SQMEET)
    if (kSqB) return;
    (void)lastSet;
#else
    if (kSqB || !lastSet) return;
#endif
    wait(it);
#if defined(PEARL_W_NOPXA)
    if (A.zero && tw[0] == 0x12345678u) part[0] = tw[0];
#endif
#if defined(PEARL_W_NOPUT)
    if (A.zero && sink == 0x12345678u) part[0] = sink;
#endif
#if defined(PEARL_W_RDH) && defined(PEARL_W_D4)
    if constexpr (kSplit) {
      v4i u = 0;
      each<2>([&](int rb) { each<8>([&](int cb) { u ^= c4[rb][cb][0] ^ c4[rb][cb][1]; }); });
      const uint32_t uu = (uint32_t)(u[0] ^ u[1] ^ u[2] ^ u[3]);
      if (A.zero && uu == 0x12345678u) part[0] = uu;
    }
#endif
#if defined(PEARL_W_XALU)
    if (A.zero && (xa[0] ^ xa[1] ^ xa[2] ^ xa[3]) == 0x12345678u) part[0] = xa[0];
#endif
    sycl::group_barrier(it.get_group());
    };
#if defined(PEARL_W_SEQUNROLL)
    // goal-arc10: the two sets unrolled: 20 dummy DPAS instead of 4 (registers the second set's set-up writes),
    // 90.7 TH/s against 92.1 for one set (card 5).
    each<SEQ>([&](int st) { run_set(st, std::false_type{}); });
#else
    if constexpr (kSqB) {
      run_set(0, std::true_type{});
      arrive(it);
    }
    // A loop, one copy of the code: 92.9 TH/s against 92.1 for one set (card 5, three rounds, 188 W).
#pragma unroll 1
    for (int set = 0; set < SEQ; set++) run_set(set, std::false_type{});
    if constexpr (kSqB) {
      wait(it);
      sycl::group_barrier(it.get_group());
    }
#endif
#if defined(PEARL_W_PADR) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    {
      uint32_t px = 0u;
      each<(PEARL_W_PADR)>([&](int i) { px ^= padr[i]; });
      if (px == 0x12345678u) part[0] = px;
    }
#endif
#if defined(PEARL_W_PFT) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    // goal-arc23 (PEARL_W_PFT=n): the next band's A' brought into L2 from the hash tail. A band's first column of
    // work-groups otherwise reads its rows from DRAM a k-step ahead, and runs ~6% slower (so does the work-group
    // sharing its Xe-core). Work-group i of band b loads k-steps n i .. n i + n - 1 of every 16-row block of band
    // b + 1 (8-lane gathers, a dword from each 64-byte line: 512 bytes of lines a message): the work-groups end in
    // about index order, so k-step k is loaded about when band b + 1's first column reaches it, a little before.
    // The loads are real (a load with no destination does nothing on this card); their data is used after the
    // hash, so their latency hides behind it. Past the last band nothing is loaded (the next launch starts cold).
    // Result: the first column runs normally, but each work-group ends ~2.5 us later (it waits for its loads before
    // it can end, longer than the hash takes): PFT 2 / 1 101.1-101.2 / 101.7-101.8 against 102.9. PEARL_W_PFTA (an
    // atomic load, nothing to wait for): 101.2. Off.
    constexpr uint32_t kPft = (PEARL_W_PFT);
    uint32_t pfv[2 * kPft];
    each<2 * (int)kPft>([&](int m) { pfv[m] = 0u; });
    if constexpr (HW) {
      const uint32_t bandG = A.band / RW > 0u ? A.band / RW : 1u, rowG = rowWins / RW;
      const uint32_t per = bandG * (colWins / CW / (uint32_t)SEQ), g = (uint32_t)it.get_group(0);
      const uint32_t b = g / per, idx = g - b * per, nb = (rowG + bandG - 1u) / bandG;
      if (b + 1u < nb && idx * kPft < 64u) {
        const uint32_t b1 = b + 1u, rows1 = rowG - b1 * bandG < bandG ? rowG - b1 * bandG : bandG;
        const uint32_t blocks = 2u * rows1 * RW;  // 16-row blocks in band b1; sub-group s takes blocks 2 s, 2 s + 1
        const uint64_t base = (uint64_t)(uintptr_t)A.Ap + 2ull * b1 * bandG * RW * A.aStride + (uint64_t)idx * kPft * 512u + lane * 64u;
        each<2 * (int)kPft>([&](int m) {
          const uint32_t blk = 2u * sgl + (uint32_t)(m & 1);
          const uint64_t a = base + (uint64_t)(blk < blocks ? blk : 0u) * A.aStride + (uint64_t)(m >> 1) * 512u;
#if defined(PEARL_W_PFTA)
          // an atomic load with no destination: done in L2, nothing to wait for
          (void)pfv;
          __asm__ volatile("lsc_atomic_load.ugm (M1, 8) %%null:d32 flat[%0]:a64 %%null %%null" : : "rw"(a));
#else
          __asm__ volatile("lsc_load.ugm." PEARL_W_PFTC " (M1, 8) %0:d32 flat[%1]:a64" : "=rw"(pfv[m]) : "rw"(a));
#endif
        });
      }
    }
#endif
    // Sub-group sgi hashes sets sgi, sgi + 2, ... (both sub-groups of the window at once). The timing variants
    // below that return end after their first set.
#pragma unroll 1
    for (uint32_t js = SEQ > 1 ? sgi : 0u; js < (uint32_t)SEQ; js += 2u) {
    if constexpr (SEQ > 1) {
      wcb = wcbS * (uint32_t)SEQ + js;
      wc = kSqI ? wcbS * (uint32_t)SEQ * CW + (widx / RW) * (uint32_t)SEQ + js : wcb * CW + widx / RW;
    }
#if defined(PEARL_W_REGWORDS)  // (SEQ 1 only: R and tw belong to the set)
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
#endif
    if ((SEQ > 1 || sgi == 0) && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
#if defined(PEARL_W_PXB)
      // goal-arc20: the window's words as two 256-byte block reads: a block read gives lane l, register e the
      // dword 8 e + l, which in PXT 2's layout is word e of region l.
      static_assert(!kRegWords, "PEARL_W_PXB: the part buffer");
      if constexpr (kPartX) {
        uint32_t *P = &part[0];
        const uint32_t pb = (SEQ > 1 && !kPart1 ? js * kPartWords : 0u) + widx * 128u;
        const v8i lo = slm_read256<HW>(P + pb, lane), hi = slm_read256<HW>(P + pb + 64u, lane);
        each<8>([&](int j) {
          words[j] = (uint32_t)lo[j];
          words[j + 8] = (uint32_t)hi[j];
        });
      } else
#endif
      for (int j = 0; j < 16; j++)
#if defined(PEARL_W_PXT) || defined(PEARL_W_PARTMIN)
        words[j] = kPartX ? part[(SEQ > 1 && !kPart1 ? js * kPartWords : 0u) + pix(widx, qr, (uint32_t)j)]
#else
        words[j] = kPartX ? part[(SEQ > 1 && !kPart1 ? js * kPartWords : 0u) + (widx * 8u + qr) * 16u + (uint32_t)j]
#endif
                          // goal-arc18: the set's offset and the window's first sub-group (both hash at SEQ 2;
                          // goal-arc8's NOPARTX failed the check for want of these). With SLOTS 2 (64 KB at SEQ 2):
                          // PASS, 100.4-100.6 against 100.2-100.4 at full speed, but 76.2-76.3 against 76.9 at
                          // duty 75 (card 5). Off.
                          : part[(SEQ > 1 ? js * kPartWords : 0u) + ((sgl & ~1u) * 8u + qr) * 16u + (uint32_t)j]
                            ^ part[(SEQ > 1 ? js * kPartWords : 0u) + ((sgl | 1u) * 8u + qr) * 16u + (uint32_t)j];
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
#if defined(PEARL_W_NOEPI)
      // goal-arc10 timing (no hits): no BLAKE3 in the kernel, the words only XORed together and kept.
      uint32_t x = 0u;
      for (int j = 0; j < 16; j++) x ^= words[j];
      if (x == A.zero + 0x9E3779B9u) A.hits->index[0] = local;
      return;
#endif
#if defined(PEARL_W_SEPEPI)
      if constexpr (HW) {
        // goal-arc10: the words go out to memory (allTr holds the words buffer here) and a second kernel hashes
        // them, so no ALU code of this kernel writes the registers B' lands in (see fold_xmxw).
#if defined(PEARL_W_SEPBLK) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
        // window (wr, wc)'s 8 regions as 16 sub-group block writes of 32 bytes: word j of lane q at
        // ((window * 16 + j) * 8 + q), one uniform address a write.
        {
          (void)local;
          __attribute__((opencl_global)) uint32_t *wp =
              (__attribute__((opencl_global)) uint32_t *)A.allTr + (size_t)(wr + rowWins * wc) * 128u;
#if defined(PEARL_W_SB1)
          uint32_t xx = 0u;  // timing: one write of the XOR of the words
          each<16>([&](int j) { xx ^= words[j]; });
          __spirv_SubgroupBlockWriteINTEL(wp, xx);
#else
          each<16>([&](int j) { __spirv_SubgroupBlockWriteINTEL(wp + j * 8, words[j]); });
#endif
          return;
        }
#endif
        each<16>([&](int j) { A.allTr[(size_t)local * 16u + (uint32_t)j] = words[j]; });
        return;
      }
#endif
#if defined(PEARL_W_EPI2)
      // goal-arc10 timing: the hash twice (the second on other words), to price the epilogue's tail.
      {
        uint32_t w2[16];
        each<16>([&](int j) { w2[j] = words[j] ^ 0x5A5A5A5Au; });
        region_epilogue(w2, local, A.test, A.hits, A.allTr);
      }
#endif
#if defined(PEARL_W_NOINL)
      region_epilogue_ni(words, local, A.test, A.hits, A.allTr);
      return;
#endif
#if defined(PEARL_W_LEAN)
      if constexpr (kPartX) {
        (void)words;
        const volatile uint32_t *pw = &part[(widx * 8u + qr) * 16u];
        region_epilogue_lean(pw, local, A.test, A.hits, A.allTr, A.zero);
        return;
      }
#endif
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
    }  // js
#if defined(PEARL_W_PFT) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    {
      // The loads' data is used here, after the hash, by an asm that IGC may not move above it (memory clobber).
      uint32_t x = 0u;
      each<2 * (int)kPft>([&](int m) {
        uint32_t y;
        __asm__ volatile("xor (M1, 8) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>" : "=rw"(y) : "rw"(x), "rw"(pfv[m]) : "memory");
        x = y;
      });
      if (A.zero && x == 0x12345678u) A.hits->index[0] = x;  // never (A.zero is 0): keeps the loads
    }
#endif
#if defined(PEARL_W_TSC) && defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
    if constexpr (HW) {
      uint32_t t0 = 0, t1 = 0;
      __asm__ volatile("mov (M1_NM, 1) %0(0,0)<1> %%tsc(0,0)<0;1,0>\nmov (M1_NM, 1) %1(0,0)<1> %%tsc(0,1)<0;1,0>" : "=rw"(t0), "=rw"(t1));
      if (lane == 0u && A.tsc)
        sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(A.tsc[(size_t)it.get_group(0) * 4u + 1u])
            .fetch_max(((uint64_t)t1 << 32) | t0);
    }
#endif
  }
};

// `groups` windows, 8 to a work-group (A.wpg must be 8: the copy is one block a sub-group).
#ifndef PEARL_W_KPS
#define PEARL_W_KPS 4
#endif
#ifndef PEARL_W_AH
#define PEARL_W_AH 1  // goal-arc12: A' a k-step ahead (with the 8x4 DPAS: +0.6%)
#endif
#ifndef PEARL_W_RW
#define PEARL_W_RW 8
#endif
#ifndef PEARL_W_CW
#define PEARL_W_CW 1
#endif
template <bool HW, int SEQ = 1, int KPS = PEARL_W_KPS, bool AH = (PEARL_W_AH != 0), int RW = PEARL_W_RW, int CW = PEARL_W_CW,
          bool SP = false, bool S24 = false>
inline sycl::event fold_xmxw(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  typedef FoldXmxW<HW, KPS, AH, RW, CW, SEQ, SP, S24> K;
  return q.submit([&](sycl::handler &h) {
#ifndef PEARL_W_SLMPAD
#define PEARL_W_SLMPAD 0  // goal-arc8 timing: extra words of local memory a work-group (fewer work-groups a core)
#endif
    sycl::local_accessor<uint32_t, 1> part((K::kRegWords ? 1u : K::kPartWords * (K::kPart1 ? 1u : (uint32_t)SEQ)) + (PEARL_W_SLMPAD), h);
#if defined(PEARL_W_NORING)
    sycl::local_accessor<uint32_t, 1> ring(64u, h);  // goal-arc16 timing: B' from global only, no ring
#else
#if defined(PEARL_W_PARTMIN)
    // goal-arc18 timing: the ring's last words lie on top of `part`, so ring and part fit 32 KB together.
#ifndef PEARL_W_RINGCUT
#define PEARL_W_RINGCUT 0
#endif
    sycl::local_accessor<uint32_t, 1> ring(K::kSlots * K::kSlotWords - K::kPartWords * (uint32_t)SEQ - (PEARL_W_RINGCUT), h);
#else
    sycl::local_accessor<uint32_t, 1> ring(K::kSlots * K::kSlotWords, h);
#endif
#endif
#if defined(PEARL_W_NC)
    constexpr uint32_t kSgs = K::kNsg + (PEARL_W_NC);  // goal-arc8: and the copier sub-groups
#else
    constexpr uint32_t kSgs = K::kNsg;
#endif
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / (RW * CW * SEQ)) * kSgs * 8u, kSgs * 8u),
#if defined(PEARL_W_ARGP)
                   K{a, part, ring, a.Ap, a.Bp});
#else
                   K{a, part, ring});
#endif
  });
}

#if defined(PEARL_W_SEPEPI)
// goal-arc10: xmx8w with its epilogue in a kernel of its own. `words` holds 16 words a region of the batch.
// goal-arc17 (PEARL_W_SEPQ): the hash kernel on a queue of its own (hq), after the fold's event, so the next
// batch's fold need not wait for it; *hev gets the hash kernel's event. bench.js, card 5, full speed: 93.1-93.6
// against 98.9-99.1 (no hash at all 99.7-100.2): the driver does not run the two kernels side by side. Off.
inline void fold_xmxw_sep(sycl::queue &q, const FoldArgs &a, uint32_t groups, uint32_t *words,
                          sycl::queue *hq = nullptr, sycl::event *hev = nullptr) {
  FoldArgs b = a;
  b.allTr = words;
  sycl::event fe;
#if defined(PEARL_W_SEPQ)
  if ((a.colBatch / 4u) % (2u * PEARL_W_CW) == 0u) fe = fold_xmxw<true, 2>(q, b, groups);
  else
#endif
  fe = fold_xmxw<true>(q, b, groups);
#if defined(PEARL_W_SEPNOK)
  return;  // timing: no hashing at all
#endif
  const FoldArgs A = a;
#ifndef PEARL_W_EPIWG
#define PEARL_W_EPIWG 256
#endif
  const size_t n = (size_t)groups * 8u, wg = (PEARL_W_EPIWG);
#ifndef PEARL_W_EPISG
#define PEARL_W_EPISG 8
#endif
  auto hk = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(PEARL_W_EPISG)]] {
    const uint32_t r = (uint32_t)it.get_global_id(0);
    if (r >= n) return;
    uint32_t w[16];
#if defined(PEARL_W_SEPBLK)
    {
      const uint32_t row = r % A.rowsValid, col = r / A.rowsValid;
      const uint32_t win = row / 2u + (A.rowsValid / 2u) * (col / 4u), q = (col % 4u) * 2u + (row & 1u);
      for (int j = 0; j < 16; j++) w[j] = words[((size_t)win * 16u + (uint32_t)j) * 8u + q];
    }
#else
    for (int j = 0; j < 16; j++) w[j] = words[(size_t)r * 16u + (uint32_t)j];
#endif
    region_epilogue(w, r, A.test, A.hits, A.allTr);
  };
  const sycl::nd_range<1> nr((n + wg - 1) / wg * wg, wg);
  if (hq) {
    sycl::event e = hq->submit([&](sycl::handler &h) {
      h.depends_on(fe);
      h.parallel_for(nr, hk);
    });
    if (hev) *hev = e;
  } else {
    q.parallel_for(nr, hk);
  }
}
#endif

#if defined(PEARL_SYCL_ESIMD) && defined(PEARL_SYCL_XMX_JIT)
// ---------------------------------------------------------------------------
// xmx8w written in ESIMD (goal-arc9 experiment, built only with -DPEARL_SYCL_ESIMD). The same work-group (8
// windows, 16 threads), ring, copy, split DPAS and words as FoldXmxW at KPS 4, but each thread's program is
// written out: IGC's SYCL path reorders loads next to their uses and reuses registers, so A' could not be
// loaded a k-step ahead (PEARL_E_AH here). In ESIMD a work-item is a hardware thread (8 lanes), a simd<int, 64>
// is a DPAS operand as it lies in registers, and a 128-byte block read is simd<uint32_t, 32>.
// The words go to a scratch buffer (16 a region) and a plain kernel runs the epilogue.
//
// Run with PEARL_SYCL_ESIMD=1 (card 5, 12 November, against xmx8w's 90.7 TH/s): it passes the fold check but is
// slower. 77.0 as first written (a `% 3` became a call to a division routine; the scratch epilogue costs 2.4%);
// 83.2 with the slot stepped; A' a k-step ahead (PEARL_E_AH) 86.1; with the finalizer's scheduler off
// (IGC_VCApiOptions="-vc-codegen -Xfinalizer -noschedule", not something a release can set) 87.8, the same work a
// clock as xmx8w but at 1,896 MHz against 1,955. DPAS row-first (PEARL_E_ROWFIRST) 85.2; an L1 prefetch of A'
// (PEARL_E_PF) 84.3 to 84.5. B' bound to fixed registers (PEARL_E_BREG) removes every dummy DPAS but fails the check.
// ---------------------------------------------------------------------------
#if defined(PEARL_E_BREG)
// goal-arc9: B' bound to registers from PEARL_E_BREG up. On this driver the finalizer puts a dummy DPAS (null
// accumulator) before a run of DPAS whose first B operand lies below r42 (seen in every dump so far).
ESIMD_PRIVATE ESIMD_REGISTER((PEARL_E_BREG) * 32) sycl::ext::intel::esimd::simd<int, 256> pearl_e_breg;
#endif
struct FoldXmxE {
  FoldArgs A;
  uint32_t *W;  // [regions][16]
  static constexpr uint32_t kRing = 3u * 8192u, kSlm = kRing + 8u * 512u;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{sycl::ext::intel::experimental::grf_size<256>};
  }

  // Lambdas inside an ESIMD kernel are not ESIMD functions to this compiler: helpers are static members.
  static SYCL_ESIMD_FUNCTION sycl::ext::intel::esimd::simd<uint32_t, 32> gload(uint64_t addr) {
    namespace es = sycl::ext::intel::esimd;
    constexpr auto gp = es::properties{es::cache_hint_L1<es::cache_hint::cached>,
                                       es::cache_hint_L2<es::cache_hint::cached>, es::alignment<16>};
    return es::block_load<uint32_t, 32>(reinterpret_cast<const uint32_t *>((uintptr_t)addr), gp);
  }
  // L1 prefetch of 512 bytes (two 256-byte messages, no data back; d64x64t does not compile).
  static SYCL_ESIMD_FUNCTION void pf512(uint64_t addr) {
    namespace es = sycl::ext::intel::esimd;
    constexpr auto pp = es::properties{es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
    es::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>((uintptr_t)addr), 0u, pp);
    es::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>((uintptr_t)addr), 256u, pp);
  }
  static SYCL_ESIMD_FUNCTION void arrive() {
    namespace es = sycl::ext::intel::esimd;
    namespace xes = sycl::ext::intel::experimental::esimd;
    es::fence<es::memory_kind::local>();
    xes::split_barrier<xes::split_barrier_action::signal>();
  }
  static SYCL_ESIMD_FUNCTION void wait() {
    namespace xes = sycl::ext::intel::experimental::esimd;
    xes::split_barrier<xes::split_barrier_action::wait>();
  }

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace es = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using es::simd;
    es::slm_init(kSlm);
    const uint32_t sgl = (uint32_t)it.get_local_id(0), sgi = sgl & 1u, widx = sgl >> 1;
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wrb, wcb;
    window_of((uint32_t)it.get_group(0), rowWins / 8u, colWins, A.band / 8u > 0u ? A.band / 8u : 1u, wrb, wcb);
    const uint32_t wr = wrb * 8u + widx, wc = wcb;
#if defined(PEARL_E_LO32)
    // 32-bit address sums, as FoldXmxW's lo32 (the host keeps each operand inside one 4 GB window).
    const uint64_t aHi = (uint64_t)(uintptr_t)A.Ap & 0xFFFFFFFF00000000ull, bHi = (uint64_t)(uintptr_t)A.Bp & 0xFFFFFFFF00000000ull;
    const uint32_t aLo = (uint32_t)(uintptr_t)A.Ap + 2u * wr * A.aStride + sgi * 128u;
    const uint32_t bLo = (uint32_t)(uintptr_t)A.Bp + (A.colStart / 4u + wcb) * A.bStride + sgl * 128u;
#define PEARL_E_ALOAD(off) gload(aHi | (uint64_t)(aLo + (off)))
#define PEARL_E_BLOAD(off) gload(bHi | (uint64_t)(bLo + (off)))
#else
    const int8_t *Ab = A.Ap + (size_t)(2u * wr) * A.aStride + sgi * 128u;
    const int8_t *Bc = A.Bp + (size_t)(A.colStart / 4u + wcb) * A.bStride + (size_t)sgl * 128u;
#define PEARL_E_ALOAD(off) gload((uint64_t)(uintptr_t)(Ab + (off)))
#define PEARL_E_BLOAD(off) gload((uint64_t)(uintptr_t)(Bc + (off)))
#endif
    const uint32_t aStride = A.aStride;
    const uint32_t mine = sgl * 128u;  // bytes: this thread's 128-byte piece of a k-step's B' in the ring
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++) es::slm_block_store<uint32_t, 32>(mine + k * 2048u, PEARL_E_BLOAD(k * 2048u));
    simd<uint32_t, 32> next = PEARL_E_BLOAD(4u * 2048u);
    simd<int, 64> c[16];
#pragma unroll
    for (int i = 0; i < 16; i++) c[i] = 0;
    simd<uint32_t, 128> words = 0;
#define PEARL_E_LOAD_A(KS, DST)                                                         _Pragma("unroll") for (uint32_t rb = 0; rb < 4u; rb++) {                                simd<uint32_t, 32> r = PEARL_E_ALOAD((KS) * 512u + (rb >> 1) * aStride + (rb & 1u) * 256u);     DST[rb] = r.template bit_cast_view<int>();                                            }
#if defined(PEARL_E_AH)
    simd<int, 32> an[4];
    PEARL_E_LOAD_A(0u, an)
#endif
    arrive();
    // The chunk's slot and the next one, stepped (a % 3 here is a call to a division routine in VC's code).
    uint32_t slot = 0u, nslot = 8192u;
    for (uint32_t ch = 0; ch < 16u; ch++) {
#pragma unroll
      for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t ks = ch * 4u + s;
        if (s == 0) wait();
#if defined(PEARL_E_PF)
        // goal-arc9: this thread's 16-row block of A' (both halves of the pair), PEARL_E_PF k-steps ahead, into L1.
        pf512((uint64_t)(uintptr_t)A.Ap + (uint64_t)(2u * wr + sgi) * aStride
              + (uint64_t)((ks + (PEARL_E_PF) < 64u ? ks + (PEARL_E_PF) : 63u) * 512u));
#endif
        {
          const uint32_t kn = ks + 5u < 64u ? ks + 5u : 63u;
          const simd<uint32_t, 32> v = PEARL_E_BLOAD(kn * 2048u);
          es::slm_block_store<uint32_t, 32>(nslot + s * 2048u + mine, next);
          next = v;
        }
        if (s == 3) arrive();
        simd<int, 64> b[4];
#pragma unroll
        for (uint32_t cb = 0; cb < 4u; cb++) {
#if defined(PEARL_E_B128)
          // two 128-byte local reads a fragment, as FoldXmxW (one 256-byte read cost it 3%, goal-arc8)
          b[cb].template select<32, 1>(0) = es::slm_block_load<int, 32>(slot + s * 2048u + (sgi * 4u + cb) * 256u);
          b[cb].template select<32, 1>(32) = es::slm_block_load<int, 32>(slot + s * 2048u + (sgi * 4u + cb) * 256u + 128u);
#elif defined(PEARL_E_BREG)
          pearl_e_breg.template select<64, 1>(cb * 64u) = es::slm_block_load<int, 64>(slot + s * 2048u + (sgi * 4u + cb) * 256u);
#else
          b[cb] = es::slm_block_load<int, 64>(slot + s * 2048u + (sgi * 4u + cb) * 256u);
#endif
        }
        simd<int, 32> a[4];
#if defined(PEARL_E_AH)
#pragma unroll
        for (int rb = 0; rb < 4; rb++) a[rb] = an[rb];
        PEARL_E_LOAD_A((ks + 1u < 64u ? ks + 1u : 63u), an)
#else
        PEARL_E_LOAD_A(ks, a)
#endif
#if defined(PEARL_E_ROWFIRST)
        // goal-arc9: one A' fragment's 4 DPAS back to back, so no two DPAS in a row read the same B (src1): the
        // finalizer's DPAS fused-EU read-suppression WA (Wa_18023229625) adds a dummy DPAS before a run whose
        // first two DPAS share src1 (IGC visa/LocalScheduler/SWSB_G4IR.cpp, needWA).
#pragma unroll
        for (int rb = 0; rb < 4; rb++) {
#pragma unroll
          for (int cb = 0; cb < 4; cb++) {
#else
#pragma unroll
        for (int cb = 0; cb < 4; cb++) {
#pragma unroll
          for (int rb = 0; rb < 4; rb++) {
#endif
            simd<int, 64> &acc = c[(rb >> 1) * 8 + (rb & 1) * 4 + cb];
#if defined(PEARL_E_BREG)
            const simd<int, 64> bb = pearl_e_breg.template select<64, 1>(cb * 64);
            acc = xmx::dpasw<8, 8, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, bb, a[rb]);
#else
            acc = xmx::dpasw<8, 8, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, b[cb], a[rb]);
#endif
          }
        }
      }
      // The chunk's word: all 16 accumulators XORed, rows 0-3 and 4-7 of each column, then lane pairs.
      simd<int, 64> t = ((c[0] ^ c[1]) ^ (c[2] ^ c[3])) ^ ((c[4] ^ c[5]) ^ (c[6] ^ c[7]));
      t ^= ((c[8] ^ c[9]) ^ (c[10] ^ c[11])) ^ ((c[12] ^ c[13]) ^ (c[14] ^ c[15]));
      simd<uint32_t, 64> tu = t.template bit_cast_view<uint32_t>();
      simd<uint32_t, 8> x0 = (tu.template select<8, 1>(0) ^ tu.template select<8, 1>(8)) ^ (tu.template select<8, 1>(16) ^ tu.template select<8, 1>(24));
      simd<uint32_t, 8> x1 = (tu.template select<8, 1>(32) ^ tu.template select<8, 1>(40)) ^ (tu.template select<8, 1>(48) ^ tu.template select<8, 1>(56));
      simd<uint32_t, 8> w;
      w.template select<4, 2>(0) = x0.template select<4, 2>(0) ^ x0.template select<4, 2>(1);
      w.template select<4, 2>(1) = x1.template select<4, 2>(0) ^ x1.template select<4, 2>(1);
      words.template select<8, 1>(ch * 8u) = w;
      slot = nslot;
      nslot = nslot == 16384u ? 0u : nslot + 8192u;
    }
    wait();
    es::barrier();
    const uint32_t pbase = kRing + widx * 512u;
    if (sgi == 1u) es::slm_block_store<uint32_t, 128>(pbase, words);
    es::barrier();
    if (sgi == 0u) {
      words ^= es::slm_block_load<uint32_t, 128>(pbase);
      // Region q of the window: row offset 4 (q & 1), column offset 2 (q >> 1); regions 2p and 2p + 1 are
      // neighbours in the batch, so one 128-byte store each pair.
#pragma unroll
      for (uint32_t p = 0; p < 4u; p++) {
        simd<uint32_t, 32> v;
        v.template select<16, 1>(0) = words.template select<16, 8>(2u * p);
        v.template select<16, 1>(16) = words.template select<16, 8>(2u * p + 1u);
        const uint32_t local = 2u * wr + A.rowsValid * (4u * wc + p);
        es::block_store<uint32_t, 32>(W + (size_t)local * 16u, v);
      }
    }
#undef PEARL_E_ALOAD
#undef PEARL_E_BLOAD
#undef PEARL_E_LOAD_A
  }
};

inline void fold_xmxe(sycl::queue &q, const FoldArgs &a, uint32_t groups, uint32_t *words) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / 8u) * 16u, 16u), FoldXmxE{a, words});
  });
#if !defined(PEARL_E_NOEPI)
  const FoldArgs A = a;
  q.parallel_for(sycl::range<1>((size_t)groups * 8u), [=](sycl::id<1> i) {
    const uint32_t r = (uint32_t)i[0];
    uint32_t w[16];
#if defined(PEARL_W_SEPBLK)
    {
      const uint32_t row = r % A.rowsValid, col = r / A.rowsValid;
      const uint32_t win = row / 2u + (A.rowsValid / 2u) * (col / 4u), q = (col % 4u) * 2u + (row & 1u);
      for (int j = 0; j < 16; j++) w[j] = words[((size_t)win * 16u + (uint32_t)j) * 8u + q];
    }
#else
    for (int j = 0; j < 16; j++) w[j] = words[(size_t)r * 16u + (uint32_t)j];
#endif
    region_epilogue(w, r, A.test, A.hits, A.allTr);
  });
#endif
}
#endif

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
