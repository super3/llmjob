// The RTX 20 folds built a second time, with ptxas's register-usage level 8 (PEARL_T75_RUL8 in
// pearl_config.h). Compiled for sm_75 alone:
//
//   nvcc ... -gencode arch=compute_75,code=sm_75 -Xptxas --register-usage-level=8 \
//     -c src/pearl_kernel_t75.cu -o cuda-build/pearl_kernel_t75.o
//
// and pearl_host.cu with -DPEARL_T75_RUL8=1, which makes an RTX 20 card launch these two folds
// in place of pearl_tile_fold_wmma / pearl_tile_fold_wmma_rg.
//
// Why a second file. nvcc hands its -Xptxas options to every architecture of a call, and CI
// builds sm_61 to sm_120 in one call, so the flag in that call would rebuild every other
// card's kernels too. This file is pearl_kernel.cu itself, included whole, with every name
// that has linkage renamed below, so the two objects link side by side and pearl_kernel.o
// (every architecture's machine code, sm_75's included) is byte for byte what it was.
// Level 8 on sm_75 measured +0.3 to +0.5% on an RTX 2080 Ti, a 2070 and a 2060 (goal/nv-20c;
// see earn/tuning/benchmark.md).
//
// Every other kernel here is built too and never launched. A kernel or a __device__ variable
// added to pearl_kernel.cu without a line below fails the link with a duplicate symbol.

#define pearl_blake3_chunk_cvs pearl_t75_blake3_chunk_cvs
#define pearl_blake3_parent_layer pearl_t75_blake3_parent_layer
#define pearl_blake3_unkeyed pearl_t75_blake3_unkeyed
#define pearl_bind_root pearl_t75_bind_root
#define pearl_gen_dense pearl_t75_gen_dense
#define pearl_gen_perm pearl_t75_gen_perm
#define pearl_gen_operand pearl_t75_gen_operand
#define pearl_hash_operands pearl_t75_hash_operands
#define pearl_materialize pearl_t75_materialize
#define pearl_materialize16 pearl_t75_materialize16
#define pearl_materialize16_kblocked pearl_t75_materialize16_kblocked
#define pearl_noise_materialize_kblocked pearl_t75_noise_materialize_kblocked
#define pearl_materialize16_tiled pearl_t75_materialize16_tiled
#define pearl_materialize16_frag pearl_t75_materialize16_frag
#define pearl_materialize16_afrag pearl_t75_materialize16_afrag
#define pearl_restamp_operand pearl_t75_restamp_operand
#define pearl_restamp_commit pearl_t75_restamp_commit
#define pearl_tile_fold_wmma pearl_t75_tile_fold_wmma
#define pearl_tile_fold_wmma_rg pearl_t75_tile_fold_wmma_rg
#define pearl_tile_fold_tall pearl_t75_tile_fold_tall
#define pearl_tile_fold_tall_pinb pearl_t75_tile_fold_tall_pinb
#define pearl_tile_fold_hopper pearl_t75_tile_fold_hopper
#define pearl_tile_fold_dp4a pearl_t75_tile_fold_dp4a
#define pearl_tile_fold_hmma pearl_t75_tile_fold_hmma
#define pearl_tall_hash80 pearl_t75_tall_hash80
#define pearl_tall_cyc pearl_t75_tall_cyc
#define pearl_tall_hcyc pearl_t75_tall_hcyc
#define pearl_tall_cyc_take pearl_t75_tall_cyc_take
#define pearl_tall_hcyc_take pearl_t75_tall_hcyc_take
#define pearl_e4_flag pearl_t75_e4_flag
#define BLAKE3_IV PEARL_T75_BLAKE3_IV
#define blake3_chunk_cv pearl_t75_blake3_chunk_cv

#include "pearl_kernel.cu"
