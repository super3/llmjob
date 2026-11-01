#!/usr/bin/env bash
# The static checks for two-lane 32x8 regions on RDNA 4 (PEARL_AMD_TWOLANE, pearl_amd_pattern.h),
# from the compiler's own output, before any GPU time. For each target it compiles the device
# code three ways:
#   base   today's readout
#   reg    -DPEARL_AMD_TWOLANE=1: the words kept in registers, stored at the end of the tile
#   store  -DPEARL_AMD_TWOLANE=1 -DPEARL_AMD_TWOLANE_STORE=1: a store each chunk
# Then readout-census.js cuts pearl_amd_fold_wmma12 out of each, counts its instructions and
# prints one table a target with the decision's gates (registers, scratch, LDS, DPP and lane
# moves, the slab loop's VALU against base, the longest v_xor3 chain, the keep step, the
# store's form in the slab loop). Exits non-zero on any miss.
#
#   test/readout-census.sh [gfx...]        (default: gfx1200 gfx1201)
#
# Needs ROCm's hip-clang (ROCM_PATH, default /opt/rocm) and node. PEARL_AMD_CFLAGS goes into all
# three builds. Trust only ROCm 7.2.0's numbers, the compiler the release is built with:
# register counts move between versions. CI's AMD job runs this (native-core.yml), after
# `PEARL_AMD_CFLAGS=-DPEARL_AMD_TWOLANE=1 ./build-amd.sh check gfx1200 gfx1201`. The assembly
# stays in build/census for a closer look.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD="$HERE/.."
SRC="$AMD/../src"
OUT="$AMD/build/census"
GEN="$AMD/build/gen"
targets="${*:-gfx1200 gfx1201}"
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || { echo "readout-census.sh: no hip-clang at $cc: set ROCM_PATH" >&2; exit 2; }
command -v node > /dev/null || { echo "readout-census.sh: needs node" >&2; exit 2; }

# The portable kernels, cut from pearl_kernel.cu at build-amd.sh's own marker, as its builds do.
marker="$(sed -n "s/^MARKER='\(.*\)'$/\1/p" "$AMD/build-amd.sh")"
line="$(grep -n "$marker" "$SRC/pearl_kernel.cu" | head -n 1 | cut -d: -f1 || true)"
if [ -z "$marker" ] || [ -z "$line" ]; then
  echo "readout-census.sh: can't find build-amd.sh's cut marker in pearl_kernel.cu" >&2
  exit 2
fi
mkdir -p "$GEN" "$OUT"
head -n $((line - 1)) "$SRC/pearl_kernel.cu" > "$GEN/pearl_kernel_portable.cu"

inc="-I$AMD/compat/gpu -I$AMD/src -I$SRC -I$GEN"
bad=0
for a in $targets; do
  for b in base reg store; do
    case $b in
      base) flags="" ;;
      reg) flags="-DPEARL_AMD_TWOLANE=1" ;;
      store) flags="-DPEARL_AMD_TWOLANE=1 -DPEARL_AMD_TWOLANE_STORE=1" ;;
    esac
    if ! "$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 -O3 ${PEARL_AMD_CFLAGS:-} $flags $inc \
        --cuda-device-only -S "$AMD/src/pearl_host_hip.cpp" -o "$OUT/$a-$b.s" \
        -Rpass-analysis=kernel-resource-usage > "$OUT/$a-$b.txt" 2>&1; then
      echo "$a $b: compile failed, see $OUT/$a-$b.txt" >&2
      bad=1
    fi
  done
done
[ "$bad" -eq 0 ] || exit 1
node "$HERE/readout-census.js" "$OUT" $targets
