#!/usr/bin/env bash
# The static checks for readout v2 (PEARL_AMD_READOUT_V2, pearl_amd_fold.h; round 4 item 4), from
# the compiler's own output, before any GPU time. For each target it compiles the device code
# these ways:
#   base   today's readout
#   v2     -DPEARL_AMD_READOUT_V2=1: one ds_xor_b64 a block a chunk
#   v2w    -DPEARL_AMD_READOUT_V2=2: two ds_xor_b32 instead
#   drop   -DPEARL_AMD_READOUT_DROP=1, gfx12 only: the timing probe, with no cross-lane steps and
#          wrong words, for reference
# Then readout-census.js cuts the target's RDNA fold out of each (pearl_amd_fold_wmma12 on gfx12,
# pearl_amd_fold_wmma11 on gfx11), counts its instructions and prints one table a target with
# the plan's gates (registers, scratch, LDS, lane moves and selects, the LDS XORs, the slab
# loop's VALU and DS waits against base). Exits non-zero on any miss.
#
#   test/readout-census.sh [gfx...]        (default: gfx1100 gfx1200 gfx1201)
#
# Needs ROCm's hip-clang (ROCM_PATH, default /opt/rocm) and node. PEARL_AMD_CFLAGS goes into every
# build, but each build sets the readout switches itself. Trust only ROCm 7.2.0's numbers, the
# compiler the release is built with: register counts move between versions. CI's AMD job runs
# this (native-core.yml). The assembly stays in build/census for a closer look.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD="$HERE/.."
SRC="$AMD/../src"
OUT="$AMD/build/census"
GEN="$AMD/build/gen"
targets="${*:-gfx1100 gfx1200 gfx1201}"
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
  builds="base v2 v2w"
  case $a in gfx12*) builds="$builds drop" ;; esac
  for b in $builds; do
    case $b in
      base) flags="" ;;
      v2) flags="-DPEARL_AMD_READOUT_V2=1" ;;
      v2w) flags="-DPEARL_AMD_READOUT_V2=2" ;;
      drop) flags="-DPEARL_AMD_READOUT_DROP=1" ;;
    esac
    if ! "$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 -O3 ${PEARL_AMD_CFLAGS:-} \
        -UPEARL_AMD_READOUT_DROP -UPEARL_AMD_READOUT_V2 $flags $inc \
        --cuda-device-only -S "$AMD/src/pearl_host_hip.cpp" -o "$OUT/$a-$b.s" \
        -Rpass-analysis=kernel-resource-usage > "$OUT/$a-$b.txt" 2>&1; then
      echo "$a $b: compile failed, see $OUT/$a-$b.txt" >&2
      bad=1
    fi
  done
done
[ "$bad" -eq 0 ] || exit 1
node "$HERE/readout-census.js" "$OUT" $targets
