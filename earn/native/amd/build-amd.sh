#!/usr/bin/env bash
# Build the AMD Pearl core. Nothing here touches the CUDA build: it reads ../src and writes
# only to ./build.
#
#   ./build-amd.sh cpu              HIP-CPU: build/pearl_core_hipcpu.node and
#                                   build/pearl_amd_foldtest_cpu, which run the AMD kernels
#                                   on the CPU. For checking without a GPU; never shipped.
#   ./build-amd.sh gpu [gfx...]     ROCm: build/pearl_core_hip.node and build/pearl_amd_foldtest
#                                   for the listed targets (default: $DEFAULT_ARCHS below).
#   ./build-amd.sh check [gfx...]   Compile the device code for each target and print each
#                                   fold's registers, scratch and LDS. Fails if a fold
#                                   spills or uses scratch memory.
#
# Environment:
#   ROCM_PATH       ROCm install with hip-clang (default /opt/rocm). gpu and check.
#   HIP_CPU_PATH    a checkout of github.com/ROCm/HIP-CPU (its include/ is used). cpu.
#   TBB_ROOT        prefix holding include/tbb and lib/libtbb.so (default: the system's). cpu.
#   CPU_CXX         C++ compiler for the cpu build (default clang++; g++ ignores the vector
#                   types the folds use).
#   NODE_ADDON_API  node-addon-api's include folder (default: require('node-addon-api')).
#   NODE_INCLUDE    Node's headers (default: the running node's include/node).
#
# The portable kernels are pearl_kernel.cu's own: this script copies that file up to the
# start of the CUDA folds into build/gen/pearl_kernel_portable.cu, and stops if the line it
# cuts at is gone.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../src"
OUT="$HERE/build"
GEN="$OUT/gen"
DEFAULT_ARCHS="gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"
MARKER='^// A 16-byte global->shared copy that does not pass through registers'

mode="${1:-}"
shift || true
archs="${*:-$DEFAULT_ARCHS}"

slice() {
  mkdir -p "$GEN"
  local line
  line="$(grep -n "$MARKER" "$SRC/pearl_kernel.cu" | head -n 1 | cut -d: -f1 || true)"
  if [ -z "$line" ]; then
    echo "build-amd.sh: the cut marker is gone from pearl_kernel.cu; update MARKER to the line" >&2
    echo "  that starts the CUDA folds, so the AMD build takes everything before it." >&2
    exit 1
  fi
  head -n $((line - 1)) "$SRC/pearl_kernel.cu" > "$GEN/pearl_kernel_portable.cu"
}

node_includes() {
  local api="${NODE_ADDON_API:-}" inc="${NODE_INCLUDE:-}"
  [ -n "$api" ] || api="$(node -p "require('node-addon-api').include" 2>/dev/null | tr -d '"' || true)"
  [ -n "$inc" ] || inc="$(node -p "require('path').join(process.execPath, '..', '..', 'include', 'node')")"
  if [ -z "$api" ] || [ ! -f "$api/napi.h" ]; then
    echo "build-amd.sh: node-addon-api not found: run 'npm install --no-save node-addon-api' in" >&2
    echo "  earn/native/amd, or set NODE_ADDON_API to its include folder." >&2
    exit 1
  fi
  if [ ! -f "$inc/node_api.h" ]; then
    echo "build-amd.sh: Node headers not found at $inc: set NODE_INCLUDE." >&2
    exit 1
  fi
  echo "-I$api -I$inc"
}

hip_clang() {
  local rocm="${ROCM_PATH:-/opt/rocm}"
  local cc="$rocm/lib/llvm/bin/clang++"
  if [ ! -x "$cc" ]; then
    echo "build-amd.sh: no hip-clang at $cc: set ROCM_PATH to a ROCm install." >&2
    exit 1
  fi
  echo "$cc"
}

offload_flags() {
  local f=""
  for a in $archs; do f="$f --offload-arch=$a"; done
  echo "$f"
}

case "$mode" in
  cpu)
    slice
    : "${HIP_CPU_PATH:?set HIP_CPU_PATH to a HIP-CPU checkout}"
    cxx="${CPU_CXX:-clang++}"
    tbb_inc="" tbb_lib=""
    if [ -n "${TBB_ROOT:-}" ]; then
      tbb_inc="-I$TBB_ROOT/include"
      tbb_dir="$TBB_ROOT/lib"
      [ -f "$tbb_dir/libtbb.so" ] || tbb_dir="$TBB_ROOT/lib/x86_64-linux-gnu"
      tbb_lib="-L$tbb_dir -Wl,-rpath,$tbb_dir"
    fi
    mkdir -p "$OUT/cpu"
    inc="-I$HERE/compat/cpu -I$HERE/src -I$SRC -I$GEN -I$HIP_CPU_PATH/include $tbb_inc"
    flags="-std=c++17 -O2 -fPIC -DNDEBUG"
    echo "cpu: kernels and host"
    $cxx $flags $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/cpu/pearl_host_hip.o"
    echo "cpu: addon"
    $cxx $flags $(node_includes) -I"$SRC" -DNAPI_DISABLE_CPP_EXCEPTIONS -c "$SRC/pearl_core.cc" -o "$OUT/cpu/pearl_core.o"
    $cxx -shared -o "$OUT/pearl_core_hipcpu.node" "$OUT/cpu/pearl_host_hip.o" "$OUT/cpu/pearl_core.o" $tbb_lib -ltbb -lpthread
    echo "cpu: fold check"
    $cxx $flags $inc "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/pearl_amd_foldtest_cpu" $tbb_lib -ltbb -lpthread
    echo "built $OUT/pearl_core_hipcpu.node and $OUT/pearl_amd_foldtest_cpu"
    ;;
  gpu)
    slice
    cc="$(hip_clang)"
    rocm="${ROCM_PATH:-/opt/rocm}"
    mkdir -p "$OUT/gpu"
    inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"
    # ROCm marks every HIP call [[nodiscard]]. The host ignores the result of the ones that
    # cannot fail on their own, as pearl_host.cu does, and reads hipGetLastError where it
    # matters.
    flags="-std=c++17 -O3 -fPIC -Wno-unused-result -Wno-unused-value"
    echo "gpu: kernels and host for $archs"
    $cc -x hip $(offload_flags) --rocm-path="$rocm" $flags $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/gpu/pearl_host_hip.o"
    echo "gpu: addon"
    $cc $flags $(node_includes) -I"$SRC" -DNAPI_DISABLE_CPP_EXCEPTIONS -c "$SRC/pearl_core.cc" -o "$OUT/gpu/pearl_core.o"
    # --allow-shlib-undefined: libamdhip64 needs the rest of the ROCm runtime, which a build
    # box without a GPU may not have. The user's machine has it (README.md).
    $cc -shared -fPIC -o "$OUT/pearl_core_hip.node" "$OUT/gpu/pearl_host_hip.o" "$OUT/gpu/pearl_core.o" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,--allow-shlib-undefined
    echo "gpu: fold check"
    $cc -x hip $(offload_flags) --rocm-path="$rocm" $flags $inc -c "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/gpu/pearl_amd_foldtest.o"
    $cc -o "$OUT/pearl_amd_foldtest" "$OUT/gpu/pearl_amd_foldtest.o" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,--allow-shlib-undefined
    echo "built $OUT/pearl_core_hip.node and $OUT/pearl_amd_foldtest for: $archs"
    ;;
  check)
    slice
    cc="$(hip_clang)"
    rocm="${ROCM_PATH:-/opt/rocm}"
    mkdir -p "$OUT/check"
    inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"
    bad=0
    printf '%-9s %-22s %6s %6s %8s %7s %8s %10s\n' target kernel VGPRs AGPRs scratch spills LDS "waves/SIMD"
    for a in $archs; do
      log="$OUT/check/$a.txt"
      if ! $cc -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 -O3 $inc --cuda-device-only \
          -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/check/$a.o" -Rpass-analysis=kernel-resource-usage > "$log" 2>&1; then
        echo "$a: compile failed, see $log" >&2
        bad=1
        continue
      fi
      # One row a kernel: the fold kernels with a body on this target, the hash and the
      # reference fold. A fold without a body here shows 0 VGPRs and is left out.
      awk -v arch="$a" '
        /remark: Function Name: / { name = $(NF-1) }
        /remark:     VGPRs: / { v[name] = $(NF-1) }
        /remark:     AGPRs: / { g[name] = $(NF-1) }
        /remark:     ScratchSize/ { s[name] = $(NF-1) }
        /remark:     VGPRs Spill: / { sp[name] = $(NF-1) }
        /remark:     SGPRs Spill: / { ss[name] = $(NF-1) }
        /remark:     Occupancy/ { o[name] = $(NF-1) }
        /remark:     LDS Size/ { l[name] = $(NF-1) }
        END {
          for (n in v) {
            if (n !~ /^pearl_amd_(fold|hash)/ || v[n] == 0) continue
            printf "%-9s %-22s %6s %6s %8s %7s %8s %10s\n", arch, n, v[n], (n in g ? g[n] : "-"), s[n], sp[n] + ss[n], l[n], o[n]
            if (s[n] + 0 > 0 || sp[n] + ss[n] > 0) bad = 1
          }
          exit bad
        }' "$log" | sort || bad=1
    done
    if [ "$bad" -ne 0 ]; then
      echo "check: a fold failed to compile, spills, or uses scratch memory" >&2
      exit 1
    fi
    echo "check: every fold compiled for $archs with no spills and no scratch"
    ;;
  *)
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
