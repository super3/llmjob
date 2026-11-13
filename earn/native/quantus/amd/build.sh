#!/usr/bin/env bash
# Build the Quantus (QPoW) core for AMD GPUs: earn/native/quantus/build/quantus_hip.node.
#
#   ./build.sh gpu [gfx...]    the addon, for the listed targets (default: $DEFAULT_ARCHS)
#   ./build.sh check [gfx...]  compile the kernel for each target and print its registers, spills,
#                              scratch and occupancy; fails on a spill or scratch
#   ./build.sh asm gfx...      the kernel's ISA, one file a target, in build/amd/
#   ./build.sh peak [gfx...]   build/amd/qh_peak, the integer instruction-rate probe (timing only)
#
# Linux with ROCm 7 (hip-clang), Node 22 and node-addon-api, as ../../amd/build-amd.sh. The addon links
# libamdhip64.so.7, looked for in /opt/rocm/lib and then in rocm/ beside the addon (the HiveOS package's).
#
# Environment:
#   ROCM_PATH       ROCm with hip-clang (default /opt/rocm)
#   NODE_ADDON_API  node-addon-api's include folder (default: require('node-addon-api').include)
#   NODE_INCLUDE    Node's headers (default: the running node's include/node)
#   QH_CFLAGS       extra compiler flags, for tuning variants
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../build"
TMP="$OUT/amd"
# RDNA 2, 3 and 4 (Radeon RX 6000, 7000, 9000 and their PRO cards).
DEFAULT_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"

mode="${1:-}"
shift || true
archs="${*:-$DEFAULT_ARCHS}"
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || { echo "build.sh: no hip-clang at $cc: set ROCM_PATH" >&2; exit 1; }
flags="-std=c++17 -O3 -fPIC -Wno-unused-result ${QH_CFLAGS:-}"
offload=""
for a in $archs; do offload="$offload --offload-arch=$a"; done
mkdir -p "$TMP"

node_includes() {
  local api="${NODE_ADDON_API:-}" inc="${NODE_INCLUDE:-}"
  [ -n "$api" ] || api="$(node -p "require('node-addon-api').include" 2>/dev/null | tr -d '"' || true)"
  [ -n "$inc" ] || inc="$(node -p "require('path').join(process.execPath, '..', '..', 'include', 'node')")"
  [ -f "$api/napi.h" ] || { echo "build.sh: node-addon-api not found: npm install --no-save node-addon-api, or set NODE_ADDON_API" >&2; exit 1; }
  [ -f "$inc/node_api.h" ] || { echo "build.sh: Node headers not found at $inc: set NODE_INCLUDE" >&2; exit 1; }
  echo "-I$api -I$inc"
}

case "$mode" in
  gpu)
    targets="-DQH_TARGETS=\"$(echo $archs | tr ' ' ',')\""
    echo "gpu: kernel and host for $archs"
    "$cc" -x hip $offload --rocm-path="$rocm" $flags $targets -I"$HERE" -c "$HERE/qh_host.cpp" -o "$TMP/qh_host.o"
    echo "gpu: addon"
    "$cc" $flags $(node_includes) -I"$HERE" -DNAPI_DISABLE_CPP_EXCEPTIONS -DNAPI_VERSION=8 -c "$HERE/qh_napi.cc" -o "$TMP/qh_napi.o"
    "$cc" -shared -fPIC -o "$OUT/quantus_hip.node" "$TMP/qh_host.o" "$TMP/qh_napi.o" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,-rpath,'$ORIGIN/rocm' -Wl,--allow-shlib-undefined
    echo "built $OUT/quantus_hip.node for: $archs"
    ;;
  check)
    bad=0
    printf '%-9s %6s %6s %8s %7s %10s\n' target VGPRs SGPRs scratch spills waves/SIMD
    for a in $archs; do
      log="$TMP/check-$a.txt"
      "$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" $flags -I"$HERE" --cuda-device-only -c "$HERE/qh_host.cpp" \
        -o "$TMP/check-$a.o" -Rpass-analysis=kernel-resource-usage > "$log" 2>&1 || { echo "$a: compile failed, see $log" >&2; bad=1; continue; }
      awk -v arch="$a" '
        /remark: Function Name: qh_search/ { on = 1; next }
        /remark: Function Name: / { on = 0 }
        on && /remark:     VGPRs: / { v = $(NF-1) }
        on && /remark:     SGPRs: / { s = $(NF-1) }
        on && /remark:     ScratchSize/ { sc = $(NF-1) }
        on && /remark:     VGPRs Spill: / { sp += $(NF-1) }
        on && /remark:     SGPRs Spill: / { sp += $(NF-1) }
        on && /remark:     Occupancy/ { o = $(NF-1) }
        END { printf "%-9s %6s %6s %8s %7s %10s\n", arch, v, s, sc, sp, o; exit (sc + 0 > 0 || sp + 0 > 0) }' "$log" || bad=1
    done
    [ "$bad" -eq 0 ] || { echo "check: a target failed to compile, spills or uses scratch" >&2; exit 1; }
    ;;
  asm)
    for a in $archs; do
      "$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" $flags -I"$HERE" --cuda-device-only -S "$HERE/qh_host.cpp" \
        -o "$TMP/qh-$a.s"
      echo "wrote $TMP/qh-$a.s"
    done
    ;;
  peak)
    "$cc" -x hip $offload --rocm-path="$rocm" -std=c++17 -O3 "$HERE/test/qh_peak.cpp" -o "$TMP/qh_peak" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,--allow-shlib-undefined
    echo "built $TMP/qh_peak for: $archs"
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
