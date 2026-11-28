#!/usr/bin/env bash
# Build the NEB core for AMD GPUs on Linux: earn/native/neb/build/neb_hip.node.
#
#   ./build.sh [gfx...]    the addon, for the listed targets (default: RDNA 2, 3 and 4)
#
# ROCm 7 (hip-clang) and Node 22, as ../../quantus/amd/build.sh. The addon links libamdhip64.so.7, looked for in
# /opt/rocm/lib and then in rocm/ beside the addon (the HiveOS package's).
#
# Environment: ROCM_PATH (default /opt/rocm), NODE_INCLUDE (default: the running node's include/node), OUT (default
# ../build/neb_hip.node), NH_CFLAGS (extra compiler flags, for tuning variants).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEB="$(dirname "$HERE")"
OUT="${OUT:-$NEB/build/neb_hip.node}"
ARCHS="${*:-gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201}"
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || { echo "build.sh: no hip-clang at $cc: set ROCM_PATH" >&2; exit 1; }
inc="${NODE_INCLUDE:-$(node -p "require('path').join(process.execPath, '..', '..', 'include', 'node')")}"
[ -f "$inc/node_api.h" ] || { echo "build.sh: Node headers not found at $inc: set NODE_INCLUDE" >&2; exit 1; }
offload=""
for a in $ARCHS; do offload="$offload --offload-arch=$a"; done
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
"$cc" -x hip $offload --rocm-path="$rocm" -std=c++17 -O3 -fPIC -Wno-unused-result ${NH_CFLAGS:-} -I"$NEB" -c "$HERE/neb_hip.cpp" -o "$T/neb_hip.o"
for f in nebhash2 shares blake3_min chacha8; do
  "$rocm/lib/llvm/bin/clang" -O3 -fPIC -I"$NEB/ref" -c "$NEB/ref/$f.c" -o "$T/$f.o"
done
"$rocm/lib/llvm/bin/clang" -O2 -fPIC -I"$inc" -I"$NEB" -c "$NEB/neb_napi.c" -o "$T/neb_napi.o"
"$cc" -shared -fPIC -o "$OUT" "$T/neb_hip.o" "$T/neb_napi.o" "$T/nebhash2.o" "$T/shares.o" "$T/blake3_min.o" "$T/chacha8.o" \
  -L"$rocm/lib" -lamdhip64 -lpthread -Wl,-rpath,/opt/rocm/lib -Wl,-rpath,'$ORIGIN/rocm' -Wl,--allow-shlib-undefined
echo "built $OUT for: $ARCHS"
