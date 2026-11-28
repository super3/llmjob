#!/usr/bin/env bash
# Build the NEB core for AMD GPUs: earn/native/neb/build/neb_hip.node.
#
#   ./build.sh [gfx...]    the addon, for the listed targets (default: RDNA 2, 3 and 4; on Windows also the
#                          built-in graphics of Ryzen processors)
#
# Linux: ROCm 7 (hip-clang) and Node 22, as ../../quantus/amd/build.sh. The addon links libamdhip64.so.7, looked for
# in /opt/rocm/lib and then in rocm/ beside the addon (the HiveOS package's).
#
# Windows: Git Bash with MSVC's environment (vcvars64, or CI's msvc-dev-cmd) and AMD's HIP SDK 7.2, found through
# HIP_PATH, as ../../quantus/amd/build.sh win builds the Quantus core. The addon is plain N-API: node.lib from
# node-gyp's cache, node.exe delay-loaded and answered by ../../quantus/win_delay_load_hook.cc, so it loads in the
# Electron app too. It uses the static C runtime and needs only amdhip64_7.dll, which comes with AMD's graphics
# driver. No Windows Radeon has run this build.
#
# Environment: ROCM_PATH (default /opt/rocm), HIP_PATH (Windows: the HIP SDK's folder), NODE_INCLUDE (default: the
# running node's include/node; on Windows node-gyp's cache), NODE_GYP_DIR (Windows: node-gyp's cache, default
# %LOCALAPPDATA%\node-gyp\Cache), OUT (default ../build/neb_hip.node), NH_CFLAGS (extra compiler flags, for tuning
# variants).
set -euo pipefail

windows() {
  case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) return 0 ;; *) return 1 ;; esac
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The compiler is a Windows program there: give it C:/... paths, which Git Bash takes too.
if windows; then HERE="$(cygpath -m "$HERE")"; fi
NEB="$(dirname "$HERE")"
OUT="${OUT:-$NEB/build/neb_hip.node}"
DEFAULT_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"
# Windows: the same, and the built-in graphics of Ryzen processors (gfx1035, gfx1036, gfx1103), as the Quantus AMD
# core's WINDOWS_ARCHS. AMD's Windows driver lists those as HIP cards, often as card 0, and HIP loads no code for
# the cards after one it has none for.
WINDOWS_ARCHS="gfx1030 gfx1031 gfx1032 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201"

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$(dirname "$OUT")"

if windows; then
  [ -n "${HIP_PATH:-}" ] || { echo "build.sh: HIP_PATH is not set: install AMD's HIP SDK, or set HIP_PATH to its folder" >&2; exit 1; }
  rocm="$(cygpath -m "${HIP_PATH%\\}")"
  rocm="${rocm%/}"
  cc="$rocm/bin/clang++.exe"
  c="$rocm/bin/clang.exe"
  [ -x "$cc" ] || { echo "build.sh: no hip-clang at $cc: set HIP_PATH" >&2; exit 1; }
  ARCHS="${*:-$WINDOWS_ARCHS}"
  W="$(cygpath -m "$T")"
  # Node's headers and node.lib, for the Node that runs this script, from node-gyp's cache.
  ver="$(node -p 'process.versions.node')"
  devdir="$(cygpath -m "${NODE_GYP_DIR:-${LOCALAPPDATA:?}/node-gyp/Cache}")"
  gyp="$devdir/$ver"
  if [ ! -f "$gyp/include/node/node_api.h" ] || [ ! -f "$gyp/x64/node.lib" ]; then
    npx --yes node-gyp install "--devdir=$devdir" > /dev/null
  fi
  [ -f "$gyp/include/node/node_api.h" ] && [ -f "$gyp/x64/node.lib" ] \
    || { echo "build.sh: no Node $ver headers and node.lib in $gyp: run npx node-gyp install" >&2; exit 1; }
  inc="$(cygpath -m "${NODE_INCLUDE:-$gyp/include/node}")"
  offload=""
  for a in $ARCHS; do offload="$offload --offload-arch=$a"; done
  # No -fPIC, which is not a Windows flag. The static C runtime, so the PC needs no Visual C++ runtime for it.
  rt="-fms-runtime-lib=static -D_CRT_SECURE_NO_WARNINGS"
  # shellcheck disable=SC2086
  "$cc" -x hip $offload --rocm-path="$rocm" -std=c++17 -O3 $rt -Wno-unused-result ${NH_CFLAGS:-} -I"$NEB" -c "$HERE/neb_hip.cpp" -o "$W/neb_hip.obj"
  for f in nebhash2 shares blake3_min chacha8; do
    # shellcheck disable=SC2086
    "$c" -O3 $rt -I"$NEB/ref" -c "$NEB/ref/$f.c" -o "$W/$f.obj"
  done
  # shellcheck disable=SC2086
  "$c" -O2 $rt -I"$inc" -I"$NEB" -c "$NEB/neb_napi.c" -o "$W/neb_napi.obj"
  "$cc" -O2 -std=c++17 -fms-runtime-lib=static -c "$NEB/../quantus/win_delay_load_hook.cc" -o "$W/win_delay_load_hook.obj"
  "$cc" -fms-runtime-lib=static -shared -o "$OUT" "$W/neb_hip.obj" "$W/neb_napi.obj" "$W/nebhash2.obj" "$W/shares.obj" \
    "$W/blake3_min.obj" "$W/chacha8.obj" "$W/win_delay_load_hook.obj" -L"$rocm/lib" -lamdhip64 "-Wl,$gyp/x64/node.lib" \
    -Wl,delayimp.lib -Wl,/DELAYLOAD:node.exe
  # The import library and exports file the linker writes beside a DLL. Nothing links against the addon.
  rm -f "${OUT%.node}.lib" "${OUT%.node}.exp"
  echo "built $OUT for: $ARCHS"
  exit 0
fi

ARCHS="${*:-$DEFAULT_ARCHS}"
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || { echo "build.sh: no hip-clang at $cc: set ROCM_PATH" >&2; exit 1; }
inc="${NODE_INCLUDE:-$(node -p "require('path').join(process.execPath, '..', '..', 'include', 'node')")}"
[ -f "$inc/node_api.h" ] || { echo "build.sh: Node headers not found at $inc: set NODE_INCLUDE" >&2; exit 1; }
offload=""
for a in $ARCHS; do offload="$offload --offload-arch=$a"; done
# shellcheck disable=SC2086
"$cc" -x hip $offload --rocm-path="$rocm" -std=c++17 -O3 -fPIC -Wno-unused-result ${NH_CFLAGS:-} -I"$NEB" -c "$HERE/neb_hip.cpp" -o "$T/neb_hip.o"
for f in nebhash2 shares blake3_min chacha8; do
  "$rocm/lib/llvm/bin/clang" -O3 -fPIC -I"$NEB/ref" -c "$NEB/ref/$f.c" -o "$T/$f.o"
done
"$rocm/lib/llvm/bin/clang" -O2 -fPIC -I"$inc" -I"$NEB" -c "$NEB/neb_napi.c" -o "$T/neb_napi.o"
"$cc" -shared -fPIC -o "$OUT" "$T/neb_hip.o" "$T/neb_napi.o" "$T/nebhash2.o" "$T/shares.o" "$T/blake3_min.o" "$T/chacha8.o" \
  -L"$rocm/lib" -lamdhip64 -lpthread -Wl,-rpath,/opt/rocm/lib -Wl,-rpath,'$ORIGIN/rocm' -Wl,--allow-shlib-undefined
echo "built $OUT for: $ARCHS"
