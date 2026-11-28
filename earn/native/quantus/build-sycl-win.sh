#!/usr/bin/env bash
# Build the Intel (SYCL) Quantus core for Windows, quantus_sycl.node. Run in Git
# Bash with MSVC's environment set (a "x64 Native Tools" prompt, or CI's
# msvc-dev-cmd), as ../sycl/build-win.sh builds the Pearl Intel core.
#
#   DPCPP_ROOT=C:/path/to/sycl_windows ./build-sycl-win.sh [--out DIR] [--stage DIR]
#
# sycl/build.sh is Linux only, and this is its Windows half, kept beside the
# sycl/ folder rather than in it. It compiles sycl/quantus_sycl.cpp the way
# build.sh --jit does: SPIR-V that the card's driver compiles at first run, one
# device image a kernel, with the open-source DPC++ release (DPCPP_ROOT: the
# unpacked sycl_windows.tar.gz of v7.2.0, which native-core.yml fetches by
# SHA-256). Windows has no ocloc, so there is no prebuilt (AOT) form.
# QTC_SYCL_EXTRA adds compiler flags, as on Linux.
#
# The addon is plain N-API. It links node.lib from node-gyp's cache (this runs
# `npx node-gyp install` when it is not there; NODE_GYP_DIR names another
# cache), delay-loads node.exe, and links win_delay_load_hook.cc, which answers
# that load with the running program. So it loads in node.exe, in the Electron
# app and in a Node with another name, as an addon node-gyp builds does.
#
# Writes DIR/quantus_sycl.node (default ./build). --stage DIR then copies what a
# release ships into DIR: the addon, the SYCL runtime DLLs it loads (the four
# the Pearl Intel core stages), their licence and a NOTICE.txt. Windows looks
# for a DLL beside the one that asks for it, so the folder loads from anywhere.
# A package puts these files in the Pearl Intel core's sycl/ folder, which holds
# the same runtime (earn/scripts/stage-quantus-sycl.mjs), and
# src/main/quantusCore.js looks for the addon there.
#
# No Intel GPU on Windows has run this build. CI checks it on the SYCL CPU device.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/sycl"
OUT="$HERE/build"
STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
: "${DPCPP_ROOT:?set DPCPP_ROOT to the unpacked sycl_windows release}"
ROOT=$(cygpath -u "$DPCPP_ROOT")
CXX="$ROOT/bin/clang++.exe"
[ -x "$CXX" ] || { echo "no $CXX" >&2; exit 1; }
# The compiler is a Windows program: every path it is given is a Windows one.
win() { cygpath -m "$1"; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# Node's headers and node.lib, for the Node that runs this script.
VER=$(node -p 'process.versions.node')
DEVDIR=$(cygpath -u "${NODE_GYP_DIR:-${LOCALAPPDATA:?}/node-gyp/Cache}")
GYP="$DEVDIR/$VER"
if [ ! -f "$GYP/include/node/node_api.h" ] || [ ! -f "$GYP/x64/node.lib" ]; then
  npx --yes node-gyp install "--devdir=$(win "$DEVDIR")" >/dev/null
fi
[ -f "$GYP/include/node/node_api.h" ] && [ -f "$GYP/x64/node.lib" ] \
  || { echo "no Node $VER headers and node.lib in $GYP: run npx node-gyp install" >&2; exit 1; }

# The dynamic C++ runtime (as /MD): sycl9.dll is built with it, and its headers refuse another.
CRT=(-fms-runtime-lib=dll)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
echo "[1/2] the delay-load hook"
"$CXX" -c -O2 -std=c++17 "${CRT[@]}" "$(win "$HERE/win_delay_load_hook.cc")" -o "$(win "$T/win_delay_load_hook.o")"

SYCL_FLAGS=(-fsycl -O3 -std=c++17 "${CRT[@]}" -Wno-psabi -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel)
if [ -n "${QTC_SYCL_EXTRA:-}" ]; then read -ra EXTRA <<< "$QTC_SYCL_EXTRA"; SYCL_FLAGS+=("${EXTRA[@]}"); fi
echo "[2/2] quantus_sycl.node (spir64, compiled by the driver)"
"$CXX" "${SYCL_FLAGS[@]}" -shared -I"$(win "$SRC")" -I"$(win "$GYP/include/node")" \
  "$(win "$SRC/quantus_sycl.cpp")" "$(win "$T/win_delay_load_hook.o")" -o "$(win "$OUT/quantus_sycl.node")" \
  "-Wl,$(win "$GYP/x64/node.lib")" -Wl,delayimp.lib -Wl,/DELAYLOAD:node.exe
# The import library and exports file the linker writes beside a DLL. Nothing links against the addon.
rm -f "$OUT/quantus_sycl.lib" "$OUT/quantus_sycl.exp"
echo "done: $OUT/quantus_sycl.node"

if [ -n "$STAGE" ]; then
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp "$OUT/quantus_sycl.node" "$STAGE/quantus_sycl.node"
  # The runtime: sycl9.dll, the two Level Zero adapters it opens by name, and the
  # proxy loader it opens them with. OpenCL is built into sycl9.dll.
  RUNTIME=(sycl9.dll ur_adapter_level_zero.dll ur_adapter_level_zero_v2.dll ur_win_proxy_loader.dll)
  for f in "${RUNTIME[@]}"; do
    [ -f "$ROOT/bin/$f" ] || { echo "stage: the DPC++ release has no bin/$f" >&2; exit 1; }
    cp "$ROOT/bin/$f" "$STAGE/$f"
  done
  cp "$HERE/../sycl/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"
  cp "$HERE/../sycl/licenses/THIRD-PARTY-DPCPP.txt" "$STAGE/THIRD-PARTY-DPCPP.txt"
  {
    echo "The Intel (SYCL) Quantus core of LLMJob Earn for Windows, and the libraries it loads."
    echo
    echo "Ours (MIT, as the rest of LLMJob Earn): quantus_sycl.node."
    echo
    echo "The SYCL runtime, unchanged, from the oneAPI DPC++ compiler's open-source release"
    echo "(https://github.com/intel/llvm, $("$CXX" --version | head -n 1)):"
    for f in "${RUNTIME[@]}"; do echo "  $f"; done
    echo "Apache License 2.0 with LLVM exceptions: LICENSE-DPCPP.txt. The ur_ files are the Unified"
    echo "Runtime, built from that repository under the same licence. THIRD-PARTY-DPCPP.txt lists"
    echo "the software built into them with each one's licence. These are the same files the Pearl"
    echo "Intel core ships in its sycl folder."
    echo
    echo "Not here, and needed on the PC: Intel's graphics driver, and Microsoft's Visual C++ runtime"
    echo "(msvcp140.dll, vcruntime140.dll), which Windows or another program has usually installed."
  } > "$STAGE/NOTICE.txt"
  echo "staged in $STAGE:"
  ( cd "$STAGE" && for f in *; do printf '  %9d  %s  %s\n' "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)" "$f"; done )
fi
