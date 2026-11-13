#!/usr/bin/env bash
# Build the SYCL Pearl core for Intel GPUs on Windows. Run in Git Bash with
# MSVC's environment set (a "x64 Native Tools" prompt, or CI's msvc-dev-cmd):
# the compiler uses MSVC's headers, libraries and linker.
#
#   DPCPP_ROOT=C:/path/to/sycl_windows ./build-win.sh [--stage DIR]
#
# DPCPP_ROOT is the unpacked open-source DPC++ release (github.com/intel/llvm,
# sycl_windows.tar.gz of v7.2.0; native-core.yml fetches it by SHA-256). It
# builds the form a release carries, as ./build.sh --jit does on Linux: SPIR-V
# the driver compiles at first run, a device image a kernel, the hardware DPAS
# where the card has XMX. Windows has no ocloc to prebuild kernels with.
#
# Writes, in ./build-win:
#   pearl_sycl.dll, pearl_sycl.lib   the host driver and every kernel
#                                    (pearl_sycl_host.cpp), its exports named in
#                                    pearl_sycl.def
#   pearl_sycl_check.exe             the fold check and bench tool
# The addon is built after it by node-gyp, with MSVC as the CUDA and AMD cores
# are (PEARL_BACKEND=sycl, ../binding.gyp), and linked against pearl_sycl.lib.
#
# --stage DIR copies what a release ships into DIR: ../build/Release/
# pearl_core.node as pearl_core_sycl.node (so run node-gyp first), pearl_sycl.dll,
# the check tool, the SYCL runtime DLLs they load and their licence. Windows
# looks for a DLL beside the one that asks for it, so the folder runs from
# anywhere.
#
# No Intel GPU has run this build: it is checked on the SYCL CPU device only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../src"
OUT="$HERE/build-win"
STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
: "${DPCPP_ROOT:?set DPCPP_ROOT to the unpacked sycl_windows release}"
ROOT=$(cygpath -u "$DPCPP_ROOT")
CXX="$ROOT/bin/clang++.exe"
[ -x "$CXX" ] || { echo "no $CXX" >&2; exit 1; }
mkdir -p "$OUT"

if [ -z "$STAGE" ] || [ ! -f "$OUT/pearl_sycl.dll" ]; then
  # The dynamic C++ runtime (as /MD): sycl9.dll is built with it, and its headers refuse another.
  CRT=(-fms-runtime-lib=dll)
  SYCL_FLAGS=(-fsycl -O3 -std=c++17 "${CRT[@]}" -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -DPEARL_SYCL_XMX_JIT)
  # The ring fold's split barrier (pearl_sycl_kernels.hpp) is a SPIR-V extension the translator leaves off.
  SYCL_FLAGS+=(-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier)
  echo "[1/2] pearl_sycl.dll (spir64, compiled by the driver)"
  "$CXX" "${SYCL_FLAGS[@]}" -shared -I"$HERE" -I"$SRC" "$HERE/pearl_sycl_host.cpp" -o "$OUT/pearl_sycl.dll" \
    "-Wl,/DEF:$(cygpath -m "$HERE/pearl_sycl.def")" "-Wl,/IMPLIB:$(cygpath -m "$OUT/pearl_sycl.lib")"
  echo "[2/2] pearl_sycl_check.exe"
  # -fsycl here only so that the driver links the same C++ runtime as the DLL has.
  "$CXX" -fsycl -O2 -std=c++17 "${CRT[@]}" -I"$HERE" -I"$SRC" "$HERE/pearl_sycl_check.cpp" "$OUT/pearl_sycl.lib" -o "$OUT/pearl_sycl_check.exe"
  echo "done: $OUT"
fi

if [ -n "$STAGE" ]; then
  ADDON="$HERE/../build/Release/pearl_core.node"
  [ -f "$ADDON" ] || { echo "stage: no $ADDON: run PEARL_BACKEND=sycl npx node-gyp rebuild in earn/native first" >&2; exit 1; }
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp "$ADDON" "$STAGE/pearl_core_sycl.node"
  cp "$OUT/pearl_sycl.dll" "$OUT/pearl_sycl_check.exe" "$STAGE/"
  # The runtime: sycl9.dll, the two Level Zero adapters it opens by name, and
  # the proxy loader it uses to open them. OpenCL is built into sycl9.dll.
  RUNTIME=(sycl9.dll ur_adapter_level_zero.dll ur_adapter_level_zero_v2.dll ur_win_proxy_loader.dll)
  for f in "${RUNTIME[@]}"; do
    [ -f "$ROOT/bin/$f" ] || { echo "stage: the DPC++ release has no bin/$f" >&2; exit 1; }
    cp "$ROOT/bin/$f" "$STAGE/$f"
  done
  cp "$HERE/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"
  cp "$HERE/licenses/THIRD-PARTY-DPCPP.txt" "$STAGE/THIRD-PARTY-DPCPP.txt"
  {
    echo "The Intel (SYCL) Pearl core of LLMJob Earn for Windows, and the libraries it loads."
    echo
    echo "Ours (MIT, as the rest of LLMJob Earn): pearl_core_sycl.node, pearl_sycl.dll, pearl_sycl_check.exe."
    echo
    echo "The SYCL runtime, unchanged, from the oneAPI DPC++ compiler's open-source release"
    echo "(https://github.com/intel/llvm, $("$CXX" --version | head -n 1)):"
    for f in "${RUNTIME[@]}"; do echo "  $f"; done
    echo "Apache License 2.0 with LLVM exceptions: LICENSE-DPCPP.txt. The ur_ files are the Unified"
    echo "Runtime, built from that repository under the same licence. THIRD-PARTY-DPCPP.txt lists"
    echo "the software built into them with each one's licence."
    echo
    echo "Not here, and needed on the PC: Intel's graphics driver, and Microsoft's Visual C++ runtime"
    echo "(msvcp140.dll, vcruntime140.dll), which Windows or another program has usually installed."
  } > "$STAGE/NOTICE.txt"
  echo "staged in $STAGE:"
  ( cd "$STAGE" && for f in *; do printf '  %9d  %s  %s\n' "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)" "$f"; done )
fi
