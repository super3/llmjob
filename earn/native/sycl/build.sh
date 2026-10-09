#!/usr/bin/env bash
# Build the SYCL Pearl core for Intel GPUs.
#
#   ./build.sh                 AOT for every Intel GPU target below plus the CPU
#                              device, with the hardware XMX folds
#   ./build.sh --targets LIST  AOT for LIST only (comma separated, e.g.
#                              intel_gpu_bmg_g21,spir64_x86_64)
#   ./build.sh --jit           one SPIR-V image, compiled by the driver at first
#                              run: any Intel GPU and the CPU device, but the
#                              XMX folds run the reference code (no hardware DPAS)
#   ./build.sh --dump DIR      also write IGC's assembly for every GPU kernel to
#                              DIR (dpas count, GRF mode, spills)
#   ./build.sh --large-grf     256-register mode on the GPU targets
#   ./build.sh --out DIR       output directory (default ./build)
#
# Writes, in the output directory:
#   libpearl_sycl.so       the host driver and every kernel (pearl_sycl_host.cpp)
#   pearl_core_sycl.node   the N-API addon, ../src/pearl_core.cc unchanged, linked
#                          against libpearl_sycl.so (found beside it)
#   pearl_sycl_check       the fold check and bench tool (pearl_sycl_check.cpp)
#
# Needs the oneAPI DPC++ compiler (icpx) on PATH, or CXX set to it, and Node 18+
# for the addon headers. install-toolchain.sh installs a compiler if there is
# none. Nothing here touches the CUDA build.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../src"
OUT="$HERE/build"
TARGETS="intel_gpu_bmg_g21,intel_gpu_bmg_g31,intel_gpu_lnl_m,intel_gpu_ptl_h,intel_gpu_pvc,intel_gpu_acm_g10,intel_gpu_acm_g11,intel_gpu_acm_g12,spir64_x86_64"
JIT=0
DUMP=""
LARGE_GRF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    --jit) JIT=1; shift ;;
    --dump) DUMP="$2"; shift 2 ;;
    --large-grf) LARGE_GRF=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

CXX=${CXX:-icpx}
command -v "$CXX" >/dev/null || { echo "no $CXX on PATH: source oneAPI's setvars.sh, or run install-toolchain.sh and source its envrc.sh" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# Build against the system's libstdc++, not one a compiler package brought:
# Node (or Electron) has already loaded the system one when it loads the addon.
GCC_FLAGS=()
if [ -z "${PEARL_NO_SYSTEM_GCC:-}" ] && command -v g++ >/dev/null; then
  GCC_DIR=$(dirname "$(g++ -print-libgcc-file-name)")
  [ -d "$GCC_DIR" ] && GCC_FLAGS=(--gcc-install-dir="$GCC_DIR")
fi

SYCL_FLAGS=(-fsycl -O3 -fPIC -std=c++17 -Wno-psabi)
if [ "$JIT" = 1 ]; then
  SYCL_FLAGS+=(-fsycl-targets=spir64)
else
  SYCL_FLAGS+=(-fsycl-targets="$TARGETS" -DPEARL_SYCL_XMX_HW)
  BACKEND=""
  [ "$LARGE_GRF" = 1 ] && BACKEND="-options -ze-opt-large-register-file"
  IFS=',' read -ra TL <<< "$TARGETS"
  for t in "${TL[@]}"; do
    case "$t" in
      intel_gpu_*)
        SYCL_FLAGS+=("-Xspirv-translator=$t" "-spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate")
        [ -n "$BACKEND" ] && SYCL_FLAGS+=("-Xsycl-target-backend=$t" "$BACKEND")
        ;;
    esac
  done
fi

# Node's headers: beside the node binary (official builds), else node-gyp's cache.
NODE_VER=$(node -p 'process.versions.node')
NODE_INC=""
for d in "$(dirname "$(command -v node)")/../include/node" "$HOME/.cache/node-gyp/$NODE_VER/include/node"; do
  [ -f "$d/node_api.h" ] && NODE_INC="$d" && break
done
if [ -z "$NODE_INC" ]; then
  npx --yes node-gyp install >/dev/null
  NODE_INC="$HOME/.cache/node-gyp/$NODE_VER/include/node"
fi
# node-addon-api: wherever require finds it from here, else a private install.
NAPI_INC=$(cd "$HERE" && node -p "require('node-addon-api').include_dir" 2>/dev/null || true)
if [ -z "$NAPI_INC" ] || [ ! -f "$NAPI_INC/napi.h" ]; then
  npm install --no-save --no-audit --no-fund --prefix "$OUT/npm" node-addon-api@8 >/dev/null
  NAPI_INC="$OUT/npm/node_modules/node-addon-api"
fi

# Run paths: the output directory, then the compiler's own lib directory, so the
# addon finds libsycl and the Intel runtime libraries without LD_LIBRARY_PATH.
RPATH=(-Wl,-rpath,'$ORIGIN')
SYCL_LIB=$(dirname "$("$CXX" -fsycl -print-file-name=libsycl.so)")
[ -f "$SYCL_LIB/libsycl.so" ] && RPATH+=(-Wl,-rpath,"$(cd "$SYCL_LIB" && pwd)")

DUMP_ENV=()
if [ -n "$DUMP" ]; then
  mkdir -p "$DUMP"
  DUMP_ENV=(env IGC_ShaderDumpEnable=1 "IGC_DumpToCustomDir=$(cd "$DUMP" && pwd)")
fi

echo "[1/3] libpearl_sycl.so ($([ "$JIT" = 1 ] && echo spir64 JIT || echo "AOT: $TARGETS"))"
"${DUMP_ENV[@]}" "$CXX" "${SYCL_FLAGS[@]}" "${GCC_FLAGS[@]}" -shared -I"$HERE" -I"$SRC" \
  "$HERE/pearl_sycl_host.cpp" -o "$OUT/libpearl_sycl.so" "${RPATH[@]}"

echo "[2/3] pearl_core_sycl.node"
"$CXX" -O2 -fPIC -std=c++17 "${GCC_FLAGS[@]}" -DNAPI_DISABLE_CPP_EXCEPTIONS -DNAPI_VERSION=8 \
  -I"$NODE_INC" -I"$NAPI_INC" -I"$SRC" -c "$SRC/pearl_core.cc" -o "$OUT/pearl_core.o"
"$CXX" -fsycl "${GCC_FLAGS[@]}" -shared "$OUT/pearl_core.o" -L"$OUT" -lpearl_sycl \
  -o "$OUT/pearl_core_sycl.node" "${RPATH[@]}"

echo "[3/3] pearl_sycl_check"
"$CXX" -fsycl -O2 -std=c++17 "${GCC_FLAGS[@]}" -I"$HERE" -I"$SRC" "$HERE/pearl_sycl_check.cpp" \
  -L"$OUT" -lpearl_sycl -o "$OUT/pearl_sycl_check" "${RPATH[@]}" -lpthread
rm -f "$OUT/pearl_core.o"
echo "done: $OUT"
