#!/usr/bin/env bash
# Build the SYCL Pearl core for Intel GPUs.
#
#   ./build.sh --jit           what a release carries: SPIR-V the driver compiles
#                              at first run, a device image a kernel. Any Intel
#                              GPU and the CPU device, with the hardware DPAS
#                              where the card has XMX. Needs no ocloc
#   ./build.sh --jit --stage DIR
#                              and copy what a release ships into DIR: the addon,
#                              libpearl_sycl.so, the check tool, the SYCL runtime
#                              libraries they load, and those libraries' licence.
#                              Only with the open-source DPC++
#                              (install-toolchain.sh --open): Intel's oneAPI
#                              runtime is not ours to hand out
#   ./build.sh                 kernels prebuilt by ocloc (AOT) for every Intel GPU
#                              with XMX (TARGETS below) plus the CPU device. Runs
#                              on those devices only. What compile-gate.sh and
#                              --dump read IGC's assembly from
#   ./build.sh --targets LIST  AOT for LIST only (comma separated, e.g.
#                              intel_gpu_bmg_g21,spir64_x86_64): faster, and the
#                              way round a compiler or ocloc that does not know
#                              one of the default targets
#   ./build.sh --dump DIR      also write IGC's assembly for every GPU kernel to
#                              DIR (dpas count, GRF mode, spills). AOT only
#   ./build.sh --large-grf     256-register mode on the GPU targets. AOT only
#   ./build.sh --peak          also build pearl_sycl_peak, the XMX peak probe
#                              (timing only; README.md, "The A750's ceiling")
#   ./build.sh --out DIR       output directory (default ./build)
#
# Writes, in the output directory:
#   libpearl_sycl.so       the host driver and every kernel (pearl_sycl_host.cpp)
#   pearl_core_sycl.node   the N-API addon, ../src/pearl_core.cc unchanged, linked
#                          against libpearl_sycl.so (found beside it)
#   pearl_sycl_check       the fold check and bench tool (pearl_sycl_check.cpp)
#
# Needs a DPC++ compiler: CXX, else icpx (Intel's oneAPI), else a clang++ that
# takes -fsycl (the open-source release). install-toolchain.sh installs either
# without root. The AOT builds also need ocloc, and Node 18+ gives the addon its
# headers. Linux only: build-win.sh builds the Windows core. Nothing here
# touches the CUDA build.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../src"
OUT="$HERE/build"
TARGETS="intel_gpu_bmg_g21,intel_gpu_bmg_g31,intel_gpu_lnl_m,intel_gpu_ptl_h,intel_gpu_ptl_u,intel_gpu_pvc,intel_gpu_acm_g10,intel_gpu_acm_g11,intel_gpu_acm_g12,spir64_x86_64"
JIT=0
DUMP=""
LARGE_GRF=0
PEAK=0
STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    --jit) JIT=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --dump) DUMP="$2"; shift 2 ;;
    --large-grf) LARGE_GRF=1; shift ;;
    --peak) PEAK=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
if [ -n "$STAGE" ] && [ "$JIT" != 1 ]; then
  echo "--stage is for the build a release carries: add --jit" >&2; exit 2
fi
if [ "$JIT" = 1 ] && { [ -n "$DUMP" ] || [ "$LARGE_GRF" = 1 ]; }; then
  echo "--dump and --large-grf are for the prebuilt (AOT) kernels: drop --jit" >&2; exit 2
fi

# The compiler: CXX, else icpx, else a clang++ that knows -fsycl.
if [ -z "${CXX:-}" ]; then
  if command -v icpx >/dev/null; then CXX=icpx
  elif command -v clang++ >/dev/null && [ -f "$(clang++ -fsycl -print-file-name=libsycl.so 2>/dev/null)" ]; then CXX=clang++
  else
    echo "no DPC++ compiler: run install-toolchain.sh (--open for the one releases are built with) and" >&2
    echo "source its envrc.sh, or source oneAPI's setvars.sh, or set CXX" >&2
    exit 1
  fi
fi
command -v "$CXX" >/dev/null || { echo "no $CXX on PATH" >&2; exit 1; }
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
# Extra compiler flags for a timing build, e.g. PEARL_SYCL_EXTRA="-DPEARL_A5=2". No release sets it.
if [ -n "${PEARL_SYCL_EXTRA:-}" ]; then read -ra EXTRA <<< "$PEARL_SYCL_EXTRA"; SYCL_FLAGS+=("${EXTRA[@]}"); fi
if [ "$JIT" = 1 ]; then
  # A device image a kernel: a hardware fold's kernel has nothing but the DPAS
  # built-in, so it must reach only the driver of a card the host launches it on
  # (pearl_sycl_kernels.hpp).
  SYCL_FLAGS+=(-fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -DPEARL_SYCL_XMX_JIT)
  # The ring fold's split barrier (pearl_sycl_kernels.hpp) is a SPIR-V extension the translator leaves off.
  SYCL_FLAGS+=(-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier)
else
  SYCL_FLAGS+=(-fsycl-targets="$TARGETS" -DPEARL_SYCL_XMX_HW)
  if [ "$LARGE_GRF" = 1 ]; then
    IFS=',' read -ra TL <<< "$TARGETS"
    for t in "${TL[@]}"; do
      case "$t" in
        intel_gpu_*) SYCL_FLAGS+=("-Xsycl-target-backend=$t" "-options -ze-opt-large-register-file") ;;
      esac
    done
  fi
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
# addon finds libsycl and the runtime libraries without LD_LIBRARY_PATH. With
# --stage only the first: the staged copy has its libraries beside it, and must
# not name a directory of the machine that built it. The build directory's own
# copies then run only with LD_LIBRARY_PATH set.
RPATH=(-Wl,-rpath,'$ORIGIN')
SYCL_LIB=$(dirname "$("$CXX" -fsycl -print-file-name=libsycl.so)")
SYCL_LIB=$(cd "$SYCL_LIB" && pwd)
[ -f "$SYCL_LIB/libsycl.so" ] || { echo "no libsycl.so beside $CXX's libraries ($SYCL_LIB)" >&2; exit 1; }
if [ -n "$STAGE" ]; then RPATH+=(-Wl,-rpath-link,"$SYCL_LIB"); else RPATH+=(-Wl,-rpath,"$SYCL_LIB"); fi

DUMP_ENV=()
if [ -n "$DUMP" ]; then
  mkdir -p "$DUMP"
  DUMP_ENV=(env IGC_ShaderDumpEnable=1 "IGC_DumpToCustomDir=$(cd "$DUMP" && pwd)")
fi

echo "[1/3] libpearl_sycl.so ($([ "$JIT" = 1 ] && echo "spir64, compiled by the driver" || echo "AOT: $TARGETS"))"
"${DUMP_ENV[@]}" "$CXX" "${SYCL_FLAGS[@]}" "${GCC_FLAGS[@]}" -shared -I"$HERE" -I"$SRC" \
  "$HERE/pearl_sycl_host.cpp" -o "$OUT/libpearl_sycl.so" "${RPATH[@]}" || {
  echo "build failed. If the error names a target (an older compiler or ocloc may not know" >&2
  echo "bmg_g31, ptl_h or ptl_u), build for your card only, e.g." >&2
  echo "  ./build.sh --targets intel_gpu_bmg_g21,spir64_x86_64   (README.md, \"Which build\")" >&2
  exit 1
}

# The addon has no SYCL of its own: it links the host library and nothing else.
# PEARL_HOST_MEM_INFO: this host can say how much memory the card has, and the
# app has no other source for an Arc card's (pearl_core.cc).
echo "[2/3] pearl_core_sycl.node"
"$CXX" -O2 -fPIC -std=c++17 "${GCC_FLAGS[@]}" -DNAPI_DISABLE_CPP_EXCEPTIONS -DNAPI_VERSION=8 -DPEARL_HOST_MEM_INFO \
  -I"$NODE_INC" -I"$NAPI_INC" -I"$SRC" -c "$SRC/pearl_core.cc" -o "$OUT/pearl_core.o"
"$CXX" "${GCC_FLAGS[@]}" -shared "$OUT/pearl_core.o" -L"$OUT" -lpearl_sycl \
  -o "$OUT/pearl_core_sycl.node" "${RPATH[@]}"

echo "[3/3] pearl_sycl_check"
"$CXX" -O2 -std=c++17 "${GCC_FLAGS[@]}" -I"$HERE" -I"$SRC" "$HERE/pearl_sycl_check.cpp" \
  -L"$OUT" -lpearl_sycl -o "$OUT/pearl_sycl_check" "${RPATH[@]}" -lpthread
rm -f "$OUT/pearl_core.o"
if [ "$PEAK" = 1 ]; then
  # Standalone: its own kernels and the fold's, with the same targets and flags.
  # With --large-grf it also has the rows for 16 rows a sub-group (--rb 2).
  PEAK_FLAGS=()
  [ "$LARGE_GRF" = 1 ] && PEAK_FLAGS=(-DPEARL_PEAK_LARGE_GRF)
  echo "[peak] pearl_sycl_peak"
  "${DUMP_ENV[@]}" "$CXX" "${SYCL_FLAGS[@]}" "${GCC_FLAGS[@]}" "${PEAK_FLAGS[@]}" -I"$HERE" -I"$SRC" "$HERE/pearl_sycl_peak.cpp" \
    -o "$OUT/pearl_sycl_peak" "${RPATH[@]}" -lpthread
fi
echo "done: $OUT"

if [ -n "$STAGE" ]; then
  bash "$HERE/stage.sh" "$OUT" "$SYCL_LIB" "$STAGE"
fi
