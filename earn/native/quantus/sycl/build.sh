#!/usr/bin/env bash
# Build the Intel (SYCL) Quantus core, quantus_sycl.node, for Intel Arc cards.
#
#   ./build.sh --jit           what a release carries: SPIR-V the driver compiles
#                              at first run. Any Intel GPU and the CPU device.
#                              Needs no ocloc
#   ./build.sh --jit --stage DIR
#                              and copy what a release ships into DIR: the addon,
#                              the SYCL runtime libraries it loads, and their
#                              licence. Only with the open-source DPC++
#                              (../../sycl/install-toolchain.sh --open)
#   ./build.sh                 the kernel prebuilt by ocloc (AOT) for TARGETS
#                              below plus the CPU device. Runs on those only
#   ./build.sh --targets LIST  AOT for LIST only (comma separated)
#   ./build.sh --dump DIR      also write IGC's assembly to DIR. AOT only (a --jit
#                              build dumps at run time instead: README.md)
#   ./build.sh --large-grf     256-register mode on the GPU targets. AOT only
#   ./build.sh --out DIR       output directory (default ../build, where
#                              src/main/quantusCore.js looks in a source tree)
#
# Writes quantus_sycl.node into the output directory. It first builds and runs
# qtc_selftest (host only, no SYCL): the five official nonce hashes through the
# same field code the kernel runs.
#
# The same toolchain as the Pearl SYCL core (../../sycl/build.sh): CXX, else
# icpx (Intel's oneAPI), else a clang++ that takes -fsycl (the open-source
# release). Node 18+ gives the addon its headers (node_api.h only: the addon is
# plain N-API, no node-addon-api). Extra compiler flags for a timing build go
# in QTC_SYCL_EXTRA, e.g. QTC_SYCL_EXTRA="-DQTC_UNROLL_PARTIAL=1". Linux only.
#
# Where the runtime comes from when the core runs: its run path is $ORIGIN and
# $ORIGIN/sycl, so libsycl and the rest are found beside the addon or in a sycl
# folder beside it, which is where a release keeps the Pearl Intel core and the
# same runtime (README.md, "Runtime libraries"). A build that is not staged also
# names the compiler's own lib directory.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../build"
TARGETS="intel_gpu_bmg_g21,intel_gpu_acm_g10,intel_gpu_acm_g11,intel_gpu_acm_g12,spir64_x86_64"
JIT=0
DUMP=""
LARGE_GRF=0
STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    --jit) JIT=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --dump) DUMP="$2"; shift 2 ;;
    --large-grf) LARGE_GRF=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
if [ -n "$STAGE" ] && [ "$JIT" != 1 ]; then
  echo "--stage is for the build a release carries: add --jit" >&2; exit 2
fi
if [ "$JIT" = 1 ] && { [ -n "$DUMP" ] || [ "$LARGE_GRF" = 1 ]; }; then
  echo "--dump and --large-grf are for the prebuilt (AOT) kernel: drop --jit" >&2; exit 2
fi

if [ -z "${CXX:-}" ]; then
  if command -v icpx >/dev/null; then CXX=icpx
  elif command -v clang++ >/dev/null && [ -f "$(clang++ -fsycl -print-file-name=libsycl.so 2>/dev/null)" ]; then CXX=clang++
  else
    echo "no DPC++ compiler: run ../../sycl/install-toolchain.sh (--open for the one releases are built with)" >&2
    echo "and source its envrc.sh, or source oneAPI's setvars.sh, or set CXX" >&2
    exit 1
  fi
fi
command -v "$CXX" >/dev/null || { echo "no $CXX on PATH" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# The system's libstdc++, as for the Pearl core: Node has loaded it already.
GCC_FLAGS=()
if [ -z "${PEARL_NO_SYSTEM_GCC:-}" ] && command -v g++ >/dev/null; then
  GCC_DIR=$(dirname "$(g++ -print-libgcc-file-name)")
  [ -d "$GCC_DIR" ] && GCC_FLAGS=(--gcc-install-dir="$GCC_DIR")
fi

echo "[1/2] qtc_selftest (host)"
"${HOST_CXX:-${CXX}}" -O2 -std=c++17 -Wno-unknown-pragmas "${GCC_FLAGS[@]}" "$HERE/qtc_selftest.cpp" -o "$OUT/qtc_selftest"
"$OUT/qtc_selftest" | tail -n 1

# A device image a kernel form: the driver compiles only the form a card runs.
SYCL_FLAGS=(-fsycl -O3 -fPIC -std=c++17 -Wno-psabi -fsycl-device-code-split=per_kernel)
if [ -n "${QTC_SYCL_EXTRA:-}" ]; then read -ra EXTRA <<< "$QTC_SYCL_EXTRA"; SYCL_FLAGS+=("${EXTRA[@]}"); fi
if [ "$JIT" = 1 ]; then
  SYCL_FLAGS+=(-fsycl-targets=spir64)
else
  SYCL_FLAGS+=(-fsycl-targets="$TARGETS")
  if [ "$LARGE_GRF" = 1 ]; then
    IFS=',' read -ra TL <<< "$TARGETS"
    for t in "${TL[@]}"; do
      case "$t" in
        intel_gpu_*) SYCL_FLAGS+=("-Xsycl-target-backend=$t" "-options -ze-opt-large-register-file") ;;
      esac
    done
  fi
fi

NODE_VER=$(node -p 'process.versions.node')
NODE_INC=""
for d in "$(dirname "$(command -v node)")/../include/node" "$HOME/.cache/node-gyp/$NODE_VER/include/node"; do
  [ -f "$d/node_api.h" ] && NODE_INC="$d" && break
done
if [ -z "$NODE_INC" ]; then
  npx --yes node-gyp install >/dev/null
  NODE_INC="$HOME/.cache/node-gyp/$NODE_VER/include/node"
fi

RPATH=(-Wl,-rpath,'$ORIGIN:$ORIGIN/sycl')
SYCL_LIB=$(dirname "$("$CXX" -fsycl -print-file-name=libsycl.so)")
SYCL_LIB=$(cd "$SYCL_LIB" && pwd)
[ -f "$SYCL_LIB/libsycl.so" ] || { echo "no libsycl.so beside $CXX's libraries ($SYCL_LIB)" >&2; exit 1; }
if [ -n "$STAGE" ]; then RPATH+=(-Wl,-rpath-link,"$SYCL_LIB"); else RPATH+=(-Wl,-rpath,"$SYCL_LIB"); fi

DUMP_ENV=()
if [ -n "$DUMP" ]; then
  mkdir -p "$DUMP"
  DUMP_ENV=(env IGC_ShaderDumpEnable=1 "IGC_DumpToCustomDir=$(cd "$DUMP" && pwd)")
fi

echo "[2/2] quantus_sycl.node ($([ "$JIT" = 1 ] && echo "spir64, compiled by the driver" || echo "AOT: $TARGETS"))"
"${DUMP_ENV[@]}" "$CXX" "${SYCL_FLAGS[@]}" "${GCC_FLAGS[@]}" -shared -I"$HERE" -I"$NODE_INC" \
  "$HERE/quantus_sycl.cpp" -o "$OUT/quantus_sycl.node" "${RPATH[@]}"
echo "done: $OUT/quantus_sycl.node"

if [ -n "$STAGE" ]; then
  needed() { readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'; }
  for n in $(needed "$OUT/quantus_sycl.node"); do
    case "$n" in
      libsvml.so*|libimf.so*|libirng.so*|libintlc.so*|libiomp5.so*)
        echo "stage: quantus_sycl.node needs $n, Intel's oneAPI runtime, which a release may not carry." >&2
        echo "Build with the open-source DPC++: ../../sycl/install-toolchain.sh --open" >&2
        exit 1 ;;
    esac
  done
  # The libsycl it links, what that needs from the same directory, and the
  # adapters libsycl opens by name: the same set the Pearl core stages.
  RUNTIME=()
  for n in $(needed "$OUT/quantus_sycl.node"); do
    case "$n" in libsycl.so.*) RUNTIME+=("$n") ;; esac
  done
  [ ${#RUNTIME[@]} -eq 1 ] || { echo "stage: quantus_sycl.node does not link one libsycl.so.N" >&2; exit 1; }
  for n in $(needed "$SYCL_LIB/${RUNTIME[0]}"); do
    [ -e "$SYCL_LIB/$n" ] && RUNTIME+=("$n")
  done
  for a in libur_adapter_level_zero.so.0 libur_adapter_level_zero_v2.so.0 libur_adapter_opencl.so.0; do
    [ -e "$SYCL_LIB/$a" ] && RUNTIME+=("$a")
  done
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  STAGE=$(cd "$STAGE" && pwd)
  cp "$OUT/quantus_sycl.node" "$STAGE/"
  strip --strip-unneeded "$STAGE/quantus_sycl.node"
  chmod 644 "$STAGE/quantus_sycl.node"
  for n in "${RUNTIME[@]}"; do
    cp -L "$SYCL_LIB/$n" "$STAGE/$n"
    chmod 644 "$STAGE/$n"
  done
  cp "$HERE/../../sycl/licenses/LICENSE-DPCPP.txt" "$HERE/../../sycl/licenses/THIRD-PARTY-DPCPP.txt" "$STAGE/"
  for f in "$STAGE"/*.so* "$STAGE"/*.node; do
    rp=$(readelf -d "$f" | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p')
    case "$rp" in
      ''|'$ORIGIN'|'$ORIGIN:$ORIGIN/sycl') ;;
      *) echo "stage: $(basename "$f") has run path '$rp'" >&2; exit 1 ;;
    esac
    for n in $(needed "$f"); do
      case "$n" in
        libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|ld-linux-x86-64.so.2|libstdc++.so.6|libgcc_s.so.1) ;;
        *) [ -e "$STAGE/$n" ] || { echo "stage: $(basename "$f") needs $n, which is not staged" >&2; exit 1; } ;;
      esac
    done
  done
  {
    echo "The Intel (SYCL) Quantus core of LLMJob Earn, and the libraries it loads."
    echo
    echo "Ours (MIT, as the rest of LLMJob Earn): quantus_sycl.node."
    echo
    echo "The SYCL runtime, unchanged, from the oneAPI DPC++ compiler's open-source release"
    echo "(https://github.com/intel/llvm, $("$SYCL_LIB/../bin/clang++" --version 2>/dev/null | head -n 1)):"
    for n in "${RUNTIME[@]}"; do echo "  $n"; done
    echo "Apache License 2.0 with LLVM exceptions: LICENSE-DPCPP.txt. THIRD-PARTY-DPCPP.txt lists the"
    echo "software built into them with each one's licence. These are the same files the Pearl Intel"
    echo "core ships in its sycl folder."
    echo
    echo "Not here, and needed on the rig: Intel's graphics driver for compute (Level Zero or OpenCL)."
  } > "$STAGE/NOTICE.txt"
  echo "staged in $STAGE:"
  ( cd "$STAGE" && for f in *; do printf '  %9d  %s\n' "$(stat -c %s "$f")" "$f"; done )
fi
