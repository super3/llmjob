#!/usr/bin/env bash
# Copy what a release ships of the SYCL core into one folder that runs from
# anywhere: build.sh --jit --stage DIR calls this.
#
#   ./stage.sh BUILD_DIR SYCL_LIB_DIR STAGE_DIR
#
# STAGE_DIR gets
#   pearl_core_sycl.node, libpearl_sycl.so, pearl_sycl_check    ours, stripped
#   libsycl.so.N, libumf.so.N, libur_adapter_*.so.N             the SYCL runtime
#       they load, from the compiler's lib directory. Each already looks for
#       the others beside itself ($ORIGIN), as ours do
#   LICENSE-DPCPP.txt, NOTICE.txt                               their licence
#
# Only the open-source DPC++ (github.com/intel/llvm, Apache-2.0 with the LLVM
# exception) is staged. A build made with Intel's oneAPI compiler links Intel's
# own math and runtime libraries, which come under Intel's end user licence for
# developer tools and are not ours to hand out: this refuses one.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${1:?usage: ./stage.sh BUILD_DIR SYCL_LIB_DIR STAGE_DIR}
LIB=${2:?usage: ./stage.sh BUILD_DIR SYCL_LIB_DIR STAGE_DIR}
STAGE=${3:?usage: ./stage.sh BUILD_DIR SYCL_LIB_DIR STAGE_DIR}

needed() { readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'; }

for f in pearl_core_sycl.node libpearl_sycl.so pearl_sycl_check; do
  [ -f "$B/$f" ] || { echo "stage: no $B/$f" >&2; exit 1; }
  for n in $(needed "$B/$f"); do
    case "$n" in
      libsvml.so*|libimf.so*|libirng.so*|libintlc.so*|libiomp5.so*)
        echo "stage: $f needs $n, Intel's oneAPI runtime, which a release may not carry." >&2
        echo "Build with the open-source DPC++: ./install-toolchain.sh --open" >&2
        exit 1 ;;
    esac
  done
done

# The runtime: the libsycl ours need, what that needs from the same directory,
# and the adapters libsycl opens by name (Level Zero, both forms, and OpenCL
# where the compiler has it as a file of its own; v7 builds it into libsycl).
RUNTIME=()
for n in $(needed "$B/libpearl_sycl.so"); do
  case "$n" in libsycl.so.*) RUNTIME+=("$n") ;; esac
done
[ ${#RUNTIME[@]} -eq 1 ] || { echo "stage: libpearl_sycl.so does not link one libsycl.so.N" >&2; exit 1; }
for n in $(needed "$LIB/${RUNTIME[0]}"); do
  [ -e "$LIB/$n" ] && RUNTIME+=("$n")
done
for a in libur_adapter_level_zero.so.0 libur_adapter_level_zero_v2.so.0 libur_adapter_opencl.so.0; do
  [ -e "$LIB/$a" ] && RUNTIME+=("$a")
done

rm -rf "$STAGE"
mkdir -p "$STAGE"
STAGE=$(cd "$STAGE" && pwd)
for f in pearl_core_sycl.node libpearl_sycl.so pearl_sycl_check; do
  cp "$B/$f" "$STAGE/$f"
  strip --strip-unneeded "$STAGE/$f"
done
chmod 644 "$STAGE/pearl_core_sycl.node" "$STAGE/libpearl_sycl.so"
chmod 755 "$STAGE/pearl_sycl_check"
for n in "${RUNTIME[@]}"; do
  cp -L "$LIB/$n" "$STAGE/$n"
  chmod 644 "$STAGE/$n"
done
cp "$HERE/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"

# Every file must find the others beside it and nowhere else: a run path that
# names the build machine's compiler would be a wrong answer waiting on a rig.
for f in "$STAGE"/*.so* "$STAGE"/*.node "$STAGE/pearl_sycl_check"; do
  rp=$(readelf -d "$f" | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p')
  case "$rp" in
    ''|'$ORIGIN') ;;
    *) echo "stage: $(basename "$f") has run path '$rp', not \$ORIGIN: build with --stage so ours are linked that way" >&2; exit 1 ;;
  esac
done
# And nothing staged may need a library that is neither staged nor part of
# every Linux system's C and C++ runtime.
for f in "$STAGE"/*.so* "$STAGE"/*.node "$STAGE/pearl_sycl_check"; do
  for n in $(needed "$f"); do
    case "$n" in
      libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|ld-linux-x86-64.so.2|libstdc++.so.6|libgcc_s.so.1) ;;
      *) [ -e "$STAGE/$n" ] || { echo "stage: $(basename "$f") needs $n, which is not staged" >&2; exit 1; } ;;
    esac
  done
done

{
  echo "The Intel (SYCL) Pearl core of LLMJob Earn, and the libraries it loads."
  echo
  echo "Ours (MIT, as the rest of LLMJob Earn): pearl_core_sycl.node, libpearl_sycl.so, pearl_sycl_check."
  echo
  echo "The SYCL runtime, unchanged, from the oneAPI DPC++ compiler's open-source release"
  echo "(https://github.com/intel/llvm, $("$LIB/../bin/clang++" --version 2>/dev/null | head -n 1)):"
  for n in "${RUNTIME[@]}"; do echo "  $n"; done
  echo "Apache License 2.0 with LLVM exceptions: LICENSE-DPCPP.txt. libumf is the Unified Memory"
  echo "Framework and the libur_adapter files the Unified Runtime, both built from that repository"
  echo "under the same licence."
  echo
  echo "Not here, and needed on the rig: Intel's graphics driver for compute (Level Zero or OpenCL)."
} > "$STAGE/NOTICE.txt"

echo "staged in $STAGE:"
( cd "$STAGE" && for f in *; do printf '  %9d  %s  %s\n' "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)" "$f"; done )
echo "  total $(du -sk "$STAGE" | cut -f1) KB"
