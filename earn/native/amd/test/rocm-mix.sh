#!/bin/bash
# Builds a ROCm tree at $1 (default /opt/rocmmix) that is the image's ROCm (/opt/rocm) with a newer compiler: AMD's
# amdrocm-llvm package of ROCm $ROCM_MIX_VER (default 7.13, clang 23) in place of lib/llvm. Headers, device libraries
# and the HIP runtime stay the image's, so the core it builds runs on the image's runtime as the release does. Only
# for racing compilers on a test box (sweep.sh's ROCM= flag); prints the clang version it set up.
set -eu
DEST=${1:-/opt/rocmmix}
VER=${ROCM_MIX_VER:-7.13}
REV=${ROCM_MIX_REV:-7.13.0-2}
POOL=https://repo.amd.com/rocm/packages/ubuntu2404/pool/main
X=/opt/rocm-$VER-x
mkdir -p "$X" /tmp/rocmmix-deb
for p in amdrocm-llvm$VER amdrocm-sysdeps$VER; do
  f=/tmp/rocmmix-deb/$p.deb
  [ -s "$f" ] || curl -fsSL --retry 3 -o "$f" "$POOL/${p}_${REV}_amd64.deb"
  dpkg-deb -x "$f" "$X"
done
LLVM=$(ls -d "$X"/opt/rocm/core-$VER/lib/llvm)
SYS=$(ls -d "$X"/opt/rocm/core-$VER/lib/rocm_sysdeps/lib 2>/dev/null || true)
rm -rf "$DEST"
mkdir -p "$DEST/lib"
for e in /opt/rocm/*; do
  [ "$(basename "$e")" = lib ] || ln -s "$(readlink -f "$e")" "$DEST/$(basename "$e")"
done
for e in /opt/rocm/lib/*; do
  [ "$(basename "$e")" = llvm ] || ln -s "$(readlink -f "$e")" "$DEST/lib/$(basename "$e")"
done
ln -s "$LLVM" "$DEST/lib/llvm"
# The sysdeps' shared libraries (zstd, elf, ...) for clang itself, if its RUNPATH doesn't already find them.
if [ -n "$SYS" ] && ! "$LLVM/bin/clang++" --version > /dev/null 2>&1; then
  echo "$SYS" > /etc/ld.so.conf.d/zz-rocmmix.conf
  ldconfig
fi
"$DEST/lib/llvm/bin/clang++" --version | head -1
