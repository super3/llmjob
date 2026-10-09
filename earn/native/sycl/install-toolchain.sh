#!/usr/bin/env bash
# Install a SYCL compiler without root: DPC++ 2026.1.1 and the OpenCL CPU runtime
# from conda-forge, through micromamba, into ./toolchain (or DIR). About 600 MB
# to download and 2.2 GB on disk after the pruning below. Then:
#
#   . ./toolchain/envrc.sh && ./build.sh
#
#   ./install-toolchain.sh [DIR] [--ocloc]
#
# --ocloc also extracts ocloc 26.31 and IGC 2.40.13 from the kobuk-team PPA
# (Ubuntu 24.04 packages, 228 MB): what AOT for GPU targets needs on a box with
# no Intel GPU driver. A box with an Intel GPU has ocloc from its driver
# packages instead (see README.md, "Driver").
#
# If oneAPI is installed already (source /opt/intel/oneapi/setvars.sh), none of
# this is needed: build.sh uses whatever icpx is on PATH.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR="$HERE/toolchain"
OCLOC=0
for a in "$@"; do
  case "$a" in
    --ocloc) OCLOC=1 ;;
    *) DIR="$a" ;;
  esac
done
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
cd "$DIR"
mkdir -p dl mm
curl -fsSL -o dl/mm.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
tar -xjf dl/mm.tar.bz2 -C mm bin/micromamba
MAMBA_ROOT_PREFIX="$DIR/mroot" ./mm/bin/micromamba create -y -q -p "$DIR/env" -c conda-forge --override-channels \
  dpcpp_impl_linux-64=2026.1.1 intel-opencl-rt=2026.1.1
rm -rf mroot dl mm   # the package cache; env/ keeps its own copies
# Pruning: duplicate copies and tools the build never runs (about 600 MB).
E="$DIR/env"
if cmp -s "$E/bin/compiler/clang" "$E/bin/compiler/clang-22"; then rm "$E/bin/compiler/clang-22"; ln -s clang "$E/bin/compiler/clang-22"; fi
if cmp -s "$E/lib/libsycl-jit.so" "$E/lib/libsycl-jit.so.22.1"; then rm "$E/lib/libsycl-jit.so"; ln -s libsycl-jit.so.22.1 "$E/lib/libsycl-jit.so"; fi
rm -f "$E"/bin/compiler/{clang-tidy,clangd,clang-include-fixer,modularize,llvm-dwp,llvm-profgen,llvm-ml,llvm-cov,llvm-profdata,clang-format} \
  "$E"/lib/libsycl-preview.so*
if [ "$OCLOC" = 1 ]; then
  P=https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/pool/main
  mkdir -p gpudl gpu
  for f in i/intel-compute-runtime/intel-ocloc_26.31.39395.14-1~24.04~ppa1_amd64.deb \
           i/intel-graphics-compiler/libigc2_2.40.13+ds1-1~24.04_amd64.deb \
           i/intel-graphics-compiler/libigdfcl2_2.40.13+ds1-1~24.04_amd64.deb \
           i/intel-gmmlib/libigdgmm12_22.10.1-1~24.04~ppa1_amd64.deb; do
    curl -fsS -o "gpudl/$(basename "$f")" "$P/$f"
  done
  for d in gpudl/*.deb; do dpkg-deb -x "$d" gpu; done
  rm -rf gpudl
  L=gpu/usr/lib/x86_64-linux-gnu
  for l in libigc libigdfcl libiga64; do ln -sf "$l.so.2.40.13+0" "$L/$l.so.2"; done
  ln -sf ocloc-26.31.1 gpu/usr/bin/ocloc
fi
cat > envrc.sh <<ENV
# source this: the SYCL compiler and runtime from install-toolchain.sh
export PATH="$E/bin:$E/bin/compiler:\$PATH"
export LD_LIBRARY_PATH="$E/lib:\${LD_LIBRARY_PATH:-}"
export OCL_ICD_VENDORS="$E/etc/OpenCL/vendors"
export CONDA_PREFIX="$E"
ENV
if [ "$OCLOC" = 1 ]; then
  cat >> envrc.sh <<ENV
export PATH="$DIR/gpu/usr/bin:\$PATH"
export LD_LIBRARY_PATH="$DIR/gpu/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
ENV
fi
du -sh "$DIR"
echo "done: . $DIR/envrc.sh"
