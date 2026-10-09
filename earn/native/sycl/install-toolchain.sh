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
# (Ubuntu 24.04 packages, 228 MB), the versions the default targets were built
# with. AOT for GPU targets needs an ocloc; a driver's own may be too old to know
# every default target (README.md, "Compiler"). Nothing is installed system-wide,
# and the driver keeps its own IGC: only ocloc runs with these libraries.
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
  # ocloc gets these IGC libraries through a wrapper, not through envrc.sh's
  # LD_LIBRARY_PATH: on a box with an Intel driver, the driver must keep
  # loading its own IGC in that shell.
  mkdir -p gpu/bin
  cat > gpu/bin/ocloc <<OCLOC
#!/bin/sh
LD_LIBRARY_PATH="$DIR/$L\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" exec "$DIR/gpu/usr/bin/ocloc-26.31.1" "\$@"
OCLOC
  chmod +x gpu/bin/ocloc
fi
# OCL_ICD_VENDORS replaces the system's list of OpenCL drivers, so pointing it
# at the CPU runtime alone hid the GPU from clinfo and sycl-ls in that shell.
# envrc.sh builds a directory with both, each time it is sourced, so a driver
# installed after the toolchain is picked up too.
cat > envrc.sh <<ENV
# source this: the SYCL compiler and runtime from install-toolchain.sh
export PATH="$E/bin:$E/bin/compiler:\$PATH"
export LD_LIBRARY_PATH="$E/lib:\${LD_LIBRARY_PATH:-}"
mkdir -p "$DIR/ocl-vendors" && rm -f "$DIR/ocl-vendors"/*.icd
for f in /etc/OpenCL/vendors/*.icd; do if [ -e "\$f" ]; then ln -sf "\$f" "$DIR/ocl-vendors/system-\${f##*/}"; fi; done
for f in "$E"/etc/OpenCL/vendors/*.icd; do if [ -e "\$f" ]; then ln -sf "\$f" "$DIR/ocl-vendors/toolchain-\${f##*/}"; fi; done
unset f
export OCL_ICD_VENDORS="$DIR/ocl-vendors"
export CONDA_PREFIX="$E"
ENV
if [ "$OCLOC" = 1 ]; then
  echo "export PATH=\"$DIR/gpu/bin:\$PATH\"" >> envrc.sh
fi
du -sh "$DIR"
echo "done: . $DIR/envrc.sh"
