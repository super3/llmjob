#!/usr/bin/env bash
# Install a SYCL compiler without root: DPC++ 2026.1.1 and the OpenCL CPU runtime
# from conda-forge, through micromamba, into ./toolchain (or DIR). About 600 MB
# to download and 2.2 GB on disk after the pruning below. Then:
#
#   . ./toolchain/envrc.sh && ./build.sh
#
#   ./install-toolchain.sh [DIR] [--ocloc]
#
# --ocloc also extracts ocloc and IGC from the kobuk-team PPA, for this Ubuntu
# release (24.04 or 26.04; PEARL_OCLOC_SERIES=noble or resolute picks one). It
# takes the versions the PPA has today and prints them: the PPA keeps only its
# newest build of each package, so the 26.31 and 2.40.13 this script named in
# October were gone a month later and it stopped with a 404. They are also the
# versions the driver from the same PPA is built from. AOT for GPU targets needs
# an ocloc; a driver's own may be too old to know every default target
# (README.md, "Compiler"). Nothing is installed system-wide, and the driver
# keeps its own IGC: only ocloc runs with these libraries.
#
# If oneAPI is installed already (source /opt/intel/oneapi/setvars.sh), none of
# this is needed: build.sh uses whatever icpx is on PATH.
#
#   ./install-toolchain.sh --open [DIR]
#
# The compiler releases are built with: the open-source DPC++ (github.com/
# intel/llvm, release v7.2.0, checked against its SHA-256) into ./toolchain-open
# (or DIR). 242 MB to download, 670 MB on disk. Its runtime libraries are under
# the Apache License 2.0 with LLVM exceptions, so build.sh --jit --stage may
# copy them into a release; the oneAPI ones above are under Intel's end user
# licence for developer tools and may not be. It has no ocloc, so it builds the
# --jit form only. It runs on glibc 2.27 or newer (Ubuntu 18.04).
#
#   ./install-toolchain.sh --open --cpu-rt [DIR]
#
# also installs Intel's OpenCL CPU runtime from conda-forge beside it, so that
# verify.sh can run on the SYCL CPU device (PEARL_SYCL_DEVICE=cpu). That runtime
# is Intel's, for testing only: nothing stages or ships it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=""
OCLOC=0
OPEN=0
CPU_RT=0
for a in "$@"; do
  case "$a" in
    --ocloc) OCLOC=1 ;;
    --open) OPEN=1 ;;
    --cpu-rt) CPU_RT=1 ;;
    *) DIR="$a" ;;
  esac
done
if [ "$OPEN" = 1 ]; then
  [ "$OCLOC" = 0 ] || { echo "--open has no ocloc: it builds ./build.sh --jit, which needs none" >&2; exit 2; }
  DIR=${DIR:-$HERE/toolchain-open}
  mkdir -p "$DIR"
  DIR=$(cd "$DIR" && pwd)
  cd "$DIR"
  # One release, by name and by digest: a rebuilt archive under the same tag is
  # a compiler nobody has checked a fold with.
  DPCPP_TAG=v7.2.0
  DPCPP_SHA256=385586a25e75ac727883f0be5e442dec0ee81b3c20f9a29037a79921ad2f1728
  if [ ! -x dpcpp/bin/clang++ ]; then
    mkdir -p dl
    curl -fsSL -o dl/sycl_linux.tar.gz "https://github.com/intel/llvm/releases/download/$DPCPP_TAG/sycl_linux.tar.gz"
    echo "$DPCPP_SHA256  dl/sycl_linux.tar.gz" | sha256sum -c - >/dev/null \
      || { echo "sycl_linux.tar.gz of $DPCPP_TAG is not the file this script was written for (SHA-256 differs)" >&2; exit 1; }
    rm -rf dpcpp && mkdir dpcpp && tar -xzf dl/sycl_linux.tar.gz -C dpcpp
    rm -rf dl
  fi
  dpcpp/bin/clang++ --version | head -n 1
  {
    echo "# source this: the open-source DPC++ compiler and runtime from install-toolchain.sh --open"
    echo "export PATH=\"$DIR/dpcpp/bin:\$PATH\""
    echo "export LD_LIBRARY_PATH=\"$DIR/dpcpp/lib:\${LD_LIBRARY_PATH:-}\""
    echo "if [ -f \"$DIR/cpu-rt.sh\" ]; then . \"$DIR/cpu-rt.sh\"; fi"
  } > envrc.sh
  if [ "$CPU_RT" = 1 ]; then
    if [ ! -d cpu-rt/etc/OpenCL/vendors ]; then
      mkdir -p dl mm
      curl -fsSL -o dl/mm.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
      tar -xjf dl/mm.tar.bz2 -C mm bin/micromamba
      MAMBA_ROOT_PREFIX="$DIR/mroot" ./mm/bin/micromamba create -y -q -p "$DIR/cpu-rt" -c conda-forge --override-channels \
        intel-opencl-rt=2026.1.1 ocl-icd
      rm -rf mroot dl mm
    fi
    # The runtime is found through the OpenCL loader, by a list of drivers that
    # holds the system's too. Its libraries go on the library path: it is built
    # against a newer libstdc++ than Ubuntu 20.04's, and a process that has
    # loaded the system's first cannot load it (the CPU device is then just
    # missing). That set has no SYCL runtime in it.
    #
    # cpu-rt.sh, which envrc.sh sources, is also sourced alone to check a staged
    # core: that needs the CPU device and must not find the compiler's copy of
    # the SYCL runtime before its own. What the staged files themselves need of
    # glibc and libstdc++ is read from them, not found out by running them here
    # (check-glibc-floor.sh).
    cat > cpu-rt.sh <<ENV
# source this: Intel's OpenCL CPU runtime from install-toolchain.sh --open --cpu-rt (testing only)
mkdir -p "$DIR/ocl-vendors" && rm -f "$DIR/ocl-vendors"/*.icd
for f in /etc/OpenCL/vendors/*.icd; do if [ -e "\$f" ]; then ln -sf "\$f" "$DIR/ocl-vendors/system-\${f##*/}"; fi; done
for f in "$DIR"/cpu-rt/etc/OpenCL/vendors/*.icd; do if [ -e "\$f" ]; then ln -sf "\$f" "$DIR/ocl-vendors/toolchain-\${f##*/}"; fi; done
unset f
export OCL_ICD_VENDORS="$DIR/ocl-vendors"
export LD_LIBRARY_PATH="$DIR/cpu-rt/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV
  fi
  du -sh "$DIR"
  echo "done: . $DIR/envrc.sh && ./build.sh --jit"
  exit 0
fi
[ "$CPU_RT" = 0 ] || { echo "--cpu-rt goes with --open: the oneAPI toolchain has the CPU runtime already" >&2; exit 2; }
DIR=${DIR:-$HERE/toolchain}
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
  # Whatever the PPA has now for this Ubuntu release, read from its package
  # index. A file name with a version in it is gone from the PPA's pool a few
  # weeks after the next build is published.
  PPA=https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu
  SERIES=${PEARL_OCLOC_SERIES:-$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-noble}")}
  mkdir -p gpudl gpu
  curl -fsSL "$PPA/dists/$SERIES/main/binary-amd64/Packages.gz" | gunzip > gpudl/Packages || {
    echo "the kobuk-team PPA has no package index for Ubuntu '$SERIES': it builds for noble (24.04) and" >&2
    echo "resolute (26.04). Set PEARL_OCLOC_SERIES to the one whose glibc this box can run." >&2
    exit 1
  }
  for pkg in intel-ocloc libigc2 libigdfcl2 libigdgmm12; do
    f=$(awk -v p="$pkg" '$1 == "Package:" { ok = ($2 == p) } ok && $1 == "Filename:" { print $2 }' gpudl/Packages \
        | sort -V | tail -1)
    [ -n "$f" ] || { echo "the PPA's $SERIES index has no $pkg" >&2; exit 1; }
    echo "ocloc toolchain: ${f##*/}"
    curl -fsS -o "gpudl/${f##*/}" "$PPA/$f"
  done
  for d in gpudl/*.deb; do dpkg-deb -x "$d" gpu; done
  rm -rf gpudl
  L=gpu/usr/lib/x86_64-linux-gnu
  # A package carries each library under its full version (libigc.so.2.42.14+0)
  # and leaves the name the loader asks for (libigc.so.2) to ldconfig, which
  # does not run here.
  for f in "$L"/lib*.so.*; do
    if [ -f "$f" ] && [ ! -L "$f" ]; then
      b=${f##*/}
      so=${b%%.so.*}.so.$(printf '%s' "${b#*.so.}" | cut -d. -f1)
      [ -e "$L/$so" ] || ln -s "$b" "$L/$so"
    fi
  done
  OCLOC_BIN=$(find "$DIR/gpu/usr/bin" -maxdepth 1 -name 'ocloc-*' 2>/dev/null | sort | head -1) || true
  [ -n "$OCLOC_BIN" ] || { echo "the intel-ocloc package has no usr/bin/ocloc-* in it" >&2; exit 1; }
  # ocloc gets these IGC libraries through a wrapper, not through envrc.sh's
  # LD_LIBRARY_PATH: on a box with an Intel driver, the driver must keep
  # loading its own IGC in that shell.
  mkdir -p gpu/bin
  cat > gpu/bin/ocloc <<OCLOC
#!/bin/sh
LD_LIBRARY_PATH="$DIR/$L\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" exec "$OCLOC_BIN" "\$@"
OCLOC
  chmod +x gpu/bin/ocloc
  # These are Ubuntu 24.04 or 26.04 binaries. On an older glibc the loader
  # refuses them; say so here, not as a failed target in build.sh.
  out=$(gpu/bin/ocloc --help 2>&1 || true)
  case "$out" in
    *"error while loading shared libraries"*|*"version \`GLIBC"*)
      echo "the extracted ocloc does not run on this box (it is built for Ubuntu $SERIES):" >&2
      printf '%s\n' "$out" | head -3 >&2
      exit 1 ;;
  esac
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
