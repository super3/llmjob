#!/usr/bin/env bash
# Stages the ROCm 7 HIP runtime the AMD core (pearl_core_hip.node) loads, for the HiveOS package:
#
#   bash scripts/fetch-hip-runtime.sh <out dir>      (CI: earn/vendor/hip-runtime)
#
# HiveOS doesn't ship ROCm 7, and its current image is Ubuntu 20.04, which ROCm 7 isn't published
# for. So the package carries the runtime in rocm/ beside the core, which looks there after
# /opt/rocm/lib (build-amd.sh). These are AMD's ROCm 7.2.0 packages for RHEL 8 (glibc 2.28), the
# release the core is built with, and three system libraries the HSA runtime needs from Rocky
# Linux 8, so every file loads on Ubuntu 20.04 and newer. Each library goes in under its soname,
# the name the others ask for; they find each other through their own $ORIGIN runpath.
#
# The packages are pinned by checksum: a different file fails this, not the rig. Needs rpm2cpio
# and cpio (apt: rpm2cpio cpio).
set -euo pipefail

out=${1:?usage: fetch-hip-runtime.sh <out dir>}
R=https://repo.radeon.com/rocm/rhel8/7.2/main
B=https://dl.rockylinux.org/pub/rocky/8/BaseOS/x86_64/os/Packages
PKGS=(
  "$R/hip-runtime-amd-7.2.26015.70200-43.el8.x86_64.rpm 09af9baf34ad328a60cf2825406b5739d18e51873ee6ca032cb8b45e25b01033"
  "$R/hsa-rocr-1.18.0.70200-43.el8.x86_64.rpm 058152320b0149f59b57f183345d1593f4d746f4466338d5fdf1eeaffc85ef95"
  "$R/comgr-3.0.0.70200-43.el8.x86_64.rpm a5404e501124bff13298c3c87a49f3e71e2c51beed9f0b739c01f69cac7d314a"
  "$R/rocprofiler-register-0.6.0.70200-43.el8.x86_64.rpm 08f952f4785f929b7537ae2e892361ecc1123fd66430b0c796b595ac1ff9dbc3"
  "$B/n/numactl-libs-2.0.16-4.el8.x86_64.rpm 08d156324fa101363d0559c9558d3df70e25b7033170ada0c5bfca9608a94fd6"
  "$B/e/elfutils-libelf-0.190-2.el8.x86_64.rpm 0cd9636d64cd3bcc98b27c959d1e21ccf4e8696313801069611eb2306e9c9957"
  "$B/l/libdrm-2.4.115-2.el8.x86_64.rpm 86becbf4c7dd367cb2e5a1dd286690f64bd3d3b63f1cbcb36493ecc07d74810e"
)
# The libraries the core's runtime needs, by soname, and the package each comes from. Everything
# else they need (libc, libstdc++, libz, libzstd, liblzma, libbz2) every Ubuntu has.
LIBS=(libamdhip64.so.7 libhsa-runtime64.so.1 libamd_comgr.so.3 librocprofiler-register.so.0
      libnuma.so.1 libelf.so.1 libdrm.so.2 libdrm_amdgpu.so.1)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for p in "${PKGS[@]}"; do
  url=${p% *} sum=${p#* }
  f="$work/$(basename "$url")"
  curl -fsSL --retry 5 --retry-all-errors -o "$f" "$url"
  echo "$sum  $f" | sha256sum -c --quiet || { echo "::error::$(basename "$url"): checksum mismatch"; exit 1; }
  (cd "$work" && rpm2cpio "$f" | cpio -idm --quiet)
done

rm -rf "$out"
mkdir -p "$out"
for lib in "${LIBS[@]}"; do
  src=$(find "$work/opt" "$work/usr" -name "$lib" 2>/dev/null | head -n 1)
  [ -n "$src" ] || { echo "::error::no $lib in the packages"; exit 1; }
  cp -L "$src" "$out/$lib"
  chmod 644 "$out/$lib"
done

cat > "$out/NOTICE.txt" <<'EOF'
The libraries in this folder are the ROCm 7.2.0 HIP runtime LLMJob Earn's AMD core loads, from
unmodified binary packages, so a HiveOS rig needn't install ROCm. A ROCm in /opt/rocm/lib is
used instead when the rig has one.

  libamdhip64.so.7            hip-runtime-amd 7.2.26015 (AMD, MIT)
  libhsa-runtime64.so.1       hsa-rocr 1.18.0 (AMD, University of Illinois/NCSA)
  libamd_comgr.so.3           comgr 3.0.0 (AMD, Apache-2.0 WITH LLVM-exception)
  librocprofiler-register.so.0  rocprofiler-register 0.6.0 (AMD, MIT)
    from https://repo.radeon.com/rocm/rhel8/7.2/main
  libnuma.so.1                numactl-libs 2.0.16 (LGPL-2.1)
  libelf.so.1                 elfutils-libelf 0.190 (LGPL-3.0-or-later or GPL-2.0-or-later)
  libdrm.so.2, libdrm_amdgpu.so.1  libdrm 2.4.115 (MIT)
    from Rocky Linux 8, https://dl.rockylinux.org/pub/rocky/8/BaseOS/x86_64/os/Packages
EOF

ls -l "$out"
du -sh "$out"
