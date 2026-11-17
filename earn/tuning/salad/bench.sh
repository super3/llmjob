#!/bin/bash
# Benchmarks PeakMiner, SRBMiner, WildRig Multi, OneZeroMiner and our miner on one Salad GPU, one after another, the
# way earn/tuning/benchmark.md's "How a host is tested" does: a 60 s unmeasured warm-up, then each miner for 5 minutes
# on Kryptex (prl-us.kryptex.network:7048; HeroMiners until 30 October, when the founder retired it). WildRig and
# OneZeroMiner joined on 9 November, as the third and fourth most downloaded Pearl miners; a box runs them only
# when ORDER names them ("wild", "ozm"). OneZeroMiner has no AMD support, so an AMD box leaves it out.
# bench.py starts it as a Salad container command and reads its "[bench]" lines from Salad's logs.
#
# Environment (set by bench.py, or by spot.sh in a fleet box): W wallet, ORDER (e.g. "peak,srb,wild,ozm,ours"), OURS_SECS (300, or 90 on a card
# our release has no build for), SRB_EXTRA (extra SRBMiner flags, usually empty), and on AMD BASE_SHA and
# VERIFY_SECS. Every line it prints for bench.py starts "[bench] HH:MM:SS".
#
# AMD: on a box with no nvidia-smi (Salad's AMD classes, run on the ROCm image) it reads the card with rocminfo
# and Salad's injected amd-smi (/opt/rocm-wsl/bin; the image's own amd-smi and rocm-smi can't reach a GPU under
# WSL2) and runs SRBMiner on AMD only. PeakMiner has no Pearl kernel for AMD and isn't downloaded or run. No
# release has an AMD build, so "ours" is the HIP core (earn/native/amd) at BASE_SHA, built here for the card's
# target and run through the app's CLI from source. That uses amdtest.sh's setup, fetch, hip, build, verify and
# mine_cli: bench.py's command fetches amdtest.sh beside this script, at the same commit. Before the miners run it
# prints the PC's device copy rate and runs the hit check (verify-hits.js) for VERIFY_SECS (600). If the core
# doesn't build or the hit check fails, our miner is left out and SRBMiner still runs. LD_LIBRARY_PATH is never
# set: Salad puts its ROCm-to-WSL bridge first on it.
#
# TAP=1 (bench.py's BENCH_TAP=1, NVIDIA only): every miner mines through sharetap.py, a relay beside this script that
# records each stratum line, and each measured run ends with that miner's shares decoded ("share ..." lines): m, n,
# k and rank, the tile's rows, and the operand bytes the proof carries. bench.py's command fetches sharetap.py at
# the same commit.
#
# VPOOL=BITS (NVIDIA only, with BASE_SHA): every miner mines on the stand-in pool (earn/native/probes/verify-pool.js
# at BASE_SHA, run here under Node 22) at the target 2^BITS instead of on Kryptex. It checks each share as the spec's
# verifier does and refuses duplicates, so each measured run ends with a "credit" line: the shares it accepted in
# the run's window, and the rate they prove, accepted x 2^(256-BITS) MACs over the window (TH/s), beside the miner's
# own figure. SECS and WARM set the run and warm-up lengths (300 and 60).
#
# OURS_SRC=1 (NVIDIA, with BASE_SHA): "ours" is not a release but the commit BASE_SHA, built here as CI builds
# it (.github/workflows/native-core.yml) for this card alone, and run through the CLI from source at that commit.
# nvtest.sh's setup, cuda and fetch do the toolkit and the checkout; bench.py's command fetches nvtest.sh beside
# this script. The toolkit is the one the app's core for this card is built with: CUDA 13.3 on an RTX 50 card on
# driver 580 or newer (pearl_core_cu13.node, coreVariant.js), 12.8 on every other. The rotation fold's cubin for
# the card (RTX 30, 40, 50) is built with probes/hand89/build_rot.py, which writes it only when its hash is one
# src/pearl_rot_<arch>.ref lists, and put beside the core, as a release ships it. FOLD=1 runs "ours" with
# --rotation-fold; "ours0" in ORDER is a second leg of the same build without it (the old kernel). After each
# leg a "fold" line says what the CLI said about the fold. A build that fails leaves both legs out.
export DEBIAN_FRONTEND=noninteractive
POOL_HOST=prl-us.kryptex.network; POOL_PORT=7048; SECS=${SECS:-300}; WARM=${WARM:-60}
# The versions a box runs: our release, and each competitor's newest with its published checksum. bench.py pins
# this file by commit, so a run keeps the versions it started with. Until 28 October: v0.6.0, PeakMiner 2.17.6
# and SRBMiner 3.7.1 (the GTX and AMD tables); until 9 November: v0.6.9, PeakMiner 2.20.3 and SRBMiner 3.8.2.
# PeakMiner, WildRig and OneZeroMiner publish no checksums, so theirs are the SHA-256 of the files as downloaded
# on 9 November.
VER=${VER:-v0.6.13}   # VER=v0.6.14 etc. runs another release of ours
PEAK_VER=2.20.9; PEAK_SHA256=2d504d9217f021b84b8f9915ac3c45cfe951558d13a0dc25c0e4fb1ace46c9d3
# SRBMiner 3.8.6 needs wget in the image. Without it, on Salad, it exited 73 within seconds and printed only its
# interactive setup prompt ("Please enter a text with length between 1 and 50"), even for --list-algorithms (two
# 2080 Tis, 9 November), so the tables of 9 to 10 November ran 3.8.2. With wget installed (the apt line below) it
# ran on all four RX 7900 boxes of the 11 November re-run. Not yet run this way on an NVIDIA box.
# 3.8.7 (11 November) changed pearlhash only on NVIDIA (RTX 30, 40, 50); on RDNA 2 it read the same as 3.8.6 (7.382
# against 7.380 TH/s on an RX 6600, 12 November). Until 12 November: 3.8.6, md5 38128fa15afff3c488e53726aa251249.
# 3.8.8 (13 November) improves pearlhash on NVIDIA RTX 30-50 and AMD RDNA 3-4. Until 14 November: 3.8.7, md5
# 7471402c42ae0b347b1eebc731eef6ce; OneZeroMiner 1.8.5, sha256 c1234876...ab6a48; PeakMiner 2.20.7, sha256
# 38fc5c4d...bcb3d (those two as downloaded on 14 November).
# PeakMiner 2.20.9 (15 November, "Pearl hashrate efficiency improved on RTX 20/30/40/50") and WildRig 0.52.3 (14
# November) now publish GitHub asset digests; these are those. Until 15 November: PeakMiner 2.20.8, sha256
# 13b556c8...b0ffb; WildRig 0.51.9, sha256 7444c6c4...5bb44.
# SRBMiner publishes an MD5 file with each release, and GitHub now gives a SHA-256 digest of each asset too. The
# check uses the SHA-256, like the other miners: on 16 November the 3.8.8 download hashed to both GitHub's digest
# and the release's own MD5 (2650fbe22162528db96640b6fba299d6).
SRB_VER=3.8.9; SRB_SHA256=186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1
WILD_VER=0.52.3; WILD_SHA256=7308c95f7c8c4c0d82aade61bb4ab3366decae2b9d04ac47b8c1dc6d1bdd2c44
OZM_VER=1.8.6; OZM_SHA256=47ea652dcc7bacd49102203aab149dd6f678471e5fd69e68622363e0747c0f00
REL=https://github.com/super3/llmjob/releases/download/$VER
say() { echo "[bench] $(date -u +%T) $*"; }
fail() { say "FAIL $*"; say "all done"; sleep infinity; }
AMD=; command -v nvidia-smi >/dev/null 2>&1 || AMD=1
SMI=/opt/rocm-wsl/bin/amd-smi
export PATH="$PATH:/opt/rocm/bin"   # append only: Salad's /opt/rocm-wsl/bin stays first

# drop NAME [WHY]: leaves a miner out of ORDER, and says why if given.
drop() {
  [ -n "${2:-}" ] && say "skip $1: $2"
  ORDER=$(echo "${ORDER:-peak,srb,ours}" | tr , '\n' | grep -vx "$1" | paste -sd, -)
}
# has NAME: whether ORDER runs that miner.
has() { case ",${ORDER:-peak,srb,ours}," in *",$1,"*) return 0 ;; esac; return 1; }

# copy_rate LOG: the GB/s of amdtest_hip.cpp's timed device copy ("copy: 564 GB/s (read + write)").
copy_rate() { grep -oE '^copy: [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+$' | head -1; }

# hitcheck LOG: one short line from verify-hits.js's JSON (the last "{" line of its log). PASS only if the JSON
# says PASS and no hit failed; returns 1 otherwise. t_verify prints the JSON too, in pieces, as it's long.
hitcheck() {
  local j v
  j=$(grep '^{' "$1" 2>/dev/null | tail -1)
  v=$(echo "$j" | jq -r 'if .error then "error=\(.error | tostring | .[0:160])"
        else "hits=\(.hits) verified=\(.verified) failed=\(.failed) salts=\(.salts)"
          + (if (.fails | length) > 0 then " why=\(.fails[0].why | tostring | .[0:120])" else "" end) end' 2>/dev/null)
  [ -n "$v" ] || v="error=no result: $(grep -v '^\s*$' "$1" 2>/dev/null | tail -1 | tr -d '\r' | head -c 160)"
  if [ "$(echo "$j" | jq -r '.PASS == true and .failed == 0' 2>/dev/null)" = true ]; then
    say "hitcheck PASS secs=${VERIFY_SECS:-600} $v"; return 0
  fi
  say "hitcheck FAIL secs=${VERIFY_SECS:-600} $v"; return 1
}

# Our miner on AMD, up to where it can mine: amdtest.sh's setup and fetch (BASE_SHA into /opt/llmjob), its hip
# task for the PC's copy rate, the core built for this card's target, and the hit check. A failed build or hit
# check leaves our miner out; any other failure ends the box, as a failed download does.
amd_ours() {
  local keep c; keep=$(declare -f say fail)
  AMDTEST_STAGE=lib . "$(dirname "$0")/amdtest.sh" || fail "no amdtest.sh beside bench.sh"
  eval "$keep"   # bench's own say and fail again, so every line is a [bench] line
  [ -n "${BASE_SHA:-}" ] || fail "no BASE_SHA"
  setup; fetch; cd /opt/b || fail "no /opt/b"
  ARCH=$(target); [ -n "$ARCH" ] || fail "no gfx target from rocminfo"
  t_hip; c=$(copy_rate /root/hip.log)
  say "copy ${c:-?} GB/s"
  say "start build $ARCH"
  t_build "$ARCH" || { drop ours "build failed"; return; }
  VERIFY_SECS=${VERIFY_SECS:-600}
  say "start hitcheck ${VERIFY_SECS}s"
  t_verify "$VERIFY_SECS"
  hitcheck /root/verify.log || drop ours "hit check failed"
  CORE="HIP core $(git -C "$R" rev-parse --short=12 HEAD) for $ARCH"
}

# Our miner on NVIDIA at BASE_SHA (OURS_SRC=1): the core built here for this card, the fold's cubin beside it, the
# CLI's packages in /opt/nm. Sets NVCORE (the core's path) and CORE (what the "ready" line calls it).
nv_ours() {
  local keep cc drv nd ndsha t0 C K f arch rot
  keep=$(declare -f say fail)
  cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
  drv=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -cd 0-9)
  [ -n "$cc" ] || fail "no compute capability from nvidia-smi"
  export CUDA_VER=12.8
  [ "${cc:0:2}" = 12 ] && [ "${drv:-0}" -ge 580 ] && CUDA_VER=13.3
  NVTEST_STAGE=lib . "$(dirname "$0")/nvtest.sh" || fail "no nvtest.sh beside bench.sh"
  set +u   # nvtest.sh sets -u; this script reads unset variables as empty
  eval "$keep"
  [ -n "${BASE_SHA:-}" ] || fail "no BASE_SHA"
  t0=$(date +%s)
  say "start build sm_$cc CUDA $CUDA_VER"
  setup; cuda; fetch
  # nvdisasm for build_rot.py (the redistrib JSONs' checksums, as nv30box.sh and nv50box.sh have them).
  if [ "$CUDA_VER" = 13.3 ]; then
    nd=cuda_nvdisasm-linux-x86_64-13.3.73-archive.tar.xz; ndsha=38b2eca90c0ffa2ce815bd392077f68b7fe94992ea0ec0f8e85df30ca01fc4e9
  else
    nd=cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz; ndsha=8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae
  fi
  if [ ! -x "$CUDA/bin/nvdisasm" ]; then
    { curl -fsSL --retry 3 "$RD/cuda_nvdisasm/linux-x86_64/$nd" -o /root/nvd.tar.xz \
        && echo "$ndsha  /root/nvd.tar.xz" | sha256sum -c --quiet - \
        && tar -xJf /root/nvd.tar.xz -C "$CUDA" --strip-components=1; } > /root/nvd.log 2>&1 || say "no nvdisasm: no fold"
  fi
  # Node 22 and the CLI's packages, and node-gyp for the addon.
  if [ "$(/opt/node/bin/node -v 2>/dev/null | cut -d. -f1)" != v22 ]; then
    mkdir -p /opt/node && curl -fsSL --retry 3 https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.gz \
      | tar xz -C /opt/node --strip-components=1 || fail "node download"
  fi
  export PATH="/opt/node/bin:$CUDA/bin:$PATH"
  npm install --prefix /opt/nm --no-audit --no-fund --loglevel=error node-addon-api node-gyp tweetnacl tweetnacl-util \
    > /root/npm.log 2>&1 || fail "npm install: $(tail -1 /root/npm.log)"
  # The addon, as native-core.yml builds it, for this card only. The 12.8 build has the RTX 20 folds at
  # register-usage level 8 too (t75), as CI's does.
  K=$R/earn/native
  cd "$K" && mkdir -p cuda-build && ln -sfn /opt/nm/node_modules node_modules || fail "no $K"
  C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_$cc,code=sm_$cc"
  { nvcc $C -c src/pearl_kernel.cu -o cuda-build/pearl_kernel.o > /root/kernel.log 2>&1; } \
    || { drop ours "build failed: $(grep -m1 -i error /root/kernel.log | cut -c1-200)"; drop ours0; cd /opt/b; return; }
  T75=; f="cuda-build/pearl_kernel.o cuda-build/pearl_host.o"
  if [ "$CUDA_VER" = 12.8 ]; then
    nvcc -O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_75,code=sm_75 \
      -Xptxas --register-usage-level=8 -c src/pearl_kernel_t75.cu -o cuda-build/pearl_kernel_t75.o > /root/t75.log 2>&1 \
      || { drop ours "build failed: t75"; drop ours0; cd /opt/b; return; }
    T75=-DPEARL_T75_RUL8=1; f="$f cuda-build/pearl_kernel_t75.o"
  fi
  nvcc $C $T75 -c src/pearl_host.cu -o cuda-build/pearl_host.o > /root/host.log 2>&1 \
    || { drop ours "build failed: $(grep -m1 -i error /root/host.log | cut -c1-200)"; drop ours0; cd /opt/b; return; }
  ar rcs cuda-build/libpearl_cuda.a $f
  # node-gyp downloads Node's headers on its first build. On an RTX 5080 PC (15 November) that download failed
  # ("gyp WARN install got an error, rolling back install") and ours was left out, so a failed build gets one more
  # try. Every failed build goes back to /opt/b before it returns: the miners are started from there, and until
  # 15 November PeakMiner then exited 127 ("./peak: No such file or directory") on such a box.
  { CUDA_PATH=$CUDA /opt/nm/node_modules/.bin/node-gyp rebuild > /root/gyp.log 2>&1 \
      || { say "node-gyp failed once, retrying: $(grep -m1 -i error /root/gyp.log | cut -c1-160)"; sleep 10
           CUDA_PATH=$CUDA /opt/nm/node_modules/.bin/node-gyp rebuild > /root/gyp.log 2>&1; }; } \
    || { drop ours "build failed: node-gyp $(grep -m1 -i error /root/gyp.log | cut -c1-160)"; drop ours0; cd /opt/b; return; }
  NVCORE=$K/build/Release/pearl_core.node
  if [ "$CUDA_VER" = 13.3 ]; then cp "$NVCORE" "$K/build/Release/pearl_core_cu13.node"; NVCORE=$K/build/Release/pearl_core_cu13.node; fi
  say "built $(basename "$NVCORE") for sm_$cc with CUDA $CUDA_VER in $(( $(date +%s) - t0 ))s"
  # The fold's cubin: RTX 30 and 40 from the 12.8 toolkit, RTX 50 from 13.3, as CI builds them.
  arch=; case "$cc:$CUDA_VER" in 86:12.8|89:12.8|120:13.3) arch=sm_$cc ;; esac
  if [ -n "$arch" ] && [ -x "$CUDA/bin/nvdisasm" ]; then
    if python3 -X utf8 probes/hand89/build_rot.py nvcc "$CUDA/bin/nvdisasm" /root/rot --arch "$arch" > /root/rot.log 2>&1 \
        && cp "/root/rot/pearl_rot_sm$cc.cubin" "$(dirname "$NVCORE")/"; then
      # build_rot.py names it pearl_rot_sm89.cubin (no "_"), the name the core looks for.
      rot="pearl_rot_sm$cc.cubin $(python3 probes/hand89/asm89.py hash "/root/rot/pearl_rot_sm$cc.cubin" 2>/dev/null | cut -c1-12)"
      say "built $rot"
    else
      say "no fold cubin: $(grep -v '^\s*$' /root/rot.log | tail -1 | tr -cd ' -~' | cut -c1-240)"
    fi
  fi
  # The RTX 5090's own file, pearl_rot_sm120w.cubin, as native-core.yml builds it: the core takes it on 128 SMs or
  # more on compute 12.0 when it is beside the module, the sm_120 file elsewhere. Until 14 November this script
  # built only the sm_120 file, so a 5090 here mined on that one, not on the file a release ships for it.
  if [ "$arch" = sm_120 ] && [ -f src/pearl_rot_sm120w.ref ]; then
    if python3 -X utf8 probes/hand89/build_rot.py nvcc "$CUDA/bin/nvdisasm" /root/rotw --arch sm_120w > /root/rotw.log 2>&1 \
        && cp /root/rotw/pearl_rot_sm120w.cubin "$(dirname "$NVCORE")/"; then
      rot="$rot, pearl_rot_sm120w.cubin $(python3 probes/hand89/asm89.py hash /root/rotw/pearl_rot_sm120w.cubin 2>/dev/null | cut -c1-12)"
      say "built pearl_rot_sm120w.cubin"
    else
      say "no 5090 fold cubin: $(grep -v '^\s*$' /root/rotw.log | tail -1 | tr -cd ' -~' | cut -c1-240)"
    fi
  fi
  CORE="$(git -C "$R" rev-parse --short=12 HEAD) built for sm_$cc (CUDA $CUDA_VER${rot:+, $rot})"
  cd /opt/b || fail "no /opt/b"
}

# WildRig mines NVIDIA cards through OpenCL. Salad's PCs run containers in WSL2, whose NVIDIA driver has no OpenCL
# library, so there it finds the card, logs "Failed to find OpenCL platform" and hashes 0 (a 3090 Ti, 9 November).
# A box without the driver's OpenCL library leaves it out rather than spend 5 minutes on a 0. One that has it gets
# the OpenCL loader, which Salad's CUDA image lacks, pointed at that library.
# Any one of the three places will do. This used to be a single ls over all three, which fails when any one is
# missing, and a box never has all three, so every NVIDIA box read as "no OpenCL".
nv_opencl() {
  local f
  for f in /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so* /usr/lib64/libnvidia-opencl.so* \
      /usr/lib/wsl/lib/libnvidia-opencl.so*; do
    [ -e "$f" ] && return 0
  done
  return 1
}
if [ -z "$AMD" ] && has wild && ! nv_opencl; then
  drop wild "no OpenCL"
fi
OCL=; [ -z "$AMD" ] && has wild && OCL=1
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq --no-install-recommends curl ca-certificates jq procps wget \
  ${AMD:+clinfo} ${TAP:+python3} ${VPOOL:+git} ${OCL:+ocl-icd-libopencl1} >/dev/null 2>&1 || fail "apt-get install"
if [ -n "$OCL" ] && [ ! -f /etc/OpenCL/vendors/nvidia.icd ]; then
  mkdir -p /etc/OpenCL/vendors && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd
fi
if [ -z "$AMD" ]; then
  say "card $(nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,memory.total,compute_cap --format=csv,noheader | head -1)"
else
  # The same comma layout as nvidia-smi's: name, driver, power limit, default limit, memory, target.
  test -c /dev/dxg || fail "no /dev/dxg"
  ri=$(timeout 60 rocminfo 2>&1)
  name=$(echo "$ri" | awk -F': +' '/Marketing Name:/{m=$2} /Device Type:/{if ($2 ~ /GPU/) {print m; exit}}' | sed 's/ *$//')
  gfx=$(echo "$ri" | grep -m1 -oE 'gfx[0-9a-f]+')
  [ -n "$name" ] || fail "no AMD GPU: $(echo "$ri" | grep -m1 -E 'HSA_STATUS|rror' | tail -c 200)"
  st=$(timeout 20 $SMI static --asic --board --limit --driver --vram --json 2>/dev/null)
  echo "$st" | tr -s ' \n' ' ' | fold -w 900 | while read -r l; do say "amdsmi $l"; done   # its keys aren't documented
  num() { echo "$st" | jq -r --arg p "$1" 'first(paths(numbers) as $q | select($q|map(tostring)|join(".")|test($p;"i")) | getpath($q)) // empty' 2>/dev/null; }
  str() { echo "$st" | jq -r --arg p "$1" 'first(paths(strings) as $q | select($q|map(tostring)|join(".")|test($p;"i")) | getpath($q)) // empty' 2>/dev/null; }
  pl=$(num 'power_limit|power_cap'); vram=$(num 'vram.*(size|total)'); drv=$(str 'driver.*version')
  say "card $name, ${drv:-?}, ${pl:+$pl W}, ${pl:+$pl W}, ${vram:-?} MiB, $gfx"
  say "opencl $(clinfo -l 2>&1 | tr -s ' \n' ' ' | head -c 300)"
  say "gpucols $(timeout 5 $SMI monitor -p -t -u --csv 2>&1 | head -1)"
  drop peak   # no Pearl kernel for AMD
  drop ozm    # NVIDIA only
fi

mkdir -p /opt/b && cd /opt/b || fail "no /opt/b"
# The two miners from their official releases, checked against the published checksums.
if [ -z "$AMD" ]; then
  curl -fsSL --retry 3 -o peak \
    "https://github.com/peakminer/peakminer/releases/download/v$PEAK_VER/peakminer-$PEAK_VER-linux-x86_64" \
    || fail "download PeakMiner"
  echo "$PEAK_SHA256  peak" | sha256sum -c --quiet || fail "PeakMiner checksum"
  chmod +x peak
fi
curl -fsSL --retry 3 -o srb.tgz \
  "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
  || fail "download SRBMiner"
echo "$SRB_SHA256  srb.tgz" | sha256sum -c --quiet || fail "SRBMiner checksum"
tar xzf srb.tgz && SRB=$(ls -d /opt/b/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI && [ -x "$SRB" ] || fail "unpack SRBMiner"
if has wild; then
  curl -fsSL --retry 3 -o wild.tgz \
    "https://github.com/andru-kun/wildrig-multi/releases/download/$WILD_VER/wildrig-multi-linux-$WILD_VER.tar.gz" \
    || fail "download WildRig"
  echo "$WILD_SHA256  wild.tgz" | sha256sum -c --quiet || fail "WildRig checksum"
  mkdir -p /opt/b/wild && tar xzf wild.tgz -C /opt/b/wild && [ -x /opt/b/wild/wildrig-multi ] || fail "unpack WildRig"
fi
if has ozm || has ozm1 || has ozm2 || has ozm3; then
  curl -fsSL --retry 3 -o ozm.tgz \
    "https://github.com/OneZeroMiner/onezerominer/releases/download/v$OZM_VER/onezerominer-linux-$OZM_VER.tar.gz" \
    || fail "download OneZeroMiner"
  echo "$OZM_SHA256  ozm.tgz" | sha256sum -c --quiet || fail "OneZeroMiner checksum"
  tar xzf ozm.tgz && [ -x /opt/b/onezerominer-linux/onezerominer ] || fail "unpack OneZeroMiner"
fi
# Our miner: the published release, as a user gets it; on AMD, the HIP core built here; with OURS_SRC=1, BASE_SHA
# built here.
case ",${ORDER:-peak,srb,ours}," in *,ours,*|*,ours0,*|*,oursa,*)
  if [ -z "$AMD" ] && [ "${OURS_SRC:-}" = 1 ]; then
    nv_ours
  elif [ -z "$AMD" ] && [ -n "${OURS_REL_DIR:-}" ]; then
    # A spot check in a fleet box (spot.sh): the release the box already mines with, rotation-fold cubins included,
    # copied from box.sh's folder rather than downloaded, so "ours" is the build the fleet runs.
    cp "$OURS_REL_DIR/cli" "$OURS_REL_DIR"/pearl_core*.node . && chmod +x cli || fail "copy ours from $OURS_REL_DIR"
    cp "$OURS_REL_DIR"/pearl_rot_sm*.cubin . 2>/dev/null
    say "ours: the box's release $VER from $OURS_REL_DIR"
  elif [ -z "$AMD" ]; then
    for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do
      curl -fsSL --retry 3 -o "$f" "$REL/$f" || fail "download $f"
    done
    mv llmjob-earn-cli-linux cli && chmod +x cli
    # The rotation fold's cubins, which a release ships beside the core. Without them the core mines on the normal
    # kernel, about 5% slower on RTX 30, 40 and 50 (found 16 Nov). Releases before the fold have none: skipped.
    for f in pearl_rot_sm86.cubin pearl_rot_sm89.cubin pearl_rot_sm120.cubin pearl_rot_sm120w.cubin; do
      curl -fsSL --retry 3 -o "$f" "$REL/$f" 2>/dev/null || rm -f "$f"
    done
    # OURS_CORE (a measuring option, nv20box.sh): the release's CLI with another pearl_core.node, so the rate of a
    # core built from a branch can be read the same way as the other miners'.
    if [ -n "${OURS_CORE:-}" ]; then
      # The release's cubins belong to the release's core, so a core from a branch runs without them, as before.
      cp "$OURS_CORE" pearl_core.node && rm -f pearl_core_cu13.node pearl_rot_sm*.cubin || fail "OURS_CORE $OURS_CORE"
      say "ours: the release's CLI with the core $OURS_CORE"
    fi
  else
    amd_ours
  fi ;;
esac
# VPOOL: the stand-in pool on 127.0.0.1:3334, for the whole box. Its share log has a line per share:
# "<ms> <connection> <k> ok|<reason> <hs>".
if [ -n "${VPOOL:-}" ] && [ -z "$AMD" ]; then
  [ -n "${BASE_SHA:-}" ] || fail "VPOOL needs BASE_SHA"
  mkdir -p /opt/node && curl -fsSL --retry 3 https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.gz \
    | tar xz -C /opt/node --strip-components=1 || fail "node download"
  # With OURS_SRC=1 nv_ours has already checked BASE_SHA out there and built our core in it: keep that checkout.
  # Until 15 November this always fetched afresh, which deleted the core, and every "ours" leg exited at once
  # with "pearl_core.node not found" (goal-5070ti).
  if [ "$(git -C /opt/llmjob rev-parse HEAD 2>/dev/null)" != "$BASE_SHA" ]; then
    { rm -rf /opt/llmjob && mkdir -p /opt/llmjob && cd /opt/llmjob && git init -q \
        && git remote add origin https://github.com/super3/llmjob.git && git fetch -q --depth 1 origin "$BASE_SHA" \
        && git checkout -q FETCH_HEAD; } > /opt/b/git.log 2>&1 || fail "fetch $BASE_SHA: $(tail -c 200 /opt/b/git.log)"
  fi
  cd /opt/b
  PEARL_SHARE_LOG=/opt/b/vp.shares /opt/node/bin/node /opt/llmjob/earn/native/probes/verify-pool.js serve 3334 864000 \
    "$VPOOL" > /opt/b/vp.out 2> /opt/b/vp.err &
  VP_PID=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q listening /opt/b/vp.err && break; sleep 1; done
  grep -q listening /opt/b/vp.err || fail "verify-pool: $(tail -c 200 /opt/b/vp.err)"
  say "$(grep -m1 listening /opt/b/vp.err) (commit ${BASE_SHA:0:12})"
  POOL_HOST=127.0.0.1; POOL_PORT=3334
  # VERIFY_SECS with VPOOL: the hit check (verify-hits.js) on each of the release's cores this card can load, first.
  if [ -n "${VERIFY_SECS:-}" ] && has ours; then
    for c in pearl_core_cu13.node pearl_core.node; do
      say "start hitcheck $c ${VERIFY_SECS}s"
      timeout $((VERIFY_SECS + 300)) /opt/node/bin/node /opt/llmjob/earn/native/probes/verify-hits.js "/opt/b/$c"         "$VERIFY_SECS" > /opt/b/verify.log 2>&1
      hitcheck /opt/b/verify.log; say "hitcheck $c $(grep '^{' /opt/b/verify.log | tail -1 | jq -c '{device, k, thLast, thMax}' 2>/dev/null | head -c 300)"
    done
  fi
fi
# credit NAME T0 T1: the stand-in's shares in [T0, T1) (epoch seconds), as one line: accepted, duplicates, other
# rejects, the rate the accepted ones prove over the window and over the window less its first 60 s, accepted a
# minute, and the mean hs (the hashrate a miner puts in its submits, if it does).
credit() {
  awk -v n="$1" -v a="$2" -v b="$3" -v bits="$VPOOL" '
    BEGIN { a *= 1000; b *= 1000; w = 2 ^ (256 - bits) }
    $1 >= a && $1 < b {
      if ($4 == "ok") { ok++; if ($1 >= a + 60000) ok60++; m = int(($1 - a) / 60000); pm[m]++ }
      else if ($4 == "\"duplicate_share\"") dup++
      else { rej++; why = $4 }
      if ($5 != "null") { hs += $5; nhs++ }
    }
    END {
      s = (b - a) / 1000; mins = ""
      for (i = 0; i < int(s / 60 + 0.999); i++) mins = mins (i ? "," : "") (pm[i] + 0)
      printf "credit %s secs=%d bits=%d ok=%d dup=%d rej=%d ths=%.2f ths60=%.2f hs=%s permin=%s%s\n", n, s, bits, ok,
        dup, rej, ok * w / s / 1e12, (s > 60 ? ok60 * w / (s - 60) / 1e12 : 0), (nhs ? sprintf("%.4g", hs / nhs) : "none"),
        mins, (rej ? " why=" substr(why, 1, 120) : "")
    }' /opt/b/vp.shares 2>/dev/null
}
# TAP=1: the miners connect to the relay, which passes everything on to the pool and records it.
if [ -n "${TAP:-}" ] && [ -z "$AMD" ]; then
  [ -f /root/sharetap.py ] || fail "no sharetap.py beside bench.sh"
  python3 /root/sharetap.py tap 3333 "$POOL_HOST" "$POOL_PORT" /opt/b/tap.jsonl > /opt/b/tap.log 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q listening /opt/b/tap.log && break; sleep 1; done
  grep -q listening /opt/b/tap.log || fail "sharetap: $(tail -c 200 /opt/b/tap.log)"
  say "$(cat /opt/b/tap.log)"
  POOL_HOST=127.0.0.1; POOL_PORT=3333
fi
others=; has wild && others="$others, WildRig $WILD_VER"; has ozm && others="$others, OneZeroMiner $OZM_VER"
if [ -z "$AMD" ]; then
  say "ready PeakMiner $PEAK_VER, SRBMiner $SRB_VER$others, ours ${CORE:-$VER}"
else
  r="SRBMiner $SRB_VER$others"; case ",$ORDER," in *,ours,*) r="$r, ours $CORE" ;; esac
  say "ready $r"
  say "srb devices $(cd "$(dirname "$SRB")" && timeout 60 "$SRB" --list-devices $SRB_EXTRA 2>&1 | tr -s '\r\n ' ' ' | tail -c 400)"
fi

peak() { cd /opt/b && ./peak -o "stratum+tcp://$POOL_HOST:$POOL_PORT" -u "$W.bench-peak" -p x -c pearl; }
srb()  { cd "$(dirname "$SRB")" && "$SRB" --algorithm-gpu pearlhash --pool "$POOL_HOST:$POOL_PORT" --wallet "$W" \
           --worker bench-srb --disable-cpu --api-enable --api-port 21550 ${AMD:+--disable-gpu-nvidia --disable-gpu-intel} \
           $SRB_EXTRA; }
# WildRig and OneZeroMiner, run as their own Pearl start scripts do, each with its API on, as their HiveOS packages
# read them (h-stats.sh).
wild() { cd /opt/b/wild && ./wildrig-multi --algo pearlhash --url "stratum+tcp://$POOL_HOST:$POOL_PORT" \
           --user "$W.bench-wild" --pass x --api-port 21551; }
ozm()  { cd /opt/b/onezerominer-linux && ./onezerominer -a pearl -w "$W" --worker bench-ozm \
           -o "stratum+tcp://$POOL_HOST:$POOL_PORT" --api-port 21552 "$@"; }
# ozm1 to ozm3 (a measuring option, goal-nv20q): OneZeroMiner with OZM1_X .. OZM3_X added to its flags, e.g.
# OZM1_X="--power-save 1", so its own settings can be read beside its default run.
ozm1() { ozm ${OZM1_X:-}; }
ozm2() { ozm ${OZM2_X:-}; }
ozm3() { ozm ${OZM3_X:-}; }
# Our CLI picks Kryptex's "wallet/worker" login from the host name, so behind the TAP relay (127.0.0.1) Kryptex
# credits the same wallet but names the worker "worker" rather than bench-ours.
ours() { ./cli -a "$W" --mode mining -w bench-ours --no-report --no-serve --no-update \
           --pool "$POOL_HOST:$POOL_PORT"; }
# On AMD: the CLI from source with the core built above (amdtest.sh's mine_cli), on SRBMiner's pool.
[ -n "$AMD" ] && ours() { WALLET=$W POOL=$POOL_HOST:$POOL_PORT WORKER=bench-ours mine_cli; }
# OURS_SRC=1: the CLI from source at BASE_SHA with the core built above; "ours" with the fold if FOLD=1, "ours0"
# without it.
src_cli() { cd "$R/earn" && PEARL_CORE_PATH=$NVCORE NODE_PATH=/opt/nm/node_modules exec node src/cli/earn-cli.js \
              -a "$W" --mode mining --no-report --no-serve --no-update --pool "$POOL_HOST:$POOL_PORT" "$@"; }
if [ -z "$AMD" ] && [ "${OURS_SRC:-}" = 1 ]; then
  ours() { src_cli -w bench-ours ${FOLD:+--rotation-fold}; }
  # Since 0.7.0 the fold is on by default; PEARL_FOLD_ROT=0 keeps it off, and an older BASE_SHA reads it the same.
  ours0() { PEARL_FOLD_ROT=0 src_cli -w bench-ours0; }
  # "oursa": the same build with OURS_A's environment (e.g. "PEARL_HIT_DEDUP=0 PEARL_ROT_SEARCH_THREADS=1", a switch
  # back to how the core worked before a change), so ORDER=ours,oursa,ours,oursa races a host-side change on one
  # build. ROTLOG=1 sets PEARL_ROT_LOG=1 in these legs: the core then logs each job's draw and how long a salt mined
  # in the definition's order, and each leg ends with a "rotlog" line summing them.
  oursa() { ( [ -n "${OURS_A:-}" ] && export ${OURS_A}; src_cli -w bench-oursa ${FOLD:+--rotation-fold} ); }
  [ "${ROTLOG:-}" = 1 ] && export PEARL_ROT_LOG=1
fi
stop_all() { pkill -f '\./peak -o'; pkill -f SRBMiner-MULTI; pkill -f wildrig-multi; pkill -f onezerominer; pkill -f '\./cli -a'; { [ -n "$AMD" ] || [ "${OURS_SRC:-}" = 1 ]; } && pkill -f 'src/cli/earn-cli\.js'; sleep 8; }

# One sample: the miner's own hashrate (H/s), accepted and rejected shares, and the card's power, clock and temperature,
# then (NVIDIA) its memory clock, GPU use, memory-controller use and memory in use (MiB), in gpu=.
sample() {
  local name=$1 t=$2 hs="" acc="" rej=""
  case $name in
    srb)  read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:21550 \
            | jq -r '[.algorithms[0].hashrate.gpu.total, .algorithms[0].shares.accepted, .algorithms[0].shares.rejected] | @tsv' 2>/dev/null) ;;
    peak) read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:4068/summary | jq -r '
            def num(p): [paths(numbers) as $q | select(($q|map(tostring)|join(".")) | test(p; "i")) | getpath($q)];
            [(num("hash") | max), (num("accept") | max), (num("reject") | max)] | @tsv' 2>/dev/null) ;;
    wild) read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:21551 | jq -r '
            [.hashrate.total[0], .results.shares_good, (.results.shares_total - .results.shares_good)] | @tsv' 2>/dev/null) ;;
    ozm|ozm[1-3])  read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:21552 | jq -r '.algos[0] |
            [.total_hashrate, .total_accepted_shares, (.total_rejected_shares + (.total_invalid_shares // 0))] | @tsv' 2>/dev/null) ;;
    ours|ours0|oursa) local l; l=$(tr '\r' '\n' < "$name.log" | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1)
          [ -n "$l" ] && { hs=$(echo "$l" | awk '{printf "%.0f", $1*1e12}'); acc=$(echo "$l" | awk '{gsub(",","",$4); print $4}')
                           rej=$(echo "$l" | awk '{gsub(",","",$7); print $7}'); } ;;
  esac
  local g
  if [ -z "$AMD" ]; then g=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,clocks.mem,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  else g=$(timeout 5 $SMI monitor -p -t -u --csv 2>/dev/null | tail -1 | tr -d ' '); fi
  say "sample $name t=$t hs=${hs:-?} acc=${acc:-?} rej=${rej:-?} gpu=$g"
}

# smisum NAME PID (SMI1=1): stops the once-a-second reader and says what it saw past the first 60 s: the mean SM and
# memory clock, watts, temperature and memory-controller use, the share of seconds each clock event reason (the
# driver's bit mask, e.g. 0x4 the power cap) was set, and the P-states seen. For OneZeroMiner legs, the lines of its
# log that name a kernel, clock, memory, power or a failure.
smisum() {
  [ -n "${2:-}" ] || return 0
  kill "$2" 2>/dev/null
  say "smi $1 $(awk -F', *' 'NR > 60 && $1 ~ /^[0-9]/ {n++; c += $1; m += $2; w += $3; t += $4; r[$5]++; p[$6]++; u += $7}
    END {if (!n) {print "n=0"; exit}; printf "n=%d mhz=%.0f mem=%.0f w=%.1f c=%.1f memutil=%.0f reasons=", n, c/n, m/n, w/n, t/n, u/n
         for (k in r) printf "%s:%.0f%%,", k, 100*r[k]/n; printf " pstate="; for (k in p) printf "%s:%d,", k, p[k]}' "/opt/b/$1.smi")"
  case $1 in ozm*) tr '\r' '\n' < "/opt/b/$1.log" | sed 's/\x1b\[[0-9;]*m//g' | LC_ALL=C tr -cd ' -~\n' \
      | grep -a -i -E 'kernel|clock|mclk|mem|power|lock|fail|error|warn|permission|turing|2080|2060|2070|intensity' \
      | awk '!s[$0]++' | head -25 | while IFS= read -r l; do say "ozmlog $1 $(echo "$l" | cut -c1-240)"; done
    tr '\r' '\n' < "/opt/b/$1.log" | sed 's/\x1b\[[0-9;]*m//g' | LC_ALL=C tr -cd ' -~\n' | grep -a -v '^\s*$' | tail -14 \
      | while IFS= read -r l; do say "ozmtail $1 $(echo "$l" | cut -c1-240)"; done ;;
  esac
}

# Which core our CLI loaded: its last "Pearl core:" line (a CUDA 13 build that fails to start is followed by the 12.8
# one it fell back to). A driver under 580 gets the 12.8 build, 3.2% slower on a 5090; the bench logged only the driver
# before 10 November, which left that to be worked out. The " · " after the label becomes "; " (ASCII only).
loaded() { say "loaded $(tr '\r' '\n' < /opt/b/ours.log | grep -a 'Pearl core: ' | tail -1 \
             | LC_ALL=C tr -cd ' -~' | sed 's/.*Pearl core: //; s/  /; /' | tail -c 300)"; }
# OURS_SRC=1: what the CLI said about the rotation fold in leg NAME, its last line that names the fold ("rotation
# fold passed its check", "asked for but not used: ...", "failed its check"), or "none" when it said nothing.
foldline() { local f; f=$(tr '\r' '\n' < "/opt/b/$1.log" | grep -a -i 'rotation fold' | tail -1 | LC_ALL=C tr -cd ' -~' \
               | sed 's/^.*rotation fold/rotation fold/' | tail -c 300)
             say "fold $1 ${f:-none}"; }

# Our CLI's 1 s lines past the first 60 (as linemean): their median, and the lines under 98.5% of it, with the
# seconds of full rate they lose. A job's draw or a share's proof shows as such a line.
linedips() {
  tr '\r' '\n' < "$1" | grep -oE '[0-9.]+ TH/s · ' | awk 'NR > 60 {print $1}' | sort -g \
    | awk '{v[NR] = $1} END {if (!NR) exit; m = v[int((NR + 1) / 2)]; for (i = 1; i <= NR; i++) if (v[i] < 0.985 * m) {d++; l += (m - v[i]) / m}
           printf "median=%.2f dips=%d lost_s=%.2f", m, d, l}'
}

run() {
  local name=$1 secs=$2 measured=$3
  say "start $name ${secs}s"
  ( $name > "/opt/b/$name.log" 2>&1; echo $? > "/opt/b/$name.rc" ) & local pid=$! t0; t0=$(date +%s)
  # SMI1=1 (NVIDIA): the card read once a second through the run, summed at its end in an "smi" line.
  local spid=""
  if [ "$measured" = 1 ] && [ -z "$AMD" ] && [ "${SMI1:-}" = 1 ]; then
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,pstate,utilization.memory \
      --format=csv,noheader,nounits -lms 1000 > "/opt/b/$name.smi" 2>&1 & spid=$!
  fi
  while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
    sleep 15; local t=$(( $(date +%s) - t0 ))
    if ! kill -0 $pid 2>/dev/null; then
      [ "$measured" = 1 ] && [ -n "${VP_PID:-}" ] && say "$(credit "$name" "$t0" "$(date +%s)")"
      say "exited $name t=$t: rc=$(cat "/opt/b/$name.rc" 2>/dev/null || echo '?') $(tr '\r\n' '  ' < "/opt/b/$name.log" | tail -c 300)"
      # SRBMiner 3.8.6 quit in its first seconds with no output on a 2080 Ti (9 November): say what it says about
      # itself, and any library it can't find.
      if [ "$name" = srb ]; then
        say "diag srb: $(cd "$(dirname "$SRB")" && { timeout 20 "$SRB" --version 2>&1; timeout 30 "$SRB" --list-algorithms 2>&1 | grep -i pearl; ldd "$SRB" 2>&1 | grep -i 'not found'; } | tr '\r\n' '  ' | tail -c 300)"
      fi
      # PeakMiner 2.20.9 exited 127 with no output on an RTX 5090 (15 November): say whether its file is there
      # and runs, and any library it can't find.
      if [ "$name" = peak ]; then
        say "diag peak: $(ls -l /opt/b/peak 2>&1 | cut -c1-120); $(cd /opt/b && sha256sum peak 2>&1 | cut -c1-16); $({ ldd /opt/b/peak 2>&1 | grep -i 'not found'; timeout 20 /opt/b/peak --version 2>&1; } | tr '\r\n' '  ' | tail -c 300)"
      fi
      [ "$name" = ours ] && [ -z "$AMD" ] && loaded
      [ "${OURS_SRC:-}" = 1 ] && case $name in ours|ours0|oursa) foldline "$name" ;; esac
      smisum "$name" "$spid"; stop_all; return
    fi
    [ "$measured" = 1 ] && sample "$name" "$t"
  done
  # Only printable ASCII from a miner's log: WildRig's "end" and "diag" lines reached Salad's logs blank (9 November).
  [ "$name" = ours ] && [ -z "$AMD" ] && loaded
  [ "${OURS_SRC:-}" = 1 ] && case $name in ours|ours0|oursa) foldline "$name" ;; esac
  [ "$measured" = 1 ] && say "end $name: $(tr '\r' '\n' < "/opt/b/$name.log" | grep -a -v '^\s*$' | tail -2 | tr '\n' ' ' | LC_ALL=C tr -cd ' -~' | tail -c 300)"
  # Our CLI prints its rate once a second, each line the time-weighted mean since the line before, so the mean of
  # every line past the first 60 is the run's own mean. The 15 s samples above read only the latest line, and on
  # RTX 20 they swing with batch boundaries (35-51 TH/s on a 2070 that hashrate.js puts at 43.6), so 8 of them move
  # about +-4% run to run. PeakMiner's samples do not swing.
  [ "$measured" = 1 ] && case $name in ours|ours0|oursa) true ;; *) false ;; esac && say "linemean $name $(tr '\r' '\n' < "/opt/b/$name.log" | grep -oE '[0-9.]+ TH/s · ' \
    | awk 'NR > 60 {s += $1; n++} END {if (n) printf "n=%d ths=%.2f", n, s / n; else printf "n=0"}') $(linedips "/opt/b/$name.log")"
  # ROTLOG=1 (PEARL_ROT_LOG): what the core said about each job's draw and each salt mined in the definition's order.
  [ "$measured" = 1 ] && [ "${ROTLOG:-}" = 1 ] && case $name in ours|oursa) true ;; *) false ;; esac \
    && say "rotlog $name $(tr '\r' '\n' < "/opt/b/$name.log" | grep -a '^\[pearl\] rot: ' | awk '
         /job drawn in/ {j++; js += $6}
         /ordered after/ {o++; os += $7; ss += $(NF - 1)}
         END {printf "jobs=%d draw_ms=%.0f unordered=%d unordered_ms=%.0f search_ms=%.0f", j, j ? js / j : 0, o, o ? os / o : 0, o ? ss / o : 0}')"
  # WildRig ran 5 minutes at 0 H/s with the card idle on a 2080 Ti (9 November): say which devices it found and
  # whether the driver's OpenCL library is in the container.
  if [ "$name" = wild ]; then
    say "diag wild: $(ls /usr/lib/x86_64-linux-gnu/libnvidia-opencl* /usr/lib64/libnvidia-opencl* 2>&1 | head -2 | tr '\n' ' ') $(tr '\r' '\n' < /opt/b/wild.log | grep -a -v '^\s*$' | grep -a -i -m6 -E 'gpu|device|platform|opencl|cuda|error' | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ' | LC_ALL=C tr -cd ' -~' | tail -c 400)"
  fi
  [ "$measured" = 1 ] && [ -n "${VP_PID:-}" ] && say "$(credit "$name" "$t0" "$(date +%s)")"
  smisum "$name" "$spid"; kill $pid 2>/dev/null; stop_all
  if [ "$measured" = 1 ] && [ -f /opt/b/tap.jsonl ]; then
    python3 /root/sharetap.py decode /opt/b/tap.jsonl --since "$t0" 2>&1 | while IFS= read -r l; do say "share $name $l"; done
  fi
}

IFS=, read -ra MINERS <<< "${ORDER:-peak,srb,ours}"
# A spot check (OURS_REL_DIR, from spot.sh): the box's own miner has mined through the downloads above. It stops
# now, with its Monero worker, so the card and CPU are the legs' alone (Gemma runs in the same CLI and stops too).
if [ -n "${OURS_REL_DIR:-}" ]; then
  stop_all; pkill -f llmjob-xmr-worker; sleep 2; say "spot: the box's miner stopped"
fi
run "${MINERS[0]}" "$WARM" 0          # warm-up: whoever runs first doesn't get a cold card
for m in "${MINERS[@]}"; do
  case $m in ours|ours0|oursa) run "$m" "${OURS_SECS:-300}" 1 ;; *) run "$m" "$SECS" 1 ;; esac
done
if [ -n "${VP_PID:-}" ]; then
  # The stand-in's summary: logins (agent and type, never a wallet), methods it refused, reasons, in 400-char pieces.
  kill -TERM "$VP_PID"; wait "$VP_PID" 2>/dev/null
  tail -n 1 /opt/b/vp.out | jq -c '{connections, jobs, byK, reasons, methods, logins}' 2>/dev/null | fold -w 400 \
    | while IFS= read -r l; do say "vpool $l"; done
fi
say "all done"
sleep infinity
