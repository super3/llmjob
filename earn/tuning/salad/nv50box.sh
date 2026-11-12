#!/bin/bash
# Box script for the rotation fold on an RTX 50 card (sm_120), branch goal/nv-50: on one Salad PC, the fold's probe
# (earn/native/probes/hand89/gprobe.cu built with GP_ROTC=1, the rotation fold compiled from C), then the fold in
# the miner's core against the release fold in the same core (verify-hits.js, then hashrate.js legs taking turns),
# then the other miners and our release through bench.sh. A measuring tool for this branch: nothing here ships.
# nv30box.sh's sm_120 twin. No hand loop: the cubin is pearl_rot.cu as ptxas compiles it for sm_120.
#
# Two toolkits: CUDA 13.3.1 (the one pearl_core_cu13.node, the core RTX 50 rigs run, is built with) for the addon,
# and CUDA 12.8.1 beside it, so the fold's cubin can be raced as each compiler makes it.
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   STAGES        any of "probe product bench", in that order (default all three)
#   PROBE_LIST    the probe's runs, ';' between them: "<exe> <nv30probe.py line>", exe c12 / c13 (the compiled
#                 rotation fold, CUDA 12.8 / 13.3) or g13 (the gather fold, 13.3)
#   PROBE_DEFS    extra nvcc flags for the c12 and c13 probes (default none)
#   ROT_DEFS      extra nvcc flags for the miner's cubins (default none)
#   VERIFY_SECS   verify-hits.js on the fold (default 150; 0 skips it), at 2^VERIFY_BITS (233), with the cubin
#                 VERIFY_CUBIN (r13)
#   HR_LEGS       hashrate.js legs: rel (the release fold), r12 or r13 (the fold's cubin from CUDA 12.8 / 13.3);
#                 a leg may carry core settings after "+" (default "rel r13 r12 r12 r13 rel"); HR_SECS (120)
#   ORDER, VPOOL, SECS, WARM, VER   passed to bench.sh (defaults ours,ozm,peak,srb; 210; 180; 60; v0.6.15)
set -u
say() { echo "[nvtest] $(date -u +%T) [nv50] $*"; sleep 0.03; }
[ -n "${BASE_SHA:-}" ] || { say "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in nvtest.sh bench.sh sharetap.py nv30probe.py; do
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { say "FAIL fetch $f"; exec sleep infinity; }
done
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }
say "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem,memory.total)"
say "card idle $(q clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
CC=$(q compute_cap | tr -d ' .')
[ "$CC" = 120 ] || say "WARN this card is compute $CC, not 12.0"
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 jq procps; } > /root/apt0.log 2>&1 || say "WARN apt-get: $(tail -1 /root/apt0.log)"
# nvtest.sh's functions: setup, cuda, fetch. Sourced once a toolkit: its variables are set when it is read.
export NVTEST_STAGE=lib
export CUDA_VER=12.8 NVTEST_CUDA=/opt/cuda12
# shellcheck disable=SC1091
. /root/nvtest.sh
say() { echo "[nvtest] $(date -u +%T) [nv50] $*"; sleep 0.03; }
fail() { say "FAIL $*"; say "all done"; exec sleep infinity; }
setup
cuda
C12=$CUDA
export CUDA_VER=13.3 NVTEST_CUDA=/opt/cuda13
# shellcheck disable=SC1091
. /root/nvtest.sh
say() { echo "[nvtest] $(date -u +%T) [nv50] $*"; sleep 0.03; }
fail() { say "FAIL $*"; say "all done"; exec sleep infinity; }
driver_ok || fail "driver too old for CUDA 13"
cuda
C13=$CUDA
unset NVTEST_STAGE
fetch
export PATH="/opt/node/bin:$PATH"
STAGES=" ${STAGES:-probe product bench} "
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
H=$R/earn/native/probes/hand89
W=/root/w
mkdir -p $W
N12="$C12/bin/nvcc -arch=sm_120 -std=c++17 -O3"
N13="$C13/bin/nvcc -arch=sm_120 -std=c++17 -O3"
regs() { grep -E 'Used [0-9]+ registers|spill' "$1" | tr -s ' ' | tr '\n' ' ' | cut -c1-300; }

# ---------------------------------------------------------------- builds (no GPU)
if has probe; then
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $N12 -Xptxas -v -DGP_STAGE=24576 -DGP_ROTC=1 ${PROBE_DEFS:-} -o $W/c12 "$H/gprobe.cu" -ldl > $W/c12.log 2>&1 || { tail -3 $W/c12.log | emit; fail "probe c12"; }
  # shellcheck disable=SC2086
  $N13 -Xptxas -v -DGP_STAGE=24576 -DGP_ROTC=1 ${PROBE_DEFS:-} -o $W/c13 "$H/gprobe.cu" -ldl > $W/c13.log 2>&1 || { tail -3 $W/c13.log | emit; fail "probe c13"; }
  $N13 -Xptxas -v -DGP_STAGE=24576 -o $W/g13 "$H/gprobe.cu" -ldl > $W/g13.log 2>&1 || { tail -3 $W/g13.log | emit; fail "probe g13"; }
  say "probes built in $(( $(date +%s) - t0 ))s; c12 $(regs $W/c12.log); c13 $(regs $W/c13.log)"
fi
if has product; then
  t0=$(date +%s)
  if [ ! -x /opt/node/bin/node ]; then
    mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz \
      | tar -xJ -C /opt/node --strip-components=1 || fail "node download"
  fi
  mkdir -p /opt/gyp && (cd /opt/gyp && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || fail "node-gyp setup: $(tail -1 /opt/gyp/npm.log)"
  # The fold's cubins: pearl_rot.cu as each compiler makes it for sm_120, nothing patched.
  # shellcheck disable=SC2086
  $N12 -Xptxas -v ${ROT_DEFS:-} -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/r12.cubin > $W/r12.log 2>&1 || fail "pearl_rot.cu 12.8"
  # shellcheck disable=SC2086
  $N13 -Xptxas -v ${ROT_DEFS:-} -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/r13.cubin > $W/r13.log 2>&1 || fail "pearl_rot.cu 13.3"
  say "rot cubins: r12 $(sha256sum $W/r12.cubin | cut -c1-12) $(regs $W/r12.log); r13 $(sha256sum $W/r13.cubin | cut -c1-12) $(regs $W/r13.log)"
  # The addon for this card, as CI builds pearl_core_cu13.node (CUDA 13.3, sm_120).
  K=$W/addon
  rm -rf $K && mkdir -p $K/cuda-build && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" $K/
  C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_120,code=sm_120"
  # shellcheck disable=SC2086
  $C13/bin/nvcc $C -c $K/src/pearl_kernel.cu -o $K/cuda-build/pearl_kernel.o > $K/kernel.log 2>&1 || fail "kernel: $(grep -m2 -i error $K/kernel.log)"
  # shellcheck disable=SC2086
  $C13/bin/nvcc $C -c $K/src/pearl_host.cu -o $K/cuda-build/pearl_host.o > $K/host.log 2>&1 || fail "host: $(grep -m2 -i error $K/host.log)"
  ar rcs $K/cuda-build/libpearl_cuda.a $K/cuda-build/pearl_kernel.o $K/cuda-build/pearl_host.o
  (cd $K && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
    CUDA_PATH=$C13 ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || fail "node-gyp: $(grep -m2 -i error $K/gyp.log)"
  ADDON=$K/build/Release/pearl_core.node
  say "addon (CUDA 13.3) built in $(( $(date +%s) - t0 ))s: $(ls -l $ADDON | awk '{print $5}') bytes"
fi

# ---------------------------------------------------------------- on the card
smi() { q clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown; }
if has probe; then
  L=${PROBE_LIST:-"c13 - c13-k2048-check 2048 8 0 64 6 0 2;c12 - c12-k2048-check 2048 8 0 64 6 0 2;c13 - c13-k65536-check8 65536 16 0 8 6 0 2;c13 - c13-k65536-60 65536 128 60 2 6 0 2;c12 - c12-k65536-60 65536 128 60 2 6 0 2;g13 - gather-k65536-60 65536 64 60 2 6 0 0;c13 - c13-k65536-60-again 65536 128 60 2 6 0 2"}
  IFS=';' read -r -a PL <<< "$L"
  for line in "${PL[@]}"; do
    [ -n "$line" ] || continue
    exe=${line%% *}
    # Salad drops lines over 400 characters and blanks a line with "%" or "|": "pc" and "/" here.
    python3 -I /root/nv30probe.py "$W/$exe" $W "${line#* }" 2>&1 | grep -v "card has" | sed -u "s/^/probe $exe /; s/%/pc/g; s/|/\//g" | emit
  done
fi
if has product; then
  cd "$R/earn/native/probes" || fail "no probes"
  if [ "${VERIFY_SECS:-150}" != 0 ]; then
    vc=${VERIFY_CUBIN:-r13}
    j=$(PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$vc.cubin timeout -k 5 $(( ${VERIFY_SECS:-150} + 120 )) \
      node verify-hits.js "$ADDON" "${VERIFY_SECS:-150}" "${VERIFY_BITS:-233}" 2> $W/verify.err | grep '^{' | tail -1)
    say "verify $vc $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, fold: .device.fold, thLast, fails: (.fails[0:2])}' 2>/dev/null | cut -c1-390)"
    [ -n "$j" ] || say "verify $vc no result: $(tail -2 $W/verify.err | tr '\n' ' ' | cut -c1-300)"
  fi
  n=0
  for leg in ${HR_LEGS:-rel r13 r12 r12 r13 rel}; do
    n=$((n + 1))
    lv=${leg%%+*}; extra=""
    [ "$lv" != "$leg" ] && extra=$(echo "${leg#*+}" | tr '+' ' ')
    case "$lv" in
      rel) e="PEARL_FOLD_ROT=0" ;;
      *) e="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$lv.cubin" ;;
    esac
    e="$e $extra"
    # shellcheck disable=SC2086
    j=$(env $e timeout -k 5 $(( ${HR_SECS:-120} + 120 )) node hashrate.js "$ADDON" "${HR_SECS:-120}" 10 2> $W/hr$n.err | grep '^{' | tail -1)
    say "hashrate $leg$n $(echo "$j" | jq -c '{k, th, sd, n, mhz, watts, tempC, memUtil, fold: .device.fold}' 2>/dev/null | cut -c1-300) end $(smi)"
    [ -n "$j" ] || say "hashrate $leg$n no result: $(tail -2 $W/hr$n.err | tr '\n' ' ' | cut -c1-300)"
  done
fi
if has bench; then
  # W: a payout address in form only, which is all our CLI checks. The stand-in pool pays nobody.
  W=prl1pgate00000000000000000000000000000000000000000000000000000 ORDER=${ORDER:-ours,ozm,peak,srb} \
    VPOOL=${VPOOL:-210} SECS=${SECS:-180} WARM=${WARM:-60} OURS_SECS=${SECS:-180} VER=${VER:-v0.6.15} \
    bash /root/bench.sh 2>&1 | sed -u 's/^\[bench\]/[nvtest] [bench]/'
fi
say "card at the end $(smi)"
say "all done"
exec sleep infinity
