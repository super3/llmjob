#!/bin/bash
# Box script for RTX 20 (sm_75) work on branch goal/nv-20: on one Salad PC, Turing's price probe
# (earn/native/probes/turing/price75.cu, timing only), then the miner's core built for this card from BASE_SHA in
# one or more variants (verify-hits.js, then hashrate.js legs taking turns), then the other miners and our release
# through bench.sh. A measuring tool for this branch: nothing here is shipped.
#
# The container command fetches this file at BASE_SHA and runs it (S/nv20/salad.py in the agent's scratchpad, a
# copy of amdtest.py's nvtest boxes with the group names nv20-<card>-<n>). Every line is
# "[nvtest] HH:MM:SS [nv20] ..." (bench.sh's "[nvtest] [bench] HH:MM:SS ..."), so the test-box tools read it.
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   STAGES        any of "probe product bench", in that order (default all three)
#   PROBE_LIST    the probe's runs, ';' between them: "<kernel> <seconds> [tiles]" (default below)
#   VARIANTS      core builds besides "rel" (BASE_SHA as it is): "name:-DX=1 -DY=2;name2:..." (nvcc defines)
#   VERIFY        builds to run verify-hits.js on (default: every variant, not rel); VERIFY_SECS (120; 0 skips),
#                 VERIFY_BITS (232)
#   HR_LEGS       hashrate.js legs, a build's name each, optionally "+K=V+K2=V2" for the core's environment
#                 (default "rel" and each variant, then the same reversed); HR_SECS (120)
#   ORDER, VPOOL, SECS, WARM, VER, TAP   passed to bench.sh (defaults ours,ozm,peak,srb; 210; 180; 60; v0.6.15)
set -u
say() { echo "[nvtest] $(date -u +%T) [nv20] $*"; sleep 0.03; }
[ -n "${BASE_SHA:-}" ] || { say "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in nvtest.sh bench.sh sharetap.py; do
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { say "FAIL fetch $f"; exec sleep infinity; }
done
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }
say "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.max_limit,clocks.max.sm,memory.total)"
say "card idle $(q clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
CC=$(q compute_cap | tr -d ' .')
[ "$CC" = 75 ] || say "WARN this card is compute $CC, not 7.5"
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 jq procps; } > /root/apt0.log 2>&1 || say "WARN apt-get: $(tail -1 /root/apt0.log)"
# nvtest.sh's functions: setup, cuda (CUDA 12.8.1's compiler, as CI builds with), fetch (the repo at BASE_SHA).
export CUDA_VER=12.8 NVTEST_STAGE=lib
# shellcheck disable=SC1091
. /root/nvtest.sh
unset NVTEST_STAGE
say() { echo "[nvtest] $(date -u +%T) [nv20] $*"; sleep 0.03; }
fail() { say "FAIL $*"; say "all done"; exec sleep infinity; }
setup
cuda
fetch
export PATH="$CUDA/bin:/opt/node/bin:$PATH"
STAGES=" ${STAGES:-probe product bench} "
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
W=/root/w
mkdir -p $W
smi() { q clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown; }

# ---------------------------------------------------------------- builds (no GPU)
if has probe; then
  t0=$(date +%s)
  nvcc -arch=sm_75 -std=c++17 -O3 -Xptxas -v -o $W/price75 "$R/earn/native/probes/turing/price75.cu" > $W/price75.log 2>&1 \
    || { grep -m3 -i error $W/price75.log | emit; fail "price75 build"; }
  say "probe built in $(( $(date +%s) - t0 ))s; spills: $(grep -c 'bytes spill stores' $W/price75.log) kernels, $(grep 'bytes spill stores' $W/price75.log | grep -vc ' 0 bytes spill stores') with any"
fi
build_core() {  # build_core <name> "<defines>": the addon for this card alone, as compilers.sh builds one
  local K=$W/core-$1 t0
  t0=$(date +%s)
  rm -rf "$K" && mkdir -p "$K/cuda-build" && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" "$K/"
  local C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_75,code=sm_75 $2"
  # shellcheck disable=SC2086
  nvcc $C -Xptxas -v -c "$K/src/pearl_kernel.cu" -o "$K/cuda-build/pearl_kernel.o" > "$K/kernel.log" 2>&1 \
    || { say "core $1 kernel FAILED: $(grep -m2 -i error "$K/kernel.log" | cut -c1-200)"; return 1; }
  # shellcheck disable=SC2086
  nvcc $C -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o" > "$K/host.log" 2>&1 \
    || { say "core $1 host FAILED: $(grep -m2 -i error "$K/host.log" | cut -c1-200)"; return 1; }
  ar rcs "$K/cuda-build/libpearl_cuda.a" "$K/cuda-build/pearl_kernel.o" "$K/cuda-build/pearl_host.o"
  (cd "$K" && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
    CUDA_PATH=$CUDA ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) \
    || { say "core $1 node-gyp FAILED: $(grep -m2 -i error "$K/gyp.log" | cut -c1-200)"; return 1; }
  local regs
  regs=$(grep -A2 'pearl_tile_fold_wmma' "$K/kernel.log" | grep -o 'Used [0-9]* registers\|[0-9]* bytes spill stores' | tr '\n' ' ' | cut -c1-120)
  say "core $1 built in $(( $(date +%s) - t0 ))s ($2): $regs"
}
NAMES="rel"
if has product; then
  if [ ! -x /opt/node/bin/node ]; then
    mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz \
      | tar -xJ -C /opt/node --strip-components=1 || fail "node download"
  fi
  mkdir -p /opt/gyp && (cd /opt/gyp && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || fail "node-gyp setup: $(tail -1 /opt/gyp/npm.log)"
  build_core rel "" || fail "the release core"
  IFS=';' read -r -a VV <<< "${VARIANTS:-}"
  for v in "${VV[@]}"; do
    [ -n "$v" ] || continue
    build_core "${v%%:*}" "${v#*:}" && NAMES="$NAMES ${v%%:*}"
  done
fi

# ---------------------------------------------------------------- on the card
if has probe; then
  L=${PROBE_LIST:-"ceil 25;rel 40;sm8 40;rot 40;rotnb 40;rotrel 40;gath 40;gathA 40;relA 30;relB 30;sm8+lds4 30;sm8+lds8 30;sm8+lds16 30;sm8+l1g8 30;sm8+l2g4 30;sm8+l2g8 30;sm8+alu32 30;sm8+alu64 30;sm8+sts4 30;rel+alu32 30;rot 40;rel 40;sm8 40"}
  python3 -I "$R/earn/tuning/salad/nv20probe.py" $W/price75 "$L" 2>&1 | sed -u 's/^/probe /; s/%/pc/g; s/|/\//g' | emit
fi
if has product; then
  cd "$R/earn/native/probes" || fail "no probes"
  for nm in ${VERIFY:-${NAMES#rel}}; do
    [ "${VERIFY_SECS:-120}" != 0 ] || break
    j=$(timeout -k 5 $(( ${VERIFY_SECS:-120} + 120 )) node verify-hits.js "$W/core-$nm/build/Release/pearl_core.node" \
      "${VERIFY_SECS:-120}" "${VERIFY_BITS:-232}" 2> $W/verify-$nm.err | grep '^{' | tail -1)
    say "verify $nm $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, fold: .device.fold, thLast, fails: (.fails[0:2])}' 2>/dev/null | cut -c1-390)"
    [ -n "$j" ] || say "verify $nm no result: $(tail -2 $W/verify-$nm.err | tr '\n' ' ' | cut -c1-300)"
  done
  legs=${HR_LEGS:-}
  if [ -z "$legs" ]; then
    rev=""; for nm in $NAMES; do rev="$nm $rev"; done
    legs="$NAMES $rev"
  fi
  n=0
  for leg in $legs; do
    n=$((n + 1))
    lv=${leg%%+*}; extra=""
    [ "$lv" != "$leg" ] && extra=$(echo "${leg#*+}" | tr '+' ' ')
    [ -f "$W/core-$lv/build/Release/pearl_core.node" ] || { say "hashrate $leg$n: no core $lv"; continue; }
    # shellcheck disable=SC2086
    j=$(env $extra timeout -k 5 $(( ${HR_SECS:-120} + 120 )) node hashrate.js "$W/core-$lv/build/Release/pearl_core.node" \
      "${HR_SECS:-120}" 10 2> $W/hr$n.err | grep '^{' | tail -1)
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
