#!/bin/bash
# Box script for the rotation fold on an RTX 30 card (sm_86), branch goal/nv-30: on one Salad PC, the fold's probe
# (earn/native/probes/hand89/gprobe.cu with the hand loop of rfold.py), then the fold in the miner's core against
# the release fold in the same core (verify-hits.js, then hashrate.js legs taking turns), then the other miners
# and our release through bench.sh. A measuring tool for this branch: nothing here is shipped.
#
# The container command fetches this file at BASE_SHA and runs it (S/nv30/salad.py in the agent's scratchpad, a
# copy of amdtest.py's nvtest boxes with the group names nv30-<card>-<n>). Every line is
# "[nvtest] HH:MM:SS [nv30] ..." (bench.sh's "[nvtest] [bench] HH:MM:SS ..."), so the test-box tools read it.
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   STAGES        any of "probe product bench", in that order (default all three)
#   PROBE_VARIANTS  hand loops to build over the probe's kernel: "name:key=v,key=v;name:..." ("h:" is rfold.py's
#                 defaults, the loop RTX 40 ships)
#   PROBE_LIST    the probe's runs, ';' between them (nv30probe.py's lines; the default is below)
#   ROT_KEYS      rfold.py keys for the miner's cubin, comma-separated (default none: the shipped loop)
#   VERIFY_SECS   verify-hits.js on the fold (default 150; 0 skips it), at the target 2^VERIFY_BITS (233: an RTX 30
#                 card is a third of a 4090, so twice the 4090's 2^234 hits keeps 400 within the run)
#   ROT_VARIANTS  more miner cubins, "name:key=v,key=v;..." (pearl_rot_<name>.cubin), for hashrate.js legs
#   HR_LEGS       hashrate.js legs: "rel" (the release fold), "rot" (the shipped loop) or a ROT_VARIANTS name
#                 (default "rel rot rot rel"); HR_SECS (120)
#   ORDER, VPOOL, SECS, WARM, VER, TAP   passed to bench.sh (defaults ours,ozm,peak,srb; 210; 180; 60; v0.6.15)
set -u
say() { echo "[nvtest] $(date -u +%T) [nv30] $*"; sleep 0.03; }
[ -n "${BASE_SHA:-}" ] || { say "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in nvtest.sh bench.sh sharetap.py; do
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { say "FAIL fetch $f"; exec sleep infinity; }
done
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }
say "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.max_limit,clocks.max.sm,memory.total)"
say "card idle $(q clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
CC=$(q compute_cap | tr -d ' .')
[ "$CC" = 86 ] || say "WARN this card is compute $CC, not 8.6"
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 jq procps; } > /root/apt0.log 2>&1 || say "WARN apt-get: $(tail -1 /root/apt0.log)"
# nvtest.sh's functions: setup, cuda (CUDA 12.8.1's compiler, as CI builds with), fetch (the repo at BASE_SHA).
export CUDA_VER=12.8 NVTEST_STAGE=lib
# shellcheck disable=SC1091
. /root/nvtest.sh
unset NVTEST_STAGE
say() { echo "[nvtest] $(date -u +%T) [nv30] $*"; sleep 0.03; }
fail() { say "FAIL $*"; say "all done"; exec sleep infinity; }
setup
cuda
# nvdisasm, which rfold.py reads every patched cubin back with (redistrib_12.8.1.json's checksum).
ND=cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz
if [ ! -x "$CUDA/bin/nvdisasm" ]; then
  curl -fsSL --retry 3 "$RD/$ND" -o /root/nvd.tar.xz || fail "nvdisasm download"
  echo "8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae  /root/nvd.tar.xz" | sha256sum -c --quiet - \
    || fail "nvdisasm checksum"
  tar -xJf /root/nvd.tar.xz -C "$CUDA" --strip-components=1 || fail "nvdisasm unpack"
fi
fetch
export PATH="$CUDA/bin:/opt/node/bin:$PATH"
STAGES=" ${STAGES:-probe product bench} "
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
H=$R/earn/native/probes/hand89
W=/root/w
mkdir -p $W
NV="nvcc -arch=sm_86 -std=c++17 -O3"

# ---------------------------------------------------------------- builds (no GPU)
if has probe; then
  t0=$(date +%s)
  $NV -DGP_STAGE=24576 -o $W/gprobe "$H/gprobe.cu" -ldl > $W/gprobe.log 2>&1 || { tail -3 $W/gprobe.log | emit; fail "gprobe build"; }
  $NV -DGP_STAGE=24576 -cubin -o $W/g86.cubin "$H/gprobe.cu" > $W/g86.log 2>&1 || fail "gprobe cubin"
  nvdisasm -c $W/g86.cubin > $W/g86.sass || fail "nvdisasm g86"
  say "probe built in $(( $(date +%s) - t0 ))s; asm89 check: $(python3 -I "$H/asm89.py" check $W/g86.cubin $W/g86.sass | grep -E '^(assembled|wrong)' | cut -c1-60 | tr '\n' ' ')"
  IFS=';' read -r -a VARS <<< "${PROBE_VARIANTS:-h:}"
  for v in "${VARS[@]}"; do
    [ -n "$v" ] || continue
    name=${v%%:*}; keys=${v#*:}
    # shellcheck disable=SC2046
    out=$(python3 -I "$H/rfold.py" $W/g86.cubin $W/g86.sass nvdisasm "$W/$name.cubin" $(echo "$keys" | tr ',' ' ') 2>&1)
    if [ -f "$W/$name.cubin" ]; then
      say "variant $name ($keys): $(echo "$out" | tail -1 | cut -c1-200)"
    else
      say "variant $name ($keys) FAILED: $(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-300)"
    fi
  done
fi
if has product; then
  t0=$(date +%s)
  # Node 22 and node-gyp, as compilers.sh sets them up.
  if [ ! -x /opt/node/bin/node ]; then
    mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz \
      | tar -xJ -C /opt/node --strip-components=1 || fail "node download"
  fi
  mkdir -p /opt/gyp && (cd /opt/gyp && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || fail "node-gyp setup: $(tail -1 /opt/gyp/npm.log)"
  # The fold's cubin for sm_86: the kernel compiled, the hand loop put in (build_rot.py's steps 1 to 5).
  $NV -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/rot_base.cubin > $W/rot.log 2>&1 || fail "pearl_rot.cu"
  nvdisasm -c $W/rot_base.cubin > $W/rot_base.sass || fail "nvdisasm rot"
  say "rot asm89 check: $(python3 -I "$H/asm89.py" check $W/rot_base.cubin $W/rot_base.sass | grep -E '^(assembled|wrong)' | cut -c1-60 | tr '\n' ' ')"
  # shellcheck disable=SC2046
  python3 -I "$H/rfold.py" $W/rot_base.cubin $W/rot_base.sass nvdisasm $W/pearl_rot_sm86.cubin \
    $(echo "${ROT_KEYS:-}" | tr ',' ' ') > $W/rfold.log 2>&1 || { tail -3 $W/rfold.log | emit; fail "rfold.py"; }
  say "rot cubin $(tail -1 $W/rfold.log | cut -c1-160); hash $(python3 -I "$H/asm89.py" hash $W/pearl_rot_sm86.cubin)"
  IFS=';' read -r -a RV <<< "${ROT_VARIANTS:-}"
  for v in "${RV[@]}"; do
    [ -n "$v" ] || continue
    # shellcheck disable=SC2046
    vn=${v%%:*}
    if python3 -I "$H/rfold.py" $W/rot_base.cubin $W/rot_base.sass nvdisasm "$W/pearl_rot_$vn.cubin" \
        $(echo "${v#*:}" | tr ',' ' ') > "$W/rfold_$vn.log" 2>&1; then
      say "rot variant $vn: $(tail -1 "$W/rfold_$vn.log" | cut -c1-120); hash $(python3 -I "$H/asm89.py" hash "$W/pearl_rot_$vn.cubin")"
    else
      say "rot variant $vn FAILED: $(tail -2 "$W/rfold_$vn.log" | tr '\n' ' ' | cut -c1-200)"
    fi
  done
  # The addon for this card alone, as compilers.sh builds one.
  K=$W/addon
  rm -rf $K && mkdir -p $K/cuda-build && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" $K/
  C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86"
  # shellcheck disable=SC2086
  nvcc $C -c $K/src/pearl_kernel.cu -o $K/cuda-build/pearl_kernel.o > $K/kernel.log 2>&1 || fail "kernel: $(grep -m2 -i error $K/kernel.log)"
  # shellcheck disable=SC2086
  nvcc $C -c $K/src/pearl_host.cu -o $K/cuda-build/pearl_host.o > $K/host.log 2>&1 || fail "host: $(grep -m2 -i error $K/host.log)"
  ar rcs $K/cuda-build/libpearl_cuda.a $K/cuda-build/pearl_kernel.o $K/cuda-build/pearl_host.o
  (cd $K && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
    CUDA_PATH=$CUDA ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || fail "node-gyp: $(grep -m2 -i error $K/gyp.log)"
  ADDON=$K/build/Release/pearl_core.node
  say "addon built in $(( $(date +%s) - t0 ))s: $(ls -l $ADDON | awk '{print $5}') bytes"
fi

# ---------------------------------------------------------------- on the card
if has probe; then
  L=${PROBE_LIST:-"h h-k2048-o6-check 2048 8 0 64 6 0 2;h h-k8192-check 8192 8 0 8 6 0 2;h h-k65536-check8 65536 16 0 8 6 0 2;- gather-k2048-check 2048 8 0 64 6 0 0;h h-k65536-60 65536 128 60 2 6 0 2;h h-k8192-60 8192 128 60 2 6 0 2;- gather-k65536-60 65536 64 60 2 6 0 0;h h-k65536-60-again 65536 128 60 2 6 0 2"}
  # emit (nvtest.sh's) cuts a line into pieces of 400 characters: Salad's log service drops a longer one. And
  # it stores a line with "%" or "|" in it as an empty one (12 November, both RTX 30 boxes): "pc" and "/" here.
  python3 -I "$R/earn/tuning/salad/nv30probe.py" $W/gprobe $W "$L" 2>&1 | sed -u 's/^/probe /; s/%/pc/g; s/|/\//g' | emit
fi
smi() { q clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown; }
if has product; then
  cd "$R/earn/native/probes" || fail "no probes"
  if [ "${VERIFY_SECS:-150}" != 0 ]; then
    j=$(PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/pearl_rot_sm86.cubin timeout -k 5 $(( ${VERIFY_SECS:-150} + 120 )) \
      node verify-hits.js "$ADDON" "${VERIFY_SECS:-150}" "${VERIFY_BITS:-233}" 2> $W/verify.err | grep '^{' | tail -1)
    say "verify rot $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, fold: .device.fold, thLast, fails: (.fails[0:2])}' 2>/dev/null | cut -c1-390)"
    [ -n "$j" ] || say "verify rot no result: $(tail -2 $W/verify.err | tr '\n' ' ' | cut -c1-300)"
  fi
  n=0
  for leg in ${HR_LEGS:-rel rot rot rel}; do
    n=$((n + 1))
    # A leg may carry settings for the core after a "+": rot+PEARL_ROT_TILES=128.
    lv=${leg%%+*}; extra=""
    [ "$lv" != "$leg" ] && extra=$(echo "${leg#*+}" | tr '+' ' ')
    case "$lv" in
      rel) e="PEARL_FOLD_ROT=0" ;;
      rot) e="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/pearl_rot_sm86.cubin" ;;
      *) e="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/pearl_rot_$lv.cubin" ;;
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
