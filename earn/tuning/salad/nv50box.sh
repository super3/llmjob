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
#   PROBE_BUILDS  the probe's builds, "name:toolkit:nvcc flags;..." (toolkit 12 or 13; flags with "," for " "),
#                 each gprobe.cu with -DGP_STAGE=24576 and the flags (GP_ROTC=1/3/5/7: pearl_rot_c.inc's forms)
#   PROBE_LIST    the probe's runs, ';' between them: "<build> <nv30probe.py line>" (default: for each build a
#                 wide check at k 2048 and 60 s at k 65536)
#   ROT_BUILDS    the miner's cubins, "name:toolkit:nvcc flags;..." (ROT_C=1/3/5/7), pearl_rot.cu for sm_120
#   PROBE_SCHED   probe cubins scheduled again by probes/hand89/sched120.py, "name:base:key=v,key=v;..." (base a
#                 PROBE_BUILDS name); run with the base's program (PD_CUBIN)
#   ROT_SCHED     the same for the miner's cubins (base a ROT_BUILDS name); the names join ROT_BUILDS' for
#                 VERIFY_CUBINS and HR_LEGS
#   VERIFY_SECS   verify-hits.js on each cubin of VERIFY_CUBINS (default all of ROT_BUILDS) for this many
#                 seconds (default 150; 0 skips it), at 2^VERIFY_BITS (233)
#   HR_LEGS       hashrate.js legs: rel (the release fold) or a ROT_BUILDS name; a leg may carry core settings
#                 after "+" (default: rel, every cubin, every cubin backwards, rel); HR_SECS (120)
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
# nvdisasm 13.3, which sched120.py reads cubins with (redistrib_13.3.1.json's checksum).
ND=cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-13.3.73-archive.tar.xz
if [ ! -x "$C13/bin/nvdisasm" ]; then
  curl -fsSL --retry 3 "$RD/$ND" -o /root/nvd.tar.xz || fail "nvdisasm download"
  echo "38b2eca90c0ffa2ce815bd392077f68b7fe94992ea0ec0f8e85df30ca01fc4e9  /root/nvd.tar.xz" | sha256sum -c --quiet - \
    || fail "nvdisasm checksum"
  tar -xJf /root/nvd.tar.xz -C "$C13" --strip-components=1 || fail "nvdisasm unpack"
fi
NVD=$C13/bin/nvdisasm
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
nv() { if [ "$1" = 12 ]; then echo "$N12"; else echo "$N13"; fi; }
# sched "name:base:key=v,key=v;...": $W/base.cubin scheduled again into $W/name.cubin
sched() {
  local IFS_OLD=$IFS b n r base keys out
  IFS=';' read -r -a SL <<< "$1"
  IFS=$IFS_OLD
  for b in "${SL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; base=${r%%:*}; keys=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2086
    out=$(python3 -I "$H/sched120.py" "$W/$base.cubin" "$NVD" "$W/$n.cubin" $keys 2>&1)
    if [ -f "$W/$n.cubin" ]; then say "sched $n ($base $keys): $(echo "$out" | tail -1 | cut -c1-330)"
    else say "sched $n ($base $keys) FAILED: $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-330)"; fi
  done
}
PB=${PROBE_BUILDS:-"c1:13:-DGP_ROTC=1;c3:13:-DGP_ROTC=3;c5:13:-DGP_ROTC=5;c7:13:-DGP_ROTC=7;c1x:12:-DGP_ROTC=1"}
RB=${ROT_BUILDS:-"r1:13:-DROT_C=1;r3:13:-DROT_C=3;r5:13:-DROT_C=5;r7:13:-DROT_C=7"}
names() { echo "$1" | tr ';' '\n' | cut -d: -f1 | tr '\n' ' '; }
if has probe; then
  IFS=';' read -r -a BL <<< "$PB"
  for b in "${BL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; tk=${r%%:*}; fl=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) -Xptxas -v -DGP_STAGE=24576 $fl -o $W/$n "$H/gprobe.cu" -ldl > $W/$n.log 2>&1 || { tail -3 $W/$n.log | emit; fail "probe $n"; }
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) -DGP_STAGE=24576 $fl -cubin -o $W/$n.cubin "$H/gprobe.cu" > /dev/null 2>&1 || fail "probe cubin $n"
    say "probe $n built ($tk: $fl): $(regs $W/$n.log)"
  done
  sched "${PROBE_SCHED:-}"
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
  # The fold's cubins: pearl_rot.cu as the compiler makes it for sm_120, nothing patched.
  IFS=';' read -r -a BL <<< "$RB"
  for b in "${BL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; tk=${r%%:*}; fl=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) -Xptxas -v $fl -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/$n.cubin > $W/$n.log 2>&1 || { tail -3 $W/$n.log | emit; fail "pearl_rot.cu $n"; }
    say "rot cubin $n ($tk: $fl): fold hash $(python3 -I "$H/asm89.py" hash $W/$n.cubin) $(grep -A3 "pearl_rot_fold' for" $W/$n.log | regs /dev/stdin)"
  done
  sched "${ROT_SCHED:-}"
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
  L=${PROBE_LIST:-}
  if [ -z "$L" ]; then
    for n in $(names "$PB"); do L="$L$n - $n-k2048-check 2048 8 0 64 6 0 2;"; done
    for s in $(echo "${PROBE_SCHED:-}" | tr ';' '\n' | cut -d: -f1,2 | tr '\n' ' '); do
      L="$L${s#*:} ${s%%:*} ${s%%:*}-k2048-check 2048 8 0 64 6 0 2;"; done
    for n in $(names "$PB"); do L="$L$n - $n-k65536-60 65536 128 60 2 6 0 2;"; done
    for s in $(echo "${PROBE_SCHED:-}" | tr ';' '\n' | cut -d: -f1,2 | tr '\n' ' '); do
      L="$L${s#*:} ${s%%:*} ${s%%:*}-k65536-60 65536 128 60 2 6 0 2;"; done
  fi
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
    for vc in ${VERIFY_CUBINS:-$(names "$RB") $(names "${ROT_SCHED:-}")}; do
      j=$(PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$vc.cubin timeout -k 5 $(( ${VERIFY_SECS:-150} + 120 )) \
        node verify-hits.js "$ADDON" "${VERIFY_SECS:-150}" "${VERIFY_BITS:-233}" 2> $W/verify.err | grep '^{' | tail -1)
      say "verify $vc $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, thLast, fails: (.fails[0:1])}' 2>/dev/null | cut -c1-390)"
      [ -n "$j" ] || say "verify $vc no result: $(tail -2 $W/verify.err | tr '\n' ' ' | cut -c1-300)"
    done
  fi
  n=0
  fw="$(names "$RB") $(names "${ROT_SCHED:-}")"; bw=$(echo $fw | tr ' ' '\n' | tac | tr '\n' ' ')
  for leg in ${HR_LEGS:-rel $fw $bw rel}; do
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
