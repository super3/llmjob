#!/bin/bash
# Box script for the gate's timing arms (branch goal/gate, 10 November): on one Salad PC, the three competitors
# and our release measured by bench.sh on the stand-in pool, while the arms build; then the arms raced
# (probes/race.cu) by compilers.sh. A test tool for this branch only.
#
# The container command fetches this file at BASE_SHA and runs it. It fetches bench.sh and nvtest.sh at the
# same commit, starts bench.sh if GATE_BENCH is set (its lines come out as "[nvtest] [bench] HH:MM:SS ..."),
# and takes nvtest.sh's own steps (setup, cuda, fetch), with compilers.sh run twice: STAGES "setup build" beside
# the miners (nvcc uses no GPU), then, once bench.sh has said "all done", STAGES "race". RACE_ONLY=1 keeps
# compilers.sh from building the addon and bench for every arm.
#
# Environment: BASE_SHA, VARIANTS_SPEC, CUDA_VER, RACE_SECS, RACE_LEG, RACE_WARM, SMI_EVERY as nvtest.sh's;
# GATE_BENCH=1 with ORDER (peak,srb,ozm,ours), SECS, WARM, OURS_SECS, VPOOL (210) and VER as bench.sh's.
set -u
gsay() { echo "[nvtest] $(date -u +%T) [gate] $*"; }
[ -n "${BASE_SHA:-}" ] || { gsay "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in bench.sh nvtest.sh; do
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { gsay "FAIL fetch $f"; exec sleep infinity; }
done
gsay "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,power.default_limit,temperature.gpu,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1)"
: > /root/bench.log
# Every package bench.sh and nvtest.sh ask for, once, before either starts: two apt-get runs at
# the same time would fight over dpkg's lock.
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 jq procps; } > /root/apt0.log 2>&1 || gsay "WARN apt-get: $(tail -1 /root/apt0.log)"
if [ -n "${GATE_BENCH:-}" ]; then
  ( W=${W:-gate} ORDER=${ORDER:-peak,srb,ozm,ours} VPOOL=${VPOOL:-210} SECS=${SECS:-240} WARM=${WARM:-60} \
      OURS_SECS=${OURS_SECS:-240} bash /root/bench.sh 2>&1 | tee -a /root/bench.log \
      | sed -u 's/^\[bench\]/[nvtest] [bench]/' ) &
else
  echo "all done" > /root/bench.log
fi
# nvtest.sh's functions and settings, without its run.
export NVTEST_ROOT=/opt/gate
NVTEST_STAGE=lib . /root/nvtest.sh
[ -n "$NVCC_VER" ] || fail "CUDA_VER must be 12.8 or 13.3, not $CUDA_VER"
driver_ok || fail "CUDA_VER=$CUDA_VER needs driver 580 or newer, and this card's is older"
setup
cuda
fetch
export GC=${GC:-$(card_gc)} FOLD=${FOLD:-pearl_tile_fold_tall} VARIANTS_SPEC=${VARIANTS_SPEC:-base|} RACE_ONLY=1
[ -n "$GC" ] || fail "no compute capability from nvidia-smi; set GC"
crun() {   # $1 = compilers.sh's stages
  local rc t0; t0=$(date +%s)
  env STAGES="$1" SRC="$R" OUT="$OUT" CUDA_HOME="$CUDA" NVCC_VER="$NVCC_VER" SMI_EVERY="${SMI_EVERY:-5}" \
    bash "$R/earn/native/probes/turing/compilers.sh" 2>&1 | tee "/root/compilers.$t0.log" \
    | grep --line-buffered -a '^\[compilers\]' | emit
  rc=${PIPESTATUS[0]}
  say "compilers [$1] done rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"
  [ "$rc" = 0 ] || grep -av '^\[compilers\]' "/root/compilers.$t0.log" | tail -n 8 | emit
}
crun "setup build"
n=0
until grep -q 'all done' /root/bench.log; do
  n=$((n + 1)); [ $n -gt 360 ] && { gsay "bench.sh still running after an hour; racing anyway"; break; }
  sleep 10
done
pkill -f /root/bench.sh 2> /dev/null
pkill -f 'peak -o' 2> /dev/null; pkill -f SRBMiner-MULTI 2> /dev/null; pkill -f onezerominer 2> /dev/null; pkill -f 'cli -a' 2> /dev/null
sleep 5
gsay "card before the race $(nvidia-smi -i 0 --query-gpu=clocks.sm,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1)"
crun "race"
gsay "card after the race $(nvidia-smi -i 0 --query-gpu=clocks.sm,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1)"
say "all done"
idle "until deleted"
