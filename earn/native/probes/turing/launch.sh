#!/bin/bash
# The RTX 20 launch sweep: how the B-direct fold's work is split up and sent to the card,
# on one Turing card. The shapes are the core's PEARL_BD_SWEEP_* overrides
# (pearl_bd_sweep in src/pearl_host.cu), read once at create and doing nothing when unset,
# so one build measures every shape.
#
#   SRC=<llmjob checkout at a pinned commit> OUT=<scratch dir> bash launch.sh
#
# SRC is only read. Assumes one idle RTX 20 card (GPU 0, or GPU=n) and a working
# nvidia-smi, on the nvidia/cuda:12.8.1-devel-ubuntu22.04 image. Installs what else it
# needs: node 22 from nodejs.org, node-gyp, and python3, make and g++ with apt.
#
# What it runs, after building the core for sm_75 with CUDA 12.8:
#   1. verify-hits.js on the default shape: 400 hits recomputed in JS, all must match.
#   2. bench.cu over band depths: the rule's band and two or three others (BANDS), each
#      a whole number of 256-column groups. 3 interleaved rounds. The best other band is
#      then checked with verify-hits.js, since only powers of two have run on a card.
#   3. hashrate.js, the full miner loop, 3 interleaved rounds of 60 s: the default
#      (ctl), that band, col_batch 4096 and 1024 (launches twice and half as long), and
#      both pipeline slots on one stream (st1).
#   4. If a shape's median beats ctl's by CONFIRM_PCT or more: 3 more A/B pairs of the
#      two, and verify-hits.js on it when it changes the tile walk.
# About 45 minutes, inside BUDGET_MIN (55) whatever the card: a phase that would run
# past it is skipped.
#
# Every result is one line starting with [launch], under 500 characters (Vast's log API
# cuts longer lines). mhz, w and C are medians of 1 s nvidia-smi samples over the
# measured part of a run. thpg is TH/s per GHz of that clock. d is the change of a
# point's median TH/s against ctl's, and wins counts the rounds it beat ctl's run of the
# same round. Raw files stay in $OUT/launch.
#
# Settings (environment): GPU=0 ROUNDS=3 HR_SECS=60 HR_WARM=10 BENCH_SECS=20
#   VERIFY_SECS=100 BANDS="5 6" CONFIRM_PCT=0.3 BUDGET_MIN=55 CUDA_HOME=/usr/local/cuda
#   NODE_VER=v22.22.0 LAUNCH_LABEL=<printed on the card line>
#   LAUNCH_BUILD_ONLY=1  stop after the build (works without a GPU)
#   LAUNCH_DRY=1         no GPU and no build: every run is made up, to check the flow
set -u
: "${SRC:?set SRC to the llmjob checkout}" "${OUT:?set OUT to a scratch dir}"
SRC=$(cd "$SRC" && pwd) || exit 1
mkdir -p "$OUT/launch" && W=$(cd "$OUT/launch" && pwd) || exit 1
HERE=$SRC/earn/native/probes/turing; PROBES=$SRC/earn/native/probes; H=$HERE/launch.js
GPU=${GPU:-0}; ROUNDS=${ROUNDS:-3}; HR_SECS=${HR_SECS:-60}; HR_WARM=${HR_WARM:-10}
BENCH_SECS=${BENCH_SECS:-20}; VERIFY_SECS=${VERIFY_SECS:-100}; CONFIRM_PCT=${CONFIRM_PCT:-0.3}
BUDGET_MIN=${BUDGET_MIN:-55}; CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NODE_VER=${NODE_VER:-v22.22.0}
DRY=${LAUNCH_DRY:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
T0=$(date +%s); RES=$W/results.tsv; : > "$RES"; mkdir -p "$W/runs"

say() { local m; m=$(printf '%s' "$*" | tr '\r\n\t' '   '); printf '[launch] %s\n' "${m:0:480}"; }
die() { say "FAIL $*"; exit 1; }
elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }
# Whether $1 more seconds still end inside the budget.
fits() { [ $(( $(date +%s) - T0 + $1 )) -le $(( BUDGET_MIN * 60 )) ]; }
SAMP=; trap '[ -n "$SAMP" ] && kill $SAMP 2>/dev/null' EXIT
say "start $(date -u +%FT%TZ) src $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)${DRY:+ (dry run)}"

# ---- tools ---------------------------------------------------------------------------
if [ -z "$DRY" ]; then
  pk=()
  command -v curl > /dev/null || pk+=(curl ca-certificates)
  command -v xz > /dev/null || pk+=(xz-utils)
  command -v python3 > /dev/null || pk+=(python3)
  command -v make > /dev/null || pk+=(make)
  command -v g++ > /dev/null || pk+=(g++)
  if [ ${#pk[@]} -gt 0 ]; then
    # Package lists without NVIDIA's CUDA repo, which can be slow or down.
    export DEBIAN_FRONTEND=noninteractive
    mkdir -p /tmp/apt-nocuda
    for f in /etc/apt/sources.list.d/*; do
      [ -f "$f" ] && ! grep -q developer.download.nvidia.com "$f" && cp "$f" /tmp/apt-nocuda/
    done
    apt-get update -o Dir::Etc::SourceParts=/tmp/apt-nocuda > "$W/apt.log" 2>&1
    apt-get install -y --no-install-recommends "${pk[@]}" >> "$W/apt.log" 2>&1 ||
      die "apt-get install ${pk[*]}: $(tail -1 "$W/apt.log")"
  fi
  if ! "$CUDA_HOME/bin/nvcc" --version 2> /dev/null | grep -q 'release 12\.8,'; then
    if /usr/local/cuda-12.8/bin/nvcc --version 2> /dev/null | grep -q 'release 12\.8,'; then
      CUDA_HOME=/usr/local/cuda-12.8
    else
      apt-get update > "$W/apt-cuda.log" 2>&1
      apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 \
        >> "$W/apt-cuda.log" 2>&1 || die "no CUDA 12.8 nvcc: $(tail -1 "$W/apt-cuda.log")"
      CUDA_HOME=/usr/local/cuda-12.8
    fi
  fi
  NVCC=$CUDA_HOME/bin/nvcc
  if [ "$(node -v 2> /dev/null | sed 's/^v\([0-9]*\).*/\1/')" -ge 22 ] 2> /dev/null; then :; else
    [ -x /opt/node-$NODE_VER-linux-x64/bin/node ] ||
      curl -fsSL "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz" | tar -xJ -C /opt ||
      die "node $NODE_VER download"
    export PATH=/opt/node-$NODE_VER-linux-x64/bin:$PATH
  fi
  if [ ! -x "$W/gyp/node_modules/.bin/node-gyp" ]; then
    mkdir -p "$W/gyp" && echo '{}' > "$W/gyp/package.json"
    (cd "$W/gyp" && npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp &&
      ./node_modules/.bin/node-gyp install) > "$W/gyp.log" 2>&1 || die "node-gyp: $(tail -2 "$W/gyp.log")"
  fi
  say "tools nvcc $("$NVCC" --version | grep -o 'V[0-9][0-9.]*') node $(node -v) $(elapsed) min"
fi

# ---- the card ------------------------------------------------------------------------
if [ -n "$DRY" ]; then
  SMS=30; L2=3145728; CC=7.5
  say "card dry run: RTX 2060, 30 SMs, L2 3072 KB"
else
  cat > "$W/devq.cu" << 'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("none\n"); return 1; }
  printf("%d %d %d.%d %d|%s\n", p.multiProcessorCount, p.l2CacheSize, p.major, p.minor,
         p.memoryBusWidth, p.name);
  return 0;
}
EOF
  "$NVCC" -o "$W/devq" "$W/devq.cu" > "$W/devq.log" 2>&1 || die "devq build: $(head -c 200 "$W/devq.log")"
  DQ=$("$W/devq" 2>&1) || [ -n "${LAUNCH_BUILD_ONLY:-}" ] || die "no CUDA device: $DQ"
  read -r SMS L2 CC BUS <<< "${DQ%%|*}"
  smi=$(nvidia-smi -i "$GPU" --query-gpu=name,driver_version,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,clocks.sm,temperature.gpu,memory.total,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,vbios_version --format=csv,noheader 2>&1)
  say "card ${LAUNCH_LABEL:-} | ${DQ#*|} | sms=$SMS l2KB=$(( ${L2:-0} / 1024 )) cc=$CC bus=$BUS | $smi"
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader 2> /dev/null | tr '\n' ' ')
  [ -n "$apps" ] && say "WARN the card is not idle: $apps"
  [ "$CC" = 7.5 ] || [ -n "${LAUNCH_BUILD_ONLY:-}" ] || die "compute $CC is not Turing (7.5)"
fi

# ---- build ---------------------------------------------------------------------------
B=$W/build; NODEF=$B/build/Release/pearl_core.node
if [ -z "$DRY" ]; then
  rm -rf "$B" && mkdir -p "$B/cuda-build" &&
    cp -r "$SRC/earn/native/src" "$SRC/earn/native/binding.gyp" "$PROBES/bench.cu" "$B/" || die "copy the source"
  echo '{}' > "$B/package.json"; ln -s "$W/gyp/node_modules" "$B/node_modules"
  GC="-gencode arch=compute_75,code=sm_75"
  (cd "$B" &&
    "$NVCC" -O3 -std=c++17 -cudart static -Xcompiler -fPIC $GC -Xptxas -v -c src/pearl_kernel.cu \
      -o cuda-build/pearl_kernel.o > ptxas.txt 2>&1 &&
    "$NVCC" -O3 -std=c++17 -cudart static -Xcompiler -fPIC $GC -c src/pearl_host.cu \
      -o cuda-build/pearl_host.o > host.log 2>&1 &&
    ar rcs cuda-build/libpearl_cuda.a cuda-build/pearl_kernel.o cuda-build/pearl_host.o &&
    CUDA_PATH=$CUDA_HOME ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1 &&
    "$NVCC" -O3 -std=c++17 -cudart static $GC -Isrc bench.cu cuda-build/pearl_kernel.o \
      cuda-build/pearl_host.o -o bench > bench-build.log 2>&1) ||
    die "build: $(cat "$B"/ptxas.txt "$B"/*.log 2> /dev/null | grep -m3 -i error | tr '\n' ' ')"
  [ -f "$NODEF" ] || die "build: no pearl_core.node"
  # Registers and spill of one kernel. The name is matched with its quotes, so the fold
  # does not also match pearl_tile_fold_wmma_rg, the second kernel (PEARL_BD_RG_KERNEL),
  # which the core runs instead on cards with 4 MB of L2 or more.
  kstat() {
    tr -d '\r' < "$B/ptxas.txt" | awk -v k="'$1'" '/Compiling entry function/ { on = index($0, "function " k " ") > 0; next }
      on && /spill stores/ { match($0, /[0-9]+ bytes spill stores/); st = substr($0, RSTART, RLENGTH - 19) }
      on && /Used [0-9]+ registers/ { match($0, /Used [0-9]+ registers/); print substr($0, RSTART + 5, RLENGTH - 15) " registers, " st " bytes spilled"; exit }'
  }
  fold=$(kstat pearl_tile_fold_wmma); rgfold=$(kstat pearl_tile_fold_wmma_rg)
  say "build ok sm_75 with $("$NVCC" --version | grep -o 'V[0-9][0-9.]*'): fold $fold; second kernel ${rgfold:-none}; $(elapsed) min"
fi
[ -n "${LAUNCH_BUILD_ONLY:-}" ] && { say "done (build only) $(elapsed) min"; exit 0; }

# ---- runs ----------------------------------------------------------------------------
# nvidia-smi samples, one a second, while a run lasts.
sample_start() {
  local s
  [ -n "$DRY" ] && { for s in $(seq 30); do echo "$(( 1835 + RANDOM % 20 )), 184.$(( RANDOM % 10 )), 71"; done > "$1"; return; }
  nvidia-smi -i "$GPU" --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits \
    -lms 1000 > "$1" 2> /dev/null &
  SAMP=$!
}
sample_stop() { [ -n "$SAMP" ] && { kill "$SAMP" 2> /dev/null; wait "$SAMP" 2> /dev/null; }; SAMP=; }
# A made-up reading for the dry run: a little noise, and +0.5% for band 6 and col_batch 4096.
fake_th() {
  local base=4780; case "$1" in *BAND=6*|*COL_BATCH=4096*) base=4804 ;; esac
  printf '%d.%02d' $(( (base + RANDOM % 20) / 100 )) $(( (base + RANDOM % 20) % 100 ))
}
fold_said() {  # the core's own line about the shape, once a point
  local f=$1 pt=$2 line
  [ -f "$W/said-$pt" ] && return
  line=$(grep -h -m1 -e 'launch sweep' -e 'fold:' "$f" 2> /dev/null | sed 's/.*fold: //' | cut -c1-300)
  [ -n "$line" ] && { touch "$W/said-$pt"; say "core $pt: $line"; }
}
# run_hr <round> <point> <env...>: hashrate.js, the number the app shows.
run_hr() {
  local r=$1 pt=$2; shift 2; local f=$W/runs/hr-$r-$pt
  sample_start "$f.smi"
  if [ -n "$DRY" ]; then
    echo "{\"th\":$(fake_th "$*"),\"sd\":0.3,\"n\":118}" > "$f.json"; : > "$f.err"
  else
    env "$@" node "$PROBES/hashrate.js" "$NODEF" "$HR_SECS" "$HR_WARM" > "$f.json" 2> "$f.err"
  fi
  sample_stop
  local th sd n mhz w c
  read -r th sd n <<< "$(node "$H" hr "$f.json")"
  read -r mhz w c <<< "$(node "$H" med "$f.smi" $(( HR_WARM + 4 )))"
  fold_said "$f.err" "$pt"
  [ "$th" = - ] && { say "FAIL hr r$r $pt: $(tail -c 300 "$f.err")"; return 1; }
  printf 'hr %s %s %s %s %s %s\n' "$r" "$pt" "$th" "$mhz" "$w" "$c" >> "$RES"
  say "hr r$r $pt th=$th sd=$sd n=$n mhz=$mhz w=$w C=$c thpg=$(awk -v t="$th" -v m="$mhz" 'BEGIN { if (m > 0) printf "%.2f", t / m * 1000; else print "-" }')"
}
# run_bench <round> <point> <env...>: the fold alone, synchronous batches, m = n = 131072.
run_bench() {
  local r=$1 pt=$2; shift 2; local f=$W/runs/bench-$r-$pt
  sample_start "$f.smi"
  if [ -n "$DRY" ]; then echo "  ->  $(fake_th "$*") TH/s" > "$f.out"
  else env "$@" "$B/bench" "$BENCH_SECS" > "$f.out" 2>&1; fi
  sample_stop
  local th mhz w c
  th=$(node "$H" bench "$f.out")
  read -r mhz w c <<< "$(node "$H" med "$f.smi" 4)"
  fold_said "$f.out" "b-$pt"
  [ "$th" = - ] && { say "FAIL bench r$r $pt: $(tail -c 300 "$f.out")"; return 1; }
  printf 'bench %s %s %s %s %s %s\n' "$r" "$pt" "$th" "$mhz" "$w" "$c" >> "$RES"
  say "bench r$r $pt th=$th mhz=$mhz w=$w C=$c"
}
# run_verify <point> <env...>: prints PASS or FAIL as its last line.
run_verify() {
  local pt=$1; shift; local f=$W/runs/verify-$pt hits ok bad salts pass
  if [ -n "$DRY" ]; then echo '{"hits":612,"verified":400,"failed":0,"salts":41,"PASS":true}' > "$f"
  else
    (cd "$SRC/earn" && env "$@" timeout -k 10 $(( VERIFY_SECS + 120 )) node native/probes/verify-hits.js \
      "$NODEF" "$VERIFY_SECS" > "$f" 2> "$f.err")
  fi
  read -r hits ok bad salts pass <<< "$(node "$H" verify "$f")"
  say "verify $pt hits=$hits verified=$ok failed=$bad salts=$salts PASS=$pass $(grep -o '"fails":\[[^]]*' "$f" | cut -c1-200)"
  [ "$pass" = true ]
}
# summarise <phase> <control> <points...>: a line a point, and BEST_PT / BEST_D.
summarise() {
  local phase=$1 a b c d e f g h; shift
  BEST_PT=-; BEST_D=-
  while read -r a b c d e f g h; do
    if [ "$a" = BEST ]; then BEST_PT=$b; BEST_D=$c
    else say "summary $phase $a runs=$b th=$c mhz=$d w=$e thpg=$f d=$g% wins=$h"; fi
  done <<< "$(node "$H" summary "$RES" "$phase" "$@")"
}

# 1. The default shape's hits.
fits $(( VERIFY_SECS + 60 )) && { run_verify ctl || die "the default shape failed verify-hits"; }

# 2. Band depth, on bench. The rule's band is what pearl_bd_band_for picks for this L2
# (column bands: 8, halved while 8 x 512 KB is over 75% of it, never below 2; then 5
# instead of 4 where 5 x 512 KB is at most 65% of it).
RULE=8; while [ $RULE -gt 2 ] && [ $(( RULE * 524288 * 100 )) -gt $(( L2 * 75 )) ]; do RULE=$(( RULE / 2 )); done
[ $RULE = 4 ] && [ $(( 5 * 524288 * 100 )) -le $(( L2 * 65 )) ] && RULE=5
if [ -z "${BANDS:-}" ]; then
  if [ "$L2" -le 3670016 ]; then BANDS="5 6"        # 3 MB: 2.5 and 3 MB of B
  elif [ "$L2" -lt 5242880 ]; then BANDS="4 6 8"    # 4 MB
  else BANDS="6 9 10"; fi                           # 5.5 MB and up
fi
BPTS=("b$RULE"); for b in $BANDS; do [ "$b" != "$RULE" ] && BPTS+=("b$b"); done
say "bands rule=$RULE others=$BANDS ($(( L2 / 1024 )) KB L2, 512 KB of B a column group)"
benv() { [ "$1" = "b$RULE" ] && return; echo "PEARL_BD_SWEEP_BAND=${1#b}"; }
for r in $(seq 1 "$ROUNDS"); do
  fits $(( ${#BPTS[@]} * (BENCH_SECS + 10) )) || { say "skip bench r$r: budget"; break; }
  order=("${BPTS[@]}"); [ $(( r % 2 )) = 0 ] && order=($(printf '%s\n' "${BPTS[@]}" | tac))
  for pt in "${order[@]}"; do run_bench "$r" "$pt" $(benv "$pt"); done
done
summarise bench "b$RULE" "${BPTS[@]}"
BAND_PT=; [ "$BEST_PT" != - ] && BAND_PT=$BEST_PT
# Only powers of two have run on a card, so the band that goes on is checked first.
if [ -n "$BAND_PT" ] && fits $(( VERIFY_SECS + 60 )); then
  run_verify "$BAND_PT" $(benv "$BAND_PT") || { say "dropping $BAND_PT: verify failed"; BAND_PT=; }
fi

# 3. The full miner loop.
declare -A ENVS=([ctl]="" [cb4096]="PEARL_BD_SWEEP_COL_BATCH=4096" [cb1024]="PEARL_BD_SWEEP_COL_BATCH=1024"
  [st1]="PEARL_BD_SWEEP_STREAMS=0")
PTS=(ctl)
[ -n "$BAND_PT" ] && { PTS+=("$BAND_PT"); ENVS[$BAND_PT]=$(benv "$BAND_PT"); }
PTS+=(cb4096 cb1024 st1)
RUN_S=$(( HR_SECS + HR_WARM + 8 ))
for r in $(seq 1 "$ROUNDS"); do
  fits $(( ${#PTS[@]} * RUN_S )) || { say "skip hr r$r: budget"; break; }
  order=("${PTS[@]}"); [ $(( r % 2 )) = 0 ] && order=($(printf '%s\n' "${PTS[@]}" | tac))
  for pt in "${order[@]}"; do run_hr "$r" "$pt" ${ENVS[$pt]}; done
done
summarise hr ctl "${PTS[@]}"

# 4. Confirm the best, if it is ahead.
CAND=$BEST_PT
if [ "$CAND" != - ] && awk -v d="$BEST_D" -v p="$CONFIRM_PCT" 'BEGIN { exit !(d >= p) }'; then
  if fits $(( 6 * RUN_S + VERIFY_SECS + 60 )); then
    say "confirm $CAND (d=$BEST_D%): 3 more A/B pairs"
    for i in 1 2 3; do
      if [ $(( i % 2 )) = 1 ]; then run_hr "c$i" ctl; run_hr "c$i" "$CAND" ${ENVS[$CAND]}
      else run_hr "c$i" "$CAND" ${ENVS[$CAND]}; run_hr "c$i" ctl; fi
    done
    sed 's/^hr /all /' "$RES" | grep '^all ' > "$W/all.tsv"; cat "$W/all.tsv" >> "$RES"
    summarise all ctl ctl "$CAND"
    case "$CAND" in b*|cb*) [ "$CAND" = "$BAND_PT" ] || run_verify "$CAND" ${ENVS[$CAND]} ;; esac
  else
    say "confirm $CAND skipped: budget"
  fi
else
  say "no shape beat ctl by ${CONFIRM_PCT}% (best $CAND $BEST_D%)"
fi
say "done $(elapsed) min; raw files in $W"
