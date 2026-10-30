#!/bin/bash
# Box script for testing the AMD (HIP) Pearl core on a Salad AMD PC. amdtest.py's container command fetches this
# file at SCRIPT_SHA (default BASE_SHA) and runs it. It installs what the build needs, fetches the repo at BASE_SHA
# into /opt/llmjob, applies PATCH_GZ_B64 if set, then runs TASKS in order and sleeps, so the logs can be read and
# `amdtest.py set-env` can start the next run on the same PC.
#
# bench.sh sources it with AMDTEST_STAGE=lib for its functions (setup, fetch, target, the t_ tasks, mine_cli)
# and runs none of the tasks.
#
# Every line it prints for amdtest.py is "[amdtest] HH:MM:SS <tag> ...". A command's output is "<tag> | <line>",
# cut into pieces of at most 400 characters ("+" starts a continuation) so Salad's log service keeps each one.
# It never prints environment variables: Salad's bridge sets some, and nothing here needs them shown. A "run"
# script is the sender's responsibility.
#
# Environment (amdtest.py sets it):
#   BASE_SHA      commit of super3/llmjob to test (required)
#   PATCH_GZ_B64  optional: `git diff --binary` against BASE_SHA, gzipped and base64'd; applied with git apply.
#                 If it changes this script, the patched script takes over after the patch.
#   TASKS         comma list, run in order (default "card,hip"). Each is name[@fold][+K=V...][:arg[:arg...]];
#                 @fold sets PEARL_AMD_FOLD and each +K=V sets K for that task only, so one run can compare
#                 settings on the same PC (hashrate+PEARL_AMD_COL_BATCH=128:60:10). V can't hold , : or +.
#     env                     OS, ROCm, node, compiler, the bridge's files (names only)
#     card                    /dev/dxg, rocminfo's GPU agents, Salad's amd-smi
#     hip                     builds and runs amdtest_hip.cpp: devices, runtime libraries loaded, kernels checked
#     check[:all|:gfx...]     ./build-amd.sh check for this card's target (or all 14, or the ones given)
#     build[:gfx...]          ./build-amd.sh gpu for this card's target (or the ones given)
#     foldtest[:m:cols:seed:reps:check]   build/pearl_amd_foldtest
#     verify[@fold][:secs:bits]           node ../probes/verify-hits.js build/pearl_core_hip.node [120]
#     profile[@fold][:secs:bits:m:n:fill] node test/verify-hits-profile.js build/pearl_core_hip.node ...
#     hashrate[@fold][:secs:warm]         node ../probes/hashrate.js build/pearl_core_hip.node [60 10]
#     mine[@fold][:secs]      the app's CLI with PEARL_CORE_VARIANT=amd on POOL; secs 0 = until stopped [600]
#     srb[:secs:warm]         SRBMiner 3.8.2 on DUEL_POOL for secs [300] after a warm-up of its own [60]
#     duel[:first:secs:warm]  benchmark.md's AMD method on this PC: first (ours or srb) [ours] warms the card for
#                             warm [60], then each runs secs [300]; ends with "duel <name> result ..." for each
#     cpuchecks               build-amd.sh cpu and test/run-cpu-checks.sh on this box's CPU (HIP-CPU, TBB)
#     run                     the bash script in RUN_GZ_B64 (gzip+base64), from earn/native/amd
#     sleep:secs              wait
#   ARCH          gfx target to build for (default: rocminfo's first GPU agent)
#   POOL, WORKER, WALLET      for mine (default prl-us.kryptex.network:7048, amdtest, the user's wallet)
#   DUEL_POOL     for srb and duel, both miners (default us.pearl.herominers.com:1200, the table's pool)
#
# Salad's PCs run the container in WSL2: the GPU is behind /dev/dxg, and Salad puts its ROCm-to-WSL bridge
# (/opt/rocm-wsl) first on PATH and LD_LIBRARY_PATH. This script only appends to PATH and never sets
# LD_LIBRARY_PATH. Builds link libamdhip64 with a RUNPATH of /opt/rocm/lib, which LD_LIBRARY_PATH overrides.
export DEBIAN_FRONTEND=noninteractive
R=${AMDTEST_ROOT:-/opt/llmjob}   # AMDTEST_ROOT and AMDTEST_NM: for trying the script off a box
A=$R/earn/native/amd
NM=${AMDTEST_NM:-/opt/nm}
NODE_VER=v22.22.0
CLANG=/opt/rocm/lib/llvm/bin/clang++
export PATH="$PATH:/opt/rocm/bin:/opt/node/bin"
export NODE_ADDON_API=$NM/node_modules/node-addon-api
POOL=${POOL:-prl-us.kryptex.network:7048}
WORKER=${WORKER:-amdtest}
WALLET=${WALLET:-prl1pw88rlj2y4cazjej2gzavhwxut2js9gws936eazajzrll5ehjul8sfp5u3t}
TASKS=${TASKS:-card,hip}

say() { echo "[amdtest] $(date -u +%T) $*"; }
# emit TAG: every non-empty stdin line as "TAG | line", in pieces of at most 400 characters.
emit() {
  local tag=$1 line
  tr '\r' '\n' | while IFS= read -r line || [ -n "$line" ]; do
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 400 ]; do say "$tag | ${line:0:400}"; line="+${line:400}"; done
    say "$tag | $line"
  done
}
idle() { say "idle $*"; sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }

setup() {
  local pkgs="" t0=$(date +%s)
  command -v git >/dev/null || pkgs="$pkgs git"
  command -v curl >/dev/null || pkgs="$pkgs curl ca-certificates"
  command -v make >/dev/null && [ -d /usr/include/c++ ] || pkgs="$pkgs build-essential"
  command -v xz >/dev/null || pkgs="$pkgs xz-utils"
  if [ -n "$pkgs" ]; then
    { apt-get update -qq && apt-get install -y -qq --no-install-recommends $pkgs; } > /root/apt.log 2>&1 \
      || { tail -5 /root/apt.log | emit setup; fail "apt-get install$pkgs"; }
  fi
  if [ "$(node -v 2>/dev/null | cut -d. -f1)" != v22 ]; then
    mkdir -p /opt/node
    curl -fsSL --retry 3 "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.gz" \
      | tar xz -C /opt/node --strip-components=1 || fail "node $NODE_VER download"
    # /opt/node/bin is last on PATH; put node first only if the image has an older one.
    hash -r; [ "$(node -v 2>/dev/null | cut -d. -f1)" = v22 ] || export PATH="/opt/node/bin:$PATH"
  fi
  if [ ! -f "$NODE_ADDON_API/napi.h" ]; then
    npm install --prefix $NM --no-audit --no-fund --loglevel=error node-addon-api tweetnacl tweetnacl-util \
      > /root/npm.log 2>&1 || { tail -5 /root/npm.log | emit setup; fail "npm install"; }
  fi
  say "setup ok in $(( $(date +%s) - t0 ))s: node $(node -v), apt:${pkgs:- none}"
}

fetch() {
  rm -rf $R && mkdir -p $R && cd $R || fail "no $R"
  if ! { git init -q && git remote add origin https://github.com/super3/llmjob.git \
         && git fetch -q --depth 1 origin "$BASE_SHA" && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1; then
    tail -3 /root/git.log | emit fetch
    fail "fetch $BASE_SHA"
  fi
  say "fetch ok $(git rev-parse --short=12 HEAD)"
  if [ -n "${PATCH_GZ_B64:-}" ]; then
    printf '%s' "$PATCH_GZ_B64" | base64 -d | gunzip > /root/patch.diff || fail "patch: not gzip+base64"
    say "patch $(wc -c < /root/patch.diff) bytes, sha256 $(sha256sum /root/patch.diff | cut -c1-12)"
    git apply --whitespace=nowarn /root/patch.diff > /root/apply.log 2>&1 \
      || { emit patch < /root/apply.log; fail "patch does not apply to $BASE_SHA"; }
    git diff --stat | tail -n 30 | emit patch
  fi
}

# The card's target, from rocminfo's first GPU agent.
target() {
  timeout 60 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1
}

t_env() {
  say "env os $(. /etc/os-release; echo "$PRETTY_NAME"), kernel $(uname -r), $(nproc) cpus, $(free -m | awk '/Mem:/{print $2}') MiB"
  say "env rocm $(cat /opt/rocm/.info/version 2>/dev/null || echo ?), hip-clang $([ -x $CLANG ] && $CLANG --version | head -1 || echo missing)"
  say "env node $(node -v 2>/dev/null), git $(git --version 2>/dev/null | cut -d' ' -f3), g++ $(g++ -dumpversion 2>/dev/null || echo none)"
  # Facts about the bridge, without printing PATH or LD_LIBRARY_PATH themselves.
  say "env rocm-wsl first on PATH: $(case "$PATH" in /opt/rocm-wsl/*) echo yes;; *) echo no;; esac)," \
      "LD_LIBRARY_PATH set: $([ -n "${LD_LIBRARY_PATH:-}" ] && echo yes || echo no)," \
      "has /opt/rocm-wsl: $(case ":${LD_LIBRARY_PATH:-}:" in *:/opt/rocm-wsl*) echo yes;; *) echo no;; esac)"
  ls /opt/rocm-wsl 2>&1 | tr '\n' ' ' | emit env
  for d in /opt/rocm-wsl/lib /opt/rocm-wsl/lib64 /usr/lib/wsl/lib; do
    [ -d $d ] && say "env $d: $(ls $d | tr '\n' ' ')"
  done
  return 0
}

t_card() {
  [ -c /dev/dxg ] && say "card /dev/dxg yes" || say "card /dev/dxg MISSING"
  ls -l /dev/kfd /dev/dri 2>&1 | head -3 | emit card
  local ri; ri=$(timeout 60 rocminfo 2>&1); local rc=$?
  echo "$ri" | awk -F': +' '/^\*+$/{a++} /Marketing Name:/{m=$2} /^  Name:/{n=$2}
    /Compute Unit:/{cu=$2} /Max Clock Freq/{mhz=$2}
    /Device Type:/{if ($2 ~ /GPU/) g=1}
    /ISA Info:/{if (g) {print "gpu " n ", " m ", CUs " cu ", " mhz " MHz"; g=0}}' | sed 's/  */ /g' | emit card
  [ $rc -eq 0 ] || { echo "$ri" | tail -5 | emit card; say "card rocminfo rc=$rc"; }
  local smi=/opt/rocm-wsl/bin/amd-smi
  [ -x $smi ] && timeout 20 $smi static --asic --vram 2>&1 | grep -iE 'market|asic|vram_size|vram_type|device_id|gfx|target' | head -12 | emit card
  say "card target ${ARCH:-$(target)}"
}

t_hip() {
  local arch=${ARCH:-$(target)} bin=/root/amdtest_hip
  [ -n "$arch" ] || { say "hip FAIL no gfx target from rocminfo"; return 1; }
  [ -x $CLANG ] || { say "hip FAIL no hip-clang at $CLANG"; return 1; }
  $CLANG -x hip --offload-arch=$arch --rocm-path=/opt/rocm -std=c++17 -O2 "$R/tuning/salad/amdtest_hip.cpp" \
    -o $bin -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/hipcc.log 2>&1 \
    || { tail -20 /root/hipcc.log | emit hip; say "hip FAIL compile"; return 1; }
  ldd $bin | grep -E 'amdhip|hsa|dxg|rocprof|not found' | sed 's/ (0x[0-9a-f]*)//' | emit hip
  timeout 120 $bin > /root/hip.log 2>&1; local rc=$?
  emit hip < /root/hip.log
  if [ $rc -ne 0 ]; then
    # Once more with HIP's own log, which says where the runtime gave up.
    AMD_LOG_LEVEL=3 timeout 120 $bin > /root/hip3.log 2>&1
    grep -vE '^\s*$' /root/hip3.log | tail -30 | emit hip
    say "hip FAIL rc=$rc"
    return 1
  fi
}

t_check() {
  local archs
  case "${1:-}" in all) archs="" ;; "") archs=${ARCH:-$(target)} ;; *) archs="$*" ;; esac
  (cd $A && ./build-amd.sh check $archs) > /root/check.log 2>&1; local rc=$?
  grep -vE '^\s*$' /root/check.log | tail -n 80 | emit check
  return $rc
}

t_build() {
  local archs="${*:-${ARCH:-$(target)}}" t0=$(date +%s)
  (cd $A && ./build-amd.sh gpu $archs) > /root/build.log 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then
    grep -E 'error|Error|build-amd' /root/build.log | head -20 | emit build
    tail -10 /root/build.log | emit build
  else
    tail -1 /root/build.log | emit build
    say "build $(grep -c 'warning:' /root/build.log) warnings, $(( $(date +%s) - t0 ))s"
  fi
  return $rc
}

need_core() { [ -f $A/build/pearl_core_hip.node ] || { say "$1 FAIL no build/pearl_core_hip.node (run build first)"; return 1; }; }

t_foldtest() {
  [ -x $A/build/pearl_amd_foldtest ] || { say "foldtest FAIL no build/pearl_amd_foldtest (run build first)"; return 1; }
  (cd $A && timeout 1800 ./build/pearl_amd_foldtest "$@") > /root/foldtest.log 2>&1; local rc=$?
  emit foldtest < /root/foldtest.log
  return $rc
}

t_verify() {
  need_core verify || return 1
  (cd $A && timeout 3600 node ../probes/verify-hits.js build/pearl_core_hip.node "${1:-120}" ${2:+"$2"}) \
    > /root/verify.log 2>&1; local rc=$?
  tail -n 20 /root/verify.log | emit verify
  return $rc
}

t_profile() {
  need_core profile || return 1
  (cd $A && timeout 3600 node test/verify-hits-profile.js build/pearl_core_hip.node "$@") > /root/profile.log 2>&1
  local rc=$?; tail -n 20 /root/profile.log | emit profile
  return $rc
}

t_hashrate() {
  need_core hashrate || return 1
  (cd $A && timeout 3600 node ../probes/hashrate.js build/pearl_core_hip.node "${1:-60}" "${2:-10}") \
    > /root/hashrate.log 2>&1; local rc=$?
  # hashrate.js asks nvidia-smi for clocks every 2 s; there is none here. PEARL_AMD_TRACE=1 adds the core's
  # "[pearl-amd]" timing lines, shown first.
  grep '^\[pearl-amd\]' /root/hashrate.log | head -n 40 | emit hashrate
  grep -v 'nvidia-smi' /root/hashrate.log | grep -v '^\[pearl-amd\]' | tail -n 10 | emit hashrate
  return $rc
}

# The app's CLI from source, with the AMD core, mining on POOL as WORKER for WALLET. It ends in exec, so call it
# in a subshell. t_mine and bench.sh's ours both run it.
mine_cli() {
  cd $R/earn && PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$A/build/pearl_core_hip.node NODE_PATH=$NM/node_modules \
    exec node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$POOL" -w "$WORKER" \
      --no-report --no-serve --no-update
}

# Every 60 s: the CLI's new lines, but only the last of its once-a-second "TH/s" lines.
t_mine() {
  need_core mine || return 1
  local secs=${1:-600} t0=$(date +%s) seen=0 n pid
  say "mine $POOL worker $WORKER for ${secs}s (0 = until stopped)"
  (mine_cli) > /root/mine.log 2>&1 &
  pid=$!
  while :; do
    sleep 60
    n=$(wc -l < /root/mine.log)
    tail -n +$((seen + 1)) /root/mine.log | head -n $((n - seen)) | tr '\r' '\n' | grep -v 'TH/s ·' | head -20 | emit mine
    tail -n +$((seen + 1)) /root/mine.log | head -n $((n - seen)) | tr '\r' '\n' | grep 'TH/s ·' | tail -1 | emit mine
    seen=$n
    kill -0 $pid 2>/dev/null || { say "mine exited"; tail -5 /root/mine.log | emit mine; wait $pid; return 1; }
    [ "$secs" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "$secs" ] && break
  done
  kill $pid; wait $pid 2>/dev/null
  return 0
}

# SRBMiner 3.8.2 from its official release, checked against the published MD5 (bench.sh downloads the same file).
SRB_DIR=/opt/srb
srb_get() {
  [ -x $SRB_DIR/SRBMiner-MULTI ] && return 0
  mkdir -p $SRB_DIR && cd $SRB_DIR || return 1
  curl -fsSL --retry 3 -o srb.tgz https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.2/SRBMiner-Multi-3-8-2-Linux.tar.gz \
    || { say "srb FAIL download"; return 1; }
  echo "9f64784a9894067b7bc3811b4912dd81  srb.tgz" | md5sum -c --quiet || { say "srb FAIL checksum"; return 1; }
  tar xzf srb.tgz && cp "$(ls -d $SRB_DIR/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI" $SRB_DIR/ || { say "srb FAIL unpack"; return 1; }
}
# Both miners mine on DUEL_POOL in srb and duel, as benchmark.md's AMD table did (bench.sh's pool).
DUEL_POOL=${DUEL_POOL:-us.pearl.herominers.com:1200}
srb_cli() {
  cd $SRB_DIR && exec ./SRBMiner-MULTI --algorithm-gpu pearlhash --pool "$DUEL_POOL" --wallet "$WALLET" \
    --worker "$WORKER-srb" --disable-cpu --api-enable --api-port 21550 --disable-gpu-nvidia --disable-gpu-intel
}
ours_cli() { POOL=$DUEL_POOL WORKER=$WORKER-ours mine_cli; }
# One sample as "TH/s accepted rejected": SRBMiner's from its API, ours from the CLI's last "TH/s ·" line.
sample_srb() {
  curl -s -m 3 http://127.0.0.1:21550 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{
    const a=JSON.parse(s).algorithms[0];console.log((a.hashrate.gpu.total/1e12).toFixed(2),a.shares.accepted,a.shares.rejected)}catch(e){}})'
}
sample_ours() {
  tr '\r' '\n' < /root/duel-ours.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1 \
    | awk '{gsub(",","",$4); gsub(",","",$7); print $1, $4, $7}'
}
# run_miner NAME SECS MEASURED: runs ours or srb for SECS. Measured, it samples every 10 s, prints a line a minute,
# and ends with "NAME result ...": the mean of the samples after the first minute, and the miner's last figure,
# which is what benchmark.md's table takes for SRBMiner ("its own figure at the end").
run_miner() {
  local name=$1 secs=$2 measured=$3 pid t0 t s th acc rej n=0 sum=0 last="" fails=0
  say "duel $name start ${secs}s$([ "$measured" = 1 ] || echo ' (warm-up, not measured)')"
  ( ${name}_cli ) > /root/duel-$name.log 2>&1 &
  pid=$!; t0=$(date +%s)
  while :; do
    sleep 10; t=$(( $(date +%s) - t0 ))
    if ! kill -0 $pid 2>/dev/null; then
      say "duel $name exited at ${t}s"; tr '\r' '\n' < /root/duel-$name.log | grep -v '^\s*$' | tail -5 | emit duel
      fails=1; break
    fi
    if [ "$measured" = 1 ]; then
      s=$(sample_$name); read -r th acc rej <<< "$s"
      if [ -n "$th" ]; then
        last="$th acc=$acc rej=$rej"
        [ $t -gt 60 ] && { n=$((n + 1)); sum=$(awk "BEGIN{print $sum + $th}"); }
      fi
      if [ $((t % 60)) -lt 10 ]; then
        say "duel $name t=${t}s th=${th:-?} acc=${acc:-?} rej=${rej:-?}"
        grep '^\[pearl-amd\]' /root/duel-$name.log | tail -n 1 | emit duel   # with PEARL_AMD_TRACE=1
      fi
    fi
    [ $t -ge "$secs" ] && break
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  pkill -f SRBMiner-MULTI; pkill -f 'src/cli/earn-cli\.js'; sleep 8
  if [ "$measured" = 1 ]; then
    tr '\r' '\n' < /root/duel-$name.log | grep -iE 'error|reject|fail|stale' | grep -v 'TH/s ·' | tail -5 | emit duel
    say "duel $name result mean_after_60s=$( [ $n -gt 0 ] && awk "BEGIN{printf \"%.2f\", $sum / $n}" || echo '?') n=$n last=${last:-?} secs=$secs"
  fi
  return $fails
}

# One miner for SECS after a WARM-second warm-up of itself.
t_srb() {
  srb_get || return 1
  local secs=${1:-300} warm=${2:-60}
  [ "$warm" -gt 0 ] && run_miner srb "$warm" 0
  run_miner srb "$secs" 1
}

# benchmark.md's AMD method on one PC: the first miner warms the card for WARM seconds, unmeasured, then each
# runs SECS seconds. FIRST is ours or srb. Ours is the CLI with this core (needs build first).
t_duel() {
  local first=${1:-ours} secs=${2:-300} warm=${3:-60} second rc=0
  need_core duel || return 1
  srb_get || return 1
  case $first in ours) second=srb ;; srb) second=ours ;; *) say "duel FAIL first must be ours or srb"; return 1 ;; esac
  say "duel pool $DUEL_POOL, first $first, ${secs}s each after a ${warm}s warm-up"
  [ "$warm" -gt 0 ] && run_miner $first "$warm" 0
  run_miner $first "$secs" 1 || rc=1
  run_miner $second "$secs" 1 || rc=1
  return $rc
}

t_cpuchecks() {
  { apt-get install -y -qq --no-install-recommends libtbb-dev clang > /root/apt2.log 2>&1 \
    && { [ -d /opt/hip-cpu ] || git clone -q --depth 1 https://github.com/ROCm/HIP-CPU /opt/hip-cpu; }; } \
    || { tail -5 /root/apt2.log | emit cpuchecks; say "cpuchecks FAIL setup"; return 1; }
  (cd $A && HIP_CPU_PATH=/opt/hip-cpu ./build-amd.sh cpu && ./test/run-cpu-checks.sh) > /root/cpuchecks.log 2>&1
  local rc=$?; grep -E '^(PASS|FAIL|built| )' /root/cpuchecks.log | tail -n 40 | emit cpuchecks
  return $rc
}

t_run() {
  [ -n "${RUN_GZ_B64:-}" ] || { say "run FAIL no RUN_GZ_B64"; return 1; }
  printf '%s' "$RUN_GZ_B64" | base64 -d | gunzip > /root/run.sh || { say "run FAIL not gzip+base64"; return 1; }
  (cd $A && bash /root/run.sh) > /root/run.log 2>&1; local rc=$?
  tail -n 60 /root/run.log | emit run
  return $rc
}

# AMDTEST_STAGE=lib: sourced for its functions only (bench.sh does this), so stop here.
[ "${AMDTEST_STAGE:-}" = lib ] && return 0

if [ "${AMDTEST_STAGE:-}" != repo ]; then
  say "boot sha ${BASE_SHA:-none}, tasks $TASKS, patch ${#PATCH_GZ_B64} chars"
  [ -n "${BASE_SHA:-}" ] || idle "no BASE_SHA"
  setup
  fetch
  # A patch that changes this script hands over to the patched copy, which skips what is done.
  if ! cmp -s "$0" "$R/tuning/salad/amdtest.sh"; then
    say "patched amdtest.sh takes over"
    AMDTEST_STAGE=repo exec bash "$R/tuning/salad/amdtest.sh"
  fi
fi

failed=""
IFS=, read -ra LIST <<< "$TASKS"
for item in "${LIST[@]}"; do
  [ -n "$item" ] || continue
  IFS=: read -ra P <<< "$item"
  IFS=+ read -ra HP <<< "${P[0]}"
  envs=("${HP[@]:1}")
  name=${HP[0]%%@*}; fold=""; [ "$name" != "${HP[0]}" ] && fold=${HP[0]#*@}
  [ -n "$fold" ] && envs+=("PEARL_AMD_FOLD=$fold")
  args=("${P[@]:1}")
  case $name in env|card|hip|check|build|foldtest|verify|profile|hashrate|mine|srb|duel|cpuchecks|run) ;;
    sleep) sleep "${args[0]:-60}"; continue ;;
    *) say "$name FAIL unknown task"; failed="$failed $item"; continue ;;
  esac
  say "$name start${envs:+ ${envs[*]}}${args:+ ${args[*]}}"
  t0=$(date +%s)
  # A subshell, so a task's own settings end with it.
  ( for kv in "${envs[@]}"; do export "$kv"; done; "t_$name" "${args[@]}" )
  rc=$?
  say "$name done rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || failed="$failed $item"
done
say "all done, failed:${failed:- none}"
idle "until the next set-env"
