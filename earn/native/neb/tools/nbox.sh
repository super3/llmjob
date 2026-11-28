#!/bin/bash
# Box script for testing the NEB cores on a Salad PC.
#
# NVIDIA: a leased test box runs it through nvtest.sh's NVTEST_SCRIPT (earn/tuning/salad/amdtest.py create ... --env
# NVTEST_SCRIPT=earn/native/neb/tools/nbox.sh). It installs CUDA 12.8.1's compiler (the archives and checksums
# nvtest.sh pins) and Node 22, fetches the repo at BASE_SHA and builds neb_cuda.node with cuda/build.sh.
#
# AMD: amdtest.sh's `run` task runs it with NBOX_VENDOR=amd (RUN_GZ_B64 is a one-line script: NBOX_VENDOR=amd bash
# ../neb/tools/nbox.sh). amdtest.sh has already installed Node and fetched the repo; this builds neb_hip.node with
# amd/build.sh for the card's target. The run task keeps a script's output to itself until it ends, so in this mode
# the lines go straight to the container's log, tagged [amdtest].
#
# Either way it downloads the network's dataset file from the NEB release (checked by SHA-256) and runs the STAGES
# in order:
#
#   verify   verify-core.js on card 0: the chain's vectors and the checks on the test dataset
#   live     verify-core.js on the network's dataset and the model the pool is on now (slot LIVE_SLOT, default 0)
#   mine     tools/nebmine.js on the pool for SECS seconds, paid to NEB_ADDRESS; shares must be accepted
#   srb      SRBMiner-MULTI SRB_VER on the same pool and address for SECS seconds: its API hashrate every 15 s, with
#            clock, power and temperature. Observation only: nothing of it is read but what it prints.
#   race     ROUNDS rounds of mine then srb, SECS seconds each
#   bench    tools/nebbench.js: the core's rates on two recorded pool models, with no pool; then the same for each
#            build in VARIANTS ("name=flags;name=flags", compiler flags such as -DNB_SCREEN_LOG2=24)
#
# Then it sleeps (NVIDIA) or returns to amdtest.sh (AMD), so the logs can be read and set-env can start the next run.
# Every line is "[nvtest] HH:MM:SS <text>" ("[amdtest] HH:MM:SS neb <text>" on AMD), at most 400 characters. Nothing here prints environment variables.
#
# Environment: BASE_SHA (required), STAGES (default "verify live mine"), SECS (300), ROUNDS (2), NEB_ADDRESS (a
# neb1 address; required by live, mine, srb and race), NEB_POOL (neb.rabidmining.com:3470), WORKER (n1), ARCH (the
# card's own compute capability), DEFS (extra nvcc -D flags), BIG (verify's big count, default 2^30), TRIPLES and
# LIVE_TRIPLES (rows of the triple table in verify and live, default 768; 0 for none), TRIPLE_GB (its size limit, 4),
# LIVE_SLOT (the dataset live checks, default 0).
set -u
R=/opt/llmjob
CUDA=/opt/cuda
NODE=/opt/node
B=/opt/nb
RD=https://developer.download.nvidia.com/compute/cuda/redist
NVCC_VER=V12.8.93
CUDA_PARTS="cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b"
NODE_TGZ=node-v22.23.2-linux-x64.tar.xz
NODE_SHA=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
# The main network's dataset file, from the NEB release v1.0.11 (its SHA256SUMS, signed by the release key).
DATASET_URL=https://github.com/SilasEigen/neb/releases/download/v1.0.11/neb-datasets.bin
DATASET_SHA=57e22a4b8aacecb2ebae62f76307ddccb04198ed5c4ba15c1dee90ca7a7c2174
DATASET=/opt/neb-datasets.bin
SRB_VER=${SRB_VER:-3.9.1}; SRB_MD5=${SRB_MD5:-7576bfb1bfdc68a53986acace25e8901}
POOL=${NEB_POOL:-neb.rabidmining.com:3470}
export DEBIAN_FRONTEND=noninteractive

AMD=0; [ "${NBOX_VENDOR:-nvidia}" = amd ] && AMD=1
ADDON=$B/neb_cuda.node; [ $AMD = 1 ] && ADDON=$B/neb_hip.node
# AMD: PID 1 is the container's command, and its output is the container's log.
say() { if [ $AMD = 1 ]; then echo "[amdtest] $(date -u +%T) neb $*" > /proc/1/fd/1; else echo "[nvtest] $(date -u +%T) $*"; fi; }
emit() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//$'\r'/ }
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 380 ]; do say "${line:0:380}"; line="+${line:380}"; done
    say "$line"
  done
}
idle() { say "idle $*"; [ $AMD = 1 ] && exit 1; sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }
smi() { [ $AMD = 1 ] && return; nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

install_cuda() {
  local dir=$1 ver=$2 parts=$3 rel sha f
  [ "$("$dir/bin/nvcc" --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)" = "$ver" ] && return
  rm -rf "$dir" && mkdir -p "$dir" /root/dl || fail "no $dir"
  while IFS='|' read -r rel sha; do
    f=/root/dl/${rel##*/}
    curl -fsSL --retry 3 "$RD/$rel" -o "$f" || fail "cuda download ${rel##*/}"
    echo "$sha  $f" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "cuda checksum ${rel##*/}"
    tar -xJf "$f" -C "$dir" --strip-components=1 || fail "cuda unpack ${rel##*/}"
    rm -f "$f"
  done <<< "$parts"
  ln -sfn lib "$dir/lib64"
  mkdir -p "$dir/targets/x86_64-linux"
  ln -sfn ../../include "$dir/targets/x86_64-linux/include"
  ln -sfn ../../lib "$dir/targets/x86_64-linux/lib"
  ln -sfn ../../lib "$dir/targets/x86_64-linux/lib64"
}

dataset() {
  if ! echo "$DATASET_SHA  $DATASET" | sha256sum -c --quiet - > /dev/null 2>&1; then
    curl -fsSL --retry 3 "$DATASET_URL" -o "$DATASET" || fail "dataset download"
    echo "$DATASET_SHA  $DATASET" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "dataset checksum"
  fi
}

setup() {
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make       wget jq procps python3; } > /root/apt.log 2>&1 || { tail -5 /root/apt.log | emit; fail "apt-get install"; }
  install_cuda "$CUDA" "$NVCC_VER" "$CUDA_PARTS"
  if [ ! -x "$NODE/bin/node" ]; then
    curl -fsSL --retry 3 "https://nodejs.org/dist/v22.23.2/$NODE_TGZ" -o /root/node.txz || fail "node download"
    echo "$NODE_SHA  /root/node.txz" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "node checksum"
    mkdir -p "$NODE" && tar -xJf /root/node.txz -C "$NODE" --strip-components=1 || fail "node unpack"
  fi
  export PATH="$NODE/bin:$CUDA/bin:$PATH"
  say "setup ok: nvcc $("$CUDA/bin/nvcc" --version | grep -oE 'V[0-9.]+' | tail -1), node $(node --version), $(nproc) threads"
}

fetch() {
  rm -rf "$R" && mkdir -p "$R" && cd "$R" || fail "no $R"
  if ! { git init -q && git remote add origin https://github.com/super3/llmjob.git \
         && git fetch -q --depth 1 origin "$BASE_SHA" && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1; then
    tail -3 /root/git.log | emit
    fail "fetch $BASE_SHA"
  fi
  say "fetch ok $(git rev-parse --short=12 HEAD)"
}

# build_addon OUT FLAGS: the core for this card, with extra compiler flags.
build_addon() {
  local out=$1 flags=$2 cc
  if [ $AMD = 1 ]; then
    cc=${ARCH:-$(timeout 60 rocminfo 2> /dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1)}
    OUT=$out NH_CFLAGS="$flags" bash $R/earn/native/neb/amd/build.sh $cc > $B/build.log 2>&1 || return 1
    echo "$cc"
    return 0
  fi
  cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d '. ')
  CUDA_HOME=$CUDA OUT=$out DEFS="$flags" bash $R/earn/native/neb/cuda/build.sh "${ARCH:-$cc}" > $B/build.log 2>&1 || return 1
  echo "sm_${ARCH:-$cc}"
}

build() {
  local t0 cc
  mkdir -p $B
  t0=$(date +%s)
  cc=$(build_addon $ADDON "${DEFS:-}") || { tail -25 $B/build.log | emit; fail "build"; }
  say "build ok $cc in $(( $(date +%s) - t0 ))s"
  regs base | emit
}

# regs NAME: each kernel's register count from the last build's log (ptxas -v), as "registers NAME kernel: Used n registers".
regs() {
  awk -v n="$1" '/Compiling entry function/ { match($0, /nbk_[a-z0-9]+/); k = substr($0, RSTART, RLENGTH) }
    /Used [0-9]+ registers/ { match($0, /Used [0-9]+ registers/); print "registers", n, k ": " substr($0, RSTART, RLENGTH) }' $B/build.log | sort -u
}

# bench: tools/nebbench.js on the core as built, then on each build in VARIANTS ("name=flags;name=flags").
bench() {
  cd $R/earn || fail "no earn"
  local spec name flags specs
  node native/neb/tools/nebbench.js --addon $ADDON --dataset $DATASET --label base --secs "${BENCH_SECS:-4}" \
    --triples "${BENCH_TRIPLES:-0}" --triple-gb "${TRIPLE_GB:-20}" 2>&1 | emit
  IFS=';' read -ra specs <<< "${VARIANTS:-}"
  for spec in "${specs[@]}"; do
    name=${spec%%=*}; flags=${spec#*=}
    build_addon $B/v_$name.node "$flags" > /dev/null || { tail -8 $B/build.log | emit; say "bench $name: build failed"; continue; }
    regs "$name" | grep -a nbk_screen | emit
    node native/neb/tools/nebbench.js --addon $B/v_$name.node --dataset $DATASET --label "$name" --secs "${BENCH_SECS:-4}" \
      --triples "${BENCH_TRIPLES:-0}" --triple-gb "${TRIPLE_GB:-20}" 2>&1 | emit
  done
}

verify() {
  cd $R/earn || fail "no earn"
  node native/neb/verify-core.js --addon $ADDON --device 0 --big "${BIG:-1073741824}" --triples "${TRIPLES:-768}" --triple-gb "${TRIPLE_GB:-4}" 2>&1 | emit
  say "verify rc=${PIPESTATUS[0]}"
}

live() {
  cd $R/earn || fail "no earn"
  [ -n "${NEB_ADDRESS:-}" ] || { say "live: no NEB_ADDRESS"; return; }
  node native/neb/verify-core.js --addon $ADDON --device 0 --big "${BIG:-1073741824}" --triples "${LIVE_TRIPLES:-768}" --triple-gb "${TRIPLE_GB:-4}" --dataset $DATASET \
    --pool "$POOL" --address "$NEB_ADDRESS" --slot "${LIVE_SLOT:-0}" 2>&1 | emit
  say "live rc=${PIPESTATUS[0]}"
}

# watch PID NAME: the card every 15 s while PID runs; then its means.
watch() {
  local pid=$1 name=$2 n=0 c=0 w=0 t=0 s sc sw st su
  while kill -0 "$pid" 2> /dev/null; do
    s=$(smi)
    if [ -n "$s" ]; then
      IFS=, read -r sc sw st su <<< "$s"
      n=$((n + 1)); c=$(awk -v a=$c -v b="$sc" 'BEGIN{print a+b}'); w=$(awk -v a=$w -v b="$sw" 'BEGIN{print a+b}')
      t=$(awk -v a=$t -v b="$st" 'BEGIN{print a+b}')
    fi
    sleep 15
  done
  say "card $name: $(awk -v n=$n -v c=$c -v w=$w -v t=$t 'BEGIN{if(n) printf "%.0f MHz %.1f W %.0f C (n=%d)", c/n, w/n, t/n, n; else print "no samples"}')"
}

mine() {
  cd $R/earn || fail "no earn"
  [ -n "${NEB_ADDRESS:-}" ] || { say "mine: no NEB_ADDRESS"; return; }
  local pid
  UV_THREADPOOL_SIZE=8 node native/neb/tools/nebmine.js --addon $ADDON --dataset $DATASET --pool "$POOL" \
    --address "$NEB_ADDRESS" --worker "${WORKER:-n1}" --secs "${SECS:-300}" > $B/mine.log 2>&1 & pid=$!
  ( tail -n +1 -f --pid=$pid $B/mine.log 2> /dev/null | grep --line-buffered -v 'share accepted' | sed -u 's/^/ours /' | emit ) &
  watch $pid ours
  wait $pid
  say "mine rc=$? $(grep -c 'share accepted' $B/mine.log) accepted lines"
}

srb() {
  [ -n "${NEB_ADDRESS:-}" ] || { say "srb: no NEB_ADDRESS"; return; }
  local d=/opt/srb srbx t0 api pid
  mkdir -p $d && cd $d || fail "no $d"
  if [ ! -d "$d/SRBMiner-Multi-${SRB_VER//./-}" ]; then
    curl -fsSL --retry 3 -o srb.tgz \
      "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
      || { say "srb download failed"; return; }
    echo "$SRB_MD5  srb.tgz" | md5sum -c --quiet - > /dev/null 2>&1 || { say "srb checksum failed"; return; }
    tar xzf srb.tgz || { say "srb unpack failed"; return; }
  fi
  srbx=$d/SRBMiner-Multi-${SRB_VER//./-}/SRBMiner-MULTI
  (cd "$(dirname "$srbx")" && exec "$srbx" --algorithm neb --pool "$POOL" --wallet "$NEB_ADDRESS.${WORKER:-n1}srb" \
     --disable-cpu --api-enable --api-port 21550 ${SRB_ARGS:-} > $d/srb.log 2>&1) & pid=$!
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "${SECS:-300}" ] && kill -0 $pid 2> /dev/null; do
    sleep 15
    api=$(curl -s -m 3 http://127.0.0.1:21550 | jq -r '[.algorithms[0].hashrate.gpu.total, .algorithms[0].shares.accepted, .algorithms[0].shares.rejected] | @tsv' 2>/dev/null | tr '\t' ' ')
    say "sample srb t=$(( $(date +%s) - t0 )) hs/accepted/rejected=${api:-?} gpu=$(smi)"
  done
  pkill -f SRBMiner-MULTI; sleep 3
  tr '\r' '\n' < $d/srb.log | grep -a -v '^\s*$' | LC_ALL=C tr -cd ' -~\n' | grep -a -i -E 'neb|hashrate|accepted|rejected|error|gpu0' | tail -12 | sed 's/^/srb /' | emit
  say "end srb"
}

race() {
  local r
  for r in $(seq 1 "${ROUNDS:-2}"); do mine; srb; done
}

say "boot nbox sha ${BASE_SHA:-none}, stages ${STAGES:-default}"
[ -n "${BASE_SHA:-}" ] || idle "no BASE_SHA"
if [ $AMD = 1 ]; then
  command -v jq > /dev/null 2>&1 || apt-get install -y -qq --no-install-recommends jq > /dev/null 2>&1
  say "card $(timeout 60 rocminfo 2> /dev/null | grep -m1 -E 'Marketing Name:.*(Radeon|AMD)' | sed 's/.*Marketing Name: *//'), $(nproc) threads"
else
  say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,clocks.max.sm,memory.total --format=csv,noheader 2>&1 | head -1)"
  setup
  fetch
fi
dataset
build
for st in ${STAGES:-verify live mine}; do
  say "stage $st"
  case $st in
    verify) verify ;;
    live) live ;;
    mine) mine ;;
    srb) srb ;;
    race) race ;;
    bench) bench ;;
    *) say "unknown stage $st" ;;
  esac
done
say "all done"
[ $AMD = 1 ] && exit 0
idle "until the next set-env"
