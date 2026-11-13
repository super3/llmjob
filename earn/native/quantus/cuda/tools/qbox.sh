#!/bin/bash
# Box script for testing the Quantus CUDA core on a Salad PC (or any Linux box with an NVIDIA card). The container
# command fetches this file at SCRIPT_SHA (default BASE_SHA) and runs it, as nvtest.sh's boxes do. It installs CUDA
# 12.8.1's compiler (the archives and checksums nvtest.sh pins) and Node 22, fetches the repo at BASE_SHA, builds
# quantus_cuda.node for this card with cuda/build.sh, and runs the STAGES in order:
#
#   verify   verify-core.js on card 0 (the five checks)
#   ubench   integer issue rates per SM per clock (tools/ubench.cu)
#   race     ROUNDS rounds of: ours, official, each SECS seconds (tools/qcbench.cu; the official kernel is
#            quantus-miner's mining.cu at a pinned commit, checked by SHA-256), with clock, power and temperature
#            every 5 s. Each leg ends with an "end" line: mean MH/s and mean clock, power, temperature.
#   srb      SRBMiner-MULTI SRB_VER for SRB_SECS seconds against tools/mockpool.js on 127.0.0.1 (no wallet, nothing
#            leaves the box but SRBMiner's own dev-fee connection): its API hashrate and the pool's valid-share rate,
#            sampled every 15 s, with clock, power and temperature. Observation only.
#
# Then it sleeps, so the logs can be read and set-env can start the next run. Every line is
# "[nvtest] HH:MM:SS <text>", at most 400 characters. Nothing here prints environment variables.
#
# Environment: BASE_SHA (required), STAGES (default "verify ubench race srb"), ROUNDS (2), SECS (60), SRB_SECS (300),
# DEFS (extra nvcc -D flags for our build), OURS_ALT (a second build of ours with these -D flags, raced as "alt").
set -u
R=/opt/llmjob
CUDA=/opt/cuda
NODE=/opt/node
B=/opt/qb
RD=https://developer.download.nvidia.com/compute/cuda/redist
NVCC_VER=V12.8.93
CUDA_PARTS="cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz|dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f"
NODE_TGZ=node-v22.23.2-linux-x64.tar.xz
NODE_SHA=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
OFF_SHA=7d6af15a245733558ca7c263515bd95bb3d25cfc
OFF_SRC_SHA=2d9de99d2c8fdf70bca69cebabd4b5ebb63f805448d1b2cf43ff5d2fc4c241f5
SRB_VER=${SRB_VER:-3.8.6}; SRB_MD5=${SRB_MD5:-38128fa15afff3c488e53726aa251249}
export DEBIAN_FRONTEND=noninteractive

say() { echo "[nvtest] $(date -u +%T) $*"; }
emit() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//$'\r'/ }
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 380 ]; do say "${line:0:380}"; line="+${line:380}"; done
    say "$line"
  done
}
idle() { say "idle $*"; sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }
smi() { nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

setup() {
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
      wget jq procps; } > /root/apt.log 2>&1 || { tail -5 /root/apt.log | emit; fail "apt-get install"; }
  if [ "$("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)" != "$NVCC_VER" ]; then
    rm -rf "$CUDA" && mkdir -p "$CUDA" /root/dl || fail "no $CUDA"
    while IFS='|' read -r rel sha; do
      f=/root/dl/${rel##*/}
      curl -fsSL --retry 3 "$RD/$rel" -o "$f" || fail "cuda download ${rel##*/}"
      echo "$sha  $f" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "cuda checksum ${rel##*/}"
      tar -xJf "$f" -C "$CUDA" --strip-components=1 || fail "cuda unpack ${rel##*/}"
      rm -f "$f"
    done <<< "$CUDA_PARTS"
    ln -sfn lib "$CUDA/lib64"
    mkdir -p "$CUDA/targets/x86_64-linux"
    ln -sfn ../../include "$CUDA/targets/x86_64-linux/include"
    ln -sfn ../../lib "$CUDA/targets/x86_64-linux/lib"
  fi
  if [ ! -x "$NODE/bin/node" ]; then
    curl -fsSL --retry 3 "https://nodejs.org/dist/v22.23.2/$NODE_TGZ" -o /root/node.txz || fail "node download"
    echo "$NODE_SHA  /root/node.txz" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "node checksum"
    mkdir -p "$NODE" && tar -xJf /root/node.txz -C "$NODE" --strip-components=1 || fail "node unpack"
  fi
  export PATH="$NODE/bin:$CUDA/bin:$PATH"
  say "setup ok: nvcc $("$CUDA/bin/nvcc" --version | grep -oE 'V[0-9.]+' | tail -1), node $(node --version)"
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

build() {
  local cc t0 Q=$R/earn/native/quantus/cuda
  t0=$(date +%s)
  cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  [ -n "$cc" ] || fail "no compute capability"
  mkdir -p $B
  OUT=$B/quantus_cuda.node DEFS="${DEFS:-}" CUDA_HOME=$CUDA NODE_INCLUDE=$NODE/include/node bash $Q/build.sh "$cc" \
    > $B/build.log 2>&1 || { tail -8 $B/build.log | emit; fail "build"; }
  G="-gencode=arch=compute_$cc,code=sm_$cc"
  nvcc -O3 -std=c++17 $G ${DEFS:-} -I $Q -o $B/ours $Q/tools/qcbench.cu $Q/qc_host.cu > $B/b2.log 2>&1 \
    || { tail -8 $B/b2.log | emit; fail "build qcbench"; }
  if [ -n "${OURS_ALT:-}" ]; then
    nvcc -O3 -std=c++17 $G $OURS_ALT -I $Q -o $B/alt $Q/tools/qcbench.cu $Q/qc_host.cu > $B/b3.log 2>&1 \
      || { tail -8 $B/b3.log | emit; fail "build alt"; }
  fi
  nvcc -O3 -std=c++17 $G -o $B/ubench $Q/tools/ubench.cu > $B/b4.log 2>&1 || { tail -8 $B/b4.log | emit; fail "build ubench"; }
  mkdir -p $B/off
  curl -fsSL --retry 3 "https://raw.githubusercontent.com/Quantus-Network/quantus-miner/$OFF_SHA/crates/engine-cuda/src/kernels/mining.cu" \
    -o $B/off/mining.cu || fail "official kernel download"
  echo "$OFF_SRC_SHA  $B/off/mining.cu" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "official kernel checksum"
  nvcc -O3 -std=c++17 $G -DOFFICIAL -I $B/off -o $B/official $Q/tools/qcbench.cu > $B/b5.log 2>&1 \
    || { tail -8 $B/b5.log | emit; fail "build official"; }
  say "build ok sm_$cc in $(( $(date +%s) - t0 ))s"
}

verify() {
  cd $R/earn || fail "no earn"
  node native/quantus/verify-core.js --addon $B/quantus_cuda.node --device 0 2>&1 | emit
  say "verify rc=${PIPESTATUS[0]}"
}

# leg NAME SECS: one timed run of $B/NAME, sampling the card every 5 s.
leg() {
  local name=$1 secs=$2 pid n=0 c=0 w=0 t=0 s
  "$B/$name" "$secs" > "$B/$name.out" 2>&1 & pid=$!
  sleep 10
  while kill -0 $pid 2> /dev/null; do
    s=$(smi)
    if [ -n "$s" ]; then
      IFS=, read -r sc sw st su <<< "$s"
      n=$((n + 1)); c=$(awk -v a=$c -v b=$sc 'BEGIN{print a+b}'); w=$(awk -v a=$w -v b=$sw 'BEGIN{print a+b}')
      t=$(awk -v a=$t -v b=$st 'BEGIN{print a+b}')
    fi
    sleep 5
  done
  wait $pid
  local mean; mean=$(grep -oE 'mean [0-9.]+' "$B/$name.out" | awk '{print $2}')
  [ -n "$mean" ] || { tail -3 "$B/$name.out" | emit; }
  say "end $name ${mean:-?} MH/s over ${secs}s; $(awk -v n=$n -v c=$c -v w=$w -v t=$t 'BEGIN{if(n) printf "%.0f MHz %.1f W %.0f C (n=%d)", c/n, w/n, t/n, n; else print "no samples"}')"
}

race() {
  local r legs="ours official"
  [ -n "${OURS_ALT:-}" ] && legs="ours alt official"
  for r in $(seq 1 "${ROUNDS:-2}"); do
    for l in $legs; do leg "$l" "${SECS:-60}"; done
  done
}

ubench() { $B/ubench 2000 2>&1 | emit; }

srb() {
  local d=/opt/srb srbx t0 api p
  mkdir -p $d && cd $d || fail "no $d"
  if [ ! -d "$d/SRBMiner-Multi-${SRB_VER//./-}" ]; then
    curl -fsSL --retry 3 -o srb.tgz \
      "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
      || { say "srb download failed"; return; }
    echo "$SRB_MD5  srb.tgz" | md5sum -c --quiet - > /dev/null 2>&1 || { say "srb checksum failed"; return; }
    tar xzf srb.tgz || { say "srb unpack failed"; return; }
  fi
  srbx=$d/SRBMiner-Multi-${SRB_VER//./-}/SRBMiner-MULTI
  # A well-formed Quantus address (SS58 prefix 189) for a random key nobody holds: only the mock pool ever sees it.
  local addr; addr=$(node $R/earn/native/quantus/cuda/tools/qzaddr.js)
  node $R/earn/native/quantus/cuda/tools/mockpool.js --port 3334 --difficulty "${SRB_DIFF:-4000000000}" > $d/pool.log 2>&1 & p=$!
  sleep 2
  (cd "$(dirname "$srbx")" && "$srbx" --algorithm quantus --pool 127.0.0.1:3334 --wallet "$addr" --disable-cpu \
     --api-enable --api-port 21550 > $d/srb.log 2>&1) &
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "${SRB_SECS:-300}" ]; do
    sleep 15
    api=$(curl -s -m 3 http://127.0.0.1:21550 | jq -r '[.algorithms[0].hashrate.gpu.total, .algorithms[0].shares.accepted] | @tsv' 2>/dev/null)
    say "sample srb t=$(( $(date +%s) - t0 )) hs=${api:-?} gpu=$(smi) pool=$(grep -a '^t ' $d/pool.log | tail -1)"
  done
  pkill -f SRBMiner-MULTI; sleep 3; kill $p 2> /dev/null
  say "end srb: $(tr '\r' '\n' < $d/srb.log | grep -a -v '^\s*$' | tail -2 | tr '\n' ' ' | LC_ALL=C tr -cd ' -~' | tail -c 300)"
  grep -a -E 'rejected|login|method' $d/pool.log | head -5 | emit
}

say "boot qbox sha ${BASE_SHA:-none}, stages ${STAGES:-default}"
[ -n "${BASE_SHA:-}" ] || idle "no BASE_SHA"
say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv,noheader 2>&1 | head -1)"
setup
fetch
build
for st in ${STAGES:-verify ubench race srb}; do
  say "stage $st"
  case $st in
    verify) verify ;;
    ubench) ubench ;;
    race) race ;;
    srb) srb ;;
    *) say "unknown stage $st" ;;
  esac
done
say "all done"
idle "until the next set-env"
