#!/bin/bash
# Box script for tuning the FP8 core (earn/native/fp8) on a rented RTX 50 card. A measuring tool: nothing here ships.
#
# It sets up once (CUDA 13 compiler, Node, and, with PEARL_RAW=1, Pearl's own miner for its raw rate), then watches
# BRANCH of super3/llmjob. For each new commit it reads earn/tuning/salad/fp8tune.txt and, for every variant line
# there, builds pearl_core_fp8.node with that line's nvcc flags, runs fp8-verify.js (when the line asks) and times
# fp8-mine.js --bench for each shape. After the variants it times Pearl's pipeline (pip3raw.py) when the file says
# "pearl", so the two rates come from the same card a few minutes apart. Push a commit to start the next round;
# no restart is needed.
#
# fp8tune.txt, one item a line (# starts a comment):
#   variant NAME [verify] -- NVCC FLAGS...
#   shapes "m n k" "m n k" ...      the shapes fp8-mine.js times (h 4), default "8192 131072 2048"
#   secs N                          seconds a shape (default 40)
#   pearl SHAPE...                  Pearl's pipeline at these m x n x k shapes after the variants
#
# Every line is "[nvtest] HH:MM:SS [fp8t] <text>", at most 200 characters, for amdtest.py's log reader.
# Environment: BRANCH (default goal/fp8core), PEARL_RAW (1: install Pearl's miner), PEARL_SHA, POLL (seconds).
set -u
BRANCH=${BRANCH:-goal/fp8core}
PEARL_SHA=${PEARL_SHA:-1dd75a52e073b4a67079c7b6758228d71865ccea}
REPO=https://github.com/super3/llmjob.git
L=/root/llmjob
P=/root/pearl
export DEBIAN_FRONTEND=noninteractive PEARL_FP8=1
say() { echo "[nvtest] $(date -u +%T) [fp8t] $*" | cut -c1-230; sleep 0.03; }
emit() { local line; while IFS= read -r line || [ -n "$line" ]; do line=${line//$'\r'/ }; [ -z "${line// /}" ] || say "$line"; done; }
smi() { nvidia-smi -i 0 --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader 2>&1 | head -1; }

say "boot branch $BRANCH; card $(nvidia-smi -i 0 --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv,noheader 2>&1 | head -1)"
say "host $(nproc) threads, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo), $(. /etc/os-release; echo "$PRETTY_NAME")"
t0=$(date +%s)
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential python3 \
    make xz-utils; } > /root/apt.log 2>&1 || { tail -3 /root/apt.log | emit; say "FAIL apt"; }
if [ ! -x /usr/local/cuda/bin/nvcc ]; then
  rel=$(. /etc/os-release; echo "ubuntu${VERSION_ID//./}")
  { curl -fsSL --retry 5 "https://developer.download.nvidia.com/compute/cuda/repos/$rel/x86_64/cuda-keyring_1.1-1_all.deb" -o /root/kr.deb \
    && dpkg -i /root/kr.deb && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 cuda-cuobjdump-13-0; } \
    > /root/cuda.log 2>&1 || { tail -3 /root/cuda.log | emit; say "FAIL cuda toolkit"; }
  [ -d /usr/local/cuda ] || ln -sfn /usr/local/cuda-13.0 /usr/local/cuda
fi
if [ ! -x /root/node/bin/node ]; then
  { curl -fsSL --retry 5 https://nodejs.org/dist/v22.21.1/node-v22.21.1-linux-x64.tar.xz -o /root/node.txz \
    && mkdir -p /root/node && tar -C /root/node --strip-components=1 -xJf /root/node.txz; } > /root/node.log 2>&1 || say "FAIL node"
fi
export PATH=/root/node/bin:/usr/local/cuda/bin:/root/.local/bin:$PATH CUDA_PATH=/usr/local/cuda
say "tools in $(( $(date +%s) - t0 ))s: node $(node --version 2>&1), nvcc $(nvcc --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)"

if [ "${PEARL_RAW:-}" = 1 ] && [ ! -f /root/pearl.ok ]; then
  ( { [ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh; } > /root/uv.log 2>&1
    mkdir -p $P && cd $P && git init -q && git remote add origin https://github.com/pearl-research-labs/pearl.git \
      && git fetch -q --depth 1 origin "$PEARL_SHA" && git checkout -q FETCH_HEAD \
      && export UV_TORCH_BACKEND=cu130 UV_LINK_MODE=copy && uv python install 3.12 && uv sync --all-packages --all-groups \
      && touch /root/pearl.ok ) > /root/pearl.log 2>&1 &
fi

pearl_raw() {  # pearl_raw SHAPE...: Pearl's whole pipeline rate (pip3raw.py) on this card
  if [ ! -f /root/pearl.ok ]; then say "pearl not installed (yet): $(tail -1 /root/pearl.log 2> /dev/null | cut -c1-120)"; return; fi
  cp $L/earn/tuning/salad/pip3raw.py $P/miner/pearl-gemm/pip3raw.py
  ( cd $P/miner/pearl-gemm && PATH=$P/.venv/bin:$PATH timeout 900 python pip3raw.py 20 "$@" ) > /root/raw.log 2>&1
  grep -a '^RAW' /root/raw.log | emit
  grep -aq '^RAW' /root/raw.log || tail -4 /root/raw.log | emit
}

run_round() {
  local f=$L/earn/tuning/salad/fp8tune.txt shapes='"8192 131072 2048"' secs=40 pearl=""
  [ -f "$f" ] || { say "no fp8tune.txt"; return; }
  shapes=$(awk '$1 == "shapes" {$1 = ""; print}' "$f" | tail -1); [ -n "$shapes" ] || shapes='"8192 131072 2048"'
  secs=$(awk '$1 == "secs" {print $2}' "$f" | tail -1); secs=${secs:-40}
  pearl=$(awk '$1 == "pearl" {$1 = ""; print}' "$f" | tail -1)
  local vars line
  mapfile -t vars < <(grep -E '^variant ' "$f")
  for line in "${vars[@]}"; do
    local name flags verify
    name=$(echo "$line" | awk '{print $2}')
    flags=${line#*-- }
    case "$line" in *" verify "*) verify=1 ;; *) verify=0 ;; esac
    ( cd $L/earn/native && rm -rf fp8/build && bash fp8/build.sh $flags ) < /dev/null > /root/build-$name.log 2>&1 \
      || { tail -5 /root/build-$name.log | emit; say "$name FAIL build"; continue; }
    say "$name built ($flags): $(grep -a 'fp8_search_kernel' -A 2 $L/earn/native/fp8/cuda-build/ptxas.txt | grep -aoE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' ')"
    local node=$L/earn/native/fp8/build/Release/pearl_core_fp8.node
    if [ $verify = 1 ]; then
      timeout 900 node $L/earn/native/probes/fp8-verify.js $node < /dev/null > /root/verify-$name.log 2>&1
      say "$name verify: $(grep -ac '^PASS' /root/verify-$name.log) pass, $(grep -ac '^FAIL' /root/verify-$name.log) fail; $(tail -1 /root/verify-$name.log | cut -c1-80)"
      grep -a '^FAIL' /root/verify-$name.log | head -4 | cut -c1-150 | emit
    fi
    eval "set -- $shapes"
    for s in "$@"; do
      set -- $s
      ( sleep $((secs / 2)); smi > /root/smi.txt ) &
      timeout $((secs + 120)) node $L/earn/native/probes/fp8-mine.js $node --bench --m $1 --n $2 --k $3 --h 4 --secs $secs \
        < /dev/null > /root/mine.log 2>&1
      wait
      say "$name m $1 n $2 k $3: $(grep -a 'T-MAC/s' /root/mine.log | tail -1 | cut -c10-150); smi $(cat /root/smi.txt)"
    done
  done
  # shellcheck disable=SC2086
  [ -n "$pearl" ] && pearl_raw $pearl
}

last=""
while :; do
  head=$(git ls-remote $REPO "refs/heads/$BRANCH" 2> /dev/null | cut -c1-40)
  if [ -n "$head" ] && [ "$head" != "$last" ]; then
    { [ -d $L/.git ] || git clone -q --depth 1 -b "$BRANCH" $REPO $L; } > /root/git.log 2>&1
    ( cd $L && git fetch -q --depth 1 origin "$BRANCH" && git checkout -q -f FETCH_HEAD ) >> /root/git.log 2>&1 \
      || { tail -2 /root/git.log | emit; say "FAIL fetch"; sleep 60; continue; }
    last=$head
    say "round at ${head:0:8}: $(cd $L && git log -1 --format=%s | cut -c1-120)"
    run_round
    say "round ${head:0:8} done"
  fi
  sleep "${POLL:-45}"
done
