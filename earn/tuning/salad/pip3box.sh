#!/bin/bash
# Box script for the PIP-3 (FP8) readiness test: Pearl's own fp8-scheme reference miner on one rented card.
# It builds the miner (a vLLM plugin with CuTe DSL kernels that compile at run time) and the fp8-scheme node, then
# runs the stages in STAGES and prints each result. A measuring tool: nothing here ships.
#
# The reference miner only loads on Hopper (sm_90) and Blackwell (sm_100, sm_120: RTX 50, RTX PRO 6000).
#
# Stages (STAGES, default "gemm raw e2e"):
#   gemm   Pearl's kernel tests (miner/pearl-gemm/tests, not the slow ones): pass, skip and fail counts
#   raw    the four-op mining pipeline without vLLM (pip3raw.py): T-MAC/s (1e12 m*n*k a second) per shape, power
#   e2e    Pearl's end-to-end test: vLLM + pearl-gateway + pearld --regtest. It passes when the node accepts a
#          block mined on this card, so it is the consensus verifier's word on this card's arithmetic.
#   vload  the reference miner serving MODEL under a prefill load for VLOAD_SECS: T-MAC/s it credits, power
#
# Every line is "[nvtest] HH:MM:SS [pip3] <text>", so amdtest.py's log reader picks it, in pieces of at most 200
# characters: the 330-character lines of the first run never reached Salad's log service. Nothing here prints
# environment variables. The same script runs over SSH on a Vast box: `bash pip3box.sh`.
#
# Environment:
#   PEARL_SHA   commit of pearl-research-labs/pearl to test (default: fp8-scheme's head on 18 November)
#   STAGES      see above
#   RAW_SECS, RAW_SHAPES   pip3raw.py's seconds per shape (20) and shapes ("m x n x k ...")
#   MODEL, VLOAD_SECS, VLOAD_THREADS, VLOAD_UTIL   the vload stage (Qwen/Qwen3-0.6B, 120, 8, 0.85)
#   BASE_SHA    commit of super3/llmjob to fetch pip3raw.py from (Salad sets it; over SSH put pip3raw.py in /root)
#   NO_IDLE     1: exit at the end instead of sleeping (SSH runs)
set -u
PEARL_SHA=${PEARL_SHA:-1dd75a52e073b4a67079c7b6758228d71865ccea}
STAGES=" ${STAGES:-gemm raw e2e} "
P=/root/pearl
export DEBIAN_FRONTEND=noninteractive
say() { echo "[nvtest] $(date -u +%T) [pip3] $*"; sleep 0.03; }
emit() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//$'\r'/ }
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 200 ]; do say "${line:0:200}"; line="+${line:200}"; done
    say "$line"
  done
}
finish() { say "all done"; [ "${NO_IDLE:-}" = 1 ] && exit 0; exec sleep infinity; }
fail() { say "FAIL $*"; finish; }
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
# Colour codes out, so a log line is plain text.
plain() { sed -e 's/\x1b\[[0-9;]*m//g'; }

say "boot pearl ${PEARL_SHA:0:12}, stages$STAGES"
say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,memory.total --format=csv,noheader 2>&1 | head -1)"
say "host $(nproc) threads, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo), kernel $(uname -r), $(. /etc/os-release; echo "$PRETTY_NAME"), disk $(df -h / | awk 'NR==2 {print $4}') free"
major=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1 | cut -d. -f1 | tr -d ' ')
case "$major" in '' | *[!0-9]*) fail "no driver version from nvidia-smi" ;; esac
[ "$major" -ge 580 ] || fail "driver $major: the miner's CUDA 13 runtime needs 580 or newer"

# ---------------------------------------------------------------- tools
t0=$(date +%s)
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential \
    pkg-config libatomic1 libssl-dev clang; } > /root/apt.log 2>&1 || { tail -4 /root/apt.log | emit; fail "apt-get"; }
# FlashInfer compiles its sampler with nvcc when vLLM first samples. A base image has no toolkit, so take CUDA
# 13.0's compiler and headers from NVIDIA's apt repository. A devel image already has nvcc.
if (has e2e || has vload) && ! command -v nvcc > /dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  rel=$(. /etc/os-release; echo "ubuntu${VERSION_ID//./}")
  { curl -fsSL --retry 5 --retry-all-errors "https://developer.download.nvidia.com/compute/cuda/repos/$rel/x86_64/cuda-keyring_1.1-1_all.deb" \
      -o /root/keyring.deb && dpkg -i /root/keyring.deb && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
      cuda-nvrtc-dev-13-0 libcurand-dev-13-0; } > /root/cuda.log 2>&1 || { tail -4 /root/cuda.log | emit; say "WARN no CUDA 13.0 toolkit: e2e and vload may fail at the sampler"; }
  [ -d /usr/local/cuda ] || ln -sfn /usr/local/cuda-13.0 /usr/local/cuda
fi
[ -x /usr/local/go/bin/go ] || { curl -fsSL --retry 5 --retry-all-errors https://go.dev/dl/go1.26.10.linux-amd64.tar.gz -o /root/go.tgz \
  && tar -C /usr/local -xzf /root/go.tgz; } || fail "go download"
[ -x /root/.cargo/bin/cargo ] || { curl -fsSL --retry 5 --retry-all-errors https://sh.rustup.rs | sh -s -- -y --profile minimal; } > /root/rustup.log 2>&1 \
  || fail "rustup"
[ -x /root/.local/bin/uv ] || { curl -LsSf --retry 5 --retry-all-errors https://astral.sh/uv/install.sh | sh; } > /root/uv.log 2>&1 || fail "uv"
export PATH=$P/.venv/bin:/usr/local/cuda/bin:/usr/local/go/bin:/root/.cargo/bin:/root/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
say "tools ok in $(( $(date +%s) - t0 ))s: $(go version | cut -d' ' -f3), $(cargo --version | cut -d' ' -f1-2), $(uv --version), nvcc $(nvcc --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)"

# ---------------------------------------------------------------- the miner and the node
t0=$(date +%s)
if [ ! -d $P/.git ]; then
  rm -rf $P && mkdir -p $P && cd $P || fail "no $P"
  { git init -q && git remote add origin https://github.com/pearl-research-labs/pearl.git \
    && git fetch -q --depth 1 origin "$PEARL_SHA" && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1 \
    || { tail -3 /root/git.log | emit; fail "fetch pearl $PEARL_SHA"; }
fi
cd $P || fail "no $P"
say "pearl $(git rev-parse --short=12 HEAD) of $(git log -1 --format=%cs)"
( export UV_TORCH_BACKEND=cu130 UV_LINK_MODE=copy
  uv python install 3.12 > /root/uvsync.log 2>&1
  uv sync --all-packages --all-groups >> /root/uvsync.log 2>&1 ) &
UVPID=$!
if has e2e && [ ! -x $P/bin/pearld ]; then
  ( cd zk-pow && cargo run --release --no-default-features --bin build_cache src/v4/api/v4_cache.bin \
      src/v2/circuit/v2_cache.bin src/v1/v1_cache.bin ) > /root/zkcache.log 2>&1 || { tail -4 /root/zkcache.log | emit; say "FAIL zk cache"; }
  ( cd zk-pow/bindings/go && cargo build --release ) > /root/zkgobind.log 2>&1 || { tail -4 /root/zkgobind.log | emit; say "FAIL zk go binding"; }
  ( cd xmss && make CC=gcc CXX=g++ ) > /root/xmss.log 2>&1 || { tail -4 /root/xmss.log | emit; say "FAIL xmss"; }
  mkdir -p bin
  CGO_ENABLED=1 CGO_LDFLAGS_ALLOW=".*zk_pow_ffi.*" go build -tags xmss,zkpow -o bin/pearld ./node > /root/pearld.log 2>&1 \
    || { tail -6 /root/pearld.log | emit; say "FAIL pearld build"; }
  say "pearld $([ -x bin/pearld ] && echo built || echo missing) at $(( $(date +%s) - t0 ))s"
fi
wait $UVPID || { tail -6 /root/uvsync.log | emit; fail "uv sync"; }
say "miner installed at $(( $(date +%s) - t0 ))s: $(python -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)' 2>&1 | tail -1)"
say "torch sees $(python -c 'import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())' 2>&1 | tail -1)"
say "committed device $(python -c 'from miner_base.devices import local_device; print(local_device())' 2>&1 | tail -1)"

# ---------------------------------------------------------------- stages
if has gemm; then
  t0=$(date +%s)
  ( cd miner/pearl-gemm && timeout 3000 python -m pytest -n "${GEMM_WORKERS:-4}" -m "not slow" tests -p no:cacheprovider \
      -p no:randomly -q -rf ) > /root/gemm.log 2>&1
  say "gemm rc=$? in $(( $(date +%s) - t0 ))s: $(grep -aE '^=+ .*(passed|failed|error).* =+$' /root/gemm.log | tail -1 | tr -d '=')"
  grep -a '^FAILED\|^ERROR' /root/gemm.log | head -12 | cut -c1-380 | emit
fi

if has raw; then
  RAW=/root/pip3raw.py
  [ -f $RAW ] || curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/${BASE_SHA:-none}/earn/tuning/salad/pip3raw.py" -o $RAW \
    || say "FAIL fetch pip3raw.py"
  if [ -f $RAW ]; then
    cp $RAW miner/pearl-gemm/pip3raw.py
    # shellcheck disable=SC2086
    ( cd miner/pearl-gemm && timeout 1800 python pip3raw.py "${RAW_SECS:-20}" ${RAW_SHAPES:-} ) > /root/raw.log 2>&1
    say "raw rc=$?"
    grep -a '^RAW' /root/raw.log | emit
    grep -aq '^RAW' /root/raw.log || tail -6 /root/raw.log | emit
  fi
fi

if has e2e; then
  t0=$(date +%s)
  rm -rf /root/e2e
  # The prover's memory: the largest resident size of any process while the test runs.
  ( while sleep 2; do ps -eo rss= | sort -n | tail -1; done > /root/rss.log ) &
  RSSPID=$!
  PEARLD_BIN=$P/bin/pearld timeout 2700 python -m pytest miner/vllm-miner/tests/test_e2e_regtest.py -x -s -m integration \
    --basetemp=/root/e2e -p no:cacheprovider > /root/e2e.log 2>&1
  rc=$?
  kill $RSSPID 2> /dev/null
  D=$(ls -d /root/e2e/test_vllm_mines_block_verified0 2> /dev/null)
  say "e2e rc=$rc in $(( $(date +%s) - t0 ))s: $(grep -aE '(passed|failed|skipped|error).* in [0-9.]+s' /root/e2e.log | tail -1 | tr -d '=')"
  say "e2e largest process $(sort -n /root/rss.log | tail -1 | awk '{printf "%.1f GB", $1 / 1048576}')"
  if [ -n "$D" ]; then
    grep -a 'Accepted block\|Block accepted\|rejected' "$D/pearld.log" | tail -3 | cut -c1-380 | emit
    grep -a 'Fp8Prover\|Submitting block\|accepted by node\|rejected\|Rejecting' "$D/gateway.log" | plain | head -6 | cut -c1-380 | emit
    say "e2e block candidates $(grep -ac 'block candidate' "$D/vllm.log"), variants $(grep -ac 'pipeline variant ready' "$D/vllm.log")"
    if [ "$rc" != 0 ]; then
      grep -a 'Error\|ERROR\|refus\|unsupported' "$D/vllm.log" | plain | tail -8 | cut -c1-380 | emit
      grep -a '^E ' /root/e2e.log | head -4 | cut -c1-380 | emit
    fi
  else
    tail -6 /root/e2e.log | plain | cut -c1-380 | emit
  fi
fi

if has vload; then
  MODEL=${MODEL:-Qwen/Qwen3-0.6B}
  L=/root/pip3load.py
  [ -f $L ] || curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/${BASE_SHA:-none}/earn/tuning/salad/pip3load.py" -o $L \
    || say "FAIL fetch pip3load.py"
  if [ -f $L ]; then
    python $L patch $P > /root/patch.log 2>&1 || { tail -3 /root/patch.log | emit; say "FAIL hash counter patch"; }
    rm -f /root/hash.log
    MINER_NO_GATEWAY=true python -m vllm.entrypoints.cli.main serve "$MODEL" --quantization pearl --port 8000 \
      --max-model-len 8192 --gpu-memory-utilization "${VLOAD_UTIL:-0.85}" > /root/vllm.log 2>&1 &
    VP=$!
    up=0
    for _ in $(seq 1 360); do
      curl -sf localhost:8000/health > /dev/null && { up=1; break; }
      kill -0 $VP 2> /dev/null || break
      sleep 5
    done
    if [ $up = 1 ]; then
      say "vload $MODEL up; $(grep -a 'Reserved' /root/vllm.log | plain | grep -o 'Reserved.*bytes' | head -1)"
      say "vload shapes: $(grep -a 'pipeline variant ready' /root/vllm.log | grep -o 'm=[0-9]*, n=[0-9]*, k=[0-9]*' | tr -d ' ' | tr '\n' ' ' | cut -c1-300)"
      ( while sleep 10; do nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm,memory.used,temperature.gpu --format=csv,noheader; done > /root/smi.log ) &
      SP=$!
      python $L load http://localhost:8000 "$MODEL" "${VLOAD_THREADS:-8}" "${VLOAD_SECS:-120}" 2>&1 | grep -a '^LOAD' | tail -2 | emit
      kill $SP 2> /dev/null
      # The first line covers the ramp, so leave it out of the mean.
      say "vload $MODEL: $(awk 'NR > 1 {s += $2; n++; if ($2 > mx) mx = $2} END {if (n) printf "mean %.2f T-MAC/s, best %.2f, over %d ten-second lines", s / n, mx, n; else print "no hash lines"}' /root/hash.log 2> /dev/null)"
      say "vload smi (middle of the run): $(sed -n "$(( ($(wc -l < /root/smi.log) + 1) / 2 ))p" /root/smi.log)"
      grep -a 'WARNING\|ERROR' /root/vllm.log | plain | sed -e 's/^.*\(WARNING\|ERROR\)/\1/' | cut -c1-200 | sort | uniq -c | sort -rn | head -4 | emit
    else
      say "vload FAIL $MODEL did not come up"
      grep -a 'Error\|ERROR' /root/vllm.log | plain | tail -6 | cut -c1-380 | emit
    fi
    kill $VP 2> /dev/null; sleep 5; pkill -f vllm.entrypoints 2> /dev/null
  fi
fi
finish
