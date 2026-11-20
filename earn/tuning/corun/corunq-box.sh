#!/bin/bash
# Box script for issue #312's co-run test with the Quantus core, and the switch-time runs for both cores, on one
# Linux NVIDIA box (a Salad PC). A measuring tool: nothing here ships.
#
# It installs CUDA 12.8.1's compiler and runtime, cuBLAS 12.8 (node-llama-cpp's CUDA build loads it), Node 22 and
# node-llama-cpp 3.22.3, downloads Gemma 4 E4B Q4_K_M, builds quantus_cuda.node for this card from QSHA (with the
# patched RTX 40 kernel, as CI does), takes v0.7.7's Linux Pearl core from the release, and runs corunq.mjs
# (BASE_SHA's copy) through the runs in STAGES. Every result is a "[nvtest] HH:MM:SS [corunq] R <json>" line.
#
# Environment: BASE_SHA (this branch's commit: the harness), QSHA (the Quantus source commit), QPER (Quantus
# blocks a SM, for the free-SM runs), STAGES (default "corun switch"), SECS (90), WARM (30).
set -u
W=/root/corun
CUDA=/opt/cuda
NODE=/opt/node
RD=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_PARTS="cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz|dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f
cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz|8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae"
CUBLAS="libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz|21718957c2cf000bacd69d36c95708a2319199e39e056f8b4f0f68e3b9f323bb"
NODE_TGZ=node-v22.23.2-linux-x64.tar.xz
NODE_SHA=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
MODEL_URL=https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
REL=https://github.com/super3/llmjob/releases/download/v0.7.7
STAGES=" ${STAGES:-corun switch} "
SECS=${SECS:-90}
WARM=${WARM:-30}
QPER=${QPER:-3}
export DEBIAN_FRONTEND=noninteractive

say() { echo "[nvtest] $(date -u +%T) [corunq] $*"; }
emit() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//$'\r'/ }
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 380 ]; do say "${line:0:380}"; line="+${line:380}"; done
    say "$line"
  done
}
idle() { say "idle $*"; exec sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
[ -n "${BASE_SHA:-}" ] && [ -n "${QSHA:-}" ] || idle "no BASE_SHA or QSHA"

say "boot harness ${BASE_SHA:0:12} quantus ${QSHA:0:12} stages$STAGES"
say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,memory.total --format=csv,noheader 2>&1 | head -1)"
say "host $(nproc) threads, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo), disk $(df -h / | awk 'NR==2 {print $4}') free"

# ---------------------------------------------------------------- setup
t0=$(date +%s)
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 procps; } > /root/apt.log 2>&1 || { tail -5 /root/apt.log | emit; fail "apt-get"; }
get_part() {  # get_part REL SHA DIR [tar members...]
  local rel=$1 sha=$2 dir=$3 f; shift 3
  f=/root/dl/${rel##*/}
  mkdir -p /root/dl "$dir"
  curl -fsSL --retry 3 "$RD/$rel" -o "$f" || fail "download ${rel##*/}"
  echo "$sha  $f" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "checksum ${rel##*/}"
  tar -xJf "$f" -C "$dir" --strip-components=1 --wildcards "$@" || fail "unpack ${rel##*/}"
  rm -f "$f"
}
while IFS='|' read -r rel sha; do get_part "$rel" "$sha" $CUDA; done <<< "$CUDA_PARTS"
ln -sfn lib $CUDA/lib64
mkdir -p $CUDA/targets/x86_64-linux
ln -sfn ../../include $CUDA/targets/x86_64-linux/include
ln -sfn ../../lib $CUDA/targets/x86_64-linux/lib
get_part "${CUBLAS%%|*}" "${CUBLAS##*|}" $CUDA '*/lib/libcublas.so*' '*/lib/libcublasLt.so*'
curl -fsSL --retry 3 "https://nodejs.org/dist/v22.23.2/$NODE_TGZ" -o /root/node.txz || fail "node download"
echo "$NODE_SHA  /root/node.txz" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "node checksum"
mkdir -p $NODE && tar -xJf /root/node.txz -C $NODE --strip-components=1 && rm -f /root/node.txz || fail "node unpack"
export PATH="$NODE/bin:$CUDA/bin:$PATH" LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
say "toolkit ok in $(( $(date +%s) - t0 ))s: nvcc $(nvcc --version | grep -oE 'V[0-9.]+' | tail -1), node $(node --version)"

mkdir -p $W && cd $W || fail "no $W"
for f in corunq.mjs power.mjs profile.json prio_shim.c; do
  curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/corun/$f" -o $W/$f \
    || fail "fetch $f"
done
for f in pearl_core.node pearl_rot_sm89.cubin; do curl -fsSL --retry 3 "$REL/$f" -o $W/$f || fail "fetch $f"; done
gcc -shared -fPIC -o $W/prio_shim.so $W/prio_shim.c -ldl || fail "prio shim"
(curl -fsSL --retry 3 "$MODEL_URL" -o $W/model.gguf > /root/model.log 2>&1; echo $? > /root/model.rc) &
mpid=$!
echo '{"name":"corunq","private":true}' > package.json
npm install --no-audit --no-fund node-llama-cpp@3.22.3 > /root/npm.log 2>&1 || { tail -5 /root/npm.log | emit; fail "npm"; }
say "npm ok in $(( $(date +%s) - t0 ))s"

# Quantus core from QSHA, as CI builds it: the patched RTX 40 kernel, then the addon for this card.
Q=/opt/q
rm -rf $Q && mkdir -p $Q && cd $Q || fail "no $Q"
{ git init -q && git remote add origin https://github.com/super3/llmjob.git && git fetch -q --depth 1 origin "$QSHA" \
  && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1 || { tail -3 /root/git.log | emit; fail "fetch $QSHA"; }
cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
C=$Q/earn/native/quantus/cuda
SMDIR=""
if [ "$cc" = 89 ] || [ "$cc" = 86 ]; then
  python3 $C/sm86/build.py nvcc nvdisasm $Q/sm$cc --arch sm_$cc > /root/sm.log 2>&1 || tail -3 /root/sm.log | emit
  [ -f $Q/sm$cc/qc_sm${cc}_cubin.h ] && SMDIR=$Q/sm$cc
fi
if [ "$cc" = 89 ]; then QV=QC_SM89_DIR; else QV=QC_SM86_DIR; fi
env "$QV=$SMDIR" OUT=$W/quantus_cuda.node CUDA_HOME=$CUDA NODE_INCLUDE=$NODE/include/node bash $C/build.sh "$cc" \
  > /root/qbuild.log 2>&1 || { tail -5 /root/qbuild.log | emit; fail "quantus build"; }
tail -1 /root/qbuild.log | emit
cd $W
wait $mpid
[ "$(cat /root/model.rc)" = 0 ] || { tail -3 /root/model.log | emit; fail "model download"; }
say "setup ok in $(( $(date +%s) - t0 ))s, model $(du -h $W/model.gguf | cut -f1)"

# ---------------------------------------------------------------- runs
export PCORE=$W/pearl_core.node QCORE=$W/quantus_cuda.node PROFILE_JSON=$W/profile.json MODEL=$W/model.gguf
# run TAG MODE [secs warm]: one corunq.mjs run (MODE "two": llm + miner processes), power and clocks over its
# window; prints "R <json>" and "P <json>".
run() {
  local tag=$1 mode=$2 secs=${3:-$SECS} warm=${4:-$WARM} pre=""
  [ -n "${SHIM:-}" ] && pre=$W/prio_shim.so
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu --format=csv,noheader,nounits \
    -lms 500 > smi-$tag.csv &
  local smi=$!
  if [ "$mode" = two ]; then
    rm -f ready-$tag
    READY_FILE=ready-$tag node corunq.mjs miner $secs $warm $tag-miner > out-$tag-miner.txt 2> err-$tag-miner.txt &
    local m=$!
    READY_FILE=ready-$tag LD_PRELOAD=$pre node corunq.mjs llm $secs $warm $tag-llm 2> err-$tag.txt | tail -1 > last-$tag.txt
    wait $m
    tail -1 out-$tag-miner.txt | sed 's/^/R /' | emit
  else
    LD_PRELOAD=$pre node corunq.mjs $mode $secs $warm $tag 2> err-$tag.txt | tail -1 > last-$tag.txt
  fi
  kill $smi 2> /dev/null
  sed 's/^/R /' last-$tag.txt | emit
  [ -s last-$tag.txt ] || { tail -3 err-$tag.txt | emit; return; }
  local a
  a=$(node -e "const j=JSON.parse(require('fs').readFileSync('last-$tag.txt','utf8'));console.log((j.windowStartUtc||'')+' '+(j.windowEndUtc||''))")
  [ "$a" != " " ] && TZ=UTC node power.mjs smi-$tag.csv $a | sed "s/^/P $tag /" | emit
}
rate_of() { node -e "const j=JSON.parse(require('fs').readFileSync('last-$1.txt','utf8'));console.log(j.rate)"; }
G=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | head -1)
SMS=$(case "$G" in *4090*) echo 128 ;; *) echo 0 ;; esac)

if has corun; then
  export MINER=quantus
  run q-miner miner; QSOLO=$(rate_of q-miner)
  run q-llm llm
  run q-two two
  for k in 0 16 32 64; do
    if [ "$k" = 0 ]; then unset QUANTUS_CUDA_GRID; else export QUANTUS_CUDA_GRID=$(( (SMS - k) * QPER )); fi
    run q-free$k both
  done
  unset QUANTUS_CUDA_GRID
  SHIM=1 run q-free0-prio both
  unset SHIM
  say "corun done"
fi

if has switch; then
  # Solo rates for "full rate" (95% of solo).
  export MINER=pearl; run p-miner miner 60 30; PSOLO=$(rate_of p-miner)
  if [ -z "${QSOLO:-}" ]; then export MINER=quantus; run q-miner miner 60 30; QSOLO=$(rate_of q-miner); fi
  for i in 1 2 3; do run cold-llm-$i coldllm 60 0; done
  for i in 1 2 3; do MINER=pearl SOLO=$PSOLO run cold-p-$i coldminer 90 0; done
  for i in 1 2 3; do MINER=quantus SOLO=$QSOLO run cold-q-$i coldminer 60 0; done
  MINER=pearl SOLO=$PSOLO REPS=4 run pause-p pause 20 20
  MINER=quantus SOLO=$QSOLO REPS=4 run pause-q pause 20 20
  # Time to first token while the miner keeps running: short answers, so a run holds several.
  for m in pearl quantus; do
    MINER=$m NTOK=8 run ttft-$m-two two 60 20
    MINER=$m NTOK=8 run ttft-$m-one both 60 20
  done
  say "switch done"
fi
idle "all done"
