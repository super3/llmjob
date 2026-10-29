#!/bin/bash
# Box script for racing our own NVIDIA builds on a Salad PC, RTX 30 (sm_86) first of all. amdtest.py's
# container command for an nvtest- group fetches this file at SCRIPT_SHA (default BASE_SHA) and runs it. It
# installs CUDA 12.8.1's compiler (nvcc V12.8.93, the one CI builds with) from NVIDIA's redistributable archives,
# each checked against the SHA-256 pinned below, fetches the repo at BASE_SHA into /opt/llmjob, applies
# PATCH_GZ_B64 if set, and runs earn/native/probes/turing/compilers.sh for this card: the builds in
# VARIANTS_SPEC, raced in the stages in STAGES. Then pool.sh, if WALLET and POOL_SECS are set. Then it sleeps,
# so the logs can be read and `amdtest.py set-env` can start the next run.
#
# Not the CUDA devel image: on Salad the 12.8.1 image sat in "preparing" for 45+ minutes (README.md, "The
# container"), and the devel one is larger still. The fleet's base image is ready at once, and the four
# archives are 82 MB.
#
# Every line is "[nvtest] HH:MM:SS <text>", with at most 400 characters of text: longer ones are cut into
# pieces, "+" starting each continuation, so Salad's log service keeps each one. compilers.sh's and pool.sh's
# lines keep their own "[compilers]" and "[pool]" starts after the time. Nothing here prints environment
# variables.
#
# Environment (amdtest.py sets it):
#   BASE_SHA        commit of super3/llmjob to test (required; the full 40 characters)
#   PATCH_GZ_B64    optional: `git diff --binary` against BASE_SHA, gzipped and base64'd; applied with git apply
#   VARIANTS_SPEC   the builds, "name|flags;name|flags" (default "base|", the default core); name one "base"
#   STAGES          compilers.sh's stages (default "setup build bench hashrate verify complete")
#   ROUNDS, BENCH_SECS, HR_ROUNDS, HR_SECS, HR_BEST, VERIFY_SECS, COMPLETE_SHAPES, COMPLETE_SECS, ONLY
#                   passed to compilers.sh as they are (see its header)
#   GC, FOLD        default: this card's compute capability, and the tall fold (pearl_tile_fold_tall)
#   WALLET, POOL, POOL_SECS   pool.sh after the race, when WALLET and POOL_SECS are both set
set -u
R=${NVTEST_ROOT:-/opt/llmjob}
OUT=${NVTEST_OUT:-/root/out}
CUDA=${NVTEST_CUDA:-/opt/cuda}
RD=https://developer.download.nvidia.com/compute/cuda/redist
# From redistrib_12.8.1.json: path|sha256.
CUDA_PARTS="cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz|dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f"
export DEBIAN_FRONTEND=noninteractive

say() { echo "[nvtest] $(date -u +%T) $*"; }
# emit: every non-empty stdin line as a say line, in pieces of at most 400 characters. Read straight from stdin
# (no tr in front, which holds a pipe's lines until 4 KB or the end), so each line goes out as it comes.
emit() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//$'\r'/ }
    [ -z "${line//[[:space:]]/}" ] && continue
    while [ ${#line} -gt 400 ]; do say "${line:0:400}"; line="+${line:400}"; done
    say "$line"
  done
}
idle() { say "idle $*"; sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }

setup() {
  local pkgs="" t0; t0=$(date +%s)
  for t in git curl xz g++ make python3; do command -v $t > /dev/null || pkgs=1; done
  if [ -n "$pkgs" ]; then
    { apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
        python3; } > /root/apt.log 2>&1 || { tail -5 /root/apt.log | emit; fail "apt-get install"; }
  fi
  say "setup ok in $(( $(date +%s) - t0 ))s"
}

# CUDA 12.8.1's compiler, runtime headers and static runtime in $CUDA, laid out as a toolkit: nvcc.profile
# looks in targets/x86_64-linux, and binding.gyp in lib64.
cuda() {
  local t0 part rel sha f; t0=$(date +%s)
  if [ "$("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)" != V12.8.93 ]; then
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
    ln -sfn ../../lib "$CUDA/targets/x86_64-linux/lib64"
  fi
  say "cuda $("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1) in $CUDA, $(( $(date +%s) - t0 ))s"
}

fetch() {
  rm -rf "$R" && mkdir -p "$R" && cd "$R" || fail "no $R"
  if ! { git init -q && git remote add origin https://github.com/super3/llmjob.git \
         && git fetch -q --depth 1 origin "$BASE_SHA" && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1; then
    tail -3 /root/git.log | emit
    fail "fetch $BASE_SHA"
  fi
  say "fetch ok $(git rev-parse --short=12 HEAD)"
  if [ -n "${PATCH_GZ_B64:-}" ]; then
    printf '%s' "$PATCH_GZ_B64" | base64 -d | gunzip > /root/patch.diff || fail "patch: not gzip+base64"
    say "patch $(wc -c < /root/patch.diff) bytes, sha256 $(sha256sum /root/patch.diff | cut -c1-12)"
    git apply --whitespace=nowarn /root/patch.diff > /root/apply.log 2>&1 \
      || { emit < /root/apply.log; fail "patch does not apply to $BASE_SHA"; }
  fi
}

# This card's -gencode flags, from its compute capability (8.6 -> sm_86).
card_gc() {
  local cc
  cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -1 | tr -d ' .')
  [ -n "$cc" ] && echo "-gencode arch=compute_$cc,code=sm_$cc"
}

# NVTEST_STAGE=lib: sourced for its functions only (the tests do this), so stop here.
[ "${NVTEST_STAGE:-}" = lib ] && return 0

patch=${PATCH_GZ_B64:-}
say "boot sha ${BASE_SHA:-none}, builds ${VARIANTS_SPEC:-base|}, stages ${STAGES:-default}, patch ${#patch} chars"
[ -n "${BASE_SHA:-}" ] || idle "no BASE_SHA"
say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,memory.total --format=csv,noheader 2>&1 | head -1)"
setup
cuda
fetch
export GC=${GC:-$(card_gc)} FOLD=${FOLD:-pearl_tile_fold_tall}
[ -n "$GC" ] || fail "no compute capability from nvidia-smi; set GC"
export VARIANTS_SPEC=${VARIANTS_SPEC:-base|} STAGES=${STAGES:-setup build bench hashrate verify complete}
# run TAG SCRIPT: the script's TAG lines as they come; its whole output in /root/TAG.log, and if it fails, the
# last other lines (a shell error never has the tag).
run() {
  local tag=$1 rc t0; t0=$(date +%s)
  shift
  "$@" 2>&1 | tee "/root/$tag.log" | grep --line-buffered -a "^\[$tag\]" | emit
  rc=${PIPESTATUS[0]}
  say "$tag done rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"
  [ "$rc" = 0 ] || grep -av "^\[$tag\]" "/root/$tag.log" | tail -n 8 | emit
}
say "compilers start: $GC, fold $FOLD"
run compilers env SRC="$R" OUT="$OUT" CUDA_HOME="$CUDA" bash "$R/earn/native/probes/turing/compilers.sh"
if [ -n "${WALLET:-}" ] && [ -n "${POOL_SECS:-}" ]; then
  run pool env SECS="$POOL_SECS" SRC="$R" OUT="$OUT/pool" CUDA_HOME="$CUDA" bash "$R/earn/native/probes/turing/pool.sh"
fi
say "all done"
idle "until the next set-env"
