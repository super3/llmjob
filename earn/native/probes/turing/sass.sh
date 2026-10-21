#!/bin/bash
# The hand-scheduled inner loop of Turing's B-direct fold, against ptxas's and
# against a pure-IMMA ceiling, per clock, on one card (probes/turing/sass/).
#
#   SRC=<llmjob checkout> OUT=<scratch dir> bash earn/native/probes/turing/sass.sh
#
# Expects an idle sm_75 GPU, nvidia-smi and the CUDA 12.8 toolkit of
# nvidia/cuda:12.8.1-devel-ubuntu22.04; installs python3 and git if missing.
# Builds everything on the box from the checkout: nvcc builds kernels.cubin
# and the harness, CuAssembler (fetched at a pinned commit) disassembles the
# cubin, hand.py writes the hand-scheduled kernels into it and checks them for
# hazards, CuAssembler assembles the result, and the assembled code is
# disassembled again and compared with what hand.py wrote. Then the harness
# checks the results and times every kernel.
#
# Every result is one line starting with [sass]. The last is the verdict.
# About 5 minutes, most of it apt and pip.
set -u
SRC=${SRC:?set SRC to the llmjob checkout}
OUT=${OUT:?set OUT to a scratch directory}
D=$SRC/earn/native/probes/turing/sass
CUASM_REPO=https://github.com/cloudcores/CuAssembler.git
CUASM_COMMIT=96a9f72baf00f40b9b299653fcef8d3e2b4a3d49
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p $OUT
say() { echo "[sass] $*" | cut -c1-490; }
fail() { say "FAIL $*"; exit 1; }

say "card $(nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader | head -1)"
say "apps on the GPU: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
command -v nvcc > /dev/null || fail "no nvcc"
say "$(nvcc --version | grep release)"

# Tools: python3 with venv, git. NVIDIA's apt repo is left out of the lists:
# when it is slow, apt waits on it for nothing we need from it.
if ! command -v python3 > /dev/null || ! python3 -c 'import ensurepip' 2> /dev/null || ! command -v git > /dev/null; then
  mkdir -p /tmp/apt-nocuda
  for f in /etc/apt/sources.list.d/*; do [ -f "$f" ] && ! grep -q developer.download.nvidia.com "$f" && cp "$f" /tmp/apt-nocuda/; done
  { apt-get update -o Dir::Etc::SourceParts=/tmp/apt-nocuda &&
    apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates; } > $OUT/apt.log 2>&1 ||
    fail "apt: $(tail -1 $OUT/apt.log)"
fi
if ! command -v nvdisasm > /dev/null || ! command -v cuobjdump > /dev/null; then
  { apt-get update && apt-get install -y --no-install-recommends cuda-nvdisasm-12-8 cuda-cuobjdump-12-8; } >> $OUT/apt.log 2>&1 ||
    fail "no nvdisasm/cuobjdump: $(tail -1 $OUT/apt.log)"
fi
python3 -m venv $OUT/venv > $OUT/pip.log 2>&1 &&
  $OUT/venv/bin/pip install -q pyelftools==0.33 sympy==1.14.0 >> $OUT/pip.log 2>&1 ||
  fail "pip: $(tail -1 $OUT/pip.log)"
PY=$OUT/venv/bin/python
rm -rf $OUT/CuAssembler && mkdir -p $OUT/CuAssembler &&
  git -C $OUT/CuAssembler init -q && git -C $OUT/CuAssembler remote add origin $CUASM_REPO &&
  git -C $OUT/CuAssembler fetch -q --depth 1 origin $CUASM_COMMIT && git -C $OUT/CuAssembler checkout -q FETCH_HEAD ||
  fail "could not fetch CuAssembler $CUASM_COMMIT"
say "CuAssembler at $(git -C $OUT/CuAssembler rev-parse HEAD)"
ND=$(command -v nvdisasm)

# Build.
nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -Xptxas -v -cubin -o $OUT/kernels.cubin $D/kernels.cu > $OUT/ptxas.txt 2>&1 ||
  fail "nvcc kernels: $(grep -m1 error $OUT/ptxas.txt)"
say "ptxas: $(grep -E 'Used [0-9]+ registers' $OUT/ptxas.txt | sed -E 's/.*Used ([0-9]+) registers.*/\1/' | tr '\n' ' ')registers (ceiling, slots, ptxas), $(grep -c ' 0 bytes spill stores' $OUT/ptxas.txt) of $(grep -c 'spill stores' $OUT/ptxas.txt) with no spills"
CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
nvcc -O3 -std=c++17 -Wno-deprecated-gpu-targets -o $OUT/harness $D/harness.cu -L$CUDA_HOME/lib64/stubs -L$CUDA_HOME/lib/stubs -lcuda > $OUT/harness.log 2>&1 ||
  fail "nvcc harness: $(grep -m1 error $OUT/harness.log)"
cuobjdump -sass $OUT/kernels.cubin > $OUT/learn.sass || fail "cuobjdump"
$PY -I $D/cuasm_drv.py $OUT/CuAssembler $ND d $OUT/kernels.cubin $OUT/kernels.cuasm > $OUT/cuasm-d.log 2>&1 ||
  fail "CuAssembler disassembly: $(tail -1 $OUT/cuasm-d.log)"
# A round trip of the untouched cubin first: CuAssembler must give back what nvcc made.
$PY -I $D/cuasm_drv.py $OUT/CuAssembler $ND a $OUT/kernels.cuasm $OUT/kernels-rt.cubin $OUT/learn.sass > $OUT/cuasm-rt.log 2>&1 ||
  fail "CuAssembler round trip: $(tail -1 $OUT/cuasm-rt.log)"
cmp -s $OUT/kernels.cubin $OUT/kernels-rt.cubin && say "round trip of the nvcc cubin: byte-identical" ||
  say "round trip of the nvcc cubin: differs ($(cmp $OUT/kernels.cubin $OUT/kernels-rt.cubin | head -1))"
python3 -I $D/hand.py $OUT/kernels.cuasm $OUT/hand.cuasm 2> $OUT/hand.log || fail "hand.py: $(head -3 $OUT/hand.log | tr '\n' ' ')"
say "hand.py: hand1 hand2 hand3 hceil written, hazard check passed"
$PY -I $D/cuasm_drv.py $OUT/CuAssembler $ND a $OUT/hand.cuasm $OUT/hand.cubin $OUT/learn.sass > $OUT/cuasm-a.log 2>&1 ||
  fail "CuAssembler assembly: $(grep -m1 -i 'error\|exception' $OUT/cuasm-a.log)"
$PY -I $D/cuasm_drv.py $OUT/CuAssembler $ND d $OUT/hand.cubin $OUT/hand-rt.cuasm > $OUT/cuasm-d2.log 2>&1 ||
  fail "disassembling hand.cubin: $(tail -1 $OUT/cuasm-d2.log)"
python3 -I $D/hand.py --verify $OUT/hand-rt.cuasm || fail "assembled code differs from hand.py's"

# Run: correctness, then 7 interleaved rounds of every kernel.
say "before: $(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader)"
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > $OUT/smi.log 2>&1 &
SMI=$!
(cd $OUT && ./harness kernels.cubin hand.cubin 600 7) 2>&1 | cut -c1-490
rc=${PIPESTATUS[0]}
kill $SMI 2> /dev/null
say "during: $(sort -t, -k2 -n $OUT/smi.log | awk -F, 'NF>=4{n++; c[n]=$0} END{if(n) print n " samples, median " c[int((n+1)/2)]}')"
[ $rc = 0 ] || fail "harness exit $rc"
say "done"
