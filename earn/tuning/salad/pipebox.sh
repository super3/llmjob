#!/bin/bash
# Box script for the tensor-pipe probe on other cards (branch goal/nv-probe-cards, 11 November): on one Salad PC,
# earn/native/probes/pipeprobe2.cu built for the card's architecture and run as the streams of the RTX 4090 table
# (probes/README.md, "What fills the tensor pipe"), with the card's clock and watts beside the long runs. Before
# that, on the same PC: OneZeroMiner and our release on the stand-in pool (bench.sh), and our release's core under
# hashrate.js. A measuring tool for this branch only: the streams compute nothing real.
#
# The container command fetches this file at BASE_SHA and runs it. Every line is "[nvtest] HH:MM:SS [probe] ..."
# (bench.sh's are "[nvtest] [bench] HH:MM:SS ..."), so the test-box tools read them as an nvtest box's.
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   CUDA_VER      the toolkit: 12.8 (default; 13.3 on a compute 12.x card whose driver is 580 or newer)
#   CUDA_VER2     a second toolkit, for a short pass of the main streams built with it (default: none)
#   PP_BENCH      1: run bench.sh first (ORDER default ozm,ours; SECS, WARM, OURS_SECS default 180, 60, 180;
#                 VPOOL 210; VER the release, default v0.6.15)
#   PP_HR_SECS    seconds of hashrate.js on the release's core after a 20 s warm-up (default 180; 0: skip)
#   PP_CHUNKS     chunks a launch (default 100000, about 0.3 s)
#   PP_SHORT      seconds a short run (default 1)
#   PP_LONG       seconds a long run (default 30; the card is read every second and the first 8 s are left out)
#   PP_ONLY_STAT  1: build only, and print what stands between each build's IMMA (nothing runs)
set -u
psay() { echo "[nvtest] $(date -u +%T) [probe] $*"; }
[ -n "${BASE_SHA:-}" ] || { psay "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in bench.sh nvtest.sh; do
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { psay "FAIL fetch $f"; exec sleep infinity; }
done
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }
psay "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,memory.total)"
psay "card idle $(q clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
CC=$(q compute_cap | tr -d ' .')
DRV=$(q driver_version | cut -d. -f1 | tr -d ' ')
case "$CC" in ''|*[!0-9]*) psay "FAIL no compute capability: $CC"; exec sleep infinity ;; esac
if [ -z "${CUDA_VER:-}" ]; then
  CUDA_VER=12.8
  [ "$CC" -ge 120 ] && [ "${DRV:-0}" -ge 580 ] 2> /dev/null && CUDA_VER=13.3
fi
export CUDA_VER
: > /root/bench.log
export DEBIAN_FRONTEND=noninteractive
# Every package bench.sh and nvtest.sh ask for, once, before either starts (two apt-get runs fight over dpkg's lock).
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
    python3 jq procps; } > /root/apt0.log 2>&1 || psay "WARN apt-get: $(tail -1 /root/apt0.log)"
if [ -n "${PP_BENCH:-}" ]; then
  # W: a payout address in form only, which is all our CLI checks. The stand-in pool pays nobody.
  ( W=${W:-prl1pgate00000000000000000000000000000000000000000000000000000} ORDER=${ORDER:-ozm,ours} VPOOL=${VPOOL:-210} \
      SECS=${SECS:-180} WARM=${WARM:-60} OURS_SECS=${OURS_SECS:-180} VER=${VER:-v0.6.15} bash /root/bench.sh 2>&1 \
      | tee -a /root/bench.log | sed -u 's/^\[bench\]/[nvtest] [bench]/' ) &
else
  echo "all done" > /root/bench.log
fi
# nvtest.sh's functions (setup, cuda, fetch), without its run. /opt/llmjob is bench.sh's own checkout.
export NVTEST_ROOT=/opt/pp-src
NVTEST_STAGE=lib . /root/nvtest.sh
[ -n "$NVCC_VER" ] || { psay "FAIL CUDA_VER must be 12.8 or 13.3, not $CUDA_VER"; exec sleep infinity; }
setup
cuda
fetch
P=/root/pp
mkdir -p $P
SRC=$R/earn/native/probes/pipeprobe2.cu
ARCH=sm_$CC
# disasm HOME VER: nvdisasm into a toolkit, which cuobjdump -sass calls and nvtest.sh's archives leave out. Its
# path and SHA-256 come from the same release's manifest on NVIDIA's server.
disasm() {
  local home=$1 rel man path sha f=/root/dl-nvdisasm.tar.xz
  [ -x "$home/bin/nvdisasm" ] && return 0
  case "$2" in 12.8) rel=12.8.1 ;; 13.3) rel=13.3.1 ;; *) return 1 ;; esac
  man=$(curl -fsSL --retry 3 "$RD/redistrib_$rel.json") || return 1
  path=$(echo "$man" | jq -r '.cuda_nvdisasm."linux-x86_64".relative_path')
  sha=$(echo "$man" | jq -r '.cuda_nvdisasm."linux-x86_64".sha256')
  curl -fsSL --retry 3 "$RD/$path" -o "$f" && echo "$sha  $f" | sha256sum -c --quiet - > /dev/null 2>&1 \
    && tar -xJf "$f" -C "$home" --strip-components=1
  rm -f "$f"
  [ -x "$home/bin/nvdisasm" ]
}
disasm "$CUDA" "$CUDA_VER" || psay "WARN no nvdisasm for $CUDA_VER: builds will show no instruction counts"

# sass.py FILE [trace]: what stands between the first IMMA of a build's kernel and the branch back after its
# last. With "trace": that stretch in order, run lengths by kind (I IMMA, X LOP3, M ldmatrix, L other loads,
# S stores and shared atomics, A other straight-line steps, control instructions by name), 360 characters a line.
cat > $P/sass.py << 'EOF'
import collections, re, sys
ops = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if m:
        ops.append(m.group(2).split(".")[0])
im = [i for i, o in enumerate(ops) if o == "IMMA"]
if not im:
    print("no IMMA in %d instructions" % len(ops)); sys.exit()
end = next((i for i in range(im[-1], len(ops)) if ops[i] == "BRA"), im[-1])   # the loop's back branch
body = ops[im[0]:end + 1]
other = [o for o in body if o != "IMMA"]
CTL = ("BRA", "BRX", "JMP", "WARPSYNC", "YIELD", "BSSY", "BSYNC", "BAR", "NANOSLEEP", "CALL", "RET", "BREAK",
       "BMOV", "MEMBAR")
ctl = [o for o in other if o in CTL]
if len(sys.argv) > 2:
    kind = lambda o: ("I" if o == "IMMA" else o if o in CTL else "X" if o in ("LOP3", "PLOP3", "LOP") else
                      "M" if o == "LDSM" else "L" if o in ("LDS", "LD", "LDG", "LDL") else
                      "S" if o in ("STS", "ST", "STG", "STL", "ATOMS", "ATOM", "ATOMG") else "A")
    toks, last, n = [], None, 0
    for o in body:
        k = kind(o)
        if k == last:
            n += 1
        else:
            if last:
                toks.append(last + (str(n) if n > 1 or last in "IXMLSA" else ""))
            last, n = k, 1
    toks.append(last + str(n))
    line = ""
    for t in toks:
        if len(line) + len(t) + 1 > 360:
            print(line); line = ""
        line += (" " if line else "") + t
    print(line)
    sys.exit()
run = best = 0
gaps = []
for o in body:
    if o == "IMMA":
        gaps.append(run); best = max(best, run); run = 0
    else:
        run += 1
top = ",".join("%s:%d" % kv for kv in collections.Counter(other).most_common(7))
print("imma=%d other=%d per_imma=%.3f longest_run=%d runs_over_8=%d ctl=%d[%s] top=%s" % (
    len(im), len(other), len(other) / len(im), best, sum(1 for g in gaps if g > 8), len(ctl),
    ",".join(sorted(set(ctl))), top))
EOF
# smi.py FILE T0 SKIP: the card's samples from SKIP seconds after T0: means, the clock's range, the limit.
cat > $P/smi.py << 'EOF'
import sys
f, t0, skip = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows, why = [], set()
for line in open(f):
    p = [x.strip() for x in line.split(",")]
    try:
        t = float(p[0]); v = [float(x) for x in p[1:6]]
    except (ValueError, IndexError):
        continue
    if t - t0 >= skip:
        rows.append(v)
        if len(p) > 6:
            why.add(p[6])
if not rows:
    print("no card samples"); sys.exit()
n = len(rows)
m = [sum(r[i] for r in rows) / n for i in range(5)]
print("%.0f MHz (%.0f-%.0f) %.1f W (max %.1f) of %.0f W, %.0f C, util %.0f%%, reasons %s, %d samples" % (
    m[0], min(r[0] for r in rows), max(r[0] for r in rows), m[1], max(r[1] for r in rows), m[4], m[2], m[3],
    "/".join(sorted(why)) or "-", n))
EOF
SQ=clocks.sm,power.draw,temperature.gpu,utilization.gpu,power.limit
for r in clocks_event_reasons.active clocks_throttle_reasons.active; do
  if nvidia-smi -i 0 --query-gpu=$r --format=csv,noheader > /dev/null 2>&1; then SQ=$SQ,$r; break; fi
done

# The streams: name|flags. The first block is the 4090 table's (lists 5 and 6 of the study); "r" after a name is
# the same stream with hashed operand bytes (FILLRND), which changes the watts and not the code between the IMMA.
F="-DRD=2 -DTRW=0 -DDRAW=4 -DDPOS=1 -DDREAL=1"
STREAMS="pure|-DKLOAD=0
base|
rd1|-DRD=1
rd1a|-DRD=1 -DTRW=0
rd2|-DRD=2
rd2a|-DRD=2 -DTRW=0
rd3|-DRD=3
d4e|-DDRAW=4 -DDPOS=0
d4m|-DDRAW=4 -DDPOS=1
d4h|-DDRAW=4 -DDPOS=2
d4s|-DDRAW=4 -DDPOS=3
d4mg|-DDRAW=4 -DDPOS=1 -DDGRP=1
d4r|-DDRAW=4 -DDPOS=1 -DDREAL=1
fullns|$F
full|$F -DDSYNC=1
full3|-DRD=2 -DTRW=0 -DDRAW=3 -DDPOS=1 -DDREAL=1 -DDSYNC=1
fulle|-DRD=2 -DTRW=0 -DDRAW=4 -DDPOS=0 -DDREAL=1 -DDSYNC=1
baser|-DFILLRND=1
rd1r|-DRD=1 -DFILLRND=1
rd2ar|-DRD=2 -DTRW=0 -DFILLRND=1
d4mr|-DDRAW=4 -DDPOS=1 -DFILLRND=1
w4|-DKLOAD=0 -DWARPS=4
w8a32|-DKLOAD=0 -DACC=32
w8a24|-DKLOAD=0 -DACC=24
w12|-DKLOAD=0 -DWARPS=12 -DACC=32
w16|-DKLOAD=0 -DWARPS=16 -DACC=24
w4k|-DWARPS=4
w12k|-DWARPS=12 -DACC=32
w16k|-DWARPS=16 -DACC=24
m01|-DKLOAD=0 -DWMASK=0x03u
m02|-DKLOAD=0 -DWMASK=0x05u
m03|-DKLOAD=0 -DWMASK=0x09u
m04|-DKLOAD=0 -DWMASK=0x11u
m05|-DKLOAD=0 -DWMASK=0x21u
m06|-DKLOAD=0 -DWMASK=0x41u
m07|-DKLOAD=0 -DWMASK=0x81u
m0|-DKLOAD=0 -DWMASK=0x01u"

# build DIR NVCC-HOME NAME FLAGS: the program, with ptxas's registers and spill and what stands between its IMMA
# (read from the program's own kernel image; from a cubin of the same flags if cuobjdump can't read the program).
build1() {
  local d=$1 home=$2 name=$3 flags=$4 used spill
  "$home/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -cudart static -Xptxas -v $flags "$SRC" -o "$d/$name" > "$d/$name.log" 2>&1 \
    || "$home/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -Xptxas -v $flags "$SRC" -o "$d/$name" > "$d/$name.log" 2>&1
  PATH="$home/bin:$PATH" "$home/bin/cuobjdump" -sass "$d/$name" > "$d/$name.sass" 2> /dev/null
  if ! grep -q IMMA "$d/$name.sass" 2> /dev/null && [ -x "$home/bin/nvdisasm" ]; then
    "$home/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -cubin $flags "$SRC" -o "$d/$name.cubin" > "$d/$name.cubin.log" 2>&1
    "$home/bin/nvdisasm" -c "$d/$name.cubin" > "$d/$name.sass" 2> /dev/null
  fi
  used=$(grep -oE 'Used [0-9]+ registers' "$d/$name.log" | head -1 | grep -oE '[0-9]+')
  spill=$(grep -oE '[0-9]+ bytes spill stores' "$d/$name.log" | head -1 | grep -oE '^[0-9]+')
  if [ -x "$d/$name" ]; then
    echo "built $name regs=${used:-?} spill=${spill:-?} $(python3 $P/sass.py "$d/$name.sass" 2>&1 | head -1)"
  else
    echo "NOT built $name: $(grep -m1 -iE 'error' "$d/$name.log" | cut -c1-240)"
  fi
}
# buildall DIR NVCC-HOME TAG [NAMES]: every stream (or the named ones), three at a time. It waits for its own
# builds by process id: a bare wait would also wait for bench.sh, which never exits. Then the main streams' loops
# as traces.
buildall() {
  local d=$1 home=$2 tag=$3 only=${4:-} name flags t0 pids=() s l
  t0=$(date +%s)
  mkdir -p "$d"
  while IFS='|' read -r name flags; do
    [ -n "$name" ] || continue
    if [ -n "$only" ]; then case " $only " in *" $name "*) ;; *) continue ;; esac; fi
    build1 "$d" "$home" "$name" "$flags" > "$d/$name.built" 2>&1 &
    pids+=($!)
    if [ ${#pids[@]} -ge 3 ]; then wait "${pids[@]}"; pids=(); fi
  done <<< "$STREAMS"
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  for f in "$d"/*.built; do psay "$tag $(cat "$f")"; done
  psay "$tag builds done in $(( $(date +%s) - t0 )) s with $("$home/bin/nvcc" --version | grep -oE 'V[0-9.]+' | tail -1) for $ARCH"
  for s in ${PP_TRACES:-base rd1 rd2a d4m d4r full3 full}; do
    grep -q IMMA "$d/$s.sass" 2> /dev/null || continue
    python3 $P/sass.py "$d/$s.sass" trace 2>&1 | while IFS= read -r l; do psay "$tag trace $s $l"; done
  done
}
# toolkit2: CUDA_VER2's compiler in /opt/cuda2, with its nvdisasm. True if it is there.
toolkit2() {
  timeout 600 bash -c 'export CUDA_VER=$0 NVTEST_CUDA=/opt/cuda2 NVTEST_STAGE=lib; . /root/nvtest.sh; fail() { echo "FAIL $*"; exit 1; }; cuda' "$CUDA_VER2" 2>&1 \
    | sed -u 's/^\[nvtest\] [0-9:]* //' | while IFS= read -r l; do psay "toolkit2 $l"; done
  [ -x /opt/cuda2/bin/nvcc ] || return 1
  disasm /opt/cuda2 "$CUDA_VER2" || psay "WARN no nvdisasm for $CUDA_VER2"
  return 0
}
L2="pure base rd1 rd2a d4m d4r fullns full full3"
buildall $P/b "$CUDA" "nvcc$CUDA_VER"
# PP_ONLY_STAT=1: the builds and what stands between their IMMA, with both toolkits, and nothing run.
if [ -n "${PP_ONLY_STAT:-}" ]; then
  if [ -n "${CUDA_VER2:-}" ] && [ "$CUDA_VER2" != "$CUDA_VER" ] && toolkit2; then
    buildall $P/b2 /opt/cuda2 "nvcc$CUDA_VER2" "$L2"
  fi
  psay "all done"
  say "all done"
  idle "until deleted"
fi

# The miners first (bench.sh, started above), then the release's core under hashrate.js.
n=0
until grep -q 'all done' /root/bench.log; do
  n=$((n + 1)); [ $n -gt 240 ] && { psay "bench.sh still running after 40 min; going on"; break; }
  sleep 10
done
pkill -f /root/bench.sh 2> /dev/null
pkill -f 'peak -o' 2> /dev/null; pkill -f SRBMiner-MULTI 2> /dev/null; pkill -f onezerominer 2> /dev/null; pkill -f 'cli -a' 2> /dev/null
sleep 5
if [ "${PP_HR_SECS:-180}" != 0 ]; then
  VER=${VER:-v0.6.15}
  core=pearl_core.node
  [ "$CC" -ge 120 ] && [ "${DRV:-0}" -ge 580 ] 2> /dev/null && core=pearl_core_cu13.node
  mkdir -p /root/rel /opt/node-pp
  if curl -fsSL --retry 3 -o "/root/rel/$core" "https://github.com/super3/llmjob/releases/download/$VER/$core" \
     && curl -fsSL --retry 3 https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.gz \
        | tar xz -C /opt/node-pp --strip-components=1; then
    psay "hashrate start $VER $core ${PP_HR_SECS:-180} s after 20 s"
    j=$(cd "$R/earn" && timeout -k 10 $(( ${PP_HR_SECS:-180} + 150 )) /opt/node-pp/bin/node native/probes/hashrate.js \
          "/root/rel/$core" "${PP_HR_SECS:-180}" 20 2>&1 | tail -3 | tr '\n' ' ')
    psay "hashrate $VER $core $(echo "$j" | cut -c1-330)"
  else
    psay "hashrate FAIL: download of $core or node"
  fi
fi
sleep 10
psay "card before the probe $(q clocks.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used)"

CH=${PP_CHUNKS:-100000}
# short DIR TAG NAME [v]: one short run, pipe use only.
short() {
  local l
  timeout 120 "$1/$3" "$3" "$CH" "${PP_SHORT:-1}" ${4:-} 2>&1 | while IFS= read -r l; do psay "$2 short $l"; done
}
# long DIR TAG NAME: a long run with the card read every second.
long() {
  local f=$P/smi.$3 sp t0 out
  : > "$f"
  ( while :; do
      printf '%s,%s\n' "$(date +%s.%N)" "$(nvidia-smi -i 0 --query-gpu=$SQ --format=csv,noheader,nounits 2> /dev/null | head -1)" >> "$f"
      sleep 0.8
    done ) &
  sp=$!
  t0=$(date +%s.%N)
  out=$(timeout 240 "$1/$3" "$3" "$CH" "${PP_LONG:-30}" 2>&1 | tail -1)
  kill $sp 2> /dev/null; wait $sp 2> /dev/null
  psay "$2 long $out | $(python3 $P/smi.py "$f" "$t0" 8)"
}
T=nvcc$CUDA_VER
# 1. What a full pipe is here: IMMA only with one to four warps a scheduler, the warp-to-pipe map, and the same
#    with the fragment loads.
for s in m0 m01 m02 m03 m04 m05 m06 m07 w4 pure w8a32 w8a24 w12 w16 w4k base w12k w16k; do
  [ -x $P/b/$s ] && short $P/b $T $s v
done
# 2. The 4090 table's streams, pipe use only.
for s in pure base rd1 rd1a rd2 rd2a rd3 d4e d4m d4h d4s d4mg d4r fullns full full3 fulle; do
  [ -x $P/b/$s ] && short $P/b $T $s
done
# 3. The long runs, with clock and watts: the offset-slice bytes first, then hashed bytes, then base again.
for s in ${PP_LONGS:-pure base rd1 rd2a d4m d4r full full3 baser rd1r rd2ar d4mr base}; do
  [ -x $P/b/$s ] && long $P/b $T $s
done
psay "card after the probe $(q clocks.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used)"
# 4. A second toolkit's builds of the main streams, short runs and two long ones.
if [ -n "${CUDA_VER2:-}" ] && [ "$CUDA_VER2" != "$CUDA_VER" ]; then
  if toolkit2; then
    T2=nvcc$CUDA_VER2
    buildall $P/b2 /opt/cuda2 $T2 "$L2"
    for s in $L2; do [ -x $P/b2/$s ] && short $P/b2 $T2 $s; done
    for s in ${PP_LONGS2:-base full3}; do [ -x $P/b2/$s ] && long $P/b2 $T2 $s; done
  else
    psay "toolkit2 $CUDA_VER2 not installed"
  fi
fi
psay "all done"
say "all done"
idle "until deleted"
