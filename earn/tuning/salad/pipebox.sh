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
#                 VPOOL 210; VER the release, default v0.6.15). TAP=1 passes through to bench.sh (share relay).
#   PP_ONLY_BENCH 1: with PP_BENCH, the miners only: no builds, no probe
#   PP_HR_SECS    seconds of hashrate.js on the release's core after a 20 s warm-up (default 180; 0: skip)
#   PP_CHUNKS     chunks a launch (default 100000, about 0.3 s)
#   PP_SHORT      seconds a short run (default 1)
#   PP_LONG       seconds a long run (default 30; the card is read every second and the first 8 s are left out)
#   PP_ONLY_STAT  1: build only, and print what stands between each build's IMMA (nothing runs)
#   PP_SKIP_SHORT 1: no short runs but PP_SHORTS's
#   PP_SHORTS     more streams to run short (default: one to three draw units, the readout with 4 and 16 chains)
#   PP_L2, PP_LONGS2   the second toolkit's streams, and those of its long runs
#   PP_LONGS      the streams of the long runs (default: the main ones, then hashed bytes, then base again)
#   PP_SET        a named pass of the sm_120 follow-up (b1 ...): its streams only are built and run short, with
#                 each warp's cycles; no miners, no hashrate.js, no long runs unless PP_LONGS names some
#   PP_BUILD      the streams to build (default: all)
#   PP_OPS        streams whose whole kernel is printed as run lengths by instruction name
#   PP_V          1: PP_SHORTS runs print each warp's own cycles an IMMA
#   PP_PERMS      NAME:TAG:RULES ...: after the short runs, each named stream again with its loop's steps placed
#                 by earn/tuning/pipeprobe/perm.py's RULES (the kernel is launched from the reordered cubin)
#   PP_PTRACE     NAME.TAG ...: the reordered loops to print as traces
#   PP_PLONGS     NAME.TAG ...: reordered kernels to run long, after PP_LONGS
#   PP_DUMPS      NAME:A:B ...: the listing of a stream's loop from its instruction A to B, as the compiler made it
set -u
# Salad's log service hands back at most 100 lines for its smallest window (about 2 s), so a burst of lines loses
# some: psay waits a moment after each.
psay() { echo "[nvtest] $(date -u +%T) [probe] $*"; sleep 0.03; }
[ -n "${BASE_SHA:-}" ] || { psay "no BASE_SHA; idle"; exec sleep infinity; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
for f in bench.sh nvtest.sh sharetap.py; do   # sharetap.py: bench.sh's share relay, used when TAP=1
  curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { psay "FAIL fetch $f"; exec sleep infinity; }
done
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }
psay "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,memory.total)"
psay "card idle $(q clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
psay "card enforced limit $(q enforced.power.limit), instant draw $(q power.draw.instant), average $(q power.draw.average)"
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
# PP_ONLY_BENCH=1: the miners alone, run normally, and nothing built or probed. With TAP=1 they mine through the
# share relay and bench.sh ends each run with that miner's shares decoded (m, n, k, rank, fill, the tile's rows
# and columns); its sample lines carry the card's watts, clock, memory-controller use and memory in use.
if [ -n "${PP_ONLY_BENCH:-}" ]; then
  n=0
  until grep -q 'all done' /root/bench.log; do n=$((n + 1)); [ $n -gt 360 ] && break; sleep 10; done
  psay "card after the miners $(q clocks.sm,power.draw,power.limit,temperature.gpu,utilization.memory,memory.used)"
  psay "all done"
  exec sleep infinity
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
    m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(@\S+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if m:
        p = m.group(2).split(".")
        ops.append(p[1] if p[0] == "IMAD" and len(p) > 1 and p[1] in ("MOV", "IADD") else p[0])
if len(sys.argv) > 2 and sys.argv[2] == "ops":   # the whole kernel, run lengths by instruction name
    toks, last, n = [], None, 0
    for o in ops + [None]:
        if o == last:
            n += 1
        else:
            if last:
                toks.append(last + ("*%d" % n if n > 1 else ""))
            last, n = o, 1
    line = ""
    for t in toks:
        if len(line) + len(t) + 1 > 360:
            print(line); line = ""
        line += (" " if line else "") + t
    print(line)
    sys.exit()
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
gaps = gaps[1:] + [run]   # the steps after each IMMA; the last gap runs to the back branch
top = ",".join("%s:%d" % kv for kv in collections.Counter(other).most_common(7))
hist = ",".join("%d:%d" % kv for kv in sorted(collections.Counter(gaps).items()))
print("imma=%d other=%d per_imma=%.3f longest_run=%d runs_over_8=%d ctl=%d[%s] top=%s gaps=%s" % (
    len(im), len(other), len(other) / len(im), best, sum(1 for g in gaps if g > 8), len(ctl),
    ",".join(sorted(set(ctl))), top, hist))
EOF
# smi.py FILE T0 SKIP [TAGS]: the card's samples from SKIP seconds after T0: means, the clock's range, the limit.
# TAGS names the columns after the fifth (why: the clock's reasons; inst: instant draw; enf: enforced limit).
cat > $P/smi.py << 'EOF'
import sys
f, t0, skip = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tags = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []
rows, why, inst, enf = [], set(), [], set()
for line in open(f):
    p = [x.strip() for x in line.split(",")]
    try:
        t = float(p[0]); v = [float(x) for x in p[1:6]]
    except (ValueError, IndexError):
        continue
    if t - t0 >= skip:
        rows.append(v)
        for i, tag in enumerate(tags):
            x = p[6 + i] if len(p) > 6 + i else ""
            if tag == "why":
                why.add(x)
            elif tag == "enf":
                enf.add(x)
            elif tag == "inst":
                try:
                    inst.append(float(x))
                except ValueError:
                    pass
if not rows:
    print("no card samples"); sys.exit()
n = len(rows)
m = [sum(r[i] for r in rows) / n for i in range(5)]
print("%.0f MHz (%.0f-%.0f) %.1f W (max %.1f) of %.0f W, %.0f C, util %.0f%%, reasons %s, %d samples" % (
    m[0], min(r[0] for r in rows), max(r[0] for r in rows), m[1], max(r[1] for r in rows), m[4], m[2], m[3],
    "/".join(sorted(why)) or "-", n)
    + (", instant %.1f W (%.1f-%.1f)" % (sum(inst) / len(inst), min(inst), max(inst)) if inst else "")
    + (", enforced %s W" % "/".join(sorted(enf)) if enf else ""))
EOF
SQ=clocks.sm,power.draw,temperature.gpu,utilization.gpu,power.limit
SQT=""
has_field() { nvidia-smi -i 0 --query-gpu=$1 --format=csv,noheader > /dev/null 2>&1; }
for r in clocks_event_reasons.active clocks_throttle_reasons.active; do
  if has_field $r; then SQ=$SQ,$r; SQT=why; break; fi
done
has_field power.draw.instant && { SQ=$SQ,power.draw.instant; SQT=$SQT,inst; }
has_field enforced.power.limit && { SQ=$SQ,enforced.power.limit; SQT=$SQT,enf; }
SQT=${SQT#,}

# The streams: name|flags. The first block is the 4090 table's (lists 5 and 6 of the study); "r" after a name is
# the same stream with hashed operand bytes (FILLRND), which changes the watts and not the code between the IMMA.
F="-DRD=2 -DTRW=0 -DDRAW=4 -DDPOS=1 -DDREAL=1"
# The sm_120 follow-up's streams (after m0 below) are not unrolled: Z0 without the fragment loads, Z1 with them.
# l, L: warp 0 alone on its SM; Q: warps 0 and 4 (one scheduler); f, F: all eight. The number is the steps after
# every IMMA (one XOR3 of sums written 8 IMMA earlier, then plain steps chained on it). U: unpinned plain steps.
# D: two XOR3 an IMMA of sums written that many IMMA earlier. B, C: one and two steps an IMMA on average, in
# gaps after every 2nd to 48th IMMA. O, BW: every IMMA reads the same fragments, and W steps write one of them.
# K: two plain steps of one kind after the XOR3. H: four or eight folding warps and four that only help.
Z0="-DNOUNROLL=1 -DKLOAD=0"
Z1="-DNOUNROLL=1"
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
d1m|-DDRAW=1 -DDPOS=1
d2m|-DDRAW=2 -DDPOS=1
d3m|-DDRAW=3 -DDPOS=1
d1e|-DDRAW=1 -DDPOS=0
d2e|-DDRAW=2 -DDPOS=0
d3e|-DDRAW=3 -DDPOS=0
d3r|-DDRAW=3 -DDPOS=1 -DDREAL=1
d2r|-DDRAW=2 -DDPOS=1 -DDREAL=1
rd1c4|-DRD=1 -DRCH=4
rd1c16|-DRD=1 -DRCH=16
rd2c4|-DRD=2 -DTRW=0 -DRCH=4
rd2c16|-DRD=2 -DTRW=0 -DRCH=16
p1k0|-DPAD=1 -DPKIND=0
p2k0|-DPAD=2 -DPKIND=0
p4k0|-DPAD=4 -DPKIND=0
p2k4|-DPAD=2 -DPKIND=4
p1k1|-DPAD=1 -DPKIND=1
p2k1|-DPAD=2 -DPKIND=1
p1k2|-DPAD=1 -DPKIND=2
p1k3|-DPAD=1 -DPKIND=3
w4p|-DKLOAD=0 -DWARPS=4 -DPAD=2 -DPKIND=0
w8p|-DKLOAD=0 -DPAD=2 -DPKIND=0
w8p4|-DKLOAD=0 -DPAD=4 -DPKIND=0
w12a24|-DKLOAD=0 -DWARPS=12 -DACC=24
w12p|-DKLOAD=0 -DWARPS=12 -DACC=24 -DPAD=2 -DPKIND=0
w12p4|-DKLOAD=0 -DWARPS=12 -DACC=24 -DPAD=4 -DPKIND=0
w16a16|-DKLOAD=0 -DWARPS=16 -DACC=16
w16p|-DKLOAD=0 -DWARPS=16 -DACC=16 -DPAD=2 -DPKIND=0
w16p4|-DKLOAD=0 -DWARPS=16 -DACC=16 -DPAD=4 -DPKIND=0
w8q|-DKLOAD=0 -DPAD=1 -DPKIND=1
w12q|-DKLOAD=0 -DWARPS=12 -DACC=24 -DPAD=1 -DPKIND=1
w16q|-DKLOAD=0 -DWARPS=16 -DACC=16 -DPAD=1 -DPKIND=1
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
m0|-DKLOAD=0 -DWMASK=0x01u
l0|$Z0 -DWMASK=0x01u
l1|$Z0 -DWMASK=0x01u -DGX=1
l2|$Z0 -DWMASK=0x01u -DGX=1 -DGA=1
l4|$Z0 -DWMASK=0x01u -DGX=1 -DGA=3
l8|$Z0 -DWMASK=0x01u -DGX=1 -DGA=7
l16|$Z0 -DWMASK=0x01u -DGX=1 -DGA=15
L0|$Z1 -DWMASK=0x01u
L1|$Z1 -DWMASK=0x01u -DGX=1
L2|$Z1 -DWMASK=0x01u -DGX=1 -DGA=1
L4|$Z1 -DWMASK=0x01u -DGX=1 -DGA=3
L8|$Z1 -DWMASK=0x01u -DGX=1 -DGA=7
L16|$Z1 -DWMASK=0x01u -DGX=1 -DGA=15
Q0|$Z1 -DWMASK=0x11u
Q1|$Z1 -DWMASK=0x11u -DGX=1
Q2|$Z1 -DWMASK=0x11u -DGX=1 -DGA=1
Q4|$Z1 -DWMASK=0x11u -DGX=1 -DGA=3
Q8|$Z1 -DWMASK=0x11u -DGX=1 -DGA=7
Q16|$Z1 -DWMASK=0x11u -DGX=1 -DGA=15
f0|$Z0
f1|$Z0 -DGX=1
f2|$Z0 -DGX=1 -DGA=1
f4|$Z0 -DGX=1 -DGA=3
f8|$Z0 -DGX=1 -DGA=7
f16|$Z0 -DGX=1 -DGA=15
F0|$Z1
F1|$Z1 -DGX=1
F2|$Z1 -DGX=1 -DGA=1
F4|$Z1 -DGX=1 -DGA=3
F8|$Z1 -DGX=1 -DGA=7
F16|$Z1 -DGX=1 -DGA=15
U1|$Z1 -DGA=1
LU1|$Z1 -DWMASK=0x01u -DGA=1
U2|$Z1 -DGA=2
LU2|$Z1 -DWMASK=0x01u -DGA=2
U4|$Z1 -DGA=4
LU4|$Z1 -DWMASK=0x01u -DGA=4
D0|$Z1 -DGX=2 -DGD=0
D1|$Z1 -DGX=2 -DGD=1
D2|$Z1 -DGX=2 -DGD=2
D4|$Z1 -DGX=2 -DGD=4
D8|$Z1 -DGX=2 -DGD=8
D16|$Z1 -DGX=2 -DGD=16
D24|$Z1 -DGX=2 -DGD=24
D40|$Z1 -DGX=2 -DGD=40
LD0|$Z1 -DWMASK=0x01u -DGX=2 -DGD=0
LD1|$Z1 -DWMASK=0x01u -DGX=2 -DGD=1
LD2|$Z1 -DWMASK=0x01u -DGX=2 -DGD=2
LD8|$Z1 -DWMASK=0x01u -DGX=2 -DGD=8
B2|$Z1 -DGX=1 -DGA=1 -DGEV=2
B4|$Z1 -DGX=1 -DGA=3 -DGEV=4
B8|$Z1 -DGX=1 -DGA=7 -DGEV=8
B16|$Z1 -DGX=1 -DGA=15 -DGEV=16
B48|$Z1 -DGX=1 -DGA=47 -DGEV=48
C2|$Z1 -DGX=1 -DGA=3 -DGEV=2
C8|$Z1 -DGX=1 -DGA=15 -DGEV=8
C48|$Z1 -DGX=1 -DGA=95 -DGEV=48
OS0|$Z1 -DOPSAME=1
OWA|$Z1 -DOPSAME=1 -DGW=1
OW0|$Z1 -DOPSAME=1 -DGX=1 -DGA=1
OW1|$Z1 -DOPSAME=1 -DGX=1 -DGW=1
BW2|$Z1 -DOPSAME=1 -DGX=1 -DGW=1 -DGEV=2
BW8|$Z1 -DOPSAME=1 -DGX=1 -DGA=6 -DGW=1 -DGEV=8
BW48|$Z1 -DOPSAME=1 -DGX=1 -DGA=46 -DGW=1 -DGEV=48
LOS0|$Z1 -DOPSAME=1 -DWMASK=0x01u
LOWA|$Z1 -DOPSAME=1 -DWMASK=0x01u -DGW=1
LOW0|$Z1 -DOPSAME=1 -DWMASK=0x01u -DGX=1 -DGA=1
LOW1|$Z1 -DOPSAME=1 -DWMASK=0x01u -DGX=1 -DGW=1
K1|$Z1 -DGX=1 -DGA=2 -DGK=1
K2|$Z1 -DGX=1 -DGA=2 -DGK=2
K3|$Z1 -DGX=1 -DGA=2 -DGK=3
K4|$Z1 -DGX=1 -DGA=2 -DGK=4
K5|$Z1 -DGX=1 -DGA=2 -DGK=5
K6|$Z1 -DGX=1 -DGA=2 -DGK=6
K8|$Z1 -DGX=1 -DGA=2 -DGK=8
K9|$Z1 -DGX=1 -DGA=2 -DGK=9
K10|$Z1 -DGX=1 -DGA=2 -DGK=10
H0|$Z0 -DHELP=0xf0u -DHK=-1
H1|$Z0 -DHELP=0xf0u -DHK=0 -DHA=8
H1B|$Z0 -DHELP=0xf0u -DHK=0 -DHA=2
H2|$Z0 -DHELP=0xf0u -DHK=1 -DHA=1
H3|$Z0 -DHELP=0xf0u -DHK=1 -DHA=4
H4|$Z0 -DHELP=0xf0u -DHK=2 -DHA=4
H1S|$Z0 -DHELP=0x10u -DWMASK=0x1fu -DHK=0 -DHA=8
H8|$Z0 -DWARPS=12 -DACC=32 -DHELP=0xf00u -DHK=-1
H8B|$Z0 -DWARPS=12 -DACC=32 -DHELP=0xf00u -DHK=0 -DHA=8
H8C|$Z0 -DWARPS=12 -DACC=32 -DHELP=0xf00u -DHK=1 -DHA=1
HL0|$Z1 -DHELP=0xf0u -DHK=-1
HL1|$Z1 -DHELP=0xf0u -DHK=0 -DHA=8
HL2|$Z1 -DHELP=0xf0u -DHK=1 -DHA=1
rd2aN|$Z1 -DRD=2 -DTRW=0
d4mN|$Z1 -DDRAW=4 -DDPOS=1
full3N|$Z1 -DRD=2 -DTRW=0 -DDRAW=3 -DDPOS=1 -DDREAL=1 -DDSYNC=1
rd2aO|$Z1 -DRD=2 -DTRW=0 -Xptxas -O1
d4mO|$Z1 -DDRAW=4 -DDPOS=1 -Xptxas -O1
full3O|$Z1 -DRD=2 -DTRW=0 -DDRAW=3 -DDPOS=1 -DDREAL=1 -DDSYNC=1 -Xptxas -O1
F4O|$Z1 -DGX=1 -DGA=3 -Xptxas -O1
A1|$Z1 -DGA=1 -DGK=1
A2|$Z1 -DGA=1 -DGK=2
A3|$Z1 -DGA=1 -DGK=3
A4|$Z1 -DGA=1 -DGK=4
A5|$Z1 -DGA=1 -DGK=5
A6|$Z1 -DGA=1 -DGK=6
A9|$Z1 -DGA=1 -DGK=9
A11|$Z1 -DGA=1 -DGK=11
A9h|$Z1 -DGA=1 -DGK=9 -DGEV=2
A11h|$Z1 -DGA=1 -DGK=11 -DGEV=2
M03|$Z1 -DGX=1 -DWMASK=0x03u
M0f|$Z1 -DGX=1 -DWMASK=0x0fu
M33|$Z1 -DGX=1 -DWMASK=0x33u
M55|$Z1 -DGX=1 -DWMASK=0x55u
M77|$Z1 -DGX=1 -DWMASK=0x77u
N0f|$Z1 -DWMASK=0x0fu
N33|$Z1 -DWMASK=0x33u
Mff|$Z1 -DGX=1 -DWMASK=0xffu
full2N|$Z1 -DRD=2 -DTRW=0 -DDRAW=2 -DDPOS=1 -DDREAL=1 -DDSYNC=1
rd4N|$Z1 -DRD=2 -DTRW=0 -DDRAW=4 -DDPOS=1
d3mN|$Z1 -DDRAW=3 -DDPOS=1
d2mN|$Z1 -DDRAW=2 -DDPOS=1
baseF|$Z1 -DFILLRND=2
rd2aF|$Z1 -DRD=2 -DTRW=0 -DFILLRND=2
d4mF|$Z1 -DDRAW=4 -DDPOS=1 -DFILLRND=2
d3mF|$Z1 -DDRAW=3 -DDPOS=1 -DFILLRND=2
rd4F|$Z1 -DRD=2 -DTRW=0 -DDRAW=4 -DDPOS=1 -DFILLRND=2
rd3F|$Z1 -DRD=2 -DTRW=0 -DDRAW=3 -DDPOS=1 -DFILLRND=2
FP1|$Z1 -DGX=1 -DPAD=1 -DPKIND=0
rd3N|$Z1 -DRD=2 -DTRW=0 -DDRAW=3 -DDPOS=1
rd2N|$Z1 -DRD=2 -DTRW=0 -DDRAW=2 -DDPOS=1
r4|$Z1 -DRD=4 -DTRW=0
r4d1|$Z1 -DRD=4 -DTRW=0 -DDRAW=1 -DDPOS=1
r4d2|$Z1 -DRD=4 -DTRW=0 -DDRAW=2 -DDPOS=1
r4d3|$Z1 -DRD=4 -DTRW=0 -DDRAW=3 -DDPOS=1
d1mN|$Z1 -DDRAW=1 -DDPOS=1
r4F|$Z1 -DRD=4 -DTRW=0 -DFILLRND=2
r4d1F|$Z1 -DRD=4 -DTRW=0 -DDRAW=1 -DDPOS=1 -DFILLRND=2
r4d2F|$Z1 -DRD=4 -DTRW=0 -DDRAW=2 -DDPOS=1 -DFILLRND=2
r4d3F|$Z1 -DRD=4 -DTRW=0 -DDRAW=3 -DDPOS=1 -DFILLRND=2
d2mF|$Z1 -DDRAW=2 -DDPOS=1 -DFILLRND=2
d1mF|$Z1 -DDRAW=1 -DDPOS=1 -DFILLRND=2"

# build DIR NVCC-HOME NAME FLAGS: the program, with ptxas's registers and spill and what stands between its IMMA
# (read from the program's own kernel image; from a cubin of the same flags if cuobjdump can't read the program).
build1() {
  local d=$1 home=$2 name=$3 flags=$4 used spill
  "$home/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -cudart static -Xptxas -v $flags "$SRC" -o "$d/$name" -ldl > "$d/$name.log" 2>&1 \
    || "$home/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -Xptxas -v $flags "$SRC" -o "$d/$name" -ldl > "$d/$name.log" 2>&1
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
L2=${PP_L2:-pure base rd1 rd2a d4m d4r fullns full full3}
B1="l0 l1 l2 l4 l8 l16 L0 L1 L2 L4 L8 L16 Q0 Q1 Q2 Q4 Q8 Q16 f0 f1 f2 f4 f8 f16 F0 F1 F2 F4 F8 F16 U1 LU1 U2 LU2 U4 LU4 D0 D1 D2 D4 D8 D16 D24 D40 LD0 LD1 LD2 LD8 B2 B4 B8 B16 B48 C2 C8 C48 OS0 OWA OW0 OW1 BW2 BW8 BW48 LOS0 LOWA LOW0 LOW1 K1 K2 K3 K4 K5 K6 K8 K9 K10 H0 H1 H1B H2 H3 H4 H1S H8 H8B H8C HL0 HL1 HL2 rd2aN d4mN full3N rd2aO d4mO full3O F4O"
case "${PP_SET:-}" in
  b1) PP_BUILD="pure base rd2a d4m full3 p2k0 p2k1 $B1"
      PP_TRACES="l4 L4 Q4 f4 F4 F8 F16 U2 U4 D0 D8 B8 B48 C48 OWA OW1 BW8 BW48 K8 K9 rd2aN d4mN full3N rd2aO d4mO full3O F4O"
      PP_OPS="H1 H2 H4 H8B" ;;
  # b2: steps placed exactly (perm.py): one step after every IMMA by kind, two to eight a gap, the readout and
  # the draw one or two a gap, and one XOR3 an IMMA with one to seven warps running.
  b2) PP_BUILD="base F0 F1 p2k0 p2k1 U1 B2 B48 rd2aN d4mN full3N A1 A2 A3 A4 A5 A6 A9 A11 A9h A11h M03 M0f M33 M55 M77 N0f N33 Mff"
      PP_TRACES="A11 A11h A9"
      PP_PTRACE="F1.e1 rd2aN.x1 d4mN.e1 A11h.s1"
      PP_PERMS="F1:o:*=orig F1:e1:X=even,1 p2k0:e1:A=even,1 p2k0:e2:A=even,2 p2k0:e4:A=even,4 p2k0:e8:A=even,8 p2k0:b:A=all,10 U1:h1:A=every,2,1 U1:h2:A=every,4,2 U1:e1:A=even,1 B2:s1:X=every,2,1,0/A=every,2,1,1 B2:s2:X=every,2,1,1/A=every,2,1,1 B2:s3:X=even,1,0/A=even,1,96 B48:e1:*=even,1 B48:b48:*=every,48,0,47 B48:b8:*=every,8,0,7 rd2aN:x1:X=even,1,8 rd2aN:x2:X=even,2,8 d4mN:e1:*=even,1 d4mN:e2:*=even,2 d4mN:e4:*=even,4 d4mN:b:*=every,48,0,23 full3N:e1:*=even,1 full3N:e2:*=even,2 A1:e1:A=even,1 A2:e1:A=even,1 A3:e1:A=even,1 A4:e1:A=even,1 A5:e1:A=even,1 A6:e1:A=even,1 A9:e1:S=even,1 A9h:e1:S=every,2,1 A11h:s1:L=every,2,1,0/X=every,2,1,1 A11h:s2:L=every,2,1,1/X=every,2,1,1 A11:e1:L=even,1/X=even,1,0 F1:x2:X=every,2,2 A9h:e2:S=even,1 p2k0:v2:A=every,2,2 p2k0:v4:A=every,4,4" ;;
  # b3: the same with steps held where their registers allow (so each kernel computes what the compiler's did):
  # the cost of one to eight independent adds a gap, what the UIADD3 that ptxas puts beside an XOR3 of sums is
  # worth, and the readout and the draw shared out one or two a gap.
  b3) PP_BUILD="base F0 F1 p2k0 rd2aN d4mN d3mN d2mN rd4N full3N full2N A9h"
      PP_DUMPS="rd2aN:0:60 d4mN:60:110"
      PP_PTRACE="rd2aN.s1b d4mN.sp full3N.sp F1.d1n F1.d0"
      PP_PERMS="p2k0:e1:A=even,1 p2k0:e2:A=even,2 p2k0:e3:A=even,3 p2k0:e4:A=even,4 p2k0:e8:A=even,8 p2k0:b:A=all,10 F1:o:*=orig F1:n9:U=all,191 F1:d0:X=even,1,191 F1:d1:X=even,1,0 F1:d1n:X=even,1,0/U=all,191 F1:d2:X=even,1,1 F1:d2n:X=even,1,1/U=all,191 F1:d4:X=even,1,3 F1:d8:X=even,1,7 F1:d24:X=even,1,23 F1:d40:X=even,1,39 rd2aN:o:*=orig rd2aN:s1:X=seg,0:32:1,144:32:1,160:32:1 rd2aN:s1b:X=seg,0:32:1,146:32:1,162:32:1 rd2aN:s1n:X=seg,0:32:1,146:32:1,162:32:1/U=all,191 rd2aN:s2:X=seg,0:32:2,146:32:2,170:32:2 d4mN:o:*=orig d4mN:sp:*=spread d4mN:e2:*=even,2 d4mN:v2:*=every,2,0,1 d4mN:v4:*=every,4,0,3 d4mN:b:*=every,48,0,23 d3mN:sp:*=spread d2mN:sp:*=spread rd4N:sp:*=spread rd4N:e2:*=even,2 full3N:sp:*=spread full3N:e2:*=even,2 full2N:sp:*=spread A9h:e1:S=every,2,1 A9h:e2:S=even,1" ;;
  # b4: the same once more with the operand-reuse flags cleared in every reordered stretch, and a spacer step
  # between an IMMA and an XOR3 of sums.
  b4) PP_BUILD="base F0 F1 FP1 p2k0 rd2aN d4mN d3mN d2mN rd4N rd3N rd2N full3N full2N"
      PP_TRACES="FP1"
      PP_PTRACE="FP1.ax FP1.axn FP1.ax1 FP1.xa1 d4mN.sp rd4N.sp full3N.sp"
      PP_PERMS="p2k0:e1:A=even,1 p2k0:e2:A=even,2 p2k0:e3:A=even,3 p2k0:e4:A=even,4 p2k0:e8:A=even,8 p2k0:b:A=all,10 F1:n9:U=all,191 FP1:o:*=orig FP1:ax:A=even,1,0/X=even,1,0 FP1:axn:A=even,1,0/X=even,1,0/U=all,191 FP1:ax1:A=even,1,0/X=even,1,1/U=all,191 FP1:xa1:A=even,1,1/X=even,1,0/U=all,191 rd2aN:s1n:X=seg,0:32:1,146:32:1,162:32:1/U=all,191 d4mN:c:R=clear d4mN:sp:*=spread d4mN:e2:*=even,2 d4mN:v2:*=every,2,0,1 d4mN:v4:*=every,4,0,3 d4mN:b:*=every,48,0,23 d3mN:sp:*=spread d2mN:sp:*=spread rd4N:sp:*=spread rd4N:e2:*=even,2 rd3N:sp:*=spread rd2N:sp:*=spread full3N:sp:*=spread full2N:sp:*=spread" ;;
  # b5: the readout written one XOR3 after every IMMA (RD=4), alone and with one to three draw units whose steps
  # are shared out over the gaps while the XOR3 stay where the compiler put them.
  b5) PP_BUILD="base F1 rd2aN d4mN d3mN d2mN d1mN rd4N rd2N r4 r4d1 r4d2 r4d3"
      PP_TRACES="r4 r4d2"
      PP_DUMPS="d4mN:60:110 r4:0:24"
      PP_PTRACE="r4d2.d r4d1.d d4mN.sp"
      PP_PERMS="d4mN:sp:*=spread d4mN:e2:*=even,2 d4mN:b:*=every,48,0,23 d3mN:sp:*=spread d2mN:sp:*=spread d1mN:sp:*=spread rd4N:sp:*=spread rd2N:sp:*=spread r4:o:*=orig r4d1:d:YLAS=spread r4d2:d:YLAS=spread r4d3:d:YLAS=spread r4d1:s:*=spread r4d2:s:*=spread" ;;
esac
if [ -n "${PP_SET:-}" ]; then
  PP_SHORTS=${PP_SHORTS:-$PP_BUILD}; PP_SKIP_SHORT=1; PP_LONGS=${PP_LONGS:-none}; PP_HR_SECS=0; PP_V=1
fi
buildall $P/b "$CUDA" "nvcc$CUDA_VER" "${PP_BUILD:-}"
for s in ${PP_OPS:-}; do
  [ -s "$P/b/$s.sass" ] && python3 $P/sass.py "$P/b/$s.sass" ops 2>&1 | while IFS= read -r l; do psay "nvcc$CUDA_VER ops $s $l"; done
done
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
  out=$(${4:+env "PP_CUBIN=$4"} timeout 240 "$1/$3" "${5:-$3}" "$CH" "${PP_LONG:-30}" 2>&1 | tail -1)
  kill $sp 2> /dev/null; wait $sp 2> /dev/null
  psay "$2 long $out | $(python3 $P/smi.py "$f" "$t0" 8 "$SQT")"
}
T=nvcc$CUDA_VER
# perm NAME TAG RULES: the stream's kernel with its loop's steps placed by perm.py's RULES, launched from the
# reordered cubin by the stream's own program. The line before the run gives the gaps asked for, and the gaps
# read back from the reordered cubin.
# cub NAME: the stream's kernel as a cubin, and its listing.
cub() {
  local name=$1 d=$P/b flags
  [ -s "$d/$name.csass" ] && return 0
  flags=$(echo "$STREAMS" | awk -F'|' -v n="$name" '$1 == n { print $2; exit }')
  "$CUDA/bin/nvcc" -arch=$ARCH -std=c++17 -O3 -cubin $flags "$SRC" -o "$d/$name.cubin" > "$d/$name.cubin.log" 2>&1
  "$CUDA/bin/nvdisasm" -c "$d/$name.cubin" > "$d/$name.csass" 2> /dev/null
  [ -s "$d/$name.csass" ]
}
perm() {
  local name=$1 tag=$2 rules=$3 d=$P/b l
  [ -x "$d/$name" ] || { psay "$T perm $name.$tag: no program"; return; }
  cub "$name"
  l=$(python3 "$R/earn/tuning/pipeprobe/perm.py" "$d/$name.cubin" "$d/$name.csass" "$d/$name.$tag.cubin" "$rules" 2>&1 | tail -1)
  [ -s "$d/$name.$tag.cubin" ] || { psay "$T perm $name.$tag FAIL $l"; return; }
  "$CUDA/bin/nvdisasm" -c "$d/$name.$tag.cubin" > "$d/$name.$tag.sass" 2> /dev/null
  psay "$T $name.$tag $l | read back $(python3 $P/sass.py "$d/$name.$tag.sass" 2>&1 | grep -oE 'gaps=.*')"
  case " ${PP_PTRACE:-} " in *" $name.$tag "*)
    python3 $P/sass.py "$d/$name.$tag.sass" trace 2>&1 | while IFS= read -r l; do psay "$T trace $name.$tag $l"; done ;;
  esac
  PP_CUBIN="$d/$name.$tag.cubin" timeout 120 "$d/$name" "$name.$tag" "$CH" "${PP_SHORT:-1}" v 2>&1 \
    | while IFS= read -r l; do psay "$T short $l"; done
}
# 1. What a full pipe is here: IMMA only with one to four warps a scheduler, the warp-to-pipe map, and the same
#    with the fragment loads.
[ -n "${PP_SKIP_SHORT:-}" ] || for s in m0 m01 m02 m03 m04 m05 m06 m07 w4 pure w8a32 w8a24 w12 w16 w4k base w12k w16k; do
  [ -x $P/b/$s ] && short $P/b $T $s v
done
# 2. The 4090 table's streams, pipe use only.
[ -n "${PP_SKIP_SHORT:-}" ] || for s in pure base rd1 rd1a rd2 rd2a rd3 d4e d4m d4h d4s d4mg d4r fullns full full3 fulle; do
  [ -x $P/b/$s ] && short $P/b $T $s
done
# 2b. What a step between the IMMA costs by count and kind: one to three draw units, the readout's chains, and
#     one kind of step after every IMMA (p<PAD>k<PKIND>; ptxas folds two adds into one, so p2k0 is one add an
#     IMMA, and p1k2 is one shared load and half an XOR).
#     w<warps>p, w<warps>p4, w<warps>q (not run unless PP_SHORTS names them): one add, two adds or one XOR3 of
#     earlier sums an IMMA with one to four warps a scheduler, IMMA only; w12a24 and w16a16 are their
#     streams without the step.
#     PP_SHORTS names other streams to run short (with PP_SKIP_SHORT, only these run short).
for s in ${PP_SHORTS:-d1m d2m d3m d1e d2e d3e d2r d3r rd1c4 rd1c16 rd2c4 rd2c16 p1k0 p2k0 p4k0 p2k4 p1k1 p2k1 p1k2 p1k3}; do
  [ -x $P/b/$s ] && short $P/b $T $s ${PP_V:+v}
done
# 2c. Reordered kernels.
for x in ${PP_DUMPS:-}; do
  IFS=: read -r pn pa pb <<< "$x"
  cub "$pn" && python3 "$R/earn/tuning/pipeprobe/perm.py" "$P/b/$pn.cubin" "$P/b/$pn.csass" - - dump "$pa" "$pb" 2>&1     | while IFS= read -r l; do psay "$T dump $pn $l"; done
done
set -f   # a rule may hold a "*"
for x in ${PP_PERMS:-}; do
  IFS=: read -r pn pt pr <<< "$x"
  perm "$pn" "$pt" "$pr"
done
set +f
# 3. The long runs, with clock and watts: the offset-slice bytes first, then hashed bytes, then base again.
for s in ${PP_LONGS:-pure base rd1 rd2a d4m d4r full full3 baser rd1r rd2ar d4mr base}; do
  [ -x $P/b/$s ] && long $P/b $T $s
done
# 3b. Long runs of reordered kernels (PP_PLONGS: NAME.TAG ..., each made by PP_PERMS above).
for x in ${PP_PLONGS:-}; do
  [ -s "$P/b/$x.cubin" ] && long $P/b $T "${x%%.*}" "$P/b/$x.cubin" "$x"
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
