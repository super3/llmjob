#!/bin/bash
# Box script for the rotation fold on an RTX 50 card (sm_120), branch goal/nv-50: on one Salad PC, the fold's probe
# (earn/native/probes/hand89/gprobe.cu built with GP_ROTC=1, the rotation fold compiled from C), then the fold in
# the miner's core against the release fold in the same core (verify-hits.js, then hashrate.js legs taking turns),
# then the other miners and our release through bench.sh. A measuring tool for this branch: nothing here ships.
# nv30box.sh's sm_120 twin. No hand loop: the cubin is pearl_rot.cu as ptxas compiles it for sm_120.
#
# Two toolkits: CUDA 13.3.1 (the one pearl_core_cu13.node, the core RTX 50 rigs run, is built with) for the addon,
# and CUDA 12.8.1 beside it, so the fold's cubin can be raced as each compiler makes it.
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   STAGES        any of "ldsm8 probe micro powcap product cli bench", in that order (default probe product bench)
#   PROBE_BUILDS  the probe's builds, "name:toolkit:nvcc flags;..." (toolkit 12 or 13; flags with "," for " "),
#                 each gprobe.cu with -DGP_STAGE=24576 and the flags (GP_ROTC=1/3/5/7: pearl_rot_c.inc's forms)
#   PROBE_LIST    the probe's runs, ';' between them: "<build> <nv30probe.py line>" (default: for each build a
#                 wide check at k 2048 and 60 s at k 65536)
#   ROT_BUILDS    the miner's cubins, "name:toolkit:nvcc flags;..." (ROT_C=1/3/5/7), pearl_rot.cu for sm_120
#   PROBE_SCHED   probe cubins scheduled again by probes/hand89/sched120.py, "name:base:key=v,key=v;..." (base a
#                 PROBE_BUILDS name); run with the base's program (PD_CUBIN)
#   PROBE_HAND    probe cubins with rfold.py's hand loop put in by probes/hand89/rfold120.py,
#                 "name:base:key=v,...;..." (base a PROBE_BUILDS name built with GP_ROTC=16387)
#   ROT_HAND      the same as PROBE_HAND for the miner's cubins (base a ROT_BUILDS name with -DROT_INLINE=0 -DROT_C=16387);
#                 list them in VERIFY_CUBINS and HR_LEGS
#   ROT_SHIP      1: also build the file CI ships (build_rot.py --arch sm_120, or ROT_SHIP_ARCH) as rship
#                 2 (goal-nv50t): both RTX 50 files CI ships, the sm_120 one as rship and the sm_120w one as rshipw,
#                 and both put beside the addon, where the core finds them when no PEARL_ROT_CUBIN is set
#   ROT_SCHED     the same for the miner's cubins (base a ROT_BUILDS name); the names join ROT_BUILDS' for
#                 VERIFY_CUBINS and HR_LEGS
#   VERIFY_SECS   verify-hits.js on each cubin of VERIFY_CUBINS (default all of ROT_BUILDS) for this many
#                 seconds (default 150; 0 skips it), at 2^VERIFY_BITS (233)
#   HR_LEGS       hashrate.js legs: rel (the release fold) or a ROT_BUILDS name; a leg may carry core settings
#                 after "+" (default: rel, every cubin, every cubin backwards, rel); HR_SECS (120)
#   CLI_LEGS      stage "cli" (goal-nv50s): our CLI from source at BASE_SHA mining on the stand-in pool
#                 (verify-pool.js at 2^CLI_BITS, default 206; a new job every CLI_JOB s, default 30) with the
#                 addon and cubins above, beside hashrate.js on the same file: "c:<cubin>" a CLI leg, "h:<cubin>"
#                 a hashrate.js leg, CLI_SECS each (180). Whether the CLI loses rate against hashrate.js.
#                 With ROT_SHIP=2 (goal-nv50t): "t:<n>" a CLI leg with the files beside the addon and a fresh shape
#                 cache, so the core's startup trial times the two tiles (its 'launch shape' lines are printed);
#                 "f:w" / "f:n" a CLI leg forced to the 96 x 512 / 192 x 256 tile (PEARL_SHAPE).
#                 "o:<n>" (goal-f-rtx50) a "t" leg with PEARL_SHAPE_TRIAL_DELAY=0: the trial from the first job.
#                 Each of these legs also prints a "whole" line, the leg scored as bench.py scores a row.
#                 CLI_ENV: settings for every such leg (e.g. PEARL_ROT_LOG=2).
#   PWLOOP        (goal-nv50w) NVML power fields by scope every PWLOOP s (nvmlpw.c), and nvidia-smi -q once under load
#   ORDER, VPOOL, SECS, WARM, VER   passed to bench.sh (defaults ours,ozm,peak,srb; 210; 180; 60; v0.6.15)
set -u
NVTEST_TAG=nv50
[ -n "${BASE_SHA:-}" ] || { echo "[nvtest] $(date -u +%T) [nv50] no BASE_SHA; idle"; exec sleep infinity; }
# The start the nvNNbox.sh scripts share: say, q, nvbox_boot, nvbox_apt.
curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad/nvbox.sh" \
  -o /root/nvbox.sh || { echo "[nvtest] $(date -u +%T) [nv50] FAIL fetch nvbox.sh"; exec sleep infinity; }
# shellcheck disable=SC1091
. /root/nvbox.sh
NVBOX_MEM=1 nvbox_boot nv30probe.py nvprobe.py
[ "$CC" = 120 ] || say "WARN this card is compute $CC, not 12.0"
nvbox_apt
# nvtest.sh's functions: setup, cuda, fetch. Sourced once a toolkit: its variables are set when it is read.
export NVTEST_STAGE=lib
export CUDA_VER=12.8 NVTEST_CUDA=/opt/cuda12
# shellcheck disable=SC1091
. /root/nvtest.sh
setup
cuda
C12=$CUDA
export CUDA_VER=13.3 NVTEST_CUDA=/opt/cuda13
# shellcheck disable=SC1091
. /root/nvtest.sh
driver_ok || fail "driver too old for CUDA 13"
cuda
C13=$CUDA
# nvdisasm 13.3, which sched120.py reads cubins with (redistrib_13.3.1.json's checksum).
ND=cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-13.3.73-archive.tar.xz
if [ ! -x "$C13/bin/nvdisasm" ]; then
  curl -fsSL --retry 3 "$RD/$ND" -o /root/nvd.tar.xz || fail "nvdisasm download"
  echo "38b2eca90c0ffa2ce815bd392077f68b7fe94992ea0ec0f8e85df30ca01fc4e9  /root/nvd.tar.xz" | sha256sum -c --quiet - \
    || fail "nvdisasm checksum"
  tar -xJf /root/nvd.tar.xz -C "$C13" --strip-components=1 || fail "nvdisasm unpack"
fi
NVD=$C13/bin/nvdisasm
unset NVTEST_STAGE
fetch
export PATH="/opt/node/bin:$PATH"
STAGES=" ${STAGES:-probe product bench} "
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
H=$R/earn/native/probes/hand89
W=/root/w
mkdir -p $W
N12="$C12/bin/nvcc -arch=sm_120 -std=c++17 -O3"
N13="$C13/bin/nvcc -arch=sm_120 -std=c++17 -O3"
regs() { grep -E 'Used [0-9]+ registers|spill' "$1" | tr -s ' ' | tr '\n' ' ' | cut -c1-300; }

# ---------------------------------------------------------------- builds (no GPU)
nv() { if [ "$1" = 12 ]; then echo "$N12"; else echo "$N13"; fi; }
# sched "name:base:key=v,key=v;...": $W/base.cubin scheduled again into $W/name.cubin
sched() {
  local IFS_OLD=$IFS b n r base keys out
  IFS=';' read -r -a SL <<< "$1"
  IFS=$IFS_OLD
  for b in "${SL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; base=${r%%:*}; keys=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2086
    out=$(python3 -I "$H/sched120.py" "$W/$base.cubin" "$NVD" "$W/$n.cubin" $keys 2>&1)
    if [ -f "$W/$n.cubin" ]; then say "sched $n ($base $keys): $(echo "$out" | tail -1 | cut -c1-330)"
    else say "sched $n ($base $keys) FAILED: $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-330)"; fi
  done
}
# hand "name:base:key=v,key=v;...": rfold.py's hand loop put into $W/base.cubin by rfold120.py, as $W/name.cubin
# (base a probe build with GP_ROTC=16387 or another form with flag 16384: room for the loop).
hand() {
  local IFS_OLD=$IFS b n r base keys out
  IFS=';' read -r -a HL <<< "$1"
  IFS=$IFS_OLD
  for b in "${HL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; base=${r%%:*}; keys=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2086
    out=$(python3 -I "$H/rfold120.py" "$W/$base.cubin" "$NVD" "$W/$n.cubin" $keys 2>&1)
    if [ -f "$W/$n.cubin" ]; then say "hand $n ($base $keys): $(echo "$out" | tail -1 | cut -c1-330)"
    else say "hand $n ($base $keys) FAILED: $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-330)"; fi
  done
}
PB=${PROBE_BUILDS:-"c1:13:-DGP_ROTC=1;c3:13:-DGP_ROTC=3;c5:13:-DGP_ROTC=5;c7:13:-DGP_ROTC=7;c1x:12:-DGP_ROTC=1"}
RB=${ROT_BUILDS:-"r1:13:-DROT_C=1;r3:13:-DROT_C=3;r5:13:-DROT_C=5;r7:13:-DROT_C=7"}
names() { echo "$1" | tr ';' '\n' | cut -d: -f1 | tr '\n' ' '; }
if has probe; then
  IFS=';' read -r -a BL <<< "$PB"
  for b in "${BL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; tk=${r%%:*}; fl=$(echo "${r#*:}" | tr ',' ' ')
    # The stage is 24 KB unless the flags name another (goal-nv50k: the 96 x 512 block's is 12 KB, -DGP_W8=1).
    case " $fl" in *" -DGP_STAGE="*) st="" ;; *) st="-DGP_STAGE=24576" ;; esac
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) -Xptxas -v $st $fl -o $W/$n "$H/gprobe.cu" -ldl > $W/$n.log 2>&1 || { tail -3 $W/$n.log | emit; fail "probe $n"; }
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) $st $fl -cubin -o $W/$n.cubin "$H/gprobe.cu" > /dev/null 2>&1 || fail "probe cubin $n"
    say "probe $n built ($tk: $fl): $(regs $W/$n.log)"
  done
  sched "${PROBE_SCHED:-}"
  hand "${PROBE_HAND:-}"
fi
if has product; then
  t0=$(date +%s)
  if [ ! -x /opt/node/bin/node ]; then
    mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz \
      | tar -xJ -C /opt/node --strip-components=1 || fail "node download"
  fi
  mkdir -p /opt/gyp && (cd /opt/gyp && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || fail "node-gyp setup: $(tail -1 /opt/gyp/npm.log)"
  # The fold's cubins: pearl_rot.cu as the compiler makes it for sm_120, nothing patched.
  IFS=';' read -r -a BL <<< "$RB"
  for b in "${BL[@]}"; do
    [ -n "$b" ] || continue
    n=${b%%:*}; r=${b#*:}; tk=${r%%:*}; fl=$(echo "${r#*:}" | tr ',' ' ')
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) -Xptxas -v $fl -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/$n.v.cubin > $W/$n.log 2>&1 || { tail -3 $W/$n.log | emit; fail "pearl_rot.cu $n"; }
    # The cubin itself without -Xptxas -v, as build_rot.py and CI build it: the options are recorded in the file's
    # .note.nv.tkinfo, so -v alone changes its hash (the code is the same).
    # shellcheck disable=SC2046,SC2086
    $(nv $tk) $fl -cubin -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o $W/$n.cubin > /dev/null 2>&1 || fail "pearl_rot.cu $n"
    say "rot cubin $n ($tk: $fl): fold hash $(python3 -I "$H/asm89.py" hash $W/$n.cubin) $(grep -A3 "pearl_rot_fold' for" $W/$n.log | regs /dev/stdin)"
  done
  sched "${ROT_SCHED:-}"
  hand "${ROT_HAND:-}"
  # ROT_SHIP=1: the file CI ships, built as CI builds it (build_rot.py --arch sm_120, CUDA 13.3), as $W/rship.cubin;
  # its hash is printed. Name it rship in VERIFY_CUBINS and HR_LEGS.
  # ROT_SHIP_ARCH=sm_120w builds the RTX 5090 file (pearl_rot_sm120w.cubin) instead.
  if [ "${ROT_SHIP:-0}" = 1 ] || [ "${ROT_SHIP:-0}" = 2 ]; then
    sa=${ROT_SHIP_ARCH:-sm_120}
    (cd "$R/earn/native" && PATH="$C13/bin:$PATH" python3 -X utf8 probes/hand89/build_rot.py "$C13/bin/nvcc" "$NVD" $W/br \
      --arch "$sa" > $W/br.log 2>&1)
    f=$W/br/pearl_rot_${sa/_/}.cubin; [ -f "$f" ] || f=$f.unverified
    if [ -f "$f" ]; then cp "$f" $W/rship.cubin; say "ship cubin (build_rot.py): fold hash $(python3 -I "$H/asm89.py" hash $W/rship.cubin)"
    else say "ship cubin FAILED: $(tail -3 $W/br.log | tr '\n' ' ' | cut -c1-300)"; fi
  fi
  if [ "${ROT_SHIP:-0}" = 2 ]; then
    (cd "$R/earn/native" && PATH="$C13/bin:$PATH" python3 -X utf8 probes/hand89/build_rot.py "$C13/bin/nvcc" "$NVD" $W/brw \
      --arch sm_120w > $W/brw.log 2>&1)
    f=$W/brw/pearl_rot_sm120w.cubin
    if [ -f "$f" ]; then cp "$f" $W/rshipw.cubin; say "ship cubin w (build_rot.py): fold hash $(python3 -I "$H/asm89.py" hash $W/rshipw.cubin)"
    else say "ship cubin w FAILED: $(tail -3 $W/brw.log | tr '\n' ' ' | cut -c1-300)"; fi
  fi
  # The addon for this card, as CI builds pearl_core_cu13.node (CUDA 13.3, sm_120).
  K=$W/addon
  rm -rf $K && mkdir -p $K/cuda-build && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" $K/
  C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_120,code=sm_120"
  # shellcheck disable=SC2086
  $C13/bin/nvcc $C -c $K/src/pearl_kernel.cu -o $K/cuda-build/pearl_kernel.o > $K/kernel.log 2>&1 || fail "kernel: $(grep -m2 -i error $K/kernel.log)"
  # shellcheck disable=SC2086
  $C13/bin/nvcc $C -c $K/src/pearl_host.cu -o $K/cuda-build/pearl_host.o > $K/host.log 2>&1 || fail "host: $(grep -m2 -i error $K/host.log)"
  ar rcs $K/cuda-build/libpearl_cuda.a $K/cuda-build/pearl_kernel.o $K/cuda-build/pearl_host.o
  (cd $K && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
    CUDA_PATH=$C13 ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || fail "node-gyp: $(grep -m2 -i error $K/gyp.log)"
  ADDON=$K/build/Release/pearl_core.node
  if [ "${ROT_SHIP:-0}" = 2 ]; then
    cp $W/rship.cubin $K/build/Release/pearl_rot_sm120.cubin 2>/dev/null
    cp $W/rshipw.cubin $K/build/Release/pearl_rot_sm120w.cubin 2>/dev/null
  fi
  say "addon (CUDA 13.3) built in $(( $(date +%s) - t0 ))s: $(ls -l $ADDON | awk '{print $5}') bytes"
  # ADDON_VARIANTS (goal-rtx50b): "name:nvcc flags;..." (flags with "," for " "), the same addon with pearl_kernel.cu
  # (the release fold) compiled with the extra flags, e.g. "O1:-Xptxas,-O1". A leg or verify "rel@name" runs on it.
  IFS=';' read -r -a AV <<< "${ADDON_VARIANTS:-}"
  for b in "${AV[@]}"; do
    [ -n "$b" ] || continue
    vn=${b%%:*}; vf=$(echo "${b#*:}" | tr ',' ' ')
    KV=$W/addon_$vn
    rm -rf $KV && mkdir -p $KV/cuda-build && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" $KV/
    # shellcheck disable=SC2086
    $C13/bin/nvcc $C $vf -Xptxas -v -c $KV/src/pearl_kernel.cu -o $KV/cuda-build/pearl_kernel.o > $KV/kernel.log 2>&1 \
      || { say "addon $vn FAILED: $(grep -m2 -i error $KV/kernel.log | cut -c1-200)"; continue; }
    cp $K/cuda-build/pearl_host.o $KV/cuda-build/
    ar rcs $KV/cuda-build/libpearl_cuda.a $KV/cuda-build/pearl_kernel.o $KV/cuda-build/pearl_host.o
    (cd $KV && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
      CUDA_PATH=$C13 ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || { say "addon $vn node-gyp FAILED"; continue; }
    [ "${ROT_SHIP:-0}" = 2 ] && cp $K/build/Release/pearl_rot_sm120*.cubin $KV/build/Release/ 2>/dev/null
    say "addon $vn ($vf) built: $(grep -c 'Used' $KV/kernel.log) kernels, spills: $(grep -o '[0-9]* bytes spill stores' $KV/kernel.log | sort -u | tr '\n' ' ' | cut -c1-120)"
  done
fi
addon_for() { case "$1" in *@*) echo "$W/addon_${1#*@}/build/Release/pearl_core.node" ;; *) echo "$ADDON" ;; esac; }

# ---------------------------------------------------------------- on the card
# SMIQ=<secs>: every <secs> a "smiq" line with what nvidia-smi -q says about power, voltage and the clock's limiters
# (the reasons that are active), for whichever miner is running then. Plain observation, nothing set.
if [ -n "${SMIQ:-}" ]; then
  ( while true; do
      say "smiq $(nvidia-smi -q -i 0 -d POWER,VOLTAGE,PERFORMANCE,CLOCK 2>&1 | awk '
        BEGIN { FS = " : " }
        /^    [A-Z]/ { s = $0; gsub(/^ +| +$/, "", s)
          t = (s == "Clocks") ? "C" : (s == "Voltage") ? "V" : (s ~ /Power Readings/) ? "P" : (s ~ /^Clocks (Event|Throttle) Reasons$/) ? "R" : (s == "Power Samples") ? "S" : "" }
        NF == 2 && t != "" { k = $1; v = $2; gsub(/^ +| +$/, "", k); gsub(/^ +| +$/, "", v)
          if (t == "R") { if (v == "Active") printf "R:%s; ", k; next }
          if (v == "N/A") next
          if (t == "P" && k !~ /^(Average Power Draw|Instantaneous Power Draw|Current Power Limit|Power Draw)$/) next
          if (t == "S" && k !~ /^(Max|Min|Avg)$/) next
          if (t == "C" && k !~ /^(Graphics|SM|Memory)$/) next
          if (t == "V" && k != "Graphics") next
          printf "%s:%s=%s; ", t, k, v }' | tr -d '%|' | cut -c1-360)"
      sleep "$SMIQ"
    done ) &
  SMIQ_PID=$!
fi
# PWLOOP=<secs> (goal-nv50w): every <secs> a "nvmlpw" block, 8 s of NVML's power fields in each scope (GPU, module,
# memory) every 50 ms, the instant draw's spread and the clock-event reason bits (nvmlpw.c), for whichever miner runs
# then. The first time the card draws over 150 W, also every line of nvidia-smi -q's power, performance, clock,
# voltage and temperature sections ("smiqfull"). Plain observation, nothing set.
if [ -n "${PWLOOP:-}" ]; then
  gcc -O2 -o /root/nvmlpw "$R/earn/tuning/salad/nvmlpw.c" -ldl > $W/nvmlpw.log 2>&1 || say "WARN nvmlpw build: $(tail -1 $W/nvmlpw.log)"
  ( full=0
    while true; do
      if [ "$full" = 0 ] && [ "$(q power.draw | cut -d. -f1)" -gt 150 ] 2>/dev/null; then
        full=1
        nvidia-smi -q -i 0 -d POWER,PERFORMANCE,CLOCK,VOLTAGE,TEMPERATURE 2>&1 | grep -v "^ *$" | tr -s ' ' | tr -d '%|' \
          | cut -c1-300 | sed "s/^/smiqfull /" | emit
      fi
      [ -x /root/nvmlpw ] && /root/nvmlpw 8 2>&1 | tr -d '%|' | cut -c1-300 | emit
      sleep "$PWLOOP"
    done ) &
  PW_PID=$!
fi
smi() { q clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown; }
if has probe; then
  L=${PROBE_LIST:-}
  if [ -z "$L" ]; then
    for n in $(names "$PB"); do L="$L$n - $n-k2048-check 2048 8 0 64 6 0 2;"; done
    for s in $(echo "${PROBE_SCHED:-}" | tr ';' '\n' | cut -d: -f1,2 | tr '\n' ' '); do
      L="$L${s#*:} ${s%%:*} ${s%%:*}-k2048-check 2048 8 0 64 6 0 2;"; done
    for n in $(names "$PB"); do L="$L$n - $n-k65536-60 65536 128 60 2 6 0 2;"; done
    for s in $(echo "${PROBE_SCHED:-}" | tr ';' '\n' | cut -d: -f1,2 | tr '\n' ' '); do
      L="$L${s#*:} ${s%%:*} ${s%%:*}-k65536-60 65536 128 60 2 6 0 2;"; done
  fi
  IFS=';' read -r -a PL <<< "$L"
  for line in "${PL[@]}"; do
    [ -n "$line" ] || continue
    # "!lgc <MHz>" locks the SM clock for the runs after it (nvidia-smi -lgc), "!rgc" lets it go: watts at a
    # locked clock price energy finer than the clock under the cap. A PC that refuses says so and runs go on.
    case "$line" in
      "!lgc "*) say "lgc ${line#!lgc }: $(nvidia-smi -i 0 -lgc "${line#!lgc }","${line#!lgc }" 2>&1 | tr '\n' ' ' | cut -c1-200)"; LGC=1; continue ;;
      "!rgc"*) say "rgc: $(nvidia-smi -i 0 -rgc 2>&1 | tr '\n' ' ' | cut -c1-200)"; continue ;;
      # "!lmc <MHz>" / "!rmc" (goal-nv50d): the memory clock, as the RTX 40 rotation fold locks it by default.
      "!lmc "*) say "lmc ${line#!lmc }: $(nvidia-smi -i 0 -lmc "${line#!lmc }","${line#!lmc }" 2>&1 | tr '\n' ' ' | cut -c1-200) now $(q clocks.mem)"; LMC=1; continue ;;
      "!rmc"*) say "rmc: $(nvidia-smi -i 0 -rmc 2>&1 | tr '\n' ' ' | cut -c1-200)"; continue ;;
    esac
    exe=${line%% *}
    # Salad drops lines over 400 characters and blanks a line with "%" or "|": "pc" and "/" here, the collision and
    # hit counts left out and the line cut to 395 (a longer name lost every probe line of a run on 12 November).
    python3 -I /root/nv30probe.py "$W/$exe" $W "${line#* }" 2>&1 | grep -v "card has" | sed -u "s/^/probe $exe /; s/%/pc/g; s/|/\//g; s/rows of 8 that collide[^/]*\/ //; s/the card.s hits [0-9]* in [0-9]* launch[a-z]*//" | cut -c1-395 | emit
  done
  [ "${LGC:-0}" = 1 ] && say "rgc at the probe's end: $(nvidia-smi -i 0 -rgc 2>&1 | tr '\n' ' ' | cut -c1-200)"
  [ "${LMC:-0}" = 1 ] && say "rmc at the probe's end: $(nvidia-smi -i 0 -rmc 2>&1 | tr '\n' ' ' | cut -c1-200)"
fi
# PROBE_SAN: probe runs under compute-sanitizer's memcheck (CUDA 13.3's, from NVIDIA's redistributables), ';'
# between them, each "<build exe> <cubin> <k> <tiles>": the first faulting access and its address, for a hand loop.
if [ -n "${PROBE_SAN:-}" ]; then
  SAN=cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-13.3.75-archive.tar.xz
  if [ ! -x /opt/san/compute-sanitizer/compute-sanitizer ]; then
    mkdir -p /opt/san && curl -fsSL --retry 3 "$RD/$SAN" -o /root/san.tar.xz &&
      echo "f409be1ba58f44a0a1c826933fa6d557659bfb5b29f4d175979d1cf9c530e3cf  /root/san.tar.xz" | sha256sum -c --quiet - &&
      tar -xJf /root/san.tar.xz -C /opt/san --strip-components=1 || say "WARN sanitizer download"
  fi
  IFS=';' read -r -a SL2 <<< "$PROBE_SAN"
  for line in "${SL2[@]}"; do
    [ -n "$line" ] || continue
    read -r exe cub k tiles <<< "$line"
    PD_CUBIN=$W/$cub.cubin timeout 300 /opt/san/compute-sanitizer/compute-sanitizer --tool memcheck --show-backtrace no \
      "$W/$exe" "san-$cub" "$k" "$tiles" 0 0 6 0 2 2>&1 | grep -vE "^=+ *$" | head -24 | cut -c1-300 | sed "s/^/san $cub /; s/%/pc/g; s/|/\//g" | emit
  done
fi
# ldsm8.cu (stage "ldsm8", goal-nv50d): which bytes sm_120a's byte-transposing ldmatrix hands each lane.
if has ldsm8; then
  $C13/bin/nvcc -arch=sm_120a -o $W/ldsm8 "$H/ldsm8.cu" > $W/ldsm8.log 2>&1 || say "FAIL ldsm8 build: $(tail -2 $W/ldsm8.log | tr '\n' ' ' | cut -c1-300)"
  [ -x $W/ldsm8 ] && $W/ldsm8 2>&1 | cut -c1-390 | emit
fi
# mmabench.cu (stage "micro"): a stream of nothing but mma, by order, and how much of the pipe it holds.
if has micro; then
  $N13 -o $W/mmabench "$H/mmabench.cu" > $W/mma.log 2>&1 || fail "mmabench: $(tail -2 $W/mma.log | tr '
' ' ')"
  $W/mmabench "${MICRO_SECS:-10}" 2>&1 | cut -c1-300 | emit
fi
# powcap.cu (stage "powcap", goal-nv50w): one kind of work alone (shared-memory loads, IMMA, FMA, mixes) and the
# board power and clock at which the card's power cap holds it. POWCAP_SECS a mode (30), POWCAP_MODES (its list).
if has powcap; then
  $N13 -o $W/powcap "$H/powcap.cu" -ldl > $W/powcap.log 2>&1 || say "FAIL powcap build: $(grep -m2 -i error $W/powcap.log | tr '\n' ' ' | cut -c1-300)"
  # One process a mode: a fault ends the CUDA context, and the modes after it would all fail.
  for m in ${POWCAP_MODES:-idle imma immaz immah lds ldsm ffma mix8}; do
    [ -x $W/powcap ] && timeout -k 5 $(( ${POWCAP_SECS:-30} + 60 )) $W/powcap "${POWCAP_SECS:-30}" "$m" 2>&1 | cut -c1-390 | emit
  done
fi
if has product; then
  cd "$R/earn/native/probes" || fail "no probes"
  if [ "${VERIFY_SECS:-150}" != 0 ]; then
    for vc in ${VERIFY_CUBINS:-$(names "$RB") $(names "${ROT_SCHED:-}")}; do
      # goal-rtx50b: "<cubin>+K=V+..." verifies with those settings too; "rel@<variant>" a release fold variant.
      vcub=${vc%%+*}; venv=""; [ "$vcub" != "$vc" ] && venv=$(echo "${vc#*+}" | tr '+' ' ')
      va=$(addon_for "$vcub"); vcub=${vcub%%@*}
      if [ "$vcub" = rel ]; then venv="PEARL_FOLD_ROT=0 $venv"; else venv="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$vcub.cubin $venv"; fi
      # shellcheck disable=SC2086
      j=$(env $venv timeout -k 5 $(( ${VERIFY_SECS:-150} + 120 )) \
        node verify-hits.js "$va" "${VERIFY_SECS:-150}" "${VERIFY_BITS:-233}" 2> $W/verify.err | grep '^{' | tail -1)
      say "verify $vc $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, thLast, fails: (.fails[0:1])}' 2>/dev/null | cut -c1-390)"
      [ -n "$j" ] || say "verify $vc no result: $(tail -2 $W/verify.err | tr '\n' ' ' | cut -c1-300)"
    done
  fi
  n=0
  fw="$(names "$RB") $(names "${ROT_SCHED:-}")"; bw=$(echo $fw | tr ' ' '\n' | tac | tr '\n' ' ')
  for leg in ${HR_LEGS:-rel $fw $bw rel}; do
    n=$((n + 1))
    lv=${leg%%+*}; extra=""
    [ "$lv" != "$leg" ] && extra=$(echo "${leg#*+}" | tr '+' ' ')
    la=$(addon_for "$lv"); lv=${lv%%@*}
    case "$lv" in
      rel) e="PEARL_FOLD_ROT=0" ;;
      *) e="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$lv.cubin" ;;
    esac
    e="$e $extra"
    # shellcheck disable=SC2086
    j=$(env $e timeout -k 5 $(( ${HR_SECS:-120} + 120 )) node hashrate.js "$la" "${HR_SECS:-120}" 10 2> $W/hr$n.err | grep '^{' | tail -1)
    say "hashrate $leg$n $(echo "$j" | jq -c '{k, th, sd, n, mhz, watts, tempC, memUtil, fold: .device.fold}' 2>/dev/null | cut -c1-300) end $(smi)"
    [ -n "$j" ] || say "hashrate $leg$n no result: $(tail -2 $W/hr$n.err | tr '\n' ' ' | cut -c1-300)"
    # goal-rtx50b: the core's own count of the leg's k-order searches and their time on the helper thread.
    grep -o '[0-9]* order searches ([0-9]* ms each' $W/hr$n.err | tail -1 | sed "s/^/hashrate $leg$n search: /" | emit
  done
fi
if has cli; then
  cd "$R/earn" || fail "no earn"
  if [ ! -d /opt/nm/node_modules/tweetnacl ]; then
    npm install --prefix /opt/nm --no-audit --no-fund --loglevel=error tweetnacl tweetnacl-util > $W/npmcli.log 2>&1 \
      || say "WARN npm install: $(tail -1 $W/npmcli.log)"
  fi
  PEARL_JOB_EVERY=${CLI_JOB:-30} node native/probes/verify-pool.js serve 3335 864000 "${CLI_BITS:-206}" > $W/vp.out 2> $W/vp.err &
  VPC=$!
  sleep 5
  say "cli pool $(tail -c 200 $W/vp.err | tr '\n' ' ')"
  n=0
  for leg in ${CLI_LEGS:-}; do
    n=$((n + 1)); kind=${leg%%:*}; cub=${leg#*:}
    if [ "$kind" = h ]; then
      j=$(PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$cub.cubin timeout -k 5 $(( ${CLI_SECS:-180} + 120 )) \
        node native/probes/hashrate.js "$ADDON" "${CLI_SECS:-180}" 10 2> $W/ch$n.err | grep '^{' | tail -1)
      say "clileg h:$cub$n $(echo "$j" | jq -c '{th, sd, mhz, watts, tempC}' 2>/dev/null) end $(smi)"
    elif [ "$kind" = t ] || [ "$kind" = f ] || [ "$kind" = o ]; then
      # goal-nv50t: the files beside the addon (ROT_SHIP=2), no PEARL_ROT_CUBIN, a fresh HOME (so a fresh shape cache).
      sh=""; [ "$leg" = f:w ] && sh="rotation 96 x 512"; [ "$leg" = f:n ] && sh="rotation 192 x 256"
      mkdir -p $W/home$n
      # "o:<n>" (goal-f-rtx50): a "t" leg whose trial starts with the search, as it did until 19 November.
      od=""; [ "$kind" = o ] && od=0
      HOME=$W/home$n PEARL_SHAPE="$sh" PEARL_CORE_PATH=$ADDON PEARL_FOLD_ROT=1 NODE_PATH=/opt/nm/node_modules \
        env ${od:+PEARL_SHAPE_TRIAL_DELAY=$od} ${CLI_ENV:-} timeout -k 5 "${CLI_SECS:-180}" node src/cli/earn-cli.js -a prl1pgate00000000000000000000000000000000000000000000000000000 \
        --mode mining -w clitest --no-report --no-serve --no-update --pool 127.0.0.1:3335 > $W/cli$n.log 2>&1
      lm=$(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9.]+ TH/s · ' | awk 'NR > 60 {s += $1; c++} END {if (c) printf "n=%d ths=%.2f", c, s / c; else printf "n=0"}')
      say "clileg $leg$n $lm acc=$(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9,]+ accepted' | tail -1) end $(smi)"
      # The leg as bench.py scores a row (goal-f-rtx50): every 1 s line from the first (whole), every 15th
      # (s15), the lines under 98.5% of the median (dips, and the share of the run's work they are short by),
      # and the means of each 15 lines in order (ser), which show a trial's two levels.
      med=$(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9.]+ TH/s · ' | sort -n | awk '{a[NR] = $1} END {print a[int((NR + 1) / 2)] + 0}')
      say "clileg $leg$n $(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9.]+ TH/s · ' | awk -v med="$med" '
        {v[NR] = $1; s += $1}
        END {if (!NR) {print "whole n=0"; exit}
             for (i = 15; i <= NR; i += 15) {s15 += v[i]; c15++}
             for (i = 1; i <= NR; i++) if (v[i] < 0.985 * med) {d++; short += med - v[i]}
             printf "whole n=%d ths=%.3f s15=%.3f med=%.2f dips=%d short=%.2f%% ser=", NR, s / NR, c15 ? s15 / c15 : 0, med, d, 100 * short / (med * NR)
             for (i = 1; i + 14 <= NR; i += 15) {t = 0; for (j = i; j < i + 15; j++) t += v[j]; printf "%s%.1f", i > 1 ? "," : "", t / 15}
             print ""}')"
      tr '\r' '\n' < $W/cli$n.log | grep -a -i 'launch shape\|rotation fold' | LC_ALL=C tr -cd ' -~\n' | cut -c1-250 | head -6 \
        | while read -r l; do say "clileg $leg$n log: $l"; done
      # The trial's verdict comes after the lines above, so the head can cut it off (goal-nv50v).
      tr '\r' '\n' < $W/cli$n.log | grep -a 'launch shape trial\|: launch shape ' | LC_ALL=C tr -cd ' -~\n' | cut -c1-250 | head -4 \
        | while read -r l; do say "clileg $leg$n pick: $l"; done
      [ -n "$lm" ] && [ "$lm" != n=0 ] || say "clileg $leg$n log: $(tr '\r\n' '  ' < $W/cli$n.log | LC_ALL=C tr -cd ' -~' | tail -c 300)"
    else
      PEARL_CORE_PATH=$ADDON PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/$cub.cubin NODE_PATH=/opt/nm/node_modules \
        timeout -k 5 "${CLI_SECS:-180}" node src/cli/earn-cli.js -a prl1pgate00000000000000000000000000000000000000000000000000000 \
        --mode mining -w clitest --no-report --no-serve --no-update --pool 127.0.0.1:3335 > $W/cli$n.log 2>&1
      lm=$(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9.]+ TH/s · ' | awk 'NR > 60 {s += $1; c++} END {if (c) printf "n=%d ths=%.2f", c, s / c; else printf "n=0"}')
      say "clileg c:$cub$n $lm acc=$(tr '\r' '\n' < $W/cli$n.log | grep -oE '[0-9,]+ accepted' | tail -1) fold=$(tr '\r' '\n' < $W/cli$n.log | grep -a -i -m1 'rotation fold' | LC_ALL=C tr -cd ' -~' | cut -c1-120) end $(smi)"
      [ -n "$lm" ] && [ "$lm" != n=0 ] || say "clileg c:$cub$n log: $(tr '\r\n' '  ' < $W/cli$n.log | LC_ALL=C tr -cd ' -~' | tail -c 300)"
    fi
  done
  # The stand-in's verdicts on every CLI leg's shares (goal-nv50v: a trial that switches tiles mid-run must not
  # send a wrong one).
  kill -TERM $VPC 2>/dev/null; wait $VPC 2>/dev/null
  say "cli pool end $(tail -n 1 $W/vp.out | jq -c '{jobs, byK, reasons}' 2>/dev/null | cut -c1-300)"
fi
if has bench; then
  # W: a payout address in form only, which is all our CLI checks. The stand-in pool pays nobody.
  W=prl1pgate00000000000000000000000000000000000000000000000000000 ORDER=${ORDER:-ours,ozm,peak,srb} \
    VPOOL=${VPOOL:-210} SECS=${SECS:-180} WARM=${WARM:-60} OURS_SECS=${SECS:-180} VER=${VER:-v0.6.15} \
    bash /root/bench.sh 2>&1 | sed -u 's/^\[bench\]/[nvtest] [bench]/'
fi
[ -n "${SMIQ_PID:-}" ] && kill "$SMIQ_PID" 2>/dev/null
[ -n "${PW_PID:-}" ] && kill "$PW_PID" 2>/dev/null
say "card at the end $(smi)"
say "all done"
exec sleep infinity
