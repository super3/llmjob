#!/bin/bash
# compilers.sh -- the RTX 20 (sm_75) fold built with other compiler options, raced on
# one rented card.
#
#   SRC=<repo checkout> OUT=<scratch dir> bash earn/native/probes/turing/compilers.sh
#
# Every result is one line starting with [compilers], under 500 characters (Vast's log
# API cuts longer lines). STAGES picks the steps: "setup build bench hashrate verify
# complete probe" by default; "setup build" runs without a GPU. ONLY="base rul8" keeps just
# those builds (to confirm a winner on a second card). Assumes an idle GPU 0 and the
# nvidia/cuda:12.8.1-devel-ubuntu22.04 image (CUDA_HOME, default /usr/local/cuda, must
# be CUDA 12.8.1, the toolkit CI builds with, or the toolkit NVCC_VER names: nvtest.sh's
# CUDA_VER=13.3 sets V13.3.73). About 35 minutes on a 2080 Ti.
#
# Why these builds. Built here first (2026-10-20), ptxas from CUDA 12.0, 12.2, 12.4,
# 12.6, 12.8, 12.9, 13.0, 13.3 and 13.4, and cicc from 12.4 through 13.4, all give the
# B-direct fold (pearl_fold_bd.cuh) the same machine code, byte for byte; 12.0 and
# 12.2 differ in one control bit before the loop. So another toolkit cannot change
# the sm_75 fold, and no run here spends GPU time on one. What does change it:
#   base        what ships, first kernel only: CUDA 12.8.1. 254 registers, 680 instructions
#               a chunk loop (512 IMMA).
#   rul8        -Xptxas --register-usage-level=8 (any of 6-10 gives the same code). 252
#               registers, 668 instructions a loop, the chunk barrier 24 IMMA later.
#   cmr255      -maxrregcount=255. ptxas already has 255 from the launch bound; what
#               changes the code is the -maxreg 255 nvcc hands cicc (254 or lower does
#               nothing, nor does __maxnreg__(255) on the kernel). 675 a loop.
#   cmr255rul8  both. 255 registers, 668 a loop.
# Lower register limits spill, and buy nothing: at 40 KB of shared and 256 threads the
# fold runs one block an SM at any register count above 128.
#
# Steps: bench.cu for every build in ROUNDS interleaved rounds (the order reverses each
# round), hashrate.js on the fastest (or HR_BEST) and on base in HR_ROUNDS interleaved rounds,
# verify-hits.js (400 hits) on every build but base, every build's hit list against
# base's (complete, below), then the driver probe: the
# base fold from ptxas 13.4, whose cubin has the CUDA 13 ELF format, and from
# ptxas 12.9, packed into a 12.8 core, to see whether this card's driver loads them.
#
# Another architecture: GC (nvcc's -gencode flags, sm_75's by default) and FOLD (the
# kernel that fold= hashes and the build line's regs= and spill= read,
# pearl_tile_fold_wmma by default) race another card's fold with the same stages, as
# tuning/salad/nvtest.sh does on RTX 30: GC="-gencode arch=compute_86,code=sm_86"
# FOLD=pearl_tile_fold_tall, with VARIANTS_SPEC naming the builds. The second kernel
# (rg=), the patch lists and the driver probe are sm_75's alone; on any other GC the
# build line says rg=none and the probe stage is skipped.
#
# Every bench and hashrate round also reads the memory controller's use, the memory in
# use and the memory clock (nvidia-smi's utilization.memory, memory.used, clocks.mem):
# on Ampere the fold's DRAM traffic costs SM clock at the power limit.
#
# Race (stage "race", not in the default list): every build in one process, on this
# card, taking turns of RACE_LEG seconds for RACE_SECS, with base as the reference
# (probes/race.cu). Separate rounds wander 1.5-2.4% between runs; a race compares builds
# seconds apart on the same clock, heat and power. RACE_AA=1 (the default) adds "aa", a
# copy of base, whose figure is the race's own noise. Each build takes about 1.4 GB at
# k 2048; A' and B' double with k, so on an 8 GB card race k 8192 with RACE_AA=0.
#
# k. A build with -DPEARL_FOLD_K=K in its flags mines only k K (pearl_host_submit refuses
# any other), so every stage runs it at that k: the race as name=lib.so:K, hashrate and
# verify with PEARL_PROFILE_K=K, bench's TH/s and the cycles' share of the peak scaled by
# K. Every rate is MACs a second, so builds at different k compare. Their hit lists do not
# (see "Completeness"). A salt at k 4096 is twice the work, and a race turn ends only on a
# whole salt: on slow cards (RTX 20) use RACE_LEG=8 to 10. A build without the flag runs
# exactly as before, at k 2048. Write the flag just so, once: bench.cu checks that the build
# mines the k read from it, and the build fails if not (-D PEARL_FOLD_K=K, say).
#
# Multicast (stage "multicast", not in the default list): probes/multicast-probe.cu on this card, last of all
# (see that stage below). Needs no build stage.
#
# SMI_EVERY=5 (nvtest.sh sets it; unset, none): while the race and hashrate stages run, a
# line every 5 s with the SM clock, power draw and power limit, so a box can be labelled
# clock-held (drawing under its limit) or power-capped (at it).
set -u
: "${SRC:?set SRC to the repo checkout}" "${OUT:?set OUT to a scratch directory}"
STAGES=" ${STAGES:-setup build bench hashrate verify complete probe} "
ROUNDS=${ROUNDS:-5}; BENCH_SECS=${BENCH_SECS:-20}
HR_ROUNDS=${HR_ROUNDS:-3}; HR_SECS=${HR_SECS:-60}; VERIFY_SECS=${VERIFY_SECS:-90}
RACE_SECS=${RACE_SECS:-600}; RACE_LEG=${RACE_LEG:-4}; RACE_WARM=${RACE_WARM:-1}; RACE_AA=${RACE_AA:-1}
COMPLETE_SHAPES=${COMPLETE_SHAPES:-131072 2048 40;8192 32 300;8192 16 300;4096 16 300}
COMPLETE_SECS=${COMPLETE_SECS:-600}
SMI_EVERY=${SMI_EVERY:-0}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC_VER=${NVCC_VER:-V12.8.93}
NODE_VER=v22.22.0
NATIVE=$SRC/earn/native
T0=$(date +%s)
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
export PATH="$CUDA_HOME/bin:$OUT/node/bin:$PATH"
say() { printf '[compilers] %s\n' "$*" | cut -c1-490; }
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }

# name|nvcc flags|the fold's machine code (md5 of its .text section, first 12 hex)
# under CUDA 12.8.1 from this source, as built without a GPU. A mismatch is reported
# and the run goes on: the comparison still holds, but it is not the code analysed.
# Two kernels. A default build has a second sm_75 kernel, pearl_tile_fold_wmma_rg
# (PEARL_BD_RG_KERNEL in pearl_config.h). The host runs it instead of the first on
# cards with 4 MB of L2 or more, such as the 2060 Super, 2070, 2080 and 2080 Ti.
# fold=, the expect check, the patch and fold.hash are all about the first kernel. A
# build that should run the first kernel on those cards takes -DPEARL_BD_RG_KERNEL=0,
# as the builds below do. A build with a patch list gets it added, so its base needs
# it too (base|-DPEARL_BD_RG_KERNEL=0|5c97fbadf64a). The build line's runs= says which
# kernel runs: fold (the first, on every card), rg (the second, on every card) or
# rg_if_L2>=4MB. The bench stage's "runs fold:" line ends in HASH_RG when the second
# one ran.
VARIANTS=(
  "base|-DPEARL_BD_RG_KERNEL=0|ac306adc8702"
  "rul8|-DPEARL_BD_RG_KERNEL=0 -Xptxas --register-usage-level=8|8fa27fbb65da"
  "cmr255|-DPEARL_BD_RG_KERNEL=0 -maxrregcount=255|a9500fa5f046"
  "cmr255rul8|-DPEARL_BD_RG_KERNEL=0 -maxrregcount=255 -Xptxas --register-usage-level=8|23d035369029"
)
# VARIANTS_SPEC="name|flags;name|flags" races these builds instead, with no expected
# hash: the fold's ablation builds (-DPEARL_ABLATE_*), or a candidate behind a -D
# switch against base. Name the shipped build "base" so the others are measured
# against it. A -D switch most often changes the second kernel too (LAST_RO's is
# e17e255bd8c0), and without -DPEARL_BD_RG_KERNEL=0 that is what 4 MB cards run (see
# "Two kernels" above). An entry may add the expected hash and a 4th field, a patch
# list ("name|flags|expect|sm75/sbfix1.json", relative to this directory): that build's
# sm_75 cubin is edited after ptxas by sm75patch.py (see patch_fold below), for a change
# no source switch can make ptxas emit, such as one control bit.
if [ -n "${VARIANTS_SPEC:-}" ]; then IFS=';' read -r -a VARIANTS <<< "$VARIANTS_SPEC"; fi
if [ -n "${ONLY:-}" ]; then
  keep=()
  for v in "${VARIANTS[@]}"; do case " $ONLY " in *" ${v%%|*} "*) keep+=("$v") ;; esac; done
  VARIANTS=("${keep[@]}")
fi
# Each build's k (see "k" above): -DPEARL_FOLD_K=K in its flags, as amd/test/sweep.sh reads
# it. kof prints a build's k, 2048 without one. kset prints it only past 2048, so a k 2048
# build's commands stay as they were.
declare -A KOF=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name flags _ <<< "$v"
  if [[ " $flags " =~ [[:space:]]-DPEARL_FOLD_K=([0-9]+)u?[[:space:]] ]]; then KOF[$name]=${BASH_REMATCH[1]}; fi
done
kof() { echo "${KOF[$1]:-2048}"; }
kset() { [ "$(kof "$1")" = 2048 ] || kof "$1"; }
GC=${GC:-"-gencode arch=compute_75,code=sm_75"}
SM=$(printf '%s' "$GC" | grep -oE 'sm_[0-9]+' | tail -1 | cut -d_ -f2)
FOLD=${FOLD:-pearl_tile_fold_wmma}
COMMON="-O3 -std=c++17 -cudart static -Xcompiler -fPIC $GC"

# Hash each kernel's machine code in the cubins given: "<kernel> <md5[:12]>".
cat > "$OUT/texthash.py" <<'EOF'
import hashlib, struct, sys
for p in sys.argv[1:]:
    b = open(p, "rb").read()
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    ent, num, stri = struct.unpack_from("<HHH", b, 0x3A)
    hs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * ent) for i in range(num)]
    names = b[hs[stri][4]:hs[stri][4] + hs[stri][5]]
    for h in hs:
        n = names[h[0]:names.index(b"\0", h[0])].decode()
        if n.startswith(".text."):
            print(n[6:], hashlib.md5(b[h[4]:h[4] + h[5]]).hexdigest()[:12])
EOF

fold_hash() {   # $1 = an object, .node or cubin; prints the fold's (FOLD's) code hash
  local d; d=$(mktemp -d "$OUT/x.XXXX")
  case "$1" in
    *.cubin) cp "$1" "$d/" ;;
    *) (cd "$d" && cuobjdump -xelf all "$1" > /dev/null 2>&1) ;;
  esac
  # Every image (the addon also carries pearl_host.cu's); the fold is in one.
  python3 "$OUT/texthash.py" "$d"/*.cubin 2>/dev/null | awk -v f="$FOLD" '$1 == f {print $2; exit}'
  rm -rf "$d"
}
rg_hash() {   # $1 = an object or .node; the second sm_75 kernel's code hash (PEARL_BD_RG_KERNEL)
  local d; d=$(mktemp -d "$OUT/x.XXXX")
  (cd "$d" && cuobjdump -xelf all "$1" > /dev/null 2>&1)
  python3 "$OUT/texthash.py" "$d"/*.cubin 2>/dev/null | awk '$1 == "pearl_tile_fold_wmma_rg" {print $2; exit}'
  rm -rf "$d"
}

if has setup; then
  need=""
  for t in g++ make python3 git curl xz; do command -v $t > /dev/null || need=1; done
  if [ -n "$need" ]; then
    export DEBIAN_FRONTEND=noninteractive
    # Without NVIDIA's CUDA repo, which the image lists: when that server is slow,
    # apt-get waits on it, and nothing needed here comes from it.
    mkdir -p "$OUT/apt-nocuda"
    for f in /etc/apt/sources.list.d/*; do
      [ -f "$f" ] && ! grep -q developer.download.nvidia.com "$f" && cp "$f" "$OUT/apt-nocuda/"
    done
    apt-get update -o Dir::Etc::SourceParts="$OUT/apt-nocuda" > "$OUT/apt.log" 2>&1
    apt-get install -y --no-install-recommends g++ make python3 git curl xz-utils ca-certificates >> "$OUT/apt.log" 2>&1 ||
      { say "FAIL: apt-get: $(tail -1 "$OUT/apt.log")"; exit 1; }
  fi
  if ! node -e 'process.exit(+process.versions.node.split(".")[0] >= 18 ? 0 : 1)' 2> /dev/null; then
    mkdir -p "$OUT/node"
    curl -fsSL "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz" |
      tar -xJ -C "$OUT/node" --strip-components=1 || { say "FAIL: node download"; exit 1; }
  fi
  mkdir -p "$OUT/gyp" && (cd "$OUT/gyp" && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || { say "FAIL: node-gyp setup: $(tail -1 "$OUT/gyp/npm.log")"; exit 1; }
fi

nv=$(nvcc --version 2>/dev/null | grep -oE 'V[0-9.]+' | tail -1)
say "start $(date -u +%FT%TZ) commit $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null) nvcc $nv node $(node --version 2>/dev/null) stages:$STAGES"
# The expected fold hashes above are 12.8.93's, so under another toolkit they cannot match.
if [ "$NVCC_VER" = V12.8.93 ]; then
  [ "$nv" = V12.8.93 ] || say "WARN: nvcc $nv is not CI's 12.8.93; the expected fold hashes will not match"
else
  [ "$nv" = "$NVCC_VER" ] || say "WARN: nvcc $nv is not the $NVCC_VER asked for (NVCC_VER)"
fi
if command -v nvidia-smi > /dev/null && nvidia-smi -i 0 > /dev/null 2>&1; then
  say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,memory.total --format=csv,noheader)"
  q=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,utilization.gpu,utilization.memory,memory.used
  say "card now $(nvidia-smi -i 0 --query-gpu=$q,clocks_event_reasons.active --format=csv,noheader 2> /dev/null ||
    nvidia-smi -i 0 --query-gpu=$q --format=csv,noheader) gpus=$(nvidia-smi -L | wc -l)"
  DRIVER=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader | tr -d ' ')
else
  say "no GPU visible"; DRIVER=none
fi

# A build with a patch list. pearl_kernel.cu was compiled with --keep, so nvcc left the
# sm_75 cubin ptxas wrote on disk. sm75patch.py edits it in place, and refuses unless the
# fold is exactly the code the list was made from. Then nvcc's own fatbinary and host
# compile steps, read from --dryrun of the same command, run again to pack it into
# pearl_kernel.o: the driver probe's replay (below), so nothing else in the object
# changes. The cubin as ptxas wrote it is kept as ptxas.cubin, to compare.
patch_fold() {   # K flags list name
  local K=$1 flags=$2 list=$3 name=$4 o=$1/cuda-build/pearl_kernel.o fb host cubin
  # shellcheck disable=SC2086
  nvcc $COMMON $flags --keep --keep-dir "$K/keep" -Xptxas -v --dryrun -c "$K/src/pearl_kernel.cu" -o "$o" > "$K/dryrun.txt" 2>&1 ||
    { say "build $name FAIL: patch: nvcc --dryrun: $(tail -1 "$K/dryrun.txt")"; return 1; }
  fb=$(grep '^#\$ fatbinary ' "$K/dryrun.txt" | sed 's/^#\$ //')
  host=$(tail -n 1 "$K/dryrun.txt" | sed 's/^#\$ //; s/[[:space:]]*$//')
  cubin=$(printf '%s\n' "$fb" | grep -oE 'sm=75,file=[^"]+' | sed 's/^sm=75,file=//')
  # One fatbinary step holding one sm_75 image, and a last step that writes this object.
  if [ "$(printf '%s\n' "$fb" | grep -c .)" != 1 ] || [ "$(printf '%s\n' "$cubin" | grep -c .)" != 1 ] ||
    [ ! -f "$cubin" ] || [[ "$host" != *" -c "*" -o \"$o\"" ]]; then
    say "build $name FAIL: patch: $K/dryrun.txt does not show one sm_75 cubin and the host step"; return 1
  fi
  cp "$cubin" "$K/ptxas.cubin"
  python3 -I "$NATIVE/probes/turing/sm75patch.py" "$cubin" "$list" > "$K/patch.txt" 2>&1 ||
    { say "build $name FAIL: patch: $(tail -1 "$K/patch.txt")"; return 1; }
  (eval "$fb" && eval "$host") > "$K/replay.txt" 2>&1 ||
    { say "build $name FAIL: patch: replay: $(tail -1 "$K/replay.txt")"; return 1; }
}

# One build: pearl_kernel.o for sm_75 with the flags, the addon around it, and bench.
build_variant() {   # name flags expect [patch list]
  local name=$1 flags=$2 expect=$3 patch=${4:-} K=$OUT/v/$1 list="" after="" keep=()
  rm -rf "$K"; mkdir -p "$K/cuda-build"
  cp -r "$NATIVE/src" "$NATIVE/binding.gyp" "$K/"
  if [ -n "$patch" ]; then
    list=$NATIVE/probes/turing/$patch
    after=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["after"])' "$list" 2> /dev/null) ||
      { say "build $name FAIL: patch: cannot read the patch list probes/turing/$patch"; return 1; }
    keep=(--keep --keep-dir "$K/keep"); mkdir -p "$K/keep"
  fi
  # sm75patch.py edits only the first kernel. Without this, cards with 4 MB of L2 or
  # more would run the second, unpatched one. Left alone if the entry sets the policy.
  if [ -n "$patch" ] && [[ "$flags" != *PEARL_BD_RG_KERNEL=* ]]; then
    flags="${flags:+$flags }-DPEARL_BD_RG_KERNEL=0"
  fi
  # shellcheck disable=SC2086
  if ! nvcc $COMMON $flags "${keep[@]}" -Xptxas -v -c "$K/src/pearl_kernel.cu" -o "$K/cuda-build/pearl_kernel.o" > "$K/ptxas.txt" 2>&1; then
    say "build $name FAIL: $(grep -m2 -iE 'error' "$K/ptxas.txt" | tr '\n' ' ')"; return 1
  fi
  if [ -n "$patch" ]; then patch_fold "$K" "$flags" "$list" "$name" || return 1; fi
  # The host with the same flags: a switch such as PEARL_BD_A_DIRECT changes what the
  # host writes as well as what the fold reads, and the two must agree.
  # shellcheck disable=SC2086
  if ! nvcc $COMMON $flags -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o" > "$K/host.txt" 2>&1; then
    say "build $name FAIL: host: $(grep -m2 -iE 'error' "$K/host.txt" | tr '\n' ' ')"; return 1
  fi
  ar rcs "$K/cuda-build/libpearl_cuda.a" "$K/cuda-build/pearl_kernel.o" "$K/cuda-build/pearl_host.o"
  (cd "$K" && ln -s "$OUT/gyp/node_modules" node_modules && echo '{}' > package.json &&
    CUDA_PATH=$CUDA_HOME ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) ||
    { say "build $name FAIL: node-gyp: $(grep -m2 -iE 'error' "$K/gyp.log" | tr '\n' ' ')"; return 1; }
  # bench.cu with the build's flags too: -DPEARL_TALL_CYC has it print the fold's cycles a tile.
  # PEARL_BENCH_K: the build fails unless it mines the k kof read from its flags (see "k").
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -cudart static $GC $flags -DPEARL_BENCH_K="$(kof "$name")" -I"$K/src" "$NATIVE/probes/bench.cu" \
    "$K/cuda-build/pearl_kernel.o" "$K/cuda-build/pearl_host.o" -o "$K/bench" > "$K/bench-build.txt" 2>&1 ||
    { say "build $name FAIL: bench: $(grep -m1 -i error "$K/bench-build.txt")"; return 1; }
  # The same two objects as a library for the race stage. -Bsymbolic: its calls stay
  # inside it when other builds with the same symbols are loaded beside it.
  nvcc -shared -cudart static -Xlinker -Bsymbolic -o "$K/libpearl.so" "$K/cuda-build/pearl_kernel.o" \
    "$K/cuda-build/pearl_host.o" > "$K/so-build.txt" 2>&1 ||
    { rm -f "$K/libpearl.so"; say "build $name WARN: libpearl.so: $(grep -m1 -i error "$K/so-build.txt")"; }
  local st h rst rg="" runs
  st=$(awk "/Compiling entry function .$FOLD. for .sm_$SM./{f=1;next} f&&/Compiling entry/{f=0} f" "$K/ptxas.txt" | tr '\n' ' ')
  # The second sm_75 kernel, which the host runs on cards with 4 MB of L2 or more
  # (PEARL_BD_RG_KERNEL): its code hash, registers and spill, or rg=none in a build
  # without it. The patch, the expect check and fold.hash stay on the first.
  rst=$(awk '/Compiling entry function .pearl_tile_fold_wmma_rg. for .sm_75./{f=1;next} f&&/Compiling entry/{f=0} f' "$K/ptxas.txt" | tr '\n' ' ')
  [ "$SM" = 75 ] && rg=$(rg_hash "$K/build/Release/pearl_core.node")
  if [ -n "$rg" ]; then
    rg="rg=$rg rg_regs=$(echo "$rst" | grep -oE 'Used [0-9]+' | grep -oE '[0-9]+') rg_spill=$(echo "$rst" | grep -oE '[0-9]+ bytes spill stores' | grep -oE '^[0-9]+')/$(echo "$rst" | grep -oE '[0-9]+ bytes spill loads' | grep -oE '^[0-9]+')"
  else
    rg="rg=none"
  fi
  # Which kernel runs (see "Two kernels" above): the policy is 2 unless the flags set it.
  case " $flags " in
    *PEARL_BD_RG_KERNEL=0*) runs=fold ;;
    *PEARL_BD_RG_KERNEL=1*) runs=rg ;;
    *) runs='rg_if_L2>=4MB' ;;
  esac
  [ "$rg" = rg=none ] && runs=fold
  h=$(fold_hash "$K/build/Release/pearl_core.node")
  echo "$h" > "$K/fold.hash"
  # The addon and bench must carry the patched fold: a build that lost the patch is base.
  if [ -n "$patch" ] && { [ "$h" != "$after" ] || [ "$(fold_hash "$K/bench")" != "$after" ]; }; then
    say "build $name FAIL: patch: the addon's fold is ${h:-unreadable}, bench's $(fold_hash "$K/bench"), not the patch list's $after"; return 1
  fi
  say "build $name regs=$(echo "$st" | grep -oE 'Used [0-9]+' | grep -oE '[0-9]+') spill=$(echo "$st" | grep -oE '[0-9]+ bytes spill stores' | grep -oE '^[0-9]+')/$(echo "$st" | grep -oE '[0-9]+ bytes spill loads' | grep -oE '^[0-9]+') $rg runs=$runs fold=$h $([ -z "$expect" ] && echo "expect=-" || { [ "$h" = "$expect" ] && echo "expect=$expect match" || echo "expect=$expect MISMATCH"; })$([ -n "$patch" ] && echo " patch=[$patch, $(grep -oE '[0-9]+ words' "$K/patch.txt")]") flags=[$flags]"
  touch "$K/ok"
}

if has build; then
  mkdir -p "$OUT/host"
  nvcc $COMMON -c "$NATIVE/src/pearl_host.cu" -o "$OUT/host/pearl_host.o" > "$OUT/host/log" 2>&1 ||
    { say "FAIL: pearl_host.cu: $(grep -m1 -i error "$OUT/host/log")"; exit 1; }
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r name flags expect patch <<< "$v"
    build_variant "$name" "$flags" "$expect" "$patch"
  done
fi

BUILT=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name _ _ <<< "$v"
  [ -f "$OUT/v/$name/ok" ] && BUILT+=("$name")
done

# The card's SM clock, power draw and power limit every SMI_EVERY seconds while a stage runs
# (see SMI_EVERY at the top), as "smi <stage> 1845 MHz 349.12/350.00 W". Off at 0. A stop
# lets a line being written finish, so none comes after the stage, and leaves no sleep behind.
# The query runs in the background like the sleep, so a stop ends at once even when
# nvidia-smi hangs (bash holds a trap until a foreground command returns).
smi_start() {   # $1 = the stage
  SMI_PID=""
  [ "${SMI_EVERY:-0}" != 0 ] || return 0
  (
    trap '[ -z "${!:-}" ] || kill "$!" 2> /dev/null; exit 0' TERM
    while :; do
      nvidia-smi -i 0 --query-gpu=clocks.sm,power.draw,power.limit --format=csv,noheader,nounits > "$OUT/smi.$1" 2> /dev/null &
      wait $!
      say "smi $1 $(head -1 "$OUT/smi.$1" | awk -F', *' '{printf "%s MHz %s/%s W", $1, $2, $3}')"
      sleep "$SMI_EVERY" > /dev/null & wait $! || break
    done
  ) &
  SMI_PID=$!
}
smi_stop() { [ -z "${SMI_PID:-}" ] || { kill "$SMI_PID" 2> /dev/null; wait "$SMI_PID" 2> /dev/null; }; SMI_PID=""; }

# The race (see "Race" at the top): base first, then aa (a copy of base's library under
# another path, which dlopen loads again), then the other builds. A build past k 2048 is
# named lib.so:K (see "k" at the top).
if has race && [ -f "$OUT/v/base/libpearl.so" ]; then
  if ! nvcc -O2 -std=c++17 -I"$NATIVE/src" "$NATIVE/probes/race.cu" -o "$OUT/race" -ldl -lpthread > "$OUT/race-build.txt" 2>&1; then
    say "race FAIL: build: $(grep -m1 -i error "$OUT/race-build.txt")"
  else
    k=$(kset base)
    arms=("base=$OUT/v/base/libpearl.so${k:+:$k}")
    if [ "$RACE_AA" = 1 ]; then
      cp "$OUT/v/base/libpearl.so" "$OUT/v/base/libpearl-aa.so" && arms+=("aa=$OUT/v/base/libpearl-aa.so${k:+:$k}")
    fi
    for name in "${BUILT[@]}"; do
      [ "$name" = base ] && continue
      k=$(kset "$name")
      if [ -f "$OUT/v/$name/libpearl.so" ]; then arms+=("$name=$OUT/v/$name/libpearl.so${k:+:$k}")
      else say "race: $name has no libpearl.so, left out"; fi
    done
    say "race: ${#arms[@]} builds, ${RACE_SECS} s, turns of ${RACE_LEG} s after ${RACE_WARM} s warm"
    smi_start race
    timeout -k 10 $((RACE_SECS + 600)) "$OUT/race" "$RACE_SECS" "$RACE_LEG" "$RACE_WARM" "${arms[@]}" 2>&1 |
      tee "$OUT/race.log" | while IFS= read -r l; do say "race $l"; done
    smi_stop
  fi
fi

# Median SM clock, power and temperature of the samples taken while the timed part ran,
# and the mean SM clock: these cards clock in 15 MHz steps, so the median of a round's
# 18 samples moves a whole step at a time, and the mean resolves finer. Then the medians
# of the memory controller's use (%), the memory in use (MiB) and the memory clock, or
# "-" where the card gave none.
cat > "$OUT/med.py" <<'EOF'
import statistics, sys
f, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows, mem = [], []
for line in open(f):
    p = [x.strip() for x in line.split(",")]
    try:
        t = float(p[0]); v = [float(x) for x in p[1:4]]
    except Exception:
        continue
    if t0 <= t <= t1:
        rows.append(v)
        m = []
        for x in p[4:7]:
            try:
                m.append(float(x))
            except Exception:
                m.append(None)
        mem.append(m + [None] * (3 - len(m)))
if not rows:
    print("0 0 0 0 0 - - -"); sys.exit()
def mm(i):
    xs = [m[i] for m in mem if m[i] is not None]
    return round(statistics.median(xs)) if xs else "-"
print(len(rows), *[round(statistics.median(r[i] for r in rows), 1) for i in range(3)],
      round(statistics.mean(r[0] for r in rows), 1), mm(0), mm(1), mm(2))
EOF
sampler() {   # $1 = file; one sample a second until killed
  while :; do
    printf '%s,%s\n' "$(date +%s.%N)" "$(nvidia-smi -i 0 --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.memory,memory.used,clocks.mem --format=csv,noheader,nounits)" >> "$1"
    sleep 1
  done
}

if has bench && [ ${#BUILT[@]} -gt 0 ]; then
  : > "$OUT/bench.tsv"
  for r in $(seq 1 "$ROUNDS"); do
    order=("${BUILT[@]}")
    if [ $((r % 2)) = 0 ]; then mapfile -t order < <(printf '%s\n' "${BUILT[@]}" | tac); fi
    for name in "${order[@]}"; do
      K=$OUT/v/$name; S=$K/smi.r$r; : > "$S"
      sampler "$S" & sp=$!
      "$K/bench" "$BENCH_SECS" > "$K/bench.r$r" 2>&1
      t1=$(date +%s.%N); kill $sp 2> /dev/null; wait $sp 2> /dev/null
      line=$(grep -E 'attempts=.*TH/s' "$K/bench.r$r" | tail -1)
      att=$(echo "$line" | grep -oE 'attempts=[0-9]+' | cut -d= -f2)
      secs=$(echo "$line" | grep -oE '[0-9.]+ s ' | grep -oE '[0-9.]+')
      if [ -z "$att" ] || [ -z "$secs" ]; then
        say "bench r$r $name FAIL: $(tail -2 "$K/bench.r$r" | tr '\n' ' ')"; continue
      fi
      # MACs a second: an attempt is a region, 256 x k MACs at the build's k.
      th=$(python3 -c "print(round($att * 256 * $(kof "$name") / $secs / 1e12, 3))")
      read -r n mhz w c mean mc mu mclk < <(python3 "$OUT/med.py" "$S" "$(python3 -c "print($t1 - $secs + 1)")" "$t1")
      pg=$(python3 -c "print(round($th / ($mhz / 1000), 3) if $mhz else 0)")
      # A -DPEARL_TALL_CYC build's cycles a tile (bench.cu prints them), or "-".
      cyc=$(grep -oE 'cycles a tile [0-9]+' "$K/bench.r$r" | tail -1 | grep -oE '[0-9]+$')
      # Column 13, for a build past k 2048 only: its k, which scales sum.py's share of the peak.
      kcol="" k=$(kset "$name"); [ -z "$k" ] || kcol=$'\t'$k
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s%s\n' "$r" "$name" "$th" "$mhz" "$w" "$c" "$pg" "$mean" "$mc" "$mu" "$mclk" "${cyc:--}" "$kcol" >> "$OUT/bench.tsv"
      [ "$r" = 1 ] && say "bench $name runs $(grep -oE 'fold: [^|]*' "$K/bench.r$r" | head -1)"
      say "bench r$r $name $th TH/s ${mhz} MHz (mean $mean) ${w} W ${c} C $pg TH/s/GHz mem ${mc}% ${mu} MiB ${mclk} MHz${cyc:+ cyc $cyc}${k:+ k $k} (n=$n)"
    done
  done
  # Per build: medians over rounds, and against base in the same round. The mean clock
  # beside the median is the mean of the rounds' mean clocks: bench.tsv's 8th column, or
  # hashrate.tsv's clock, which hashrate.js already averages. A round with no clock
  # reading writes 0, which the mean leaves out: one such round would pull it far down.
  # Columns 9 to 11, where there are any, are the memory readings: the medians of the
  # rounds' memory-controller use and memory in use ("-" where a round had none). Column
  # 12, a -DPEARL_TALL_CYC build's cycles a tile: their median, and its share of the peak,
  # where a tile takes 48 x k cycles (98304 at k 2048, as bench.cu counts it). Column 13, a
  # build's k where it is past 2048.
  cat > "$OUT/sum.py" <<'EOF'
import statistics as st, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
tag = sys.argv[2]
by = {}
for r in rows:
    by.setdefault(r[1], {})[int(r[0])] = (float(r[2]), float(r[3]), float(r[4]), float(r[6]),
                                         float(r[7]) if len(r) > 7 else float(r[3]))
base = by.get("base", {})
best, bestv = None, None
def num(x):
    try:
        return float(x)
    except ValueError:
        return None
mem, cyc, kof = {}, {}, {}
for r in rows:
    if len(r) > 9:
        mem.setdefault(r[1], []).append((num(r[8]), num(r[9])))
    if len(r) > 11 and num(r[11]):
        cyc.setdefault(r[1], []).append(num(r[11]))
    if len(r) > 12 and num(r[12]):
        kof[r[1]] = int(num(r[12]))
def memtxt(name):
    ms = mem.get(name) or []
    mc = [m[0] for m in ms if m[0] is not None]; mu = [m[1] for m in ms if m[1] is not None]
    t = ""
    if mc or mu:
        t = (f" mem {st.median(mc):.0f}%" if mc else " mem -%") + (f" {st.median(mu):.0f} MiB" if mu else "")
    if cyc.get(name):
        t += f" cyc {st.median(cyc[name]):.0f} ({48 * kof.get(name, 2048) / st.median(cyc[name]) * 100:.2f}%)"
    return t
for name, d in by.items():
    th = [v[0] for v in d.values()]; mhz = [v[1] for v in d.values()]; w = [v[2] for v in d.values()]
    pg = [v[3] for v in d.values()]; clk = [v[4] for v in d.values() if v[4]] or [0]
    raw =[d[k][0] / base[k][0] - 1 for k in d if k in base and name != "base"]
    per = [d[k][3] / base[k][3] - 1 for k in d if k in base and name != "base" and base[k][3]]
    vs = ""
    if raw:
        vs = (f" vs base: raw {st.median(raw) * 100:+.2f}% per-GHz {st.median(per) * 100:+.2f}% "
              f"(rounds {' '.join(f'{x * 100:+.1f}' for x in raw)})")
        if bestv is None or st.median(raw) > bestv:
            best, bestv = name, st.median(raw)
    print(f"[compilers] {tag} {name} n={len(th)} {st.median(th):.3f} TH/s ({min(th):.2f}-{max(th):.2f}) "
          f"{st.median(mhz):.0f} MHz (mean {st.mean(clk):.0f}) {st.median(w):.0f} W "
          f"{st.median(pg):.3f} TH/s/GHz{memtxt(name)}{vs}"[:490])
if best:
    open(sys.argv[3], "w").write(best)
EOF
  python3 "$OUT/sum.py" "$OUT/bench.tsv" bench-sum "$OUT/best"
fi

# HR_BEST names the build to race against base instead (when the fastest in the bench
# is an ablation, whose rate means nothing to the miner).
if has hashrate && { [ -n "${HR_BEST:-}" ] || [ -f "$OUT/best" ]; } && [ -f "$OUT/v/base/ok" ]; then
  best=${HR_BEST:-$(cat "$OUT/best")}
  say "hashrate: base against $best, $HR_ROUNDS rounds of ${HR_SECS} s"
  : > "$OUT/hashrate.tsv"
  smi_start hashrate
  for r in $(seq 1 "$HR_ROUNDS"); do
    pair=(base "$best"); [ $((r % 2)) = 0 ] && pair=("$best" base)
    for name in "${pair[@]}"; do
      # A build past k 2048 at its own k (see "k" at the top); hashrate.js then skips cardK.
      k=$(kset "$name")
      j=$(cd "$SRC/earn" && timeout -k 10 $((HR_SECS + 120)) ${k:+env PEARL_PROFILE_K=$k} node native/probes/hashrate.js "$OUT/v/$name/build/Release/pearl_core.node" "$HR_SECS" 10 2>&1 | grep '^{' | tail -1)
      th=$(echo "$j" | grep -oE '"th":[0-9.]+' | cut -d: -f2)
      mhz=$(echo "$j" | grep -oE '"mhz":[0-9.]+' | cut -d: -f2)
      w=$(echo "$j" | grep -oE '"watts":[0-9.]+' | cut -d: -f2)
      c=$(echo "$j" | grep -oE '"tempC":[0-9.]+' | cut -d: -f2)
      mc=$(echo "$j" | grep -oE '"memUtil":[0-9.]+' | cut -d: -f2)
      mu=$(echo "$j" | grep -oE '"memUsedMb":[0-9.]+' | cut -d: -f2)
      mclk=$(echo "$j" | grep -oE '"memMhz":[0-9.]+' | cut -d: -f2)
      mhz=${mhz:-0}
      if [ -z "$th" ]; then say "hashrate r$r $name FAIL: $j"; continue; fi
      pg=$(python3 -c "print(round($th / ($mhz / 1000), 3) if '$mhz' and $mhz else 0)")
      # The 8th column is the clock again (hashrate.js's is already a mean), so the memory
      # readings sit where bench.tsv's do.
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$name" "$th" "$mhz" "$w" "$c" "$pg" "$mhz" "${mc:--}" "${mu:--}" "${mclk:--}" >> "$OUT/hashrate.tsv"
      say "hashrate r$r $name $th TH/s sd=$(echo "$j" | grep -oE '"sd":[0-9.]+' | cut -d: -f2) ${mhz} MHz ${w} W ${c} C $pg TH/s/GHz mem ${mc:--}% ${mu:--} MiB ${mclk:--} MHz${k:+ k $k}"
    done
  done
  smi_stop
  python3 "$OUT/sum.py" "$OUT/hashrate.tsv" hashrate-sum "$OUT/best-hr"
fi

if has verify; then
  for name in "${BUILT[@]}"; do
    [ "$name" = base ] && continue
    k=$(kset "$name")
    out=$(cd "$SRC/earn" && timeout -k 10 $((VERIFY_SECS + 120)) ${k:+env PEARL_PROFILE_K=$k} node native/probes/verify-hits.js "$OUT/v/$name/build/Release/pearl_core.node" "$VERIFY_SECS" 2>&1)
    j=$(echo "$out" | grep '^{' | tail -1)
    if [ -z "$j" ]; then say "verify $name FAIL: no result: $(echo "$out" | tail -3 | tr '\n' ' ')"; continue; fi
    say "verify $name${k:+ k $k} $(echo "$j" | grep -oE '"(hits|verified|failed|salts|PASS)":[a-z0-9]+' | tr '\n' ' ')$(echo "$j" | grep -q '"PASS":true' || echo " FAILS: $(echo "$j" | grep -oE '"fails":.*' | cut -c1-300)")"
  done
fi

# Completeness: every build must find exactly base's hits. The fold reports a hit only
# after it recomputes the whole hash from the transcript, so a build whose filter misses
# (PEARL_BD_HASH_SPLIT reading a stale state, say) drops hits and reports no wrong one:
# verify-hits.js passes it, and the bench reads the lost work as speed. bench's
# PEARL_HITLIST lists each batch's hits (16 to 32 on average) on the same job and nonces,
# in each of COMPLETE_SHAPES ("m col_batch batches", ';' between): the default shape,
# where a block walks hundreds of tiles, and batches of 128, 64 and 32 tiles, where a
# block has one, two or a few on any RTX 20 card (30 to 72 blocks): a persistent
# block's first and last hand-offs. Each build's lines must match base's, batch for
# batch; an ablation build (output meaningless) fails. A run stops after COMPLETE_SECS.
# The lines only line up between builds that search the same batch width: on RTX 30 a
# shape's col_batch at or below every build's PEARL_AMPERE_COL_BATCH (the narrower
# wins), such as "131072 128 160" for builds at 128 and 512. A wider shape fails a
# narrower build that finds the same hits (31 October, RTX 3090).
# Nor do they line up across k: k is in config52, so the job key differs, and the operands
# are m x k, so every hash and every hit differs. Each build is held to the first build of
# its own k (base, for k 2048), which is that k's reference. A k with one build has nothing
# to match; verify-hits.js, at its k, is its check.
if has complete && [ -f "$OUT/v/base/ok" ]; then
  IFS=';' read -r -a shapes <<< "$COMPLETE_SHAPES"
  order=(base)
  for name in "${BUILT[@]}"; do [ "$name" = base ] || order+=("$name"); done
  declare -A REF=()
  for name in "${order[@]}"; do k=$(kof "$name"); [ -n "${REF[$k]:-}" ] || REF[$k]=$name; done
  for sh in "${shapes[@]}"; do
    read -r m cb nb <<< "$sh"
    tag=m$m-cb$cb
    for name in "${order[@]}"; do
      f=$OUT/v/$name/hitlist.$tag; k=$(kof "$name"); B=$OUT/v/${REF[$k]}/hitlist.$tag
      PEARL_HITLIST=$nb timeout -k 10 "$COMPLETE_SECS" "$OUT/v/$name/bench" 0 "$m" "$cb" > "$f" 2>&1
      sum=$(grep '^hitlist ' "$f" | tail -1)
      if [ -z "$sum" ] || [[ "$sum" == *FAIL* ]]; then
        say "complete $tag $name FAIL: $(tail -2 "$f" | tr '\n' ' ')"; continue
      fi
      if [ "$name" = base ]; then say "complete $tag base ${sum#hitlist }"; continue; fi
      if [ "$name" = "${REF[$k]}" ]; then say "complete $tag $name ${sum#hitlist } (the reference for k $k)"; continue; fi
      vs=""; [ "${REF[$k]}" = base ] || vs=" against ${REF[$k]}"
      if cmp -s <(grep '^HL ' "$B") <(grep '^HL ' "$f"); then
        say "complete $tag $name PASS$vs ${sum#hitlist }"
      else
        say "complete $tag $name FAIL$vs ${sum#hitlist }, first difference: $(diff <(grep '^HL ' "$B") <(grep '^HL ' "$f") | grep -m2 '^[<>]' | tr '\n' ' ')"
      fi
    done
  done
fi

# Driver probe: does this driver load an sm_75 cubin from a newer ptxas inside the 12.8
# core? 13.x cubins use a new ELF format (OS ABI 0x41, ABI version 8, the architecture
# moved within e_flags); 12.9's keep the old one. The fold's code is the same either way.
if has probe && [ -f "$OUT/v/base/ok" ] && [ "$SM" = 75 ]; then
  P=$OUT/probe; mkdir -p "$P/keep"
  R=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64
  nvcc $COMMON --keep --keep-dir "$P/keep" -c "$NATIVE/src/pearl_kernel.cu" -o "$P/pearl_kernel.o" > "$P/log" 2>&1 &&
    nvcc $COMMON --keep --keep-dir "$P/keep" --dryrun -c "$NATIVE/src/pearl_kernel.cu" -o "$P/pearl_kernel.o" > "$P/dryrun.txt" 2>&1 ||
    say "probe FAIL: keep build: $(grep -m1 -i error "$P/log")"
  for spec in "13.4|cuda_nvcc-linux-x86_64-13.4.92-archive.tar.xz|60998f40cc9df5826b2a846e2dcc328133fbc5ef539398ce0a21cf5dcac7193d" \
              "12.9|cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz|7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"; do
    IFS='|' read -r ver file sha <<< "$spec"
    D=$P/$ver; mkdir -p "$D"
    if ! { curl -fsSL "$R/$file" -o "$D/a.tar.xz" && echo "$sha  $D/a.tar.xz" | sha256sum -c --quiet - &&
           tar -xJf "$D/a.tar.xz" -C "$D" --strip-components=2 --wildcards '*/bin/ptxas'; }; then
      say "probe ptxas $ver FAIL: download"; continue
    fi
    rm -f "$D/a.tar.xz"
    ptx=$(grep -oE '"[^"]*\.ptx"' "$P/dryrun.txt" | grep -v compute_ | head -1 | tr -d '"')
    [ -n "$ptx" ] || ptx=$(ls "$P"/keep/*.ptx | head -1)
    "$D/ptxas" -arch=sm_75 -m64 "$ptx" -o "$D/sm75.cubin" > "$D/ptxas.log" 2>&1 || { say "probe ptxas $ver FAIL: $(head -1 "$D/ptxas.log")"; continue; }
    abi=$(python3 -c "b=open('$D/sm75.cubin','rb').read(); print(f'osabi=0x{b[7]:02x} abi={b[8]}')")
    fh=$(python3 "$OUT/texthash.py" "$D/sm75.cubin" | awk '$1 == "pearl_tile_fold_wmma" {print $2}')
    K=$D/core; mkdir -p "$K/cuda-build"; cp -r "$NATIVE/src" "$NATIVE/binding.gyp" "$K/"
    # The fatbinary and host-compile steps of the same compile, with this sm_75 cubin.
    fb=$(grep '^#\$ fatbinary ' "$P/dryrun.txt" | sed 's/^#\$ //')
    host=$(tail -n 1 "$P/dryrun.txt" | sed 's/^#\$ //; s/[[:space:]]*$//')
    orig=$(printf '%s\n' "$fb" | grep -oE 'sm=75,file=[^"]+' | sed 's/^sm=75,file=//')
    cp "$orig" "$orig.nvcc" && cp "$D/sm75.cubin" "$orig"
    ok=1; (eval "$fb" && eval "$host") > "$D/swap.log" 2>&1 || ok=0
    mv "$orig.nvcc" "$orig"
    [ $ok = 1 ] || { say "probe ptxas $ver FAIL: swap: $(tail -1 "$D/swap.log")"; continue; }
    cp "$P/pearl_kernel.o" "$K/cuda-build/pearl_kernel.o"
    ar rcs "$K/cuda-build/libpearl_cuda.a" "$K/cuda-build/pearl_kernel.o" "$OUT/host/pearl_host.o"
    (cd "$K" && ln -s "$OUT/gyp/node_modules" node_modules && echo '{}' > package.json &&
      CUDA_PATH=$CUDA_HOME ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || { say "probe ptxas $ver FAIL: node-gyp"; continue; }
    inside=$(fold_hash "$K/build/Release/pearl_core.node")
    # The first kernel on every card, the one fold= and in-node= hash.
    out=$(cd "$SRC/earn" && PEARL_BD_SWEEP_HASH_RG=0 timeout -k 10 120 node native/probes/hashrate.js "$K/build/Release/pearl_core.node" 20 5 2>&1 | tr '\n' ' ')
    th=$(echo "$out" | grep -oE '"th":[0-9.]+' | cut -d: -f2)
    if [ -n "$th" ]; then res="loads, $th TH/s"
    else res="does not run: $(echo "$out" | grep -oE '(error|Error)[^{}]{0,200}' | head -1 | sed 's/ \{2,\}at .*//')"; fi
    say "probe ptxas $ver cubin $abi fold=$fh in-node=$inside driver $DRIVER: $res"
  done
fi

# Multicast probe (stage "multicast", not in the default list; sm_90 and newer): probes/multicast-probe.cu,
# the TMA fold's B' feed in clusters of two without the mma, which prices what TMA multicast saves and shows
# whether the cluster build's barriers hang (see its header). Its own build, with GC; MC_SECS seconds an arm a
# round (6), MC_ROUNDS rounds (3), MC_PACE cycles a stage (3240), under a timeout of MC_TIMEOUT seconds (900).
# Every line it prints goes out as "multicast ...", then multicast-read.js's reading. About 4 minutes. Last,
# so a card that the probe leaves stuck has raced already.
if has multicast; then
  if [ "${SM:-0}" -lt 90 ]; then
    say "multicast: skipped, sm_$SM has no clusters"
  elif ! nvcc -O3 -std=c++17 $GC "$NATIVE/probes/multicast-probe.cu" -o "$OUT/multicast-probe" -ldl -lpthread \
      > "$OUT/multicast-build.txt" 2>&1; then
    say "multicast FAIL: build: $(grep -m2 -iE 'error' "$OUT/multicast-build.txt" | tr '\n' ' ')"
  else
    timeout -k 10 "${MC_TIMEOUT:-900}" "$OUT/multicast-probe" "${MC_SECS:-6}" "${MC_ROUNDS:-3}" "${MC_PACE:-3240}" 2>&1 |
      tee "$OUT/multicast.log" | while IFS= read -r l; do say "multicast $l"; done
    say "multicast exit ${PIPESTATUS[0]}"
    node "$NATIVE/probes/multicast-read.js" "$OUT/multicast.log" 2>&1 | while IFS= read -r l; do say "multicast $l"; done
  fi
fi
say "done in $(( ($(date +%s) - T0) / 60 )) min"
