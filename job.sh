# job 1: SRBMiner 3.8.6 with wget and a terminal, then the duel, 2 turns each and a clock turn
G_PR=5d2b012eaba6e514a0a543c985ded658d025bd73
G_CODE=f9ddc2965a1b6ff53a53cbcf113d114d59aa3348
G_PARTS=srb\ duel
G_TURNS=2
G_DSECS=300
G_CLKSECS=180
G_SRB_ALSO=3.8.2
cd /opt/llmjob/earn/native/amd || exit 1
# goal/amd job body (PR #290, 10 November). The box agent (agent.sh) runs it from earn/native/amd; the job's header
# sets G_*. Parts (G_PARTS), in this order:
#   srb     pick the SRBMiner version: 3.8.6 if it mines in this image (plain, then with a terminal), else 3.8.2
#   duel    ours (the CLI on the PR head's core, built here from G_PR) against SRBMiner on Kryptex, taking turns,
#           G_TURNS turns each of G_DSECS, every turn scored the same way: the mean of its 10 s samples after the
#           first 60 s. Then one turn each of G_CLKSECS with the clock probe beside the miner.
#   build   G_CODE checked out, and sweep.sh builds and checks the builds in G_SWEEP (no timing of its own)
#   fold    the fold alone at the launch shapes in G_SHAPES
#   stamps  clock stamps of the fold alone at G_STSHAPES, and where each launch's time went
#   hash    hashrate.js on every variant in G_VARIANTS (defined in G_VDEF), G_HROUNDS rounds, order rotating; then
#           G_DIAG once more with PEARL_AMD_TRACE and the clock probe
#   verify  verify-hits.js on the variants in G_VERIFY, G_VSECS each
# Lines go straight to the container's log as "[amdtest] HH:MM:SS g ...".
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
PARTS=${G_PARTS:-srb duel build fold stamps hash}
TURNS=${G_TURNS:-2}; DSECS=${G_DSECS:-300}; CLKSECS=${G_CLKSECS:-180}
HROUNDS=${G_HROUNDS:-4}; IFS=: read -r HSECS HWARM <<< "${G_HASH:-50:15}"
VARIANTS=${G_VARIANTS:-base twin}
DIAGV=${G_DIAG:-}
FROUNDS=${G_FROUNDS:-3}
# w512: 4 of the miner's 128-offset batches in one launch by PEARL_AMD_WALK's supertiles at 8 x 1024 (a strip of 8
# column tiles through every row). strip: the same launch as strips of the grid's z (PEARL_AMD_STRIP), the host
# choosing how many at run time (PEARL_AMD_STRIPS).
WSTRIP="-DPEARL_AMD_WALK=1 -DPEARL_AMD_WALK_W=8 -DPEARL_AMD_WALK_H=1024"
SWEEPV=${G_SWEEP:-base=;twin=-DPEARL_AMD_GOALTWIN=1;strip=-DPEARL_AMD_STRIP=1;cb256=-DPEARL_AMD_COL_BATCH=256;cb512=-DPEARL_AMD_COL_BATCH=512;w512=$WSTRIP -DPEARL_AMD_COL_BATCH=512;sbase=-DPEARL_AMD_STAMP=1;sw=-DPEARL_AMD_STAMP=1 $WSTRIP}
# A variant is a build and the environment it runs in: "name=build K=V ...;...". One not named here is the build of
# its name.
VDEF="hq0=base PEARL_AMD_HASHQ=0;hq0t=twin PEARL_AMD_HASHQ=0;hz=base PEARL_AMD_HASHQ_ZERO=1;hzw=base PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;hw=base PEARL_AMD_HASHQ_WAIT=0;s1=strip PEARL_AMD_STRIPS=1;s2=strip PEARL_AMD_STRIPS=2;s4=strip PEARL_AMD_STRIPS=4;s8=strip PEARL_AMD_STRIPS=8;s16=strip PEARL_AMD_STRIPS=16;s4z=strip PEARL_AMD_STRIPS=4 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s8z=strip PEARL_AMD_STRIPS=8 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s2z=strip PEARL_AMD_STRIPS=2 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s4q0=strip PEARL_AMD_STRIPS=4 PEARL_AMD_HASHQ=0;${G_VDEF:-}"
ARCHX=${ARCH:-$(target)}
FOLD=$(case "$ARCHX" in gfx11*) echo wmma11 ;; gfx12*) echo wmma12 ;; *) echo ref ;; esac)
SW=/root/sw
has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
stats() {   # FILE of numbers: mean, median, min, max, n
  node -e 'const v=require("fs").readFileSync(process.argv[1],"utf8").split(/\s+/).filter(Boolean).map(Number).filter(Number.isFinite);
    if(!v.length){console.log("n=0");process.exit(0)}v.sort((a,b)=>a-b);const m=v.reduce((s,x)=>s+x,0)/v.length;
    console.log("mean="+m.toFixed(2)+" median="+v[v.length>>1].toFixed(2)+" min="+v[0].toFixed(2)+" max="+v[v.length-1].toFixed(2)+" n="+v.length)' "$1" 2>/dev/null
}
say "job start $ARCHX ($FOLD), parts: $PARTS"
# The PR head's core, for the duel: built once a box and kept in /root/core-pr.
tree() {   # SHA: /opt/llmjob at that commit, and build/ built from it
  [ "$(git -C $R rev-parse HEAD)" = "$1" ] && [ -f build/pearl_core_hip.node ] && return 0
  ( cd $R && git fetch -q --depth 1 origin "$1" && git checkout -q -f FETCH_HEAD ) > /root/tree.log 2>&1 \
    || { say "FAIL fetch $1: $(tail -n 2 /root/tree.log | tr '\n' '|' | cut -c1-200)"; return 1; }
  ./build-amd.sh gpu "$ARCHX" > /root/tree-build.log 2>&1 \
    || { say "FAIL build $1"; grep -E 'error' /root/tree-build.log | head -6 | while IFS= read -r l; do say "build | ${l:0:300}"; done; return 1; }
  say "tree $(git -C $R rev-parse --short=12 HEAD) built, core md5 $(md5sum build/pearl_core_hip.node | cut -c1-8)"
}
if [ -n "${G_PR:-}" ] && [ ! -f /root/core-pr/pearl_core_hip.node ]; then
  tree "$G_PR" && mkdir -p /root/core-pr && cp build/pearl_core_hip.node /root/core-pr/ && echo "$G_PR" > /root/core-pr/sha
fi
OURS_CORE=/root/core-pr/pearl_core_hip.node
[ -f $OURS_CORE ] || OURS_CORE=$A/build/pearl_core_hip.node

# The clock probe: one wave reads SHADER_CYCLES and REALTIME (100 MHz) around a short loop, every few seconds.
if [ ! -x /root/clkprobe ]; then
cat > /root/clkprobe.cpp <<'EOF_CP'
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
#define CP_GFX12 1
#else
#define CP_GFX12 0
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#define CP_GFX11 1
#else
#define CP_GFX11 0
#endif
__device__ __forceinline__ uint32_t cp_cycles() {
#if CP_GFX12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);
#elif CP_GFX11
  return (uint32_t)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;
#else
  return 0u;
#endif
}
__device__ __forceinline__ uint32_t cp_realtime() {
#if CP_GFX12 || CP_GFX11
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#else
  return 0u;
#endif
}
__global__ __launch_bounds__(32) void cp_kernel(uint32_t steps, uint32_t spin, uint32_t *__restrict__ out) {
  const uint32_t r0 = cp_realtime();
  uint32_t last = cp_cycles();
  uint32_t lo = 0u, hi = 0u, mx = 0u;
  uint32_t x = (uint32_t)threadIdx.x + 1u;
  for (uint32_t s = 0; s < steps; s++) {
    for (uint32_t i = 0; i < spin; i++) {
      x = x * 1664525u + 1013904223u;
#if CP_GFX12 || CP_GFX11
      asm volatile("" : "+v"(x));
#endif
    }
    const uint32_t c = cp_cycles();
    uint32_t d = c - last;
#if CP_GFX11
    d &= 0xFFFFFu;
#endif
    last = c;
    const uint32_t nlo = lo + d;
    hi += nlo < lo ? 1u : 0u;
    lo = nlo;
    mx = d > mx ? d : mx;
  }
  const uint32_t r1 = cp_realtime();
  if (threadIdx.x == 0) {
    out[0] = r1 - r0; out[1] = lo; out[2] = hi; out[3] = mx; out[4] = x;
  }
}
int main(int argc, char **argv) {
  const double every = argc > 1 ? atof(argv[1]) : 10.0;
  const int count = argc > 2 ? atoi(argv[2]) : 1;
  uint32_t *d = nullptr;
  if (hipMalloc(&d, 8 * sizeof(uint32_t)) != hipSuccess) { printf("clkfail malloc\n"); return 2; }
  int wallKhz = 0;
  if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0) != hipSuccess || wallKhz <= 0) wallKhz = 100000;
  (void)hipGetLastError();
  for (int i = 0; i < count; i++) {
    uint32_t h[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    hipLaunchKernelGGL(cp_kernel, dim3(1), dim3(32), 0, 0, 256u, 256u, d);
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost) != hipSuccess) {
      printf("clkfail launch\n");
      fflush(stdout);
      return 2;
    }
    const double cyc = (double)h[1] + 4294967296.0 * (double)h[2];
    const double secs = (double)h[0] / ((double)wallKhz * 1e3);
    printf("clk %.0f MHz %.0f cycles %.3f ms longest %u\n", secs > 0 ? cyc / secs / 1e6 : 0.0, cyc, secs * 1e3, h[3]);
    fflush(stdout);
    if (i + 1 < count) std::this_thread::sleep_for(std::chrono::duration<double>(every));
  }
  hipFree(d);
  return 0;
}
EOF_CP
if $CLANG -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 /root/clkprobe.cpp -o /root/clkprobe \
    -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/clkprobe-build.log 2>&1; then
  say "clkprobe idle card: $(timeout 60 /root/clkprobe 1 3 2>&1 | tr '\n' ';' | cut -c1-300)"
else
  say "clkprobe build FAIL"; grep -E 'error' /root/clkprobe-build.log | head -6 | while IFS= read -r l; do say "clkprobe | ${l:0:300}"; done
  rm -f /root/clkprobe
fi
fi
# clk_start FILE EVERY COUNT: the probe beside whatever runs now; clk_of FILE: "median MHz (n)".
CLKPID=""
clk_start() { CLKPID=""; [ -x /root/clkprobe ] || return 0; ( exec /root/clkprobe "$2" "$3" ) > "$1" 2>&1 & CLKPID=$!; }
clk_stop() { [ -n "$CLKPID" ] && kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null; CLKPID=""; }
clk_of() {
  [ -s "$1" ] || { echo "none"; return; }
  grep '^clk ' "$1" | awk '{print $2}' > "$1.n"
  echo "$(stats "$1.n")$(grep -m1 clkfail "$1" | sed 's/^/ /')"
}

# ---- SRBMiner: which version runs here -------------------------------------------------------------------------
SRBV=""; SRBPTY=0
srb_fetch() {   # VER MD5 -> /opt/srb-VER, the whole release folder
  local ver=$1 md5=$2 d=/opt/srb-$1
  [ -x $d/SRBMiner-MULTI ] && return 0
  mkdir -p $d
  ( cd $d && curl -fsSL --retry 3 -o srb.tgz "https://github.com/doktor83/SRBMiner-Multi/releases/download/$ver/SRBMiner-Multi-${ver//./-}-Linux.tar.gz" \
      && echo "$md5  srb.tgz" | md5sum -c --quiet && tar xzf srb.tgz --strip-components=1 && [ -x SRBMiner-MULTI ] ) \
    || { say "srb $ver FAIL download, checksum or unpack"; return 1; }
}
srb_args() {
  echo "--algorithm-gpu pearlhash --pool $DUEL_POOL --wallet $WALLET --worker $WORKER-srb --disable-cpu --api-enable --api-port 21550 --disable-gpu-nvidia --disable-gpu-intel"
}
srb_start() {
  cd /opt/srb-$SRBV || exit 1
  # With a terminal: script gives it one, and its input stays open (tail), so nothing reads an end of file.
  if [ "$SRBPTY" = 1 ]; then tail -f /dev/null | script -qfec "./SRBMiner-MULTI $(srb_args)" /dev/null
  else exec ./SRBMiner-MULTI $(srb_args); fi
}
ours_start() {
  cd $R/earn && PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$OURS_CORE NODE_PATH=$NM/node_modules \
    exec node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$DUEL_POOL" -w "$WORKER-ours" \
      --no-report --no-serve --no-update
}
stop_miners() {
  pkill -f SRBMiner-MULTI; pkill -f 'src/cli/earn-cli\.js'; sleep 8
  pkill -9 -f SRBMiner-MULTI 2>/dev/null; pkill -f 'tail -f /dev/null' 2>/dev/null
}
# srb_try VER PTY: does it mine? Up to 75 s for a rate from its API.
srb_try() {
  local ver=$1 pty=$2 pid t=0 th=""
  SRBV=$ver; SRBPTY=$pty
  ( srb_start ) > /root/srb-try.log 2>&1 < /dev/null &
  pid=$!
  while [ $t -lt 75 ]; do
    sleep 5; t=$((t + 5))
    kill -0 $pid 2>/dev/null || break
    th=$(sample_srb | awk '{print $1}')
    [ -n "$th" ] && awk "BEGIN{exit !($th > 1)}" && break
    th=""
  done
  if [ -n "$th" ]; then
    say "srb $ver$([ "$pty" = 1 ] && echo ' (given a terminal)') mines here: $th TH/s after ${t}s"
    tr '\r' '\n' < /root/srb-try.log | sed 's/\x1b\[[0-9;]*m//g' | grep -iE 'SRBMiner|GPU0|gpu.*\[|OpenCL|intensity|kernel|driver' | head -12 | while IFS= read -r l; do say "srb | ${l:0:240}"; done
    stop_miners; return 0
  fi
  say "srb $ver$([ "$pty" = 1 ] && echo ' (given a terminal)') does not mine here (${t}s, $(kill -0 $pid 2>/dev/null && echo still running || echo exited)): $(tr '\r' '\n' < /root/srb-try.log | sed 's/\x1b\[[0-9;]*m//g' | grep -v '^\s*$' | tail -4 | tr '\n' '|' | cut -c1-330)"
  stop_miners; SRBV=""; return 1
}
if { has srb || has duel; } && [ -s /root/srbv ] && [ -z "${G_SRBV:-}" ]; then
  read -r SRBV SRBPTY < /root/srbv
  say "srb version for the duel: $SRBV (picked earlier on this box)"
elif has srb || has duel; then
  command -v script >/dev/null || apt-get install -y -qq --no-install-recommends bsdutils > /dev/null 2>&1
  # 3.8.6 stopped here without wget ("Make sure you have 'wget' installed"), and printed nothing without a terminal.
  command -v wget >/dev/null || apt-get install -y -qq --no-install-recommends wget > /dev/null 2>&1
  say "wget: $(command -v wget || echo missing)"
  if [ -n "${G_SRBV:-}" ]; then
    case $G_SRBV in 3.8.2) m5=9f64784a9894067b7bc3811b4912dd81 ;; 3.8.6) m5=38128fa15afff3c488e53726aa251249 ;; esac
    srb_fetch "$G_SRBV" "$m5" && srb_try "$G_SRBV" "${G_SRBPTY:-0}"
  else
    srb_fetch 3.8.6 38128fa15afff3c488e53726aa251249 && { srb_try 3.8.6 0 || srb_try 3.8.6 1; }
    [ -n "$SRBV" ] || { srb_fetch 3.8.2 9f64784a9894067b7bc3811b4912dd81 && srb_try 3.8.2 0; }
  fi
  say "srb version for the duel: ${SRBV:-none}"
  [ -n "$SRBV" ] && echo "$SRBV $SRBPTY" > /root/srbv
fi

# ---- the duel --------------------------------------------------------------------------------------------------
ours_rates() { tr '\r' '\n' < /root/duel-ours.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted' | awk '{print $1}'; }
# turn NAME LABEL SECS PROBE: one miner for SECS. PROBE 1: the clock probe runs beside it from 40 s on.
# Ends with "duel LABEL result": the mean of the 10 s samples after the first 60 s. For ours, also the mean of
# every line the CLI printed after then (one a second).
turn() {
  local name=$1 label=$2 secs=$3 probe=$4 pid t0 t s th acc rej last="" skip="" f=/root/duel-$2
  rm -f $f.samples $f.clk $f.all; : > $f.samples
  stop_miners
  say "duel $label start ${secs}s$([ "$probe" = 1 ] && echo ', clock probe beside it')"
  ( ${name}_start ) > /root/duel-$name.log 2>&1 < /dev/null &
  pid=$!; t0=$(date +%s)
  while :; do
    sleep 10; t=$(( $(date +%s) - t0 ))
    if ! kill -0 $pid 2>/dev/null; then
      say "duel $label exited at ${t}s: $(tr '\r' '\n' < /root/duel-$name.log | grep -v '^\s*$' | tail -3 | tr '\n' '|' | cut -c1-300)"
      break
    fi
    [ "$probe" = 1 ] && [ -z "$CLKPID" ] && [ $t -ge 40 ] && clk_start $f.clk 10 $(( (secs - t) / 10 ))
    s=$(sample_$name); read -r th acc rej <<< "$s"
    if [ -n "$th" ]; then
      last="$th acc=$acc rej=$rej"
      if [ $t -gt 60 ]; then
        echo "$th" >> $f.samples
        [ -z "$skip" ] && [ "$name" = ours ] && skip=$(ours_rates | wc -l)
      fi
    fi
    [ $((t % 60)) -lt 10 ] && say "duel $label t=${t}s th=${th:-?} acc=${acc:-?} rej=${rej:-?}$([ -s $f.clk ] && echo " $(tail -n 1 $f.clk | cut -c1-40)")"
    [ $t -ge "$secs" ] && break
  done
  [ "$name" = ours ] && [ -n "$skip" ] && ours_rates | tail -n +$((skip + 1)) > $f.all
  if [ "$name" = srb ]; then
    say "duel $label api: $(curl -s -m 3 http://127.0.0.1:21550 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s),a=j.algorithms[0];console.log(JSON.stringify({ver:j.miner_version,hashrate:a.hashrate,shares:a.shares}).slice(0,330))}catch(e){console.log("unreadable")}})')"
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  clk_stop; stop_miners
  say "duel $label result $(stats $f.samples) last=${last:-?} secs=$secs$([ -s $f.all ] && echo " | every line: $(stats $f.all)")$([ "$probe" = 1 ] && echo " | clock MHz: $(clk_of $f.clk)")"
}
if has duel && [ -n "$SRBV" ]; then
  say "duel pool $DUEL_POOL: ours (HIP core $(cut -c1-12 /root/core-pr/sha 2>/dev/null || git -C $R rev-parse --short=12 HEAD), the default build, md5 $(md5sum $OURS_CORE | cut -c1-8)) against SRBMiner $SRBV$([ "$SRBPTY" = 1 ] && echo ' (given a terminal)'), $TURNS turns each of ${DSECS}s"
  # The card warm first: ours for 60 s, not measured.
  ( ours_start ) > /root/duel-ours.log 2>&1 < /dev/null & wp=$!; sleep 60; kill $wp 2>/dev/null; wait $wp 2>/dev/null
  for ((i = 1; i <= TURNS; i++)); do
    turn ours ours$i "$DSECS" 0
    turn srb srb$i "$DSECS" 0
  done
  if [ "$CLKSECS" -gt 0 ] && [ -x /root/clkprobe ]; then
    turn ours oursclk "$CLKSECS" 1
    turn srb srbclk "$CLKSECS" 1
  fi
  if [ -n "${G_SRB_ALSO:-}" ] && [ "$G_SRB_ALSO" != "$SRBV" ]; then
    keep=$SRBV
    keeppty=$SRBPTY
    srb_fetch "$G_SRB_ALSO" 9f64784a9894067b7bc3811b4912dd81 && { SRBV=$G_SRB_ALSO; SRBPTY=0; turn srb srb-$G_SRB_ALSO "$DSECS" 0; }
    SRBV=$keep; SRBPTY=$keeppty
  fi
  tr '\r' '\n' < /root/duel-ours.log | grep -iE 'mining on|error|reject' | grep -v 'TH/s ·' | tail -n 4 | while IFS= read -r l; do say "duel ours | ${l:0:300}"; done
fi

# ---- our builds ------------------------------------------------------------------------------------------------
if has build; then
  mkdir -p $SW
  if [ -n "${G_CODE:-}" ]; then tree "$G_CODE" || { say "job done: no tree"; exit 1; }; fi
  SWEEP="$SWEEPV" SWEEP_FOLD= SWEEP_ROUNDS=0 SWEEP_ROOT=$SW bash test/sweep.sh > /root/sweep.log 2>&1
  grep -E ' sweep .*(FAIL|not timed)' /root/sweep.log | head -8 | while IFS= read -r l; do say "build | ${l:0:300}"; done
  say "builds at $(git -C $R rev-parse --short=12 HEAD): $(for d in $SW/var/*/; do echo "$(basename $d) $(md5sum $d/pearl_core_hip.node | cut -c1-6)"; done | tr '\n' ' ')"
fi

# The fold alone. Shapes are m:cols:launches, the same MACs each: the miner's launch (8 tiles wide, 128 column
# offsets), 2 and 4 of them in one launch (taller: B' is the same 4 MiB), and 16 and 32 tiles wide (batch 256, 512).
SHAPES=${G_SHAPES:-base:131072:2048:400 base:262144:2048:200 base:524288:2048:100 base:131072:4096:200 base:131072:8192:100 w512:131072:4096:200 w512:131072:8192:100}
if has fold && [ -x $SW/var/base/pearl_amd_foldtest ]; then
  for ((i = 1; i <= FROUNDS; i++)); do
    line=""
    for sh in $SHAPES; do
      IFS=: read -r FV FM FC FR <<< "$sh"
      [ -x $SW/var/$FV/pearl_amd_foldtest ] || continue
      r=$(timeout 900 $SW/var/$FV/pearl_amd_foldtest "$FM" "$FC" 1 "$FR" 0 "$FOLD" 0 miner 2>&1 | grep -E "^$FOLD" | sed 's/  */ /g')
      t=$(printf '%s\n' "$r" | sed -nE 's/.* ([0-9]+) launches, ([0-9.]+) s, ([0-9.]+) TMAC\/s$/\3 TMAC\/s, \2 s \/ \1/p')
      ms=$(printf '%s\n' "$r" | sed -nE 's/.* ([0-9]+) launches, ([0-9.]+) s, .*/\1 \2/p' | awk '{printf "%.3f", 1000 * $2 / $1}')
      line="$line; $FV ${FM}x${FC}: ${t:-?} = ${ms:-?} ms a launch"
    done
    say "fold round $i of $FROUNDS$line"
  done
fi

# Clock stamps of the fold alone, and where a launch's time went (tl.js): the blocks' span against the launch's
# period (the gap is what a launch costs outside its blocks), the ramp to full occupancy and the tail.
cat > /root/tl.js <<'EOF_TL'
'use strict';
const ss = require(process.argv[2]);
const data = ss.load(require('fs').readFileSync(process.argv[3]));
const s = ss.summarize(data);
const launchMs = Number(process.argv[4] || 0);
const msPerTick = 1 / (s.tickMhz * 1e3);
const lead = data.waves.filter((w) => w.bits > 0 && w.hw0 === w.hw1 && w.wave === 0 && Number.isFinite(w.start));
if (!lead.length) { console.log('tl: no blocks'); process.exit(0); }
let t0 = Infinity, t1 = -Infinity, busy = 0;
const ev = [];
for (const w of lead) {
  t0 = Math.min(t0, w.start); t1 = Math.max(t1, w.end); busy += w.end - w.start;
  ev.push([w.start, 1], [w.end, -1]);
}
ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
let c = 0, peak = 0;
for (const e of ev) { c += e[1]; if (c > peak) peak = c; }
const need = Math.ceil(0.97 * peak);
c = 0;
let full0 = null, full1 = null;
for (const e of ev) {
  const was = c; c += e[1];
  if (was < need && c >= need && full0 === null) full0 = e[0];
  if (was >= need && c < need) full1 = e[0];
}
const span = (t1 - t0) * msPerTick, ideal = busy * msPerTick / peak;
const durs = lead.map((w) => (w.end - w.start) * msPerTick).sort((a, b) => a - b);
const f = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '-');
const r = s.rest || {};
console.log('tl: ' + lead.length + ' blocks, ' + peak + ' at once; launch ' + f(launchMs) + ' ms, its blocks span '
  + f(span) + ' ms, gap ' + f(launchMs - span) + ' ms (' + f(100 * (launchMs - span) / launchMs, 2) + '%); ramp '
  + f((full0 - t0) * msPerTick) + ' ms, tail ' + f((t1 - full1) * msPerTick) + ' ms; busy time over ' + peak + ' slots '
  + f(ideal) + ' ms (' + f(100 * ideal / launchMs, 2) + '% of the launch); block ' + f(durs[durs.length >> 1], 4)
  + ' ms median; ' + (r.mhz ? Math.round(r.mhz.med) + ' MHz, slab ' + Math.round(r.slab.med) + ' cycles' : ''));
EOF_TL
STSHAPES=${G_STSHAPES:-sbase:131072:2048:400 sbase:524288:2048:100 sbase:131072:4096:200 sw:131072:8192:100}
if has stamps; then
  for ((i = 1; i <= ${G_SROUNDS:-2}; i++)); do
    for sh in $STSHAPES; do
      IFS=: read -r FV FM FC FR <<< "$sh"
      [ -x $SW/var/$FV/pearl_amd_foldtest ] || continue
      rm -f /root/st.bin
      r=$(timeout 900 $SW/var/$FV/pearl_amd_foldtest "$FM" "$FC" 1 "$FR" 0 "$FOLD" 0 miner /root/st.bin 2>&1)
      ms=$(printf '%s\n' "$r" | grep -E "^$FOLD" | sed -nE 's/.* ([0-9]+) launches, ([0-9.]+) s, .*/\1 \2/p' | awk '{printf "%.3f", 1000 * $2 / $1}')
      say "stamp r$i $FV ${FM}x${FC}: $(printf '%s\n' "$r" | grep -E "^$FOLD" | sed 's/  */ /g' | cut -c1-200)"
      [ -s /root/st.bin ] || { say "stamp r$i $FV ${FM}x${FC}: no stamps written: $(printf '%s\n' "$r" | tail -n 2 | tr '\n' '|' | cut -c1-200)"; continue; }
      node test/stamp-summary.js summary /root/st.bin 2>&1 | grep -E '^(first|rest)' | while IFS= read -r l; do say "stamp r$i $FV ${FM}x${FC} ${l:0:330}"; done
      node /root/tl.js "$A/test/stamp-summary.js" /root/st.bin "${ms:-0}" 2>&1 | while IFS= read -r l; do say "stamp r$i $FV ${FM}x${FC} ${l:0:380}"; done
    done
  done
fi

# hashrate.js on each variant (VDEF above).
# vdef NAME: sets VCORE and VENV from VDEF.
vdef() {
  local d; VCORE=$1; VENV=()
  IFS=';' read -ra DS <<< "$VDEF"
  for d in "${DS[@]}"; do
    [ "${d%%=*}" = "$1" ] || continue
    read -ra W <<< "${d#*=}"
    VCORE=${W[0]}; VENV=("${W[@]:1}")
    return 0
  done
}
# hr LABEL VARIANT [K=V ...]: one run; prints "hash LABEL VARIANT th=.. sd=.. n=..".
hr() {
  local label=$1 v=$2 core envs log r; shift 2
  vdef "$v"; core=$VCORE; envs=("${VENV[@]}")
  [ -n "${G_JOBEVERY:-}" ] && envs+=("PEARL_JOB_EVERY=$G_JOBEVERY")
  [ -f $SW/var/$core/pearl_core_hip.node ] || { say "hash $label $v: no build"; return; }
  log=$(env PEARL_AMD_UNTESTED=1 "${envs[@]}" "$@" timeout 600 node ../probes/hashrate.js $SW/var/$core/pearl_core_hip.node "$HSECS" "$HWARM" 2>&1 | grep -v nvidia-smi)
  r=$(printf '%s\n' "$log" | grep -E '"th"' | tail -n 1 | sed -nE 's/.*"th":([0-9.]+),"sd":([0-9.]+),"n":([0-9]+).*/th=\1 sd=\2 n=\3/p')
  [ -n "$r" ] || r="FAIL $(printf '%s\n' "$log" | tail -n 2 | tr '\n' '|' | cut -c1-240)"
  say "hash $label $v $r"
  r=$(printf '%s\n' "$log" | grep '^\[pearl-amd\]' | tail -n 1)
  [ -z "$r" ] || say "hash $label $v trace: ${r:0:330}"
}
if has hash; then
  read -ra VS <<< "$VARIANTS"
  n=${#VS[@]}
  for ((i = 1; i <= HROUNDS; i++)); do
    for ((j = 0; j < n; j++)); do hr "r$i" "${VS[$(( (j + i - 1) % n ))]}"; done
  done
  # One more round with the core's trace, and the clock probe beside it (every 5 s after the warm-up).
  for v in $DIAGV; do
    rm -f /root/hr.clk
    ( sleep $((HWARM + 5)); exec /root/clkprobe 5 $(( (HSECS - 8) / 5 )) ) > /root/hr.clk 2>&1 &
    cp=$!
    hr diag "$v" PEARL_AMD_TRACE=1
    kill $cp 2>/dev/null; wait $cp 2>/dev/null; pkill -f /root/clkprobe 2>/dev/null
    say "hash diag $v clock MHz: $(clk_of /root/hr.clk)"
  done
  if [ -n "${G_COST:-}" ]; then
  # What the trace and the probe cost, each alone, on base.
  hr trace-only base PEARL_AMD_TRACE=1
  rm -f /root/hr.clk
  ( sleep $((HWARM + 5)); exec /root/clkprobe 5 $(( (HSECS - 8) / 5 )) ) > /root/hr.clk 2>&1 &
  cp=$!
  hr probe-only base
  kill $cp 2>/dev/null; wait $cp 2>/dev/null; pkill -f /root/clkprobe 2>/dev/null
  say "hash probe-only base clock MHz: $(clk_of /root/hr.clk)"
  hr plain base
  fi
fi

# verify-hits.js on a variant: the pool's check of every hit, 400 of them.
if has verify; then
  for v in ${G_VERIFY:-}; do
    vdef "$v"
    [ -f $SW/var/$VCORE/pearl_core_hip.node ] || { say "verify $v: no build"; continue; }
    env "${VENV[@]}" timeout 3600 node ../probes/verify-hits.js $SW/var/$VCORE/pearl_core_hip.node "${G_VSECS:-600}" > /root/verify-$v.log 2>&1
    rc=$?
    j=$(grep '^{' /root/verify-$v.log | tail -n 1)
    say "verify $v rc=$rc $(printf '%s' "$j" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);console.log(JSON.stringify({PASS:j.PASS,hits:j.hits,verified:j.verified,failed:j.failed,salts:j.salts,secs:j.secs,fold:j.device&&j.device.fold,error:j.error}).slice(0,330))}catch(e){console.log("no result: "+s.slice(-200))}})')"
    [ -n "$j" ] || tail -n 3 /root/verify-$v.log | while IFS= read -r l; do say "verify $v | ${l:0:300}"; done
  done
fi
say "job done"
