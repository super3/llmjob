# y11: W64GB=2 (second B' set a slab ahead) against base, twin and W64GB; fold check, regs, isa, 3 rounds, clock diag, verify-hits
G_CODE=28474d6c6e1cdde8f6c6db878d82d22655ac4e42
G_PARTS=build\ regs\ isa\ hash\ verify
G_SWEEP=base=\;twin=-DPEARL_AMD_GOALTWIN=1\;w64gb=-DPEARL_AMD_W64=1\ -DPEARL_AMD_W64GB=1\ -mwavefrontsize64\;w64gb2=-DPEARL_AMD_W64=1\ -DPEARL_AMD_W64GB=2\ -mwavefrontsize64
G_VARIANTS=base\ twin\ w64gb\ w64gb2
G_HROUNDS=3
G_DIAG=base\ w64gb2
G_ISA=w64gb2
G_VERIFY=w64gb2
G_VSECS=300
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
VDEF="hq0=base PEARL_AMD_HASHQ=0;hq0t=twin PEARL_AMD_HASHQ=0;hz=base PEARL_AMD_HASHQ_ZERO=1;hzw=base PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;hw=base PEARL_AMD_HASHQ_WAIT=0;s1=strip PEARL_AMD_STRIPS=1;s2=strip PEARL_AMD_STRIPS=2;s4=strip PEARL_AMD_STRIPS=4;s8=strip PEARL_AMD_STRIPS=8;s16=strip PEARL_AMD_STRIPS=16;s4z=strip PEARL_AMD_STRIPS=4 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s8z=strip PEARL_AMD_STRIPS=8 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s2z=strip PEARL_AMD_STRIPS=2 PEARL_AMD_HASHQ_ZERO=1 PEARL_AMD_HASHQ_WAIT=0;s4q0=strip PEARL_AMD_STRIPS=4 PEARL_AMD_HASHQ=0;bt2=base TICK=0.002;bt10=base TICK=0.010;bt1=base TICK=0.001;hq0t2=base PEARL_AMD_HASHQ=0 TICK=0.002;s4t2=strip PEARL_AMD_STRIPS=4 TICK=0.002;${G_VDEF:-}"
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
clk_stop() { if [ -n "$CLKPID" ]; then kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null; fi; CLKPID=""; }
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
      && { if [ ${#md5} = 64 ]; then echo "$md5  srb.tgz" | sha256sum -c --quiet; else echo "$md5  srb.tgz" | md5sum -c --quiet; fi; } && tar xzf srb.tgz --strip-components=1 && [ -x SRBMiner-MULTI ] ) \
    || { say "srb $ver FAIL download, checksum or unpack"; return 1; }
}
srb_args() {
  echo "--algorithm-gpu pearlhash --pool ${SRB_POOL:-$DUEL_POOL} --wallet $WALLET --worker $WORKER-srb --disable-cpu --api-enable --api-port 21550 --disable-gpu-nvidia --disable-gpu-intel ${G_SRBX:-}"
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
# A second build of ours in the duel (G_OURSX, a variant of VDEF): its core, with its settings.
oursx_start() {
  vdef "$G_OURSX"
  cd $R/earn && exec env "${VENV[@]}" PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$SW/var/$VCORE/pearl_core_hip.node \
    NODE_PATH=$NM/node_modules node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$DUEL_POOL" \
      -w "$WORKER-ours" --no-report --no-serve --no-update
}
stop_miners() {
  say "stop_miners"; pkill -f SRBMiner-MULTI; pkill -f 'src/cli/earn-cli\.js'; sleep 8
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
    case $G_SRBV in 3.8.2) m5=9f64784a9894067b7bc3811b4912dd81 ;; 3.8.6) m5=38128fa15afff3c488e53726aa251249 ;; 3.8.8) m5=2650fbe22162528db96640b6fba299d6 ;; 3.8.9) m5=186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1 ;; esac
    srb_fetch "$G_SRBV" "$m5" && srb_try "$G_SRBV" "${G_SRBPTY:-0}"
  else
    srb_fetch 3.8.9 186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1 && { srb_try 3.8.9 0 || srb_try 3.8.9 1; }
  fi
  say "srb version for the duel: ${SRBV:-none}"
  [ -n "$SRBV" ] && echo "$SRBV $SRBPTY" > /root/srbv
fi

# ---- the duel --------------------------------------------------------------------------------------------------
# The rates the CLI printed in the running turn ($name is turn's: ours, or oursx for the second build of ours).
ours_rates() { tr '\r' '\n' < /root/duel-$name.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted' | awk '{print $1}'; }
sample_oursx() {
  tr '\r' '\n' < /root/duel-oursx.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1 \
    | awk '{gsub(",","",$4); gsub(",","",$7); print $1, $4, $7}'
}
# turn NAME LABEL SECS PROBE: one miner for SECS. PROBE 1: the clock probe runs beside it from 40 s on.
# Ends with "duel LABEL result": the mean of the 10 s samples after the first 60 s. For ours, also the mean of
# every line the CLI printed after then (one a second).
turn() {
  local name=$1 label=$2 secs=$3 probe=$4 pid t0 t s th acc rej last="" skip="" f=/root/duel-$2
  rm -f $f.samples $f.clk $f.all; : > $f.samples
  stop_miners
  say "duel $label start ${secs}s$([ "$probe" = 1 ] && echo ', clock probe beside it')"
  SRB_POOL=""; STAP=""
  if [ "$name" = srb ] && [ -n "${G_TAPSRB:-}" ]; then
    # SRBMiner through sharetap.py's relay, to read the k, rank and tile of the shares it sends.
    command -v python3 >/dev/null || apt-get install -y -qq --no-install-recommends python3 > /root/apt3.log 2>&1
    rm -f /root/srbtap.jsonl
    python3 $(tdir)/sharetap.py tap 3333 "${DUEL_POOL%:*}" "${DUEL_POOL##*:}" /root/srbtap.jsonl > /root/srbtap.log 2>&1 &
    STAP=$!; sleep 3
    if grep -q listening /root/srbtap.log; then SRB_POOL=127.0.0.1:3333; else say "duel $label sharetap FAIL $(tail -c 200 /root/srbtap.log)"; fi
  fi
  ( ${name}_start ) > /root/duel-$name.log 2>&1 < /dev/null &
  pid=$!; t0=$(date +%s); TURN_T0=$t0
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
        [ -z "$skip" ] && [ "$name" != srb ] && skip=$(ours_rates | wc -l)
      fi
    fi
    [ $((t % 60)) -lt 10 ] && say "duel $label t=${t}s th=${th:-?} acc=${acc:-?} rej=${rej:-?}$([ -s $f.clk ] && echo " $(tail -n 1 $f.clk | cut -c1-40)")"
    # The card's own figures (Salad's amd-smi, if it reads anything through WSL2), once a turn at about 2 minutes.
    if [ -n "${G_SMI:-}" ] && [ $t -ge 120 ] && [ $t -lt 130 ] && [ -x /opt/rocm-wsl/bin/amd-smi ]; then
      timeout 20 /opt/rocm-wsl/bin/amd-smi metric 2>&1 | grep -iE 'power|socket|gfx_clk|clk|temp|usage|activity|voltage' | grep -v 'N/A' | head -n 24 | tr -s ' ' | while IFS= read -r l; do say "duel $label smi | ${l:0:200}"; done
    fi
    [ $t -ge "$secs" ] && break
  done; TURN_T1=$(date +%s)
  [ "$name" != srb ] && [ -n "$skip" ] && ours_rates | tail -n +$((skip + 1)) > $f.all
  if [ "$name" = srb ]; then
    say "duel $label api: $(curl -s -m 3 http://127.0.0.1:21550 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s),a=j.algorithms[0];console.log(JSON.stringify({ver:j.miner_version,hashrate:a.hashrate,shares:a.shares}).slice(0,330))}catch(e){console.log("unreadable")}})')"
    if [ -n "${G_SRBLOG:-}" ]; then
      # Everything its API says (wallet left out), and what its log says about the job: k, difficulty, devices.
      curl -s -m 3 http://127.0.0.1:21550 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{let t=s.replace(/prl1[0-9a-z]{20,}/g,"<addr>");for(let i=0;i<Math.min(t.length,3600);i+=330)console.log(t.slice(i,i+330))})' | while IFS= read -r l; do say "duel $label apijson | $l"; done
      tr '\r' '\n' < /root/duel-$name.log | sed 's/\x1b\[[0-9;]*m//g; s/prl1[0-9a-z]\{20,\}/<addr>/g' | grep -v '^\s*$' | awk '!seen[$0]++' | head -n 70 | while IFS= read -r l; do say "duel $label log | ${l:0:300}"; done
    fi
  fi
  kill $pid 2>/dev/null; for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  clk_stop; stop_miners
  if [ -n "$STAP" ]; then
    kill -9 $STAP 2>/dev/null; wait $STAP 2>/dev/null
    timeout 120 python3 $(tdir)/sharetap.py decode /root/srbtap.jsonl --max 2 2>&1 | while IFS= read -r l; do say "duel $label share | ${l:0:420}"; done
    timeout 60 python3 $(tdir)/sharetap.py jobs /root/srbtap.jsonl 2>&1 | head -n 4 | while IFS= read -r l; do say "duel $label share | ${l:0:300}"; done
    STAP=""; SRB_POOL=""
  fi
  tr '\r' '\n' < /root/duel-$name.log | sed 's/\x1b\[[0-9;]*m//g' | grep -iE 'reject|error|stale|invalid' | grep -v 'TH/s ·' | tail -n 3 | while IFS= read -r l; do say "duel $label | ${l:0:300}"; done
  say "duel $label result $(stats $f.samples) last=${last:-?} secs=$secs$([ -s $f.all ] && echo " | every line: $(stats $f.all)")$([ "$probe" = 1 ] && echo " | clock MHz: $(clk_of $f.clk)")"
}
if has duel && [ -n "$SRBV" ]; then
  say "duel pool $DUEL_POOL: ours (HIP core $(cut -c1-12 /root/core-pr/sha 2>/dev/null || git -C $R rev-parse --short=12 HEAD), the default build, md5 $(md5sum $OURS_CORE | cut -c1-8)) against SRBMiner $SRBV$([ "$SRBPTY" = 1 ] && echo ' (given a terminal)'), $TURNS turns each of ${DSECS}s"
  # The card warm first: ours for 60 s, not measured.
  ( ours_start ) > /root/duel-ours.log 2>&1 < /dev/null & wp=$!; sleep ${G_WARM:-60}; kill $wp 2>/dev/null; wait $wp 2>/dev/null
  # G_NO_OURS=1: the PR head's default sits out, so the turns are SRBMiner and the second build (G_OURSX) only.
  for ((i = 1; i <= TURNS; i++)); do
    [ -n "${G_NO_OURS:-}" ] || turn ours ours$i "$DSECS" 0
    turn srb srb$i "$DSECS" 0
    [ -n "${G_OURSX:-}" ] && turn oursx oursx$i "$DSECS" 0
  done
  if [ "$CLKSECS" -gt 0 ] && [ -x /root/clkprobe ]; then
    if [ -n "${G_NO_OURS:-}" ] && [ -n "${G_OURSX:-}" ]; then turn oursx oursxclk "$CLKSECS" 1; else turn ours oursclk "$CLKSECS" 1; fi
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


# srbx (goal-rdna4o): SRBMiner alone with each of G_SRBXS's argument sets (";" between them, "-" for none), one turn
# of G_DSECS each, in order, G_TURNS times: its public knobs (--gpu-intensity) observed, nothing else.
if has srbx && [ -n "$SRBV" ]; then
  IFS=';' read -ra XS <<< "${G_SRBXS:--}"
  for ((i = 1; i <= TURNS; i++)); do
    j=0
    for x in "${XS[@]}"; do
      j=$((j + 1)); G_SRBX=""; [ "$x" = - ] || G_SRBX=$x
      say "srbx set $j: ${G_SRBX:-default}"
      turn srb srbx$i-$j "$DSECS" $([ "$i" = 1 ] && echo 1 || echo 0)
    done
  done
  G_SRBX=""
fi
# Ours alone on the pool for G_WSECS, to see how steady it is: every second's rate after the first minute, and how
# many of them read under G_WLOW (a PC that stops the queue reads about 50 on an RX 7900 XTX).
if has watch; then
  turn ours watch "${G_WSECS:-600}" 0
  if [ -s /root/duel-watch.all ]; then
    say "watch: $(awk -v T="${G_WLOW:-52.5}" '{ n++; if ($1 < T) lo++ } END { printf "%d of %d seconds under %s TH/s", lo, n, T }' /root/duel-watch.all); by minute: $(awk '{ s += $1; if (NR % 60 == 0) { printf "%.1f ", s / 60; s = 0 } }' /root/duel-watch.all)"
  fi
fi

# ---- the stand-in pool: is each miner credited what it shows? ---------------------------------------------------
# G_VBITS (210): the target 2^BITS. Each miner mines on verify-pool.js (it checks every share as the spec's verifier
# does) for G_VSECS, G_VTURNS turns each; a "credit" line gives the shares accepted after the first 60 s, the rate
# they prove (accepted x 2^(256-BITS) MACs a second) and the k of the shares.
if has vpool && [ -n "$SRBV" ]; then
  VB=${G_VBITS:-210}
  rm -f /root/vp.shares
  ( cd $R/earn/native/probes && PEARL_SHARE_LOG=/root/vp.shares exec node verify-pool.js serve 3334 864000 "$VB" ) > /root/vp.out 2> /root/vp.err &
  VPP=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q listening /root/vp.err && break; sleep 1; done
  if grep -q listening /root/vp.err; then
    say "vpool $(grep -m1 listening /root/vp.err | cut -c1-200)"
    keep_pool=$DUEL_POOL; keep_tap=${G_TAPSRB:-}; DUEL_POOL=127.0.0.1:3334; G_TAPSRB=""
    for ((i = 1; i <= ${G_VTURNS:-2}; i++)); do
      for nm in ${G_VNAMES:-ours srb}; do
        t0=$(date +%s); turn $nm v$nm$i "${G_VSECS:-360}" 0; t1=$(date +%s)
        say "credit v$nm$i $(awk -v a=$(( (t0 + 60) * 1000 )) -v b=$(( t1 * 1000 )) -v bits=$VB '
          BEGIN { w = 2 ^ (256 - bits) }
          $1 >= a && $1 < b { if ($4 == "ok") { ok++; kk[$3]++ } else { rej++; why = $4 } }
          END { s = (b - a) / 1000; ks = ""; for (x in kk) ks = ks " k" x "=" kk[x];
                printf "secs=%d bits=%d ok=%d rej=%d ths=%.2f +-%.1f%%%s%s", s, bits, ok, rej, ok * w / s / 1e12, (ok ? 100 / sqrt(ok) : 0), ks, (rej ? " why=" substr(why, 1, 100) : "") }' /root/vp.shares)"
      done
    done
    DUEL_POOL=$keep_pool; G_TAPSRB=$keep_tap
  else
    say "vpool FAIL $(tail -c 300 /root/vp.err | tr '\n' '|')"
  fi
  kill $VPP 2>/dev/null; wait $VPP 2>/dev/null
fi


# ---- credit (goal-rdna4v): each miner on the stand-in pool, what it shows and what the pool credits -------------
# G_CBITS (211): the target 2^BITS; a share proves 2^(256-BITS) MACs whatever its k or tile. G_CROUNDS (6) rounds of
# G_CSECS (540) a turn; odd rounds ours first, even rounds SRBMiner first. A turn's credit window is [the miner's
# start + 60 s, the moment the turn stopped sampling it): the time spent stopping one miner and starting the next
# counts for neither. (The vpool part's window ran to after stop_miners, so it took 10-20 s of no mining per turn.)
# One "cturn" line a turn: shares accepted in the window, the rate they prove, the mean of the miner's own figure
# over the same turn (ours: every CLI line after the first minute; SRBMiner: its API every 10 s after the first
# minute), the mean hs it put in its submits, and the longest gap between two accepted shares.
if has credit && [ -n "$SRBV" ]; then
  CB=${G_CBITS:-211}; CROUNDS=${G_CROUNDS:-6}; CSECS=${G_CSECS:-540}
  rm -f /root/cp.shares
  ( cd $R/earn/native/probes && PEARL_JOB_EVERY=${G_JOBEVERY:-44} PEARL_SHARE_LOG=/root/cp.shares exec node verify-pool.js serve 3334 864000 "$CB" ) > /root/cp.out 2> /root/cp.err &
  CPP=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q listening /root/cp.err && break; sleep 1; done
  if grep -q listening /root/cp.err; then
    say "cpool $(grep -m1 listening /root/cp.err | cut -c1-200), a job every ${G_JOBEVERY:-44} s, $CROUNDS rounds of ${CSECS}s"
    keep_pool=$DUEL_POOL; DUEL_POOL=127.0.0.1:3334; G_TAPSRB=""
    for ((i = 1; i <= CROUNDS; i++)); do
      if (( i % 2 )); then order="ours srb"; else order="srb ours"; fi
      for nm in $order; do
        turn $nm c$nm$i "$CSECS" 0
        if [ "$nm" = srb ]; then sf=/root/duel-c$nm$i.samples; else sf=/root/duel-c$nm$i.all; fi
        say "cturn $nm $i t0=$TURN_T0 t1=$TURN_T1 $(awk -v a=$(( (TURN_T0 + 60) * 1000 )) -v b=$(( TURN_T1 * 1000 )) -v bits=$CB '
          BEGIN { w = 2 ^ (256 - bits) }
          $1 >= a && $1 < b {
            if ($4 == "ok") { ok++; kk[$3]++; if (last) { g = $1 - last; if (g > gap) gap = g }; last = $1 }
            else if ($4 == "\"duplicate_share\"") dup++
            else { rej++; why = $4 }
            if ($5 != "null") { hs += $5; nhs++ }
          }
          END { s = (b - a) / 1000; ks = ""; for (x in kk) ks = ks " k" x "=" kk[x];
                printf "secs=%d bits=%d ok=%d dup=%d rej=%d credited=%.3f hs=%s maxgap_s=%.1f%s%s", s, bits, ok, dup, rej,
                  ok * w / s / 1e12, (nhs ? sprintf("%.2f", hs / nhs / 1e12) : "none"), gap / 1000, ks, (rej ? " why=" substr(why, 1, 100) : "") }' /root/cp.shares) shown $(stats $sf)"
      done
    done
    if [ "${G_CCLK:-180}" -gt 0 ] && [ -x /root/clkprobe ]; then
      turn ours cclkours "${G_CCLK:-180}" 1
      turn srb cclksrb "${G_CCLK:-180}" 1
    fi
    DUEL_POOL=$keep_pool
    say "cpool shares logged: $(wc -l < /root/cp.shares), by result: $(awk '{print $4}' /root/cp.shares | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  else
    say "cpool FAIL $(tail -c 300 /root/cp.err | tr '\n' '|')"
  fi
  kill $CPP 2>/dev/null; wait $CPP 2>/dev/null
fi


# ---- ratecheck (goal-rdna4v): our core's ticks against the CLI's meter and the stand-in pool's credit ---------
if has ratecheck; then
cat > /root/ratecheck.js <<'EOF_RC'
// ratecheck.js CORE.node SECS [BITS=211]: our miner (PearlMiner, one card) on the stand-in pool in this process.
// Records every hashrate tick the core sends (each is the work of the batches finished since the tick before, over
// the time since the tick before) and replays them through the CLI's own rate meter (shared/miningStats). Prints
// one JSON line comparing, over the same span (after the first SKIP seconds):
//   work:    sum of th_i * (t_i - t_{i-1}) / span, the core's finished work over wall time (each tick's rate over
//            the window it covers)
//   held:    the time integral of the latest tick, held until the next one, over the span (what the meter averages)
//   lines:   the plain mean of the 1 s lines the CLI would print (how the duel scores ours)
//   credited: accepted shares in the span x 2^(256-BITS) MACs / span
'use strict';
const path = require('path');
const net = require('net');
const E = process.env.EARN || '/opt/llmjob/earn';
const { servePool } = require(path.join(E, 'native', 'probes', 'verify-pool.js'));
const { PearlMiner } = require(path.join(E, 'src', 'main', 'pearlMiner'));
const { createRateMeter, meterSample, meterRead } = require(path.join(E, 'src', 'shared', 'miningStats'));

const [coreArg, secsArg, bitsArg] = process.argv.slice(2);
const secs = Number(secsArg || 300);
const bits = Number(bitsArg || 211);
const skip = Number(process.env.SKIP || 60) * 1000;
const addon = require(path.resolve(coreArg));
const shares = [];
const ticks = [];
const lines = [];
const meter = createRateMeter();
let lastLine = 0;
let jobs = 0;

const pool = servePool({
  port: 0, bits, jobEvery: Number(process.env.PEARL_JOB_EVERY || 44),
  onShare: (e) => shares.push(e),
});
pool.server.on('listening', () => {
  const endpoint = '127.0.0.1:' + pool.server.address().port;
  const miner = new PearlMiner({
    connect: (host, port) => net.connect({ host, port }),
    createCore: (p, o) => addon.createCore(p, o),
  });
  miner.on('log', (l) => console.error('[miner] ' + l.line));
  miner.on('error', (e) => console.error('[miner] error: ' + ((e && e.message) || e)));
  miner.on('job', () => { jobs++; });
  miner.on('hashrate', (th) => {
    const now = Date.now();
    ticks.push([now, th]);
    meterSample(meter, th, now);
    if (now - lastLine >= 1000) {
      lastLine = now;
      lines.push([now, meterRead(meter, now)]);
    }
  });
  miner.start({
    endpoint, address: 'prl1pstandin', worker: 'ratecheck',
    gpus: [{ index: Number(process.env.PEARL_GPU_INDEX || 0), name: 'GPU' }],
  });
  setTimeout(() => {
    miner.stop();
    pool.close();
    const t0 = ticks.length ? ticks[0][0] : 0;
    const A = t0 + skip;
    const B = ticks.length ? ticks[ticks.length - 1][0] : 0;
    let work = 0, held = 0, n = 0, from = null;
    const wins = [];
    for (let i = 1; i < ticks.length; i++) {
      const [t, th] = ticks[i], [tp, thp] = ticks[i - 1];
      if (tp >= A) {
        if (from === null) from = tp;
        work += th * (t - tp);
        held += thp * (t - tp);
        n++;
        wins.push([t - tp, th]);
      }
    }
    const span = from === null ? 0 : B - from;
    const ln = lines.filter(([t]) => t > A && t <= B).map(([, v]) => v);
    const ok = shares.filter((s) => s.t >= from && s.t < B && !s.why).length;
    const bad = shares.filter((s) => s.t >= from && s.t < B && s.why).length;
    const w = 2 ** (256 - bits);
    wins.sort((a, b) => b[0] - a[0]);
    const durs = wins.map((x) => x[0]).sort((a, b) => a - b);
    const out = {
      secs, bits, spanS: +(span / 1000).toFixed(1), ticks: n, jobs,
      work: +(work / span).toFixed(3), held: +(held / span).toFixed(3),
      lines: ln.length ? +(ln.reduce((s, v) => s + v, 0) / ln.length).toFixed(3) : null, nLines: ln.length,
      credited: +(ok * w / (span / 1000) / 1e12).toFixed(3), ok, bad, errPct: ok ? +(100 / Math.sqrt(ok)).toFixed(2) : null,
      tickMedianMs: durs.length ? durs[durs.length >> 1] : null,
      longest: wins.slice(0, 6).map(([d, th]) => d + 'ms@' + th.toFixed(1)),
    };
    console.log('RATECHECK ' + JSON.stringify(out));
    process.exit(0);
  }, secs * 1000);
});
EOF_RC
  # goal-rdna4y: the stand-in pool's job headers are a fixed byte after the sequence number, and our salts start at
  # 0, so two fresh runs mine the same shares. G_HDRFILL=1 gives each run its own header byte.
  if [ -n "${G_HDRFILL:-}" ]; then
    sed -i 's/header.fill(0x5A, 4);/header.fill(Number(process.env.PEARL_HDR_FILL || 0x5A), 4);/' $R/earn/native/probes/verify-pool.js
    say "ratecheck header fill patched: $(grep -c PEARL_HDR_FILL $R/earn/native/probes/verify-pool.js)"
  fi
  for ((i = 1; i <= ${G_RCTURNS:-2}; i++)); do
    [ -n "${G_HDRFILL:-}" ] && export PEARL_HDR_FILL=$(( (RANDOM % 254) + 1 )) && say "ratecheck $i header byte $PEARL_HDR_FILL"
    stop_miners
    say "ratecheck $i start ${G_RCSECS:-600}s at 2^${G_CBITS:-211}"
    ( cd $R/earn && NODE_PATH=$NM/node_modules PEARL_JOB_EVERY=${G_JOBEVERY:-44} timeout $(( ${G_RCSECS:-600} + 120 )) node /root/ratecheck.js $OURS_CORE ${G_RCSECS:-600} ${G_CBITS:-211} ) > /root/rc$i.log 2>&1
    if grep -q RATECHECK /root/rc$i.log; then say "ratecheck $i $(grep -m1 RATECHECK /root/rc$i.log | cut -c1-700)"
    else say "ratecheck $i FAIL $(tail -n 4 /root/rc$i.log | tr '\n' '|' | cut -c1-400)"; fi
    grep -E 'k trial|k [0-9]+|fold' /root/rc$i.log | head -n 4 | while IFS= read -r l; do say "ratecheck $i log | ${l:0:250}"; done
  done
fi


# ---- prof (goal-rdna4y): rocprofv3 (or rocprof) on each miner, observe only -------------------------------------
# Kernel trace (names, grid, workgroup, LDS, VGPRs, SGPRs, scratch, durations as the runtime reports them) and, if
# the profiler can read counters through Salad's WSL2 bridge, G_PMC's counter sets. No code object is dumped.
if has prof; then
  command -v python3 >/dev/null || apt-get install -y -qq --no-install-recommends python3 > /root/apt3.log 2>&1
  apt-get install -y -qq --no-install-recommends libdw1 libelf1 > /root/apt-dw.log 2>&1; say "prof libdw: $(ls /usr/lib/x86_64-linux-gnu/libdw.so.1 2>&1 | cut -c1-100)"
  PV=$(command -v rocprofv3 || ls /opt/rocm/bin/rocprofv3 2>/dev/null)
  if [ -z "$PV" ]; then
    apt-get install -y -qq --no-install-recommends rocprofiler-sdk > /root/apt-prof.log 2>&1
    PV=$(command -v rocprofv3 || ls /opt/rocm/bin/rocprofv3 2>/dev/null)
  fi
  say "prof rocprofv3: ${PV:-missing} $(tail -n 1 /root/apt-prof.log 2>/dev/null | cut -c1-120)"
  if [ -n "$PV" ]; then
    $PV -L > /root/prof-list.log 2>&1
    say "prof list rc=$? lines=$(wc -l < /root/prof-list.log): $(grep -oE '\b(SQ|GRBM|TA|TCP|TCC|GL2C|SQC|GRBM)_[A-Z0-9_]+' /root/prof-list.log | sort -u | tr '\n' ' ' | cut -c1-3000)"
    tail -n 3 /root/prof-list.log | while IFS= read -r l; do say "prof list | ${l:0:300}"; done
    # One run: prof_run LABEL SECS ARGS... ; the miner command is the rest.
    prof_run() {
      local label=$1 secs=$2; shift 2
      local d=/root/prof-$label; rm -rf $d; mkdir -p $d
      stop_miners
      ( cd ${PDIR:-/root} && exec timeout -s INT $secs $PV "$@" -d $d -o out --output-format csv ) > $d.log 2>&1
      local rc=$?
      sleep 5; stop_miners
      say "prof $label rc=$rc files: $(cd $d && find . -name '*.csv' | tr '\n' ' ' | cut -c1-300)"
      tail -n 4 $d.log | while IFS= read -r l; do say "prof $label log | ${l:0:300}"; done
      local kt=$(find $d -name '*kernel_trace.csv' | head -1)
      if [ -n "$kt" ]; then
        say "prof $label kt header: $(head -n 1 $kt | cut -c1-600)"
        # Unique kernels with their resources, and the duration stats.
        python3 - "$kt" <<'EOF_PY' 2>&1 | while IFS= read -r l; do say "prof $label kt | ${l:0:600}"; done
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.OrderedDict()
for r in rows:
    k = r.get("Kernel_Name", "?")[:80]
    by.setdefault(k, []).append(r)
for k, rs in by.items():
    r = rs[0]
    ds = sorted(int(x["End_Timestamp"]) - int(x["Start_Timestamp"]) for x in rs)
    keys = ["Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z", "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
            "LDS_Block_Size", "Lds_Size", "Group_Segment_Size", "Private_Segment_Size", "Scratch_Size", "Arch_VGPR_Count",
            "Accum_VGPR_Count", "SGPR_Count", "VGPR_Count"]
    res = " ".join(f"{x}={r[x]}" for x in keys if x in r)
    tot = sum(ds)
    print(f"{k} n={len(rs)} med_us={ds[len(ds)//2]/1e3:.1f} max_us={ds[-1]/1e3:.1f} total_ms={tot/1e6:.1f} {res}")
if rows:
    t0 = min(int(r["Start_Timestamp"]) for r in rows); t1 = max(int(r["End_Timestamp"]) for r in rows)
    busy = sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in rows)
    print(f"span_s={(t1-t0)/1e9:.2f} busy_s={busy/1e9:.2f} kernels={len(rows)}")
    # gaps between consecutive kernels (sorted by start), the 6 longest
    ss = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r.get("Kernel_Name", "?")[:30]) for r in rows)
    gaps = sorted(((ss[i+1][0] - ss[i][1], ss[i][2], ss[i+1][2], (ss[i][1]-t0)/1e9) for i in range(len(ss)-1)), reverse=True)[:6]
    print("gaps_ms " + "; ".join(f"{g/1e6:.1f} after {a} before {b} at {t:.1f}s" for g, a, b, t in gaps))
EOF_PY
      fi
      local cc=$(find $d -name '*counter_collection.csv' | head -1)
      if [ -n "$cc" ]; then
        say "prof $label cc header: $(head -n 1 $cc | cut -c1-400)"
        python3 - "$cc" <<'EOF_PY' 2>&1 | while IFS= read -r l; do say "prof $label cc | ${l:0:600}"; done
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    k = r.get("Kernel_Name", "?")[:60]
    agg[k][r.get("Counter_Name", "?")].append(float(r.get("Counter_Value", 0) or 0))
for k, cs in agg.items():
    print(k + " " + " ".join(f"{c}={sum(v)/len(v):.4g}" for c, v in sorted(cs.items())))
EOF_PY
      fi
    }
    PSECS=${G_PSECS:-40}
    # ours: hashrate.js on the default build (the miner's own fold and hash launches)
    PDIR=$R/earn/native/amd prof_run ours-kt $((PSECS + 20)) --kernel-trace --stats -- env PEARL_AMD_UNTESTED=1 node ../probes/hashrate.js $OURS_CORE $PSECS 5
    if [ -n "$SRBV" ]; then
      PDIR=/opt/srb-$SRBV prof_run srb-kt $((PSECS + 30)) --kernel-trace --stats -- ./SRBMiner-MULTI $(srb_args)
    fi
    IFS=';' read -ra PMCS <<< "${G_PMC:-SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS;SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_INST_ANY SQ_INSTS_VMEM}"
    j=0
    for p in "${PMCS[@]}"; do
      j=$((j + 1))
      PDIR=$R/earn/native/amd prof_run ours-pmc$j $((PSECS / 2 + 20)) --pmc $p -- env PEARL_AMD_UNTESTED=1 node ../probes/hashrate.js $OURS_CORE $((PSECS / 2)) 5
      [ -n "$SRBV" ] && PDIR=/opt/srb-$SRBV prof_run srb-pmc$j $((PSECS / 2 + 30)) --pmc $p -- ./SRBMiner-MULTI $(srb_args)
    done
  fi
fi


# ---- mtrace (goal-rdna4y): ours mining on DUEL_POOL with PEARL_AMD_TRACE=1, the core's trace lines as they come
if has mtrace; then
  stop_miners
  ( export PEARL_AMD_TRACE=1; ours_start ) > /root/mtrace.log 2>&1 < /dev/null &
  mp=$!
  sleep ${G_TRSECS:-300}
  kill $mp 2>/dev/null; sleep 5; stop_miners
  tr '\r' '\n' < /root/mtrace.log | grep '\[pearl-amd\]' | tail -n ${G_TRN:-30} | while IFS= read -r l; do say "mtrace | ${l:0:400}"; done
  tr '\r' '\n' < /root/mtrace.log | grep -oE '[0-9.]+ TH/s' | awk '{print $1}' > /root/mtrace.th
  say "mtrace TH/s: $(stats /root/mtrace.th)"
fi


# ---- cfg (goal-rdna4y): what each miner declares in its shares (m, n, k, rank, tile rows and columns), read by the
# stand-in pool from the shares it accepts; and amd-smi's whole metric list while each mines.
if has cfg; then
  command -v python3 >/dev/null || apt-get install -y -qq --no-install-recommends python3 > /root/apt3.log 2>&1
  python3 - "$R/earn/native/probes/verify-pool.js" <<'EOF_PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "            const pp = decodePlainProof(p.plain_proof, { gzip });\n"
assert a in s
b = a + "            if (process.env.PEARL_CFG_LOG && (stats.cfgN = (stats.cfgN || 0) + 1) <= 3) log('cfg ' + JSON.stringify({ m: pp.m, n: pp.n, k: pp.k, r: pp.noiseRank, h: pp.a.rowIndices.length, w: pp.bt.rowIndices.length, rows: pp.a.rowIndices, cols: pp.bt.rowIndices }));\n"
open(p, 'w').write(s.replace(a, b))
EOF_PY
  for nm in srb ours; do
    ( cd $R/earn/native/probes && PEARL_CFG_LOG=1 PEARL_JOB_EVERY=600 exec node verify-pool.js serve 3335 864000 ${G_CBITS:-211} ) > /root/cfg.out 2> /root/cfg-$nm.err &
    CFP=$!; sleep 3
    keep_pool=$DUEL_POOL; DUEL_POOL=127.0.0.1:3335
    ( ${nm}_start ) > /root/cfg-$nm.log 2>&1 < /dev/null &
    mp=$!
    sleep 120
    timeout 20 /opt/rocm-wsl/bin/amd-smi metric > /root/smi-$nm.txt 2>&1
    grep -v 'N/A' /root/smi-$nm.txt | tr -s ' ' | grep -iE 'mem|umc|vram|bw|bandwidth|activity|clk|power|volt|pcie|xgmi|fabric|fclk|soc' | head -40 | while IFS= read -r l; do say "cfg $nm smi | ${l:0:200}"; done
    sleep 60
    kill $mp 2>/dev/null; sleep 4; stop_miners
    DUEL_POOL=$keep_pool
    kill $CFP 2>/dev/null; wait $CFP 2>/dev/null
    grep ' cfg ' /root/cfg-$nm.err | head -3 | while IFS= read -r l; do say "cfg $nm | ${l:0:900}"; done
    say "cfg $nm shares: $(grep -c 'share k=' /root/cfg-$nm.err) ($(grep 'share k=' /root/cfg-$nm.err | awk '{print $3}' | sort | uniq -c | tr '\n' ' '))"
  done
fi

# ---- our builds ------------------------------------------------------------------------------------------------
if has build; then
  mkdir -p $SW
  if [ -n "${G_CODE:-}" ]; then tree "$G_CODE" || { say "job done: no tree"; exit 1; }; fi
  SWEEP="$SWEEPV" SWEEP_FOLD= SWEEP_ROUNDS=0 SWEEP_ROOT=$SW bash test/sweep.sh > /root/sweep.log 2>&1
  grep -E ' sweep .*(FAIL|not timed)' /root/sweep.log | head -8 | while IFS= read -r l; do say "build | ${l:0:300}"; done
  say "builds at $(git -C $R rev-parse --short=12 HEAD): $(for d in $SW/var/*/; do echo "$(basename $d) $(md5sum $d/pearl_core_hip.node | cut -c1-6)"; done | tr '\n' ' ')"
fi

# test/pearl_amd_peak.cpp on this card: the WMMA alone, with VALU, with the fold's readout, with LDS reads, and the
# no-LDS loads; each a TMAC/s and the clock it ran at. G_PEAK: its arguments (seconds, then a variant name or none).
if has peak; then
  if /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 \
       test/pearl_amd_peak.cpp -o /root/peak -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/peak-build.log 2>&1; then
    for pk in ${G_PEAKONLY:-all}; do
      if [ "$pk" = all ]; then a=""; else a="$pk"; fi
      timeout 1200 /root/peak ${G_PEAKSECS:-2.5} $a > /root/peak.log 2>&1
      grep -E '^(peak|device)' /root/peak.log | tr -s ' ' | while IFS= read -r l; do say "peak | ${l:0:300}"; done
    done
  else
    say "peak build FAIL: $(grep -m2 error /root/peak-build.log | tr '\n' '|' | cut -c1-250)"
  fi
fi

# asmtest (goal-rdna4f): which instructions this card's assembler takes. G_ASM: lines separated by "|".
if has asmtest; then
  IFS='|' read -ra AL <<< "${G_ASM:-}"
  for l in "${AL[@]}"; do
    o=$(printf '%s\n' "$l" | /opt/rocm/lib/llvm/bin/llvm-mc -arch=amdgcn -mcpu="$ARCHX" -show-encoding 2>&1 | grep -vE '^\s*\.text' | tr '\n' ' ' | tr -s ' ')
    say "asm [$l] -> ${o:0:220}"
  done
fi

# test/pearl_amd_cost.cpp (goal-rdna4f): what one instruction beside the WMMAs costs. Fetched at G_CODE, so no build.
# G_COSTONLY: name filters (all if unset); G_COSTBPW: blocks a WGP (default 4); G_COSTSECS.
if has cost; then
  if curl -fsSL --retry 2 -m 60 "https://raw.githubusercontent.com/super3/llmjob/${G_CODE}/earn/native/amd/test/pearl_amd_cost.cpp" -o /root/cost.cpp \
     && /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 \
       /root/cost.cpp -o /root/cost -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/cost-build.log 2>&1; then
    for bpw in ${G_COSTBPW:-4}; do
      for pk in ${G_COSTONLY:-all}; do
        if [ "$pk" = all ]; then a=""; else a="$pk"; fi
        COST_BPW=$bpw timeout 1500 /root/cost ${G_COSTSECS:-2} $a > /root/cost.log 2>&1
        grep -E '^(cost|device)' /root/cost.log | tr -s ' ' | while IFS= read -r l; do say "cost | ${l:0:300}"; done
        grep -vE '^(cost|device)' /root/cost.log | head -3 | while IFS= read -r l; do say "cost err | ${l:0:300}"; done
      done
    done
    if [ -n "${G_COSTISA:-}" ]; then
      /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 \
        --cuda-device-only -S -c /root/cost.cpp -o /root/cost.s > /root/costisa.err 2>&1 || say "cost isa FAIL"
      for kn in $G_COSTISA; do
        # kn: a mangled-name fragment such as ILi1ELi8ELi2ELi0E (K, N, PLACE, BAR); the loop's lines, compressed
        awk -v k="$kn" '/^_Z[^ ]*:/ && index($0, k) { on = 1; next } on && /^\.Lfunc_end/ { on = 0 } on' /root/cost.s > /root/costk.s
        say "cost isa $kn: $(awk '/^\.LBB/ { if (n > best) { best = n; out = cur }; cur = ""; n = 0 } $1 ~ /^(v_|ds_|s_wait|s_barrier|s_delay|s_nop|buffer_|s_add|s_cbranch)/ { o = $1; sub(/^v_wmma.*/, "W", o); cur = cur " " o; if ($1 ~ /^v_wmma/) n++ } END { if (n > best) out = cur; print out }' /root/costk.s | tr ' ' '\n' | uniq -c | awk '{printf "%s%s ", ($1 > 1 ? $1 "x" : ""), $2}' | cut -c1-1500)"
      done
    fi
  else
    say "cost build FAIL: $(grep -m3 error /root/cost-build.log | tr '\n' '|' | cut -c1-400)"
  fi
fi

# test/pearl_amd_lay64.cpp (goal-rdna4b): the wave64 WMMA's operand and result layout.
if has lay64; then
  if curl -fsSL --retry 2 -m 60 "https://raw.githubusercontent.com/super3/llmjob/${G_P64SHA}/earn/native/amd/test/pearl_amd_lay64.cpp" -o /root/lay64.cpp      && /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" -mwavefrontsize64 --rocm-path=/opt/rocm -std=c++17 -O2         /root/lay64.cpp -o /root/lay64 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/lay64-build.log 2>&1; then
    timeout 120 /root/lay64 2>&1 | head -40 | while IFS= read -r l; do say "${l:0:330}"; done
  else
    say "lay64 build FAIL: $(grep -m2 error /root/lay64-build.log | tr '
' '|' | cut -c1-300)"
  fi
fi

# test/pearl_amd_peak64.cpp (goal-rdna4b): the wave64 WMMA probe, fetched at G_P64SHA, built with -mwavefrontsize64.
if has peak64; then
  curl -fsSL --retry 2 -m 60 "https://raw.githubusercontent.com/super3/llmjob/${G_P64SHA}/earn/native/amd/test/pearl_amd_peak64.cpp" -o /root/peak64.cpp \
    || say "peak64 fetch FAIL"
  if /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" -mwavefrontsize64 --rocm-path=/opt/rocm -std=c++17 -O3 \
       /root/peak64.cpp -o /root/peak64 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/peak64-build.log 2>&1; then
    if [ -n "${G_P64ISA:-}" ]; then
      /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" -mwavefrontsize64 --rocm-path=/opt/rocm -std=c++17 -O3 \
        --cuda-device-only -S -c /root/peak64.cpp -o /root/peak64.s > /root/p64isa.err 2>&1 || say "peak64 isa FAIL: $(grep -m2 error /root/p64isa.err | tr "
" "|" | cut -c1-300)"
      say "peak64 isa: $(wc -c < /root/peak64.s) bytes, $(grep -c p64_kernel /root/peak64.s) lines name a kernel; $(grep -m3 -E '^[_a-zA-Z].*p64_kernel.*:' /root/peak64.s | cut -c1-90 | tr '
' ' ')"
      for kn in $G_P64ISA; do
        awk -v k="$kn" '/^_Z[^ ]*:/ && index($0, k) { on = 1; next } on && /^\.Lfunc_end/ { on = 0 } on' /root/peak64.s > /root/p64k.s
        say "peak64 isa $kn loop: $(awk '/^\.LBB/ { lab = $0 } { print }' /root/p64k.s | awk '/^\.LBB/ { if (n > best) { best = n; out = cur }; cur = ""; n = 0 } $1 ~ /^(v_wmma|ds_|s_wait|s_barrier|v_|s_cbranch)/ { o = $1; sub(/^v_wmma.*/, "W", o); sub(/^ds_load_b64/, "L", o); cur = cur " " o; if ($1 ~ /^v_wmma/) n++ } END { if (n > best) out = cur; print out }' | tr ' ' '
' | uniq -c | awk '{printf "%s%s ", ($1 > 1 ? $1 "x" : ""), $2}' | cut -c1-1400)"
        say "peak64 isa $kn: $(awk '$1 ~ /^[vsdbg][a-z0-9_]+$/ { n[$1]++ } END { for (o in n) print n[o], o }' /root/p64k.s | sort -rn | head -n 20 | awk '{ printf "%s %s, ", $2, $1 }')"
      done
      grep -E '^_Z.*p64_kernel.*:$' /root/peak64.s | head -12 | while IFS= read -r l; do say "peak64 sym ${l:0:200}"; done
    fi
    if [ -n "${G_P32ONLY:-}" ] && curl -fsSL --retry 2 -m 60 "https://raw.githubusercontent.com/super3/llmjob/${G_P64SHA}/earn/native/amd/test/pearl_amd_peak.cpp" -o /root/peak32.cpp         && /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 /root/peak32.cpp -o /root/peak32            -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/peak32-build.log 2>&1; then
      for pk in $G_P32ONLY; do
        timeout 1200 /root/peak32 ${G_PEAKSECS:-2.5} $pk > /root/peak32.log 2>&1
        grep -E '^peak' /root/peak32.log | tr -s ' ' | while IFS= read -r l; do say "p32 | ${l:0:300}"; done
      done
    elif [ -n "${G_P32ONLY:-}" ]; then say "peak32 FAIL: $(grep -m2 error /root/peak32-build.log | tr '
' '|' | cut -c1-300)"; fi
    if [ -n "${G_P64T512:-}" ]; then
      /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" -mwavefrontsize64 --rocm-path=/opt/rocm -std=c++17 -O3 -DP64_T=512         /root/peak64.cpp -o /root/peak64t -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/peak64t-build.log 2>&1 || say "peak64 T512 build FAIL"
    fi
    for pk in ${G_P64ONLY:-all}; do
      if [ "$pk" = all ]; then a=""; else a="$pk"; fi
      [ -x /root/peak64t ] && timeout 1200 /root/peak64t ${G_PEAKSECS:-2.5} $a 2>&1 | grep -E '^peak' | tr -s ' ' | while IFS= read -r l; do say "p64 | ${l:0:300}"; done
      timeout 1200 /root/peak64 ${G_PEAKSECS:-2.5} $a > /root/peak64.log 2>&1
      grep -E '^(peak|device)' /root/peak64.log | tr -s ' ' | while IFS= read -r l; do say "p64 | ${l:0:300}"; done
      grep -vE '^(peak|device)' /root/peak64.log | head -4 | while IFS= read -r l; do say "p64 err | ${l:0:300}"; done
    done
  else
    say "peak64 build FAIL: $(grep -m3 error /root/peak64-build.log | tr '\n' '|' | cut -c1-400)"
  fi
fi

# The wmma12 fold's instructions for each variant in G_ISA (names from G_SWEEP): the device pass compiled to assembly
# as build-amd.sh check compiles it. One line of counts over the whole kernel, and one over the hot loop (the
# stretch from a label to the last branch back to it with the most v_wmma).
if has isa; then
  IFS=';' read -ra RV <<< "$SWEEPV"
  for want in ${G_ISA:-base}; do
    fl=""
    for v in "${RV[@]}"; do [ "${v%%=*}" = "$want" ] && fl="${v#*=}"; done
    /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 $fl \
      -I$A/compat/gpu -I$A/src -I$R/earn/native/src -I$A/build/gen --cuda-device-only -S \
      -c "$A/src/pearl_host_hip.cpp" -o /root/isa-$want.s > /root/isa-err.log 2>&1 \
      || { say "isa $want: compile failed: $(grep -m2 error /root/isa-err.log | tr '\n' '|' | cut -c1-250)"; continue; }
    awk '/^pearl_amd_fold_wmma12:/ { on = 1; next } on && /^\.Lfunc_end/ { on = 0 } on' /root/isa-$want.s > /root/isa-$want.k
    say "isa $want kernel: $(awk '$1 ~ /^[vsdbg][a-z0-9_]+$/ { n[$1]++ } END { for (o in n) print n[o], o }' /root/isa-$want.k | sort -rn | head -n 24 | awk '{ printf "%s %s, ", $2, $1 }')"
    awk '
      { line[NR] = $0; op[NR] = $1
        if ($0 ~ /^\.LBB[0-9_]+:/) { l = $0; sub(/:.*/, "", l); lab[l] = NR }
        if ($1 ~ /^s_cbranch|^s_branch/) tgt[NR] = $2 }
      END {
        best = 0; bs = 0; be = 0
        for (i in tgt) { t = tgt[i]; if ((t in lab) && lab[t] < i) { s = 0; for (j = lab[t]; j <= i; j++) if (op[j] ~ /^v_wmma/) s++
            if (s > best || (s == best && i - lab[t] > be - bs)) { best = s; bs = lab[t]; be = i } } }
        tot = 0
        for (j = bs; j <= be; j++) if (op[j] ~ /^[vsdbg][a-z0-9_]+$/) { n[op[j]]++; tot++ }
        printf "loop: %d instructions, %d v_wmma; ", tot, best
        for (j = bs; j <= be; j++) print line[j] > "/root/isa-loop-lines.txt"
        for (o in n) printf "%d %s\n", n[o], o > "/root/isa-loop.txt"
      }' /root/isa-$want.k > /root/isa-loop-head.txt 2>&1
    say "isa $want $(cut -c1-80 /root/isa-loop-head.txt)$(sort -rn /root/isa-loop.txt 2>/dev/null | head -n 26 | awk '{ printf "%s %s, ", $2, $1 }')"
    if [ -n "${G_ISASEQ:-}" ]; then
      # the loop as one compressed sequence: W for a WMMA, other ops by name, runs counted
      seq=$(awk '{ o = $1; if (o ~ /^v_wmma/) o = "W"; if (o ~ /^(;|\.)/ && o !~ /^\.LBB/) next; if (o != "") print o }' /root/isa-loop-lines.txt | uniq -c | awk '{ printf "%s%s ", ($1 > 1 ? $1 "x" : ""), $2 }')
      for ((p = 0; p < ${#seq}; p += 300)); do say "isaseq $want | ${seq:$p:300}"; sleep 0.3; done
    fi
    if [ -n "${G_ISADUMP:-}" ]; then
      grep -E "^[[:space:]]+(v_|ds_|buffer_|s_|.LBB)|^.LBB" /root/isa-loop-lines.txt | grep -vE "${G_ISAX:-v_wmma}" | tr -s ' 	' ' ' | head -n ${G_ISADUMP} | while IFS= read -r l; do say "isa $want | ${l:0:160}"; sleep 0.3; done
    fi
    rm -f /root/isa-loop.txt /root/isa-loop-lines.txt
  done
fi

# Registers, LDS and waves of each sweep variant's wmma12 fold (build-amd.sh check for this card's target).
if has regs; then
  IFS=';' read -ra RV <<< "$SWEEPV"
  for v in "${RV[@]}"; do
    [ -n "$v" ] || continue
    r=$(PEARL_AMD_CFLAGS="${v#*=}" timeout 600 ./build-amd.sh check "$ARCHX" 2>&1 | grep -E 'wmma12|failed' | tr -s ' ' | head -n 2 | tr '
' ';')
    say "regs ${v%%=*}: ${r:0:300}"
    case "$r" in *"compile failed"*) grep -m3 -E "error" build/check/$ARCHX.txt 2>/dev/null | while IFS= read -r l; do say "regs ${v%%=*} | ${l:0:280}"; done ;; esac
  done
fi

# The fold alone (one queue, launches back to back) beside a ticker: the clock probe's kernel from another process
# every G_TICKS seconds (0: no ticker). Does a second busy queue change what the fold's queue gets?
if has tick && [ -x /root/clkprobe ] && [ -x $SW/var/base/pearl_amd_foldtest ]; then
  for ((i = 1; i <= ${G_TROUNDS:-3}; i++)); do
    line=""
    for tk in ${G_TICKS:-0 0.010 0.002}; do
      cp=""
      if [ "$tk" != 0 ]; then ( exec /root/clkprobe "$tk" 100000000 ) > /root/tick.log 2>&1 & cp=$!; sleep 1; fi
      r=$(timeout 900 $SW/var/base/pearl_amd_foldtest ${G_TICKSHAPE:-131072 2048 1 400} 0 "$FOLD" "${G_FOLDK:-0}" miner 2>&1 | grep -E "^$FOLD" | sed 's/  */ /g')
      n=""
      if [ -n "$cp" ]; then kill $cp 2>/dev/null; wait $cp 2>/dev/null; n=", $(grep -c '^clk ' /root/tick.log) ticks, last $(grep '^clk ' /root/tick.log | tail -n 1 | cut -c1-60)"; fi
      line="$line; every ${tk}s: $(printf '%s\n' "$r" | sed -nE 's/.* ([0-9.]+) TMAC\/s$/\1/p') TMAC/s$n"
    done
    say "tick round $i$line"
  done
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
      r=$(timeout 900 $SW/var/$FV/pearl_amd_foldtest "$FM" "$FC" 1 "$FR" 0 "$FOLD" "${G_FOLDK:-0}" miner 2>&1 | grep -E "^$FOLD" | sed 's/  */ /g')
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
// Between the launch reaching full occupancy and leaving it for the last time: the share of slot time in use.
let area = 0, last = null;
c = 0;
for (const e of ev) {
  if (last !== null && e[0] > full0 && last < full1) area += c * (Math.min(e[0], full1) - Math.max(last, full0));
  c += e[1]; last = e[0];
}
const steady = full1 > full0 ? area / (peak * (full1 - full0)) : NaN;
const durs = lead.map((w) => (w.end - w.start) * msPerTick).sort((a, b) => a - b);
const f = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '-');
const r = s.rest || {};
console.log('tl: ' + lead.length + ' blocks, ' + peak + ' at once; launch ' + f(launchMs) + ' ms, its blocks span '
  + f(span) + ' ms, gap ' + f(launchMs - span) + ' ms (' + f(100 * (launchMs - span) / launchMs, 2) + '%); ramp '
  + f((full0 - t0) * msPerTick) + ' ms, tail ' + f((t1 - full1) * msPerTick) + ' ms; busy time over ' + peak + ' slots '
  + f(ideal) + ' ms (' + f(100 * ideal / launchMs, 2) + '% of the launch); slots in use between ramp and tail '
  + f(100 * steady, 2) + '%; block ' + f(durs[durs.length >> 1], 4)
  + ' ms median; ' + (r.mhz ? Math.round(r.mhz.med) + ' MHz, slab ' + Math.round(r.slab.med) + ' cycles' : ''));
EOF_TL
cat > /root/tl2.js <<'EOF_TL2'
'use strict';
// tl2.js <stamp-summary.js> <stamps file>: where a launch's slot time went, from every wave's start and end on
// REALTIME (pearl_amd_foldtest's stamps file, the last launch).
//   stalls  periods with no block starting although blocks remained: how many, how long, when
//   moved   waves whose HW_ID1 changed between start and end (the wave was taken off its SIMD and put back)
//   use     the share of the span at full, part and no occupancy, and the busiest and idlest WGP
const fs = require('fs');
const ss = require(process.argv[2]);
const data = ss.load(fs.readFileSync(process.argv[3]));
const tickMhz = (data.head.wallKhz || 100000) / 1000;
const ms = (ticks) => ticks / (tickMhz * 1e3);
const all = data.waves.filter((w) => w.bits > 0);
if (!all.length) { console.log('tl2: no waves'); process.exit(0); }
const ref = all[0].rt0;
for (const w of all) {
  w.s = (w.rt0 - ref) | 0;
  w.e = w.s + ((w.rt1 - w.rt0) >>> 0);
}
const base = all.reduce((m, w) => Math.min(m, w.s), Infinity);
for (const w of all) { w.s -= base; w.e -= base; }
const lead = all.filter((w) => w.wave === 0).sort((a, b) => a.s - b.s);
const span = lead.reduce((m, w) => Math.max(m, w.e), 0);
const f = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '-');
const q = (v, p) => (v.length ? v[Math.min(v.length - 1, Math.round(p * (v.length - 1)))] : NaN);

// Stalls: gaps with no block starting and none ending. Blocks start and end about every 0.001 ms when the card is
// full and nothing holds it up.
const STALL = 0.05;   // ms
const times = [];
for (const w of lead) times.push(w.s, w.e);
times.sort((x, y) => x - y);
const gaps = [];
for (let i = 1; i < times.length; i++) {
  const g = ms(times[i] - times[i - 1]);
  if (g >= STALL) gaps.push({ at: ms(times[i - 1]), g });
}
const stallMs = gaps.reduce((s, x) => s + x.g, 0);
const long = gaps.filter((x) => x.g >= 0.3);
console.log('tl2 stalls: ' + lead.length + ' blocks over ' + f(ms(span)) + ' ms; ' + gaps.length + ' gaps of ' + STALL
  + ' ms or more with no block starting or ending, ' + f(stallMs) + ' ms in all (' + f(100 * stallMs / ms(span), 1) + '% of the span); '
  + long.length + ' of 0.3 ms or more: ' + long.slice(0, 14).map((x) => f(x.g, 2) + ' ms at ' + f(x.at, 1)).join(', '));
// Every gap, as its length at its time in the launch (ms), the first 24: is there a pattern?
if (process.argv[4] === 'gaps') {
  console.log('tl2 gaps: ' + gaps.slice(0, 24).map((x) => f(x.g, 3) + '@' + f(x.at, 2)).join(' '));
}
if (long.length > 1) {
  const between = [];
  for (let i = 1; i < long.length; i++) between.push(long[i].at - long[i - 1].at);
  between.sort((a, b) => a - b);
  console.log('tl2 stalls: from one long gap to the next ' + f(q(between, 0.5), 2) + ' ms median (min ' + f(between[0], 2)
    + ', max ' + f(between[between.length - 1], 2) + '): ' + f(1000 / q(between, 0.5), 0) + ' a second; a long gap is '
    + f(q(long.map((x) => x.g).sort((a, b) => a - b), 0.5), 2) + ' ms median');
}

// Moved waves.
const moved = all.filter((w) => w.hw0 !== w.hw1);
const stay = all.filter((w) => w.hw0 === w.hw1);
const dur = (v) => v.map((w) => ms(w.e - w.s)).sort((a, b) => a - b);
const dm = dur(moved), dsv = dur(stay);
const movedBlocks = new Set(moved.map((w) => w.block)).size;
// When the moved waves ended, in clusters 0.2 ms apart or more.
const ends = moved.map((w) => ms(w.e)).sort((a, b) => a - b);
const clusters = [];
for (const t of ends) {
  const c = clusters[clusters.length - 1];
  if (c && t - c.last < 0.2) { c.last = t; c.n++; } else clusters.push({ first: t, last: t, n: 1 });
}
console.log('tl2 moved: ' + moved.length + ' of ' + all.length + ' waves changed HW_ID1 (' + movedBlocks + ' blocks); they ran '
  + f(q(dm, 0.5)) + ' ms median (max ' + f(dm[dm.length - 1]) + '), the others ' + f(q(dsv, 0.5)) + ' (p99 ' + f(q(dsv, 0.99))
  + ', max ' + f(dsv[dsv.length - 1]) + '); ' + clusters.length + ' groups of them, ending at '
  + clusters.slice(0, 12).map((c) => f(c.first, 1) + ' (' + c.n + ')').join(', '));

// Occupancy over time, from every block's lead wave.
const ev = [];
for (const w of lead) ev.push([w.s, 1], [w.e, -1]);
ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
let c = 0, peak = 0;
for (const e of ev) { c += e[1]; if (c > peak) peak = c; }
const share = { full: 0, part: 0, low: 0, none: 0 };
c = 0;
let last = 0;
for (const e of ev) {
  const dt = e[0] - last;
  if (c >= 0.97 * peak) share.full += dt; else if (c >= 0.5 * peak) share.part += dt; else if (c > 0) share.low += dt; else share.none += dt;
  c += e[1]; last = e[0];
}
// With "prof" as an argument: the mean number of blocks running in each millisecond of the span (the first 70), and
// the blocks that started in it, so a stop and what follows it can be read off.
if (process.argv.includes('prof')) {
  const tpm = tickMhz * 1e3;   // ticks a millisecond
  const n = Math.min(70, Math.ceil(span / tpm));
  const area = new Array(n).fill(0), starts = new Array(n).fill(0);
  c = 0; last = 0;
  for (const e of ev) {
    for (let b = Math.floor(last / tpm); b <= Math.floor(e[0] / tpm) && b < n; b++) {
      const lo = Math.max(last, b * tpm), hi = Math.min(e[0], (b + 1) * tpm);
      if (hi > lo) area[b] += c * (hi - lo);
    }
    if (e[1] > 0 && Math.floor(e[0] / tpm) < n) starts[Math.floor(e[0] / tpm)]++;
    c += e[1]; last = e[0];
  }
  console.log('tl2 prof running: ' + area.map((a) => Math.round(a / tpm)).join(' '));
  console.log('tl2 prof starts: ' + starts.join(' '));
}
const wgpOf = (h) => (((h >>> 18) & 7) << 5) | (((h >>> 16) & 1) << 4) | ((h >>> 10) & 15);
const busy = new Map();
for (const w of lead) busy.set(wgpOf(w.hw0), (busy.get(wgpOf(w.hw0)) || 0) + (w.e - w.s));
const per = [...busy.values()].map((b) => b / span / 3).sort((a, b) => a - b);
console.log('tl2 use: ' + peak + ' blocks at once at most; of the span ' + f(100 * share.full / span, 1) + '% at 97% of that or more, '
  + f(100 * share.part / span, 1) + '% at 50-97%, ' + f(100 * share.low / span, 1) + '% under 50%, ' + f(100 * share.none / span, 1)
  + '% with no block running; ' + per.length + ' WGPs, each 3 blocks: in use ' + f(100 * per[0], 1) + '% (the idlest), '
  + f(100 * q(per, 0.5), 1) + '% (median), ' + f(100 * per[per.length - 1], 1) + '% (the busiest)');
EOF_TL2
STSHAPES=${G_STSHAPES:-sbase:131072:2048:400 sbase:524288:2048:100 sbase:131072:4096:200 sw:131072:8192:100}
if has stamps; then
  for ((i = 1; i <= ${G_SROUNDS:-2}; i++)); do
    for sh in $STSHAPES; do
      IFS=: read -r FV FM FC FR <<< "$sh"
      [ -x $SW/var/$FV/pearl_amd_foldtest ] || continue
      rm -f /root/st.bin
      r=$(timeout 900 $SW/var/$FV/pearl_amd_foldtest "$FM" "$FC" 1 "$FR" 0 "$FOLD" "${G_FOLDK:-0}" miner /root/st.bin 2>&1)
      ms=$(printf '%s\n' "$r" | grep -E "^$FOLD" | sed -nE 's/.* ([0-9]+) launches, ([0-9.]+) s, .*/\1 \2/p' | awk '{printf "%.3f", 1000 * $2 / $1}')
      say "stamp r$i $FV ${FM}x${FC}: $(printf '%s\n' "$r" | grep -E "^$FOLD" | sed 's/  */ /g' | cut -c1-200)"
      [ -s /root/st.bin ] || { say "stamp r$i $FV ${FM}x${FC}: no stamps written: $(printf '%s\n' "$r" | tail -n 2 | tr '\n' '|' | cut -c1-200)"; continue; }
      node test/stamp-summary.js summary /root/st.bin 2>&1 | grep -E '^(first|rest)' | while IFS= read -r l; do say "stamp r$i $FV ${FM}x${FC} ${l:0:330}"; done
      node /root/tl.js "$A/test/stamp-summary.js" /root/st.bin "${ms:-0}" 2>&1 | while IFS= read -r l; do say "stamp r$i $FV ${FM}x${FC} ${l:0:380}"; done
      node /root/tl2.js "$A/test/stamp-summary.js" /root/st.bin ${G_PROF:+prof} 2>&1 | while IFS= read -r l; do say "stamp r$i $FV ${FM}x${FC} ${l:0:390}"; done
    done
  done
fi

# hashrate.js on each variant (VDEF and vdef above).
# hr LABEL VARIANT [K=V ...]: one run; prints "hash LABEL VARIANT th=.. sd=.. n=..".
hr() {
  local label=$1 v=$2 core envs log r; shift 2
  vdef "$v"; core=$VCORE; envs=("${VENV[@]}")
  [ -n "${G_JOBEVERY:-}" ] && envs+=("PEARL_JOB_EVERY=$G_JOBEVERY")
  # TICK=secs in a variant's settings: the ticker (the clock probe's kernel from another process) runs beside it.
  local tk="" tp="" e
  for e in "${envs[@]}"; do case $e in TICK=*) tk=${e#TICK=} ;; esac; done
  if [ -n "$tk" ] && [ -x /root/clkprobe ]; then ( exec /root/clkprobe "$tk" 100000000 ) > /root/tick.log 2>&1 & tp=$!; fi
  [ -f $SW/var/$core/pearl_core_hip.node ] || { say "hash $label $v: no build"; return; }
  # The run's CPU seconds (user+system, the whole process) beside its rate: what a ticker thread costs the PC.
  local TIMEFORMAT='cpu=%U+%S real=%R'
  log=$( { time env PEARL_AMD_UNTESTED=1 "${envs[@]}" "$@" timeout 600 node ../probes/hashrate.js $SW/var/$core/pearl_core_hip.node "$HSECS" "$HWARM" 2>&1 | grep -v nvidia-smi; } 2> /root/hr.time )
  if [ -n "$tp" ]; then kill $tp 2>/dev/null; wait $tp 2>/dev/null; fi
  r=$(printf '%s\n' "$log" | grep -E '"th"' | tail -n 1 | sed -nE 's/.*"th":([0-9.]+),"sd":([0-9.]+),"n":([0-9]+).*/th=\1 sd=\2 n=\3/p')
  [ -n "$r" ] || r="FAIL $(printf '%s\n' "$log" | tail -n 2 | tr '\n' '|' | cut -c1-240)"
  say "hash $label $v $r $(tail -n 1 /root/hr.time 2>/dev/null | cut -c1-60)"
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

# The miner's own launches, stamped: hashrate.js on a stamp build writes the stamps of one launch, the one after
# batch N (PEARL_AMD_STAMP_DUMP, pearl_host_hip.cpp), G_MSN times for each variant in G_MST, and tl2.js reads each:
# did the queue stop in that launch, and were waves moved?
if has mstamps; then
  for v in ${G_MST:-msq1 msq0}; do
    vdef "$v"
    [ -f $SW/var/$VCORE/pearl_core_hip.node ] || { say "mst $v: no build"; continue; }
    for ((i = 1; i <= ${G_MSN:-6}; i++)); do
      rm -f /root/mst.bin
      env PEARL_AMD_UNTESTED=1 PEARL_AMD_STAMP_DUMP=/root/mst.bin PEARL_AMD_STAMP_DUMP_AT=$((${G_MSAT:-400} + ${G_MSSTEP:-61} * i)) "${VENV[@]}" \
        timeout 120 node ../probes/hashrate.js $SW/var/$VCORE/pearl_core_hip.node 8 2 > /root/mst.log 2>&1
      if [ -s /root/mst.bin ]; then
        node /root/tl2.js "$A/test/stamp-summary.js" /root/mst.bin ${G_MSGAPS:+gaps} ${G_PROF:+prof} 2>&1 | grep -E 'tl2 (stalls: [0-9]+ blocks|moved|gaps|prof|use)' | sed -E 's/; they ran .*//; s/ with no block starting or ending//' | while IFS= read -r l; do say "mst $v $i ${l:0:390}"; done
      else
        say "mst $v $i: no stamps written: $(grep -v nvidia-smi /root/mst.log | tail -n 2 | tr '\n' '|' | cut -c1-240)"
      fi
    done
  done
fi

# verify-hits.js on a variant: the pool's check of every hit, 400 of them.
if has verify; then
  for v in ${G_VERIFY:-}; do
    vdef "$v"
    [ -f $SW/var/$VCORE/pearl_core_hip.node ] || { say "verify $v: no build"; continue; }
    env "${VENV[@]}" timeout 3600 node ../probes/verify-hits.js $SW/var/$VCORE/pearl_core_hip.node "${G_VSECS:-600}" > /root/verify-$v.log 2>&1
    rc=$?
    j=$(grep '^{' /root/verify-$v.log | tail -n 1)
    say "verify $v rc=$rc $(printf '%s' "$j" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);console.log(JSON.stringify({PASS:j.PASS,hits:j.hits,verified:j.verified,failed:j.failed,salts:j.salts,secs:j.secs,fold:j.device&&j.device.fold,error:j.error,fails:j.fails}).slice(0,900))}catch(e){console.log("no result: "+s.slice(-200))}})')"
    [ -n "$j" ] || tail -n 3 /root/verify-$v.log | while IFS= read -r l; do say "verify $v | ${l:0:300}"; done
  done
fi

# Which of the miner's threads keep the CPU busy: ps -L 22 s into a 30 s run, for each variant in G_THREADS.
if has threads; then
  for v in ${G_THREADS:-base hq1}; do
    vdef "$v"
    [ -f $SW/var/$VCORE/pearl_core_hip.node ] || { say "threads $v: no build"; continue; }
    env PEARL_AMD_UNTESTED=1 "${VENV[@]}" timeout 120 node ../probes/hashrate.js $SW/var/$VCORE/pearl_core_hip.node 25 5 > /root/thr.log 2>&1 &
    hp=$!
    sleep 22
    np=$(pgrep -f '^node ../probes/hashrate.js' | head -n 1)
    say "threads $v (pid ${np:-?}, $(nproc) CPUs): $(ps -L -o pcpu=,comm= -p "${np:-1}" 2>/dev/null | sort -rn | head -n 9 | awk '{printf "%s %s; ", $1, $2}')"
    wait $hp
  done
fi

# What the busy threads are doing: the same run as gdb's child, stopped 24 s in with SIGINT, and the call chain of
# every thread that took 30% of a core or more (ps -L just before the stop), for each variant in G_GDB.
if has gdb; then
  command -v gdb > /dev/null || apt-get install -y -qq gdb > /root/gdb-apt.log 2>&1 || say "gdb: apt failed: $(tail -n 1 /root/gdb-apt.log | cut -c1-200)"
  for v in ${G_GDB:-base}; do
    vdef "$v"
    [ -f $SW/var/$VCORE/pearl_core_hip.node ] || { say "gdb $v: no build"; continue; }
    command -v gdb > /dev/null || { say "gdb $v: no gdb"; continue; }
    env PEARL_AMD_UNTESTED=1 "${VENV[@]}" timeout 150 gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex run -ex "thread apply all bt 12" -ex kill --args node ../probes/hashrate.js $SW/var/$VCORE/pearl_core_hip.node 60 5 > /root/gdb-$v.log 2>&1 &
    gp=$!
    sleep 24
    np=$(pgrep -x node | head -n 1)
    ps -L -o tid=,pcpu= -p "${np:-1}" 2>/dev/null | sort -k2 -rn | head -n 9 > /root/gdb-busy.txt
    say "gdb $v busy threads (tid cpu%): $(tr -s ' ' < /root/gdb-busy.txt | tr '\n' ';' | cut -c1-200)"
    kill -INT "${np:-1}" 2>/dev/null
    wait $gp
    awk 'NR == FNR { busy[$1] = $2; next }
      /^Thread / { if (line != "") print line; line = ""; lwp = ""; for (i = 1; i < NF; i++) if ($i == "(LWP") { lwp = $(i + 1); gsub(/[^0-9]/, "", lwp) }; n = 0; if ((lwp in busy) && busy[lwp] + 0 >= 30) line = "tid " lwp " " busy[lwp] "%:"; next }
      /^#/ && line != "" && n < 12 { f = ($3 == "in") ? $4 : $2; lib = ""; if ($(NF - 1) == "from") { m = split($NF, a, "/"); lib = "@" a[m] } line = line " " f lib " <"; n++ }
      END { if (line != "") print line }' /root/gdb-busy.txt /root/gdb-$v.log > /root/gdb-$v.txt
    [ -s /root/gdb-$v.txt ] || say "gdb $v: no chains: $(grep -v '^\[' /root/gdb-$v.log | tail -n 3 | tr '\n' '|' | cut -c1-300)"
    while IFS= read -r l; do say "gdb $v ${l:0:420}"; done < /root/gdb-$v.txt
  done
fi
# goal-rdna4n: for each variant in G_SCAN (names from G_SWEEP), the wmma12 fold's basic blocks that touch scratch
# (isascan.sh's report), and the VGPRs each loop block names.
if has scan; then
  IFS=';' read -ra RV <<< "$SWEEPV"
  for want in ${G_SCAN:-}; do
    fl=""
    for v in "${RV[@]}"; do [ "${v%%=*}" = "$want" ] && fl="${v#*=}"; done
    /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 $fl \
      -I$A/compat/gpu -I$A/src -I$R/earn/native/src -I$A/build/gen --cuda-device-only -S \
      -c "$A/src/pearl_host_hip.cpp" -o /root/scan.s > /root/scan.err 2>&1 \
      || { say "scan $want: compile failed: $(grep -m2 error /root/scan.err | tr '\n' '|' | cut -c1-250)"; continue; }
    awk '/^pearl_amd_fold_wmma12:/ { on = 1; next } on && /^\.Lfunc_end/ { on = 0 } on' /root/scan.s > /root/scan.k
    say "scan $want: lines $(wc -l < /root/scan.k), scratch_store $(grep -c scratch_store /root/scan.k), scratch_load $(grep -c scratch_load /root/scan.k)"
    awk '
      function flush() { if (blk != "" && (n > 20 || ss + sl > 0)) printf "%s n=%d W=%d ss=%d sl=%d bw=%d bl=%d ds=%d dl=%d xor=%d\n", blk, n, w, ss, sl, bw, bl, dst, dld, x }
      /^\.LBB[0-9_]+:/ { flush(); blk = $1; n = w = ss = sl = bw = bl = dst = dld = x = 0; next }
      blk == "" { blk = "entry" }
      $1 ~ /^[vsdbg][a-z0-9_]+$/ { n++ }
      $1 ~ /^v_wmma/ { w++ }
      $1 ~ /^scratch_store/ { ss++ }
      $1 ~ /^scratch_load/ { sl++ }
      $1 == "s_barrier_wait" { bw++ }
      $1 ~ /^buffer_load/ { bl++ }
      $1 ~ /^ds_store/ { dst++ }
      $1 ~ /^ds_load/ { dld++ }
      $1 ~ /^v_xor/ { x++ }
      END { flush() }' /root/scan.k | while IFS= read -r l; do say "scan $want | $l"; sleep 0.2; done
    # G_SCANDUMP: the block with the most WMMAs, every instruction, 8 to a log line (WMMA as "W d a b").
    if case " ${G_SCANDUMP:-} " in *" $want "*) true ;; *) false ;; esac; then
      awk '
        /^\.LBB[0-9_]+:/ { if (w > bw) { bw = w; bb = b }; b = $1; w = 0; next }
        $1 ~ /^v_wmma/ { w++ }
        END { if (w > bw) { bw = w; bb = b }; print bb }' /root/scan.k > /root/scan.best
      bb=$(cat /root/scan.best)
      awk -v bb="$bb" '
        $1 == bb { on = 1; print bb; next }
        on && /^\.LBB[0-9_]+:/ { exit }
        on && $1 ~ /^[vsdbg][a-z0-9_]+$/ {
          l = $0; sub(/;.*/, "", l); gsub(/[ \t]+/, " ", l); sub(/^ /, "", l)
          if ($1 ~ /^v_wmma/) { sub(/^v_wmma[a-z0-9_]* /, "W ", l); gsub(/v\[/, "", l); gsub(/\]/, "", l); gsub(/, /, " ", l) }
          print l }' /root/scan.k | awk '{ buf = buf (n ? " ; " : "") $0; n++; if (n == 8) { print buf; buf = ""; n = 0 } } END { if (n) print buf }' > /root/scan.dump
      say "scandump $want $bb: $(wc -l < /root/scan.dump) lines"
      i=0; while IFS= read -r l; do i=$((i + 1)); say "sd $want $i | ${l:0:420}"; sleep 0.25; done < /root/scan.dump
    fi
    # G_SCANREST: every block with fewer than 100 WMMAs (the prologue, tail and epilogue), the same way.
    if case " ${G_SCANREST:-} " in *" $want "*) true ;; *) false ;; esac; then
      awk '
        FNR == NR { if ($0 ~ /^\.LBB[0-9_]+:/) b = $1; else if ($1 ~ /^v_wmma/) w[b]++; next }
        /^\.LBB[0-9_]+:/ { b = $1; on = (w[b] < 100); if (on) print b; next }
        FNR == 1 { b = ""; on = 1 }
        on && $1 ~ /^[vsdbg][a-z0-9_]+$/ {
          l = $0; sub(/;.*/, "", l); gsub(/[ \t]+/, " ", l); sub(/^ /, "", l)
          if ($1 ~ /^v_wmma/) { sub(/^v_wmma[a-z0-9_]* /, "W ", l); gsub(/v\[/, "", l); gsub(/\]/, "", l); gsub(/, /, " ", l); sub(/ neg_lo.*/, "", l) }
          print l }' /root/scan.k /root/scan.k | awk '{ buf = buf (n ? " ; " : "") $0; n++; if (n == 8) { print buf; buf = ""; n = 0 } } END { if (n) print buf }' > /root/scan.rest
      say "scanrest $want: $(wc -l < /root/scan.rest) lines"
      i=0; while IFS= read -r l; do i=$((i + 1)); say "sr $want $i | ${l:0:420}"; sleep 0.25; done < /root/scan.rest
    fi
    # Lines around the first few scratch ops
    awk '/scratch_(store|load)/ { print NR }' /root/scan.k | head -n 4 | while read -r ln; do
      say "scan $want @$ln | $(sed -n "$((ln - 5)),$((ln + 2))p" /root/scan.k | tr -s ' \t' ' ' | tr '\n' '|' | cut -c1-380)"
    done
  done
fi
say "job done"

# goal-rdna4e: G_CMD run here (from earn/native/amd), its first 40 lines to the log.
if has cmd; then
  eval "${G_CMD:-true}" > /root/cmd.log 2>&1
  head -n 40 /root/cmd.log | while IFS= read -r l; do say "cmd | ${l:0:400}"; done
fi

# goal-rdna4q: the CLI's k trial, G_KRUNS starts of G_KSECS each on the base build, every one timing afresh
# (LLMJOB_K_TRIAL_RETRY=1), on DUEL_POOL. G_KENV: more environment for it. The trial's lines, the card line, and the
# rate line at the end, to the log.
if has ktrial; then
  for i in $(seq 1 ${G_KRUNS:-3}); do
    say "ktrial run $i start ${G_KSECS:-150}s"
    ( cd $R/earn && exec env ${G_KENV:-} LLMJOB_K_TRIAL_RETRY=1 PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$SW/var/base/pearl_core_hip.node \
        NODE_PATH=$NM/node_modules node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$DUEL_POOL" \
        -w "$WORKER-kq" --no-report --no-serve --no-update ) 2>&1 < /dev/null | while IFS= read -r l; do echo "$(date +%s.%N | cut -c1-14) $l"; done > /root/ktrial-$i.log &
    kp=$!
    sleep ${G_KSECS:-150}
    pkill -f 'src/cli/earn-cli\.js'; sleep 6; pkill -9 -f 'src/cli/earn-cli\.js' 2>/dev/null; wait $kp 2>/dev/null
    tr '\r' '\n' < /root/ktrial-$i.log | sed 's/\x1b\[[0-9;]*m//g; s/prl1[0-9a-z]\{20,\}/<addr>/g' \
      | grep -E 'k trial|timing k|mining on|from its k|rotation fold|error|reject' | grep -v 'TH/s ·' | head -n 20 \
      | while IFS= read -r l; do say "ktrial $i | ${l:0:330}"; done
    # The gaps: each card line (a core opened) and the first rate line after it, with their times.
    tr '\r' '\n' < /root/ktrial-$i.log | awk '/mining on GPU/ { print; want = 1; next } want && /TH\/s/ { print; want = 0 }' \
      | cut -c1-120 | while IFS= read -r l; do say "ktrial $i gap | $l"; done
    say "ktrial $i rate | $(tr '\r' '\n' < /root/ktrial-$i.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1)"
  done
fi
# goal-rdna4q: k 8192 against k 16384 in the CLI itself (PEARL_AMD_K, trial off), G_KAB rounds of G_KABSECS each, order
# turning ABBA; each run's mean of the CLI's rate line after its first 60 s.
if has kab; then
  for r in $(seq 1 ${G_KAB:-3}); do
    case $((r % 2)) in 1) order="8192 16384" ;; *) order="16384 8192" ;; esac
    for kk in $order; do
      ( cd $R/earn && exec env PEARL_AMD_K=$kk LLMJOB_K_TRIAL=0 PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$SW/var/base/pearl_core_hip.node \
          NODE_PATH=$NM/node_modules node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$DUEL_POOL" \
          -w "$WORKER-kq" --no-report --no-serve --no-update ) > /root/kab.log 2>&1 < /dev/null &
      kp=$!; : > /root/kab.th; t=0
      while [ $t -lt ${G_KABSECS:-240} ]; do
        sleep 10; t=$((t + 10))
        th=$(tr '\r' '\n' < /root/kab.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted' | tail -1 | awk '{print $1}')
        [ $t -gt 60 ] && [ -n "$th" ] && echo "$th" >> /root/kab.th
      done
      pkill -f 'src/cli/earn-cli\.js'; sleep 6; pkill -9 -f 'src/cli/earn-cli\.js' 2>/dev/null; wait $kp 2>/dev/null
      say "kab r$r k$kk $(stats /root/kab.th) | $(tr '\r' '\n' < /root/kab.log | grep -oE '[0-9,]+ accepted · [0-9,]+ rejected' | tail -1) | $(tr '\r' '\n' < /root/kab.log | grep -m1 'mining on' | cut -c1-120)"
    done
  done
fi
# goal-rdna4q: one more start of G_KMEMSECS without LLMJOB_K_TRIAL_RETRY, so the last trial's answer is used.
if has kmem; then
  ( cd $R/earn && exec env PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$SW/var/base/pearl_core_hip.node \
      NODE_PATH=$NM/node_modules node src/cli/earn-cli.js --address "$WALLET" --mode mining --pool "$DUEL_POOL" \
      -w "$WORKER-kq" --no-report --no-serve --no-update ) > /root/kmem.log 2>&1 < /dev/null &
  kp=$!; sleep ${G_KMEMSECS:-90}
  pkill -f 'src/cli/earn-cli\.js'; sleep 6; pkill -9 -f 'src/cli/earn-cli\.js' 2>/dev/null; wait $kp 2>/dev/null
  grep -E 'k trial|timing k|mining on|from its k' /root/kmem.log | head -n 8 | while IFS= read -r l; do say "kmem | ${l:0:330}"; done
  say "kmem rate | $(tr '\r' '\n' < /root/kmem.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1)"
fi
say "job done (rdna4q)"
