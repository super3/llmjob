# q4-amd job 17: RX 9070 XT: QH_PFOLD (partial rounds' element-0 constant in the S-box's multiply-add, offsets taken off with term0) and QH_DBL with the check skipped (job 16: it spilled); verify-core
J_PARTS="build var verify"
J_VARS="def= pfold=-DQH_PFOLD=1 dblnc=-DQH_DBL=1"
J_ROUNDS=3; J_SECS=22; J_WARM=5
J_VERIFY="pfold dblnc"
J_VEXACT=262144
J_DUMP=""
J_RACE_CFLAGS=""
J_WORKER=g21-q4amd-x9070
# A box job for the q4-amd agent (Quantus on AMD, round 4, 21 Nov). The box agent runs it with J_SHA set to the
# commit it came from. Not part of the miner: the agent drops this file when the round ends.
# Lines go straight to the container's log as "[amdtest] HH:MM:SS q4 ...".
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) q4 $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
sayf() { local tag=$1 l; while IFS= read -r l; do [ -n "${l//[[:space:]]/}" ] && say "$tag | ${l:0:380}"; done; }
QZ=qzk7L38XY4MBHgjvqiudDvVR7gv5asYpZiTKLWJwZjSbzLab6
PARTS=${J_PARTS}
has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
Q=$R/earn/native/quantus
ARCHX=${ARCH:-$(target)}
export NODE_PATH=$NM/node_modules
say "job start $ARCHX, code ${J_SHA:0:12}, parts: $PARTS"

if [ "$(git -C $R rev-parse HEAD)" != "$J_SHA" ]; then
  ( cd $R && git fetch -q --depth 1 origin "$J_SHA" && git checkout -q -f FETCH_HEAD ) > /root/tree.log 2>&1 \
    || { say "FAIL fetch $J_SHA"; exit 1; }
fi
say "tree $(git -C $R rev-parse --short=12 HEAD)"

# The clock probe: one wave from another process reads SHADER_CYCLES and REALTIME (100 MHz) around a short loop.
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
      asm volatile("" : "+v"(x));
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
  say "clkprobe build FAIL"; grep -E 'error' /root/clkprobe-build.log | head -6 | sayf clkprobe
  rm -f /root/clkprobe
fi
fi
# clk_start FILE EVERY COUNT: the probe beside whatever runs now, each line stamped with the time (UTC seconds).
clk_start() {
  [ -x /root/clkprobe ] || return 0
  ( /root/clkprobe "$2" "$3" 2>&1 | while IFS= read -r l; do echo "$(date -u +%s) $l"; done > "$1" ) &
  CLKPID=$!
}
clk_stop() { [ -n "${CLKPID:-}" ] && { pkill -f /root/clkprobe 2>/dev/null; kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null; }; CLKPID=; }
# clk_of FILE [FROM TO]: "median MHz (n)" of the probe's lines (between UTC seconds FROM and TO).
clk_of() {
  awk -v a="${2:-0}" -v b="${3:-9999999999}" '$2=="clk" && $1>=a && $1<=b {print $3}' "$1" | sort -n \
    | awk '{v[NR]=$1} END {if (NR) printf "%s MHz (%d)", v[int((NR+1)/2)], NR; else printf "? (0)"}'
}

# bld NAME FLAGS: the addon for this card's target with QH_CFLAGS=FLAGS, kept in /root/var/NAME/.
bld() {
  local name=$1 flags=$2
  mkdir -p /root/var/$name
  # A name ending in "nc" skips build.sh check, so a build with a few spills can still be timed.
  local chk="bash build.sh check $ARCHX"
  case $name in *nc) chk="true" ;; esac
  if ! (cd $Q/amd && QH_CFLAGS="$flags" $chk && QH_CFLAGS="$flags" bash build.sh gpu "$ARCHX") > /root/var/$name/build.log 2>&1; then
    say "FAIL build $name"; grep -iE 'error|build.sh|failed|^gfx|^target' /root/var/$name/build.log | head -10 | sayf build
    grep -A16 'Function Name: qh_search' $Q/build/amd/check-$ARCHX.txt 2>/dev/null | grep -E 'VGPRs|SGPRs|Spill|Scratch|Occupancy' | sed 's/.*remark: *//' | tr -s ' ' | tr '\n' ';' | { cat; echo; } | sayf builderr
    grep -E -A2 'error:' $Q/build/amd/check-$ARCHX.txt 2>/dev/null | head -12 | sayf builderr; return 1
  fi
  cp $Q/build/quantus_hip.node /root/var/$name/
  say "built $name ($flags): $(grep -E "^$ARCHX" /root/var/$name/build.log | tr -s ' ' | head -1)"
}

# Builds from an earlier job on this box are never reused: a name can mean other flags or older code.
rm -rf /root/var
if has build; then
  t0=$(date +%s)
  bld base "" || exit 1
  say "base built in $(( $(date +%s) - t0 ))s"
fi

write_isa_py() {
  cat > /root/isa.py <<'EOF_PY'
import re, sys, collections
ins = [l.strip() for l in open(sys.argv[1]) if l.strip()]
blocks, cur = [], {"label": "entry", "ins": []}
for s in ins:
    if s.endswith(":"):
        blocks.append(cur); cur = {"label": s[:-1], "ins": []}; continue
    cur["ins"].append(s)
blocks.append(cur)
def cls(op):
    if op == "v_mad_u64_u32": return "mad"
    if op.startswith("v_dual_"): return "vopd"
    if op.startswith("v_"): return "valu"
    if op == "s_delay_alu": return "dly"
    if op.startswith("s_waitcnt"): return "wait"
    if op.startswith("s_load") or op.startswith("s_buffer_load"): return "smem"
    if op.startswith("s_cbranch") or op == "s_branch": return "br"
    if op.startswith("s_"): return "salu"
    return "other"
tot = collections.Counter(); ops = collections.Counter()
for b in blocks:
    c = collections.Counter(); tgt = []
    for s in b["ins"]:
        op = s.split()[0]; c[cls(op)] += 1; tot[cls(op)] += 1; ops[op] += 1
        if cls(op) == "br": tgt.append(op.replace("s_cbranch_", "") + ">" + s.split()[-1].replace(".LBB0_", ""))
    if not b["ins"]: continue
    print("blk %s n=%d mad=%d valu=%d vopd=%d dly=%d salu=%d smem=%d wait=%d br=%s" % (b["label"].replace(".LBB0_", "B"), len(b["ins"]),
          c["mad"], c["valu"], c["vopd"], c["dly"], c["salu"], c["smem"], c["wait"], ",".join(tgt)))
print("total " + " ".join("%s=%d" % kv for kv in sorted(tot.items())))
print("ops " + " ".join("%s=%d" % kv for kv in ops.most_common(40)))
EOF_PY
}
write_isa_py

if has isa; then
  t0=$(date +%s)
  (cd $Q/amd && bash build.sh asm gfx1100 gfx1101 gfx1201 gfx1030 gfx1031 gfx1032) > /root/asm.log 2>&1 || { say "FAIL asm"; tail -n 5 /root/asm.log | sayf asm; }
  # The instruction lines of each target's qh_search (no comments or directives), and their md5: the RDNA 2
  # targets' must stay the same through this round.
  for a in gfx1100 gfx1101 gfx1201 gfx1030 gfx1031 gfx1032; do
    s=$Q/build/amd/qh-$a.s
    [ -f $s ] || continue
    awk '/^qh_search:/{on=1;next} /^\.Lfunc_end/{on=0} on' $s | sed 's/;.*//; s/^[ \t]*//; s/[ \t]*$//' | grep -v '^\.' | grep -v '^$' > /root/isa-$a.txt
    say "isa $a: $(wc -l < /root/isa-$a.txt) lines, md5 $(md5sum < /root/isa-$a.txt | cut -c1-12)"
  done
  cat > /root/isa.py <<'EOF_PY'
import re, sys, collections
ins = [l.strip() for l in open(sys.argv[1]) if l.strip()]
blocks, cur = [], {"label": "entry", "ins": []}
for s in ins:
    if s.endswith(":"):
        blocks.append(cur); cur = {"label": s[:-1], "ins": []}; continue
    cur["ins"].append(s)
blocks.append(cur)
def cls(op):
    if op == "v_mad_u64_u32": return "mad"
    if op.startswith("v_dual_"): return "vopd"
    if op.startswith("v_"): return "valu"
    if op == "s_delay_alu": return "dly"
    if op.startswith("s_waitcnt"): return "wait"
    if op.startswith("s_load") or op.startswith("s_buffer_load"): return "smem"
    if op.startswith("s_cbranch") or op == "s_branch": return "br"
    if op.startswith("s_"): return "salu"
    return "other"
tot = collections.Counter(); ops = collections.Counter()
for b in blocks:
    c = collections.Counter(); tgt = []
    for s in b["ins"]:
        op = s.split()[0]; c[cls(op)] += 1; tot[cls(op)] += 1; ops[op] += 1
        if cls(op) == "br": tgt.append(op.replace("s_cbranch_", "") + ">" + s.split()[-1].replace(".LBB0_", ""))
    if not b["ins"]: continue
    print("blk %s n=%d mad=%d valu=%d vopd=%d dly=%d salu=%d smem=%d wait=%d br=%s" % (b["label"].replace(".LBB0_", "B"), len(b["ins"]),
          c["mad"], c["valu"], c["vopd"], c["dly"], c["salu"], c["smem"], c["wait"], ",".join(tgt)))
print("total " + " ".join("%s=%d" % kv for kv in sorted(tot.items())))
print("ops " + " ".join("%s=%d" % kv for kv in ops.most_common(40)))
EOF_PY
  python3 /root/isa.py /root/isa-$ARCHX.txt | sayf isa
  # The whole gfx1100 kernel, gzip + base64, numbered, for reading off the box (J_ISADUMP=1).
  [ "${J_ISADUMP:-0}" = 1 ] && gzip -9c /root/isa-gfx1100.txt | base64 -w 360 > /root/isa64.txt
  n=0; while IFS= read -r l; do n=$((n + 1)); say "isa64 $(printf %04d $n) $l"; [ $((n % 25)) -eq 0 ] && sleep 0.3; done < /root/isa64.txt
  say "isa64 end $n lines, md5 $(md5sum < /root/isa-gfx1100.txt | cut -c1-12)"
  say "isa done in $(( $(date +%s) - t0 ))s"
fi

# rate1 NAME: rate.js on that build (J_SECS after J_WARM), the clock probe beside it.
rate1() {
  local name=$1 r
  clk_start /root/clk-rate.log 5 100
  r=$(timeout 300 node $Q/amd/test/rate.js /root/var/$name/quantus_hip.node --secs $J_SECS --warm $J_WARM 2>&1 | tail -n 1)
  clk_stop
  say "rate $name $(echo "$r" | grep -oE '"mhs":[0-9.]+,"sd":[0-9.]+,"scans":[0-9]+') clk $(clk_of /root/clk-rate.log)"
}

if has rate; then
  rate1 base; rate1 base
fi

if has var; then
  for v in $J_VARS; do
    n=${v%%=*}; f=${v#*=}; f=${f//__/ }
    bld "$n" "$f" || continue
  done
  names=""; for v in $J_VARS; do n=${v%%=*}; [ -f /root/var/$n/quantus_hip.node ] && names="$names $n"; done
  for r in $(seq 1 $J_ROUNDS); do
    for n in $names; do rate1 $n; done
    names="$(echo $names | awk '{for (i=2;i<=NF;i++) printf "%s ", $i; print $1}')"
  done
fi

if has carry; then
  if $CLANG -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 $Q/amd/test/qh_carry.cpp -o /root/qh_carry \
      -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/carry-build.log 2>&1; then
    timeout 300 /root/qh_carry 0 2>&1 | sayf carry
  else
    say "carry build FAIL"; grep -iE 'error' /root/carry-build.log | head -8 | sayf carry
  fi
fi

if has parts; then
  for v in $J_PFLAGS; do
    n=${v%%=*}; f=${v#*=}; f=${f//__/ }
    if $CLANG -x hip --offload-arch="$ARCHX" --rocm-path=/opt/rocm -std=c++17 -O3 $f -I$Q/amd $Q/amd/test/qh_parts.cpp \
        -o /root/qh_parts_$n -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/parts-build-$n.log 2>&1; then
      timeout 300 /root/qh_parts_$n 0 2>&1 | sed "s/^/$n: /" | sayf parts
    else
      say "parts $n build FAIL"; grep -iE 'error' /root/parts-build-$n.log | head -6 | sayf parts
    fi
  done
fi

if has isavar; then
  for v in $J_VARS; do
    n=${v%%=*}; f=${v#*=}; f=${f//__/ }
    (cd $Q/amd && QH_CFLAGS="$f" bash build.sh asm "$ARCHX") > /root/asmv.log 2>&1 || { say "isavar $n asm FAIL"; continue; }
    awk '/^qh_search:/{on=1;next} /^\.Lfunc_end/{on=0} on' $Q/build/amd/qh-$ARCHX.s | sed 's/;.*//; s/^[ \t]*//; s/[ \t]*$//' | grep -v '^\.[a-z]' | grep -v '^$' > /root/isa-$n.txt
    say "isavar $n: $(python3 /root/isa.py /root/isa-$n.txt | grep '^total' | cut -c1-200)"
    case " $J_DUMP " in *" $n "*)
      gzip -9c /root/isa-$n.txt | base64 -w 360 > /root/isa64-$n.txt
      i=0; while IFS= read -r l; do i=$((i + 1)); say "isa64 $n $(printf %04d $i) $l"; [ $((i % 25)) -eq 0 ] && sleep 0.3; done < /root/isa64-$n.txt
      say "isa64 $n end $i lines, md5 $(md5sum < /root/isa-$n.txt | cut -c1-12)" ;;
    esac
  done
fi

if has verify; then
  for n in $J_VERIFY; do
    [ -f /root/var/$n/quantus_hip.node ] || continue
    (cd $R/earn && timeout 900 node native/quantus/verify-core.js --addon /root/var/$n/quantus_hip.node --exact ${J_VEXACT:-4096}) > /root/verify-$n.log 2>&1
    say "verify $n rc=$? $(grep -E '^(PASS|FAIL)' /root/verify-$n.log | tr '\n' ' ' | cut -c1-300)"
  done
fi

if has peak; then
  if (cd $Q/amd && bash build.sh peak "$ARCHX") > /root/peak-build.log 2>&1; then
    timeout 600 $Q/build/amd/qh_peak 0 2>&1 | sayf peak
  else
    say "peak build FAIL"; grep -iE 'error' /root/peak-build.log | head -6 | sayf peak
  fi
fi

if has prof; then
  if command -v rocprofv3 > /dev/null 2>&1; then
    say "prof rocprofv3 at $(command -v rocprofv3)"
    rm -rf /root/prof
    timeout 240 rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_BUSY_CYCLES SQ_WAVE_CYCLES --output-format csv -d /root/prof -o run \
      -- node $Q/amd/test/rate.js /root/var/base/quantus_hip.node --secs 2 --warm 0 --count 16777216 > /root/prof.log 2>&1
    say "prof rc=$?"
    tail -n 12 /root/prof.log | sayf prof
    find /root/prof -name '*.csv' 2>/dev/null | head -3 | while read -r f; do say "prof file $f"; head -n 4 "$f" | sayf prof; done
  else
    say "prof no rocprofv3 in this image: $(ls /opt/rocm/bin | grep -i prof | tr '\n' ' ' | cut -c1-200)"
  fi
fi

if has race; then
  curl -fsSL "https://raw.githubusercontent.com/super3/llmjob/$J_SHA/earn/native/quantus/qrace.sh" -o /root/qrace.sh || { say "FAIL qrace fetch"; exit 1; }
  clk_start /root/clk-race.log 10 400
  ( BASE_SHA=$J_SHA WALLET=$QZ WORKER=${J_WORKER:-g21-q4amd} QH_CFLAGS="${J_RACE_CFLAGS:-}" POOL=us.quantus.herominers.com:1155 POOL2=qtc-us.kryptex.network:7049 WARM=30 SECS=90 LEGS="ours srb srb ours" timeout 2400 bash /root/qrace.sh > /root/qrace.log 2>&1 ) &
  qp=$!
  seen=0
  while kill -0 $qp 2>/dev/null; do
    sleep 15
    n=$(wc -l < /root/qrace.log)
    tail -n +$((seen + 1)) /root/qrace.log | head -n $((n - seen)) | grep -E 'build ok|verify \||verify rc|leg .* (start|result)|FAIL|exited|all done|setup ok|card ' | sed 's/^\[amdtest\] [0-9:]* //' | sayf race
    seen=$n
    grep -qE 'all done|idle after a failure' /root/qrace.log && break
  done
  pkill -f /root/qrace.sh 2>/dev/null; kill $qp 2>/dev/null
  clk_stop
  # The clock under each leg: the probe's lines between the leg's warm-up end and its result line.
  python3 - /root/qrace.log /root/clk-race.log <<'EOF_PY' | sayf raceclk
import re, sys, datetime as dt, statistics as st
legs, cur = [], None
day = dt.datetime.now(dt.timezone.utc).date()
def ts(hms):
    h, m, s = map(int, hms.split(":"))
    return dt.datetime(day.year, day.month, day.day, h, m, s, tzinfo=dt.timezone.utc).timestamp()
for l in open(sys.argv[1], errors="replace"):
    m = re.search(r"\] (\d\d:\d\d:\d\d) leg (\w+) start", l)
    if m: cur = [m.group(2), ts(m.group(1)) + 30]
    m = re.search(r"\] (\d\d:\d\d:\d\d) leg (\w+) result (.*)", l)
    if m and cur: legs.append((cur[0], cur[1], ts(m.group(1)) - 5, m.group(3))); cur = None
clk = []
for l in open(sys.argv[2], errors="replace"):
    p = l.split()
    if len(p) > 3 and p[1] == "clk": clk.append((int(p[0]), float(p[2])))
for name, a, b, res in legs:
    v = [m for t, m in clk if a <= t <= b]
    print("%s clk median %s MHz (n=%d, %s-%s) %s" % (name, st.median(v) if v else "?", len(v), min(v) if v else "?", max(v) if v else "?", res))
EOF_PY
fi
say "job done"
