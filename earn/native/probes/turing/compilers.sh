#!/bin/bash
# compilers.sh -- the RTX 20 (sm_75) fold built with other compiler options, raced on
# one rented card.
#
#   SRC=<repo checkout> OUT=<scratch dir> bash earn/native/probes/turing/compilers.sh
#
# Every result is one line starting with [compilers], under 500 characters (Vast's log
# API cuts longer lines). STAGES picks the steps: "setup build bench hashrate verify
# probe" by default; "setup build" runs without a GPU. ONLY="base rul8" keeps just
# those builds (to confirm a winner on a second card). Assumes an idle GPU 0 and the
# nvidia/cuda:12.8.1-devel-ubuntu22.04 image (CUDA_HOME, default /usr/local/cuda, must
# be CUDA 12.8.1, the toolkit CI builds with). About 35 minutes on a 2080 Ti.
#
# Why these builds. Built here first (2026-10-20), ptxas from CUDA 12.0, 12.2, 12.4,
# 12.6, 12.8, 12.9, 13.0, 13.3 and 13.4, and cicc from 12.4 through 13.4, all give the
# B-direct fold (pearl_fold_bd.cuh) the same machine code, byte for byte; 12.0 and
# 12.2 differ in one control bit before the loop. So another toolkit cannot change
# the sm_75 fold, and no run here spends GPU time on one. What does change it:
#   base        what ships: CUDA 12.8.1, no extra flags. 254 registers, 680 instructions
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
# round), hashrate.js on the fastest and on base in HR_ROUNDS interleaved rounds,
# verify-hits.js (400 hits) on every build but base, then the driver probe: the
# base fold from ptxas 13.4, whose cubin has the CUDA 13 ELF format, and from
# ptxas 12.9, packed into a 12.8 core, to see whether this card's driver loads them.
set -u
: "${SRC:?set SRC to the repo checkout}" "${OUT:?set OUT to a scratch directory}"
STAGES=" ${STAGES:-setup build bench hashrate verify probe} "
ROUNDS=${ROUNDS:-5}; BENCH_SECS=${BENCH_SECS:-20}
HR_ROUNDS=${HR_ROUNDS:-3}; HR_SECS=${HR_SECS:-60}; VERIFY_SECS=${VERIFY_SECS:-90}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
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
VARIANTS=(
  "base||ac306adc8702"
  "rul8|-Xptxas --register-usage-level=8|8fa27fbb65da"
  "cmr255|-maxrregcount=255|a9500fa5f046"
  "cmr255rul8|-maxrregcount=255 -Xptxas --register-usage-level=8|23d035369029"
)
if [ -n "${ONLY:-}" ]; then
  keep=()
  for v in "${VARIANTS[@]}"; do case " $ONLY " in *" ${v%%|*} "*) keep+=("$v") ;; esac; done
  VARIANTS=("${keep[@]}")
fi
GC="-gencode arch=compute_75,code=sm_75"
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

fold_hash() {   # $1 = an object, .node or cubin; prints the sm_75 fold's code hash
  local d; d=$(mktemp -d "$OUT/x.XXXX")
  case "$1" in
    *.cubin) cp "$1" "$d/" ;;
    *) (cd "$d" && cuobjdump -xelf all "$1" > /dev/null 2>&1) ;;
  esac
  # Every sm_75 image (the addon also carries pearl_host.cu's); the fold is in one.
  python3 "$OUT/texthash.py" "$d"/*.cubin 2>/dev/null | awk '$1 == "pearl_tile_fold_wmma" {print $2; exit}'
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
[ "$nv" = V12.8.93 ] || say "WARN: nvcc $nv is not CI's 12.8.93; the expected fold hashes will not match"
if command -v nvidia-smi > /dev/null && nvidia-smi -i 0 > /dev/null 2>&1; then
  say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,memory.total --format=csv,noheader)"
  q=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,utilization.gpu
  say "card now $(nvidia-smi -i 0 --query-gpu=$q,clocks_event_reasons.active --format=csv,noheader 2> /dev/null ||
    nvidia-smi -i 0 --query-gpu=$q --format=csv,noheader) gpus=$(nvidia-smi -L | wc -l)"
  DRIVER=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader | tr -d ' ')
else
  say "no GPU visible"; DRIVER=none
fi

# One build: pearl_kernel.o for sm_75 with the flags, the addon around it, and bench.
build_variant() {   # name flags expect
  local name=$1 flags=$2 expect=$3 K=$OUT/v/$1
  rm -rf "$K"; mkdir -p "$K/cuda-build"
  cp -r "$NATIVE/src" "$NATIVE/binding.gyp" "$K/"
  # shellcheck disable=SC2086
  if ! nvcc $COMMON $flags -Xptxas -v -c "$K/src/pearl_kernel.cu" -o "$K/cuda-build/pearl_kernel.o" > "$K/ptxas.txt" 2>&1; then
    say "build $name FAIL: $(grep -m2 -iE 'error' "$K/ptxas.txt" | tr '\n' ' ')"; return 1
  fi
  cp "$OUT/host/pearl_host.o" "$K/cuda-build/"
  ar rcs "$K/cuda-build/libpearl_cuda.a" "$K/cuda-build/pearl_kernel.o" "$K/cuda-build/pearl_host.o"
  (cd "$K" && ln -s "$OUT/gyp/node_modules" node_modules && echo '{}' > package.json &&
    CUDA_PATH=$CUDA_HOME ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) ||
    { say "build $name FAIL: node-gyp: $(grep -m2 -iE 'error' "$K/gyp.log" | tr '\n' ' ')"; return 1; }
  nvcc -O3 -std=c++17 -cudart static $GC -I"$K/src" "$NATIVE/probes/bench.cu" \
    "$K/cuda-build/pearl_kernel.o" "$K/cuda-build/pearl_host.o" -o "$K/bench" > "$K/bench-build.txt" 2>&1 ||
    { say "build $name FAIL: bench: $(grep -m1 -i error "$K/bench-build.txt")"; return 1; }
  local st h
  st=$(awk '/Compiling entry function .pearl_tile_fold_wmma. for .sm_75./{f=1;next} f&&/Compiling entry/{f=0} f' "$K/ptxas.txt" | tr '\n' ' ')
  h=$(fold_hash "$K/build/Release/pearl_core.node")
  echo "$h" > "$K/fold.hash"
  say "build $name regs=$(echo "$st" | grep -oE 'Used [0-9]+' | grep -oE '[0-9]+') spill=$(echo "$st" | grep -oE '[0-9]+ bytes spill stores' | grep -oE '^[0-9]+')/$(echo "$st" | grep -oE '[0-9]+ bytes spill loads' | grep -oE '^[0-9]+') fold=$h expect=$expect $([ "$h" = "$expect" ] && echo match || echo MISMATCH) flags=[$flags]"
  touch "$K/ok"
}

if has build; then
  mkdir -p "$OUT/host"
  nvcc $COMMON -c "$NATIVE/src/pearl_host.cu" -o "$OUT/host/pearl_host.o" > "$OUT/host/log" 2>&1 ||
    { say "FAIL: pearl_host.cu: $(grep -m1 -i error "$OUT/host/log")"; exit 1; }
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r name flags expect <<< "$v"
    build_variant "$name" "$flags" "$expect"
  done
fi

BUILT=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name _ _ <<< "$v"
  [ -f "$OUT/v/$name/ok" ] && BUILT+=("$name")
done

# Median SM clock, power and temperature of the samples taken while the timed part ran.
cat > "$OUT/med.py" <<'EOF'
import statistics, sys
f, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for line in open(f):
    p = [x.strip() for x in line.split(",")]
    try:
        t = float(p[0]); v = [float(x) for x in p[1:4]]
    except Exception:
        continue
    if t0 <= t <= t1:
        rows.append(v)
if not rows:
    print("0 0 0 0"); sys.exit()
print(len(rows), *[round(statistics.median(r[i] for r in rows), 1) for i in range(3)])
EOF
sampler() {   # $1 = file; one sample a second until killed
  while :; do
    printf '%s,%s\n' "$(date +%s.%N)" "$(nvidia-smi -i 0 --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits)" >> "$1"
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
      th=$(python3 -c "print(round($att * 256 * 2048 / $secs / 1e12, 3))")
      read -r n mhz w c < <(python3 "$OUT/med.py" "$S" "$(python3 -c "print($t1 - $secs + 1)")" "$t1")
      pg=$(python3 -c "print(round($th / ($mhz / 1000), 3) if $mhz else 0)")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$name" "$th" "$mhz" "$w" "$c" "$pg" >> "$OUT/bench.tsv"
      [ "$r" = 1 ] && say "bench $name runs $(grep -oE 'fold: [^|]*' "$K/bench.r$r" | head -1)"
      say "bench r$r $name $th TH/s ${mhz} MHz ${w} W ${c} C $pg TH/s/GHz (n=$n)"
    done
  done
  # Per build: medians over rounds, and against base in the same round.
  cat > "$OUT/sum.py" <<'EOF'
import statistics as st, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
tag = sys.argv[2]
by = {}
for r in rows:
    by.setdefault(r[1], {})[int(r[0])] = (float(r[2]), float(r[3]), float(r[4]), float(r[6]))
base = by.get("base", {})
best, bestv = None, None
for name, d in by.items():
    th = [v[0] for v in d.values()]; mhz = [v[1] for v in d.values()]; w = [v[2] for v in d.values()]
    pg = [v[3] for v in d.values()]
    raw = [d[k][0] / base[k][0] - 1 for k in d if k in base and name != "base"]
    per = [d[k][3] / base[k][3] - 1 for k in d if k in base and name != "base" and base[k][3]]
    vs = ""
    if raw:
        vs = (f" vs base: raw {st.median(raw) * 100:+.2f}% per-GHz {st.median(per) * 100:+.2f}% "
              f"(rounds {' '.join(f'{x * 100:+.1f}' for x in raw)})")
        if bestv is None or st.median(raw) > bestv:
            best, bestv = name, st.median(raw)
    print(f"[compilers] {tag} {name} n={len(th)} {st.median(th):.3f} TH/s ({min(th):.2f}-{max(th):.2f}) "
          f"{st.median(mhz):.0f} MHz {st.median(w):.0f} W {st.median(pg):.3f} TH/s/GHz{vs}"[:490])
if best:
    open(sys.argv[3], "w").write(best)
EOF
  python3 "$OUT/sum.py" "$OUT/bench.tsv" bench-sum "$OUT/best"
fi

if has hashrate && [ -f "$OUT/best" ] && [ -f "$OUT/v/base/ok" ]; then
  best=$(cat "$OUT/best")
  say "hashrate: base against $best, $HR_ROUNDS rounds of ${HR_SECS} s"
  : > "$OUT/hashrate.tsv"
  for r in $(seq 1 "$HR_ROUNDS"); do
    pair=(base "$best"); [ $((r % 2)) = 0 ] && pair=("$best" base)
    for name in "${pair[@]}"; do
      j=$(cd "$SRC/earn" && timeout -k 10 $((HR_SECS + 120)) node native/probes/hashrate.js "$OUT/v/$name/build/Release/pearl_core.node" "$HR_SECS" 10 2>&1 | grep '^{' | tail -1)
      th=$(echo "$j" | grep -oE '"th":[0-9.]+' | cut -d: -f2)
      mhz=$(echo "$j" | grep -oE '"mhz":[0-9.]+' | cut -d: -f2)
      w=$(echo "$j" | grep -oE '"watts":[0-9.]+' | cut -d: -f2)
      c=$(echo "$j" | grep -oE '"tempC":[0-9.]+' | cut -d: -f2)
      mhz=${mhz:-0}
      if [ -z "$th" ]; then say "hashrate r$r $name FAIL: $j"; continue; fi
      pg=$(python3 -c "print(round($th / ($mhz / 1000), 3) if '$mhz' and $mhz else 0)")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$name" "$th" "$mhz" "$w" "$c" "$pg" >> "$OUT/hashrate.tsv"
      say "hashrate r$r $name $th TH/s sd=$(echo "$j" | grep -oE '"sd":[0-9.]+' | cut -d: -f2) ${mhz} MHz ${w} W ${c} C $pg TH/s/GHz"
    done
  done
  python3 "$OUT/sum.py" "$OUT/hashrate.tsv" hashrate-sum "$OUT/best-hr"
fi

if has verify; then
  for name in "${BUILT[@]}"; do
    [ "$name" = base ] && continue
    out=$(cd "$SRC/earn" && timeout -k 10 $((VERIFY_SECS + 120)) node native/probes/verify-hits.js "$OUT/v/$name/build/Release/pearl_core.node" "$VERIFY_SECS" 2>&1)
    j=$(echo "$out" | grep '^{' | tail -1)
    if [ -z "$j" ]; then say "verify $name FAIL: no result: $(echo "$out" | tail -3 | tr '\n' ' ')"; continue; fi
    say "verify $name $(echo "$j" | grep -oE '"(hits|verified|failed|salts|PASS)":[a-z0-9]+' | tr '\n' ' ')$(echo "$j" | grep -q '"PASS":true' || echo " FAILS: $(echo "$j" | grep -oE '"fails":.*' | cut -c1-300)")"
  done
fi

# Driver probe: does this driver load an sm_75 cubin from a newer ptxas inside the 12.8
# core? 13.x cubins use a new ELF format (OS ABI 0x41, ABI version 8, the architecture
# moved within e_flags); 12.9's keep the old one. The fold's code is the same either way.
if has probe && [ -f "$OUT/v/base/ok" ]; then
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
    out=$(cd "$SRC/earn" && timeout -k 10 120 node native/probes/hashrate.js "$K/build/Release/pearl_core.node" 20 5 2>&1 | tr '\n' ' ')
    th=$(echo "$out" | grep -oE '"th":[0-9.]+' | cut -d: -f2)
    if [ -n "$th" ]; then res="loads, $th TH/s"
    else res="does not run: $(echo "$out" | grep -oE '(error|Error)[^{}]{0,200}' | head -1 | sed 's/ \{2,\}at .*//')"; fi
    say "probe ptxas $ver cubin $abi fold=$fh in-node=$inside driver $DRIVER: $res"
  done
fi
say "done in $(( ($(date +%s) - T0) / 60 )) min"
