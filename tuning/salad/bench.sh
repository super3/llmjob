#!/bin/bash
# Benchmarks PeakMiner, SRBMiner and our miner on one Salad GPU, one after another, the way benchmark.md's
# "How a host is tested" does: a 60 s unmeasured warm-up, then each miner for 5 minutes on HeroMiners.
# bench.py starts it as a Salad container command and reads its "[bench]" lines from Salad's logs.
#
# Environment (set by bench.py): W wallet, ORDER (e.g. "peak,srb,ours"), OURS_SECS (300, or 90 on a card
# our release has no build for), SRB_EXTRA (extra SRBMiner flags, usually empty), and on AMD BASE_SHA and
# VERIFY_SECS. Every line it prints for bench.py starts "[bench] HH:MM:SS".
#
# AMD: on a box with no nvidia-smi (Salad's AMD classes, run on the ROCm image) it reads the card with rocminfo
# and Salad's injected amd-smi (/opt/rocm-wsl/bin; the image's own amd-smi and rocm-smi can't reach a GPU under
# WSL2) and runs SRBMiner on AMD only. PeakMiner has no Pearl kernel for AMD and isn't downloaded or run. No
# release has an AMD build, so "ours" is the HIP core (earn/native/amd) at BASE_SHA, built here for the card's
# target and run through the app's CLI from source. That uses amdtest.sh's setup, fetch, hip, build, verify and
# mine_cli: bench.py's command fetches amdtest.sh beside this script, at the same commit. Before the miners run it
# prints the PC's device copy rate and runs the hit check (verify-hits.js) for VERIFY_SECS (600). If the core
# doesn't build or the hit check fails, our miner is left out and SRBMiner still runs. LD_LIBRARY_PATH is never
# set: Salad puts its ROCm-to-WSL bridge first on it.
export DEBIAN_FRONTEND=noninteractive
POOL_HOST=us.pearl.herominers.com; POOL_PORT=1200; SECS=300; WARM=60
# The versions a box runs: our release, and each competitor's newest with its published checksum. bench.py pins
# this file by commit, so a run keeps the versions it started with. Until 28 October: v0.6.0, PeakMiner 2.17.6
# and SRBMiner 3.7.1 (the GTX and AMD tables).
VER=v0.6.9
PEAK_VER=2.20.3; PEAK_SHA256=345a4446acaa9213e6cd4622460b002a966dad4a8ec127f965d3a8878375fc2a
SRB_VER=3.8.0; SRB_MD5=01cdd7ffdca43f3b6968fbaacea449b5
REL=https://github.com/super3/llmjob/releases/download/$VER
say() { echo "[bench] $(date -u +%T) $*"; }
fail() { say "FAIL $*"; say "all done"; sleep infinity; }
AMD=; command -v nvidia-smi >/dev/null 2>&1 || AMD=1
SMI=/opt/rocm-wsl/bin/amd-smi
export PATH="$PATH:/opt/rocm/bin"   # append only: Salad's /opt/rocm-wsl/bin stays first

# drop NAME [WHY]: leaves a miner out of ORDER, and says why if given.
drop() {
  [ -n "${2:-}" ] && say "skip $1: $2"
  ORDER=$(echo "${ORDER:-peak,srb,ours}" | tr , '\n' | grep -vx "$1" | paste -sd, -)
}

# copy_rate LOG: the GB/s of amdtest_hip.cpp's timed device copy ("copy: 564 GB/s (read + write)").
copy_rate() { grep -oE '^copy: [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+$' | head -1; }

# hitcheck LOG: one short line from verify-hits.js's JSON (the last "{" line of its log). PASS only if the JSON
# says PASS and no hit failed; returns 1 otherwise. t_verify prints the JSON too, in pieces, as it's long.
hitcheck() {
  local j v
  j=$(grep '^{' "$1" 2>/dev/null | tail -1)
  v=$(echo "$j" | jq -r 'if .error then "error=\(.error | tostring | .[0:160])"
        else "hits=\(.hits) verified=\(.verified) failed=\(.failed) salts=\(.salts)"
          + (if (.fails | length) > 0 then " why=\(.fails[0].why | tostring | .[0:120])" else "" end) end' 2>/dev/null)
  [ -n "$v" ] || v="error=no result: $(grep -v '^\s*$' "$1" 2>/dev/null | tail -1 | tr -d '\r' | head -c 160)"
  if [ "$(echo "$j" | jq -r '.PASS == true and .failed == 0' 2>/dev/null)" = true ]; then
    say "hitcheck PASS secs=${VERIFY_SECS:-600} $v"; return 0
  fi
  say "hitcheck FAIL secs=${VERIFY_SECS:-600} $v"; return 1
}

# Our miner on AMD, up to where it can mine: amdtest.sh's setup and fetch (BASE_SHA into /opt/llmjob), its hip
# task for the PC's copy rate, the core built for this card's target, and the hit check. A failed build or hit
# check leaves our miner out; any other failure ends the box, as a failed download does.
amd_ours() {
  local keep c; keep=$(declare -f say fail)
  AMDTEST_STAGE=lib . "$(dirname "$0")/amdtest.sh" || fail "no amdtest.sh beside bench.sh"
  eval "$keep"   # bench's own say and fail again, so every line is a [bench] line
  [ -n "${BASE_SHA:-}" ] || fail "no BASE_SHA"
  setup; fetch; cd /opt/b || fail "no /opt/b"
  ARCH=$(target); [ -n "$ARCH" ] || fail "no gfx target from rocminfo"
  t_hip; c=$(copy_rate /root/hip.log)
  say "copy ${c:-?} GB/s"
  say "start build $ARCH"
  t_build "$ARCH" || { drop ours "build failed"; return; }
  VERIFY_SECS=${VERIFY_SECS:-600}
  say "start hitcheck ${VERIFY_SECS}s"
  t_verify "$VERIFY_SECS"
  hitcheck /root/verify.log || drop ours "hit check failed"
  CORE="HIP core $(git -C "$R" rev-parse --short=12 HEAD) for $ARCH"
}

apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq --no-install-recommends curl ca-certificates jq procps \
  ${AMD:+clinfo} >/dev/null 2>&1 || fail "apt-get install"
if [ -z "$AMD" ]; then
  say "card $(nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,memory.total,compute_cap --format=csv,noheader | head -1)"
else
  # The same comma layout as nvidia-smi's: name, driver, power limit, default limit, memory, target.
  test -c /dev/dxg || fail "no /dev/dxg"
  ri=$(timeout 60 rocminfo 2>&1)
  name=$(echo "$ri" | awk -F': +' '/Marketing Name:/{m=$2} /Device Type:/{if ($2 ~ /GPU/) {print m; exit}}' | sed 's/ *$//')
  gfx=$(echo "$ri" | grep -m1 -oE 'gfx[0-9a-f]+')
  [ -n "$name" ] || fail "no AMD GPU: $(echo "$ri" | grep -m1 -E 'HSA_STATUS|rror' | tail -c 200)"
  st=$(timeout 20 $SMI static --asic --board --limit --driver --vram --json 2>/dev/null)
  echo "$st" | tr -s ' \n' ' ' | fold -w 900 | while read -r l; do say "amdsmi $l"; done   # its keys aren't documented
  num() { echo "$st" | jq -r --arg p "$1" 'first(paths(numbers) as $q | select($q|map(tostring)|join(".")|test($p;"i")) | getpath($q)) // empty' 2>/dev/null; }
  str() { echo "$st" | jq -r --arg p "$1" 'first(paths(strings) as $q | select($q|map(tostring)|join(".")|test($p;"i")) | getpath($q)) // empty' 2>/dev/null; }
  pl=$(num 'power_limit|power_cap'); vram=$(num 'vram.*(size|total)'); drv=$(str 'driver.*version')
  say "card $name, ${drv:-?}, ${pl:+$pl W}, ${pl:+$pl W}, ${vram:-?} MiB, $gfx"
  say "opencl $(clinfo -l 2>&1 | tr -s ' \n' ' ' | head -c 300)"
  say "gpucols $(timeout 5 $SMI monitor -p -t -u --csv 2>&1 | head -1)"
  drop peak   # no Pearl kernel for AMD
fi

mkdir -p /opt/b && cd /opt/b || fail "no /opt/b"
# The two miners from their official releases, checked against the published checksums.
if [ -z "$AMD" ]; then
  curl -fsSL --retry 3 -o peak \
    "https://github.com/peakminer/peakminer/releases/download/v$PEAK_VER/peakminer-$PEAK_VER-linux-x86_64" \
    || fail "download PeakMiner"
  echo "$PEAK_SHA256  peak" | sha256sum -c --quiet || fail "PeakMiner checksum"
  chmod +x peak
fi
curl -fsSL --retry 3 -o srb.tgz \
  "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
  || fail "download SRBMiner"
echo "$SRB_MD5  srb.tgz" | md5sum -c --quiet || fail "SRBMiner checksum"
tar xzf srb.tgz && SRB=$(ls -d /opt/b/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI && [ -x "$SRB" ] || fail "unpack SRBMiner"
# Our miner: the published release, as a user gets it; on AMD, the HIP core built here.
case ",${ORDER:-peak,srb,ours}," in *,ours,*)
  if [ -z "$AMD" ]; then
    for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do
      curl -fsSL --retry 3 -o "$f" "$REL/$f" || fail "download $f"
    done
    mv llmjob-earn-cli-linux cli && chmod +x cli
  else
    amd_ours
  fi ;;
esac
if [ -z "$AMD" ]; then
  say "ready PeakMiner $PEAK_VER, SRBMiner $SRB_VER, ours $VER"
else
  r="SRBMiner $SRB_VER"; case ",$ORDER," in *,ours,*) r="$r, ours $CORE" ;; esac
  say "ready $r"
  say "srb devices $(cd "$(dirname "$SRB")" && timeout 60 "$SRB" --list-devices $SRB_EXTRA 2>&1 | tr -s '\r\n ' ' ' | tail -c 400)"
fi

peak() { ./peak -o "stratum+tcp://$POOL_HOST:$POOL_PORT" -u "$W.bench-peak" -p x -c pearl; }
srb()  { cd "$(dirname "$SRB")" && "$SRB" --algorithm-gpu pearlhash --pool "$POOL_HOST:$POOL_PORT" --wallet "$W" \
           --worker bench-srb --disable-cpu --api-enable --api-port 21550 ${AMD:+--disable-gpu-nvidia --disable-gpu-intel} \
           $SRB_EXTRA; }
ours() { ./cli -a "$W" --mode mining -w bench-ours --no-report --no-serve --no-update; }
# On AMD: the CLI from source with the core built above (amdtest.sh's mine_cli), on SRBMiner's pool.
[ -n "$AMD" ] && ours() { WALLET=$W POOL=$POOL_HOST:$POOL_PORT WORKER=bench-ours mine_cli; }
stop_all() { pkill -f '\./peak -o'; pkill -f SRBMiner-MULTI; pkill -f '\./cli -a'; [ -n "$AMD" ] && pkill -f 'src/cli/earn-cli\.js'; sleep 8; }

# One sample: the miner's own hashrate (H/s), accepted and rejected shares, and the card's power, clock and temperature,
# then (NVIDIA) its memory clock, GPU use, memory-controller use and memory in use (MiB), in gpu=.
sample() {
  local name=$1 t=$2 hs="" acc="" rej=""
  case $name in
    srb)  read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:21550 \
            | jq -r '[.algorithms[0].hashrate.gpu.total, .algorithms[0].shares.accepted, .algorithms[0].shares.rejected] | @tsv' 2>/dev/null) ;;
    peak) read -r hs acc rej < <(curl -s -m 3 http://127.0.0.1:4068/summary | jq -r '
            def num(p): [paths(numbers) as $q | select(($q|map(tostring)|join(".")) | test(p; "i")) | getpath($q)];
            [(num("hash") | max), (num("accept") | max), (num("reject") | max)] | @tsv' 2>/dev/null) ;;
    ours) local l; l=$(tr '\r' '\n' < ours.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1)
          [ -n "$l" ] && { hs=$(echo "$l" | awk '{printf "%.0f", $1*1e12}'); acc=$(echo "$l" | awk '{gsub(",","",$4); print $4}')
                           rej=$(echo "$l" | awk '{gsub(",","",$7); print $7}'); } ;;
  esac
  local g
  if [ -z "$AMD" ]; then g=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,clocks.mem,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  else g=$(timeout 5 $SMI monitor -p -t -u --csv 2>/dev/null | tail -1 | tr -d ' '); fi
  say "sample $name t=$t hs=${hs:-?} acc=${acc:-?} rej=${rej:-?} gpu=$g"
}

run() {
  local name=$1 secs=$2 measured=$3
  say "start $name ${secs}s"
  ( $name > "/opt/b/$name.log" 2>&1 ) & local pid=$! t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
    sleep 15; local t=$(( $(date +%s) - t0 ))
    if ! kill -0 $pid 2>/dev/null; then
      say "exited $name t=$t: $(tr '\r\n' '  ' < "/opt/b/$name.log" | tail -c 300)"; stop_all; return
    fi
    [ "$measured" = 1 ] && sample "$name" "$t"
  done
  [ "$measured" = 1 ] && say "end $name: $(tr '\r' '\n' < "/opt/b/$name.log" | grep -v '^\s*$' | tail -2 | tr '\n' ' ' | tail -c 300)"
  kill $pid 2>/dev/null; stop_all
}

IFS=, read -ra MINERS <<< "${ORDER:-peak,srb,ours}"
run "${MINERS[0]}" "$WARM" 0          # warm-up: whoever runs first doesn't get a cold card
for m in "${MINERS[@]}"; do
  if [ "$m" = ours ]; then run ours "${OURS_SECS:-300}" 1; else run "$m" "$SECS" 1; fi
done
say "all done"
sleep infinity
