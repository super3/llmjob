#!/bin/bash
# Benchmarks PeakMiner, SRBMiner and our miner on one Salad GPU, one after another, the way benchmark.md's
# "How a host is tested" does: a 60 s unmeasured warm-up, then each miner for 5 minutes on HeroMiners.
# bench.py starts it as a Salad container command and reads its "[bench]" lines from Salad's logs.
#
# Environment (set by bench.py): W wallet, ORDER (e.g. "peak,srb,ours"), OURS_SECS (300, or 90 on a card
# our release has no build for). Every line it prints for bench.py starts "[bench] HH:MM:SS".
export DEBIAN_FRONTEND=noninteractive
POOL_HOST=us.pearl.herominers.com; POOL_PORT=1200; SECS=300; WARM=60; VER=v0.6.0
REL=https://github.com/super3/llmjob/releases/download/$VER
say() { echo "[bench] $(date -u +%T) $*"; }
fail() { say "FAIL $*"; say "all done"; sleep infinity; }

apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq --no-install-recommends curl ca-certificates jq procps \
  >/dev/null 2>&1 || fail "apt-get install"
say "card $(nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,memory.total,compute_cap --format=csv,noheader | head -1)"

mkdir -p /opt/b && cd /opt/b || fail "no /opt/b"
# The two miners from their official releases, checked against the published checksums.
curl -fsSL --retry 3 -o peak https://github.com/peakminer/peakminer/releases/download/v2.17.6/peakminer-2.17.6-linux-x86_64 \
  || fail "download PeakMiner"
echo "573c3412ddac99fe2c0eca790dabebf9a029cbe38a2811784e93c4eddc2ff65c  peak" | sha256sum -c --quiet || fail "PeakMiner checksum"
chmod +x peak
curl -fsSL --retry 3 -o srb.tgz https://github.com/doktor83/SRBMiner-Multi/releases/download/3.7.1/SRBMiner-Multi-3-7-1-Linux.tar.gz \
  || fail "download SRBMiner"
echo "112223a663506023e31d7f9dbce3dfe2  srb.tgz" | md5sum -c --quiet || fail "SRBMiner checksum"
tar xzf srb.tgz && SRB=$(ls -d /opt/b/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI && [ -x "$SRB" ] || fail "unpack SRBMiner"
# Our miner: the published release, as a user gets it.
for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do
  curl -fsSL --retry 3 -o "$f" "$REL/$f" || fail "download $f"
done
mv llmjob-earn-cli-linux cli && chmod +x cli
say "ready PeakMiner 2.17.6, SRBMiner 3.7.1, ours $VER"

peak() { ./peak -o "stratum+tcp://$POOL_HOST:$POOL_PORT" -u "$W.bench-peak" -p x -c pearl; }
srb()  { cd "$(dirname "$SRB")" && "$SRB" --algorithm-gpu pearlhash --pool "$POOL_HOST:$POOL_PORT" --wallet "$W" \
           --worker bench-srb --disable-cpu --api-enable --api-port 21550; }
ours() { ./cli -a "$W" --mode mining -w bench-ours --no-report --no-serve --no-update; }
stop_all() { pkill -f '\./peak -o'; pkill -f SRBMiner-MULTI; pkill -f '\./cli -a'; sleep 8; }

# One sample: the miner's own hashrate (H/s), accepted and rejected shares, and the card's power, clock and temperature.
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
  local g; g=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
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
