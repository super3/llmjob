#!/bin/bash
# compare-miners.sh -- the number a user compares: each miner's OWN displayed
# hashrate over a 5-minute run, back to back on one card, same pool, same wallet.
#
#   WALLET=prl1... ./compare-miners.sh            (Git Bash on Windows, or Linux)
#
# Runs, in order: a 60 s unmeasured warm-up (so the first miner gets no cold-card
# advantage), our core, SRBMiner, PeakMiner -- SECS each (default 300). Samples every
# 5 s: the miner's reported hashrate (our CLI's status line; SRBMiner's and
# PeakMiner's local JSON APIs) and the card's SM clock, power, temperature and load.
# Then summarises with compare-summary.js.
#
# Nothing here is bundled: SRBMiner and PeakMiner are proprietary and must already be
# installed where SRB_EXE / PEAK_EXE point. Stop anything else on the card first --
# a second process silently corrupts every reading.
set -u
: "${WALLET:?set WALLET to a prl1 address}"
HOST=${HOST:-us2.pearl.herominers.com}; PORT=${PORT:-1200}; SECS=${SECS:-300}
HERE="$(cd "$(dirname "$0")" && pwd)"; EARN="$HERE/../.."
CORE=${CORE:-$EARN/native/build/Release/pearl_core.node}
SRB_EXE=${SRB_EXE:-/c/miners/srb-latest/SRBMiner-Multi-3-6-9/SRBMiner-MULTI.exe}
PEAK_EXE=${PEAK_EXE:-/c/miners/peak/peakminer.exe}
OUT=${OUT:-$HERE/compare-$(date +%Y%m%d-%H%M)}; mkdir -p "$OUT"; echo "out: $OUT"
native() { command -v cygpath >/dev/null && cygpath -w "$1" || echo "$1"; }
ours() { cd "$EARN" && PEARL_CORE_PATH="$(native "$CORE")" node src/cli/earn-cli.js -a "$WALLET" \
  --mode mining -w cmp-ours --no-report --no-serve --no-update; }
srb()  { cd "$(dirname "$SRB_EXE")" && "$SRB_EXE" --algorithm-gpu pearlhash --pool "$HOST:$PORT" \
  --wallet "$WALLET" --worker cmp-srb --disable-cpu --api-enable --api-port 21550; }
peak() { cd "$(dirname "$PEAK_EXE")" && "$PEAK_EXE" -o "stratum+tcp://$HOST:$PORT" -u "$WALLET.cmp-peak" -p x -c pearl; }
stop_all() {
  if command -v taskkill >/dev/null; then
    taskkill //F //T //IM SRBMiner-MULTI.exe >/dev/null 2>&1; taskkill //F //T //IM peakminer.exe >/dev/null 2>&1
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | Where-Object { \$_.CommandLine -like '*earn-cli.js*cmp-ours*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
  else
    pkill -f SRBMiner-MULTI; pkill -f peakminer; pkill -f 'earn-cli.js.*cmp-ours'
  fi
}
run() { name=$1; fn=$2; secs=$3; echo "== $name ($secs s) $(date +%T)"
  ( $fn > "$OUT/$name.log" 2>&1 ) & pid=$!; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
    ts=$(( $(date +%s) - t0 ))
    echo "$ts,$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" >> "$OUT/$name.gpu.csv"
    case $name in
      srb)  echo "$ts $(curl -s -m 2 http://127.0.0.1:21550 | tr -d '\n')" >> "$OUT/srb.api" ;;
      peak) echo "$ts $(curl -s -m 2 http://127.0.0.1:4068/summary | tr -d '\n')" >> "$OUT/peak.api" ;;
    esac
    sleep 5
  done
  kill $pid 2>/dev/null; stop_all; sleep 8; }
stop_all
run warmup ours 60; rm -f "$OUT"/warmup.*
run ours ours "$SECS"
[ -x "$SRB_EXE" ] && run srb srb "$SECS" || echo "(no SRBMiner at $SRB_EXE)"
[ -x "$PEAK_EXE" ] && run peak peak "$SECS" || echo "(no PeakMiner at $PEAK_EXE)"
node "$HERE/compare-summary.js" "$OUT"
