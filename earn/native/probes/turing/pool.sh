#!/bin/bash
# Our miner on a pool, the way benchmark.md's "How a host is tested" runs it, but with
# this checkout's core instead of the published release files: a different method, so
# a result goes in a section's notes, never into a row.
#
#   WALLET=prl1... SRC=<llmjob checkout> OUT=<scratch dir> bash earn/native/probes/turing/pool.sh
#
# Builds the default core with compilers.sh, runs verify-hits.js on it for 90 s, then
# runs the CLI on POOL (default us.pearl.herominers.com:1200, the benchmark's pool): a
# 60 s warm-up that is not measured, then SECS (300) measured. Our rate is the mean of
# the CLI's readings after the first minute of the measured run, each the average
# since the one before, as benchmark.md takes it; the SM clock, power and temperature
# are medians of nvidia-smi samples every 5 s over the same span. Only our miner runs:
# PeakMiner's and SRBMiner's figures are the rows' fixed targets.
#
# Every result is one line starting with [pool], under 500 characters (Vast's log API
# cuts longer lines). Assumes an idle GPU 0 and the nvidia/cuda:12.8.1-devel image that
# compilers.sh does.
set -u
: "${WALLET:?set WALLET to a prl1 address}" "${SRC:?set SRC to the repo checkout}" "${OUT:?set OUT to a scratch directory}"
POOL=${POOL:-us.pearl.herominers.com:1200}; SECS=${SECS:-300}; WARM=${WARM:-60}
WORKER=${WORKER:-bench-$(hostname | tr -cd 'a-zA-Z0-9' | cut -c1-12)}
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
say() { printf '[pool] %s\n' "$*" | cut -c1-490; }

say "start $(date -u +%FT%TZ) commit $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null) pool $POOL worker $WORKER"
say "apps on the GPU: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"

# 1. The default core, built as compilers.sh builds base.
STAGES="setup build" VARIANTS_SPEC='base|' SRC="$SRC" OUT="$OUT/cs" \
  bash "$SRC/earn/native/probes/turing/compilers.sh" > "$OUT/compilers.log" 2>&1
grep -a '^\[compilers\] \(card \|build \)' "$OUT/compilers.log" | sed 's/^\[compilers\] /[pool] core: /' | cut -c1-490
CORE=$OUT/cs/v/base/build/Release/pearl_core.node
[ -f "$CORE" ] || { say "FAIL no core: $(tail -2 "$OUT/compilers.log" | tr '\n' ' ')"; exit 1; }
export PATH="$OUT/cs/node/bin:$PATH"

# 2. The CLI's own dependencies (three small packages; no Electron).
if ! (cd "$SRC/earn" && npm ci --omit=dev --ignore-scripts --no-audit --no-fund > "$OUT/npm.log" 2>&1); then
  say "FAIL npm ci: $(tail -2 "$OUT/npm.log" | tr '\n' ' ')"; exit 1
fi

# 3. The hit check: 400 hits recomputed from scratch, as the pool's verifier would.
v=$(cd "$SRC/earn" && timeout -k 10 300 node native/probes/verify-hits.js "$CORE" 90 2>&1 | grep -o '"hits".*' | tail -1)
say "verify ${v:-FAIL no result}"

# 4. The runs. The CLI runs in its own process group, so stopping it stops the core's
# worker too.
run() { # name seconds
  local name=$1 secs=$2 t0 pid
  (cd "$SRC/earn" && exec setsid env PEARL_CORE_PATH="$CORE" node src/cli/earn-cli.js -a "$WALLET" \
     --mode mining --pool "$POOL" -w "$WORKER" --no-report --no-serve --no-update) > "$OUT/$name.log" 2>&1 &
  pid=$!; t0=$(date +%s)
  : > "$OUT/$name.gpu.csv"
  while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
    echo "$(( $(date +%s) - t0 )),$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> "$OUT/$name.gpu.csv"
    sleep 5
  done
  kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # In case setsid forked and the group is not $pid's: no other process here runs the CLI.
  pkill -KILL -f 'src/cli/earn-cli.js' 2>/dev/null
  sleep 8
}
run warmup "$WARM"
say "warm-up done: $(grep -a 'TH/s' "$OUT/warmup.log" | tail -1 | tr -cd '[:print:]' | cut -c1-200)"
run ours "$SECS"

# 5. The summary, from the CLI's status lines ("<rate> TH/s · <n> accepted · <m>
# rejected · up <M>m <S>s", one a second) and the GPU samples.
node -e '
const fs = require("fs"); const [log, csv] = process.argv.slice(1);
const pts = fs.readFileSync(log, "utf8").split("\n").filter((l) => l.includes("TH/s")).map((l) => {
  const th = l.match(/([\d.]+) TH\/s/); const up = l.match(/up (?:(\d+)h )?(\d+)m (\d+)s/);
  return th && up ? { t: (+(up[1] || 0)) * 3600 + +up[2] * 60 + +up[3], th: +th[1],
    acc: +(l.match(/([\d,]+) accepted/) || [0, "0"])[1].replace(/,/g, ""), rej: +(l.match(/(\d+) rejected/) || [0, 0])[1] } : null;
}).filter(Boolean);
const steady = pts.filter((p) => p.t >= 60); const last = pts[pts.length - 1] || {};
const med = (a) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const g = fs.readFileSync(csv, "utf8").trim().split("\n").map((l) => l.split(",").map(Number)).filter((r) => r[0] >= 60 && r.length >= 5);
const mean = steady.reduce((s, p) => s + p.th, 0) / steady.length;
console.log("[pool] result " + (steady.length ? mean.toFixed(2) : "FAIL") + " TH/s mean of " + steady.length +
  " readings after the first minute; at the end " + last.th + " TH/s, up " + last.t + " s; shares " + last.acc + "/" + last.rej +
  "; SM " + med(g.map((r) => r[1])) + " MHz, " + med(g.map((r) => r[2])) + " W, " + med(g.map((r) => r[3])) + " C, use " + med(g.map((r) => r[4])) + "% (medians of " + g.length + ")");
' "$OUT/ours.log" "$OUT/ours.gpu.csv"
grep -a -i 'error\|fail' "$OUT/ours.log" | head -3 | while IFS= read -r l; do say "cli: $(echo "$l" | tr -cd '[:print:]')"; done
say "done $(date -u +%T)"
