#!/usr/bin/env bash
# A real pool run with the CLI, through the app's existing opt-in for another
# core (PEARL_CORE_PATH). Nothing in the app changes.
#
#   ./pool-run.sh <prl1p...address> [seconds=300] [region]
#
# Mines with --mode mining and nothing else (no LLM, no board report, no job
# serving, no self-update), writes the CLI's log to pool-run.log and its stats
# to pool-run-stats.json, and prints the last stats at the end: accepted and
# rejected shares and the rate. Regions: us us2 ca br de fi fr tr sg hk kr au
# (default: the CLI picks the fastest).
#
# One card mines unless PEARL_GPU_INDEX lists more: PEARL_GPU_INDEX=0,1,2,3,4,5
# for a rig of six. The CLI finds cards with nvidia-smi, which lists no Intel
# card, so it cannot count them itself.
#
# Every Intel GPU is an untested kind of card, so the CLI recomputes each hit
# before it sends it and turns a card off at its first wrong one (README.md,
# "Untested cards"). The log says so at the first hit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${PEARL_SYCL_BUILD:-$HERE/build}
EARN=$(cd "$HERE/../.." && pwd)
ADDR=${1:?usage: ./pool-run.sh <prl1p...address> [seconds=300] [region]}
SECS=${2:-300}
REGION=${3:-}
# The CLI is built and tested on Node 22 (earn/package.json's engines).
node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' \
  || { echo "the CLI needs Node 22 or newer; this is $(node -v)" >&2; exit 1; }
# The CLI needs two small packages, tweetnacl and tweetnacl-util. A checkout
# without them gets them in build/cli-deps, through NODE_PATH: npm install in
# earn/ would install every package the desktop app uses, Electron among them.
if ! (cd "$EARN" && node -e "require('tweetnacl'); require('tweetnacl-util')" 2>/dev/null); then
  DEPS="$B/cli-deps"
  [ -d "$DEPS/node_modules/tweetnacl-util" ] \
    || npm install --no-save --no-package-lock --no-audit --no-fund --prefix "$DEPS" tweetnacl@1 tweetnacl-util@0.15 >/dev/null
  export NODE_PATH="$DEPS/node_modules${NODE_PATH:+:$NODE_PATH}"
fi
STATS="$HERE/pool-run-stats.json"
rm -f "$STATS"
ARGS=(--address "$ADDR" --mode mining --no-report --no-serve --no-update --stats-file "$STATS")
[ -n "$REGION" ] && ARGS+=(--region "$REGION")
echo "== $SECS s on the pool with $B/pearl_core_sycl.node"
PEARL_CORE_PATH="$B/pearl_core_sycl.node" timeout --signal=INT "$SECS" \
  node "$EARN/src/cli/earn-cli.js" "${ARGS[@]}" 2>&1 | tee "$HERE/pool-run.log" || true
echo "== last stats ($STATS)"
cat "$STATS" 2>/dev/null || echo "no stats file was written"
