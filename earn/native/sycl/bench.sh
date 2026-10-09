#!/usr/bin/env bash
# The speed test, at the mainnet profile. Run verify.sh first.
#
#   ./bench.sh             each fold alone for 20 s (pearl_sycl_check --bench),
#                          then the full miner loop for 60 s (bench.js)
#   ./bench.sh --sweep     also the core's fold at other batch widths and bands
#
# Prints JSON lines. TH/s is MACs a second / 1e12, the unit the pool and the app
# use, so it compares directly with benchmark.md and with other miners.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${PEARL_SYCL_BUILD:-$HERE/build}
SECS=${BENCH_SECONDS:-20}
echo "== each fold alone, ${SECS} s"
"$B/pearl_sycl_check" --bench "$SECS" --folds dot,auto
if [ "${1:-}" = "--sweep" ]; then
  for cb in 64 128 512 1024; do
    echo "== PEARL_SYCL_COL_BATCH=$cb"
    PEARL_SYCL_COL_BATCH=$cb "$B/pearl_sycl_check" --bench "$SECS" --folds auto
  done
  for band in 4 16 64 256; do
    echo "== PEARL_SYCL_BAND=$band"
    PEARL_SYCL_BAND=$band "$B/pearl_sycl_check" --bench "$SECS" --folds auto
  done
fi
echo "== the full miner loop, 60 s after 10 s of warm-up"
node "$HERE/bench.js" "$B/pearl_core_sycl.node" --seconds 60 --warmup 10
