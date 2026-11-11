#!/usr/bin/env bash
# The speed test, at the mainnet profile. Run verify.sh first.
#
#   ./bench.sh             each fold alone for 20 s (pearl_sycl_check --bench),
#                          then the full miner loop for 60 s (bench.js)
#   ./bench.sh --sweep     also the core's fold at other batch widths and bands
#
# Prints JSON lines. TH/s is MACs a second / 1e12, the unit the pool and the app
# use, so it compares directly with benchmark.md and with other miners.
#
# PEARL_GPU_INDEX=N times card N of a rig (one number; default: the core's pick).
# bench.js checks no hits and sends no shares, so it is given
# PEARL_SYCL_UNTESTED=1, without which the core opens no GPU for it. That turns
# no check off: the app and verify-hits.js still check every hit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${PEARL_SYCL_BUILD:-$HERE/build}
SECS=${BENCH_SECONDS:-20}
# Each fold in a core that would mine with it: the band depends on the fold.
echo "== each fold alone, ${SECS} s"
PEARL_SYCL_FOLD=dot "$B/pearl_sycl_check" --bench "$SECS" --folds dot
"$B/pearl_sycl_check" --bench "$SECS" --folds auto
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
PEARL_SYCL_UNTESTED=1 node "$HERE/bench.js" "$B/pearl_core_sycl.node" --seconds 60 --warmup 10
