#!/usr/bin/env bash
# The correctness gate. Run it on an Intel GPU first; nothing else is worth
# measuring until it passes.
#
#   ./verify.sh            fold check, then 400 hits through the addon with the
#                          core's own fold and 400 with the plain fold (on the
#                          CPU device: the plain fold and both XMX folds)
#   ./verify.sh --quick    100 hits each
#
# 1. pearl_sycl_check: every region of 4 batches, with the plain fold, the core's
#    own fold and (on an XMX GPU) the reference code of that fold, against a
#    scalar fold on the host. Hit counts must agree as well. On an XMX GPU this is
#    the hardware self-check: a DPAS layout mistake fails "auto" while "auto-emu"
#    passes.
# 2. The same check at the mainnet m and n, on the last column batch (the
#    largest offsets), so a fold that loses hits only at full size fails too.
# 3. verify-hits.js: hits from the addon (pearl_core.cc unchanged), each
#    recomputed from scratch in JS the way the pool's verifier would. At the
#    mainnet m and n on a GPU; at m = n = 4096 with PEARL_SYCL_DEVICE=cpu.
#    Its expectedHits and hitRatio are a rough check that none go missing.
#
# Exits non-zero on the first failure. PEARL_SYCL_BUILD names the build
# directory (default ./build).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${PEARL_SYCL_BUILD:-$HERE/build}
HITS=400
[ "${1:-}" = "--quick" ] && HITS=100
# The mainnet m and n, 4 column offsets a batch, and the last of the 4096
# batches: 32,768 regions with the largest row and column offsets.
MAIN=(--m 131072 --n 262144 --col-batch 4 --first 4095 --batches 1 --bits 247)
if [ "${PEARL_SYCL_DEVICE:-gpu}" = "cpu" ]; then
  # The CPU device has no XMX, so its own fold is the plain one; the XMX folds
  # (the specification's reference code there) are asked for by name.
  CHECK=(--m 1024 --n 1024 --col-batch 16 --batches 4 --bits 248 --folds dot,xmx16,xmx8)
  MAIN+=(--folds dot,xmx16)
  HITARGS=(--m 4096 --n 4096 --col-batch 64 --bits 246)
  FOLDS=(dot xmx16 xmx8)
else
  CHECK=(--m 2048 --n 2048 --col-batch 32 --batches 4 --bits 247 --folds dot,auto,auto-emu)
  MAIN+=(--folds dot,auto,auto-emu)
  HITARGS=(--bits 235)
  FOLDS=(auto dot)
fi
echo "== fold check: pearl_sycl_check ${CHECK[*]}"
"$B/pearl_sycl_check" "${CHECK[@]}"
echo "== fold check at the mainnet m and n: pearl_sycl_check ${MAIN[*]}"
"$B/pearl_sycl_check" "${MAIN[@]}"
for f in "${FOLDS[@]}"; do
  echo "== $HITS hits, PEARL_SYCL_FOLD=$f"
  PEARL_SYCL_FOLD=$f node "$HERE/verify-hits.js" "$B/pearl_core_sycl.node" --hits "$HITS" --seconds 1800 "${HITARGS[@]}"
done
echo "== PASS"
