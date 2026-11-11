#!/usr/bin/env bash
# The correctness gate. Run it on an Intel GPU first; nothing else is worth
# measuring until it passes.
#
#   ./verify.sh            fold check, then 400 hits through the addon with the
#                          core's own fold and 400 with the plain fold (on the
#                          CPU device: the plain fold and the three XMX folds)
#   ./verify.sh --quick    100 hits each, at half the bound, for half the batches
#
# 1. pearl_sycl_check: every region of 4 batches, with the plain fold and the
#    core's own fold, each against a scalar fold on the host. Hit counts must
#    agree as well. On an XMX GPU the core's own fold is the hardware DPAS, so
#    this is the hardware check. The reference code of that fold is run after
#    it and only reported: it mines nothing, and on an Arc A750 it is the one
#    thing that differs from the host (README.md, "Results on an Arc A750").
# 2. The same check at the mainnet m and n, on the last column batch (the
#    largest offsets), so a fold that loses hits only at full size fails too.
# 3. verify-hits.js: hits from the addon (pearl_core.cc unchanged), each
#    recomputed from scratch in JS the way the pool's verifier would. At the
#    mainnet m and n on a GPU; at m = n = 4096 with PEARL_SYCL_DEVICE=cpu.
#    Its expectedHits and hitRatio are a rough check that none go missing.
#
# Exits non-zero on the first failure. PEARL_SYCL_BUILD names the build
# directory (default ./build). PEARL_GPU_INDEX=N checks card N of a rig (one
# number; default: the core's pick), so a rig runs this once a card.
#
# Both tools check what the core gives them and send nothing, so the core opens
# a GPU for them (README.md, "Untested cards").
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${PEARL_SYCL_BUILD:-$HERE/build}
HITS=400
QUICK=0
EMU=()
if [ "${1:-}" = "--quick" ]; then HITS=100; QUICK=1; fi
# The mainnet m and n, 4 column offsets a batch, and the last of the 4096
# batches: 32,768 regions with the largest row and column offsets.
MAIN=(--m 131072 --n 262144 --col-batch 4 --first 4095 --batches 1 --bits 247)
# The bound for verify-hits (--bits) gives about 64 hits a salt at either
# size below: 400 hits come from 6 salts, and it wants at least 3.
if [ "${PEARL_SYCL_DEVICE:-gpu}" = "cpu" ]; then
  # The CPU device has no XMX, so its own fold is the plain one; the XMX folds
  # (the specification's reference code there) are asked for by name.
  CHECK=(--m 1024 --n 1024 --col-batch 16 --batches 4 --bits 248 --folds dot,xmx16,xmx8,xmx8f)
  MAIN+=(--folds dot,xmx16,xmx8f)
  HITARGS=(--m 4096 --n 4096 --col-batch 64)
  BITS=246
  FOLDS=(dot xmx16 xmx8 xmx8f)
else
  # The folds that mine, each against the host: the plain fold and the core's
  # own. The reference code of the core's fold is run too, but only reported
  # (EMU): on an Arc A750 the hardware DPAS matches the host and the
  # sub-group 8 reference code does not (README.md, "Results on an Arc A750").
  CHECK=(--m 2048 --n 2048 --col-batch 32 --batches 4 --bits 247 --folds dot,auto)
  EMU=(--m 2048 --n 2048 --col-batch 32 --batches 4 --bits 247 --folds auto-emu)
  MAIN+=(--folds dot,auto)
  HITARGS=()
  BITS=235
  FOLDS=(auto dot)
fi
# --quick's 100 hits would come from 2 salts at that bound, one short of the
# rule, so it could not pass. At half the bound a salt gives 32, so they come
# from 3 or 4 salts, in half the batches 400 take.
if [ "$QUICK" = 1 ]; then BITS=$((BITS - 1)); fi
HITARGS+=(--bits "$BITS")
echo "== fold check: pearl_sycl_check ${CHECK[*]}"
"$B/pearl_sycl_check" "${CHECK[@]}"
if [ ${#EMU[@]} -gt 0 ]; then
  echo "== for information, not part of PASS: the reference code of the core's fold"
  if ! "$B/pearl_sycl_check" "${EMU[@]}"; then
    echo "note: the reference code differs from the host on this device; the folds checked above are the ones that mine"
  fi
fi
echo "== fold check at the mainnet m and n: pearl_sycl_check ${MAIN[*]}"
"$B/pearl_sycl_check" "${MAIN[@]}"
for f in "${FOLDS[@]}"; do
  echo "== $HITS hits, PEARL_SYCL_FOLD=$f"
  PEARL_SYCL_FOLD=$f node "$HERE/verify-hits.js" "$B/pearl_core_sycl.node" --hits "$HITS" --seconds 1800 "${HITARGS[@]}"
done
echo "== PASS"
