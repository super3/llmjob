#!/usr/bin/env bash
# Compile the SYCL core for each Intel GPU target on its own and print what IGC
# made of the folds: SIMD width, register mode, DPAS count, spill. One build a
# target, because IGC names its dump files by shader hash and targets of one
# build overwrite each other's.
#
#   ./compile-gate.sh [TARGET ...]     (default: every target build.sh uses)
#
# Fails if a target does not build, if a hardware XMX fold has no DPAS, or if a
# fold spills on its own hardware target. On Alchemist that is two folds: the
# fragment fold the card mines with, which must also get the 256-register mode
# its kernel property asks for, and fold_xmx, which PEARL_SYCL_FOLD=xmx8 still
# picks. The emulated folds (emu) are for the CPU device and the hardware
# self-check; they are listed, not gated.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(intel_gpu_bmg_g21 intel_gpu_bmg_g31 intel_gpu_lnl_m intel_gpu_ptl_h \
  intel_gpu_ptl_u intel_gpu_pvc intel_gpu_acm_g10 intel_gpu_acm_g11 intel_gpu_acm_g12)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
for t in "${TARGETS[@]}"; do
  echo "== $t"
  if ! "$HERE/build.sh" --targets "$t" --dump "$WORK/dump-$t" --out "$WORK/out-$t" > "$WORK/log-$t" 2>&1; then
    echo "BUILD FAILED"; tail -20 "$WORK/log-$t"; fail=1; continue
  fi
  grep -i "spill" "$WORK/log-$t" || true
  node "$HERE/isa-report.js" "$WORK/dump-$t" | awk 'NR == 1 || /fold/'
  # The hardware folds for this target's sub-group size. Columns: 2 the
  # kernel, 4 the register mode, 5 DPAS, 6 spill.
  case "$t" in
    intel_gpu_acm_*) hws=('fold_xmxf<sg8,rb2,hw>' 'fold_xmx<sg8,rb1,hw>') ;;
    *) hws=('fold_xmx<sg16,rb2,hw>') ;;
  esac
  for hw in "${hws[@]}"; do
    node "$HERE/isa-report.js" "$WORK/dump-$t" | awk -v k="$hw" '$2 == k && ($5 == 0 || $6 != 0) { bad = 1 } $2 == k { seen = 1 }
      $2 == "fold_xmxf<sg8,rb2,hw>" && $4 != 256 { bad = 1 }
      $2 == "fold_dot" && $6 != 0 { bad = 1 } END { exit (bad || !seen) }' || { echo "GATE FAILED: $hw needs DPAS and no spill (the fragment fold 256 registers too), fold_dot no spill"; fail=1; }
  done
  rm -rf "$WORK/dump-$t" "$WORK/out-$t"
done
exit $fail
