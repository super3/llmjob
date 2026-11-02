#!/usr/bin/env bash
# Every check the AMD core can pass without a GPU, on the HIP-CPU build (build-amd.sh cpu
# first). Prints one line a check and exits non-zero if any fails. About ten minutes on
# four cores.
#
#   1. pearl_amd_foldtest_cpu: each of the six folds, the matrix ones through the emulated
#      matrix instructions, against a plain CPU computation of every transcript word.
#      Also built with the RDNA 3 loop (PF2) and the RDNA 4 one (PF2 + PFX + TID), with
#      the wrap readout RDNA 4 mines k 4096 with (_kwrap), at k 2048 and k 4096, and dot4
#      with its other k-steps (_dot4k8, _dot4k16).
#   2. verify-hits-profile.js on pearl_core_hipcpu.node: the whole core -- draws,
#      restamps, the fold, the hash, the hit list, the share proofs -- with every fold,
#      both operand fills, and a profile whose salt takes two batches.
#   3. The same on pearl_core_hipcpu_kwrap.node with cardK, as the app opens a core: the
#      core picks k 4096, as on an RDNA 4 card, and every hit is checked at that k. The
#      default core with cardK stays at k 2048, as every other card does.
#
# SHARE_DIR, when set, is a folder the k 4096 runs write their shares to (verify-hits-profile.js's
# PEARL_SHARE_DUMP), for the reference verifier (probes/refverify.rs) to check afterwards.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$HERE/../build"
SECS="${SECS:-60}"
fail=0

run() {
  local name="$1"
  shift
  if out="$("$@" 2>&1)"; then
    echo "PASS  $name  $(echo "$out" | tail -n 1)"
  else
    echo "FAIL  $name"
    echo "$out" | tail -n 8 | sed 's/^/      /'
    fail=1
  fi
}

# The k a verify-hits-profile.js run checked, from its JSON line: it must be the one the core was expected to pick.
want_k() {
  local k="$1"
  shift
  out="$("$@" 2>&1)"; local rc=$?
  echo "$out"
  [ $rc -eq 0 ] && echo "$out" | tail -n 1 | grep -q "\"k\":$k[,}]"
}

dump() { [ -n "${SHARE_DIR:-}" ] && echo "PEARL_SHARE_DUMP=$SHARE_DIR/$1.jsonl" || echo "PEARL_SHARE_DUMP="; }
[ -n "${SHARE_DIR:-}" ] && mkdir -p "$SHARE_DIR"

run "foldtest 256x512" "$B/pearl_amd_foldtest_cpu" 256 512 1
run "foldtest 512x1024" "$B/pearl_amd_foldtest_cpu" 512 1024 7
run "foldtest 256x512, RDNA 3 loop (PF2)" "$B/pearl_amd_foldtest_cpu_pf2" 256 512 1
run "foldtest 256x512, RDNA 4 loop (PF2 + PFX + TID)" "$B/pearl_amd_foldtest_cpu_pfx" 256 512 1
run "foldtest 256x512, wrap readout, k 2048" "$B/pearl_amd_foldtest_cpu_kwrap" 256 512 1 1 1 all 2048
run "foldtest 256x512, wrap readout, k 4096 (its default)" "$B/pearl_amd_foldtest_cpu_kwrap" 256 512 3
for dk in 8 16; do
  run "foldtest 256x512, dot4, k-step $dk" "$B/pearl_amd_foldtest_cpu_dot4k$dk" 256 512 5 1 1 dot4
done
for f in ref mfma16 mfma8 wmma11 wmma12 dot4; do
  run "core $f, constant fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 1
done
for f in mfma16 wmma12; do
  run "core $f, hashed fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 0
done
run "core mfma16, two batches a salt" env PEARL_AMD_FOLD=mfma16 node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 254 256 8192 1
for f in wmma12 ref dot4; do
  run "core $f, wrap readout, cardK: k 4096" want_k 4096 env PEARL_AMD_FOLD=$f PEARL_CARD_K=1 "$(dump "kwrap-$f")" \
    node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu_kwrap.node" "$SECS" 252 256 512 1
done
run "core wmma12, wrap readout, cardK, hashed fill: k 4096" want_k 4096 env PEARL_AMD_FOLD=wmma12 PEARL_CARD_K=1 \
  "$(dump kwrap-wmma12-hashed)" node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu_kwrap.node" "$SECS" 252 256 512 0
run "core wmma12, wrap readout, cardK, PEARL_AMD_K=2048: k 2048" want_k 2048 env PEARL_AMD_FOLD=wmma12 PEARL_CARD_K=1 \
  PEARL_AMD_K=2048 node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu_kwrap.node" "$SECS" 252 256 512 1
run "core wmma12, today's readout, cardK: k 2048" want_k 2048 env PEARL_AMD_FOLD=wmma12 PEARL_CARD_K=1 \
  node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 1
exit $fail
