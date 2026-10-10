#!/usr/bin/env bash
# Every check the AMD core can pass without a GPU, on the HIP-CPU build (build-amd.sh cpu
# first). Prints one line a check and exits non-zero if any fails. About seven minutes on
# four cores.
#
#   1. pearl_amd_foldtest_cpu: each of the five folds, through the emulated matrix
#      instructions, against a plain CPU computation of every transcript word.
#   2. verify-hits-profile.js on pearl_core_hipcpu.node: the whole core -- draws,
#      restamps, the fold, the hash, the hit list, the share proofs -- with every fold,
#      both operand fills, and a profile whose salt takes two batches.
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

run "foldtest 256x512" "$B/pearl_amd_foldtest_cpu" 256 512 1
run "foldtest 512x1024" "$B/pearl_amd_foldtest_cpu" 512 1024 7
for f in ref mfma16 mfma8 wmma11 wmma12; do
  run "core $f, constant fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 1
done
for f in mfma16 wmma12; do
  run "core $f, hashed fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 0
done
run "core mfma16, two batches a salt" env PEARL_AMD_FOLD=mfma16 node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 254 256 8192 1
exit $fail
