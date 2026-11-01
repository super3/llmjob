#!/usr/bin/env bash
# Every check the AMD core can pass without a GPU, on the HIP-CPU build (build-amd.sh cpu
# first). Prints one line a check and exits non-zero if any fails. About seven minutes on
# four cores before the readout v2 checks; those add a few more (not yet timed).
#
#   1. pearl_amd_foldtest_cpu: each of the five folds, through the emulated matrix
#      instructions, against a plain CPU computation of every transcript word.
#      Also built with the RDNA 3 loop (PF2) and the RDNA 4 one (PF2 + PFX + TID), and with
#      readout v2 in wmma11 and wmma12 (PEARL_AMD_READOUT_V2) on each loop.
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
run "foldtest 256x512, RDNA 3 loop (PF2)" "$B/pearl_amd_foldtest_cpu_pf2" 256 512 1
run "foldtest 256x512, RDNA 4 loop (PF2 + PFX + TID)" "$B/pearl_amd_foldtest_cpu_pfx" 256 512 1
# Readout v2 against the same 16x16 reference. The first runs every fold, so the CDNA and
# reference folds show they're unchanged in that build. At 512x1024 (16 blocks) a block may
# start on shared memory an earlier one left full of words; on a card, the fold test at
# 1024x4096 (128 blocks, more than fit at once) is the check that v2 zeroes its words.
run "foldtest 256x512, readout v2" "$B/pearl_amd_foldtest_cpu_v2" 256 512 1
run "foldtest 512x1024, readout v2, wmma12" "$B/pearl_amd_foldtest_cpu_v2" 512 1024 7 1 1 wmma12
run "foldtest 256x512, readout v2, RDNA 3 loop (PF2), wmma11" "$B/pearl_amd_foldtest_cpu_v2_pf2" 256 512 1 1 1 wmma11
run "foldtest 512x1024, readout v2, RDNA 4 loop, wmma12" "$B/pearl_amd_foldtest_cpu_v2_pfx" 512 1024 7 1 1 wmma12
run "foldtest 256x512, readout v2 as ds_xor_b32, RDNA 4 loop, wmma12" "$B/pearl_amd_foldtest_cpu_v2w_pfx" 256 512 1 1 1 wmma12
for f in ref mfma16 mfma8 wmma11 wmma12; do
  run "core $f, constant fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 1
done
for f in mfma16 wmma12; do
  run "core $f, hashed fill" env PEARL_AMD_FOLD=$f node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 252 256 512 0
done
run "core mfma16, two batches a salt" env PEARL_AMD_FOLD=mfma16 node "$HERE/verify-hits-profile.js" "$B/pearl_core_hipcpu.node" "$SECS" 254 256 8192 1
exit $fail
