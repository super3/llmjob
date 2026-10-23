#!/bin/bash
# A tuning sweep of the AMD folds on one PC, so variants are compared on the same card. Run from
# earn/native/amd by amdtest.sh's `run` task (RUN_GZ_B64 holding "bash test/lane-a-sweep.sh"), after `build`.
#
# For each variant in SWEEP it builds the core with those compiler flags (PEARL_AMD_CFLAGS), checks the fold
# bit-exact against the CPU (pearl_amd_foldtest at 1024 x 1024), and times the fold alone at the mainnet m. Then it
# runs hashrate.js on every variant, SWEEP_ROUNDS rounds in turn, and on the first variant once for each batch
# width in SWEEP_CB (PEARL_AMD_COL_BATCH, no rebuild).
#
# Environment:
#   SWEEP         "name=flags;name=flags", e.g. "tiled=;rowmajor=-DPEARL_AMD_TILED=0" (default: one build, "base=")
#   SWEEP_FOLD    m:cols:reps for the fold-only timing, several space-separated (default 131072:4096:10); ""
#                 skips it
#   SWEEP_HASH    secs:warm for each hashrate.js run (default 40:10)
#   SWEEP_ROUNDS  rounds of hashrate.js over the variants (default 2)
#   SWEEP_CB      batch widths to try on one variant, e.g. "64 128" (default none)
#   SWEEP_CBV     the variant SWEEP_CB runs on (default the first)
#
# Each result is one "[amdtest] HH:MM:SS sweep ..." line, written straight to the container's log too, so it shows
# while the sweep runs (the run task prints only the end of its output, when it finishes).
set -u
A=$(pwd)
ARCH=${ARCH:-$(timeout 60 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1)}
FOLD=$(case "$ARCH" in gfx11*) echo wmma11 ;; gfx12*) echo wmma12 ;; gfx94*|gfx95*) echo mfma16 ;; gfx90*) echo mfma8 ;; *) echo ref ;; esac)
out() {   # to the container's log now, and to the run task's output for its summary at the end
  local l="[amdtest] $(date -u +%T) sweep $*"
  echo "$l"
  { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true
}

SWEEP=${SWEEP:-base=}
SWEEP_FOLD=${SWEEP_FOLD-131072:4096:10}
IFS=: read -r HSECS HWARM <<< "${SWEEP_HASH:-40:10}"
ROUNDS=${SWEEP_ROUNDS:-2}
out "start $ARCH fold $FOLD; variants: $SWEEP; fold timing ${SWEEP_FOLD:-none}; hashrate ${HSECS}s+${HWARM}s x $ROUNDS"

names=()
IFS=';' read -ra VARS <<< "$SWEEP"
for v in "${VARS[@]}"; do
  [ -n "$v" ] || continue
  name=${v%%=*}; flags=${v#*=}
  t0=$(date +%s)
  if ! PEARL_AMD_CFLAGS="$flags" ./build-amd.sh gpu "$ARCH" > /root/sweep-build-$name.log 2>&1; then
    out "$name build FAIL ($flags)"
    grep -E 'error' /root/sweep-build-$name.log | head -5 | while IFS= read -r l; do out "$name | ${l:0:300}"; done
    continue
  fi
  mkdir -p /root/var/$name
  cp build/pearl_core_hip.node build/pearl_amd_foldtest /root/var/$name/
  chk=$(timeout 600 /root/var/$name/pearl_amd_foldtest 1024 1024 7 1 1 "$FOLD" 2>&1 | grep -E "^($FOLD|PASS|FAIL|no fold)" | tr '\n' ' ')
  out "$name built in $(( $(date +%s) - t0 ))s ($flags); check: $chk"
  case "$chk" in *PASS*) names+=("$name") ;; *) out "$name not timed: the check failed" ; continue ;; esac
  for spec in $SWEEP_FOLD; do
    IFS=: read -r FM FC FR <<< "$spec"
    r=$(timeout 900 /root/var/$name/pearl_amd_foldtest "$FM" "$FC" 1 "$FR" 0 "$FOLD" 2>&1 | grep -E "^$FOLD" | sed 's/  */ /g')
    out "$name fold $FM x $FC: $r"
  done
done

hrun() {   # name [col batch]
  local r
  r=$(cd $A && PEARL_AMD_COL_BATCH=${2:-} timeout 600 node ../probes/hashrate.js /root/var/$1/pearl_core_hip.node "$HSECS" "$HWARM" 2>&1 \
      | grep -v nvidia-smi | grep -E '"th"|error' | tail -1)
  out "$1${2:+ cb=$2} hashrate: ${r:0:300}"
}
for ((i = 1; i <= ROUNDS; i++)); do
  for n in "${names[@]}"; do hrun "$n"; done
  if [ -n "${SWEEP_CB:-}" ] && [ ${#names[@]} -gt 0 ]; then
    for cb in $SWEEP_CB; do hrun "${SWEEP_CBV:-${names[0]}}" "$cb"; done
  fi
done
out "done"
