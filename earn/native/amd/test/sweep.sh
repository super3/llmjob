#!/bin/bash
# A tuning sweep of the AMD folds on one PC, so variants are compared on the same card. Run from
# earn/native/amd by amdtest.sh's `run` task (RUN_GZ_B64 holding "bash test/sweep.sh"), after `build`.
# When the variants are built it builds the default again, so a task after `run` (verify, duel, mine) runs the
# shipping code, not the last variant.
#
# For each variant in SWEEP it builds the core with those compiler flags (PEARL_AMD_CFLAGS), checks the fold
# bit-exact against the CPU (pearl_amd_foldtest at 1024 x 1024), and times the fold alone at the mainnet m. Then it
# runs hashrate.js on every variant, SWEEP_ROUNDS rounds in turn, and on the first variant once for each batch
# width in SWEEP_CB (PEARL_AMD_COL_BATCH, no rebuild).
#
# A variant whose flags set -DPEARL_FOLD_K=K (a k 4096 build, pearl_amd_fold.h) is checked and timed at that k:
# its fold test folds k K, its fold timing runs launches x 2048 / K launches (the same MACs as a k 2048 build),
# and hashrate.js runs it with PEARL_PROFILE_K=K. The core reports MACs a second at any k, so the rates compare.
#
# Environment:
#   SWEEP         "name=flags;name=flags", e.g. "tiled=;rowmajor=-DPEARL_AMD_TILED=0" (default: one build, "base=").
#   SWEEP_FOLD    m:cols:launches for the fold-only timing, several space-separated (default 131072:4096:10);
#                 "" skips it
#   SWEEP_FOLD_ROUNDS  N: time the fold alone N rounds, every variant in turn each round, once all are built,
#                 instead of each variant once right after its build. Each round ends with a line of every
#                 variant's TMAC/s and its change against the first variant. Score from round 2: the first
#                 carries a warm-up bias. Default 0, the old way.
#   SWEEP_HASH    secs:warm for each hashrate.js run (default 40:10)
#   SWEEP_ROUNDS  rounds of hashrate.js over the variants (default 2)
#   SWEEP_CB      batch widths to try on one variant, e.g. "64 128" (default none)
#   SWEEP_CBV     the variant SWEEP_CB runs on (default the first)
#   SWEEP_TIMEONLY  variants to time although their check finds wrong words, space-separated, e.g. "drop" for
#                 a -DPEARL_AMD_READOUT_DROP=1 build, whose transcripts are wrong on purpose (pearl_amd_fold.h).
#                 One whose check passes is not timed: its flags did not change the fold. Nor is one whose
#                 fold did not run to the end.
#   SWEEP_ROOT    where the build logs and each variant's copy go (default /root)
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
FOLD_ROUNDS=${SWEEP_FOLD_ROUNDS:-0}
ROOT=${SWEEP_ROOT:-/root}
case "$FOLD_ROUNDS" in '' | *[!0-9]*) out "SWEEP_FOLD_ROUNDS must be a whole number, not $FOLD_ROUNDS"; exit 2 ;; esac
out "start $ARCH fold $FOLD; variants: $SWEEP${SWEEP_TIMEONLY:+ (timing only: $SWEEP_TIMEONLY)}; fold timing ${SWEEP_FOLD:-none}$([ "$FOLD_ROUNDS" -gt 0 ] && echo " x $FOLD_ROUNDS rounds in turn"); hashrate ${HSECS}s+${HWARM}s x $ROUNDS"

# Each variant's k, from -DPEARL_FOLD_K=K in its flags; unset for a k 2048 build (pearl_config.h's default).
declare -A KOF=()
# Launches for variant $1 given $2 at k 2048: the same MACs at its own k, and at least one.
launches() {
  local k=${KOF[$1]:-} n
  [ -n "$k" ] || { echo "$2"; return; }
  n=$(( ${2:-1} * 2048 / k ))
  [ "$n" -ge 1 ] || n=1
  echo "$n"
}
# The fold alone, timed: the foldtest's line for variant $1 at m $2, cols $3, launches $4 (at k 2048), spaces squeezed.
fold_rate() {
  timeout 900 $ROOT/var/$1/pearl_amd_foldtest "$2" "$3" 1 "$(launches "$1" "$4")" 0 "$FOLD" 2>&1 | grep -E "^$FOLD" | sed 's/  */ /g'
}

# SWEEP_FOLD_ROUNDS: every variant in turn, each round, so a round compares them on the card as it is then.
fold_rounds() {
  local i spec FM FC FR n r t base line
  for ((i = 1; i <= FOLD_ROUNDS; i++)); do
    for spec in $SWEEP_FOLD; do
      IFS=: read -r FM FC FR <<< "$spec"
      base="" line=""
      for n in "${names[@]}"; do
        r=$(fold_rate "$n" "$FM" "$FC" "$FR")
        out "$n fold $FM x $FC, round $i of $FOLD_ROUNDS: $r"
        t=$(printf '%s\n' "$r" | sed -nE 's/.* ([0-9]+(\.[0-9]+)?) TMAC\/s$/\1/p')
        if [ "$n" = "${names[0]}" ]; then
          base=$t line="$n ${t:-?}"
        elif [ -n "$t" ] && [ -n "$base" ]; then
          line="$line, $n $t ($(awk -v a="$t" -v b="$base" 'BEGIN { printf "%+.2f%%", (a / b - 1) * 100 }'))"
        else
          line="$line, $n ${t:-?}"
        fi
      done
      [ -n "$line" ] && out "fold $FM x $FC, round $i of $FOLD_ROUNDS, TMAC/s: $line"
    done
  done
}

names=()
IFS=';' read -ra VARS <<< "$SWEEP"
for v in "${VARS[@]}"; do
  [ -n "$v" ] || continue
  name=${v%%=*}; flags=${v#*=}
  if [[ " $flags " =~ [[:space:]]-DPEARL_FOLD_K=([0-9]+)u?[[:space:]] ]]; then KOF[$name]=${BASH_REMATCH[1]}; fi
  t0=$(date +%s)
  if ! PEARL_AMD_CFLAGS="$flags" ./build-amd.sh gpu "$ARCH" > $ROOT/sweep-build-$name.log 2>&1; then
    out "$name build FAIL ($flags)"
    grep -E 'error|build-amd' $ROOT/sweep-build-$name.log | head -5 | while IFS= read -r l; do out "$name | ${l:0:300}"; done
    continue
  fi
  mkdir -p $ROOT/var/$name
  cp build/pearl_core_hip.node build/pearl_amd_foldtest $ROOT/var/$name/
  chk=$(timeout 600 $ROOT/var/$name/pearl_amd_foldtest 1024 1024 7 1 1 "$FOLD" 2>&1 | grep -E "^($FOLD|PASS|FAIL|no fold)" | tr '\n' ' ')
  out "$name built in $(( $(date +%s) - t0 ))s ($flags)${KOF[$name]:+, k ${KOF[$name]}}; check: $chk"
  case " ${SWEEP_TIMEONLY:-} " in *" $name "*) timeonly=1 ;; *) timeonly=0 ;; esac
  case "$timeonly:$chk" in
    0:*PASS*) names+=("$name") ;;
    1:*PASS*) out "$name not timed: it is timing-only, yet the check passed, so its flags did not change the fold"
              continue ;;
    1:*"words differ"*) out "$name check FAIL (expected: timing only)"; names+=("$name") ;;
    *) out "$name not timed: the check failed" ; continue ;;
  esac
  [ "$FOLD_ROUNDS" -gt 0 ] && continue   # timed in turn below
  for spec in $SWEEP_FOLD; do
    IFS=: read -r FM FC FR <<< "$spec"
    out "$name fold $FM x $FC: $(fold_rate "$name" "$FM" "$FC" "$FR")"
  done
done
fold_rounds

# Leave build/ holding the default, not the last variant.
if ./build-amd.sh gpu "$ARCH" > $ROOT/sweep-build-default.log 2>&1; then out "default rebuilt in build/"
else out "default rebuild FAIL: build/ holds the last variant; run the build task before verify, duel or mine"; fi

hrun() {   # name [col batch]
  local r k=${KOF[$1]:-}
  r=$(cd $A && PEARL_AMD_COL_BATCH=${2:-} timeout 600 env ${k:+PEARL_PROFILE_K=$k} node ../probes/hashrate.js \
      $ROOT/var/$1/pearl_core_hip.node "$HSECS" "$HWARM" 2>&1 | grep -v nvidia-smi | grep -E '"th"|error' | tail -1)
  out "$1${2:+ cb=$2}${k:+ k=$k} hashrate: ${r:0:300}"
}
for ((i = 1; i <= ROUNDS; i++)); do
  for n in "${names[@]}"; do hrun "$n"; done
  if [ -n "${SWEEP_CB:-}" ] && [ ${#names[@]} -gt 0 ]; then
    for cb in $SWEEP_CB; do hrun "${SWEEP_CBV:-${names[0]}}" "$cb"; done
  fi
done
out "done"
