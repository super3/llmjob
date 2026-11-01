#!/bin/bash
# Item 14, the start skew (PEARL_AMD_SKEW, pearl_amd_fold.h), raced against the release on one PC: an RDNA 3 card
# (gfx1100, gfx1101 or gfx1151, k 2048) or an RDNA 4 card (gfx1200 or gfx1201, k 4096), each as it mines. Run from
# earn/native/amd by amdtest.sh's `run` task (RUN_GZ_B64 holding "SKEW_CLASS=rdna3 bash test/skew-race.sh", or
# rdna4). It builds what it races, so no build task is needed first. About 35 minutes.
#
# One skew build races every wait. The host reads the waits at run time (PEARL_AMD_SKEW_WAIT=a:b:c: places 0, 1 and
# 2, in percent of a slab), so each rung below is the same build in another environment.
# 1. The gate: test/variant-check.sh on the card's target. The skew build is raced only with the release's VGPRs or
#    fewer, 6 waves, no spills or scratch, LDS 36864, and the release's LDS stores and global loads in the fold.
# 2. test/sweep.sh builds both and checks each fold bit-exact against the CPU at 1024 x 1024. That launch has too few
#    blocks to fill every WGP, so most likely no block waits in it.
# 3. The probe: the skew build's fold test launches a launch's first blocks (three a WGP), which take places as the
#    fold's do and hold their WGP meanwhile. Every counter must get three blocks, places 0, 1 and 2. If not, the
#    counters don't tell the WGPs apart (or HIP's count isn't WGPs), and nothing is raced.
# 4. The skew build checked bit-exact at 2048 x 4096 with SKEW 1's waits: every WGP's first three blocks wait there.
# 5. r1 and r2: base's fold alone at one generation of blocks (m 128 x WGPs) with one, two and three blocks a WGP
#    (cols 256, 512 and 768), 3 runs each. r1 is how much faster a block runs with the WGP to itself than with three,
#    r2 with one other. A wait is time, not work: while places 1 and 2 wait, place 0 runs r1 and then r2 times as
#    fast. So a work-aimed rung waits L / r1 at place 1 and L / r2 more at place 2, for leads of L and 2L slabs.
# 6. The rungs, places 0:1:2 in percent of a slab:
#      t8 0:8:17, t17 0:17:33, t33 0:33:67 (SKEW 1's)   the same in time: leads of 1/12, 1/6 and 1/3 if r1 = r2 = 1
#      w17, w33                                        aimed at leads in work of 1/6 and 1/3, from r1 and r2
#      ctl 67:67:67                                    the control: every first block waits t33's longest, the
#                                                       same delay with no skew
#    The fold alone on the miner's operands, base and the rungs in turn, 4 rounds at each size:
#      one generation (m 128 x WGPs, cols 768), where every block takes a place: what de-phasing buys with nothing
#        to decay. Read a rung against ctl too: both launches end on a block that waited.
#      eight generations (m x 8, cols 768)
#      the mined launch (131072 x 2048 on gfx1100 and gfx1101, at their batch 128; 131072 x 4096 elsewhere), for
#        the SKEW_FOLD rungs that gained most at eight generations.
#    A gain at one generation and none at the mined launch is decay. None at one generation: de-phasing doesn't help.
# 7. hashrate.js, base and the SKEW_RACE rungs that gained most at the mined launch, 4 rounds of 60 s after 20 s.
# 8. A verdict for each: +0.3% or more in every scored round? Then verify-hits.js on the best rung that says yes (the
#    best mean if none does): the first 400 hits recomputed as a pool would. Its shares go to
#    $RUN_ROOT/shares/verify-<rung>.jsonl, for a refverify task after run. It exits 1 unless that is 400 of 400.
# Rounds 2-4 are scored everywhere: the first carries a warm-up bias. A rung ships on +0.3% or more in every scored
# round on two PCs of a class, bit-exact, with verify-hits 400 of 400.
#
# Environment:
#   SKEW_CLASS   rdna3 or rdna4: refuse a card of the other class (default: either)
#   SKEW_FOLD    rungs timed at the mined launch (default 3)
#   SKEW_RACE    rungs raced in hashrate.js (default 2)
#   SKEW_EXTRA   more rungs, "name=a:b:c ...", each wait at most 400, timed at every size and always raced in
#                hashrate.js, on top of the SKEW_RACE best: say, the waits a rung won with on the first PC of a class,
#                so a second PC races exactly those (w17 and w33 come out of each PC's own r1 and r2)
#   SKEW_VERIFY  seconds of verify-hits.js (default 150: an XTX found 400 hits within 120 s, a 7800 XT 387)
#   RUN_ROOT     where the builds and logs go (default /root)
#   ARCH         the target (default: the card's, from rocminfo)
#
# Each result is one "[amdtest] HH:MM:SS skew ..." line, written straight to the container's log too.
set -u
R=${RUN_ROOT:-/root}
say() {   # to the container's log now, and to the run task's output for its summary at the end
  local l="[amdtest] $(date -u +%T) skew $*"
  echo "$l"
  { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true
}
case "${SKEW_CLASS:-}" in '' | rdna3 | rdna4) ;; *) say "SKEW_CLASS is rdna3 or rdna4, not $SKEW_CLASS"; exit 2 ;; esac
TOPF=${SKEW_FOLD:-3}
RACE=${SKEW_RACE:-2}
VSECS=${SKEW_VERIFY:-150}
for v in "$TOPF" "$RACE" "$VSECS"; do
  case "$v" in '' | *[!0-9]* | 0*) say "SKEW_FOLD, SKEW_RACE and SKEW_VERIFY are whole numbers above 0"; exit 2 ;; esac
done
for x in ${SKEW_EXTRA:-}; do
  if ! [[ "$x" =~ ^[a-z][a-z0-9]*=([0-9]{1,3}):([0-9]{1,3}):([0-9]{1,3})$ ]] \
    || [ "${x%%=*}" = ctl ] || [ "${x%%=*}" = base ] \
    || (( 10#${BASH_REMATCH[1]} > 400 || 10#${BASH_REMATCH[2]} > 400 || 10#${BASH_REMATCH[3]} > 400 )); then
    say "SKEW_EXTRA takes name=a:b:c, each wait at most 400 and the name not ctl or base, not $x"
    exit 2
  fi
done
A=${ARCH:-$(timeout 60 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -n 1)}
case "$A" in
  gfx1100* | gfx1101* | gfx1151*) C=rdna3 F=wmma11 ;;
  gfx1200* | gfx1201*) C=rdna4 F=wmma12 ;;
  gfx11* | gfx12*) say "$A has no skew: its WGP holds two blocks of the fold, not three (pearl_amd_fold.h)"; exit 2 ;;
  *) say "no RDNA 3 or RDNA 4 card here (${A:-none}): the skew is in their folds only"; exit 2 ;;
esac
if [ -n "${SKEW_CLASS:-}" ] && [ "$SKEW_CLASS" != "$C" ]; then
  say "this card is $A, not ${SKEW_CLASS}: run the other class's script"
  exit 2
fi
# The launch the card mines, as m:cols:launches: 2048 columns on gfx1100 and gfx1101 (their batch 128, amd_col_batch)
# and 4096 elsewhere; launches for the MACs of 600 at 131072 x 4096 x 2048, so half as many at RDNA 4's k 4096.
case "$A" in
  gfx1100* | gfx1101*) MINED=131072:2048:1200 ;;
  gfx11*) MINED=131072:4096:600 ;;
  *) MINED=131072:4096:300 ;;
esac
# Launches a run at one and eight generations: about half a second of fold each.
case "$C" in rdna3) N1=3000 N8=400 ;; *) N1=4000 N8=500 ;; esac
say "start $A fold $F; the mined launch $MINED; up to $TOPF rungs at the mined launch, $RACE in hashrate.js;" \
  "verify-hits ${VSECS}s"

# 1. The gate.
KEEP=$R/skew-keep.txt
VARIANTS='base=;skew=-DPEARL_AMD_SKEW=1' TWINS='skew<base' ARCH=$A KEEP_TO=$KEEP CHECK_ROOT=$R/variant-check \
  bash test/variant-check.sh
declare -A FL=()
if [ -f "$KEEP" ]; then while IFS= read -r l; do [ -n "$l" ] && FL[${l%%=*}]=${l#*=}; done < "$KEEP"; fi
[ -n "${FL[base]+x}" ] || { say "base was dropped or not built: nothing to race against"; exit 1; }
[ -n "${FL[skew]+x}" ] || { say "the skew build was dropped by the gate: nothing to race"; exit 0; }

# 2. Both built and checked.
ARCH=$A SWEEP_ROOT=$R SWEEP="base=;skew=${FL[skew]}" SWEEP_FOLD='' SWEEP_ROUNDS=0 bash test/sweep.sh \
  | tee "$R/skew-sweep.log"
for n in base skew; do
  grep -qE " sweep $n built in .*PASS" "$R/skew-sweep.log" \
    || { say "$n's fold check at 1024 x 1024 didn't pass: nothing to race"; exit 1; }
done
FT=$R/var/skew/pearl_amd_foldtest

# 3. The probe, with a rung's waits from the environment, to see the fold test take them.
p=$(PEARL_AMD_SKEW_WAIT=0:8:17 timeout 300 "$FT" 128 256 1 1 0 "$F" 2>&1)
printf '%s\n' "$p" | grep -E '^(device|skew)' | while IFS= read -r l; do say "probe: ${l:0:300}"; done
first=$(printf '%s\n' "$p" | sed -nE 's/^skew: .*; first ([0-9]+) blocks;.*/\1/p' | head -n 1)
case "$p" in
  *"waits 0:8:17% "*"from PEARL_AMD_SKEW_WAIT"*) ;;
  *) say "the fold test didn't take PEARL_AMD_SKEW_WAIT=0:8:17: nothing to race"; exit 1 ;;
esac
case "$p" in
  *"skew probe: "*": as designed"*) ;;
  *) say "the probe's places aren't as designed: nothing to race"; exit 1 ;;
esac
[ -n "$first" ] && [ "$first" -ge 3 ] || { say "the probe gave no block count: nothing to race"; exit 1; }
WGPS=$((first / 3))

# 4. The big check, with the build's waits (SKEW 1's).
r=$(PEARL_AMD_SKEW_WAIT='' timeout 900 "$FT" 2048 4096 7 1 1 "$F" 2>&1 | grep -E "^($F|PASS|FAIL|no fold|skew)" \
  | tr '\n' ' ')
say "skew check at 2048 x 4096, the build's waits: ${r:0:500}"
case "$r" in *PASS*) ;; *) say "the skew build's check at 2048 x 4096 didn't pass: nothing to race"; exit 1 ;; esac

# The fold alone on the miner's operands: build $1 with waits $2 at m $3, cols $4, $5 launches. Its TMAC/s, or nothing.
fold() {
  PEARL_AMD_SKEW_WAIT=$2 timeout 900 "$R/var/$1/pearl_amd_foldtest" "$3" "$4" 1 "$5" 0 "$F" 0 miner 2>&1 \
    | sed -nE "s/^$F +timed only: .* ([0-9]+(\\.[0-9]+)?) TMAC\\/s\$/\\1/p" | head -n 1
}
mean() { printf '%s\n' "$@" | awk 'NF { s += $1; n++ } END { if (n) printf "%+.2f", s / n }'; }
avg() { printf '%s\n' "$@" | awk 'NF { s += $1; n++ } END { if (n) printf "%.3f", s / n }'; }

# 5. r1 and r2.
M1=$((128 * WGPS))
declare -A RT=()
for ((i = 1; i <= 3; i++)); do
  for c in 256 512 768; do RT[$c]="${RT[$c]:-} $(fold base '' "$M1" "$c" "$N1")"; done
done
t1=$(avg ${RT[256]}) t2=$(avg ${RT[512]}) t3=$(avg ${RT[768]})
rr=$(awk -v a="$t1" -v b="$t2" -v c="$t3" 'BEGIN {
  if (a == "" || b == "" || c == "" || c <= 0) exit
  r1 = 3 * a / c; r2 = 1.5 * b / c
  if (r1 < 1) r1 = 1; if (r1 > 3) r1 = 3; if (r2 < 1) r2 = 1; if (r2 > 1.5) r2 = 1.5
  printf "%.2f %.2f", r1, r2 }')

# 6. The rungs, and the fold alone.
declare -A W=([t8]=0:8:17 [t17]=0:17:33 [t33]=0:33:67)
RUNGS='t8 t17 t33'
if [ -n "$rr" ]; then
  read -r r1 r2 <<< "$rr"
  say "base at one generation (m $M1), TMAC/s: one block a WGP ${t1}, two ${t2}, three ${t3}: r1 $r1, r2 $r2" \
    "(each held to 1-3 and 1-1.5)"
  for L in 17 33; do
    W[w$L]=$(awk -v L="$L" -v a="$r1" -v b="$r2" \
      'BEGIN { l = (L == 17 ? 100 / 6 : 100 / 3); x = l / a; printf "0:%d:%d", x + 0.5, x + l / b + 0.5 }')
    RUNGS="$RUNGS w$L"
  done
else
  say "base at one generation (m $M1): not every rate was read (${t1:-?}, ${t2:-?}, ${t3:-?}), so no work-aimed rungs"
fi
OWN=$RUNGS EXTRA=''   # the ladder's own rungs, and SKEW_EXTRA's
for x in ${SKEW_EXTRA:-}; do
  case " $OWN $EXTRA " in
    *" ${x%%=*} "*) say "SKEW_EXTRA: ${x%%=*} is a rung already: not raced twice"; continue ;;
  esac
  W[${x%%=*}]=${x#*=}
  EXTRA="$EXTRA ${x%%=*}"
done
EXTRA=${EXTRA# }
W[ctl]=67:67:67
RUNGS="$OWN${EXTRA:+ $EXTRA} ctl"
say "rungs, places 0:1:2 in percent of a slab: $(for g in $RUNGS; do printf '%s %s, ' "$g" "${W[$g]}"; done \
  | sed 's/, $//')"

declare -A G=()   # G[size:rung]: its scored changes against base, rounds 2-4, in %
# Four rounds of the fold alone at size $1 (m $2, cols $3, $4 launches): base, then each rung after it in turn.
rounds() {
  local size=$1 m=$2 c=$3 n=$4 i b t d line g; shift 4
  local -a s
  say "fold $size: m $m, cols $c, $n launches a run"
  for g in "$@"; do G[$size:$g]=''; done
  for ((i = 1; i <= 4; i++)); do
    b=$(fold base '' "$m" "$c" "$n")
    line="base ${b:-?}"
    for g in "$@"; do
      t=$(fold skew "${W[$g]}" "$m" "$c" "$n")
      if [ -n "$t" ] && [ -n "$b" ]; then
        d=$(awk -v a="$t" -v b="$b" 'BEGIN { printf "%+.2f", (a / b - 1) * 100 }')
        line="$line, $g $t ($d%)"
        [ "$i" -lt 2 ] || G[$size:$g]="${G[$size:$g]}${G[$size:$g]:+ }$d"
      else
        line="$line, $g ${t:-?}"
      fi
    done
    say "fold $size, round $i of 4, TMAC/s: $line"
  done
  for g in "$@"; do
    read -ra s <<< "${G[$size:$g]}"
    if [ ${#s[@]} -eq 3 ]; then
      say "fold $size, $g (${W[$g]}): rounds 2-4 ${s[0]}, ${s[1]}, ${s[2]}%; mean $(mean "${s[@]}")%"
    else
      say "fold $size, $g (${W[$g]}): ${#s[@]} of 3 scored rounds read"
    fi
  done
}
# The $2 rungs of $3... with the largest mean gain at size $1, largest first; ties keep the order given. Only rungs
# with all three scored rounds.
top() {
  local size=$1 k=$2 g; shift 2
  for g in "$@"; do printf '%s %s\n' "$g" "${G[$size:$g]:-}"; done | awk -v k="$k" '
    NF == 4 { n++; v[n] = $1; m[n] = ($2 + $3 + $4) / 3 }
    END {
      for (j = 1; j <= n && j <= k; j++) {
        b = 0
        for (i = 1; i <= n; i++) if (!u[i] && (!b || m[i] > m[b])) b = i
        u[b] = 1; print v[b]
      }
    }'
}
rounds one-gen "$M1" 768 "$N1" $RUNGS
# At one generation each launch ends on a block that waited, so a rung is read against ctl: the same delay, no skew.
read -ra s <<< "${G[one-gen:ctl]}"
if [ ${#s[@]} -eq 3 ]; then
  c=$(mean "${s[@]}") line=''
  for g in ${RUNGS% ctl}; do
    read -ra s <<< "${G[one-gen:$g]}"
    [ ${#s[@]} -eq 3 ] && line="$line, $g $(awk -v a="$(mean "${s[@]}")" -v b="$c" 'BEGIN { printf "%+.2f", a - b }')"
  done
  say "fold one-gen against ctl, the mean change less ctl's (${c}%), in points: ${line#, }"
fi
rounds eight-gen "$((M1 * 8))" 768 "$N8" $RUNGS
own=$(top eight-gen "$TOPF" $OWN)
pick=$(echo $own $EXTRA)
[ -n "$pick" ] || { say "no rung has three scored rounds at eight generations: nothing to race"; exit 1; }
IFS=: read -r MM MC MN <<< "$MINED"
rounds mined "$MM" "$MC" "$MN" $pick
race=$(echo $(top mined "$RACE" $own) $EXTRA)
[ -n "$race" ] || { say "no rung has three scored rounds at the mined launch: nothing to race"; exit 1; }

# 7. hashrate.js in turn: base, then each raced rung on the skew build.
hr() {   # build waits: hashrate.js's last line
  PEARL_AMD_SKEW_WAIT=$2 timeout 600 node ../probes/hashrate.js "$R/var/$1/pearl_core_hip.node" 60 20 2>&1 \
    | grep -v nvidia-smi | grep -E '"th"|error' | tail -n 1
}
declare -A SC=() SUM=() CNT=()
for ((i = 1; i <= 4; i++)); do
  j=$(hr base '')
  say "base hashrate, round $i of 4: ${j:0:300}"
  b=$(printf '%s\n' "$j" | sed -nE 's/.*"th":([0-9.]+).*/\1/p')
  line="base ${b:-?}"
  for g in $race; do
    j=$(hr skew "${W[$g]}")
    say "$g hashrate, round $i of 4: ${j:0:300}"
    t=$(printf '%s\n' "$j" | sed -nE 's/.*"th":([0-9.]+).*/\1/p')
    if [ -n "$t" ] && [ -n "$b" ]; then
      d=$(awk -v a="$t" -v b="$b" 'BEGIN { printf "%+.2f", (a / b - 1) * 100 }')
      line="$line, $g $t (${d}%)"
      if [ "$i" -ge 2 ]; then
        SC[$g]="${SC[$g]:-}${SC[$g]:+, }$d"
        SUM[$g]=$(awk -v s="${SUM[$g]:-0}" -v d="$d" 'BEGIN { print s + d }')
        CNT[$g]=$(( ${CNT[$g]:-0} + 1 ))
      fi
    else
      line="$line, $g ${t:-?}"
    fi
  done
  say "hashrate round $i of 4, TH/s: $line"
done

# 8. The verdicts, and verify-hits on the best rung that passes, or the best mean while none does.
cand=''   # "passes mean rung", a line each
for g in $race; do
  c=${CNT[$g]:-0}
  if [ "$c" -lt 3 ]; then say "$g verdict: $c of 3 scored rounds read; no verdict"; continue; fi
  m=$(awk -v s="${SUM[$g]}" -v c="$c" 'BEGIN { printf "%+.2f", s / c }')
  low=$(printf '%s\n' "${SC[$g]}" | tr ',' '\n' | awk '$1 + 0 < 0.3 { n++ } END { print n + 0 }')
  if [ "$low" -eq 0 ]; then v='yes'; else v="no ($low under)"; fi
  say "$g verdict (${W[$g]}): hashrate rounds 2-4 ${SC[$g]}%; mean ${m}%; +0.3% or more in every scored round: $v"
  cand="$cand$([ "$low" -eq 0 ] && echo 1 || echo 0) $m $g"$'\n'
done
best=$(printf '%s' "$cand" | awk '
  NF == 3 && (!n || $1 > p || ($1 == p && $2 + 0 > m + 0)) { n = 1; p = $1; m = $2; b = $3 }
  END { print b }')
[ -n "$best" ] || { say "no raced rung has three scored rounds: no verify-hits"; exit 1; }
mkdir -p "$R/shares"
dump=$R/shares/verify-$best.jsonl
rm -f "$dump"   # verify-hits appends: shares from an earlier run would be checked as this one's
r=$(PEARL_AMD_SKEW_WAIT=${W[$best]} PEARL_SHARE_DUMP=$dump timeout 3600 node ../probes/verify-hits.js \
  "$R/var/skew/pearl_core_hip.node" "$VSECS" 2>&1 | tail -n 1)
say "$best verify-hits (${W[$best]}): ${r:0:400}"
ok=$(printf '%s\n' "$r" | sed -nE 's/.*"verified":([0-9]+).*/\1/p')
bad=$(printf '%s\n' "$r" | sed -nE 's/.*"failed":([0-9]+).*/\1/p')
if [ "${ok:-0}" -ge 400 ] && [ "${bad:-1}" -eq 0 ]; then
  say "$best: verify-hits ${ok} of ${ok}, 0 failed: the rule's 400 of 400 holds"
  say "done"
  exit 0
fi
say "$best: verify-hits ${ok:-?} verified, ${bad:-?} failed: not the rule's 400 of 400"
say "done"
exit 1
