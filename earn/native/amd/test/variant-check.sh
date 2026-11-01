#!/bin/bash
# What each build variant compiles to on this card's target, with nothing timed. For each: its row from build-amd.sh
# check, its SGPRs, and from its device assembly the matrix fold's LDS stores, global loads and LDS loads, the WMMAs
# between each barrier and the next wait on LDS, the LDS loads between the last WMMA and a barrier, and whether its
# code is the base's. Then which variants a timed run should keep. Run from earn/native/amd after the build task
# (build/gen must exist), by amdtest.sh's `run` task (RUN_GZ_B64), before test/sweep.sh. About 5 seconds a variant.
#
# Environment:
#   VARIANTS    "name=flags;name=flags", as sweep.sh's SWEEP (default "base="). The first is the base the others are
#               held to.
#   TWINS       "name=twin other~twin ...". With =, name is dropped unless it has twin's VGPRs (a stamp build and
#               its unstamped twin); with ~, the two are only compared, in a note. A twin comes before its variant.
#   ARCH        the target (default: the card's, from rocminfo). gfx11 checks the wmma11 fold, gfx12 the wmma12.
#   KEEP_TO     a file for the kept variants, one "name=flags" a line, in VARIANTS's order
#   CHECK_ROOT  where the assembly goes (default /root/variant-check)
#   CHECK_CC    the HIP compiler (default ROCm's, as build-amd.sh's)
#
# A variant is kept with no scratch and no spills, LDS 36864 bytes and 6 waves a SIMD (the release's on RDNA 3 and
# RDNA 4), the base's LDS stores and global loads (so the compiler kept all of the operand staging), and its twin's
# VGPRs. If the base's code can't be read, the counts aren't compared, and a line says so. On gfx12 a
# PEARL_AMD_LDS_AHEAD build is kept only with no LDS load between its last WMMA and a barrier: the release fence
# before the barrier waits for every LDS load, so a load there puts its wait back on the barrier.
#
# The WMMAs after a barrier are counted up to the first s_wait_dscnt (gfx12) or lgkmcnt wait (gfx11) after it; "16+"
# means 16 and no LDS wait before the next barrier or the end. The release waits for its fragments right after each
# slab barrier (0); PEARL_AMD_LDS_AHEAD's slab barrier is followed by the last k-step's 16 WMMAs. After the slab
# loop's first barrier the ops up to that wait are listed too (G global load, S LDS store, D LDS load, W WMMA), to
# show whether the LDS loads come among the WMMAs.
#
# Each result is one "[amdtest] HH:MM:SS check ..." line, written straight to the container's log too.
set -u
ROCM=${ROCM_PATH:-/opt/rocm}
CHECK_CC=${CHECK_CC:-$ROCM/lib/llvm/bin/clang++}
ARCH=${ARCH:-$(timeout 60 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1)}
OUTD=${CHECK_ROOT:-/root/variant-check}
say() {   # to the container's log now, and to the run task's output for its summary at the end
  local l="[amdtest] $(date -u +%T) check $*"
  echo "$l"
  { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true
}
case "$ARCH" in
  gfx11*) F=wmma11 ;;
  gfx12*) F=wmma12 ;;
  *) say "no RDNA 3 or RDNA 4 target here (${ARCH:-none}): nothing to check"; exit 2 ;;
esac
mkdir -p "$OUTD" || exit 2
declare -A FL=() V=() N=() H=() TWIN=() NOTE=()
names=()
IFS=';' read -ra VARS <<< "${VARIANTS:-base=}"
for v in "${VARS[@]}"; do
  [ -n "$v" ] || continue
  names+=("${v%%=*}")
  FL[${v%%=*}]=${v#*=}
done
[ ${#names[@]} -gt 0 ] || { say "VARIANTS names no variant"; exit 2; }
base=${names[0]}
for p in ${TWINS:-}; do
  case "$p" in
    *=*) TWIN[${p%%=*}]=${p#*=} ;;
    *~*) NOTE[${p%%~*}]=${p#*~} ;;
    *) say "TWINS takes name=twin or name~twin, not $p"; exit 2 ;;
  esac
done
[ -z "${KEEP_TO:-}" ] || : > "$KEEP_TO" || exit 2
say "start $ARCH fold $F; variants: ${VARIANTS:-base=}${TWINS:+; twins: $TWINS}"

# The fold's instructions in variant $1's device assembly, compiled with flags $2 as build-amd.sh check compiles it,
# into $OUTD/$1.fold: comments and directives dropped, block labels numbered alike. Empty if it can't be read.
fold_code() {
  rm -f "$OUTD/$1.s" "$OUTD/$1.fold"
  "$CHECK_CC" -x hip --offload-arch="$ARCH" --rocm-path="$ROCM" -std=c++17 -O3 $2 -Icompat/gpu -Isrc -I../src -Ibuild/gen \
    --cuda-device-only -S src/pearl_host_hip.cpp -o "$OUTD/$1.s" > "$OUTD/$1.log" 2>&1
  awk -v f="pearl_amd_fold_$F:" '
    $1 == f { on = 1; next }
    on && /^\.Lfunc_end/ { exit }
    on {
      sub(/;.*/, ""); sub(/[ \t]+$/, "")
      if (NF == 0 || ($1 ~ /^\./ && $1 !~ /:$/)) next
      gsub(/\.LBB[0-9]+_/, ".LBB_"); print
    }' "$OUTD/$1.s" > "$OUTD/$1.fold" 2> /dev/null
}
# From $OUTD/$1.fold: "LDS stores, global loads, LDS loads|WMMAs after each barrier|the second barrier's ops|late LDS
# loads". The ops after the second barrier (the slab loop's first), up to the first LDS wait: G a global load, S an
# LDS store, D an LDS load, W a WMMA. The late LDS loads: those after the last WMMA before each barrier, summed.
counts() {
  awk '
    $1 ~ /^ds_(store|write)/ { st++; op = "S" }
    $1 ~ /^(global|buffer)_load/ { gl++; op = "G" }
    $1 ~ /^ds_(load|read)/ { ld++; op = "D"; late++ }
    $1 ~ /^v_wmma/ { late = 0 }
    $1 == "s_barrier" || $1 == "s_barrier_signal" { lt += late; late = 0 }
    $1 == "s_barrier" || $1 == "s_barrier_wait" { if (inb) cov = cov " " w "+"; inb = 1; w = 0; nb++; op = ""; next }
    inb && $1 ~ /^v_wmma/ { w++; op = "W" }
    inb && nb == 2 && op != "" && length(seq) < 60 { seq = seq op }
    { op = "" }
    inb && ($1 ~ /^s_wait_(.*_)?dscnt/ || ($1 ~ /^s_waitcnt/ && /lgkmcnt/)) { cov = cov " " w; inb = 0 }
    END { if (inb) cov = cov " " w "+"; printf "%d %d %d|%s|%s|%d\n", st, gl, ld, substr(cov, 2), seq, lt }' "$OUTD/$1.fold"
}
# The fold's SGPRs, from the remarks build-amd.sh check just wrote.
sgprs() {
  awk -v f="pearl_amd_fold_$F" '
    /remark: Function Name: / { n = $(NF-1) }
    n == f && /remark:     SGPRs: / { print $(NF-1); exit }' "build/check/$ARCH.txt" 2> /dev/null
}

kept=()
for n in "${names[@]}"; do
  flags=${FL[$n]}
  row=$(PEARL_AMD_CFLAGS="$flags" ./build-amd.sh check "$ARCH" 2>&1 | grep -E "pearl_amd_fold_$F " | head -n 1)
  sg=$(sgprs)
  fold_code "$n" "$flags"
  same='' cov='' seq='' late=''
  if [ -s "$OUTD/$n.fold" ]; then
    c=$(counts "$n")
    N[$n]=${c%%|*}
    c=${c#*|}
    cov=${c%%|*}
    c=${c#*|}
    seq=${c%%|*}
    late=${c#*|}
    H[$n]=$(md5sum < "$OUTD/$n.fold" | cut -c1-32)
    if [ "$n" != "$base" ] && [ -n "${H[$base]:-}" ]; then
      if [ "${H[$n]}" = "${H[$base]}" ]; then same="; the same code as $base"; else same="; code differs from $base"; fi
    fi
  fi
  say "$n (${flags:-release}): ${row:-no row}; SGPRs ${sg:-?}; fold code ${N[$n]:-not read}" \
    "(LDS stores, global loads, LDS loads); WMMAs after each barrier before an LDS wait: ${cov:-?}" \
    "(after the loop's first: ${seq:-none}); LDS loads between the last WMMA and a barrier: ${late:-?}$same"
  if [ "$n" = "$base" ] && [ -z "${N[$base]:-}" ]; then
    say "$base: the fold's code couldn't be read, so the variants are kept on their row alone"
  fi
  set -- $row   # target kernel VGPRs AGPRs scratch spills LDS waves wrap
  V[$n]=${3:-}
  why=''
  t=${TWIN[$n]:-}
  if ! { [ "${5:-x}" = 0 ] && [ "${6:-x}" = 0 ] && [ "${7:-x}" = 36864 ] && [ "${8:-x}" = 6 ]; }; then
    why='not LDS 36864 and 6 waves with no scratch and no spills'
  elif [ -n "$t" ] && [ "${V[$n]}" != "${V[$t]:-}" ]; then
    why="${V[$n]} VGPRs, not $t's ${V[$t]:-?}"
  elif [ -n "${N[$base]:-}" ] && [ "${N[$n]:-}" = '' ]; then
    why="its code couldn't be read, so its LDS stores and global loads can't be held to $base's"
  elif [ -n "${N[$base]:-}" ] && [ "${N[$n]% *}" != "${N[$base]% *}" ]; then
    why="LDS stores and global loads ${N[$n]% *}, not $base's ${N[$base]% *}"
  elif [ "$F" = wmma12 ] && [[ $flags == *-DPEARL_AMD_LDS_AHEAD=[12]* ]] && [ "${late:-?}" != 0 ]; then
    why="${late:-?} LDS loads between the last WMMA and a barrier, whose release fence waits for them"
  fi
  o=${NOTE[$n]:-}
  if [ -n "$o" ] && [ -n "${V[$n]}" ] && [ "${V[$n]}" != "${V[$o]:-}" ]; then
    say "$n has ${V[$n]} VGPRs, $o ${V[$o]:-?}: comparing them compares two schedules"
  fi
  if [ -z "$why" ]; then
    kept+=("$n")
    [ -z "${KEEP_TO:-}" ] || printf '%s=%s\n' "$n" "$flags" >> "$KEEP_TO"
  else
    say "$n dropped: $why"
  fi
done
say "kept: ${kept[*]:-none}"
