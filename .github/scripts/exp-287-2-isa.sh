#!/bin/bash
# Temporary, for issue #287 item 2 (exp/287-2 only; removed before merge). In the release's ROCm 7.2.0 image: the
# RDNA 3 fold (wmma11) with and without buffer loads (PEARL_AMD_BUFLD) and PEARL_AMD_TID, against the release this
# branch was cut from. For each: build-amd.sh check's rows on every gfx11 target, and on gfx1100 and gfx1101 the
# fold's device assembly, its md5 and what its slab loop issues (exp-287-2-isa.js). Also that the default build is
# the release's code on gfx11 and gfx12, and that -DPEARL_AMD_BUFLD=1 leaves CDNA and RDNA 2 code alone.
# Run from the repo root. Writes build/isa/ for the artifact.
set -u
ROOT=$(pwd)
OUT=$ROOT/build/isa
CC=/opt/rocm/lib/llvm/bin/clang++
mkdir -p "$OUT"
# The release this branch was cut from, checked out into rel/ by the workflow (the image has no git).
[ -f "$ROOT/rel/earn/native/amd/build-amd.sh" ] || { echo "no release tree in rel/"; exit 1; }
rm -rf /tmp/rel && cp -r "$ROOT/rel" /tmp/rel

GFX11="gfx1100 gfx1101 gfx1102 gfx1150 gfx1151"
fold_of() { case "$1" in gfx11*) echo wmma11 ;; gfx12*) echo wmma12 ;; gfx90*) echo mfma8 ;; gfx103*) echo dot4 ;; esac; }

# fold_code TREE NAME ARCH FLAGS: the fold's code into $OUT/NAME-ARCH.fold (as test/variant-check.sh does).
fold_code() {
  local tree=$1 name=$2 a=$3 flags=$4 f
  f=$(fold_of "$a")
  (cd "$tree/earn/native/amd" && "$CC" -x hip --offload-arch="$a" --rocm-path=/opt/rocm -std=c++17 -O3 $flags \
    -Icompat/gpu -Isrc -I../src -Ibuild/gen --cuda-device-only -S src/pearl_host_hip.cpp -o "$OUT/$name-$a.s") \
    > "$OUT/$name-$a.log" 2>&1 || { echo "$name $a: compile failed"; cat "$OUT/$name-$a.log" | head -20; return 1; }
  awk -v f="pearl_amd_fold_$f:" '
    $1 == f { on = 1; next }
    on && /^\.Lfunc_end/ { exit }
    on {
      sub(/;.*/, ""); sub(/[ \t]+$/, "")
      if (NF == 0 || ($1 ~ /^\./ && $1 !~ /:$/)) next
      gsub(/\.LBB[0-9]+_/, ".LBB_"); print
    }' "$OUT/$name-$a.s" > "$OUT/$name-$a.fold"
  echo "$name $a $(fold_of "$a") md5 $(md5sum < "$OUT/$name-$a.fold" | cut -c1-12) lines $(wc -l < "$OUT/$name-$a.fold")"
}

rows() {   # TREE NAME FLAGS: build-amd.sh check's fold rows on the gfx11 targets
  local tree=$1 name=$2 flags=$3
  (cd "$tree/earn/native/amd" && PEARL_AMD_CFLAGS="$flags" ./build-amd.sh check $GFX11 2>&1) \
    | grep -E 'pearl_amd_fold_wmma11|check:' | sed "s/^/$name: /"
}

echo "== check rows (target kernel VGPRs AGPRs scratch spills LDS waves wrap)"
rows /tmp/rel release ""
VARS=("base|" "bufld|-DPEARL_AMD_BUFLD=1" "tid|-DPEARL_AMD_TID=1" "bufldtid|-DPEARL_AMD_BUFLD=1 -DPEARL_AMD_TID=1")
for v in "${VARS[@]}"; do rows "$ROOT" "${v%%|*}" "${v#*|}"; done

echo "== the default build against the release (md5 of the fold's code)"
for a in gfx1100 gfx1101 gfx1201 gfx90a gfx1030; do fold_code /tmp/rel release "$a" ""; fold_code "$ROOT" base "$a" ""; done
echo "== -DPEARL_AMD_BUFLD=1 on CDNA and RDNA 2 (must match base)"
for a in gfx90a gfx1030; do fold_code "$ROOT" bufld "$a" "-DPEARL_AMD_BUFLD=1"; done

echo "== RDNA 3 variants"
for a in gfx1100 gfx1101; do
  for v in "${VARS[@]:1}"; do fold_code "$ROOT" "${v%%|*}" "$a" "${v#*|}"; done
done
for a in gfx1100 gfx1101; do
  for n in release base bufld tid bufldtid; do
    [ -s "$OUT/$n-$a.fold" ] && node "$ROOT/.github/scripts/exp-287-2-isa.js" "$OUT/$n-$a.fold" "$n $a"
  done
done
echo "== RDNA 4 release fold for reference (BUFLD and KPAIR on)"
node "$ROOT/.github/scripts/exp-287-2-isa.js" "$OUT/base-gfx1201.fold" "base gfx1201"
echo "== diff of the slab loop, base against bufldtid on gfx1100 (first 200 lines)"
diff "$OUT/base-gfx1100.fold" "$OUT/bufldtid-gfx1100.fold" | head -n 200
echo "== the box run's twin (-DPEARL_AMD_TWIN=1, a macro nothing reads) must be the base's code"
for a in gfx1100 gfx1101; do fold_code "$ROOT" twin "$a" "-DPEARL_AMD_TWIN=1"; done
echo "== the peak probe (test/pearl_amd_peak.cpp) for gfx11, which no box has built yet"
for a in gfx1100 gfx1101; do
  (cd earn/native/amd && "$CC" -x hip --offload-arch="$a" --rocm-path=/opt/rocm -std=c++17 -O3 test/pearl_amd_peak.cpp \
    -o "$OUT/peak-$a" -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib) > "$OUT/peak-$a.log" 2>&1 \
    && echo "peak $a: built" || { echo "peak $a: build FAILED"; head -20 "$OUT/peak-$a.log"; }
done
