#!/bin/bash
# Box script for the rotation fold on an RTX 30 or RTX 40 card (sm_86 or sm_89), branch goal/nv-k25: on one Salad
# PC, the fold's cubins built from rfold.py keys, checked with verify-hits.js and raced with hashrate.js legs
# taking turns, then the other miners and ours through bench.sh, with the card read beside every leg. nv30box.sh
# for either architecture, with nv50box.sh's "smiq" lines. A measuring tool: nothing here is shipped.
#
# The container command fetches this file at BASE_SHA and runs it (an amdtest.py nvtest box whose boot fetches
# nv34box.sh). Every line is "[nvtest] HH:MM:SS [nv34] ..." (bench.sh's "[nvtest] [bench] HH:MM:SS ...").
#
# Environment:
#   BASE_SHA      commit of super3/llmjob to fetch (required, 40 characters)
#   STAGES        any of "product bench", in that order (default both)
#   ROT_VARIANTS  the fold's cubins, "name:key=v,key=v;..." (pearl_rot_<name>.cubin), rfold.py over pearl_rot.cu
#                 compiled for this card (with ROT_SR=1 / ROT_BX=1 when the keys have sr=1 / bx=1, as
#                 build_rot.py does). "ship" is always built: the file a release ships for this card
#                 (build_rot.py --arch sm_86 or sm_89), its hash printed
#   VERIFY_LIST   cubins verify-hits.js checks (default "ship"), VERIFY_SECS (150; 0 skips), VERIFY_BITS (233)
#   HR_LEGS       hashrate.js legs: "rel" (no fold), a cubin name, or "auto" (no file named: the shipped files
#                 beside the core, as a release has them, so the host's own pick of loop runs; with PEARL_ROT_LOG),
#                 optionally "+K=V" core settings after it (default "ship ship"); HR_SECS (120). A leg's lines about
#                 the loop rule are printed after it ("rule ...")
#   SMIQ          every SMIQ seconds a "smiq" line: power, voltage, clocks and the clock's active limiters
#   ORDER, VPOOL, SECS, WARM, VER   passed to bench.sh (defaults ours,ozm,peak,srb; 210; 180; 60; v0.7.2). With
#                 the product stage, "ours" is the release's CLI with the core built here (OURS_CORE) and the cubin
#                 BENCH_CUBIN (default ship) through PEARL_ROT_CUBIN. (OURS_SRC=1 failed here on 15 November: the
#                 CLI from source said pearl_core.node was not found.)
set -u
NVTEST_TAG=nv34
[ -n "${BASE_SHA:-}" ] || { echo "[nvtest] $(date -u +%T) [nv34] no BASE_SHA; idle"; exec sleep infinity; }
# The start the nvNNbox.sh scripts share: say, q, nvbox_boot, nvbox_apt.
curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad/nvbox.sh" \
  -o /root/nvbox.sh || { echo "[nvtest] $(date -u +%T) [nv34] FAIL fetch nvbox.sh"; exec sleep infinity; }
# shellcheck disable=SC1091
. /root/nvbox.sh
NVBOX_MEM=1 nvbox_boot
case "$CC" in 86|89) ;; *) say "FAIL this card is compute $CC, not 8.6 or 8.9"; say "all done"; exec sleep infinity ;; esac
nvbox_apt
export CUDA_VER=12.8 NVTEST_STAGE=lib
# shellcheck disable=SC1091
. /root/nvtest.sh
unset NVTEST_STAGE
setup
cuda
ND=cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz
if [ ! -x "$CUDA/bin/nvdisasm" ]; then
  curl -fsSL --retry 3 "$RD/$ND" -o /root/nvd.tar.xz || fail "nvdisasm download"
  echo "8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae  /root/nvd.tar.xz" | sha256sum -c --quiet - \
    || fail "nvdisasm checksum"
  tar -xJf /root/nvd.tar.xz -C "$CUDA" --strip-components=1 || fail "nvdisasm unpack"
fi
fetch
export PATH="$CUDA/bin:/opt/node/bin:$PATH"
STAGES=" ${STAGES:-product bench} "
has() { case "$STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
H=$R/earn/native/probes/hand89
W=/root/w
mkdir -p $W
NV="nvcc -arch=sm_$CC -std=c++17 -O3"

# SMIQ=<secs>: what nvidia-smi -q says about power, voltage, clocks and the clock's limiters (the active ones),
# for whichever miner runs then. Plain observation, nothing set.
if [ -n "${SMIQ:-}" ]; then
  ( while true; do
      say "smiq $(nvidia-smi -q -i 0 -d POWER,VOLTAGE,PERFORMANCE,CLOCK 2>&1 | awk '
        BEGIN { FS = " : " }
        /^    [A-Z]/ { s = $0; gsub(/^ +| +$/, "", s)
          t = (s == "Clocks") ? "C" : (s == "Voltage") ? "V" : (s ~ /Power Readings/) ? "P" : (s ~ /^Clocks (Event|Throttle) Reasons$/) ? "R" : (s == "Power Samples") ? "S" : "" }
        NF == 2 && t != "" { k = $1; v = $2; gsub(/^ +| +$/, "", k); gsub(/^ +| +$/, "", v)
          if (t == "R") { if (v == "Active") printf "R:%s; ", k; next }
          if (v == "N/A") next
          if (t == "P" && k !~ /^(Average Power Draw|Instantaneous Power Draw|Current Power Limit|Power Draw)$/) next
          if (t == "S" && k !~ /^(Max|Min|Avg)$/) next
          if (t == "C" && k !~ /^(Graphics|SM|Memory)$/) next
          if (t == "V" && k != "Graphics") next
          printf "%s:%s=%s; ", t, k, v }' | tr -d '%|' | cut -c1-360)"
      sleep "$SMIQ"
    done ) &
  SMIQ_PID=$!
fi

if has product; then
  t0=$(date +%s)
  if [ ! -x /opt/node/bin/node ]; then
    mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz \
      | tar -xJ -C /opt/node --strip-components=1 || fail "node download"
  fi
  mkdir -p /opt/gyp && (cd /opt/gyp && echo '{}' > package.json &&
    npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp > npm.log 2>&1 &&
    ./node_modules/.bin/node-gyp install >> npm.log 2>&1) || fail "node-gyp setup: $(tail -1 /opt/gyp/npm.log)"
  # The file a release ships for this card, as CI builds it.
  (cd "$R/earn/native" && python3 -X utf8 probes/hand89/build_rot.py nvcc "$CUDA/bin/nvdisasm" $W/br --arch "sm_$CC" > $W/br.log 2>&1)
  f=$W/br/pearl_rot_sm$CC.cubin; [ -f "$f" ] || f=$f.unverified
  if [ -f "$f" ]; then cp "$f" $W/pearl_rot_ship.cubin; say "rot ship (build_rot.py sm_$CC): hash $(python3 -I "$H/asm89.py" hash $W/pearl_rot_ship.cubin)"
  else say "rot ship FAILED: $(tail -3 $W/br.log | tr '\n' ' ' | cut -c1-300)"; fi
  # On an RTX 40 card, also the file a release ships for RTX 30 (ship86): ptxas gives sm_86 and sm_89 the same code
  # for this kernel, so the card can run it, and it is beside every core already (goal-r1-rtx40).
  if [ "$CC" = 89 ]; then
    (cd "$R/earn/native" && python3 -X utf8 probes/hand89/build_rot.py nvcc "$CUDA/bin/nvdisasm" $W/br86 --arch sm_86 > $W/br86.log 2>&1)
    f=$W/br86/pearl_rot_sm86.cubin; [ -f "$f" ] || f=$f.unverified
    if [ -f "$f" ]; then cp "$f" $W/pearl_rot_ship86.cubin; say "rot ship86 (build_rot.py sm_86): hash $(python3 -I "$H/asm89.py" hash $W/pearl_rot_ship86.cubin)"
    else say "rot ship86 FAILED: $(tail -3 $W/br86.log | tr '\n' ' ' | cut -c1-300)"; fi
  fi
  IFS=';' read -r -a RV <<< "${ROT_VARIANTS:-}"
  for v in "${RV[@]}"; do
    [ -n "$v" ] || continue
    vn=${v%%:*}; d=""; b=vbase
    case ",${v#*:}," in *,sr=1,*) d="$d -DROT_SR=1"; b=${b}_sr ;; esac
    case ",${v#*:}," in *,bx=1,*) d="$d -DROT_BX=1"; b=${b}_bx ;; esac
    if [ ! -f "$W/$b.sass" ]; then
      # shellcheck disable=SC2086
      $NV -cubin $d -I "$R/earn/native/src" "$R/earn/native/src/pearl_rot.cu" -o "$W/$b.cubin" > "$W/$b.log" 2>&1 \
        && nvdisasm -c "$W/$b.cubin" > "$W/$b.sass" || { say "rot base$d FAILED"; continue; }
    fi
    # shellcheck disable=SC2046
    if python3 -I "$H/rfold.py" "$W/$b.cubin" "$W/$b.sass" nvdisasm "$W/pearl_rot_$vn.cubin" \
        $(echo "${v#*:}" | tr ',' ' ') > "$W/rfold_$vn.log" 2>&1; then
      say "rot $vn: $(tail -1 "$W/rfold_$vn.log" | cut -c1-200); hash $(python3 -I "$H/asm89.py" hash "$W/pearl_rot_$vn.cubin")"
    else
      say "rot $vn FAILED: $(tail -2 "$W/rfold_$vn.log" | tr '\n' ' ' | cut -c1-260)"
    fi
  done
  K=$W/addon
  rm -rf $K && mkdir -p $K/cuda-build && cp -r "$R/earn/native/src" "$R/earn/native/binding.gyp" $K/
  C="-O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_$CC,code=sm_$CC"
  # shellcheck disable=SC2086
  nvcc $C -c $K/src/pearl_kernel.cu -o $K/cuda-build/pearl_kernel.o > $K/kernel.log 2>&1 || fail "kernel: $(grep -m2 -i error $K/kernel.log)"
  # shellcheck disable=SC2086
  nvcc $C -c $K/src/pearl_host.cu -o $K/cuda-build/pearl_host.o > $K/host.log 2>&1 || fail "host: $(grep -m2 -i error $K/host.log)"
  ar rcs $K/cuda-build/libpearl_cuda.a $K/cuda-build/pearl_kernel.o $K/cuda-build/pearl_host.o
  (cd $K && ln -s /opt/gyp/node_modules node_modules && echo '{}' > package.json &&
    CUDA_PATH=$CUDA ./node_modules/.bin/node-gyp rebuild > gyp.log 2>&1) || fail "node-gyp: $(grep -m2 -i error $K/gyp.log)"
  ADDON=$K/build/Release/pearl_core.node
  say "addon built in $(( $(date +%s) - t0 ))s"
  smi() { q clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown; }
  # The shipped files beside the core, as a release has them ("auto" legs).
  cp $W/pearl_rot_ship.cubin "$(dirname "$ADDON")/pearl_rot_sm$CC.cubin" 2>/dev/null
  [ -f $W/pearl_rot_ship86.cubin ] && cp $W/pearl_rot_ship86.cubin "$(dirname "$ADDON")/pearl_rot_sm86.cubin"
  cd "$R/earn/native/probes" || fail "no probes"
  [ "${VERIFY_SECS:-150}" != 0 ] && for vv in ${VERIFY_LIST:-ship}; do
    ve="PEARL_ROT_CUBIN=$W/pearl_rot_$vv.cubin"; [ "$vv" = auto ] && ve="PEARL_ROT_LOG=1"   # auto: the host's own pick
    # shellcheck disable=SC2086
    j=$(env PEARL_FOLD_ROT=1 $ve timeout -k 5 $(( ${VERIFY_SECS:-150} + 120 )) \
      node verify-hits.js "$ADDON" "${VERIFY_SECS:-150}" "${VERIFY_BITS:-233}" 2> $W/verify.err | grep '^{' | tail -1)
    say "verify $vv $(echo "$j" | jq -c '{PASS, hits, verified, failed, salts, fold: .device.fold, thLast, fails: (.fails[0:2])}' 2>/dev/null | cut -c1-390)"
    grep -a -E 'clock event reasons|RTX 30 loop|RTX 40 loop' $W/verify.err | head -8 | while IFS= read -r l; do say "rule verify $vv $(echo "$l" | cut -c1-260)"; done
    [ -n "$j" ] || say "verify $vv no result: $(tail -2 $W/verify.err | tr '\n' ' ' | cut -c1-300)"
  done
  n=0
  for leg in ${HR_LEGS:-ship ship}; do
    n=$((n + 1))
    lv=${leg%%+*}; extra=""
    [ "$lv" != "$leg" ] && extra=$(echo "${leg#*+}" | tr '+' ' ')
    case "$lv" in
      rel) e="PEARL_FOLD_ROT=0" ;;
      auto) e="PEARL_FOLD_ROT=1 PEARL_ROT_LOG=1" ;;
      *) e="PEARL_FOLD_ROT=1 PEARL_ROT_CUBIN=$W/pearl_rot_$lv.cubin" ;;
    esac
    # shellcheck disable=SC2086
    j=$(env $e $extra timeout -k 5 $(( ${HR_SECS:-120} + 120 )) node hashrate.js "$ADDON" "${HR_SECS:-120}" 10 2> $W/hr$n.err | grep '^{' | tail -1)
    say "hashrate $leg$n $(echo "$j" | jq -c '{k, th, sd, n, mhz, watts, tempC, memUtil, fold: .device.fold}' 2>/dev/null | cut -c1-300) end $(smi)"
    [ -n "$j" ] || say "hashrate $leg$n no result: $(tail -2 $W/hr$n.err | tr '\n' ' ' | cut -c1-300)"
    grep -a -E 'clock event reasons|RTX 30 loop|RTX 40 loop' $W/hr$n.err | head -8 | while IFS= read -r l; do say "rule $leg$n $(echo "$l" | cut -c1-260)"; done
  done
fi
if has bench; then
  # BENCH_CUBIN=auto: no file named. The shipped files go beside the core, and the host picks file and loop itself.
  if [ -n "${ADDON:-}" ] && [ "${BENCH_CUBIN:-ship}" = auto ]; then export OURS_ROT_DIR; OURS_ROT_DIR=$(dirname "$ADDON"); export ROTLOG=1
  elif [ -n "${ADDON:-}" ]; then export PEARL_ROT_CUBIN=/root/w/pearl_rot_${BENCH_CUBIN:-ship}.cubin; fi
  # W: a payout address in form only, which is all our CLI checks. The stand-in pool pays nobody.
  W=prl1pgate00000000000000000000000000000000000000000000000000000 ORDER=${ORDER:-ours,ozm,peak,srb} \
    VPOOL=${VPOOL:-210} SECS=${SECS:-180} WARM=${WARM:-60} OURS_SECS=${SECS:-180} VER=${VER:-v0.7.2} \
    OURS_CORE=${ADDON:-} bash /root/bench.sh 2>&1 | sed -u 's/^\[bench\]/[nvtest] [bench]/'
fi
[ -n "${SMIQ_PID:-}" ] && kill "$SMIQ_PID" 2>/dev/null
say "card at the end $(q clocks.sm,clocks.mem,power.draw,temperature.gpu)"
say "all done"
exec sleep infinity
