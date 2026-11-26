#!/bin/bash
# Box script (goal/r1-rtx30, a measuring tool, nothing ships): probes/hand89/trhome.cu on one Salad PC, its
# transcript homes raced in turns, with the card read once a second beside each leg (SM clock, watts,
# memory-controller use, the clock's limiter mask). Lines read "[nvtest] HH:MM:SS [trh] ...".
#   BASE_SHA   commit to fetch (40 characters)
#   HOMES      homes in turn order (default "0 1 2 3 4 5"), ROUNDS (3), LEG_SECS (60)
set -u
NVTEST_TAG=trh
[ -n "${BASE_SHA:-}" ] || { echo "[nvtest] $(date -u +%T) [trh] no BASE_SHA; idle"; exec sleep infinity; }
curl -fsSL --retry 3 "https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad/nvbox.sh" \
  -o /root/nvbox.sh || { echo "[nvtest] $(date -u +%T) [trh] FAIL fetch nvbox.sh"; exec sleep infinity; }
# shellcheck disable=SC1091
. /root/nvbox.sh
nvbox_boot
nvbox_apt
export CUDA_VER=12.8 NVTEST_STAGE=lib
# shellcheck disable=SC1091
. /root/nvtest.sh
unset NVTEST_STAGE
setup
cuda
fetch
export PATH="$CUDA/bin:$PATH"
W=/root/trh
mkdir -p $W
for h in ${HOMES:-0 1 2 3 4 5}; do
  nvcc -arch=sm_$CC -O3 -DHOME=$h -o $W/trhome$h "$R/earn/native/probes/hand89/trhome.cu" > $W/b$h.log 2>&1 \
    || { say "build $h FAILED: $(grep -m2 -i error $W/b$h.log)"; continue; }
  cuobjdump -sass $W/trhome$h 2>/dev/null | grep -E 'LDL|STL|LDG|STG|LDS|STS' | sed 's/^ *//; s/ *;.*//' | sort | uniq -c \
    | awk '{$1=$1; printf "%s; ", $0}' | cut -c1-380 | while IFS= read -r l; do say "sass $h $l"; done
done
for r in $(seq "${ROUNDS:-3}"); do
  for h in ${HOMES:-0 1 2 3 4 5}; do
    [ -x $W/trhome$h ] || continue
    nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.memory,clocks_event_reasons.active,temperature.gpu \
      --format=csv,noheader,nounits -lms 1000 > $W/smi.csv 2>&1 & sp=$!
    j=$($W/trhome$h "${LEG_SECS:-60}" 2>&1 | tail -1)
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    s=$(awk -F', *' 'NR > 10 && $1 ~ /^[0-9]/ {n++; c += $1; w += $2; m += $3; t += $5; r[$4]++}
         END {if (!n) {print "n=0"; exit}; printf "n=%d mhz=%.0f w=%.1f mem=%.1f c=%.0f reasons=", n, c/n, w/n, m/n, t/n
              for (k in r) printf "%s:%.0f%%,", k, 100*r[k]/n}' $W/smi.csv)
    say "leg r$r home=$h $j $s"
  done
done
say "all done"
exec sleep infinity
