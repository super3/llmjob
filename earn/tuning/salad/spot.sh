# A spot check inside a fleet box: our release against each competitor, 5 minutes each, on the PC the box is
# already mining on, then back to normal mining. box.sh sources this file before its main loop (the watcher's
# cmd_for adds the line when its spot config picks the group), so it has box.sh's D, N, W, VER, now, stop_gpu and
# start_gpu. Changing a running box's command would make Salad restart it, often on another PC; this keeps the PC.
#
# SPOT_AFTER_MIN: how long the box mines first (120). The check waits for a steady miner: no 0 TH/s in its last 10
# minutes of rate lines, trying every 30 minutes, 3 times at most. SPOT_BENCH: the URL of bench.sh at a pinned
# commit. SPOT_ORDER: the miners, in order ("ours,peak,srb,ozm").
#
# While it runs, $D/spot.on tells box.sh's loop to leave the card alone (no [mining] line, no restart of our miner),
# and the watcher holds its rules for the box (see W.spotting). bench.sh stops our miner when its first leg starts,
# which stops Gemma too: no row comes from a card the LLM shares. Every miner mines to the box's own wallet on
# Kryptex, so the box keeps earning through the check, less the competitors' fees and startups. The box prints:
#   [spot] start HH:MM:SS                 when it begins
#   [bench] ...                           bench.sh's own lines, which the watcher scores as bench.py does
#   [spot] end HH:MM:SS <how>             when our miner is mining again ("done", "timeout", "FAIL ...")
#   [spot] skip HH:MM:SS <why>            when it doesn't run at all
# A box checks once ($D/spot.done); a restart on the same PC doesn't check again.

# Steady means a rate in each of the last 600 rate lines (about 10 minutes), or in all of them since the wait
# began when SPOT_AFTER_MIN is shorter: the founder asked on 17 November for a check after 5 minutes of mining.
spot_steady() {
  local n z k=$(( ${SPOT_AFTER_MIN:-120} * 60 < 600 ? ${SPOT_AFTER_MIN:-120} * 60 : 600 ))
  n=$(tr -d '\r' < $D/g0.log 2>/dev/null | grep -aoE '[0-9.]+ TH/s · ' | tail -$k | wc -l)
  z=$(tr -d '\r' < $D/g0.log 2>/dev/null | grep -aoE '[0-9.]+ TH/s · ' | tail -$k | awk '$1 == 0 {z++} END {print z + 0}')
  [ "$n" -ge "$k" ] && [ "$z" = 0 ]
}

spot_run() {
  [ -f $D/spot.done ] && return
  sleep $(( ${SPOT_AFTER_MIN:-120} * 60 ))
  [ "$N" = 1 ] || { echo "[spot] skip $(now) $N cards"; touch $D/spot.done; return; }
  local try=1
  until spot_steady; do
    [ $try -ge 3 ] && { echo "[spot] skip $(now) miner not steady"; touch $D/spot.done; return; }
    try=$((try + 1)); sleep $(( ${SPOT_AFTER_MIN:-120} < 30 ? 300 : 1800 ))
  done
  touch $D/spot.done $D/spot.on
  echo "[spot] start $(now)"
  if ! curl -fsSL --retry 5 --retry-delay 5 -o $D/spot-bench.sh "$SPOT_BENCH"; then
    rm -f $D/spot.on; echo "[spot] end $(now) FAIL download bench.sh"; return
  fi
  ( cd / && W=$W ORDER=${SPOT_ORDER:-ours,peak,srb,ozm} OURS_SECS=300 VER=$VER OURS_REL_DIR=$D/rel \
      exec bash $D/spot-bench.sh ) > $D/spot.log 2>&1 &
  local bp=$! from=0 how=timeout t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 2700 ]; do   # 45 minutes at most
    sleep 15
    local size; size=$(wc -c < $D/spot.log)
    tail -c +$((from + 1)) $D/spot.log | head -c $((size - from)) | tr -d '\r' | grep -a '^\[bench\] ' | cut -c1-400
    from=$size
    if grep -aq '^\[bench\] [0-9:]* all done' $D/spot.log; then
      how=done; grep -aq '^\[bench\] [0-9:]* FAIL ' $D/spot.log && how="FAIL $(grep -a -m1 '^\[bench\] [0-9:]* FAIL ' $D/spot.log | cut -d' ' -f4- | cut -c1-200)"
      break
    fi
  done
  # bench.sh sleeps once it is done; stop it and any miner it left running, then mine as before.
  pkill -P $bp 2>/dev/null; kill $bp 2>/dev/null
  pkill -f '\./peak -o'; pkill -f SRBMiner-MULTI; pkill -f wildrig-multi; pkill -f onezerominer; pkill -f 'spot-bench\.sh'
  for p in $(pgrep -f '\./cli -a'); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /opt/b ] && kill -9 $p 2>/dev/null; done
  sleep 5
  stop_gpu 0; rm -f $D/spot.on; start_gpu 0 "$(cat $D/g0.cfg)"
  echo "[spot] end $(now) $how"
}

spot_run &
