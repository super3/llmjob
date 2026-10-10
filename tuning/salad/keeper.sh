#!/bin/bash
# Keeps watcher.py running: every 30 s, starts it if it isn't. Start this detached, e.g.
#   PRL_WALLET=prl1... SALAD_MINE_DIR=/some/dir setsid nohup bash tuning/salad/keeper.sh >/dev/null 2>&1 </dev/null &
H=$(cd "$(dirname "$0")" && pwd)
M=${SALAD_MINE_DIR:-$H/run}; export SALAD_MINE_DIR=$M
mkdir -p "$M"; echo $$ > "$M/keeper.pid"
while true; do
  if ! kill -0 "$(cat "$M/watcher.pid" 2>/dev/null)" 2>/dev/null; then
    [ -f "$M/state.json" ] && echo "$(date -u +%H:%M)Z KEEPER watcher was not running; started it" >> "$M/events.log"
    nohup python3 -I "$H/watcher.py" >> "$M/watcher.err" 2>&1 < /dev/null &
    echo $! > "$M/watcher.pid"
  fi
  sleep 30
done
