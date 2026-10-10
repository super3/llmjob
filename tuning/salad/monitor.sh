#!/bin/bash
# Keeps watch_all.py and hourly.py running (starts either one that isn't), and prints the log lines worth reading.
# Run it under a monitor; each printed line is an event.
cd "$(dirname "$0")"
alive() { for p in $(pgrep -x python3); do [ "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" = "python3 -u $1 " ] && [ "$(readlink /proc/$p/cwd)" = "$PWD" ] && return 0; done; return 1; }
touch watch_all.log hourly.log; W=$(wc -l < watch_all.log); H=$(wc -l < hourly.log)
while true; do
  for s in watch_all.py hourly.py; do
    log=$([ $s = hourly.py ] && echo hourly.log || echo watch_all.log)
    if ! alive $s; then (setsid nohup python3 -u $s >> $log 2>&1 < /dev/null &); echo "salad: $s was not running: started"; fi
  done
  sleep 15
  m=$(wc -l < watch_all.log); [ "$m" -gt "$W" ] && sed -n "$((W+1)),${m}p" watch_all.log | grep -E 'running on|moved to|moved it|not running|first share|stopping|stopped|why:|started|summary:|credit:|check failed|Traceback|Error' | sed 's/^/salad: /'; W=$m
  m=$(wc -l < hourly.log); [ "$m" -gt "$H" ] && sed -n "$((H+1)),${m}p" hourly.log | grep -E 'search|create|Traceback|Error' | sed 's/^/salad hourly: /'; H=$m
done
