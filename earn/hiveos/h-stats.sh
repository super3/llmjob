#!/usr/bin/env bash
# Sourced by the HiveOS agent about every 10 s, in a subshell with a 20 s limit.
# It must set two variables:
#   khs    the rig's total hashrate in kH/s
#   stats  JSON for the dashboard: per-card hs (kH/s), bus_numbers, temp and
#          fan, plus ar, uptime, ver and algo
#
# Everything comes from the stats file the CLI writes every 10 s (--stats-file,
# set by h-config.sh). Hashrates there are in TH/s; 1 TH/s is 1e9 kH/s.
#
# Each card the CLI mines on gets its own hs entry, and its PCI bus number in
# bus_numbers. That is how HiveOS puts a number on the right GPU row. Without it
# HiveOS matches by position, and gpu-detect lists an iGPU or a server's BMC
# display first, so every number would land one row off. The CLI writes each
# card's bus (nvidia-smi's pci.bus_id) next to that card's index. If a card has
# none, the n-th NVIDIA card in $gpu_detect_json stands in, since both are in
# PCI bus order. If that doesn't cover every card either, bus_numbers is left
# out.
#
# Temps and fans come from $gpu_stats, which the agent has already read, matched
# by bus. This never runs gpu-stats itself: that can take longer than the 20 s
# limit. A card gpu-stats doesn't list gets the CLI's own temp and fan 0.
#
# A stats file older than 2 minutes means the miner is hung or gone, so khs is 0
# and stats is null rather than old numbers.
#
# LLMJOB_EARN_STATS_FILE overrides the file's path, for tests.
llmjob_earn_stats() {
  local file=${LLMJOB_EARN_STATS_FILE:-/run/hive/llmjob-earn-stats.json}
  khs=0
  stats=null

  [[ -f $file ]] || return 0
  local mtime
  mtime=$(stat -c %Y "$file" 2>/dev/null) || return 0
  (( $(date +%s) - mtime <= 120 )) || return 0

  # Read once: the CLI replaces the file every 10 s.
  local data
  data=$(< "$file") || return 0

  local gs=${gpu_stats:-$(cat "${GPU_STATS_JSON:-/run/hive/gpu-stats.json}" 2>/dev/null)}
  local gd=${gpu_detect_json:-$(cat "${GPU_DETECT_JSON:-/run/hive/gpu-detect.json}" 2>/dev/null)}
  # jq 1.6, which HiveOS ships, exits 0 on empty input, hence the -n tests.
  [[ -n $gs ]] && jq -e 'type == "object"' >/dev/null 2>&1 <<< "$gs" || gs='{}'
  [[ -n $gd ]] && jq -e 'type == "array"' >/dev/null 2>&1 <<< "$gd" || gd='[]'

  local out
  out=$(jq -c --argjson gs "$gs" --argjson gd "$gd" '
    # "00000000:0A:00.0" or "0a:00.0" -> 10, the decimal bus HiveOS wants.
    def bus:
      if type == "string" then
        (capture("(?<b>[0-9A-Fa-f]{2}):[0-9A-Fa-f]{2}\\.[0-9A-Fa-f]+$").b
          | ascii_downcase | explode | map(if . >= 97 then . - 87 else . - 48 end)
          | .[0] * 16 + .[1]) // null
      else null end;
    def num: tonumber? // 0;

    (.gpus // []) as $cards
    | (($gs.busids // []) | map(bus)) as $gsbus
    | [$gd[]? | select(.brand == "nvidia" and .vbios != "DISABLED") | .busid | bus] as $nvbus
    | ($cards | map(.pciBusId | bus)) as $own
    | (if ($own | all(. != null)) then $own
       elif ($cards | all((.index | type) == "number" and .index >= 0 and $nvbus[.index] != null))
       then [$cards[] | $nvbus[.index]]
       else null end) as $bus
    | [range(0; $cards | length) as $k
        | $cards[$k] as $c
        | (if $bus == null then null else ($gsbus | index($bus[$k])) end) as $row
        | { hs: (($c.hashrate // 0) * 1000000000 | floor),
            temp: (if $row == null then ($c.temp // 0) else ($gs.temp[$row] | num) end),
            fan: (if $row == null then 0 else ($gs.fan[$row] | num) end) }] as $rows
    | { hs: [$rows[].hs],
        hs_units: "khs",
        temp: [$rows[].temp],
        fan: [$rows[].fan],
        uptime: (.uptimeSec // 0),
        ver: (.ver // ""),
        ar: [(.accepted // 0), (.rejected // 0)],
        algo: "pearlhash" }
      + (if ($bus // []) | length > 0 then { bus_numbers: $bus } else {} end)
  ' <<< "$data" 2>/dev/null) || return 0
  [[ -n $out ]] || return 0

  local total
  total=$(jq -r '(.ths // 0) * 1000000000 | floor' <<< "$data" 2>/dev/null)
  [[ $total =~ ^[0-9]+$ ]] || return 0
  khs=$total
  stats=$out
}

llmjob_earn_stats
unset -f llmjob_earn_stats
