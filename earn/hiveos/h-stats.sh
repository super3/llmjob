#!/usr/bin/env bash
# Sourced by the HiveOS agent — must set two variables:
#   khs    total hashrate in kH/s
#   stats  JSON: { hs, hs_units, temp, fan, uptime, ver, ar, algo }
#
# The CLI writes /run/hive/llmjob-earn-stats.json every 10s (--stats-file).
# `ths` in that file is TH/s: ×1e9 → kH/s for khs, ×1e6 → MH/s for hs. Temps
# and fans come from HiveOS's own `gpu-stats`. A stale file (miner hung or
# stopped >2 min) reports zeros so the dashboard shows the truth.
LLE_SF=/run/hive/llmjob-earn-stats.json
khs=0
stats="null"

if [[ -f $LLE_SF ]]; then
  LLE_NOW=$(date +%s)
  LLE_MT=$(stat -c %Y "$LLE_SF" 2>/dev/null || echo 0)
  if (( LLE_NOW - LLE_MT <= 120 )); then
    LLE_GPUS=$(gpu-stats 2>/dev/null)
    LLE_TEMP=$(jq -c '[.temp[]? // 0]' <<< "$LLE_GPUS" 2>/dev/null)
    LLE_FAN=$(jq -c '[.fan[]? // 0]' <<< "$LLE_GPUS" 2>/dev/null)
    [[ -z $LLE_TEMP || $LLE_TEMP == "null" ]] && LLE_TEMP='[]'
    [[ -z $LLE_FAN || $LLE_FAN == "null" ]] && LLE_FAN='[]'

    khs=$(jq -r '(.ths // 0) * 1000000000' "$LLE_SF" 2>/dev/null)
    [[ -z $khs || $khs == "null" ]] && khs=0

    stats=$(jq -c --argjson temp "$LLE_TEMP" --argjson fan "$LLE_FAN" '{
      hs: [((.ths // 0) * 1000000)],
      hs_units: "mhs",
      temp: $temp,
      fan: $fan,
      uptime: (.uptimeSec // 0),
      ver: (.ver // "0"),
      ar: [(.accepted // 0), (.rejected // 0)],
      algo: "pearlhash"
    }' "$LLE_SF" 2>/dev/null)
    if [[ -z $stats ]]; then stats="null"; khs=0; fi
  fi
fi
