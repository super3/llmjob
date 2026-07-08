#!/usr/bin/env bash
# Build the CLI argument line from the flight sheet fields. The HiveOS agent
# exports these before calling us:
#   CUSTOM_TEMPLATE     the wallet field — prl1p… or prl1p…+mdl1p… (merge mining)
#   CUSTOM_USER_CONFIG  extra raw CLI args, e.g. --region eu1 --difficulty 131072
#   WORKER_NAME         this rig's HiveOS worker name
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./h-manifest.conf

if [[ -z $CUSTOM_TEMPLATE ]]; then
  echo -e "${RED}No wallet set — put your prl1p… address in the flight sheet's wallet field${NOCOLOR}"
  exit 1
fi

# Split a prl1…+mdl1… combined wallet into the CLI's separate flags.
ADDR=${CUSTOM_TEMPLATE%%+*}
MDL=""
[[ $CUSTOM_TEMPLATE == *"+"* ]] && MDL=${CUSTOM_TEMPLATE#*+}

ARGS="--address $ADDR"
[[ -n $MDL ]] && ARGS="$ARGS --mdl $MDL"
[[ -n $WORKER_NAME ]] && ARGS="$ARGS --worker $WORKER_NAME"
# Self-update stays off under HiveOS — the agent owns the miner lifecycle, and
# updates arrive by reinstalling the package. The stats file feeds h-stats.sh.
ARGS="$ARGS --no-update --stats-file /run/hive/llmjob-earn-stats.json"
[[ -n $CUSTOM_USER_CONFIG ]] && ARGS="$ARGS $CUSTOM_USER_CONFIG"

echo "$ARGS" > "$CUSTOM_CONFIG_FILENAME"
