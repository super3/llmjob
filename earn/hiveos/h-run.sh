#!/usr/bin/env bash
# Start the miner. The agent runs this inside a screen session; stdout is also
# tee'd to the log so `miner log` and the web console both work.
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./h-manifest.conf

[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && ./h-config.sh
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

ARGS=$(< "$CUSTOM_CONFIG_FILENAME")
./llmjob-earn-cli-linux $ARGS 2>&1 | tee "$CUSTOM_LOG_BASENAME.log"
