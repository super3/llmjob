#!/usr/bin/env bash
# Starts the miner. HiveOS runs this inside its miner screen. Output goes to the
# screen and to the log, so `miner log` and the web console both show it.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./h-manifest.conf

# HiveOS runs h-config.sh, which writes this file, before every start.
if [[ ! -f $CUSTOM_CONFIG_FILENAME ]]; then
  echo -e "${RED}No $CUSTOM_CONFIG_FILENAME. Run miner restart to write it.${NOCOLOR}"
  exit 1
fi
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

ARGS=$(< "$CUSTOM_CONFIG_FILENAME")
# `miner stop` sends Ctrl+C, which reaches everything in this pipeline. Plain
# tee dies at once, so the CLI's shutdown lines never reach the log. tee -i
# ignores it and keeps writing until the CLI exits. -a appends, as HiveOS's own
# miners do: a restart within 30 s skips log rotation, and that crashed run's
# log is the one worth keeping.
./llmjob-earn-cli-linux $ARGS 2>&1 | tee -i -a "$CUSTOM_LOG_BASENAME.log"
