#!/usr/bin/env bash
# Starts the miner. HiveOS runs this inside its miner screen, in a loop that
# starts it again 3 s after it ends. Output goes to the screen and to the log,
# so `miner log` and the web console both show it.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./h-manifest.conf

# HiveOS runs h-config.sh, which writes this file, before every start.
if [[ ! -f $CUSTOM_CONFIG_FILENAME ]]; then
  echo -e "${RED}No $CUSTOM_CONFIG_FILENAME. Run miner restart to write it.${NOCOLOR}"
  exit 1
fi
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
log=$CUSTOM_LOG_BASENAME.log

ARGS=$(< "$CUSTOM_CONFIG_FILENAME")
# LLMJOB_HIVEOS=1 tells the CLI it runs here: it updates this whole folder
# itself and exits, so the loop starts the new version, and it posts problems
# to the HiveOS dashboard.
#
# `miner stop` sends Ctrl+C, which reaches everything in this pipeline. Plain
# tee dies at once, so the CLI's shutdown lines never reach the log. tee -i
# ignores it and keeps writing until the CLI exits. -a appends, as HiveOS's own
# miners do: a restart within 30 s skips log rotation, and that crashed run's
# log is the one worth keeping.
LLMJOB_HIVEOS=1 ./llmjob-earn-cli-linux $ARGS 2>&1 | tee -i -a "$log"
code=${PIPESTATUS[0]}

# A version an update swapped in is on trial (.update-trial) until it hashes,
# and the version before it waits in llmjob-earn.prev. If it ends with an error
# before then, put the old version back, and have it skip this one for a day.
# The loop then starts the old version's h-run.sh.
dir=$(pwd -P)
if [[ $code -ne 0 && -f .update-trial && -d $dir.prev ]]; then
  from=$(sed -n 's/^UPDATE_FROM=//p' .update-trial)
  to=$(sed -n 's/^UPDATE_TO=//p' .update-trial)
  # The argument line, when it lives in this folder: HiveOS may have rewritten
  # it since the update.
  [[ $(realpath -m "$(dirname "$CUSTOM_CONFIG_FILENAME")") == "$dir" ]] &&
    cp -f "$CUSTOM_CONFIG_FILENAME" "$dir.prev/"
  rm -rf "$dir.failed"
  if mv "$dir" "$dir.failed"; then
    if mv "$dir.prev" "$dir"; then
      rm -rf "$dir.failed"
      printf 'SKIP_VERSION=%s\nSKIP_UNTIL=%s\n' "$to" "$(( $(date +%s) + 86400 ))" > "$dir/.update-skip"
      text="LLMJob Earn v$to stopped with an error (exit code $code), so this rig went back to v$from. It tries v$to again in a day."
      echo "$text" | tee -a "$log"
      command -v message >/dev/null && message warning "$text" >/dev/null 2>&1
    else
      # Never leave HiveOS without a miner folder.
      mv "$dir.failed" "$dir"
    fi
  fi
fi
exit "$code"
