#!/usr/bin/env bash
# Writes the CLI's argument line from the flight sheet. HiveOS's custom-miner
# wrapper sources this in a subshell before every start, after rig.conf and
# wallet.conf are loaded, so these are set:
#   CUSTOM_TEMPLATE     "Wallet and worker template", macros already filled in:
#                       prl1p… for %WAL%, prl1p….rig01 for %WAL%.%WORKER_NAME%
#   CUSTOM_USER_CONFIG  "Extra config arguments": more CLI flags, e.g. --region de
#   WORKER_NAME         this rig's name in HiveOS
# Pool URL and Pass (CUSTOM_URL, CUSTOM_PASS) are not used: the CLI picks the
# fastest HeroMiners region itself, or the one --region names.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./h-manifest.conf

# No address or worker name contains whitespace, and a stray space from the
# form would split the argument line.
template=${CUSTOM_TEMPLATE//[[:space:]]/}

# %WAL%.%WORKER_NAME% is HiveOS's usual template. Pearl addresses never contain
# '.', so everything after the first '.' is the worker name. It wins over
# WORKER_NAME.
wallet=${template%%.*}
worker=$WORKER_NAME
[[ $template == *.* ]] && worker=${template#*.}

# prl1…+mdl1… is the old merge-mining form. Merge mining is retired, but the CLI
# still takes --mdl so old flight sheets keep mining.
address=${wallet%%+*}
mdl=
[[ $wallet == *+* ]] && mdl=${wallet#*+}

if [[ -z $address ]]; then
  echo -e "${RED}No wallet set. Put %WAL% (your prl1p… address) in the flight sheet's wallet template${NOCOLOR}"
  exit 1
fi

# h-run.sh splits the argument line on whitespace and expands * ? [ ], so the
# worker name can't carry those. "--worker=" so a name starting with '-' isn't
# read as a flag.
worker=${worker//[][[:space:]*?]/_}

args="--address $address"
[[ -n $mdl ]] && args+=" --mdl $mdl"
[[ -n $worker ]] && args+=" --worker=$worker"

# A HiveOS rig is a mining rig, so mine only unless Extra config sets a mode.
# The CLI's own default (auto) also serves a local LLM, which downloads a
# 5-18 GB model onto the HiveOS drive.
[[ " $CUSTOM_USER_CONFIG " =~ [[:space:]]--mode([[:space:]]|=) ]] || args+=" --mode mining"

# Self-update stays off: the agent owns the miner's lifecycle, and a new version
# arrives as a new Installation URL. The stats file feeds h-stats.sh.
args+=" --no-update --stats-file /run/hive/llmjob-earn-stats.json"
[[ -n $CUSTOM_USER_CONFIG ]] && args+=" $CUSTOM_USER_CONFIG"

echo "$args" > "$CUSTOM_CONFIG_FILENAME"
