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

# Cards turned off in HiveOS (power limit 1 in the NVIDIA overclock, or
# GPU_DISABLE) must not mine. HiveOS's `miner start` says which are off through
# CUDA_VISIBLE_DEVICES: the CUDA numbers of the NVIDIA cards still on, such as
# 0,2, or a single space when every one is off. It is not set when none is off.
# The CLI drops that variable when it starts (see shared/gpu.js), so pass the
# choice on as --gpu-index. The numbers mean the same cards to both: HiveOS and
# the CLI number them in PCI bus order (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Anything else in it (GPU UUIDs, MIG ids) did not come from HiveOS, so it is
# not passed on, and the CLI logs that it ignored it. A --gpu-index in Extra
# config replaces HiveOS's choice.
if [[ -v CUDA_VISIBLE_DEVICES && ! " $CUSTOM_USER_CONFIG " =~ [[:space:]]--gpu-index([[:space:]]|=) ]]; then
  if [[ $CUDA_VISIBLE_DEVICES =~ ^[[:space:]]*$ ]]; then
    args+=" --gpu-index=none"
  elif [[ $CUDA_VISIBLE_DEVICES =~ ^[[:space:]]*([0-9]+(,[0-9]+)*)[[:space:]]*$ ]]; then
    args+=" --gpu-index=${BASH_REMATCH[1]}"
  fi
fi

# Self-update stays off: the agent owns the miner's lifecycle, and a new version
# arrives as a new Installation URL. The stats file feeds h-stats.sh.
args+=" --no-update --stats-file /run/hive/llmjob-earn-stats.json"
[[ -n $CUSTOM_USER_CONFIG ]] && args+=" $CUSTOM_USER_CONFIG"

echo "$args" > "$CUSTOM_CONFIG_FILENAME"
