#!/usr/bin/env bash
# Writes the CLI's argument line from the flight sheet. HiveOS's custom-miner
# wrapper sources this in a subshell before every start, after rig.conf and
# wallet.conf are loaded, so these are set:
#   CUSTOM_TEMPLATE     "Wallet and worker template", macros already filled in:
#                       prl1p… for %WAL%, prl1p….rig01 for %WAL%.%WORKER_NAME%.
#                       A Quantus address (qz…) there mines Quantus.
#   CUSTOM_USER_CONFIG  "Extra config arguments": more CLI flags, e.g. --region de
#   WORKER_NAME         this rig's name in HiveOS
#   CUSTOM_URL          "Pool URL", as typed. A HeroMiners address leaves the
#                       choice of region to the CLI, as an empty field does.
#                       Another pool's host:port is mined on, as --pool does.
#                       --pool in Extra config beats it. On a rig that mines
#                       Quantus it is the Quantus pool instead, and
#                       --quantus-pool beats it.
# Pass (CUSTOM_PASS) is not used.
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
  echo -e "${RED}No wallet set. Put your prl1p… address (or a qz… one for Quantus) in the flight sheet's wallet template${NOCOLOR}"
  exit 1
fi

# h-run.sh splits the argument line on whitespace and expands * ? [ ], so the
# worker name can't carry those. "--worker=" so a name starting with '-' isn't
# read as a flag.
worker=${worker//[][[:space:]*?]/_}

# A Quantus address (qz…; Pearl's start prl1) means a Quantus flight sheet. The
# CLI takes it as --quantus-address, and mines Quantus (--coin quantus) unless
# Extra config chooses a coin itself. It needs no Pearl address.
if [[ $address == qz* ]]; then
  args="--quantus-address $address"
  [[ " $CUSTOM_USER_CONFIG " =~ [[:space:]]--coin([[:space:]]|=) ]] || args+=" --coin quantus"
else
  args="--address $address"
  [[ -n $mdl ]] && args+=" --mdl $mdl"
fi
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
# With --core amd it isn't passed on either: those numbers are NVIDIA cards, and
# the AMD core would take them as AMD card numbers and mine on the wrong cards.
# To leave AMD cards out, put --gpu-index in Extra config. --core intel is the
# same: the Intel core's card numbers are its own.
if [[ -v CUDA_VISIBLE_DEVICES && ! " $CUSTOM_USER_CONFIG " =~ [[:space:]]--gpu-index([[:space:]]|=)
      && ! " $CUSTOM_USER_CONFIG " =~ [[:space:]]--core([[:space:]]+|=)([aA][mM][dD]|[iI][nN][tT][eE][lL])[[:space:]] ]]; then
  if [[ $CUDA_VISIBLE_DEVICES =~ ^[[:space:]]*$ ]]; then
    args+=" --gpu-index=none"
  elif [[ $CUDA_VISIBLE_DEVICES =~ ^[[:space:]]*([0-9]+(,[0-9]+)*)[[:space:]]*$ ]]; then
    args+=" --gpu-index=${BASH_REMATCH[1]}"
  fi
fi

# Pool URL goes to the CLI as it is, for either coin, and the CLI decides what
# it means (see shared/hiveos.js). Only the CLI knows the coin for sure: Extra
# config can ask for Quantus with a prl1p… wallet. It must stay one word for h-run.sh: whitespace, as between
# several URLs, becomes commas, and glob characters become '_'. The CLI reads
# the first address and says so. Nothing is passed for an empty field.
pool_url=${CUSTOM_URL//[[:space:]]/,}
pool_url=${pool_url//[][*?]/_}
[[ $pool_url == *[!,]* ]] && args+=" --hive-pool-url=$pool_url"

# The CLI updates this package itself (h-run.sh says it runs under HiveOS), so
# the Installation URL never has to change. --no-update in Extra config keeps a
# rig on the version it has. The stats file feeds h-stats.sh.
args+=" --stats-file /run/hive/llmjob-earn-stats.json"
[[ -n $CUSTOM_USER_CONFIG ]] && args+=" $CUSTOM_USER_CONFIG"

echo "$args" > "$CUSTOM_CONFIG_FILENAME"
