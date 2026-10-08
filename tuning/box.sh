export DEBIAN_FRONTEND=noninteractive
# A rented Vast box that mines PRL with one miner per GPU and takes its settings from
# tuning/control/<worker>.json on this PR's branch, so each GPU can be tuned on its own
# while the others keep mining.
#
# Starts mining at once with the published v0.5.12 release (the CLI picks its CUDA 12 or
# CUDA 13 core). Build tools install in the background afterwards, so a core built from
# this branch can be tried on one GPU when its control file asks for one.
#
# Every 2 minutes the control file is fetched. Each GPU's settings are the file's "default"
# with its "gpus" entry laid over them; a GPU whose settings changed is restarted on its own,
# and only that GPU. A built core is checked first (400 hits recomputed from scratch, the
# way the pool verifies them) on its GPU, and used only if every hit matches.
#
# Control file:
#   {"default": {"core": "release", "flags": "", "env": {}},
#    "gpus":    {"0": {"core": "cu12"}},
#    "builds":  {"tallapt4": {"ref": "<commit or branch>", "defines": "-DPEARL_TALL_APT=4"}}}
#   core: release (CLI's choice) | cu12 | cu13 | build:<name>
#   flags: extra CLI flags, e.g. "--mine-mem-clock 0"; env: extra environment for the miner.
#
# Lines it prints (read by the supervisor and the tuning agents):
#   [mining gN] HH:MM:SS <TH/s> TH/s · <a> accepted · <r> rejected · up ...   every minute per GPU
#   [gpu gN] HH:MM:SS sm=.. mem=.. W=../.. C=.. util=..                       every 5 minutes
#   [ctl] / [build] / [verify] / [run]                                         as they happen
W=prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud
WORKER=__WORKER__
REPO=super3/llmjob; BRANCH=claude/sleepy-noether-lt2mh9; NODE_VER=v22.22.0
REL=https://github.com/super3/llmjob/releases/download/v0.5.12
CTL_API="https://api.github.com/repos/$REPO/contents/tuning/control/$WORKER.json?ref=$BRANCH"
CTL_RAW="https://raw.githubusercontent.com/$REPO/$BRANCH/tuning/control/$WORKER.json"
D=/opt/m; mkdir -p $D/rel $D/builds $D/run
export PATH=/opt/node-$NODE_VER-linux-x64/bin:/usr/local/cuda/bin:$PATH
now() { date -u +%T; }

if [ -f $D/setup ]; then
  echo "[run] resumed $(now)"
else
  echo "[run] setup $(now)"
  ok=0; for h in us us2 ca de; do timeout 5 bash -c "exec 3<>/dev/tcp/$h.pearl.herominers.com/1200" 2>/dev/null && ok=1; done
  [ $ok = 1 ] || { echo "[run] FAIL: outbound port 1200 blocked"; sleep infinity; }
  apt-get update 2>&1 | tail -1; apt-get install -y --no-install-recommends curl ca-certificates procps jq git 2>&1 | tail -1
  nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm,compute_cap --format=csv,noheader | sed 's/^/[card] /'
  for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do
    curl -fsSL --retry 3 -o $D/rel/$f "$REL/$f" || { echo "[run] FAIL: download $f"; sleep infinity; }
  done
  mv $D/rel/llmjob-earn-cli-linux $D/rel/cli && chmod +x $D/rel/cli
  echo "[run] release v0.5.12 ready"
  touch $D/setup
fi

# Build tools, in the background and at low priority so mining is not slowed.
if [ ! -f $D/tools ]; then
  ( nice -n 19 bash -c "
      apt-get install -y --no-install-recommends xz-utils python3 make g++ cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 > $D/tools.log 2>&1 &&
      curl -fsSL https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz | tar -xJ -C /opt &&
      git clone -q --depth 50 -b $BRANCH https://github.com/$REPO.git $D/src >> $D/tools.log 2>&1 &&
      mkdir -p $D/gyp && cd $D/gyp && echo '{}' > package.json &&
      npm install --no-save --ignore-scripts --no-audit --no-fund node-addon-api node-gyp >> $D/tools.log 2>&1 &&
      ./node_modules/.bin/node-gyp install >> $D/tools.log 2>&1 && touch $D/tools && echo \"[build] tools ready \$(date -u +%T)\" ||
      echo \"[build] tools FAIL: \$(tail -2 $D/tools.log | tr '\n' ' ' | cut -c1-200)\"" ) &
fi

N=$(nvidia-smi -L | wc -l)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')

# Build core <name> from <ref> with <defines>; writes $D/builds/<name>/pearl_core.node or .../FAIL.
build_core() {
  local name=$1 ref=$2 defines=$3 B=$D/builds/$1
  mkdir -p $B
  ( while [ ! -f $D/tools ]; do sleep 10; done
    cd $D/src && git fetch -q --depth 50 origin $BRANCH && git checkout -q --detach "$ref" 2>>$B/log || git checkout -q --detach "origin/$ref" 2>>$B/log ||
      { echo "[build] $name FAIL: no ref $ref"; touch $B/FAIL; exit; }
    rm -rf $B/a && mkdir -p $B/a/cuda-build && cp -r $D/src/earn/native/src $D/src/earn/native/binding.gyp $B/a/ && cd $B/a &&
    ln -s $D/gyp/node_modules node_modules && echo '{}' > package.json && GC="-gencode arch=compute_$CC,code=sm_$CC" &&
    nice -n 10 nvcc -O3 -std=c++17 -cudart static -Xcompiler -fPIC $GC $defines -c src/pearl_kernel.cu -o cuda-build/pearl_kernel.o >> $B/log 2>&1 &&
    nice -n 10 nvcc -O3 -std=c++17 -cudart static -Xcompiler -fPIC $GC $defines -c src/pearl_host.cu -o cuda-build/pearl_host.o >> $B/log 2>&1 &&
    ar rcs cuda-build/libpearl_cuda.a cuda-build/pearl_kernel.o cuda-build/pearl_host.o &&
    CUDA_PATH=/usr/local/cuda nice -n 10 ./node_modules/.bin/node-gyp rebuild >> $B/log 2>&1 &&
    cp build/Release/pearl_core.node $B/pearl_core.node && echo "[build] $name ok for sm_$CC at $(git -C $D/src rev-parse --short HEAD) $(date -u +%T)" ||
    { echo "[build] $name FAIL: $(grep -m2 -i 'error' $B/log | tr '\n' ' ' | cut -c1-240)"; touch $B/FAIL; } ) &
}

# The settings GPU $1 should run: default with gpus[$1] laid over it, as compact sorted JSON.
BASE='{"core":"release","env":{},"flags":""}'
want() { jq -cS --arg g "$1" --argjson b "$BASE" '$b * (.default // {}) * (.gpus[$g] // {})' $D/control.json 2>/dev/null || echo "$BASE"; }

start_gpu() {   # $1 = gpu, $2 = settings JSON
  local g=$1 cfg=$2 R=$D/run/g$1
  local core flags; core=$(echo "$cfg" | jq -r '.core // "release"'); flags=$(echo "$cfg" | jq -r '.flags // ""')
  rm -rf $R && mkdir -p $R && cp $D/rel/cli $R/
  local envs="PEARL_GPU_INDEX=$g"
  case "$core" in
    release) cp $D/rel/pearl_core.node $D/rel/pearl_core_cu13.node $R/ ;;
    cu12)    cp $D/rel/pearl_core.node $R/; envs="$envs PEARL_CORE_VARIANT=cu12" ;;
    cu13)    cp $D/rel/pearl_core.node $D/rel/pearl_core_cu13.node $R/; envs="$envs PEARL_CORE_VARIANT=cu13" ;;
    build:*) cp $D/builds/${core#build:}/pearl_core.node $R/pearl_core.node; envs="$envs PEARL_CORE_VARIANT=cu12" ;;
  esac
  envs="$envs $(echo "$cfg" | jq -r '(.env // {}) | to_entries | map("\(.key)=\(.value)") | join(" ")')"
  ( cd $R && env $envs ./cli -a $W --mode mining -w $WORKER-g$g --no-serve --no-update $flags >> $D/g$g.log 2>&1 & echo $! > $D/g$g.pid )
  echo "$cfg" > $D/g$g.cfg; echo 0 > $D/g$g.restarts
  echo "[ctl] g$g running $cfg $(now)"
}

stop_gpu() { [ -f $D/g$1.pid ] && kill $(cat $D/g$1.pid) 2>/dev/null; sleep 5; }

# A built core runs on a GPU only after the hit check passes on that GPU.
verified() {   # $1 = gpu, $2 = build name; prints PASS or FAIL
  local B=$D/builds/$2
  [ -f $B/PASS ] && { echo PASS; return; }; [ -f $B/BAD ] && { echo FAIL; return; }
  local out; out=$(cd $D/src/earn && CUDA_VISIBLE_DEVICES=$1 timeout -k 10 200 node native/probes/verify-hits.js $B/pearl_core.node 90 2>&1 | grep '^{' | tail -1)
  echo "[verify] $2 on g$1: $out" >&2
  echo "$out" | grep -q '"PASS":true' && { touch $B/PASS; echo PASS; } || { touch $B/BAD; echo FAIL; }
}

# After a pause the container restarts this script: keep the last control file, so tuned GPUs come straight
# back on their settings. A built core starts only if it already passed the hit check on this box.
[ -f $D/control.json ] || echo '{}' > $D/control.json; rm -f $D/g*.lastbad
for g in $(seq 0 $((N-1))); do
  cfg=$(want $g); core=$(echo "$cfg" | jq -r '.core')
  if [ "${core#build:}" != "$core" ] && [ ! -f $D/builds/${core#build:}/PASS ]; then cfg=$BASE; fi
  start_gpu $g "$cfg"
done
tick=0
while true; do
  sleep 60; tick=$((tick+1))
  for g in $(seq 0 $((N-1))); do
    line=$(tr -d '\r' < $D/g$g.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected · up [0-9]+[hm] [0-9]+[ms]' | tail -1)
    echo "[mining g$g] $(now) $line"
    if ! kill -0 $(cat $D/g$g.pid) 2>/dev/null; then
      r=$(( $(cat $D/g$g.restarts) + 1 ))
      echo "[run] g$g miner exited $(now) (restart $r): $(tail -c 600 $D/g$g.log | tr '\r\n' '  ' | cut -c1-300)"
      if [ $r -ge 4 ]; then echo "[ctl] g$g: these settings keep exiting; back to the release default"; cat $D/g$g.cfg > $D/g$g.lastbad; start_gpu $g "$BASE"
      else cfg=$(cat $D/g$g.cfg); start_gpu $g "$cfg"; echo $r > $D/g$g.restarts; fi
    fi
  done
  [ $((tick % 5)) = 0 ] && nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv,noheader,nounits |
    while IFS=, read i sm mem pw pl t u; do echo "[gpu g$i] $(now) sm=${sm// /} mem=${mem// /} W=${pw// /}/${pl// /} C=${t// /} util=${u// /}"; done
  [ $((tick % 2)) = 0 ] || continue
  # Fetch the control file (API first: the raw CDN can lag five minutes).
  if curl -fsS -m 20 -H 'Accept: application/vnd.github.raw' "$CTL_API" -o $D/control.new 2>/dev/null || curl -fsS -m 20 "$CTL_RAW" -o $D/control.new 2>/dev/null; then
    jq -e . $D/control.new > /dev/null 2>&1 && mv $D/control.new $D/control.json || echo "[ctl] control file is not valid JSON; keeping the last one"
  fi
  for name in $(jq -r '(.builds // {}) | keys[]' $D/control.json 2>/dev/null); do
    [ -d $D/builds/$name ] || build_core $name "$(jq -r --arg n $name '.builds[$n].ref' $D/control.json)" "$(jq -r --arg n $name '.builds[$n].defines // ""' $D/control.json)"
  done
  for g in $(seq 0 $((N-1))); do
    cfg=$(want $g); [ "$cfg" = "$(cat $D/g$g.cfg)" ] && continue
    [ "$cfg" = "$(cat $D/g$g.lastbad 2>/dev/null)" ] && continue   # failed before; wait for different settings
    core=$(echo "$cfg" | jq -r '.core // "release"')
    if [ "${core#build:}" != "$core" ]; then
      name=${core#build:}
      [ -f $D/builds/$name/FAIL ] && { echo "[ctl] g$g: build $name failed; not switching"; echo "$cfg" > $D/g$g.lastbad; continue; }
      [ -f $D/builds/$name/pearl_core.node ] || continue    # still building
      stop_gpu $g
      if [ "$(verified $g $name)" != PASS ]; then
        echo "[ctl] g$g: build $name failed the hit check; staying on $(cat $D/g$g.cfg)"; echo "$cfg" > $D/g$g.lastbad
        start_gpu $g "$(cat $D/g$g.cfg)"; continue
      fi
    else
      stop_gpu $g
    fi
    start_gpu $g "$cfg"
  done
done
