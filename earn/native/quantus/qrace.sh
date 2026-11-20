#!/bin/bash
# qrace.sh: our Quantus core against SRBMiner on one Salad PC (NVIDIA or AMD), taking turns on the same card, both
# mining to WALLET on a real pool. A box's container command fetches this file at SCRIPT_SHA (default BASE_SHA) and
# runs it, as qbox.sh's boxes do.
#
#   1. the card: name, driver, compute capability or gfx target, power limit
#   2. setup: apt packages, Node 22 (checked by SHA-256), tweetnacl for the CLI (and node-addon-api on AMD), on
#      NVIDIA CUDA 12.8.1's compiler from NVIDIA's archives, pinned as in qbox.sh; SRBMiner SRB_VER (checked by MD5)
#   3. the repo at BASE_SHA, and this card's core: cuda/build.sh <cc> or amd/build.sh gpu <gfx>; on an RTX 30 card first
#      cuda/sm86/build.py, as CI runs it, so the core embeds the patched kernel (SM86=0 leaves it out)
#   4. verify: verify-core.js on card 0 (the five checks). If it fails, ours is not raced; SRBMiner still is.
#   5. race: the legs in LEGS (default "ours srb srb ours"). Each runs WARM seconds unmeasured, then SECS seconds in
#      which the miner's displayed rate and share counts are read every 10 s and the card every 5 s (NVIDIA). Ours is
#      the app's CLI (--coin quantus) with the core just built; SRBMiner is its own binary. Both log in to POOL (or
#      POOL2 when POOL's port can't be reached) as WALLET, workers WORKER-ours and WORKER-srb.
#      Each leg ends with
#        leg <name> result mean=<MH/s> n=<samples> acc=<a> rej=<r> clk=<MHz> w=<W> t=<C>
#      acc and rej count the whole leg, warm-up included.
#
# Then it sleeps, so the logs can be read. Every line is "[nvtest] HH:MM:SS <text>" ("[amdtest]" on AMD), at most
# about 400 characters. Nothing here prints environment variables.
#
# Environment: BASE_SHA (required), WALLET (required), WORKER (default qrace), POOL (us.quantus.herominers.com:1155),
# POOL2 (qtc-us.kryptex.network:7049), LEGS, WARM (30), SECS (90), STAGES (default "verify race"), SRB_VER (3.8.9),
# SRB_MD5 (that release's own .tar.md5), DEFS (extra nvcc -D flags), QH_CFLAGS (extra AMD flags).
set -u
export DEBIAN_FRONTEND=noninteractive
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then V=nv; TAGN=nvtest; else V=amd; TAGN=amdtest; export PATH="$PATH:/opt/rocm/bin"; fi
R=/opt/llmjob
Q=$R/earn/native/quantus
CUDA=/opt/cuda
NODE=/opt/node
NM=/opt/nm
SRBD=/opt/srb
RD=https://developer.download.nvidia.com/compute/cuda/redist
NVCC_VER=V12.8.93
CUDA_PARTS="cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz|8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae"
NODE_TGZ=node-v22.23.2-linux-x64.tar.xz
NODE_SHA=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
SRB_VER=${SRB_VER:-3.8.9}; SRB_MD5=${SRB_MD5:-ced48cbc4ff4a70b7da78835f9aaf0ae}
POOL=${POOL:-us.quantus.herominers.com:1155}
POOL2=${POOL2:-qtc-us.kryptex.network:7049}
WORKER=${WORKER:-qrace}
WARM=${WARM:-30}
SECS=${SECS:-90}
LEGS=${LEGS:-ours srb srb ours}
VERIFIED=0

say() { echo "[$TAGN] $(date -u +%T) $*"; }
emit() {
  local tag=$1 line
  tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | while IFS= read -r line || [ -n "$line" ]; do
    [ -z "${line//[[:space:]]/}" ] && continue
    line=$(printf '%s' "$line" | LC_ALL=C tr -cd '\11\40-\176')
    while [ ${#line} -gt 380 ]; do say "$tag | ${line:0:380}"; line="+${line:380}"; done
    say "$tag | $line"
  done
}
idle() { say "idle $*"; sleep infinity; }
fail() { say "FAIL $*"; idle "after a failure"; }
APT="apt-get -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::Retries=3"
smi() { nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
target() { timeout 60 rocminfo 2> /dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1; }

card() {
  if [ $V = nv ]; then
    say "card $(nvidia-smi -i 0 --query-gpu=name,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv,noheader 2>&1 | head -1)"
    say "card count $(nvidia-smi -L 2> /dev/null | grep -c '^GPU')"
  else
    timeout 60 rocminfo 2>&1 | awk -F': +' '/Marketing Name:/{m=$2} /^  Name:/{n=$2} /Compute Unit:/{cu=$2}
      /Max Clock Freq/{mhz=$2} /Device Type:/{if ($2 ~ /GPU/) g=1}
      /ISA Info:/{if (g) {print "gpu " n ", " m ", CUs " cu ", " mhz " MHz"; g=0}}' | sed 's/  */ /g' | emit card
    say "card target $(target)"
  fi
}

setup() {
  local t0; t0=$(date +%s)
  # SRBMiner needs wget: without it, it says its dev-fee connection is blocked and exits.
  local pkgs="git curl wget ca-certificates xz-utils procps"
  [ $V = nv ] && pkgs="$pkgs g++ make python3"
  { timeout 300 $APT update -qq && timeout 600 $APT install -y -qq --no-install-recommends $pkgs; } > /root/apt.log 2>&1 \
    || { tail -5 /root/apt.log | emit setup; fail "apt-get install"; }
  if [ ! -x "$NODE/bin/node" ]; then
    curl -fsSL --retry 3 -m 300 "https://nodejs.org/dist/v22.23.2/$NODE_TGZ" -o /root/node.txz || fail "node download"
    echo "$NODE_SHA  /root/node.txz" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "node checksum"
    mkdir -p "$NODE" && tar -xJf /root/node.txz -C "$NODE" --strip-components=1 || fail "node unpack"
  fi
  export PATH="$NODE/bin:$PATH"
  local mods="tweetnacl tweetnacl-util"
  [ $V = amd ] && mods="$mods node-addon-api"
  npm install --prefix $NM --no-audit --no-fund --loglevel=error $mods > /root/npm.log 2>&1 \
    || { tail -5 /root/npm.log | emit setup; fail "npm install"; }
  if [ $V = nv ] && { [ "$("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE 'V[0-9.]+' | tail -1)" != "$NVCC_VER" ] \
                      || [ ! -x "$CUDA/bin/nvdisasm" ]; }; then
    rm -rf "$CUDA" && mkdir -p "$CUDA" /root/dl || fail "no $CUDA"
    while IFS='|' read -r rel sha; do
      f=/root/dl/${rel##*/}
      curl -fsSL --retry 3 -m 600 "$RD/$rel" -o "$f" || fail "cuda download ${rel##*/}"
      echo "$sha  $f" | sha256sum -c --quiet - > /dev/null 2>&1 || fail "cuda checksum ${rel##*/}"
      tar -xJf "$f" -C "$CUDA" --strip-components=1 || fail "cuda unpack ${rel##*/}"
      rm -f "$f"
    done <<< "$CUDA_PARTS"
    ln -sfn lib "$CUDA/lib64"
  fi
  if [ ! -x $SRBD/SRBMiner-MULTI ]; then
    mkdir -p $SRBD && cd $SRBD || fail "no $SRBD"
    curl -fsSL --retry 3 -m 300 -o srb.tgz \
      "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
      || fail "srb download"
    echo "$SRB_MD5  srb.tgz" | md5sum -c --quiet - > /dev/null 2>&1 || fail "srb checksum"
    tar xzf srb.tgz && cp "$(ls -d $SRBD/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI" $SRBD/ || fail "srb unpack"
  fi
  say "setup ok in $(( $(date +%s) - t0 ))s: node $(node --version)$([ $V = nv ] && echo ", nvcc $("$CUDA/bin/nvcc" --version | grep -oE 'V[0-9.]+' | tail -1)"), SRBMiner $SRB_VER"
}

fetch() {
  rm -rf "$R" && mkdir -p "$R" && cd "$R" || fail "no $R"
  if ! { git init -q && git remote add origin https://github.com/super3/llmjob.git \
         && timeout 600 git fetch -q --depth 1 origin "$BASE_SHA" && git checkout -q FETCH_HEAD; } > /root/git.log 2>&1; then
    tail -3 /root/git.log | emit fetch
    fail "fetch $BASE_SHA"
  fi
  say "fetch ok $(git rev-parse --short=12 HEAD)"
}

build() {
  local t0 cc gfx; t0=$(date +%s)
  if [ $V = nv ]; then
    cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
    [ -n "$cc" ] || fail "no compute capability"
    ADDON=$Q/build/quantus_cuda.node
    # RTX 30: the patched kernel (cuda/sm86), as CI builds it, when this commit has one and SM86 is not 0.
    local sm86=""
    if [ "$cc" = 86 ] && [ -f $Q/cuda/sm86/build.py ] && [ "${SM86:-1}" != 0 ]; then
      python3 $Q/cuda/sm86/build.py $CUDA/bin/nvcc $CUDA/bin/nvdisasm $Q/build/sm86 > /root/sm86.log 2>&1 && sm86=$Q/build/sm86
      tail -2 /root/sm86.log | emit sm86
    fi
    QC_SM86_DIR=$sm86 OUT=$ADDON DEFS="${DEFS:-}" CUDA_HOME=$CUDA NODE_INCLUDE=$NODE/include/node bash $Q/cuda/build.sh "$cc" \
      > /root/build.log 2>&1 || { grep -m6 -iE "error" /root/build.log | emit build; tail -3 /root/build.log | emit build; fail "build"; }
    say "build ok sm_$cc in $(( $(date +%s) - t0 ))s"
  else
    gfx=$(target)
    [ -n "$gfx" ] || fail "no gfx target from rocminfo"
    ADDON=$Q/build/quantus_hip.node
    NODE_ADDON_API=$NM/node_modules/node-addon-api NODE_INCLUDE=$NODE/include/node bash $Q/amd/build.sh gpu "$gfx" \
      > /root/build.log 2>&1 || { grep -m6 -iE "error" /root/build.log | emit build; tail -3 /root/build.log | emit build; fail "build"; }
    say "build ok $gfx in $(( $(date +%s) - t0 ))s"
  fi
}

verify() {
  cd $R/earn || fail "no earn"
  QUANTUS_CUDA_LOG=1 timeout 1200 node native/quantus/verify-core.js --addon "$ADDON" --device 0 > /root/verify.log 2>&1
  local rc=$?
  emit verify < /root/verify.log
  say "verify rc=$rc"
  [ $rc -eq 0 ] && VERIFIED=1
}

# The pool both miners use: POOL, or POOL2 when POOL's port can't be reached from this PC.
pick_pool() {
  local p
  for p in "$POOL" "$POOL2"; do
    if timeout 10 bash -c "exec 3<>/dev/tcp/${p%:*}/${p##*:}" 2> /dev/null; then echo "$p"; return; fi
    say "pool $p unreachable from this PC" >&2
  done
}

# Both miners use card 0 only: some PCs have more than one card (a Salad RTX 2080 box had a 2080 SUPER and a
# 2080 Ti, and both miners mined on both). CUDA then numbers the cards in PCI order, as nvidia-smi does.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ours_cli() {
  cd $R/earn && CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 QUANTUS_CORE_PATH=$ADDON NODE_PATH=$NM/node_modules QUANTUS_CUDA_LOG=1 \
    exec node src/cli/earn-cli.js --coin quantus --quantus-address "$WALLET" --quantus-pool "$P" -w "$WORKER-ours" \
    --gpu-index 0 --mode mining --no-report --no-update --no-serve
}
srb_cli() {
  local off="--disable-gpu-amd"
  [ $V = amd ] && off="--disable-gpu-nvidia"
  cd $SRBD && CUDA_VISIBLE_DEVICES=0 exec ./SRBMiner-MULTI --algorithm quantus --gpu-id 0 --pool "$P" --wallet "$WALLET" \
    --password "$WORKER-srb" --worker "$WORKER-srb" --disable-cpu $off --disable-gpu-intel --api-enable --api-port 21550
}
# One sample as "MH/s accepted rejected": SRBMiner's from its API, ours from the CLI's last status line.
sample_srb() {
  curl -s -m 3 http://127.0.0.1:21550 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{
    const a=JSON.parse(s).algorithms[0];console.log((a.hashrate.gpu.total/1e6).toFixed(2),a.shares.accepted,a.shares.rejected)}catch(e){}})'
}
sample_ours() {
  tr '\r' '\n' < /root/leg-ours.log | grep -aoE '[0-9.]+ [kMGT]?H/s · [0-9,]+ accepted · [0-9,]+ rejected' | tail -1 \
    | awk '{u=$2; m=(u=="kH/s")?1e-3:(u=="GH/s")?1e3:(u=="TH/s")?1e6:(u=="H/s")?1e-6:1; gsub(",","",$4); gsub(",","",$7);
            printf "%.2f %s %s\n", $1*m, $4, $7}'
}

# leg NAME: WARM s unmeasured, then SECS s measured. Ends with "leg NAME result ...".
leg() {
  local name=$1 pid t0 t s mh acc rej n=0 sum=0 nc=0 c=0 w=0 tc=0 sc sw st last="? ? ?"
  say "leg $name start: ${WARM}s warm-up, ${SECS}s measured, pool $P"
  ( ${name}_cli ) > /root/leg-$name.log 2>&1 &
  pid=$!; t0=$(date +%s)
  while :; do
    sleep 5; t=$(( $(date +%s) - t0 ))
    if ! kill -0 $pid 2> /dev/null; then
      say "leg $name: the miner exited at ${t}s"; tr '\r' '\n' < /root/leg-$name.log | grep -av '^\s*$' | tail -8 | emit $name
      break
    fi
    [ $t -lt "$WARM" ] && continue
    if [ $V = nv ]; then
      s=$(smi)
      if [ -n "$s" ]; then
        IFS=, read -r sc sw st <<< "$s"
        nc=$((nc + 1)); c=$(awk -v a=$c -v b=$sc 'BEGIN{print a+b}'); w=$(awk -v a=$w -v b=$sw 'BEGIN{print a+b}')
        tc=$(awk -v a=$tc -v b=$st 'BEGIN{print a+b}')
      fi
    elif [ $(( t - WARM )) -ge 40 ] && [ $(( t - WARM )) -lt 45 ] && [ -x /opt/rocm-wsl/bin/amd-smi ]; then
      timeout 20 /opt/rocm-wsl/bin/amd-smi metric 2>&1 | grep -iE 'power|gfx_clk|clk|temp' | grep -v 'N/A' | head -12 \
        | tr -s ' ' | emit "$name smi"
    fi
    if [ $(( (t - WARM) % 10 )) -lt 5 ]; then
      s=$(sample_$name); read -r mh acc rej <<< "$s"
      if [ -n "$mh" ]; then last="$mh $acc $rej"; n=$((n + 1)); sum=$(awk -v a=$sum -v b=$mh 'BEGIN{print a+b}'); fi
      [ $(( (t - WARM) % 30 )) -lt 5 ] && say "leg $name t=${t}s rate=${mh:-?} acc=${acc:-?} rej=${rej:-?}$([ $V = nv ] && echo " gpu=$(smi)")"
    fi
    [ $t -ge $(( WARM + SECS )) ] && break
  done
  kill $pid 2> /dev/null; wait $pid 2> /dev/null
  pkill -f SRBMiner-MULTI; pkill -f 'src/cli/earn-cli\.js'; sleep 5
  read -r mh acc rej <<< "$last"
  tr '\r' '\n' < /root/leg-$name.log | grep -aiE 'reject|error|invalid|stale|disconnect|fail' | grep -av 'H/s · ' | tail -5 | emit "$name log"
  say "leg $name result mean=$([ $n -gt 0 ] && awk -v s=$sum -v n=$n 'BEGIN{printf "%.2f", s/n}' || echo '?') n=$n acc=$acc rej=$rej" \
      "clk=$([ $nc -gt 0 ] && awk -v s=$c -v n=$nc 'BEGIN{printf "%.0f", s/n}' || echo '?')" \
      "w=$([ $nc -gt 0 ] && awk -v s=$w -v n=$nc 'BEGIN{printf "%.1f", s/n}' || echo '?')" \
      "t=$([ $nc -gt 0 ] && awk -v s=$tc -v n=$nc 'BEGIN{printf "%.0f", s/n}' || echo '?')"
}

race() {
  P=$(pick_pool)
  [ -n "$P" ] || fail "no pool reachable from this PC"
  local l first=1
  for l in $LEGS; do
    if [ "$l" = ours ] && [ $VERIFIED != 1 ]; then say "leg ours skipped: verify-core did not pass"; continue; fi
    leg "$l"
    # The CLI's first lines say what the pool sent at login.
    if [ "$l" = ours ] && [ $first = 1 ]; then
      first=0; tr '\r' '\n' < /root/leg-ours.log | grep -aE '^quantus(_cuda)?:' | head -7 | emit "ours start"
    fi
  done
}

say "boot qrace sha ${BASE_SHA:-none}, vendor $V, stages ${STAGES:-verify race}, legs $LEGS"
[ -n "${BASE_SHA:-}" ] || idle "no BASE_SHA"
[ -n "${WALLET:-}" ] || idle "no WALLET"
card
setup
fetch
build
for st in ${STAGES:-verify race}; do
  say "stage $st"
  case $st in
    verify) verify ;;
    race) race ;;
    *) say "unknown stage $st" ;;
  esac
done
say "all done"
idle "until the group is deleted"
