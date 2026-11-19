# job 3: a speed gate (slow PCs stop), then the v0.7.5 baseline against SRBMiner 3.8.9 (Kryptex shown, stand-in credit), then launch-setting rounds
# Run by the g18-rx9060 box agent from earn/native/amd after `build`. Removed from the branch when the agent is done.
PARTS=${PARTS:-"gate kry cred sweep"}; has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
AMDTEST_STAGE=lib source /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) j $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
A=/opt/llmjob/earn/native/amd; cd $A
WORKER=g18rx9060
ARCHQ=$(target); [ -n "$ARCHQ" ] || { say "FAIL no AMD GPU on this PC"; exit 1; }
SRB_VER=3.8.9; SRB_SHA256=186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1
if [ ! -x /opt/srb/SRBMiner-MULTI ] || [ "$(cat /opt/srb/ver 2>/dev/null)" != $SRB_VER ]; then
  rm -rf /opt/srb; mkdir -p /opt/srb && cd /opt/srb && curl -fsSL --retry 3 -o srb.tgz \
    "https://github.com/doktor83/SRBMiner-Multi/releases/download/$SRB_VER/SRBMiner-Multi-${SRB_VER//./-}-Linux.tar.gz" \
    && echo "$SRB_SHA256  srb.tgz" | sha256sum -c --quiet && tar xzf srb.tgz \
    && cp "$(ls -d /opt/srb/SRBMiner-Multi-*/ | head -1)SRBMiner-MULTI" /opt/srb/ && echo $SRB_VER > /opt/srb/ver \
    || { say "FAIL srb $SRB_VER"; exit 1; }
  cd $A
fi
if has gate; then
  out=$(timeout 300 node ../probes/hashrate.js build/pearl_core_hip.node 40 15 2>&1 | grep -v nvidia-smi | tail -n 1)
  th=$(echo "$out" | grep -oE '"th":[0-9.]+' | cut -d: -f2); say "gate th=$th $(echo "$out" | grep -oE '"k":[0-9]+')"
  awk -v t="${th:-0}" -v m="${GATE_MIN:-72}" 'BEGIN{exit !(t < m)}' && { say "slow PC: gate under ${GATE_MIN:-72}, stopping"; exit 3; }
fi
if has kry; then
# Kryptex, shown rate: a warm-up, then ours srb srb ours, 300 s each.
say "kry pool $DUEL_POOL"
run_miner ours 60 0
for o in ours srb srb ours; do run_miner $o 300 1; done
fi
if has cred; then
# Stand-in pool (verify-pool.js) at 2^CRED_BITS: every share checked as the verifier does; credit counted after the
# first minute of each turn up to the end of sampling.
CRED_BITS=${CRED_BITS:-210}
rm -f /root/cred.log
( PEARL_SHARE_LOG=/root/cred.log PEARL_JOB_EVERY=60 exec node ../probes/verify-pool.js serve 3334 4000 $CRED_BITS ) > /root/vp.log 2> /root/vp.err &
vp=$!; sleep 3
DUEL_POOL=127.0.0.1:3334
say "cred stand-in 2^$CRED_BITS"
for o in srb ours ours srb; do
  t0=$(( $(date +%s%3N) + 60000 )); run_miner $o 540 1; t1=$(( $(date +%s%3N) - 8000 ))
  ok=$(awk -v a=$t0 -v b=$t1 '$1>=a && $1<=b && $4=="ok"' /root/cred.log | wc -l)
  bad=$(awk -v a=$t0 -v b=$t1 '$1>=a && $1<=b && $4!="ok"' /root/cred.log | wc -l)
  say "cred $o window_s=$(( (t1 - t0) / 1000 )) ok=$ok bad=$bad credited_th=$(awk -v n=$ok -v ms=$((t1 - t0)) -v b=$CRED_BITS 'BEGIN{printf "%.2f", n * 2^(256-b) / (ms/1000) / 1e12}')"
done
kill -TERM $vp; wait $vp; tail -n 2 /root/vp.log | emit cred; tail -n 3 /root/vp.err | emit cred
fi
if has sweep; then
BUILDS=${BUILDS:-"twin=-DPEARL_AMD_GOALTWIN=1"}
VARS=${VARS:-"base=base;twin=twin;cb64=base PEARL_AMD_COL_BATCH=64;k4096=base PEARL_AMD_K=4096;k4096cb128=base PEARL_AMD_K=4096 PEARL_AMD_COL_BATCH=128;job30=base PEARL_JOB_EVERY=30;k4096job30=base PEARL_AMD_K=4096 PEARL_JOB_EVERY=30"}
ROUNDS=${ROUNDS:-4}; HSECS=${HSECS:-50}; HWARM=${HWARM:-15}
mkdir -p /root/var/base && cp build/pearl_core_hip.node /root/var/base/
IFS=';' read -ra BL <<< "$BUILDS"
for b in "${BL[@]}"; do
  [ -n "$b" ] || continue
  n=${b%%=*}; f=${b#*=}
  if PEARL_AMD_CFLAGS="$f" ./build-amd.sh gpu "$ARCHQ" > /root/build-$n.log 2>&1; then
    mkdir -p /root/var/$n && cp build/pearl_core_hip.node /root/var/$n/
    say "built $n ($f) md5 $(md5sum build/pearl_core_hip.node | cut -c1-8) $(grep -oE 'vgpr[^,;]*|spill[^,;]*' /root/build-$n.log | sort -u | tr '\n' ' ' | cut -c1-160)"
  else
    say "build $n FAIL: $(grep -m3 -E 'error' /root/build-$n.log | tr '\n' ' ' | cut -c1-300)"
  fi
done
cp /root/var/base/pearl_core_hip.node build/
IFS=';' read -ra VL <<< "$VARS"
nv=${#VL[@]}
for r in $(seq 1 $ROUNDS); do
  for i in $(seq 0 $((nv - 1))); do
    v=${VL[$(( (i + r - 1) % nv ))]}; name=${v%%=*}; read -ra W <<< "${v#*=}"
    core=/root/var/${W[0]}/pearl_core_hip.node; [ -f $core ] || continue
    out=$(cd $A && env PEARL_AMD_UNTESTED=1 "${W[@]:1}" timeout 600 node ../probes/hashrate.js $core $HSECS $HWARM 2>&1 | grep -v nvidia-smi | tail -n 1)
    say "hash r$r $name $(echo "$out" | grep -oE '"k":[0-9]+|"th":[0-9.]+|"sd":[0-9.]+|"jobs":[0-9]+' | tr '\n' ' ')"
  done
done
fi
say "job done"
