# job 2: hashrate.js rounds on the shipped core (main) with launch settings the 9060 XT has not raced since FUSE and GB
# Run by the g18-rx9060 box agent from earn/native/amd after `build`. Removed from the branch when the agent is done.
# Variants: "name=build K=V ..." where build is base (build/) or a flag build made here. Rounds rotate the order.
AMDTEST_STAGE=lib source /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) j $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
A=/opt/llmjob/earn/native/amd; cd $A
ARCHQ=$(target); [ -n "$ARCHQ" ] || { say "FAIL no AMD GPU on this PC"; exit 1; }
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
say "job done"
