# job 1: baseline, ours (main, v0.7.5 core) against SRBMiner 3.8.9: Kryptex shown rate, then stand-in pool credit
# Run by the g18-rx9060 box agent from earn/native/amd after `build`. Removed from the branch when the agent is done.
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
say "base $ARCHQ srb $SRB_VER, core md5 $(md5sum build/pearl_core_hip.node | cut -c1-8), tree $(git -C /opt/llmjob rev-parse --short=12 HEAD)"
for t in rocprofv3 rocprof; do say "tool $t: $(command -v $t || echo none); kfd: $(ls /dev/kfd 2>&1 | head -1)"; done
# Kryptex, shown rate: a warm-up, then ours srb srb ours, 300 s each.
say "kry pool $DUEL_POOL"
run_miner ours 60 0
for o in ours srb srb ours; do run_miner $o 300 1; done
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
say "job done"
