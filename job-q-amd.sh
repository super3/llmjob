# q-amd job 4: is this PC slow, or our path? rate.js twice and verify-core on the card, the PC's clocks, then 3 min of the CLI from source
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) q $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
sayf() { local tag=$1 l; while IFS= read -r l; do [ -n "${l//[[:space:]]/}" ] && say "$tag | ${l:0:380}"; done; }
QZ=qzk7L38XY4MBHgjvqiudDvVR7gv5asYpZiTKLWJwZjSbzLab6
Q=$R/earn/native/quantus
say "job start $(target), $(nproc) cpus, load $(cut -d' ' -f1-3 /proc/loadavg)"
ps -eo pid,pcpu,etime,args --sort=-pcpu | head -n 6 | cut -c1-200 | sayf ps
for i in 1 2; do say "rate $i $(timeout 300 node $Q/amd/test/rate.js $Q/build/quantus_hip.node --secs 30 --warm 5 2>&1 | tail -n 1 | cut -c1-200)"; done
(cd $R/earn && timeout 1200 node native/quantus/verify-core.js --addon native/quantus/build/quantus_hip.node) 2>&1 | tail -n 3 | sayf verify
s=/opt/rocm-wsl/bin/amd-smi
[ -x $s ] && timeout 20 $s metric 2>&1 | grep -iE 'gfx_0|clk|power|edge|hotspot|usage' | tr -s ' ' | head -n 12 | sayf smi
cd $R/earn && NODE_PATH=$NM/node_modules timeout -s INT 180 node src/cli/earn-cli.js --coin quantus --quantus-address $QZ --mode mining --region us -w g19q-amd-cli4 --no-report --no-serve --no-update > /root/cli4.log 2>&1
grep -E 'H/s ·' /root/cli4.log | tail -n 3 | sayf cli
grep -c 'QTC share accepted' /root/cli4.log | sayf cli
say "job done"
