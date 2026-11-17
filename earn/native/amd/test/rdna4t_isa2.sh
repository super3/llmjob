#!/bin/bash
# goal-rdna4t: the wmma12 fold's device assembly from each compiler in /root/r2 (rdna4t_newllvm.sh), for flag sets
# given as NAME=FLAGS;..., run from earn/native/amd. Counts over the kernel, then over the hot loop, and the loop as a
# compressed sequence (SEQ=1).
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
A=$(pwd); R=$(cd ../../.. && pwd)
ARCHX=${ARCHX:-gfx1201}
IFS=';' read -ra VS <<< "${ISAV:-obase=-DRDNA4T_OLDCC;nbase=;nesm=-mllvm -amdgpu-expert-scheduling-mode=1}"
for v in "${VS[@]}"; do
  n=${v%%=*}; fl=${v#*=}
  /root/r2/lib/llvm/bin/clang++ -x hip --offload-arch="$ARCHX" --rocm-path=/root/r2 -std=c++17 -O3 $fl \
    -I$A/compat/gpu -I$A/src -I$R/earn/native/src -I$A/build/gen --cuda-device-only -S \
    -c "$A/src/pearl_host_hip.cpp" -o /root/isa2-$n.s > /root/isa2-err.log 2>&1 \
    || { say "isa2 $n: compile failed: $(grep -m2 error /root/isa2-err.log | tr '\n' '|' | cut -c1-250)"; continue; }
  awk '/^pearl_amd_fold_wmma12:/ { on = 1; next } on && /^\.Lfunc_end/ { on = 0 } on' /root/isa2-$n.s > /root/isa2-$n.k
  say "isa2 $n kernel: $(grep -cE '^\s+[vsdbg][a-z0-9_]+' /root/isa2-$n.k) instructions; s_setreg $(grep -c s_setreg /root/isa2-$n.k), s_wait_alu $(grep -c s_wait_alu /root/isa2-$n.k), depctr $(grep -cE 'depctr|s_waitcnt_depctr' /root/isa2-$n.k), s_delay_alu $(grep -c s_delay_alu /root/isa2-$n.k)"
  awk '
    { line[NR] = $0; op[NR] = $1
      if ($0 ~ /^\.LBB[0-9_]+:/) { l = $0; sub(/:.*/, "", l); lab[l] = NR }
      if ($1 ~ /^s_cbranch|^s_branch/) tgt[NR] = $2 }
    END {
      best = 0; bs = 0; be = 0
      for (i in tgt) { t = tgt[i]; if ((t in lab) && lab[t] < i + 0) { s = 0; for (j = lab[t]; j <= i + 0; j++) if (op[j] ~ /^v_wmma/) s++
          if (s > best || (s == best && i - lab[t] > be - bs)) { best = s; bs = lab[t]; be = i + 0 } } }
      for (j = bs; j <= be; j++) print line[j]
    }' /root/isa2-$n.k > /root/isa2-$n.loop
  say "isa2 $n loop: $(grep -cE '^\s+[vsdbg][a-z0-9_]+' /root/isa2-$n.loop) instructions, $(grep -c v_wmma /root/isa2-$n.loop) v_wmma; $(awk '$1 ~ /^[vsdbg][a-z0-9_]+$/ { c[$1]++ } END { for (o in c) print c[o], o }' /root/isa2-$n.loop | sort -rn | head -n 22 | awk '{ printf "%s %s, ", $2, $1 }')"
  grep -E 's_wait_alu|depctr' /root/isa2-$n.loop | tr -s ' \t' ' ' | sort | uniq -c | sort -rn | head -n 6 | while IFS= read -r l; do say "isa2 $n waits | $l"; done
  if [ -n "${SEQ:-}" ]; then
    seq=$(awk '{ o = $1; if (o ~ /^v_wmma/) o = "W"; if (o ~ /^(;|\.)/ && o !~ /^\.LBB/) next; if (o == "s_wait_alu") o = "WA:" $2; if (o != "") print o }' /root/isa2-$n.loop | uniq -c | awk '{ printf "%s%s ", ($1 > 1 ? $1 "x" : ""), $2 }')
    for ((p = 0; p < ${#seq} && p < 6000; p += 300)); do say "isa2seq $n | ${seq:$p:300}"; sleep 0.3; done
  fi
done
