#!/bin/bash
# goal-rdna4t: install hipBLASLt from ROCm's apt repository, time its int8 GEMM on this card beside the clock probe,
# and print the main loop of the kernel it picks, from the shipped code object. Run from earn/native/amd.
#   SHAPES  M:N:K list (default 8192:8192:16384 16384:8192:16384)
#   SECS    seconds per algorithm (default 2)
#   LOOPLINES  loop instructions to print (default 0: the compressed sequence only)
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
SHAPES=${SHAPES:-8192:8192:16384 16384:8192:16384}
SECS=${SECS:-2}
ARCHX=${ARCHX:-$(timeout 60 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]{3,4}' | grep -v '^gfx000' | head -1)}
L=/opt/rocm/lib/llvm/bin
if [ ! -f /opt/rocm/include/hipblaslt/hipblaslt.h ]; then
  t0=$(date +%s)
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends hipblaslt-dev; } \
    > /root/blaslt-apt.log 2>&1 || { say "blaslt apt FAIL: $(tail -n 3 /root/blaslt-apt.log | tr '\n' '|' | cut -c1-300)"; exit 1; }
  say "blaslt installed in $(( $(date +%s) - t0 ))s: $(dpkg -s hipblaslt 2>/dev/null | grep -m1 Version)"
fi
B=/root/rdna4t_blaslt
if $L/clang++ -x hip --offload-arch=$ARCHX --rocm-path=/opt/rocm -std=c++17 -O2 -DRDNA4T_KNAME test/rdna4t_blaslt.cpp \
     -o $B -L/opt/rocm/lib -lhipblaslt -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/blaslt-build.log 2>&1 \
   || $L/clang++ -x hip --offload-arch=$ARCHX --rocm-path=/opt/rocm -std=c++17 -O2 test/rdna4t_blaslt.cpp \
     -o $B -L/opt/rocm/lib -lhipblaslt -lamdhip64 -Wl,-rpath,/opt/rocm/lib >> /root/blaslt-build.log 2>&1; then
  say "blaslt built"
else
  say "blaslt build FAIL: $(grep -m3 error /root/blaslt-build.log | tr '\n' '|' | cut -c1-300)"; exit 1
fi
CLK=""
[ -x /root/clkprobe ] && { /root/clkprobe 3 100000 > /root/blaslt-clk.log 2>&1 & CLK=$!; }
timeout 1800 $B $SECS $SHAPES > /root/blaslt.log 2>&1
[ -n "$CLK" ] && kill $CLK 2>/dev/null
grep -E '^blaslt' /root/blaslt.log | sed -E 's/(Cijk_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9_]{0,60}).*_(MT[0-9x]+_MI[0-9x]+).*/\1.._\2/' \
  | while IFS= read -r l; do say "${l:0:330}"; sleep 0.2; done
say "blaslt clk: $(grep '^clk' /root/blaslt-clk.log | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END {print "median " a[int((NR+1)/2)] " MHz, n " NR}')"
# The best kernel's name (full), for the disassembly.
best=$(grep -E '^blaslt best' /root/blaslt.log | tail -n 1 | awk '{print $4}')
kn=$(grep -E "^blaslt algo $best " /root/blaslt.log | tail -n 1 | awk '{print $NF}')
for sh in $SHAPES; do
  b=$(grep -E "^blaslt best $sh " /root/blaslt.log | awk '{print $4}')
  k=$(grep -A40 "^blaslt shape $sh " /root/blaslt.log | grep -m1 -E "^blaslt algo $b " | awk '{print $NF}')
  [ -n "$k" ] && kn=$k
done
say "blaslt kernel: ${kn:0:300}"
[ -n "$kn" ] && [ "$kn" != "?" ] || exit 0
f=$(grep -l -a -F "$kn" /opt/rocm/lib/hipblaslt/library/*$ARCHX* 2>/dev/null | head -n 1)
say "blaslt code object: ${f:-none}"
[ -n "$f" ] || exit 0
$L/llvm-objdump -d --mcpu=$ARCHX "$f" > /root/blaslt-dis.txt 2>/root/blaslt-dis.err
awk -v k="$kn" 'index($0, "<" k ">:") { on = 1; next } on && /^[0-9a-f]+ <.*>:/ { on = 0 } on' /root/blaslt-dis.txt > /root/blaslt-k.txt
say "blaslt kernel disassembly: $(wc -l < /root/blaslt-k.txt) lines; ops: $(awk '$1 ~ /^[vsdbg][a-z0-9_]+$/ { n[$1]++ } END { for (o in n) print n[o], o }' /root/blaslt-k.txt | sort -rn | head -n 30 | awk '{ printf "%s %s, ", $2, $1 }')"
# The loop: the backward branch span with the most v_wmma (addresses from the "// 0000..: " comments).
awk '
  function hex(h,   i, c, v) { v = 0; h = toupper(h); for (i = 1; i <= length(h); i++) { c = index("0123456789ABCDEF", substr(h, i, 1)) - 1; v = v * 16 + c }; return v }
  { line[NR] = $0; op[NR] = $1
    if (match($0, /\/\/ [0-9A-Fa-f]+:/)) { a = hex(substr($0, RSTART + 3, RLENGTH - 4)); addr[NR] = a; at[a] = NR }
    if ($1 ~ /^s_cbranch|^s_branch/ && $2 ~ /^-?[0-9]+$/) { v = $2 + 0; if (v > 32767) v -= 65536; br[NR] = v } }
  END {
    best = 0; bs = 0; be = 0
    for (i in br) { if (!(i in addr)) continue; ta = addr[i] + 4 + br[i] * 4
      if (ta in at) { s0 = at[ta]; if (s0 < i + 0) { s = 0; for (j = s0; j <= i + 0; j++) if (op[j] ~ /^v_wmma/) s++
        if (s > best) { best = s; bs = s0; be = i + 0 } } } }
    for (j = bs; j <= be; j++) print line[j]
  }' /root/blaslt-k.txt > /root/blaslt-loop.txt
say "blaslt loop: $(grep -c . /root/blaslt-loop.txt) lines, $(grep -c v_wmma /root/blaslt-loop.txt) v_wmma; ops: $(awk '$1 ~ /^[vsdbg][a-z0-9_]+$/ { n[$1]++ } END { for (o in n) print n[o], o }' /root/blaslt-loop.txt | sort -rn | head -n 30 | awk '{ printf "%s %s, ", $2, $1 }')"
seq=$(awk '{ o = $1; if (o ~ /^v_wmma/) o = "W"; if (o !~ /^[vsdbg][a-z0-9_]+$/ && o != "W") next; print o }' /root/blaslt-loop.txt | uniq -c | awk '{ printf "%s%s ", ($1 > 1 ? $1 "x" : ""), $2 }')
for ((p = 0; p < ${#seq}; p += 300)); do say "blaslt seq | ${seq:$p:300}"; sleep 0.3; done
if [ "${LOOPLINES:-0}" -gt 0 ]; then
  sed -E 's@ *// [0-9A-F]+:.*@@' /root/blaslt-loop.txt | grep -E '^\s+[a-z]' | tr -s ' \t' ' ' | head -n $LOOPLINES \
    | while IFS= read -r l; do say "blaslt isa |${l:0:200}"; sleep 0.25; done
fi
