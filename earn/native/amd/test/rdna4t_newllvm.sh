#!/bin/bash
# goal-rdna4t: a newer AMD LLVM (TheRock nightly amdrocm-llvm, which has gfx12 expert scheduling mode) beside the
# image's ROCm 7.2, and /root/r2: a ROCm root that is /opt/rocm with lib/llvm and llvm swapped for it. A build with
# ROCM_PATH=/root/r2 compiles with the new clang and links the same runtime.
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
U=${NLURL:-https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260822-32539019050/pool/main/amdrocm-llvm10.1_10.1.0~20260822-32539019050_amd64.deb}
if [ ! -d /root/nl ]; then
  curl -fsSL --retry 3 -o /root/nl.deb "$U" || { say "newllvm FAIL download"; exit 1; }
  mkdir -p /root/nl && dpkg -x /root/nl.deb /root/nl || { say "newllvm FAIL dpkg -x"; exit 1; }
  rm -f /root/nl.deb
fi
cl=$(find /root/nl -type f -name clang++ -o -type l -name clang++ | head -n 1)
[ -n "$cl" ] || cl=$(find /root/nl -name 'clang-[0-9]*' -type f | head -n 1)
say "newllvm clang: ${cl:-none}"
[ -n "$cl" ] || exit 1
lb=$(dirname "$(dirname "$cl")")   # the llvm root (bin/..)
say "newllvm root $lb: $("$lb/bin/clang" --version 2>&1 | head -n 1)"
say "newllvm expert option: $("$lb/bin/llc" -march=amdgcn -mcpu=gfx1201 --help-hidden 2>&1 | grep -iE 'expert' | tr -s ' ' | head -n 2 | tr '\n' '|')"
rm -rf /root/r2 && mkdir -p /root/r2/lib
for e in /opt/rocm/*; do b=$(basename "$e"); [ "$b" = lib ] || [ "$b" = llvm ] || ln -s "$(readlink -f "$e")" "/root/r2/$b"; done
for e in /opt/rocm/lib/*; do b=$(basename "$e"); [ "$b" = llvm ] || ln -s "$(readlink -f "$e")" "/root/r2/lib/$b"; done
ln -s "$lb" /root/r2/lib/llvm
ln -s "$lb" /root/r2/llvm
# device libraries: the new clang looks under --rocm-path/amdgcn/bitcode (the image's) unless its own resource dir has them
say "newllvm r2: $(ls /root/r2 | tr '\n' ' '); bitcode: $(ls /root/r2/amdgcn/bitcode 2>/dev/null | head -n 3 | tr '\n' ' ')"
# a smoke test: the clock probe source, built with the new clang against r2
if /root/r2/lib/llvm/bin/clang++ -x hip --offload-arch=${ARCHX:-gfx1201} --rocm-path=/root/r2 -std=c++17 -O3 \
     /root/clkprobe.cpp -o /root/clk2 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/nl-smoke.log 2>&1; then
  say "newllvm smoke: $(timeout 60 /root/clk2 1 1 2>&1 | head -n 1)"
else
  say "newllvm smoke FAIL: $(grep -m3 -iE 'error|fatal' /root/nl-smoke.log | tr '\n' '|' | cut -c1-300)"
fi
