#!/bin/bash
# goal-rdna4t: a newer AMD LLVM (TheRock nightly amdrocm-llvm, which has gfx12 expert scheduling mode) beside the
# image's ROCm 7.2, and /root/r2: a ROCm root that is /opt/rocm with lib/llvm and llvm swapped for it. A build with
# ROCM_PATH=/root/r2 compiles with the new clang and links the same runtime.
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
P=${NLPOOL:-https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260822-32539019050/pool/main}
V=${NLVER:-10.1_10.1.0~20260822-32539019050}
if [ ! -f /root/nl/.ok ]; then
  rm -rf /root/nl && mkdir -p /root/nl
  for p in amdrocm-llvm amdrocm-sysdeps; do
    curl -fsSL --retry 3 -o /root/nl.deb "$P/${p}${V}_amd64.deb" || { say "newllvm FAIL download $p"; exit 1; }
    dpkg -x /root/nl.deb /root/nl || { say "newllvm FAIL dpkg -x $p"; exit 1; }
  done
  rm -f /root/nl.deb; touch /root/nl/.ok
fi
cl=$(find /root/nl -type f -name clang++ -o -type l -name clang++ | head -n 1)
[ -n "$cl" ] || cl=$(find /root/nl -name 'clang-[0-9]*' -type f | head -n 1)
say "newllvm clang: ${cl:-none}"
[ -n "$cl" ] || exit 1
lb=$(dirname "$(dirname "$cl")")   # the llvm root (bin/..)
SD=$(dirname "$(find /root/nl -name 'librocm_sysdeps_z.so*' | head -n 1)")
export LD_LIBRARY_PATH="$SD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
say "newllvm sysdeps dir $SD; LD_LIBRARY_PATH ${LD_LIBRARY_PATH:-unset}"
say "newllvm root $lb: $("$lb/bin/clang" --version 2>&1 | head -n 1)"
ldd "$lb/bin/clang" 2>&1 | grep -i "not found" | head -n 3 | while IFS= read -r l; do say "newllvm missing | $l"; done
say "newllvm expert option: $("$lb/bin/llc" -march=amdgcn -mcpu=gfx1201 --help-hidden 2>&1 | grep -iE 'expert' | tr -s ' ' | head -n 2 | tr '\n' '|')"
rm -rf /root/r2 && mkdir -p /root/r2/lib
for e in /opt/rocm/*; do b=$(basename "$e"); [ "$b" = lib ] || [ "$b" = llvm ] || ln -s "$(readlink -f "$e")" "/root/r2/$b"; done
for e in /opt/rocm/lib/*; do b=$(basename "$e"); [ "$b" = llvm ] || ln -s "$(readlink -f "$e")" "/root/r2/lib/$b"; done
# r2/lib/llvm: the new llvm, but clang and clang++ are wrappers: with -DRDNA4T_OLDCC in the arguments they run the
# image's ROCm 7.2 clang, so one sweep can build with both compilers. The new one gets its sysdeps libraries.
mkdir -p /root/r2/lib/llvm/bin
for e in "$lb"/*; do b=$(basename "$e"); [ "$b" = bin ] || ln -s "$e" "/root/r2/lib/llvm/$b"; done
for e in "$lb"/bin/*; do b=$(basename "$e"); case "$b" in clang|clang++) ;; *) ln -s "$e" "/root/r2/lib/llvm/bin/$b" ;; esac; done
for b in clang clang++; do
  cat > /root/r2/lib/llvm/bin/$b <<EOW
#!/bin/bash
case " \$* " in *" -DRDNA4T_OLDCC "*) exec /opt/rocm/lib/llvm/bin/$b "\$@" ;; esac
LD_LIBRARY_PATH="$SD\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" exec "$lb/bin/$b" "\$@"
EOW
  chmod +x /root/r2/lib/llvm/bin/$b
done
ln -s /root/r2/lib/llvm /root/r2/llvm
# device libraries: the new clang looks under --rocm-path/amdgcn/bitcode (the image's) unless its own resource dir has them
say "newllvm r2: $(ls /root/r2 | tr '\n' ' '); bitcode: $(ls /root/r2/amdgcn/bitcode 2>/dev/null | head -n 3 | tr '\n' ' ')"
# a smoke test: the clock probe source, built with the new clang against r2
if /root/r2/lib/llvm/bin/clang++ -x hip --offload-arch=${ARCHX:-gfx1201} --rocm-path=/root/r2 -std=c++17 -O3 \
     /root/clkprobe.cpp -o /root/clk2 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /root/nl-smoke.log 2>&1; then
  say "newllvm smoke: $(timeout 60 /root/clk2 1 1 2>&1 | head -n 1); wrapper old: $(/root/r2/lib/llvm/bin/clang++ -DRDNA4T_OLDCC --version | head -n 1); new: $(/root/r2/lib/llvm/bin/clang++ --version | head -n 1)"
else
  say "newllvm smoke FAIL: $(grep -m3 -iE 'error|fatal' /root/nl-smoke.log | tr '\n' '|' | cut -c1-300)"
fi
