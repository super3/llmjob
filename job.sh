# l4e: why the QG builds fail (build-amd.sh gpu errors)
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
cd /opt/llmjob/earn/native/amd || exit 1
for f in /root/sw*/sweep-build-qg.log /opt/llmjob/earn/native/amd/build/sweep*/sweep-build-qg.log; do [ -f "$f" ] && say "log $f"; done
L=$(ls -t $(find / -name 'sweep-build-qg.log' 2>/dev/null) 2>/dev/null | head -1)
say "file $L"
grep -n -m8 -iE "error|fail" "$L" | while IFS= read -r l; do say "e | ${l:0:280}"; done
tail -n 4 "$L" | while IFS= read -r l; do say "t | ${l:0:280}"; done
