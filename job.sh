# restore /opt/llmjob at 0490e409 (a bad G_CODE checked out a job-only commit), then rebuild the default core
say() { local l="[amdtest] $(date -u +%T) g $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
cd /opt/llmjob && git fetch -q --depth 1 origin 0490e409410bb15ef9333cc11c06bbefe52fd971 && git checkout -q -f FETCH_HEAD
say "restore: HEAD $(git -C /opt/llmjob rev-parse --short HEAD), $(ls /opt/llmjob/earn/native/amd | tr '\n' ' ')"
cd /opt/llmjob/earn/native/amd || exit 1
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say "restore: target $(target), R=$R A=$A"
./build-amd.sh gpu "$(target)" > /root/rb.log 2>&1; say "restore: build rc=$? md5 $(md5sum build/pearl_core_hip.node | cut -c1-8)"
