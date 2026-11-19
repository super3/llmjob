# q-amd job 3: the HiveOS gaps: hiveosScripts tests under real bash and jq, then a Quantus-only flight sheet (qz wallet) through the package, with h-stats.sh read while it mines
# A box job for the q-amd agent (Quantus on AMD, 19 Nov). The box agent runs it from earn/native/amd with J_SHA set
# to the commit it came from. Not part of the miner: the agent drops this file when the round ends.
# Lines go straight to the container's log as "[amdtest] HH:MM:SS q ...".
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) q $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
sayf() { local tag=$1 l; while IFS= read -r l; do [ -n "${l//[[:space:]]/}" ] && say "$tag | ${l:0:380}"; done; }
QZ=qzk7L38XY4MBHgjvqiudDvVR7gv5asYpZiTKLWJwZjSbzLab6
HIVE_SECS=${J_HIVE_SECS:-420}
Q=$R/earn/native/quantus
ARCHX=${ARCH:-$(target)}
say "job start $ARCHX, code ${J_SHA:0:12}"

if [ "$(git -C $R rev-parse HEAD)" != "$J_SHA" ]; then
  ( cd $R && git fetch -q --depth 1 origin "$J_SHA" && git checkout -q -f FETCH_HEAD ) > /root/tree.log 2>&1 \
    || { say "FAIL fetch $J_SHA"; exit 1; }
fi
say "tree $(git -C $R rev-parse --short=12 HEAD)"

if ! (cd $Q/amd && bash build.sh gpu "$ARCHX") > /root/qbuild.log 2>&1; then
  say "FAIL build"; grep -iE 'error' /root/qbuild.log | head -10 | sayf build; exit 1
fi
say "built quantus_hip.node for $ARCHX"

# The HiveOS hook scripts' tests need Linux, bash and jq, so they are skipped on the Windows dev box: run them here.
cd $R/earn || exit 1
command -v jq > /dev/null || apt-get install -y -qq --no-install-recommends jq > /root/apt-jq.log 2>&1 || { apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq --no-install-recommends jq > /root/apt-jq.log 2>&1; }
say "jq $(jq --version 2>&1)"
if ! npm ci --ignore-scripts --no-audit --no-fund --loglevel=error > /root/npmci.log 2>&1; then
  say "FAIL npm ci"; tail -n 8 /root/npmci.log | sayf tests; exit 1
fi
npx jest test/hiveosScripts.test.js test/statsFile.test.js test/hiveosPackage.test.js --coverage=false > /root/jest.log 2>&1
rc=$?
grep -E '^(Tests|Test Suites):|✕|skipped' /root/jest.log | head -n 30 | sayf tests
grep -A 12 '●' /root/jest.log | head -n 60 | sayf tests
say "tests rc=$rc"

# mine NAME SECS DIR CMD...: as in job 1, and every 30 s h-stats.sh sourced the way HiveOS's agent does.
mine() {
  local name=$1 secs=$2 dir=$3; shift 3
  local log=/root/qmine-$name.log pid t0 seen=0 n
  say "mine $name start ${secs}s"
  ( cd "$dir" && exec "$@" ) > $log 2>&1 &
  pid=$!; t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    n=$(wc -l < $log)
    tail -n +$((seen + 1)) $log | head -n $((n - seen)) | grep -v 'H/s ·' | grep -v 'QTC share accepted' | head -n 25 | sayf "$name"
    tail -n +$((seen + 1)) $log | head -n $((n - seen)) | grep 'H/s ·' | tail -n 1 | sayf "$name"
    seen=$n
    ( source /hive/miners/custom/llmjob-earn/h-stats.sh; echo "khs=$khs stats=$stats" ) 2>&1 | sayf hstats
    [ $(( $(date +%s) - t0 )) -ge "$secs" ] && break
  done
  say "stats file: $(head -c 700 /run/hive/llmjob-earn-stats.json)"
  pkill -INT -f llmjob-earn-cli-linux 2>/dev/null; kill -INT $pid 2>/dev/null; sleep 8
  pkill -f llmjob-earn-cli-linux 2>/dev/null; kill $pid 2>/dev/null; wait $pid 2>/dev/null
  tail -n +$((seen + 1)) $log | grep -v 'H/s ·' | grep -v 'QTC share accepted' | tail -n 15 | sayf "$name"
  say "mine $name result accepted=$(grep -c 'QTC share accepted' $log) rejected=$(grep -c 'QTC share rejected' $log) secs=$(( $(date +%s) - t0 ))"
}

mkdir -p vendor/native && cp $Q/build/quantus_hip.node vendor/native/
if ! { ALLOW_MISSING_CORE=1 node scripts/build-cli.mjs && ALLOW_MISSING_CORE=1 node scripts/build-hiveos.mjs; } > /root/dist.log 2>&1; then
  say "FAIL package"; tail -n 12 /root/dist.log | sayf hive; exit 1
fi
tb=$(ls dist/llmjob-earn-[0-9]*.tar.gz | head -n 1)
rm -rf /hive/miners/custom /run/hive/llmjob-earn-stats.json && mkdir -p /hive/miners/custom /run/hive && tar xzf "$tb" -C /hive/miners/custom || { say "FAIL untar $tb"; exit 1; }
P=/hive/miners/custom/llmjob-earn
say "package $(basename $tb): $(cd $P && ls | tr '\n' ' ')"
# A Quantus-only flight sheet: the qz address as the wallet (%WAL%.%WORKER_NAME%), a Quantus Pool URL, and no
# --coin or Pearl address anywhere.
(
  cd $P && . ./h-manifest.conf
  export CUSTOM_TEMPLATE="$QZ.g19q-amd-hive3" WORKER_NAME=g19q-amd-x78-2 CUSTOM_URL="us.quantus.herominers.com:1155"
  export CUSTOM_USER_CONFIG="--no-update --no-report"
  bash h-config.sh 2>&1 | sayf hiveconf
  sayf hiveconf < "$CUSTOM_CONFIG_FILENAME"
)
mine hive "$HIVE_SECS" $P bash h-run.sh
say "job done"
