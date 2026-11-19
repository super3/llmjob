# q-amd job 1: quantus_hip for every target, verify-core, rate twice, the CLI on HeroMiners (its default) and Kryptex, then the HiveOS package on HeroMiners
# A box job for the q-amd agent (Quantus on AMD, 19 Nov). The box agent runs it from earn/native/amd with J_SHA set
# to the commit it came from. Not part of the miner: the agent drops this file when the round ends.
# Lines go straight to the container's log as "[amdtest] HH:MM:SS q ...".
AMDTEST_STAGE=lib . /opt/llmjob/earn/tuning/salad/amdtest.sh
say() { local l="[amdtest] $(date -u +%T) q $*"; echo "$l"; { echo "$l" > /proc/1/fd/1; } 2>/dev/null || true; }
sayf() { local tag=$1 l; while IFS= read -r l; do [ -n "${l//[[:space:]]/}" ] && say "$tag | ${l:0:380}"; done; }
QZ=qzk7L38XY4MBHgjvqiudDvVR7gv5asYpZiTKLWJwZjSbzLab6
PARTS=${J_PARTS:-build verify rate cli kryptex hive}
CLI_SECS=${J_CLI_SECS:-600}; KX_SECS=${J_KX_SECS:-360}; HIVE_SECS=${J_HIVE_SECS:-480}
has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
Q=$R/earn/native/quantus
ARCHX=${ARCH:-$(target)}
say "job start $ARCHX, code ${J_SHA:0:12}, parts: $PARTS"

# The tree at the job's commit.
if [ "$(git -C $R rev-parse HEAD)" != "$J_SHA" ]; then
  ( cd $R && git fetch -q --depth 1 origin "$J_SHA" && git checkout -q -f FETCH_HEAD ) > /root/tree.log 2>&1 \
    || { say "FAIL fetch $J_SHA"; exit 1; }
fi
say "tree $(git -C $R rev-parse --short=12 HEAD)"

if has build; then
  t0=$(date +%s)
  if ! (cd $Q/amd && bash build.sh check "$ARCHX" && bash build.sh gpu) > /root/qbuild.log 2>&1; then
    say "FAIL build"; grep -iE 'error|build.sh' /root/qbuild.log | head -10 | sayf build; exit 1
  fi
  grep -E '^(target|gfx)' /root/qbuild.log | sayf regs
  f=$Q/build/quantus_hip.node
  say "built in $(( $(date +%s) - t0 ))s: $(stat -c %s $f) bytes, md5 $(md5sum $f | cut -c1-8), targets $(strings -a $f | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f]+' | sed 's/.*--//' | sort -u | tr '\n' ' ')"
fi

if has verify; then
  (cd $R/earn && timeout 1200 node native/quantus/verify-core.js --addon native/quantus/build/quantus_hip.node) > /root/qverify.log 2>&1
  rc=$?
  sayf verify < /root/qverify.log
  say "verify rc=$rc"
fi

if has rate; then
  for i in 1 2; do
    r=$(timeout 300 node $Q/amd/test/rate.js $Q/build/quantus_hip.node --secs 60 --warm 10 2>&1 | tail -n 1)
    say "rate $i $r"
  done
  s=/opt/rocm-wsl/bin/amd-smi
  [ -x $s ] && say "smi idle: $(timeout 15 $s metric -c -p -t 2>&1 | grep -iE 'gfx_0|clk|socket_power|edge|hotspot' | tr -s ' ' | head -8 | tr '\n' ' ' | cut -c1-300)"
fi

# mine NAME SECS DIR CMD...: runs CMD in DIR for SECS, then SIGINT. Every 60 s, the new lines other than rate lines,
# and the last rate line. At the end: shares accepted and rejected, and the mean displayed rate after 60 s.
mine() {
  local name=$1 secs=$2 dir=$3; shift 3
  local log=/root/qmine-$name.log pid t0 seen=0 n
  say "mine $name start ${secs}s"
  ( cd "$dir" && exec "$@" ) > $log 2>&1 &
  pid=$!; t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    n=$(wc -l < $log)
    if [ $(( ($(date +%s) - t0) % 60 )) -lt 20 ] || [ $(( $(date +%s) - t0 )) -lt 60 ]; then
      tail -n +$((seen + 1)) $log | head -n $((n - seen)) | grep -v 'H/s ·' | grep -v 'QTC share accepted' | head -n 25 | sayf "$name"
      tail -n +$((seen + 1)) $log | head -n $((n - seen)) | grep 'H/s ·' | tail -n 1 | sayf "$name"
      seen=$n
    fi
    [ $(( $(date +%s) - t0 )) -ge "$secs" ] && break
  done
  pkill -INT -f llmjob-earn-cli-linux 2>/dev/null; kill -INT $pid 2>/dev/null; sleep 8
  pkill -f llmjob-earn-cli-linux 2>/dev/null; kill $pid 2>/dev/null; wait $pid 2>/dev/null
  tail -n +$((seen + 1)) $log | grep -v 'H/s ·' | grep -v 'QTC share accepted' | tail -n 15 | sayf "$name"
  local acc rej mean
  acc=$(grep -c 'QTC share accepted' $log); rej=$(grep -c 'QTC share rejected' $log)
  mean=$(grep -oE '[0-9.]+ [KMG]H/s · [0-9,]+ accepted · [0-9,]+ rejected · up [0-9hms ]+' $log | node -e '
    let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const u={K:1e-3,M:1,G:1e3};const v=[];
    for(const l of s.split("\n")){const m=/^([0-9.]+) ([KMG])H\/s .* up (?:(\d+)h ?)?(?:(\d+)m ?)?(?:(\d+)s)?/.exec(l);if(!m)continue;
    const t=(+(m[3]||0))*3600+(+(m[4]||0))*60+(+(m[5]||0));if(t>60)v.push(+m[1]*u[m[2]]);}
    if(!v.length){console.log("n=0");return}const a=v.reduce((x,y)=>x+y,0)/v.length;
    console.log("mean_after_60s="+a.toFixed(1)+" MH/s min="+Math.min(...v).toFixed(1)+" max="+Math.max(...v).toFixed(1)+" n="+v.length)})')
  say "mine $name result accepted=$acc rejected=$rej $mean secs=$(( $(date +%s) - t0 ))"
}

CLI=(node src/cli/earn-cli.js --coin quantus --quantus-address $QZ --mode mining --no-report --no-serve --no-update)
export NODE_PATH=$NM/node_modules
has cli && mine cli "$CLI_SECS" $R/earn "${CLI[@]}" --region us -w g19q-amd-cli
has kryptex && mine kryptex "$KX_SECS" $R/earn "${CLI[@]}" --quantus-pool qtc-us.kryptex.network:7049 -w g19q-amd-kx

if has hive; then
  t0=$(date +%s)
  cd $R/earn || exit 1
  if ! npm ci --ignore-scripts --no-audit --no-fund --loglevel=error > /root/npmci.log 2>&1; then
    say "FAIL npm ci"; tail -n 8 /root/npmci.log | sayf hive; exit 1
  fi
  mkdir -p vendor/native && cp $Q/build/quantus_hip.node vendor/native/
  if ! { ALLOW_MISSING_CORE=1 node scripts/build-cli.mjs && ALLOW_MISSING_CORE=1 node scripts/build-hiveos.mjs; } > /root/dist.log 2>&1; then
    say "FAIL package"; tail -n 12 /root/dist.log | sayf hive; exit 1
  fi
  grep -iE 'quantus|warning|wrote|tar' /root/dist.log | head -n 12 | sayf hive
  tb=$(ls dist/llmjob-earn-[0-9]*.tar.gz | head -n 1)
  rm -rf /hive/miners/custom && mkdir -p /hive/miners/custom /run/hive && tar xzf "$tb" -C /hive/miners/custom || { say "FAIL untar $tb"; exit 1; }
  P=/hive/miners/custom/llmjob-earn
  say "package $(basename $tb) in $(( $(date +%s) - t0 ))s: $(cd $P && ls | tr '\n' ' ')"
  # As HiveOS starts a custom miner: h-config.sh writes the argument line from the flight sheet, h-run.sh runs it.
  (
    cd $P && . ./h-manifest.conf
    export CUSTOM_TEMPLATE="$WALLET.g19q-amd-hive" WORKER_NAME=g19q-amd-hive CUSTOM_URL=""
    export CUSTOM_USER_CONFIG="--coin quantus --quantus-address $QZ --region us --no-update --no-report"
    bash h-config.sh && sed "s/$WALLET/<prl>/" "$CUSTOM_CONFIG_FILENAME" | sayf hiveconf
  )
  mine hive "$HIVE_SECS" $P bash h-run.sh
fi
say "job done"
