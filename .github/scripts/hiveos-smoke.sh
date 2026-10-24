#!/usr/bin/env bash
# Runs a build's HiveOS package the way a HiveOS rig runs it, in a plain Ubuntu container.
#
#   hiveos-smoke.sh <image> <mines|refuses> <dir> <version>
#
# <dir> holds llmjob-earn-<version>.tar.gz, as the build's llmjob-earn-cli-linux artifact does.
#
# The unit tests run the hook scripts against a fake CLI. This runs the real package: the
# binary, the cores and the scripts, on the system HiveOS is built on. v0.6.7 passed every test
# and still mined nothing on HiveOS, because its cores needed a newer glibc than Ubuntu 22.04
# has. Here that is a core that won't load, and the run fails.
#
# In order, it:
#   - unpacks the package where HiveOS's custom-get puts it, /hive/miners/custom/llmjob-earn
#   - checks the files, their modes and the version
#   - lists each core's libraries with ldd. A glibc or libstdc++ version the system lacks fails,
#     and so does a library it lacks, other than ROCm's for the AMD core (README: the rig
#     installs ROCm itself)
#   - writes the argument line with h-config.sh from a sample flight sheet, starts h-run.sh, and
#     waits for the CUDA 12.8 core to load and ask for a GPU. There is none, so the miner then
#     stops, as it does on a rig without an NVIDIA driver
#   - does the same with --core cu13 for the CUDA 13 core, and checks that --core amd says ROCm
#     is missing (the container has none)
#   - with every GPU turned off in HiveOS (CUDA_VISIBLE_DEVICES=" "), mines Monero on the CPU,
#     waits for the worker's self-test, reads the stats file through h-stats.sh as the HiveOS
#     agent does, and stops the miner with Ctrl+C, as `miner stop` does, checking its shutdown
#     line is logged
#
# With "refuses" (Ubuntu 20.04, HiveOS's older focal image) it checks the opposite: the cores
# don't load there, and the miner says why rather than mining nothing in silence.
#
# There is no GPU, so nothing is mined. The container's network is cut before the miner starts:
# it must not reach a real pool or put a row on the network board.
set -euo pipefail

if [ "${1:-}" != --inside ]; then
  image=$1 expect=$2 dir=$3 version=$4
  tarball="$dir/llmjob-earn-$version.tar.gz"
  [ -f "$tarball" ] || { echo "::error::$tarball is not in the build's artifact"; exit 1; }
  name="hiveos-smoke-$$"
  docker run -d --name "$name" "$image" sleep infinity >/dev/null
  trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
  # A HiveOS rig has jq (h-stats.sh needs it) and procps; the base image has neither. They go
  # in while the container still has a network.
  docker exec -e DEBIAN_FRONTEND=noninteractive "$name" sh -c \
    'apt-get update -qq && apt-get install -y -qq --no-install-recommends jq procps >/dev/null'
  docker cp "$tarball" "$name:/tmp/"
  docker cp "${BASH_SOURCE[0]}" "$name:/tmp/hiveos-smoke.sh"
  docker network disconnect bridge "$name"
  docker exec "$name" bash /tmp/hiveos-smoke.sh --inside "$image" "$expect" "$version"
  exit 0
fi

image=$2 expect=$3 version=$4
d=/hive/miners/custom/llmjob-earn
log=/var/log/miner/custom/llmjob-earn/llmjob-earn.log
stats_file=/run/hive/llmjob-earn-stats.json
# Any prl1p… address passes the CLI's check. The Monero one is the Monero Project's donation
# address, which the unit tests use too. Neither reaches a pool: the network is off.
address=prl1p$(printf 'q%.0s' {1..58})
xmr=44AFFq5kSiGBoZ4NMDwYtN18obc8AemS33DBLWs3H7otXft3XjrpDtQGv7SqSsaBYBb98uNbr2VBBEt7f2wfn3RVGQBEP3A

fail() {
  echo "::error::$image: $*"
  if [ -s "$log" ]; then
    echo "--- the miner's log ---"
    cat "$log"
  fi
  exit 1
}

if timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
  fail "the container can still reach the internet, so the miner isn't started: it could reach a real pool"
fi

# ── Install, as custom-get does ──────────────────────────────────────────────────────────
mkdir -p /hive/miners/custom /run/hive "$(dirname "$log")"
: >> "$log"
tar -xzf "/tmp/llmjob-earn-$version.tar.gz" -C /hive/miners/custom
cd "$d"
ls -l

for f in h-manifest.conf h-config.sh h-run.sh h-stats.sh llmjob-earn-cli-linux pearl_core.node; do
  [ -f "$f" ] || fail "the package has no $f"
done
# The optional files: a build can ship without each, and says so in its own run summary.
for f in pearl_core_cu13.node pearl_core_hip.node llmjob-xmr-worker; do
  [ -f "$f" ] || echo "::warning::$image: the package has no $f, so it isn't tested"
done
mode_is() {
  [ ! -e "$1" ] || [ "$(stat -c %a "$1")" = "$2" ] || fail "$1 is mode $(stat -c %a "$1"), not $2"
}
for f in h-config.sh h-run.sh h-stats.sh llmjob-earn-cli-linux llmjob-xmr-worker; do mode_is "$f" 755; done
for f in h-manifest.conf pearl_core.node pearl_core_cu13.node pearl_core_hip.node; do mode_is "$f" 644; done

got=$(. ./h-manifest.conf && echo "$CUSTOM_VERSION")
[ "$got" = "$version" ] || fail "h-manifest.conf says CUSTOM_VERSION=$got, not $version"
got=$(./llmjob-earn-cli-linux --version 2>&1) || fail "llmjob-earn-cli-linux --version failed: $got"
[ "$got" = "$version" ] || fail "llmjob-earn-cli-linux --version says $got, not $version"

# ── What each core needs from the system ─────────────────────────────────────────────────
for core in pearl_core.node pearl_core_cu13.node pearl_core_hip.node; do
  [ -f "$core" ] || continue
  deps=$(ldd "./$core" 2>&1 || true)
  echo "ldd $core:"
  sed 's/^/  /' <<< "$deps"
  # "./x.node: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by …)"
  versions=$(grep -F "version \`" <<< "$deps" || true)
  missing=$(grep -F '=> not found' <<< "$deps" | awk '{print $1}' || true)
  if [ "$core" = pearl_core_hip.node ]; then
    missing=$(grep -vE '^lib(amdhip64|hsa-runtime64|amd_comgr|rocprofiler-register)\.' <<< "$missing" || true)
  fi
  if [ "$expect" = mines ]; then
    [ -z "$versions" ] || fail "$core needs versions this system doesn't have: $versions"
    [ -z "$missing" ] || fail "$core needs libraries this system doesn't have: $missing"
  elif [ "$core" = pearl_core.node ] && [ -z "$versions" ]; then
    fail "pearl_core.node has every version it needs here. h-manifest.conf and the README say the cores need Ubuntu 22.04; if they now run on $image, say so there and make this a 'mines' run."
  fi
done

# ── The flight sheet, h-config.sh and h-run.sh ───────────────────────────────────────────
# h-config.sh as HiveOS's wrapper runs it: in a subshell, with the flight sheet's fields set.
# $2, when given, is CUDA_VISIBLE_DEVICES, which `miner start` sets when cards are turned off.
configure() {
  (
    export CUSTOM_TEMPLATE="$address.smoke" WORKER_NAME=hive-smoke CUSTOM_USER_CONFIG="$1"
    [ $# -lt 2 ] || export CUDA_VISIBLE_DEVICES="$2"
    bash ./h-config.sh
  ) || fail "h-config.sh failed"
  echo "argument line: $(< llmjob-earn.conf)"
}

# h-run.sh in a session of its own, as HiveOS runs it in a screen, so that Ctrl+C reaches the
# whole pipeline. Each run appends to the one log; `from` is where this run's lines start.
start() {
  from=$(( $(wc -l < "$log") + 1 ))
  setsid bash ./h-run.sh < /dev/null > /dev/null 2>&1 &
  run=$!
}
run_log() { tail -n +"$from" "$log"; }

# Waits up to $2 seconds for a line containing $1 in this run's log. Fails early when the run
# has ended, or has said a core won't load.
wait_for() {
  local text=$1 secs=$2 i
  for ((i = 0; i < secs; i++)); do
    grep -qF -- "$text" <<< "$(run_log)" && return 0
    grep -qF ' is there but would not load: ' <<< "$(run_log)" && fail "a core would not load (waiting for \"$text\")"
    kill -0 "$run" 2>/dev/null || fail "the miner exited before it logged \"$text\""
    sleep 1
  done
  fail "no \"$text\" in the miner's log after $secs s"
}

# Waits up to $1 seconds for the run to end by itself.
wait_exit() {
  local i
  for ((i = 0; i < $1; i++)); do
    kill -0 "$run" 2>/dev/null || { wait "$run" || true; return 0; }
    sleep 1
  done
  kill -KILL -- "-$run" 2>/dev/null || true
  fail "the miner was still running after $1 s"
}

# Ctrl+C, as `miner stop` sends it.
stop() {
  kill -INT -- "-$run" 2>/dev/null || true
  wait_exit 30
  grep -qF 'shutting down' <<< "$(run_log)" || fail "after Ctrl+C the log has no \"shutting down\" line"
}

if [ "$expect" = refuses ]; then
  configure "--region us"
  start
  wait_exit 60
  grep -qF 'pearl_core.node is there but would not load: ' <<< "$(run_log)" \
    || fail "the miner didn't say why its core won't load"
  grep -qE "version .(GLIBC|GLIBCXX|CXXABI)_[0-9.]+' not found" <<< "$(run_log)" \
    || fail "the miner's reason doesn't name the version its core needs"
  echo "the miner's log:"
  run_log | sed 's/^/  /'
  echo "ok: on $image the cores don't load, and the miner says why"
  exit 0
fi

# Logged when a core loaded and asked CUDA for a GPU. Here there is none.
no_gpu="no CUDA device found"

# 1. The CUDA 12.8 core, the one every NVIDIA rig gets unless it has only RTX 50 cards.
configure "--region us"
conf=" $(< llmjob-earn.conf) "
for want in "--address $address" --worker=smoke "--mode mining" --no-update "--stats-file $stats_file"; do
  [[ $conf == *" $want "* ]] || fail "the argument line has no \"$want\":$conf"
done
start
wait_for "LLMJob Earn CLI v$version" 30
wait_for "Pearl core: CUDA 12.8 build" 60
wait_for "$no_gpu" 60
wait_exit 60

# 2. The CUDA 13 core, which a rig gets on its own only with RTX 50 cards and driver 580+.
if [ -f pearl_core_cu13.node ]; then
  configure "--region us --core cu13"
  start
  wait_for "Pearl core: CUDA 13 build" 60
  wait_for "$no_gpu" 60
  wait_exit 60
fi

# 3. The AMD core. Without ROCm it can't load, so the miner exits, and the log must say ROCm
#    is what's missing, not something about the system.
if [ -f pearl_core_hip.node ]; then
  configure "--region us --core amd"
  start
  wait_exit 60
  grep -qF 'pearl_core_hip.node is there but would not load: ' <<< "$(run_log)" \
    || fail "with --core amd the miner didn't say why the AMD core won't load"
  grep -qF 'It needs the ROCm 7 HIP runtime' <<< "$(run_log)" \
    || fail "with --core amd the miner's reason isn't that ROCm is missing"
fi

# 4. Every GPU turned off in HiveOS, mining Monero on the CPU. The miner keeps running, so this
#    is the run that writes the stats file and is stopped with Ctrl+C.
extra="--region us"
[ -f llmjob-xmr-worker ] && extra+=" --xmr-address $xmr --cpu-threads 1"
configure "$extra" " "
[[ " $(< llmjob-earn.conf) " == *" --gpu-index=none "* ]] \
  || fail "with every GPU turned off the argument line has no --gpu-index=none: $(< llmjob-earn.conf)"
rm -f "$stats_file"
start
if [ -f llmjob-xmr-worker ]; then
  wait_for "RandomX self-test passed" 60
else
  wait_for "waiting until stopped" 60
fi

for ((i = 0; i < 15; i++)); do [ -f "$stats_file" ] && break; sleep 1; done
[ -f "$stats_file" ] || fail "no $stats_file, so h-stats.sh has nothing to read"
# The agent sources h-stats.sh with its own gpu_stats and gpu_detect_json. Without them it
# reads files that aren't here, as on a rig whose agent hasn't read its cards yet.
read_back=$(unset gpu_stats gpu_detect_json && . ./h-stats.sh && printf '%s\n%s\n' "$khs" "$stats")
khs=$(sed -n 1p <<< "$read_back")
stats=$(sed -n 2p <<< "$read_back")
echo "h-stats.sh: khs=$khs stats=$stats"
[[ $khs =~ ^[0-9]+$ ]] || fail "h-stats.sh set khs to \"$khs\""
jq -e --arg v "$version" '.algo == "pearlhash" and .ver == $v and .hs_units == "khs"' <<< "$stats" > /dev/null \
  || fail "h-stats.sh's stats aren't what the agent expects: $stats"
if [ -f llmjob-xmr-worker ]; then
  jq -e '.algo2 == "randomx"' <<< "$stats" > /dev/null || fail "h-stats.sh's stats have no Monero: $stats"
fi
stop

echo "the miner's log, every run:"
sed 's/^/  /' "$log"
echo "ok: the HiveOS package installs, loads its cores and reports to h-stats.sh on $image"
