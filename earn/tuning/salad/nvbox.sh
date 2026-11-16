# The start nv20box.sh, nv30box.sh, nv34box.sh and nv50box.sh share. Each box script fetches this file at
# BASE_SHA and sources it, with NVTEST_TAG set to its own name ("nv20"). Every line is then
# "[nvtest] HH:MM:SS [nv20] ...", here and in nvtest.sh's functions, which read the same NVTEST_TAG.
#
#   nvbox_boot [file...]  fetch nvtest.sh, bench.sh, sharetap.py and any extra files from earn/tuning/salad into
#                         /root, then log the card and set CC (its compute capability, "86"). NVBOX_MEM=1 adds the
#                         memory clocks to the card lines.
#   nvbox_apt             the packages the boxes use (a warning, not a failure, if apt-get fails)
#   q <fields>            one nvidia-smi query of card 0, as CSV with no header
say() { echo "[nvtest] $(date -u +%T) [$NVTEST_TAG] $*"; sleep 0.03; }
U="https://raw.githubusercontent.com/super3/llmjob/$BASE_SHA/earn/tuning/salad"
q() { nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader 2>&1 | head -1; }

nvbox_boot() {
  local f
  for f in nvtest.sh bench.sh sharetap.py "$@"; do
    curl -fsSL --retry 3 "$U/$f" -o "/root/$f" || { say "FAIL fetch $f"; exec sleep infinity; }
  done
  say "card $(q name,driver_version,compute_cap,power.limit,power.default_limit,power.max_limit,clocks.max.sm${NVBOX_MEM:+,clocks.max.mem},memory.total)"
  say "card idle $(q clocks.sm${NVBOX_MEM:+,clocks.mem},power.draw,temperature.gpu,utilization.gpu,memory.used) gpus=$(nvidia-smi -L | wc -l)"
  CC=$(q compute_cap | tr -d ' .')
}

nvbox_apt() {
  export DEBIAN_FRONTEND=noninteractive
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates xz-utils g++ make \
      python3 jq procps; } > /root/apt0.log 2>&1 || say "WARN apt-get: $(tail -1 /root/apt0.log)"
}
