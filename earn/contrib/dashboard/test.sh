#!/bin/bash
# Regression tests for gpu-dashboard's mode handling.
#
#   ./test.sh
#
# Not wired into CI (the JS suites are what test.yml runs); these exist because the
# two bugs below cost a live rig its config and ~35 minutes of downtime, and both
# are invisible to a syntax check.
#
#   1. set_mode truncated $EARN_ENV instead of updating the EARN_MODE line, wiping
#      EARN_BIN/EARN_ADDRESS/EARN_WORKER/EARN_ARGS on the first mode switch.
#   2. the selector treated its own stale value as authoritative, so a unit started
#      outside the dashboard was "corrected" back -- firing (1) with nobody at the
#      keyboard.
cd "$(dirname "$0")" || exit 1
DASH=./gpu-dashboard
fail=0
ok()  { printf '  PASS  %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }

bash -n "$DASH" && ok "syntax" || bad "syntax"

# ---------------------------------------------------------------- write_mode ---
# Pull the function out; sourcing the dashboard would start the UI loop.
eval "$(sed -n '/^write_mode() {/,/^}/p' "$DASH")"

full() { cat <<'EOF'
EARN_ADDRESS=prl1pTESTADDRESS
EARN_WORKER=rig-1
EARN_MODE=auto
EARN_BIN="/usr/bin/node /opt/llmjob/earn/src/cli/earn-cli.js"
EARN_ARGS="--no-update --llm-binary /opt/llama/llama-server"
PEARL_CORE_PATH=/opt/llmjob/earn/native/build/Release/pearl_core.node
EOF
}

EARN_ENV=$(mktemp); full > "$EARN_ENV"
write_mode mining
for key in EARN_ADDRESS EARN_WORKER EARN_BIN EARN_ARGS PEARL_CORE_PATH; do
  grep -q "^${key}=" "$EARN_ENV" && ok "switch preserves $key" || bad "switch LOST $key"
done
grep -qx 'EARN_MODE=mining' "$EARN_ENV" && ok "mode updated" || bad "mode not updated"
[ "$(grep -c '^EARN_MODE=' "$EARN_ENV")" = 1 ] && ok "one EARN_MODE line" || bad "duplicate EARN_MODE"
grep -qx 'EARN_BIN="/usr/bin/node /opt/llmjob/earn/src/cli/earn-cli.js"' "$EARN_ENV" \
  && ok "quoted value with spaces round-trips" || bad "quoted value mangled"

write_mode llm; write_mode off; write_mode auto
[ "$(grep -c '^EARN_MODE=' "$EARN_ENV")" = 1 ] && ok "repeated switches do not accumulate" || bad "EARN_MODE accumulated"
grep -q '^EARN_BIN=' "$EARN_ENV" && ok "settings survive repeated switches" || bad "settings lost over switches"

E2=$(mktemp); printf 'EARN_ADDRESS=prl1pX\nEARN_BIN="/a b"\n' > "$E2"
EARN_ENV=$E2; write_mode mining
grep -qx 'EARN_MODE=mining' "$E2" && ok "appends when no EARN_MODE line exists" || bad "did not append"
grep -q '^EARN_ADDRESS=' "$E2" && ok "append keeps other keys" || bad "append lost other keys"

E3=$(mktemp); full > "$E3"; chmod 600 "$E3"
EARN_ENV=$E3; write_mode llm
[ "$(stat -c %a "$E3")" = 600 ] && ok "file mode preserved" || bad "file mode became $(stat -c %a "$E3")"
[ "$(ls "${E3}".?????? 2>/dev/null | wc -l)" = 0 ] && ok "no temp files left behind" || bad "temp files left behind"
rm -f "$EARN_ENV" "$E2" "$E3"

# ------------------------------------------------------------ selector state ---
# Replay of the loop's resync + settle-fire, same order as the real thing.
fired=(); mode_switch() { fired+=("$1"); }
target=''; last_fired=''; sw_since=0; user_sel=0; last_actual=''
tick() {
  local actual=$1 press=${2:-}
  [ -z "$target" ] && target=$actual
  if [ "$actual" != "$last_actual" ]; then
    (( user_sel )) || { target=$actual; last_fired=$actual; }
    last_actual=$actual
  fi
  if (( user_sel )) && [ "$target" != "$last_fired" ] && [ "$target" != "$actual" ] && (( $(date +%s) - sw_since >= 1 )); then
    last_fired=$target; user_sel=0; mode_switch "$target"
  fi
  if [ -n "$press" ]; then
    local old=$target; target=$press
    [ "$target" != "$old" ] && { sw_since=$(( $(date +%s) - 5 )); user_sel=1; }
  fi
}

tick off; tick off; tick auto; tick auto; tick auto
[ ${#fired[@]} -eq 0 ] && ok "externally started unit is not reverted" || bad "reverted the unit: ${fired[*]}"
[ "$target" = auto ] && ok "selector resyncs to real state" || bad "selector stuck on $target"

fired=(); tick auto mine; tick auto
[ "${fired[0]}" = mine ] && ok "keypress still switches" || bad "keypress did not switch"

fired=(); tick auto; tick auto; tick auto
[ ${#fired[@]} -eq 0 ] && ok "no re-fire while a switch is in flight" || bad "re-fired ${#fired[@]}x"

fired=(); tick off; tick off; tick off
[ ${#fired[@]} -eq 0 ] && ok "failed restart is not retried in a loop" || bad "retry loop: ${fired[*]}"

echo
[ "$fail" = 0 ] && echo "ALL PASS" || echo "FAILURES"
exit "$fail"
