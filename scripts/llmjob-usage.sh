#!/bin/bash
# LLMJob usage log-shipper — tails the llama.cpp (llama-server) journal and
# posts one record to POST /api/usage per completed inference, so the
# dashboard Logs table and 24h activity chart populate.
#
# llama-server emits three timing lines per request, sharing one task id:
#
#   slot print_timing: id  0 | task 6080 | prompt eval time =   11352.50 ms / 32734 tokens (    0.35 ms per token,  2883.42 tokens per second)
#   slot print_timing: id  0 | task 6080 |        eval time =     817.82 ms /    43 tokens (   19.02 ms per token,    52.58 tokens per second)
#   slot print_timing: id  0 | task 6080 |       total time =   12170.32 ms / 32777 tokens
#
# Mapping: in = prompt-eval tokens, out = eval tokens, speed = eval tok/s
# (1 decimal). The journal does not expose a true finish reason, so finish is
# always "stop" (documented approximation — see scripts/README.md).
#
# Usage:
#   llmjob-usage.sh           follow the journal and ship records (needs
#                             LLMJOB_API_KEY, an lj-… key from the dashboard)
#   llmjob-usage.sh --parse   read journal lines from stdin and print the
#                             assembled JSON records to stdout (no network;
#                             used by the test suite)

set -u
export LC_ALL=C

SERVER="${LLMJOB_SERVER:-https://llmjob-production.up.railway.app}"
API_KEY="${LLMJOB_API_KEY:-}"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://127.0.0.1:8000}"
LLAMA_UNIT="${LLAMA_UNIT:-llama-qwen}"
APP="${LLMJOB_APP:-hermes}"
CONFIG_DIR="${LLMJOB_CONFIG_DIR:-$HOME/.llmjob}"
MODEL="${LLMJOB_MODEL:-}"
MAX_TASKS=64  # cap the partial-record map so a dropped line can't leak memory

PARSE_ONLY=0
[ "${1:-}" = "--parse" ] && PARSE_ONLY=1

RE_PROMPT='task +([0-9]+) \| +prompt eval time += +[0-9.]+ ms / +([0-9]+) tokens'
RE_EVAL='task +([0-9]+) \| +eval time += +[0-9.]+ ms / +([0-9]+) tokens \( *[0-9.]+ ms per token, +([0-9.]+) tokens per second'
RE_TOTAL='task +([0-9]+) \| +total time'

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# Model alias served by llama.cpp (e.g. "qwen3.6-27b"), from /v1/models.
resolve_model() {
  curl -fsS --max-time 5 "$LLAMA_ENDPOINT/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["models"][0]["name"])' 2>/dev/null
}

# Node attribution: LLMJOB_NODE_NAME, else the nodeId saved by install.sh,
# else the hostname.
resolve_node() {
  if [ -n "${LLMJOB_NODE_NAME:-}" ]; then
    printf '%s' "$LLMJOB_NODE_NAME"
  elif [ -r "$CONFIG_DIR/node_id" ]; then
    printf 'node-%s' "$(head -n 1 "$CONFIG_DIR/node_id")"
  else
    hostname 2>/dev/null || uname -n
  fi
}

declare -A T_IN T_OUT T_SPEED T_SEEN
declare -a TASK_ORDER=()

# Register a task id, evicting the oldest partial when the map is full.
touch_task() {
  local id=$1
  if [ -z "${T_SEEN[$id]:-}" ]; then
    T_SEEN[$id]=1
    TASK_ORDER+=("$id")
    if [ "${#TASK_ORDER[@]}" -gt "$MAX_TASKS" ]; then
      local old=${TASK_ORDER[0]}
      TASK_ORDER=("${TASK_ORDER[@]:1}")
      unset "T_IN[$old]" "T_OUT[$old]" "T_SPEED[$old]" "T_SEEN[$old]"
    fi
  fi
}

# POST one usage record; on failure retry 3x with backoff, then drop it.
post_usage() {
  local body=$1 attempt code delay=2
  for attempt in 1 2 3; do
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$SERVER/api/usage" \
      -H "Authorization: Bearer $API_KEY" \
      -H 'Content-Type: application/json' \
      -d "$body" 2>/dev/null) || code=000
    case "$code" in 2??) return 0 ;; esac
    echo "llmjob-usage: POST /api/usage failed (HTTP $code), attempt $attempt/3" >&2
    sleep "$delay"; delay=$((delay * 2))
  done
  echo "llmjob-usage: dropping usage record after 3 failed attempts: $body" >&2
  return 1
}

# Emit the record for a task once both its prompt-eval and eval lines have
# been seen; clearing the entry afterwards guarantees a task id is never
# emitted twice.
emit_task() {
  local id=$1
  [ -n "${T_IN[$id]:-}" ] && [ -n "${T_OUT[$id]:-}" ] || return 0

  if [ -z "$MODEL" ] && [ "$PARSE_ONLY" -eq 0 ]; then
    MODEL=$(resolve_model)
  fi
  local model=${MODEL:-unknown}

  local body
  body=$(printf '{"model":"%s","node":"%s","app":"%s","in":%s,"out":%s,"speed":%s,"finish":"stop"}' \
    "$(json_escape "$model")" "$(json_escape "$NODE")" "$(json_escape "$APP")" \
    "${T_IN[$id]}" "${T_OUT[$id]}" "${T_SPEED[$id]}")
  unset "T_IN[$id]" "T_OUT[$id]" "T_SPEED[$id]"

  if [ "$PARSE_ONLY" -eq 1 ]; then
    printf '%s\n' "$body"
  else
    post_usage "$body" &  # background so a slow server never blocks the tail
  fi
}

process_line() {
  local line=$1 id
  if [[ $line =~ $RE_PROMPT ]]; then
    id=${BASH_REMATCH[1]}
    touch_task "$id"
    T_IN[$id]=${BASH_REMATCH[2]}
  elif [[ $line =~ $RE_EVAL ]]; then
    id=${BASH_REMATCH[1]}
    touch_task "$id"
    T_OUT[$id]=${BASH_REMATCH[2]}
    T_SPEED[$id]=$(printf '%.1f' "${BASH_REMATCH[3]}")
  elif [[ $line =~ $RE_TOTAL ]]; then
    emit_task "${BASH_REMATCH[1]}"
  fi
}

main() {
  NODE=$(resolve_node)

  if [ "$PARSE_ONLY" -eq 1 ]; then
    while IFS= read -r line; do process_line "$line"; done
    return 0
  fi

  if [ -z "$API_KEY" ]; then
    echo "llmjob-usage: error: LLMJOB_API_KEY is not set." >&2
    echo "llmjob-usage: create an lj-… key in the dashboard (API Keys) and put it in /etc/llmjob/agent.env." >&2
    exit 1
  fi

  [ -z "$MODEL" ] && MODEL=$(resolve_model)
  echo "llmjob-usage: shipping usage for node '$NODE' (model '${MODEL:-unknown}') from unit '$LLAMA_UNIT' to $SERVER" >&2

  # Follow only new journal lines; journalctl reconnects across unit restarts.
  while IFS= read -r line; do
    process_line "$line"
  done < <(journalctl -u "$LLAMA_UNIT" -f -o cat --since now)
}

main
