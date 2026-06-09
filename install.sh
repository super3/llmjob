#!/bin/sh
# LLMJob node agent — pure shell, no Node and no npm. Needs only curl + openssl
# (preinstalled on most Linux). Served by the app and meant to be run from a pipe:
#
#   curl -fsSL <server>/install.sh/<token> | bash
#
# The app bakes the server and join token into this script when it serves the
# per-account URL, so no arguments are needed. You can still override with
# --server <url> / --token <token> / --name <name>, or the LLMJOB_SERVER /
# LLMJOB_TOKEN / LLMJOB_NODE_NAME environment variables (used by the systemd
# unit in scripts/systemd/).
#
# It creates an Ed25519 key locally (only the public key ever leaves the
# machine), joins this machine to your account with the join token, then pings
# so the node shows as online. Each ping also carries best-effort telemetry
# (GPU, VRAM, served model, quant, tok/s) gathered from nvidia-smi, the local
# llama.cpp server and journald — any source that is missing is simply omitted.

SERVER="${LLMJOB_SERVER:-https://llmjob-production.up.railway.app}"
TOKEN="${LLMJOB_TOKEN:-}"
NAME="${LLMJOB_NODE_NAME:-}"
PING_INTERVAL=300  # seconds
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://127.0.0.1:8000}"
LLAMA_UNIT="${LLAMA_UNIT:-llama-qwen}"

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# Emit ,"key":"value" (JSON-escaped string) only when the value is non-empty,
# so absent telemetry is omitted entirely — the server keeps its prior value.
json_str_field() {
  if [ -n "$2" ]; then printf ',"%s":"%s"' "$1" "$(json_escape "$2")"; fi
}

# Emit ,"key":value (unquoted number) only when the value is a plain number.
json_num_field() {
  case "$2" in
    ''|*[!0-9.]*|.|*.*.*) ;;
    *) printf ',"%s":%s' "$1" "$2" ;;
  esac
}

# Extract the GGUF quant token from a model filename, e.g.
# Qwen_Qwen3.6-27B-Q6_K.gguf -> Q6_K. Prints nothing when no token is present.
parse_quant() {
  printf '%s\n' "$1" \
    | grep -oiE 'I?Q[0-9](_[A-Z0-9]+)*|BF16|F16|F32' \
    | tail -n 1 | tr '[:lower:]' '[:upper:]'
}

# Round a MiB count to whole GB (97887 -> 96). Prints nothing for non-numbers.
mib_to_gb() {
  case "$1" in ''|*[!0-9]*) return 0 ;; esac
  printf '%s' $(( ($1 + 512) / 1024 ))
}

# Pull the most recent generation speed (tok/s, 1 decimal) out of llama.cpp
# journal output: the last "eval time" timing line — NOT the prefill line,
# which reads "prompt eval time" and reports a much higher tok/s.
parse_tps() {
  printf '%s\n' "$1" \
    | grep -F 'tokens per second' | grep -F 'eval time =' | grep -Fv 'prompt eval time' \
    | tail -n 1 \
    | sed -nE 's/.*[ (]([0-9]+\.?[0-9]*) tokens per second.*/\1/p' \
    | LC_ALL=C awk '{ printf "%.1f", $0 }'
}

# Best-effort telemetry, gathered fresh before every ping. Every collector
# tolerates a missing source (no GPU, no llama.cpp, no journald) by leaving
# its variable empty, which omits the field from the ping body.
collect_telemetry() {
  DEVICE=""; VRAM_TOTAL=""; VRAM_USED=""; MODEL=""; QUANT=""; TPS=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    if [ -n "$_gpu" ]; then
      DEVICE=$(printf '%s' "$_gpu" | awk -F', *' '{print $1}')
      VRAM_TOTAL=$(mib_to_gb "$(printf '%s' "$_gpu" | awk -F', *' '{print $2}')")
      VRAM_USED=$(mib_to_gb "$(printf '%s' "$_gpu" | awk -F', *' '{print $3}')")
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    MODEL=$(curl -fsS --max-time 5 "$LLAMA_ENDPOINT/v1/models" 2>/dev/null \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["models"][0]["name"])' 2>/dev/null)
    _mp=$(curl -fsS --max-time 5 "$LLAMA_ENDPOINT/props" 2>/dev/null \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["model_path"])' 2>/dev/null)
    if [ -n "$_mp" ]; then QUANT=$(parse_quant "${_mp##*/}"); fi
  fi

  if command -v journalctl >/dev/null 2>&1; then
    TPS=$(parse_tps "$(journalctl -u "$LLAMA_UNIT" --no-pager -n 400 -o cat 2>/dev/null)")
  fi
}

# Build the ping JSON: the four base fields are mandatory and unchanged (the
# Ed25519 signature still covers only "nodeId:timestamp"); telemetry fields
# are appended only when collected.
build_ping_body() { # nodeId publicKey signature timestamp
  printf '{"nodeId":"%s","publicKey":"%s","signature":"%s","timestamp":%s' \
    "$1" "$2" "$3" "$4"
  json_str_field device "$DEVICE"
  json_num_field vramTotal "$VRAM_TOTAL"
  json_num_field vramUsed "$VRAM_USED"
  json_str_field model "$MODEL"
  json_str_field quant "$QUANT"
  json_num_field tps "$TPS"
  printf '}'
}

# Sourced with LLMJOB_TEST_MODE set, the script defines its helpers and stops
# before any side effects, so the parsers above can be unit-tested.
if [ -n "${LLMJOB_TEST_MODE:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --token)  TOKEN="$2";  shift 2 ;;
    --name)   NAME="$2";   shift 2 ;;
    --server=*) SERVER="${1#*=}"; shift ;;
    --token=*)  TOKEN="${1#*=}";  shift ;;
    --name=*)   NAME="${1#*=}";   shift ;;
    *) echo "Unknown option: $1" >&2; shift ;;
  esac
done

if [ -z "$TOKEN" ]; then
  echo "Error: --token is required (copy the full command from your dashboard)." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Error: 'curl' is required but was not found." >&2
  exit 1
fi

# Pick an OpenSSL that can actually generate Ed25519 keys. macOS ships
# LibreSSL, which only gained Ed25519 support in 3.3 (macOS 13+); on older
# systems fall back to a Homebrew OpenSSL if one is installed.
OPENSSL=""
for c in openssl \
         /opt/homebrew/opt/openssl/bin/openssl \
         /usr/local/opt/openssl/bin/openssl \
         /opt/homebrew/bin/openssl \
         /usr/local/bin/openssl; do
  command -v "$c" >/dev/null 2>&1 || continue
  _probe=$(mktemp)
  if "$c" genpkey -algorithm ed25519 -out "$_probe" 2>/dev/null; then
    OPENSSL="$c"; rm -f "$_probe"; break
  fi
  rm -f "$_probe"
done
if [ -z "$OPENSSL" ]; then
  echo "Error: no OpenSSL with Ed25519 support was found (need OpenSSL 1.1.1+ or LibreSSL 3.3+)." >&2
  echo "On macOS: run 'brew install openssl' and try again." >&2
  exit 1
fi

CONFIG_DIR="${LLMJOB_CONFIG_DIR:-$HOME/.llmjob}"
KEY="$CONFIG_DIR/node.pem"
mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR" 2>/dev/null || true

if [ ! -f "$KEY" ]; then
  ( umask 077; "$OPENSSL" genpkey -algorithm ed25519 -out "$KEY" 2>/dev/null ) || {
    echo "Error: could not generate an Ed25519 key (needs OpenSSL 1.1.1+)." >&2
    exit 1
  }
fi

# Raw 32-byte public key (tail of the SPKI DER), base64; nodeId is its fingerprint.
PUBKEY=$("$OPENSSL" pkey -in "$KEY" -pubout -outform DER 2>/dev/null | tail -c 32 | "$OPENSSL" base64 -A)
NODEID=$(printf '%s' "$PUBKEY" | "$OPENSSL" dgst -sha256 | awk '{print $NF}' | cut -c1-6)

# Persist the nodeId so companion tools (the usage log-shipper) can attribute
# their records to this node without re-deriving it from the key.
printf '%s\n' "$NODEID" > "$CONFIG_DIR/node_id" 2>/dev/null || true

RESP=$(mktemp)
trap 'rm -f "$RESP"' EXIT

# sign <message> -> base64 signature (handles OpenSSL 3 -rawin and 1.1.1)
sign() {
  _m=$(mktemp); _s=$(mktemp)
  printf '%s' "$1" > "$_m"
  if ! "$OPENSSL" pkeyutl -sign -inkey "$KEY" -rawin -in "$_m" -out "$_s" 2>/dev/null; then
    "$OPENSSL" pkeyutl -sign -inkey "$KEY" -in "$_m" -out "$_s" 2>/dev/null
  fi
  "$OPENSSL" base64 -A < "$_s"
  rm -f "$_m" "$_s"
}

# post <path> <json-body> -> prints HTTP status, response body in $RESP
post() {
  curl -sS -o "$RESP" -w '%{http_code}' -X POST "$SERVER$1" \
    -H 'Content-Type: application/json' -d "$2" 2>/dev/null || echo "000"
}

NAME_FINAL=${NAME:-node-$NODEID}
echo "LLMJob node $NODEID -> $SERVER"

join_body=$(printf '{"token":"%s","nodeId":"%s","publicKey":"%s","name":"%s"}' \
  "$(json_escape "$TOKEN")" "$NODEID" "$PUBKEY" "$(json_escape "$NAME_FINAL")")
code=$(post /api/nodes/join "$join_body")
case "$code" in
  200|201) echo "✓ Joined and claimed to your account" ;;
  *) echo "✗ Failed to join (HTTP $code): $(cat "$RESP")" >&2; exit 1 ;;
esac

ping_once() {
  collect_telemetry
  ts=$(( $(date +%s) * 1000 ))
  sig=$(sign "${NODEID}:${ts}")
  body=$(build_ping_body "$NODEID" "$PUBKEY" "$sig" "$ts")
  code=$(post /api/nodes/ping "$body")
  t=$(date '+%H:%M:%S')
  case "$code" in
    200) echo "[$t] ✓ ping" ;;
    *) echo "[$t] ✗ ping failed (HTTP $code): $(cat "$RESP")" >&2 ;;
  esac
}

ping_once
while true; do
  sleep "$PING_INTERVAL"
  ping_once
done
