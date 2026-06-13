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

# Detect the system package manager. Prints its name, fails when none found.
detect_pkg_manager() {
  for _pm in apt-get dnf yum pacman zypper apk brew; do
    if command -v "$_pm" >/dev/null 2>&1; then printf '%s' "$_pm"; return 0; fi
  done
  return 1
}

# Install a package with the detected manager, non-interactively. Uses sudo
# without a password prompt (-n) when not root, so a piped install can never
# hang waiting for input; fails quietly when installation isn't possible.
pkg_install() {
  _pm=$(detect_pkg_manager) || return 1
  _sudo=""
  if [ "$(id -u 2>/dev/null)" != "0" ] && [ "$_pm" != "brew" ]; then
    command -v sudo >/dev/null 2>&1 || return 1
    _sudo="sudo -n"
  fi
  case "$_pm" in
    apt-get) $_sudo apt-get update -qq >/dev/null 2>&1
             $_sudo apt-get install -y -qq "$1" >/dev/null 2>&1 ;;
    dnf|yum) $_sudo "$_pm" install -y "$1" >/dev/null 2>&1 ;;
    pacman)  $_sudo pacman -Sy --noconfirm "$1" >/dev/null 2>&1 ;;
    zypper)  $_sudo zypper --non-interactive install "$1" >/dev/null 2>&1 ;;
    apk)     $_sudo apk add "$1" >/dev/null 2>&1 ;;
    brew)    brew install "$1" >/dev/null 2>&1 ;;
  esac
}

# Make sure command $1 exists, attempting to install package $2 (defaults to
# the command name) when it doesn't. Succeeds only if the command ends up
# available; the caller decides whether that is fatal.
ensure_dep() {
  command -v "$1" >/dev/null 2>&1 && return 0
  echo "Dependency '$1' not found — attempting to install '${2:-$1}'..."
  pkg_install "${2:-$1}" \
    || echo "Warning: could not install '${2:-$1}' automatically." >&2
  command -v "$1" >/dev/null 2>&1
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
if ! ensure_dep curl curl; then
  echo "Error: 'curl' is required and could not be installed automatically." >&2
  exit 1
fi

# python3 is only used to parse JSON from the local llama.cpp server; without
# it the model/quant telemetry fields are simply omitted from pings.
_py_pkg=python3
[ "$(detect_pkg_manager 2>/dev/null)" = "pacman" ] && _py_pkg=python
if ! ensure_dep python3 "$_py_pkg"; then
  echo "Note: 'python3' is unavailable; model/quant telemetry will be omitted." >&2
fi

# Pick an OpenSSL that can actually generate Ed25519 keys. macOS ships
# LibreSSL, which only gained Ed25519 support in 3.3 (macOS 13+); on older
# systems fall back to a Homebrew OpenSSL if one is installed.
find_openssl() {
  for c in openssl \
           /opt/homebrew/opt/openssl/bin/openssl \
           /usr/local/opt/openssl/bin/openssl \
           /opt/homebrew/bin/openssl \
           /usr/local/bin/openssl; do
    command -v "$c" >/dev/null 2>&1 || continue
    _probe=$(mktemp)
    if "$c" genpkey -algorithm ed25519 -out "$_probe" 2>/dev/null; then
      rm -f "$_probe"; printf '%s' "$c"; return 0
    fi
    rm -f "$_probe"
  done
  return 1
}

OPENSSL=$(find_openssl) || OPENSSL=""
if [ -z "$OPENSSL" ]; then
  # Either openssl is missing or it's a LibreSSL without Ed25519 — try to
  # install a modern one and probe again.
  echo "No Ed25519-capable OpenSSL found — attempting to install 'openssl'..."
  pkg_install openssl || echo "Warning: could not install 'openssl' automatically." >&2
  OPENSSL=$(find_openssl) || OPENSSL=""
fi
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
