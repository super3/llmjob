#!/bin/sh
# LLMJob node agent — pure shell, no Node and no npm. Needs only curl + openssl
# (preinstalled on most Linux). Served by the app and meant to be run from a pipe:
#
#   curl -fsSL <server>/install.sh/<token> | bash
#
# The app bakes the server and join token into this script when it serves the
# per-account URL, so no arguments are needed. You can still override with
# --server <url> / --token <token> / --name <name>.
#
# It creates an Ed25519 key locally (only the public key ever leaves the
# machine), joins this machine to your account with the join token, then pings
# so the node shows as online.

SERVER="https://llmjob-production.up.railway.app"
TOKEN=""
NAME=""
PING_INTERVAL=300  # seconds

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

RESP=$(mktemp)
trap 'rm -f "$RESP"' EXIT

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

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
  ts=$(( $(date +%s) * 1000 ))
  sig=$(sign "${NODEID}:${ts}")
  body=$(printf '{"nodeId":"%s","publicKey":"%s","signature":"%s","timestamp":%s}' \
    "$NODEID" "$PUBKEY" "$sig" "$ts")
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
