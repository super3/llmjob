#!/bin/sh
# LLMJob node installer.
#
# Fetches the LLMJob node client from source (we host it; it is not published to
# npm) and joins this machine to your account using a join token from your
# dashboard. The node's secret key is generated locally and never leaves this
# machine.
#
# Usage (copy the exact command from your dashboard's "Add node" dialog):
#   curl -fsSL https://llmjob.example/install.sh | sh -s -- --server <url> --token <token> [--name <name>]

set -e

SERVER="https://llmjob-production.up.railway.app"
TOKEN=""
NAME=""
# Source tarball for the client. Defaults to the repo's main branch; override
# with --source <url> (e.g. to pin a branch or tag) if needed.
SOURCE="https://github.com/super3/llmjob/archive/refs/heads/main.tar.gz"

while [ $# -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --token)  TOKEN="$2";  shift 2 ;;
    --name)   NAME="$2";   shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --server=*) SERVER="${1#*=}"; shift ;;
    --token=*)  TOKEN="${1#*=}";  shift ;;
    --name=*)   NAME="${1#*=}";   shift ;;
    --source=*) SOURCE="${1#*=}"; shift ;;
    *) echo "Unknown option: $1" >&2; shift ;;
  esac
done

if [ -z "$TOKEN" ]; then
  echo "Error: a --token is required. Copy the full command from your dashboard." >&2
  exit 1
fi

for cmd in curl tar node npm; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is required but was not found." >&2
    if [ "$cmd" = "node" ] || [ "$cmd" = "npm" ]; then
      echo "Install Node.js 18+ from https://nodejs.org and re-run this command." >&2
    fi
    exit 1
  fi
done

APP_DIR="${HOME}/.llmjob/app"

echo "Fetching the LLMJob node client..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"
curl -fsSL "$SOURCE" | tar xz -C "$APP_DIR" --strip-components=1

echo "Installing dependencies..."
cd "$APP_DIR/client"
npm install --omit=dev --no-audit --no-fund

echo "Joining this machine to your account..."
if [ -n "$NAME" ]; then
  exec node bin/llmjob-node join --server "$SERVER" --token "$TOKEN" --name "$NAME"
else
  exec node bin/llmjob-node join --server "$SERVER" --token "$TOKEN"
fi
