#!/bin/sh
# LLMJob node installer.
#
# Installs the LLMJob node client and joins this machine to your account using
# a join token from your dashboard. The node's secret key is generated locally
# and never leaves this machine.
#
# Usage (copy the exact command from your dashboard's "Add node" dialog):
#   curl -fsSL https://llmjob.example/install.sh | sh -s -- --server <url> --token <token> [--name <name>]

set -e

SERVER="https://llmjob-production.up.railway.app"
TOKEN=""
NAME=""

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
  echo "Error: a --token is required. Copy the full command from your dashboard." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm (Node.js) is required but was not found." >&2
  echo "Install Node.js 18+ from https://nodejs.org and re-run this command." >&2
  exit 1
fi

echo "Installing the LLMJob node client..."
npm install -g llmjob-node

echo "Joining this machine to your account..."
if [ -n "$NAME" ]; then
  exec llmjob-node join --server "$SERVER" --token "$TOKEN" --name "$NAME"
else
  exec llmjob-node join --server "$SERVER" --token "$TOKEN"
fi
