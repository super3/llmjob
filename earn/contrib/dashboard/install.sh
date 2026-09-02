#!/bin/bash
# Install the llmjob-earn unit and the gpu-dashboard script.
#
#   sudo ./install.sh
#
# Reverse with ./uninstall.sh. Requires /etc/default/llmjob-earn to exist —
# copy llmjob-earn.default.example and edit it first.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE=/etc/default/llmjob-earn
UNIT=llmjob-earn.service

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo" >&2; exit 1; }

if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE — copy llmjob-earn.default.example there and edit it first" >&2
  exit 1
fi
# A unit that starts without an address just fails in a loop, so say so here.
grep -qE '^EARN_ADDRESS=prl1p.' "$ENV_FILE" || {
  echo "set a real EARN_ADDRESS in $ENV_FILE first" >&2; exit 1; }
grep -q '^EARN_BIN=' "$ENV_FILE" || { echo "set EARN_BIN in $ENV_FILE first" >&2; exit 1; }

# Never clobber an existing dashboard without a copy to go back to.
if [ -f /usr/local/bin/gpu-dashboard ] && [ ! -f /usr/local/bin/gpu-dashboard.pre-llmjob ]; then
  cp -a /usr/local/bin/gpu-dashboard /usr/local/bin/gpu-dashboard.pre-llmjob
  echo "kept your previous dashboard at /usr/local/bin/gpu-dashboard.pre-llmjob"
fi

# The unit has to name a concrete user; take the one that invoked sudo.
RUN_USER="${SUDO_USER:-root}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
[ -n "$RUN_HOME" ] || RUN_HOME=/
sed -e "s|__EARN_USER__|$RUN_USER|" -e "s|__EARN_HOME__|$RUN_HOME|" \
  "$D/llmjob-earn.service" > "/etc/systemd/system/$UNIT"
chmod 0644 "/etc/systemd/system/$UNIT"
echo "unit will run as $RUN_USER" 
install -m 0755 "$D/gpu-dashboard" /usr/local/bin/gpu-dashboard
chmod 0644 "$ENV_FILE"
systemctl daemon-reload
systemctl enable "$UNIT"
systemctl restart "$UNIT"

sleep 5
echo
echo "  unit: $(systemctl is-active "$UNIT")"
echo "  mode: $(grep -E '^EARN_MODE=' "$ENV_FILE" || echo 'EARN_MODE unset (defaults to auto)')"
echo
echo "Run 'gpu-dashboard' in a terminal at least 128 columns wide."
