#!/bin/bash
# Remove the unit and the dashboard, restoring whatever was there before.
#
#   sudo ./uninstall.sh
#
# /etc/default/llmjob-earn is left alone: it holds your address, and removing a
# dashboard should not lose it.
set -euo pipefail
UNIT=llmjob-earn.service

[ "$(id -u)" -eq 0 ] || { echo "run me with sudo" >&2; exit 1; }

systemctl stop "$UNIT" 2>/dev/null || true
systemctl disable "$UNIT" 2>/dev/null || true
rm -f "/etc/systemd/system/$UNIT"
systemctl daemon-reload

if [ -f /usr/local/bin/gpu-dashboard.pre-llmjob ]; then
  mv /usr/local/bin/gpu-dashboard.pre-llmjob /usr/local/bin/gpu-dashboard
  echo "restored your previous dashboard"
else
  rm -f /usr/local/bin/gpu-dashboard
fi

echo "done. $UNIT removed; /etc/default/llmjob-earn left in place."
