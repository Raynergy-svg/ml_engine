#!/usr/bin/env bash
# Uninstall the Buddy TUI launchd user agent.
set -euo pipefail

LABEL="com.buddy.trader"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [ -f "${PLIST_PATH}" ]; then
    echo "Unloading ${LABEL}..."
    launchctl unload -w "${PLIST_PATH}" 2>/dev/null || true
    rm -f "${PLIST_PATH}"
    echo "Removed ${PLIST_PATH}"
else
    echo "No plist found at ${PLIST_PATH} — nothing to uninstall."
fi

echo ""
echo "Verify removal:"
echo "  launchctl list | grep ${LABEL} || echo 'not loaded'"
