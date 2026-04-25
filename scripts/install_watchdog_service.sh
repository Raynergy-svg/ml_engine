#!/usr/bin/env bash
# US-603: Install launchd user agent for the heartbeat watchdog daemon.
# Plist is written to ~/Library/LaunchAgents/com.buddy.watchdog.plist.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.buddy.watchdog"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${REPO_DIR}/logs"
STDOUT_LOG="${LOG_DIR}/watchdog.stdout.log"
STDERR_LOG="${LOG_DIR}/watchdog.stderr.log"
WATCHDOG_BIN="${REPO_DIR}/scripts/heartbeat_watchdog.sh"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PLIST_PATH}")"

if [ ! -x "${WATCHDOG_BIN}" ]; then
    chmod +x "${WATCHDOG_BIN}" 2>/dev/null || true
fi
if [ ! -x "${WATCHDOG_BIN}" ]; then
    echo "ERROR: ${WATCHDOG_BIN} is not executable" >&2
    exit 1
fi

cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${WATCHDOG_BIN}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ExitTimeOut</key>
    <integer>30</integer>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>${STDOUT_LOG}</string>
    <key>StandardErrorPath</key>
    <string>${STDERR_LOG}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
PLIST

chmod 644 "${PLIST_PATH}"

echo "Wrote watchdog plist: ${PLIST_PATH}"
echo ""
echo "To load: launchctl load -w ${PLIST_PATH}"
echo "Logs:    tail -f ${LOG_DIR}/watchdog.log"
