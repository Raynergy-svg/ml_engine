#!/usr/bin/env bash
# Install launchd user agent that keeps the Buddy TUI running across reboots.
# Plist is written to ~/Library/LaunchAgents/com.buddy.trader.plist.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.buddy.trader"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${REPO_DIR}/logs"
STDOUT_LOG="${LOG_DIR}/launchd_tui.stdout.log"
STDERR_LOG="${LOG_DIR}/launchd_tui.stderr.log"
TMUX_BIN="$(command -v tmux || echo /opt/homebrew/bin/tmux)"
BUDDY_BIN="${REPO_DIR}/buddy"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PLIST_PATH}")"

if [ ! -x "${BUDDY_BIN}" ]; then
    echo "ERROR: ${BUDDY_BIN} is not executable" >&2
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
        <string>${TMUX_BIN}</string>
        <string>new-session</string>
        <string>-A</string>
        <string>-d</string>
        <string>-s</string>
        <string>buddy</string>
        <string>${BUDDY_BIN}</string>
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
    <integer>60</integer>
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

echo "Wrote launchd plist: ${PLIST_PATH}"
echo ""
echo "To load the agent:"
echo "  launchctl load -w ${PLIST_PATH}"
echo ""
echo "To check status:"
echo "  launchctl list | grep ${LABEL}"
echo ""
echo "To attach to the running TUI:"
echo "  tmux attach -t buddy"
echo ""
echo "Logs:"
echo "  tail -f ${STDOUT_LOG}"
echo "  tail -f ${STDERR_LOG}"
