#!/usr/bin/env bash
# US-603: External heartbeat watchdog daemon.
#
# Reads .claude/heartbeat.json every CHECK_INTERVAL_S seconds. If the timestamp
# is older than STALE_THRESHOLD_S, kickstart the com.buddy.trader launchd
# service. Each check appends a 1-line JSON record to logs/watchdog.log.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEARTBEAT_FILE="${REPO_DIR}/.claude/heartbeat.json"
LOG_DIR="${REPO_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
SERVICE_LABEL="com.buddy.trader"
CHECK_INTERVAL_S="${WATCHDOG_CHECK_INTERVAL_S:-20}"
STALE_THRESHOLD_S="${WATCHDOG_STALE_THRESHOLD_S:-60}"

mkdir -p "${LOG_DIR}"

# Convert ISO8601 (with optional trailing Z or +00:00) to epoch seconds.
iso_to_epoch() {
    local iso="$1"
    # Strip fractional seconds and normalize timezone for /usr/bin/date.
    local clean="${iso%%.*}"
    clean="${clean//Z/+0000}"
    clean="${clean//+00:00/+0000}"
    # macOS /usr/bin/date accepts -j -f
    /usr/bin/date -j -f "%Y-%m-%dT%H:%M:%S%z" "${clean}" "+%s" 2>/dev/null || echo "0"
}

log_event() {
    local status="$1"
    local age="$2"
    local action="$3"
    local now_iso
    now_iso="$(/usr/bin/date -u "+%Y-%m-%dT%H:%M:%SZ")"
    printf '{"ts":"%s","status":"%s","age_s":%s,"action":"%s","heartbeat":"%s"}\n' \
        "${now_iso}" "${status}" "${age}" "${action}" "${HEARTBEAT_FILE}" >> "${LOG_FILE}"
}

restart_service() {
    /bin/launchctl kickstart -k "gui/$(/usr/bin/id -u)/${SERVICE_LABEL}" >/dev/null 2>&1 || true
}

while true; do
    now_epoch="$(/usr/bin/date -u "+%s")"

    if [ ! -f "${HEARTBEAT_FILE}" ]; then
        log_event "missing" "-1" "kickstart"
        restart_service
        sleep "${CHECK_INTERVAL_S}"
        continue
    fi

    ts_iso="$(/usr/bin/python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get("ts_iso",""))
except Exception:
    print("")' "${HEARTBEAT_FILE}" 2>/dev/null || echo "")"

    if [ -z "${ts_iso}" ]; then
        log_event "unreadable" "-1" "kickstart"
        restart_service
        sleep "${CHECK_INTERVAL_S}"
        continue
    fi

    hb_epoch="$(iso_to_epoch "${ts_iso}")"
    age=$((now_epoch - hb_epoch))

    if [ "${hb_epoch}" -eq 0 ] || [ "${age}" -gt "${STALE_THRESHOLD_S}" ]; then
        log_event "stale" "${age}" "kickstart"
        restart_service
    else
        log_event "ok" "${age}" "noop"
    fi

    sleep "${CHECK_INTERVAL_S}"
done
