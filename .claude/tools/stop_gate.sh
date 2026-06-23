#!/usr/bin/env bash
# stop_gate.sh — Stop hook: runs the risk monitor at turn-end, deterministically, regardless of
# whether the model remembered to. This makes the one genuinely-inert piece (risk_monitor.sh) fire
# on its own. Fail-closed per operator ethos: on ALARM it BLOCKS the stop (exit 2) and feeds the
# alarm back to the model to address.
#
# Loop guard: Claude sets stop_hook_active=true on the continuation it triggers; we exit 0 then, so
# this blocks AT MOST once per stop attempt — it surfaces the alarm and forces one corrective turn,
# it does NOT trap the session. It is an ALERTING tripwire (defense-in-depth); the hard trading
# guards remain in code (execution.py halt, HARD_MAX_GAP quarantine).
#
# Reads the Claude hook JSON on stdin. REPO overridable for tests (RISK_MONITOR_REPO).

set -uo pipefail
input="$(cat 2>/dev/null || true)"

# Loop guard — never block a stop that is itself the product of a prior stop-hook block.
active="$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("stop_hook_active", False))
except Exception: print(False)' 2>/dev/null || echo False)"
if [ "$active" = "True" ]; then
  exit 0
fi

REPO="${RISK_MONITOR_REPO:-/Users/buddy/Documents/ml_engine}"
MON="$REPO/.claude/tools/risk_monitor.sh"
if [ ! -x "$MON" ]; then
  # Missing tripwire == unsafe: block and say so.
  echo "STOP-GATE: risk_monitor.sh missing/non-executable at $MON — cannot verify safety; blocking." >&2
  exit 2
fi

out="$(RISK_MONITOR_REPO="$REPO" "$MON" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  {
    echo "STOP-GATE: HALT-SAFETY — risk monitor ALARM. Address before ending the turn:"
    echo "$out"
  } >&2
  exit 2
fi
exit 0
