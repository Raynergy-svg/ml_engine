#!/usr/bin/env bash
# Parallel risk monitor — independent safety tripwire for the self-improver loop (LOOP.md).
#
# Runs CONCURRENTLY with any work session: foreground spot-check, or `run_in_background: true`
# so safety does not depend on the worker remembering to pause and look. Re-derives every safety
# invariant straight from DISK — it never trusts the worker's narrative or in-memory beliefs.
#
# Exit 0 = GREEN  (all invariants hold)
# Exit 2 = ALARM  (>=1 invariant violated, OR state unreadable) -> the loop MUST HALT (HALT-SAFETY).
# Fail-closed: unknown == unsafe. macOS/BSD-grep safe (uses [[:space:]], ERE via -E).

set -uo pipefail
# REPO is overridable (RISK_MONITOR_REPO) so the tripwire can be unit-tested against
# synthetic fixtures without touching the real tree. Defaults to the live repo.
REPO="${RISK_MONITOR_REPO:-/Users/buddy/Documents/ml_engine}"
cd "$REPO" 2>/dev/null || { echo "RISK MONITOR: ALARM — cannot cd $REPO"; exit 2; }

STATE=".claude/state.json"
CONFIG="src/scanner/config.py"
EXEC="src/scanner/execution.py"
RULES=".claude/rules/improvement.md"
alarms=()

# --- HARD NO 1: oanda_environment must default to "practice" (L-003) ---
if ! grep -Eq 'oanda_environment:[[:space:]]*str[[:space:]]*=[[:space:]]*"practice"' "$CONFIG"; then
  alarms+=("oanda_environment default is no longer \"practice\" in ${CONFIG} (HARD NO / L-003)")
fi
# working-tree must not flip it to live, nor introduce the live OANDA endpoint
if git diff -- "$CONFIG" 2>/dev/null | grep -Eq '^\+.*oanda_environment.*"live"'; then
  alarms+=("working tree sets oanda_environment=\"live\" (HARD NO / L-003)")
fi
if git diff 2>/dev/null | grep -Eq '^\+.*api-fxtrade\.oanda\.com'; then
  alarms+=("working tree introduces live OANDA endpoint api-fxtrade.oanda.com (HARD NO)")
fi

# --- HARD NO 2: ship gate must stay <= 0.10 (L-002) ---
# Search src AND scripts (the constant actually lives in scripts/ — verified 2026-06-23).
if grep -REq 'HARD_MAX_GAP[[:space:]]*=[[:space:]]*0\.1[1-9]|HARD_MAX_GAP[[:space:]]*=[[:space:]]*0\.[2-9]' src scripts 2>/dev/null; then
  alarms+=("HARD_MAX_GAP raised above 0.10 in src/ or scripts/ (HARD NO / L-002)")
fi

# --- HARD NO 3: halt enforcement (L-004) — mid-cycle guard must still exist ---
# Require BOTH the guard's log string AND the get_halted() call: a rename or a gutted
# guard that leaves a stale log line must not pass.
if ! grep -q "execute_trade BLOCKED" "$EXEC" 2>/dev/null; then
  alarms+=("mid-cycle halt guard log missing from ${EXEC} — halt enforcement may be removed (L-004)")
fi
if ! grep -q "get_halted()" "$EXEC" 2>/dev/null; then
  alarms+=("get_halted() call missing from ${EXEC} — halt is no longer checked in the trade path (L-004)")
fi

# --- HARD NO 4: no real money / no live env — no oanda_environment="live" ASSIGNMENT in src/scripts.
# (The annotated default `oanda_environment: str = "practice"` and the live-URL constant are NOT
# matched — only an actual assignment to "live", which is the flip that matters.)
if grep -rEq --exclude-dir=_quarantine 'oanda_environment[[:space:]]*=[[:space:]]*"live"' src scripts 2>/dev/null; then
  alarms+=("oanda_environment assigned \"live\" in src/scripts (HARD NO — no real money)")
fi

# --- Runtime state must be readable; live mode is incoherent on this practice/halted system ---
halted="?"; mode="?"
if [ ! -r "$STATE" ]; then
  alarms+=("cannot read ${STATE} — runtime state UNKNOWN == unsafe (fail-closed)")
else
  halted="$(python3 -c "import json;print(json.load(open('$STATE')).get('halted'))" 2>/dev/null || echo ERR)"
  mode="$(python3 -c "import json;print(json.load(open('$STATE')).get('mode'))" 2>/dev/null || echo ERR)"
  if [ "$halted" = "ERR" ] || [ "$mode" = "ERR" ]; then
    alarms+=("${STATE} unparseable — state UNKNOWN == unsafe")
  elif [ "$mode" = "live" ] && ! grep -Eq 'oanda_environment:[[:space:]]*str[[:space:]]*=[[:space:]]*"practice"' "$CONFIG"; then
    # mode=live is EXECUTION mode (place orders), NOT real money. Real money == oanda_environment=live,
    # caught hard by HARD NO 1/4 above. Operator-directed un-halt on PRACTICE runs mode=live on the
    # demo/paper account (broker is OandaPracticeClient, api-fxpractice-only). So flag here ONLY when
    # env is NOT practice = real-money execution. (operator-directed 2026-06-24; practice account.)
    alarms+=("state.mode=live AND oanda_environment NOT practice — REAL-MONEY EXECUTION (HARD NO)")
  fi
fi

if [ ${#alarms[@]} -eq 0 ]; then
  echo "RISK MONITOR: GREEN — practice default ok · ship-gate<=0.10 · halt-guard present · state readable (halted=${halted} mode=${mode})"
  exit 0
fi
echo "RISK MONITOR: ALARM (${#alarms[@]}) — self-improver loop must HALT (HALT-SAFETY):"
for a in "${alarms[@]}"; do echo "  ✗ ${a}"; done
exit 2
