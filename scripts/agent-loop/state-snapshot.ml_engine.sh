#!/usr/bin/env bash
# state-snapshot.sh (ml_engine) — writes STATE.md: a lean, current picture the agent
# reads at the top of each loop. Includes the SAFETY POSTURE up front so the agent
# always sees whether the system is fail-closed. Read-only: never mutates state.
#
# Usage:  ./state-snapshot.sh /path/to/ml_engine
set -uo pipefail
REPO="${1:-$(pwd)}"
cd "$REPO" || { echo "Cannot cd to $REPO"; exit 1; }
OUT="$REPO/STATE.md"

section() {
  local title="$1"; shift
  { echo "### $title"; echo '```'; } >> "$OUT"
  if "$@" > /tmp/_snap 2>&1; then echo "STATUS: PASS" >> "$OUT"
  else echo "STATUS: FAIL (exit $?)" >> "$OUT"; fi
  tail -n 25 /tmp/_snap >> "$OUT"
  { echo '```'; echo; } >> "$OUT"
}

{
  echo "# ml_engine PROJECT STATE"
  echo "_Generated: $(date -u '+%Y-%m-%d %H:%M:%SZ')_"
  echo
  echo "## SAFETY POSTURE (read-only)"
  echo '```'
  python - <<'PY' 2>/dev/null || echo "could not read .claude/state.json"
import json
try:
    s = json.load(open(".claude/state.json"))
    print("state.json  halted:", s.get("halted"), "| mode:", s.get("mode"), "| status:", s.get("status"))
except Exception as e:
    print("state.json unreadable:", e)
try:
    h = json.load(open(".claude/heartbeat.json"))
    print("heartbeat   alive:", h.get("scanner_alive"), "| mode:", h.get("mode"), "(stale; state.json is authoritative)")
except Exception as e:
    print("heartbeat unreadable:", e)
PY
  echo -n "config oanda_environment: "
  grep -n "oanda_environment" src/scanner/config.py 2>/dev/null | head -1 || echo "not found"
  echo "REQUIRED for loop to run: oanda_environment=practice (demo). halted/mode not gated."
  echo '```'
  echo
  echo "## Git"
  echo '```'
  echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "Clean:   $([ -z "$(git status --porcelain)" ] && echo yes || echo 'NO — uncommitted')"
  git log --oneline -5 2>/dev/null
  git diff --stat 2>/dev/null | tail -n 12
  echo '```'
  echo
  echo "## Open work markers"
  echo '```'
  echo "TODO/FIXME/HACK: $(grep -rIn -E 'TODO|FIXME|HACK' --include='*.py' src tests 2>/dev/null | wc -l | tr -d ' ')"
  echo '```'
  echo
  echo "## Loop budget"
  echo '```'
  if [ -f .loop/cost_usd ]; then
    echo "cumulative loop spend: \$$(cat .loop/cost_usd) (cap: LOOP_MAX_COST_USD, runner-enforced)"
  else
    echo "no spend recorded yet"
  fi
  echo '```'
  echo
} > "$OUT"

section "pytest"  python -m pytest tests/ -q --tb=line
section "flake8"  python -m flake8 src/ --config=.flake8

echo "Wrote $OUT"
