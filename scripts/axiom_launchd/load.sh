#!/usr/bin/env bash
# Install + load the AXIOM / Buddy LaunchAgents so a reboot or logout brings the
# whole stack back automatically. Idempotent: re-running re-installs and reloads.
#
# Services:
#   com.axiom.api   — FastAPI read/control data layer  (:8888, loopback-only)
#   com.axiom.web   — Next.js operator cockpit          (:51999)
#   com.buddy.trend — OANDA daily-trend loop            (PRACTICE; respects halt)
#   com.buddy.tier7 — Tier 7 bounded self-heal loop     (respects halt + practice pin)
#
# NOTE: com.buddy.learning_loop.plist (market-closed continual-learning batch,
# scripts/offline_learning_cycle.py) is deliberately NOT in this script's
# LABELS list. It is a scheduled batch job, not a resident daemon — bundling
# it here would mean every routine reload of the four services above also
# silently installs/enables it. Activate it as an explicit, separate step
# (see README.md "Enabling the learning-loop schedule") once you've decided
# you want it resident-scheduled.
#
# SAFETY INVARIANT (see README.md): KeepAlive restores the operator's LAST state only.
# The trading loops re-check state.json halt + the practice pin EVERY cycle, so a
# respawn NEVER trades while halted and NEVER flips off practice. KeepAlive is process
# resilience, not a halt bypass.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
LABELS=(com.axiom.api com.axiom.web com.buddy.trend com.buddy.tier7)

mkdir -p "$DEST"
mkdir -p "$(cd "$SRC_DIR/../.." && pwd)/trained_data/axiom"

for label in "${LABELS[@]}"; do
  plist="$SRC_DIR/$label.plist"
  dest="$DEST/$label.plist"
  cp "$plist" "$dest"
  # Tear down any prior copy (ignore errors if not loaded), then bootstrap fresh.
  launchctl bootout "gui/$UID_NUM/$label" 2>/dev/null || true
  if launchctl bootstrap "gui/$UID_NUM" "$dest" 2>/dev/null; then
    echo "bootstrapped $label"
  else
    # Fallback for older launchctl semantics.
    launchctl load -w "$dest" && echo "loaded $label (legacy)"
  fi
  launchctl enable "gui/$UID_NUM/$label" 2>/dev/null || true
done

echo "---"
echo "Loaded. Status:"
for label in "${LABELS[@]}"; do
  printf '  %-18s ' "$label"
  launchctl print "gui/$UID_NUM/$label" 2>/dev/null | awk -F'= ' '/state =/{print "state="$2; f=1} END{if(!f)print "(not found)"}'
done
