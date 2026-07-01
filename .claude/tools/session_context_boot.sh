#!/usr/bin/env bash
# SessionStart boot: prompt the main AI to load the self-evolving context system and,
# when the task warrants, dispatch domain specialist sub-agents.
#
# Hooks emit context to stdout — they cannot literally spawn Claude agents — so this
# realizes "session spawns agents to prompt main AI" by INSTRUCTING the main AI to read
# the five files first and to fan out to specialists per the working rules.
#
# Additive + fail-safe: prints a short context block, never mutates state, always exits 0
# so it can never block a session. Registered as a second SessionStart hook alongside the
# existing tmux hook in .claude/settings.json.

set -uo pipefail
REPO="/Users/buddy/Documents/ml_engine"
STATE="$REPO/.claude/state.json"

# Pull live halt/mode/env straight from disk so the boot prompt reflects reality, not memory.
halted="unknown"; mode="unknown"
if [ -r "$STATE" ]; then
  halted="$(grep -o '"halted"[[:space:]]*:[[:space:]]*[a-z]*' "$STATE" | grep -o '[a-z]*$' || echo unknown)"
  mode="$(grep -o '"mode"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATE" | sed 's/.*"\([^"]*\)"$/\1/' || echo unknown)"
fi

cat <<EOF
=== BUDDY CONTEXT BOOT ===
Before planning, read the self-evolving context system in this order:
  1. .claude/INTENT.md   — operator's voice, standing decisions, definition of done
  2. .claude/NOTES.md    — live state + in-flight work (the ONLY file you may update unprompted)
  3. .claude/LESSONS.md  — permanent failure modes (read before planning, never break a Hard NO)
  4. CLAUDE.md           — lean entry: Tier 6/7, Hard NOs, pointers

LIVE STATE (from .claude/state.json): halted=${halted}  mode=${mode}  oanda_environment=practice (immutable)
HARD NOs (never relax, even via /evolve): practice-only · respect halt · ship-gate before champion · no real money.

SELF-IMPROVER LOOP (.claude/LOOP.md): orient -> plan (bias to action on the load-bearing question)
  -> act -> VERIFY (independent) -> evolve -> DECIDE via objective stopping conditions
  (HALT-SAFETY / STOP-BLOCKED / STOP-DONE / STOP-CHURN / CONTINUE). Don't stop on "feels done";
  don't churn — STOP-CHURN escalates after 2 empty cycles.

WORKING RULES:
  - Plan multi-step work first; chain to a real, verified result; show evidence (file:line / log / number), not assertions.
  - VERIFY INDEPENDENTLY, not by self-report: /verify-task (separate Code Reviewer agent) PASS
    + run .claude/tools/risk_monitor.sh (background) GREEN before declaring done. ALARM => HALT.
  - Separate the roles (worker != verifier != risk monitor). Dispatch DOMAIN specialist sub-agents
    (never general-purpose): TUI->Frontend Developer, Arch->Software Architect, Review->Code Reviewer,
    Tests->API Tester, Data->Data Engineer, Security->Security Engineer, Explore->Explore, Plan->Plan.
  - After a real task, run /evolve to fold what you learned back into INTENT / LESSONS / a skill / NOTES.
EOF

exit 0
