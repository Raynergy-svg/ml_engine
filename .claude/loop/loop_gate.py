#!/usr/bin/env python3
"""loop_gate.py — OBJECTIVE stopping conditions, computed from disk (LOOP.md).

The loop's stop/continue decision must be measurable, not a vibe. This reads the loop run-state
artifact (.claude/loop/state.json) plus the safety/verify gates and returns exactly one decision:
HALT-SAFETY / STOP-BLOCKED / STOP-DONE / STOP-CHURN / CONTINUE — so "done" and "stalled" cannot be
gamed by narration. The anti-stall detector (STOP-CHURN) is the guard against a high-tier prompt
responder looping with no new verified information.

state.json schema (the worker appends one entry per cycle; /evolve keeps it honest):
  {
    "task": "<short>",
    "open_questions": <int>,            # current count of open load-bearing questions
    "blocked_on_irreversible": <bool>,  # only an operator fork remains (unhalt / live / force-push)
    "cycles": [
      {"n":1, "new_lessons":0, "new_verified_facts":3, "open_questions_after":2, "verdict":"PASS|FAIL|null"},
      ...
    ]
  }

Decision precedence (first match wins), each from a concrete test:
  HALT-SAFETY  : risk monitor ALARM, or a Hard-NO check fails              -> exit 2
  STOP-BLOCKED : blocked_on_irreversible == true                          -> exit 0
  STOP-DONE    : last verdict PASS AND last cycle produced no new info     -> exit 0
                 (new_lessons==0 and new_verified_facts==0) AND open_questions==0
  STOP-CHURN   : >=2 cycles AND last 2 produced no new info AND open_qs    -> exit 0
                 not decreasing  (anti-stall: escalate, don't loop)
  CONTINUE     : otherwise                                                 -> exit 0
Exit 2 only on HALT (so a Stop/PreTool hook can block). Decision is always in stdout JSON.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path("/Users/buddy/Documents/ml_engine")


def _risk_status(repo: Path, override: str | None) -> tuple[str, str]:
    """Return ('green'|'alarm', detail). Override lets tests inject status deterministically."""
    if override:
        return override, f"injected:{override}"
    mon = repo / ".claude/tools/risk_monitor.sh"
    if not mon.exists():
        return "alarm", "risk_monitor.sh missing == unsafe"
    try:
        r = subprocess.run(["bash", str(mon)], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "RISK_MONITOR_REPO": str(repo)})
        return ("green" if r.returncode == 0 else "alarm"), r.stdout.strip().splitlines()[-1] if r.stdout else ""
    except Exception as e:
        return "alarm", f"risk_monitor error == unsafe ({e})"


def decide(state: dict, risk: str) -> tuple[str, str]:
    cycles = state.get("cycles", [])
    open_q = int(state.get("open_questions", 0))

    if risk == "alarm":
        return "HALT-SAFETY", "risk monitor ALARM — stop and address before anything else"
    if state.get("blocked_on_irreversible"):
        return "STOP-BLOCKED", "only an irreversible/operator fork remains — escalate, do not self-authorize"
    if not cycles:
        return "CONTINUE", "no cycles recorded yet — run the first cycle"

    last = cycles[-1]
    last_new = int(last.get("new_lessons", 0)) + int(last.get("new_verified_facts", 0))
    if last.get("verdict") == "PASS" and last_new == 0 and open_q == 0:
        return "STOP-DONE", "verifier PASS, no new info this cycle, zero open questions"

    if len(cycles) >= 2:
        prev = cycles[-2]
        prev_new = int(prev.get("new_lessons", 0)) + int(prev.get("new_verified_facts", 0))
        oq_last = int(last.get("open_questions_after", open_q))
        oq_prev = int(prev.get("open_questions_after", open_q))
        if last_new == 0 and prev_new == 0 and oq_last >= oq_prev:
            return ("STOP-CHURN",
                    f"2 consecutive cycles with no new verified info and open questions not "
                    f"decreasing ({oq_prev}->{oq_last}) — stalled; escalate to operator")

    return "CONTINUE", "progress is being made (new info this cycle or open questions shrinking)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--state", default=None, help="path to loop state.json (default <repo>/.claude/loop/state.json)")
    ap.add_argument("--risk-status", default=None, choices=["green", "alarm"], help="inject risk status (tests)")
    a = ap.parse_args()
    repo = Path(a.repo)
    state_path = Path(a.state) if a.state else (repo / ".claude/loop/state.json")

    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception as e:
        # Fail-closed: an unreadable loop state is treated as a halt, not a free pass.
        print(json.dumps({"decision": "HALT-SAFETY",
                          "reason": f"loop state unreadable == unsafe ({e})"}))
        return 2

    risk, risk_detail = _risk_status(repo, a.risk_status)
    decision, reason = decide(state, risk)
    print(json.dumps({"decision": decision, "reason": reason,
                      "risk": risk, "risk_detail": risk_detail,
                      "cycles": len(state.get("cycles", [])),
                      "open_questions": state.get("open_questions")}, indent=2))
    return 2 if decision == "HALT-SAFETY" else 0


if __name__ == "__main__":
    sys.exit(main())
