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
  STOP-CHURN   : >=2 cycles AND over the last min(3) cycles open questions  -> exit 0
                 did NOT net-decrease AND no lesson was learned (anti-stall:
                 progress = close a question OR learn a lesson; self-reported
                 "facts" that do neither don't count — closes the masked-stall
                 evasion of the old 2-empty-cycle rule). Escalate, don't loop.
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
        # Rolling window: progress over the window must CLOSE an open question or LEARN a lesson.
        # Self-reported new_verified_facts that do neither are not progress — this closes the
        # masked-stall evasion where one trivial fact per cycle dodged the old 2-empty-cycle rule.
        window = cycles[-3:] if len(cycles) >= 3 else cycles[-2:]
        oq_start = int(window[0].get("open_questions_after", open_q))
        oq_end = int(window[-1].get("open_questions_after", open_q))
        w_lessons = sum(int(c.get("new_lessons", 0)) for c in window)
        if oq_end >= oq_start and w_lessons == 0:
            return ("STOP-CHURN",
                    f"no net progress over last {len(window)} cycles (open questions "
                    f"{oq_start}->{oq_end} not decreasing, no lesson learned) — stalled; escalate")

    # Absolute backstop: a lesson buys a short reprieve, but if NOT A SINGLE open question has
    # net-closed over a long window (>=6 cycles), you are stalled regardless of lesson cadence.
    # (Closes the "one fabricated lesson every 3rd cycle keeps it alive forever" residual.)
    if len(cycles) >= 6:
        longw = cycles[-6:]
        if int(longw[-1].get("open_questions_after", open_q)) >= int(longw[0].get("open_questions_after", open_q)):
            return ("STOP-CHURN",
                    "no open question net-closed in the last 6 cycles — stalled regardless of "
                    "lesson cadence; escalate to operator")

    # NOTE (honest limit): these signals are read from worker-written state.json. They catch
    # accidental/lazy stalls; a worker that deliberately falsifies open_questions/new_lessons can
    # still evade — the backstop there is the human review, not this gate. Documented, not hidden.
    return "CONTINUE", "progress: an open question closed or a lesson learned"


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
