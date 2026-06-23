#!/usr/bin/env python3
"""loop_gate.py — OBJECTIVE stopping conditions, derived from observable reality (LOOP.md).

The stop/continue decision must not be fakeable by typing numbers into state.json. So the signals
are DERIVED, not self-reported:
  - open_questions   <- count of status=="open" in .claude/loop/questions.json (a tracked LIST).
  - new_verified_facts <- per-cycle delta of `tests_passed` (real suite count) + a verify-verdict
                          flip (FAIL->PASS). Recorded by record_cycle.py, which runs the suite.
  - new_lessons      <- per-cycle delta of `lessons_count` (count of `## L-NNN` in LESSONS.md).
  - verdict          <- LIVE verify_gate.py run at decide time (or injected for tests).

ANTI-TAMPER (fail-closed -> HALT): the LATEST recorded cycle's `verify_gate_pass` and
`open_questions_after` must match LIVE reality (a fresh verify_gate run + the live questions.json
count). A hand-edit that fakes a PASS verdict or a closed question is caught here. So a worker that
lies in state.json can neither keep the loop alive nor declare STOP-DONE falsely.

state.json cycle schema (written by record_cycle.py — do not hand-edit; loop_gate re-checks it):
  {"n":1, "summary":"...", "tests_passed":<int>, "verify_gate_pass":<bool>,
   "lessons_count":<int>, "open_questions_after":<int>}

Decision precedence (first match wins):
  HALT-SAFETY  : state/questions unreadable, OR latest cycle stale/tampered vs live, OR risk ALARM -> exit 2
  STOP-BLOCKED : blocked_on_irreversible == true                                                   -> exit 0
  STOP-DONE    : LIVE verify_gate PASS AND latest cycle added no new facts/lessons AND live open==0 -> exit 0
  STOP-CHURN   : rolling window (<=3): open questions did NOT net-decrease AND no lesson learned;   -> exit 0
                 OR backstop (>=6 cycles): no open question net-closed over 6 cycles. (anti-stall)
  CONTINUE     : otherwise                                                                          -> exit 0

Keeps the 6-cycle backstop as belt-and-suspenders. Honest limit: a worker could still falsify the
recorded `tests_passed`/`lessons_count` for the latest cycle, but the latest verdict + open-count are
re-derived live, and record_cycle.py is the sanctioned writer that measures them — human review
backstops deliberate falsification of the test/lesson counts.
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
    if override:
        return override, f"injected:{override}"
    mon = repo / ".claude/tools/risk_monitor.sh"
    if not mon.exists():
        return "alarm", "risk_monitor.sh missing == unsafe"
    try:
        r = subprocess.run(["bash", str(mon)], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "RISK_MONITOR_REPO": str(repo)})
        return ("green" if r.returncode == 0 else "alarm"), (r.stdout.strip().splitlines()[-1] if r.stdout else "")
    except Exception as e:
        return "alarm", f"risk_monitor error == unsafe ({e})"


def _verify_status(repo: Path, override: str | None) -> tuple[bool, str]:
    """Return (passed, detail). Override ('pass'/'fail') lets tests inject without running it."""
    if override:
        return override == "pass", f"injected:{override}"
    vg = repo / ".claude/loop/verify_gate.py"
    if not vg.exists():
        return False, "verify_gate.py missing == unsafe"
    try:
        r = subprocess.run([sys.executable, str(vg), "--repo", str(repo), "--quiet"],
                           capture_output=True, text=True, timeout=90)
        return (r.returncode == 0), ("verify_gate PASS" if r.returncode == 0 else "verify_gate FAIL")
    except Exception as e:
        return False, f"verify_gate error == unsafe ({e})"


def load_open_questions(path: Path) -> tuple[int | None, str | None]:
    """Count status=='open' entries. Missing/malformed -> (None, error) so caller fails closed."""
    if not path.exists():
        return None, f"questions.json missing at {path} — cannot derive open questions (fail-closed)"
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return None, f"questions.json unreadable ({e})"
    qs = data.get("questions")
    if not isinstance(qs, list):
        return None, "questions.json malformed ('questions' is not a list)"
    return sum(1 for q in qs if isinstance(q, dict) and q.get("status") == "open"), None


def _derive(cycles: list[dict]) -> list[tuple[int, int]]:
    """Per-cycle (new_verified_facts, new_lessons) from objective deltas. First cycle = (0,0)."""
    out: list[tuple[int, int]] = []
    ptp = ppass = plc = None
    for c in cycles:
        tp = int(c.get("tests_passed", 0))
        vp = bool(c.get("verify_gate_pass", False))
        lc = int(c.get("lessons_count", 0))
        nf = nl = 0
        if ptp is not None:
            nf = max(0, tp - ptp) + (1 if (vp and ppass is False) else 0)
            nl = max(0, lc - plc)
        out.append((nf, nl))
        ptp, ppass, plc = tp, vp, lc
    return out


def decide(state: dict, live_open: int, live_pass: bool, risk: str) -> tuple[str, str]:
    cycles = state.get("cycles", [])

    # ANTI-TAMPER (fail-closed): latest recorded measurement must match live reality.
    if cycles:
        last = cycles[-1]
        if bool(last.get("verify_gate_pass", False)) != live_pass:
            return ("HALT-SAFETY",
                    f"state.json stale/tampered: latest verify_gate_pass={last.get('verify_gate_pass')} "
                    f"but live verify_gate={'PASS' if live_pass else 'FAIL'} — re-record the cycle (record_cycle.py)")
        rec_open = last.get("open_questions_after")
        if rec_open is None or int(rec_open) != live_open:
            return ("HALT-SAFETY",
                    f"state.json stale/tampered: latest open_questions_after={rec_open} but tracked "
                    f"questions.json has {live_open} open — re-record the cycle (record_cycle.py)")

    if risk == "alarm":
        return "HALT-SAFETY", "risk monitor ALARM — stop and address before anything else"
    if state.get("blocked_on_irreversible"):
        return "STOP-BLOCKED", "only an irreversible/operator fork remains — escalate, do not self-authorize"
    if not cycles:
        return "CONTINUE", "no cycles recorded yet — run the first cycle"

    derived = _derive(cycles)
    last_nf, last_nl = derived[-1]
    # STOP-DONE uses LIVE verify_gate + LIVE open-count (observable, not self-reported).
    if live_pass and last_nf == 0 and last_nl == 0 and live_open == 0:
        return ("STOP-DONE",
                "verify_gate PASS (live), zero open questions (tracked list), no new verified facts "
                "or lessons this cycle (test-count + verdict + lesson-count deltas == 0)")

    # STOP-CHURN rolling window: progress = close a question OR learn a lesson (derived).
    if len(cycles) >= 2:
        window = cycles[-3:] if len(cycles) >= 3 else cycles[-2:]
        oq_start = int(window[0].get("open_questions_after", live_open))
        oq_end = int(window[-1].get("open_questions_after", live_open))
        w_lessons = sum(nl for nf, nl in derived[-len(window):])
        if oq_end >= oq_start and w_lessons == 0:
            return ("STOP-CHURN",
                    f"no net progress over last {len(window)} cycles (open questions "
                    f"{oq_start}->{oq_end} not decreasing, no lesson learned) — stalled; escalate")

    # Absolute backstop: no open question net-closed over >=6 cycles -> stalled regardless of lessons.
    if len(cycles) >= 6:
        longw = cycles[-6:]
        if int(longw[-1].get("open_questions_after", live_open)) >= int(longw[0].get("open_questions_after", live_open)):
            return ("STOP-CHURN",
                    "no open question net-closed in the last 6 cycles — stalled regardless of "
                    "lesson cadence; escalate to operator")

    return "CONTINUE", "progress: an open question closed or a lesson learned"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--state", default=None)
    ap.add_argument("--questions", default=None, help="path to questions.json (default <repo>/.claude/loop/questions.json)")
    ap.add_argument("--risk-status", default=None, choices=["green", "alarm"], help="inject risk status (tests)")
    ap.add_argument("--verify-status", default=None, choices=["pass", "fail"], help="inject verify_gate status (tests)")
    a = ap.parse_args()
    repo = Path(a.repo)
    state_path = Path(a.state) if a.state else (repo / ".claude/loop/state.json")
    questions_path = Path(a.questions) if a.questions else (repo / ".claude/loop/questions.json")

    def halt(reason: str) -> int:
        print(json.dumps({"decision": "HALT-SAFETY", "reason": reason}, indent=2))
        return 2

    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception as e:
        return halt(f"loop state unreadable == unsafe ({e})")

    live_open, q_err = load_open_questions(questions_path)
    if q_err:
        return halt(q_err)

    live_pass, verify_detail = _verify_status(repo, a.verify_status)
    risk, risk_detail = _risk_status(repo, a.risk_status)

    decision, reason = decide(state, live_open, live_pass, risk)
    print(json.dumps({"decision": decision, "reason": reason,
                      "live_open_questions": live_open, "live_verify": "PASS" if live_pass else "FAIL",
                      "risk": risk, "cycles": len(state.get("cycles", []))}, indent=2))
    return 2 if decision == "HALT-SAFETY" else 0


if __name__ == "__main__":
    sys.exit(main())
