#!/usr/bin/env python3
"""record_cycle.py — the sanctioned writer of loop cycles (objective measurement, not self-report).

A worker does NOT hand-type progress numbers into state.json. Instead it calls this at cycle close;
this MEASURES reality and appends one cycle:
  - tests_passed     : parsed from running the test suite (--test-cmd)
  - verify_gate_pass : from running verify_gate.py (or --verify-status inject for tests)
  - lessons_count    : count of `## L-NNN` headers in LESSONS.md
  - open_questions_after : count of status=="open" in questions.json
loop_gate.py then DERIVES new_verified_facts / new_lessons from deltas, and re-checks the latest
cycle's verdict + open-count against live reality (so a later hand-edit is caught as tamper).

Usage:
  python3 record_cycle.py --summary "what this cycle did" [--repo R] [--test-cmd CMD]
                          [--verify-status pass|fail] [--state S] [--questions Q] [--lessons L]
Atomic write. Stdlib only. --repo override + injectable cmd/status make it unit-testable.
"""
from __future__ import annotations
import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path("/Users/buddy/Documents/ml_engine")


def _measure_tests(repo: Path, test_cmd: str | None) -> tuple[int, int]:
    """Run the suite (or injected cmd) and parse 'N passed, M failed'. Failure => fail-closed (0,1).

    No shell: --test-cmd is shlex-split and run as an argv list (avoids command injection)."""
    argv = shlex.split(test_cmd) if test_cmd else [sys.executable, str(repo / ".claude/loop/tests/test_loop_enforcement.py")]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=300, cwd=str(repo))
        out = r.stdout + r.stderr
    except Exception:
        return 0, 1
    passed = int(m.group(1)) if (m := re.search(r"(\d+)\s+passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+)\s+failed", out)) else (0 if passed else 1)
    return passed, failed


def _measure_verify(repo: Path, override: str | None) -> bool:
    if override:
        return override == "pass"
    vg = repo / ".claude/loop/verify_gate.py"
    if not vg.exists():
        return False
    try:
        return subprocess.run([sys.executable, str(vg), "--repo", str(repo), "--quiet"],
                              capture_output=True, text=True, timeout=90).returncode == 0
    except Exception:
        return False


def _lessons_count(lessons_path: Path) -> int:
    try:
        return len(re.findall(r'^##\s+L-\d+\b', lessons_path.read_text(), re.M))
    except Exception:
        return 0


def _open_count(questions_path: Path) -> int:
    try:
        qs = json.loads(questions_path.read_text()).get("questions", [])
        return sum(1 for q in qs if isinstance(q, dict) and q.get("status") == "open")
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--test-cmd", default=None)
    ap.add_argument("--verify-status", default=None, choices=["pass", "fail"])
    ap.add_argument("--state", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--lessons", default=None)
    a = ap.parse_args()
    repo = Path(a.repo)
    state_path = Path(a.state) if a.state else (repo / ".claude/loop/state.json")
    questions_path = Path(a.questions) if a.questions else (repo / ".claude/loop/questions.json")
    lessons_path = Path(a.lessons) if a.lessons else (repo / ".claude/LESSONS.md")

    passed, failed = _measure_tests(repo, a.test_cmd)
    cycle = {
        "n": 0,
        "summary": a.summary,
        "tests_passed": passed,
        "tests_failed": failed,
        "verify_gate_pass": _measure_verify(repo, a.verify_status),
        "lessons_count": _lessons_count(lessons_path),
        "open_questions_after": _open_count(questions_path),
    }

    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception:
        state = {}
    cycles = state.get("cycles", [])
    if not isinstance(cycles, list):
        cycles = []
    cycle["n"] = len(cycles) + 1
    cycles.append(cycle)
    state["cycles"] = cycles

    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)
    print(json.dumps({"recorded_cycle": cycle["n"], **{k: cycle[k] for k in
          ("tests_passed", "tests_failed", "verify_gate_pass", "lessons_count", "open_questions_after")}}))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
