#!/usr/bin/env python3
"""record_verdict.py — record the SEPARATE agent-verifier's verdict, bound to the current loop state.

After dispatching the separate Code Reviewer verifier (verifier.md / /verify-task) and getting its
gate, run this. loop_gate requires a FRESH PASS verdict — one whose bound measurements match the
current last cycle — before STOP-DONE. So a worker can't reach "done" by SKIPPING the separate
verifier (laziness): no verdict → no done; and changing anything after verifying makes the verdict
stale → must re-verify.

IRREDUCIBLE FLOOR (documented, L-011): the `gate` value here is a CLAIM the orchestrator makes — a
worker can write --gate PASS without truly dispatching anyone. No static gate can prove an LLM was
dispatched and reasoned honestly. The DETERMINISTIC half (verify_gate, re-run live by loop_gate,
hash-pinned, cross-protected) is the enforced verification floor; this artifact closes the lazy-skip
of the AGENT half, not the lie. Human review backstops the agent half's honesty.

Usage: python3 record_verdict.py --gate PASS|FAIL [--repo R] [--note ...] [--state S] [--out O]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPO = Path("/Users/buddy/Documents/ml_engine")
BOUND_KEYS = ("tests_passed", "lessons_count", "open_questions_after")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["PASS", "FAIL"])
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--note", default="")
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(a.repo)
    state_path = Path(a.state) if a.state else repo / ".claude/loop/state.json"
    try:
        last = json.loads(state_path.read_text())["cycles"][-1]
        bound = {k: last.get(k) for k in BOUND_KEYS}
    except Exception as e:
        print(f"record_verdict: cannot read loop state ({e})", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else repo / ".claude/loop/agent_verdict.json"
    out.write_text(json.dumps({"gate": a.gate, "note": a.note, "bound": bound}, indent=2))
    print(json.dumps({"recorded_verdict": a.gate, "bound": bound}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
