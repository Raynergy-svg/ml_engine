#!/usr/bin/env python3
"""Deterministic running-status oracle for the equity harvester (L-017).

Re-derives **running: YES/NO** from disk, fail-closed — read-only, no side
effects. The point: a status/completion report that claims the harvester is
"running" must AGREE with this oracle. "Shipped + unit-tested" is not running.

running:YES requires BOTH (a) a live non-test process executing the harvester
AND (b) evidence a real cycle happened (a live state artifact / non-empty
ledger). If either is absent, or anything is unreadable, the verdict is NO
(fail-closed). `tmp_path` unit-test execution is invisible here by design.

Usage:
    python3 .claude/loop/running_status.py              # prints status, exit 0
    python3 .claude/loop/running_status.py --assert-running   # exit 3 if NOT running
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root from .claude/loop/
_PROCESS_NEEDLES = (
    "run_equity_harvester",
    "AutonomousLoop",
    "embedded_scanner",
    "continuous.py",
    "buddy_scanner",
)
_ARTIFACTS = (
    "cycle_ledger.jsonl",
    "rebalance_state.json",
    "portfolio_state.json",
    "loop_state.json",
)


def _process_running() -> bool:
    try:
        out = subprocess.run(
            ["ps", "axo", "command"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False  # fail-closed: can't tell -> NO
    for line in out.splitlines():
        if "pytest" in line or "grep" in line or "running_status" in line:
            continue
        if any(n in line for n in _PROCESS_NEEDLES):
            return True
    return False


def _cycles_executed() -> int:
    ledger = ROOT / "trained_data" / "equity" / "cycle_ledger.jsonl"
    if not ledger.exists():
        return 0
    try:
        return sum(1 for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return 0


def _live_artifacts() -> bool:
    d = ROOT / "trained_data" / "equity"
    if not d.exists():
        return False
    return any((d / f).exists() for f in _ARTIFACTS)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    proc = _process_running()
    cycles = _cycles_executed()
    artifacts = _live_artifacts()
    running = proc and (cycles > 0 or artifacts)  # fail-closed: need BOTH
    print(
        "HARVESTER STATUS — running: %s · process: %s · cycles_executed: %d · "
        "live_artifacts: %s"
        % (
            "YES" if running else "NO",
            "YES" if proc else "NO",
            cycles,
            "YES" if artifacts else "NO",
        )
    )
    if "--assert-running" in argv and not running:
        return 3  # fail-closed: a "running" claim does not hold
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
