#!/usr/bin/env python3
"""Sonnet brain loop — headless cycle runner (research orchestration only).

Runs ONE deterministic cycle of src.brain_loop.cycle.run_cycle: check halt,
check rails (de-risk on breach), and — if a hypothesis file is supplied —
register it, gate it against the real ship-gate/decision-gate, and (only on a
CONTINUE verdict) propose SHADOW promotion. Never proposes LIVE promotion,
never arms anything, never calls flatten/set_gross_leverage/start_loop/unhalt.

This script can run standalone (no LLM in the loop at all) — that's by design:
the deterministic mechanics work without Sonnet present, so they are
independently testable and independently safe. The actual "Sonnet is the
brain" invocation is a SEPARATE, scheduled `claude -p` session (not built by
this script) that decides which hypothesis to propose next and calls this
script as one of its tools each cycle — see .claude/brain_loop/PROMPT.md.

HARD BOUNDS (by construction, verified in
tests/test_brain_loop_capability_absence.py): this script and everything it
imports contain no reference to flatten / set_gross_leverage / start_loop /
unhalt / arming, and no import of src.scanner.execution or any broker client.

Usage:
    python3 scripts/run_brain_loop.py --once
    python3 scripts/run_brain_loop.py --once --propose path/to/hypothesis.json
    python3 scripts/run_brain_loop.py --loop --max-cycles 5 --interval 3600

NOT scheduled/cron'd by this task. Running --loop against the live repo is an
operator decision, not something this script does on its own.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brain_loop.cycle import run_cycle  # noqa: E402
from src.scanner.config import ScannerConfig  # noqa: E402

logger = logging.getLogger("run_brain_loop")

_DEFAULT_LEDGER_REL = "trained_data/brain_loop/hypotheses.jsonl"
_DEFAULT_REQUESTS_REL = "trained_data/brain_loop/promotion_requests"


def _load_hypothesis(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"hypothesis_id", "name", "mechanism", "novelty_justification", "params", "harness_cmd"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"hypothesis file missing required keys: {sorted(missing)}")
    harness_cmd = payload["harness_cmd"]
    if not isinstance(harness_cmd, list) or not all(isinstance(x, str) for x in harness_cmd):
        raise ValueError("hypothesis file 'harness_cmd' must be a list of strings")
    return payload


def _run_once(project_root: Path, hypothesis_path: Optional[str]) -> dict:
    result = run_cycle(
        project_root=project_root,
        config=ScannerConfig(),
        ledger_path=project_root / _DEFAULT_LEDGER_REL,
        requests_dir=project_root / _DEFAULT_REQUESTS_REL,
        hypothesis=_load_hypothesis(hypothesis_path),
    )
    return {
        "halted": result.halted,
        "breach_derisked": result.breach_derisked,
        "derisk_action": result.derisk_action,
        "hypothesis_result": result.hypothesis_result,
        "rails": result.rails,
    }


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run exactly one cycle and exit")
    parser.add_argument("--loop", action="store_true", help="run cycles repeatedly (operator decision to invoke)")
    parser.add_argument("--max-cycles", type=int, default=1, help="cycles to run when --loop (default 1)")
    parser.add_argument("--interval", type=float, default=3600.0, help="seconds between cycles when --loop")
    parser.add_argument("--propose", type=str, default=None, help="path to a hypothesis JSON file to register+gate this cycle")
    parser.add_argument("--project-root", type=str, default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    cycles = 1 if args.once or not args.loop else args.max_cycles

    for i in range(cycles):
        try:
            summary = _run_once(project_root, args.propose)
        except Exception as exc:  # noqa: BLE001 - a stuck/crashed harness must not kill --loop mode
            if not args.loop:
                raise
            logger.exception("brain loop cycle %d failed; continuing loop: %s", i, exc)
            if args.loop and i < cycles - 1:
                time.sleep(args.interval)
            continue
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["halted"]:
            logger.warning("brain loop cycle %d: halted (breach_derisked=%s) — stopping", i, summary["breach_derisked"])
            break
        if args.loop and i < cycles - 1:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
