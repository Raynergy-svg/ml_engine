#!/usr/bin/env python3
"""Feed closed positions through the deterministic learning spine.

Reads the broker transaction journal, reconstructs closed round trips, runs
deterministic attribution on each, appends new ones to the outcome ledger, and
rebuilds the performance-memory snapshot.

Fully deterministic and idempotent: re-running over the whole journal appends
nothing new. Safe to schedule.

This script does NOT trade, does NOT call any model, and does NOT propose
config changes. It turns realized outcomes into arithmetic. Whether any of that
arithmetic ever becomes a rule is a separate, gated decision
(``.claude/rules/improvement.md``: 3+ observations, then replay, then the
signed disposition chain in ``src/evidence/``).

Exit codes:
    0  ran; attribution reconciled
    1  ran; RESIDUAL ALARM -- the attribution model disagrees with the broker
    2  could not run (missing journal, unreadable ledger)

Usage:
    python scripts/run_outcome_attribution.py
    python scripts/run_outcome_attribution.py --dry-run
    python scripts/run_outcome_attribution.py --transactions PATH --ledger PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.learning.oanda_outcome_adapter import (  # noqa: E402
    DEFAULT_TRANSACTIONS_PATH,
    load_outcome_records,
)
from src.learning.outcome_attribution import attribute  # noqa: E402
from src.learning.outcome_ledger import (  # noqa: E402
    LEDGER_PATH,
    append_rows,
    build_row,
    load_ledger,
)
from src.learning.performance_memory import build_memory  # noqa: E402

MEMORY_PATH = REPO_ROOT / "trained_data" / "learning" / "performance_memory.json"

logger = logging.getLogger("run_outcome_attribution")


def _write_json_atomic(payload: dict, path: Path) -> None:
    """tmp + os.replace, so a reader never observes a half-written snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a .tmp sibling behind on failure.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--memory", type=Path, default=MEMORY_PATH)
    parser.add_argument("--dry-run", action="store_true", help="attribute and report, write nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Refuse a ledger contaminated with decision records BEFORE anything else.
    # A candidate is an intention, not money that moved; averaging the two would
    # overstate performance. This check deliberately precedes the journal load:
    # contamination is fatal on its own, and gating it behind an unrelated
    # missing-file exit made it unreachable on a clean checkout.
    if args.ledger.exists():
        try:
            contaminated = [
                r for r in load_ledger(args.ledger) if r.get("record_kind") == "decision"
            ]
        except OSError as exc:
            logger.error("ledger unreadable: %s", exc)
            return 2
        if contaminated:
            logger.error(
                "REFUSING: %d decision record(s) found in the realized ledger %s. "
                "Decisions belong in the decision ledger and must never be attributed "
                "as realized P&L.", len(contaminated), args.ledger,
            )
            return 2

    try:
        records, unlinkable = load_outcome_records(args.transactions)
    except FileNotFoundError as exc:
        logger.error("cannot run: %s", exc)
        return 2

    if not records:
        logger.warning("no closed round trips reconstructed from %s", args.transactions)

    reports = [(rec, attribute(rec)) for rec in records]
    rows = [build_row(rec, rep) for rec, rep in reports]

    if args.dry_run:
        stats = {"appended": 0, "skipped_duplicate": 0, "skipped_keyless": 0, "dry_run": True}
        ledger_rows = rows
    else:
        stats = append_rows(rows, args.ledger)
        try:
            ledger_rows = load_ledger(args.ledger)
        except OSError as exc:
            logger.error("ledger unreadable after append: %s", exc)
            return 2

    memory = build_memory(ledger_rows)
    if not args.dry_run:
        _write_json_atomic(memory, args.memory)

    health = memory["residual_health"]
    overall = memory["overall"]

    print(json.dumps({
        "closed_slices_attributed": len(records),
        "unlinkable_trade_ids": sorted(set(unlinkable)),
        "ledger": stats,
        "ledger_total_rows": len(ledger_rows),
        "residual_health": health["status"],
        "max_abs_residual_home": health.get("max_abs_residual_home"),
        "unreconciled_count": health.get("unreconciled_count", 0),
        "overall_expectancy_home": overall.get("expectancy_home"),
        "overall_n": overall.get("n"),
        "memory_path": None if args.dry_run else str(args.memory),
    }, indent=2, sort_keys=True))

    if health["status"] == "ALARM":
        logger.error(
            "RESIDUAL ALARM: %d slice(s) failed to reconcile. %s",
            health["unreconciled_count"], health["interpretation"],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
