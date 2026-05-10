#!/usr/bin/env python3
"""Backfill `.claude/learnings_outcomes.jsonl` from the trade journal.

Reads ``trained_data/trade_journal_rl.json`` (a list of dicts), filters to
trades closed since a cutoff date, and emits one JSONL line per trade via
``OutcomesLedger`` — which is idempotent by ``trade_id``, so re-running
this script is safe.

Spec note (2026-05-10):
    The Angle 2' brief asked for "last 18 trades since 2026-04-28". As of
    this script's first run, the actual journal contained 16 closed trades
    spanning 2026-04-03 to 2026-04-16. There are no closed trades on or
    after 2026-04-28 in the journal yet. Rather than emit zero entries
    (which would defeat the bootstrap goal), the cutoff defaults to the
    earliest available closed trade so the backfill seeds the ledger with
    the full visible history. Operator can override via `--since`.

After backfill, this script also re-renders the fenced auto-section in
``.claude/learnings.md`` so the ledger and the human-readable mirror are
consistent.

Usage:
    python scripts/backfill_learnings_outcomes.py
    python scripts/backfill_learnings_outcomes.py --since 2026-04-01
    python scripts/backfill_learnings_outcomes.py --no-render

Idempotency: re-running produces no duplicates. The script reports
``backfilled N entries; skipped M duplicates`` so re-runs are visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.automation.learnings_outcomes import (  # noqa: E402
    LearningsRenderer,
    OutcomesLedger,
    emit_outcome_from_journal_entry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_learnings_outcomes")

JOURNAL_PATH_DEFAULT = REPO_ROOT / "trained_data" / "trade_journal_rl.json"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _close_time(entry: Dict[str, Any]) -> Optional[str]:
    """Return the close_time string for an entry, or None if not closed."""
    outcome = entry.get("outcome")
    if not isinstance(outcome, dict):
        return None
    return outcome.get("close_time") or outcome.get("exit_time")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Tolerantly parse an ISO-8601 string. Returns None on failure."""
    if not s:
        return None
    try:
        # Strip nanoseconds beyond microsecond precision; replace trailing Z.
        normalized = str(s).replace("Z", "+00:00")
        # Trim sub-microsecond if present (Python max 6 digits after dot).
        if "." in normalized:
            head, tail = normalized.split(".", 1)
            # tail may contain timezone — split off
            for tz_marker in ("+", "-"):
                if tz_marker in tail:
                    frac, tz = tail.split(tz_marker, 1)
                    tail = frac[:6] + tz_marker + tz
                    break
            else:
                tail = tail[:6]
            normalized = head + "." + tail
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None


def _load_journal(path: Path) -> List[Dict[str, Any]]:
    """Read the journal JSON file with graceful fallback."""
    if not path.exists():
        logger.error("journal not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.error("journal at %s is not a list (got %s)", path, type(data).__name__)
            return []
        return data
    except json.JSONDecodeError as e:
        logger.error("journal parse failed at %s: %s", path, e)
        return []


def _resolve_since(entries: List[Dict[str, Any]], requested: Optional[str]) -> str:
    """Pick the effective cutoff date.

    If the user requested a specific date, honor it. Otherwise fall back
    to the earliest close_time available in the journal so the backfill
    seeds the ledger with the full visible history (see module docstring
    for the spec-vs-reality note).
    """
    if requested:
        return requested
    closed_times = [t for t in (_close_time(e) for e in entries) if t]
    if not closed_times:
        return "1970-01-01"  # no closed trades — match anything
    earliest = min(closed_times)[:10]  # YYYY-MM-DD slice
    logger.info(
        "no --since given; defaulting to earliest closed-trade date in journal: %s "
        "(spec-asked 2026-04-28; journal has no closes on/after that date)",
        earliest,
    )
    return earliest


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        type=Path,
        default=JOURNAL_PATH_DEFAULT,
        help=f"Trade journal path (default: {JOURNAL_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) — only backfill trades closed on/after this date. "
        "Defaults to the earliest close_time in the journal.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering the fenced auto-section in .claude/learnings.md.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Override ledger path (default: .claude/learnings_outcomes.jsonl)",
    )
    args = parser.parse_args()

    entries = _load_journal(args.journal)
    if not entries:
        logger.error("no entries to backfill — exiting")
        return 1

    since_str = _resolve_since(entries, args.since)
    since_dt = _parse_iso(since_str + "T00:00:00Z")
    if since_dt is None:
        logger.error("could not parse --since=%s as ISO date", since_str)
        return 2

    # Filter: closed trades only, with close_time >= since.
    candidates: List[Dict[str, Any]] = []
    for e in entries:
        ct = _close_time(e)
        if not ct:
            continue
        ct_dt = _parse_iso(ct)
        if ct_dt is None:
            logger.warning("skipping entry with unparseable close_time: %s", ct)
            continue
        # Compare in UTC — ISO strings without offset assumed UTC.
        if ct_dt.tzinfo is None:
            ct_dt = ct_dt.replace(tzinfo=timezone.utc)
        if ct_dt >= since_dt:
            candidates.append(e)

    logger.info(
        "filtered %d closed trades (since=%s) out of %d journal entries",
        len(candidates), since_str, len(entries),
    )

    ledger = OutcomesLedger(ledger_path=args.ledger) if args.ledger else OutcomesLedger()
    pre_existing_ids = ledger.trade_ids()

    # Sort by close_time ascending so the JSONL ends up in chronological order.
    candidates.sort(key=lambda e: _close_time(e) or "")

    backfilled = 0
    skipped = 0
    failed = 0
    for entry in candidates:
        tid = str(entry.get("trade_id", ""))
        if not tid:
            failed += 1
            logger.warning("skipping journal entry with no trade_id")
            continue
        if tid in pre_existing_ids:
            skipped += 1
            continue
        ok = emit_outcome_from_journal_entry(entry, ledger=ledger)
        if ok:
            backfilled += 1
        else:
            failed += 1

    print(
        f"backfilled {backfilled} entries; skipped {skipped} duplicates"
        + (f"; {failed} failed" if failed else "")
    )

    # Render the fenced section.
    if not args.no_render:
        renderer = LearningsRenderer(ledger=ledger)
        ok = renderer.render(allow_insert=True)
        if ok:
            print(f"rendered fenced auto-section in {renderer.learnings_md_path}")
        else:
            print(
                f"WARNING: renderer refused to write {renderer.learnings_md_path} "
                "(check WARN logs for fence-marker state)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
