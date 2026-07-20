"""Durable, idempotent ledger of attributed closed positions.

One row per closed position slice: the outcome as observed, plus the
deterministic attribution of it. This is the artifact every later stage reads
-- performance memory, pattern accumulation, and (only when triggered) any
reasoning layer.

Durability rules (from ``.claude/rules/improvement.md``):

* **Locked + fsynced appends.** A scheduled runner can overlap with itself, and
  two interleaved appends on the same file produce torn lines. Writes take an
  exclusive ``fcntl.flock`` and ``fsync`` before releasing.
* **Idempotent by content.** Re-running the attribution pass over the whole
  broker journal must not duplicate rows. Each row carries an ``outcome_key``
  -- a canonical digest of the material outcome fields -- and an append whose
  key is already present is skipped.
* **Corrupt lines never abort a read.** A torn final write is possible on any
  append-only file; a malformed line is logged and skipped, never fatal.

The digest reuses ``src.evidence`` canonicalization so a ledger row hashes the
same way the evidence layer would hash it. Deliberate: these rows are candidate
inputs to the evidence pipeline, and two different canonical forms for the same
fact is exactly the drift that makes provenance untrustworthy.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from src.evidence.hashing import content_digest
from src.learning.decision_record import RECORD_KIND_DECISION, RECORD_KIND_REALIZED
from src.learning.outcome_attribution import AttributionReport, OutcomeRecord

logger = logging.getLogger(__name__)


class OutcomeLedgerKindError(TypeError):
    """A decision record was offered to the realized-outcome ledger.

    The two corpora must never blend: a decision is an intention, a realized
    outcome is money that moved. Raised rather than skipped, because silently
    dropping the row would hide a caller that has confused the two.
    """


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "trained_data" / "learning" / "outcome_ledger.jsonl"
LEDGER_SCHEMA_VERSION = 1

# Fields that MATERIALLY identify a closed slice. Two rows agreeing on all of
# these describe the same real-world event and must not both be stored.
# Deliberately excludes derived//cost fields: a broker restating a financing
# figure should not mint a duplicate position close.
_MATERIAL_FIELDS = ("trade_id", "instrument", "entry_time", "exit_time", "units", "realized_pl_home")


def outcome_key(outcome: OutcomeRecord) -> str:
    """Stable content digest identifying one closed slice."""
    material = {field: getattr(outcome, field) for field in _MATERIAL_FIELDS}
    return content_digest(material)


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("outcome_ledger: skipping malformed line %d in %s: %s", lineno, path, exc)
            continue
        if isinstance(row, dict):
            yield row
        else:
            logger.warning("outcome_ledger: line %d in %s is not an object -- skipped", lineno, path)


def load_ledger(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read every well-formed ledger row. Missing file reads as empty."""
    return list(_iter_rows(Path(path) if path is not None else LEDGER_PATH))


def existing_keys(path: Optional[Path] = None) -> Set[str]:
    """Set of ``outcome_key`` values already recorded."""
    return {row["outcome_key"] for row in load_ledger(path) if row.get("outcome_key")}


def build_row(
    outcome: OutcomeRecord,
    report: AttributionReport,
    *,
    source: str = "oanda_transactions",
) -> Dict[str, Any]:
    """Assemble one ledger row from an outcome and its attribution."""
    return {
        "record_kind": RECORD_KIND_REALIZED,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "outcome_key": outcome_key(outcome),
        "source": source,
        "lane": outcome.lane,
        "decision_id": outcome.decision_id,
        "outcome": {
            "trade_id": outcome.trade_id,
            "instrument": outcome.instrument,
            "entry_time": outcome.entry_time,
            "exit_time": outcome.exit_time,
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "units": outcome.units,
            "realized_pl_home": outcome.realized_pl_home,
            "exit_reason": outcome.exit_reason,
        },
        "attribution": report.to_dict(),
    }


def append_rows(
    rows: Iterable[Dict[str, Any]],
    path: Optional[Path] = None,
) -> Dict[str, int]:
    """Append rows under an exclusive lock, skipping keys already present.

    The duplicate check happens INSIDE the lock. Checking outside would let two
    concurrent runners both observe "not present" and both append.

    Returns:
        ``{"appended": n, "skipped_duplicate": n, "skipped_keyless": n}``.
    """
    target = Path(path) if path is not None else LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = list(rows)
    stats = {"appended": 0, "skipped_duplicate": 0, "skipped_keyless": 0}
    if not pending:
        return stats

    # Open in append mode; the lock is held across read-back and write so the
    # dedupe decision cannot be raced.
    with open(target, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            seen: Set[str] = set()
            for line in fh.read().splitlines():
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, dict) and existing.get("outcome_key"):
                    seen.add(existing["outcome_key"])

            for row in pending:
                # A DecisionRecord describes an INTENTION; this ledger holds
                # money that actually moved. Letting a candidate in would let a
                # counterfactual inflate realized performance, so the boundary
                # is enforced here rather than trusted to the caller.
                kind = row.get("record_kind")
                if kind == RECORD_KIND_DECISION:
                    raise OutcomeLedgerKindError(
                        "realized outcome ledger refuses a decision record; "
                        "decisions belong in src.learning.decision_ledger"
                    )
                key = row.get("outcome_key")
                if not key:
                    logger.warning("outcome_ledger: row without outcome_key refused")
                    stats["skipped_keyless"] += 1
                    continue
                if key in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                seen.add(key)
                stats["appended"] += 1

            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return stats


__all__ = [
    "LEDGER_PATH",
    "OutcomeLedgerKindError",
    "LEDGER_SCHEMA_VERSION",
    "append_rows",
    "build_row",
    "existing_keys",
    "load_ledger",
    "outcome_key",
]
