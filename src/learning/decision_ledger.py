"""Append-only ledger of DecisionRecords — structurally isolated from realized P&L.

Separate file, separate API, mutually exclusive record kinds. A decision record
describes an *intention*; a realized outcome describes *money that moved*.
Blending them would let counterfactual candidates inflate performance
statistics, so the boundary is mechanical:

* ``append_decisions`` refuses any row not marked ``record_kind="decision"``.
* ``src.learning.outcome_ledger.append_rows`` refuses rows marked as decisions.
* The realized attribution runner refuses a ledger containing decision rows.

Durability matches the proven outcome-ledger pattern: exclusive ``fcntl.flock``
held across read-back and write, ``fsync`` before release, dedupe INSIDE the
lock, malformed lines skipped rather than fatal.

Idempotency is by ``decision_digest`` — a canonical digest of the material
decision fields. A retried cycle reproduces the digest and appends nothing; a
materially different decision produces a new digest and a new row.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from src.learning.decision_record import (
    RECORD_KIND_DECISION,
    DecisionRecord,
    DecisionRecordError,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_LEDGER_PATH = REPO_ROOT / "trained_data" / "learning" / "decision_ledger.jsonl"


class DecisionLedgerError(RuntimeError):
    """Raised when a decision cannot be durably persisted.

    Callers on the executable path MUST treat this as fatal to the trade: the
    cycle converts to ABSTAIN and no order is submitted. An unrecorded
    execution is an unattributable execution.
    """


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("decision_ledger: skipping malformed line %d in %s: %s", lineno, path, exc)
            continue
        if isinstance(row, dict):
            yield row
        else:
            logger.warning("decision_ledger: line %d in %s is not an object -- skipped", lineno, path)


def load_decisions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read every well-formed decision row. Missing file reads as empty."""
    return list(_iter_rows(Path(path) if path is not None else DECISION_LEDGER_PATH))


def existing_digests(path: Optional[Path] = None) -> Set[str]:
    return {r["decision_digest"] for r in load_decisions(path) if r.get("decision_digest")}


def _validate_kind(row: Mapping[str, Any]) -> None:
    kind = row.get("record_kind")
    if kind != RECORD_KIND_DECISION:
        raise DecisionRecordError(
            f"decision ledger refuses record_kind={kind!r}; only "
            f"{RECORD_KIND_DECISION!r} rows belong here"
        )


def append_decisions(
    records: Iterable[DecisionRecord],
    path: Optional[Path] = None,
) -> Dict[str, int]:
    """Persist decisions atomically under an exclusive lock.

    Returns ``{"appended": n, "skipped_duplicate": n}``.

    Raises:
        DecisionLedgerError: if the rows could not be durably written. The
            caller decides what that means; on an executable decision it must
            mean "do not submit the order".
    """
    pending = list(records)
    stats = {"appended": 0, "skipped_duplicate": 0}
    if not pending:
        return stats

    target = Path(path) if path is not None else DECISION_LEDGER_PATH
    rows = [r.to_row() for r in pending]
    for row in rows:
        _validate_kind(row)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
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
                    if isinstance(existing, dict) and existing.get("decision_digest"):
                        seen.add(existing["decision_digest"])

                for row in rows:
                    digest = row["decision_digest"]
                    if digest in seen:
                        stats["skipped_duplicate"] += 1
                        continue
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    seen.add(digest)
                    stats["appended"] += 1

                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise DecisionLedgerError(f"could not persist decision records to {target}: {exc}") from exc

    return stats


def link_broker_order(
    decision_digest: str,
    *,
    broker_order_id: str,
    client_order_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> bool:
    """Append a linkage row binding a broker order to its decision.

    Appended rather than mutated: the ledger stays append-only, so the original
    decision is never rewritten after the fact. Every replacement, partial fill,
    cancellation and close should carry the same ``decision_digest`` so the
    whole order lifecycle remains attributable to one decision.

    Returns True if the linkage row was written.
    """
    target = Path(path) if path is not None else DECISION_LEDGER_PATH
    row = {
        "record_kind": RECORD_KIND_DECISION,
        "row_type": "broker_link",
        "decision_digest": f"{decision_digest}:link:{broker_order_id}",
        "links_decision_digest": decision_digest,
        "broker_order_id": broker_order_id,
        "client_order_id": client_order_id,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                for line in fh.read().splitlines():
                    if not line.strip():
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(existing, dict) and existing.get("decision_digest") == row["decision_digest"]:
                        return False
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise DecisionLedgerError(f"could not link broker order to {target}: {exc}") from exc
    return True


__all__ = [
    "DECISION_LEDGER_PATH",
    "DecisionLedgerError",
    "append_decisions",
    "existing_digests",
    "link_broker_order",
    "load_decisions",
]
