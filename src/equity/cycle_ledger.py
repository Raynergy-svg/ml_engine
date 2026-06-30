"""Pillar 2 — tamper-evident per-cycle ledger + recall-triggers (shadow lane).

The harvester records what it learns each cycle — the Pillar-3 decision plus the
re-derived facts (target-weight fingerprint, gross, order count) — to an
append-only, **hash-chained** JSONL ledger. Each record commits to the prior
record's hash, so any edit / insert / delete is detectable by
:func:`verify_chain` (anti-falsification, cf. L-007/L-008: the record is
objective and cannot be silently rewritten). :func:`recall_triggers` aggregates
recurring conditions (executions, stale abstains, drawdown halts, ship-gate
blocks, halt refusals) so the loop sharpens over cycles.

NON-hot-path / shadow-only: imports nothing from the FX runtime; writes only
under ``trained_data/equity/``. Atomic appends (read-all + tmp + ``os.replace``)
keep the file recoverable on a crash mid-write.

Known limitations (documented, not yet closed — verifier 2026-06-24):
* **Tail truncation is undetectable.** The chain is unanchored, so dropping the
  last K records leaves a valid shorter prefix. Closing this needs a signed head
  record or an external length anchor — deferred until the H1 live driver, where
  it matters; in the shadow lane the full ledger is always present.
* **Unbounded + O(n) per append** (read-all-rewrite). Negligible for a daily
  shadow lane for years; add rotation before high-frequency use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64
LEDGER_PATH_DEFAULT = "trained_data/equity/cycle_ledger.jsonl"

# Re-derived-fact field set committed to the hash (record_hash is appended after).
_HASHED_FIELDS = (
    "seq",
    "asof",
    "decision",
    "reasons",
    "actionable",
    "universe_hash",
    "n_orders",
    "target_weight_hash",
    "gross",
    "prev_hash",
)


@dataclass(frozen=True)
class CycleRecord:
    """One harvester cycle: the decision + objective re-derived facts."""

    seq: int
    asof: str
    decision: str
    reasons: Tuple[str, ...]
    actionable: bool
    universe_hash: str
    n_orders: int
    target_weight_hash: str
    gross: float
    prev_hash: str
    record_hash: str

    def payload(self) -> dict:
        """The hash-committed payload (everything except ``record_hash``)."""
        return {
            "seq": int(self.seq),
            "asof": str(self.asof),
            "decision": str(self.decision),
            "reasons": list(self.reasons),
            "actionable": bool(self.actionable),
            "universe_hash": str(self.universe_hash),
            "n_orders": int(self.n_orders),
            "target_weight_hash": str(self.target_weight_hash),
            "gross": float(self.gross),
            "prev_hash": str(self.prev_hash),
        }

    def to_line(self) -> dict:
        d = self.payload()
        d["record_hash"] = self.record_hash
        return d


def target_weight_hash(weights: Optional[Mapping[str, float]]) -> str:
    """Order-invariant, value-sensitive sha256 fingerprint of a target book.

    Pillar 1 (verifier) recomputes the strategy's weights and compares this
    fingerprint — so the recorded decision cannot be forged without detection.
    """
    items = sorted(
        (str(k), round(float(v), 8)) for k, v in (weights or {}).items()
    )
    canonical = json.dumps(
        items, separators=(",", ":"), sort_keys=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record_hash(payload: dict) -> str:
    """sha256 over the canonical payload; ``prev_hash`` is inside ``payload``."""
    canonical = json.dumps(
        {k: payload[k] for k in _HASHED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_append_line(path: Path, line: str) -> None:
    """Append one line by rewriting the file atomically (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(existing + line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _last_hash_and_seq(path: Path) -> Tuple[str, int]:
    """Return ``(prev_hash, next_seq)`` from the ledger tail (genesis if empty)."""
    if not path.exists():
        return (GENESIS_HASH, 0)
    last = None
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        count += 1
        last = raw
    if last is None:
        return (GENESIS_HASH, 0)
    try:
        obj = json.loads(last)
        return (str(obj["record_hash"]), count)
    except (json.JSONDecodeError, KeyError, TypeError):
        # Corrupt tail: chain from genesis-of-rest is impossible; surface loudly.
        logger.error("cycle_ledger tail unreadable at %s — refusing to chain", path)
        raise


def append_cycle(
    *,
    ledger_path: Path,
    asof: str,
    decision: str,
    reasons,
    actionable: bool,
    universe_hash: str,
    n_orders: int = 0,
    target_weights: Optional[Mapping[str, float]] = None,
) -> CycleRecord:
    """Append one cycle record, chained to the prior record's hash."""
    path = Path(ledger_path)
    prev_hash, seq = _last_hash_and_seq(path)
    weights = dict(target_weights or {})
    gross = float(sum(abs(float(v)) for v in weights.values()))
    payload = {
        "seq": int(seq),
        "asof": str(asof),
        "decision": str(decision),
        "reasons": [str(r) for r in (reasons or ())],
        "actionable": bool(actionable),
        "universe_hash": str(universe_hash),
        "n_orders": int(n_orders),
        "target_weight_hash": target_weight_hash(weights),
        "gross": gross,
        "prev_hash": prev_hash,
    }
    rec_hash = _record_hash(payload)
    line = dict(payload)
    line["record_hash"] = rec_hash
    _atomic_append_line(
        path,
        json.dumps(line, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
    return CycleRecord(
        seq=seq,
        asof=payload["asof"],
        decision=payload["decision"],
        reasons=tuple(payload["reasons"]),
        actionable=payload["actionable"],
        universe_hash=payload["universe_hash"],
        n_orders=payload["n_orders"],
        target_weight_hash=payload["target_weight_hash"],
        gross=payload["gross"],
        prev_hash=prev_hash,
        record_hash=rec_hash,
    )


def read_ledger(ledger_path: Path) -> List[CycleRecord]:
    """Parse the ledger into records. Missing file -> empty list."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    out: List[CycleRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        out.append(
            CycleRecord(
                seq=int(obj["seq"]),
                asof=str(obj["asof"]),
                decision=str(obj["decision"]),
                reasons=tuple(obj.get("reasons", [])),
                actionable=bool(obj["actionable"]),
                universe_hash=str(obj["universe_hash"]),
                n_orders=int(obj["n_orders"]),
                target_weight_hash=str(obj["target_weight_hash"]),
                gross=float(obj["gross"]),
                prev_hash=str(obj["prev_hash"]),
                record_hash=str(obj["record_hash"]),
            )
        )
    return out


def verify_chain(ledger_path: Path) -> Tuple[bool, Optional[int]]:
    """Recompute the hash chain; return ``(ok, first_broken_seq)``.

    Detects any tampered field, reordered record, or broken link. Missing file
    is vacuously OK.
    """
    path = Path(ledger_path)
    if not path.exists():
        return (True, None)
    prev = GENESIS_HASH
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            seq = int(obj["seq"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return (False, None)
        if str(obj.get("prev_hash")) != prev:
            return (False, seq)
        expected = _record_hash({k: obj.get(k) for k in _HASHED_FIELDS})
        if expected != str(obj.get("record_hash")):
            return (False, seq)
        prev = str(obj["record_hash"])
    return (True, None)


def _trigger_for(decision: str, reasons: Tuple[str, ...]) -> str:
    """Map a (decision, reasons) pair to a named recall-trigger."""
    if decision == "continue":
        return "executed"
    if decision == "halt":
        # Exact reason-code prefix (not a loose substring) so an unrelated reason
        # merely containing "drawdown" can't misclassify a non-drawdown halt.
        is_dd = any(r.startswith("drawdown_breach") for r in reasons)
        return "drawdown_halt" if is_dd else "halt_other"
    if decision == "no_act":
        return "ship_gate_block"
    if decision == "abstain":
        return "stale_abstain"
    if decision == "refuse":
        return "halt_refuse"
    return "other"


def recall_triggers(ledger_path: Path) -> dict:
    """Aggregate recurring conditions across all recorded cycles."""
    recs = read_ledger(ledger_path)
    by_decision: dict = {}
    triggers: dict = {}
    for r in recs:
        by_decision[r.decision] = by_decision.get(r.decision, 0) + 1
        trig = _trigger_for(r.decision, r.reasons)
        triggers[trig] = triggers.get(trig, 0) + 1
    return {
        "total_cycles": len(recs),
        "by_decision": by_decision,
        "triggers": triggers,
    }
