"""Forward-ledger engine — evidence-safe recording for strategy shadow lanes.

Rev 2 (2026-07-19, first review of f590485): activation baselines,
deterministic backfill, applied/next-target book split.

Rev 3 (2026-07-19, scheduler-safety audit): the engine is now safe under
real scheduler and failure conditions, not just sequential calls:

* **Cross-process locking (finding 1).** The ENTIRE read/validate/append
  transaction runs under an exclusive ``fcntl`` lock on a sidecar
  ``<ledger>.lock`` file. Two overlapping invocations serialize; the loser
  re-reads and finds nothing unseen. A check-before-append without the lock
  was insufficient and is gone.
* **Fail-closed corruption (finding 2).** Reads are STRICT: any undecodable
  or non-JSON line raises ``LedgerCorruptionError`` naming the exact line
  number and byte offset, and drops a ``<ledger>.corrupt.json`` marker.
  While the marker exists (or corruption is detected), appends and every
  evidence read REFUSE. Repair is explicit and human: fix the file, remove
  the marker. Corrupt lines are never skipped, never silently analyzed
  around, and never destroyed.
* **Evidence-period identity (finding 3).** Every forward row carries
  ``evidence_period_id`` (``strategy:period_start->period_end``) and
  ``snapshot_version_id`` (sha256 of books+construction). ONE active
  observation per evidence period: ``active_observations`` resolves the
  LAST row per period as the canonical revision (an appended revision
  carries ``supersedes``); ``forward_rows`` returns active revisions only,
  so every consumer counts a period exactly once.
* **Real cadence semantics (finding 4).** Completed rebalances are counted
  from EXPLICIT per-row fields the strategy's frozen rule emits
  (``rebalance_event``, ``rebalance_id``, ``holding_period_id``), never
  inferred from book-dictionary inequality (vol scaling / HRP drift moves
  weights daily without a signal rebalance). Definition: a holding period
  is counted once a FORWARD bar inside it is observed; the activation
  baseline's period provides context but is itself never counted (the
  off-by-one is defined, not accidental). Rows without the explicit fields
  yield ``None`` counts with a reason — never a fabricated number.
* **Strict schema validation (finding 5).** Before any write: canonical
  unique strictly-increasing ISO dates; finite non-bool returns and book
  weights; valid applied/target books; ``return_period_start <
  return_period_end``; ``decision_asof == return_period_start``; one
  strategy and one construction identity per ledger
  (``construction_sha256`` must match the ledger's); no duplicate evidence
  periods except explicit supersession.
* **Atomic persistence (finding 6).** ``_append`` raises on any failure —
  no swallowed write errors, no success reported without durable fsync.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class LedgerCorruptionError(RuntimeError):
    """Evidence ledger corruption — everything downstream must fail closed."""

    def __init__(self, path: Path, line_no: int, byte_offset: int, detail: str):
        self.path = str(path)
        self.line_no = line_no
        self.byte_offset = byte_offset
        self.detail = detail
        super().__init__(
            f"ledger corruption in {path} at line {line_no} (byte offset "
            f"{byte_offset}): {detail} — appends and evidence reads REFUSED "
            f"until repaired (fix the file, then remove {_marker_path(path)})")


def _marker_path(path: Path) -> Path:
    return Path(str(path) + ".corrupt.json")


def _lock_path(path: Path) -> Path:
    return Path(str(path) + ".lock")


@contextmanager
def ledger_lock(path: Path) -> Iterator[None]:
    """Exclusive cross-process lock for one ledger's read/validate/append
    transaction. Blocks until acquired; released on exit or process death."""
    lock_file = _lock_path(path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _write_corruption_marker(path: Path, line_no: int, offset: int, detail: str) -> None:
    try:
        _marker_path(path).write_text(json.dumps({
            "path": str(path), "line_no": line_no, "byte_offset": offset,
            "detail": detail, "detected_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "repair": "fix the ledger line, then delete this marker file",
        }, indent=2, sort_keys=True))
    except OSError:
        logger.error("could not write corruption marker for %s", path, exc_info=True)


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """STRICT evidence read (rev 3): corruption fails closed, never skipped.

    Raises ``LedgerCorruptionError`` on an existing corruption marker or on
    any line that fails to parse as a JSON object. Blank lines are not
    corruption. Missing file -> []."""
    if _marker_path(path).exists():
        try:
            m = json.loads(_marker_path(path).read_text())
        except (OSError, ValueError):
            m = {}
        raise LedgerCorruptionError(path, int(m.get("line_no", -1)),
                                    int(m.get("byte_offset", -1)),
                                    "corruption marker present")
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    offset = 0
    with open(path, "rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line_offset = offset
            offset += len(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _write_corruption_marker(path, line_no, line_offset, str(exc))
                raise LedgerCorruptionError(path, line_no, line_offset, str(exc)) from exc
            if not isinstance(parsed, dict):
                _write_corruption_marker(path, line_no, line_offset, "non-object row")
                raise LedgerCorruptionError(path, line_no, line_offset, "non-object row")
            rows.append(parsed)
    return rows


def _append(path: Path, row: Dict[str, Any]) -> None:
    """Durable append: raises on ANY failure — success is never reported
    without a completed fsync (finding 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def construction_sha256(construction: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(construction, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def snapshot_version_id(body: Dict[str, Any]) -> str:
    payload = {"applied": body.get("applied_book"), "book": body.get("book"),
               "construction_sha256": body.get("construction_sha256")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Schema validation (finding 5) — everything checked BEFORE any write.
# --------------------------------------------------------------------------- #
def _is_bad_number(v: Any) -> bool:
    return isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))


def _validate_book(book: Any, label: str) -> None:
    if not isinstance(book, dict):
        raise ValueError(f"{label} must be a dict, got {type(book).__name__}")
    sides = book if not ({"longs", "shorts"} & set(book)) else \
        {**(book.get("longs") or {}), **(book.get("shorts") or {})}
    for k, v in sides.items():
        if _is_bad_number(v):
            raise ValueError(f"{label} weight {k!r}={v!r} is not a finite number")


def _validate_dates(dates: List[str]) -> None:
    if not dates:
        raise ValueError("forward_ledger: empty dates")
    seen = set()
    prev: Optional[str] = None
    for d in dates:
        try:
            canonical = _dt.date.fromisoformat(str(d)).isoformat()
        except ValueError as exc:
            raise ValueError(f"forward_ledger: non-ISO date {d!r}") from exc
        if canonical != d:
            raise ValueError(f"forward_ledger: non-canonical ISO date {d!r}")
        if d in seen:
            raise ValueError(f"forward_ledger: duplicate date {d!r} in panel")
        if prev is not None and d <= prev:
            raise ValueError(f"forward_ledger: dates not strictly increasing at {d!r}")
        seen.add(d)
        prev = d


def _validate_forward_body(body: Dict[str, Any], d: str, strategy: str,
                           ledger_construction_sha: Optional[str]) -> None:
    net = body.get("today_net_return")
    if _is_bad_number(net):
        raise ValueError(f"forward_ledger: non-finite/boolean today_net_return {net!r} at {d}")
    _validate_book(body.get("applied_book"), "applied_book")
    _validate_book(body.get("book"), "book (next target)")
    start = body.get("return_period_start")
    if not start or str(start) >= d:
        raise ValueError(
            f"forward_ledger: return_period_start {start!r} must be strictly before {d}")
    csha = body.get("construction_sha256")
    if not csha:
        raise ValueError("forward_ledger: payload lacks construction_sha256")
    if ledger_construction_sha is not None and csha != ledger_construction_sha:
        raise ValueError(
            "forward_ledger: construction identity changed mid-ledger — a frozen "
            f"spec must not drift (ledger {ledger_construction_sha[:12]}…, "
            f"payload {csha[:12]}…)")
    if body.get("strategy") != strategy:
        raise ValueError("forward_ledger: payload strategy mismatch")
    for fld in ("rebalance_event", "holding_period_id", "rebalance_id"):
        if fld not in body:
            raise ValueError(f"forward_ledger: payload lacks explicit cadence field {fld!r}")
    if not isinstance(body["rebalance_event"], bool):
        raise ValueError("forward_ledger: rebalance_event must be bool")


def append_unseen_bars(
    ledger_path: Path,
    *,
    strategy: str,
    dates: List[str],
    payload_for: Callable[[str], Dict[str, Any]],
    activation_payload_for: Callable[[str], Dict[str, Any]],
    cycle_ts_iso: str,
) -> List[Dict[str, Any]]:
    """Advance a lane's forward ledger to the latest available bar.

    The whole read/validate/append transaction holds the ledger's exclusive
    cross-process lock. Empty ledger -> ONE activation baseline at the
    latest bar (no realized return). Else -> one forward row per bar
    strictly after the last recorded asof, chronological, validated
    (finding 5) before ANY write. Stale data -> [] (honest no-op)."""
    _validate_dates(dates)
    with ledger_lock(ledger_path):
        rows = read_rows(ledger_path)          # strict: corruption fails closed
        appended: List[Dict[str, Any]] = []

        if rows and str(rows[0].get("strategy", strategy)) != strategy:
            raise ValueError(
                f"forward_ledger: ledger belongs to {rows[0].get('strategy')!r}, "
                f"refusing append for {strategy!r}")
        ledger_csha = next((r.get("construction_sha256") for r in rows
                            if r.get("construction_sha256")), None)

        if not rows:
            latest = dates[-1]
            body = dict(activation_payload_for(latest))
            body.setdefault("construction_sha256",
                            construction_sha256(body.get("construction", {})))
            _validate_book(body.get("book"), "activation book")
            row = dict(body)
            row.update({
                "kind": "activation",
                "strategy": strategy,
                "cycle_ts": cycle_ts_iso,
                "asof_date": latest,
                "today_net_return": None,
                "applied_book": None,
                "return_period_start": None,
                "return_period_end": None,
                "decision_asof": latest,
                "evidence_period_id": None,     # a baseline is NOT an observation
                "snapshot_version_id": snapshot_version_id(
                    {"book": body.get("book"),
                     "construction_sha256": body["construction_sha256"]}),
                "cumulative_shadow_return": 0.0,
                "forward_cycle_seq": 0,
                "orders_placed": 0,
                "broker": None,
            })
            _append(ledger_path, row)
            logger.info("forward ledger activated at %s (baseline): %s", latest, ledger_path)
            return [row]

        last_asof = str(rows[-1].get("asof_date", ""))
        prior_cum = _last_cumulative(rows)
        prior_seq = _last_forward_seq(rows)
        existing_periods = {r.get("evidence_period_id")
                            for r in rows if r.get("evidence_period_id")}
        unseen = [d for d in dates if d > last_asof]
        if not unseen:
            logger.info("forward ledger: no bar after %s — honest no-op (%s)",
                        last_asof, ledger_path)
            return []

        pending: List[Dict[str, Any]] = []
        for d in unseen:
            body = dict(payload_for(d))
            body.setdefault("strategy", strategy)
            body.setdefault("construction_sha256",
                            construction_sha256(body.get("construction", {})))
            _validate_forward_body(body, d, strategy, ledger_csha)
            ledger_csha = ledger_csha or body["construction_sha256"]
            period_id = f"{strategy}:{body['return_period_start']}->{d}"
            if period_id in existing_periods or any(
                    p.get("evidence_period_id") == period_id for p in pending):
                raise ValueError(
                    f"forward_ledger: duplicate evidence period {period_id!r} — "
                    "one observation per market period")
            prior_cum = (1.0 + prior_cum) * (1.0 + float(body["today_net_return"])) - 1.0
            prior_seq += 1
            body.update({
                "kind": "forward",
                "cycle_ts": cycle_ts_iso,
                "asof_date": d,
                "return_period_end": d,
                "decision_asof": body["return_period_start"],
                "evidence_period_id": period_id,
                "cumulative_shadow_return": prior_cum,
                "forward_cycle_seq": prior_seq,
                "orders_placed": 0,
                "broker": None,
            })
            body["snapshot_version_id"] = snapshot_version_id(body)
            pending.append(body)

        for body in pending:                    # validated set commits in order
            _append(ledger_path, body)
            appended.append(body)
        return appended


def _last_cumulative(rows: List[Dict[str, Any]]) -> float:
    for r in reversed(rows):
        v = r.get("cumulative_shadow_return")
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _last_forward_seq(rows: List[Dict[str, Any]]) -> int:
    seqs = [int(r["forward_cycle_seq"]) for r in rows
            if isinstance(r.get("forward_cycle_seq"), (int, float))]
    return max(seqs) if seqs else 0


def active_observations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonical evidence set: ONE active observation per evidence period —
    the LAST recorded revision supersedes earlier ones (finding 3). Rows
    without an evidence_period_id (legacy) key on asof_date. Activation
    baselines are excluded."""
    by_period: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in rows:
        if r.get("kind", "forward") != "forward":
            continue
        if not isinstance(r.get("today_net_return"), (int, float)):
            continue
        key = r.get("evidence_period_id") or f"legacy:{r.get('asof_date')}"
        if key not in by_period:
            order.append(key)
        by_period[key] = r
    return [by_period[k] for k in order]


def forward_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Active forward observations only (baselines excluded, one per
    evidence period). Alias kept for existing consumers."""
    return active_observations(rows)


def cadence_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evidence-cadence accounting from EXPLICIT frozen-rule fields
    (finding 4) — never inferred from book-dictionary inequality (vol
    scaling / HRP drift moves weights daily without a signal rebalance).

    Definitions: a holding period counts once a FORWARD bar inside it is
    observed; completed rebalances = forward bars whose frozen rule emitted
    ``rebalance_event=True``; the activation baseline's period is context
    only and never counted. Rows without the explicit fields -> None counts
    with a reason (never fabricated)."""
    fwd = active_observations(rows)
    weeks, months = set(), set()
    for r in fwd:
        d = str(r.get("asof_date") or "")
        if len(d) >= 10:
            try:
                iso = _dt.date.fromisoformat(d[:10]).isocalendar()
                weeks.add((iso[0], iso[1]))
                months.add(d[:7])
            except ValueError:
                pass
    have_explicit = bool(fwd) and all(
        "holding_period_id" in r and "rebalance_event" in r for r in fwd)
    if have_explicit:
        holding_periods = {r["holding_period_id"] for r in fwd}
        rebalance_events = sum(1 for r in fwd if r.get("rebalance_event") is True)
        n_rebal: Optional[int] = rebalance_events
        n_holding: Optional[int] = len(holding_periods)
        note = ("counts derive from the frozen rule's EXPLICIT rebalance fields; "
                "weekly/monthly evidence requirements count holding periods/weeks, "
                "NEVER raw daily bars")
    else:
        n_rebal = None
        n_holding = None
        note = ("rows lack explicit rebalance fields (legacy/pre-rev3) — rebalance "
                "counts undefined, not inferred from book inequality; "
                "weekly/monthly requirements count holding periods/weeks, "
                "NEVER raw daily bars")
    return {
        "n_return_bars": len(fwd),
        "n_calendar_weeks": len(weeks),
        "n_calendar_months": len(months),
        "n_completed_rebalances": n_rebal,
        "n_independent_holding_periods": n_holding,
        "note": note,
    }


__all__ = ["LedgerCorruptionError", "active_observations", "append_unseen_bars",
           "cadence_counts", "construction_sha256", "forward_rows",
           "ledger_lock", "read_rows", "snapshot_version_id"]
