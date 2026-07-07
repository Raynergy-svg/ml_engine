"""Pre-registration ledger for brain-loop hypotheses — frozen before results.

A hypothesis's parameters are hashed at registration time. A result can never
attach to a hypothesis whose params don't match that frozen hash — this is the
mechanized version of the pre-registration discipline this repo already uses in
``docs/experiment-*.md`` (freeze the design, then run it), operationalized so it
can't be silently reshaped to fit a favorable result after the fact.

Single append-only, hash-chained JSONL ledger (same tamper-evidence pattern as
``src/equity/cycle_ledger.py``: each record commits to the previous record's
hash, so any edit/insert/delete is detectable by :func:`verify_chain`). Three
record kinds share one chain: ``register``, ``result``, ``gate_verdict``.

NON-hot-path: imports nothing from the FX/OANDA runtime. Writes only under
whatever path the caller supplies (by convention, ``trained_data/brain_loop/``).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

GENESIS_HASH = "0" * 64

# Fields committed to the hash for each record kind (record_hash is appended after).
_HASHED_FIELDS = (
    "seq",
    "ts",
    "kind",
    "hypothesis_id",
    "name",
    "mechanism",
    "novelty_justification",
    "params_hash",
    "status",
    "result_summary",
    "params_used_hash",
    "decision",
    "reasons",
    "prev_hash",
)


class HypothesisNotRegistered(LookupError):
    """Raised when a result/verdict is attached to a hypothesis_id never registered."""


class FrozenParamsViolation(ValueError):
    """Raised when a result's params don't match the hash frozen at registration."""


def canonical_hash(payload: Optional[Mapping[str, Any]]) -> str:
    """Order-invariant, value-sensitive sha256 fingerprint of a params dict.

    Mirrors ``src.equity.cycle_ledger.target_weight_hash``'s canonicalization
    approach (sorted items, fixed separators, no NaN) generalized to arbitrary
    JSON-safe param dicts rather than just float weights.
    """
    items = sorted((str(k), v) for k, v in (payload or {}).items())
    canonical = json.dumps(items, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive cross-process flock for a read-modify-write section.

    Same pattern as ``src.equity.market_calendar._TTLHaltStore._locked``: the
    lock lives on a sibling ``<ledger>.lock`` file (never the ledger itself,
    so the atomic tmp+rename in ``_atomic_append_line`` never swaps the inode
    the lock is held on). Without this, two concurrent writers could each
    read the same tail (``prev_hash``/``seq``), and the second's write would
    silently clobber the first's freshly-appended record — a lost-update
    race that ``verify_chain`` cannot detect after the fact, since the
    surviving file's chain still looks internally consistent.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


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
    obj = json.loads(last)
    return (str(obj["record_hash"]), count)


def _record_hash(payload: dict) -> str:
    canonical = json.dumps(
        {k: payload.get(k) for k in _HASHED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append_record(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    with _locked(path):
        prev_hash, seq = _last_hash_and_seq(path)
        full = dict(payload)
        full["seq"] = seq
        full["ts"] = datetime.now(timezone.utc).isoformat()
        full["prev_hash"] = prev_hash
        rec_hash = _record_hash(full)
        full["record_hash"] = rec_hash
        _atomic_append_line(
            path, json.dumps(full, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    return full


def _find_registration(records: List[dict], hypothesis_id: str) -> Optional[dict]:
    for rec in records:
        if rec.get("kind") == "register" and rec.get("hypothesis_id") == hypothesis_id:
            return rec
    return None


def register_hypothesis(
    ledger_path: Path,
    *,
    hypothesis_id: str,
    name: str,
    mechanism: str,
    novelty_justification: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze a hypothesis's params BEFORE any backtest runs. Append-only."""
    path = Path(ledger_path)
    payload = {
        "kind": "register",
        "hypothesis_id": str(hypothesis_id),
        "name": str(name),
        "mechanism": str(mechanism),
        "novelty_justification": str(novelty_justification),
        "params_hash": canonical_hash(params),
        "status": "REGISTERED",
    }
    return _append_record(path, payload)


def attach_result(
    ledger_path: Path,
    hypothesis_id: str,
    *,
    params_used: Mapping[str, Any],
    result_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach a backtest result to a previously-registered hypothesis.

    Refuses (``FrozenParamsViolation``) if ``params_used`` doesn't hash to the
    value frozen at registration — the params cannot be quietly reshaped to fit
    a favorable result after the fact.
    """
    path = Path(ledger_path)
    records = read_ledger(path)
    reg = _find_registration(records, hypothesis_id)
    if reg is None:
        raise HypothesisNotRegistered(
            f"attach_result: hypothesis_id={hypothesis_id!r} was never registered"
        )
    used_hash = canonical_hash(params_used)
    if used_hash != reg["params_hash"]:
        raise FrozenParamsViolation(
            f"attach_result: params_used hash {used_hash} != frozen hash "
            f"{reg['params_hash']} for hypothesis_id={hypothesis_id!r}"
        )
    payload = {
        "kind": "result",
        "hypothesis_id": str(hypothesis_id),
        "params_used_hash": used_hash,
        "result_summary": dict(result_summary),
    }
    return _append_record(path, payload)


def record_gate_verdict(
    ledger_path: Path,
    hypothesis_id: str,
    *,
    decision: str,
    reasons: Sequence[str],
) -> Dict[str, Any]:
    """Record the ship-gate verdict for a previously-registered hypothesis."""
    path = Path(ledger_path)
    records = read_ledger(path)
    if _find_registration(records, hypothesis_id) is None:
        raise HypothesisNotRegistered(
            f"record_gate_verdict: hypothesis_id={hypothesis_id!r} was never registered"
        )
    payload = {
        "kind": "gate_verdict",
        "hypothesis_id": str(hypothesis_id),
        "decision": str(decision),
        "reasons": [str(r) for r in reasons],
    }
    return _append_record(path, payload)


def read_ledger(ledger_path: Path) -> List[Dict[str, Any]]:
    """Parse the ledger into dicts. Missing file -> empty list."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        out.append(json.loads(raw))
    return out


def verify_chain(ledger_path: Path) -> Tuple[bool, Optional[int]]:
    """Recompute the hash chain; return ``(ok, first_broken_seq)``."""
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
