"""Versioned history of agent_weights.json writes for rollback safety.

Every successful write to `trained_data/models/agent_weights.json` should
append one JSONL line to `trained_data/weight_history.jsonl` via
`record_weight_version()`. The `restore_version()` helper writes a chosen
historical snapshot back to the weights file and appends a new history
entry tagged ``rollback:<from-hash>`` so the chain stays monotonic.

Locking contract
----------------
The existing sidecar lock at
``trained_data/models/agent_weights.json.lock`` (created by
``TrainingSignalApplicator``) is reused as the cross-process critical
section guarding BOTH the weights file AND this history file.

Callers that already hold that flock (currently only
``TrainingSignalApplicator._write_weights_atomic``) must pass
``_under_lock=True`` to skip re-acquisition. All other call sites pass the
default (``_under_lock=False``) and we acquire+release here.

Record schema (one JSON object per JSONL line)::

    {
      "ts_iso":            "<UTC ISO8601 timestamp>",
      "source":            "<rl_update|training_signal:...|self_heal:...|...>",
      "snapshot_hash":     "<sha256 of canonical-form weights>",
      "weights_snapshot":  {<full regime-keyed weights dict>},
      "reason":            "<free-text annotation, optional>"
    }

Rotation: when the JSONL file exceeds ``MAX_LINES`` entries the live file
is gzip-archived to ``weight_history.jsonl.<ts>.gz`` and a fresh empty
JSONL is started — both moves happen under the same lock.

Deduplication: if the new snapshot's hash matches the most-recent entry's
hash, the append is skipped (no-op). Prevents flooding the history with
identical entries on repeated saves of unchanged weights.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = _PROJECT_ROOT / "trained_data" / "weight_history.jsonl"
LOCK_PATH = _PROJECT_ROOT / "trained_data" / "models" / "agent_weights.json.lock"
MAX_LINES = 5000  # rotation trigger


def canonical_hash(weights: Dict[str, Any]) -> str:
    """SHA256 of canonical-form weights JSON.

    Canonical form: sorted keys, compact separators, no trailing whitespace.
    Two semantically-equal dicts produce the same hash regardless of insertion order.
    """
    payload = json.dumps(weights, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_last_hash(path: Path) -> Optional[str]:
    """Return the snapshot_hash of the last valid JSONL line, or None."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-4096, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("snapshot_hash")
            if isinstance(h, str):
                return h
        return None
    except OSError as e:
        logger.debug("weight_history: failed to read last hash from %s: %s", path, e)
        return None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _rotate(path: Path) -> None:
    """Gzip-archive the current JSONL file and start fresh.

    Caller is responsible for holding the flock. Uses tmp+rename so a crash
    mid-rotation leaves either the original file OR the gzip archive intact,
    never a half-written state.
    """
    if not path.exists():
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_suffix(path.suffix + f".{ts}.gz")
    tmp = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.rename(tmp, archive)
        # Truncate live file to empty; preserves inode + permissions.
        with open(path, "w", encoding="utf-8"):
            pass
        logger.info("weight_history: rotated %s -> %s", path.name, archive.name)
    except OSError as e:
        logger.warning("weight_history: rotation failed for %s: %s", path, e)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class _FlockGuard:
    """Context manager wrapping fcntl.flock(LOCK_EX) on the sidecar lock file.

    Reused by both weight_history.record_weight_version() and any caller
    that needs to coordinate with TrainingSignalApplicator's existing lock.
    """

    def __init__(self, lock_path: Optional[Path] = None):
        self._lock_path = lock_path if lock_path is not None else LOCK_PATH
        self._fh = None

    def __enter__(self) -> "_FlockGuard":
        # Resolve at enter time so test monkeypatching of LOCK_PATH takes effect.
        if self._lock_path is LOCK_PATH or self._lock_path is None:
            self._lock_path = LOCK_PATH
        _ensure_parent(self._lock_path)
        self._fh = open(self._lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._fh.close()
            self._fh = None


def record_weight_version(
    weights: Dict[str, Any],
    *,
    source: str,
    reason: str = "",
    path: Optional[Path] = None,
    _under_lock: bool = False,
) -> Optional[str]:
    """Append one history entry for the given weights snapshot.

    Returns the snapshot hash on append, or ``None`` if the call was
    skipped (duplicate hash, or write failure).

    :param weights: the full regime-keyed weights dict that was just saved.
    :param source: short tag identifying the writer (e.g. ``"rl_update"``,
        ``"training_signal:<homework_id>"``, ``"self_heal:soft_reset:trend"``).
    :param reason: optional free-text annotation (operator notes, etc.).
    :param path: history JSONL path; defaults to :data:`HISTORY_PATH`.
    :param _under_lock: pass ``True`` only if the caller already holds the
        sidecar flock on :data:`LOCK_PATH`. Default ``False`` acquires it.
    """
    if not isinstance(weights, dict):
        logger.warning("weight_history: refusing to record non-dict weights (got %s)", type(weights).__name__)
        return None

    if path is None:
        path = HISTORY_PATH
    snapshot_hash = canonical_hash(weights)

    def _append_unlocked() -> Optional[str]:
        _ensure_parent(path)
        last = _read_last_hash(path)
        if last == snapshot_hash:
            return None  # dedup
        record = {
            "ts_iso": _utc_now_iso(),
            "source": str(source),
            "snapshot_hash": snapshot_hash,
            "weights_snapshot": weights,
            "reason": str(reason),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        try:
            with open(path, "ab") as fh:
                fh.write(line.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as e:
            logger.warning("weight_history: append to %s failed: %s", path, e)
            return None
        if _count_lines(path) > MAX_LINES:
            _rotate(path)
        return snapshot_hash

    try:
        if _under_lock:
            return _append_unlocked()
        with _FlockGuard():
            return _append_unlocked()
    except OSError as e:
        logger.warning("weight_history: lock acquire failed: %s", e)
        return None


def list_versions(limit: int = 20, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the most-recent ``limit`` history entries, newest first.

    Malformed JSONL lines are skipped with a debug log. Returns an empty list
    if the history file does not exist or contains no valid entries.
    """
    if path is None:
        path = HISTORY_PATH
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        logger.warning("weight_history: list_versions read failed: %s", e)
        return []

    out: List[Dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("weight_history: skipping malformed line")
            continue
        if not isinstance(rec, dict):
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def find_version(hash_prefix: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the most-recent history entry whose snapshot_hash starts with ``hash_prefix``.

    Returns ``None`` if not found. Hash prefixes shorter than 4 chars are rejected
    to avoid accidental wildcard matches.
    """
    if not hash_prefix or len(hash_prefix) < 4:
        logger.warning("weight_history: hash_prefix '%s' too short (need >= 4 chars)", hash_prefix)
        return None
    if path is None:
        path = HISTORY_PATH
    for rec in list_versions(limit=10_000, path=path):
        h = rec.get("snapshot_hash")
        if isinstance(h, str) and h.startswith(hash_prefix):
            return rec
    return None


def restore_version(
    hash_prefix: str,
    *,
    weights_path: Path,
    history_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Restore the weights file to the snapshot matching ``hash_prefix``.

    Writes the historical ``weights_snapshot`` back to ``weights_path`` via
    tmp+fsync+rename, then appends a new history entry tagged
    ``rollback:<from-hash-prefix>`` so the chain remains monotonic.

    Acquires the sidecar flock for the entire critical section (weights
    write + history append). Raises ``LookupError`` if no matching entry
    is found.
    """
    if history_path is None:
        history_path = HISTORY_PATH
    with _FlockGuard():
        rec = find_version(hash_prefix, path=history_path)
        if rec is None:
            raise LookupError(f"No history entry matches hash prefix '{hash_prefix}'")
        snapshot = rec.get("weights_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(f"History entry {rec.get('snapshot_hash')} has invalid weights_snapshot")

        # tmp+fsync+rename
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights_path.with_suffix(weights_path.suffix + ".restore.tmp")
        payload = json.dumps(snapshot, indent=2, sort_keys=True, default=str)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, weights_path)

        from_hash_short = (rec.get("snapshot_hash") or "")[:8]
        record_weight_version(
            snapshot,
            source=f"rollback:{from_hash_short}",
            reason=f"Restored from history entry ts={rec.get('ts_iso')} source={rec.get('source')}",
            path=history_path,
            _under_lock=True,
        )
        logger.info(
            "weight_history: restored %s from %s (source=%s)",
            weights_path,
            from_hash_short,
            rec.get("source"),
        )
        return rec
