"""Atomic dashboard job-state journal for Phase-M observability."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.evidence.hashing import sha256_bytes

_TRANSITIONS = {
    None: {"submitted"},
    "submitted": {"running", "failed"},
    "running": {"completed", "failed"},
    "failed": set(),
    "completed": set(),
}
_RESERVED_FIELDS = {"job_id", "status", "updated_at", "submitted_at", "started_at", "completed_at"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingJobJournal:
    """One atomic JSON projection per authorized dashboard training request."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        return self.root / f"{sha256_bytes(job_id.encode('utf-8'))}.json"

    def transition(
        self,
        job_id: str,
        status: str,
        *,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in {"submitted", "running", "failed", "completed"}:
            raise ValueError(f"unsupported job status {status!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".jobs.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                path = self._path(job_id)
                existing: dict[str, Any] = {}
                if path.exists():
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise ValueError(f"job journal entry is unreadable: {exc}") from exc
                    if not isinstance(loaded, dict) or loaded.get("job_id") != job_id:
                        raise ValueError("job journal identity mismatch")
                    existing = loaded
                previous = existing.get("status")
                if status not in _TRANSITIONS.get(previous, set()):
                    raise ValueError(f"invalid job transition {previous!r} -> {status!r}")
                timestamp = _now()
                supplied = {key: value for key, value in dict(fields or {}).items() if key not in _RESERVED_FIELDS}
                row = {**existing, **supplied, "job_id": job_id, "status": status, "updated_at": timestamp}
                if status == "submitted":
                    row["submitted_at"] = timestamp
                elif status == "running":
                    row["started_at"] = timestamp
                elif status in {"failed", "completed"}:
                    row["completed_at"] = timestamp
                self._atomic_write(path, row)
                return row
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".job.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


__all__ = ["TrainingJobJournal"]
