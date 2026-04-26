"""HomeworkStore — atomic .jsonl I/O for pending and history files.

Mirrors AdjustmentApprover's two-file pattern: pending.jsonl + history.jsonl.
All writes are atomic (tmp + rename). All reads quarantine corrupt lines
rather than crash. fcntl advisory locks protect against concurrent operator
actions when running alongside the TUI.

See spec §3.2 (Storage) and §7 (Error handling).
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.scanner.automation.homework.types import HomeworkEntry

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_PENDING_PATH = _PROJECT_ROOT / ".claude" / "homework_pending.jsonl"
DEFAULT_HISTORY_PATH = _PROJECT_ROOT / ".claude" / "homework_history.jsonl"
DEFAULT_QUARANTINE_PATH = _PROJECT_ROOT / ".claude" / "homework_quarantine.jsonl"


class HomeworkStore:
    """File-backed store for HomeworkEntry pending → history transitions.

    Args:
        pending_path: where new homework entries land. Default: .claude/homework_pending.jsonl
        history_path: where graded entries are appended forever. Default: .claude/homework_history.jsonl
        quarantine_path: where corrupt JSONL lines go. Default: .claude/homework_quarantine.jsonl
    """

    def __init__(
        self,
        pending_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
        quarantine_path: Optional[Path] = None,
    ) -> None:
        self.pending_path = pending_path or DEFAULT_PENDING_PATH
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.quarantine_path = quarantine_path or DEFAULT_QUARANTINE_PATH

    # ---------------- public API ----------------

    def add(self, entry: HomeworkEntry) -> None:
        """Append entry to pending.jsonl with file lock + fsync."""
        self._append_locked(self.pending_path, dataclasses.asdict(entry))

    def list_pending(self) -> List[HomeworkEntry]:
        """Read all pending entries. Corrupt lines are quarantined.

        Idempotency note: any entry whose homework_id is already present in
        history is filtered out. This closes the crash window in
        move_to_history (history append succeeded, pending rewrite did not):
        the next list_pending call simply doesn't see the duplicate, and a
        retry of move_to_history will skip the history append (already in
        history) and proceed to rewrite pending. End state matches the
        no-crash path.
        """
        raw = self._read_jsonl(self.pending_path)
        history_ids = self._history_id_set()
        if not history_ids:
            return raw
        return [e for e in raw if e.homework_id not in history_ids]

    def list_history(self) -> List[HomeworkEntry]:
        """Read all graded entries from history."""
        return self._read_jsonl(self.history_path)

    def move_to_history(
        self,
        homework_id: str,
        grade: str,
        note: Optional[str],
        edits: Optional[dict],
    ) -> bool:
        """Find entry in pending by id, mark with grade, move to history.

        Returns True on success, False if homework_id not found.

        Atomicity: history append happens first (durable), then pending is
        rewritten without the moved entry. If we crash between those two
        steps the entry is briefly in both files, but list_pending filters
        out any id already in history, so the system stays consistent and
        a retry is safe. The history append itself is idempotent — if the
        entry is already in history we skip the append and only do the
        pending rewrite.
        """
        # Use the raw pending file (not the dedupe-filtered list_pending)
        # because we may be recovering from a crash where the entry is
        # both in pending and in history.
        pending = self._read_jsonl(self.pending_path)
        target_idx = next(
            (i for i, e in enumerate(pending) if e.homework_id == homework_id),
            None,
        )
        if target_idx is None:
            logger.warning("HomeworkStore.move_to_history: id %s not found", homework_id)
            return False

        target = pending.pop(target_idx)
        graded = dataclasses.replace(
            target,
            status=grade,
            operator_grade=grade,
            operator_note=note,
            operator_edits=edits,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Idempotent history append: skip if already there (crash-recovery path).
        if homework_id not in self._history_id_set():
            self._append_locked(self.history_path, dataclasses.asdict(graded))
        else:
            logger.info(
                "HomeworkStore.move_to_history: %s already in history; "
                "skipping append (recovery from prior crash)",
                homework_id,
            )

        # Then rewrite pending without the moved entry (atomic via tmp+rename).
        self._rewrite_atomic(
            self.pending_path,
            [dataclasses.asdict(e) for e in pending],
        )
        return True

    def rewrite_pending(self, entries: List[HomeworkEntry]) -> None:
        """Atomically rewrite the entire pending file with the given entries.

        Used by snooze (and any future bulk-edit operation) to update entries
        in place without going through the pending → history move path.
        """
        payloads = [dataclasses.asdict(e) for e in entries]
        self._rewrite_atomic(self.pending_path, payloads)

    def _history_id_set(self) -> set:
        """Set of homework_ids currently in history. Used for dedupe."""
        return {e.homework_id for e in self._read_jsonl(self.history_path)}

    # ---------------- internals ----------------

    def _append_locked(self, path: Path, payload: dict) -> None:
        """JSONL append protected by exclusive file lock + fsync.

        Not "atomic" in the rename-replace sense (that would clobber prior
        lines for an append). The durability guarantee here is: under
        concurrent writers, lines do not interleave (flock), and a written
        line survives a crash (fsync). Append-only files are inherently
        crash-safe at line granularity.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _rewrite_atomic(self, path: Path, payloads: List[dict]) -> None:
        """Rewrite the entire file atomically via tmp + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as tmp:
            for p in payloads:
                tmp.write(json.dumps(p, sort_keys=True, default=str) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.rename(tmp_name, str(path))

    def _read_jsonl(self, path: Path) -> List[HomeworkEntry]:
        if not path.exists():
            return []
        entries: List[HomeworkEntry] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entries.append(HomeworkEntry(**obj))
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    self._quarantine(path, line_no, line, str(e))
        return entries

    def _quarantine(self, source: Path, line_no: int, line: str, reason: str) -> None:
        """Append a corrupt line to the quarantine file with context."""
        try:
            self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.quarantine_path, "a", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(json.dumps({
                    "source": str(source),
                    "line_no": line_no,
                    "line": line[:1000],
                    "reason": reason,
                    "quarantined_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            logger.warning(
                "HomeworkStore: quarantined corrupt line %d in %s: %s",
                line_no, source, reason,
            )
        except OSError as e:
            logger.exception("HomeworkStore._quarantine failed: %s", e)
