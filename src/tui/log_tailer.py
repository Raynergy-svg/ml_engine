"""LogTailer (pick #7) — incremental tail of a growing log file.

Adapted from the seek-resume pattern in `ReflectionLogReader` (app.py:194)
but pulled out into a reusable, side-effect-free module so it can be unit
tested without spinning up Textual. The LogViewerModal mounts a LogTailer
and polls it on a Textual interval.

Behavior:
  * read_new_lines() returns lines appended since the last call.
  * If the file shrinks (rotation, truncate), the offset resets to 0 and
    the next call returns the file from the start.
  * If the file disappears, the next call returns []; once it reappears
    the tail resumes from offset 0.
  * Long lines (> per_line_cap) are truncated with a "…[truncated]" tail
    so a mega-traceback can't spike memory.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_PER_LINE_CAP = 10 * 1024  # 10 KB
DEFAULT_INITIAL_TAIL_BYTES = 64 * 1024  # 64 KB — on first read, show last 64KB
DEFAULT_HISTORY_LINES = 500


@dataclass
class TailChunk:
    """Result of one `read_new_lines()` call."""

    lines: list[str]
    rotated: bool = False  # True if the file shrank since last read


class LogTailer:
    """Stateful, single-file tailer. Not thread-safe — callers must externally
    serialize access (each modal owns one tailer, polled from one worker).

    Args:
        path: file to tail
        per_line_cap: max characters per emitted line
        initial_tail_bytes: on the very first read, seek `-initial_tail_bytes`
            from end so the user sees recent context instead of waiting for
            new lines.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        per_line_cap: int = DEFAULT_PER_LINE_CAP,
        initial_tail_bytes: int = DEFAULT_INITIAL_TAIL_BYTES,
    ) -> None:
        self._path = Path(path)
        self._per_line_cap = per_line_cap
        self._initial_tail_bytes = initial_tail_bytes
        self._offset: int = 0
        self._first_read: bool = True

    @property
    def path(self) -> Path:
        return self._path

    def reset(self) -> None:
        """Re-seek to the initial-tail position on the next call."""
        self._offset = 0
        self._first_read = True

    def read_new_lines(self) -> TailChunk:
        """Read whatever has been appended since the last call.

        Empty list when nothing new is available (or when the file is
        missing). Caller may treat ``rotated=True`` as a signal to clear
        the displayed buffer.
        """
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            # File rotated out from under us; reset and wait for it to reappear.
            self._offset = 0
            self._first_read = True
            return TailChunk(lines=[], rotated=False)
        except OSError as e:
            logger.debug("LogTailer: stat failed for %s: %s", self._path, e)
            return TailChunk(lines=[], rotated=False)

        rotated = False
        start = self._offset
        if self._first_read:
            # On first read, seek backward from end so we surface recent context.
            start = max(0, size - self._initial_tail_bytes)
            self._first_read = False
        elif size < self._offset:
            # File shrank → rotation/truncate → start over.
            start = 0
            rotated = True

        if start >= size:
            self._offset = size
            return TailChunk(lines=[], rotated=rotated)

        try:
            with self._path.open("rb") as f:
                f.seek(start)
                data = f.read(size - start)
        except OSError as e:
            logger.debug("LogTailer: read failed for %s: %s", self._path, e)
            return TailChunk(lines=[], rotated=rotated)

        text = data.decode("utf-8", errors="replace")
        # If we seeked into the middle of a UTF-8 sequence (very rare with
        # replace=errors) or into the middle of a logical line, drop the
        # partial leading line on the first-from-tail read so we don't
        # show half-lines.
        if start > 0 and self._offset == 0 and "\n" in text:
            text = text.split("\n", 1)[1]
            start += len(data) - len(text.encode("utf-8"))

        lines: list[str] = []
        for raw in text.splitlines():
            if len(raw) > self._per_line_cap:
                lines.append(raw[: self._per_line_cap] + "…[truncated]")
            else:
                lines.append(raw)

        self._offset = size
        return TailChunk(lines=lines, rotated=rotated)


def tail_last_n_lines(
    path: Path | str, *, n: int = DEFAULT_HISTORY_LINES,
    per_line_cap: int = DEFAULT_PER_LINE_CAP,
) -> list[str]:
    """One-shot helper: return the last `n` non-empty lines of a file.

    Used by the modal on first open to seed the RichLog before the
    incremental tailer takes over. Safe on missing files (returns []).
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = p.read_bytes()
    except OSError as e:
        logger.debug("tail_last_n_lines: %s: %s", p, e)
        return []

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > n:
        lines = lines[-n:]
    out: list[str] = []
    for raw in lines:
        if len(raw) > per_line_cap:
            out.append(raw[:per_line_cap] + "…[truncated]")
        else:
            out.append(raw)
    return out


def filter_lines(lines: list[str], query: str) -> list[str]:
    """Pure, case-insensitive substring filter — empty query is identity."""
    q = query.strip().lower()
    if not q:
        return list(lines)
    return [ln for ln in lines if q in ln.lower()]


def known_log_files(repo_root: Path) -> list[Path]:
    """List the log files the viewer should offer.

    Order matters: first entry is the default selection. Only existing
    files are returned so the dropdown never shows ghosts; the modal
    falls back to a placeholder when none exist.
    """
    candidates = [
        repo_root / "logs" / "buddy_tui.stderr.log",
        repo_root / "logs" / "buddy_boot_trace.log",
        repo_root / "logs" / "reflection_log.jsonl",
    ]
    return [p for p in candidates if p.exists()]
