"""Code-freshness detector — lets a long-running daemon notice it is running stale code.

Root cause this closes (2026-07-03 autonomy audit): daemons run whatever code was on
disk when they started; nothing ever restarted them, so committed fixes never went
live ("code-on-disk vs code-running", .claude/rules/honesty.md #4). Each daemon
constructs a CodeFreshness at startup and polls ``changed()`` once per loop tick:

- the tier7 supervisor re-execs itself (os.execv) on change (it runs outside launchd);
- the OANDA trend runner does a CLEAN exit(0) on change and lets its launchd
  KeepAlive respawn it with fresh code (it never restarts itself mid-cycle).

Detection signals (both cheap, both read-only):
1. ``git rev-parse HEAD`` — any committed change to anything in the repo.
2. mtime of an explicit watch-list of files (the daemon's own entrypoint by
   default) — catches uncommitted edits to the entrypoint itself.

Fail-open BY DESIGN toward "not changed": if git is unavailable or a watched file
is unreadable, that signal is skipped rather than forcing a restart loop. A restart
is an availability action, not a safety gate — the safety gates (halt, practice pin)
are enforced elsewhere and are re-read by the fresh process anyway.

A change must be OBSERVED ON TWO CONSECUTIVE POLLS before ``changed()`` reports it
(debounce): a half-written file save or mid-rebase HEAD never triggers a restart on
the first sighting.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_S = 5.0


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
        )
        head = out.stdout.strip()
        return head if out.returncode == 0 and head else None
    except (OSError, subprocess.SubprocessError):
        return None


def _mtimes(paths: Sequence[Path]) -> Tuple[Tuple[str, float], ...]:
    out: List[Tuple[str, float]] = []
    for p in paths:
        try:
            out.append((str(p), p.stat().st_mtime))
        except OSError:
            # Unreadable/missing watch file: skip the signal (see module docstring).
            continue
    return tuple(out)


class CodeFreshness:
    """Snapshot code identity at construction; ``changed()`` says if disk moved on."""

    def __init__(self, repo_root: Path, watch_files: Optional[Sequence[Path]] = None):
        self.repo_root = Path(repo_root)
        self.watch_files: Tuple[Path, ...] = tuple(watch_files or ())
        self.baseline_head = _git_head(self.repo_root)
        if self.baseline_head is None and (self.repo_root / ".git").exists():
            # Verifier finding (2026-07-03): without this line the git signal degrades
            # SILENTLY for the whole process lifetime — make the hole observable.
            logger.warning("code_freshness: .git exists but HEAD unreadable at baseline — "
                           "committed changes will NOT trigger restarts for this process")
        self.baseline_mtimes = _mtimes(self.watch_files)
        # Debounce: the pending (not yet confirmed) observation from the last poll.
        self._pending: Optional[str] = None

    def _observe(self) -> Optional[str]:
        """One raw observation. Returns a reason string if disk differs from baseline."""
        head = _git_head(self.repo_root)
        if self.baseline_head and head and head != self.baseline_head:
            return f"git HEAD moved {self.baseline_head[:8]} -> {head[:8]}"
        current = dict(_mtimes(self.watch_files))
        for path, base_mtime in self.baseline_mtimes:
            cur = current.get(path)
            if cur is not None and cur != base_mtime:
                return f"watched file changed on disk: {path}"
        return None

    def changed(self) -> Optional[str]:
        """Reason string once the SAME change is seen on two consecutive polls, else None."""
        reason = self._observe()
        if reason is None:
            self._pending = None
            return None
        if self._pending == reason:
            return reason
        self._pending = reason  # first sighting — confirm on the next poll (debounce)
        return None
