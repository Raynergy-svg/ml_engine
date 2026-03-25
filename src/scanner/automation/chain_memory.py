"""
Chain Memory — Hot-state persistence for the PRD improvement loop.

Written by:
  - Ralph (or user) as stories complete
  - PRDAgentChain as chain events fire
  - Scheduled sweep on each run

Read by:
  - Scheduled sweep to know where we are
  - Chain agents to avoid duplicate work
  - Any session that needs loop context

File: .claude/ralph/memory.json

This is NOT a replacement for state.json (which tracks phases) or
watcher_state.json (which tracks fired completions). This is the
real-time progress tracker that bridges the gap between fast iteration
and the 5-minute sweep interval.
"""

import fcntl
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = ".claude/ralph/memory.json"


class ChainMemory:
    """
    Thread-safe, file-locked read/write access to the chain memory file.
    Designed for multiple writers (Ralph, chain, sweep, user sessions).
    """

    def __init__(self, project_root: Optional[Path] = None):
        self._root = project_root or Path(".")
        self._path = self._root / DEFAULT_MEMORY_PATH
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write_locked({
                "version": 2,
                "created": datetime.now(timezone.utc).isoformat(),
                "current_prd": {
                    "file": ".claude/ralph/prd.json",
                    "phase": "unknown",
                    "total_stories": 0,
                    "passed_stories": 0,
                    "story_status": {},
                    "last_story_completed": None,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
                "loop_history": {
                    "total_phases_completed": 0,
                    "total_chain_runs": 0,
                    "total_stories_implemented": 0,
                    "phases": [],
                    "last_chain_result": None,
                },
                "pending_actions": [],
                "flags": {
                    "ralph_in_progress": False,
                    "review_in_progress": False,
                    "chain_in_progress": False,
                    "manual_override": False,
                },
            })

    def _read_locked(self) -> dict:
        """Read with shared file lock."""
        try:
            with open(self._path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"ChainMemory: read error ({e}), returning empty")
            return {}

    def _write_locked(self, data: dict) -> None:
        """Write with exclusive file lock."""
        try:
            with open(self._path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"ChainMemory: write error: {e}")

    def _update(self, updater) -> dict:
        """Read-modify-write with exclusive lock."""
        try:
            with open(self._path, "r+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                data = json.load(f)
                updater(data)
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except Exception as e:
            logger.error(f"ChainMemory: update error: {e}")
            return self._read_locked()

    # ─── Public API ──────────────────────────────────────────────

    def read(self) -> dict:
        """Get current memory state."""
        return self._read_locked()

    def get_flags(self) -> dict:
        """Get current flags."""
        return self._read_locked().get("flags", {})

    def get_current_prd(self) -> dict:
        """Get current PRD tracking info."""
        return self._read_locked().get("current_prd", {})

    # ─── Story Progress (called as Ralph/user completes stories) ─

    def mark_story_complete(self, story_id: str, title: str = "") -> None:
        """Mark a single story as complete in memory."""
        now = datetime.now(timezone.utc).isoformat()

        def _update(data):
            prd = data.setdefault("current_prd", {})
            status = prd.setdefault("story_status", {})
            status[story_id] = {"passed": True, "completed_at": now, "title": title}
            prd["passed_stories"] = sum(1 for v in status.values() if v.get("passed"))
            prd["last_story_completed"] = story_id
            prd["last_updated"] = now

        self._update(_update)
        logger.info(f"ChainMemory: story {story_id} marked complete")

    def sync_from_prd(self, prd_path: Optional[Path] = None) -> dict:
        """
        Sync memory from the actual prd.json file.
        Call this to reconcile memory with disk after fast iteration.
        Returns the synced PRD status.
        """
        prd_file = prd_path or (self._root / ".claude" / "ralph" / "prd.json")
        if not prd_file.exists():
            return {}

        try:
            prd_data = json.loads(prd_file.read_text())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"ChainMemory: can't read PRD: {e}")
            return {}

        stories = prd_data.get("userStories", [])
        now = datetime.now(timezone.utc).isoformat()

        story_status = {}
        passed = 0
        for s in stories:
            sid = s.get("id", "unknown")
            is_passed = s.get("passes", False)
            if is_passed:
                passed += 1
            story_status[sid] = {
                "passed": is_passed,
                "completed_at": now if is_passed else None,
                "title": s.get("title", "")[:80],
            }

        synced = {
            "file": str(prd_file.relative_to(self._root)) if self._root in prd_file.parents else str(prd_file),
            "phase": prd_data.get("branchName", prd_data.get("project", "unknown")),
            "total_stories": len(stories),
            "passed_stories": passed,
            "story_status": story_status,
            "last_story_completed": next(
                (sid for sid, v in story_status.items() if v["passed"]), None
            ),
            "last_updated": now,
            "all_complete": passed == len(stories) and len(stories) > 0,
        }

        def _update(data):
            data["current_prd"] = synced

        self._update(_update)
        logger.info(
            f"ChainMemory: synced from PRD — {passed}/{len(stories)} stories "
            f"({'COMPLETE' if synced['all_complete'] else 'in progress'})"
        )
        return synced

    # ─── Flag Management (called by chain agents) ───────────────

    def set_flag(self, flag: str, value: bool) -> None:
        """Set a flag (ralph_in_progress, review_in_progress, etc.)."""
        def _update(data):
            data.setdefault("flags", {})[flag] = value

        self._update(_update)

    def is_busy(self) -> bool:
        """Check if any agent is currently working."""
        flags = self.get_flags()
        return any([
            flags.get("ralph_in_progress", False),
            flags.get("review_in_progress", False),
            flags.get("chain_in_progress", False),
        ])

    # ─── Chain Events (called by PRDAgentChain) ─────────────────

    def record_chain_start(self, phase: str, gaps: int) -> None:
        """Record that a chain run has started."""
        now = datetime.now(timezone.utc).isoformat()

        def _update(data):
            data.setdefault("flags", {})["chain_in_progress"] = True
            data.setdefault("pending_actions", []).append({
                "action": "chain_started",
                "phase": phase,
                "gaps": gaps,
                "timestamp": now,
            })

        self._update(_update)

    def record_chain_complete(self, result: dict) -> None:
        """Record chain completion and update loop history."""
        now = datetime.now(timezone.utc).isoformat()

        def _update(data):
            # Clear busy flags
            flags = data.setdefault("flags", {})
            flags["chain_in_progress"] = False
            flags["ralph_in_progress"] = False
            flags["review_in_progress"] = False

            # Update loop history
            history = data.setdefault("loop_history", {})
            history["total_chain_runs"] = history.get("total_chain_runs", 0) + 1
            history["last_chain_result"] = {
                "timestamp": now,
                "review_passed": result.get("review_passed", True),
                "blockers": result.get("blockers", 0),
                "review_cycles_used": result.get("review_cycles_used", 0),
            }

            # Clear pending actions for this chain
            data["pending_actions"] = [
                a for a in data.get("pending_actions", [])
                if a.get("action") != "chain_started"
            ]

        self._update(_update)
        logger.info(f"ChainMemory: chain complete recorded")

    def record_phase_complete(self, phase_name: str, stories_count: int) -> None:
        """Record that a phase (full PRD) completed."""
        def _update(data):
            history = data.setdefault("loop_history", {})
            phases = history.setdefault("phases", [])
            if phase_name not in phases:
                phases.append(phase_name)
            history["total_phases_completed"] = len(phases)
            history["total_stories_implemented"] = (
                history.get("total_stories_implemented", 0) + stories_count
            )

        self._update(_update)

    # ─── Quick Status (for sweep/status checks) ─────────────────

    def summary(self) -> str:
        """One-line status string."""
        data = self._read_locked()
        prd = data.get("current_prd", {})
        flags = data.get("flags", {})
        history = data.get("loop_history", {})

        busy_parts = []
        if flags.get("ralph_in_progress"):
            busy_parts.append("ralph")
        if flags.get("review_in_progress"):
            busy_parts.append("review")
        if flags.get("chain_in_progress"):
            busy_parts.append("chain")
        busy = f" | BUSY: {'+'.join(busy_parts)}" if busy_parts else ""

        return (
            f"PRD {prd.get('passed_stories', 0)}/{prd.get('total_stories', 0)} "
            f"({prd.get('phase', '?')}) | "
            f"Phases: {history.get('total_phases_completed', 0)} | "
            f"Chains: {history.get('total_chain_runs', 0)}"
            f"{busy}"
        )


# ─── Module-level convenience ────────────────────────────────────

_instance: Optional[ChainMemory] = None


def get_chain_memory(project_root: Optional[Path] = None) -> ChainMemory:
    global _instance
    if _instance is None:
        _instance = ChainMemory(project_root=project_root)
    return _instance


def reset_chain_memory() -> None:
    global _instance
    _instance = None
