"""
PRD Watcher — Filesystem event listener that detects PRD completion.

Monitors .claude/ralph/prd*.json files. When ALL user stories in a PRD
have passes=true, fires a prd_complete event on the EventBus.

Supports two backends:
  1. watchdog (inotify) — instant detection on file modify
  2. Polling fallback — checks every N seconds

Phase 28 (US-169): Event-driven PRD completion detection.
"""

import json
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from src.scanner.automation.background_activity import get_background_activity_tracker
from src.scanner.automation.event_bus import Event, EventBus, EventType, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class PRDStatus:
    """Snapshot of a PRD file's completion state."""
    filepath: str
    project: str = ""
    branch_name: str = ""
    description: str = ""
    total_stories: int = 0
    passed_stories: int = 0
    failed_story_ids: List[str] = field(default_factory=list)
    is_complete: bool = False
    last_hash: str = ""
    last_checked: str = ""
    completed_at: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        if self.total_stories == 0:
            return 0.0
        return round(self.passed_stories / self.total_stories * 100, 1)


class PRDCompletionChecker:
    """
    Stateless checker that reads a PRD JSON and determines completion.
    """

    @staticmethod
    def check_file(filepath: Path) -> PRDStatus:
        """Parse a PRD file and return its completion status."""
        status = PRDStatus(filepath=str(filepath))
        try:
            content = filepath.read_text(encoding="utf-8")
            status.last_hash = hashlib.md5(content.encode()).hexdigest()
            status.last_checked = datetime.now(timezone.utc).isoformat()

            data = json.loads(content)
            status.project = data.get("project", "")
            status.branch_name = data.get("branchName", "")
            status.description = data.get("description", "")

            stories = data.get("userStories", [])
            status.total_stories = len(stories)
            status.passed_stories = 0
            status.failed_story_ids = []

            for story in stories:
                if story.get("passes", False) is True:
                    status.passed_stories += 1
                else:
                    status.failed_story_ids.append(story.get("id", "unknown"))

            status.is_complete = (
                status.total_stories > 0
                and status.passed_stories == status.total_stories
            )
            if status.is_complete:
                status.completed_at = status.last_checked

        except json.JSONDecodeError as e:
            logger.warning(f"PRDChecker: invalid JSON in {filepath}: {e}")
        except Exception as e:
            logger.error(f"PRDChecker: error reading {filepath}: {e}")

        return status


class PRDWatcher:
    """
    Filesystem watcher that monitors .claude/ralph/ for PRD completion events.

    On detection of a completed PRD (all stories pass), emits EventType.PRD_COMPLETE
    on the EventBus with the full PRDStatus as payload.

    Deduplication: tracks file hashes to avoid re-firing for the same content.
    """

    def __init__(
        self,
        ralph_dir: Optional[Path] = None,
        event_bus: Optional[EventBus] = None,
        poll_interval_secs: float = 10.0,
        glob_pattern: str = "prd*.json",
        on_complete_callback: Optional[Callable[[PRDStatus], None]] = None,
    ):
        self._ralph_dir = ralph_dir or Path(".claude/ralph")
        self._bus = event_bus or get_event_bus()
        self._poll_interval = poll_interval_secs
        self._glob_pattern = glob_pattern
        self._on_complete = on_complete_callback

        # Dedup state: {filepath: last_hash_when_complete_fired}
        self._fired_completions: Dict[str, str] = {}
        # Track in-progress hashes for change detection
        self._known_hashes: Dict[str, str] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog_observer = None
        self._activity_id: Optional[str] = None

        # State persistence
        self._state_file = self._ralph_dir / ".watcher_state.json"
        self._load_state()

    def scan_all(self) -> List[PRDStatus]:
        """
        Scan all PRD files once and return their statuses.
        Fires prd_complete for any newly completed PRDs.
        """
        if not self._ralph_dir.exists():
            logger.warning(f"PRDWatcher: ralph dir not found: {self._ralph_dir}")
            return []

        statuses = []
        for prd_file in sorted(self._ralph_dir.glob(self._glob_pattern)):
            if not prd_file.is_file():
                continue
            status = PRDCompletionChecker.check_file(prd_file)
            statuses.append(status)

            filepath_key = str(prd_file)
            old_hash = self._known_hashes.get(filepath_key)
            self._known_hashes[filepath_key] = status.last_hash

            # Skip if hash unchanged
            if old_hash == status.last_hash:
                continue

            # Check if newly complete and not already fired for this hash
            if status.is_complete:
                fired_hash = self._fired_completions.get(filepath_key)
                if fired_hash != status.last_hash:
                    self._fire_completion(status)
                    self._fired_completions[filepath_key] = status.last_hash
                    self._save_state()
            else:
                logger.debug(
                    f"PRDWatcher: {prd_file.name} → {status.passed_stories}/{status.total_stories} "
                    f"({status.progress_pct}%) — pending: {status.failed_story_ids}"
                )

        return statuses

    def _fire_completion(self, status: PRDStatus) -> None:
        """Emit prd_complete event and invoke optional callback."""
        logger.info(
            f"🎯 PRD COMPLETE: {Path(status.filepath).name} "
            f"({status.total_stories} stories, branch: {status.branch_name})"
        )

        event = Event(
            event_type=EventType.PRD_COMPLETE,
            source="prd_watcher",
            payload={
                "filepath": status.filepath,
                "project": status.project,
                "branch_name": status.branch_name,
                "description": status.description,
                "total_stories": status.total_stories,
                "passed_stories": status.passed_stories,
                "completed_at": status.completed_at,
            },
        )
        self._bus.emit(event)

        if self._on_complete:
            try:
                self._on_complete(status)
            except Exception as e:
                logger.error(f"PRDWatcher: on_complete callback error: {e}")

    # ─── Polling Backend ────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start the polling watcher on a background daemon thread."""
        if self._running:
            logger.warning("PRDWatcher: already running")
            return

        self._running = True
        tracker = get_background_activity_tracker()
        self._activity_id = tracker.start_activity(
            kind="watcher",
            title="PRD completion watcher",
            source="prd_watcher",
            status="running",
            activity_id=self._activity_id,
            metadata={"backend": "polling", "path": str(self._ralph_dir), "interval_secs": self._poll_interval},
        )
        tracker.append_event(self._activity_id, "Started PRD watcher polling backend")
        self._thread = threading.Thread(
            target=self._poll_loop, name="prd-watcher-poll", daemon=True
        )
        self._thread.start()
        logger.info(f"PRDWatcher: polling started (interval={self._poll_interval}s)")

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                self.scan_all()
            except Exception as e:
                logger.error(f"PRDWatcher: poll error: {e}")
            time.sleep(self._poll_interval)

    # ─── Watchdog (inotify) Backend ─────────────────────────────────

    def start_watchdog(self) -> bool:
        """
        Start inotify-based file watching via watchdog library.
        Returns False if watchdog not available (falls back to polling).
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
        except ImportError:
            logger.warning("PRDWatcher: watchdog not installed, falling back to polling")
            self.start_polling()
            return False

        watcher = self  # closure reference

        class PRDFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).name.startswith("prd") and event.src_path.endswith(".json"):
                    logger.debug(f"PRDWatcher: file modified: {event.src_path}")
                    watcher.scan_all()

            def on_created(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).name.startswith("prd") and event.src_path.endswith(".json"):
                    logger.debug(f"PRDWatcher: file created: {event.src_path}")
                    watcher.scan_all()

        self._running = True
        tracker = get_background_activity_tracker()
        self._activity_id = tracker.start_activity(
            kind="watcher",
            title="PRD completion watcher",
            source="prd_watcher",
            status="running",
            activity_id=self._activity_id,
            metadata={"backend": "watchdog", "path": str(self._ralph_dir)},
        )
        tracker.append_event(self._activity_id, "Started PRD watcher watchdog backend")
        observer = Observer()
        observer.schedule(PRDFileHandler(), str(self._ralph_dir), recursive=False)
        observer.start()
        self._watchdog_observer = observer
        logger.info(f"PRDWatcher: watchdog started on {self._ralph_dir}")

        # Initial scan
        self.scan_all()
        return True

    def start(self) -> None:
        """Start the best available backend (watchdog → polling fallback)."""
        if not self.start_watchdog():
            # Already fell back to polling inside start_watchdog
            pass

    def stop(self) -> None:
        """Stop the watcher (either backend)."""
        self._running = False
        if self._watchdog_observer:
            self._watchdog_observer.stop()
            self._watchdog_observer.join(timeout=5)
            self._watchdog_observer = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._activity_id:
            tracker = get_background_activity_tracker()
            tracker.update_activity(self._activity_id, status="completed", summary="PRD watcher stopped")
        self._save_state()
        logger.info("PRDWatcher: stopped")

    # ─── State Persistence ──────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist fired completions to avoid re-firing after restart."""
        try:
            state = {
                "fired_completions": self._fired_completions,
                "last_saved": datetime.now(timezone.utc).isoformat(),
            }
            self._state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"PRDWatcher: state save error: {e}")

    def _load_state(self) -> None:
        """Load previous fired completions from disk."""
        try:
            if self._state_file.exists():
                state = json.loads(self._state_file.read_text())
                self._fired_completions = state.get("fired_completions", {})
                logger.info(
                    f"PRDWatcher: loaded state — {len(self._fired_completions)} previous completions tracked"
                )
        except Exception as e:
            logger.warning(f"PRDWatcher: state load error: {e}")
            self._fired_completions = {}

    # ─── Status / Inspection ────────────────────────────────────────

    def get_status(self) -> dict:
        """Return watcher status for system health checks."""
        return {
            "running": self._running,
            "backend": "watchdog" if self._watchdog_observer else ("polling" if self._thread else "stopped"),
            "ralph_dir": str(self._ralph_dir),
            "poll_interval_secs": self._poll_interval,
            "tracked_files": len(self._known_hashes),
            "fired_completions": len(self._fired_completions),
            "fired_files": list(self._fired_completions.keys()),
        }

    def reset_fired(self, filepath: Optional[str] = None) -> None:
        """Reset fired state so a PRD can re-fire. None = reset all."""
        if filepath:
            self._fired_completions.pop(filepath, None)
        else:
            self._fired_completions.clear()
        self._save_state()
        logger.info(f"PRDWatcher: reset fired state for {'all' if not filepath else filepath}")
