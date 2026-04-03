"""Session registry for tracking active Buddy watch-mode sessions.

Persists session state to trained_data/session_registry.json via safe_json
for operator visibility across restarts. Thread-safe via threading.Lock.

Tier 7 PR 5 — Control Plane: Session Registry.
"""

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = structlog.get_logger(__name__)

_REGISTRY_PATH = Path(
    os.environ.get(
        "BUDDY_SESSION_REGISTRY_PATH",
        str(Path(__file__).resolve().parents[3] / "trained_data" / "session_registry.json"),
    )
)

_MAX_PROCESSED_EVENT_IDS = 50

_VALID_TRANSPORT_STATES = frozenset({
    "connecting",
    "connected",
    "degraded",
    "reconnecting",
    "closed",
})


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _empty_session() -> Dict[str, Any]:
    """Return a blank session dict with all fields initialised."""
    return {
        "session_id": "",
        "started_at": "",
        "transport_state": "connecting",
        "last_heartbeat": "",
        "reconnect_attempts": 0,
        "degraded_mode": False,
        "degraded_reason": "",
        "degraded_since": "",
        "queue_depth_summary": {
            "depth": 0,
            "in_flight": 0,
            "failed_count": 0,
        },
        "outstanding_jobs": [],
        "last_processed_event_ids": [],
    }


class SessionRegistry:
    """Thread-safe registry that tracks the current Buddy session.

    Persists to ``trained_data/session_registry.json`` using atomic
    JSON writes (safe_json_write) so operators can inspect state at
    any time, even across process restarts.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path: Path = path or _REGISTRY_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = _empty_session()
        self.load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Start a new session, replacing any previous one.

        Returns:
            The new session_id (UUID4 hex string).
        """
        with self._lock:
            now = _now_iso()
            sid = uuid.uuid4().hex
            self._data = _empty_session()
            self._data["session_id"] = sid
            self._data["started_at"] = now
            self._data["last_heartbeat"] = now
            self._data["transport_state"] = "connecting"
            logger.info("session_registry.created", session_id=sid)
            self._save_locked()
            return sid

    # ------------------------------------------------------------------
    # Transport state
    # ------------------------------------------------------------------

    def update_transport_state(self, state: str) -> None:
        """Transition the transport to *state*.

        Args:
            state: One of ``connecting``, ``connected``, ``degraded``,
                   ``reconnecting``, ``closed``.

        Raises:
            ValueError: If *state* is not a recognised transport state.
        """
        if state not in _VALID_TRANSPORT_STATES:
            raise ValueError(
                f"Invalid transport state '{state}'. "
                f"Must be one of {sorted(_VALID_TRANSPORT_STATES)}"
            )
        with self._lock:
            prev = self._data.get("transport_state", "unknown")
            self._data["transport_state"] = state
            logger.info(
                "session_registry.transport_state",
                prev=prev,
                new=state,
                session_id=self._data.get("session_id", ""),
            )
            self._save_locked()

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def record_heartbeat(self) -> None:
        """Stamp the current UTC time as the last heartbeat."""
        with self._lock:
            self._data["last_heartbeat"] = _now_iso()
            self._save_locked()

    # ------------------------------------------------------------------
    # Reconnect tracking
    # ------------------------------------------------------------------

    def record_reconnect_attempt(self) -> None:
        """Increment the reconnect attempt counter."""
        with self._lock:
            self._data["reconnect_attempts"] = (
                self._data.get("reconnect_attempts", 0) + 1
            )
            logger.info(
                "session_registry.reconnect_attempt",
                attempts=self._data["reconnect_attempts"],
                session_id=self._data.get("session_id", ""),
            )
            self._save_locked()

    # ------------------------------------------------------------------
    # Degraded mode
    # ------------------------------------------------------------------

    def enter_degraded(self, reason: str) -> None:
        """Mark the session as degraded with a human-readable reason.

        Args:
            reason: Why the session entered degraded mode
                    (e.g. ``"broker API timeout"``).
        """
        with self._lock:
            self._data["degraded_mode"] = True
            self._data["degraded_reason"] = reason
            self._data["degraded_since"] = _now_iso()
            logger.warning(
                "session_registry.degraded",
                reason=reason,
                session_id=self._data.get("session_id", ""),
            )
            self._save_locked()

    def exit_degraded(self) -> None:
        """Clear the degraded flag and associated metadata."""
        with self._lock:
            was_degraded = self._data.get("degraded_mode", False)
            self._data["degraded_mode"] = False
            self._data["degraded_reason"] = ""
            self._data["degraded_since"] = ""
            if was_degraded:
                logger.info(
                    "session_registry.degraded_cleared",
                    session_id=self._data.get("session_id", ""),
                )
            self._save_locked()

    # ------------------------------------------------------------------
    # Queue summary
    # ------------------------------------------------------------------

    def update_queue_summary(
        self,
        depth: int,
        in_flight: int,
        failed_count: int,
    ) -> None:
        """Snapshot the current event-queue health.

        Args:
            depth: Total items waiting in the queue.
            in_flight: Items currently being processed.
            failed_count: Cumulative items that failed processing.
        """
        with self._lock:
            self._data["queue_depth_summary"] = {
                "depth": depth,
                "in_flight": in_flight,
                "failed_count": failed_count,
            }
            self._save_locked()

    # ------------------------------------------------------------------
    # Outstanding jobs
    # ------------------------------------------------------------------

    def set_outstanding_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """Replace the outstanding-jobs list wholesale.

        Args:
            jobs: List of job dicts (caller defines schema).
        """
        with self._lock:
            self._data["outstanding_jobs"] = list(jobs)
            self._save_locked()

    # ------------------------------------------------------------------
    # Processed events
    # ------------------------------------------------------------------

    def record_event_processed(self, event_id: str) -> None:
        """Append *event_id* to the recent-events ring buffer (max 50).

        Args:
            event_id: Unique identifier of the processed event.
        """
        with self._lock:
            ids: List[str] = self._data.get("last_processed_event_ids", [])
            ids.append(event_id)
            if len(ids) > _MAX_PROCESSED_EVENT_IDS:
                ids = ids[-_MAX_PROCESSED_EVENT_IDS:]
            self._data["last_processed_event_ids"] = ids
            self._save_locked()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the current session state to disk (thread-safe)."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Internal save — caller MUST hold ``self._lock``."""
        ok = safe_json_write(self._path, self._data, indent=2, create_backup=True)
        if not ok:
            logger.error(
                "session_registry.save_failed",
                path=str(self._path),
                session_id=self._data.get("session_id", ""),
            )

    def load(self) -> None:
        """Load session state from disk, or reset to empty defaults."""
        with self._lock:
            loaded = safe_json_read(self._path, default=None)
            if loaded is not None and isinstance(loaded, dict):
                # Merge with defaults so new fields are always present
                merged = _empty_session()
                merged.update(loaded)
                self._data = merged
                logger.info(
                    "session_registry.loaded",
                    session_id=self._data.get("session_id", ""),
                    transport_state=self._data.get("transport_state", ""),
                )
            else:
                self._data = _empty_session()
                logger.info("session_registry.initialised_empty")

    # ------------------------------------------------------------------
    # Operator summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a snapshot dict suitable for operator display.

        The returned dict is a shallow copy; mutations do not affect
        internal state.
        """
        with self._lock:
            summary = dict(self._data)
            # Add computed convenience fields
            if summary.get("last_heartbeat"):
                try:
                    hb = datetime.fromisoformat(summary["last_heartbeat"])
                    age_s = (datetime.now(timezone.utc) - hb).total_seconds()
                    summary["heartbeat_age_seconds"] = round(age_s, 1)
                except (ValueError, TypeError):
                    summary["heartbeat_age_seconds"] = -1
            else:
                summary["heartbeat_age_seconds"] = -1
            return summary
