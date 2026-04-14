"""Unified visibility for Buddy background work.

Tracks long-running or off-thread/off-process work in a small JSON registry so
CLI status, dashboard views, and operators can see what Buddy is doing.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("trained_data/background_activity.json")
_MAX_ITEMS = 200
_MAX_EVENTS = 25
_STATUSES_ACTIVE = {"queued", "running", "waiting", "starting"}
_TRACKERS: Dict[str, "BackgroundActivityTracker"] = {}
_TRACKERS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundActivityTracker:
    """Thread-safe registry for inspectable background work."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path or _DEFAULT_PATH)
        self._lock = threading.RLock()

    def start_activity(
        self,
        *,
        kind: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "running",
        activity_id: Optional[str] = None,
    ) -> str:
        with self._lock:
            state = self._load_state()
            now = _now_iso()
            act_id = activity_id or f"bg_{uuid.uuid4().hex[:12]}"
            existing = self._find(state, act_id)
            if existing is None:
                existing = {
                    "id": act_id,
                    "kind": kind,
                    "title": title,
                    "source": source,
                    "status": status,
                    "started_at": now,
                    "updated_at": now,
                    "ended_at": None,
                    "summary": "",
                    "error": None,
                    "metadata": {},
                    "events": [],
                }
                state["activities"].append(existing)
            else:
                existing["kind"] = kind or existing.get("kind", "")
                existing["title"] = title or existing.get("title", "")
                existing["source"] = source or existing.get("source", "")
                existing["status"] = status or existing.get("status", "running")
                existing["updated_at"] = now
                existing["ended_at"] = None if existing["status"] in _STATUSES_ACTIVE else existing.get("ended_at")
            if metadata:
                existing["metadata"] = self._merge_dicts(existing.get("metadata") or {}, metadata)
            self._persist(state)
            return act_id

    def append_event(
        self,
        activity_id: str,
        message: str,
        *,
        level: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            state = self._load_state()
            activity = self._find(state, activity_id)
            if activity is None:
                return False
            activity.setdefault("events", []).append({
                "timestamp": _now_iso(),
                "level": level,
                "message": message,
                "details": deepcopy(details) if isinstance(details, dict) else {},
            })
            activity["events"] = activity["events"][-_MAX_EVENTS:]
            activity["updated_at"] = _now_iso()
            self._persist(state)
            return True

    def update_activity(
        self,
        activity_id: str,
        *,
        status: Optional[str] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            state = self._load_state()
            activity = self._find(state, activity_id)
            if activity is None:
                return False
            now = _now_iso()
            if status:
                activity["status"] = status
                if status not in _STATUSES_ACTIVE:
                    activity["ended_at"] = now
            if summary is not None:
                activity["summary"] = summary
            if error is not None:
                activity["error"] = error
            if metadata:
                activity["metadata"] = self._merge_dicts(activity.get("metadata") or {}, metadata)
            activity["updated_at"] = now
            self._persist(state)
            return True

    def complete_activity(
        self,
        activity_id: str,
        *,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.update_activity(
            activity_id,
            status="completed",
            summary=summary,
            error=None,
            metadata=metadata,
        )

    def fail_activity(
        self,
        activity_id: str,
        *,
        error: str,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.update_activity(
            activity_id,
            status="failed",
            summary=summary,
            error=error,
            metadata=metadata,
        )

    def get_snapshot(self, limit: int = 25) -> Dict[str, Any]:
        with self._lock:
            state = self._load_state()
            changed = self._refresh_process_states(state)
            activities = list(reversed(state.get("activities", [])))
            active = [a for a in activities if str(a.get("status", "")).lower() in _STATUSES_ACTIVE]
            history = [a for a in activities if str(a.get("status", "")).lower() not in _STATUSES_ACTIVE][:limit]
            snapshot = {
                "updated_at": state.get("updated_at"),
                "active_count": len(active),
                "history_count": len(history),
                "active": deepcopy(active[:limit]),
                "history": deepcopy(history),
            }
            if changed:
                self._persist(state)
            return snapshot

    def _load_state(self) -> Dict[str, Any]:
        state = safe_json_read(self._path, default={"version": 1, "updated_at": _now_iso(), "activities": []})
        if not isinstance(state, dict):
            return {"version": 1, "updated_at": _now_iso(), "activities": []}
        state.setdefault("version", 1)
        state.setdefault("updated_at", _now_iso())
        activities = state.get("activities")
        if not isinstance(activities, list):
            state["activities"] = []
        return state

    def _persist(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _now_iso()
        activities = state.get("activities", [])
        activities.sort(key=lambda item: str(item.get("started_at", "")))
        state["activities"] = activities[-_MAX_ITEMS:]
        safe_json_write(self._path, state)

    @staticmethod
    def _find(state: Dict[str, Any], activity_id: str) -> Optional[Dict[str, Any]]:
        for activity in state.get("activities", []):
            if activity.get("id") == activity_id:
                return activity
        return None

    @staticmethod
    def _merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def _refresh_process_states(self, state: Dict[str, Any]) -> bool:
        changed = False
        for activity in state.get("activities", []):
            if str(activity.get("status", "")).lower() not in _STATUSES_ACTIVE:
                continue
            metadata = activity.get("metadata") or {}
            pid = metadata.get("pid")
            if not pid:
                continue
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            if self._pid_running(pid_int):
                continue
            activity["status"] = "completed"
            activity["summary"] = activity.get("summary") or "Process exited"
            activity["ended_at"] = _now_iso()
            activity["updated_at"] = _now_iso()
            activity.setdefault("events", []).append({
                "timestamp": _now_iso(),
                "level": "info",
                "message": "Tracked process exited",
                "details": {"pid": pid_int},
            })
            activity["events"] = activity["events"][-_MAX_EVENTS:]
            changed = True
        return changed

    @staticmethod
    def _pid_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def get_background_activity_tracker(path: Optional[Path] = None) -> BackgroundActivityTracker:
    key = str(Path(path or _DEFAULT_PATH).resolve())
    with _TRACKERS_LOCK:
        tracker = _TRACKERS.get(key)
        if tracker is None:
            tracker = BackgroundActivityTracker(path=Path(path or _DEFAULT_PATH))
            _TRACKERS[key] = tracker
        return tracker
