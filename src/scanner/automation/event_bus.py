"""
Event Bus — Lightweight pub/sub system for PRD lifecycle events.

Events flow:
  prd_complete → gap_wirer → gaps_wired → code_reviewer → review_complete

Phase 28 (US-169): Event-driven PRD completion → agent chain automation.
"""

import logging
import threading
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types in the PRD lifecycle pipeline."""
    PRD_COMPLETE = "prd_complete"
    GAPS_IDENTIFIED = "gaps_identified"
    GAPS_WIRED = "gaps_wired"
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETE = "review_complete"
    CHAIN_COMPLETE = "chain_complete"
    CHAIN_ERROR = "chain_error"


@dataclass
class Event:
    """Immutable event payload."""
    event_type: EventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class EventResult:
    """Result from an event handler."""
    success: bool
    handler_name: str
    duration_secs: float = 0.0
    output: Any = None
    error: Optional[str] = None


class EventBus:
    """
    Thread-safe pub/sub event bus with ordered handler execution.

    Handlers for the same event type run in registration order (priority).
    Each handler receives the Event and can return an EventResult.
    Results are collected and passed to downstream events via payload enrichment.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._handlers: Dict[EventType, List[tuple]] = defaultdict(list)  # {type: [(priority, name, fn)]}
        self._lock = threading.Lock()
        self._event_log: List[dict] = []
        self._log_path = log_path or Path(".claude/ralph/event_log.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._async_threads: set = set()
        self._async_lock = threading.Lock()

    def subscribe(self, event_type: EventType, handler: Callable[[Event], EventResult],
                  name: str = "", priority: int = 100) -> None:
        """Register a handler for an event type. Lower priority = runs first."""
        handler_name = name or getattr(handler, "__name__", str(handler))
        with self._lock:
            self._handlers[event_type].append((priority, handler_name, handler))
            self._handlers[event_type].sort(key=lambda x: x[0])
        logger.info(f"EventBus: subscribed '{handler_name}' to {event_type.value} (priority={priority})")

    def unsubscribe(self, event_type: EventType, name: str) -> bool:
        """Remove a named handler."""
        with self._lock:
            before = len(self._handlers[event_type])
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h[1] != name
            ]
            removed = before - len(self._handlers[event_type])
        if removed:
            logger.info(f"EventBus: unsubscribed '{name}' from {event_type.value}")
        return removed > 0

    def emit(self, event: Event) -> List[EventResult]:
        """
        Emit an event and run all subscribed handlers synchronously in priority order.
        Returns list of EventResults from each handler.
        """
        results = []
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))

        logger.info(f"EventBus: emitting {event.event_type.value} → {len(handlers)} handler(s)")

        for priority, name, handler in handlers:
            start = time.monotonic()
            try:
                result = handler(event)
                if result is None:
                    result = EventResult(success=True, handler_name=name)
                result.duration_secs = time.monotonic() - start
                results.append(result)
                logger.info(f"  ✓ {name} ({result.duration_secs:.2f}s)")
            except Exception as e:
                duration = time.monotonic() - start
                result = EventResult(
                    success=False, handler_name=name,
                    duration_secs=duration, error=str(e)
                )
                results.append(result)
                logger.error(f"  ✗ {name} failed: {e}")

        # Log the event + results
        log_entry = {
            "event": event.to_dict(),
            "results": [
                {"handler": r.handler_name, "success": r.success,
                 "duration": round(r.duration_secs, 3), "error": r.error}
                for r in results
            ]
        }
        self._event_log.append(log_entry)
        self._persist_log(log_entry)

        return results

    def emit_async(self, event: Event, callback: Optional[Callable[[List[EventResult]], None]] = None) -> threading.Thread:
        """Emit an event on a background thread. Optional callback with results."""
        def _run():
            try:
                results = self.emit(event)
                if callback:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"EventBus: async callback error: {e}")
            finally:
                with self._async_lock:
                    self._async_threads.discard(thread)

        thread = threading.Thread(target=_run, name=f"event-{event.event_type.value}", daemon=True)
        with self._async_lock:
            self._async_threads.add(thread)
        thread.start()
        return thread

    def shutdown(self, timeout_secs: float = 5.0) -> int:
        """Wait for active async threads to finish. Returns count of threads that timed out."""
        with self._async_lock:
            threads = list(self._async_threads)
        timed_out = 0
        for t in threads:
            t.join(timeout=timeout_secs)
            if t.is_alive():
                timed_out += 1
        with self._async_lock:
            self._async_threads.clear()
        return timed_out

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 50) -> List[dict]:
        """Get recent event history, optionally filtered by type."""
        history = self._event_log
        if event_type:
            history = [e for e in history if e["event"]["event_type"] == event_type.value]
        return history[-limit:]

    def handler_count(self, event_type: Optional[EventType] = None) -> int:
        """Count registered handlers, optionally for a specific event type."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values())

    def _persist_log(self, entry: dict) -> None:
        """Append event to JSONL log file."""
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"EventBus: failed to persist log: {e}")


# Singleton instance for the PRD lifecycle pipeline
_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus(log_path: Optional[Path] = None) -> EventBus:
    """Get or create the global EventBus singleton."""
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = EventBus(log_path=log_path)
        return _global_bus


def reset_event_bus() -> None:
    """Reset the global bus (for testing)."""
    global _global_bus
    with _bus_lock:
        _global_bus = None
