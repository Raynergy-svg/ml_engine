"""
Event Bus — two buses in one module:

1. PrdEventBus (sync callback-based) — PRD lifecycle pipeline events.
   Legacy callers (gap_wirer, code_reviewer, prd_watcher, prd_agent_chain) use this.

2. EventBus (asyncio.Queue-based) — trading signals and control events (US-501).
   Thread-safe publish, async iterator subscribe, 100-event replay buffer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Optional,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRD Lifecycle Bus  (renamed from EventBus → PrdEventBus for US-501)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


class PrdEventBus:
    """
    Thread-safe pub/sub event bus with ordered handler execution (PRD lifecycle).

    Handlers for the same event type run in registration order (priority).
    Each handler receives the Event and can return an EventResult.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._handlers: Dict[EventType, List[tuple]] = defaultdict(list)
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
        logger.info(f"PrdEventBus: subscribed '{handler_name}' to {event_type.value} (priority={priority})")

    def unsubscribe(self, event_type: EventType, name: str) -> bool:
        """Remove a named handler."""
        with self._lock:
            before = len(self._handlers[event_type])
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h[1] != name
            ]
            removed = before - len(self._handlers[event_type])
        if removed:
            logger.info(f"PrdEventBus: unsubscribed '{name}' from {event_type.value}")
        return removed > 0

    def emit(self, event: Event) -> List[EventResult]:
        """Emit an event and run all subscribed handlers synchronously."""
        results = []
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))

        logger.info(f"PrdEventBus: emitting {event.event_type.value} → {len(handlers)} handler(s)")

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
        """Emit an event on a background thread."""
        def _run():
            try:
                results = self.emit(event)
                if callback:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"PrdEventBus: async callback error: {e}")
            finally:
                with self._async_lock:
                    self._async_threads.discard(thread)

        thread = threading.Thread(target=_run, name=f"event-{event.event_type.value}", daemon=True)
        with self._async_lock:
            self._async_threads.add(thread)
        thread.start()
        return thread

    def shutdown(self, timeout_secs: float = 5.0) -> int:
        """Wait for active async threads. Returns count of threads that timed out."""
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
        """Count registered handlers."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values())

    def _persist_log(self, entry: dict) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"PrdEventBus: failed to persist log: {e}")


# PRD bus singleton
_prd_global_bus: Optional[PrdEventBus] = None
_prd_bus_lock = threading.Lock()


def get_prd_event_bus(log_path: Optional[Path] = None) -> PrdEventBus:
    """Get or create the global PrdEventBus singleton."""
    global _prd_global_bus
    with _prd_bus_lock:
        if _prd_global_bus is None:
            _prd_global_bus = PrdEventBus(log_path=log_path)
        return _prd_global_bus


def reset_prd_event_bus() -> None:
    """Reset the global PRD bus (for testing)."""
    global _prd_global_bus
    with _prd_bus_lock:
        _prd_global_bus = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trading Event Bus  (US-501) — asyncio.Queue-based pub/sub
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#: All valid event type strings for the trading bus.
TRADING_EVENT_TYPES = frozenset({
    "signal.pending",
    "signal.executed",
    "signal.rejected",
    "signal.vetoed",
    "adjustment.proposed",
    "adjustment.applied",
    "adjustment.rejected",
    "control.kill",
    "control.pause",
    "control.resume",
    "control.mode_change",
    "control.signal_veto",
    "control.reconciler_offline",
    "trade.manual_close",
})

_REPLAY_SIZE = 100


class EventBus:
    """
    asyncio.Queue-based pub/sub bus for trading signals and control events.

    Thread-safe publish side: any thread can call publish().
    Async subscriber side: subscribe() is an async generator consumed in an
    event loop (e.g. the Textual TUI loop). The bridge is asyncio's
    call_soon_threadsafe which schedules queue.put_nowait on the subscriber's loop.

    Late subscribers receive up to REPLAY_BUFFER_SIZE buffered events before
    live ones, so TUI panels that connect after startup see recent history.
    """

    REPLAY_BUFFER_SIZE: int = _REPLAY_SIZE

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each subscriber has its own (loop, queue) pair
        self._subscribers: List[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Dict[str, Any]]]] = []
        self._replay: deque[Dict[str, Any]] = deque(maxlen=_REPLAY_SIZE)
        # US-511: veto window registry — signal_id → threading.Event
        self._veto_events: Dict[str, threading.Event] = {}

    def register_veto_wait(self, signal_id: str) -> threading.Event:
        """US-511: Register a threading.Event for veto_window synchronization.

        The event is set when control.signal_veto with matching signal_id is published.
        Must be called before publishing signal.pending to avoid a race.
        """
        evt = threading.Event()
        with self._lock:
            self._veto_events[signal_id] = evt
        return evt

    def cancel_veto_wait(self, signal_id: str) -> None:
        """US-511: Remove a veto waiter after the window expires or veto fires."""
        with self._lock:
            self._veto_events.pop(signal_id, None)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Thread-safe publish. Enqueues to all active subscriber asyncio queues.

        Safely handles subscribers whose event loops have been closed.
        """
        event: Dict[str, Any] = {
            "event_type": event_type,
            "payload": payload,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._replay.append(event)
            subs = list(self._subscribers)
            # US-511: if this is a veto control event, fire the waiting thread event
            veto_evt: Optional[threading.Event] = None
            if event_type == "control.signal_veto":
                sig_id = payload.get("signal_id", "")
                veto_evt = self._veto_events.pop(sig_id, None)

        dead: List[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Dict[str, Any]]]] = []
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                # Loop is closed — mark for removal
                dead.append((loop, q))

        if dead:
            with self._lock:
                for item in dead:
                    try:
                        self._subscribers.remove(item)
                    except ValueError:
                        pass

        if veto_evt is not None:
            veto_evt.set()

    async def subscribe(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator yielding events. Late subscribers first get the replay buffer."""
        loop = asyncio.get_event_loop()
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        with self._lock:
            replay_snapshot = list(self._replay)
            self._subscribers.append((loop, q))

        try:
            for event in replay_snapshot:
                yield event
            while True:
                event = await q.get()
                yield event
        finally:
            with self._lock:
                self._subscribers = [
                    (l, sq) for l, sq in self._subscribers if sq is not q
                ]

    def subscriber_count(self) -> int:
        """Return the number of active async subscribers."""
        with self._lock:
            return len(self._subscribers)

    def replay_buffer(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the current replay buffer (most-recent last)."""
        with self._lock:
            return list(self._replay)


# Trading EventBus singleton
_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get or create the global trading EventBus singleton."""
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = EventBus()
        return _global_bus


def reset_event_bus() -> None:
    """Reset the global trading bus (for testing)."""
    global _global_bus
    with _bus_lock:
        _global_bus = None
