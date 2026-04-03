"""Trading Event Bus — blinker-based pub/sub with priority ordering and idempotency.

Thread-safe event dispatch for post-trade fan-out. Handlers are called in
priority order (lower number = higher priority). Duplicate events are skipped
via an LRU cache of processed event IDs. All events are persisted to JSONL.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from blinker import Signal
from cachetools import LRUCache

from src.scanner.automation.safe_json import safe_jsonl_append

_HISTORY_PATH = Path("trained_data/trading_events.jsonl")

logger = structlog.get_logger(__name__)


class _Handler:
    """Wrapper storing handler metadata for priority sorting."""

    __slots__ = ("fn", "priority", "name")

    def __init__(self, fn: Callable, priority: int, name: str):
        self.fn = fn
        self.priority = priority
        self.name = name


class TradingEventBus:
    """Priority-ordered, idempotent, thread-safe trading event bus."""

    def __init__(self) -> None:
        self._signals: Dict[str, Signal] = {}
        self._handlers: Dict[str, List[_Handler]] = {}  # event_type -> sorted handlers
        self._handler_names: Dict[str, _Handler] = {}   # handler_name -> _Handler
        self._seen = LRUCache(maxsize=10000)
        self._lock = threading.Lock()

    # ── registration ────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler_fn: Callable,
        priority: int = 100,
        handler_name: Optional[str] = None,
    ) -> None:
        """Register *handler_fn* for *event_type*.

        Args:
            event_type: Event name to listen for.
            handler_fn: Callback receiving the event dict.
            priority: Lower number = called first.
            handler_name: Unique name for this handler (defaults to fn.__name__).
        """
        name = handler_name or handler_fn.__name__
        handler = _Handler(handler_fn, priority, name)

        with self._lock:
            if event_type not in self._signals:
                self._signals[event_type] = Signal(event_type)
            self._handlers.setdefault(event_type, []).append(handler)
            self._handlers[event_type].sort(key=lambda h: h.priority)
            self._handler_names[name] = handler

    def unsubscribe(self, handler_name: str) -> None:
        """Remove handler by name from all event types."""
        with self._lock:
            handler = self._handler_names.pop(handler_name, None)
            if handler is None:
                return
            for et, handlers in self._handlers.items():
                self._handlers[et] = [h for h in handlers if h.name != handler_name]

    # ── dispatch ────────────────────────────────────────────────────

    def emit(self, event: Dict[str, Any]) -> None:
        """Dispatch *event* to all subscribers of its type, in priority order.

        Skips events whose ``event_id`` has already been processed (idempotency).
        Persists every dispatched event to the JSONL history file.
        """
        event_id = event.get("event_id")
        event_type = event.get("event_type", "unknown")
        log = logger.bind(event_type=event_type, event_id=event_id)

        # Idempotency check
        if event_id is not None and event_id in self._seen:
            log.debug("trading_event_bus.duplicate_skipped")
            return
        if event_id is not None:
            self._seen[event_id] = True

        log.info("trading_event_bus.emit")

        # Persist to JSONL
        record = {**event, "dispatched_at": datetime.now(timezone.utc).isoformat()}
        safe_jsonl_append(_HISTORY_PATH, record)

        # Dispatch in priority order
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))

        for handler in handlers:
            try:
                handler.fn(event)
            except Exception:
                log.exception(
                    "trading_event_bus.handler_error",
                    handler_name=handler.name,
                )

        # Also fire blinker signal for any direct signal subscribers
        sig = self._signals.get(event_type)
        if sig is not None:
            sig.send(event)

    def emit_to_queue(self, event: Dict[str, Any]) -> None:
        """Enqueue *event* for async dispatch (currently delegates to emit)."""
        self.emit(event)

    # ── history ─────────────────────────────────────────────────────

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent events from the JSONL history file.

        Args:
            event_type: Filter to this type, or None for all.
            limit: Max records to return (most recent first).
        """
        import json

        if not _HISTORY_PATH.exists():
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event_type is None or rec.get("event_type") == event_type:
                        records.append(rec)
        except Exception:
            logger.exception("trading_event_bus.history_read_error")
            return []

        return records[-limit:]


# ── singleton ───────────────────────────────────────────────────────

_bus_instance: Optional[TradingEventBus] = None
_bus_lock = threading.Lock()


def get_trading_event_bus() -> TradingEventBus:
    """Return the module-level singleton TradingEventBus."""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = TradingEventBus()
    return _bus_instance


def reset_trading_event_bus() -> None:
    """Reset the singleton for testing."""
    global _bus_instance
    with _bus_lock:
        _bus_instance = None
