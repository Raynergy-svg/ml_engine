"""Bounded serial event queue with tenacity-based retry for post-trade fan-out."""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import structlog
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = structlog.get_logger(__name__)


@dataclass(order=True)
class _WorkItem:
    """Priority-ordered work item. Lower priority number = higher priority."""

    priority: int
    seq: int = field(compare=True)  # tie-breaker for FIFO within same priority
    handler_name: str = field(compare=False)
    handler_fn: Callable = field(compare=False)
    event: Any = field(compare=False)
    event_id: str = field(compare=False, default_factory=lambda: uuid.uuid4().hex[:12])


class EventQueue:
    """Bounded, serial event queue that drains handlers one at a time with retry.

    Designed for post-trade fan-out: enqueue handlers that must run reliably
    without blocking the scanner loop.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self._max_size = max_size
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self._seq = 0
        self._seq_lock = threading.Lock()

        self._shutdown = threading.Event()
        self._flush_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Stats
        self._in_flight = 0
        self._failed_count = 0
        self._retry_count = 0
        self._last_failure: str | None = None
        self._stats_lock = threading.Lock()

    def enqueue(
        self,
        handler_name: str,
        handler_fn: Callable,
        event: Any,
        priority: int = 0,
    ) -> None:
        """Add a work item to the queue.

        Args:
            handler_name: Human-readable name for logging.
            handler_fn: Callable to invoke with (event,).
            event: The event payload passed to handler_fn.
            priority: Lower number = higher priority. Default 0.
        """
        with self._seq_lock:
            seq = self._seq
            self._seq += 1

        item = _WorkItem(
            priority=priority,
            seq=seq,
            handler_name=handler_name,
            handler_fn=handler_fn,
            event=event,
        )
        self._queue.put(item)

    def start(self) -> None:
        """Launch the consumer daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self._thread.start()

    def stop(self, flush_first: bool = True) -> None:
        """Signal shutdown.

        Args:
            flush_first: If True, drain remaining items before stopping.
                         If False, drop them immediately.
        """
        if not flush_first:
            # Clear the queue
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break

        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
            self._thread = None

    def flush(self) -> None:
        """Block calling thread until the queue is empty."""
        self._queue.join()

    def get_status(self) -> Dict[str, Any]:
        """Return queue status snapshot."""
        with self._stats_lock:
            return {
                "depth": self._queue.qsize(),
                "in_flight": self._in_flight,
                "failed_count": self._failed_count,
                "retry_count": self._retry_count,
                "last_failure": self._last_failure,
            }

    def _consumer_loop(self) -> None:
        """Main consumer loop — runs in daemon thread."""
        while not self._shutdown.is_set():
            try:
                item: _WorkItem = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._execute_with_retry(item)
            finally:
                self._queue.task_done()

    def _execute_with_retry(self, item: _WorkItem) -> None:
        """Execute a handler with tenacity retry."""
        log = logger.bind(handler_name=item.handler_name, event_id=item.event_id)

        with self._stats_lock:
            self._in_flight = 1

        # Build a retrying wrapper with the instance's retry config
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._base_delay, max=self._max_delay)
            + wait_random(0, 1),
            reraise=True,
        )
        def _run(attempt_number_holder: list) -> None:
            attempt = attempt_number_holder[0]
            attempt_number_holder[0] += 1

            if attempt > 1:
                with self._stats_lock:
                    self._retry_count += 1

            log.info(
                "handler_executing",
                attempt_number=attempt,
            )
            item.handler_fn(item.event)

        try:
            _run([1])
            log.info("handler_succeeded")
        except RetryError as exc:
            with self._stats_lock:
                self._failed_count += 1
                self._last_failure = (
                    f"{item.handler_name}: {exc.last_attempt.exception()}"
                )
            log.error(
                "handler_failed_permanently",
                error=str(exc.last_attempt.exception()),
                max_retries=self._max_retries,
            )
        except Exception as exc:
            with self._stats_lock:
                self._failed_count += 1
                self._last_failure = f"{item.handler_name}: {exc}"
            log.error(
                "handler_failed_unexpectedly",
                error=str(exc),
            )
        finally:
            with self._stats_lock:
                self._in_flight = 0


class FlushGate:
    """Transport-safe buffer gate for broker reconnection scenarios.

    Buffers operations while active so no writes are lost during transport
    transitions. Items are drained in FIFO order on end().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._buffer: List[Any] = []

    @property
    def is_active(self) -> bool:
        """Return current gate state."""
        with self._lock:
            return self._active

    def start(self) -> None:
        """Activate the gate — subsequent enqueue() calls will buffer items."""
        with self._lock:
            self._active = True

    def enqueue(self, item: Any) -> bool:
        """Buffer an item if the gate is active.

        Returns:
            True if the item was buffered (gate active).
            False if the gate is not active (caller should send directly).
        """
        with self._lock:
            if not self._active:
                return False
            self._buffer.append(item)
            return True

    def end(self) -> List[Any]:
        """Deactivate the gate and return all buffered items in FIFO order."""
        with self._lock:
            self._active = False
            items = list(self._buffer)
            self._buffer.clear()
            return items

    def drop(self) -> None:
        """Discard all buffered items and deactivate the gate."""
        with self._lock:
            self._active = False
            self._buffer.clear()
