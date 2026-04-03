"""Tests for EventQueue: ordering, retry, shutdown, and status reporting."""

import queue
import threading
import time

import pytest

from src.scanner.automation.event_queue import EventQueue


class TestFIFOOrdering:
    """Verify handlers enqueued at the same priority execute in FIFO order."""

    def test_fifo_ordering_three_handlers(self):
        """Three handlers enqueued execute in enqueue order."""
        results = []
        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq.enqueue("first", lambda e: results.append(1), "evt1")
        eq.enqueue("second", lambda e: results.append(2), "evt2")
        eq.enqueue("third", lambda e: results.append(3), "evt3")
        eq.start()
        eq.flush()
        eq.stop()
        assert results == [1, 2, 3]

    def test_priority_ordering(self):
        """Lower priority number executes first regardless of enqueue order."""
        results = []
        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        # Enqueue high-priority-number first
        eq.enqueue("low_pri", lambda e: results.append("low"), "evt", priority=5)
        eq.enqueue("high_pri", lambda e: results.append("high"), "evt", priority=1)
        eq.start()
        eq.flush()
        eq.stop()
        assert results == ["high", "low"]


class TestSerialExecution:
    """Verify only one handler runs at a time."""

    def test_serial_execution_threading_event(self):
        """Use threading.Event to prove no two handlers overlap."""
        running = threading.Event()
        overlap_detected = threading.Event()
        results = []

        def handler(event):
            if running.is_set():
                overlap_detected.set()  # Another handler is already running!
            running.set()
            time.sleep(0.05)
            results.append(event)
            running.clear()

        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq.enqueue("h1", handler, "a")
        eq.enqueue("h2", handler, "b")
        eq.enqueue("h3", handler, "c")
        eq.start()
        eq.flush()
        eq.stop()

        assert not overlap_detected.is_set(), "Handlers ran concurrently!"
        assert len(results) == 3


class TestBoundedCapacity:
    """Verify queue respects max_size."""

    def test_bounded_capacity_blocks_or_raises(self):
        """Enqueue on a full queue blocks (put with timeout would raise Full)."""
        eq = EventQueue(max_size=2, max_retries=1, base_delay=0, max_delay=0)
        # Don't start consumer — items stay in queue
        eq.enqueue("h1", lambda e: None, "evt1")
        eq.enqueue("h2", lambda e: None, "evt2")

        # Queue is full (maxsize=2). A non-blocking put should raise queue.Full.
        with pytest.raises(queue.Full):
            eq._queue.put_nowait(None)


class TestRetryOnFailure:
    """Verify retry mechanics."""

    def test_handler_fails_once_then_succeeds(self):
        """Handler that fails on first call succeeds on retry — called twice."""
        call_count = {"n": 0}

        def flaky_handler(event):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("transient failure")

        eq = EventQueue(max_size=10, max_retries=3, base_delay=0, max_delay=0)
        eq.start()
        eq.enqueue("flaky", flaky_handler, "evt")
        eq.flush()
        eq.stop()

        assert call_count["n"] == 2  # failed once, succeeded on retry

    def test_max_retries_exhaustion(self):
        """Handler that always fails increments failed_count; consumer continues."""
        results = []

        def always_fails(event):
            raise RuntimeError("permanent failure")

        def after_failure(event):
            results.append("ok")

        eq = EventQueue(max_size=10, max_retries=2, base_delay=0, max_delay=0)
        eq.start()
        eq.enqueue("bad", always_fails, "evt1")
        eq.enqueue("good", after_failure, "evt2")
        eq.flush()
        eq.stop()

        status = eq.get_status()
        assert status["failed_count"] >= 1
        assert results == ["ok"], "Consumer must continue after permanent failure"


class TestFlush:
    """Verify flush blocks until queue is empty."""

    def test_flush_blocks_until_empty(self):
        """flush() returns only after all enqueued items have been processed."""
        results = []

        def slow_handler(event):
            time.sleep(0.05)
            results.append(event)

        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq.start()
        for i in range(5):
            eq.enqueue(f"h{i}", slow_handler, i)
        eq.flush()  # Should block until all 5 are done
        assert len(results) == 5
        eq.stop()


class TestStop:
    """Verify stop behavior with and without flush_first."""

    def test_stop_flush_first_true_drains_remaining(self):
        """stop(flush_first=True) lets consumer finish; flush() ensures drain."""
        results = []

        def handler(event):
            results.append(event)

        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq.start()
        for i in range(3):
            eq.enqueue(f"h{i}", handler, i)
        eq.flush()  # Block until all items processed
        eq.stop(flush_first=True)  # Graceful shutdown — doesn't clear queue
        assert results == [0, 1, 2]

    def test_stop_flush_first_false_drops_remaining(self):
        """stop(flush_first=False) drops unprocessed items."""
        results = []
        gate = threading.Event()

        def blocking_handler(event):
            gate.wait(timeout=5)  # Block until gate is opened
            results.append(event)

        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        # Fill queue without starting consumer — items stay queued
        for i in range(5):
            eq.enqueue(f"h{i}", blocking_handler, i)

        eq.start()
        time.sleep(0.1)  # Let consumer pick up at most the first item
        eq.stop(flush_first=False)
        gate.set()  # Release any in-flight handler

        # Most items should have been dropped
        assert len(results) < 5


class TestHandlerExceptionIsolation:
    """Failing handler doesn't block the next handler."""

    def test_exception_isolation(self):
        """A handler that raises doesn't prevent subsequent handlers from running."""
        results = []

        def bad_handler(event):
            raise RuntimeError("boom")

        def good_handler(event):
            results.append(event)

        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq.start()
        eq.enqueue("bad", bad_handler, "fail")
        eq.enqueue("good", good_handler, "success")
        eq.flush()
        eq.stop()

        assert results == ["success"]


class TestGetStatus:
    """Verify get_status returns accurate depth and failed_count."""

    def test_get_status_depth_and_failed(self):
        """get_status reflects queue depth and failed_count accurately."""
        eq = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)

        # Before start — enqueued items show in depth
        eq.enqueue("h1", lambda e: None, "evt1")
        eq.enqueue("h2", lambda e: None, "evt2")
        status = eq.get_status()
        assert status["depth"] == 2
        assert status["failed_count"] == 0

        # After processing a failing handler
        def always_fails(event):
            raise RuntimeError("fail")

        eq2 = EventQueue(max_size=10, max_retries=1, base_delay=0, max_delay=0)
        eq2.start()
        eq2.enqueue("bad", always_fails, "evt")
        eq2.flush()
        eq2.stop()

        status2 = eq2.get_status()
        assert status2["depth"] == 0
        assert status2["failed_count"] == 1
        assert status2["in_flight"] == 0
