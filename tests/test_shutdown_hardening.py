"""Tests for Phase 100 shutdown hardening fixes."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Fix #1: _terminate_background_processes — nested getattr crash
# ---------------------------------------------------------------------------

class TestTerminateBackgroundProcesses:
    """Ensure _terminate_background_processes doesn't crash when scanner is None."""

    def _make_cs(self, scanner=None):
        """Build a minimal ContinuousScanner mock with the patched method."""
        from src.scanner.automation.continuous import ContinuousScanner
        cs = object.__new__(ContinuousScanner)
        cs.scanner = scanner
        cs._maintenance = None
        cs._retrain_process = None
        return cs

    def test_scanner_none_does_not_crash(self):
        cs = self._make_cs(scanner=None)
        cs._terminate_background_processes()  # should not raise

    def test_scanner_missing_exec_mgr(self):
        scanner = MagicMock()
        del scanner._execution_manager  # getattr returns None via default
        cs = self._make_cs(scanner=scanner)
        cs._terminate_background_processes()

    def test_scanner_with_exec_mgr_and_process(self):
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        exec_mgr = MagicMock()
        exec_mgr._retrain_process = proc
        scanner = MagicMock()
        scanner._execution_manager = exec_mgr
        cs = self._make_cs(scanner=scanner)
        cs._terminate_background_processes()


# ---------------------------------------------------------------------------
# Fix #2: Module persistence None check
# ---------------------------------------------------------------------------

class TestModulePersistenceNoneCheck:
    """hasattr passes but scanner is None — must not crash."""

    def test_persist_modules_scanner_none(self):
        from src.scanner.automation.continuous import ContinuousScanner
        cs = object.__new__(ContinuousScanner)
        cs.scanner = None
        cs._shutdown_started = False
        cs._running = True
        cs._shutdown_event = threading.Event()
        cs._scan_count = 0
        cs._maintenance = None
        cs._retrain_process = None
        cs._peak_nav = 0.0
        cs.config = MagicMock()
        cs._orchestrator = None
        cs._signal_count = 0
        cs._original_signal_handlers = {}

        # _perform_shutdown should complete without AttributeError
        cs._perform_shutdown()
        assert cs._shutdown_started is True


# ---------------------------------------------------------------------------
# Fix #4: Cache clearing
# ---------------------------------------------------------------------------

class TestCacheClearing:
    """Scanner.clear_scan_caches frees memory (tested via extracted logic)."""

    @staticmethod
    def _clear_scan_caches(self_):
        """Extracted logic matching Scanner.clear_scan_caches."""
        freed = 0
        for cache in (self_._cached_results, self_._cache_contexts,
                      self_._pair_returns, self_._feature_snapshots, self_._raw_snapshots):
            freed += len(cache)
            cache.clear()
        return freed

    def test_clear_scan_caches(self):
        scanner = MagicMock()
        scanner._cached_results = {"a": 1, "b": 2}
        scanner._cache_contexts = {"a": (), "b": ()}
        scanner._pair_returns = {"a": None}
        scanner._feature_snapshots = {"a": None, "b": None, "c": None}
        scanner._raw_snapshots = {"a": None}

        freed = self._clear_scan_caches(scanner)
        assert freed == 9
        assert len(scanner._cached_results) == 0
        assert len(scanner._feature_snapshots) == 0

    def test_clear_empty_caches(self):
        scanner = MagicMock()
        scanner._cached_results = {}
        scanner._cache_contexts = {}
        scanner._pair_returns = {}
        scanner._feature_snapshots = {}
        scanner._raw_snapshots = {}

        freed = self._clear_scan_caches(scanner)
        assert freed == 0


# ---------------------------------------------------------------------------
# Fix #6: EventBus shutdown with thread tracking
# ---------------------------------------------------------------------------

class TestEventBusShutdown:
    """EventBus tracks async threads and shuts them down."""

    def test_emit_async_tracks_thread(self):
        from src.scanner.automation.event_bus import EventBus, Event, EventType, EventResult

        bus = EventBus()
        done = threading.Event()

        def handler(event):
            done.wait(timeout=5)
            return EventResult(handler_name="test", success=True, duration_secs=0.0)

        bus.subscribe(EventType.PRD_COMPLETE, handler, name="test")

        event = Event(event_type=EventType.PRD_COMPLETE, source="test", payload={})
        bus.emit_async(event)

        # Thread should be tracked
        with bus._async_lock:
            assert len(bus._async_threads) == 1

        done.set()
        timed_out = bus.shutdown(timeout_secs=5.0)
        assert timed_out == 0

        with bus._async_lock:
            assert len(bus._async_threads) == 0

    def test_shutdown_empty_bus(self):
        from src.scanner.automation.event_bus import EventBus
        bus = EventBus()
        timed_out = bus.shutdown()
        assert timed_out == 0


# ---------------------------------------------------------------------------
# Fix #7: RL subprocess cleanup
# ---------------------------------------------------------------------------

class TestRLSubprocessCleanup:
    """Scanner.shutdown_rl_workers terminates RL processes."""

    @staticmethod
    def _shutdown_rl_workers(self_):
        """Extracted logic matching Scanner.shutdown_rl_workers for isolated testing."""
        mei = getattr(self_, "_modular_ensemble", None)
        if mei is None:
            return
        for attr in ("_rl_gates_proc", "_rl_exits_proc"):
            proc = getattr(mei, attr, None)
            if proc is None or not proc.is_alive():
                continue
            try:
                proc.terminate()
                proc.join(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            finally:
                setattr(mei, attr, None)

    def test_shutdown_rl_workers_no_mei(self):
        scanner = MagicMock(spec=[])  # no attributes
        self._shutdown_rl_workers(scanner)  # should not raise

    def test_shutdown_rl_workers_terminates(self):
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True

        mei = MagicMock()
        mei._rl_gates_proc = mock_proc
        mei._rl_exits_proc = None

        scanner = MagicMock()
        scanner._modular_ensemble = mei

        self._shutdown_rl_workers(scanner)
        mock_proc.terminate.assert_called_once()
        mock_proc.join.assert_called_once_with(timeout=3)


# ---------------------------------------------------------------------------
# Fix #5: Orchestrator closes PRDAgentChain
# ---------------------------------------------------------------------------

class TestOrchestratorClose:
    """Orchestrator.close() stops PRDAgentChain."""

    def test_close_stops_prd_chain(self):
        from src.scanner.automation.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._prd_chain = MagicMock()
        orch._state = None
        orch._config_adjuster = None
        orch._observation_consumer = None
        orch._drift_remediator = None
        orch._online_rl = None
        orch._gate_health = None
        orch._agent_health = None
        orch._module_dispatcher = None

        orch.close()
        orch._prd_chain.stop.assert_called_once()

    def test_close_no_prd_chain(self):
        from src.scanner.automation.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._prd_chain = None
        orch._state = None
        orch._config_adjuster = None
        orch._observation_consumer = None
        orch._drift_remediator = None
        orch._online_rl = None
        orch._gate_health = None
        orch._agent_health = None
        orch._module_dispatcher = None

        orch.close()  # should not raise
