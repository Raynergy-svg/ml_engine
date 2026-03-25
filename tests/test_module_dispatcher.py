"""Phase 36 (US-232): Unit tests for ModuleDispatcher.

Tests frequency scheduling, module registration, tick dispatch,
error handling, and status reporting.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.module_dispatcher import (
    DEFAULT_FREQUENCIES,
    ModuleDispatcher,
)


# ---------------------------------------------------------------------------
# Tests: DEFAULT_FREQUENCIES
# ---------------------------------------------------------------------------
class TestDefaultFrequencies:
    def test_all_frequencies_positive(self):
        for name, freq in DEFAULT_FREQUENCIES.items():
            assert freq >= 1, f"{name} has invalid frequency {freq}"

    def test_known_modules_present(self):
        assert "memory_manager" in DEFAULT_FREQUENCIES
        assert "health_registry" in DEFAULT_FREQUENCIES
        assert "replay_validator" in DEFAULT_FREQUENCIES
        assert "regime_tracker" in DEFAULT_FREQUENCIES

    def test_frequent_modules_run_every_cycle(self):
        assert DEFAULT_FREQUENCIES["memory_manager"] == 1
        assert DEFAULT_FREQUENCIES["health_registry"] == 1
        assert DEFAULT_FREQUENCIES["position_manager"] == 1


# ---------------------------------------------------------------------------
# Tests: Module registration
# ---------------------------------------------------------------------------
class TestModuleRegistration:
    def test_register_single_module(self):
        dispatcher = ModuleDispatcher()
        fn = MagicMock()
        dispatcher.register_module("test_mod", fn, run_every_n=5)
        assert "test_mod" in dispatcher._modules
        assert dispatcher._modules["test_mod"] == (fn, 5)

    def test_register_clamps_frequency_minimum(self):
        dispatcher = ModuleDispatcher()
        fn = MagicMock()
        dispatcher.register_module("test_mod", fn, run_every_n=0)
        _, freq = dispatcher._modules["test_mod"]
        assert freq >= 1

    def test_register_all_without_scanner(self):
        dispatcher = ModuleDispatcher(scanner=None)
        count = dispatcher.register_all_modules()
        assert count == 0

    def test_register_all_with_mock_scanner(self):
        scanner = MagicMock()
        # Set up mock modules with save_state
        mock_mm = MagicMock()
        mock_mm.save_state = MagicMock()
        scanner._memory_manager = mock_mm

        mock_hr = MagicMock()
        mock_hr.save_state = MagicMock()
        scanner._health_registry = mock_hr

        # Everything else returns None
        scanner._module_activation = None

        dispatcher = ModuleDispatcher(scanner=scanner)
        count = dispatcher.register_all_modules()
        assert count >= 2  # At least memory_manager and health_registry


# ---------------------------------------------------------------------------
# Tests: Cycle scheduling
# ---------------------------------------------------------------------------
class TestCycleScheduling:
    def test_get_runnable_cycle_1(self):
        """Cycle 1 should not trigger any freq>1 modules (since 1 % freq != 0 for freq > 1)."""
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("every_1", MagicMock(), 1)
        dispatcher.register_module("every_5", MagicMock(), 5)
        dispatcher.register_module("every_10", MagicMock(), 10)

        runnable = dispatcher.get_runnable_modules(1)
        assert "every_1" in runnable
        assert "every_5" not in runnable
        assert "every_10" not in runnable

    def test_get_runnable_cycle_10(self):
        """Cycle 10 should trigger freq 1, 5, and 10 modules."""
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("every_1", MagicMock(), 1)
        dispatcher.register_module("every_5", MagicMock(), 5)
        dispatcher.register_module("every_10", MagicMock(), 10)

        runnable = dispatcher.get_runnable_modules(10)
        assert "every_1" in runnable
        assert "every_5" in runnable
        assert "every_10" in runnable

    def test_get_runnable_cycle_0(self):
        """Cycle 0: all modules run (0 % n == 0 for all n)."""
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("mod_a", MagicMock(), 3)
        dispatcher.register_module("mod_b", MagicMock(), 7)

        runnable = dispatcher.get_runnable_modules(0)
        assert "mod_a" in runnable
        assert "mod_b" in runnable


# ---------------------------------------------------------------------------
# Tests: Execute cycle
# ---------------------------------------------------------------------------
class TestExecuteCycle:
    def test_execute_calls_run_fn(self):
        dispatcher = ModuleDispatcher()
        fn = MagicMock()
        dispatcher.register_module("test_mod", fn, 1)

        results = dispatcher.execute_cycle(1)
        fn.assert_called_once()
        assert results["test_mod"] is True

    def test_execute_handles_failure(self):
        dispatcher = ModuleDispatcher()
        fn = MagicMock(side_effect=RuntimeError("boom"))
        dispatcher.register_module("failing_mod", fn, 1)

        results = dispatcher.execute_cycle(1)
        assert results["failing_mod"] is False
        assert dispatcher._dispatch_errors == 1

    def test_execute_tracks_total_dispatches(self):
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("mod_a", MagicMock(), 1)
        dispatcher.register_module("mod_b", MagicMock(), 1)

        dispatcher.execute_cycle(1)
        assert dispatcher._total_dispatches == 2

    def test_execute_logs_at_cycle_10(self):
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("mod_a", MagicMock(), 1)

        dispatcher.execute_cycle(10)
        assert len(dispatcher._dispatch_log) == 1
        assert dispatcher._dispatch_log[0]["cycle"] == 10


# ---------------------------------------------------------------------------
# Tests: Status reporting
# ---------------------------------------------------------------------------
class TestStatusReporting:
    def test_get_status_structure(self):
        dispatcher = ModuleDispatcher()
        dispatcher.register_module("mod_a", MagicMock(), 3)

        status = dispatcher.get_status()
        assert "registered_modules" in status
        assert "total_dispatches" in status
        assert "dispatch_errors" in status
        assert "module_frequencies" in status
        assert status["registered_modules"] == 1
        assert status["module_frequencies"]["mod_a"] == 3


# ---------------------------------------------------------------------------
# Tests: State persistence
# ---------------------------------------------------------------------------
class TestStatePersistence:
    def test_save_state_writes_json(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "dispatcher_state.json"
        dispatcher = ModuleDispatcher(persistence_path=path)
        dispatcher._total_dispatches = 42
        dispatcher._cycle_count = 100

        dispatcher.save_state()

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_dispatches"] == 42
        assert data["cycle_count"] == 100
        assert data["version"] == 1
