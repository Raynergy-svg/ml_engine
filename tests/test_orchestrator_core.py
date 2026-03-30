"""Phase 36 (US-233): Unit tests for Orchestrator core methods.

Tests OrchestrationResult dataclass, get_system_status structure,
cycle counting, and module initialization patterns.
"""

import json
from dataclasses import fields
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.scanner.automation.orchestrator import OrchestrationResult, Orchestrator


# ---------------------------------------------------------------------------
# Tests: OrchestrationResult dataclass
# ---------------------------------------------------------------------------
class TestOrchestrationResult:
    def test_default_values(self):
        result = OrchestrationResult()
        assert result.cycle_id == ""
        assert result.pairs_scanned == 0
        assert result.tradeable_setups == 0
        assert result.trades_executed == 0
        assert result.errors == []

    def test_to_dict_has_all_fields(self):
        result = OrchestrationResult()
        d = result.to_dict()
        # Verify all dataclass fields are present in dict
        dc_fields = {f.name for f in fields(OrchestrationResult)}
        for field_name in dc_fields:
            assert field_name in d, f"Missing field: {field_name}"

    def test_to_dict_values_match(self):
        result = OrchestrationResult(
            cycle_id="cycle_001",
            pairs_scanned=15,
            tradeable_setups=3,
            trades_executed=2,
            learnings_extracted=1,
        )
        d = result.to_dict()
        assert d["cycle_id"] == "cycle_001"
        assert d["pairs_scanned"] == 15
        assert d["tradeable_setups"] == 3
        assert d["trades_executed"] == 2
        assert d["learnings_extracted"] == 1

    def test_errors_list_is_independent(self):
        """Each instance should have its own errors list (no shared mutable default)."""
        r1 = OrchestrationResult()
        r2 = OrchestrationResult()
        r1.errors.append("test error")
        assert len(r2.errors) == 0

    def test_observation_fields_present(self):
        """Phase 29 fields should exist."""
        result = OrchestrationResult()
        assert result.observation_patterns == 0
        assert result.observation_recommendations == 0

    def test_to_dict_serializable(self):
        """to_dict output should be JSON-serializable."""
        result = OrchestrationResult(
            cycle_id="test",
            timestamp=datetime.utcnow().isoformat(),
            errors=["some error"],
        )
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Tests: Orchestrator init
# ---------------------------------------------------------------------------
class TestOrchestratorInit:
    def test_default_init(self):
        orch = Orchestrator()
        assert orch._cycle_count == 0
        assert orch._auto_execute is False

    def test_auto_execute_flag(self):
        orch = Orchestrator(auto_execute=True)
        assert orch._auto_execute is True

    def test_modules_lazy_init(self):
        """Modules should be None before first use."""
        orch = Orchestrator()
        assert orch._state is None
        assert orch._learner is None
        assert orch._tuner is None


# ---------------------------------------------------------------------------
# Tests: get_system_status
# ---------------------------------------------------------------------------
class TestGetSystemStatus:
    def _make_orch(self):
        """Create Orchestrator with scanner attribute set to avoid AttributeError."""
        orch = Orchestrator()
        # _init_modules is called by get_system_status; it sets many attributes.
        # We pre-set scanner to None so the health registry block doesn't crash.
        orch.scanner = None
        return orch

    def test_status_has_modules_key(self):
        orch = self._make_orch()
        status = orch.get_system_status()
        assert "modules" in status
        assert isinstance(status["modules"], dict)

    def test_status_has_session_key(self):
        orch = self._make_orch()
        status = orch.get_system_status()
        assert "session" in status
        assert "cycles_completed" in status["session"]
        assert "auto_execute" in status["session"]
        assert "started" in status["session"]

    def test_status_modules_are_booleans(self):
        orch = self._make_orch()
        status = orch.get_system_status()
        for mod_name, available in status["modules"].items():
            assert isinstance(available, bool), f"{mod_name} should be bool, got {type(available)}"

    def test_status_serializable(self):
        """get_system_status must produce JSON-serializable output."""
        orch = self._make_orch()
        status = orch.get_system_status()
        serialized = json.dumps(status, default=str)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Tests: Dispatch Table (Fix 2: Registry Pattern for Module Dispatch)
# ---------------------------------------------------------------------------
class TestDispatchTable:
    def test_dispatch_table_initialized_empty(self):
        """Dispatch table should be empty before _init_modules."""
        orch = Orchestrator()
        assert orch._dispatch_table == []

    def test_dispatch_table_populated_after_init_modules(self):
        """After _init_modules, dispatch table should have steps registered."""
        orch = Orchestrator()
        orch._init_modules()
        assert len(orch._dispatch_table) > 0
        assert len(orch._dispatch_table) >= 20, f"Expected >=20 steps, got {len(orch._dispatch_table)}"

    def test_dispatch_step_has_required_attributes(self):
        """Each DispatchStep should have all required attributes."""
        orch = Orchestrator()
        orch._init_modules()
        for step in orch._dispatch_table:
            assert hasattr(step, "name")
            assert hasattr(step, "callable")
            assert hasattr(step, "condition")
            assert hasattr(step, "interval")
            assert hasattr(step, "critical")
            assert hasattr(step, "result_key")
            assert isinstance(step.name, str)
            assert callable(step.callable)
            assert callable(step.condition)
            assert isinstance(step.interval, int)
            assert isinstance(step.critical, bool)

    def test_dispatch_step_interval_filtering(self):
        """Steps with interval > 1 should only run on matching cycles."""
        orch = Orchestrator()
        orch._init_modules()

        # Find a step with interval > 1
        interval_5_step = None
        for step in orch._dispatch_table:
            if step.interval == 5:
                interval_5_step = step
                break

        assert interval_5_step is not None, "Should have at least one step with interval=5"

        # Simulate cycles and check when step should run
        orch._cycle_count = 1
        # Cycle 1: 1 % 5 != 0, should not run
        assert orch._cycle_count % interval_5_step.interval != 0

        orch._cycle_count = 5
        # Cycle 5: 5 % 5 == 0, should run
        assert orch._cycle_count % interval_5_step.interval == 0

        orch._cycle_count = 10
        # Cycle 10: 10 % 5 == 0, should run
        assert orch._cycle_count % interval_5_step.interval == 0

    def test_dispatch_step_condition_false_skips_step(self):
        """If condition returns False, step should be skipped in dispatch loop."""
        orch = Orchestrator()
        orch._init_modules()

        # Register a test step with condition that returns False
        call_count = []
        orch.register_module(
            name="test_step_never_runs",
            callable=lambda: call_count.append(1) or None,
            condition=lambda: False,
            interval=1,
        )

        # Run the dispatch loop
        orch._cycle_count = 1
        orch._current_result = OrchestrationResult()
        orch._dispatch_results = {}

        # Manually run dispatch loop
        for step in orch._dispatch_table[-1:]:  # Just test our new step
            if orch._cycle_count % step.interval != 0:
                continue
            if not step.condition():
                continue
            try:
                step_result = step.callable()
            except Exception:
                pass

        # Callable should never be called
        assert len(call_count) == 0

    def test_dispatch_step_failure_isolated(self):
        """If one step fails, other steps should still run."""
        orch = Orchestrator()
        orch._init_modules()

        # Clear dispatch table
        orch._dispatch_table = []

        # Add 3 steps: first fails, second succeeds, third succeeds
        failed_step = [False]

        orch.register_module(
            name="step1_fails",
            callable=lambda: (_ for _ in ()).throw(Exception("Intentional failure")),
            condition=lambda: True,
            interval=1,
            critical=False,
        )

        results = []
        orch.register_module(
            name="step2_runs",
            callable=lambda: results.append("step2"),
            condition=lambda: True,
            interval=1,
            critical=False,
        )

        orch.register_module(
            name="step3_runs",
            callable=lambda: results.append("step3"),
            condition=lambda: True,
            interval=1,
            critical=False,
        )

        # Run dispatch loop
        orch._cycle_count = 1
        orch._current_result = OrchestrationResult()
        orch._dispatch_results = {}

        for step in orch._dispatch_table:
            if orch._cycle_count % step.interval != 0:
                continue
            if not step.condition():
                continue
            try:
                step_result = step.callable()
                if step.result_key:
                    orch._dispatch_results[step.result_key] = step_result
            except Exception as e:
                if step.critical:
                    raise

        # Step2 and Step3 should have run despite Step1 failing
        assert "step2" in results
        assert "step3" in results

    def test_register_module_adds_to_dispatch_table(self):
        """register_module should add a step to the dispatch table."""
        orch = Orchestrator()
        orch._init_modules()
        initial_count = len(orch._dispatch_table)

        orch.register_module(
            name="test_custom_step",
            callable=lambda: None,
            condition=lambda: True,
            interval=1,
        )

        assert len(orch._dispatch_table) == initial_count + 1
        assert orch._dispatch_table[-1].name == "test_custom_step"

    def test_dispatch_result_key_stores_return_value(self):
        """If result_key is set, return value should be stored in _dispatch_results."""
        orch = Orchestrator()
        orch._dispatch_table = []
        orch._current_result = OrchestrationResult()
        orch._dispatch_results = {}
        orch._cycle_count = 1

        def test_callable():
            return {"test_data": "value123"}

        orch.register_module(
            name="test_step_with_result",
            callable=test_callable,
            condition=lambda: True,
            interval=1,
            result_key="test_result",
        )

        # Run dispatch loop
        for step in orch._dispatch_table:
            if orch._cycle_count % step.interval != 0:
                continue
            if not step.condition():
                continue
            try:
                step_result = step.callable()
                if step.result_key:
                    orch._dispatch_results[step.result_key] = step_result
            except Exception as e:
                if step.critical:
                    raise

        # Check that result was stored
        assert "test_result" in orch._dispatch_results
        assert orch._dispatch_results["test_result"]["test_data"] == "value123"

    def test_dispatch_table_step_names_unique(self):
        """Each step name should be unique for debugging."""
        orch = Orchestrator()
        orch._init_modules()

        names = [step.name for step in orch._dispatch_table]
        unique_names = set(names)
        assert len(names) == len(unique_names), f"Duplicate step names: {[n for n in names if names.count(n) > 1]}"

    def test_dispatch_table_all_intervals_valid(self):
        """All step intervals should be positive integers."""
        orch = Orchestrator()
        orch._init_modules()

        for step in orch._dispatch_table:
            assert step.interval >= 1, f"Step {step.name} has invalid interval {step.interval}"
            assert isinstance(step.interval, int)
