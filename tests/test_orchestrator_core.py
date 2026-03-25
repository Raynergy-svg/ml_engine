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
