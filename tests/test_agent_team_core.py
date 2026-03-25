"""Phase 36 (US-230): Unit tests for ScannerAgentTeam core methods.

Tests agent weight management, vote aggregation, regime-aware weights,
decay mechanics, and data structure validation.
Uses mock-based approach to avoid OANDA dependencies.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the utilities and classes we can test directly
from src.scanner.agents._team import (
    AgentVerdict,
    ScannerAgentTeam,
    _clip01,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Helper: create a minimal config mock
# ---------------------------------------------------------------------------
def _mock_config(**overrides):
    """Create a minimal config object for ScannerAgentTeam."""
    cfg = MagicMock()
    cfg.regime_disabled_agents = {}
    cfg.enable_trend_agent = True
    cfg.enable_mean_reversion_agent = True
    cfg.enable_volatility_agent = True
    cfg.enable_risk_sentinel_agent = True
    cfg.enable_uncertainty_agent = True
    cfg.enable_execution_quality_agent = True
    cfg.enable_news_risk_agent = False
    cfg.enable_multi_timeframe_agent = False
    cfg.enable_pair_performance_agent = False
    cfg.enable_momentum_agent = True
    cfg.enable_session_timing_agent = False
    cfg.enable_support_resistance_agent = False
    cfg.enable_trader_readiness_agent = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Tests: utility functions
# ---------------------------------------------------------------------------
class TestClip01:
    def test_normal_value(self):
        assert _clip01(0.5) == 0.5

    def test_below_zero(self):
        assert _clip01(-0.3) == 0.0

    def test_above_one(self):
        assert _clip01(1.7) == 1.0

    def test_boundary_zero(self):
        assert _clip01(0.0) == 0.0

    def test_boundary_one(self):
        assert _clip01(1.0) == 1.0


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_default(self):
        assert _safe_float(None, 1.0) == 1.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan"), 0.5) == 0.5

    def test_invalid_string_returns_default(self):
        assert _safe_float("not_a_number", 0.0) == 0.0

    def test_integer_input(self):
        assert _safe_float(42) == 42.0


# ---------------------------------------------------------------------------
# Tests: AgentVerdict dataclass
# ---------------------------------------------------------------------------
class TestAgentVerdict:
    def test_to_dict_has_all_fields(self):
        v = AgentVerdict(
            name="trend",
            score=0.75,
            passed=True,
            weight=1.15,
            reason="Strong uptrend",
            reason_code="TREND_UP",
            confidence_delta=0.02,
        )
        d = v.to_dict()
        assert d["name"] == "trend"
        assert d["score"] == 0.75
        assert d["passed"] is True
        assert d["weight"] == 1.15
        assert d["reason"] == "Strong uptrend"
        assert d["reason_code"] == "TREND_UP"

    def test_default_metadata_is_empty(self):
        v = AgentVerdict(
            name="risk", score=0.5, passed=True, weight=1.0,
            reason="OK", reason_code="OK",
        )
        assert v.metadata == {}
        assert v.block_trade is False
        assert v.confidence_delta == 0.0


# ---------------------------------------------------------------------------
# Tests: ScannerAgentTeam weight management
# ---------------------------------------------------------------------------
class TestAgentTeamWeights:
    """Test weight loading, saving, validation, and regime awareness."""

    def _create_team_with_weights(self, weights_data=None):
        """Create a ScannerAgentTeam with a temp weights file."""
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)
        weights_file = os.path.join(weights_path, "agent_weights.json")

        if weights_data is not None:
            with open(weights_file, "w") as f:
                json.dump(weights_data, f)

        # Patch the class-level path
        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE", weights_file):
            team = ScannerAgentTeam(_mock_config())
        return team, tmpdir

    def test_init_creates_regime_weights(self):
        """Without existing weights file, team initializes regime-aware structure."""
        team, _ = self._create_team_with_weights(None)
        assert "_global" in team._learned_weights
        assert "NORMAL" in team._learned_weights
        assert "HIGH" in team._learned_weights
        assert "EXTREME" in team._learned_weights

    def test_load_regime_aware_weights(self):
        """Loads regime-aware weights from disk correctly."""
        data = {
            "_global": {"trend": 1.20, "volatility": 0.95},
            "NORMAL": {"trend": 1.18},
            "HIGH": {},
            "EXTREME": {},
            "_meta": {"min_trades_per_regime": 10},
        }
        team, _ = self._create_team_with_weights(data)
        assert team._learned_weights["_global"]["trend"] == 1.20
        assert team._learned_weights["NORMAL"]["trend"] == 1.18

    def test_migrate_legacy_flat_weights(self):
        """Legacy flat dict is migrated to regime-aware format."""
        legacy = {"trend": 1.30, "volatility": 0.80}
        team, _ = self._create_team_with_weights(legacy)
        # Should have been migrated
        assert "_global" in team._learned_weights
        assert team._learned_weights["_global"]["trend"] == 1.30

    def test_corrupted_json_falls_back(self):
        """Corrupted weights file falls back to baseline."""
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)
        weights_file = os.path.join(weights_path, "agent_weights.json")

        with open(weights_file, "w") as f:
            f.write("{broken json!!!")

        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE", weights_file):
            team = ScannerAgentTeam(_mock_config())

        # Should have baseline regime structure
        assert "_global" in team._learned_weights

    def test_get_weights_for_regime_normal(self):
        """get_weights_for_regime returns weights dict for requested regime."""
        data = {
            "_global": {"trend": 1.25, "volatility": 0.95},
            "NORMAL": {"trend": 1.22},
            "HIGH": {},
            "EXTREME": {},
            "_meta": {"min_trades_per_regime": 10},
        }
        team, _ = self._create_team_with_weights(data)
        weights = team.get_weights_for_regime("NORMAL")
        # Should return a dict with at least the learned keys
        assert isinstance(weights, dict)
        assert "trend" in weights

    def test_base_weights_all_positive(self):
        """All base weights must be positive."""
        for name, w in ScannerAgentTeam._BASE_WEIGHTS.items():
            assert w > 0, f"Base weight for {name} is {w}, expected positive"

    def test_apply_weight_decay_moves_toward_base(self):
        """Weight decay should move learned weights toward base."""
        data = {
            "_global": {"trend": 1.50},  # Base is 1.15
            "NORMAL": {"trend": 1.50},
            "HIGH": {},
            "EXTREME": {},
            "_meta": {"min_trades_per_regime": 10},
        }
        team, tmpdir = self._create_team_with_weights(data)

        weights_file = os.path.join(tmpdir, "trained_data", "models", "agent_weights.json")
        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE", weights_file):
            result = team.apply_weight_decay(decay_rate=0.10)

        # trend should have moved from 1.50 toward 1.15
        if "trend" in team._learned_weights.get("_global", {}):
            assert team._learned_weights["_global"]["trend"] < 1.50


# ---------------------------------------------------------------------------
# Tests: Validate weights rejects bad data
# ---------------------------------------------------------------------------
class TestWeightValidation:
    def _make_team(self):
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)
        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE",
                          os.path.join(weights_path, "agent_weights.json")):
            return ScannerAgentTeam(_mock_config())

    def test_validate_rejects_nan(self):
        """NaN weight values should be rejected."""
        team = self._make_team()
        data = {
            "_global": {"trend": float("nan")},
            "NORMAL": {},
            "HIGH": {},
            "EXTREME": {},
            "_meta": {},
        }
        result = team._validate_weights(data)
        # Should either return cleaned data or None
        if result is not None:
            global_trend = result.get("_global", {}).get("trend")
            assert global_trend is None or not math.isnan(global_trend)

    def test_validate_rejects_inf(self):
        """Inf weight values should be rejected."""
        team = self._make_team()
        data = {
            "_global": {"trend": float("inf")},
            "NORMAL": {},
            "HIGH": {},
            "EXTREME": {},
            "_meta": {},
        }
        result = team._validate_weights(data)
        if result is not None:
            global_trend = result.get("_global", {}).get("trend")
            assert global_trend is None or not math.isinf(global_trend)
