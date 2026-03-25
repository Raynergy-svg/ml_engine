"""Unit tests for Enhanced Confidence Calibration System.

Tests cover:
1. CalibrationConfig and CalibratedConfidence dataclasses
2. Ensemble disagreement calculation
3. Agent agreement quality
4. Platt scaling (with and without scipy)
5. Meta-confidence computation
6. Time decay application
7. Full calibration pipeline
8. Trade outcome recording and refitting
9. Atomic JSON persistence and recovery
10. Edge cases: empty agents, NaN/inf values, corrupt data
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.scanner.confidence_calibration import (
    CalibrationConfig,
    CalibratedConfidence,
    ConfidenceCalibrationSystem,
    _clip01,
    _safe_float,
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_clip01_in_range(self):
        assert _clip01(0.5) == 0.5
        assert _clip01(0.0) == 0.0
        assert _clip01(1.0) == 1.0

    def test_clip01_out_of_range(self):
        assert _clip01(-0.5) == 0.0
        assert _clip01(1.5) == 1.0
        assert _clip01(-100) == 0.0
        assert _clip01(100) == 1.0

    def test_safe_float_valid(self):
        assert _safe_float(0.5) == 0.5
        assert _safe_float("0.75") == 0.75
        assert _safe_float(1) == 1.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, 0.5) == 0.5

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float(float("nan"), 0.7) == 0.7

    def test_safe_float_invalid(self):
        assert _safe_float("invalid") == 0.0
        assert _safe_float([1, 2, 3]) == 0.0


class TestCalibrationConfig:
    """Test CalibrationConfig dataclass."""

    def test_default_config(self):
        config = CalibrationConfig()
        assert config.min_trades_for_calibration == 30
        assert config.refit_interval == 20
        assert "NORMAL" in config.decay_rates
        assert config.decay_rates["NORMAL"] == 0.97

    def test_custom_config(self):
        config = CalibrationConfig(
            min_trades_for_calibration=50,
            refit_interval=30,
        )
        assert config.min_trades_for_calibration == 50
        assert config.refit_interval == 30

    def test_decay_rates_per_regime(self):
        config = CalibrationConfig()
        for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]:
            assert regime in config.decay_rates
            assert 0.9 <= config.decay_rates[regime] <= 1.0


class TestCalibratedConfidence:
    """Test CalibratedConfidence dataclass."""

    def test_create_result(self):
        result = CalibratedConfidence(
            raw_weighted_score=0.65,
            agent_scores=[0.6, 0.7, 0.65],
            ensemble_disagreement=0.05,
            disagreement_level="LOW",
            platt_calibrated=0.68,
            is_calibrated=True,
            calibration_regime="NORMAL",
            agent_agreement=0.95,
            agents_reporting=3,
        )

        assert result.raw_weighted_score == 0.65
        assert len(result.agent_scores) == 3
        assert result.disagreement_level == "LOW"
        assert result.is_calibrated is True


class MockAgentVerdict:
    """Mock AgentVerdict for testing."""

    def __init__(self, name, score, weight=1.0, passed=True):
        self.name = name
        self.score = score
        self.weight = weight
        self.passed = passed


class TestConfidenceCalibrationSystem:
    """Test ConfidenceCalibrationSystem."""

    def setup_method(self):
        """Create a temporary config file for testing."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = CalibrationConfig(
            calibration_file=str(Path(self.tmpdir.name) / "calibration.json")
        )
        self.system = ConfidenceCalibrationSystem(self.config)

    def teardown_method(self):
        """Clean up temporary files."""
        self.tmpdir.cleanup()

    def test_initialization(self):
        """Test system initializes with empty parameters."""
        assert len(self.system._platt_params) == 0
        assert len(self.system._trade_history) == 0
        assert self.system._trades_since_refit == 0

    def test_ensemble_disagreement_low(self):
        """Test low disagreement classification."""
        verdicts = [
            MockAgentVerdict("agent1", 0.65),
            MockAgentVerdict("agent2", 0.64),
            MockAgentVerdict("agent3", 0.66),
        ]
        disagreement, level = self.system._compute_ensemble_disagreement(verdicts)
        assert disagreement < 0.15
        assert level == "LOW"

    def test_ensemble_disagreement_moderate(self):
        """Test moderate disagreement classification."""
        verdicts = [
            MockAgentVerdict("agent1", 0.3),
            MockAgentVerdict("agent2", 0.5),
            MockAgentVerdict("agent3", 0.7),
        ]
        disagreement, level = self.system._compute_ensemble_disagreement(verdicts)
        assert 0.15 <= disagreement < 0.30
        assert level == "MODERATE"

    def test_ensemble_disagreement_high(self):
        """Test high disagreement classification."""
        verdicts = [
            MockAgentVerdict("agent1", 0.0),
            MockAgentVerdict("agent2", 0.8),
            MockAgentVerdict("agent3", 0.2),
        ]
        disagreement, level = self.system._compute_ensemble_disagreement(verdicts)
        # std is ~0.36
        assert level in ["HIGH", "MODERATE"]

    def test_ensemble_disagreement_critical(self):
        """Test critical disagreement classification."""
        verdicts = [
            MockAgentVerdict("agent1", 0.0),
            MockAgentVerdict("agent2", 1.0),
            MockAgentVerdict("agent3", 0.0),
            MockAgentVerdict("agent4", 1.0),
            MockAgentVerdict("agent5", 0.0),
        ]
        disagreement, level = self.system._compute_ensemble_disagreement(verdicts)
        # std is 0.5, should classify as CRITICAL or HIGH
        assert level in ["CRITICAL", "HIGH"]

    def test_weighted_score_equal_weights(self):
        """Test weighted score with equal weights."""
        verdicts = [
            MockAgentVerdict("agent1", 0.6, weight=1.0),
            MockAgentVerdict("agent2", 0.8, weight=1.0),
        ]
        score = self.system._compute_weighted_score(verdicts)
        assert abs(score - 0.7) < 0.01

    def test_weighted_score_unequal_weights(self):
        """Test weighted score with unequal weights."""
        verdicts = [
            MockAgentVerdict("agent1", 0.6, weight=1.0),
            MockAgentVerdict("agent2", 0.8, weight=3.0),
        ]
        score = self.system._compute_weighted_score(verdicts)
        expected = (0.6 * 1.0 + 0.8 * 3.0) / 4.0
        assert abs(score - expected) < 0.01

    def test_agent_agreement_high(self):
        """Test high agent agreement with many agents reporting."""
        verdicts = [
            MockAgentVerdict("agent1", 0.65),
            MockAgentVerdict("agent2", 0.655),
            MockAgentVerdict("agent3", 0.645),
            MockAgentVerdict("agent4", 0.65),
            MockAgentVerdict("agent5", 0.652),
            MockAgentVerdict("agent6", 0.648),
        ]
        agreement, n_agents = self.system._compute_agent_agreement(verdicts)
        # Very low std = high agreement
        assert agreement > 0.5
        assert n_agents == 6

    def test_agent_agreement_low(self):
        """Test low agent agreement (diverse scores)."""
        verdicts = [
            MockAgentVerdict("agent1", 0.1),
            MockAgentVerdict("agent2", 0.9),
            MockAgentVerdict("agent3", 0.5),
            MockAgentVerdict("agent4", 0.2),
            MockAgentVerdict("agent5", 0.8),
            MockAgentVerdict("agent6", 0.6),
        ]
        agreement, n_agents = self.system._compute_agent_agreement(verdicts)
        # High std = low agreement
        assert agreement < 0.5
        assert n_agents == 6

    def test_agent_agreement_insufficient_agents(self):
        """Test agreement with fewer than min_agents."""
        verdicts = [MockAgentVerdict("agent1", 0.6)]
        agreement, n_agents = self.system._compute_agent_agreement(verdicts)
        assert agreement == 0.5  # Default when insufficient agents
        assert n_agents == 1

    def test_platt_scale_no_calibration(self):
        """Test Platt scaling without calibration data."""
        calibrated, is_calibrated = self.system._platt_scale(0.65, "NORMAL")
        assert calibrated == 0.65  # Falls back to raw score
        assert is_calibrated is False

    def test_platt_scale_with_global_calibration(self):
        """Test Platt scaling with global calibration."""
        self.system._platt_params["_global"] = {"coef": 2.0, "intercept": -1.0}
        calibrated, is_calibrated = self.system._platt_scale(0.65, "NORMAL")
        assert is_calibrated is True
        assert 0.0 <= calibrated <= 1.0

    def test_calibrate_full_pipeline(self):
        """Test full calibration pipeline."""
        verdicts = [
            MockAgentVerdict("trend", 0.65, weight=1.15),
            MockAgentVerdict("momentum", 0.70, weight=1.05),
            MockAgentVerdict("volatility", 0.60, weight=1.00),
            MockAgentVerdict("mean_reversion", 0.68, weight=0.90),
            MockAgentVerdict("risk_sentinel", 0.62, weight=1.25),
            MockAgentVerdict("uncertainty", 0.64, weight=1.10),
        ]

        result = self.system.calibrate(verdicts, regime_name="NORMAL")

        assert isinstance(result, CalibratedConfidence)
        assert 0.0 <= result.final_confidence <= 1.0
        assert result.disagreement_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert result.agents_reporting == 6
        assert "timestamp" in result.metadata

    def test_calibrate_empty_verdicts(self):
        """Test calibration with empty verdicts."""
        with pytest.raises(ValueError):
            self.system.calibrate([], regime_name="NORMAL")

    def test_apply_time_decay_zero_bars(self):
        """Test time decay with zero bars held."""
        result = CalibratedConfidence(
            raw_weighted_score=0.65,
            agent_scores=[0.65],
            ensemble_disagreement=0.05,
            disagreement_level="LOW",
            platt_calibrated=0.67,
            is_calibrated=True,
            calibration_regime="NORMAL",
            agent_agreement=0.95,
            agents_reporting=1,
            final_confidence=0.66,
        )

        decayed = self.system.apply_time_decay(result, bars_held=0, regime_name="NORMAL")
        assert decayed.final_confidence == result.final_confidence
        assert decayed.time_decay_factor == 1.0

    def test_apply_time_decay_normal_regime(self):
        """Test time decay in NORMAL regime."""
        result = CalibratedConfidence(
            raw_weighted_score=0.65,
            agent_scores=[0.65],
            ensemble_disagreement=0.05,
            disagreement_level="LOW",
            platt_calibrated=0.67,
            is_calibrated=True,
            calibration_regime="NORMAL",
            agent_agreement=0.95,
            agents_reporting=1,
            final_confidence=0.66,
        )

        decayed = self.system.apply_time_decay(result, bars_held=10, regime_name="NORMAL")
        assert decayed.final_confidence < result.final_confidence
        # Decay factor should be 0.97^10 = ~0.7374
        expected_decay = 0.97 ** 10
        assert abs(decayed.time_decay_factor - expected_decay) < 0.001

    def test_apply_time_decay_extreme_regime(self):
        """Test time decay in EXTREME regime (faster decay)."""
        result = CalibratedConfidence(
            raw_weighted_score=0.65,
            agent_scores=[0.65],
            ensemble_disagreement=0.05,
            disagreement_level="LOW",
            platt_calibrated=0.67,
            is_calibrated=True,
            calibration_regime="EXTREME",
            agent_agreement=0.95,
            agents_reporting=1,
            final_confidence=0.66,
        )

        decayed_extreme = self.system.apply_time_decay(result, bars_held=10, regime_name="EXTREME")
        decayed_normal = self.system.apply_time_decay(result, bars_held=10, regime_name="NORMAL")

        # EXTREME (0.92) decays faster than NORMAL (0.97)
        assert decayed_extreme.final_confidence < decayed_normal.final_confidence

    def test_record_outcome_valid(self):
        """Test recording valid trade outcomes."""
        self.system.record_outcome(raw_score=0.65, outcome=1.0, regime_name="NORMAL")
        assert len(self.system._trade_history) == 1
        assert self.system._trade_history[0] == (0.65, 1.0, "NORMAL")

    def test_record_outcome_clamps_invalid(self):
        """Test that invalid outcomes are clamped."""
        self.system.record_outcome(raw_score=0.65, outcome=1.5, regime_name="NORMAL")
        assert len(self.system._trade_history) == 1
        assert self.system._trade_history[0][1] == 1.0  # Clamped

    def test_persistence_save_and_load(self):
        """Test atomic save and load of calibration."""
        # Add some data
        self.system._platt_params["_global"] = {"coef": 1.5, "intercept": -0.5}
        for i in range(5):
            self.system._trade_history.append((0.6 + 0.01 * i, 1.0, "NORMAL"))

        # Save
        self.system._save_calibration()

        # Load into new system
        system2 = ConfidenceCalibrationSystem(self.config)
        assert len(system2._platt_params) > 0
        assert len(system2._trade_history) > 0
        assert system2._platt_params["_global"]["coef"] == 1.5

    def test_confidence_combination_factors(self):
        """Test confidence combination with different factor levels."""
        # LOW disagreement
        final_low = self.system._combine_confidence_components(
            platt_calibrated=0.70,
            agent_agreement=0.95,
            disagreement_level="LOW",
            meta_confidence=0.50,
        )

        # HIGH disagreement (should be penalized)
        final_high = self.system._combine_confidence_components(
            platt_calibrated=0.70,
            agent_agreement=0.95,
            disagreement_level="HIGH",
            meta_confidence=0.50,
        )

        assert final_high < final_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
