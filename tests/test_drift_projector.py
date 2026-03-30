"""Unit tests for DriftProjector and SelfModelState.

Tests cover all major code paths and edge cases:
1. Empty drift history → conservative projection
2. With history → valid curve generation
3. Velocity estimation → correct rate calculation
4. Hours until critical → accurate threshold detection
5. Recommended action mapping → correct action/urgency assignment
6. All-pairs projection → batch processing
7. Projection persistence → save and load
8. DriftMonitor integration → projection included in report
9. SelfModelState → summary generation
10. Risk factor detection → correct classification
"""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scanner.automation.drift_projector import (
    DriftProjector,
    DriftProjectionCurve,
    DriftProjectionPoint,
)
from src.scanner.automation.self_model_state import SelfModelState


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "trained_data").mkdir(parents=True)
        yield tmppath


@pytest.fixture
def projector(temp_dirs):
    """Create a DriftProjector with temp paths."""
    return DriftProjector(
        drift_log_path=temp_dirs / "trained_data" / "drift_monitor_log.json",
        projection_cache_path=temp_dirs / "trained_data" / "drift_projections.json",
        feature_baseline_path=temp_dirs / "trained_data" / "feature_baseline.json",
        regime_path=temp_dirs / "trained_data" / "regime_transitions.json",
    )


@pytest.fixture
def self_model_state(temp_dirs):
    """Create a SelfModelState with temp path."""
    return SelfModelState(
        projection_cache_path=temp_dirs / "trained_data" / "drift_projections.json"
    )


class TestDriftProjectionPoint:
    """Test DriftProjectionPoint dataclass."""

    def test_creation(self):
        """Test point creation with valid data."""
        point = DriftProjectionPoint(
            hours_ahead=6,
            predicted_accuracy=0.52,
            confidence_interval_low=0.47,
            confidence_interval_high=0.57,
            degradation_driver="historical_pattern",
        )
        assert point.hours_ahead == 6
        assert point.predicted_accuracy == 0.52

    def test_to_dict(self):
        """Test serialization to dict."""
        point = DriftProjectionPoint(
            hours_ahead=12,
            predicted_accuracy=0.50,
            confidence_interval_low=0.45,
            confidence_interval_high=0.55,
            degradation_driver="regime_shift",
        )
        d = point.to_dict()
        assert d["hours_ahead"] == 12
        assert d["degradation_driver"] == "regime_shift"


class TestDriftProjectionCurve:
    """Test DriftProjectionCurve dataclass."""

    def test_creation(self):
        """Test curve creation."""
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
        )
        assert curve.pair == "EUR_USD"
        assert curve.current_accuracy == 0.55

    def test_hours_until_critical_not_reached(self):
        """Test when accuracy never reaches critical."""
        points = [
            DriftProjectionPoint(6, 0.54, 0.49, 0.59, "stable"),
            DriftProjectionPoint(12, 0.53, 0.48, 0.58, "stable"),
            DriftProjectionPoint(24, 0.52, 0.47, 0.57, "stable"),
            DriftProjectionPoint(48, 0.51, 0.46, 0.56, "stable"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        assert curve.hours_until_critical() is None

    def test_hours_until_critical_reached_at_24h(self):
        """Test when critical is reached at 24h."""
        points = [
            DriftProjectionPoint(6, 0.54, 0.49, 0.59, "stable"),
            DriftProjectionPoint(12, 0.52, 0.47, 0.57, "stable"),
            DriftProjectionPoint(24, 0.47, 0.42, 0.52, "degrading"),
            DriftProjectionPoint(48, 0.45, 0.40, 0.50, "degrading"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        assert curve.hours_until_critical() == 24

    def test_hours_until_critical_reached_soon(self):
        """Test when critical is reached within 6h."""
        points = [
            DriftProjectionPoint(6, 0.47, 0.42, 0.52, "degrading"),
            DriftProjectionPoint(12, 0.45, 0.40, 0.50, "degrading"),
            DriftProjectionPoint(24, 0.40, 0.35, 0.45, "critical"),
            DriftProjectionPoint(48, 0.35, 0.30, 0.40, "critical"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        assert curve.hours_until_critical() == 6

    def test_to_dict(self):
        """Test curve serialization."""
        points = [
            DriftProjectionPoint(6, 0.54, 0.49, 0.59, "stable"),
        ]
        curve = DriftProjectionCurve(
            pair="GBP_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
            recommended_action="monitor",
            action_urgency="low",
        )
        d = curve.to_dict()
        assert d["pair"] == "GBP_USD"
        assert d["current_accuracy"] == 0.55
        assert len(d["projection_points"]) == 1


class TestDriftProjector:
    """Test DriftProjector core functionality."""

    def test_project_empty_history(self, projector):
        """Test projection with no drift history → returns conservative projection."""
        curve = projector.project(
            pair="EUR_USD",
            current_accuracy=0.55,
            baseline_accuracy=0.55,
        )
        assert curve.pair == "EUR_USD"
        assert curve.current_accuracy == 0.55
        assert curve.recommended_action == "monitor"
        assert curve.action_urgency == "low"
        assert len(curve.projection_points) == 4  # t+6h, t+12h, t+24h, t+48h

    def test_project_with_history(self, projector, temp_dirs):
        """Test projection with drift history → returns valid curve."""
        # Create mock drift history
        now = datetime.now(timezone.utc)
        history = {
            "drift_history": {
                "EUR_USD": [
                    {
                        "timestamp": (now - timedelta(hours=24)).isoformat(),
                        "drift_score": -0.05,
                        "current_accuracy": 0.54,
                        "severity": "none",
                    },
                    {
                        "timestamp": (now - timedelta(hours=20)).isoformat(),
                        "drift_score": -0.08,
                        "current_accuracy": 0.52,
                        "severity": "mild",
                    },
                    {
                        "timestamp": (now - timedelta(hours=16)).isoformat(),
                        "drift_score": -0.10,
                        "current_accuracy": 0.50,
                        "severity": "mild",
                    },
                    {
                        "timestamp": (now - timedelta(hours=12)).isoformat(),
                        "drift_score": -0.12,
                        "current_accuracy": 0.48,
                        "severity": "mild",
                    },
                    {
                        "timestamp": (now - timedelta(hours=8)).isoformat(),
                        "drift_score": -0.15,
                        "current_accuracy": 0.46,
                        "severity": "critical",
                    },
                    {
                        "timestamp": now.isoformat(),
                        "drift_score": -0.18,
                        "current_accuracy": 0.44,
                        "severity": "critical",
                    },
                ]
            }
        }
        drift_log = temp_dirs / "trained_data" / "drift_monitor_log.json"
        drift_log.write_text(json.dumps(history, indent=2))

        curve = projector.project(
            pair="EUR_USD",
            current_accuracy=0.44,
            baseline_accuracy=0.55,
        )

        assert curve.pair == "EUR_USD"
        assert curve.current_accuracy == 0.44
        assert len(curve.projection_points) == 4
        # With strong negative velocity, should eventually hit critical
        assert curve.hours_until_critical() is not None or curve.recommended_action != "monitor"

    def test_velocity_estimation_empty_history(self, projector):
        """Test velocity with empty history → returns 0."""
        velocity = projector._estimate_velocity([], 0.55)
        assert velocity == 0.0

    def test_velocity_estimation_insufficient_history(self, projector):
        """Test velocity with too few records → returns 0."""
        history = [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "current_accuracy": 0.55}
        ]
        velocity = projector._estimate_velocity(history, 0.55)
        assert velocity == 0.0

    def test_velocity_estimation_declining_accuracy(self, projector):
        """Test velocity estimation with declining accuracy."""
        now = datetime.now(timezone.utc)
        history = [
            {
                "timestamp": (now - timedelta(hours=5 - i)).isoformat(),
                "current_accuracy": 0.55 - (i * 0.02),
            }
            for i in range(0, 5)
        ]
        velocity = projector._estimate_velocity(history, 0.45)
        # Should be negative (declining)
        assert velocity < 0

    def test_velocity_estimation_improving_accuracy(self, projector):
        """Test velocity estimation with improving accuracy."""
        now = datetime.now(timezone.utc)
        history = [
            {
                "timestamp": (now - timedelta(hours=5 - i)).isoformat(),
                "current_accuracy": 0.45 + (i * 0.02),
            }
            for i in range(0, 5)
        ]
        velocity = projector._estimate_velocity(history, 0.55)
        # Should be positive (improving)
        assert velocity > 0

    def test_regime_risk_factor_no_data(self, projector):
        """Test regime risk when regime data unavailable."""
        risk = projector._detect_regime_risk_factor("EUR_USD")
        assert 0.0 <= risk <= 1.0

    def test_regime_risk_factor_normal_regime(self, projector, temp_dirs):
        """Test regime risk with normal regime."""
        regime_data = {"_global": {"current_regime": "NORMAL"}}
        regime_file = temp_dirs / "trained_data" / "regime_transitions.json"
        regime_file.write_text(json.dumps(regime_data, indent=2))

        risk = projector._detect_regime_risk_factor("EUR_USD")
        # NORMAL regime should have low risk
        assert 0.0 <= risk <= 0.2

    def test_regime_risk_factor_extreme_regime(self, projector, temp_dirs):
        """Test regime risk with extreme regime."""
        regime_data = {"_global": {"current_regime": "EXTREME"}}
        regime_file = temp_dirs / "trained_data" / "regime_transitions.json"
        regime_file.write_text(json.dumps(regime_data, indent=2))

        risk = projector._detect_regime_risk_factor("EUR_USD")
        # EXTREME regime should have high risk
        assert risk > 0.5

    def test_feature_drift_score_no_features(self, projector):
        """Test feature drift when no features provided."""
        score = projector._compute_feature_drift_score(None)
        assert score == 0.0

    def test_feature_drift_score_no_baseline(self, projector):
        """Test feature drift when baseline unavailable."""
        features = np.array([1.0, 2.0, 3.0])
        score = projector._compute_feature_drift_score(features)
        assert score == 0.0

    def test_feature_drift_score_within_baseline(self, projector, temp_dirs):
        """Test feature drift when features are within baseline."""
        baseline = {"mean": [1.0, 2.0, 3.0], "std": [0.5, 0.5, 0.5]}
        baseline_file = temp_dirs / "trained_data" / "feature_baseline.json"
        baseline_file.write_text(json.dumps(baseline, indent=2))

        # Features very close to mean
        features = np.array([1.01, 2.01, 3.01])
        score = projector._compute_feature_drift_score(features)
        assert 0.0 <= score < 0.1

    def test_feature_drift_score_far_from_baseline(self, projector, temp_dirs):
        """Test feature drift when features deviate significantly."""
        baseline = {"mean": [1.0, 2.0, 3.0], "std": [0.5, 0.5, 0.5]}
        baseline_file = temp_dirs / "trained_data" / "feature_baseline.json"
        baseline_file.write_text(json.dumps(baseline, indent=2))

        # Features far from mean (>2 std deviations)
        features = np.array([3.0, 5.0, 7.0])
        score = projector._compute_feature_drift_score(features)
        assert score > 0.1

    def test_project_all_pairs(self, projector):
        """Test batch projection for multiple pairs."""
        pairs = ["EUR_USD", "GBP_USD", "JPY_USD"]
        projections = projector.project_all_pairs(pairs)

        assert len(projections) == 3
        for pair in pairs:
            assert pair in projections
            assert isinstance(projections[pair], DriftProjectionCurve)

    def test_project_all_pairs_with_accuracies(self, projector):
        """Test batch projection with specific accuracies."""
        pairs = ["EUR_USD", "GBP_USD"]
        accuracies = {"EUR_USD": 0.52, "GBP_USD": 0.48}
        projections = projector.project_all_pairs(pairs, accuracies)

        assert projections["EUR_USD"].current_accuracy == 0.52
        assert projections["GBP_USD"].current_accuracy == 0.48

    def test_save_projections(self, projector, temp_dirs):
        """Test projection persistence."""
        curve1 = projector.project("EUR_USD", 0.55)
        curve2 = projector.project("GBP_USD", 0.52)

        projections = {"EUR_USD": curve1, "GBP_USD": curve2}
        projector.save_projections(projections)

        # Verify file was created
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        assert cache_file.exists()

        # Verify content
        data = json.loads(cache_file.read_text())
        assert "projections" in data
        assert "EUR_USD" in data["projections"]
        assert "GBP_USD" in data["projections"]

    def test_map_to_action_stable(self, projector):
        """Test action mapping for stable projection."""
        points = [
            DriftProjectionPoint(6, 0.55, 0.50, 0.60, "stable"),
            DriftProjectionPoint(12, 0.55, 0.50, 0.60, "stable"),
            DriftProjectionPoint(24, 0.54, 0.49, 0.59, "stable"),
            DriftProjectionPoint(48, 0.54, 0.49, 0.59, "stable"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        action, urgency = projector._map_to_action(curve)
        assert action == "monitor"
        assert urgency == "low"

    def test_map_to_action_degrading_within_48h(self, projector):
        """Test action mapping for degradation within 48h."""
        points = [
            DriftProjectionPoint(6, 0.54, 0.49, 0.59, "stable"),
            DriftProjectionPoint(12, 0.52, 0.47, 0.57, "stable"),
            DriftProjectionPoint(24, 0.49, 0.44, 0.54, "degrading"),
            DriftProjectionPoint(48, 0.47, 0.42, 0.52, "degrading"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        action, urgency = projector._map_to_action(curve)
        assert action == "prepare_retrain"
        assert urgency in ["low", "medium"]

    def test_map_to_action_critical_soon(self, projector):
        """Test action mapping for critical degradation within 12h."""
        points = [
            DriftProjectionPoint(6, 0.52, 0.47, 0.57, "degrading"),
            DriftProjectionPoint(12, 0.47, 0.42, 0.52, "critical"),
            DriftProjectionPoint(24, 0.42, 0.37, 0.47, "critical"),
            DriftProjectionPoint(48, 0.37, 0.32, 0.42, "critical"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.55,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        action, urgency = projector._map_to_action(curve)
        assert action == "reduce_sizing"
        assert urgency == "high"

    def test_map_to_action_currently_critical(self, projector):
        """Test action mapping when already below critical threshold."""
        points = [
            DriftProjectionPoint(6, 0.47, 0.42, 0.52, "critical"),
            DriftProjectionPoint(12, 0.46, 0.41, 0.51, "critical"),
            DriftProjectionPoint(24, 0.45, 0.40, 0.50, "critical"),
            DriftProjectionPoint(48, 0.44, 0.39, 0.49, "critical"),
        ]
        curve = DriftProjectionCurve(
            pair="EUR_USD",
            generated_at=datetime.now(timezone.utc).isoformat(),
            current_accuracy=0.47,
            baseline_accuracy=0.55,
            projection_points=points,
        )
        action, urgency = projector._map_to_action(curve)
        assert action == "immediate_retrain"
        assert urgency == "critical"


class TestSelfModelState:
    """Test SelfModelState."""

    def test_get_summary_no_projections(self, self_model_state):
        """Test summary when no projections available."""
        summary = self_model_state.get_summary()
        assert summary["pairs_monitored"] == 0
        assert summary["overall_health"] == "unknown"
        assert summary["worst_pair"] is None

    def test_get_summary_stable(self, self_model_state, temp_dirs):
        """Test summary with stable projections."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "current_accuracy": 0.55,
                    "action_urgency": "low",
                    "recommended_action": "monitor",
                    "hours_until_critical": None,
                },
                "GBP_USD": {
                    "pair": "GBP_USD",
                    "current_accuracy": 0.54,
                    "action_urgency": "low",
                    "recommended_action": "monitor",
                    "hours_until_critical": None,
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        summary = self_model_state.get_summary()
        assert summary["pairs_monitored"] == 2
        assert summary["overall_health"] == "stable"
        assert len(summary["pairs_critical"]) == 0
        assert len(summary["pairs_warning"]) == 0

    def test_get_summary_with_critical(self, self_model_state, temp_dirs):
        """Test summary with critical pair."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "current_accuracy": 0.47,
                    "action_urgency": "critical",
                    "recommended_action": "immediate_retrain",
                    "hours_until_critical": 2,
                },
                "GBP_USD": {
                    "pair": "GBP_USD",
                    "current_accuracy": 0.54,
                    "action_urgency": "low",
                    "recommended_action": "monitor",
                    "hours_until_critical": None,
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        summary = self_model_state.get_summary()
        assert summary["overall_health"] == "critical"
        assert "EUR_USD" in summary["pairs_critical"]

    def test_get_worst_pair_stable(self, self_model_state, temp_dirs):
        """Test worst_pair with all stable — returns first stable pair."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "action_urgency": "low",
                    "hours_until_critical": None,
                },
                "GBP_USD": {
                    "pair": "GBP_USD",
                    "action_urgency": "low",
                    "hours_until_critical": None,
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        worst = self_model_state.get_worst_pair()
        # All stable, so returns first encountered (order depends on dict iteration)
        # Just verify it returns a valid pair or None
        assert worst in [None, "EUR_USD", "GBP_USD"]

    def test_get_worst_pair_with_critical(self, self_model_state, temp_dirs):
        """Test worst_pair returns critical pair."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "action_urgency": "low",
                    "hours_until_critical": 48,
                },
                "GBP_USD": {
                    "pair": "GBP_USD",
                    "action_urgency": "critical",
                    "hours_until_critical": 6,
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        worst = self_model_state.get_worst_pair()
        assert worst == "GBP_USD"

    def test_needs_immediate_action_false(self, self_model_state, temp_dirs):
        """Test immediate action flag when not needed."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "action_urgency": "low",
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        assert self_model_state.needs_immediate_action() is False

    def test_needs_immediate_action_true(self, self_model_state, temp_dirs):
        """Test immediate action flag when critical."""
        projections = {
            "projections": {
                "EUR_USD": {
                    "pair": "EUR_USD",
                    "action_urgency": "critical",
                },
            }
        }
        cache_file = temp_dirs / "trained_data" / "drift_projections.json"
        cache_file.write_text(json.dumps(projections, indent=2))

        assert self_model_state.needs_immediate_action() is True


class TestDriftMonitorIntegration:
    """Test integration of DriftProjector with DriftMonitor."""

    def test_drift_monitor_includes_projection(self, temp_dirs):
        """Test that DriftMonitor report includes projection when available."""
        from src.scanner.automation.drift_monitor import DriftMonitor

        # Create mock replay_validator
        mock_validator = MagicMock()
        mock_validator.replay_scan.return_value = {
            "n_trades": 10,
            "win_rate": 0.50,
        }
        mock_validator.detect_drift.return_value = {
            "drift_score": -0.05,
            "is_drifting": False,
            "current_accuracy": 0.50,
            "n_trades": 10,
        }

        monitor = DriftMonitor(
            replay_validator=mock_validator,
            persistence_path=temp_dirs / "trained_data" / "drift_monitor_log.json",
        )

        # Create sample candles
        candles = [{"time": i, "open": 1.1, "close": 1.1} for i in range(100)]

        # Run check
        report = monitor.run_drift_check("EUR_USD", candles, baseline_accuracy=0.55)

        # Verify report has projection key
        assert "projection" in report
        assert "hours_until_critical" in report["projection"]
        assert "recommended_action" in report["projection"]
        assert "action_urgency" in report["projection"]

    def test_drift_monitor_projection_robustness(self, temp_dirs):
        """Test that DriftMonitor handles projection errors gracefully."""
        from src.scanner.automation.drift_monitor import DriftMonitor

        mock_validator = MagicMock()
        mock_validator.replay_scan.return_value = {"n_trades": 10}
        mock_validator.detect_drift.return_value = {
            "drift_score": 0.0,
            "is_drifting": False,
            "current_accuracy": 0.55,
            "n_trades": 10,
        }

        monitor = DriftMonitor(
            replay_validator=mock_validator,
            persistence_path=temp_dirs / "trained_data" / "drift_monitor_log.json",
        )

        candles = [{"time": i, "open": 1.1, "close": 1.1} for i in range(100)]

        # Should not raise even if projection fails
        report = monitor.run_drift_check("EUR_USD", candles)
        assert "error" not in report or report.get("error") is None
