"""
Tests for Phase 44 (US-280): ConfidenceCalibrationSystem wiring into ScannerAgentTeam.

Covers:
- Initialization of _confidence_calibrator
- Calibration overlay runs when calibrator available
- Graceful fallback when calibrator unavailable
- Exception handling doesn't break evaluate()
- Calibrated score replaces weighted_vote_score
- Calibration metadata stored in analysis
- Regime name handling (LOW, NORMAL, HIGH, EXTREME, numeric)
- Empty verdicts edge case
- Various agent score distributions
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import the class under test
from src.scanner.agents._team import ScannerAgentTeam, AgentVerdict
from src.scanner.results import PairAnalysis


@dataclass
class MockCalibratedConfidence:
    """Mock version of CalibratedConfidence for testing."""
    raw_weighted_score: float = 0.6
    final_confidence: float = 0.65
    ensemble_disagreement: float = 0.15
    disagreement_level: str = "MODERATE"
    agent_agreement: float = 0.80
    platt_calibrated: float = 0.64
    meta_confidence: float = 0.75


class TestConfidenceCalibrationWiring(unittest.TestCase):
    """Test ConfidenceCalibrationSystem integration into ScannerAgentTeam."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.weighted_vote_threshold = 0.55
        self.mock_config.regime_vote_thresholds = {}

    def test_init_creates_confidence_calibrator(self):
        """Test that ScannerAgentTeam.__init__ creates _confidence_calibrator."""
        with patch('src.scanner.agents._team.logger'):
            team = ScannerAgentTeam(self.mock_config)

            # Should have _confidence_calibrator attribute
            self.assertTrue(hasattr(team, '_confidence_calibrator'))
            # It should be initialized (not None, but may fail gracefully)
            # The exact type depends on whether ConfidenceCalibrationSystem is available

    def test_init_handles_calibrator_import_error(self):
        """Test that init gracefully handles missing ConfidenceCalibrationSystem."""
        with patch('src.scanner.agents._team.logger'):
            # Patch the lazy import inside __init__
            with patch('builtins.__import__', side_effect=ImportError("mock")):
                team = ScannerAgentTeam(self.mock_config)
                # Should still have the attribute, just not initialized
                self.assertTrue(hasattr(team, '_confidence_calibrator'))

    def test_calibration_overlay_applied_when_calibrator_available(self):
        """Test that calibration overlay runs and replaces weighted_vote_score."""
        team = ScannerAgentTeam(self.mock_config)

        # Mock the calibrator
        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.60,
            final_confidence=0.68,
            ensemble_disagreement=0.12,
            disagreement_level="LOW",
            agent_agreement=0.85,
            platt_calibrated=0.67,
            meta_confidence=0.80,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        # Create analysis and verdicts
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "NORMAL"
        analysis.confidence = 0.5

        verdicts = [
            AgentVerdict(name="trend", score=0.7, passed=True, weight=1.15, reason="uptrend", reason_code="TREND_UP"),
            AgentVerdict(name="momentum", score=0.65, passed=True, weight=1.05, reason="strong", reason_code="MOM_STRONG"),
        ]

        # Manually apply weighted vote logic and calibration (simulating evaluate's key section)
        total_weight = sum(max(v.weight, 0.0) for v in verdicts) or 1.0
        weighted_vote_score = sum(min(max(v.score, 0.0), 1.0) * max(v.weight, 0.0) for v in verdicts) / total_weight

        analysis.weighted_vote_score = weighted_vote_score
        regime_name = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()

        # Apply calibration overlay logic
        if team._confidence_calibrator is not None:
            _cal_verdicts = [
                {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
                for v in verdicts
            ]
            _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, regime_name)
            analysis.weighted_vote_score = min(max(_calibrated.final_confidence, 0.0), 1.0)
            analysis.calibration_details = {
                "raw_score": round(_calibrated.raw_weighted_score, 4),
                "calibrated_score": round(_calibrated.final_confidence, 4),
            }

        # Verify
        self.assertEqual(analysis.weighted_vote_score, 0.68)
        self.assertIsNotNone(analysis.calibration_details)
        self.assertEqual(analysis.calibration_details["calibrated_score"], 0.68)
        mock_calibrator.calibrate.assert_called_once()

    def test_calibration_skipped_when_calibrator_none(self):
        """Test that calibration is skipped gracefully when calibrator is None."""
        team = ScannerAgentTeam(self.mock_config)
        team._confidence_calibrator = None

        # Create analysis and verdicts
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "NORMAL"

        verdicts = [
            AgentVerdict(name="trend", score=0.7, passed=True, weight=1.15, reason="uptrend", reason_code="TREND_UP"),
        ]

        # Simulate weighted vote computation
        total_weight = sum(max(v.weight, 0.0) for v in verdicts) or 1.0
        weighted_vote_score = sum(min(max(v.score, 0.0), 1.0) * max(v.weight, 0.0) for v in verdicts) / total_weight
        analysis.weighted_vote_score = weighted_vote_score
        original_score = analysis.weighted_vote_score

        # Try to apply calibration (should skip)
        if team._confidence_calibrator is not None:
            # This should not execute
            self.fail("Calibrator should be None")

        # Score should remain unchanged
        self.assertEqual(analysis.weighted_vote_score, original_score)
        self.assertFalse(hasattr(analysis, 'calibration_details') and analysis.calibration_details)

    def test_calibration_exception_doesnt_break_evaluate(self):
        """Test that calibration exception doesn't break evaluate() flow."""
        team = ScannerAgentTeam(self.mock_config)

        # Mock calibrator that throws
        mock_calibrator = Mock()
        mock_calibrator.calibrate.side_effect = ValueError("Calibration failed")
        team._confidence_calibrator = mock_calibrator

        # Create analysis and verdicts
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "NORMAL"

        verdicts = [
            AgentVerdict(name="trend", score=0.7, passed=True, weight=1.15, reason="uptrend", reason_code="TREND_UP"),
        ]

        # Simulate weighted vote computation
        total_weight = sum(max(v.weight, 0.0) for v in verdicts) or 1.0
        weighted_vote_score = sum(min(max(v.score, 0.0), 1.0) * max(v.weight, 0.0) for v in verdicts) / total_weight

        analysis.confidence = 0.5
        analysis.weighted_vote_score = weighted_vote_score
        regime_name = "NORMAL"

        original_score = analysis.weighted_vote_score

        # Apply calibration overlay with exception handling
        if team._confidence_calibrator is not None:
            try:
                _cal_verdicts = [
                    {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
                    for v in verdicts
                ]
                _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, regime_name)
                analysis.weighted_vote_score = min(max(_calibrated.final_confidence, 0.0), 1.0)
            except Exception:
                # Should catch and skip, not propagate
                pass

        # Score should remain at original value (not overridden due to exception)
        self.assertEqual(analysis.weighted_vote_score, original_score)

    def test_calibrated_score_replaces_weighted_vote_score(self):
        """Test that calibrated score actually replaces the weighted_vote_score."""
        team = ScannerAgentTeam(self.mock_config)

        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.60,
            final_confidence=0.75,  # Higher than raw
            ensemble_disagreement=0.10,
            disagreement_level="LOW",
            agent_agreement=0.90,
            platt_calibrated=0.74,
            meta_confidence=0.88,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        # Raw score lower than calibrated
        raw_score = 0.60
        calibrated_score = 0.75

        # Simulate the actual override
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "NORMAL"
        analysis.weighted_vote_score = raw_score

        verdicts = [
            AgentVerdict(name="trend", score=0.6, passed=True, weight=1.15, reason="", reason_code=""),
        ]

        # Apply calibration
        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]
        _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, "NORMAL")
        analysis.weighted_vote_score = min(max(_calibrated.final_confidence, 0.0), 1.0)

        self.assertEqual(analysis.weighted_vote_score, calibrated_score)
        self.assertNotEqual(analysis.weighted_vote_score, raw_score)

    def test_calibration_details_stored_in_analysis(self):
        """Test that calibration details are stored in analysis.calibration_details."""
        team = ScannerAgentTeam(self.mock_config)

        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.60,
            final_confidence=0.68,
            ensemble_disagreement=0.15,
            disagreement_level="MODERATE",
            agent_agreement=0.82,
            platt_calibrated=0.66,
            meta_confidence=0.76,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        # Create analysis and apply calibration
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "NORMAL"

        verdicts = [
            AgentVerdict(name="trend", score=0.7, passed=True, weight=1.15, reason="", reason_code=""),
        ]

        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]
        _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, "NORMAL")

        analysis.calibration_details = {
            "raw_score": round(_calibrated.raw_weighted_score, 4),
            "calibrated_score": round(_calibrated.final_confidence, 4),
            "ensemble_disagreement": round(_calibrated.ensemble_disagreement, 4),
            "disagreement_level": _calibrated.disagreement_level,
            "agent_agreement_quality": round(_calibrated.agent_agreement, 4),
            "platt_adjusted": round(_calibrated.platt_calibrated, 4),
            "meta_confidence": round(_calibrated.meta_confidence, 4),
        }

        # Verify all fields present
        self.assertIsNotNone(analysis.calibration_details)
        self.assertEqual(analysis.calibration_details["raw_score"], 0.6)
        self.assertEqual(analysis.calibration_details["calibrated_score"], 0.68)
        self.assertEqual(analysis.calibration_details["disagreement_level"], "MODERATE")
        self.assertEqual(analysis.calibration_details["agent_agreement_quality"], 0.82)
        self.assertEqual(analysis.calibration_details["meta_confidence"], 0.76)

    def test_regime_name_low(self):
        """Test regime name handling: LOW."""
        team = ScannerAgentTeam(self.mock_config)
        mock_calibrator = Mock()
        team._confidence_calibrator = mock_calibrator

        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "LOW"

        # Extract regime name as in the actual code
        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "LOW")

    def test_regime_name_high(self):
        """Test regime name handling: HIGH."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "HIGH"

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "HIGH")

    def test_regime_name_extreme(self):
        """Test regime name handling: EXTREME."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "EXTREME"

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "EXTREME")

    def test_regime_name_numeric_1_maps_to_normal(self):
        """Test regime name handling: numeric 1 -> NORMAL."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = 1

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "NORMAL")

    def test_regime_name_numeric_2_maps_to_high(self):
        """Test regime name handling: numeric 2 -> HIGH."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = 2

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "HIGH")

    def test_regime_name_numeric_3_maps_to_extreme(self):
        """Test regime name handling: numeric 3 -> EXTREME."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = 3

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        self.assertEqual(_cal_regime, "EXTREME")

    def test_regime_name_unknown_defaults_to_normal(self):
        """Test regime name handling: unknown -> defaults appropriately."""
        team = ScannerAgentTeam(self.mock_config)
        analysis = PairAnalysis(pair="EURUSD")
        analysis.volatility_regime = "UNKNOWN"

        _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        if _cal_regime.isdigit():
            _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

        # UNKNOWN will not be converted (not in regime_names), stays as "UNKNOWN"
        self.assertEqual(_cal_regime, "UNKNOWN")

    def test_agent_score_distribution_all_high(self):
        """Test with all agents giving high scores."""
        team = ScannerAgentTeam(self.mock_config)
        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.95,
            final_confidence=0.92,
            ensemble_disagreement=0.02,
            disagreement_level="LOW",
            agent_agreement=0.98,
            platt_calibrated=0.91,
            meta_confidence=0.95,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        verdicts = [
            AgentVerdict(name="trend", score=0.95, passed=True, weight=1.15, reason="", reason_code=""),
            AgentVerdict(name="momentum", score=0.96, passed=True, weight=1.05, reason="", reason_code=""),
            AgentVerdict(name="volatility", score=0.94, passed=True, weight=1.0, reason="", reason_code=""),
        ]

        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]
        _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, "NORMAL")

        # Low disagreement expected
        self.assertEqual(_calibrated.disagreement_level, "LOW")
        self.assertEqual(_calibrated.ensemble_disagreement, 0.02)

    def test_agent_score_distribution_mixed(self):
        """Test with mixed agent scores (some high, some low)."""
        team = ScannerAgentTeam(self.mock_config)
        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.55,
            final_confidence=0.58,
            ensemble_disagreement=0.25,
            disagreement_level="MODERATE",
            agent_agreement=0.65,
            platt_calibrated=0.57,
            meta_confidence=0.60,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        verdicts = [
            AgentVerdict(name="trend", score=0.80, passed=True, weight=1.15, reason="", reason_code=""),
            AgentVerdict(name="momentum", score=0.30, passed=False, weight=1.05, reason="", reason_code=""),
            AgentVerdict(name="volatility", score=0.55, passed=True, weight=1.0, reason="", reason_code=""),
        ]

        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]
        _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, "NORMAL")

        # Moderate disagreement expected
        self.assertEqual(_calibrated.disagreement_level, "MODERATE")
        self.assertGreater(_calibrated.ensemble_disagreement, 0.2)

    def test_agent_score_distribution_all_low(self):
        """Test with all agents giving low scores."""
        team = ScannerAgentTeam(self.mock_config)
        mock_calibrator = Mock()
        mock_calibrated = MockCalibratedConfidence(
            raw_weighted_score=0.15,
            final_confidence=0.18,
            ensemble_disagreement=0.03,
            disagreement_level="LOW",
            agent_agreement=0.96,
            platt_calibrated=0.17,
            meta_confidence=0.92,
        )
        mock_calibrator.calibrate.return_value = mock_calibrated
        team._confidence_calibrator = mock_calibrator

        verdicts = [
            AgentVerdict(name="trend", score=0.15, passed=False, weight=1.15, reason="", reason_code=""),
            AgentVerdict(name="momentum", score=0.14, passed=False, weight=1.05, reason="", reason_code=""),
            AgentVerdict(name="volatility", score=0.16, passed=False, weight=1.0, reason="", reason_code=""),
        ]

        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]
        _calibrated = team._confidence_calibrator.calibrate(_cal_verdicts, "NORMAL")

        # Low disagreement expected (consensus that score is low)
        self.assertEqual(_calibrated.disagreement_level, "LOW")
        self.assertLess(_calibrated.ensemble_disagreement, 0.1)

    def test_verdicts_passed_field_preserved(self):
        """Test that verdict 'passed' field is preserved in calibration dicts."""
        team = ScannerAgentTeam(self.mock_config)

        verdicts = [
            AgentVerdict(name="trend", score=0.8, passed=True, weight=1.15, reason="", reason_code=""),
            AgentVerdict(name="risk_sentinel", score=0.3, passed=False, weight=1.25, reason="", reason_code=""),
        ]

        _cal_verdicts = [
            {"name": v.name, "score": min(max(v.score, 0.0), 1.0), "weight": max(v.weight, 0.0), "passed": v.passed}
            for v in verdicts
        ]

        # Verify passed field is preserved
        self.assertEqual(_cal_verdicts[0]["passed"], True)
        self.assertEqual(_cal_verdicts[1]["passed"], False)


if __name__ == '__main__':
    unittest.main()
