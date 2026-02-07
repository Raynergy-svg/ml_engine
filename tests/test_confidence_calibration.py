"""Unit tests for confidence calibration module."""

import numpy as np
import pytest
from src.risk.confidence_calibration import (
    CalibrationConfig,
    CalibrationResult,
    ConfidenceCalibrator,
    ConfidenceAdjuster,
    create_default_calibrator,
    create_adjuster,
)


class TestCalibrationConfig:
    """Test CalibrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        assert config.method == 'platt'
        assert config.min_confidence_threshold == 0.5
        assert config.max_confidence_threshold == 0.95
        assert config.apply_directional_adjustment is True
        assert config.apply_win_probability_adjustment is True
        assert config.apply_trading_context_adjustment is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CalibrationConfig(
            method='isotonic',
            min_confidence_threshold=0.6,
            max_confidence_threshold=0.9,
            apply_directional_adjustment=False,
            apply_win_probability_adjustment=False,
            apply_trading_context_adjustment=False
        )
        assert config.method == 'isotonic'
        assert config.min_confidence_threshold == 0.6
        assert config.max_confidence_threshold == 0.9
        assert config.apply_directional_adjustment is False
        assert config.apply_win_probability_adjustment is False
        assert config.apply_trading_context_adjustment is False


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""
    
    def test_result_creation(self):
        """Test CalibrationResult creation."""
        result = CalibrationResult(
            original_confidence=0.8,
            calibrated_confidence=0.75,
            adjustment_factors={'directional': 0.95},
            calibration_method='platt',
            is_valid=True,
            reason='Valid'
        )
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence == 0.75
        assert result.adjustment_factors == {'directional': 0.95}
        assert result.calibration_method == 'platt'
        assert result.is_valid is True
        assert result.reason == 'Valid'


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator class."""
    
    def test_default_calibrator_creation(self):
        """Test creation of default calibrator."""
        calibrator = create_default_calibrator()
        assert isinstance(calibrator, ConfidenceCalibrator)
        assert calibrator.config.method == 'platt'
        assert calibrator.config.min_confidence_threshold == 0.5
        assert calibrator.config.max_confidence_threshold == 0.95
    
    def test_calibrator_fit_insufficient_data(self):
        """Test calibrator fit with insufficient data."""
        calibrator = create_default_calibrator()
        confidence_scores = [0.7, 0.8]  # Less than minimum 10 samples
        outcomes = [True, False]
        
        # Should not raise an error but log a warning
        calibrator.fit(confidence_scores, outcomes)
        assert calibrator.is_fitted is False
    
    def test_calibrator_fit_sufficient_data(self):
        """Test calibrator fit with sufficient data."""
        calibrator = create_default_calibrator()
        np.random.seed(42)
        confidence_scores = np.random.uniform(0.5, 1.0, 20)
        outcomes = np.random.binomial(1, confidence_scores)
        
        calibrator.fit(confidence_scores, outcomes)
        assert calibrator.is_fitted is True
    
    def test_calibrate_confidence_unfitted(self):
        """Test calibrate_confidence when model is not fitted."""
        calibrator = create_default_calibrator()
        result = calibrator.calibrate_confidence(0.8)
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence == 0.8  # Returns original when not fitted
        assert result.is_valid is True
        assert result.reason == "Models not fitted"
    
    def test_calibrate_confidence_fitted(self):
        """Test calibrate_confidence when model is fitted."""
        calibrator = create_default_calibrator()
        np.random.seed(42)
        confidence_scores = np.random.uniform(0.5, 1.0, 20)
        outcomes = np.random.binomial(1, confidence_scores)
        
        calibrator.fit(confidence_scores, outcomes)
        result = calibrator.calibrate_confidence(0.8)
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence != 0.8  # Should be different after calibration
        assert result.is_valid is True
    
    def test_calibrate_confidence_thresholds(self):
        """Test that calibrated confidence respects thresholds."""
        calibrator = create_default_calibrator()
        result = calibrator.calibrate_confidence(0.3)  # Below min threshold
        
        assert result.calibrated_confidence >= 0.5  # Should be clipped to min
        assert result.calibrated_confidence <= 0.95  # Should be clipped to max
    
    def test_evaluate_calibration(self):
        """Test calibration evaluation."""
        calibrator = create_default_calibrator()
        np.random.seed(42)
        confidence_scores = np.random.uniform(0.5, 1.0, 20)
        outcomes = np.random.binomial(1, confidence_scores)
        
        calibrator.fit(confidence_scores, outcomes)
        evaluation = calibrator.evaluate_calibration(confidence_scores, outcomes)
        
        assert 'original_brier_score' in evaluation
        assert 'calibrated_brier_score' in evaluation
        assert 'improvement' in evaluation
        assert 'relative_improvement' in evaluation


class TestConfidenceAdjuster:
    """Test ConfidenceAdjuster class."""
    
    def test_default_adjuster_creation(self):
        """Test creation of default adjuster."""
        adjuster = create_adjuster()
        assert isinstance(adjuster, ConfidenceAdjuster)
        assert adjuster.config.min_confidence_threshold == 0.3
        assert adjuster.config.max_confidence_threshold == 0.95
    
    def test_adjust_confidence_no_prediction_direction(self):
        """Test adjust_confidence without prediction direction."""
        adjuster = create_adjuster()
        result = adjuster.adjust_confidence(
            original_confidence=0.8,
            prediction_direction=None
        )
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence == 0.8  # No adjustment without direction
        assert result.is_valid is True
    
    def test_adjust_confidence_with_prediction_direction(self):
        """Test adjust_confidence with prediction direction."""
        adjuster = create_adjuster()
        result = adjuster.adjust_confidence(
            original_confidence=0.8,
            prediction_direction=0.7  # 0.2 away from neutral
        )
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence != 0.8  # Should be adjusted
        assert result.is_valid is True
    
    def test_adjust_confidence_with_tier2_win_probability(self):
        """Test adjust_confidence with Tier2 win probability."""
        adjuster = create_adjuster()
        result = adjuster.adjust_confidence(
            original_confidence=0.8,
            prediction_direction=0.7,
            tier2_win_probability=0.4  # Below 0.5 threshold
        )
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence < 0.8  # Should be reduced due to low win probability
        assert result.is_valid is True
    
    def test_adjust_confidence_with_trading_context(self):
        """Test adjust_confidence with trading context."""
        adjuster = create_adjuster()
        result = adjuster.adjust_confidence(
            original_confidence=0.8,
            prediction_direction=0.7,
            trading_context='forex'
        )
        
        assert result.original_confidence == 0.8
        assert result.calibrated_confidence < 0.8  # Should be reduced for trading context
        assert result.is_valid is True
    
    def test_adjust_confidence_invalid(self):
        """Test adjust_confidence with invalid inputs."""
        adjuster = create_adjuster()
        result = adjuster.adjust_confidence(
            original_confidence=0.2,  # Below threshold
            prediction_direction=0.5
        )
        
        assert result.original_confidence == 0.2
        assert result.is_valid is False
        assert "Below threshold" in result.reason


class TestConfidenceAdjustmentFactors:
    """Test specific adjustment factors."""
    
    def test_directional_adjustment_factor(self):
        """Test directional adjustment factor calculation."""
        adjuster = create_adjuster()
        
        # Test neutral prediction (distance from 0.5 is 0)
        factor = adjuster._calculate_direction_factor(0.5)
        assert factor < 1.0  # Should apply penalty for neutral
        
        # Test strong directional prediction
        factor = adjuster._calculate_direction_factor(0.8)
        assert factor == 1.0  # Should not apply penalty for strong direction
    
    def test_win_probability_factor(self):
        """Test win probability adjustment factor."""
        adjuster = create_adjuster()
        
        # Test low win probability
        factor = adjuster._calculate_win_probability_factor(0.3)
        assert factor < 1.0  # Should reduce confidence
        
        # Test high win probability
        factor = adjuster._calculate_win_probability_factor(0.8)
        assert factor == 1.0  # Should not reduce confidence
    
    def test_trading_context_factor(self):
        """Test trading context adjustment factor."""
        adjuster = create_adjuster()
        
        factor = adjuster._calculate_trading_factor('forex')
        assert factor < 1.0  # Should apply trading penalty
    
    def test_validity_reason(self):
        """Test validity reason generation."""
        adjuster = create_adjuster()
        
        # Test valid confidence
        reason = adjuster._get_validity_reason(0.6, True)
        assert "Valid" in reason
        
        # Test invalid confidence
        reason = adjuster._get_validity_reason(0.2, False)
        assert "Below threshold" in reason


if __name__ == "__main__":
    pytest.main([__file__])
# — Raynergy-svg —
