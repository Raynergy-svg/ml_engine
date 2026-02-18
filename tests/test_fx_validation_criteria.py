"""
Unit tests for FX-specific validation criteria.

Tests cover:
- FXValidationCriteria dataclass default values
- ValidationLevel enum values
- get_validation_criteria() function with various parameters
- validate_model_performance() function with passing/failing metrics
"""

import pytest
from dataclasses import fields

from src.training.fx_validation_criteria import (
    FXValidationCriteria,
    ValidationLevel,
    get_validation_criteria,
    validate_model_performance,
    quick_validate,
    compute_collapse_ratio,
    compute_cv_degradation,
    _get_criteria_by_level,
    PAIR_CRITERIA_ADJUSTMENTS,
    REGIME_ADJUSTMENTS,
)


class TestValidationLevelEnum:
    """Tests for ValidationLevel enum."""

    def test_enum_values_exist(self):
        """Verify all expected validation levels exist."""
        assert hasattr(ValidationLevel, "DEVELOPMENT")
        assert hasattr(ValidationLevel, "STAGING")
        assert hasattr(ValidationLevel, "PRODUCTION")
        assert hasattr(ValidationLevel, "CONSERVATIVE")

    def test_enum_string_values(self):
        """Verify enum string values are lowercase."""
        assert ValidationLevel.DEVELOPMENT.value == "development"
        assert ValidationLevel.STAGING.value == "staging"
        assert ValidationLevel.PRODUCTION.value == "production"
        assert ValidationLevel.CONSERVATIVE.value == "conservative"

    def test_enum_count(self):
        """Verify exactly 4 validation levels."""
        assert len(ValidationLevel) == 4


class TestFXValidationCriteriaDataclass:
    """Tests for FXValidationCriteria dataclass default values."""

    def test_default_direction_accuracy_min(self):
        """Default direction_accuracy_min should be 0.53 (53%)."""
        criteria = FXValidationCriteria()
        assert criteria.direction_accuracy_min == 0.53

    def test_default_balanced_accuracy_min(self):
        """Default balanced_accuracy_min should be 0.53 (53%)."""
        criteria = FXValidationCriteria()
        assert criteria.balanced_accuracy_min == 0.53

    def test_default_cv_std_max(self):
        """Default cv_std_max should be 0.08 (8%)."""
        criteria = FXValidationCriteria()
        assert criteria.cv_std_max == 0.08

    def test_default_bootstrap_ci_lower(self):
        """Default bootstrap_ci_lower should be 0.51 (51%)."""
        criteria = FXValidationCriteria()
        assert criteria.bootstrap_ci_lower == 0.51

    def test_default_max_cv_degradation(self):
        """Default max_cv_degradation should be 0.15 (15%)."""
        criteria = FXValidationCriteria()
        assert criteria.max_cv_degradation == 0.15

    def test_default_max_prediction_collapse(self):
        """Default max_prediction_collapse should be 0.15 (15%)."""
        criteria = FXValidationCriteria()
        assert criteria.max_prediction_collapse == 0.15

    def test_default_min_r2_score(self):
        """Default min_r2_score should be 0.01 (near zero for FX)."""
        criteria = FXValidationCriteria()
        assert criteria.min_r2_score == 0.01

    def test_default_min_acceleration_accuracy(self):
        """Default min_acceleration_accuracy should be 0.55 (55%)."""
        criteria = FXValidationCriteria()
        assert criteria.min_acceleration_accuracy == 0.55

    def test_default_max_momentum_mae(self):
        """Default max_momentum_mae should be 0.15."""
        criteria = FXValidationCriteria()
        assert criteria.max_momentum_mae == 0.15

    def test_default_max_drawdown_mae_bps(self):
        """Default max_drawdown_mae_bps should be 100.0 bps."""
        criteria = FXValidationCriteria()
        assert criteria.max_drawdown_mae_bps == 100.0

    def test_default_max_streak_prob_mae(self):
        """Default max_streak_prob_mae should be 0.15."""
        criteria = FXValidationCriteria()
        assert criteria.max_streak_prob_mae == 0.15

    def test_default_min_data_size(self):
        """Default min_data_size should be 2000 samples."""
        criteria = FXValidationCriteria()
        assert criteria.min_data_size == 2000

    def test_default_max_class_imbalance(self):
        """Default max_class_imbalance should be 0.70 (70%)."""
        criteria = FXValidationCriteria()
        assert criteria.max_class_imbalance == 0.70

    def test_to_dict_contains_all_fields(self):
        """to_dict() should return all dataclass fields."""
        criteria = FXValidationCriteria()
        result = criteria.to_dict()
        
        field_names = {f.name for f in fields(FXValidationCriteria)}
        assert set(result.keys()) == field_names

    def test_to_dict_values_match(self):
        """to_dict() values should match dataclass attribute values."""
        criteria = FXValidationCriteria(
            direction_accuracy_min=0.55,
            cv_std_max=0.06,
        )
        result = criteria.to_dict()
        
        assert result["direction_accuracy_min"] == 0.55
        assert result["cv_std_max"] == 0.06


class TestGetCriteriaByLevel:
    """Tests for _get_criteria_by_level function."""

    def test_development_level_lenient(self):
        """Development level should have most lenient thresholds."""
        criteria = _get_criteria_by_level(ValidationLevel.DEVELOPMENT)
        
        assert criteria.direction_accuracy_min == 0.51
        assert criteria.cv_std_max == 0.10
        assert criteria.max_cv_degradation == 0.20

    def test_staging_level_default(self):
        """Staging level should have default thresholds."""
        criteria = _get_criteria_by_level(ValidationLevel.STAGING)
        
        assert criteria.direction_accuracy_min == 0.53
        assert criteria.cv_std_max == 0.08
        assert criteria.max_cv_degradation == 0.15

    def test_production_level_stricter(self):
        """Production level should have stricter thresholds than staging."""
        staging = _get_criteria_by_level(ValidationLevel.STAGING)
        production = _get_criteria_by_level(ValidationLevel.PRODUCTION)
        
        assert production.direction_accuracy_min > staging.direction_accuracy_min
        assert production.cv_std_max < staging.cv_std_max
        assert production.max_cv_degradation < staging.max_cv_degradation

    def test_conservative_level_strictest(self):
        """Conservative level should have strictest thresholds."""
        criteria = _get_criteria_by_level(ValidationLevel.CONSERVATIVE)
        
        assert criteria.direction_accuracy_min == 0.55
        assert criteria.cv_std_max == 0.05
        assert criteria.max_cv_degradation == 0.10


class TestGetValidationCriteria:
    """Tests for get_validation_criteria function with various parameters."""

    # === Tests for different currency pairs ===

    def test_eur_usd_pair_adjustment(self):
        """EUR_USD should apply stricter accuracy multiplier (1.02)."""
        criteria = get_validation_criteria("EUR_USD", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        # EUR_USD has accuracy_mult=1.02, so should be stricter
        expected_accuracy = base.direction_accuracy_min * 1.02
        assert abs(criteria.direction_accuracy_min - expected_accuracy) < 0.001

    def test_gbp_usd_pair_default(self):
        """GBP_USD should use default multipliers (1.0)."""
        criteria = get_validation_criteria("GBP_USD", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        # GBP_USD has default multipliers
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    def test_usd_jpy_pair_adjustment(self):
        """USD_JPY should apply slightly stricter accuracy (1.01)."""
        criteria = get_validation_criteria("USD_JPY", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        expected_accuracy = base.direction_accuracy_min * 1.01
        assert abs(criteria.direction_accuracy_min - expected_accuracy) < 0.001

    def test_gbp_jpy_pair_lenient(self):
        """GBP_JPY should have more lenient criteria due to high volatility."""
        criteria = get_validation_criteria("GBP_JPY", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        # GBP_JPY has accuracy_mult=0.97 (more lenient)
        expected_accuracy = base.direction_accuracy_min * 0.97
        assert abs(criteria.direction_accuracy_min - expected_accuracy) < 0.001
        
        # GBP_JPY has cv_std_mult=1.15 (wider CV std allowed)
        expected_cv_std = base.cv_std_max * 1.15
        assert abs(criteria.cv_std_max - expected_cv_std) < 0.001

    def test_unknown_pair_uses_defaults(self):
        """Unknown pairs should use default multipliers (1.0)."""
        criteria = get_validation_criteria("UNKNOWN_PAIR", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    # === Tests for different regimes ===

    def test_normal_regime(self):
        """Normal regime should use base thresholds."""
        criteria = get_validation_criteria("GBP_USD", regime="normal")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        # Normal regime has multipliers of 1.0
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    def test_high_volatility_regime_lenient(self):
        """High volatility regime should be more lenient."""
        normal = get_validation_criteria("GBP_USD", regime="normal")
        high_vol = get_validation_criteria("GBP_USD", regime="high_volatility")
        
        # High volatility has accuracy_mult=0.97
        assert high_vol.direction_accuracy_min < normal.direction_accuracy_min
        # And cv_std_mult=1.15
        assert high_vol.cv_std_max > normal.cv_std_max

    def test_low_volatility_regime_stricter(self):
        """Low volatility regime should be stricter."""
        normal = get_validation_criteria("GBP_USD", regime="normal")
        low_vol = get_validation_criteria("GBP_USD", regime="low_volatility")
        
        # Low volatility has accuracy_mult=1.03
        assert low_vol.direction_accuracy_min > normal.direction_accuracy_min
        # And cv_std_mult=0.90
        assert low_vol.cv_std_max < normal.cv_std_max

    def test_crisis_regime_most_lenient(self):
        """Crisis regime should be most lenient."""
        normal = get_validation_criteria("GBP_USD", regime="normal")
        crisis = get_validation_criteria("GBP_USD", regime="crisis")
        
        # Crisis has accuracy_mult=0.95
        assert crisis.direction_accuracy_min < normal.direction_accuracy_min
        # And cv_std_mult=1.25
        assert crisis.cv_std_max > normal.cv_std_max

    def test_unknown_regime_uses_defaults(self):
        """Unknown regime should use default multipliers."""
        criteria = get_validation_criteria("GBP_USD", regime="unknown_regime")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    # === Tests for different validation levels ===

    def test_development_level(self):
        """Development level should return development criteria."""
        criteria = get_validation_criteria(
            "GBP_USD", regime="normal", level=ValidationLevel.DEVELOPMENT
        )
        base = _get_criteria_by_level(ValidationLevel.DEVELOPMENT)
        
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    def test_production_level(self):
        """Production level should return production criteria."""
        criteria = get_validation_criteria(
            "GBP_USD", regime="normal", level=ValidationLevel.PRODUCTION
        )
        base = _get_criteria_by_level(ValidationLevel.PRODUCTION)
        
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    def test_conservative_level(self):
        """Conservative level should return conservative criteria."""
        criteria = get_validation_criteria(
            "GBP_USD", regime="normal", level=ValidationLevel.CONSERVATIVE
        )
        base = _get_criteria_by_level(ValidationLevel.CONSERVATIVE)
        
        assert criteria.direction_accuracy_min == base.direction_accuracy_min

    # === Combined tests ===

    def test_pair_and_regime_combined(self):
        """Pair and regime adjustments should combine multiplicatively."""
        # EUR_USD with accuracy_mult=1.02, high_volatility with accuracy_mult=0.97
        # Combined: 1.02 * 0.97 = 0.9894
        criteria = get_validation_criteria("EUR_USD", regime="high_volatility")
        base = _get_criteria_by_level(ValidationLevel.STAGING)
        
        expected_accuracy = base.direction_accuracy_min * 1.02 * 0.97
        assert abs(criteria.direction_accuracy_min - expected_accuracy) < 0.001


class TestValidateModelPerformance:
    """Tests for validate_model_performance function."""

    @pytest.fixture
    def default_criteria(self):
        """Return default validation criteria."""
        return FXValidationCriteria()

    # === Passing validation tests ===

    def test_passing_validation_with_realistic_metrics(self, default_criteria):
        """Validation should pass with realistic FX metrics."""
        metrics = {
            "val_accuracy": 0.54,  # Above 0.53 threshold
            "cv_std": 0.06,  # Below 0.08 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is True
        assert failures == []

    def test_passing_validation_with_all_metrics(self, default_criteria):
        """Validation should pass when all metrics meet thresholds."""
        metrics = {
            "val_accuracy": 0.55,
            "val_balanced_accuracy": 0.54,
            "cv_std": 0.05,
            "cv_degradation": 0.10,
            "bootstrap_ci_lower": 0.52,
            "collapse_ratio": 0.10,
            "r2_score": 0.05,
            "acceleration_accuracy": 0.57,
            "momentum_mae": 0.10,
            "drawdown_mae_bps": 80.0,
            "streak_prob_mae": 0.10,
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is True
        assert failures == []

    def test_passing_validation_at_threshold(self, default_criteria):
        """Validation should pass when metrics are exactly at threshold."""
        metrics = {
            "val_accuracy": 0.53,  # Exactly at threshold
            "cv_std": 0.08,  # Exactly at threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is True

    def test_balanced_accuracy_defaults_to_val_accuracy(self, default_criteria):
        """Balanced accuracy should default to val_accuracy if not provided."""
        metrics = {
            "val_accuracy": 0.54,
            # No val_balanced_accuracy provided
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is True

    # === Failing validation tests ===

    def test_failing_validation_low_accuracy(self, default_criteria):
        """Validation should fail with accuracy below threshold."""
        metrics = {
            "val_accuracy": 0.50,  # Below 0.53 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        # Both direction accuracy and balanced accuracy fail (balanced defaults to val_accuracy)
        assert len(failures) == 2
        assert "Direction accuracy" in failures[0]

    def test_failing_validation_high_cv_std(self, default_criteria):
        """Validation should fail with CV std above threshold."""
        metrics = {
            "val_accuracy": 0.55,
            "cv_std": 0.10,  # Above 0.08 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        assert any("CV std" in f for f in failures)

    def test_failing_validation_high_cv_degradation(self, default_criteria):
        """Validation should fail with CV degradation above threshold."""
        metrics = {
            "val_accuracy": 0.55,
            "cv_degradation": 0.20,  # Above 0.15 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        assert any("CV degradation" in f for f in failures)

    def test_failing_validation_low_bootstrap_ci(self, default_criteria):
        """Validation should fail with bootstrap CI below threshold."""
        metrics = {
            "val_accuracy": 0.55,
            "bootstrap_ci_lower": 0.48,  # Below 0.51 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        assert any("Bootstrap CI" in f for f in failures)

    def test_failing_validation_prediction_collapse(self, default_criteria):
        """Validation should fail with prediction collapse above threshold."""
        metrics = {
            "val_accuracy": 0.55,
            "collapse_ratio": 0.20,  # Above 0.15 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        assert any("collapse" in f.lower() for f in failures)

    def test_failing_validation_low_r2_score(self, default_criteria):
        """Validation should fail with R² score below threshold."""
        metrics = {
            "val_accuracy": 0.55,
            "r2_score": -0.05,  # Below 0.01 threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        assert any("R²" in f for f in failures)

    # === Multiple failure tests ===

    def test_multiple_failure_reasons(self, default_criteria):
        """Validation should report all failure reasons."""
        metrics = {
            "val_accuracy": 0.50,  # Below threshold
            "cv_std": 0.12,  # Above threshold
            "cv_degradation": 0.25,  # Above threshold
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is False
        # 4 failures: direction accuracy, balanced accuracy, cv_std, cv_degradation
        assert len(failures) == 4

    def test_failure_messages_are_descriptive(self, default_criteria):
        """Failure messages should contain actual and threshold values."""
        metrics = {
            "val_accuracy": 0.50,
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert "0.50" in failures[0]  # Actual value
        assert "0.53" in failures[0]  # Threshold value

    # === Optional parameter tests ===

    def test_check_collapse_false_skips_collapse_check(self, default_criteria):
        """Setting check_collapse=False should skip collapse check."""
        metrics = {
            "val_accuracy": 0.55,
            "collapse_ratio": 0.30,  # Would fail if checked
        }
        
        passed, failures = validate_model_performance(
            metrics, default_criteria, check_collapse=False
        )
        
        assert passed is True

    def test_optional_metrics_not_provided(self, default_criteria):
        """Validation should pass if optional metrics are not provided."""
        metrics = {
            "val_accuracy": 0.55,
            # No cv_std, cv_degradation, etc.
        }
        
        passed, failures = validate_model_performance(metrics, default_criteria)
        
        assert passed is True


class TestQuickValidate:
    """Tests for quick_validate convenience function."""

    def test_quick_validate_passing(self):
        """Quick validate should pass with good accuracy."""
        # Use GBP_USD which has default multiplier (1.0), so 0.54 > 0.53 threshold
        passed, failures = quick_validate(0.54, cv_std=0.06, pair="GBP_USD")
        
        assert passed is True

    def test_quick_validate_failing(self):
        """Quick validate should fail with low accuracy."""
        passed, failures = quick_validate(0.50, cv_std=0.06, pair="EUR_USD")
        
        assert passed is False

    def test_quick_validate_without_cv_std(self):
        """Quick validate should work without cv_std."""
        # Use GBP_USD which has default multiplier (1.0)
        passed, failures = quick_validate(0.54, pair="GBP_USD")
        
        assert passed is True


class TestComputeCollapseRatio:
    """Tests for compute_collapse_ratio function."""

    def test_no_collapse_with_varied_predictions(self):
        """Should return 0.0 when predictions are varied."""
        predictions = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8]
        
        collapse = compute_collapse_ratio(predictions)
        
        assert collapse == 0.0

    def test_collapse_with_constant_predictions(self):
        """Should return high value when predictions are constant."""
        predictions = [0.51, 0.51, 0.51, 0.51, 0.51]
        
        collapse = compute_collapse_ratio(predictions)
        
        assert collapse > 0.0

    def test_collapse_with_narrow_range(self):
        """Should detect collapse with narrow prediction range."""
        predictions = [0.50, 0.51, 0.50, 0.51, 0.50]
        
        collapse = compute_collapse_ratio(predictions, threshold=0.05)
        
        assert collapse > 0.0

    def test_empty_predictions_returns_zero(self):
        """Should return 0.0 for empty predictions list."""
        collapse = compute_collapse_ratio([])
        
        assert collapse == 0.0


class TestComputeCVDegradation:
    """Tests for compute_cv_degradation function."""

    def test_positive_degradation(self):
        """Should return positive value when CV is lower than training."""
        degradation = compute_cv_degradation(0.70, 0.55)
        
        # (0.70 - 0.55) / 0.70 = 0.214
        assert abs(degradation - 0.214) < 0.01

    def test_zero_degradation(self):
        """Should return 0 when CV equals training."""
        degradation = compute_cv_degradation(0.55, 0.55)
        
        assert degradation == 0.0

    def test_negative_degradation_improvement(self):
        """Should return negative value when CV is higher than training."""
        degradation = compute_cv_degradation(0.55, 0.60)
        
        assert degradation < 0.0

    def test_zero_train_accuracy(self):
        """Should return 0.0 when train accuracy is 0."""
        degradation = compute_cv_degradation(0.0, 0.55)
        
        assert degradation == 0.0


class TestPairCriteriaAdjustments:
    """Tests for PAIR_CRITERIA_ADJUSTMENTS dictionary."""

    def test_all_major_pairs_have_adjustments(self):
        """All major pairs should have entries in PAIR_CRITERIA_ADJUSTMENTS."""
        major_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "GBP_JPY"]
        
        for pair in major_pairs:
            assert pair in PAIR_CRITERIA_ADJUSTMENTS, f"Missing adjustment for {pair}"

    def test_adjustments_have_required_keys(self):
        """Each pair adjustment should have accuracy_mult and cv_std_mult."""
        required_keys = ["accuracy_mult", "cv_std_mult"]
        
        for pair, adjustments in PAIR_CRITERIA_ADJUSTMENTS.items():
            for key in required_keys:
                assert key in adjustments, f"Missing {key} for {pair}"

    def test_accuracy_multipliers_are_reasonable(self):
        """Accuracy multipliers should be between 0.95 and 1.05."""
        for pair, adjustments in PAIR_CRITERIA_ADJUSTMENTS.items():
            mult = adjustments["accuracy_mult"]
            assert 0.95 <= mult <= 1.05, f"Unreasonable accuracy_mult for {pair}: {mult}"

    def test_cv_std_multipliers_are_reasonable(self):
        """CV std multipliers should be between 0.90 and 1.25."""
        for pair, adjustments in PAIR_CRITERIA_ADJUSTMENTS.items():
            mult = adjustments["cv_std_mult"]
            assert 0.90 <= mult <= 1.25, f"Unreasonable cv_std_mult for {pair}: {mult}"


class TestRegimeAdjustments:
    """Tests for REGIME_ADJUSTMENTS dictionary."""

    def test_all_regimes_have_adjustments(self):
        """All expected regimes should have entries."""
        expected_regimes = ["normal", "high_volatility", "low_volatility", "crisis"]
        
        for regime in expected_regimes:
            assert regime in REGIME_ADJUSTMENTS, f"Missing adjustment for {regime}"

    def test_adjustments_have_required_keys(self):
        """Each regime adjustment should have accuracy_mult and cv_std_mult."""
        required_keys = ["accuracy_mult", "cv_std_mult"]
        
        for regime, adjustments in REGIME_ADJUSTMENTS.items():
            for key in required_keys:
                assert key in adjustments, f"Missing {key} for {regime}"

    def test_high_volatility_is_more_lenient(self):
        """High volatility should have lower accuracy_mult than normal."""
        normal_mult = REGIME_ADJUSTMENTS["normal"]["accuracy_mult"]
        high_vol_mult = REGIME_ADJUSTMENTS["high_volatility"]["accuracy_mult"]
        
        assert high_vol_mult < normal_mult

    def test_low_volatility_is_stricter(self):
        """Low volatility should have higher accuracy_mult than normal."""
        normal_mult = REGIME_ADJUSTMENTS["normal"]["accuracy_mult"]
        low_vol_mult = REGIME_ADJUSTMENTS["low_volatility"]["accuracy_mult"]
        
        assert low_vol_mult > normal_mult

    def test_crisis_is_most_lenient(self):
        """Crisis regime should have lowest accuracy_mult."""
        crisis_mult = REGIME_ADJUSTMENTS["crisis"]["accuracy_mult"]
        
        for regime, adjustments in REGIME_ADJUSTMENTS.items():
            if regime != "crisis":
                assert crisis_mult <= adjustments["accuracy_mult"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
