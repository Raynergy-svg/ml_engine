"""
Integration test for FX Validation.

This test creates a mock model with FX-realistic metrics and validates it
against the new FX-specific criteria.
"""

import pytest

from src.training.fx_validation_criteria import (
    FXValidationCriteria,
    ValidationLevel,
    get_validation_criteria,
    validate_model_performance,
)


class TestFXValidationIntegration:
    """Integration tests for FX validation system."""

    def test_realistic_eur_usd_model_passes_validation(self):
        """A realistic EUR_USD model should pass validation."""
        # Simulate realistic FX model metrics
        # EUR_USD has stricter thresholds (1.02x multiplier), so we need slightly higher metrics
        metrics = {
            "val_accuracy": 0.555,  # 55.5% direction accuracy (above 54.06% threshold)
            "val_balanced_accuracy": 0.55,
            "cv_std": 0.06,  # 6% CV std (acceptable)
            "cv_degradation": 0.10,  # 10% degradation (acceptable)
            "bootstrap_ci_lower": 0.53,  # CI lower bound above adjusted threshold
            "collapse_ratio": 0.08,  # Low collapse ratio
        }
        
        # Get criteria for EUR_USD in normal regime
        criteria = get_validation_criteria("EUR_USD", regime="normal")
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is True, f"Expected pass but got failures: {failures}"

    def test_realistic_gbp_jpy_model_passes_with_lenient_criteria(self):
        """A realistic GBP_JPY model should pass with lenient criteria."""
        # GBP_JPY is high volatility, so metrics can be slightly worse
        metrics = {
            "val_accuracy": 0.535,  # 53.5% (slightly lower due to volatility)
            "val_balanced_accuracy": 0.53,
            "cv_std": 0.08,  # Higher CV std expected for volatile pair
            "cv_degradation": 0.12,
            "bootstrap_ci_lower": 0.51,
            "collapse_ratio": 0.10,
        }
        
        # Get criteria for GBP_JPY in normal regime (has lenient multipliers)
        criteria = get_validation_criteria("GBP_JPY", regime="normal")
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is True, f"Expected pass but got failures: {failures}"

    def test_poor_model_fails_validation(self):
        """A poor model should fail validation with clear reasons."""
        metrics = {
            "val_accuracy": 0.48,  # Below random (48%)
            "cv_std": 0.15,  # Very high variance
            "cv_degradation": 0.30,  # Severe overfitting
            "collapse_ratio": 0.40,  # High collapse (predicting one class)
        }
        
        criteria = get_validation_criteria("EUR_USD", regime="normal")
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is False
        assert len(failures) >= 3  # Should have multiple failures

    def test_high_volatility_regime_adjustment(self):
        """High volatility regime should adjust thresholds appropriately."""
        # Get criteria for normal vs high volatility
        normal = get_validation_criteria("GBP_USD", regime="normal")
        high_vol = get_validation_criteria("GBP_USD", regime="high_volatility")
        
        # High volatility should be more lenient
        assert high_vol.direction_accuracy_min < normal.direction_accuracy_min
        assert high_vol.cv_std_max > normal.cv_std_max

    def test_production_level_stricter(self):
        """Production level should be stricter than staging."""
        staging = get_validation_criteria("EUR_USD", regime="normal", level=ValidationLevel.STAGING)
        production = get_validation_criteria("EUR_USD", regime="normal", level=ValidationLevel.PRODUCTION)
        
        assert production.direction_accuracy_min > staging.direction_accuracy_min
        assert production.cv_std_max < staging.cv_std_max

    def test_edge_case_model_at_threshold(self):
        """Model exactly at threshold should pass."""
        criteria = FXValidationCriteria(
            direction_accuracy_min=0.53,
            cv_std_max=0.08,
        )
        
        metrics = {
            "val_accuracy": 0.53,  # Exactly at threshold
            "cv_std": 0.08,  # Exactly at threshold
        }
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is True

    def test_comprehensive_model_with_all_metrics(self):
        """Test model with all possible metrics."""
        # Use GBP_USD which has default multipliers (1.0)
        metrics = {
            "val_accuracy": 0.55,
            "val_balanced_accuracy": 0.54,
            "cv_std": 0.06,
            "cv_degradation": 0.10,
            "bootstrap_ci_lower": 0.52,
            "collapse_ratio": 0.08,
            "r2_score": 0.05,
            "acceleration_accuracy": 0.57,
            "momentum_mae": 0.10,
            "drawdown_mae_bps": 80.0,
            "streak_prob_mae": 0.10,
        }
        
        criteria = get_validation_criteria("GBP_USD", regime="normal")
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is True, f"Expected pass but got failures: {failures}"

    def test_model_with_momentum_failures(self):
        """Test model that fails momentum-specific checks."""
        metrics = {
            "val_accuracy": 0.55,
            "acceleration_accuracy": 0.50,  # Below 0.55 threshold
            "momentum_mae": 0.20,  # Above 0.15 threshold
        }
        
        criteria = FXValidationCriteria()
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is False
        assert any("Acceleration" in f for f in failures)
        assert any("Momentum" in f for f in failures)

    def test_model_with_risk_model_failures(self):
        """Test model that fails risk model checks."""
        metrics = {
            "val_accuracy": 0.55,
            "drawdown_mae_bps": 150.0,  # Above 100 bps threshold
            "streak_prob_mae": 0.20,  # Above 0.15 threshold
        }
        
        criteria = FXValidationCriteria()
        
        passed, failures = validate_model_performance(metrics, criteria)
        
        assert passed is False
        assert any("Drawdown" in f for f in failures)
        assert any("Streak" in f for f in failures)


class TestValidationWorkflow:
    """Test realistic validation workflows."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow for a trained model."""
        # Step 1: Get criteria for the pair (use GBP_USD with default multipliers)
        pair = "GBP_USD"
        regime = "normal"
        criteria = get_validation_criteria(pair, regime)
        
        # Step 2: Simulate model metrics from training
        model_metrics = {
            "val_accuracy": 0.542,
            "val_balanced_accuracy": 0.538,
            "cv_std": 0.065,
            "cv_degradation": 0.12,
            "bootstrap_ci_lower": 0.515,
            "collapse_ratio": 0.05,
        }
        
        # Step 3: Validate
        passed, failures = validate_model_performance(model_metrics, criteria)
        
        # Step 4: Check results
        if not passed:
            # In real workflow, would log failures and potentially reject model
            pytest.fail(f"Model validation failed: {failures}")
        
        assert passed is True

    def test_multi_pair_validation(self):
        """Test validation across multiple pairs."""
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "GBP_JPY"]
        
        # Use higher metrics that will pass all pair-specific thresholds
        base_metrics = {
            "val_accuracy": 0.56,  # High enough for all pairs
            "cv_std": 0.06,  # Low enough for all pairs
        }
        
        results = {}
        for pair in pairs:
            criteria = get_validation_criteria(pair, regime="normal")
            passed, failures = validate_model_performance(base_metrics, criteria)
            results[pair] = {"passed": passed, "failures": failures}
        
        # All pairs should pass with these metrics
        for pair, result in results.items():
            assert result["passed"], f"{pair} failed: {result['failures']}"

    def test_deployment_decision_workflow(self):
        """Test workflow for making deployment decisions."""
        # Simulate three models with different quality levels
        models = {
            "model_a": {"val_accuracy": 0.58, "cv_std": 0.05},  # Good - high enough for production
            "model_b": {"val_accuracy": 0.53, "cv_std": 0.08},  # Marginal
            "model_c": {"val_accuracy": 0.50, "cv_std": 0.12},  # Poor
        }
        
        # Use GBP_USD which has default multipliers (1.0) for cleaner testing
        criteria = get_validation_criteria("GBP_USD", regime="normal", level=ValidationLevel.PRODUCTION)
        
        deployment_decisions = {}
        for model_name, metrics in models.items():
            passed, failures = validate_model_performance(metrics, criteria)
            deployment_decisions[model_name] = {
                "deploy": passed,
                "reason": "Passed all checks" if passed else "; ".join(failures),
            }
        
        # Model A should be deployable (0.58 > 0.55 production threshold)
        assert deployment_decisions["model_a"]["deploy"] is True
        
        # Model C should not be deployable
        assert deployment_decisions["model_c"]["deploy"] is False


class TestConfiguredThresholds:
    """Test that configured thresholds match expected values."""

    def test_staging_thresholds_match_expected(self):
        """Verify staging level thresholds match expected values."""
        criteria = FXValidationCriteria()
        
        # Expected values from task specification
        assert criteria.direction_accuracy_min == 0.53
        assert criteria.balanced_accuracy_min == 0.53
        assert criteria.cv_std_max == 0.08
        assert criteria.bootstrap_ci_lower == 0.51
        assert criteria.max_cv_degradation == 0.15

    def test_production_thresholds_are_stricter(self):
        """Production thresholds should be stricter than staging."""
        staging = FXValidationCriteria()
        production = get_validation_criteria("GBP_USD", level=ValidationLevel.PRODUCTION)
        
        # Production should require higher accuracy
        assert production.direction_accuracy_min >= staging.direction_accuracy_min
        
        # Production should require lower variance
        assert production.cv_std_max <= staging.cv_std_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
