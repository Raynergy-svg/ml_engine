#!/usr/bin/env python3
"""
Unit tests for DeploymentValidator.

Tests validation logic, threshold checks, and deployment decisions.
"""

import pytest
from src.training.deployment_gate import (
    DeploymentValidator,
    ValidationCriteria,
    ValidationResult,
)


class TestValidationCriteria:
    """Test ValidationCriteria dataclass."""

    def test_default_criteria(self):
        """Test default validation criteria."""
        criteria = ValidationCriteria()

        # Check defaults
        assert criteria.min_accuracy == 0.65
        assert criteria.min_balanced_accuracy == 0.60
        assert criteria.max_cv_std == 0.05
        assert criteria.min_acceleration_accuracy == 0.60
        assert criteria.min_ridge_r2 == 0.30

    def test_custom_criteria(self):
        """Test custom validation criteria."""
        criteria = ValidationCriteria(
            min_accuracy=0.70,
            min_balanced_accuracy=0.65,
            max_cv_std=0.03,
        )

        assert criteria.min_accuracy == 0.70
        assert criteria.min_balanced_accuracy == 0.65
        assert criteria.max_cv_std == 0.03


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_empty_result(self):
        """Test empty validation result."""
        result = ValidationResult(deployment_approved=False)

        assert result.deployment_approved is False
        assert result.total_checks == 0
        assert result.checks_failed == 0
        assert len(result.failure_reasons) == 0
        assert len(result.warnings) == 0
        assert len(result.recommendations) == 0

    def test_add_check_passed(self):
        """Test adding a passed check."""
        result = ValidationResult(deployment_approved=False)
        result.add_check(
            name="Test Check",
            passed=True,
            value=0.75,
            threshold=0.65,
            critical=False,
        )

        assert result.total_checks == 1
        assert result.checks_failed == 0
        assert result.checks_passed["Test Check"] is True
        assert len(result.failure_reasons) == 0

    def test_add_check_failed(self):
        """Test adding a failed check."""
        result = ValidationResult(deployment_approved=False)
        result.add_check(
            name="Test Check",
            passed=False,
            value=0.55,
            threshold=0.65,
            critical=True,
        )

        assert result.total_checks == 1
        assert result.checks_failed == 1
        assert result.critical_failures == 1
        assert result.checks_passed["Test Check"] is False
        assert len(result.failure_reasons) == 1

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(deployment_approved=False)
        result.add_warning("Test warning")

        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

    def test_add_recommendation(self):
        """Test adding recommendations."""
        result = ValidationResult(deployment_approved=False)
        result.add_recommendation("Test recommendation")

        assert len(result.recommendations) == 1
        assert result.recommendations[0] == "Test recommendation"

    def test_get_summary(self):
        """Test getting summary statistics."""
        result = ValidationResult(deployment_approved=True)
        result.add_check("Check 1", passed=True, value=0.7, threshold=0.6)
        result.add_check("Check 2", passed=False, value=0.5, threshold=0.6)
        result.add_check("Check 3", passed=True, value=0.8, threshold=0.6)

        summary = result.get_summary()

        assert summary['total_checks'] == 3
        assert summary['checks_passed'] == 2
        assert summary['checks_failed'] == 1
        assert summary['pass_rate'] == pytest.approx(2/3)


class TestDeploymentValidator:
    """Test DeploymentValidator class."""

    def test_initialization_default(self):
        """Test validator initialization with defaults."""
        validator = DeploymentValidator()

        assert validator.criteria is not None
        assert isinstance(validator.criteria, ValidationCriteria)

    def test_initialization_custom(self):
        """Test validator initialization with custom criteria."""
        criteria = ValidationCriteria(min_accuracy=0.70)
        validator = DeploymentValidator(criteria=criteria)

        assert validator.criteria.min_accuracy == 0.70

    def test_validate_all_pass(self):
        """Test validation with all checks passing."""
        validator = DeploymentValidator()

        dir_metrics = {
            'val_accuracy': 0.75,
            'val_balanced_accuracy': 0.70,
            'cv_mean': 0.73,
            'cv_std': 0.03,
            'bootstrap_ci_lower': 0.68,
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.65,
            'momentum_mae': 0.10,
        }

        rf_metrics = {
            'drawdown_mae_bps': 80.0,
            'streak_prob_mae': 0.12,
        }

        ridge_metrics = {
            'r2_score': 0.45,
            'confidence_mae': 12.0,
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=5000,
        )

        assert result.deployment_approved is True
        assert result.critical_failures == 0
        summary = result.get_summary()
        assert summary['pass_rate'] == 1.0

    def test_validate_critical_failure(self):
        """Test validation with critical failure."""
        validator = DeploymentValidator()

        dir_metrics = {
            'val_accuracy': 0.55,  # Below threshold (0.65)
            'val_balanced_accuracy': 0.52,
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.65,
            'momentum_mae': 0.10,
        }

        rf_metrics = {
            'drawdown_mae_bps': 80.0,
            'streak_prob_mae': 0.12,
        }

        ridge_metrics = {
            'r2_score': 0.45,
            'confidence_mae': 12.0,
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=5000,
        )

        assert result.deployment_approved is False
        assert result.critical_failures > 0
        assert len(result.failure_reasons) > 0

    def test_validate_non_critical_failure(self):
        """Test validation with non-critical failures only."""
        validator = DeploymentValidator()

        dir_metrics = {
            'val_accuracy': 0.75,
            'val_balanced_accuracy': 0.70,
            'cv_std': 0.08,  # Above threshold (0.05), but non-critical
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.65,
            'momentum_mae': 0.20,  # Above threshold (0.15), non-critical
        }

        rf_metrics = {
            'drawdown_mae_bps': 80.0,
            'streak_prob_mae': 0.12,
        }

        ridge_metrics = {
            'r2_score': 0.45,
            'confidence_mae': 12.0,
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=5000,
        )

        # Should still approve if no critical failures
        assert result.deployment_approved is True
        assert result.critical_failures == 0
        # But should have some failed checks
        assert result.checks_failed > 0

    def test_validate_data_size_failure(self):
        """Test validation with insufficient data size."""
        validator = DeploymentValidator()

        dir_metrics = {
            'val_accuracy': 0.75,
            'val_balanced_accuracy': 0.70,
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.65,
            'momentum_mae': 0.10,
        }

        rf_metrics = {
            'drawdown_mae_bps': 80.0,
            'streak_prob_mae': 0.12,
        }

        ridge_metrics = {
            'r2_score': 0.45,
            'confidence_mae': 12.0,
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=500,  # Below threshold (1000)
        )

        assert result.deployment_approved is False
        assert result.critical_failures > 0
        assert any('Data: Minimum Size' in reason for reason in result.failure_reasons)

    def test_validate_missing_cv_data(self):
        """Test validation when CV data is missing but required."""
        criteria = ValidationCriteria(require_cv_validation=True)
        validator = DeploymentValidator(criteria=criteria)

        dir_metrics = {
            'val_accuracy': 0.75,
            'val_balanced_accuracy': 0.70,
            # Missing cv_mean and cv_std
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.65,
            'momentum_mae': 0.10,
        }

        rf_metrics = {
            'drawdown_mae_bps': 80.0,
            'streak_prob_mae': 0.12,
        }

        ridge_metrics = {
            'r2_score': 0.45,
            'confidence_mae': 12.0,
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=5000,
        )

        # Should have a failed check for CV availability
        assert 'Stability: Walk-Forward CV Available' in result.checks_passed
        assert result.checks_passed['Stability: Walk-Forward CV Available'] is False

    def test_format_report(self):
        """Test report formatting."""
        validator = DeploymentValidator()

        dir_metrics = {'val_accuracy': 0.75, 'val_balanced_accuracy': 0.70}
        xgb_metrics = {'acceleration_accuracy': 0.65, 'momentum_mae': 0.10}
        rf_metrics = {'drawdown_mae_bps': 80.0, 'streak_prob_mae': 0.12}
        ridge_metrics = {'r2_score': 0.45, 'confidence_mae': 12.0}

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=5000,
        )

        report = validator.format_report(result)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "DEPLOYMENT VALIDATION REPORT" in report
        assert "Individual Checks:" in report
        # Should contain approval status
        assert "✓ APPROVED" in report or "✗ REJECTED" in report


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_at_threshold(self):
        """Test metrics exactly at threshold values."""
        validator = DeploymentValidator()

        dir_metrics = {
            'val_accuracy': 0.65,  # Exactly at threshold
            'val_balanced_accuracy': 0.60,  # Exactly at threshold
        }

        xgb_metrics = {
            'acceleration_accuracy': 0.60,  # Exactly at threshold
            'momentum_mae': 0.15,  # Exactly at threshold
        }

        rf_metrics = {
            'drawdown_mae_bps': 100.0,  # Exactly at threshold
            'streak_prob_mae': 0.15,  # Exactly at threshold
        }

        ridge_metrics = {
            'r2_score': 0.30,  # Exactly at threshold
            'confidence_mae': 15.0,  # Exactly at threshold
        }

        result = validator.validate(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=1000,  # Exactly at threshold
        )

        # All checks should pass when exactly at threshold
        assert result.deployment_approved is True

    def test_empty_metrics(self):
        """Test handling of empty metrics dictionaries."""
        validator = DeploymentValidator()

        # Should handle missing keys gracefully
        result = validator.validate(
            dir_metrics={},
            xgb_metrics={},
            rf_metrics={},
            ridge_metrics={},
            training_data_size=None,
        )

        # Should fail but not crash
        assert result.deployment_approved is False
        assert result.total_checks > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
