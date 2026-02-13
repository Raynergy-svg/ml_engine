#!/usr/bin/env python3
"""
Demo script showing DeploymentValidator usage and output.

This script demonstrates the deployment validation gate with various
scenarios: approved deployment, failed validation, and partial failures.
"""

from src.training.deployment_gate import (
    DeploymentValidator,
    ValidationCriteria,
)


def demo_approved_deployment():
    """Demo: All checks pass - deployment approved."""
    print("\n" + "=" * 70)
    print("DEMO 1: Successful Deployment (All Checks Pass)")
    print("=" * 70)

    validator = DeploymentValidator()

    # High-quality metrics that should pass all checks
    dir_metrics = {
        'val_accuracy': 0.78,
        'val_balanced_accuracy': 0.75,
        'cv_mean': 0.76,
        'cv_std': 0.03,
        'bootstrap_ci_lower': 0.72,
        'bootstrap_ci_upper': 0.80,
    }

    xgb_metrics = {
        'acceleration_accuracy': 0.68,
        'momentum_mae': 0.08,
    }

    rf_metrics = {
        'drawdown_mae_bps': 65.0,
        'streak_prob_mae': 0.10,
    }

    ridge_metrics = {
        'r2_score': 0.52,
        'confidence_mae': 10.5,
    }

    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        training_data_size=8000,
    )

    print(validator.format_report(result))
    print("\n✓ This model is APPROVED for production deployment")


def demo_failed_deployment():
    """Demo: Critical checks fail - deployment blocked."""
    print("\n" + "=" * 70)
    print("DEMO 2: Failed Deployment (Critical Failures)")
    print("=" * 70)

    validator = DeploymentValidator()

    # Poor metrics that fail critical checks
    dir_metrics = {
        'val_accuracy': 0.58,  # Below 65% threshold - CRITICAL
        'val_balanced_accuracy': 0.55,  # Below 60% threshold - CRITICAL
    }

    xgb_metrics = {
        'acceleration_accuracy': 0.52,  # Below 60% threshold - CRITICAL
        'momentum_mae': 0.18,
    }

    rf_metrics = {
        'drawdown_mae_bps': 120.0,
        'streak_prob_mae': 0.18,
    }

    ridge_metrics = {
        'r2_score': 0.15,  # Below 30% threshold - CRITICAL
        'confidence_mae': 18.0,
    }

    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        training_data_size=500,  # Below 1000 threshold - CRITICAL
    )

    print(validator.format_report(result))
    print("\n✗ This model is REJECTED - critical quality issues detected")


def demo_partial_failures():
    """Demo: Non-critical failures only - deployment approved with warnings."""
    print("\n" + "=" * 70)
    print("DEMO 3: Approved with Warnings (Non-Critical Failures)")
    print("=" * 70)

    validator = DeploymentValidator()

    # Metrics that pass critical checks but have some issues
    dir_metrics = {
        'val_accuracy': 0.72,  # Good
        'val_balanced_accuracy': 0.68,  # Good
        'cv_std': 0.08,  # High variance (non-critical)
    }

    xgb_metrics = {
        'acceleration_accuracy': 0.65,  # Good
        'momentum_mae': 0.20,  # Higher than ideal (non-critical)
    }

    rf_metrics = {
        'drawdown_mae_bps': 110.0,  # Slightly high (non-critical)
        'streak_prob_mae': 0.12,  # Good
    }

    ridge_metrics = {
        'r2_score': 0.38,  # Good
        'confidence_mae': 16.5,  # Slightly high (non-critical)
    }

    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        training_data_size=5000,
    )

    print(validator.format_report(result))
    print("\n⚠ This model is APPROVED but requires monitoring")


def demo_custom_criteria():
    """Demo: Custom validation criteria for stricter requirements."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Validation Criteria (Stricter Requirements)")
    print("=" * 70)

    # Create custom criteria with higher standards
    custom_criteria = ValidationCriteria(
        min_accuracy=0.75,  # Raised from 0.65
        min_balanced_accuracy=0.70,  # Raised from 0.60
        max_cv_std=0.03,  # Lowered from 0.05
        min_acceleration_accuracy=0.65,  # Raised from 0.60
        min_ridge_r2=0.40,  # Raised from 0.30
    )

    validator = DeploymentValidator(criteria=custom_criteria)

    # Metrics that would pass default criteria
    dir_metrics = {
        'val_accuracy': 0.70,  # Would pass default (0.65) but fails custom (0.75)
        'val_balanced_accuracy': 0.68,  # Would pass default (0.60) but fails custom (0.70)
        'cv_std': 0.04,  # Would pass default (0.05) but fails custom (0.03)
    }

    xgb_metrics = {
        'acceleration_accuracy': 0.62,  # Would pass default (0.60) but fails custom (0.65)
        'momentum_mae': 0.10,
    }

    rf_metrics = {
        'drawdown_mae_bps': 85.0,
        'streak_prob_mae': 0.11,
    }

    ridge_metrics = {
        'r2_score': 0.35,  # Would pass default (0.30) but fails custom (0.40)
        'confidence_mae': 12.0,
    }

    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        training_data_size=5000,
    )

    print(validator.format_report(result))
    print("\n✗ This model fails stricter custom criteria")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "DEPLOYMENT VALIDATOR DEMONSTRATION" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run all demos
    demo_approved_deployment()
    demo_failed_deployment()
    demo_partial_failures()
    demo_custom_criteria()

    print("\n" + "=" * 70)
    print("END OF DEMONSTRATION")
    print("=" * 70)
    print("\nSummary:")
    print("  • Demo 1: All checks pass → APPROVED")
    print("  • Demo 2: Critical failures → REJECTED")
    print("  • Demo 3: Non-critical failures only → APPROVED (with warnings)")
    print("  • Demo 4: Custom strict criteria → REJECTED")
    print("\nThe deployment gate ensures only production-ready models are deployed!")
    print()
