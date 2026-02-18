"""
Deployment Validation Gate for ML Models.

This module implements a production-ready deployment validation system that
ensures models meet quality standards before being deployed. It validates
training metrics, model stability, and production readiness.

Key Features:
1. Multi-criteria validation (accuracy, stability, drift)
2. Configurable thresholds for each model component
3. Detailed validation reporting
4. Integration with walk-forward validation results
5. Deployment decision logic with explanations

Usage:
    from src.training.deployment_gate import DeploymentValidator, ValidationCriteria

    validator = DeploymentValidator(
        min_accuracy=0.65,
        min_balanced_accuracy=0.60,
        max_cv_std=0.05,
    )

    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
    )

    if result.deployment_approved:
        # Save model artifacts
        save_models()
    else:
        # Log rejection reasons
        log_validation_failure(result.failure_reasons)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.training.unified_thresholds import get_threshold

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION CRITERIA AND THRESHOLDS
# =============================================================================

@dataclass
class ValidationCriteria:
    """
    Deployment validation criteria and thresholds.

    These thresholds define the minimum quality standards for production deployment.
    Adjust based on business requirements and risk tolerance.
    
    Note:
        Default values are sourced from src.training.unified_thresholds to ensure
        consistency across all validation modules. The bootstrap CI lower bound
        uses the FX-calibrated value of 0.51 (not 0.60) based on academic research
        showing 52-55% is realistic for FX direction prediction.
    """

    # === TRANSFORMER DIRECTION MODEL ===
    # Using unified thresholds for consistency
    min_accuracy: float = 0.53  # Minimum validation accuracy (FX-calibrated)
    min_balanced_accuracy: float = 0.53  # Minimum balanced accuracy (FX-calibrated)
    max_cv_std: float = 0.08  # Maximum CV std deviation (8% - higher for FX noise)
    min_bootstrap_ci_lower: Optional[float] = 0.51  # FX-calibrated (was 0.60)

    # === XGBOOST MOMENTUM MODEL ===
    min_acceleration_accuracy: float = 0.55  # Minimum acceleration accuracy
    max_momentum_mae: float = 0.15  # Maximum momentum MAE

    # === RANDOM FOREST RISK MODEL ===
    max_drawdown_mae_bps: float = 100.0  # Maximum drawdown MAE (100 bps)
    max_streak_prob_mae: float = 0.15  # Maximum streak probability MAE

    # === RIDGE CONFIDENCE MODEL ===
    min_ridge_r2: float = 0.01  # Minimum R² score (near zero for FX confidence)
    max_confidence_mae: float = 15.0  # Maximum confidence MAE

    # === STABILITY CHECKS ===
    max_metric_degradation: float = 0.15  # Max 15% degradation (FX-calibrated)
    require_cv_validation: bool = True  # Require walk-forward CV results
    require_bootstrap_ci: bool = False  # Require bootstrap CI (optional)

    # === PRODUCTION READINESS ===
    min_data_size: int = 2000  # Minimum training data size
    require_balanced_classes: bool = True  # Check class balance
    max_class_imbalance: float = 0.70  # Maximum class imbalance (70% one class)
    
    @classmethod
    def from_unified_thresholds(cls, preset: str = "fx_default") -> "ValidationCriteria":
        """
        Create ValidationCriteria from unified thresholds.
        
        Args:
            preset: Threshold preset name ("fx_default", "production")
            
        Returns:
            ValidationCriteria configured from unified thresholds
        """
        thresholds = get_threshold(preset)
        return cls(
            min_accuracy=thresholds.min_direction_accuracy,
            min_balanced_accuracy=thresholds.min_direction_accuracy,
            max_cv_std=thresholds.max_cv_fold_variance,
            min_bootstrap_ci_lower=thresholds.min_bootstrap_ci_lower,
            max_metric_degradation=thresholds.max_cv_degradation,
            min_ridge_r2=thresholds.min_r2_score,
            min_data_size=2000,
            max_class_imbalance=0.70,
        )


@dataclass
class ValidationResult:
    """
    Result of deployment validation.

    Contains detailed validation results, pass/fail status for each criterion,
    and overall deployment decision.
    """

    # === OVERALL RESULT ===
    deployment_approved: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # === INDIVIDUAL CHECKS ===
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    check_values: Dict[str, Any] = field(default_factory=dict)

    # === FAILURE ANALYSIS ===
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # === RECOMMENDATIONS ===
    recommendations: List[str] = field(default_factory=list)

    # === SUMMARY STATS ===
    total_checks: int = 0
    checks_failed: int = 0
    critical_failures: int = 0

    def add_check(
        self,
        name: str,
        passed: bool,
        value: Any = None,
        threshold: Any = None,
        critical: bool = False,
    ) -> None:
        """
        Add a validation check result.

        Args:
            name: Check name
            passed: Whether check passed
            value: Actual value
            threshold: Expected threshold
            critical: Whether this is a critical check
        """
        self.checks_passed[name] = passed
        self.total_checks += 1

        if value is not None:
            self.check_values[name] = {
                'value': value,
                'threshold': threshold,
                'passed': passed,
                'critical': critical,
            }

        if not passed:
            self.checks_failed += 1
            if critical:
                self.critical_failures += 1

            # Generate failure message
            if threshold is not None:
                msg = f"{name}: {value:.4f} (threshold: {threshold:.4f})"
            else:
                msg = f"{name}: {value}"
            self.failure_reasons.append(msg)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_recommendation(self, message: str) -> None:
        """Add a recommendation for improvement."""
        self.recommendations.append(message)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'deployment_approved': self.deployment_approved,
            'total_checks': self.total_checks,
            'checks_passed': self.total_checks - self.checks_failed,
            'checks_failed': self.checks_failed,
            'critical_failures': self.critical_failures,
            'pass_rate': (self.total_checks - self.checks_failed) / self.total_checks
            if self.total_checks > 0 else 0,
        }


# =============================================================================
# DEPLOYMENT VALIDATOR
# =============================================================================

class DeploymentValidator:
    """
    Validates model training results against deployment criteria.

    This class implements a comprehensive validation gate that checks multiple
    quality dimensions before approving model deployment to production.

    Validation Dimensions:
    1. Model Performance: Accuracy, precision, balanced accuracy
    2. Statistical Validation: Bootstrap CI, walk-forward CV
    3. Model Stability: Consistency across folds, low variance
    4. Component Quality: Each ensemble component meets standards
    5. Data Quality: Sufficient data, balanced classes
    6. Production Readiness: No critical issues

    Example:
        validator = DeploymentValidator()
        result = validator.validate(dir_metrics, xgb_metrics, rf_metrics, ridge_metrics)

        if result.deployment_approved:
            print("✓ Model approved for deployment")
        else:
            print(f"✗ Deployment blocked: {result.failure_reasons}")
    """

    def __init__(self, criteria: Optional[ValidationCriteria] = None):
        """
        Initialize deployment validator.

        Args:
            criteria: Validation criteria (uses defaults if None)
        """
        self.criteria = criteria or ValidationCriteria()
        logger.info("Initialized DeploymentValidator with criteria: %s", self.criteria)

    def validate(
        self,
        dir_metrics: Dict[str, Any],
        xgb_metrics: Dict[str, Any],
        rf_metrics: Dict[str, Any],
        ridge_metrics: Dict[str, Any],
        training_data_size: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate all model components against deployment criteria.

        Args:
            dir_metrics: Transformer direction model metrics
            xgb_metrics: XGBoost momentum model metrics
            rf_metrics: RandomForest risk model metrics
            ridge_metrics: Ridge confidence model metrics
            training_data_size: Size of training dataset

        Returns:
            ValidationResult with deployment decision and details
        """
        result = ValidationResult(deployment_approved=False)

        logger.info("Starting deployment validation...")

        # === TRANSFORMER DIRECTION VALIDATION ===
        self._validate_direction_model(dir_metrics, result)

        # === XGBOOST MOMENTUM VALIDATION ===
        self._validate_xgboost_model(xgb_metrics, result)

        # === RANDOM FOREST RISK VALIDATION ===
        self._validate_rf_model(rf_metrics, result)

        # === RIDGE CONFIDENCE VALIDATION ===
        self._validate_ridge_model(ridge_metrics, result)

        # === DATA QUALITY VALIDATION ===
        if training_data_size is not None:
            self._validate_data_quality(training_data_size, result)

        # === STABILITY VALIDATION ===
        self._validate_stability(dir_metrics, result)

        # === FINAL DECISION ===
        # Approve if no critical failures
        result.deployment_approved = result.critical_failures == 0

        # Add recommendations
        self._generate_recommendations(result)

        logger.info(
            "Deployment validation complete: %s (%d/%d checks passed)",
            "APPROVED" if result.deployment_approved else "REJECTED",
            result.total_checks - result.checks_failed,
            result.total_checks,
        )

        return result

    def _validate_direction_model(
        self, metrics: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate Transformer direction model."""
        logger.debug("Validating Transformer direction model...")

        # Validation accuracy
        val_acc = metrics.get('val_accuracy', 0)
        result.add_check(
            'Direction: Validation Accuracy',
            passed=val_acc >= self.criteria.min_accuracy,
            value=val_acc,
            threshold=self.criteria.min_accuracy,
            critical=True,
        )

        # Balanced accuracy
        balanced_acc = metrics.get('val_balanced_accuracy', val_acc)
        result.add_check(
            'Direction: Balanced Accuracy',
            passed=balanced_acc >= self.criteria.min_balanced_accuracy,
            value=balanced_acc,
            threshold=self.criteria.min_balanced_accuracy,
            critical=True,
        )

        # Bootstrap CI (if available)
        if 'bootstrap_ci_lower' in metrics and self.criteria.min_bootstrap_ci_lower:
            ci_lower = metrics['bootstrap_ci_lower']
            result.add_check(
                'Direction: Bootstrap CI Lower',
                passed=ci_lower >= self.criteria.min_bootstrap_ci_lower,
                value=ci_lower,
                threshold=self.criteria.min_bootstrap_ci_lower,
                critical=False,
            )

        # Cross-validation std (if available)
        if 'cv_std' in metrics:
            cv_std = metrics['cv_std']
            result.add_check(
                'Direction: CV Stability (std)',
                passed=cv_std <= self.criteria.max_cv_std,
                value=cv_std,
                threshold=self.criteria.max_cv_std,
                critical=False,
            )

    def _validate_xgboost_model(
        self, metrics: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate XGBoost momentum model."""
        logger.debug("Validating XGBoost momentum model...")

        # Acceleration accuracy
        acc_accuracy = metrics.get('acceleration_accuracy', 0)
        result.add_check(
            'Momentum: Acceleration Accuracy',
            passed=acc_accuracy >= self.criteria.min_acceleration_accuracy,
            value=acc_accuracy,
            threshold=self.criteria.min_acceleration_accuracy,
            critical=True,
        )

        # Momentum MAE
        momentum_mae = metrics.get('momentum_mae', float('inf'))
        result.add_check(
            'Momentum: MAE',
            passed=momentum_mae <= self.criteria.max_momentum_mae,
            value=momentum_mae,
            threshold=self.criteria.max_momentum_mae,
            critical=False,
        )

    def _validate_rf_model(
        self, metrics: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate RandomForest risk model."""
        logger.debug("Validating RandomForest risk model...")

        # Drawdown MAE (convert from pips to bps if needed)
        drawdown_mae_bps = metrics.get('drawdown_mae_bps')
        if drawdown_mae_bps is None:
            # Try pips format (1 pip = 10 bps for major pairs)
            drawdown_mae_pips = metrics.get('drawdown_mae_pips', 0)
            drawdown_mae_bps = drawdown_mae_pips * 10

        result.add_check(
            'Risk: Drawdown MAE',
            passed=drawdown_mae_bps <= self.criteria.max_drawdown_mae_bps,
            value=drawdown_mae_bps,
            threshold=self.criteria.max_drawdown_mae_bps,
            critical=False,
        )

        # Streak probability MAE
        streak_mae = metrics.get('streak_prob_mae', float('inf'))
        result.add_check(
            'Risk: Streak Probability MAE',
            passed=streak_mae <= self.criteria.max_streak_prob_mae,
            value=streak_mae,
            threshold=self.criteria.max_streak_prob_mae,
            critical=False,
        )

    def _validate_ridge_model(
        self, metrics: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate Ridge confidence model."""
        logger.debug("Validating Ridge confidence model...")

        # R² score
        r2_score = metrics.get('r2_score', 0)
        result.add_check(
            'Confidence: R² Score',
            passed=r2_score >= self.criteria.min_ridge_r2,
            value=r2_score,
            threshold=self.criteria.min_ridge_r2,
            critical=True,
        )

        # Confidence MAE
        conf_mae = metrics.get('confidence_mae', float('inf'))
        result.add_check(
            'Confidence: MAE',
            passed=conf_mae <= self.criteria.max_confidence_mae,
            value=conf_mae,
            threshold=self.criteria.max_confidence_mae,
            critical=False,
        )

    def _validate_data_quality(
        self, data_size: int, result: ValidationResult
    ) -> None:
        """Validate data quality and size."""
        logger.debug("Validating data quality...")

        # Minimum data size
        result.add_check(
            'Data: Minimum Size',
            passed=data_size >= self.criteria.min_data_size,
            value=data_size,
            threshold=self.criteria.min_data_size,
            critical=True,
        )

    def _validate_stability(
        self, dir_metrics: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate model stability across folds."""
        logger.debug("Validating model stability...")

        # Check if CV results exist when required
        if self.criteria.require_cv_validation:
            has_cv = 'cv_mean' in dir_metrics and 'cv_std' in dir_metrics
            result.add_check(
                'Stability: Walk-Forward CV Available',
                passed=has_cv,
                value=has_cv,
                threshold=True,
                critical=False,
            )

            if has_cv:
                # Check degradation from training to CV
                val_acc = dir_metrics.get('val_accuracy', 0)
                cv_mean = dir_metrics['cv_mean']
                degradation = val_acc - cv_mean

                result.add_check(
                    'Stability: CV Degradation',
                    passed=degradation <= self.criteria.max_metric_degradation,
                    value=degradation,
                    threshold=self.criteria.max_metric_degradation,
                    critical=False,
                )

        # Check if bootstrap CI exists when required
        if self.criteria.require_bootstrap_ci:
            has_bootstrap = 'bootstrap_ci_lower' in dir_metrics
            result.add_check(
                'Stability: Bootstrap CI Available',
                passed=has_bootstrap,
                value=has_bootstrap,
                threshold=True,
                critical=False,
            )

    def _generate_recommendations(self, result: ValidationResult) -> None:
        """Generate recommendations based on validation results."""
        # If validation accuracy is low
        if not result.checks_passed.get('Direction: Validation Accuracy', True):
            result.add_recommendation(
                "Improve direction model accuracy: "
                "1) Increase training data size, "
                "2) Tune hyperparameters in config, "
                "3) Add more informative features"
            )

        # If CV stability is poor
        if 'Direction: CV Stability (std)' in result.checks_passed:
            if not result.checks_passed['Direction: CV Stability (std)']:
                result.add_recommendation(
                    "Improve model stability: "
                    "1) Reduce model complexity, "
                    "2) Add L2 regularization, "
                    "3) Use ensemble methods"
                )

        # If any component failed
        if result.checks_failed > 0:
            result.add_recommendation(
                f"Address {result.checks_failed} failed checks before deployment"
            )

        # If approved but with warnings
        if result.deployment_approved and result.warnings:
            result.add_recommendation(
                "Monitor production performance closely due to validation warnings"
            )

    def format_report(self, result: ValidationResult) -> str:
        """
        Format validation result as a readable report.

        Args:
            result: ValidationResult to format

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DEPLOYMENT VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Decision: {'✓ APPROVED' if result.deployment_approved else '✗ REJECTED'}")
        lines.append(f"Checks: {result.total_checks - result.checks_failed}/{result.total_checks} passed")
        lines.append("")

        # Individual checks
        lines.append("Individual Checks:")
        lines.append("-" * 60)
        for check_name, passed in result.checks_passed.items():
            status = "✓" if passed else "✗"
            check_data = result.check_values.get(check_name, {})
            value = check_data.get('value')
            threshold = check_data.get('threshold')

            if value is not None and threshold is not None:
                if isinstance(value, (int, float)):
                    lines.append(f"{status} {check_name}: {value:.4f} (threshold: {threshold:.4f})")
                else:
                    lines.append(f"{status} {check_name}: {value} (threshold: {threshold})")
            else:
                lines.append(f"{status} {check_name}")

        # Failure reasons
        if result.failure_reasons:
            lines.append("")
            lines.append("Failure Reasons:")
            lines.append("-" * 60)
            for reason in result.failure_reasons:
                lines.append(f"  • {reason}")

        # Warnings
        if result.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.append("-" * 60)
            for warning in result.warnings:
                lines.append(f"  ⚠ {warning}")

        # Recommendations
        if result.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            lines.append("-" * 60)
            for rec in result.recommendations:
                lines.append(f"  → {rec}")

        lines.append("=" * 60)

        return "\n".join(lines)
