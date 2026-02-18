"""
FX-Specific Validation Criteria for ML Model Deployment.

This module provides realistic validation thresholds calibrated for FX market
data prediction, where academic research shows 52-55% direction accuracy is
typical and achievable.

Key Features:
1. FX-calibrated validation thresholds (lower than general ML)
2. Tiered validation levels (development, staging, production, conservative)
3. Per-currency-pair criteria adjustments
4. Helper functions for model performance validation

Reference:
    - Cheung et al. (2005): FX direction prediction typically 52-55%
    - Menkhoff et al. (2012): Transaction costs require >52% for profitability

Usage:
    from src.training.fx_validation_criteria import (
        FXValidationCriteria,
        get_validation_criteria,
        validate_model_performance,
    )

    criteria = get_validation_criteria("EUR_USD", regime="normal")
    passed, failures = validate_model_performance(metrics, criteria)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from src.training.unified_thresholds import (
    FX_THRESHOLDS,
    PRODUCTION_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """
    Validation strictness levels for different deployment scenarios.

    Attributes:
        DEVELOPMENT: Lenient thresholds for experimentation
        STAGING: Default thresholds for pre-production validation
        PRODUCTION: Stricter thresholds for live trading
        CONSERVATIVE: Most strict thresholds for risk-averse deployment
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CONSERVATIVE = "conservative"


@dataclass
class FXValidationCriteria:
    """
    FX-specific validation criteria calibrated for financial time series.

    These thresholds are based on academic research showing that FX direction
    prediction typically achieves 52-55% accuracy, with transaction costs
    requiring >52% for profitability.

    Attributes:
        direction_accuracy_min: Minimum direction prediction accuracy (default: 0.53)
        balanced_accuracy_min: Minimum balanced accuracy for both classes (default: 0.53)
        cv_std_max: Maximum cross-validation standard deviation (default: 0.08)
        bootstrap_ci_lower: Minimum bootstrap confidence interval lower bound (default: 0.51)
        max_cv_degradation: Maximum allowed CV degradation from training (default: 0.15)
        min_r2_score: Minimum R² score for confidence regression (default: 0.01)
        max_prediction_collapse: Maximum ratio of predictions in single class (default: 0.15)
        min_acceleration_accuracy: Minimum momentum acceleration accuracy (default: 0.55)
        max_momentum_mae: Maximum momentum mean absolute error (default: 0.15)
        max_drawdown_mae_bps: Maximum drawdown MAE in basis points (default: 100.0)
        max_streak_prob_mae: Maximum streak probability MAE (default: 0.15)
        min_data_size: Minimum training data samples required (default: 2000)
        max_class_imbalance: Maximum class imbalance ratio (default: 0.70)
    """

    # === TRANSFORMER DIRECTION MODEL ===
    # Research shows 52-55% is realistic for FX direction prediction
    direction_accuracy_min: float = 0.53  # Minimum validation accuracy (53%)
    balanced_accuracy_min: float = 0.53  # Minimum balanced accuracy (53%)

    # Allow higher variance due to market noise and regime changes
    cv_std_max: float = 0.08  # Maximum CV std deviation (8%)

    # Bootstrap CI should account for regime uncertainty
    bootstrap_ci_lower: float = 0.51  # Minimum 51% CI lower bound

    # === CV DEGRADATION ===
    # Higher degradation acceptable due to concept drift in financial data
    max_cv_degradation: float = 0.15  # Max 15% degradation

    # === PREDICTION COLLAPSE ===
    # Allow some collapse but penalize heavily
    max_prediction_collapse: float = 0.15  # Max 15% collapse ratio

    # === RIDGE CONFIDENCE MODEL ===
    # R² for confidence is harder in FX - near zero is acceptable
    min_r2_score: float = 0.01  # Near zero for FX confidence estimation

    # === XGBOOST MOMENTUM MODEL ===
    # Momentum signals are inherently noisy in FX
    min_acceleration_accuracy: float = 0.55  # Minimum acceleration accuracy
    max_momentum_mae: float = 0.15  # Maximum momentum MAE

    # === RANDOM FOREST RISK MODEL ===
    max_drawdown_mae_bps: float = 100.0  # Maximum drawdown MAE (100 bps)
    max_streak_prob_mae: float = 0.15  # Maximum streak probability MAE

    # === DATA REQUIREMENTS ===
    min_data_size: int = 2000  # Minimum training samples
    max_class_imbalance: float = 0.70  # Maximum class imbalance (70%)

    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary representation."""
        return {
            "direction_accuracy_min": self.direction_accuracy_min,
            "balanced_accuracy_min": self.balanced_accuracy_min,
            "cv_std_max": self.cv_std_max,
            "bootstrap_ci_lower": self.bootstrap_ci_lower,
            "max_cv_degradation": self.max_cv_degradation,
            "max_prediction_collapse": self.max_prediction_collapse,
            "min_r2_score": self.min_r2_score,
            "min_acceleration_accuracy": self.min_acceleration_accuracy,
            "max_momentum_mae": self.max_momentum_mae,
            "max_drawdown_mae_bps": self.max_drawdown_mae_bps,
            "max_streak_prob_mae": self.max_streak_prob_mae,
            "min_data_size": self.min_data_size,
            "max_class_imbalance": self.max_class_imbalance,
        }


# =============================================================================
# TIERED VALIDATION CRITERIA BY LEVEL
# =============================================================================

def _get_criteria_by_level(level: ValidationLevel) -> FXValidationCriteria:
    """
    Get validation criteria based on deployment level.

    Args:
        level: Validation strictness level

    Returns:
        FXValidationCriteria configured for the specified level
        
    Note:
        Uses unified thresholds from src.training.unified_thresholds for
        key thresholds like bootstrap_ci_lower to ensure consistency
        with deployment_gate.py and other validation modules.
    """
    # Get unified thresholds based on level
    if level == ValidationLevel.PRODUCTION or level == ValidationLevel.CONSERVATIVE:
        unified = PRODUCTION_THRESHOLDS
    else:
        unified = FX_THRESHOLDS
    
    criteria_map = {
        ValidationLevel.DEVELOPMENT: FXValidationCriteria(
            direction_accuracy_min=0.51,
            balanced_accuracy_min=0.51,
            cv_std_max=0.10,
            bootstrap_ci_lower=0.50,  # Lenient for development
            max_cv_degradation=0.20,
            min_r2_score=0.0,
            max_prediction_collapse=0.20,
        ),
        ValidationLevel.STAGING: FXValidationCriteria(
            direction_accuracy_min=unified.min_direction_accuracy,  # 0.53
            balanced_accuracy_min=unified.min_direction_accuracy,  # 0.53
            cv_std_max=0.08,  # FX noise allows higher CV std
            bootstrap_ci_lower=unified.min_bootstrap_ci_lower,  # Uses unified 0.51
            max_cv_degradation=unified.max_cv_degradation,  # 0.15
            min_r2_score=0.01,  # Near zero for FX confidence
            max_prediction_collapse=0.15,
        ),
        ValidationLevel.PRODUCTION: FXValidationCriteria(
            direction_accuracy_min=unified.min_direction_accuracy,  # 0.55 for production
            balanced_accuracy_min=unified.min_direction_accuracy,
            cv_std_max=0.07,
            bootstrap_ci_lower=unified.min_bootstrap_ci_lower,  # Uses unified 0.55 for production
            max_cv_degradation=unified.max_cv_degradation,  # 0.10 for production
            min_r2_score=unified.min_r2_score,  # 0.45 for production
            max_prediction_collapse=0.12,
        ),
        ValidationLevel.CONSERVATIVE: FXValidationCriteria(
            direction_accuracy_min=0.55,
            balanced_accuracy_min=0.55,
            cv_std_max=0.05,
            bootstrap_ci_lower=0.53,
            max_cv_degradation=unified.max_cv_degradation,
            min_r2_score=unified.min_r2_score,
            max_prediction_collapse=0.10,
        ),
    }
    return criteria_map[level]


# =============================================================================
# PER-CURRENCY-PAIR CRITERIA ADJUSTMENTS
# =============================================================================

# Pair-specific criteria multipliers based on volatility characteristics
PAIR_CRITERIA_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # EUR_USD: Low volatility, tight spreads, mean-reverting
    # Can use slightly stricter criteria
    "EUR_USD": {
        "accuracy_mult": 1.02,  # 2% stricter
        "cv_std_mult": 0.95,   # 5% tighter CV std
    },
    # GBP_USD: Medium volatility, trending behavior
    # Use default criteria
    "GBP_USD": {
        "accuracy_mult": 1.0,
        "cv_std_mult": 1.0,
    },
    # USD_JPY: Low-medium volatility, safe haven behavior
    # Can use slightly stricter criteria
    "USD_JPY": {
        "accuracy_mult": 1.01,
        "cv_std_mult": 0.98,
    },
    # GBP_JPY: High volatility, wide spreads
    # Use more lenient criteria
    "GBP_JPY": {
        "accuracy_mult": 0.97,  # 3% more lenient
        "cv_std_mult": 1.15,   # 15% wider CV std allowed
    },
    # AUD_USD: Medium volatility, commodity-linked
    "AUD_USD": {
        "accuracy_mult": 1.0,
        "cv_std_mult": 1.0,
    },
    # USD_CHF: Low volatility, safe haven
    "USD_CHF": {
        "accuracy_mult": 1.01,
        "cv_std_mult": 0.98,
    },
    # USD_CAD: Medium volatility, oil-linked
    "USD_CAD": {
        "accuracy_mult": 0.99,
        "cv_std_mult": 1.02,
    },
    # NZD_USD: Medium volatility
    "NZD_USD": {
        "accuracy_mult": 0.99,
        "cv_std_mult": 1.02,
    },
    # EUR_GBP: Low volatility
    "EUR_GBP": {
        "accuracy_mult": 1.02,
        "cv_std_mult": 0.95,
    },
    # EUR_JPY: Medium-high volatility
    "EUR_JPY": {
        "accuracy_mult": 0.98,
        "cv_std_mult": 1.05,
    },
}

# Market regime adjustments
REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # Normal market conditions
    "normal": {
        "accuracy_mult": 1.0,
        "cv_std_mult": 1.0,
    },
    # High volatility regime - more lenient
    "high_volatility": {
        "accuracy_mult": 0.97,
        "cv_std_mult": 1.15,
    },
    # Low volatility regime - stricter
    "low_volatility": {
        "accuracy_mult": 1.03,
        "cv_std_mult": 0.90,
    },
    # Crisis/turbulent regime - most lenient
    "crisis": {
        "accuracy_mult": 0.95,
        "cv_std_mult": 1.25,
    },
}


def get_validation_criteria(
    pair: str = "default",
    regime: str = "normal",
    level: ValidationLevel = ValidationLevel.STAGING,
) -> FXValidationCriteria:
    """
    Get validation criteria adjusted for currency pair and market regime.

    Args:
        pair: Currency pair identifier (e.g., "EUR_USD", "GBP_JPY")
        regime: Market regime ("normal", "high_volatility", "low_volatility", "crisis")
        level: Validation strictness level

    Returns:
        FXValidationCriteria adjusted for the specified parameters

    Example:
        >>> criteria = get_validation_criteria("EUR_USD", regime="normal")
        >>> print(criteria.direction_accuracy_min)
        0.5406  # 0.53 * 1.02 (EUR_USD adjustment)
    """
    # Get base criteria by level
    base_criteria = _get_criteria_by_level(level)

    # Get pair-specific adjustments
    pair_adj = PAIR_CRITERIA_ADJUSTMENTS.get(pair, {"accuracy_mult": 1.0, "cv_std_mult": 1.0})

    # Get regime adjustments
    regime_adj = REGIME_ADJUSTMENTS.get(regime, {"accuracy_mult": 1.0, "cv_std_mult": 1.0})

    # Combine adjustments
    accuracy_mult = pair_adj["accuracy_mult"] * regime_adj["accuracy_mult"]
    cv_std_mult = pair_adj["cv_std_mult"] * regime_adj["cv_std_mult"]

    # Apply adjustments to create new criteria
    adjusted_criteria = FXValidationCriteria(
        direction_accuracy_min=base_criteria.direction_accuracy_min * accuracy_mult,
        balanced_accuracy_min=base_criteria.balanced_accuracy_min * accuracy_mult,
        cv_std_max=base_criteria.cv_std_max * cv_std_mult,
        bootstrap_ci_lower=min(1.0, base_criteria.bootstrap_ci_lower * accuracy_mult),
        max_cv_degradation=base_criteria.max_cv_degradation / accuracy_mult,
        max_prediction_collapse=base_criteria.max_prediction_collapse / accuracy_mult,
        min_r2_score=base_criteria.min_r2_score,
        min_acceleration_accuracy=base_criteria.min_acceleration_accuracy * accuracy_mult,
        max_momentum_mae=base_criteria.max_momentum_mae / accuracy_mult,
        max_drawdown_mae_bps=base_criteria.max_drawdown_mae_bps,
        max_streak_prob_mae=base_criteria.max_streak_prob_mae,
        min_data_size=base_criteria.min_data_size,
        max_class_imbalance=base_criteria.max_class_imbalance,
    )

    logger.debug(
        f"Validation criteria for {pair} ({regime}, {level.value}): "
        f"accuracy_min={adjusted_criteria.direction_accuracy_min:.4f}, "
        f"cv_std_max={adjusted_criteria.cv_std_max:.4f}"
    )

    return adjusted_criteria


def validate_model_performance(
    metrics: Dict[str, Any],
    criteria: FXValidationCriteria,
    check_collapse: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate model performance against FX-specific criteria.

    Args:
        metrics: Dictionary containing model performance metrics:
            - val_accuracy: Validation accuracy
            - val_balanced_accuracy: Balanced accuracy (optional, defaults to val_accuracy)
            - cv_std: Cross-validation standard deviation (optional)
            - cv_degradation: CV degradation from training (optional)
            - bootstrap_ci_lower: Bootstrap CI lower bound (optional)
            - collapse_ratio: Prediction collapse ratio (optional)
            - r2_score: R² score for confidence model (optional)
            - acceleration_accuracy: Momentum acceleration accuracy (optional)
            - momentum_mae: Momentum MAE (optional)
            - drawdown_mae_bps: Drawdown MAE in bps (optional)
            - streak_prob_mae: Streak probability MAE (optional)
        criteria: FXValidationCriteria to validate against
        check_collapse: Whether to check prediction collapse (default: True)

    Returns:
        Tuple of (passed: bool, failure_reasons: List[str])

    Example:
        >>> metrics = {"val_accuracy": 0.54, "cv_std": 0.06}
        >>> criteria = get_validation_criteria("EUR_USD")
        >>> passed, failures = validate_model_performance(metrics, criteria)
        >>> if not passed:
        ...     print(f"Validation failed: {failures}")
    """
    failure_reasons: List[str] = []
    passed = True

    # === DIRECTION ACCURACY CHECK ===
    val_accuracy = metrics.get("val_accuracy", 0)
    if val_accuracy < criteria.direction_accuracy_min:
        passed = False
        failure_reasons.append(
            f"Direction accuracy {val_accuracy:.4f} < {criteria.direction_accuracy_min:.4f}"
        )

    # === BALANCED ACCURACY CHECK ===
    balanced_accuracy = metrics.get("val_balanced_accuracy", val_accuracy)
    if balanced_accuracy < criteria.balanced_accuracy_min:
        passed = False
        failure_reasons.append(
            f"Balanced accuracy {balanced_accuracy:.4f} < {criteria.balanced_accuracy_min:.4f}"
        )

    # === CV STABILITY CHECK ===
    if "cv_std" in metrics:
        cv_std = metrics["cv_std"]
        if cv_std > criteria.cv_std_max:
            passed = False
            failure_reasons.append(
                f"CV std {cv_std:.4f} > {criteria.cv_std_max:.4f}"
            )

    # === CV DEGRADATION CHECK ===
    if "cv_degradation" in metrics:
        cv_degradation = metrics["cv_degradation"]
        if cv_degradation > criteria.max_cv_degradation:
            passed = False
            failure_reasons.append(
                f"CV degradation {cv_degradation:.4f} > {criteria.max_cv_degradation:.4f}"
            )

    # === BOOTSTRAP CI CHECK ===
    if "bootstrap_ci_lower" in metrics:
        ci_lower = metrics["bootstrap_ci_lower"]
        if ci_lower < criteria.bootstrap_ci_lower:
            passed = False
            failure_reasons.append(
                f"Bootstrap CI lower {ci_lower:.4f} < {criteria.bootstrap_ci_lower:.4f}"
            )

    # === PREDICTION COLLAPSE CHECK ===
    if check_collapse and "collapse_ratio" in metrics:
        collapse_ratio = metrics["collapse_ratio"]
        if collapse_ratio > criteria.max_prediction_collapse:
            passed = False
            failure_reasons.append(
                f"Prediction collapse {collapse_ratio:.4f} > {criteria.max_prediction_collapse:.4f}"
            )

    # === R² SCORE CHECK ===
    if "r2_score" in metrics:
        r2_score = metrics["r2_score"]
        if r2_score < criteria.min_r2_score:
            passed = False
            failure_reasons.append(
                f"R² score {r2_score:.4f} < {criteria.min_r2_score:.4f}"
            )

    # === ACCELERATION ACCURACY CHECK ===
    if "acceleration_accuracy" in metrics:
        accel_acc = metrics["acceleration_accuracy"]
        if accel_acc < criteria.min_acceleration_accuracy:
            passed = False
            failure_reasons.append(
                f"Acceleration accuracy {accel_acc:.4f} < {criteria.min_acceleration_accuracy:.4f}"
            )

    # === MOMENTUM MAE CHECK ===
    if "momentum_mae" in metrics:
        momentum_mae = metrics["momentum_mae"]
        if momentum_mae > criteria.max_momentum_mae:
            passed = False
            failure_reasons.append(
                f"Momentum MAE {momentum_mae:.4f} > {criteria.max_momentum_mae:.4f}"
            )

    # === DRAWDOWN MAE CHECK ===
    if "drawdown_mae_bps" in metrics:
        drawdown_mae = metrics["drawdown_mae_bps"]
        if drawdown_mae > criteria.max_drawdown_mae_bps:
            passed = False
            failure_reasons.append(
                f"Drawdown MAE {drawdown_mae:.2f} bps > {criteria.max_drawdown_mae_bps:.2f} bps"
            )

    # === STREAK PROBABILITY MAE CHECK ===
    if "streak_prob_mae" in metrics:
        streak_mae = metrics["streak_prob_mae"]
        if streak_mae > criteria.max_streak_prob_mae:
            passed = False
            failure_reasons.append(
                f"Streak prob MAE {streak_mae:.4f} > {criteria.max_streak_prob_mae:.4f}"
            )

    if passed:
        logger.info("Model validation PASSED all FX criteria checks")
    else:
        logger.warning(f"Model validation FAILED: {failure_reasons}")

    return passed, failure_reasons


def compute_collapse_ratio(predictions: List[float], threshold: float = 0.05) -> float:
    """
    Compute the prediction collapse ratio.

    Collapse occurs when predictions cluster too closely around a single value,
    indicating the model has learned to output constant predictions.

    Args:
        predictions: List of prediction probabilities
        threshold: Threshold for considering predictions as collapsed (default: 0.05)

    Returns:
        Collapse ratio (0.0 = no collapse, 1.0 = complete collapse)

    Example:
        >>> preds = [0.51, 0.52, 0.50, 0.51, 0.52]
        >>> collapse = compute_collapse_ratio(preds)
        >>> print(f"Collapse ratio: {collapse:.2%}")
    """
    if not predictions:
        return 0.0

    import numpy as np
    preds = np.array(predictions)

    # Check if predictions cluster around mean
    pred_std = np.std(preds)
    pred_range = np.max(preds) - np.min(preds)

    # Collapse if std is very low or range is very narrow
    if pred_std < threshold or pred_range < 2 * threshold:
        return min(1.0, threshold / max(pred_std, 1e-6))

    return 0.0


def compute_cv_degradation(
    train_accuracy: float,
    cv_mean_accuracy: float,
) -> float:
    """
    Compute CV degradation from training to cross-validation.

    Args:
        train_accuracy: Training set accuracy
        cv_mean_accuracy: Mean cross-validation accuracy

    Returns:
        Degradation ratio (positive = degradation, negative = improvement)

    Example:
        >>> degradation = compute_cv_degradation(0.70, 0.55)
        >>> print(f"CV degradation: {degradation:.2%}")
    """
    if train_accuracy == 0:
        return 0.0

    return (train_accuracy - cv_mean_accuracy) / train_accuracy


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate(
    val_accuracy: float,
    cv_std: Optional[float] = None,
    pair: str = "default",
    regime: str = "normal",
) -> Tuple[bool, List[str]]:
    """
    Quick validation with minimal required metrics.

    Args:
        val_accuracy: Validation accuracy
        cv_std: Cross-validation standard deviation (optional)
        pair: Currency pair for criteria adjustment
        regime: Market regime

    Returns:
        Tuple of (passed: bool, failure_reasons: List[str])

    Example:
        >>> passed, failures = quick_validate(0.54, cv_std=0.06, pair="EUR_USD")
    """
    metrics = {"val_accuracy": val_accuracy}
    if cv_std is not None:
        metrics["cv_std"] = cv_std

    criteria = get_validation_criteria(pair, regime)
    return validate_model_performance(metrics, criteria)


def get_threshold_summary(level: ValidationLevel = ValidationLevel.STAGING) -> str:
    """
    Get a human-readable summary of validation thresholds.

    Args:
        level: Validation level to summarize

    Returns:
        Formatted string with threshold summary
    """
    criteria = _get_criteria_by_level(level)

    summary = f"""
FX Validation Criteria Summary ({level.value.upper()})
================================================

Direction Model:
  - Min Accuracy:          {criteria.direction_accuracy_min:.1%}
  - Min Balanced Accuracy: {criteria.balanced_accuracy_min:.1%}
  - Max CV Std:            {criteria.cv_std_max:.1%}
  - Min Bootstrap CI:      {criteria.bootstrap_ci_lower:.1%}

Stability:
  - Max CV Degradation:    {criteria.max_cv_degradation:.1%}
  - Max Prediction Collapse: {criteria.max_prediction_collapse:.1%}

Momentum Model:
  - Min Acceleration Acc:  {criteria.min_acceleration_accuracy:.1%}
  - Max Momentum MAE:      {criteria.max_momentum_mae:.2f}

Confidence Model:
  - Min R² Score:          {criteria.min_r2_score:.2f}

Risk Model:
  - Max Drawdown MAE:      {criteria.max_drawdown_mae_bps:.0f} bps
  - Max Streak Prob MAE:   {criteria.max_streak_prob_mae:.2f}

Data Requirements:
  - Min Data Size:         {criteria.min_data_size:,} samples
  - Max Class Imbalance:   {criteria.max_class_imbalance:.0%}
"""
    return summary


if __name__ == "__main__":
    # Example usage
    print(get_threshold_summary(ValidationLevel.STAGING))

    # Quick validation example
    metrics = {
        "val_accuracy": 0.54,
        "cv_std": 0.06,
        "cv_degradation": 0.12,
    }

    criteria = get_validation_criteria("EUR_USD", "normal")
    passed, failures = validate_model_performance(metrics, criteria)

    print(f"\nValidation: {'PASSED' if passed else 'FAILED'}")
    if failures:
        for f in failures:
            print(f"  - {f}")
