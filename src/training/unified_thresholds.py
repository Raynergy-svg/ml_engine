"""
Unified Validation Thresholds for Financial ML Ensemble.

This module centralizes all validation thresholds to ensure consistency
across deployment_gate.py, fx_validation_criteria.py, and other validation modules.

Usage:
    from src.training.unified_thresholds import (
        FX_THRESHOLDS, 
        get_threshold,
        validate_threshold_config
    )
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model type enumeration for threshold selection."""
    FX_DIRECTION = "fx_direction"
    FX_REGRESSION = "fx_regression"
    ENSEMBLE = "ensemble"
    META_LABELER = "meta_labeler"


@dataclass
class ValidationThresholds:
    """Container for validation thresholds."""
    
    # Directional accuracy thresholds
    min_direction_accuracy: float = 0.53  # 53% minimum (FX-calibrated)
    target_direction_accuracy: float = 0.65  # 65% target
    warmstart_baseline_min: float = 0.60  # 60% minimum to preserve
    
    # Cross-validation thresholds
    max_cv_degradation: float = 0.15  # Max 15% CV degradation
    max_cv_fold_variance: float = 0.03  # Max 3% variance across folds
    min_cv_mean_accuracy: float = 0.51  # Minimum CV mean
    
    # Bootstrap thresholds
    min_bootstrap_ci_lower: float = 0.51  # Minimum CI lower bound (FX-calibrated)
    min_bootstrap_ci_width: float = 0.15  # Maximum CI width (narrower = better)
    bootstrap_n_samples: int = 1000  # Number of bootstrap samples
    
    # Regression thresholds
    min_r2_score: float = 0.40  # Minimum R² for regression
    min_direction_alignment: float = 0.60  # Direction alignment for regression
    
    # Ensemble thresholds
    max_brier_score: float = 0.20  # Maximum Brier score (lower = better)
    max_log_loss: float = 0.65  # Maximum log loss
    
    # Performance thresholds
    max_inference_latency_ms: float = 500.0  # Maximum inference latency (ms)
    
    # Statistical significance
    min_p_value: float = 0.05  # p-value threshold for significance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'min_direction_accuracy': self.min_direction_accuracy,
            'target_direction_accuracy': self.target_direction_accuracy,
            'warmstart_baseline_min': self.warmstart_baseline_min,
            'max_cv_degradation': self.max_cv_degradation,
            'max_cv_fold_variance': self.max_cv_fold_variance,
            'min_cv_mean_accuracy': self.min_cv_mean_accuracy,
            'min_bootstrap_ci_lower': self.min_bootstrap_ci_lower,
            'min_bootstrap_ci_width': self.min_bootstrap_ci_width,
            'bootstrap_n_samples': self.bootstrap_n_samples,
            'min_r2_score': self.min_r2_score,
            'min_direction_alignment': self.min_direction_alignment,
            'max_brier_score': self.max_brier_score,
            'max_log_loss': self.max_log_loss,
            'max_inference_latency_ms': self.max_inference_latency_ms,
            'min_p_value': self.min_p_value,
        }


# Default FX-specific thresholds (calibrated for financial data)
FX_THRESHOLDS = ValidationThresholds(
    min_direction_accuracy=0.53,
    target_direction_accuracy=0.65,
    warmstart_baseline_min=0.60,
    max_cv_degradation=0.15,
    max_cv_fold_variance=0.03,
    min_cv_mean_accuracy=0.51,
    min_bootstrap_ci_lower=0.51,  # FX-calibrated (was 0.60 in deployment_gate)
    min_bootstrap_ci_width=0.15,
    bootstrap_n_samples=1000,
    min_r2_score=0.40,
    min_direction_alignment=0.60,
    max_brier_score=0.20,
    max_log_loss=0.65,
    max_inference_latency_ms=500.0,
    min_p_value=0.05,
)

# Stricter production thresholds
PRODUCTION_THRESHOLDS = ValidationThresholds(
    min_direction_accuracy=0.55,
    target_direction_accuracy=0.65,
    warmstart_baseline_min=0.62,
    max_cv_degradation=0.10,
    max_cv_fold_variance=0.02,
    min_cv_mean_accuracy=0.53,
    min_bootstrap_ci_lower=0.55,
    min_bootstrap_ci_width=0.12,
    bootstrap_n_samples=2000,
    min_r2_score=0.45,
    min_direction_alignment=0.65,
    max_brier_score=0.18,
    max_log_loss=0.60,
    max_inference_latency_ms=300.0,
    min_p_value=0.05,
)


# Threshold registry
_THRESHOLD_REGISTRY: Dict[str, ValidationThresholds] = {
    "fx_default": FX_THRESHOLDS,
    "production": PRODUCTION_THRESHOLDS,
}


def get_threshold(name: str = "fx_default") -> ValidationThresholds:
    """Get validation thresholds by name.
    
    Args:
        name: Threshold preset name ("fx_default", "production")
        
    Returns:
        ValidationThresholds instance
        
    Raises:
        KeyError: If threshold name not found
    """
    if name not in _THRESHOLD_REGISTRY:
        available = list(_THRESHOLD_REGISTRY.keys())
        raise KeyError(f"Threshold '{name}' not found. Available: {available}")
    return _THRESHOLD_REGISTRY[name]


def register_threshold(name: str, thresholds: ValidationThresholds) -> None:
    """Register custom validation thresholds.
    
    Args:
        name: Name for the threshold preset
        thresholds: ValidationThresholds instance
    """
    _THRESHOLD_REGISTRY[name] = thresholds
    logger.info(f"Registered validation thresholds: {name}")


def validate_threshold_config(config: Dict[str, Any]) -> bool:
    """Validate threshold configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        'min_direction_accuracy',
        'max_cv_degradation',
        'min_bootstrap_ci_lower',
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required threshold: {key}")
        if not isinstance(config[key], (int, float)):
            raise ValueError(f"Threshold {key} must be numeric, got {type(config[key])}")
        if config[key] < 0 or config[key] > 1:
            raise ValueError(f"Threshold {key} must be between 0 and 1, got {config[key]}")
    
    return True


def compare_with_thresholds(
    metrics: Dict[str, float],
    thresholds: Optional[ValidationThresholds] = None
) -> Dict[str, Dict[str, Any]]:
    """Compare metrics against thresholds.
    
    Args:
        metrics: Dictionary of metric name → value
        thresholds: Thresholds to use (default: FX_THRESHOLDS)
        
    Returns:
        Dictionary of comparison results with 'passed', 'value', 'threshold' keys
    """
    if thresholds is None:
        thresholds = FX_THRESHOLDS
    
    threshold_dict = thresholds.to_dict()
    results = {}
    
    for metric_name, value in metrics.items():
        if metric_name not in threshold_dict:
            logger.warning(f"Unknown metric: {metric_name}")
            continue
        
        threshold = threshold_dict[metric_name]
        
        # Determine comparison direction
        if metric_name.startswith('min_') or metric_name == 'target_direction_accuracy':
            passed = value >= threshold
        else:  # max_* metrics
            passed = value <= threshold
        
        results[metric_name] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'direction': 'min' if metric_name.startswith('min_') else 'max'
        }
    
    return results
