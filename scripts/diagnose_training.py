#!/usr/bin/env python3
"""
Comprehensive Training Diagnostic Script for Buddy FX Model.

This script performs automated diagnostics to identify why training
is producing poor results. It checks:

1. DATA QUALITY
   - Missing values, NaN/Inf
   - Feature distributions (outliers, constant features)
   - Label distribution and balance
   - Autocorrelation (data leakage check)

2. LABEL QUALITY  
   - Direction label predictability (XGBoost baseline)
   - Label noise estimation
   - Class balance
   - Triple-barrier vs next-bar comparison

3. FEATURE QUALITY
   - Feature-target correlation
   - Multicollinearity
   - Feature importance ranking
   - Redundant features

4. MODEL DIAGNOSTICS
   - Gradient flow check
   - Loss curve analysis
   - Overfitting detection
   - Learning rate sensitivity

5. RECOMMENDATIONS
   - Actionable fixes based on findings

Usage:
    python diagnose_training.py --csv market_data/USD_JPY_M5.csv
    python diagnose_training.py --csv market_data/USD_JPY_M5.csv --output diagnostic_report.json
    python diagnose_training.py --help

Author: Training Debug Session
Date: 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Diagnostic Result Classes
# =============================================================================

@dataclass
class DataQualityReport:
    """Results of data quality checks."""
    n_samples: int = 0
    n_features: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    nan_columns: List[str] = field(default_factory=list)
    inf_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    high_outlier_columns: List[str] = field(default_factory=list)
    low_variance_columns: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    severity: str = "ok"  # ok, warning, critical


@dataclass
class LabelQualityReport:
    """Results of label quality checks."""
    n_positive: int = 0
    n_negative: int = 0
    balance_ratio: float = 0.5
    xgboost_accuracy: float = 0.5
    xgboost_auc: float = 0.5
    random_baseline: float = 0.5
    is_learnable: bool = False
    label_noise_estimate: float = 0.0
    autocorrelation: float = 0.0
    triple_barrier_stats: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    severity: str = "ok"


@dataclass
class FeatureQualityReport:
    """Results of feature quality checks."""
    n_features: int = 0
    top_correlated_features: List[Tuple[str, float]] = field(default_factory=list)
    multicollinear_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    redundant_features: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    recommended_features: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    severity: str = "ok"


@dataclass
class ModelDiagnosticReport:
    """Results of model diagnostics."""
    gradient_health: str = "unknown"
    loss_trend: str = "unknown"  # decreasing, flat, increasing, oscillating
    overfitting_detected: bool = False
    train_val_gap: float = 0.0
    best_epoch: int = 0
    convergence_speed: str = "unknown"  # fast, normal, slow, not_converging
    issues: List[str] = field(default_factory=list)
    severity: str = "ok"


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    timestamp: str = ""
    csv_path: str = ""
    runtime_seconds: float = 0.0
    data_quality: DataQualityReport = field(default_factory=DataQualityReport)
    label_quality: LabelQualityReport = field(default_factory=LabelQualityReport)
    feature_quality: FeatureQualityReport = field(default_factory=FeatureQualityReport)
    model_diagnostic: ModelDiagnosticReport = field(default_factory=ModelDiagnosticReport)
    recommendations: List[str] = field(default_factory=list)
    overall_severity: str = "ok"


# =============================================================================
# Data Quality Checks
# =============================================================================

def check_data_quality(df: pd.DataFrame, feature_cols: List[str]) -> DataQualityReport:
    """
    Check data quality for training.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
    
    Returns:
        DataQualityReport with findings
    """
    report = DataQualityReport()
    report.n_samples = len(df)
    report.n_features = len(feature_cols)
    
    # Check for missing values
    for col in feature_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                report.missing_values[col] = int(n_missing)
    
    if report.missing_values:
        total_missing = sum(report.missing_values.values())
        pct_missing = total_missing / (report.n_samples * report.n_features) * 100
        if pct_missing > 5:
            report.issues.append(f"High missing values: {pct_missing:.1f}% of data")
            report.severity = "warning"
    
    # Check for NaN/Inf
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if numeric_df[col].isna().any():
            report.nan_columns.append(col)
        if np.isinf(numeric_df[col]).any():
            report.inf_columns.append(col)
    
    if report.nan_columns or report.inf_columns:
        report.issues.append(f"Found NaN in {len(report.nan_columns)} cols, Inf in {len(report.inf_columns)} cols")
        report.severity = "critical"
    
    # Check for constant columns
    for col in numeric_df.columns:
        if numeric_df[col].std() < 1e-10:
            report.constant_columns.append(col)
    
    if report.constant_columns:
        report.issues.append(f"Found {len(report.constant_columns)} constant columns (no variance)")
        report.severity = max(report.severity, "warning", key=lambda x: ["ok", "warning", "critical"].index(x))
    
    # Check for extreme outliers (>10 std from mean)
    for col in numeric_df.columns:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        if std > 0:
            outlier_pct = ((numeric_df[col] - mean).abs() > 10 * std).mean()
            if outlier_pct > 0.01:  # >1% outliers
                report.high_outlier_columns.append(col)
    
    if report.high_outlier_columns:
        report.issues.append(f"Found {len(report.high_outlier_columns)} columns with extreme outliers")
    
    # Check for low variance columns
    for col in numeric_df.columns:
        variance = numeric_df[col].var()
        if 0 < variance < 1e-6:
            report.low_variance_columns.append(col)
    
    if report.low_variance_columns:
        report.issues.append(f"Found {len(report.low_variance_columns)} low-variance columns")
    
    if not report.issues:
        report.issues.append("Data quality checks passed")
    
    return report


# =============================================================================
# Label Quality Checks
# =============================================================================

def check_label_quality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ohlc_df: Optional[pd.DataFrame] = None,
    instrument: str = "USD_JPY",
) -> LabelQualityReport:
    """
    Check label quality and predictability.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        ohlc_df: Optional OHLC DataFrame for triple-barrier comparison
        instrument: Trading pair
    
    Returns:
        LabelQualityReport with findings
    """
    report = LabelQualityReport()
    
    # Class balance
    y_train_bin = (y_train >= 0.5).astype(int).ravel()

    report.n_positive = int((y_train_bin == 1).sum())
    report.n_negative = int((y_train_bin == 0).sum())
    report.balance_ratio = report.n_positive / max(report.n_positive + report.n_negative, 1)
    report.random_baseline = max(report.balance_ratio, 1 - report.balance_ratio)
    
    # Check class imbalance
    if report.balance_ratio < 0.4 or report.balance_ratio > 0.6:
        report.issues.append(f"Class imbalance: {report.balance_ratio:.1%} positive rate")
        report.severity = "warning"
    
    # XGBoost baseline
    try:
        from training_improvements import train_xgboost_baseline
        
        _, xgb_metrics = train_xgboost_baseline(
            X_train, y_train,
            X_val, y_val,
            task="classification",
            n_estimators=100,  # Faster for diagnostics
            max_depth=4,
            verbose=False,
        )
        
        report.xgboost_accuracy = xgb_metrics.get("val_accuracy", 0.5)
        report.xgboost_auc = xgb_metrics.get("val_auc", 0.5)
        
        # Determine if labels are learnable
        if report.xgboost_accuracy > report.random_baseline + 0.03:
            report.is_learnable = True
        else:
            report.is_learnable = False
            report.issues.append(
                f"Labels not learnable: XGBoost {report.xgboost_accuracy:.1%} vs random {report.random_baseline:.1%}"
            )
            report.severity = "critical"
            
    except Exception as e:
        logger.warning(f"XGBoost baseline failed: {e}")
        report.issues.append(f"XGBoost baseline failed: {e}")
    
    # Label autocorrelation (check for data leakage)
    if len(y_train_bin) > 100:
        autocorr = np.corrcoef(y_train_bin[:-1], y_train_bin[1:])[0, 1]
        report.autocorrelation = float(autocorr) if np.isfinite(autocorr) else 0.0
        
        if abs(report.autocorrelation) > 0.3:
            report.issues.append(f"High label autocorrelation: {report.autocorrelation:.2f}")
            report.severity = max(report.severity, "warning", key=lambda x: ["ok", "warning", "critical"].index(x))
    
    # Estimate label noise
    # Use XGBoost training accuracy as proxy - if train acc is low, labels may be noisy
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Fit on train, predict on train
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        
        # Flatten if needed
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) == 3 else X_train
        
        rf.fit(X_train_flat, y_train_bin)
        train_pred = rf.predict(X_train_flat)
        train_acc = (train_pred == y_train_bin).mean()
        
        # If even training accuracy is low, labels are noisy
        report.label_noise_estimate = max(0, 1 - train_acc * 2)  # Rough estimate
        
        if train_acc < 0.6:
            report.issues.append(f"High label noise estimated: train accuracy only {train_acc:.1%}")
            
    except Exception as e:
        logger.debug(f"Label noise estimation failed: {e}")
    
    # Triple-barrier comparison (if OHLC data available)
    if ohlc_df is not None and len(ohlc_df) > 100:
        try:
            from training_improvements import (
                TripleBarrierConfig,
                compute_triple_barrier_direction_labels,
            )
            
            tb_config = TripleBarrierConfig(
                stop_loss_pips=30.0,
                take_profit_pips=60.0,
                max_horizon_candles=30,
            )
            
            tb_labels, tb_weights, tb_stats = compute_triple_barrier_direction_labels(
                ohlc_df,
                instrument=instrument,
                config=tb_config,
                seq_len=60,
                label_stride=5,  # Faster for diagnostics
                verbose=False,
            )
            
            report.triple_barrier_stats = tb_stats
            
            # Check if triple-barrier produces clearer signals
            tb_clear = (tb_weights == 1.0).sum()
            tb_unclear = ((tb_weights > 0) & (tb_weights < 1.0)).sum()
            
            if tb_clear > 0:
                tb_pos_rate = tb_labels[tb_weights == 1.0].mean()
                report.triple_barrier_stats["clear_positive_rate"] = float(tb_pos_rate)
                report.triple_barrier_stats["n_clear"] = int(tb_clear)
                report.triple_barrier_stats["n_unclear"] = int(tb_unclear)
                
        except Exception as e:
            logger.debug(f"Triple-barrier comparison failed: {e}")
    
    if not report.issues:
        report.issues.append("Label quality checks passed")
    
    return report


# =============================================================================
# Feature Quality Checks
# =============================================================================

def check_feature_quality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> FeatureQualityReport:
    """
    Check feature quality and importance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
    
    Returns:
        FeatureQualityReport with findings
    """
    report = FeatureQualityReport()
    report.n_features = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    
    # Flatten sequences if needed
    if len(X_train.shape) == 3:
        # Use last timestep for correlation analysis
        X_flat = X_train[:, -1, :]
    else:
        X_flat = X_train
    
    y_flat = y_train.ravel()
    
    # Feature-target correlation
    correlations = []
    for i in range(min(X_flat.shape[1], len(feature_names))):
        try:
            corr = np.corrcoef(X_flat[:, i], y_flat)[0, 1]
            if np.isfinite(corr):
                correlations.append((feature_names[i], abs(corr)))
        except Exception:
            pass
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    report.top_correlated_features = correlations[:20]
    
    # Check if any features have meaningful correlation
    max_corr = correlations[0][1] if correlations else 0
    if max_corr < 0.05:
        report.issues.append(f"No features have significant correlation with target (max: {max_corr:.3f})")
        report.severity = "critical"
    elif max_corr < 0.1:
        report.issues.append(f"Weak feature-target correlations (max: {max_corr:.3f})")
        report.severity = "warning"
    
    # Multicollinearity check
    try:
        corr_matrix = np.corrcoef(X_flat.T)
        n_feats = corr_matrix.shape[0]
        
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                if abs(corr_matrix[i, j]) > 0.95:
                    if i < len(feature_names) and j < len(feature_names):
                        report.multicollinear_pairs.append(
                            (feature_names[i], feature_names[j], float(corr_matrix[i, j]))
                        )
        
        if len(report.multicollinear_pairs) > 10:
            report.issues.append(f"High multicollinearity: {len(report.multicollinear_pairs)} highly correlated pairs")
            
    except Exception as e:
        logger.debug(f"Multicollinearity check failed: {e}")
    
    # Feature importance using Random Forest
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        y_bin = (y_flat >= 0.5).astype(int)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_flat, y_bin)
        
        importances = rf.feature_importances_
        for i, imp in enumerate(importances):
            if i < len(feature_names):
                report.feature_importance[feature_names[i]] = float(imp)
        
        # Sort by importance
        sorted_features = sorted(report.feature_importance.items(), key=lambda x: x[1], reverse=True)
        report.recommended_features = [f[0] for f in sorted_features[:40]]
        
        # Check if importance is concentrated
        top_10_importance = sum(f[1] for f in sorted_features[:10])
        if top_10_importance > 0.8:
            report.issues.append(f"Feature importance concentrated: top 10 features = {top_10_importance:.1%}")
            
    except Exception as e:
        logger.debug(f"Feature importance calculation failed: {e}")
    
    # Identify redundant features (low importance + high correlation with important feature)
    if report.feature_importance and report.multicollinear_pairs:
        important_features = set(report.recommended_features[:20])
        for f1, f2, corr in report.multicollinear_pairs:
            if f1 in important_features and f2 not in important_features:
                report.redundant_features.append(f2)
            elif f2 in important_features and f1 not in important_features:
                report.redundant_features.append(f1)
    
    if report.redundant_features:
        report.issues.append(f"Found {len(report.redundant_features)} redundant features")
    
    if not report.issues:
        report.issues.append("Feature quality checks passed")
    
    return report


# =============================================================================
# Model Diagnostics
# =============================================================================

def check_model_diagnostics(
    tensorboard_dir: Optional[str] = None,
    training_log: Optional[str] = None,
) -> ModelDiagnosticReport:
    """
    Analyze model training diagnostics from logs.
    
    Args:
        tensorboard_dir: Path to TensorBoard logs
        training_log: Path to training log file
    
    Returns:
        ModelDiagnosticReport with findings
    """
    report = ModelDiagnosticReport()
    
    # Try to read TensorBoard logs
    if tensorboard_dir and Path(tensorboard_dir).exists():
        try:
            from tensorflow.python.summary.summary_iterator import summary_iterator
            
            # Find event files
            event_files = list(Path(tensorboard_dir).glob("**/events.out.tfevents.*"))
            
            if event_files:
                train_losses = []
                val_losses = []
                
                for event_file in event_files[:3]:  # Limit to recent files
                    try:
                        for event in summary_iterator(str(event_file)):
                            for value in event.summary.value:
                                if "loss" in value.tag.lower():
                                    if "val" in value.tag.lower():
                                        val_losses.append(value.simple_value)
                                    else:
                                        train_losses.append(value.simple_value)
                    except Exception:
                        pass
                
                if train_losses and val_losses:
                    # Analyze loss trend
                    if len(val_losses) > 10:
                        early_val = np.mean(val_losses[:5])
                        late_val = np.mean(val_losses[-5:])
                        
                        if late_val < early_val * 0.9:
                            report.loss_trend = "decreasing"
                        elif late_val > early_val * 1.1:
                            report.loss_trend = "increasing"
                            report.issues.append("Validation loss increasing - possible divergence")
                            report.severity = "warning"
                        else:
                            report.loss_trend = "flat"
                            report.issues.append("Validation loss flat - model not learning")
                            report.severity = "warning"
                    
                    # Check overfitting
                    if train_losses and val_losses:
                        train_final = np.mean(train_losses[-5:])
                        val_final = np.mean(val_losses[-5:])
                        report.train_val_gap = float(val_final - train_final)
                        
                        if report.train_val_gap > 0.1:
                            report.overfitting_detected = True
                            report.issues.append(f"Overfitting detected: train-val gap = {report.train_val_gap:.3f}")
                            report.severity = "warning"
                            
        except Exception as e:
            logger.debug(f"TensorBoard analysis failed: {e}")
    
    # Try to analyze training log
    if training_log and Path(training_log).exists():
        try:
            with open(training_log, "r") as f:
                log_content = f.read()
            
            # Look for common issues
            if "nan" in log_content.lower() or "NaN" in log_content:
                report.issues.append("NaN values detected in training log")
                report.gradient_health = "unstable"
                report.severity = "critical"
            
            if "gradient" in log_content.lower() and "explod" in log_content.lower():
                report.issues.append("Exploding gradients detected")
                report.gradient_health = "exploding"
                report.severity = "critical"
            
            if "early stopping" in log_content.lower():
                # Try to extract best epoch
                import re
                match = re.search(r"epoch[:\s]+(\d+)", log_content.lower())
                if match:
                    report.best_epoch = int(match.group(1))
                    
        except Exception as e:
            logger.debug(f"Training log analysis failed: {e}")
    
    if not report.issues:
        report.issues.append("Model diagnostics: no issues detected (limited data)")
    
    return report


# =============================================================================
# Generate Recommendations
# =============================================================================

def generate_recommendations(report: DiagnosticReport) -> List[str]:
    """
    Generate actionable recommendations based on diagnostic findings.
    
    Args:
        report: Complete diagnostic report
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Data quality recommendations
    if report.data_quality.nan_columns or report.data_quality.inf_columns:
        recommendations.append(
            "🔴 CRITICAL: Clean NaN/Inf values in data. Use df.fillna() or drop affected rows."
        )
    
    if report.data_quality.constant_columns:
        recommendations.append(
            f"⚠️ Remove {len(report.data_quality.constant_columns)} constant columns: "
            f"{report.data_quality.constant_columns[:5]}"
        )
    
    # Label quality recommendations
    if not report.label_quality.is_learnable:
        recommendations.append(
            "🔴 CRITICAL: Labels are not learnable from features. Consider:\n"
            "   1. Use triple-barrier labeling instead of next-bar direction\n"
            "   2. Increase lookahead period (12+ bars)\n"
            "   3. Add minimum price movement threshold\n"
            "   4. Check if features are calculated correctly"
        )
    
    if report.label_quality.xgboost_accuracy < 0.55:
        recommendations.append(
            f"⚠️ XGBoost baseline accuracy is only {report.label_quality.xgboost_accuracy:.1%}. "
            "Neural network unlikely to do better. Focus on improving labels first."
        )
    
    if abs(report.label_quality.autocorrelation) > 0.3:
        recommendations.append(
            f"⚠️ High label autocorrelation ({report.label_quality.autocorrelation:.2f}). "
            "Consider using time-series aware validation split."
        )
    
    # Feature quality recommendations
    if report.feature_quality.top_correlated_features:
        max_corr = report.feature_quality.top_correlated_features[0][1]
        if max_corr < 0.1:
            recommendations.append(
                "🔴 CRITICAL: No features correlate with target. Consider:\n"
                "   1. Different feature engineering approach\n"
                "   2. Longer lookback periods for indicators\n"
                "   3. Add momentum/trend features\n"
                "   4. Check if target is truly predictable"
            )
    
    if len(report.feature_quality.multicollinear_pairs) > 20:
        recommendations.append(
            f"⚠️ High multicollinearity ({len(report.feature_quality.multicollinear_pairs)} pairs). "
            "Use PCA or feature selection to reduce to 40-50 features."
        )
    
    if report.feature_quality.recommended_features:
        recommendations.append(
            f"✅ Top 10 features by importance: {report.feature_quality.recommended_features[:10]}"
        )
    
    # Model recommendations
    if report.model_diagnostic.overfitting_detected:
        recommendations.append(
            "⚠️ Overfitting detected. Consider:\n"
            "   1. Increase dropout (0.3-0.5)\n"
            "   2. Add L2 regularization\n"
            "   3. Reduce model size\n"
            "   4. Use data augmentation"
        )
    
    if report.model_diagnostic.loss_trend == "flat":
        recommendations.append(
            "⚠️ Loss not decreasing. Consider:\n"
            "   1. Increase learning rate (try 0.001)\n"
            "   2. Check for gradient issues\n"
            "   3. Simplify model architecture\n"
            "   4. Verify data preprocessing"
        )
    
    if report.model_diagnostic.loss_trend == "increasing":
        recommendations.append(
            "🔴 Loss increasing. Consider:\n"
            "   1. Reduce learning rate (try 0.0001)\n"
            "   2. Add gradient clipping\n"
            "   3. Check for NaN in data\n"
            "   4. Reduce batch size"
        )
    
    # General recommendations
    if report.label_quality.xgboost_accuracy > 0.55:
        recommendations.append(
            f"✅ XGBoost achieves {report.label_quality.xgboost_accuracy:.1%} accuracy. "
            "Neural network should be able to match or exceed this."
        )
    
    recommendations.append(
        "💡 Quick wins:\n"
        "   1. Use config_improved.yaml instead of config_m1_optimized.yaml\n"
        "   2. Enable triple-barrier labeling (buddy.triple_barrier.enabled: true)\n"
        "   3. Reduce features to top 40 (feature_selection.top_k: 40)\n"
        "   4. Set min_epochs: 50 to allow proper warmup"
    )
    
    return recommendations


# =============================================================================
# Main Diagnostic Function
# =============================================================================

def run_diagnostics(
    csv_path: str,
    seq_len: int = 60,
    instrument: str = "USD_JPY",
    tensorboard_dir: Optional[str] = None,
    training_log: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> DiagnosticReport:
    """
    Run all diagnostics on training data.
    
    Args:
        csv_path: Path to training CSV
        seq_len: Sequence length for features
        instrument: Trading pair
        tensorboard_dir: Optional TensorBoard log directory
        training_log: Optional training log file
        output_path: Optional path to save report
        verbose: Print progress
    
    Returns:
        DiagnosticReport with all findings
    """
    t0 = time.perf_counter()
    
    report = DiagnosticReport()
    report.timestamp = datetime.now().isoformat()
    report.csv_path = csv_path
    
    if verbose:
        logger.info("=" * 60)
        logger.info("BUDDY TRAINING DIAGNOSTIC REPORT")
        logger.info("=" * 60)
        logger.info(f"CSV: {csv_path}")
        logger.info(f"Instrument: {instrument}")
        logger.info("")
    
    # Load data
    if verbose:
        logger.info("1. Loading data...")
    
    try:
        df = pd.read_csv(csv_path)
        if verbose:
            logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        report.recommendations.append(f"🔴 CRITICAL: Failed to load CSV: {e}")
        return report
    
    # Identify feature columns
    exclude_cols = {"time", "date", "datetime", "timestamp", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns if c.lower() not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    if verbose:
        logger.info(f"   Identified {len(feature_cols)} feature columns")
    
    # Run feature engineering if needed
    if len(feature_cols) < 20:
        if verbose:
            logger.info("   Running feature engineering...")
        try:
            from feature_engineering import FeatureEngineering
            fe = FeatureEngineering()
            df = fe.create_features(df)
            feature_cols = [c for c in df.columns if c.lower() not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            if verbose:
                logger.info(f"   After feature engineering: {len(feature_cols)} features")
        except Exception as e:
            logger.warning(f"   Feature engineering failed: {e}")
    
    # Data quality check
    if verbose:
        logger.info("")
        logger.info("2. Checking data quality...")
    
    report.data_quality = check_data_quality(df, feature_cols)
    
    if verbose:
        logger.info(f"   Samples: {report.data_quality.n_samples}")
        logger.info(f"   Features: {report.data_quality.n_features}")
        for issue in report.data_quality.issues:
            logger.info(f"   {issue}")
    
    # Prepare features and labels
    if verbose:
        logger.info("")
        logger.info("3. Preparing features and labels...")
    
    # Clean data
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    numeric_df = numeric_df.fillna(0).replace([np.inf, -np.inf], 0)
    
    X = numeric_df.values.astype(np.float32)
    
    # Create labels (next-bar direction)
    if "close" in df.columns:
        closes = df["close"].values
        y = (closes[1:] > closes[:-1]).astype(np.float32)
        X = X[:-1]  # Align with labels
    else:
        logger.error("No 'close' column found")
        return report
    
    # Train/val split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    if verbose:
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Val: {len(X_val)} samples")
    
    # Label quality check
    if verbose:
        logger.info("")
        logger.info("4. Checking label quality...")
    
    report.label_quality = check_label_quality(
        X_train, y_train, X_val, y_val,
        ohlc_df=df if "close" in df.columns else None,
        instrument=instrument,
    )
    
    if verbose:
        logger.info(f"   Positive rate: {report.label_quality.balance_ratio:.1%}")
        logger.info(f"   XGBoost accuracy: {report.label_quality.xgboost_accuracy:.1%}")
        logger.info(f"   XGBoost AUC: {report.label_quality.xgboost_auc:.3f}")
        logger.info(f"   Labels learnable: {report.label_quality.is_learnable}")
        for issue in report.label_quality.issues:
            logger.info(f"   {issue}")
    
    # Feature quality check
    if verbose:
        logger.info("")
        logger.info("5. Checking feature quality...")
    
    report.feature_quality = check_feature_quality(X_train, y_train, feature_cols)
    
    if verbose:
        logger.info(f"   Features analyzed: {report.feature_quality.n_features}")
        if report.feature_quality.top_correlated_features:
            top_feat, top_corr = report.feature_quality.top_correlated_features[0]
            logger.info(f"   Best correlation: {top_feat} ({top_corr:.3f})")
        logger.info(f"   Multicollinear pairs: {len(report.feature_quality.multicollinear_pairs)}")
        for issue in report.feature_quality.issues:
            logger.info(f"   {issue}")
    
    # Model diagnostics
    if verbose:
        logger.info("")
        logger.info("6. Checking model diagnostics...")
    
    tb_dir = tensorboard_dir or "trained_data/tensorboard"
    log_file = training_log or "cli.log"
    
    report.model_diagnostic = check_model_diagnostics(tb_dir, log_file)
    
    if verbose:
        logger.info(f"   Loss trend: {report.model_diagnostic.loss_trend}")
        logger.info(f"   Overfitting: {report.model_diagnostic.overfitting_detected}")
        for issue in report.model_diagnostic.issues:
            logger.info(f"   {issue}")
    
    # Generate recommendations
    if verbose:
        logger.info("")
        logger.info("7. Generating recommendations...")
    
    report.recommendations = generate_recommendations(report)
    
    # Determine overall severity
    severities = [
        report.data_quality.severity,
        report.label_quality.severity,
        report.feature_quality.severity,
        report.model_diagnostic.severity,
    ]
    if "critical" in severities:
        report.overall_severity = "critical"
    elif "warning" in severities:
        report.overall_severity = "warning"
    else:
        report.overall_severity = "ok"
    
    report.runtime_seconds = time.perf_counter() - t0
    
    # Print recommendations
    if verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 60)
        for rec in report.recommendations:
            logger.info(rec)
            logger.info("")
        
        logger.info("=" * 60)
        logger.info(f"Overall severity: {report.overall_severity.upper()}")
        logger.info(f"Diagnostic runtime: {report.runtime_seconds:.1f}s")
        logger.info("=" * 60)
    
    # Save report
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            if verbose:
                logger.info(f"Report saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")
    
    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose Buddy FX training issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diagnose_training.py --csv market_data/USD_JPY_M5.csv
    python diagnose_training.py --csv market_data/EUR_USD_H1.csv --instrument EUR_USD
    python diagnose_training.py --csv data.csv --output report.json --tensorboard trained_data/tensorboard
        """,
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    
    parser.add_argument(
        "--instrument",
        type=str,
        default="USD_JPY",
        help="Trading instrument (default: USD_JPY)",
    )
    
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length (default: 60)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON report",
    )
    
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="TensorBoard log directory",
    )
    
    parser.add_argument(
        "--training-log",
        type=str,
        default=None,
        help="Training log file",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Run diagnostics
    report = run_diagnostics(
        csv_path=args.csv,
        seq_len=args.seq_len,
        instrument=args.instrument,
        tensorboard_dir=args.tensorboard,
        training_log=args.training_log,
        output_path=args.output,
        verbose=not args.quiet,
    )
    
    # Exit with appropriate code
    if report.overall_severity == "critical":
        sys.exit(2)
    elif report.overall_severity == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
