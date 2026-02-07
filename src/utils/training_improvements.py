"""
Training Improvements Module for Buddy FX Model.

Implements fixes for poor training results:
1. Triple-barrier labeling for direction (aligns with trading strategy)
2. Feature dimensionality reduction
3. Longer lookahead direction labels
4. Gradient monitoring and diagnostics
5. XGBoost baseline verification

Author: Training Debug Session
Date: 2025-01-01
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Fix 1: Triple-Barrier Direction Labels
# =============================================================================

class BarrierOutcome(Enum):
    """Outcome of triple-barrier labeling."""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TIMEOUT = "timeout"


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    stop_loss_pips: float = 30.0
    take_profit_pips: float = 60.0
    max_horizon_candles: int = 30  # 2.5 hours on M5
    spread_pips: float = 1.5
    use_both_directions: bool = True  # Try both long/short, pick best
    timeout_label: float = 0.5  # Label for timeout (unclear signal)
    timeout_weight: float = 0.3  # Weight for timeout samples
    exclude_timeouts: bool = False  # Whether to exclude timeout samples


def compute_triple_barrier_direction_labels(
    ohlc_df: pd.DataFrame,
    *,
    instrument: str,
    config: TripleBarrierConfig,
    seq_len: int = 60,
    label_stride: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute direction labels using triple-barrier method.
    
    Instead of predicting next-bar direction, predicts which direction
    would have been profitable given TP/SL targets.
    
    Args:
        ohlc_df: DataFrame with OHLC data
        instrument: Trading pair (e.g., "USD_JPY")
        config: Triple-barrier configuration
        seq_len: Sequence length for features
        label_stride: Compute labels every N bars
        verbose: Print progress
    
    Returns:
        Tuple of:
        - direction_labels: (n,) array, 1=long profitable, 0=short profitable, 0.5=unclear
        - sample_weights: (n,) array of weights (0 for excluded, 0.3 for timeout, 1.0 for clear)
        - stats: Dictionary with labeling statistics
    """
    try:
        from src.utils.fx_paper import pip_size, simulate_tp_sl_outcome
    except ImportError:
        logger.warning("fx_paper not available, falling back to simple labels")
        n = len(ohlc_df)
        return (
            np.full(n, 0.5, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            {"error": "fx_paper not available"}
        )
    
    n = len(ohlc_df)
    direction_labels = np.full(n, 0.5, dtype=np.float32)
    sample_weights = np.zeros(n, dtype=np.float32)
    
    pip = float(pip_size(instrument))
    start_idx = max(0, seq_len - 1)
    end_idx = n - config.max_horizon_candles - 2
    
    if end_idx <= start_idx:
        logger.warning(f"Insufficient data for triple-barrier: start={start_idx}, end={end_idx}")
        return direction_labels, sample_weights, {"error": "insufficient_data"}
    
    stats = {
        "n_samples": 0,
        "n_long_wins": 0,
        "n_short_wins": 0,
        "n_timeouts": 0,
        "n_excluded": 0,
    }
    
    t0 = time.perf_counter()
    
    for i in range(start_idx, end_idx, label_stride):
        try:
            # Get spread at entry
            spread = config.spread_pips
            if "bid_close" in ohlc_df.columns and "ask_close" in ohlc_df.columns:
                try:
                    bid = float(ohlc_df["bid_close"].iloc[i + 1])
                    ask = float(ohlc_df["ask_close"].iloc[i + 1])
                    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid > 0:
                        spread = (ask - bid) / pip
                except Exception:
                    pass
            
            best_direction = None
            best_outcome = None
            best_pnl = float("-inf")
            
            directions_to_try = ["long", "short"] if config.use_both_directions else ["long"]
            
            for direction in directions_to_try:
                try:
                    res = simulate_tp_sl_outcome(
                        ohlc_df,
                        instrument=instrument,
                        signal_index=i,
                        direction=direction,
                        stop_loss_pips=config.stop_loss_pips,
                        take_profit_pips=config.take_profit_pips,
                        spread_pips=spread,
                        max_horizon_candles=config.max_horizon_candles,
                        tie_break="sl",
                    )
                    
                    # Calculate PnL
                    if direction == "long":
                        pnl = (res.exit_price - res.entry_price) / pip
                    else:
                        pnl = (res.entry_price - res.exit_price) / pip
                    
                    outcome = (
                        BarrierOutcome.TAKE_PROFIT if res.hit_type == "tp"
                        else BarrierOutcome.STOP_LOSS if res.hit_type == "sl"
                        else BarrierOutcome.TIMEOUT
                    )
                    
                    # Keep best direction (prefer TP hit, then highest PnL)
                    if outcome == BarrierOutcome.TAKE_PROFIT or pnl > best_pnl:
                        best_direction = direction
                        best_outcome = outcome
                        best_pnl = pnl
                        
                        if outcome == BarrierOutcome.TAKE_PROFIT:
                            break  # Found a winner, stop searching
                            
                except Exception as e:
                    logger.debug(f"Triple-barrier failed at idx={i}, dir={direction}: {e}")
                    continue
            
            if best_direction is not None and best_outcome is not None:
                stats["n_samples"] += 1
                
                if best_outcome == BarrierOutcome.TIMEOUT:
                    stats["n_timeouts"] += 1
                    if config.exclude_timeouts:
                        stats["n_excluded"] += 1
                        continue
                    direction_labels[i] = config.timeout_label
                    sample_weights[i] = config.timeout_weight
                else:
                    # Clear outcome
                    if best_outcome == BarrierOutcome.TAKE_PROFIT:
                        # TP hit = intended direction was correct
                        direction_labels[i] = 1.0 if best_direction == "long" else 0.0
                        if best_direction == "long":
                            stats["n_long_wins"] += 1
                        else:
                            stats["n_short_wins"] += 1
                    else:
                        # SL hit = intended direction was wrong
                        direction_labels[i] = 0.0 if best_direction == "long" else 1.0
                    sample_weights[i] = 1.0
                    
        except Exception as e:
            logger.debug(f"Triple-barrier labeling failed at idx={i}: {e}")
            continue
    
    elapsed = time.perf_counter() - t0
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["n_samples"] / max(elapsed, 0.001)
    
    if verbose:
        logger.info(
            f"Triple-barrier labeling: {stats['n_samples']} samples, "
            f"{stats['n_long_wins']} long wins, {stats['n_short_wins']} short wins, "
            f"{stats['n_timeouts']} timeouts in {elapsed:.2f}s"
        )
    
    return direction_labels, sample_weights, stats


# =============================================================================
# Fix 2: Feature Dimensionality Reduction
# =============================================================================

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection/reduction."""
    method: Literal["top_k", "pca", "mutual_info", "variance", "none"] = "top_k"
    top_k: int = 40  # Number of features to keep for top_k method
    pca_components: int = 50  # Number of PCA components
    variance_threshold: float = 0.01  # Min variance for variance method
    mi_k: int = 40  # Number of features for mutual_info method


def select_top_k_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    k: int = 40,
    method: Literal["correlation", "mutual_info", "combined"] = "combined",
) -> Tuple[List[int], List[str], Dict[str, float]]:
    """
    Select top-K features based on predictive power.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        feature_names: List of feature names
        k: Number of features to select
        method: Selection method
    
    Returns:
        Tuple of (selected_indices, selected_names, importance_scores)
    """
    n_features = X_train.shape[1]
    k = min(k, n_features)
    
    scores = np.zeros(n_features)
    
    if method in ("correlation", "combined"):
        # Pearson correlation with target
        for i in range(n_features):
            try:
                corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
                if np.isfinite(corr):
                    scores[i] += abs(corr)
            except Exception:
                pass
    
    if method in ("mutual_info", "combined"):
        try:
            from sklearn.feature_selection import mutual_info_classif
            
            # Discretize target if continuous
            y_discrete = (y_train >= 0.5).astype(int)
            mi_scores = mutual_info_classif(
                X_train, y_discrete,
                discrete_features=False,
                random_state=42,
                n_neighbors=5,
            )
            # Normalize to [0, 1]
            mi_max = mi_scores.max()
            if mi_max > 0:
                mi_scores = mi_scores / mi_max
            scores += mi_scores
        except Exception as e:
            logger.warning(f"Mutual info failed: {e}")
    
    # Normalize combined scores
    if method == "combined" and scores.max() > 0:
        scores = scores / scores.max()
    
    # Select top-K
    top_indices = np.argsort(scores)[-k:][::-1]
    selected_names = [feature_names[i] for i in top_indices]
    importance = {feature_names[i]: float(scores[i]) for i in top_indices}
    
    return list(top_indices), selected_names, importance


def apply_pca_reduction(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_components: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Apply PCA dimensionality reduction.
    
    Args:
        X_train: Training features
        X_val: Validation features
        n_components: Number of PCA components
    
    Returns:
        Tuple of (X_train_pca, X_val_pca, pca_model)
    """
    from sklearn.decomposition import PCA
    
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_val_pca = pca.transform(X_val).astype(np.float32)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {X_train.shape[1]} -> {n_components} components, "
                f"explained variance: {explained_var:.2%}")
    
    return X_train_pca, X_val_pca, pca


# =============================================================================
# Fix 4: Longer Lookahead Direction Labels
# =============================================================================

def compute_lookahead_direction_labels(
    closes: np.ndarray,
    lookahead: int = 12,
    threshold: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute direction labels with longer lookahead and threshold filtering.
    
    Args:
        closes: Close prices array
        lookahead: Number of bars to look ahead
        threshold: Minimum price change (as fraction) to assign label
    
    Returns:
        Tuple of (direction_labels, sample_weights)
    """
    n = len(closes)
    direction_labels = np.full(n, 0.5, dtype=np.float32)
    sample_weights = np.zeros(n, dtype=np.float32)
    
    for i in range(n - lookahead):
        future_close = closes[i + lookahead]
        current_close = closes[i]
        
        if current_close <= 0:
            continue
        
        pct_change = (future_close - current_close) / current_close
        
        if abs(pct_change) >= threshold:
            direction_labels[i] = 1.0 if pct_change > 0 else 0.0
            sample_weights[i] = 1.0
        else:
            # Below threshold = unclear signal
            direction_labels[i] = 0.5
            sample_weights[i] = 0.3  # Downweight unclear samples
    
    n_clear = (sample_weights == 1.0).sum()
    n_unclear = ((sample_weights > 0) & (sample_weights < 1.0)).sum()
    logger.info(f"Lookahead labels: {n_clear} clear signals, {n_unclear} unclear")
    
    return direction_labels, sample_weights


# =============================================================================
# Fix 5: Gradient Monitoring Callback
# =============================================================================

def create_gradient_monitor_callback():
    """Create a Keras callback for gradient monitoring."""
    import tensorflow as tf
    
    class GradientMonitorCallback(tf.keras.callbacks.Callback):
        """Monitor gradient flow during training."""
        
        def __init__(self, check_every_n_batches: int = 50):
            super().__init__()
            self.check_every_n_batches = check_every_n_batches
            self.batch_count = 0
            self.gradient_stats = []
        
        def on_train_batch_end(self, batch, logs=None):
            self.batch_count += 1
            if self.batch_count % self.check_every_n_batches != 0:
                return
            
            # Check weight norms (proxy for gradient health)
            stats = {"batch": self.batch_count}
            vanishing_layers = []
            exploding_layers = []
            
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weight_norm = float(tf.norm(layer.kernel).numpy())
                    stats[f"{layer.name}_norm"] = weight_norm
                    
                    if weight_norm < 1e-7:
                        vanishing_layers.append(layer.name)
                    elif weight_norm > 1000:
                        exploding_layers.append(layer.name)
            
            if vanishing_layers:
                logger.warning(f"Batch {self.batch_count}: Potential vanishing gradients in {vanishing_layers}")
            if exploding_layers:
                logger.warning(f"Batch {self.batch_count}: Potential exploding gradients in {exploding_layers}")
            
            self.gradient_stats.append(stats)
        
        def on_epoch_end(self, epoch, logs=None):
            # Summary at end of epoch
            if not self.gradient_stats:
                return
            
            # Check for consistent issues
            last_stats = self.gradient_stats[-1]
            issues = []
            
            for key, value in last_stats.items():
                if key == "batch":
                    continue
                if value < 1e-6:
                    issues.append(f"{key}: very small ({value:.2e})")
                elif value > 100:
                    issues.append(f"{key}: large ({value:.2e})")
            
            if issues:
                logger.info(f"Epoch {epoch} gradient health issues: {issues[:5]}")
    
    return GradientMonitorCallback


# =============================================================================
# Fix 6: XGBoost Baseline
# =============================================================================

def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost baseline model to verify data quality.
    
    If XGBoost can't beat random, the problem is in the data/labels.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        task: "classification" or "regression"
        n_estimators: Number of trees
        max_depth: Max tree depth
        learning_rate: Learning rate
        verbose: Print results
    
    Returns:
        Tuple of (model, metrics)
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return None, {"error": "xgboost_not_installed"}
    
    # Flatten sequences if needed
    if len(X_train.shape) == 3:
        # (samples, seq_len, features) -> (samples, seq_len * features)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Ensure labels are appropriate
    if task == "classification":
        y_train_bin = (y_train >= 0.5).astype(int).ravel()
        y_val_bin = (y_val >= 0.5).astype(int).ravel()
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        
        model.fit(
            X_train, y_train_bin,
            eval_set=[(X_val, y_val_bin)],
            verbose=False,
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = (train_pred == y_train_bin).mean()
        val_acc = (val_pred == y_val_bin).mean()
        
        # Probability predictions for AUC
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import roc_auc_score
        try:
            train_auc = roc_auc_score(y_train_bin, train_proba)
            val_auc = roc_auc_score(y_val_bin, val_proba)
        except Exception:
            train_auc = val_auc = 0.5
        
        metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_auc": float(train_auc),
            "val_auc": float(val_auc),
            "overfit_gap": float(train_acc - val_acc),
        }
        
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        
        model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_val, y_val.ravel())],
            verbose=False,
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(y_train.ravel(), train_pred)
        val_mse = mean_squared_error(y_val.ravel(), val_pred)
        train_r2 = r2_score(y_train.ravel(), train_pred)
        val_r2 = r2_score(y_val.ravel(), val_pred)
        
        metrics = {
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
        }
    
    if verbose:
        logger.info(f"XGBoost baseline results: {metrics}")
        
        if task == "classification":
            if val_acc < 0.52:
                logger.warning(
                    "⚠️ XGBoost baseline accuracy < 52%. "
                    "This suggests the labels may not be predictable from features. "
                    "Consider changing the labeling strategy."
                )
            elif val_acc < 0.55:
                logger.info(
                    "XGBoost baseline accuracy 52-55%. "
                    "Labels have weak signal. Neural network may not improve much."
                )
            else:
                logger.info(
                    f"✓ XGBoost baseline accuracy {val_acc:.1%}. "
                    "Labels appear learnable. Neural network should be able to match or exceed."
                )
    
    return model, metrics


# =============================================================================
# Combined Labeling Strategy
# =============================================================================

@dataclass
class ImprovedLabelingConfig:
    """Configuration for improved labeling strategy."""
    
    # Primary labeling method
    method: Literal["triple_barrier", "lookahead", "next_bar"] = "triple_barrier"
    
    # Triple-barrier settings
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    
    # Lookahead settings
    lookahead_bars: int = 12  # 1 hour on M5
    lookahead_threshold: float = 0.002  # 0.2% min move
    
    # Feature selection
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    
    # Training settings
    min_epochs: int = 50
    early_stopping_patience: int = 30
    
    # XGBoost baseline
    run_xgboost_baseline: bool = True


def create_improved_labels(
    ohlc_df: pd.DataFrame,
    closes: np.ndarray,
    *,
    instrument: str,
    config: ImprovedLabelingConfig,
    seq_len: int = 60,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create improved direction labels using the configured strategy.
    
    Args:
        ohlc_df: OHLC DataFrame
        closes: Close prices array
        instrument: Trading pair
        config: Labeling configuration
        seq_len: Sequence length
        verbose: Print progress
    
    Returns:
        Tuple of (direction_labels, sample_weights, stats)
    """
    n = len(closes)
    
    if config.method == "triple_barrier":
        direction_labels, sample_weights, stats = compute_triple_barrier_direction_labels(
            ohlc_df,
            instrument=instrument,
            config=config.triple_barrier,
            seq_len=seq_len,
            label_stride=1,
            verbose=verbose,
        )
        stats["method"] = "triple_barrier"
        
    elif config.method == "lookahead":
        direction_labels, sample_weights = compute_lookahead_direction_labels(
            closes,
            lookahead=config.lookahead_bars,
            threshold=config.lookahead_threshold,
        )
        stats = {
            "method": "lookahead",
            "lookahead_bars": config.lookahead_bars,
            "threshold": config.lookahead_threshold,
        }
        
    else:  # next_bar (original method)
        next_close = closes[1:]
        cur_close = closes[:-1]
        direction_labels = np.zeros(n, dtype=np.float32)
        direction_labels[:-1] = (next_close > cur_close).astype(np.float32)
        sample_weights = np.ones(n, dtype=np.float32)
        sample_weights[-1] = 0  # Last sample has no label
        stats = {"method": "next_bar"}
    
    # Compute label statistics
    valid_mask = sample_weights > 0
    if valid_mask.any():
        stats["n_valid_samples"] = int(valid_mask.sum())
        stats["positive_rate"] = float(direction_labels[valid_mask].mean())
        stats["avg_weight"] = float(sample_weights[valid_mask].mean())
    
    return direction_labels, sample_weights, stats


# =============================================================================
# Integration Helper
# =============================================================================

def apply_training_improvements(
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    feature_names: List[str],
    config: ImprovedLabelingConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Apply all training improvements to prepared data.
    
    Args:
        train_feats, val_feats: Feature arrays
        train_labels, val_labels: Label arrays
        feature_names: List of feature names
        config: Improvement configuration
        verbose: Print progress
    
    Returns:
        Tuple of (train_feats, val_feats, train_labels, val_labels, feature_names, stats)
    """
    stats = {}
    
    # 1. Feature selection/reduction
    if config.feature_selection.method == "top_k":
        selected_idx, selected_names, importance = select_top_k_features(
            train_feats, train_labels, feature_names,
            k=config.feature_selection.top_k,
            method="combined",
        )
        train_feats = train_feats[:, selected_idx]
        val_feats = val_feats[:, selected_idx]
        feature_names = selected_names
        stats["feature_selection"] = {
            "method": "top_k",
            "original_features": len(importance) + len(selected_idx),
            "selected_features": len(selected_idx),
            "top_features": list(importance.keys())[:10],
        }
        if verbose:
            logger.info(f"Feature selection: {stats['feature_selection']['original_features']} -> {len(selected_idx)}")
            
    elif config.feature_selection.method == "pca":
        train_feats, val_feats, pca_model = apply_pca_reduction(
            train_feats, val_feats,
            n_components=config.feature_selection.pca_components,
        )
        feature_names = [f"pca_{i}" for i in range(train_feats.shape[1])]
        stats["feature_selection"] = {
            "method": "pca",
            "n_components": train_feats.shape[1],
            "explained_variance": float(pca_model.explained_variance_ratio_.sum()),
        }
    
    # 2. XGBoost baseline (if enabled)
    if config.run_xgboost_baseline:
        try:
            _, xgb_metrics = train_xgboost_baseline(
                train_feats, train_labels,
                val_feats, val_labels,
                task="classification",
                verbose=verbose,
            )
            stats["xgboost_baseline"] = xgb_metrics
        except Exception as e:
            logger.warning(f"XGBoost baseline failed: {e}")
            stats["xgboost_baseline"] = {"error": str(e)}
    
    return train_feats, val_feats, train_labels, val_labels, feature_names, stats

