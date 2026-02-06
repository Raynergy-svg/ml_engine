"""
Walk-Forward Validation for Time Series Trading Models.

This module implements proper time-series cross-validation that avoids look-ahead bias.
Critical for FX trading where temporal ordering matters.

Key Features:
1. Expanding window walk-forward validation
2. Rolling window walk-forward validation  
3. Purged cross-validation (with gap between train/test)
4. Out-of-sample performance metrics

Usage:
    from walkforward_validation import WalkForwardValidator, purged_kfold_split
    
    validator = WalkForwardValidator(
        n_splits=5,
        train_size=0.6,
        test_size=0.1,
        gap=10,  # 10 samples gap to prevent leakage
    )
    
    for train_idx, val_idx, test_idx in validator.split(X):
        # Train on train_idx, validate on val_idx, test on test_idx
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Tuple, List, Optional, Dict, Any, Iterator
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransactionCosts:
    """Transaction cost model for backtesting."""
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    commission_pct: float = 0.0  # Percentage of notional
    pip_value: float = 10.0  # Value of 1 pip for standard lot


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""
    mean_sharpe: float
    std_sharpe: float
    percentile_5: float
    percentile_95: float
    confidence_interval_95: Tuple[float, float]
    all_sharpe_ratios: np.ndarray
    all_total_returns: np.ndarray
    probability_positive_sharpe: float
    probability_positive_return: float


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""
    fold: int
    train_size: int
    val_size: int
    test_size: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_period: Tuple[int, int]
    val_period: Tuple[int, int]
    test_period: Tuple[int, int]


@dataclass  
class WalkForwardSummary:
    """Summary of walk-forward validation results."""
    n_folds: int
    fold_results: List[WalkForwardResult]
    
    # Aggregated metrics
    mean_test_metrics: Dict[str, float] = field(default_factory=dict)
    std_test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Stability metrics
    metric_stability: Dict[str, float] = field(default_factory=dict)
    
    def compute_summary(self) -> None:
        """Compute summary statistics from fold results."""
        if not self.fold_results:
            return
        
        # Collect all test metrics
        all_metrics = defaultdict(list)
        for result in self.fold_results:
            for key, value in result.test_metrics.items():
                if np.isfinite(value):
                    all_metrics[key].append(value)
        
        # Compute mean and std
        for key, values in all_metrics.items():
            if values:
                self.mean_test_metrics[key] = float(np.mean(values))
                self.std_test_metrics[key] = float(np.std(values))
                # Coefficient of variation as stability metric
                if self.mean_test_metrics[key] != 0:
                    self.metric_stability[key] = (
                        self.std_test_metrics[key] / abs(self.mean_test_metrics[key])
                    )


class WalkForwardValidator:
    """
    Walk-forward validation for time series data.
    
    Unlike standard K-fold, this maintains temporal ordering and provides
    out-of-sample testing periods that never overlap with training data.
    
    Modes:
    - expanding: Training window grows with each fold
    - rolling: Training window stays fixed size, slides forward
    
    Example visualization (expanding mode):
    
    Fold 1: |--TRAIN--|--GAP--|--VAL--|--TEST--|.................|
    Fold 2: |------TRAIN------|--GAP--|--VAL--|--TEST--|.........|
    Fold 3: |----------TRAIN----------|--GAP--|--VAL--|--TEST--| |
    Fold 4: |--------------TRAIN--------------|--GAP--|--VAL--|--TEST--|
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.6,
        val_size: float = 0.1,
        test_size: float = 0.1,
        gap: int = 10,
        mode: str = "expanding",
        min_train_size: int = 500,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of walk-forward splits
            train_size: Initial training size as fraction (for expanding) or fixed (for rolling)
            val_size: Validation size as fraction
            test_size: Test size as fraction
            gap: Number of samples to skip between train and val (prevents leakage)
            mode: "expanding" or "rolling"
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap
        self.mode = mode
        self.min_train_size = min_train_size
    
    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate train/val/test indices for each fold.
        
        Args:
            X: Feature array
            y: Target array (optional, not used but required for sklearn compatibility)
        
        Yields:
            (train_indices, val_indices, test_indices) for each fold
        """
        n_samples = len(X)
        
        # Calculate sizes for each period
        val_samples = max(int(n_samples * self.val_size), 1)
        test_samples = max(int(n_samples * self.test_size), 1)
        fold_size = val_samples + test_samples + self.gap
        
        # Calculate how much data we need to reserve for all validation/test periods
        total_test_data = self.n_splits * fold_size
        
        if total_test_data >= n_samples - self.min_train_size:
            logger.warning(
                f"Insufficient data for {self.n_splits} folds with current settings. "
                f"Reducing number of splits."
            )
            self.n_splits = max(1, (n_samples - self.min_train_size) // fold_size)
        
        # Generate splits
        for fold in range(self.n_splits):
            if self.mode == "expanding":
                # Expanding window: training grows, test window slides
                train_end = int(n_samples * self.train_size) + fold * fold_size
                train_start = 0
            else:
                # Rolling window: fixed training size, everything slides
                train_start = fold * fold_size
                train_end = train_start + int(n_samples * self.train_size)
            
            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                train_end = train_start + self.min_train_size
            
            # Validation period (after gap)
            val_start = train_end + self.gap
            val_end = min(val_start + val_samples, n_samples)
            
            # Test period (after validation)
            test_start = val_end
            test_end = min(test_start + test_samples, n_samples)
            
            # Ensure we have valid ranges
            if val_start >= n_samples or val_end - val_start < 1 or test_end - test_start < 1:
                break
            
            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, val_idx, test_idx
    
    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


def purged_kfold_split(
    n_samples: int,
    n_splits: int = 5,
    purge_gap: int = 10,
    embargo_gap: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold split for time series with embargo.
    
    This implements the "purged" cross-validation from "Advances in Financial Machine Learning"
    by Marcos Lopez de Prado. It adds:
    1. Purge gap: Removes samples from training that are too close to test set
    2. Embargo gap: Removes samples from test that are too close to training history
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        purge_gap: Samples to remove from train before test
        embargo_gap: Samples to remove from test after train
    
    Yields:
        (train_indices, test_indices) for each fold
    """
    fold_size = n_samples // n_splits
    
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        # Purge: remove training samples too close to test
        train_before_test = np.arange(0, max(0, test_start - purge_gap))
        
        # Embargo: start test after some gap
        test_start_embargoed = test_start + embargo_gap if i > 0 else test_start
        
        # Training samples after test (if any)
        train_after_test = np.arange(test_end + purge_gap, n_samples)
        
        train_idx = np.concatenate([train_before_test, train_after_test])
        test_idx = np.arange(test_start_embargoed, test_end)
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    More sophisticated CV that tests multiple combinations of training/test periods.
    Provides more robust estimates of out-of-sample performance.
    
    From "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 10,
    ):
        """
        Args:
            n_splits: Total number of time periods
            n_test_splits: Number of periods to use as test in each fold
            purge_gap: Samples to purge between train and test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
    
    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test splits."""
        from itertools import combinations
        
        n_samples = len(X)
        split_size = n_samples // self.n_splits
        
        # Create all combinations of test splits
        all_splits = list(range(self.n_splits))
        
        for test_splits in combinations(all_splits, self.n_test_splits):
            train_idx_list = []
            test_idx_list = []
            
            for i in range(self.n_splits):
                start = i * split_size
                end = (i + 1) * split_size if i < self.n_splits - 1 else n_samples
                
                if i in test_splits:
                    # This is a test period
                    test_idx_list.extend(range(start, end))
                else:
                    # This is a training period - but purge near test periods
                    idx_range = list(range(start, end))
                    
                    # Remove samples too close to any test period
                    for test_split in test_splits:
                        test_start = test_split * split_size
                        test_end = (test_split + 1) * split_size
                        
                        idx_range = [
                            j for j in idx_range
                            if j < test_start - self.purge_gap or j >= test_end + self.purge_gap
                        ]
                    
                    train_idx_list.extend(idx_range)
            
            if train_idx_list and test_idx_list:
                yield np.array(train_idx_list), np.array(test_idx_list)


# =============================================================================
# Validation Metrics
# =============================================================================

def calculate_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.
    
    Args:
        y_true: True direction labels (0/1 or -1/1)
        y_pred: Predicted directions or probabilities
        prices: Optional price series for return calculations
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert predictions to binary if needed
    if y_pred.max() <= 1 and y_pred.min() >= 0:
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = (y_pred > 0).astype(int)
    
    y_true_binary = (np.array(y_true) > 0).astype(int)
    
    # Directional accuracy
    metrics['directional_accuracy'] = float(np.mean(y_pred_binary == y_true_binary))
    
    # Precision/Recall for long signals
    true_positives = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    actual_positives = np.sum(y_true_binary == 1)
    
    metrics['precision_long'] = float(true_positives / max(predicted_positives, 1))
    metrics['recall_long'] = float(true_positives / max(actual_positives, 1))
    
    # F1 score
    if metrics['precision_long'] + metrics['recall_long'] > 0:
        metrics['f1_long'] = (
            2 * metrics['precision_long'] * metrics['recall_long'] 
            / (metrics['precision_long'] + metrics['recall_long'])
        )
    else:
        metrics['f1_long'] = 0.0
    
    # If prices available, calculate returns-based metrics
    if prices is not None and len(prices) > 1:
        returns = np.diff(prices) / prices[:-1]
        
        # Align with predictions
        if len(returns) >= len(y_pred_binary):
            returns = returns[:len(y_pred_binary)]
        else:
            y_pred_binary = y_pred_binary[:len(returns)]
        
        # Strategy returns (long when predicted up, cash otherwise)
        strategy_returns = returns * (2 * y_pred_binary - 1)
        
        # Sharpe ratio (annualized assuming 252 trading days)
        if np.std(strategy_returns) > 0:
            metrics['sharpe_ratio'] = float(
                np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            )
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Sortino ratio (using downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            metrics['sortino_ratio'] = float(
                np.sqrt(252) * np.mean(strategy_returns) / np.std(downside_returns)
            )
        else:
            metrics['sortino_ratio'] = 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        metrics['max_drawdown'] = float(np.max(drawdowns))
        
        # Calmar ratio
        if metrics['max_drawdown'] > 0:
            total_return = cumulative_returns[-1] - 1
            metrics['calmar_ratio'] = float(total_return / metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = len(strategy_returns)
        metrics['win_rate'] = float(winning_trades / max(total_trades, 1))
        
        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        metrics['profit_factor'] = float(gross_profit / max(gross_loss, 1e-10))
    
    return metrics


# =============================================================================
# Walk-Forward Analysis Runner
# =============================================================================

def run_walkforward_analysis(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    prices: Optional[np.ndarray] = None,
    n_splits: int = 5,
    train_size: float = 0.6,
    gap: int = 10,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> WalkForwardSummary:
    """
    Run complete walk-forward analysis.
    
    Args:
        model_fn: Function that returns a compiled model
        X: Features [samples, timesteps, features]
        y: Targets (dict for multi-task or array)
        prices: Optional price series for return metrics
        n_splits: Number of walk-forward splits
        train_size: Training set fraction
        gap: Gap between train and validation
        fit_kwargs: Additional kwargs for model.fit()
    
    Returns:
        WalkForwardSummary with aggregated results
    """
    import tensorflow as tf
    
    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_size=train_size,
        gap=gap,
    )
    
    fit_kwargs = fit_kwargs or {}
    fold_results = []
    
    for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Fold {fold + 1}/{n_splits}")
        logger.info(f"Train: {len(train_idx)} samples, Val: {len(val_idx)}, Test: {len(test_idx)}")
        logger.info(f"{'='*60}")
        
        # Prepare data
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        
        if isinstance(y, dict):
            y_train = {k: v[train_idx] for k, v in y.items()}
            y_val = {k: v[val_idx] for k, v in y.items()}
            y_test = {k: v[test_idx] for k, v in y.items()}
        else:
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        
        # Create fresh model for this fold
        model = model_fn()
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **fit_kwargs
        )
        
        # Evaluate
        train_eval = model.evaluate(X_train, y_train, verbose=0, return_dict=True)
        val_eval = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
        test_eval = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        
        # Get predictions for trading metrics
        y_pred = model.predict(X_test, verbose=0)
        
        # Extract direction predictions
        if isinstance(y_pred, dict) and 'direction' in y_pred:
            direction_pred = y_pred['direction']
        else:
            direction_pred = y_pred
        
        if isinstance(y_test, dict) and 'direction' in y_test:
            direction_true = y_test['direction']
        else:
            direction_true = y_test
        
        # Calculate trading metrics
        test_prices = prices[test_idx] if prices is not None else None
        trading_metrics = calculate_trading_metrics(
            direction_true.flatten(),
            direction_pred.flatten(),
            test_prices,
        )
        
        # Combine metrics
        test_metrics = {**test_eval, **trading_metrics}
        
        result = WalkForwardResult(
            fold=fold,
            train_size=len(train_idx),
            val_size=len(val_idx),
            test_size=len(test_idx),
            train_metrics=train_eval,
            val_metrics=val_eval,
            test_metrics=test_metrics,
            train_period=(int(train_idx[0]), int(train_idx[-1])),
            val_period=(int(val_idx[0]), int(val_idx[-1])),
            test_period=(int(test_idx[0]), int(test_idx[-1])),
        )
        
        fold_results.append(result)
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    # Create summary
    summary = WalkForwardSummary(n_folds=len(fold_results), fold_results=fold_results)
    summary.compute_summary()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Walk-Forward Validation Summary")
    logger.info(f"{'='*60}")
    
    for metric, mean_val in summary.mean_test_metrics.items():
        std_val = summary.std_test_metrics.get(metric, 0)
        logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    return summary


# =============================================================================
# Walk-Forward Training for Direction Prediction
# =============================================================================

def train_direction_with_walkforward(
    trainer,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    w: Optional[np.ndarray] = None,
    n_splits: int = 5,
    train_size: float = 0.6,
    gap: int = 10,
    mode: str = "expanding",
) -> Dict[str, Any]:
    """
    Train direction model with walk-forward validation.
    
    This provides a more robust estimate of out-of-sample performance
    by training multiple models on different time periods and averaging.
    
    Args:
        trainer: TransformerDirectionTrainer instance (will be cloned for each fold)
        X: Feature array [samples, features]
        y: Direction labels [samples]
        feature_names: Optional feature names
        w: Optional sample weights (1=clear label, 0=unclear)
        n_splits: Number of walk-forward folds
        train_size: Initial training size fraction
        gap: Gap between train and val to prevent leakage
        mode: "expanding" (growing train) or "rolling" (fixed train size)
    
    Returns:
        Dict with:
        - fold_metrics: List of metrics per fold
        - mean_val_accuracy: Mean val accuracy ± std
        - mean_balanced_accuracy: Mean balanced accuracy ± std
        - best_fold: Index of best performing fold
        - stability_score: Coefficient of variation (lower = more stable)
    """
    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_size=train_size,
        val_size=0.1,  # 10% for validation
        test_size=0.1,  # 10% for test (holdout)
        gap=gap,
        mode=mode,
        min_train_size=500,
    )
    
    fold_metrics = []
    val_accuracies = []
    balanced_accuracies = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Walk-Forward Validation ({n_splits} folds, {mode} mode)")
    logger.info(f"Gap: {gap} samples | Initial train: {train_size*100:.0f}%")
    logger.info(f"{'='*60}")
    
    for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
        logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        X_test_fold = X[test_idx] if len(test_idx) > 0 else None
        y_test_fold = y[test_idx] if len(test_idx) > 0 else None
        
        # Get weights if provided
        w_train_fold = w[train_idx] if w is not None else None
        w_val_fold = w[val_idx] if w is not None else None
        
        # Create fresh trainer instance for this fold
        # Import here to avoid circular import
        from modular_trainers import TransformerDirectionTrainer, TrainerConfig
        fold_trainer = TransformerDirectionTrainer(trainer.config)
        
        # Train on this fold
        metrics = fold_trainer.train(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            feature_names=feature_names,
            w_train=w_train_fold,
            w_val=w_val_fold,
        )
        
        # Store metrics
        fold_metrics.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            **metrics
        })
        
        val_accuracies.append(metrics.get('val_accuracy', 0))
        balanced_accuracies.append(metrics.get('val_balanced_accuracy', 0))
        
        # Log fold results
        logger.info(
            f"Fold {fold + 1} Results: val_acc={metrics.get('val_accuracy', 0):.4f}, "
            f"balanced={metrics.get('val_balanced_accuracy', 0):.4f}"
        )
        
        # Clean up to free memory
        import tensorflow as tf
        tf.keras.backend.clear_session()
        del fold_trainer
    
    # Calculate summary statistics
    mean_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)
    mean_balanced = np.mean(balanced_accuracies)
    std_balanced = np.std(balanced_accuracies)
    
    # Stability score (coefficient of variation - lower is better)
    stability = std_val_acc / mean_val_acc if mean_val_acc > 0 else float('inf')
    
    # Best fold
    best_fold = int(np.argmax(val_accuracies))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Walk-Forward Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Val Accuracy:      {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    logger.info(f"Mean Balanced Accuracy: {mean_balanced:.4f} ± {std_balanced:.4f}")
    logger.info(f"Best Fold:              {best_fold + 1} ({val_accuracies[best_fold]:.4f})")
    logger.info(f"Stability Score (CV):   {stability:.4f} {'(stable)' if stability < 0.1 else '(unstable)' if stability > 0.2 else ''}")
    
    # Check for overfitting signals
    acc_range = max(val_accuracies) - min(val_accuracies)
    if acc_range > 0.1:
        logger.warning(f"High variance across folds ({acc_range:.2f}). Model may be unstable or overfitting to specific periods.")
    
    return {
        'fold_metrics': fold_metrics,
        'mean_val_accuracy': mean_val_acc,
        'std_val_accuracy': std_val_acc,
        'mean_balanced_accuracy': mean_balanced,
        'std_balanced_accuracy': std_balanced,
        'best_fold': best_fold,
        'stability_score': stability,
        'val_accuracies': val_accuracies,
    }


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def apply_transaction_costs(
    returns: np.ndarray,
    positions: np.ndarray,
    costs: TransactionCosts,
) -> np.ndarray:
    """
    Apply transaction costs to returns based on position changes.
    
    Args:
        returns: Array of price returns
        positions: Array of positions (1=long, 0=flat, -1=short)
        costs: TransactionCosts object
    
    Returns:
        Adjusted returns after transaction costs
    """
    adjusted_returns = returns.copy()
    
    # Calculate position changes (trades)
    position_changes = np.diff(positions, prepend=0)
    trades = np.abs(position_changes)
    
    # Spread cost (paid on every trade)
    spread_cost_pct = (costs.spread_pips * costs.pip_value) / 100000  # Assuming 100k notional
    
    # Slippage cost (paid on every trade)
    slippage_cost_pct = (costs.slippage_pips * costs.pip_value) / 100000
    
    # Total cost per trade
    total_cost_pct = spread_cost_pct + slippage_cost_pct + costs.commission_pct
    
    # Subtract costs where trades occur
    cost_impact = trades * total_cost_pct
    adjusted_returns = adjusted_returns - cost_impact[:len(adjusted_returns)]
    
    return adjusted_returns


def monte_carlo_simulation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    n_simulations: int = 1000,
    costs: Optional[TransactionCosts] = None,
    confidence_noise: float = 0.05,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation to estimate confidence intervals for trading performance.
    
    This simulates the impact of:
    1. Prediction uncertainty (adding noise to probabilities)
    2. Transaction costs and slippage
    3. Random market conditions (bootstrapping)
    
    Args:
        y_true: True direction labels (0/1 or -1/1)
        y_pred: Predicted probabilities
        prices: Price series for return calculations
        n_simulations: Number of Monte Carlo runs
        costs: TransactionCosts object (None = no costs)
        confidence_noise: Std dev of noise to add to predictions
        random_seed: Random seed for reproducibility
    
    Returns:
        MonteCarloResult with statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if costs is None:
        costs = TransactionCosts()
    
    # Convert to binary
    y_true_binary = (np.array(y_true) > 0).astype(int)
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    returns = returns[:min(len(returns), len(y_pred))]
    y_pred = y_pred[:len(returns)]
    y_true_binary = y_true_binary[:len(returns)]
    
    sharpe_ratios = []
    total_returns = []
    
    for _ in range(n_simulations):
        # Add noise to predictions (simulate uncertainty)
        noisy_pred = y_pred + np.random.normal(0, confidence_noise, len(y_pred))
        noisy_pred = np.clip(noisy_pred, 0, 1)
        
        # Bootstrap sampling (simulate different market samples)
        bootstrap_idx = np.random.choice(len(returns), len(returns), replace=True)
        boot_returns = returns[bootstrap_idx]
        boot_pred = noisy_pred[bootstrap_idx]
        
        # Generate positions based on predictions
        positions = (boot_pred > 0.5).astype(int)
        
        # Calculate strategy returns
        strategy_returns = boot_returns * (2 * positions - 1)
        
        # Apply transaction costs
        strategy_returns = apply_transaction_costs(
            strategy_returns,
            positions,
            costs,
        )
        
        # Calculate metrics
        if np.std(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe = 0.0
        
        total_return = np.prod(1 + strategy_returns) - 1
        
        sharpe_ratios.append(sharpe)
        total_returns.append(total_return)
    
    sharpe_array = np.array(sharpe_ratios)
    returns_array = np.array(total_returns)
    
    # Calculate statistics
    mean_sharpe = float(np.mean(sharpe_array))
    std_sharpe = float(np.std(sharpe_array))
    percentile_5 = float(np.percentile(sharpe_array, 5))
    percentile_95 = float(np.percentile(sharpe_array, 95))
    
    prob_positive_sharpe = float(np.mean(sharpe_array > 0))
    prob_positive_return = float(np.mean(returns_array > 0))
    
    return MonteCarloResult(
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
        percentile_5=percentile_5,
        percentile_95=percentile_95,
        confidence_interval_95=(percentile_5, percentile_95),
        all_sharpe_ratios=sharpe_array,
        all_total_returns=returns_array,
        probability_positive_sharpe=prob_positive_sharpe,
        probability_positive_return=prob_positive_return,
    )


def run_monte_carlo_walkforward(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    n_splits: int = 5,
    n_simulations: int = 1000,
    costs: Optional[TransactionCosts] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run walk-forward validation with Monte Carlo simulation for each fold.
    
    Args:
        model_fn: Function that returns a compiled model
        X: Features
        y: Targets
        prices: Price series
        n_splits: Number of walk-forward splits
        n_simulations: Monte Carlo simulations per fold
        costs: Transaction costs
        **kwargs: Additional arguments for walk-forward
    
    Returns:
        Dictionary with walk-forward results and Monte Carlo statistics
    """
    import tensorflow as tf
    
    # Run standard walk-forward
    wf_summary = run_walkforward_analysis(
        model_fn, X, y, prices, n_splits=n_splits, **kwargs
    )
    
    # Run Monte Carlo on each fold
    mc_results = []
    validator = WalkForwardValidator(n_splits=n_splits)
    
    for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
        logger.info(f"Running Monte Carlo for fold {fold + 1}/{n_splits}...")
        
        X_test = X[test_idx]
        y_test = y[test_idx] if not isinstance(y, dict) else y['direction'][test_idx]
        prices_test = prices[test_idx]
        
        # Train model for this fold
        model = model_fn()
        X_train = X[train_idx]
        y_train = y[train_idx] if not isinstance(y, dict) else {k: v[train_idx] for k, v in y.items()}
        X_val = X[val_idx]
        y_val = y[val_idx] if not isinstance(y, dict) else {k: v[val_idx] for k, v in y.items()}
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, epochs=50)
        
        # Get predictions
        y_pred = model.predict(X_test, verbose=0)
        if isinstance(y_pred, dict):
            y_pred = y_pred.get('direction', y_pred)
        
        # Run Monte Carlo
        mc_result = monte_carlo_simulation(
            y_test.flatten(),
            y_pred.flatten(),
            prices_test,
            n_simulations=n_simulations,
            costs=costs,
        )
        
        mc_results.append({
            'fold': fold + 1,
            'mean_sharpe': mc_result.mean_sharpe,
            'sharpe_ci_95': mc_result.confidence_interval_95,
            'prob_positive_sharpe': mc_result.probability_positive_sharpe,
            'prob_positive_return': mc_result.probability_positive_return,
        })
        
        logger.info(
            f"  Fold {fold + 1} MC: Sharpe {mc_result.mean_sharpe:.3f} "
            f"(95% CI: {mc_result.confidence_interval_95[0]:.3f} - {mc_result.confidence_interval_95[1]:.3f})"
        )
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    # Aggregate Monte Carlo results
    all_sharpes = [r['mean_sharpe'] for r in mc_results]
    aggregate_prob_positive = np.mean([r['prob_positive_sharpe'] for r in mc_results])
    
    logger.info(f"\n{'='*60}")
    logger.info("Monte Carlo Aggregate Results")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Sharpe across folds: {np.mean(all_sharpes):.3f} ± {np.std(all_sharpes):.3f}")
    logger.info(f"Probability of positive Sharpe: {aggregate_prob_positive:.1%}")
    
    return {
        'walkforward_summary': wf_summary,
        'monte_carlo_results': mc_results,
        'aggregate_sharpe_mean': np.mean(all_sharpes),
        'aggregate_sharpe_std': np.std(all_sharpes),
        'probability_profitable': aggregate_prob_positive,
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Walk-Forward Validation Module")
    print("=" * 60)
    
    # Test with dummy data
    n_samples = 1000
    X_dummy = np.random.randn(n_samples, 60, 10)
    y_dummy = (np.random.rand(n_samples) > 0.5).astype(float)
    
    # Test validator
    validator = WalkForwardValidator(n_splits=5, train_size=0.6, gap=10)
    
    print("\nWalk-Forward Splits:")
    for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X_dummy)):
        print(f"  Fold {fold+1}: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
        print(f"    Ranges: Train[{train_idx[0]}-{train_idx[-1]}], "
              f"Val[{val_idx[0]}-{val_idx[-1]}], Test[{test_idx[0]}-{test_idx[-1]}]")
    
    # Test purged k-fold
    print("\nPurged K-Fold Splits:")
    for fold, (train_idx, test_idx) in enumerate(purged_kfold_split(n_samples, n_splits=5)):
        print(f"  Fold {fold+1}: Train {len(train_idx)}, Test {len(test_idx)}")
    
    # Test trading metrics
    print("\nTrading Metrics Test:")
    y_true = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0.7, 0.6, 0.3, 0.8, 0.4, 0.9, 0.5, 0.2, 0.3, 0.6])
    prices = np.array([100, 101, 100.5, 101.5, 101, 102, 101.5, 101, 100.5, 101])
    
    metrics = calculate_trading_metrics(y_true, y_pred, prices)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test Monte Carlo simulation
    print("\nMonte Carlo Simulation Test:")
    mc_result = monte_carlo_simulation(
        y_true, y_pred, prices,
        n_simulations=100,
        costs=TransactionCosts(spread_pips=1.0, slippage_pips=0.5),
        random_seed=42,
    )
    print(f"  Mean Sharpe: {mc_result.mean_sharpe:.4f} ± {mc_result.std_sharpe:.4f}")
    print(f"  95% CI: [{mc_result.confidence_interval_95[0]:.4f}, {mc_result.confidence_interval_95[1]:.4f}]")
    print(f"  P(positive Sharpe): {mc_result.probability_positive_sharpe:.2%}")
    print(f"  P(positive return): {mc_result.probability_positive_return:.2%}")
    
    # Test transaction costs
    print("\nTransaction Costs Test:")
    returns_test = np.array([0.01, -0.005, 0.008, -0.003, 0.012])
    positions_test = np.array([1, 1, 0, 1, 1])
    costs_test = TransactionCosts(spread_pips=1.0, slippage_pips=0.5)
    adjusted = apply_transaction_costs(returns_test, positions_test, costs_test)
    print(f"  Original returns: {returns_test}")
    print(f"  Adjusted returns: {adjusted}")
    print(f"  Cost impact: {returns_test - adjusted}")
    
    print("\n✓ Walk-forward validation module ready")


