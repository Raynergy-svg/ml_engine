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

from src.research.drawdown_convention import drawdown_fraction

# Import validation metrics modules (optional - graceful fallback if not available)
try:
    from src.risk.var_backtesting import compute_ewma_var, backtest_var, VaRConfig
    HAS_VAR_MODULE = True
except ImportError:
    HAS_VAR_MODULE = False

try:
    from src.risk.trading_metrics import compute_rapi, TradingCostsConfig, PairTier  # noqa: F401
    HAS_RAPI_MODULE = True
except ImportError:
    HAS_RAPI_MODULE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Selection Helpers
# =============================================================================

def load_model_metadata(model_path: str) -> dict:
    """
    Load model metadata including selected feature indices.

    This is used to ensure consistent feature selection across walk-forward folds.
    The metadata is stored in .meta.pkl files alongside the model.

    Args:
        model_path: Path to the model file (e.g., 'trained_data/models/joint/transformer_direction.keras')

    Returns:
        Dictionary containing model metadata (selected_indices, feature_names, n_features, etc.)
        Returns empty dict if metadata file not found
    """
    import pickle
    from pathlib import Path

    try:
        model_path = Path(model_path)

        # Try .meta.pkl (primary format for trainers)
        meta_path = model_path.with_suffix('.meta.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"✓ Loaded model metadata from {meta_path}")
            return metadata

        # Try in parent directory (for ensemble models)
        meta_path = model_path.parent / 'transformer_direction.meta.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"✓ Loaded model metadata from {meta_path}")
            return metadata

        # Try JSON format (older format)
        import json
        json_meta_path = model_path.parent / 'model_metadata.json'
        if json_meta_path.exists():
            with open(json_meta_path) as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded model metadata from {json_meta_path}")
            return metadata

        logger.warning(f"No valid metadata found for {model_path}")
        return {}

    except Exception as e:
        logger.error(f"Failed to load model metadata from {model_path}: {e}")
        return {}


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward cross-validation.

    This class encapsulates all settings for walk-forward validation,
    making it easy to load from YAML config and pass to training functions.
    """
    enabled: bool = True                     # Enable walk-forward validation
    mode: str = "rolling"                    # "rolling" (sliding) or "expanding"
    n_splits: int = 5                        # Number of folds
    train_size: float = 0.60                 # Training window size
    val_size: float = 0.10                   # Validation size
    test_size: float = 0.10                  # Test/holdout size
    gap: int = 24                            # Gap between train/val (prevent leakage)
    min_train_size: int = 2000               # Minimum training samples per fold

    # Purged K-Fold settings
    use_purged_kfold: bool = True            # Use purged cross-validation
    purge_gap: int = 24                      # Purge samples near test
    embargo_gap: int = 12                    # Embargo after train

    # Model retraining per fold
    retrain_per_fold: bool = True            # Retrain model for each fold
    aggregate_method: str = "best"           # "best", "average", or "ensemble"

    # Regime-aware validation
    ensure_regime_balance: bool = False      # Ensure each fold has all regimes
    min_samples_per_regime: int = 50         # Min samples per regime in fold

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WalkForwardConfig':
        """Create WalkForwardConfig from dictionary (e.g., from YAML)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'mode': self.mode,
            'n_splits': self.n_splits,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'gap': self.gap,
            'min_train_size': self.min_train_size,
            'use_purged_kfold': self.use_purged_kfold,
            'purge_gap': self.purge_gap,
            'embargo_gap': self.embargo_gap,
            'retrain_per_fold': self.retrain_per_fold,
            'aggregate_method': self.aggregate_method,
            'ensure_regime_balance': self.ensure_regime_balance,
            'min_samples_per_regime': self.min_samples_per_regime,
        }


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

    # Baseline comparison (computed on final summary only)
    baseline_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Validation metrics (VaR, RAPI)
    var_metrics: Dict[str, float] = field(default_factory=dict)
    rapi_metrics: Dict[str, float] = field(default_factory=dict)

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
        # Canonical NON-NEGATIVE fraction -- see src.research.drawdown_convention.
        metrics['max_drawdown'] = drawdown_fraction(
            float(np.max(drawdowns)),
            source="training.walkforward_validation._compute_metrics",
        )

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
# Baseline Benchmarking (Final Summary Only)
# =============================================================================

def calculate_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline strategy metrics for comparison.

    Compares model performance against simple baseline strategies to ensure
    the model adds value beyond naive approaches.

    Baselines:
    - buy_hold: Always long (benchmark for trend-following)
    - random_walk: Random 50/50 predictions (lower bound)
    - sma_crossover: SMA(5) > SMA(20) (simple technical)
    - momentum: Sign of 5-period return (momentum strategy)

    Args:
        y_true: True direction labels (0/1)
        y_pred: Model predictions (0/1 or probabilities)
        prices: Price series for SMA/momentum calculations
        returns: Return series (computed from prices if None)

    Returns:
        Dict mapping baseline name to metrics dict
    """
    results = {}

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n = len(y_true)

    # Compute returns if not provided
    if returns is None and prices is not None:
        returns = np.diff(prices) / prices[:-1]
        # Align with y_true
        if len(returns) >= n:
            returns = returns[:n]
        else:
            n = len(returns)
            y_true = y_true[:n]
            y_pred = y_pred[:n]

    # Model predictions (for comparison)
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else (y_pred > 0).astype(int)
    model_accuracy = float(np.mean(y_pred_binary == y_true))

    # 1. Buy & Hold Baseline (always long)
    buy_hold_pred = np.ones(n, dtype=int)
    buy_hold_accuracy = float(np.mean(buy_hold_pred == y_true))
    results['buy_hold'] = {
        'accuracy': buy_hold_accuracy,
        'vs_model': model_accuracy - buy_hold_accuracy,
    }

    if returns is not None:
        buy_hold_returns = returns  # Always long
        results['buy_hold']['total_return'] = float(np.sum(buy_hold_returns))
        if np.std(buy_hold_returns) > 0:
            results['buy_hold']['sharpe'] = float(
                np.sqrt(252) * np.mean(buy_hold_returns) / np.std(buy_hold_returns)
            )

    # 2. Random Walk Baseline (50/50)
    np.random.seed(42)  # Reproducible
    random_pred = np.random.randint(0, 2, n)
    random_accuracy = float(np.mean(random_pred == y_true))
    results['random_walk'] = {
        'accuracy': random_accuracy,
        'vs_model': model_accuracy - random_accuracy,
    }

    if returns is not None:
        random_returns = returns * (2 * random_pred - 1)
        results['random_walk']['total_return'] = float(np.sum(random_returns))
        if np.std(random_returns) > 0:
            results['random_walk']['sharpe'] = float(
                np.sqrt(252) * np.mean(random_returns) / np.std(random_returns)
            )

    # 3. SMA Crossover (if prices available)
    if prices is not None and len(prices) > 20:
        prices_series = pd.Series(prices[:n+1] if len(prices) > n else prices)
        sma_5 = prices_series.rolling(5).mean()
        sma_20 = prices_series.rolling(20).mean()

        # Signal: 1 when SMA5 > SMA20 (bullish)
        sma_signal_raw = (sma_5 > sma_20).astype(int).values
        # Align to match y_true length
        sma_signal = sma_signal_raw[:n] if len(sma_signal_raw) > n else np.pad(sma_signal_raw, (0, n - len(sma_signal_raw)), constant_values=0)

        # Handle NaN from rolling window (first 20 periods)
        valid_mask = ~np.isnan(sma_signal.astype(float))
        valid_mask[:20] = False  # First 20 periods have no valid SMA20

        if np.sum(valid_mask) > 0:
            sma_accuracy = float(np.mean(sma_signal[valid_mask] == y_true[valid_mask]))
            results['sma_crossover'] = {
                'accuracy': sma_accuracy,
                'vs_model': model_accuracy - sma_accuracy,
            }

            if returns is not None:
                valid_returns = valid_mask[:len(returns)]
                sma_returns = returns[valid_returns] * (2 * sma_signal[:len(returns)][valid_returns] - 1)
                results['sma_crossover']['total_return'] = float(np.sum(sma_returns))
                if len(sma_returns) > 0 and np.std(sma_returns) > 0:
                    results['sma_crossover']['sharpe'] = float(
                        np.sqrt(252) * np.mean(sma_returns) / np.std(sma_returns)
                    )

    # 4. Momentum Baseline (5-period return sign)
    if prices is not None and len(prices) > 5:
        prices_arr = np.asarray(prices)
        mom_5 = np.zeros(n)
        for i in range(5, min(n, len(prices_arr))):
            mom_5[i] = 1 if prices_arr[i] > prices_arr[i-5] else 0

        mom_accuracy = float(np.mean(mom_5[5:] == y_true[5:]))
        results['momentum'] = {
            'accuracy': mom_accuracy,
            'vs_model': model_accuracy - mom_accuracy,
        }

        if returns is not None and len(returns) > 5:
            mom_returns = returns[5:] * (2 * mom_5[5:len(returns)] - 1)
            results['momentum']['total_return'] = float(np.sum(mom_returns))
            if len(mom_returns) > 0 and np.std(mom_returns) > 0:
                results['momentum']['sharpe'] = float(
                    np.sqrt(252) * np.mean(mom_returns) / np.std(mom_returns)
                )

    # Add model metrics for comparison
    results['model'] = {
        'accuracy': model_accuracy,
    }
    if returns is not None:
        model_returns = returns * (2 * y_pred_binary[:len(returns)] - 1)
        results['model']['total_return'] = float(np.sum(model_returns))
        if np.std(model_returns) > 0:
            results['model']['sharpe'] = float(
                np.sqrt(252) * np.mean(model_returns) / np.std(model_returns)
            )

    return results


def compute_validation_metrics(
    y_pred: np.ndarray,
    returns: np.ndarray,
    instrument: str = "EUR_USD",
) -> Dict[str, Dict[str, float]]:
    """
    Compute VaR and RAPI validation metrics.

    This is called on the final summary to minimize compute overhead.

    Args:
        y_pred: Model predictions
        returns: Actual returns
        instrument: Currency pair for cost calculation

    Returns:
        Dict with 'var' and 'rapi' metric dictionaries
    """
    results = {'var': {}, 'rapi': {}}

    # VaR Backtesting
    if HAS_VAR_MODULE:
        try:
            config = VaRConfig(
                lambda_param=0.94,
                confidence_level=0.95,
                violation_target=0.05,
            )
            var_series = compute_ewma_var(returns, config)
            var_result = backtest_var(returns, var_series, config)

            results['var'] = var_result.to_dict()
        except Exception as e:
            logger.warning(f"VaR calculation failed: {e}")

    # RAPI Calculation
    if HAS_RAPI_MODULE:
        try:
            cost_config = TradingCostsConfig.for_instrument(instrument)
            rapi_result = compute_rapi(y_pred, returns, cost_config)

            results['rapi'] = rapi_result.to_dict()
        except Exception as e:
            logger.warning(f"RAPI calculation failed: {e}")

    return results


# =============================================================================
# Regime-Specific Stress Testing (Phase 3)
# =============================================================================

# Target accuracy per regime (based on research)
REGIME_ACCURACY_TARGETS = {
    'STRONG_TREND': 0.60,   # 60% - trends are predictable
    'WEAK_TREND': 0.55,     # 55% - moderate edge
    'CHOP': 0.50,           # 50% - basically random, avoid trading
    'MEAN_REVERT': 0.58,    # 58% - reversals are somewhat predictable
    'BREAKOUT': 0.55,       # 55% - breakouts need quick reaction
}


def stress_test_regime(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    regimes: np.ndarray,
    returns: Optional[np.ndarray] = None,
    min_samples: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Stress test model performance under different market regimes.

    Calculates accuracy, F1, and other metrics per regime to identify
    where the model excels or struggles.

    Args:
        y_pred: Predicted labels (0 or 1 for direction)
        y_true: Actual labels
        regimes: Regime labels for each sample (0-4 or string names)
        returns: Optional returns for profit-based metrics
        min_samples: Minimum samples to report a regime (flag low-sample regimes)

    Returns:
        Dict mapping regime name to metrics dict with:
        - accuracy: Overall accuracy
        - f1_score: F1 score for positive class
        - sample_count: Number of samples
        - target_met: Whether accuracy meets target
        - low_sample_warning: True if sample count < min_samples
        - sharpe (optional): If returns provided
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # Import regime names
    try:
        from src.core.modular_data_loaders import REGIME_NAMES
    except ImportError:
        REGIME_NAMES = {
            0: 'STRONG_TREND', 1: 'WEAK_TREND', 2: 'CHOP',
            3: 'MEAN_REVERT', 4: 'BREAKOUT'
        }

    # Convert regime labels to strings if numeric
    if isinstance(regimes[0], (int, np.integer)):
        regime_labels = [REGIME_NAMES.get(r, f'REGIME_{r}') for r in regimes]
    else:
        regime_labels = list(regimes)

    results = {}
    unique_regimes = list(set(regime_labels))

    for regime in unique_regimes:
        # Filter samples for this regime
        mask = np.array([r == regime for r in regime_labels])
        n_samples = mask.sum()

        if n_samples == 0:
            results[regime] = {
                'status': 'no_samples',
                'sample_count': 0,
            }
            continue

        y_pred_regime = y_pred[mask]
        y_true_regime = y_true[mask]

        # Calculate metrics
        accuracy = accuracy_score(y_true_regime, y_pred_regime)
        f1 = f1_score(y_true_regime, y_pred_regime, zero_division=0)
        precision = precision_score(y_true_regime, y_pred_regime, zero_division=0)
        recall = recall_score(y_true_regime, y_pred_regime, zero_division=0)

        # Check against target
        target = REGIME_ACCURACY_TARGETS.get(regime, 0.50)
        target_met = accuracy >= target

        # Calculate Sharpe if returns provided
        sharpe = None
        if returns is not None:
            returns_regime = returns[mask]
            # Strategy returns: correct predictions get +1, wrong get -1, times actual return
            correct_mask = y_pred_regime == y_true_regime
            strategy_returns = np.where(correct_mask, np.abs(returns_regime), -np.abs(returns_regime))

            if len(strategy_returns) > 1 and strategy_returns.std() > 0:
                sharpe = float(np.sqrt(252) * strategy_returns.mean() / strategy_returns.std())

        results[regime] = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'sample_count': int(n_samples),
            'target': float(target),
            'target_met': target_met,
            'low_sample_warning': n_samples < min_samples,
            'sharpe': sharpe,
        }

    # Add summary statistics
    total_samples = sum(r.get('sample_count', 0) for r in results.values())
    targets_met = sum(1 for r in results.values() if r.get('target_met', False))

    results['_summary'] = {
        'total_samples': total_samples,
        'regimes_tested': len([r for r in results if r != '_summary']),
        'targets_met': targets_met,
        'low_sample_regimes': [
            r for r, m in results.items()
            if r != '_summary' and m.get('low_sample_warning', False)
        ],
    }

    return results


def stress_test_regime_transitions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    regimes: np.ndarray,
    window_before: int = 5,
    window_after: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze model performance around regime transitions.

    Tests how well the model performs in the bars immediately before
    and after a regime change, which are often the most challenging.

    Args:
        y_pred: Predicted labels
        y_true: Actual labels
        regimes: Regime labels per sample
        window_before: Bars before transition to analyze
        window_after: Bars after transition to analyze

    Returns:
        Dict with 'before_transition', 'after_transition', 'stable' metrics
    """
    from sklearn.metrics import accuracy_score

    n = len(regimes)

    # Find transition points
    transitions = [i for i in range(1, n) if regimes[i] != regimes[i-1]]

    if not transitions:
        return {
            'status': 'no_transitions',
            'message': 'No regime transitions found in data',
        }

    # Collect indices for before/after/stable
    before_indices = []
    after_indices = []
    stable_indices = []

    for t in transitions:
        # Before transition
        start = max(0, t - window_before)
        before_indices.extend(range(start, t))

        # After transition
        end = min(n, t + window_after + 1)
        after_indices.extend(range(t, end))

    # Remove duplicates
    before_indices = list(set(before_indices))
    after_indices = list(set(after_indices))

    # Stable = not in transition zone
    transition_zone = set(before_indices + after_indices)
    stable_indices = [i for i in range(n) if i not in transition_zone]

    results = {}

    # Before transition accuracy
    if before_indices:
        before_acc = accuracy_score(
            y_true[before_indices],
            y_pred[before_indices]
        )
        results['before_transition'] = {
            'accuracy': float(before_acc),
            'sample_count': len(before_indices),
        }

    # After transition accuracy
    if after_indices:
        after_acc = accuracy_score(
            y_true[after_indices],
            y_pred[after_indices]
        )
        results['after_transition'] = {
            'accuracy': float(after_acc),
            'sample_count': len(after_indices),
        }

    # Stable regime accuracy
    if stable_indices:
        stable_acc = accuracy_score(
            y_true[stable_indices],
            y_pred[stable_indices]
        )
        results['stable'] = {
            'accuracy': float(stable_acc),
            'sample_count': len(stable_indices),
        }

    # Summary
    results['_summary'] = {
        'n_transitions': len(transitions),
        'window_before': window_before,
        'window_after': window_after,
        'performance_gap': (
            results.get('stable', {}).get('accuracy', 0) -
            results.get('after_transition', {}).get('accuracy', 0)
        ) if 'stable' in results and 'after_transition' in results else None,
    }

    return results


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
        model.fit(
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

    # === FEATURE SELECTION CONSISTENCY ===
    # Get selected_indices from the passed trainer to ensure consistent features across folds
    # This prevents dimension mismatch errors when each fold does its own feature selection
    selected_indices = getattr(trainer, 'selected_indices', None)
    getattr(trainer, 'n_features', None)

    if selected_indices is not None and len(selected_indices) > 0:
        logger.info(f"🔧 Using inherited feature selection: {len(selected_indices)} features")
        logger.info(f"   Input X shape: {X.shape}, will select {len(selected_indices)} features")

        # Validate indices are within bounds
        if X.ndim == 2:
            max_feature_idx = X.shape[1] - 1
        else:
            max_feature_idx = X.shape[-1] - 1

        if max(selected_indices) > max_feature_idx:
            logger.warning(f"⚠️ selected_indices max ({max(selected_indices)}) exceeds input features ({max_feature_idx}). Resetting to None.")
            selected_indices = None
        else:
            # Apply feature selection to full X array upfront
            X = X[:, selected_indices] if X.ndim == 2 else X[..., selected_indices]
            logger.info(f"✓ Applied feature selection: X shape now {X.shape}")

            # Update feature_names if provided
            if feature_names is not None:
                try:
                    feature_names = [feature_names[i] for i in selected_indices]
                    logger.info(f"✓ Updated feature_names: {len(feature_names)} features")
                except IndexError:
                    logger.warning("Could not update feature_names - index mismatch")
    else:
        logger.info("ℹ️ No inherited feature selection - each fold will select features independently")

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

        # Get weights if provided
        w_train_fold = w[train_idx] if w is not None else None
        w_val_fold = w[val_idx] if w is not None else None

        # Create fresh trainer instance for this fold
        # Import here to avoid circular import
        from src.training.modular_trainers import TransformerDirectionTrainer
        fold_trainer = TransformerDirectionTrainer(trainer.config)

        # === INHERIT FEATURE SELECTION ===
        # If we applied feature selection upfront, tell the fold_trainer not to do it again
        # by setting selected_indices and n_features (trainer will skip selection if already set)
        if selected_indices is not None:
            fold_trainer.selected_indices = list(range(len(selected_indices)))  # Already selected
            fold_trainer.n_features = len(selected_indices)
            # Disable feature selection in config for this fold
            if hasattr(fold_trainer.config, 'use_feature_selection'):
                fold_trainer.config.use_feature_selection = False
            logger.debug(f"Fold {fold + 1}: Inherited feature selection, disabled per-fold selection")

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
