"""
Model evaluation and backtesting framework for trading strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def predict_with_uncertainty(
    model: Any,
    X: Any,
    n_iterations: int = 30,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with uncertainty estimation using Monte Carlo Dropout.
    
    Args:
        model: PyTorch model with dropout layers
        X: Input tensor
        n_iterations: Number of forward passes with dropout enabled
        device: Device to run on
        
    Returns:
        Tuple of (mean_predictions, std_predictions)
    """
    raise RuntimeError(
        "predict_with_uncertainty is retired because PyTorch was removed. "
        "Use the TensorFlow-only Buddy pipeline via main.py."
    )


def calibrate_predictions(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    actual: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calibrate and evaluate prediction uncertainties.
    
    Args:
        predictions: Point predictions
        uncertainties: Prediction uncertainties (std)
        actual: Actual values
        confidence_level: Desired confidence level
        
    Returns:
        Dictionary with calibration metrics
    """
    from scipy import stats
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate prediction intervals
    lower_bound = predictions - z_score * uncertainties
    upper_bound = predictions + z_score * uncertainties
    
    # Check coverage (how many actual values fall within intervals)
    coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound))
    
    # Calculate interval width
    avg_interval_width = np.mean(upper_bound - lower_bound)
    
    # Calculate sharpness (how tight are the intervals)
    sharpness = uncertainties.mean()
    
    metrics = {
        'coverage': coverage * 100,
        'target_coverage': confidence_level * 100,
        'avg_interval_width': avg_interval_width,
        'sharpness': sharpness,
        'calibration_error': abs(coverage - confidence_level) * 100
    }
    
    logger.info(f"Calibration: {coverage*100:.1f}% coverage (target: {confidence_level*100:.1f}%)")
    
    return metrics


def smooth_predictions(
    predictions: np.ndarray,
    method: str = "ema",
    window: int = 5,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Smooth predictions to reduce noise and improve stability.
    
    Args:
        predictions: Raw predictions
        method: Smoothing method ('ema', 'sma', 'median')
        window: Window size for smoothing
        alpha: Alpha parameter for EMA
        
    Returns:
        Smoothed predictions
    """
    if len(predictions) < window:
        logger.warning(f"Predictions length {len(predictions)} < window {window}, returning original")
        return predictions
    
    if method == "ema":
        # Exponential Moving Average - vectorized implementation for better performance
        import pandas as pd
        smoothed = pd.Series(predictions).ewm(alpha=alpha, adjust=False).mean().values
        return smoothed
    
    elif method == "sma":
        # Simple Moving Average
        smoothed = np.copy(predictions)
        for i in range(window, len(predictions)):
            smoothed[i] = np.mean(predictions[i-window:i])
        return smoothed
    
    elif method == "median":
        # Median filter - more robust to outliers
        smoothed = np.copy(predictions)
        for i in range(window, len(predictions)):
            smoothed[i] = np.median(predictions[i-window:i])
        return smoothed
    
    else:
        logger.warning(f"Unknown smoothing method: {method}, returning original")
        return predictions


def ensemble_predictions(
    predictions_list: List[np.ndarray],
    method: str = "mean",
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Ensemble multiple predictions for better accuracy.
    
    Args:
        predictions_list: List of prediction arrays
        method: Ensemble method ('mean', 'median', 'weighted')
        weights: Weights for weighted averaging
        
    Returns:
        Ensembled predictions
    """
    if not predictions_list:
        raise ValueError("predictions_list cannot be empty")
    
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    predictions_array = np.array(predictions_list)
    
    if method == "mean":
        return np.mean(predictions_array, axis=0)
    elif method == "median":
        return np.median(predictions_array, axis=0)
    elif method == "weighted":
        if weights is None:
            logger.warning("No weights provided, using equal weights")
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        return np.average(predictions_array, axis=0, weights=weights)
    else:
        logger.warning(f"Unknown ensemble method: {method}, using mean")
        return np.mean(predictions_array, axis=0)


class ModelEvaluator:
    """Comprehensive model evaluation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize evaluator."""
        self.config = config or {}
        self.metrics_history = []

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Regression metrics
        metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred)
        metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
        metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
        metrics[f"{prefix}mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100

        # Custom metrics
        metrics[f"{prefix}max_error"] = np.max(np.abs(y_true - y_pred))
        metrics[f"{prefix}median_error"] = np.median(np.abs(y_true - y_pred))

        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics[f"{prefix}direction_accuracy"] = (
                np.mean(true_direction == pred_direction) * 100
            )

        self.metrics_history.append(metrics)

        return metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print metrics in a formatted way."""
        logger.info("=" * 50)
        logger.info("Model Evaluation Metrics")
        logger.info("=" * 50)

        for key, value in metrics.items():
            logger.info(f"{key:.<40} {value:.6f}")

        logger.info("=" * 50)

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot predictions against actual values."""
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series plot
        axes[0, 0].plot(y_true, label="Actual", alpha=0.7)
        axes[0, 0].plot(y_pred, label="Predicted", alpha=0.7)
        axes[0, 0].set_title("Time Series: Actual vs Predicted")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0, 1].set_title("Scatter: Predicted vs Actual")
        axes[0, 1].set_xlabel("Actual")
        axes[0, 1].set_ylabel("Predicted")
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals plot
        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_title("Residuals Plot")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals distribution
        axes[1, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1, 1].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 1].set_title("Residuals Distribution")
        axes[1, 1].set_xlabel("Residual Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.close()


class Backtester:
    """Backtesting framework for trading strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
    ):
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.reset()

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.returns = []

    def execute_trade(
        self, signal: int, price: float, timestamp: Optional[pd.Timestamp] = None
    ) -> None:
        """Execute a trade based on signal."""
        # signal: 1 = buy, -1 = sell, 0 = hold

        if signal == 1 and self.position == 0:
            # Buy
            shares = (self.capital * (1 - self.commission)) / (
                price * (1 + self.slippage)
            )
            cost = shares * price * (1 + self.slippage)
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost

            if total_cost <= self.capital:
                self.position = shares
                self.capital -= total_cost

                self.trades.append(
                    {
                        "timestamp": timestamp,
                        "action": "buy",
                        "price": price,
                        "shares": shares,
                        "cost": total_cost,
                    }
                )

        elif signal == -1 and self.position > 0:
            # Sell
            revenue = self.position * price * (1 - self.slippage)
            commission_cost = revenue * self.commission
            net_revenue = revenue - commission_cost

            self.capital += net_revenue

            self.trades.append(
                {
                    "timestamp": timestamp,
                    "action": "sell",
                    "price": price,
                    "shares": self.position,
                    "revenue": net_revenue,
                }
            )

            self.position = 0

        # Calculate current portfolio value
        portfolio_value = self.capital + (
            self.position * price if self.position > 0 else 0
        )
        self.portfolio_values.append(portfolio_value)

    def run_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, float]:
        """Run backtest with given prices and signals."""
        self.reset()

        if timestamps is None:
            timestamps = [None] * len(prices)

        for i, (price, signal, timestamp) in enumerate(
            zip(prices, signals, timestamps)
        ):
            self.execute_trade(signal, price, timestamp)

        # Close any open position at the end
        if self.position > 0:
            final_price = prices[-1]
            self.execute_trade(
                -1, final_price, timestamps[-1] if timestamps[-1] else None
            )

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()

        return metrics

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_values:
            return {}

        portfolio_values = np.asarray(self.portfolio_values, dtype=float)
        returns = (
            np.diff(portfolio_values) / portfolio_values[:-1]
            if len(portfolio_values) > 1
            else np.array([])
        )

        metrics: Dict[str, float] = {}
        metrics.update(self._calculate_return_metrics(portfolio_values))
        metrics.update(self._calculate_risk_metrics(returns))
        metrics.update(self._calculate_trade_metrics())

        return metrics

    def _calculate_return_metrics(self, portfolio_values: np.ndarray) -> Dict[str, float]:
        total_return_pct = (
            (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        )
        return {
            "total_return_pct": total_return_pct,
            "final_value": float(portfolio_values[-1]),
        }

    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        if returns.size == 0:
            return {}

        std = float(np.std(returns))
        volatility = std * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = (float(np.mean(returns)) / std) * np.sqrt(252) if std > 0 else 0

        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = float(np.min(drawdown)) * 100

        return {
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown_pct),
        }

    def _calculate_trade_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"num_trades": len(self.trades)}

        if len(self.trades) <= 1:
            return metrics

        profitable_trades = sum(
            1
            for i in range(0, len(self.trades) - 1, 2)
            if self.trades[i + 1]["revenue"] > self.trades[i]["cost"]
        )
        num_round_trips = len(self.trades) // 2
        metrics["win_rate_pct"] = (
            (profitable_trades / num_round_trips) * 100 if num_round_trips > 0 else 0
        )

        return metrics

    def plot_backtest_results(
        self,
        prices: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot backtest results."""
        _, axes = plt.subplots(2, 1, figsize=(15, 10))

        x_axis = timestamps if timestamps is not None else range(len(prices))

        # Price and trades
        axes[0].plot(x_axis, prices, label="Price", color="blue", alpha=0.7)

        # Mark buy and sell trades
        for trade in self.trades:
            if trade["action"] == "buy":
                idx = (
                    list(timestamps).index(trade["timestamp"])
                    if timestamps is not None
                    else 0
                )
                axes[0].scatter(
                    x_axis[idx],
                    trade["price"],
                    color="green",
                    marker="^",
                    s=100,
                    zorder=5,
                )
            elif trade["action"] == "sell":
                idx = (
                    list(timestamps).index(trade["timestamp"])
                    if timestamps is not None
                    else 0
                )
                axes[0].scatter(
                    x_axis[idx],
                    trade["price"],
                    color="red",
                    marker="v",
                    s=100,
                    zorder=5,
                )

        axes[0].set_title("Price and Trading Signals")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Portfolio value
        axes[1].plot(
            range(len(self.portfolio_values)),
            self.portfolio_values,
            label="Portfolio Value",
            color="green",
        )
        axes[1].axhline(
            y=self.initial_capital, color="r", linestyle="--", label="Initial Capital"
        )
        axes[1].set_title("Portfolio Value Over Time")
        axes[1].set_xlabel("Trade Number")
        axes[1].set_ylabel("Portfolio Value ($)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Backtest plot saved to {save_path}")

        plt.close()


def generate_simple_strategy_signals(
    predictions: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    """Generate simple trading signals based on predictions."""
    signals = np.zeros(len(predictions))

    for i in range(1, len(predictions)):
        # Buy if predicted to go up
        if predictions[i] > actual[i - 1]:
            signals[i] = 1
        # Sell if predicted to go down
        elif predictions[i] < actual[i - 1]:
            signals[i] = -1
        # Hold otherwise
        else:
            signals[i] = 0

    return signals
# — Raynergy-svg —
