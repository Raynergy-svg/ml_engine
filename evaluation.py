"""
Model evaluation and backtesting framework for trading strategies.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


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
        # Exponential Moving Average - gives more weight to recent predictions
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]
        for i in range(1, len(predictions)):
            smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
        if len(self.portfolio_values) == 0:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = {}

        # Return metrics
        total_return = (
            (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        )
        metrics["total_return_pct"] = total_return
        metrics["final_value"] = portfolio_values[-1]

        # Risk metrics
        if len(returns) > 0:
            metrics["volatility"] = np.std(returns) * np.sqrt(252) * 100  # Annualized
            metrics["sharpe_ratio"] = (
                (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            metrics["max_drawdown_pct"] = np.min(drawdown) * 100

        # Trade metrics
        metrics["num_trades"] = len(self.trades)

        if len(self.trades) > 1:
            # Win rate
            profitable_trades = 0
            for i in range(0, len(self.trades), 2):
                if i + 1 < len(self.trades):
                    buy_trade = self.trades[i]
                    sell_trade = self.trades[i + 1]
                    if sell_trade["revenue"] > buy_trade["cost"]:
                        profitable_trades += 1

            metrics["win_rate_pct"] = (
                (profitable_trades / (len(self.trades) // 2)) * 100
                if len(self.trades) > 0
                else 0
            )

        return metrics

    def plot_backtest_results(
        self,
        prices: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

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
