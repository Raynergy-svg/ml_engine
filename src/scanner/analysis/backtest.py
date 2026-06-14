"""
Quick Backtest Module.

Provides rapid backtesting on recent candles to validate signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of quick backtest.

    Attributes:
        win_rate: Percentage of winning trades (0-1)
        sharpe: Annualized Sharpe ratio for H1 timeframe
        num_trades: Number of simulated trades
        total_pnl: Total PnL in percentage points
        max_drawdown: Maximum drawdown observed
        success: Whether backtest completed successfully
        error: Error message if failed
    """
    win_rate: float = 0.0
    sharpe: float = 0.0
    num_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    success: bool = False
    error: Optional[str] = None


class QuickBacktester:
    """Quick backtester for signal validation.

    Runs rapid backtests on recent candles to validate trading signals
    before execution. Calculates win rate, Sharpe ratio, and PnL.

    Features:
    - 50-candle default window (configurable)
    - Annualized Sharpe for H1 timeframe
    - Per-bar signal simulation
    - Drawdown tracking
    """

    def __init__(
        self,
        window: int = 50,
        min_trades: int = 10,
        granularity: str = "H1",
    ):
        """Initialize quick backtester.

        Args:
            window: Number of candles to backtest (default 50)
            min_trades: Minimum trades required for valid result
            granularity: Timeframe for Sharpe annualization
        """
        self.window = window
        self.min_trades = min_trades
        self.granularity = granularity

        # Annualization factor for Sharpe ratio
        # H1 = 24 bars/day * 252 trading days
        # M15 = 96 bars/day * 252 trading days
        self._annualization_factors = {
            "H1": np.sqrt(252 * 24),
            "H4": np.sqrt(252 * 6),
            "D": np.sqrt(252),
            "M15": np.sqrt(252 * 96),
            "M5": np.sqrt(252 * 288),
        }

    def run(
        self,
        df: pd.DataFrame,
        direction: str,
        window: Optional[int] = None,
    ) -> BacktestResult:
        """Run quick backtest on recent data.

        Simulates entering at each bar and holding for 1 bar in the
        predicted direction. Returns win rate, Sharpe, and other metrics.

        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            direction: Predicted direction ("LONG" or "SHORT")
            window: Override default window size

        Returns:
            BacktestResult with metrics
        """
        window = window or self.window

        # Validate input
        if len(df) < window + 10:
            return BacktestResult(
                success=False,
                error=f"Insufficient data ({len(df)} < {window + 10})"
            )

        if "close" not in df.columns:
            return BacktestResult(
                success=False,
                error="Missing 'close' column"
            )

        try:
            # Get recent slice with buffer
            df_backtest = df.tail(window + 10).copy()

            # Calculate returns
            df_backtest["returns"] = df_backtest["close"].pct_change()

            # Generate signal returns based on direction
            if direction.upper() == "LONG":
                df_backtest["signal_returns"] = df_backtest["returns"]
            else:
                df_backtest["signal_returns"] = -df_backtest["returns"]

            # Remove NaN and take window
            signal_returns = df_backtest["signal_returns"].dropna().tail(window)

            if len(signal_returns) < self.min_trades:
                return BacktestResult(
                    success=False,
                    error=f"Insufficient trades ({len(signal_returns)} < {self.min_trades})"
                )

            # Calculate metrics
            wins = (signal_returns > 0).sum()
            trades = len(signal_returns)
            win_rate = wins / trades if trades > 0 else 0.0

            # Total PnL
            total_pnl = signal_returns.sum()

            # Sharpe ratio (annualized)
            mean_ret = signal_returns.mean()
            std_ret = signal_returns.std()

            annualization = self._annualization_factors.get(
                self.granularity, np.sqrt(252 * 24)
            )
            sharpe = (mean_ret / std_ret) * annualization if std_ret > 0 else 0.0

            # Maximum drawdown
            cumulative = (1 + signal_returns).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

            return BacktestResult(
                win_rate=win_rate,
                sharpe=sharpe,
                num_trades=trades,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                success=True,
            )

        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return BacktestResult(
                success=False,
                error=str(e)
            )

    def stress_test(
        self,
        df: pd.DataFrame,
        direction: str,
        regime: str = "NORMAL",
        n_paths: int = 100,
    ) -> dict:
        """Run stress test using DiffusionMarketModel synthetic paths (US-009)."""
        try:
            from src.simulation.diffusion_market_model import DiffusionMarketModel, DiffusionConfig
            diffusion = DiffusionMarketModel(DiffusionConfig(random_seed=42))
            # Extract factor returns from price data
            returns = df["close"].pct_change().dropna().values
            if len(returns) < 20:
                return {"success": False, "error": "Insufficient data for diffusion"}
            # Fit diffusion model on returns
            factors = returns.reshape(-1, 1)
            diffusion.fit(factors)
            # Generate synthetic paths
            synthetic = diffusion.generate_paths(n_paths=n_paths, n_steps=len(returns))
            # Evaluate strategy on synthetic paths
            path_results = []
            for path in synthetic:
                path_returns = path.flatten()
                if direction.upper() == "LONG":
                    pnl = path_returns.sum()
                else:
                    pnl = (-path_returns).sum()
                path_results.append(pnl)
            return {
                "success": True,
                "regime": regime,
                "n_paths": n_paths,
                "mean_pnl": float(np.mean(path_results)),
                "std_pnl": float(np.std(path_results)),
                "var_95": float(np.percentile(path_results, 5)),
                "var_99": float(np.percentile(path_results, 1)),
                "max_pnl": float(np.max(path_results)),
                "min_pnl": float(np.min(path_results)),
            }
        except Exception as e:
            logger.warning("US-009: Stress test failed: %s", e)
            return {"success": False, "error": str(e)}

    def run_multiple(
        self,
        results: list,
        data_cache: dict,
    ) -> dict:
        """Run backtests on multiple pairs.

        Args:
            results: List of PairAnalysis results
            data_cache: Dict mapping pair -> DataFrame

        Returns:
            Dict mapping pair -> BacktestResult
        """
        backtest_results = {}

        for result in results:
            pair = result.pair
            if pair not in data_cache:
                backtest_results[pair] = BacktestResult(
                    success=False,
                    error="No data available"
                )
                continue

            df = data_cache[pair]
            direction = result.direction

            backtest_results[pair] = self.run(df, direction)

        return backtest_results
