"""
Training callbacks for RL modules.

Provides callbacks for monitoring training progress, early stopping
based on Sharpe ratio, and custom logging for trading metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

# Lazy import to avoid 8+ second startup penalty from stable-baselines3
SB3_AVAILABLE = None
BaseCallback = None


def _ensure_sb3_imported():
    """Lazy import stable-baselines3 only when needed."""
    global SB3_AVAILABLE, BaseCallback
    if SB3_AVAILABLE is None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
            BaseCallback = _BaseCallback
            SB3_AVAILABLE = True
        except ImportError:
            SB3_AVAILABLE = False
            BaseCallback = object
    return SB3_AVAILABLE


def _get_callback_base():
    """Get BaseCallback class lazily."""
    _ensure_sb3_imported()
    return BaseCallback if SB3_AVAILABLE else object


logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Custom callback for monitoring trading-specific metrics during RL training.

    Tracks:
    - Sharpe ratio
    - Win rate
    - Max drawdown
    - Average trade P/L
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        if SB3_AVAILABLE:
            super().__init__(verbose)

        self.log_freq = log_freq
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._trade_pnls: List[float] = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get info from environment
        infos = self.locals.get('infos', [{}])

        for info in infos:
            if 'pnl' in info:
                self._trade_pnls.append(info['pnl'])

        # Log at intervals
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout (data collection phase)."""
        if hasattr(self, 'logger') and self.logger is not None:
            # Log to TensorBoard if available
            if len(self._trade_pnls) > 0:
                self.logger.record('trading/avg_pnl', np.mean(self._trade_pnls))
                self.logger.record(
                    'trading/win_rate',
                    sum(1 for p in self._trade_pnls if p > 0) / len(self._trade_pnls))

                # Sharpe-like metric
                if np.std(self._trade_pnls) > 0:
                    sharpe = np.mean(self._trade_pnls) / np.std(self._trade_pnls)
                    self.logger.record('trading/sharpe', sharpe)

    def _log_metrics(self):
        """Log trading metrics to console."""
        if len(self._trade_pnls) < 5:
            return

        avg_pnl = np.mean(self._trade_pnls[-100:])
        win_rate = sum(1 for p in self._trade_pnls[-100:] if p > 0) / min(100, len(self._trade_pnls))

        if np.std(self._trade_pnls[-100:]) > 0:
            sharpe = np.mean(self._trade_pnls[-100:]) / np.std(self._trade_pnls[-100:])
        else:
            sharpe = 0.0

        if self.verbose > 0:
            logger.info(
                f"Step {self.n_calls}: "
                f"Trades={len(self._trade_pnls)}, "
                f"AvgPnL={avg_pnl:.4f}, "
                f"WR={win_rate:.1%}, "
                f"Sharpe={sharpe:.2f}"
            )


class SharpeStopCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Stop training when Sharpe ratio reaches a target threshold.

    Monitors the rolling Sharpe ratio of trade returns and stops
    training early when the target is consistently achieved.
    """

    def __init__(
        self,
        target_sharpe: float = 1.5,
        patience: int = 5,
        min_trades: int = 50,
        verbose: int = 1,
    ):
        """
        Args:
            target_sharpe: Target Sharpe ratio to achieve
            patience: Number of consecutive checks at/above target before stopping
            min_trades: Minimum trades before checking Sharpe
            verbose: Verbosity level
        """
        if SB3_AVAILABLE:
            super().__init__(verbose)

        self.target_sharpe = target_sharpe
        self.patience = patience
        self.min_trades = min_trades

        self._trade_pnls: List[float] = []
        self._consecutive_above_target = 0
        self._check_freq = 1000

    def _on_step(self) -> bool:
        """Check Sharpe ratio and potentially stop training."""
        # Collect P/L data
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'pnl' in info:
                self._trade_pnls.append(info['pnl'])

        # Check at intervals
        if self.n_calls % self._check_freq != 0:
            return True

        if len(self._trade_pnls) < self.min_trades:
            return True

        # Calculate Sharpe
        returns = np.array(self._trade_pnls[-200:])
        if np.std(returns) > 1e-8:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        if sharpe >= self.target_sharpe:
            self._consecutive_above_target += 1
            if self.verbose > 0:
                logger.info(
                    f"Sharpe {sharpe:.2f} >= {self.target_sharpe} "
                    f"({self._consecutive_above_target}/{self.patience})")

            if self._consecutive_above_target >= self.patience:
                logger.info(f"🎯 Target Sharpe {self.target_sharpe} achieved! Stopping training.")
                return False
        else:
            self._consecutive_above_target = 0

        return True


class SaveBestTradingCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Save model when best trading performance is achieved.

    Monitors a composite score of Sharpe, win rate, and drawdown.
    """

    def __init__(
        self,
        save_path: str,
        check_freq: int = 5000,
        verbose: int = 1,
    ):
        if SB3_AVAILABLE:
            super().__init__(verbose)

        self.save_path = Path(save_path)
        self.check_freq = check_freq

        self._trade_pnls: List[float] = []
        self._best_score = -np.inf
        self._best_sharpe = -np.inf
        self._equity_curve: List[float] = [1.0]

    def _on_step(self) -> bool:
        """Collect data and check for best performance."""
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'pnl' in info:
                self._trade_pnls.append(info['pnl'])
                self._equity_curve.append(self._equity_curve[-1] * (1 + info['pnl']))

        if self.n_calls % self.check_freq != 0:
            return True

        if len(self._trade_pnls) < 20:
            return True

        # Calculate metrics
        returns = np.array(self._trade_pnls[-200:])
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        # Max drawdown
        equity = np.array(self._equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # Composite score
        score = sharpe * 0.5 + win_rate * 0.3 - max_dd * 0.2

        if score > self._best_score:
            self._best_score = score
            self._best_sharpe = sharpe

            # Save model
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))

            if self.verbose > 0:
                logger.info(
                    f"💾 New best model saved: score={score:.3f}, "
                    f"sharpe={sharpe:.2f}, WR={win_rate:.1%}, DD={max_dd:.1%}")

        return True


class ProgressCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Simple progress callback for visual feedback during training.
    """

    def __init__(self, total_timesteps: int, update_freq: int = 10000):
        if SB3_AVAILABLE:
            super().__init__(verbose=1)

        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            progress = self.n_calls / self.total_timesteps * 100
            logger.info(f"Training progress: {progress:.1f}% ({self.n_calls}/{self.total_timesteps})")
        return True
