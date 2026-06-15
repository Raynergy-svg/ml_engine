"""
Parallel backtest harness: run legacy ensemble and SOTA raw-sequence model
on IDENTICAL historical bars.  Same data, same SL/TP rules, same costs.

For each bar we collect:
  - legacy_pred: direction + confidence from ModularEnsembleInference
  - sota_pred:   direction + confidence from SOTAInference
  - actual:      next-bar return (ground truth)

This is the ONLY way to answer "does it beat the existing ensemble?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BarPrediction:
    """Predictions from one model at one bar."""
    timestamp: str
    direction: str  # "LONG", "SHORT", "HOLD"
    confidence: float  # [0, 1]
    score: float  # raw model output (probability for LONG)


@dataclass
class ModelComparisonResult:
    """Outcome of running two models on the same historical bars."""
    pair: str
    n_bars: int
    legacy_predictions: List[BarPrediction] = field(default_factory=list)
    sota_predictions: List[BarPrediction] = field(default_factory=list)
    actual_returns: List[float] = field(default_factory=list)  # next-bar log-returns

    # Accuracy
    legacy_accuracy: float = 0.0
    sota_accuracy: float = 0.0
    legacy_long_accuracy: float = 0.0
    legacy_short_accuracy: float = 0.0
    sota_long_accuracy: float = 0.0
    sota_short_accuracy: float = 0.0

    # PnL (simple 1-bar hold, no SL/TP)
    legacy_pnl: float = 0.0
    sota_pnl: float = 0.0
    legacy_sharpe: float = 0.0
    sota_sharpe: float = 0.0
    legacy_max_dd: float = 0.0
    sota_max_dd: float = 0.0

    # Decorrelation
    error_correlation: float = 0.0
    prediction_correlation: float = 0.0
    win_overlap: float = 0.0  # fraction of bars where BOTH got it right

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "n_bars": self.n_bars,
            "legacy_accuracy": round(self.legacy_accuracy, 4),
            "sota_accuracy": round(self.sota_accuracy, 4),
            "legacy_pnl": round(self.legacy_pnl, 2),
            "sota_pnl": round(self.sota_pnl, 2),
            "legacy_sharpe": round(self.legacy_sharpe, 4),
            "sota_sharpe": round(self.sota_sharpe, 4),
            "legacy_max_dd": round(self.legacy_max_dd, 4),
            "sota_max_dd": round(self.sota_max_dd, 4),
            "error_correlation": round(self.error_correlation, 4),
            "prediction_correlation": round(self.prediction_correlation, 4),
            "win_overlap": round(self.win_overlap, 4),
        }


def _next_bar_log_return(df: pd.DataFrame, idx: int) -> float:
    """Return log-return of the next bar (for ground truth)."""
    if idx + 1 >= len(df):
        return 0.0
    close_now = float(df["close"].iloc[idx])
    close_next = float(df["close"].iloc[idx + 1])
    if close_now <= 0:
        return 0.0
    return float(np.log(close_next / close_now))


def _direction_correct(pred_dir: str, actual_ret: float) -> bool:
    """Was the predicted direction correct?"""
    if pred_dir == "LONG":
        return actual_ret > 0
    elif pred_dir == "SHORT":
        return actual_ret < 0
    return False  # HOLD is never "correct" in this framing


def _compute_accuracy(preds: List[BarPrediction], actuals: List[float]) -> Tuple[float, float, float]:
    """Compute overall, long-only, and short-only accuracy."""
    total = 0
    correct = 0
    long_total = 0
    long_correct = 0
    short_total = 0
    short_correct = 0
    for p, a in zip(preds, actuals):
        if p.direction in ("LONG", "SHORT"):
            total += 1
            if _direction_correct(p.direction, a):
                correct += 1
            if p.direction == "LONG":
                long_total += 1
                if a > 0:
                    long_correct += 1
            else:
                short_total += 1
                if a < 0:
                    short_correct += 1
    acc = correct / total if total > 0 else 0.0
    long_acc = long_correct / long_total if long_total > 0 else 0.0
    short_acc = short_correct / short_total if short_total > 0 else 0.0
    return acc, long_acc, short_acc


def _compute_pnl(preds: List[BarPrediction], actuals: List[float]) -> Tuple[float, float, float]:
    """Compute simple 1-bar PnL, Sharpe, and max drawdown.

    PnL logic: enter at bar close, hold 1 bar, exit at next close.
      LONG:  PnL = actual_ret (in log-return space)
      SHORT: PnL = -actual_ret
      HOLD:  PnL = 0
    """
    pnls = []
    for p, a in zip(preds, actuals):
        if p.direction == "LONG":
            pnls.append(a)
        elif p.direction == "SHORT":
            pnls.append(-a)
        else:
            pnls.append(0.0)

    pnls = np.array(pnls)
    total_pnl = float(pnls.sum())
    sharpe = 0.0
    if len(pnls) > 1:
        std = float(np.std(pnls, ddof=1))
        if std > 1e-8:
            sharpe = float(np.mean(pnls) / std) * np.sqrt(len(pnls))

    # Max drawdown from equity curve
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return total_pnl, sharpe, max_dd


def compare_models_on_history(
    pair: str,
    df: pd.DataFrame,
    legacy_signal_fn,  # Callable[[pd.DataFrame], BarPrediction]
    sota_signal_fn,     # Callable[[pd.DataFrame], BarPrediction]
    warmup_bars: int = 128,
) -> ModelComparisonResult:
    """Run both models on every bar of historical data and compare.

    Args:
        pair: Instrument name (e.g. "EUR_USD").
        df: DataFrame with OHLCV data.
        legacy_signal_fn: Function that takes df window and returns BarPrediction.
        sota_signal_fn:   Function that takes df window and returns BarPrediction.
        warmup_bars: Number of initial bars to skip (model needs history).

    Returns:
        ModelComparisonResult with accuracy, PnL, Sharpe, decorrelation metrics.
    """
    result = ModelComparisonResult(pair=pair, n_bars=len(df))

    if len(df) < warmup_bars + 10:
        logger.warning(f"{pair}: insufficient data ({len(df)} bars, need {warmup_bars + 10})")
        return result

    legacy_preds: List[BarPrediction] = []
    sota_preds: List[BarPrediction] = []
    actuals: List[float] = []

    for idx in range(warmup_bars, len(df) - 1):
        window = df.iloc[: idx + 1]
        ts = str(df.index[idx])

        try:
            leg = legacy_signal_fn(window)
        except Exception as e:
            logger.debug(f"{pair} legacy signal error at {ts}: {e}")
            leg = BarPrediction(timestamp=ts, direction="HOLD", confidence=0.0, score=0.5)

        try:
            sota = sota_signal_fn(window)
        except Exception as e:
            logger.debug(f"{pair} SOTA signal error at {ts}: {e}")
            sota = BarPrediction(timestamp=ts, direction="HOLD", confidence=0.0, score=0.5)

        actual = _next_bar_log_return(df, idx)

        legacy_preds.append(leg)
        sota_preds.append(sota)
        actuals.append(actual)

    # --- Accuracy ---
    result.legacy_accuracy, result.legacy_long_accuracy, result.legacy_short_accuracy = (
        _compute_accuracy(legacy_preds, actuals)
    )
    result.sota_accuracy, result.sota_long_accuracy, result.sota_short_accuracy = (
        _compute_accuracy(sota_preds, actuals)
    )

    # --- PnL ---
    result.legacy_pnl, result.legacy_sharpe, result.legacy_max_dd = _compute_pnl(legacy_preds, actuals)
    result.sota_pnl, result.sota_sharpe, result.sota_max_dd = _compute_pnl(sota_preds, actuals)

    # --- Decorrelation ---
    # Prediction correlation: do they agree on direction?
    leg_scores = np.array([1.0 if p.direction == "LONG" else (-1.0 if p.direction == "SHORT" else 0.0) for p in legacy_preds])
    sota_scores = np.array([1.0 if p.direction == "LONG" else (-1.0 if p.direction == "SHORT" else 0.0) for p in sota_preds])
    if np.std(leg_scores) > 0 and np.std(sota_scores) > 0:
        result.prediction_correlation = float(np.corrcoef(leg_scores, sota_scores)[0, 1])

    # Error correlation: do they make the SAME mistakes?
    leg_errors = np.array([1.0 if _direction_correct(p.direction, a) else 0.0 for p, a in zip(legacy_preds, actuals)])
    sota_errors = np.array([1.0 if _direction_correct(p.direction, a) else 0.0 for p, a in zip(sota_preds, actuals)])
    if np.std(leg_errors) > 0 and np.std(sota_errors) > 0:
        result.error_correlation = float(np.corrcoef(leg_errors, sota_errors)[0, 1])

    # Win overlap: fraction of bars where BOTH got it right
    both_correct = np.logical_and(leg_errors > 0.5, sota_errors > 0.5)
    result.win_overlap = float(both_correct.sum() / len(both_correct)) if len(both_correct) > 0 else 0.0

    result.legacy_predictions = legacy_preds
    result.sota_predictions = sota_preds
    result.actual_returns = actuals

    logger.info(
        f"{pair}: legacy_acc={result.legacy_accuracy:.3f} sota_acc={result.sota_accuracy:.3f} "
        f"legacy_pnl={result.legacy_pnl:.3f} sota_pnl={result.sota_pnl:.3f} "
        f"err_corr={result.error_correlation:.3f} pred_corr={result.prediction_correlation:.3f}"
    )
    return result
