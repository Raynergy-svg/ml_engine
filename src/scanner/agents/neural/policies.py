"""
Concrete neural policies for each specialist agent.

Each subclass defines:
  - `extract_features`: what raw / engineered features the policy sees
  - `name` and `base_weight`: identity in the voting system

The feature extractors deliberately overlap (e.g. trend and momentum both
see price history) because the *network* learns to specialize, not the
human designer.  This is the key difference from rule-based agents.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.scanner.agents._team import AgentDecisionContext, _safe_float, _last_value
from .neural_agent_base import NeuralAgentBase, NeuralAgentConfig

logger = logging.getLogger(__name__)


class TrendPolicy(NeuralAgentBase):
    """Learned trend agent.

    Instead of SMA crossover, the policy sees raw price history + volume
    and learns what trend alignment looks like for this pair/regime.
    """

    name = "neural_trend"
    base_weight = 1.15

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        # Raw OHLCV window (last 20 bars) — let the network discover patterns
        window = self._price_window(ctx.df_raw, bars=20)
        # Classic features as supplementary inputs (network can ignore them)
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        adx = _last_value(ctx.df_feat, "adx", 20.0)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        features = np.concatenate([
            window.flatten(),
            np.array([
                close / sma_20 - 1.0 if sma_20 > 0 else 0.0,
                sma_20 / sma_50 - 1.0 if sma_50 > 0 else 0.0,
                adx / 50.0,
                direction,
                _safe_float(ctx.analysis.atr_pips) / 50.0,
            ], dtype=np.float32),
        ])
        return features

    @staticmethod
    def _price_window(df: pd.DataFrame, bars: int = 20) -> np.ndarray:
        if df is None or len(df) < 2:
            return np.zeros((bars, 5), dtype=np.float32)
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        arr = df[cols].values.astype(np.float32)
        # Normalize by last close
        last_close = arr[-1, 3] if len(cols) >= 4 else 1.0
        if last_close <= 0:
            last_close = 1.0
        arr = arr / last_close - 1.0
        if len(arr) >= bars:
            return arr[-bars:]
        pad = bars - len(arr)
        return np.concatenate([np.zeros((pad, len(cols)), dtype=np.float32), arr], axis=0)


class MomentumPolicy(NeuralAgentBase):
    """Learned momentum agent.

    Instead of MACD histogram + ROC, the policy sees rate-of-change
    history and learns momentum follow-through patterns.
    """

    name = "neural_momentum"
    base_weight = 1.05

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        # Returns over last 20 bars
        window = self._returns_window(ctx.df_raw, bars=20)
        macd = _last_value(ctx.df_feat, "macd_hist", 0.0)
        roc = _last_value(ctx.df_feat, "roc_10", 0.0)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        features = np.concatenate([
            window,
            np.array([
                macd,
                roc / 100.0,
                direction,
                _safe_float(ctx.analysis.volatility_percentile),
                _safe_float(ctx.analysis.atr_pips) / 50.0,
            ], dtype=np.float32),
        ])
        return features

    @staticmethod
    def _returns_window(df: pd.DataFrame, bars: int = 20) -> np.ndarray:
        if df is None or len(df) < 2:
            return np.zeros(bars, dtype=np.float32)
        returns = df["close"].pct_change().fillna(0).values.astype(np.float32)[-bars:]
        if len(returns) >= bars:
            return returns[-bars:]
        pad = bars - len(returns)
        return np.concatenate([np.zeros(pad, dtype=np.float32), returns], axis=0)


class MeanReversionPolicy(NeuralAgentBase):
    """Learned mean-reversion agent.

    Instead of RSI thresholds, the policy sees deviation from recent mean
    and learns when pullbacks are likely to reverse.
    """

    name = "neural_mean_reversion"
    base_weight = 0.90

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        bb_upper = _last_value(ctx.df_feat, "bb_upper", close * 1.01)
        bb_lower = _last_value(ctx.df_feat, "bb_lower", close * 0.99)
        bb_width = (bb_upper - bb_lower) / close if close > 0 else 0.01
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        # Deviation from SMA window
        dev_window = self._deviation_window(ctx.df_raw, sma_20, bars=15)

        features = np.concatenate([
            dev_window,
            np.array([
                (rsi - 50.0) / 50.0,
                (close - sma_20) / sma_20 if sma_20 > 0 else 0.0,
                (close - sma_50) / sma_50 if sma_50 > 0 else 0.0,
                bb_width,
                direction,
                _safe_float(ctx.analysis.volatility_percentile),
            ], dtype=np.float32),
        ])
        return features

    @staticmethod
    def _deviation_window(df: pd.DataFrame, sma: float, bars: int = 15) -> np.ndarray:
        if df is None or len(df) < 2 or sma <= 0:
            return np.zeros(bars, dtype=np.float32)
        close = df["close"].values.astype(np.float32)
        dev = (close / sma - 1.0)[-bars:]
        if len(dev) >= bars:
            return dev[-bars:]
        pad = bars - len(dev)
        return np.concatenate([np.zeros(pad, dtype=np.float32), dev], axis=0)


class VolatilityPolicy(NeuralAgentBase):
    """Learned volatility agent.

    Instead of fixed ATR thresholds, the policy sees realized volatility
    history and learns when vol is supportive vs dangerous.
    """

    name = "neural_volatility"
    base_weight = 1.00

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        atr = _safe_float(ctx.analysis.atr_pips)
        vol_pct = _safe_float(ctx.analysis.volatility_percentile, 0.5)
        regime = str(getattr(ctx.analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        regime_onehot = [0.0, 0.0, 0.0, 0.0]
        for i, r in enumerate(["LOW", "NORMAL", "HIGH", "EXTREME"]):
            if r == regime:
                regime_onehot[i] = 1.0
                break

        # ATR history
        atr_window = self._atr_window(ctx.df_raw, bars=15)

        features = np.concatenate([
            atr_window,
            np.array([
                atr / 50.0,
                vol_pct,
                *regime_onehot,
                _safe_float(ctx.analysis.trend_strength),
                _safe_float(ctx.analysis.entry_score),
            ], dtype=np.float32),
        ])
        return features

    @staticmethod
    def _atr_window(df: pd.DataFrame, bars: int = 15) -> np.ndarray:
        if df is None or len(df) < 2:
            return np.zeros(bars, dtype=np.float32)
        high = df["high"].values.astype(np.float32)
        low = df["low"].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        # Simple ATR as rolling mean of TR
        atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values.astype(np.float32)
        atr = atr[-bars:]
        if len(atr) >= bars:
            return atr[-bars:]
        pad = bars - len(atr)
        return np.concatenate([np.zeros(pad, dtype=np.float32), atr], axis=0)
