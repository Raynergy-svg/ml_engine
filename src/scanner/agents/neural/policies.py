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

from src.scanner.agents._team import AgentDecisionContext, _safe_float, _last_value, _clip01
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


class RiskSentinelPolicy(NeuralAgentBase):
    """Learned risk sentinel.

    Replaces drawdown-ratio arithmetic with a policy that sees portfolio
    heat, drawdown, ATR and recent price action and learns when risk is
    actually dangerous vs merely elevated.
    """

    name = "neural_risk_sentinel"
    base_weight = 1.25

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        drawdown = _safe_float(ctx.analysis.drawdown)
        max_dd = max(_safe_float(getattr(ctx.config, "max_drawdown_pct", 0.025), 0.025), 1e-6)
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        risk_passed = 1.0 if bool(ctx.analysis.risk_passed) else 0.0
        trend_strength = _safe_float(ctx.analysis.trend_strength)
        entry_score = _safe_float(ctx.analysis.entry_score)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)

        returns = self._returns_window(ctx.df_raw, bars=5)

        features = np.concatenate([
            returns,
            np.array([
                drawdown / max_dd,
                risk_passed,
                atr_pips / 50.0,
                trend_strength,
                entry_score,
                direction,
                vol_pct,
                close / sma_20 - 1.0 if sma_20 > 0 else 0.0,
                _safe_float(ctx.analysis.confidence),
            ], dtype=np.float32),
        ])
        return features

    @staticmethod
    def _returns_window(df: pd.DataFrame, bars: int = 5) -> np.ndarray:
        if df is None or len(df) < 2:
            return np.zeros(bars, dtype=np.float32)
        returns = df["close"].pct_change().fillna(0).values.astype(np.float32)[-bars:]
        if len(returns) >= bars:
            return returns[-bars:]
        pad = bars - len(returns)
        return np.concatenate([np.zeros(pad, dtype=np.float32), returns], axis=0)


class UncertaintyPolicy(NeuralAgentBase):
    """Learned uncertainty / model-disagreement agent.

    Instead of counting heuristic oppositions, the policy sees confidence
    variance, price vs SMA divergence, and MACD/RSI/return signals and
    learns when the ensemble is truly conflicted.
    """

    name = "neural_uncertainty"
    base_weight = 1.10

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        macd_hist = _last_value(ctx.df_feat, "macd_hist", 0.0)
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        trend_strength = _safe_float(ctx.analysis.trend_strength)

        close_sma20 = 1.0 if close > sma_20 else -1.0 if close < sma_20 else 0.0
        close_sma50 = 1.0 if close > sma_50 else -1.0 if close < sma_50 else 0.0
        rsi_bias = 1.0 if rsi > 53.0 else -1.0 if rsi < 47.0 else 0.0
        macd_bias = 1.0 if macd_hist > 0 else -1.0 if macd_hist < 0 else 0.0

        features = np.array([
            confidence,
            abs(confidence - 0.62) / 0.20,
            (confidence - 0.60) / 0.10,
            close_sma20,
            close_sma50,
            macd_bias,
            rsi_bias,
            (rsi - 50.0) / 50.0,
            vol_pct,
            trend_strength,
            _safe_float(ctx.analysis.tcn_confidence),
            _safe_float(ctx.analysis.ridge_confidence) / 100.0,
        ], dtype=np.float32)
        return features


class ExecutionQualityPolicy(NeuralAgentBase):
    """Learned execution-quality agent.

    Sees spread, slippage, liquidity and session context and learns when
    execution conditions are genuinely poor vs acceptable.
    """

    name = "neural_execution_quality"
    base_weight = 1.05

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        spread_pips = _safe_float(ctx.analysis.spread_pips)
        if spread_pips <= 0.0:
            spread_pips = max(0.1, atr_pips * 0.04)
        slippage_pips = _safe_float(ctx.analysis.est_slippage_pips)
        if slippage_pips <= 0.0:
            slippage_pips = spread_pips * 0.45
        volume_ratio = _last_value(ctx.df_feat, "volume_ratio", 1.0)
        max_spread = max(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0), 0.1)
        max_slippage = max(_safe_float(getattr(ctx.config, "max_slippage_pips", 2.0), 2.0), 0.1)
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        features = np.array([
            spread_pips / max_spread,
            slippage_pips / max_slippage,
            volume_ratio,
            atr_pips / 50.0,
            vol_pct,
            direction,
            _safe_float(ctx.analysis.entry_score),
        ], dtype=np.float32)
        return features


class NewsRiskPolicy(NeuralAgentBase):
    """Learned news-risk agent.

    Does NOT call external RSS APIs at inference time.  Instead it uses
    time-of-day, pair-currency, volatility-regime and directional context
    as proxies — the network learns which sessions/regimes carry elevated
    headline risk from historical outcome labels.
    """

    name = "neural_news_risk"
    base_weight = 0.95

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour / 24.0
        weekday = now.weekday() / 7.0
        pair = str(getattr(ctx.analysis, "pair", "")).upper()
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        trend_strength = _safe_float(ctx.analysis.trend_strength)
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)

        london_ny_overlap = 1.0 if (13 <= now.hour < 16) else 0.0
        is_nfp_week = 1.0 if (now.day >= 1 and now.day <= 7 and now.weekday() == 4) else 0.0
        has_usd = 1.0 if "USD" in pair else 0.0
        has_eur = 1.0 if "EUR" in pair else 0.0
        has_gbp = 1.0 if "GBP" in pair else 0.0
        has_jpy = 1.0 if "JPY" in pair else 0.0

        features = np.array([
            hour,
            weekday,
            london_ny_overlap,
            is_nfp_week,
            has_usd,
            has_eur,
            has_gbp,
            has_jpy,
            direction,
            vol_pct,
            trend_strength,
            confidence,
            atr_pips / 50.0,
        ], dtype=np.float32)
        return features


class MultiTimeframePolicy(NeuralAgentBase):
    """Learned multi-timeframe confluence agent.

    Computes pseudo-H4 / H1 / M15 trend proxies from the available raw
    candles (no external API calls) and lets the network learn confluence.
    """

    name = "neural_multi_timeframe"
    base_weight = 1.10

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        df = ctx.df_raw
        close = _last_value(df, "close", _safe_float(ctx.analysis.current_price))
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        if df is None or len(df) < 20 or "close" not in df.columns:
            return np.array([
                direction,
                _safe_float(ctx.analysis.trend_strength),
                _safe_float(ctx.analysis.confidence),
                _safe_float(ctx.analysis.volatility_percentile),
            ], dtype=np.float32)

        closes = df["close"].values.astype(np.float32)

        short_sma = np.mean(closes[-20:]) if len(closes) >= 20 else close
        short_slope = (closes[-1] - closes[-min(len(closes), 20)]) / short_sma if short_sma > 0 else 0.0

        med_sma = np.mean(closes[-60:]) if len(closes) >= 60 else short_sma
        med_slope = (closes[-1] - closes[-min(len(closes), 60)]) / med_sma if med_sma > 0 else 0.0

        long_sma = np.mean(closes[-120:]) if len(closes) >= 120 else med_sma
        long_slope = (closes[-1] - closes[-min(len(closes), 120)]) / long_sma if long_sma > 0 else 0.0

        recent_high = np.max(closes[-60:]) if len(closes) >= 60 else np.max(closes)
        recent_low = np.min(closes[-60:]) if len(closes) >= 60 else np.min(closes)
        range_pct = (closes[-1] - recent_low) / (recent_high - recent_low + 1e-9)

        features = np.array([
            direction,
            closes[-1] / short_sma - 1.0,
            closes[-1] / med_sma - 1.0,
            closes[-1] / long_sma - 1.0,
            short_slope * 100.0,
            med_slope * 100.0,
            long_slope * 100.0,
            range_pct,
            _safe_float(ctx.analysis.trend_strength),
            _safe_float(ctx.analysis.confidence),
        ], dtype=np.float32)
        return features


class PairPerformancePolicy(NeuralAgentBase):
    """Learned pair-performance agent.

    Reads the pair-performance JSON file when available; otherwise uses
    pair metadata and recent trade-frequency proxies.
    """

    name = "neural_pair_performance"
    base_weight = 0.85

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        from pathlib import Path
        pair = str(getattr(ctx.analysis, "pair", "")).upper()
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        trend_strength = _safe_float(ctx.analysis.trend_strength)

        perf_path = Path("trained_data/models/pair_performance.json")
        win_rate = 0.5
        total_pnl = 0.0
        trades = 0
        if perf_path.exists():
            try:
                import json as _json
                data = _json.loads(perf_path.read_text())
                stats = data.get(pair)
                if stats:
                    win_rate = _safe_float(stats.get("win_rate"), 0.5)
                    total_pnl = _safe_float(stats.get("total_pnl_pips"), 0.0)
                    trades = int(stats.get("trades", 0))
            except Exception:
                pass

        features = np.array([
            win_rate,
            np.tanh(total_pnl / 100.0),
            min(trades / 50.0, 1.0),
            confidence,
            direction,
            vol_pct,
            trend_strength,
        ], dtype=np.float32)
        return features


class SessionTimingPolicy(NeuralAgentBase):
    """Learned session-timing agent.

    Replaces hard-coded London/NY overlap bonuses with a policy that
    learns which sessions are genuinely favorable per pair from outcomes.
    """

    name = "neural_session_timing"
    base_weight = 0.80

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()
        pair = str(getattr(ctx.analysis, "pair", "")).upper()
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        trend_strength = _safe_float(ctx.analysis.trend_strength)

        tokyo = 1.0 if (0 <= hour < 9) else 0.0
        london = 1.0 if (7 <= hour < 16) else 0.0
        ny = 1.0 if (13 <= hour < 22) else 0.0
        overlap = 1.0 if (13 <= hour < 16) else 0.0
        weekend = 1.0 if (weekday >= 5) else 0.0

        jpy = 1.0 if "JPY" in pair else 0.0
        eur_gbp = 1.0 if any(c in pair for c in ("EUR", "GBP", "CHF")) else 0.0
        usd_cad = 1.0 if any(c in pair for c in ("USD", "CAD")) else 0.0
        aud_nzd = 1.0 if any(c in pair for c in ("AUD", "NZD")) else 0.0

        features = np.array([
            hour / 24.0,
            weekday / 7.0,
            tokyo,
            london,
            ny,
            overlap,
            weekend,
            jpy,
            eur_gbp,
            usd_cad,
            aud_nzd,
            direction,
            vol_pct,
            trend_strength,
        ], dtype=np.float32)
        return features


class SupportResistancePolicy(NeuralAgentBase):
    """Learned support/resistance proximity agent.

    Computes swing pivots from recent candles and lets the network learn
    whether being near a level is supportive or dangerous for the trade.
    """

    name = "neural_support_resistance"
    base_weight = 1.00

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        df = ctx.df_raw
        if df is None or len(df) < 10 or "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            return np.zeros(8, dtype=np.float32)

        close = float(df["close"].iloc[-1])
        highs = df["high"].values[-60:] if len(df) >= 60 else df["high"].values
        lows = df["low"].values[-60:] if len(df) >= 60 else df["low"].values
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        pip_value = 0.01 if "JPY" in str(getattr(ctx.analysis, "pair", "")).upper() else 0.0001
        atr_price = atr_pips * pip_value

        resistance_levels = []
        support_levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] >= max(highs[i - 2], highs[i - 1], highs[i + 1], highs[i + 2]):
                resistance_levels.append(float(highs[i]))
            if lows[i] <= min(lows[i - 2], lows[i - 1], lows[i + 1], lows[i + 2]):
                support_levels.append(float(lows[i]))

        if not resistance_levels:
            resistance_levels = [float(max(highs[-20:]))]
        if not support_levels:
            support_levels = [float(min(lows[-20:]))]

        nearest_resistance = min(resistance_levels, key=lambda r: abs(r - close))
        nearest_support = min(support_levels, key=lambda s: abs(s - close))
        dist_res = (nearest_resistance - close) / atr_price if atr_price > 0 else 5.0
        dist_sup = (close - nearest_support) / atr_price if atr_price > 0 else 5.0

        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0
        range_pct = (close - min(lows)) / (max(highs) - min(lows) + 1e-9)

        features = np.array([
            dist_res / 10.0,
            dist_sup / 10.0,
            len(resistance_levels) / 10.0,
            len(support_levels) / 10.0,
            range_pct,
            direction,
            _safe_float(ctx.analysis.volatility_percentile),
            _safe_float(ctx.analysis.trend_strength),
        ], dtype=np.float32)
        return features


class OrderFlowPolicy(NeuralAgentBase):
    """Learned order-flow agent.

    No live OANDA API calls at inference.  Uses spread, volume, price
    velocity and volatility as proxies for order-flow toxicity.
    """

    name = "neural_order_flow"
    base_weight = 0.95

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        df = ctx.df_raw
        close = _last_value(df, "close", _safe_float(ctx.analysis.current_price))
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        spread_pips = _safe_float(ctx.analysis.spread_pips)
        if spread_pips <= 0.0:
            spread_pips = atr_pips * 0.04
        volume_ratio = _last_value(ctx.df_feat, "volume_ratio", 1.0)
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        velocity = np.zeros(5, dtype=np.float32)
        if df is not None and len(df) >= 6 and "close" in df.columns:
            closes = df["close"].values.astype(np.float32)
            diff = np.diff(closes[-6:])
            atr_price = atr_pips * (0.01 if "JPY" in str(getattr(ctx.analysis, "pair", "")).upper() else 0.0001)
            if atr_price > 0:
                velocity = diff / atr_price

        features = np.concatenate([
            velocity,
            np.array([
                spread_pips / atr_pips,
                volume_ratio,
                vol_pct,
                direction,
                _safe_float(ctx.analysis.entry_score),
                _safe_float(ctx.analysis.trend_strength),
            ], dtype=np.float32),
        ])
        return features


class TraderReadinessPolicy(NeuralAgentBase):
    """Learned trader-readiness agent (Aura bridge).

    Reads the readiness signal file if it exists; otherwise abstains
    gracefully with a neutral feature vector.
    """

    name = "neural_trader_readiness"
    base_weight = 0.50

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        from pathlib import Path
        signal_path = Path(".aura/bridge/readiness_signal.json")
        readiness = 0.5
        signal_age_hours = 999.0
        if signal_path.exists():
            try:
                import json as _json
                data = _json.loads(signal_path.read_text())
                readiness = _safe_float(data.get("readiness"), 0.5) / 100.0
                ts = data.get("timestamp")
                if ts:
                    from datetime import datetime, timezone
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        signal_age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
                    except Exception:
                        pass
            except Exception:
                pass

        features = np.array([
            readiness,
            min(signal_age_hours / 24.0, 1.0),
            1.0 if signal_age_hours > 24.0 else 0.0,
            _safe_float(ctx.analysis.confidence),
            _safe_float(ctx.analysis.volatility_percentile),
        ], dtype=np.float32)
        return features


class DevilsAdvocatePolicy(NeuralAgentBase):
    """Learned adversarial evaluator.

    Sees the same "bear factors" as the rule-based DA (S/R conflict,
    spread limit, regime mismatch, uncertainty, episodic warnings, R:R)
    but lets the network learn nonlinear combinations and thresholds.
    """

    name = "neural_devil_advocate"
    base_weight = 1.30

    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        spread_pips = _safe_float(ctx.analysis.spread_pips)
        if spread_pips <= 0.0:
            spread_pips = atr_pips * 0.04
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        vol_pct = _safe_float(ctx.analysis.volatility_percentile)
        vol_regime = str(getattr(ctx.analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        trend_strength = _safe_float(ctx.analysis.trend_strength)
        entry_score = _safe_float(ctx.analysis.entry_score)
        direction = 1.0 if getattr(ctx.analysis, "direction", "HOLD") == "LONG" else 0.0

        sr_conflict = abs(close / sma_20 - 1.0) * 100.0 if sma_20 > 0 else 0.0
        spread_limit = spread_pips / max(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0), 0.1)
        regime_mismatch = 1.0 if vol_regime == "EXTREME" else 0.5 if vol_regime == "HIGH" else 0.0
        uncertainty_high = abs(confidence - 0.62) / 0.20
        rr_marginal = max(0.0, 1.0 - entry_score)
        momentum_fade = 1.0 - trend_strength
        vol_spike = vol_pct

        features = np.array([
            sr_conflict,
            spread_limit,
            regime_mismatch,
            uncertainty_high,
            rr_marginal,
            momentum_fade,
            vol_spike,
            direction,
            confidence,
            vol_pct,
            trend_strength,
        ], dtype=np.float32)
        return features
