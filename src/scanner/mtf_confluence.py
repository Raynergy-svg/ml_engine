"""
Multi-Timeframe Confluence Scoring System — Elder's Triple Screen Methodology

Implements a three-level timeframe analysis (Trend, Wave, Entry) with weighted
confluence scoring to filter entries and improve win rate.

Screens:
- Screen 1 (Trend):  4H — SMA alignment, ADX strength, slope magnitude
- Screen 2 (Wave):   1H — RSI pullback quality, momentum alignment
- Screen 3 (Entry):  15M — EMA alignment, candle quality, S/R proximity

Weights:    Screen1(50%) + Screen2(30%) + Screen3(20%)
Penalty:    -0.25 if screens disagree (misalignment)
Thresholds: PROCEED (>=0.65), CAUTION (>=0.45), REJECT (<0.45)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScreenResult:
    """Result from a single timeframe screen."""
    screen_name: str       # "trend_4h", "wave_1h", "entry_15m"
    direction: str         # "BULLISH", "BEARISH", "NEUTRAL"
    score: float           # 0.0 to 1.0
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for logging/serialization."""
        return {
            "screen_name": self.screen_name,
            "direction": self.direction,
            "score": float(self.score),
            "details": self.details,
        }


@dataclass
class ConfluenceResult:
    """Final confluence result combining all three screens."""
    confluence_score: float        # Weighted combination (0.0-1.0)
    direction_aligned: bool        # All non-NEUTRAL screens agree
    screen_results: List[ScreenResult]
    penalty_applied: float         # Misalignment penalty (-0.25 or 0.0)
    recommendation: str            # "PROCEED", "CAUTION", "REJECT"
    details: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for logging/serialization."""
        return {
            "confluence_score": float(self.confluence_score),
            "direction_aligned": bool(self.direction_aligned),
            "screen_results": [r.to_dict() for r in self.screen_results],
            "penalty_applied": float(self.penalty_applied),
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class MTFConfig:
    """Configuration for multi-timeframe confluence."""
    # Screen 1 (Trend) weights
    sma_alignment_weight: float = 0.40
    adx_weight: float = 0.30
    slope_weight: float = 0.30

    # Screen 2 (Wave) weights
    rsi_weight: float = 0.60
    momentum_weight: float = 0.40

    # Screen 3 (Entry) weights
    ema_weight: float = 0.40
    candle_quality_weight: float = 0.30
    sr_proximity_weight: float = 0.30

    # Timeframe weights
    trend_weight: float = 0.50
    wave_weight: float = 0.30
    entry_weight: float = 0.20

    # Thresholds
    proceed_threshold: float = 0.65
    caution_threshold: float = 0.45
    misalignment_penalty: float = -0.25

    # Indicator parameters
    sma_short_period: int = 50
    sma_long_period: int = 200
    rsi_period: int = 14
    ema_period: int = 20
    atr_period: int = 14
    adx_period: int = 14

    # RSI pullback thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Candle quality threshold (body/wick ratio)
    candle_body_ratio_threshold: float = 0.6

    # S/R proximity: within X ATRs
    sr_proximity_atr_multiple: float = 0.5


class MultiTimeframeConfluence:
    """
    Elder's Triple Screen confluence scoring system.

    Each screen produces a score (0.0-1.0) and direction (BULLISH/BEARISH/NEUTRAL).
    Final confluence is a weighted combination with penalty for misalignment.
    """

    def __init__(self, config: Optional[MTFConfig] = None):
        """Initialize with optional config."""
        self.config = config or MTFConfig()
        logger.debug(f"MTF Confluence initialized with config: {self.config}")

    def score_confluence(
        self,
        data_4h: pd.DataFrame,
        data_1h: pd.DataFrame,
        data_15m: pd.DataFrame,
    ) -> ConfluenceResult:
        """
        Score confluence across three timeframes using Elder's Triple Screen.

        Args:
            data_4h: 4H OHLCV data
            data_1h: 1H OHLCV data
            data_15m: 15M OHLCV data

        Returns:
            ConfluenceResult with score, recommendation, and screen details.
        """
        try:
            # Screen 1: 4H Trend
            screen1 = self._score_trend_screen(data_4h)
            if screen1 is None:
                logger.warning("Failed to score trend screen")
                return self._error_result("Insufficient data for trend screen")

            # Screen 2: 1H Wave (aligned with trend direction)
            screen2 = self._score_wave_screen(data_1h, screen1.direction)
            if screen2 is None:
                logger.warning("Failed to score wave screen")
                return self._error_result("Insufficient data for wave screen")

            # Screen 3: 15M Entry (aligned with trend direction)
            screen3 = self._score_entry_screen(data_15m, screen1.direction)
            if screen3 is None:
                logger.warning("Failed to score entry screen")
                return self._error_result("Insufficient data for entry screen")

            # Weighted confluence: Screen1(50%) + Screen2(30%) + Screen3(20%)
            raw_score = (
                screen1.score * self.config.trend_weight +
                screen2.score * self.config.wave_weight +
                screen3.score * self.config.entry_weight
            )

            # Direction alignment check: all non-NEUTRAL screens must agree
            directions = [
                s.direction for s in [screen1, screen2, screen3]
                if s.direction != "NEUTRAL"
            ]
            direction_aligned = len(set(directions)) <= 1

            # Misalignment penalty
            penalty = (
                self.config.misalignment_penalty
                if not direction_aligned else 0.0
            )
            final_score = max(0.0, min(1.0, raw_score + penalty))

            # Generate recommendation
            recommendation = self._get_recommendation(final_score, direction_aligned)

            return ConfluenceResult(
                confluence_score=final_score,
                direction_aligned=direction_aligned,
                screen_results=[screen1, screen2, screen3],
                penalty_applied=penalty,
                recommendation=recommendation,
                details={
                    "raw_score": float(raw_score),
                    "trend_direction": screen1.direction,
                    "aligned_directions": directions,
                },
            )

        except Exception as e:
            logger.error(f"Error in score_confluence: {e}", exc_info=True)
            return self._error_result(f"Exception: {str(e)}")

    def _score_trend_screen(self, data: pd.DataFrame) -> Optional[ScreenResult]:
        """
        Screen 1 — 4H Trend Analysis.

        Evaluates:
        - SMA alignment (price > SMA50 > SMA200 for bullish, inverse for bearish)
        - ADX strength (>25 trending, >40 strong, normalized 0-1)
        - Slope magnitude (normalized SMA slope)

        Score components:
        - sma_alignment: 0.4
        - adx_strength: 0.3
        - slope_magnitude: 0.3
        """
        # M-1: explicit empty check before any iloc usage
        if data is None or data.empty or len(data) < self.config.sma_long_period:
            return None

        try:
            # Calculate SMAs
            sma50 = data["close"].rolling(window=self.config.sma_short_period).mean()
            sma200 = data["close"].rolling(
                window=self.config.sma_long_period
            ).mean()

            # Get latest values
            latest_close = float(data["close"].iloc[-1])
            latest_sma50 = float(sma50.iloc[-1])
            latest_sma200 = float(sma200.iloc[-1])

            # SMA alignment score (0.0-1.0)
            if latest_sma50 > latest_sma200:
                # Bullish alignment
                sma_alignment = min(1.0, (latest_close / latest_sma50 - 0.95))
                direction = "BULLISH"
            elif latest_sma50 < latest_sma200:
                # Bearish alignment
                sma_alignment = min(1.0, (latest_sma50 / latest_close - 0.95))
                direction = "BEARISH"
            else:
                # Neutral
                sma_alignment = 0.5
                direction = "NEUTRAL"

            # ADX strength (normalized to 0-1)
            adx = self._calculate_adx(data, period=self.config.adx_period)
            if adx is None or np.isnan(adx):
                adx_strength = 0.5
            else:
                # ADX 0-40: scale to 0-1
                adx_strength = min(1.0, adx / 40.0)

            # SMA slope magnitude
            if len(sma50) >= 2 and not np.isnan(sma50.iloc[-1]):
                # M-2: use abs() on denominator to prevent extreme values when sma ≈ -1e-8
                slope_change = (sma50.iloc[-1] - sma50.iloc[-2]) / max(
                    abs(float(sma50.iloc[-2])), 1e-8
                )
                slope_magnitude = min(1.0, abs(slope_change) * 100)  # Normalize
            else:
                slope_magnitude = 0.5

            # Weighted score
            score = (
                sma_alignment * self.config.sma_alignment_weight +
                adx_strength * self.config.adx_weight +
                slope_magnitude * self.config.slope_weight
            )

            return ScreenResult(
                screen_name="trend_4h",
                direction=direction,
                score=float(np.clip(score, 0.0, 1.0)),
                details={
                    "sma_alignment": float(sma_alignment),
                    "adx_strength": float(adx_strength),
                    "slope_magnitude": float(slope_magnitude),
                    "sma50": float(latest_sma50),
                    "sma200": float(latest_sma200),
                    "close": float(latest_close),
                    "adx": float(adx) if not np.isnan(adx) else 0.0,
                },
            )

        except Exception as e:
            logger.error(f"Error in _score_trend_screen: {e}", exc_info=True)
            return None

    def _score_wave_screen(
        self,
        data: pd.DataFrame,
        trend_direction: str,
    ) -> Optional[ScreenResult]:
        """
        Screen 2 — 1H Wave Analysis (pullback quality within trend).

        Evaluates RSI position relative to trend direction:
        - BULLISH trend: RSI near oversold (30-45) = good wave entry
        - BEARISH trend: RSI near overbought (55-70) = good wave entry
        - NEUTRAL trend: RSI midrange (40-60) = okay entry

        Score components:
        - rsi_alignment: 0.6 (RSI pullback from extreme)
        - momentum_change: 0.4 (recent momentum shift rate)
        """
        if data is None or len(data) < self.config.rsi_period:
            return None

        try:
            # Calculate RSI
            rsi = self._calculate_rsi(data, period=self.config.rsi_period)
            if rsi is None or np.isnan(rsi):
                return None

            # RSI pullback quality: depends on trend direction
            if trend_direction == "BULLISH":
                # Good wave = oversold pullback (RSI 30-45)
                if rsi < self.config.rsi_oversold:
                    rsi_alignment = 0.9  # Very oversold, strong pullback
                elif rsi < 45:
                    rsi_alignment = 0.8  # Oversold zone
                elif rsi < 50:
                    rsi_alignment = 0.6  # Mild pullback
                else:
                    rsi_alignment = 0.3  # No pullback
                direction = "BULLISH"
            elif trend_direction == "BEARISH":
                # Good wave = overbought pullback (RSI 55-70)
                if rsi > self.config.rsi_overbought:
                    rsi_alignment = 0.9  # Very overbought, strong pullback
                elif rsi > 55:
                    rsi_alignment = 0.8  # Overbought zone
                elif rsi > 50:
                    rsi_alignment = 0.6  # Mild pullback
                else:
                    rsi_alignment = 0.3  # No pullback
                direction = "BEARISH"
            else:
                # NEUTRAL trend: middle range (40-60) is okay
                rsi_alignment = max(0.3, 1.0 - abs(rsi - 50) / 50.0)
                direction = "NEUTRAL"

            # Momentum change rate (how quickly is RSI moving?)
            if len(data) >= 2:
                rsi_vals = self._calculate_rsi(
                    data.iloc[-10:], period=min(self.config.rsi_period, 5)
                )
                if isinstance(rsi_vals, pd.Series) and len(rsi_vals) >= 2:
                    momentum_change = abs(rsi_vals.iloc[-1] - rsi_vals.iloc[-2]) / 50.0
                    momentum_change = min(1.0, momentum_change)
                else:
                    momentum_change = 0.5
            else:
                momentum_change = 0.5

            # Weighted score
            score = (
                rsi_alignment * self.config.rsi_weight +
                momentum_change * self.config.momentum_weight
            )

            return ScreenResult(
                screen_name="wave_1h",
                direction=direction,
                score=float(np.clip(score, 0.0, 1.0)),
                details={
                    "rsi": float(rsi),
                    "rsi_alignment": float(rsi_alignment),
                    "momentum_change": float(momentum_change),
                    "trend_direction": trend_direction,
                },
            )

        except Exception as e:
            logger.error(f"Error in _score_wave_screen: {e}", exc_info=True)
            return None

    def _score_entry_screen(
        self,
        data: pd.DataFrame,
        trend_direction: str,
    ) -> Optional[ScreenResult]:
        """
        Screen 3 — 15M Entry Confirmation (micro-trend alignment).

        Evaluates:
        - EMA20 alignment (price above EMA20 for bullish, below for bearish)
        - Candle quality (body-to-wick ratio >0.6)
        - S/R proximity (within 0.5 ATR of support/resistance)

        Score components:
        - ema_alignment: 0.4
        - candle_quality: 0.3
        - sr_proximity: 0.3
        """
        if data is None or len(data) < self.config.ema_period:
            return None

        try:
            # Calculate EMA20
            ema20 = data["close"].ewm(
                span=self.config.ema_period, adjust=False
            ).mean()
            latest_close = float(data["close"].iloc[-1])
            latest_ema20 = float(ema20.iloc[-1])

            # EMA alignment
            if trend_direction == "BULLISH":
                if latest_close > latest_ema20:
                    ema_alignment = min(1.0, (latest_close / latest_ema20 - 1.0) * 2)
                    direction = "BULLISH"
                else:
                    ema_alignment = 0.3
                    direction = "BEARISH"
            elif trend_direction == "BEARISH":
                if latest_close < latest_ema20:
                    ema_alignment = min(1.0, (latest_ema20 / latest_close - 1.0) * 2)
                    direction = "BEARISH"
                else:
                    ema_alignment = 0.3
                    direction = "BULLISH"
            else:
                ema_alignment = 0.5
                direction = "NEUTRAL"

            # Candle quality (body-to-wick ratio)
            candle_quality = self._score_candle_quality(
                data.iloc[-1], threshold=self.config.candle_body_ratio_threshold
            )

            # S/R proximity
            atr = self._calculate_atr(data, period=self.config.atr_period)
            if atr is None or np.isnan(atr):
                atr = 0.001
            sr_proximity = self._score_sr_proximity(
                data.iloc[-5:], atr, proximity_atr=self.config.sr_proximity_atr_multiple
            )

            # Weighted score
            score = (
                ema_alignment * self.config.ema_weight +
                candle_quality * self.config.candle_quality_weight +
                sr_proximity * self.config.sr_proximity_weight
            )

            return ScreenResult(
                screen_name="entry_15m",
                direction=direction,
                score=float(np.clip(score, 0.0, 1.0)),
                details={
                    "close": float(latest_close),
                    "ema20": float(latest_ema20),
                    "ema_alignment": float(ema_alignment),
                    "candle_quality": float(candle_quality),
                    "sr_proximity": float(sr_proximity),
                    "atr": float(atr),
                },
            )

        except Exception as e:
            logger.error(f"Error in _score_entry_screen: {e}", exc_info=True)
            return None

    # =========================================================================
    # Indicator Calculations
    # =========================================================================

    def _calculate_rsi(
        self, data: pd.DataFrame, period: int = 14
    ) -> Optional[float]:
        """Calculate RSI using Wilder's exponential smoothing."""
        if data is None or len(data) < period:
            return None

        try:
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).fillna(50.0).clip(0.01, 99.99)

            return float(rsi.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate ADX (Average Directional Index)."""
        if data is None or len(data) < period * 2:
            return None

        try:
            # True Range
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Directional movements
            plus_dm = data["high"].diff()
            minus_dm = -data["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # DI+ and DI-
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            # DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()

            return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else None
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate ATR (Average True Range)."""
        if data is None or len(data) < period:
            return None

        try:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None

    def _score_candle_quality(self, candle: pd.Series, threshold: float = 0.6) -> float:
        """
        Score candle quality based on body-to-wick ratio.

        Clean candles (ratio > threshold) score higher.
        Returns: 0.0-1.0
        """
        try:
            open_val = float(candle["open"])
            close_val = float(candle["close"])
            high_val = float(candle["high"])
            low_val = float(candle["low"])

            body = abs(close_val - open_val)
            wick = (high_val - low_val)

            if wick < 1e-8:
                return 0.5  # Doji-like, neutral

            ratio = body / wick
            if ratio >= threshold:
                return min(1.0, ratio)  # Clean candle
            else:
                return max(0.1, ratio)  # Weak candle

        except Exception as e:
            logger.error(f"Error scoring candle quality: {e}")
            return 0.5

    def _score_sr_proximity(
        self,
        data: pd.DataFrame,
        atr: float,
        proximity_atr: float = 0.5,
    ) -> float:
        """
        Score support/resistance proximity.

        Returns higher score if price is near (within proximity_atr * ATR)
        of a recent swing level.
        """
        try:
            if len(data) < 2 or atr < 1e-8:
                return 0.5

            latest_close = float(data["close"].iloc[-1])

            # Find recent swing high and low
            swing_high = float(data["high"].max())
            swing_low = float(data["low"].min())

            # Distance from price to nearest level
            dist_to_high = abs(latest_close - swing_high)
            dist_to_low = abs(latest_close - swing_low)
            min_dist = min(dist_to_high, dist_to_low)

            # Proximity threshold
            threshold = atr * proximity_atr

            # Score: closer to level = higher score
            if min_dist <= threshold:
                return min(1.0, 1.0 - (min_dist / threshold) * 0.5)
            else:
                return max(0.1, 1.0 - (min_dist / (threshold * 5)))

        except Exception as e:
            logger.error(f"Error scoring S/R proximity: {e}")
            return 0.5

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_recommendation(self, score: float, aligned: bool) -> str:
        """Generate recommendation based on score and alignment."""
        if score >= self.config.proceed_threshold and aligned:
            return "PROCEED"
        elif score >= self.config.caution_threshold:
            return "CAUTION"
        else:
            return "REJECT"

    def _error_result(self, reason: str) -> ConfluenceResult:
        """Return a neutral error result."""
        return ConfluenceResult(
            confluence_score=0.5,
            direction_aligned=False,
            screen_results=[],
            penalty_applied=0.0,
            recommendation="REJECT",
            details={"error": reason},
        )


def create_default_mtf_confluence() -> MultiTimeframeConfluence:
    """Factory: create MTF confluence with default config."""
    return MultiTimeframeConfluence(MTFConfig())
