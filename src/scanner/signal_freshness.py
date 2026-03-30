"""
Signal Freshness Guard — Phase 53 (US-330).

Recomputes RSI and ADX at execution time and compares to scan-time values.
If |delta| > 15 percentile points, the signal is flagged as stale.

Stale action: reduce confidence by 0.10 (configurable).
Reject if confidence drops below 0.45.

Tracks staleness rate (% of executions flagged stale, target <5%).

Usage:
    guard = SignalFreshnessGuard()
    result = guard.check(
        pair="EUR_USD",
        scan_rsi=65.0,
        scan_adx=28.0,
        current_prices=[...],  # Recent close prices for recomputation
        confidence=0.60,
    )
    if result.stale:
        confidence = result.adjusted_confidence
        if result.rejected:
            # Block trade
    # result.staleness_rate gives rolling rate
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FreshnessConfig:
    """Configuration for signal freshness guard."""
    stale_threshold: float = 15.0       # Percentile points delta to flag stale
    confidence_penalty: float = 0.10    # Confidence reduction on stale signal
    rejection_floor: float = 0.45       # Reject if confidence drops below this
    rsi_period: int = 14                # RSI calculation period
    adx_period: int = 14                # ADX calculation period
    staleness_window: int = 200         # Rolling window for rate tracking
    target_staleness_rate: float = 0.05 # Target <5% staleness


@dataclass
class FreshnessResult:
    """Result of signal freshness check."""
    stale: bool = False
    rejected: bool = False
    adjusted_confidence: float = 0.0
    confidence_penalty: float = 0.0
    scan_rsi: float = 0.0
    exec_rsi: float = 0.0
    rsi_delta: float = 0.0
    scan_adx: float = 0.0
    exec_adx: float = 0.0
    adx_delta: float = 0.0
    reason: str = ""
    staleness_rate: float = 0.0


class SignalFreshnessGuard:
    """Recomputes RSI/ADX at execution time to detect stale signals.

    Stale signals occur when market conditions change between scan time
    and execution time. Common during fast moves or delayed execution.
    """

    def __init__(self, config: Optional[FreshnessConfig] = None):
        self.config = config or FreshnessConfig()
        self._history: List[bool] = []  # True = stale, False = fresh

    @staticmethod
    def compute_rsi(prices: List[float], period: int = 14) -> float:
        """Compute RSI from a list of close prices.

        Args:
            prices: List of close prices (most recent last)
            period: RSI period (default 14)

        Returns:
            RSI value (0-100), or -1 if insufficient data
        """
        if len(prices) < period + 1:
            return -1.0

        arr = np.array(prices, dtype=np.float64)
        deltas = np.diff(arr)

        # Use last `period` deltas
        recent_deltas = deltas[-(period):]
        gains = np.where(recent_deltas > 0, recent_deltas, 0.0)
        losses = np.where(recent_deltas < 0, -recent_deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    @staticmethod
    def compute_adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> float:
        """Compute ADX from OHLC data.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: ADX period (default 14)

        Returns:
            ADX value (0-100), or -1 if insufficient data
        """
        n = len(closes)
        if n < period + 1 or len(highs) != n or len(lows) != n:
            return -1.0

        h = np.array(highs, dtype=np.float64)
        l = np.array(lows, dtype=np.float64)
        c = np.array(closes, dtype=np.float64)

        # True Range
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )

        # Directional movement
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with simple moving averages (approximation)
        if len(tr) < period:
            return -1.0

        atr = np.mean(tr[-period:])
        if atr == 0:
            return 0.0

        plus_di = 100.0 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100.0 * np.mean(minus_dm[-period:]) / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = 100.0 * abs(plus_di - minus_di) / di_sum
        return float(dx)

    def check(
        self,
        pair: str = "",
        scan_rsi: float = 0.0,
        scan_adx: float = 0.0,
        current_prices: Optional[List[float]] = None,
        current_highs: Optional[List[float]] = None,
        current_lows: Optional[List[float]] = None,
        confidence: float = 0.50,
    ) -> FreshnessResult:
        """Check if scan-time signals are still fresh.

        Args:
            pair: Instrument name (for logging)
            scan_rsi: RSI value at scan time
            scan_adx: ADX value at scan time
            current_prices: Recent close prices for RSI recomputation
            current_highs: Recent high prices for ADX recomputation
            current_lows: Recent low prices for ADX recomputation
            confidence: Current confidence score

        Returns:
            FreshnessResult with stale/rejected status and adjusted confidence
        """
        c = self.config
        result = FreshnessResult(
            adjusted_confidence=confidence,
            scan_rsi=scan_rsi,
            scan_adx=scan_adx,
        )

        # Recompute RSI
        rsi_stale = False
        if current_prices and scan_rsi > 0:
            exec_rsi = self.compute_rsi(current_prices, c.rsi_period)
            if exec_rsi >= 0:
                result.exec_rsi = exec_rsi
                result.rsi_delta = abs(exec_rsi - scan_rsi)
                if result.rsi_delta > c.stale_threshold:
                    rsi_stale = True

        # Recompute ADX
        adx_stale = False
        if current_prices and current_highs and current_lows and scan_adx > 0:
            exec_adx = self.compute_adx(
                current_highs, current_lows, current_prices, c.adx_period,
            )
            if exec_adx >= 0:
                result.exec_adx = exec_adx
                result.adx_delta = abs(exec_adx - scan_adx)
                if result.adx_delta > c.stale_threshold:
                    adx_stale = True

        # Determine staleness
        is_stale = rsi_stale or adx_stale
        result.stale = is_stale

        # Track staleness rate
        self._history.append(is_stale)
        if len(self._history) > c.staleness_window:
            self._history = self._history[-c.staleness_window:]
        result.staleness_rate = sum(self._history) / len(self._history) if self._history else 0.0

        if is_stale:
            result.confidence_penalty = c.confidence_penalty
            result.adjusted_confidence = confidence - c.confidence_penalty

            stale_reasons = []
            if rsi_stale:
                stale_reasons.append(f"RSI {scan_rsi:.1f}→{result.exec_rsi:.1f} (Δ{result.rsi_delta:.1f})")
            if adx_stale:
                stale_reasons.append(f"ADX {scan_adx:.1f}→{result.exec_adx:.1f} (Δ{result.adx_delta:.1f})")
            result.reason = f"STALE: {'; '.join(stale_reasons)}"

            # Reject if below floor
            if result.adjusted_confidence < c.rejection_floor:
                result.rejected = True
                result.reason += f" → REJECTED (conf {result.adjusted_confidence:.3f} < {c.rejection_floor})"

            logger.info(
                "Phase 53 (US-330): %s signal stale — %s (conf %.3f → %.3f, rate %.1f%%)",
                pair, result.reason, confidence, result.adjusted_confidence,
                result.staleness_rate * 100,
            )

        return result

    def get_staleness_rate(self) -> float:
        """Get current staleness rate."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def get_check_count(self) -> int:
        """Get total checks performed."""
        return len(self._history)
