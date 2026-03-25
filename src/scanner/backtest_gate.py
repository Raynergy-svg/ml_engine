"""
Pre-Trade Backtest Validation Gate.

Lightweight pre-execution validation gate that backtests the current signal
on recent candle data before committing to a trade. Implements SOFT gating:
reduces confidence by penalty if backtest win rate is below threshold, never
hard-blocks trades.

Pattern: Matches existing uncertainty agent soft penalty approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestGateResult:
    """Result of backtest gate validation.

    Attributes:
        passed: Whether backtest validation passed (win_rate >= min_win_rate)
        win_rate: Percentage of winning trades (0-1)
        trades_tested: Number of simulated trades in backtest
        confidence_adjustment: Adjustment to apply to final confidence score
        reason: Human-readable explanation of result
    """
    passed: bool
    win_rate: float
    trades_tested: int
    confidence_adjustment: float
    reason: str


class BacktestGate:
    """Pre-trade validation: runs quick backtest on recent candles to validate signal quality.

    Uses simple candle-by-candle forward-walk simulation:
    - For each of the last N candles, simulate entry at close with given SL/TP
    - Count winning vs losing trades
    - If win_rate >= min_win_rate: PASS (no confidence adjustment)
    - If win_rate < min_win_rate: SOFT FAIL (apply confidence penalty)

    This gate operates in SOFT mode: never hard-blocks trades, only adjusts confidence.
    This matches the pattern used by the uncertainty agent.
    """

    def __init__(
        self,
        min_win_rate: float = 0.55,
        lookback_candles: int = 20,
        confidence_penalty: float = 0.15,
    ):
        """Initialize backtest gate.

        Args:
            min_win_rate: Minimum acceptable win rate (0-1, default 55%)
            lookback_candles: Number of recent candles to backtest on (default 20)
            confidence_penalty: Confidence reduction if backtest fails (default 15%)
        """
        self.min_win_rate = min_win_rate
        self.lookback_candles = lookback_candles
        self.confidence_penalty = confidence_penalty

    def validate(
        self,
        pair: str,
        direction: str,
        sl_pips: float,
        tp_pips: float,
        recent_candles: Optional[pd.DataFrame] = None,
        pip_value: float = 0.0001,
    ) -> BacktestGateResult:
        """Run quick backtest on recent candles for this signal.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            direction: Trade direction ("LONG" or "SHORT")
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            recent_candles: DataFrame with OHLCV data (must have 'close' column)
            pip_value: Pip value for this pair (default 0.0001)

        Returns:
            BacktestGateResult with pass/fail and confidence adjustment
        """
        # No data available: PASS with warning (don't block on missing data)
        if recent_candles is None or len(recent_candles) == 0:
            return BacktestGateResult(
                passed=True,
                win_rate=0.5,  # Neutral
                trades_tested=0,
                confidence_adjustment=0.0,  # No adjustment on missing data
                reason=f"No candle data available for {pair}, skipping backtest validation",
            )

        # Insufficient data: PASS with warning
        if len(recent_candles) < 5:
            return BacktestGateResult(
                passed=True,
                win_rate=0.5,
                trades_tested=0,
                confidence_adjustment=0.0,
                reason=f"Insufficient candles ({len(recent_candles)} < 5), skipping backtest",
            )

        # Validate direction
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            return BacktestGateResult(
                passed=False,
                win_rate=0.0,
                trades_tested=0,
                confidence_adjustment=-self.confidence_penalty,
                reason=f"Invalid direction: {direction}",
            )

        # Validate SL/TP (CRITICAL-4: bound to reasonable ranges)
        if sl_pips <= 0 or sl_pips > 500:
            return BacktestGateResult(
                passed=True,
                win_rate=0.0,
                trades_tested=0,
                confidence_adjustment=0.0,
                reason=f"SL pips out of range ({sl_pips}), skipping backtest",
            )
        if tp_pips <= 0 or tp_pips > 1000:
            return BacktestGateResult(
                passed=True,
                win_rate=0.0,
                trades_tested=0,
                confidence_adjustment=0.0,
                reason=f"TP pips out of range ({tp_pips}), skipping backtest",
            )

        try:
            # Prepare data
            df = recent_candles.copy()
            if "close" not in df.columns:
                return BacktestGateResult(
                    passed=True,
                    win_rate=0.5,
                    trades_tested=0,
                    confidence_adjustment=0.0,
                    reason="Missing 'close' column in candle data",
                )

            # Simulate SL/TP in absolute price terms
            sl_price_offset = sl_pips * pip_value
            tp_price_offset = tp_pips * pip_value

            # Get recent candles (default 20)
            lookback = min(self.lookback_candles, len(df))
            df_backtest = df.tail(lookback).copy()

            # Simulate entry at each candle's close
            wins = 0
            losses = 0

            for i in range(len(df_backtest) - 1):
                entry_price = df_backtest.iloc[i]["close"]

                # Look at the next candle's high/low to see if we hit SL or TP
                next_candle = df_backtest.iloc[i + 1]
                candle_high = next_candle["high"] if "high" in next_candle else next_candle["close"]
                candle_low = next_candle["low"] if "low" in next_candle else next_candle["close"]

                # CRITICAL-3: Validate high >= low (corrupt candle check)
                if candle_high < candle_low:
                    logger.warning(f"Corrupt candle data: high ({candle_high}) < low ({candle_low}), skipping")
                    continue

                if direction == "LONG":
                    # For long: TP is above entry, SL is below entry
                    tp_level = entry_price + tp_price_offset
                    sl_level = entry_price - sl_price_offset

                    # CRITICAL-2: If both SL and TP hit in same candle, assume SL hit first (conservative)
                    if candle_low <= sl_level and candle_high >= tp_level:
                        losses += 1  # Conservative: assume SL first
                    elif candle_high >= tp_level:
                        wins += 1
                    elif candle_low <= sl_level:
                        losses += 1
                    # If neither, skip (incomplete candle)

                elif direction == "SHORT":
                    # For short: TP is below entry, SL is above entry
                    tp_level = entry_price - tp_price_offset
                    sl_level = entry_price + sl_price_offset

                    # CRITICAL-2: If both SL and TP hit in same candle, assume SL hit first (conservative)
                    if candle_high >= sl_level and candle_low <= tp_level:
                        losses += 1  # Conservative: assume SL first
                    elif candle_low <= tp_level:
                        wins += 1
                    elif candle_high >= sl_level:
                        losses += 1
                    # If neither, skip (incomplete candle)

            trades_tested = wins + losses

            # Not enough completed trades: PASS with neutral confidence
            if trades_tested < 3:
                return BacktestGateResult(
                    passed=True,
                    win_rate=0.5,
                    trades_tested=trades_tested,
                    confidence_adjustment=0.0,
                    reason=f"Insufficient completed trades ({trades_tested} < 3) in backtest",
                )

            win_rate = wins / trades_tested

            # Evaluate result
            if win_rate >= self.min_win_rate:
                return BacktestGateResult(
                    passed=True,
                    win_rate=win_rate,
                    trades_tested=trades_tested,
                    confidence_adjustment=0.0,
                    reason=f"Backtest passed: {wins}/{trades_tested} wins ({win_rate:.1%} >= {self.min_win_rate:.1%})",
                )
            else:
                return BacktestGateResult(
                    passed=False,
                    win_rate=win_rate,
                    trades_tested=trades_tested,
                    confidence_adjustment=-self.confidence_penalty,
                    reason=f"Backtest failed: {wins}/{trades_tested} wins ({win_rate:.1%} < {self.min_win_rate:.1%}) → -{self.confidence_penalty*100:.0f}% confidence penalty",
                )

        except Exception as e:
            logger.warning(f"Backtest validation failed for {pair}: {e}")
            # Graceful degradation: PASS with no adjustment (don't block on errors)
            return BacktestGateResult(
                passed=True,
                win_rate=0.5,
                trades_tested=0,
                confidence_adjustment=0.0,
                reason=f"Backtest error: {str(e)}, skipping validation",
            )
