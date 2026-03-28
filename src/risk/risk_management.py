"""Confidence-based risk management module for trading bot.

This module implements risk management features that adjust stop loss and take profit
levels based on confidence levels, as specified in the trading bot improvements requirements.

Key features:
- Confidence-based SL/TP adjustments
- Dynamic risk-reward ratios based on confidence
- Integration with confidence calibration system
- Position sizing integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.risk.confidence_calibration import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class RiskManagementConfig:
    """Configuration for confidence-based risk management.

    H1 Timeframe Best Practices (Professional FX Trading):
    - ATR Period: 14 (standard for H1)
    - SL Multiplier: 1.5-2.0x ATR (gives room for noise)
    - TP Multiplier: 2.0-3.0x ATR (2:1 minimum R:R)
    - Hold Time: 4-48 hours typical
    - Best Sessions: London/NY overlap (08:00-17:00 UTC)
    - Trailing Stop: Move to breakeven after 1R profit
    """

    # Base risk-reward ratios for different confidence levels
    # H1 OPTIMIZED: Wider targets suit swing trading style
    low_confidence_rr: float = 2.0  # 1:2 - minimum for H1 swing
    medium_confidence_rr: float = 2.5  # 1:2.5 - good edge
    high_confidence_rr: float = 3.0  # 1:3 - strong edge, let it run

    # Confidence thresholds for risk-reward adjustments
    low_confidence_threshold: float = 0.55  # Below 55% won't pass gates anyway
    medium_confidence_threshold: float = 0.65  # 65%+ gets better R:R
    high_confidence_threshold: float = 0.75  # 75%+ gets best R:R

    # Maximum and minimum allowed risk-reward ratios
    min_rr_ratio: float = 1.5  # Never go below 1.5:1 on H1
    max_rr_ratio: float = 5.0

    # ATR-based stop loss settings (H1 SPECIFIC)
    atr_period: int = 14  # Standard for H1
    atr_sl_multiplier: float = 1.5  # 1.5x ATR for SL (gives room for H1 noise)
    atr_tp_multiplier: float = 3.0  # 3.0x ATR for TP (2:1 R:R with 1.5 SL)

    # Stop loss adjustment factors based on confidence
    low_confidence_sl_multiplier: float = 1.3  # Wider stops for low confidence
    medium_confidence_sl_multiplier: float = 1.0  # Standard stops (1.5x ATR)
    high_confidence_sl_multiplier: float = 0.85  # Tighter stops for high confidence

    # Take profit adjustment factors based on confidence
    low_confidence_tp_multiplier: float = 0.8  # Conservative targets
    medium_confidence_tp_multiplier: float = 1.0  # Standard targets
    high_confidence_tp_multiplier: float = 1.5  # Let winners run

    # Maximum stop loss distance in pips (H1 can have larger moves)
    max_stop_loss_pips: float = 80.0  # EUR_USD H1 ATR ~10-15 pips, so max ~80
    min_stop_loss_pips: float = 15.0  # Don't go too tight on H1

    # Minimum take profit distance in pips
    min_take_profit_pips: float = 20.0  # At least 20 pips on H1

    # H1 Session filters (UTC hours)
    # Best trading: London open (08:00) to NY close (21:00)
    # Peak: London/NY overlap (13:00-17:00)
    enable_session_filter: bool = True
    active_sessions_utc: tuple = (8, 21)  # 08:00 - 21:00 UTC
    peak_sessions_utc: tuple = (13, 17)  # 13:00 - 17:00 UTC (overlap)

    # Trailing stop settings (move SL to breakeven after 1R)
    enable_trailing_stop: bool = True
    trailing_trigger_r: float = 1.0  # Move to BE after 1R profit
    trailing_distance_r: float = 0.5  # Trail 0.5R behind price

    # Maximum hold time in hours (H1 swing trades)
    max_hold_hours: int = 72  # 3 days max for H1 swing
    target_hold_hours: int = 24  # Target 24h hold (typical H1 swing)


@dataclass
class RiskManagementResult:
    """Result of risk management calculation."""

    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    confidence_level: str
    sl_adjustment_factor: float
    tp_adjustment_factor: float
    is_valid: bool
    reason: str


class ConfidenceBasedRiskManager:
    """Confidence-based risk manager.

    This class implements risk management that adjusts stop loss and take profit
    levels based on confidence levels, with higher confidence trades getting
    tighter stops and wider targets.
    """

    def __init__(self, config: RiskManagementConfig):
        """Initialize the risk manager.

        Args:
            config: Risk management configuration
        """
        self.config = config

    def calculate_risk_levels(
        self,
        entry_price: float,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None,
        base_stop_loss_pips: Optional[float] = None,
        base_take_profit_pips: Optional[float] = None,
        instrument: str = "USD_JPY"
    ) -> RiskManagementResult:
        """Calculate risk levels based on confidence.

        Args:
            entry_price: Entry price for the trade
            calibrated_confidence: Calibrated confidence result
            raw_confidence: Raw confidence score
            base_stop_loss_pips: Base stop loss distance in pips
            base_take_profit_pips: Base take profit distance in pips
            instrument: Trading instrument

        Returns:
            RiskManagementResult with calculated risk levels
        """
        # Determine confidence score to use
        if calibrated_confidence is not None:
            confidence_score = calibrated_confidence.calibrated_confidence
            is_valid = calibrated_confidence.is_valid
            reason = calibrated_confidence.reason
        elif raw_confidence is not None:
            confidence_score = raw_confidence
            is_valid = confidence_score >= self.config.low_confidence_threshold
            reason = "Valid" if is_valid else f"Below threshold ({self.config.low_confidence_threshold})"
        else:
            return RiskManagementResult(
                stop_loss_pips=0.0,
                take_profit_pips=0.0,
                risk_reward_ratio=0.0,
                confidence_level="invalid",
                sl_adjustment_factor=0.0,
                tp_adjustment_factor=0.0,
                is_valid=False,
                reason="No confidence provided"
            )

        # Check if trade should be executed based on confidence
        if not is_valid:
            return RiskManagementResult(
                stop_loss_pips=0.0,
                take_profit_pips=0.0,
                risk_reward_ratio=0.0,
                confidence_level="invalid",
                sl_adjustment_factor=0.0,
                tp_adjustment_factor=0.0,
                is_valid=False,
                reason=f"Confidence too low: {reason}"
            )

        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)

        # Get adjustment factors based on confidence
        sl_factor, tp_factor = self._get_adjustment_factors(confidence_level)

        # Calculate base risk-reward ratio
        base_rr = self._get_base_risk_reward_ratio(confidence_level)

        # Calculate stop loss and take profit
        stop_loss_pips, take_profit_pips = self._calculate_sl_tp(
            base_stop_loss_pips, base_take_profit_pips, sl_factor, tp_factor, base_rr
        )

        # Apply constraints
        stop_loss_pips = self._apply_sl_constraints(stop_loss_pips, instrument)
        # After stop-loss constraints, recompute TP to maintain the configured
        # risk-reward ratio for the confidence level (tests rely on this).
        target_rr = float(np.clip(base_rr, self.config.min_rr_ratio, self.config.max_rr_ratio))
        take_profit_pips = float(stop_loss_pips) * target_rr if float(stop_loss_pips) > 0 else 0.0
        take_profit_pips = self._apply_tp_constraints(take_profit_pips, instrument)

        # Calculate final risk-reward ratio — reject if SL invalid
        if stop_loss_pips <= 0:
            logger.warning(
                "R:R rejected: stop_loss_pips=%.4f for %s — invalid SL makes R:R undefined",
                stop_loss_pips, instrument,
            )
            return RiskManagementResult(
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
                risk_reward_ratio=0.0,
                confidence_level=confidence_level,
                sl_adjustment_factor=sl_factor,
                tp_adjustment_factor=tp_factor,
                is_valid=False,
                reason=f"Rejected: stop_loss_pips={stop_loss_pips:.4f} <= 0, R:R undefined",
            )
        final_rr = take_profit_pips / stop_loss_pips

        return RiskManagementResult(
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            risk_reward_ratio=final_rr,
            confidence_level=confidence_level,
            sl_adjustment_factor=sl_factor,
            tp_adjustment_factor=tp_factor,
            is_valid=True,
            reason=f"Confidence {confidence_score:.3f} -> {confidence_level} level"
        )

    def _get_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level based on score."""
        if confidence_score >= self.config.high_confidence_threshold:
            return "high"
        elif confidence_score >= self.config.medium_confidence_threshold:
            return "medium"
        elif confidence_score >= self.config.low_confidence_threshold:
            return "low"
        else:
            return "invalid"

    def _get_adjustment_factors(self, confidence_level: str) -> tuple[float, float]:
        """Get adjustment factors for stop loss and take profit based on confidence."""
        if confidence_level == "high":
            sl_factor = self.config.high_confidence_sl_multiplier
            tp_factor = self.config.high_confidence_tp_multiplier
        elif confidence_level == "medium":
            sl_factor = self.config.medium_confidence_sl_multiplier
            tp_factor = self.config.medium_confidence_tp_multiplier
        elif confidence_level == "low":
            sl_factor = self.config.low_confidence_sl_multiplier
            tp_factor = self.config.low_confidence_tp_multiplier
        else:
            sl_factor = 1.0
            tp_factor = 1.0

        return sl_factor, tp_factor

    def _get_base_risk_reward_ratio(self, confidence_level: str) -> float:
        """Get base risk-reward ratio for the confidence level."""
        return {
            "high": self.config.high_confidence_rr,
            "medium": self.config.medium_confidence_rr,
            "low": self.config.low_confidence_rr,
        }.get(confidence_level, 1.0)

    def _calculate_sl_tp(
        self,
        base_sl_pips: Optional[float],
        base_tp_pips: Optional[float],
        sl_factor: float,
        tp_factor: float,
        base_rr: float
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit based on base values and factors."""
        # If base values are provided, use them with adjustment factors
        if base_sl_pips is not None and base_tp_pips is not None:
            adjusted_sl = base_sl_pips * sl_factor
            adjusted_tp = base_tp_pips * tp_factor
        # If only base SL is provided, calculate TP based on risk-reward ratio
        elif base_sl_pips is not None:
            adjusted_sl = base_sl_pips * sl_factor
            adjusted_tp = adjusted_sl * base_rr
        # If only base TP is provided, calculate SL based on risk-reward ratio
        elif base_tp_pips is not None:
            adjusted_tp = base_tp_pips * tp_factor
            adjusted_sl = adjusted_tp / base_rr
        # If no base values provided, log WARNING and use configured minimum only.
        # The hardcoded 10.0 pip literal is removed — callers MUST provide ATR-based
        # SL/TP values. Only self.config.min_stop_loss_pips is used as the floor.
        else:
            logger.warning(
                "No base SL/TP values provided — ATR-based values are required. "
                "Falling back to config.min_stop_loss_pips only (no hardcoded pips)."
            )
            adjusted_sl = self.config.min_stop_loss_pips * sl_factor
            adjusted_tp = adjusted_sl * base_rr

        return adjusted_sl, adjusted_tp

    def _apply_sl_constraints(self, stop_loss_pips: float, instrument: str) -> float:
        """Apply constraints to stop loss."""
        # Apply maximum stop loss constraint
        stop_loss_pips = min(stop_loss_pips, self.config.max_stop_loss_pips)

        # Apply minimum stop loss based on instrument
        min_sl = self._get_minimum_stop_loss(instrument)
        stop_loss_pips = max(stop_loss_pips, min_sl)

        return stop_loss_pips

    def _apply_tp_constraints(self, take_profit_pips: float, instrument: str) -> float:
        """Apply constraints to take profit."""
        # Apply minimum take profit constraint
        take_profit_pips = max(take_profit_pips, self.config.min_take_profit_pips)

        return take_profit_pips

    def _get_minimum_stop_loss(self, instrument: str) -> float:
        """Get minimum stop loss for the instrument."""
        # For JPY pairs, minimum stop loss is typically higher due to volatility
        if "JPY" in instrument.upper():
            return 10.0
        else:
            return 5.0

    def get_risk_recommendation(self, confidence_score: float) -> str:
        """Get risk management recommendation based on confidence level."""
        confidence_level = self._get_confidence_level(confidence_score)

        return {
            "high": "AGGRESSIVE - Tight stops, wide targets, high R:R",
            "medium": "MODERATE - Standard stops and targets, balanced R:R",
            "low": "CONSERVATIVE - Wider stops, tighter targets, lower R:R",
        }.get(confidence_level, "AVOID - Confidence too low for risk management")


class FixedRiskManager:
    """Fixed risk manager for comparison or fallback.

    This provides fixed stop loss and take profit levels regardless of confidence,
    useful for testing or when confidence-based risk management is not desired.
    """

    def __init__(self, fixed_sl_pips: float = 20.0, fixed_tp_pips: float = 40.0):
        """Initialize with fixed risk levels.

        Args:
            fixed_sl_pips: Fixed stop loss in pips
            fixed_tp_pips: Fixed take profit in pips
        """
        self.fixed_sl_pips = fixed_sl_pips
        self.fixed_tp_pips = fixed_tp_pips

    def calculate_risk_levels(
        self,
        entry_price: float,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None,
        base_stop_loss_pips: Optional[float] = None,
        base_take_profit_pips: Optional[float] = None,
        instrument: str = "USD_JPY"
    ) -> RiskManagementResult:
        """Calculate fixed risk levels."""
        return RiskManagementResult(
            stop_loss_pips=self.fixed_sl_pips,
            take_profit_pips=self.fixed_tp_pips,
            risk_reward_ratio=self.fixed_tp_pips / self.fixed_sl_pips,
            confidence_level="fixed",
            sl_adjustment_factor=1.0,
            tp_adjustment_factor=1.0,
            is_valid=True,
            reason="Fixed risk levels"
        )


def create_default_risk_manager() -> ConfidenceBasedRiskManager:
    """Create a default risk manager with recommended settings."""
    config = RiskManagementConfig(
        low_confidence_rr=1.5,
        medium_confidence_rr=2.0,
        high_confidence_rr=3.0,
        low_confidence_threshold=0.5,
        medium_confidence_threshold=0.65,
        high_confidence_threshold=0.8,
        min_rr_ratio=1.0,
        max_rr_ratio=5.0,
        low_confidence_sl_multiplier=1.2,
        medium_confidence_sl_multiplier=1.0,
        high_confidence_sl_multiplier=0.8,
        low_confidence_tp_multiplier=0.8,
        medium_confidence_tp_multiplier=1.0,
        high_confidence_tp_multiplier=1.5,
        max_stop_loss_pips=100.0,
        min_take_profit_pips=5.0
    )
    return ConfidenceBasedRiskManager(config)


def create_conservative_risk_manager() -> ConfidenceBasedRiskManager:
    """Create a conservative risk manager."""
    config = RiskManagementConfig(
        low_confidence_rr=1.2,
        medium_confidence_rr=1.5,
        high_confidence_rr=2.0,
        low_confidence_threshold=0.6,
        medium_confidence_threshold=0.8,
        high_confidence_threshold=0.9,
        min_rr_ratio=1.0,
        max_rr_ratio=3.0,
        low_confidence_sl_multiplier=1.5,
        medium_confidence_sl_multiplier=1.1,
        high_confidence_sl_multiplier=0.9,
        low_confidence_tp_multiplier=0.7,
        medium_confidence_tp_multiplier=0.9,
        high_confidence_tp_multiplier=1.2,
        max_stop_loss_pips=50.0,
        min_take_profit_pips=10.0
    )
    return ConfidenceBasedRiskManager(config)
# — Raynergy-svg —
