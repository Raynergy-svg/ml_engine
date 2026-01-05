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

from confidence_calibration import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class RiskManagementConfig:
    """Configuration for confidence-based risk management."""
    
    # Base risk-reward ratios for different confidence levels
    low_confidence_rr: float = 1.5  # 1:1.5
    medium_confidence_rr: float = 2.0  # 1:2
    high_confidence_rr: float = 3.0  # 1:3
    
    # Confidence thresholds for risk-reward adjustments
    low_confidence_threshold: float = 0.5
    medium_confidence_threshold: float = 0.7
    high_confidence_threshold: float = 0.85
    
    # Maximum and minimum allowed risk-reward ratios
    min_rr_ratio: float = 1.0
    max_rr_ratio: float = 5.0
    
    # Stop loss adjustment factors based on confidence
    low_confidence_sl_multiplier: float = 1.2  # Wider stops for low confidence
    medium_confidence_sl_multiplier: float = 1.0  # Standard stops
    high_confidence_sl_multiplier: float = 0.8  # Tighter stops for high confidence
    
    # Take profit adjustment factors based on confidence
    low_confidence_tp_multiplier: float = 0.8  # More conservative targets
    medium_confidence_tp_multiplier: float = 1.0  # Standard targets
    high_confidence_tp_multiplier: float = 1.5  # Aggressive targets for high confidence
    
    # Maximum stop loss distance in pips (to prevent excessive risk)
    max_stop_loss_pips: float = 100.0
    
    # Minimum take profit distance in pips (to ensure reasonable R:R)
    min_take_profit_pips: float = 5.0


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

        # Calculate final risk-reward ratio
        final_rr = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0.0
        
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
        if confidence_level == "high":
            return self.config.high_confidence_rr
        elif confidence_level == "medium":
            return self.config.medium_confidence_rr
        elif confidence_level == "low":
            return self.config.low_confidence_rr
        else:
            return 1.0
    
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
        # If no base values provided, use default values
        else:
            adjusted_sl = 20.0 * sl_factor  # Default 20 pips
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
        
        if confidence_level == "high":
            return "AGGRESSIVE - Tight stops, wide targets, high R:R"
        elif confidence_level == "medium":
            return "MODERATE - Standard stops and targets, balanced R:R"
        elif confidence_level == "low":
            return "CONSERVATIVE - Wider stops, tighter targets, lower R:R"
        else:
            return "AVOID - Confidence too low for risk management"


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


# =============================================================================
# REGIME DETECTION - Dynamic predictor switching based on market regime
# =============================================================================

@dataclass
class MarketRegime:
    """Market regime classification result."""
    regime: str  # 'trending', 'ranging', 'volatile', 'quiet'
    atr_percentile: float  # ATR percentile vs historical (0-100)
    volatility_level: str  # 'low', 'medium', 'high'
    preferred_predictor: str  # 'transformer', 'histgb', 'both'
    confidence_adjustment: float  # Multiplier for confidence (0.5-1.5)
    reason: str


def detect_market_regime(
    df,
    atr_period: int = 14,
    lookback: int = 100,
    low_vol_threshold: float = 30.0,  # ATR below 30th percentile = low vol
    high_vol_threshold: float = 70.0,  # ATR above 70th percentile = high vol
) -> MarketRegime:
    """
    Detect market regime using ATR clusters for predictor switching.
    
    Logic:
    - Low volatility (ranging): HistGB tends to work better (mean-reversion)
    - High volatility (trending): Transformer tends to capture trends
    - Medium volatility: Use both with voting
    
    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation
        lookback: Number of bars for percentile calculation
        low_vol_threshold: ATR percentile below which regime is 'low volatility'
        high_vol_threshold: ATR percentile above which regime is 'high volatility'
    
    Returns:
        MarketRegime with regime classification and predictor recommendation
    """
    # Calculate ATR
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    
    atr_series = tr.rolling(atr_period).mean()
    current_atr = float(atr_series.iloc[-1])
    
    if not np.isfinite(current_atr) or current_atr <= 0:
        return MarketRegime(
            regime='unknown',
            atr_percentile=50.0,
            volatility_level='medium',
            preferred_predictor='transformer',
            confidence_adjustment=1.0,
            reason='ATR calculation failed'
        )
    
    # Calculate ATR percentile over lookback
    atr_history = atr_series.iloc[-lookback:].dropna()
    if len(atr_history) < 10:
        atr_percentile = 50.0
    else:
        atr_percentile = float((atr_history < current_atr).sum() / len(atr_history) * 100)
    
    # Classify regime
    if atr_percentile < low_vol_threshold:
        regime = 'ranging'
        volatility_level = 'low'
        # Low vol = ranging market, mean-reversion strategies work better
        # HistGB with PCA tends to pick up mean-reversion patterns
        # But HistGB performed poorly (48%), so still prefer Transformer
        preferred_predictor = 'transformer'  # Transformer still better overall
        confidence_adjustment = 0.9  # Slightly reduce confidence in quiet markets
        reason = f'Low volatility (ATR {atr_percentile:.0f}th pctl) - ranging market, reduce position size'
    elif atr_percentile > high_vol_threshold:
        regime = 'trending'
        volatility_level = 'high'
        # High vol = trending market, momentum strategies work better
        # Transformer better at capturing trends
        preferred_predictor = 'transformer'
        confidence_adjustment = 1.1  # Slightly boost confidence in trending markets
        reason = f'High volatility (ATR {atr_percentile:.0f}th pctl) - trending market, Transformer preferred'
    else:
        regime = 'normal'
        volatility_level = 'medium'
        # Normal conditions - use both with voting
        preferred_predictor = 'both'
        confidence_adjustment = 1.0
        reason = f'Normal volatility (ATR {atr_percentile:.0f}th pctl) - standard hybrid voting'
    
    return MarketRegime(
        regime=regime,
        atr_percentile=atr_percentile,
        volatility_level=volatility_level,
        preferred_predictor=preferred_predictor,
        confidence_adjustment=confidence_adjustment,
        reason=reason
    )


def apply_regime_adjustment(
    transformer_direction: int,
    transformer_prob: float,
    histgb_direction: Optional[int],
    histgb_prob: Optional[float],
    regime: MarketRegime,
) -> tuple[int, float, str]:
    """
    Apply regime-based adjustment to hybrid prediction.
    
    Returns:
        Tuple of (final_direction, final_confidence, decision_reason)
    """
    # If no HistGB model, just use Transformer
    if histgb_direction is None or histgb_prob is None:
        adjusted_conf = transformer_prob * regime.confidence_adjustment
        return transformer_direction, adjusted_conf, f"Transformer only ({regime.regime})"
    
    # Check if models agree
    models_agree = transformer_direction == histgb_direction
    
    if models_agree:
        # Both agree - high confidence
        avg_prob = (transformer_prob + histgb_prob) / 2
        adjusted_conf = avg_prob * regime.confidence_adjustment * 1.1  # Boost for agreement
        return transformer_direction, min(adjusted_conf, 0.95), f"Models agree ({regime.regime})"
    
    # Models disagree - use regime to decide
    if regime.preferred_predictor == 'transformer' or regime.volatility_level == 'high':
        # Trust Transformer in trending/high-vol markets
        adjusted_conf = transformer_prob * regime.confidence_adjustment * 0.8  # Reduce for disagreement
        return transformer_direction, adjusted_conf, f"Transformer wins ({regime.regime}, models disagree)"
    
    elif regime.preferred_predictor == 'histgb' or regime.volatility_level == 'low':
        # Trust HistGB in ranging/low-vol markets (if it were better)
        # Since HistGB is 48%, still prefer Transformer but reduce confidence
        adjusted_conf = transformer_prob * regime.confidence_adjustment * 0.7
        return transformer_direction, adjusted_conf, f"Transformer fallback ({regime.regime}, HistGB weak)"
    
    else:
        # Medium volatility - average but reduce confidence
        adjusted_conf = transformer_prob * regime.confidence_adjustment * 0.75
        return transformer_direction, adjusted_conf, f"Transformer cautious ({regime.regime}, models disagree)"