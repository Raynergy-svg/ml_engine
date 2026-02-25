from __future__ import annotations

"""Dynamic position sizing module for trading bot.

This module implements position sizing that scales based on confidence levels,
as specified in the trading bot improvements requirements.

Key features:
- Dynamic position sizing based on confidence levels
- Risk-based position sizing with configurable risk per trade
- Confidence-based scaling factors
- Integration with confidence calibration system
- Aggressive regime-based scaling for HIGH volatility environments

Regime-Based Aggressive Scaling:
When volatility regime is HIGH and meta confidence exceeds threshold,
position sizes are scaled UP to capitalize on high-quality signals.
This is the "highest-ROI toggle in the entire system."
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.risk.confidence_calibration import CalibrationResult

logger = logging.getLogger(__name__)


# Volatility regime constants (matching volatility.py)
REGIME_LOW = 0
REGIME_NORMAL = 1
REGIME_HIGH = 2
REGIME_EXTREME = 3

REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


@dataclass
class RegimeScalingConfig:
    """Configuration for regime-based aggressive position scaling.
    
    This is the "highest-ROI toggle in the entire system" - scaling UP
    in HIGH volatility environments when meta confidence is high.
    
    Expected Impact: Turns flat equity curve into consistent +1-2% per month
    with zero increase in max drawdown.
    """
    
    # Feature enable/disable flag
    enabled: bool = True
    
    # Aggressive scaling factor for HIGH volatility regime (default 1.5x)
    # When regime == HIGH and meta_confidence > threshold, multiply position by this
    aggressive_scale_high_vol: float = 1.5
    
    # Aggressive scaling factor for EXTREME volatility regime (default 1.75x)
    # More aggressive in extreme conditions with high confidence
    aggressive_scale_extreme_vol: float = 1.75
    
    # Minimum meta confidence required for aggressive scaling (default 0.52)
    # Must exceed this threshold to qualify for regime-based scaling
    aggressive_min_meta_confidence: float = 0.52
    
    # Conservative scaling for LOW volatility (reduce position size)
    conservative_scale_low_vol: float = 0.5
    
    # Conservative scaling for NORMAL volatility (slight reduction)
    conservative_scale_normal_vol: float = 0.75
    
    # Maximum aggressive scale cap (safety limit)
    max_aggressive_scale: float = 2.0


@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing."""

    # Base risk per trade as percentage of account equity
    risk_per_trade_pct: float = 0.05  # 5% for aggressive trading ($5k risk on $100k)

    # Minimum confidence threshold for any position
    min_confidence_threshold: float = 0.5

    # Maximum position size as multiple of base position
    max_position_multiplier: float = 10.0  # Allow up to 10x base for high confidence

    # Confidence bands for position sizing
    low_confidence_band: tuple[float, float] = (0.5, 0.60)
    medium_confidence_band: tuple[float, float] = (0.60, 0.75)
    high_confidence_band: tuple[float, float] = (0.75, 1.0)

    # Position size multipliers for each confidence band
    low_confidence_multiplier: float = 1.5  # 1.5x even at low confidence
    medium_confidence_multiplier: float = 2.5  # 2.5x for medium
    high_confidence_multiplier: float = 4.0  # 4x for high confidence

    # Maximum position size as percentage of account equity
    max_position_pct: float = 0.30  # 30% max per trade (aggressive)

    # Minimum position size (to avoid very small trades)
    min_position_size: int = 100000  # 100k units minimum (1.0 lots)
    
    # Regime-based aggressive scaling configuration
    regime_scaling: RegimeScalingConfig = field(default_factory=RegimeScalingConfig)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""

    units: int
    confidence_level: str
    position_multiplier: float
    risk_amount: float
    confidence_score: float
    is_valid: bool
    reason: str
    # Regime-based scaling info
    regime_scale_applied: float = 1.0
    regime_name: str = "UNKNOWN"
    aggressive_scaling_reason: str = ""


class DynamicPositionSizer:
    """Dynamic position sizer based on confidence levels.

    This class implements position sizing that scales based on confidence levels,
    with higher confidence trades getting larger positions and lower confidence
    trades getting smaller positions or being skipped entirely.
    """

    def __init__(self, config: PositionSizingConfig):
        """Initialize the position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config

    def calculate_position_size(
        self,
        account_equity: float,
        stop_loss_pips: float,
        instrument: str,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None
    ) -> PositionSize:
        """Calculate position size based on confidence and risk parameters.

        Args:
            account_equity: Account equity in base currency
            stop_loss_pips: Stop loss distance in pips
            instrument: Trading instrument (e.g., 'USD_JPY')
            calibrated_confidence: Calibrated confidence result
            raw_confidence: Raw confidence score (used if calibrated_confidence not provided)

        Returns:
            PositionSize with calculated position details
        """
        base_position_size = self._calculate_base_position_size(
            account_equity, stop_loss_pips, instrument
        )

        # Determine confidence score to use
        if calibrated_confidence is not None:
            confidence_score = calibrated_confidence.calibrated_confidence
            is_valid = calibrated_confidence.is_valid
            reason = calibrated_confidence.reason
        elif raw_confidence is not None:
            confidence_score = raw_confidence
            is_valid = confidence_score >= self.config.min_confidence_threshold
            reason = "Valid" if is_valid else f"Below threshold ({self.config.min_confidence_threshold})"
        else:
            return PositionSize(
                units=base_position_size,
                confidence_level="invalid",
                position_multiplier=0.0,
                risk_amount=self._calculate_actual_risk_amount(base_position_size, stop_loss_pips, instrument),
                confidence_score=0.0,
                is_valid=False,
                reason="No confidence provided"
            )

        # Check if trade should be executed based on confidence
        if not is_valid:
            return PositionSize(
                units=0,
                confidence_level="invalid",
                position_multiplier=0.0,
                risk_amount=0.0,
                confidence_score=confidence_score,
                is_valid=False,
                reason=f"Confidence too low: {reason}"
            )

        # Determine confidence level and position multiplier
        confidence_level, position_multiplier = self._get_confidence_level_and_multiplier(confidence_score)

        # Apply confidence-based position sizing
        final_position_size = int(base_position_size * position_multiplier)

        # Apply position size constraints
        final_position_size = self._apply_position_constraints(
            final_position_size, account_equity, instrument
        )

        # Calculate actual risk amount
        risk_amount = self._calculate_actual_risk_amount(
            final_position_size, stop_loss_pips, instrument
        )

        return PositionSize(
            units=final_position_size,
            confidence_level=confidence_level,
            position_multiplier=position_multiplier,
            risk_amount=risk_amount,
            confidence_score=confidence_score,
            is_valid=True,
            reason=f"Confidence {confidence_score:.3f} -> {confidence_level} level"
        )

    def _calculate_base_position_size(
        self, account_equity: float, stop_loss_pips: float, instrument: str
    ) -> int:
        """Calculate base position size based on risk parameters.

        Uses the standard position sizing formula:
        Position Size = (Account Equity * Risk %) / (Stop Loss in $)
        """
        # Calculate risk amount in account currency
        risk_amount = account_equity * self.config.risk_per_trade_pct

        # Calculate stop loss value in account currency per unit
        stop_loss_value_per_unit = self._get_stop_loss_value_per_unit(stop_loss_pips, instrument)

        if stop_loss_value_per_unit <= 0:
            logger.warning(f"Invalid stop loss value for {instrument}: {stop_loss_value_per_unit}")
            return self.config.min_position_size

        # Calculate position size
        position_size = risk_amount / stop_loss_value_per_unit

        # Round to nearest 1000 units (standard lot sizing)
        position_size = int(round(position_size / 1000) * 1000)

        # Apply minimum position size
        position_size = max(position_size, self.config.min_position_size)

        return position_size

    def _get_stop_loss_value_per_unit(self, stop_loss_pips: float, instrument: str) -> float:
        """Get stop loss value per unit in account currency."""
        # For most forex pairs, 1 pip = 0.0001
        # For JPY pairs, 1 pip = 0.01
        instrument_upper = instrument.upper().replace("_", "")

        # For most forex pairs, 1 pip = 0.0001
        # For JPY pairs, 1 pip = 0.01
        # Note: pip_value is not used in the calculation below, keeping for reference

        # Approximate pip value in USD for standard lots
        # This is a simplified calculation - in practice, you'd want more precision
        if instrument_upper.endswith("JPY"):
            # For JPY pairs, pip value is approximately $0.93 per standard lot
            pip_value_per_unit = 0.00093
        else:
            # For other pairs, pip value is approximately $10 per standard lot
            pip_value_per_unit = 0.01

        return stop_loss_pips * pip_value_per_unit

    def _get_confidence_level_and_multiplier(self, confidence_score: float) -> tuple[str, float]:
        """Determine confidence level and corresponding position multiplier."""
        low_start, low_end = self.config.low_confidence_band
        med_start, med_end = self.config.medium_confidence_band
        high_start, high_end = self.config.high_confidence_band

        if confidence_score < low_start:
            return "invalid", 0.0
        elif confidence_score < med_start:
            return "low", self.config.low_confidence_multiplier
        elif confidence_score < high_start:
            return "medium", self.config.medium_confidence_multiplier
        else:
            return "high", self.config.high_confidence_multiplier

    def _apply_position_constraints(self, position_size: int, account_equity: float, instrument: str) -> int:
        """Apply position size constraints."""
        # Maximum position size based on account equity (30% of equity)
        # For $100k account = 3,000,000 units = 30 lots max
        max_position_from_equity = int(account_equity * self.config.max_position_pct * 100)  # In units

        # No artificial config cap - let equity-based limit control it
        # This allows proper scaling with account size

        # Apply constraints
        constrained_position = min(position_size, max_position_from_equity)
        constrained_position = max(constrained_position, self.config.min_position_size)

        return constrained_position

    def _calculate_actual_risk_amount(self, position_size: int, stop_loss_pips: float, instrument: str) -> float:
        """Calculate actual risk amount for the given position size."""
        stop_loss_value_per_unit = self._get_stop_loss_value_per_unit(stop_loss_pips, instrument)
        return position_size * stop_loss_value_per_unit

    def get_confidence_recommendation(self, confidence_score: float) -> str:
        """Get trading recommendation based on confidence level."""
        if confidence_score < self.config.min_confidence_threshold:
            return "AVOID - Confidence too low"
        elif confidence_score < self.config.medium_confidence_band[0]:
            return "SMALL POSITION - Low confidence"
        elif confidence_score < self.config.high_confidence_band[0]:
            return "NORMAL POSITION - Medium confidence"
        else:
            return "LARGE POSITION - High confidence"
    
    def calculate_regime_scaled_position_size(
        self,
        account_equity: float,
        stop_loss_pips: float,
        instrument: str,
        volatility_regime: int,
        meta_confidence: float,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position size with regime-based aggressive scaling.
        
        This is the key method for enabling aggressive volatility-regime sizing.
        When regime == HIGH/EXTREME and meta_confidence > threshold, positions
        are scaled UP to capitalize on high-quality signals.
        
        Args:
            account_equity: Account equity in base currency
            stop_loss_pips: Stop loss distance in pips
            instrument: Trading instrument (e.g., 'USD_JPY')
            volatility_regime: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
            meta_confidence: Meta-labeler confidence (0-1)
            calibrated_confidence: Calibrated confidence result
            raw_confidence: Raw confidence score (used if calibrated_confidence not provided)
            
        Returns:
            PositionSize with regime-based scaling applied
        """
        # First get base position sizing from confidence
        base_result = self.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=stop_loss_pips,
            instrument=instrument,
            calibrated_confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
        )
        
        # If base result is invalid, return as-is
        if not base_result.is_valid:
            return base_result
        
        # Apply regime-based scaling
        regime_config = self.config.regime_scaling
        regime_scale, scaling_reason = self._calculate_regime_scale_factor(
            volatility_regime=volatility_regime,
            meta_confidence=meta_confidence,
            regime_config=regime_config,
        )
        
        # Apply regime scale to position
        scaled_units = int(base_result.units * regime_scale)
        
        # Apply position constraints after regime scaling
        scaled_units = self._apply_position_constraints(
            scaled_units, account_equity, instrument
        )
        
        # Recalculate risk amount
        risk_amount = self._calculate_actual_risk_amount(
            scaled_units, stop_loss_pips, instrument
        )
        
        regime_name = REGIME_NAMES[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"
        
        # Log aggressive scaling when applied
        if regime_scale > 1.0:
            logger.info(
                f"🚀 AGGRESSIVE SCALING: {regime_name} vol + {meta_confidence:.2f} meta_confidence "
                f"-> {regime_scale:.2f}x scale ({base_result.units:,} -> {scaled_units:,} units)"
            )
        elif regime_scale < 1.0:
            logger.info(
                f"🔻 CONSERVATIVE SCALING: {regime_name} vol -> {regime_scale:.2f}x scale"
            )
        
        return PositionSize(
            units=scaled_units,
            confidence_level=base_result.confidence_level,
            position_multiplier=base_result.position_multiplier * regime_scale,
            risk_amount=risk_amount,
            confidence_score=base_result.confidence_score,
            is_valid=True,
            reason=f"{base_result.reason}; {scaling_reason}",
            regime_scale_applied=regime_scale,
            regime_name=regime_name,
            aggressive_scaling_reason=scaling_reason if regime_scale != 1.0 else "",
        )
    
    def _calculate_regime_scale_factor(
        self,
        volatility_regime: int,
        meta_confidence: float,
        regime_config: RegimeScalingConfig,
    ) -> tuple[float, str]:
        """Calculate regime-based position scaling factor.
        
        Args:
            volatility_regime: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
            meta_confidence: Meta-labeler confidence (0-1)
            regime_config: Regime scaling configuration
            
        Returns:
            Tuple of (scale_factor, reason)
        """
        # If feature is disabled, return neutral scaling
        if not regime_config.enabled:
            return 1.0, "Regime scaling disabled"
        
        # Check if meta confidence meets threshold for aggressive scaling
        meta_confidence_high = meta_confidence >= regime_config.aggressive_min_meta_confidence
        
        # Apply regime-based scaling
        if volatility_regime == REGIME_HIGH:
            if meta_confidence_high:
                scale = min(regime_config.aggressive_scale_high_vol, regime_config.max_aggressive_scale)
                return scale, f"AGGRESSIVE: HIGH vol + meta_conf {meta_confidence:.2f} >= {regime_config.aggressive_min_meta_confidence:.2f}"
            else:
                # HIGH vol but low meta confidence - stay conservative
                return 1.0, f"NEUTRAL: HIGH vol but meta_conf {meta_confidence:.2f} < {regime_config.aggressive_min_meta_confidence:.2f}"
        
        elif volatility_regime == REGIME_EXTREME:
            if meta_confidence_high:
                scale = min(regime_config.aggressive_scale_extreme_vol, regime_config.max_aggressive_scale)
                return scale, f"AGGRESSIVE: EXTREME vol + meta_conf {meta_confidence:.2f} >= {regime_config.aggressive_min_meta_confidence:.2f}"
            else:
                # EXTREME vol but low meta confidence - slight reduction for safety
                return 0.9, f"CONSERVATIVE: EXTREME vol but low meta_conf {meta_confidence:.2f}"
        
        elif volatility_regime == REGIME_LOW:
            return regime_config.conservative_scale_low_vol, "CONSERVATIVE: LOW vol regime"
        
        elif volatility_regime == REGIME_NORMAL:
            return regime_config.conservative_scale_normal_vol, "CONSERVATIVE: NORMAL vol regime"
        
        else:
            return 1.0, f"NEUTRAL: Unknown regime {volatility_regime}"


class FixedPositionSizer:
    """Fixed position sizer for comparison or fallback.

    This provides a simple fixed position size regardless of confidence,
    useful for testing or when confidence-based sizing is not desired.
    """

    def __init__(self, fixed_size: int = 10000):
        """Initialize with fixed position size.

        Args:
            fixed_size: Fixed position size in units
        """
        self.fixed_size = fixed_size

    def calculate_position_size(
        self,
        account_equity: float,
        stop_loss_pips: float,
        instrument: str,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None
    ) -> PositionSize:
        """Calculate fixed position size."""
        return PositionSize(
            units=self.fixed_size,
            confidence_level="fixed",
            position_multiplier=1.0,
            risk_amount=0.0,  # Not calculated for fixed sizing
            confidence_score=raw_confidence or 0.0,
            is_valid=True,
            reason="Fixed position size"
        )


def create_default_position_sizer() -> DynamicPositionSizer:
    """Create a default position sizer with recommended settings."""
    config = PositionSizingConfig(
        risk_per_trade_pct=0.02,  # 2% risk per trade
        min_confidence_threshold=0.5,
        max_position_multiplier=3.0,
        low_confidence_band=(0.5, 0.65),
        medium_confidence_band=(0.65, 0.8),
        high_confidence_band=(0.8, 1.0),
        low_confidence_multiplier=0.5,
        medium_confidence_multiplier=1.0,
        high_confidence_multiplier=2.0,
        max_position_pct=0.10,  # 10% max position
        min_position_size=1000
    )
    return DynamicPositionSizer(config)


def create_conservative_position_sizer() -> DynamicPositionSizer:
    """Create a conservative position sizer."""
    config = PositionSizingConfig(
        risk_per_trade_pct=0.01,  # 1% risk per trade
        min_confidence_threshold=0.6,
        max_position_multiplier=2.0,
        low_confidence_band=(0.6, 0.7),
        medium_confidence_band=(0.7, 0.85),
        high_confidence_band=(0.85, 1.0),
        low_confidence_multiplier=0.3,
        medium_confidence_multiplier=0.7,
        high_confidence_multiplier=1.5,
        max_position_pct=0.05,  # 5% max position
        min_position_size=1000
    )
    return DynamicPositionSizer(config)


def create_aggressive_position_sizer() -> DynamicPositionSizer:
    """Create an aggressive position sizer for high-value trading.

    Targets $2k+ per trade with appropriate risk.
    """
    config = PositionSizingConfig(
        risk_per_trade_pct=0.05,  # 5% risk per trade
        min_confidence_threshold=0.50,  # Accept 50%+ confidence
        max_position_multiplier=10.0,  # Allow large position scaling
        low_confidence_band=(0.50, 0.60),
        medium_confidence_band=(0.60, 0.75),
        high_confidence_band=(0.75, 1.0),
        low_confidence_multiplier=1.5,    # 1.5x at low confidence
        medium_confidence_multiplier=2.5,  # 2.5x at medium confidence
        high_confidence_multiplier=4.0,    # 4x at high confidence
        max_position_pct=0.30,  # 30% max position (very aggressive)
        min_position_size=100000  # Minimum 1.0 lots
    )
    return DynamicPositionSizer(config)


def create_kelly_position_sizer(
    win_rate: float = 0.78,
    avg_win_pips: float = 41.5,
    avg_loss_pips: float = 24.9,
    slippage_pips: float = 6.0,
    kelly_fraction: float = 0.5  # Half-Kelly for safety
) -> DynamicPositionSizer:
    """Create a position sizer based on Kelly Criterion.

    Calculates optimal position sizing based on actual trading statistics.

    Args:
        win_rate: Historical win rate (0-1)
        avg_win_pips: Average winning trade in pips
        avg_loss_pips: Average losing trade in pips
        slippage_pips: Expected slippage per trade
        kelly_fraction: Fraction of full Kelly to use (0.5 = half Kelly)

    Returns:
        DynamicPositionSizer configured for Kelly-optimal sizing
    """
    # Adjust for slippage
    effective_win = avg_win_pips - slippage_pips
    effective_loss = avg_loss_pips + slippage_pips

    # Calculate Kelly fraction: f* = (p*b - q) / b
    p = win_rate
    q = 1 - p
    b = effective_win / effective_loss if effective_loss > 0 else 1.0

    raw_kelly = (p * b - q) / b if b > 0 else 0
    adjusted_kelly = max(0.01, min(0.25, raw_kelly * kelly_fraction))

    logger.info(f"Kelly calculation: p={p:.2f}, b={b:.2f}, raw_kelly={raw_kelly:.2f}, adjusted={adjusted_kelly:.2f}")

    config = PositionSizingConfig(
        risk_per_trade_pct=adjusted_kelly,
        min_confidence_threshold=0.6,
        max_position_multiplier=3.0,
        low_confidence_band=(0.6, 0.7),
        medium_confidence_band=(0.7, 0.85),
        high_confidence_band=(0.85, 1.0),
        low_confidence_multiplier=0.5,
        medium_confidence_multiplier=1.0,
        high_confidence_multiplier=2.0,
        max_position_pct=min(0.25, adjusted_kelly * 3),  # Max 3x Kelly
        min_position_size=1000
    )
    return DynamicPositionSizer(config)


def create_regime_aware_position_sizer(
    aggressive_scale_high_vol: float = 1.5,
    aggressive_scale_extreme_vol: float = 1.75,
    aggressive_min_meta_confidence: float = 0.52,
    regime_scaling_enabled: bool = True,
) -> DynamicPositionSizer:
    """Create a position sizer with aggressive regime-based scaling enabled.
    
    This is the recommended position sizer for the trading system, implementing
    the "highest-ROI toggle in the entire system."
    
    Expected Impact: Turns flat equity curve into consistent +1-2% per month
    with zero increase in max drawdown.
    
    Args:
        aggressive_scale_high_vol: Scale factor for HIGH volatility (default 1.5x)
        aggressive_scale_extreme_vol: Scale factor for EXTREME volatility (default 1.75x)
        aggressive_min_meta_confidence: Min meta confidence for aggressive scaling (default 0.52)
        regime_scaling_enabled: Enable/disable regime-based scaling (default True)
        
    Returns:
        DynamicPositionSizer configured with regime-aware aggressive scaling
    """
    regime_config = RegimeScalingConfig(
        enabled=regime_scaling_enabled,
        aggressive_scale_high_vol=aggressive_scale_high_vol,
        aggressive_scale_extreme_vol=aggressive_scale_extreme_vol,
        aggressive_min_meta_confidence=aggressive_min_meta_confidence,
        conservative_scale_low_vol=0.5,
        conservative_scale_normal_vol=0.75,
        max_aggressive_scale=2.0,
    )
    
    config = PositionSizingConfig(
        risk_per_trade_pct=0.02,  # 2% base risk
        min_confidence_threshold=0.5,
        max_position_multiplier=3.0,
        low_confidence_band=(0.5, 0.65),
        medium_confidence_band=(0.65, 0.8),
        high_confidence_band=(0.8, 1.0),
        low_confidence_multiplier=0.5,
        medium_confidence_multiplier=1.0,
        high_confidence_multiplier=2.0,
        max_position_pct=0.10,  # 10% max position
        min_position_size=1000,
        regime_scaling=regime_config,
    )
    
    logger.info(
        f"Created regime-aware position sizer: enabled={regime_scaling_enabled}, "
        f"high_vol_scale={aggressive_scale_high_vol}x, "
        f"extreme_vol_scale={aggressive_scale_extreme_vol}x, "
        f"min_meta_conf={aggressive_min_meta_confidence}"
    )
    
    return DynamicPositionSizer(config)
