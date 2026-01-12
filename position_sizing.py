from __future__ import annotations

"""Dynamic position sizing module for trading bot.

This module implements position sizing that scales based on confidence levels,
as specified in the trading bot improvements requirements.

Key features:
- Dynamic position sizing based on confidence levels
- Risk-based position sizing with configurable risk per trade
- Confidence-based scaling factors
- Integration with confidence calibration system
"""

import logging
from dataclasses import dataclass
from typing import Optional

from confidence_calibration import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing."""
    
    # Base risk per trade as percentage of account equity
    risk_per_trade_pct: float = 0.02  # 2%
    
    # Minimum confidence threshold for any position
    min_confidence_threshold: float = 0.5
    
    # Maximum position size as multiple of base position
    max_position_multiplier: float = 3.0
    
    # Confidence bands for position sizing
    low_confidence_band: tuple[float, float] = (0.5, 0.65)
    medium_confidence_band: tuple[float, float] = (0.65, 0.8)
    high_confidence_band: tuple[float, float] = (0.8, 1.0)
    
    # Position size multipliers for each confidence band
    low_confidence_multiplier: float = 0.5
    medium_confidence_multiplier: float = 1.0
    high_confidence_multiplier: float = 2.0
    
    # Maximum position size as percentage of account equity
    max_position_pct: float = 0.10  # 10%
    
    # Minimum position size (to avoid very small trades)
    min_position_size: int = 1000  # 1k units minimum


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
        # Maximum position size based on account equity
        max_position_from_equity = int(account_equity * self.config.max_position_pct / 1000) * 1000
        
        # Maximum position size based on configuration
        base_position = self.config.min_position_size
        max_position_from_config = int(base_position * self.config.max_position_multiplier)
        
        # Apply constraints
        max_allowed_position = min(max_position_from_equity, max_position_from_config)
        
        constrained_position = min(position_size, max_allowed_position)
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


def create_aggressive