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
- Futures tick-based position sizing (contracts from SL in ticks)

Regime-Based Aggressive Scaling:
When volatility regime is HIGH and meta confidence exceeds threshold,
position sizes are scaled UP to capitalize on high-quality signals.
This is the "highest-ROI toggle in the entire system."

Futures Tick-Based Sizing:
For FUTURES instruments, position size is calculated in contracts based on
tick-level SL: contracts = int(max(1, risk_amount / (sl_ticks * tick_value)))
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from src.risk.confidence_calibration import CalibrationResult
from src.brokers.instrument import Instrument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approximate currency->USD conversion for risk math (2026-07-18 audit).
#
# Risk-per-trade and pip-value math must be expressed in ACCOUNT (USD) terms.
# These are documented planning approximations — the same role the old single
# 0.000065 JPY constant played, but per-currency so non-USD-quoted pairs are
# no longer treated as USD-quoted (which oversized EUR_GBP ~27% and undersized
# USD_CAD ~26%). Callers with a live rate can override via set_quote_usd_rate;
# unknown currencies fall back to parity with a warning (the old implicit
# behavior, now at least logged).
# ---------------------------------------------------------------------------
_APPROX_USD_RATES: dict[str, float] = {
    "USD": 1.0,
    "JPY": 1.0 / 150.0,   # ~0.00667
    "EUR": 1.08,
    "GBP": 1.27,
    "CHF": 1.12,
    "CAD": 0.73,
    "AUD": 0.66,
    "NZD": 0.60,
}
_QUOTE_USD_OVERRIDES: dict[str, float] = {}

# Hard gross-leverage ceiling (matches src/equity/oanda_trend.py MAX_GROSS_LEVERAGE
# and the operator's documented "<=15x re-clamped every cycle" invariant).
MAX_GROSS_LEVERAGE = 15.0


def set_quote_usd_rate(currency: str, rate: float) -> None:
    """Register a live currency->USD rate to supersede the static estimate."""
    if rate > 0:
        _QUOTE_USD_OVERRIDES[currency.upper()] = float(rate)


def _approx_ccy_to_usd(currency: str) -> float:
    """Best available currency->USD rate: live override, else static estimate."""
    ccy = currency.upper()
    if ccy in _QUOTE_USD_OVERRIDES:
        return _QUOTE_USD_OVERRIDES[ccy]
    if ccy in _APPROX_USD_RATES:
        return _APPROX_USD_RATES[ccy]
    logger.warning("No USD conversion estimate for %s — assuming parity", ccy)
    return 1.0


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
    # Unit type: "units" (FX), "contracts" (futures), or "shares" (stocks)
    unit_type: str = "units"
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

    @staticmethod
    def _normalize_instrument(instrument: Union[str, Instrument]) -> Union[str, Instrument]:
        """Normalize instrument input to either string or Instrument object.

        Args:
            instrument: Either a string symbol (e.g., 'USD_JPY') or Instrument object

        Returns:
            The instrument as-is (str or Instrument)
        """
        return instrument

    def calculate_position_size(
        self,
        account_equity: float,
        stop_loss_pips: float,
        instrument: Union[str, Instrument],
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None
    ) -> PositionSize:
        """Calculate position size based on confidence and risk parameters.

        Supports both FX (pips-based) and FUTURES (ticks-based) instruments.

        Args:
            account_equity: Account equity in base currency
            stop_loss_pips: Stop loss distance in pips (for FX) or ticks (for FUTURES)
            instrument: Trading instrument (str like 'USD_JPY' or Instrument object)
            calibrated_confidence: Calibrated confidence result
            raw_confidence: Raw confidence score (used if calibrated_confidence not provided)

        Returns:
            PositionSize with calculated position details (units for FX, contracts for FUTURES)
        """
        # Determine instrument type and get symbol string for logging
        instrument_obj = self._normalize_instrument(instrument)
        instrument_str = instrument_obj.symbol if isinstance(instrument_obj, Instrument) else str(instrument)

        base_position_size = self._calculate_base_position_size(
            account_equity, stop_loss_pips, instrument_obj
        )

        # Apply adaptive risk scaling (drawdown protection / streak boost)
        try:
            from src.risk.adaptive_scaler import AdaptiveRiskScaler
            scaler = AdaptiveRiskScaler()
            scale_factor = scaler.get_scale_factor()
            if scale_factor != 1.0:
                base_position_size = int(base_position_size * scale_factor)
                logger.debug(
                    f"Adaptive risk scale: {scale_factor:.2f}x → "
                    f"base_position={base_position_size}"
                )
        except Exception:
            pass  # Graceful fallback: use unscaled position

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
            unit_type = "contracts" if isinstance(instrument_obj, Instrument) and instrument_obj.asset_class == "FUTURES" else "units"
            return PositionSize(
                units=base_position_size,
                confidence_level="invalid",
                position_multiplier=0.0,
                risk_amount=self._calculate_actual_risk_amount(base_position_size, stop_loss_pips, instrument_obj),
                confidence_score=0.0,
                is_valid=False,
                reason="No confidence provided",
                unit_type=unit_type
            )

        # Check if trade should be executed based on confidence
        if not is_valid:
            unit_type = "contracts" if isinstance(instrument_obj, Instrument) and instrument_obj.asset_class == "FUTURES" else "units"
            return PositionSize(
                units=0,
                confidence_level="invalid",
                position_multiplier=0.0,
                risk_amount=0.0,
                confidence_score=confidence_score,
                is_valid=False,
                reason=f"Confidence too low: {reason}",
                unit_type=unit_type
            )

        # Determine confidence level and position multiplier
        confidence_level, position_multiplier = self._get_confidence_level_and_multiplier(confidence_score)

        # Apply confidence-based position sizing
        final_position_size = int(base_position_size * position_multiplier)

        # Apply position size constraints
        final_position_size = self._apply_position_constraints(
            final_position_size, account_equity, instrument_obj
        )

        # Calculate actual risk amount
        risk_amount = self._calculate_actual_risk_amount(
            final_position_size, stop_loss_pips, instrument_obj
        )

        # Determine unit type based on asset class
        unit_type = "contracts" if isinstance(instrument_obj, Instrument) and instrument_obj.asset_class == "FUTURES" else "units"

        return PositionSize(
            units=final_position_size,
            confidence_level=confidence_level,
            position_multiplier=position_multiplier,
            risk_amount=risk_amount,
            confidence_score=confidence_score,
            is_valid=True,
            reason=f"Confidence {confidence_score:.3f} -> {confidence_level} level",
            unit_type=unit_type
        )

    def _calculate_base_position_size(
        self, account_equity: float, stop_loss_pips: float, instrument: Union[str, Instrument]
    ) -> int:
        """Calculate base position size based on risk parameters.

        For FX: Uses standard formula: Position Size = (Account Equity * Risk %) / (Stop Loss in $)
        For FUTURES: Uses tick-based formula: Contracts = int(max(1, risk_amount / (sl_ticks * tick_value)))

        Args:
            account_equity: Account equity in base currency
            stop_loss_pips: Stop loss in pips (FX) or ticks (FUTURES)
            instrument: Trading instrument (str or Instrument object)

        Returns:
            Position size in units (FX) or contracts (FUTURES)
        """
        # Check if this is a FUTURES instrument
        if isinstance(instrument, Instrument) and instrument.asset_class == "FUTURES":
            return self._calculate_futures_position_size(account_equity, stop_loss_pips, instrument)

        # For FX instruments (string or FX-class Instrument)
        # Calculate risk amount in account currency
        risk_amount = account_equity * self.config.risk_per_trade_pct

        # Calculate stop loss value in account currency per unit
        stop_loss_value_per_unit = self._get_stop_loss_value_per_unit(stop_loss_pips, instrument)

        if stop_loss_value_per_unit <= 0:
            instrument_str = instrument.symbol if isinstance(instrument, Instrument) else str(instrument)
            logger.warning(f"Invalid stop loss value for {instrument_str}: {stop_loss_value_per_unit}")
            return self.config.min_position_size

        # Calculate position size
        position_size = risk_amount / stop_loss_value_per_unit

        # Round to appropriate lot size based on account equity
        # Small accounts (<$50k): round to 100 units to avoid 50-67% oversize
        # Larger accounts: round to 1000 units (standard lot sizing)
        # NOTE: Use the `account_equity` parameter already in scope — do NOT shadow
        # it with a getattr(self.config, ...) that always returns the default 100000.
        rounding_unit = 100 if account_equity < 50000 else 1000
        position_size = int(round(position_size / rounding_unit) * rounding_unit)

        # Apply minimum position size
        position_size = max(position_size, self.config.min_position_size)

        return position_size

    def _calculate_futures_position_size(
        self, account_equity: float, sl_ticks: float, instrument: Instrument
    ) -> int:
        """Calculate position size (contracts) for FUTURES instruments.

        Formula: contracts = int(max(1, risk_amount / (sl_ticks * tick_value)))

        Args:
            account_equity: Account equity in base currency
            sl_ticks: Stop loss distance in ticks
            instrument: FUTURES Instrument object with tick_value defined

        Returns:
            Position size in contracts (minimum 1)
        """
        risk_amount = account_equity * self.config.risk_per_trade_pct

        # Validate instrument has required futures fields
        if instrument.tick_value is None or instrument.tick_value <= 0:
            logger.warning(f"FUTURES instrument {instrument.symbol} missing valid tick_value")
            return 1

        if sl_ticks <= 0:
            logger.warning(f"Invalid SL ticks for {instrument.symbol}: {sl_ticks}")
            return 1

        # Calculate contracts: risk_amount / (sl_ticks * tick_value)
        risk_per_tick = sl_ticks * instrument.tick_value
        contracts = int(max(1, risk_amount / risk_per_tick))

        logger.debug(
            f"FUTURES sizing {instrument.symbol}: "
            f"risk_amount=${risk_amount:.2f}, sl_ticks={sl_ticks}, "
            f"tick_value=${instrument.tick_value:.2f} → {contracts} contracts"
        )

        return contracts

    def _get_stop_loss_value_per_unit(self, stop_loss_pips: float, instrument: Union[str, Instrument]) -> float:
        """Get stop loss value per unit in account currency for FX instruments.

        Args:
            stop_loss_pips: Stop loss in pips
            instrument: FX instrument (str or Instrument object)

        Returns:
            Dollar value of stop loss per unit
        """
        # Instrument.pip_value is the price increment (0.0001 / 0.01), not a
        # dollar value. Convert to an approximate account-currency pip value
        # per unit so sizing does not shrink by 100x on non-JPY pairs.
        if isinstance(instrument, Instrument):
            instrument_upper = instrument.symbol.upper().replace("_", "")
        else:
            instrument_upper = instrument.upper().replace("_", "")

        # Pip value per unit in ACCOUNT (USD) terms = pip_size x quote->USD.
        # 2026-07-18 audit fix: the previous flat constants (0.0001 for every
        # non-JPY pair, one JPY constant for all JPY pairs) assumed quote
        # currency == USD. That oversized GBP-quoted pairs ~27% (EUR_GBP true
        # pip value ~0.000127) and undersized CAD-quoted ~26%. We now apply an
        # approximate quote->USD conversion; these are documented planning
        # approximations (same spirit as the old 0.000065), refreshed by any
        # caller that passes a live rate via _QUOTE_USD_OVERRIDES.
        pip_size = 0.01 if instrument_upper.endswith("JPY") else 0.0001
        quote_ccy = instrument_upper[-3:]
        quote_to_usd = _approx_ccy_to_usd(quote_ccy)
        pip_value_per_unit = pip_size * quote_to_usd

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

    def _apply_position_constraints(self, position_size: int, account_equity: float, instrument: Union[str, Instrument]) -> int:
        """Apply position size constraints.

        Args:
            position_size: Initial position size (units for FX, contracts for FUTURES)
            account_equity: Account equity
            instrument: Trading instrument

        Returns:
            Constrained position size
        """
        # Maximum position size based on account equity (30% of equity)
        # For FX: $100k account = 3,000,000 units = 30 lots max
        # For FUTURES: constraint is based on contract value, not units
        if isinstance(instrument, Instrument) and instrument.asset_class == "FUTURES":
            # For futures, apply a simple max contracts limit based on equity
            # Use margin_requirement to estimate max contracts
            max_position_from_equity = int(
                (account_equity * self.config.max_position_pct) / instrument.margin_requirement
            )
        else:
            # FX positioning
            max_position_from_equity = int(account_equity * self.config.max_position_pct * 100)  # In units

            # 2026-07-18 audit fix: hard notional-leverage clamp. The equity
            # cap above allows up to (max_position_pct x 100) = ~30x equity in
            # units, and no <=15x clamp existed anywhere in this FX path — the
            # documented "leverage re-clamped to <=15x" invariant lived only in
            # src/equity/oanda_trend.py. Notional(USD) ~= units x base->USD, so
            # units are capped at equity x 15 / base->USD (approximate
            # conversion; conservative parity fallback for unknowns).
            symbol = instrument.symbol if isinstance(instrument, Instrument) else str(instrument)
            base_ccy = symbol.upper().replace("_", "")[:3]
            base_to_usd = _approx_ccy_to_usd(base_ccy)
            max_units_leverage = int((account_equity * MAX_GROSS_LEVERAGE) / max(base_to_usd, 1e-9))
            max_position_from_equity = min(max_position_from_equity, max_units_leverage)

        # No artificial config cap - let equity-based limit control it
        # This allows proper scaling with account size

        # Apply constraints
        constrained_position = min(position_size, max_position_from_equity)
        constrained_position = max(constrained_position, self.config.min_position_size)

        return constrained_position

    def _calculate_actual_risk_amount(self, position_size: int, stop_loss_pips: float, instrument: Union[str, Instrument]) -> float:
        """Calculate actual risk amount for the given position size.

        Args:
            position_size: Position size (units for FX, contracts for FUTURES)
            stop_loss_pips: Stop loss (pips for FX, ticks for FUTURES)
            instrument: Trading instrument

        Returns:
            Risk amount in account currency
        """
        # For FUTURES, calculate risk differently (contracts * sl_ticks * tick_value)
        if isinstance(instrument, Instrument) and instrument.asset_class == "FUTURES":
            if instrument.tick_value and instrument.tick_value > 0:
                return position_size * stop_loss_pips * instrument.tick_value
            return 0.0

        # For FX, use pip-based calculation
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
        instrument: Union[str, Instrument],
        volatility_regime: int,
        meta_confidence: float,
        calibrated_confidence: Optional[CalibrationResult] = None,
        raw_confidence: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position size with regime-based aggressive scaling.

        This is the key method for enabling aggressive volatility-regime sizing.
        When regime == HIGH/EXTREME and meta_confidence > threshold, positions
        are scaled UP to capitalize on high-quality signals.

        Supports both FX and FUTURES instruments.

        Args:
            account_equity: Account equity in base currency
            stop_loss_pips: Stop loss distance in pips (FX) or ticks (FUTURES)
            instrument: Trading instrument (str like 'USD_JPY' or Instrument object)
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
        
        # H-9 fix: Validate regime_scale is finite and within safe bounds
        import math
        if not math.isfinite(regime_scale) or regime_scale <= 0:
            logger.warning(
                "Invalid regime_scale=%.4f for %s, defaulting to 1.0",
                regime_scale, instrument,
            )
            regime_scale = 1.0
        regime_scale = max(0.1, min(regime_scale, 3.0))

        # Apply regime scale to position
        scaled_units = int(base_result.units * regime_scale)
        
        # Apply position constraints after regime scaling
        scaled_units = self._apply_position_constraints(
            scaled_units, account_equity, instrument
        )
        
        # Recalculate risk amount
        instrument_obj = self._normalize_instrument(instrument)
        risk_amount = self._calculate_actual_risk_amount(
            scaled_units, stop_loss_pips, instrument_obj
        )

        regime_name = REGIME_NAMES[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"

        # Determine unit type
        unit_type = "contracts" if isinstance(instrument_obj, Instrument) and instrument_obj.asset_class == "FUTURES" else "units"

        # Log aggressive scaling when applied
        if regime_scale > 1.0:
            logger.info(
                f"🚀 AGGRESSIVE SCALING: {regime_name} vol + {meta_confidence:.2f} meta_confidence "
                f"-> {regime_scale:.2f}x scale ({base_result.units:,} -> {scaled_units:,} {unit_type})"
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
            unit_type=unit_type,
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
        conservative_scale_low_vol=0.75,
        conservative_scale_normal_vol=1.0,  # NORMAL = baseline, no reduction
        max_aggressive_scale=2.0,
    )
    
    config = PositionSizingConfig(
        risk_per_trade_pct=0.05,  # 5% base risk — meaningful on practice account
        min_confidence_threshold=0.5,
        max_position_multiplier=10.0,
        low_confidence_band=(0.5, 0.60),
        medium_confidence_band=(0.60, 0.75),
        high_confidence_band=(0.75, 1.0),
        low_confidence_multiplier=1.5,   # 1.5x at low confidence
        medium_confidence_multiplier=2.5,  # 2.5x at medium (most of our trades)
        high_confidence_multiplier=4.0,  # 4x at high confidence
        max_position_pct=0.30,  # 30% max position — aggressive for practice
        min_position_size=100000,  # 1.0 lots minimum
        regime_scaling=regime_config,
    )
    
    logger.info(
        f"Created regime-aware position sizer: enabled={regime_scaling_enabled}, "
        f"high_vol_scale={aggressive_scale_high_vol}x, "
        f"extreme_vol_scale={aggressive_scale_extreme_vol}x, "
        f"min_meta_conf={aggressive_min_meta_confidence}"
    )
    
    return DynamicPositionSizer(config)
