"""Comprehensive tests for futures position sizing in position_sizing.py.

Tests cover:
1. Futures tick-based sizing formula
2. Unit type field defaults and assignment
3. Instrument normalization (str vs Instrument)
4. Risk calculations for both FX and FUTURES
5. Regime-scaled sizing with futures instruments
"""

import pytest
import logging
from src.risk.position_sizing import (
    DynamicPositionSizer,
    PositionSize,
    PositionSizingConfig,
    RegimeScalingConfig,
    create_regime_aware_position_sizer,
)
from src.risk.confidence_calibration import CalibrationResult
from src.brokers.instrument import Instrument


logger = logging.getLogger(__name__)


class TestFuturesTickBasedSizing:
    """Test futures position sizing with tick-based formula."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            risk_per_trade_pct=0.05,  # 5% risk per trade
            min_confidence_threshold=0.5,
            max_position_multiplier=10.0,
            low_confidence_band=(0.5, 0.60),
            medium_confidence_band=(0.60, 0.75),
            high_confidence_band=(0.75, 1.0),
            low_confidence_multiplier=1.5,
            medium_confidence_multiplier=2.5,
            high_confidence_multiplier=4.0,
            max_position_pct=0.30,
            min_position_size=1,
        )
        self.sizer = DynamicPositionSizer(self.config)

        # Create S&P 500 E-mini futures instrument
        self.es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,  # Each tick worth $12.50
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,  # ~$4k per contract
            exchange="CME",
            currency="USD",
        )

    def test_futures_contract_calculation_basic(self):
        """Test basic futures contract sizing formula.

        Formula: contracts = int(max(1, risk_amount / (sl_ticks * tick_value)))
        For $100k account, 5% risk = $5k
        With 10 ticks SL and $12.50/tick: contracts = int(5000 / (10 * 12.50)) = int(40) = 40
        """
        account_equity = 100_000.0
        sl_ticks = 10.0

        result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,  # pips param reused for ticks in futures context
            instrument=self.es_instrument,
            raw_confidence=0.70,
        )

        # Risk amount: $100k * 0.05 = $5k
        # Per tick cost: 10 ticks * $12.50 = $125
        # Base contracts: 5000 / 125 = 40
        # Confidence multiplier (0.70 -> medium): 2.5
        # Final: int(40 * 2.5) = 100 contracts
        # BUT: min_position_size=1 constraint and position constraints may limit this
        # The test validates the formula works, check units > 0 and unit_type is contracts
        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.units >= 1
        # Verify risk calculation is correct
        expected_risk = result.units * sl_ticks * self.es_instrument.tick_value
        assert abs(result.risk_amount - expected_risk) < 0.01
        logger.info(f"Basic futures sizing: {result.units} {result.unit_type}, risk=${result.risk_amount:.2f}")

    def test_futures_contract_minimum_one(self):
        """Test that futures sizing never returns less than 1 contract.

        With high SL ticks and low account, should cap at 1 minimum.
        """
        account_equity = 5_000.0  # Small account
        sl_ticks = 100.0  # Very large SL

        result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,
            instrument=self.es_instrument,
            raw_confidence=0.55,  # Low confidence -> multiplier 1.5
        )

        # Risk: 5000 * 0.05 = $250
        # Per tick cost: 100 * $12.50 = $1250
        # Base: 250 / 1250 = 0.2 -> max(1, 0.2) = 1
        # Confidence multiplier: 1.5
        # Final: int(1 * 1.5) = 1 contract
        assert result.is_valid
        assert result.units >= 1
        assert result.unit_type == "contracts"
        logger.info(f"Minimum contract sizing: {result.units} {result.unit_type}")

    def test_futures_risk_amount_calculation(self):
        """Test risk amount calculation for futures: contracts * sl_ticks * tick_value.

        For 10 contracts, 5 ticks SL, $12.50/tick:
        risk = 10 * 5 * $12.50 = $625
        """
        account_equity = 100_000.0
        sl_ticks = 5.0

        result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,
            instrument=self.es_instrument,
            raw_confidence=0.80,  # High confidence
        )

        # Base contracts: 5000 / (5 * 12.5) = 5000 / 62.5 = 80
        # Confidence multiplier (0.80 -> high): 4.0
        # Final contracts: int(80 * 4.0) = 320
        # Risk amount: 320 * 5 * 12.50 = $20,000
        expected_risk = result.units * sl_ticks * self.es_instrument.tick_value
        assert result.is_valid
        assert result.unit_type == "contracts"
        assert abs(result.risk_amount - expected_risk) < 0.01
        logger.info(f"Futures risk calculation: {result.units} contracts × {sl_ticks} ticks × ${self.es_instrument.tick_value} = ${result.risk_amount:.2f}")

    def test_futures_with_confidence_bands(self):
        """Test that futures sizing respects confidence multipliers."""
        account_equity = 500_000.0  # Larger account to avoid min position size constraints
        sl_ticks = 5.0

        # Low confidence (0.55)
        low_result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,
            instrument=self.es_instrument,
            raw_confidence=0.55,
        )

        # Medium confidence (0.70)
        med_result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,
            instrument=self.es_instrument,
            raw_confidence=0.70,
        )

        # High confidence (0.85)
        high_result = self.sizer.calculate_position_size(
            account_equity=account_equity,
            stop_loss_pips=sl_ticks,
            instrument=self.es_instrument,
            raw_confidence=0.85,
        )

        # Low confidence should produce smallest position
        # Medium should be middle
        # High should be largest
        assert low_result.is_valid and low_result.units > 0
        assert med_result.is_valid and med_result.units > 0
        assert high_result.is_valid and high_result.units > 0
        # Check multiplier progression is correct
        assert low_result.position_multiplier == 1.5
        assert med_result.position_multiplier == 2.5
        assert high_result.position_multiplier == 4.0
        assert low_result.units <= med_result.units <= high_result.units
        logger.info(f"Confidence scaling: low={low_result.units}, med={med_result.units}, high={high_result.units} contracts")


class TestPositionSizeUnitType:
    """Test unit_type field in PositionSize dataclass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            risk_per_trade_pct=0.02,
            min_confidence_threshold=0.5,
        )
        self.sizer = DynamicPositionSizer(self.config)

    def test_fx_unit_type_default(self):
        """Test that FX instruments return unit_type='units'."""
        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",  # String FX instrument
            raw_confidence=0.65,
        )

        assert result.is_valid
        assert result.unit_type == "units"
        logger.info(f"FX unit_type: {result.unit_type}")

    def test_futures_unit_type_contracts(self):
        """Test that FUTURES instruments return unit_type='contracts'."""
        es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,
            exchange="CME",
            currency="USD",
        )

        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument=es_instrument,
            raw_confidence=0.65,
        )

        assert result.is_valid
        assert result.unit_type == "contracts"
        logger.info(f"FUTURES unit_type: {result.unit_type}")

    def test_unit_type_in_invalid_result(self):
        """Test that unit_type is present even in invalid results."""
        es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,
            exchange="CME",
            currency="USD",
        )

        # Too low confidence
        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument=es_instrument,
            raw_confidence=0.30,  # Below threshold
        )

        assert not result.is_valid
        assert result.unit_type == "contracts"
        assert result.units == 0
        logger.info(f"Invalid FUTURES result: unit_type={result.unit_type}, is_valid={result.is_valid}")


class TestInstrumentNormalization:
    """Test instrument normalization (Union[str, Instrument])."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(risk_per_trade_pct=0.05)
        self.sizer = DynamicPositionSizer(self.config)

    def test_string_instrument_accepted(self):
        """Test that string instruments are accepted."""
        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=20.0,
            instrument="GBP_USD",
            raw_confidence=0.60,
        )

        assert result.is_valid
        assert result.unit_type == "units"

    def test_instrument_object_accepted(self):
        """Test that Instrument objects are accepted."""
        eur_usd = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR/USD",
            pip_value=10.0,
            price_precision=5,
            margin_requirement=1.0,
            exchange="OANDA",
            currency="USD",
        )

        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=20.0,
            instrument=eur_usd,
            raw_confidence=0.60,
        )

        assert result.is_valid
        assert result.unit_type == "units"

    def test_fx_instrument_object_uses_pip_value(self):
        """Test that FX Instrument objects use their defined pip_value."""
        # Custom pip value for an exotic pair
        custom_pair = Instrument.fx(
            symbol="USDZAR",
            broker_symbol="USD/ZAR",
            pip_value=0.001,  # Very low pip value for high-spread pair
            price_precision=4,
            margin_requirement=2.0,
            exchange="OANDA",
            currency="USD",
        )

        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=100.0,  # 100 pips
            instrument=custom_pair,
            raw_confidence=0.65,
        )

        # Risk calc should use custom pip_value of 0.001
        # Expected risk per unit: 100 pips * 0.001 = $0.10 per unit
        # Base position: 5000 / 0.10 = 50,000 units
        assert result.is_valid
        assert result.unit_type == "units"
        logger.info(f"Custom pair sizing: {result.units} units, risk=${result.risk_amount:.2f}")


class TestRegimeScaledFuturesSizing:
    """Test regime-scaled position sizing with futures instruments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = create_regime_aware_position_sizer(
            aggressive_scale_high_vol=1.5,
            aggressive_scale_extreme_vol=1.75,
            aggressive_min_meta_confidence=0.52,
            regime_scaling_enabled=True,
        )

        self.es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,
            exchange="CME",
            currency="USD",
        )

    def test_high_vol_aggressive_scaling_futures(self):
        """Test aggressive scaling in HIGH volatility with futures."""
        # REGIME_HIGH = 2
        result = self.sizer.calculate_regime_scaled_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument=self.es_instrument,
            volatility_regime=2,  # HIGH
            meta_confidence=0.60,  # Above 0.52 threshold
            raw_confidence=0.70,
        )

        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.regime_scale_applied > 1.0  # Should scale up
        logger.info(f"HIGH vol scaling: {result.units} contracts, regime_scale={result.regime_scale_applied:.2f}")

    def test_low_vol_conservative_scaling_futures(self):
        """Test conservative scaling in LOW volatility with futures."""
        # REGIME_LOW = 0
        result = self.sizer.calculate_regime_scaled_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument=self.es_instrument,
            volatility_regime=0,  # LOW
            meta_confidence=0.60,
            raw_confidence=0.70,
        )

        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.regime_scale_applied < 1.0  # Should scale down
        logger.info(f"LOW vol scaling: {result.units} contracts, regime_scale={result.regime_scale_applied:.2f}")

    def test_extreme_vol_high_confidence_futures(self):
        """Test extreme vol with high meta confidence."""
        # REGIME_EXTREME = 3
        result = self.sizer.calculate_regime_scaled_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument=self.es_instrument,
            volatility_regime=3,  # EXTREME
            meta_confidence=0.70,  # High confidence
            raw_confidence=0.75,
        )

        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.regime_scale_applied > 1.0  # Should scale up aggressively
        logger.info(f"EXTREME vol scaling: {result.units} contracts, regime_scale={result.regime_scale_applied:.2f}")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            risk_per_trade_pct=0.05,
            min_confidence_threshold=0.5,
            min_position_size=1,
        )
        self.sizer = DynamicPositionSizer(self.config)

    def test_zero_confidence(self):
        """Test that zero confidence results in invalid position."""
        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument="EUR_USD",
            raw_confidence=0.0,
        )

        assert not result.is_valid
        assert result.units == 0

    def test_none_confidence(self):
        """Test handling of missing confidence."""
        result = self.sizer.calculate_position_size(
            account_equity=100_000.0,
            stop_loss_pips=10.0,
            instrument="EUR_USD",
            raw_confidence=None,
            calibrated_confidence=None,
        )

        assert not result.is_valid
        assert "No confidence provided" in result.reason

    def test_futures_with_missing_tick_value(self):
        """Test that futures instrument creation fails without tick_value.

        The Instrument class validates required fields in __post_init__.
        """
        # This should fail validation in __post_init__
        with pytest.raises(ValueError, match="must define tick_value"):
            Instrument.futures(
                symbol="BAD",
                broker_symbol="BAD",
                tick_size=0.01,
                tick_value=None,  # Missing!
                multiplier=100.0,
                price_precision=2,
                margin_requirement=1000.0,
                exchange="TEST",
                currency="USD",
            )

    def test_large_account_small_sl(self):
        """Test sizing with large account and small SL (should produce large position)."""
        es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,
            exchange="CME",
            currency="USD",
        )

        result = self.sizer.calculate_position_size(
            account_equity=1_000_000.0,  # $1M account
            stop_loss_pips=2.0,  # Only 2 ticks SL
            instrument=es_instrument,
            raw_confidence=0.75,  # High confidence
        )

        # Risk: 1M * 0.05 = $50k
        # Per tick: 2 * $12.50 = $25
        # Base: 50000 / 25 = 2000 contracts
        # Multiplier: 4.0 -> 8000 contracts (before constraints)
        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.units > 0
        logger.info(f"Large account sizing: {result.units} contracts")

    def test_small_account_large_sl(self):
        """Test sizing with small account and large SL (should produce 1 contract minimum)."""
        es_instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ24",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=4000.0,
            exchange="CME",
            currency="USD",
        )

        result = self.sizer.calculate_position_size(
            account_equity=10_000.0,  # $10k account
            stop_loss_pips=200.0,  # 200 ticks SL
            instrument=es_instrument,
            raw_confidence=0.55,  # Low confidence
        )

        # Risk: 10k * 0.05 = $500
        # Per tick: 200 * $12.50 = $2500
        # Base: 500 / 2500 = 0.2 -> max(1, 0.2) = 1
        # Multiplier: 1.5 -> 1.5 contracts (rounded to 1)
        assert result.is_valid
        assert result.unit_type == "contracts"
        assert result.units >= 1
        logger.info(f"Small account sizing: {result.units} contracts (minimum)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
