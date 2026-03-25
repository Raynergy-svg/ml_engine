"""Phase 36 (US-234): Unit tests for DynamicPositionSizer.

Tests base position sizing, confidence bands, regime scaling,
factory functions, and edge cases.
"""

import pytest

from src.risk.position_sizing import (
    DynamicPositionSizer,
    FixedPositionSizer,
    PositionSize,
    PositionSizingConfig,
    RegimeScalingConfig,
    create_aggressive_position_sizer,
    create_conservative_position_sizer,
    create_default_position_sizer,
)


# ---------------------------------------------------------------------------
# Tests: PositionSizingConfig defaults
# ---------------------------------------------------------------------------
class TestPositionSizingConfig:
    def test_default_risk_pct(self):
        cfg = PositionSizingConfig()
        assert cfg.risk_per_trade_pct == 0.05

    def test_default_min_confidence(self):
        cfg = PositionSizingConfig()
        assert cfg.min_confidence_threshold == 0.5

    def test_default_bands_ordered(self):
        """Confidence bands must be non-overlapping and increasing."""
        cfg = PositionSizingConfig()
        assert cfg.low_confidence_band[1] <= cfg.medium_confidence_band[0]
        assert cfg.medium_confidence_band[1] <= cfg.high_confidence_band[0]

    def test_regime_scaling_defaults(self):
        cfg = PositionSizingConfig()
        assert cfg.regime_scaling.enabled is True
        assert cfg.regime_scaling.aggressive_scale_high_vol > 1.0


# ---------------------------------------------------------------------------
# Tests: PositionSize dataclass
# ---------------------------------------------------------------------------
class TestPositionSize:
    def test_default_regime_fields(self):
        ps = PositionSize(
            units=100000, confidence_level="medium",
            position_multiplier=2.5, risk_amount=500.0,
            confidence_score=0.65, is_valid=True, reason="ok",
        )
        assert ps.regime_scale_applied == 1.0
        assert ps.regime_name == "UNKNOWN"
        assert ps.aggressive_scaling_reason == ""


# ---------------------------------------------------------------------------
# Tests: DynamicPositionSizer core calculations
# ---------------------------------------------------------------------------
class TestDynamicPositionSizer:
    def _make_sizer(self, **overrides):
        cfg = PositionSizingConfig(**overrides)
        return DynamicPositionSizer(cfg)

    def test_no_confidence_returns_invalid(self):
        """Without confidence, position should be invalid."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD",
        )
        assert result.is_valid is False
        assert result.confidence_level == "invalid"

    def test_low_confidence_rejected(self):
        """Confidence below threshold returns zero units."""
        sizer = self._make_sizer(min_confidence_threshold=0.5)
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD", raw_confidence=0.3,
        )
        assert result.is_valid is False
        assert result.units == 0

    def test_medium_confidence_valid(self):
        """Medium confidence should produce valid position."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD", raw_confidence=0.65,
        )
        assert result.is_valid is True
        assert result.units > 0
        assert result.confidence_level == "medium"

    def test_high_confidence_larger_position(self):
        """High confidence should produce larger position than medium."""
        sizer = self._make_sizer()
        medium = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD", raw_confidence=0.65,
        )
        high = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD", raw_confidence=0.85,
        )
        assert high.units >= medium.units
        assert high.confidence_level == "high"

    def test_zero_stop_loss_returns_minimum(self):
        """Zero stop loss should return minimum position size."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=0.0,
            instrument="EUR_USD", raw_confidence=0.7,
        )
        assert result.units >= sizer.config.min_position_size

    def test_jpy_pair_pip_value(self):
        """JPY pairs should use different pip value calculation."""
        sizer = self._make_sizer()
        eur_result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD", raw_confidence=0.65,
        )
        jpy_result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="USD_JPY", raw_confidence=0.65,
        )
        # JPY pairs have smaller pip value, so position should be larger
        assert jpy_result.units > eur_result.units


# ---------------------------------------------------------------------------
# Tests: Confidence level classification
# ---------------------------------------------------------------------------
class TestConfidenceLevels:
    def test_invalid_below_low_band(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        level, mult = sizer._get_confidence_level_and_multiplier(0.3)
        assert level == "invalid"
        assert mult == 0.0

    def test_low_confidence_band(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        level, mult = sizer._get_confidence_level_and_multiplier(0.55)
        assert level == "low"
        assert mult == sizer.config.low_confidence_multiplier

    def test_medium_confidence_band(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        level, mult = sizer._get_confidence_level_and_multiplier(0.68)
        assert level == "medium"

    def test_high_confidence_band(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        level, mult = sizer._get_confidence_level_and_multiplier(0.80)
        assert level == "high"


# ---------------------------------------------------------------------------
# Tests: Confidence recommendation
# ---------------------------------------------------------------------------
class TestConfidenceRecommendation:
    def test_avoid_low_confidence(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        rec = sizer.get_confidence_recommendation(0.3)
        assert "AVOID" in rec

    def test_large_position_high_confidence(self):
        sizer = DynamicPositionSizer(PositionSizingConfig())
        rec = sizer.get_confidence_recommendation(0.85)
        assert "LARGE" in rec


# ---------------------------------------------------------------------------
# Tests: FixedPositionSizer
# ---------------------------------------------------------------------------
class TestFixedPositionSizer:
    def test_default_fixed_size(self):
        sizer = FixedPositionSizer()
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD",
        )
        assert result.units == 10000

    def test_custom_fixed_size(self):
        sizer = FixedPositionSizer(fixed_size=50000)
        result = sizer.calculate_position_size(
            account_equity=100000, stop_loss_pips=20.0,
            instrument="EUR_USD",
        )
        assert result.units == 50000


# ---------------------------------------------------------------------------
# Tests: Factory functions
# ---------------------------------------------------------------------------
class TestFactoryFunctions:
    def test_create_default(self):
        sizer = create_default_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)
        assert 0.0 < sizer.config.risk_per_trade_pct <= 0.10  # Reasonable range

    def test_create_conservative(self):
        sizer = create_conservative_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)
        assert sizer.config.risk_per_trade_pct <= 0.05

    def test_create_aggressive(self):
        sizer = create_aggressive_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)
        assert sizer.config.risk_per_trade_pct >= 0.05

    def test_conservative_smaller_than_aggressive(self):
        """Conservative should have smaller risk than aggressive."""
        conservative = create_conservative_position_sizer()
        aggressive = create_aggressive_position_sizer()
        assert conservative.config.risk_per_trade_pct < aggressive.config.risk_per_trade_pct
