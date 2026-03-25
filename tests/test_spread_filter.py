"""
Tests for SpreadFilter (US-300, Phase 48).

Covers: pass/fail thresholds, pair overrides, disabled filter,
zero ATR fallback, size multiplier, config validation, edge cases.
"""

import pytest
from src.scanner.spread_filter import (
    SpreadFilter,
    SpreadFilterConfig,
    SpreadCheckResult,
    create_default_spread_filter,
)


class TestSpreadCheckResult:
    """Test SpreadCheckResult dataclass and size_multiplier property."""

    def test_size_multiplier_tight_spread(self):
        """Very tight spread should return 1.0 multiplier."""
        result = SpreadCheckResult(
            passed=True, spread_pips=0.5, atr_pips=30.0,
            normalized_spread=0.017, threshold=0.20, reason="ok"
        )
        assert result.size_multiplier == 1.0

    def test_size_multiplier_moderate_spread(self):
        """Moderate spread should reduce multiplier slightly."""
        result = SpreadCheckResult(
            passed=True, spread_pips=3.0, atr_pips=30.0,
            normalized_spread=0.10, threshold=0.20, reason="ok"
        )
        mult = result.size_multiplier
        assert 0.7 < mult < 1.0

    def test_size_multiplier_at_threshold(self):
        """At threshold boundary, multiplier should be ~0.7."""
        result = SpreadCheckResult(
            passed=True, spread_pips=6.0, atr_pips=30.0,
            normalized_spread=0.20, threshold=0.20, reason="ok"
        )
        mult = result.size_multiplier
        assert abs(mult - 0.7) < 0.05

    def test_size_multiplier_failed(self):
        """Failed check should return 0.0 multiplier."""
        result = SpreadCheckResult(
            passed=False, spread_pips=10.0, atr_pips=30.0,
            normalized_spread=0.33, threshold=0.20, reason="blocked"
        )
        assert result.size_multiplier == 0.0


class TestSpreadFilterConfig:
    """Test SpreadFilterConfig validation and defaults."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = SpreadFilterConfig()
        assert config.max_normalized_spread == 0.20
        assert config.enabled is True
        assert config.version == 1
        assert "EUR_USD" in config.pair_overrides

    def test_pair_overrides_populated(self):
        """Default pair overrides include majors and crosses."""
        config = SpreadFilterConfig()
        assert config.pair_overrides["EUR_USD"] == 0.15
        assert config.pair_overrides["GBP_NZD"] == 0.25

    def test_get_threshold_with_override(self):
        """Pair with override returns override value."""
        config = SpreadFilterConfig()
        assert config.get_threshold("EUR_USD") == 0.15

    def test_get_threshold_without_override(self):
        """Pair without override returns default."""
        config = SpreadFilterConfig()
        assert config.get_threshold("USD_SGD") == 0.20

    def test_validate_valid_config(self):
        """Valid config passes validation."""
        config = SpreadFilterConfig(max_normalized_spread=0.15)
        config.validate()  # Should not raise

    def test_validate_invalid_threshold_too_high(self):
        """Threshold > 1.0 fails validation."""
        config = SpreadFilterConfig(max_normalized_spread=1.5)
        with pytest.raises(ValueError, match="max_normalized_spread"):
            config.validate()

    def test_validate_invalid_threshold_too_low(self):
        """Threshold < 0.01 fails validation."""
        config = SpreadFilterConfig(max_normalized_spread=0.005)
        with pytest.raises(ValueError, match="max_normalized_spread"):
            config.validate()

    def test_validate_invalid_pair_override(self):
        """Invalid pair override value fails validation."""
        config = SpreadFilterConfig(pair_overrides={"EUR_USD": 2.0})
        with pytest.raises(ValueError, match="pair override"):
            config.validate()


class TestSpreadFilter:
    """Test SpreadFilter core functionality."""

    def test_spread_passes_tight(self):
        """Tight spread well below threshold passes."""
        f = SpreadFilter()
        result = f.check_spread("EUR_USD", bid=1.0850, ask=1.0852, atr_pips=30.0)
        assert result.passed is True
        assert result.spread_pips == pytest.approx(2.0, abs=0.1)
        assert result.normalized_spread < 0.15

    def test_spread_fails_wide(self):
        """Wide spread above threshold fails."""
        f = SpreadFilter()
        # 10 pip spread on 30 pip ATR = 33% > 20% default
        result = f.check_spread("USD_CHF", bid=0.9100, ask=0.9110, atr_pips=30.0)
        assert result.passed is False
        assert "spread_too_wide" in result.reason

    def test_spread_at_threshold_boundary(self):
        """Spread exactly at threshold passes (<=)."""
        config = SpreadFilterConfig(
            max_normalized_spread=0.20,
            pair_overrides={}
        )
        f = SpreadFilter(config)
        # 6 pip spread on 30 pip ATR = 20% exactly
        result = f.check_spread("EUR_USD", bid=1.0850, ask=1.0856, atr_pips=30.0)
        assert result.passed is True
        assert result.normalized_spread == pytest.approx(0.20, abs=0.01)

    def test_spread_disabled_filter(self):
        """Disabled filter always passes."""
        config = SpreadFilterConfig(enabled=False)
        f = SpreadFilter(config)
        result = f.check_spread("EUR_USD", bid=1.0800, ask=1.0900, atr_pips=5.0)
        assert result.passed is True
        assert result.reason == "spread_filter_disabled"

    def test_spread_zero_atr_fallback(self):
        """Zero ATR triggers graceful fallback (allow with warning)."""
        f = SpreadFilter()
        result = f.check_spread("EUR_USD", bid=1.0850, ask=1.0852, atr_pips=0.0)
        assert result.passed is True
        assert result.reason == "atr_zero_or_negative_fallback"

    def test_spread_negative_atr_fallback(self):
        """Negative ATR triggers graceful fallback."""
        f = SpreadFilter()
        result = f.check_spread("EUR_USD", bid=1.0850, ask=1.0852, atr_pips=-5.0)
        assert result.passed is True
        assert result.reason == "atr_zero_or_negative_fallback"

    def test_jpy_pair_spread_calculation(self):
        """JPY pairs use pip value of 0.01."""
        f = SpreadFilter()
        # 3 pip spread (0.03 in USD_JPY terms)
        result = f.check_spread("USD_JPY", bid=150.00, ask=150.03, atr_pips=50.0)
        assert result.passed is True
        assert result.spread_pips == pytest.approx(3.0, abs=0.1)
        assert result.normalized_spread == pytest.approx(0.06, abs=0.01)

    def test_pair_override_threshold(self):
        """Pair with custom threshold uses that threshold."""
        f = SpreadFilter()
        # EUR_USD has 0.15 override; 5 pip spread on 30 ATR = 16.7% > 15%
        result = f.check_spread("EUR_USD", bid=1.0850, ask=1.0855, atr_pips=30.0)
        assert result.passed is False  # Because EUR_USD threshold is 0.15

    def test_cross_pair_wider_threshold(self):
        """Cross pairs with wider default threshold allow wider spreads."""
        f = SpreadFilter()
        # GBP_NZD has 0.25 override; 6 pip spread on 30 ATR = 20% < 25%
        result = f.check_spread("GBP_NZD", bid=2.0800, ask=2.0806, atr_pips=30.0)
        assert result.passed is True

    def test_check_spread_from_price(self):
        """check_spread_from_price works with known spread_pips."""
        f = SpreadFilter()
        result = f.check_spread_from_price("EUR_USD", price=1.0850, spread_pips=2.0, atr_pips=30.0)
        assert result.passed is True
        assert result.spread_pips == 2.0
        assert result.normalized_spread == pytest.approx(0.067, abs=0.01)

    def test_check_spread_from_price_fails(self):
        """check_spread_from_price fails on wide spread."""
        config = SpreadFilterConfig(pair_overrides={})
        f = SpreadFilter(config)
        result = f.check_spread_from_price("EUR_USD", price=1.0850, spread_pips=8.0, atr_pips=30.0)
        assert result.passed is False  # 8/30 = 26.7% > 20%

    def test_check_spread_from_price_disabled(self):
        """Disabled filter passes in check_spread_from_price too."""
        config = SpreadFilterConfig(enabled=False)
        f = SpreadFilter(config)
        result = f.check_spread_from_price("EUR_USD", price=1.0850, spread_pips=100.0, atr_pips=10.0)
        assert result.passed is True

    def test_check_spread_from_price_zero_atr(self):
        """Zero ATR in check_spread_from_price triggers fallback."""
        f = SpreadFilter()
        result = f.check_spread_from_price("EUR_USD", price=1.0850, spread_pips=2.0, atr_pips=0.0)
        assert result.passed is True
        assert result.reason == "atr_zero_or_negative_fallback"


class TestSpreadFilterFactory:
    """Test factory function."""

    def test_create_default(self):
        """Factory creates working filter with defaults."""
        f = create_default_spread_filter()
        assert isinstance(f, SpreadFilter)
        assert f.config.enabled is True

    def test_create_with_custom_config(self):
        """Factory accepts custom config."""
        config = SpreadFilterConfig(max_normalized_spread=0.10)
        f = create_default_spread_filter(config=config)
        assert f.config.max_normalized_spread == 0.10

    def test_create_with_invalid_config_fallback(self):
        """Factory falls back to defaults on invalid config."""
        config = SpreadFilterConfig(max_normalized_spread=5.0)
        f = create_default_spread_filter(config=config)
        # Should fallback to default 0.20
        assert f.config.max_normalized_spread == 0.20
