"""Unit tests for InstrumentRegistry.

Tests registry loading, lookup, filtering, and validation of FX and futures instruments.
"""

import pytest

from src.brokers.instrument import Instrument
from src.brokers.registry import (
    InstrumentRegistry,
    get_default_registry,
    get_registry,
    reset_registry,
)


class TestInstrumentRegistry:
    """Tests for InstrumentRegistry class."""

    def test_registry_creation(self):
        """Test creating an empty registry."""
        registry = InstrumentRegistry()
        assert len(registry) == 0
        assert registry.get_all_symbols() == []

    def test_register_instrument(self):
        """Test registering a single instrument."""
        registry = InstrumentRegistry()
        instrument = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        registry.register(instrument)
        assert len(registry) == 1
        assert registry.contains("EUR_USD")

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate symbol raises ValueError."""
        registry = InstrumentRegistry()
        instrument = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        registry.register(instrument)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(instrument)

    def test_get_instrument(self):
        """Test retrieving an instrument by symbol."""
        registry = InstrumentRegistry()
        instrument = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        registry.register(instrument)
        retrieved = registry.get("EUR_USD")
        assert retrieved.symbol == "EUR_USD"
        assert retrieved.pip_value == 0.0001

    def test_get_nonexistent_raises_keyerror(self):
        """Test that getting nonexistent symbol raises KeyError."""
        registry = InstrumentRegistry()
        with pytest.raises(KeyError):
            registry.get("NONEXISTENT")

    def test_get_optional_returns_none(self):
        """Test that get_optional returns None for nonexistent symbol."""
        registry = InstrumentRegistry()
        result = registry.get_optional("NONEXISTENT")
        assert result is None

    def test_get_optional_returns_instrument(self):
        """Test that get_optional returns instrument when found."""
        registry = InstrumentRegistry()
        instrument = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        registry.register(instrument)
        result = registry.get_optional("EUR_USD")
        assert result is not None
        assert result.symbol == "EUR_USD"

    def test_get_by_asset_class_fx(self):
        """Test filtering instruments by FX asset class."""
        registry = InstrumentRegistry()
        fx_instr = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )
        registry.register(fx_instr)
        registry.register(fut_instr)

        fx_list = registry.get_by_asset_class("FX")
        assert len(fx_list) == 1
        assert fx_list[0].symbol == "EUR_USD"

    def test_get_by_asset_class_futures(self):
        """Test filtering instruments by FUTURES asset class."""
        registry = InstrumentRegistry()
        fx_instr = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )
        registry.register(fx_instr)
        registry.register(fut_instr)

        futures_list = registry.get_by_asset_class("FUTURES")
        assert len(futures_list) == 1
        assert futures_list[0].symbol == "ES"

    def test_get_all_instruments(self):
        """Test retrieving all instruments."""
        registry = InstrumentRegistry()
        fx_instr = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
        )
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )
        registry.register(fx_instr)
        registry.register(fut_instr)

        all_instr = registry.get_all()
        assert len(all_instr) == 2
        symbols = {i.symbol for i in all_instr}
        assert symbols == {"EUR_USD", "ES"}

    def test_get_all_symbols(self):
        """Test retrieving all symbols."""
        registry = InstrumentRegistry()
        registry.register(
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR_USD",
                pip_value=0.0001,
            )
        )
        registry.register(
            Instrument.fx(
                symbol="GBP_USD",
                broker_symbol="GBP_USD",
                pip_value=0.0001,
            )
        )
        symbols = registry.get_all_symbols()
        assert set(symbols) == {"EUR_USD", "GBP_USD"}

    def test_registry_repr(self):
        """Test string representation."""
        registry = InstrumentRegistry()
        registry.register(
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR_USD",
                pip_value=0.0001,
            )
        )
        registry.register(
            Instrument.futures(
                symbol="ES",
                broker_symbol="ES",
                tick_size=0.25,
                tick_value=12.50,
                multiplier=50.0,
            )
        )
        repr_str = repr(registry)
        assert "1 FX instruments" in repr_str
        assert "1 FUTURES instruments" in repr_str


class TestDefaultRegistry:
    """Tests for default registry creation and content."""

    def test_default_registry_has_fx_pairs(self):
        """Test that default registry includes FX pairs."""
        registry = get_default_registry()
        fx_list = registry.get_by_asset_class("FX")
        assert len(fx_list) >= 23  # Minimum expected FX pairs
        symbols = {i.symbol for i in fx_list}
        assert "EUR_USD" in symbols
        assert "GBP_USD" in symbols
        assert "USD_JPY" in symbols

    def test_default_registry_has_futures(self):
        """Test that default registry includes futures contracts."""
        registry = get_default_registry()
        futures_list = registry.get_by_asset_class("FUTURES")
        assert len(futures_list) >= 4
        symbols = {i.symbol for i in futures_list}
        assert "ES" in symbols
        assert "NQ" in symbols
        assert "CL" in symbols
        assert "GC" in symbols

    def test_default_registry_fx_count(self):
        """Test that default registry has exactly 24 FX pairs."""
        registry = get_default_registry()
        fx_list = registry.get_by_asset_class("FX")
        assert len(fx_list) == 24

    def test_default_registry_futures_count(self):
        """Test that default registry has exactly 4 futures contracts."""
        registry = get_default_registry()
        futures_list = registry.get_by_asset_class("FUTURES")
        assert len(futures_list) == 4

    def test_default_registry_total_count(self):
        """Test that default registry has 28 total instruments."""
        registry = get_default_registry()
        assert len(registry) == 28

    def test_default_registry_eur_usd(self):
        """Test EUR_USD instrument properties."""
        registry = get_default_registry()
        eur_usd = registry.get("EUR_USD")
        assert eur_usd.asset_class == "FX"
        assert eur_usd.pip_value == 0.0001
        assert eur_usd.price_precision == 5
        assert eur_usd.exchange == "OANDA"
        assert eur_usd.currency == "USD"

    def test_default_registry_usd_jpy(self):
        """Test USD_JPY instrument properties."""
        registry = get_default_registry()
        usd_jpy = registry.get("USD_JPY")
        assert usd_jpy.asset_class == "FX"
        assert usd_jpy.pip_value == 0.01
        assert usd_jpy.price_precision == 3
        assert usd_jpy.exchange == "OANDA"

    def test_default_registry_es_futures(self):
        """Test ES futures instrument properties."""
        registry = get_default_registry()
        es = registry.get("ES")
        assert es.asset_class == "FUTURES"
        assert es.tick_size == 0.25
        assert es.tick_value == 12.50
        assert es.multiplier == 50.0
        assert es.price_precision == 2
        assert es.exchange == "CME"
        assert es.currency == "USD"

    def test_default_registry_nq_futures(self):
        """Test NQ futures instrument properties."""
        registry = get_default_registry()
        nq = registry.get("NQ")
        assert nq.asset_class == "FUTURES"
        assert nq.tick_size == 0.25
        assert nq.tick_value == 5.00
        assert nq.multiplier == 20.0
        assert nq.exchange == "CME"

    def test_default_registry_cl_futures(self):
        """Test CL futures instrument properties."""
        registry = get_default_registry()
        cl = registry.get("CL")
        assert cl.asset_class == "FUTURES"
        assert cl.tick_size == 0.01
        assert cl.tick_value == 10.00
        assert cl.multiplier == 1000.0
        assert cl.exchange == "NYMEX"

    def test_default_registry_gc_futures(self):
        """Test GC futures instrument properties."""
        registry = get_default_registry()
        gc = registry.get("GC")
        assert gc.asset_class == "FUTURES"
        assert gc.tick_size == 0.10
        assert gc.tick_value == 10.00
        assert gc.multiplier == 100.0
        assert gc.exchange == "NYMEX"

    def test_all_major_fx_pairs_present(self):
        """Test that all major FX pairs are present."""
        registry = get_default_registry()
        major_pairs = [
            "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD",
            "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD",
            "GBP_AUD", "EUR_CHF", "GBP_CHF",
        ]
        for pair in major_pairs:
            assert registry.contains(pair), f"Missing pair: {pair}"

    def test_all_extended_fx_pairs_present(self):
        """Test that all extended FX pairs are present."""
        registry = get_default_registry()
        extended_pairs = [
            "EUR_NZD", "GBP_NZD", "AUD_NZD", "NZD_JPY", "CAD_JPY", "CHF_JPY",
            "USD_SGD", "EUR_CAD", "GBP_CAD",
        ]
        for pair in extended_pairs:
            assert registry.contains(pair), f"Missing pair: {pair}"

    def test_singleton_pattern(self):
        """Test that get_registry returns the same instance."""
        reset_registry()  # Clear singleton
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_reset_registry(self):
        """Test that reset_registry clears the singleton."""
        # Get initial registry
        _ = get_registry()
        # Reset
        reset_registry()
        # Should create new instance
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
        assert len(registry1) > 0  # New instance populated

    def test_fx_price_precision_major_pairs(self):
        """Test that major (non-JPY) pairs have precision 5."""
        registry = get_default_registry()
        major_non_jpy = [
            "EUR_USD", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD",
            "NZD_USD", "EUR_GBP", "EUR_AUD", "GBP_AUD", "EUR_CHF",
            "GBP_CHF", "EUR_NZD", "GBP_NZD", "AUD_NZD", "USD_SGD",
            "EUR_CAD", "GBP_CAD",
        ]
        for pair in major_non_jpy:
            instrument = registry.get(pair)
            assert instrument.price_precision == 5, f"{pair} should have precision 5"

    def test_fx_price_precision_jpy_pairs(self):
        """Test that JPY pairs have precision 3."""
        registry = get_default_registry()
        jpy_pairs = [
            "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY",
        ]
        for pair in jpy_pairs:
            instrument = registry.get(pair)
            assert instrument.price_precision == 3, f"{pair} should have precision 3"

    def test_fx_pip_values_major_pairs(self):
        """Test that major (non-JPY) pairs have pip_value 0.0001."""
        registry = get_default_registry()
        major_non_jpy = [
            "EUR_USD", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD",
            "NZD_USD", "EUR_GBP", "EUR_AUD", "GBP_AUD", "EUR_CHF",
            "GBP_CHF", "EUR_NZD", "GBP_NZD", "AUD_NZD", "USD_SGD",
            "EUR_CAD", "GBP_CAD",
        ]
        for pair in major_non_jpy:
            instrument = registry.get(pair)
            assert instrument.pip_value == 0.0001, f"{pair} should have pip_value 0.0001"

    def test_fx_pip_values_jpy_pairs(self):
        """Test that JPY pairs have pip_value 0.01."""
        registry = get_default_registry()
        jpy_pairs = [
            "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY",
        ]
        for pair in jpy_pairs:
            instrument = registry.get(pair)
            assert instrument.pip_value == 0.01, f"{pair} should have pip_value 0.01"

    def test_all_instruments_valid(self):
        """Test that all instruments pass validation."""
        registry = get_default_registry()
        for instrument in registry.get_all():
            # Just accessing the instrument ensures validation passed
            assert instrument.symbol
            assert instrument.broker_symbol
            assert instrument.asset_class
            assert instrument.exchange
            assert instrument.currency
