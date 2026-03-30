"""
Test suite for PIP_VALUES migration to InstrumentRegistry (US-017).

Verifies that:
1. All historical PIP_VALUES have been moved to InstrumentRegistry
2. Registry values match old hardcoded PIP_VALUES exactly
3. All code paths use registry.get(symbol).pip_value
4. FX behavior is unchanged (same pip values as before)
5. Futures instruments return tick_value from registry
"""

import pytest

from src.brokers.registry import get_registry
from src.scanner.config import ScannerConfig
from src.risk.trading_metrics import _get_pip_value as trading_metrics_get_pip_value
from src.scanner.spread_filter import _get_pip_value as spread_filter_get_pip_value


class TestPipValuesMigration:
    """Verify PIP_VALUES migration to InstrumentRegistry."""

    def test_registry_has_all_fx_pairs(self):
        """Registry should contain all major and cross pairs."""
        registry = get_registry()
        fx_instruments = registry.get_by_asset_class("FX")
        fx_symbols = {inst.symbol for inst in fx_instruments}

        # Major pairs
        assert "EUR_USD" in fx_symbols
        assert "GBP_USD" in fx_symbols
        assert "USD_JPY" in fx_symbols
        assert "USD_CHF" in fx_symbols

        # Commodity pairs
        assert "AUD_USD" in fx_symbols
        assert "USD_CAD" in fx_symbols
        assert "NZD_USD" in fx_symbols

        # Crosses
        assert "EUR_GBP" in fx_symbols
        assert "EUR_JPY" in fx_symbols
        assert "GBP_JPY" in fx_symbols
        assert "AUD_JPY" in fx_symbols

    def test_registry_fx_pip_values_match_old_hardcoded_values(self):
        """Verify registry pip values match the old PIP_VALUES dict exactly."""
        registry = get_registry()

        # Historical pip values from old PIP_VALUES dicts
        expected_pip_values = {
            "EUR_USD": 0.0001,
            "GBP_USD": 0.0001,
            "USD_JPY": 0.01,
            "USD_CHF": 0.0001,
            "AUD_USD": 0.0001,
            "USD_CAD": 0.0001,
            "NZD_USD": 0.0001,
            "EUR_GBP": 0.0001,
            "EUR_JPY": 0.01,
            "GBP_JPY": 0.01,
            "AUD_JPY": 0.01,
            "EUR_AUD": 0.0001,
            "GBP_AUD": 0.0001,
            "EUR_CHF": 0.0001,
            "GBP_CHF": 0.0001,
            "EUR_NZD": 0.0001,
            "GBP_NZD": 0.0001,
            "AUD_NZD": 0.0001,
            "NZD_JPY": 0.01,
            "CAD_JPY": 0.01,
            "CHF_JPY": 0.01,
            "USD_SGD": 0.0001,
            "EUR_CAD": 0.0001,
            "GBP_CAD": 0.0001,
        }

        for pair, expected_pip in expected_pip_values.items():
            inst = registry.get(pair)
            assert (
                inst.pip_value == expected_pip
            ), f"{pair}: expected {expected_pip}, got {inst.pip_value}"

    def test_registry_has_futures_with_tick_values(self):
        """Registry should include futures contracts with tick_value."""
        registry = get_registry()
        futures = registry.get_by_asset_class("FUTURES")
        futures_symbols = {inst.symbol for inst in futures}

        # Verify futures are registered
        assert "ES" in futures_symbols
        assert "NQ" in futures_symbols
        assert "CL" in futures_symbols
        assert "GC" in futures_symbols

        # Verify tick values are set
        es = registry.get("ES")
        assert es.tick_value == 12.50
        assert es.tick_size == 0.25

        nq = registry.get("NQ")
        assert nq.tick_value == 5.00

    def test_scanner_config_get_pip_value_uses_registry(self):
        """ScannerConfig.get_pip_value() should return registry values."""
        cfg = ScannerConfig()

        assert cfg.get_pip_value("EUR_USD") == 0.0001
        assert cfg.get_pip_value("USD_JPY") == 0.01
        assert cfg.get_pip_value("GBP_AUD") == 0.0001

    def test_scanner_config_unknown_pair_fallback(self):
        """Unknown pairs should fallback to 0.0001."""
        cfg = ScannerConfig()
        assert cfg.get_pip_value("FAKE_PAIR") == 0.0001

    def test_trading_metrics_get_pip_value_uses_registry(self):
        """trading_metrics._get_pip_value() should return registry values."""
        assert trading_metrics_get_pip_value("EUR_USD") == 0.0001
        assert trading_metrics_get_pip_value("USD_JPY") == 0.01
        assert trading_metrics_get_pip_value("NZD_JPY") == 0.01

    def test_trading_metrics_unknown_pair_fallback(self):
        """trading_metrics unknown pairs should fallback to 0.0001."""
        assert trading_metrics_get_pip_value("UNKNOWN_PAIR") == 0.0001

    def test_spread_filter_get_pip_value_uses_registry(self):
        """spread_filter._get_pip_value() should return registry values."""
        assert spread_filter_get_pip_value("EUR_USD") == 0.0001
        assert spread_filter_get_pip_value("USD_JPY") == 0.01
        assert spread_filter_get_pip_value("EUR_GBP") == 0.0001

    def test_spread_filter_unknown_pair_fallback(self):
        """spread_filter unknown pairs should fallback to 0.0001."""
        assert spread_filter_get_pip_value("UNKNOWN_PAIR") == 0.0001

    def test_all_fx_pip_values_are_positive(self):
        """All FX instruments must have positive pip_value."""
        registry = get_registry()
        for inst in registry.get_by_asset_class("FX"):
            assert inst.pip_value > 0, f"{inst.symbol} has non-positive pip_value: {inst.pip_value}"

    def test_all_futures_tick_values_are_positive(self):
        """All futures instruments must have positive tick_value."""
        registry = get_registry()
        for inst in registry.get_by_asset_class("FUTURES"):
            assert inst.tick_value > 0, f"{inst.symbol} has non-positive tick_value: {inst.tick_value}"

    def test_no_pip_values_dict_in_config_module(self):
        """Verify PIP_VALUES dict has been removed from config module."""
        from src.scanner import config as config_module

        # The module should not have PIP_VALUES as a module-level variable
        # (it may have a note in comments, but not as a dict)
        if hasattr(config_module, "PIP_VALUES"):
            # If it exists, it should not be a dict (it should be a comment reference)
            val = getattr(config_module, "PIP_VALUES")
            assert not isinstance(val, dict), "PIP_VALUES dict should not exist in config module"

    def test_no_pip_values_dict_in_execution_module(self):
        """Verify PIP_VALUES dict has been removed from execution module."""
        from src.scanner import execution as execution_module

        # The module should not have PIP_VALUES as a dict
        if hasattr(execution_module, "PIP_VALUES"):
            val = getattr(execution_module, "PIP_VALUES")
            assert not isinstance(val, dict), "PIP_VALUES dict should not exist in execution module"

    def test_registry_get_by_asset_class_fx(self):
        """Registry.get_by_asset_class('FX') should return FX instruments."""
        registry = get_registry()
        fx_instruments = registry.get_by_asset_class("FX")

        assert len(fx_instruments) > 0
        for inst in fx_instruments:
            assert inst.asset_class == "FX"
            assert inst.pip_value is not None
            assert inst.pip_value > 0

    def test_registry_get_by_asset_class_futures(self):
        """Registry.get_by_asset_class('FUTURES') should return FUTURES."""
        registry = get_registry()
        futures_instruments = registry.get_by_asset_class("FUTURES")

        assert len(futures_instruments) > 0
        for inst in futures_instruments:
            assert inst.asset_class == "FUTURES"
            assert inst.tick_value is not None
            assert inst.tick_value > 0

    def test_registry_get_all_symbols(self):
        """Registry.get_all_symbols() should return all instrument symbols."""
        registry = get_registry()
        symbols = registry.get_all_symbols()

        assert len(symbols) > 0
        assert "EUR_USD" in symbols
        assert "ES" in symbols

    def test_registry_contains_method(self):
        """Registry.contains() should check if symbol is registered."""
        registry = get_registry()

        assert registry.contains("EUR_USD")
        assert registry.contains("ES")
        assert not registry.contains("FAKE_PAIR")

    def test_registry_get_raises_keyerror_for_unknown(self):
        """Registry.get() should raise KeyError for unknown symbols."""
        registry = get_registry()

        with pytest.raises(KeyError):
            registry.get("UNKNOWN_PAIR")

    def test_registry_get_optional_returns_none_for_unknown(self):
        """Registry.get_optional() should return None for unknown symbols."""
        registry = get_registry()

        result = registry.get_optional("UNKNOWN_PAIR")
        assert result is None

    def test_registry_get_optional_returns_instrument_for_known(self):
        """Registry.get_optional() should return Instrument for known symbols."""
        registry = get_registry()

        inst = registry.get_optional("EUR_USD")
        assert inst is not None
        assert inst.symbol == "EUR_USD"
        assert inst.pip_value == 0.0001


class TestPipValuesUsageReplacement:
    """Verify that all PIP_VALUES usages have been replaced with registry calls."""

    def test_scanner_config_initializes_pip_values_from_registry(self):
        """ScannerConfig.pip_values field should be initialized from registry."""
        cfg = ScannerConfig()

        # The pip_values field should contain registry values
        assert "EUR_USD" in cfg.pip_values
        assert cfg.pip_values["EUR_USD"] == 0.0001
        assert cfg.pip_values["USD_JPY"] == 0.01

    def test_pip_value_fallback_behavior_consistent(self):
        """Fallback behavior should be consistent: unknown pairs → 0.0001."""
        cfg = ScannerConfig()
        registry = get_registry()

        # All three modules should return the same fallback value
        cfg_result = cfg.get_pip_value("FAKE_PAIR")
        metrics_result = trading_metrics_get_pip_value("FAKE_PAIR")
        filter_result = spread_filter_get_pip_value("FAKE_PAIR")

        assert cfg_result == 0.0001
        assert metrics_result == 0.0001
        assert filter_result == 0.0001
