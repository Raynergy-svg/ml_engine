"""Instrument Registry for FX and Futures definitions.

Provides centralized registry of supported trading instruments with factory methods
for FX pairs and futures contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import logging

from src.brokers.instrument import Instrument, AssetClass

logger = logging.getLogger(__name__)


class InstrumentRegistry:
    """Registry of supported trading instruments.

    Manages FX pairs and futures contracts with lookup methods by symbol,
    asset class, or other criteria.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._instruments: Dict[str, Instrument] = {}

    def register(self, instrument: Instrument) -> None:
        """Register an instrument in the registry.

        Args:
            instrument: Instrument to register.

        Raises:
            ValueError: If instrument symbol already registered.
        """
        if instrument.symbol in self._instruments:
            raise ValueError(
                f"Instrument {instrument.symbol} already registered"
            )
        self._instruments[instrument.symbol] = instrument
        logger.debug(f"Registered {instrument.asset_class} instrument: {instrument.symbol}")

    def get(self, symbol: str) -> Instrument:
        """Get instrument by symbol.

        Args:
            symbol: Instrument symbol (e.g., "EUR_USD", "ES").

        Returns:
            Instrument object.

        Raises:
            KeyError: If instrument not found.
        """
        if symbol not in self._instruments:
            raise KeyError(f"Instrument not found: {symbol}")
        return self._instruments[symbol]

    def get_optional(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol, or None if not found.

        Args:
            symbol: Instrument symbol.

        Returns:
            Instrument object or None.
        """
        return self._instruments.get(symbol)

    def get_by_asset_class(self, asset_class: AssetClass) -> List[Instrument]:
        """Get all instruments of a given asset class.

        Args:
            asset_class: "FX" or "FUTURES".

        Returns:
            List of instruments matching the asset class.
        """
        return [
            instrument
            for instrument in self._instruments.values()
            if instrument.asset_class == asset_class
        ]

    def get_all(self) -> List[Instrument]:
        """Get all registered instruments.

        Returns:
            List of all instruments.
        """
        return list(self._instruments.values())

    def get_all_symbols(self) -> List[str]:
        """Get all registered instrument symbols.

        Returns:
            List of symbol strings.
        """
        return list(self._instruments.keys())

    def contains(self, symbol: str) -> bool:
        """Check if symbol is registered.

        Args:
            symbol: Instrument symbol.

        Returns:
            True if registered, False otherwise.
        """
        return symbol in self._instruments

    def __len__(self) -> int:
        """Return count of registered instruments."""
        return len(self._instruments)

    def __repr__(self) -> str:
        """Return string representation."""
        fx_count = len(self.get_by_asset_class("FX"))
        fut_count = len(self.get_by_asset_class("FUTURES"))
        return (
            f"InstrumentRegistry({fx_count} FX instruments, "
            f"{fut_count} FUTURES instruments)"
        )


def get_default_registry() -> InstrumentRegistry:
    """Create and populate the default instrument registry.

    Includes all major FX pairs and standard futures contracts.

    Returns:
        Fully populated InstrumentRegistry.
    """
    registry = InstrumentRegistry()

    # FX Pairs (24 instruments)
    # Major pairs (4)
    registry.register(
        Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="GBP_USD",
            broker_symbol="GBP_USD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="USD_JPY",
            broker_symbol="USD_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="USD_CHF",
            broker_symbol="USD_CHF",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )

    # Commodity pairs (3)
    registry.register(
        Instrument.fx(
            symbol="AUD_USD",
            broker_symbol="AUD_USD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="USD_CAD",
            broker_symbol="USD_CAD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="NZD_USD",
            broker_symbol="NZD_USD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )

    # Cross pairs (5)
    registry.register(
        Instrument.fx(
            symbol="EUR_GBP",
            broker_symbol="EUR_GBP",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="EUR_JPY",
            broker_symbol="EUR_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="GBP_JPY",
            broker_symbol="GBP_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="AUD_JPY",
            broker_symbol="AUD_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="EUR_AUD",
            broker_symbol="EUR_AUD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )

    # Additional crosses (7)
    registry.register(
        Instrument.fx(
            symbol="GBP_AUD",
            broker_symbol="GBP_AUD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="EUR_CHF",
            broker_symbol="EUR_CHF",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="GBP_CHF",
            broker_symbol="GBP_CHF",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="EUR_NZD",
            broker_symbol="EUR_NZD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="GBP_NZD",
            broker_symbol="GBP_NZD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="AUD_NZD",
            broker_symbol="AUD_NZD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="NZD_JPY",
            broker_symbol="NZD_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )

    # JPY crosses (3)
    registry.register(
        Instrument.fx(
            symbol="CAD_JPY",
            broker_symbol="CAD_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="CHF_JPY",
            broker_symbol="CHF_JPY",
            pip_value=0.01,
            price_precision=3,
            margin_requirement=1.0,
        )
    )

    # Asian pairs (3)
    registry.register(
        Instrument.fx(
            symbol="USD_SGD",
            broker_symbol="USD_SGD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="EUR_CAD",
            broker_symbol="EUR_CAD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )
    registry.register(
        Instrument.fx(
            symbol="GBP_CAD",
            broker_symbol="GBP_CAD",
            pip_value=0.0001,
            price_precision=5,
            margin_requirement=1.0,
        )
    )

    # Futures Contracts (4 instruments)
    # ES (E-mini S&P 500)
    registry.register(
        Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
            price_precision=2,
            margin_requirement=10.0,
            exchange="CME",
        )
    )

    # NQ (E-mini NASDAQ 100)
    registry.register(
        Instrument.futures(
            symbol="NQ",
            broker_symbol="NQ",
            tick_size=0.25,
            tick_value=5.00,
            multiplier=20.0,
            price_precision=2,
            margin_requirement=10.0,
            exchange="CME",
        )
    )

    # CL (Crude Oil)
    registry.register(
        Instrument.futures(
            symbol="CL",
            broker_symbol="CL",
            tick_size=0.01,
            tick_value=10.00,
            multiplier=1000.0,
            price_precision=2,
            margin_requirement=10.0,
            exchange="NYMEX",
        )
    )

    # GC (Gold)
    registry.register(
        Instrument.futures(
            symbol="GC",
            broker_symbol="GC",
            tick_size=0.10,
            tick_value=10.00,
            multiplier=100.0,
            price_precision=1,
            margin_requirement=10.0,
            exchange="NYMEX",
        )
    )

    # ZB (30-Year US Treasury Bond)
    registry.register(
        Instrument.futures(
            symbol="ZB",
            broker_symbol="ZB",
            tick_size=1 / 32,  # 1/32 of a point
            tick_value=31.25,
            multiplier=1000.0,
            price_precision=5,
            margin_requirement=5.0,
            exchange="CBOT",
        )
    )

    # 6E (EUR/USD Futures — same underlying as FX but with real CME volume)
    registry.register(
        Instrument.futures(
            symbol="6E",
            broker_symbol="6E",
            tick_size=0.00005,
            tick_value=6.25,
            multiplier=125000.0,
            price_precision=5,
            margin_requirement=3.0,
            exchange="CME",
        )
    )

    logger.info(
        f"Initialized default registry: {len(registry.get_by_asset_class('FX'))} FX, "
        f"{len(registry.get_by_asset_class('FUTURES'))} FUTURES"
    )
    return registry


# Module-level singleton
_default_registry: Optional[InstrumentRegistry] = None


def get_registry() -> InstrumentRegistry:
    """Get or create the default registry singleton.

    Returns:
        The default InstrumentRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = get_default_registry()
    return _default_registry


def reset_registry() -> None:
    """Reset the global singleton (for testing)."""
    global _default_registry
    _default_registry = None
