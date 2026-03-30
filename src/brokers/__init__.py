"""Broker abstraction layer.

Provides base classes and types for multi-broker integration.

Lazy-loaded modules:
- types: Shared types (CandleData, OrderResult, PositionInfo, TradeInfo, AccountSummary)
- instrument: Instrument dataclass with FX/FUTURES factory methods
- registry: InstrumentRegistry for managing FX and futures definitions
- base: BrokerClient ABC
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy imports: only load when explicitly accessed
_TYPES_LOADED = False
_INSTRUMENT_LOADED = False
_REGISTRY_LOADED = False
_BASE_LOADED = False


def __getattr__(name: str):
    """Lazy-load broker modules on demand."""
    global _TYPES_LOADED, _INSTRUMENT_LOADED, _REGISTRY_LOADED, _BASE_LOADED

    if name == "types":
        if not _TYPES_LOADED:
            logger.debug("Lazy-loading src.brokers.types")
        from src.brokers import types as _types
        _TYPES_LOADED = True
        return _types

    if name == "instrument":
        if not _INSTRUMENT_LOADED:
            logger.debug("Lazy-loading src.brokers.instrument")
        from src.brokers import instrument as _instrument
        _INSTRUMENT_LOADED = True
        return _instrument

    if name == "registry":
        if not _REGISTRY_LOADED:
            logger.debug("Lazy-loading src.brokers.registry")
        from src.brokers import registry as _registry
        _REGISTRY_LOADED = True
        return _registry

    if name == "base":
        if not _BASE_LOADED:
            logger.debug("Lazy-loading src.brokers.base")
        from src.brokers import base as _base
        _BASE_LOADED = True
        return _base

    # Explicitly export common types for convenience
    if name == "Instrument":
        from src.brokers.instrument import Instrument
        return Instrument

    if name == "InstrumentRegistry":
        from src.brokers.registry import InstrumentRegistry
        return InstrumentRegistry

    if name == "get_registry":
        from src.brokers.registry import get_registry
        return get_registry

    if name == "get_default_registry":
        from src.brokers.registry import get_default_registry
        return get_default_registry

    if name == "BrokerClient":
        from src.brokers.base import BrokerClient
        return BrokerClient

    if name == "CandleData":
        from src.brokers.types import CandleData
        return CandleData

    if name == "OrderResult":
        from src.brokers.types import OrderResult
        return OrderResult

    if name == "PositionInfo":
        from src.brokers.types import PositionInfo
        return PositionInfo

    if name == "TradeInfo":
        from src.brokers.types import TradeInfo
        return TradeInfo

    if name == "AccountSummary":
        from src.brokers.types import AccountSummary
        return AccountSummary

    if name == "RollCalendar":
        from src.brokers.roll_calendar import RollCalendar
        return RollCalendar

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Instrument",
    "InstrumentRegistry",
    "get_registry",
    "get_default_registry",
    "BrokerClient",
    "CandleData",
    "OrderResult",
    "PositionInfo",
    "TradeInfo",
    "AccountSummary",
    "RollCalendar",
]
