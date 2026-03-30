"""Abstract base class for broker client implementations.

Defines the interface that all broker integrations must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.brokers.instrument import Instrument
from src.brokers.types import (
    AccountSummary,
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
)

logger = logging.getLogger(__name__)


class BrokerClient(ABC):
    """Abstract base class for broker integrations.

    All broker implementations must inherit from this class and implement
    all abstract methods. Returns are typed objects (from src.brokers.types),
    never raw dictionaries.

    Example usage:
        client = OandaBrokerClient(config)
        client.connect()
        candles = client.fetch_candles(instrument, "H1", 100)
        result = client.place_order(instrument, "BUY", 10000, 1.0850, 1.0950, 1.0900)
        if result.status == "FILLED":
            positions = client.get_open_positions()
            pnl = client.get_nav()
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to broker.

        Raises:
            Exception: If connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to broker.

        Can be called multiple times safely.
        """
        pass

    @abstractmethod
    def fetch_candles(
        self,
        instrument: Instrument,
        granularity: str,
        count: int,
    ) -> List[CandleData]:
        """Fetch historical candle data.

        Args:
            instrument: Trading instrument.
            granularity: Candle size (e.g., "M1", "H1", "D").
            count: Number of candles to fetch.

        Returns:
            List of CandleData objects, oldest first.

        Raises:
            Exception: If fetch fails.
        """
        pass

    @abstractmethod
    def place_order(
        self,
        instrument: Instrument,
        direction: str,
        quantity: float,
        sl_price: float,
        tp_price: float,
        entry_price: float,
    ) -> OrderResult:
        """Place a new order.

        Args:
            instrument: Trading instrument.
            direction: "BUY" or "SELL".
            quantity: Lot size or contract count.
            sl_price: Stop loss price.
            tp_price: Take profit price.
            entry_price: Entry price (used for validation/logging).

        Returns:
            OrderResult with trade_id, fill_price, and status.

        Raises:
            Exception: If placement fails.
        """
        pass

    @abstractmethod
    def close_trade(self, trade_id: str) -> bool:
        """Close/cancel a specific trade.

        Args:
            trade_id: Trade identifier from place_order().

        Returns:
            True if trade was closed, False if not found or already closed.

        Raises:
            Exception: If close operation fails.
        """
        pass

    @abstractmethod
    def get_nav(self) -> float:
        """Get current net asset value (account balance).

        Returns:
            NAV in account currency.

        Raises:
            Exception: If fetch fails.
        """
        pass

    @abstractmethod
    def get_open_positions(self) -> List[PositionInfo]:
        """Get all currently open positions.

        Returns:
            List of PositionInfo objects. Empty list if no open positions.

        Raises:
            Exception: If fetch fails.
        """
        pass

    @abstractmethod
    def get_price_quote(self, instrument: Instrument) -> Dict[str, float]:
        """Get current bid/ask prices for an instrument.

        Args:
            instrument: Trading instrument.

        Returns:
            Dictionary with 'bid' and 'ask' keys.

        Raises:
            Exception: If fetch fails.
        """
        pass

    @abstractmethod
    def get_trades(self) -> List[TradeInfo]:
        """Get list of recent trades (closed or closing).

        Returns:
            List of TradeInfo objects.

        Raises:
            Exception: If fetch fails.
        """
        pass

    @abstractmethod
    def get_account_summary(self) -> AccountSummary:
        """Get account-level summary (NAV, balance, margin, unrealized PnL).

        Returns:
            AccountSummary object.

        Raises:
            Exception: If fetch fails.
        """
        pass
