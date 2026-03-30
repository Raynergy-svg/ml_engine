"""Shared types for broker integrations.

Provides dataclasses for common broker operations and data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class CandleData:
    """OHLCV candle data from broker.

    Attributes:
        time: Candle open time (UTC).
        open: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Volume traded.
    """
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """Validate candle data integrity."""
        if self.high < self.low:
            raise ValueError(
                f"Invalid candle: high ({self.high}) < low ({self.low})"
            )
        if self.open < 0 or self.high < 0 or self.low < 0 or self.close < 0:
            raise ValueError("Candle prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Candle volume cannot be negative")


@dataclass(frozen=True)
class OrderResult:
    """Result of a place_order() call.

    Attributes:
        trade_id: Unique trade identifier from broker.
        fill_price: Price at which order was filled.
        status: Order status (e.g., "PENDING", "FILLED", "REJECTED").
        error_message: Optional error details if status is REJECTED.
    """
    trade_id: str
    fill_price: float
    status: str
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order result."""
        if self.fill_price < 0:
            raise ValueError("Fill price cannot be negative")
        if not self.trade_id:
            raise ValueError("trade_id cannot be empty")
        if not self.status:
            raise ValueError("status cannot be empty")


@dataclass(frozen=True)
class PositionInfo:
    """Current open position info.

    Attributes:
        instrument: Instrument symbol (e.g., "EUR_USD").
        direction: Position direction ("BUY" or "SELL").
        quantity: Lot size / contract count.
        avg_price: Average entry price.
        unrealized_pnl: Current unrealized P/L in account currency.
    """
    instrument: str
    direction: str
    quantity: float
    avg_price: float
    unrealized_pnl: float

    def __post_init__(self) -> None:
        """Validate position info."""
        if not self.instrument:
            raise ValueError("instrument cannot be empty")
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.avg_price < 0:
            raise ValueError("avg_price cannot be negative")


@dataclass(frozen=True)
class TradeInfo:
    """Closed or closing trade info.

    Attributes:
        trade_id: Unique trade identifier.
        instrument: Instrument symbol (e.g., "EUR_USD").
        direction: Trade direction ("BUY" or "SELL").
        quantity: Lot size / contract count.
        entry_price: Entry price.
        sl_price: Optional stop loss price.
        tp_price: Optional take profit price.
    """
    trade_id: str
    instrument: str
    direction: str
    quantity: float
    entry_price: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate trade info."""
        if not self.trade_id:
            raise ValueError("trade_id cannot be empty")
        if not self.instrument:
            raise ValueError("instrument cannot be empty")
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.entry_price < 0:
            raise ValueError("entry_price cannot be negative")
        if self.sl_price is not None and self.sl_price < 0:
            raise ValueError("sl_price cannot be negative")
        if self.tp_price is not None and self.tp_price < 0:
            raise ValueError("tp_price cannot be negative")


@dataclass(frozen=True)
class AccountSummary:
    """Account-level summary info.

    Attributes:
        nav: Net asset value (account balance).
        balance: Base balance.
        unrealized_pnl: Total unrealized P/L across all positions.
        margin_used: Margin currently in use.
    """
    nav: float
    balance: float
    unrealized_pnl: float
    margin_used: float

    def __post_init__(self) -> None:
        """Validate account summary."""
        if self.nav < 0:
            raise ValueError("nav cannot be negative")
        if self.balance < 0:
            raise ValueError("balance cannot be negative")
        if self.margin_used < 0:
            raise ValueError("margin_used cannot be negative")
