"""FX Trade Execution Module.

This module handles trade execution via OANDA API with proper error handling,
logging, and result tracking.

Extracted from main.py lines 5556-6100 as part of Phase 3a decomposition.

Example usage:
    from src.trading.execution import TradeExecutor, OrderRequest

    executor = TradeExecutor.from_env()
    result = executor.execute(OrderRequest(
        instrument="EUR_USD",
        direction="long",
        units=10000,
        stop_loss_pips=15.0,
        take_profit_pips=30.0,
    ))

    if result.success:
        print(f"Order filled: {result.order_id} @ {result.fill_price}")
    else:
        print(f"Order failed: {result.error_message}")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a trade execution attempt.

    Attributes:
        success: Whether the order was successfully filled.
        order_id: OANDA order ID if created.
        trade_id: OANDA trade ID if opened.
        fill_price: Actual fill price (may differ from requested).
        units: Number of units filled (positive=long, negative=short).
        error_message: Human-readable error description if failed.
        error_code: OANDA error code if applicable.
        reject_reason: OANDA reject reason if order was rejected.
        status: Detailed order status.
        raw_response: Full OANDA API response for debugging.
    """
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    units: Optional[int] = None
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    reject_reason: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.success:
            return f"ExecutionResult(SUCCESS, trade={self.trade_id}, filled={self.units}@{self.fill_price})"
        return f"ExecutionResult(FAILED: {self.error_message})"

    @classmethod
    def from_oanda_response(cls, response: Dict[str, Any]) -> "ExecutionResult":
        """Parse an OANDA order response into an ExecutionResult.

        Args:
            response: Raw OANDA API response from create_market_order.

        Returns:
            ExecutionResult with parsed data.
        """
        # Check for fill transaction (successful market order)
        fill_tx = response.get("orderFillTransaction") or {}
        create_tx = response.get("orderCreateTransaction") or {}
        cancel_tx = response.get("orderCancelTransaction") or {}
        reject_tx = response.get("orderRejectTransaction") or {}

        # Extract trade ID from fill transaction
        trade_id = None
        if fill_tx:
            trade_opened = fill_tx.get("tradeOpened") or {}
            trade_id = trade_opened.get("tradeID") or trade_opened.get("id")
            if not trade_id:
                trades_opened = fill_tx.get("tradesOpened") or []
                if trades_opened:
                    trade_id = trades_opened[0].get("tradeID") or trades_opened[0].get("id")

        # Parse fill price and units
        fill_price = None
        units = None
        if fill_tx:
            try:
                fill_price = float(fill_tx.get("price", 0))
            except (TypeError, ValueError):
                pass
            try:
                units = int(fill_tx.get("units", 0))
            except (TypeError, ValueError):
                pass

        # Determine status
        if fill_tx and trade_id:
            return cls(
                success=True,
                order_id=fill_tx.get("orderID") or create_tx.get("id"),
                trade_id=str(trade_id),
                fill_price=fill_price,
                units=units,
                status=OrderStatus.FILLED,
                raw_response=response,
            )

        # Check for cancellation
        if cancel_tx:
            cancel_reason = cancel_tx.get("reason", "")
            return cls(
                success=False,
                order_id=cancel_tx.get("orderID") or create_tx.get("id"),
                error_message=f"Order cancelled: {cancel_reason}",
                reject_reason=cancel_reason,
                status=OrderStatus.CANCELLED,
                raw_response=response,
            )

        # Check for rejection
        if reject_tx:
            reject_reason = reject_tx.get("rejectReason", "")
            return cls(
                success=False,
                order_id=reject_tx.get("orderID"),
                error_message=f"Order rejected: {reject_reason}",
                reject_reason=reject_reason,
                status=OrderStatus.REJECTED,
                raw_response=response,
            )

        # Generic error case
        error_msg = response.get("errorMessage", "Unknown error")
        return cls(
            success=False,
            error_message=error_msg,
            error_code=response.get("errorCode"),
            status=OrderStatus.REJECTED,
            raw_response=response,
        )


@dataclass
class OrderRequest:
    """Request to place a trade order.

    Attributes:
        instrument: OANDA instrument name (e.g., "EUR_USD").
        direction: Trade direction ("long" or "short").
        units: Number of units to trade (absolute value, direction from direction field).
        stop_loss_pips: Stop loss distance in pips.
        take_profit_pips: Take profit distance in pips.
        entry_price: Reference price for SL/TP calculation (defaults to current).
        trailing_stop_pips: Optional trailing stop distance in pips.
        price_bound: Max slippage price bound.
        client_tag: Tag for order tracking.
        metadata: Additional order metadata.
    """
    instrument: str
    direction: Literal["long", "short"]
    units: int
    stop_loss_pips: float = 15.0
    take_profit_pips: float = 30.0
    entry_price: Optional[float] = None
    trailing_stop_pips: Optional[float] = None
    price_bound: Optional[float] = None
    client_tag: str = "ml_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate order request."""
        if not self.instrument:
            raise ValueError("instrument is required")
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got: {self.direction}")
        if self.units <= 0:
            raise ValueError(f"units must be positive, got: {self.units}")
        if self.stop_loss_pips <= 0:
            raise ValueError(f"stop_loss_pips must be positive, got: {self.stop_loss_pips}")
        if self.take_profit_pips <= 0:
            raise ValueError(f"take_profit_pips must be positive, got: {self.take_profit_pips}")

    @property
    def signed_units(self) -> int:
        """Return units with sign based on direction."""
        return self.units if self.direction == "long" else -self.units

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "units": self.units,
            "stop_loss_pips": self.stop_loss_pips,
            "take_profit_pips": self.take_profit_pips,
            "entry_price": self.entry_price,
            "trailing_stop_pips": self.trailing_stop_pips,
            "price_bound": self.price_bound,
            "client_tag": self.client_tag,
        }


@dataclass(frozen=True)
class QuoteInfo:
    """Current market quote for an instrument.

    Attributes:
        instrument: OANDA instrument name.
        bid: Best bid price.
        ask: Best ask price.
        spread_pips: Bid-ask spread in pips.
        timestamp: Quote timestamp.
    """
    instrument: str
    bid: float
    ask: float
    spread_pips: float
    timestamp: Optional[str] = None

    @property
    def mid(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid + self.ask) / 2.0

    @classmethod
    def from_oanda_pricing(cls, instrument: str, pricing: Dict[str, Any]) -> "QuoteInfo":
        """Create QuoteInfo from OANDA pricing response."""
        bid = float(pricing.get("bid", 0))
        ask = float(pricing.get("ask", 0))

        # Calculate spread in pips
        pip_factor = 100.0 if instrument.endswith("_JPY") else 10000.0
        spread_pips = (ask - bid) * pip_factor

        return cls(
            instrument=instrument,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            timestamp=pricing.get("time"),
        )


@dataclass(frozen=True)
class PositionInfo:
    """Information about an open position.

    Attributes:
        instrument: OANDA instrument name.
        units: Net units (positive=long, negative=short).
        unrealized_pl: Current unrealized P&L.
        average_price: Average entry price.
    """
    instrument: str
    units: int
    unrealized_pl: float
    average_price: float

    @property
    def direction(self) -> str:
        """Position direction based on units."""
        return "long" if self.units > 0 else "short"


# =============================================================================
# Trade Executor
# =============================================================================

class TradeExecutor:
    """Wrapper for OANDA trade execution with enhanced error handling.

    This class provides a clean interface for trade execution, handling:
    - Order building with proper SL/TP prices
    - Price bound calculation for slippage protection
    - Error handling and result parsing
    - Position and quote queries
    - Auto-close scheduling for scalping

    Example:
        executor = TradeExecutor.from_env()

        # Get current quote
        quote = executor.get_quote("EUR_USD")
        print(f"EUR/USD: {quote.bid}/{quote.ask} (spread: {quote.spread_pips:.1f} pips)")

        # Place a trade
        order = OrderRequest(
            instrument="EUR_USD",
            direction="long",
            units=10000,
            stop_loss_pips=15.0,
            take_profit_pips=30.0,
        )
        result = executor.execute(order)
    """

    def __init__(
        self,
        client: Any,  # OandaPracticeClient
        *,
        default_price_bound_buffer_pips: float = 1.0,
        verbose: bool = False,
    ):
        """Initialize TradeExecutor.

        Args:
            client: OANDA practice client instance.
            default_price_bound_buffer_pips: Default buffer for price bounds in pips.
            verbose: Enable verbose logging.
        """
        self._client = client
        self._default_price_bound_buffer_pips = default_price_bound_buffer_pips
        self._verbose = verbose
        self._last_trade_id: Optional[str] = None

    @classmethod
    def from_env(cls, *, verbose: bool = False) -> "TradeExecutor":
        """Create TradeExecutor from environment variables.

        Requires OANDA_API_TOKEN and OANDA_ACCOUNT_ID environment variables.

        Args:
            verbose: Enable verbose logging.

        Returns:
            TradeExecutor instance.

        Raises:
            EnvironmentError: If required environment variables are missing.
        """
        from src.utils.oanda_practice import OandaPracticeClient
        client = OandaPracticeClient.from_env()
        return cls(client, verbose=verbose)

    @property
    def client(self) -> Any:
        """Access underlying OANDA client."""
        return self._client

    @property
    def last_trade_id(self) -> Optional[str]:
        """ID of the last trade opened by this executor."""
        return self._last_trade_id

    # =========================================================================
    # Quote and Position Methods
    # =========================================================================

    def get_quote(self, instrument: str) -> QuoteInfo:
        """Get current quote for an instrument.

        Args:
            instrument: OANDA instrument name (e.g., "EUR_USD").

        Returns:
            QuoteInfo with bid/ask prices.

        Raises:
            ValueError: If quote cannot be retrieved.
        """
        try:
            pricing = self._client.get_price_quote(instrument=instrument)
            return QuoteInfo.from_oanda_pricing(instrument, pricing)
        except Exception as e:
            logger.error(f"Failed to get quote for {instrument}: {e}")
            raise ValueError(f"Could not fetch quote for {instrument}: {e}") from e

    def get_open_positions(self) -> List[PositionInfo]:
        """Get all open positions.

        Returns:
            List of PositionInfo for each open position.
        """
        try:
            response = self._client.get_open_positions()
            positions = response.get("positions") or []

            result = []
            for pos in positions:
                instrument = pos.get("instrument")
                if not instrument:
                    continue

                # Parse long and short units
                long_info = pos.get("long") or {}
                short_info = pos.get("short") or {}

                long_units = int(long_info.get("units", 0) or 0)
                short_units = int(short_info.get("units", 0) or 0)
                net_units = long_units + short_units

                if net_units == 0:
                    continue

                # Get unrealized P&L
                unrealized_pl = float(pos.get("unrealizedPL", 0) or 0)

                # Get average price
                if net_units > 0:
                    avg_price = float(long_info.get("averagePrice", 0) or 0)
                else:
                    avg_price = float(short_info.get("averagePrice", 0) or 0)

                result.append(PositionInfo(
                    instrument=instrument,
                    units=net_units,
                    unrealized_pl=unrealized_pl,
                    average_price=avg_price,
                ))

            return result

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def get_position(self, instrument: str) -> Optional[PositionInfo]:
        """Get position for a specific instrument.

        Args:
            instrument: OANDA instrument name.

        Returns:
            PositionInfo if position exists, None otherwise.
        """
        positions = self.get_open_positions()
        for pos in positions:
            if pos.instrument == instrument:
                return pos
        return None

    # =========================================================================
    # Price Calculation Helpers
    # =========================================================================

    @staticmethod
    def pip_size(instrument: str) -> float:
        """Get pip size for an instrument.

        Args:
            instrument: OANDA instrument name.

        Returns:
            Pip size (0.01 for JPY pairs, 0.0001 for others).
        """
        return 0.01 if instrument.endswith("_JPY") else 0.0001

    @staticmethod
    def format_price(instrument: str, price: float) -> str:
        """Format price with appropriate precision for instrument.

        Args:
            instrument: OANDA instrument name.
            price: Price to format.

        Returns:
            Formatted price string.
        """
        decimals = 3 if instrument.endswith("_JPY") else 5
        return f"{float(price):.{decimals}f}"

    def calculate_sl_tp_prices(
        self,
        order: OrderRequest,
        current_price: float,
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit prices.

        Args:
            order: Order request with SL/TP pips.
            current_price: Reference price for calculation.

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        pip = self.pip_size(order.instrument)
        sl_distance = order.stop_loss_pips * pip
        tp_distance = order.take_profit_pips * pip

        if order.direction == "long":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        return sl_price, tp_price

    def calculate_price_bound(
        self,
        instrument: str,
        direction: str,
        current_quote: QuoteInfo,
        buffer_pips: Optional[float] = None,
    ) -> float:
        """Calculate price bound for slippage protection.

        Args:
            instrument: OANDA instrument name.
            direction: Trade direction ("long" or "short").
            current_quote: Current market quote.
            buffer_pips: Buffer in pips (defaults to instance default).

        Returns:
            Price bound for the order.
        """
        if buffer_pips is None:
            buffer_pips = self._default_price_bound_buffer_pips

        pip = self.pip_size(instrument)
        buffer_price = buffer_pips * pip

        if direction == "long":
            return current_quote.ask + buffer_price
        return current_quote.bid - buffer_price

    # =========================================================================
    # Order Execution
    # =========================================================================

    def execute(
        self,
        order: OrderRequest,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute a trade order.

        Args:
            order: Order request to execute.
            dry_run: If True, validate but don't place the order.

        Returns:
            ExecutionResult with execution details.
        """
        instrument = order.instrument

        # Get current quote
        try:
            quote = self.get_quote(instrument)
        except ValueError as e:
            return ExecutionResult(
                success=False,
                error_message=str(e),
                status=OrderStatus.REJECTED,
            )

        # Use provided price or current market price
        reference_price = order.entry_price or (quote.ask if order.direction == "long" else quote.bid)

        # Calculate SL/TP prices
        sl_price, tp_price = self.calculate_sl_tp_prices(order, reference_price)

        # Calculate price bound
        if order.price_bound is not None:
            price_bound = order.price_bound
        else:
            price_bound = self.calculate_price_bound(instrument, order.direction, quote)

        # Log order details
        if self._verbose:
            logger.info(
                f"Executing order: {order.direction.upper()} {order.units} {instrument} "
                f"@ ~{reference_price:.5f} SL={sl_price:.5f} TP={tp_price:.5f}"
            )

        # Dry run - just validate
        if dry_run:
            return ExecutionResult(
                success=True,
                units=order.signed_units,
                fill_price=reference_price,
                status=OrderStatus.PENDING,
                error_message="DRY RUN - order not placed",
            )

        # Calculate trailing stop if requested
        trailing_stop_distance = None
        if order.trailing_stop_pips is not None and order.trailing_stop_pips > 0:
            trailing_stop_distance = order.trailing_stop_pips * self.pip_size(instrument)

        # Place market order
        try:
            response = self._client.create_market_order(
                instrument=instrument,
                units=order.signed_units,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                trailing_stop_distance=trailing_stop_distance,
                price_bound=price_bound,
                client_tag=order.client_tag,
            )
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                status=OrderStatus.REJECTED,
            )

        # Parse result
        result = ExecutionResult.from_oanda_response(response)

        # Track last trade ID
        if result.trade_id:
            self._last_trade_id = result.trade_id

        if self._verbose:
            logger.info(f"Order result: {result}")

        return result

    def execute_with_smart_splitting(
        self,
        order: OrderRequest,
        *,
        split_threshold_units: int = 500000,
        max_splits: int = 5,
        split_delay_seconds: float = 0.5,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[ExecutionResult]:
        """Execute a large order with smart splitting.

        For large orders, split into smaller chunks to reduce slippage.

        Args:
            order: Order request to execute.
            split_threshold_units: Only split if order exceeds this size.
            max_splits: Maximum number of splits.
            split_delay_seconds: Delay between split orders.
            progress_callback: Called with (split_index, total_splits, filled_units).

        Returns:
            List of ExecutionResults for each split.
        """
        if order.units < split_threshold_units:
            return [self.execute(order)]

        # Calculate split sizes
        num_splits = min(max_splits, (order.units // split_threshold_units) + 1)
        base_size = order.units // num_splits
        remainder = order.units % num_splits

        results = []
        total_filled = 0

        for i in range(num_splits):
            # Add remainder to first split
            split_units = base_size + (1 if i < remainder else 0)

            # Create split order
            split_order = OrderRequest(
                instrument=order.instrument,
                direction=order.direction,
                units=split_units,
                stop_loss_pips=order.stop_loss_pips,
                take_profit_pips=order.take_profit_pips,
                entry_price=order.entry_price,
                trailing_stop_pips=order.trailing_stop_pips,
                price_bound=order.price_bound,
                client_tag=f"{order.client_tag}_split_{i+1}",
                metadata={**order.metadata, "split": i + 1, "total_splits": num_splits},
            )

            result = self.execute(split_order)
            results.append(result)

            if result.success and result.units:
                total_filled += abs(result.units)

            if progress_callback:
                progress_callback(i + 1, num_splits, total_filled)

            # Delay between splits (except last)
            if i < num_splits - 1:
                time.sleep(split_delay_seconds)

        return results

    # =========================================================================
    # Position Management
    # =========================================================================

    def close_position(self, instrument: str) -> ExecutionResult:
        """Close all positions for an instrument.

        Args:
            instrument: OANDA instrument name.

        Returns:
            ExecutionResult with close details.
        """
        try:
            response = self._client.close_position(instrument=instrument)

            # Parse response - look for close transaction
            long_close = response.get("longOrderFillTransaction") or {}
            short_close = response.get("shortOrderFillTransaction") or {}

            # Prefer whichever side had units
            close_tx = long_close if long_close else short_close

            if close_tx:
                return ExecutionResult(
                    success=True,
                    trade_id=close_tx.get("tradesClosed", [{}])[0].get("tradeID") if close_tx.get("tradesClosed") else None,
                    fill_price=float(close_tx.get("price", 0)) if close_tx.get("price") else None,
                    units=int(close_tx.get("units", 0)) if close_tx.get("units") else None,
                    status=OrderStatus.FILLED,
                    raw_response=response,
                )

            return ExecutionResult(
                success=True,
                status=OrderStatus.FILLED,
                error_message="Position closed (no fill details)",
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Close position failed for {instrument}: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                status=OrderStatus.REJECTED,
            )

    def close_trade(
        self,
        trade_id: str,
        *,
        ensemble: Optional[Any] = None,
        prediction: float = 0.5,
        confidence: float = 50.0,
        features: Optional[Any] = None,
    ) -> ExecutionResult:
        """Close a specific trade by ID.

        Args:
            trade_id: OANDA trade ID.
            ensemble: Optional ensemble for drift tracking.
            prediction: Model prediction at entry.
            confidence: Confidence score at entry.
            features: Feature array at entry.

        Returns:
            ExecutionResult with close details.
        """
        try:
            response = self._client.close_trade(
                trade_id=trade_id,
                ensemble=ensemble,
                prediction=prediction,
                confidence=confidence,
                features=features,
            )

            close_tx = response.get("orderFillTransaction") or {}

            return ExecutionResult(
                success=True,
                trade_id=trade_id,
                fill_price=float(close_tx.get("price", 0)) if close_tx.get("price") else None,
                units=int(close_tx.get("units", 0)) if close_tx.get("units") else None,
                status=OrderStatus.FILLED,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Close trade failed for {trade_id}: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                status=OrderStatus.REJECTED,
            )

    def close_all_positions(self) -> List[ExecutionResult]:
        """Close all open positions.

        Returns:
            List of ExecutionResults for each closed position.
        """
        positions = self.get_open_positions()
        results = []

        for pos in positions:
            result = self.close_position(pos.instrument)
            results.append(result)

        return results

    # =========================================================================
    # Auto-Close Scheduling
    # =========================================================================

    def schedule_auto_close(
        self,
        instrument: str,
        delay_seconds: float,
        *,
        trade_id: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Schedule automatic position close after a delay.

        This is useful for scalping strategies with time-based exits.
        Runs in a daemon thread and is best-effort only.

        Args:
            instrument: OANDA instrument name.
            delay_seconds: Seconds until close.
            trade_id: Specific trade ID to close (defaults to last trade).
            verbose: Enable verbose logging.
        """
        target_trade_id = trade_id or self._last_trade_id

        def _worker():
            try:
                if verbose:
                    logger.info(f"Auto-close: sleeping {delay_seconds:.1f}s before closing {instrument}")

                time.sleep(max(0.0, delay_seconds))

                # Try to close specific trade first, fallback to position
                if target_trade_id:
                    try:
                        result = self.close_trade(target_trade_id)
                        if verbose:
                            logger.info(f"Auto-close: closed trade {target_trade_id}: {result}")
                        return
                    except Exception:
                        pass

                result = self.close_position(instrument)
                if verbose:
                    logger.info(f"Auto-close: closed position {instrument}: {result}")

            except Exception as e:
                logger.warning(f"Auto-close failed for {instrument}: {e}")

        thread = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"auto-close-{instrument}",
        )
        thread.start()
