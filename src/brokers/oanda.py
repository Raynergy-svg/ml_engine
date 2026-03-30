"""OANDA broker client implementation.

Wraps OandaPracticeClient and implements the BrokerClient interface.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.brokers.base import BrokerClient
from src.brokers.instrument import Instrument
from src.brokers.types import (
    AccountSummary,
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
)
from src.utils.oanda_practice import (
    OandaApiError,
    OandaPracticeClient,
    OandaPracticeConfig,
)

logger = logging.getLogger(__name__)


class OandaBroker(BrokerClient):
    """OANDA broker client implementation.

    Thin adapter that wraps OandaPracticeClient and provides the BrokerClient
    interface. Converts OANDA-specific responses to typed dataclasses.

    Example:
        client = OandaBroker.from_env()
        client.connect()
        candles = client.fetch_candles(
            Instrument.fx("EUR_USD", "EUR_USD", pip_value=10.0),
            "H1",
            100
        )
        quote = client.get_price_quote(instrument)
        order = client.place_order(instrument, "BUY", 1000, 1.0800, 1.1000, 1.0900)
    """

    def __init__(self, client: Optional[OandaPracticeClient] = None) -> None:
        """Initialize OandaBroker with an optional OandaPracticeClient.

        If client is None, creates one from environment variables.

        Args:
            client: Optional OandaPracticeClient instance. If None, creates from env.

        Raises:
            OSError: If client is None and required env vars are missing.
        """
        if client is None:
            self._client = OandaPracticeClient.from_env()
        else:
            self._client = client

    @classmethod
    def from_env(cls) -> OandaBroker:
        """Create OandaBroker from environment variables.

        Expected env vars:
        - OANDA_API_TOKEN (or OANDA_API_KEY)
        - OANDA_ACCOUNT_ID

        Returns:
            OandaBroker instance.

        Raises:
            OSError: If required env vars are missing.
        """
        client = OandaPracticeClient.from_env()
        return cls(client)

    def connect(self) -> None:
        """Establish connection to broker.

        OANDA is REST-based, so this is a no-op. Connection happens per-request.
        """
        logger.debug("OandaBroker.connect() called (no-op for REST API)")

    def disconnect(self) -> None:
        """Close connection to broker.

        OANDA is REST-based, so this is a no-op. Can be called multiple times.
        """
        logger.debug("OandaBroker.disconnect() called (no-op for REST API)")

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
        try:
            logger.debug(
                f"Fetching {count} {granularity} candles for {instrument.broker_symbol}"
            )
            response = self._client.get_candles(
                instrument.broker_symbol,
                granularity=granularity,
                count=count,
            )

            candles: List[CandleData] = []
            candle_list = response.get("candles", [])

            for candle_dict in candle_list:
                try:
                    # OANDA candle format: {time, bid: {o,h,l,c}, ask: {o,h,l,c}, mid: {o,h,l,c}, volume, complete}
                    # We use 'mid' prices for OHLCV
                    mid = candle_dict.get("mid", {})
                    time_str = candle_dict.get("time")
                    volume = float(candle_dict.get("volume", 0.0))

                    # Parse ISO 8601 timestamp
                    if time_str:
                        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    else:
                        dt = datetime.now(tz=timezone.utc)

                    candle = CandleData(
                        time=dt,
                        open=float(mid.get("o", 0.0)),
                        high=float(mid.get("h", 0.0)),
                        low=float(mid.get("l", 0.0)),
                        close=float(mid.get("c", 0.0)),
                        volume=volume,
                    )
                    candles.append(candle)
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse candle data: {e}. Skipping candle."
                    )
                    continue

            logger.debug(f"Fetched {len(candles)} candles for {instrument.broker_symbol}")
            return candles

        except OandaApiError as e:
            logger.error(
                f"OANDA API error fetching candles: {e.status_code} {str(e)}"
            )
            raise
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            raise

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
        """
        try:
            # Validate direction
            if direction not in ("BUY", "SELL"):
                error_msg = f"Invalid direction: {direction}"
                logger.error(error_msg)
                return OrderResult(
                    trade_id="error_invalid_direction",
                    fill_price=entry_price,
                    status="REJECTED",
                    error_message=error_msg,
                )

            # Convert direction to signed units for OANDA
            units = int(quantity)
            if direction == "SELL":
                units = -units

            logger.debug(
                f"Placing {direction} order: {quantity} units of {instrument.broker_symbol} "
                f"SL={sl_price}, TP={tp_price}"
            )

            response = self._client.create_market_order(
                instrument=instrument.broker_symbol,
                units=units,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )

            # Extract trade details from response
            try:
                trade_opened = (
                    (response.get("orderFillTransaction") or {})
                    .get("tradeOpened", {})
                )
                trade_id = str(
                    trade_opened.get("tradeID") or trade_opened.get("id") or ""
                )

                if not trade_id:
                    # Try alternative extraction path
                    tx = (
                        response.get("orderFillTransaction")
                        or response.get("orderCreateTransaction")
                        or {}
                    )
                    trades_opened = tx.get("tradesOpened", [])
                    if trades_opened and isinstance(trades_opened, list):
                        trade_id = str(
                            trades_opened[0].get("tradeID")
                            or trades_opened[0].get("id")
                            or ""
                        )

                # Extract fill price (use entry price if not in response)
                fill_price = float(
                    (response.get("orderFillTransaction") or {}).get("price")
                    or entry_price
                )

                if not trade_id:
                    error_msg = "No trade ID in OANDA response"
                    logger.error(error_msg)
                    return OrderResult(
                        trade_id="error_no_trade_id",
                        fill_price=fill_price,
                        status="REJECTED",
                        error_message=error_msg,
                    )

                logger.info(
                    f"Order placed successfully: trade_id={trade_id}, "
                    f"fill_price={fill_price}, direction={direction}"
                )
                return OrderResult(
                    trade_id=trade_id,
                    fill_price=fill_price,
                    status="FILLED",
                )

            except (ValueError, KeyError, TypeError) as e:
                error_msg = f"Failed to parse OANDA order response: {e}"
                logger.error(error_msg)
                return OrderResult(
                    trade_id="error_parse_response",
                    fill_price=entry_price,
                    status="REJECTED",
                    error_message=error_msg,
                )

        except OandaApiError as e:
            error_msg = f"OANDA API error: {e.status_code} {str(e)}"
            logger.error(error_msg)
            return OrderResult(
                trade_id="error_api_error",
                fill_price=entry_price,
                status="REJECTED",
                error_message=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error placing order: {e}"
            logger.error(error_msg)
            return OrderResult(
                trade_id="error_unexpected",
                fill_price=entry_price,
                status="REJECTED",
                error_message=error_msg,
            )

    def close_trade(self, trade_id: str) -> bool:
        """Close/cancel a specific trade.

        Args:
            trade_id: Trade identifier from place_order().

        Returns:
            True if trade was closed, False if not found or already closed.

        Raises:
            Exception: If close operation fails.
        """
        try:
            if not trade_id:
                logger.warning("close_trade called with empty trade_id")
                return False

            logger.debug(f"Closing trade {trade_id}")
            response = self._client.close_trade(trade_id=trade_id)

            # Check if close was successful
            if response and response.get("orderFillTransaction"):
                logger.info(f"Trade {trade_id} closed successfully")
                return True

            logger.warning(f"Trade {trade_id} close returned unexpected response")
            return False

        except OandaApiError as e:
            logger.error(f"OANDA API error closing trade {trade_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            raise

    def get_nav(self) -> float:
        """Get current net asset value (account balance).

        Returns:
            NAV in account currency.

        Raises:
            Exception: If fetch fails.
        """
        try:
            logger.debug("Fetching account NAV")
            summary = self.get_account_summary()
            nav = summary.nav
            logger.debug(f"Account NAV: {nav}")
            return nav

        except Exception as e:
            logger.error(f"Error fetching NAV: {e}")
            raise

    def get_open_positions(self) -> List[PositionInfo]:
        """Get all currently open positions.

        Returns:
            List of PositionInfo objects. Empty list if no open positions.

        Raises:
            Exception: If fetch fails.
        """
        try:
            logger.debug("Fetching open positions")
            response = self._client.get_open_positions()

            positions: List[PositionInfo] = []
            positions_list = response.get("positions", [])

            for pos_dict in positions_list:
                try:
                    instrument = str(pos_dict.get("instrument", ""))

                    # OANDA position format: {long: {units, averagePrice, ...}, short: {...}, ...}
                    long_side = pos_dict.get("long", {})
                    short_side = pos_dict.get("short", {})

                    long_units = float(long_side.get("units", 0.0) or 0.0)
                    short_units = float(short_side.get("units", 0.0) or 0.0)
                    long_avg_price = float(long_side.get("averagePrice", 0.0) or 0.0)
                    short_avg_price = float(short_side.get("averagePrice", 0.0) or 0.0)
                    long_pnl = float(long_side.get("unrealizedPL", 0.0) or 0.0)
                    short_pnl = float(short_side.get("unrealizedPL", 0.0) or 0.0)

                    # Create PositionInfo for each active side
                    if long_units > 0:
                        pos = PositionInfo(
                            instrument=instrument,
                            direction="BUY",
                            quantity=long_units,
                            avg_price=long_avg_price,
                            unrealized_pnl=long_pnl,
                        )
                        positions.append(pos)

                    if short_units != 0:  # short_units is negative in OANDA
                        pos = PositionInfo(
                            instrument=instrument,
                            direction="SELL",
                            quantity=abs(short_units),
                            avg_price=short_avg_price,
                            unrealized_pnl=short_pnl,
                        )
                        positions.append(pos)

                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse position data: {e}. Skipping position.")
                    continue

            logger.debug(f"Fetched {len(positions)} open positions")
            return positions

        except OandaApiError as e:
            logger.error(f"OANDA API error fetching positions: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            raise

    def get_price_quote(self, instrument: Instrument) -> Dict[str, float]:
        """Get current bid/ask prices for an instrument.

        Args:
            instrument: Trading instrument.

        Returns:
            Dictionary with 'bid' and 'ask' keys.

        Raises:
            Exception: If fetch fails.
        """
        try:
            logger.debug(f"Fetching price quote for {instrument.broker_symbol}")
            quote = self._client.get_price_quote(instrument=instrument.broker_symbol)
            logger.debug(
                f"Price quote for {instrument.broker_symbol}: "
                f"bid={quote.get('bid')}, ask={quote.get('ask')}"
            )
            return quote

        except Exception as e:
            logger.error(
                f"Error fetching price quote for {instrument.broker_symbol}: {e}"
            )
            raise

    def get_trades(self) -> List[TradeInfo]:
        """Get list of recent trades (closed or closing).

        Returns:
            List of TradeInfo objects.

        Raises:
            Exception: If fetch fails.
        """
        try:
            logger.debug("Fetching trades")
            response = self._client.get_trades(state="ALL", count=500)

            trades: List[TradeInfo] = []
            trades_list = response.get("trades", [])

            for trade_dict in trades_list:
                try:
                    trade_id = str(trade_dict.get("id", ""))
                    instrument = str(trade_dict.get("instrument", ""))
                    units = float(trade_dict.get("initialUnits", 0.0) or 0.0)
                    entry_price = float(trade_dict.get("price", 0.0))
                    sl_price = None
                    tp_price = None

                    # Extract SL/TP if present
                    stop_loss = trade_dict.get("stopLossOrder")
                    if stop_loss and isinstance(stop_loss, dict):
                        try:
                            sl_price = float(stop_loss.get("priceBound") or stop_loss.get("price"))
                        except (ValueError, TypeError):
                            pass

                    take_profit = trade_dict.get("takeProfitOrder")
                    if take_profit and isinstance(take_profit, dict):
                        try:
                            tp_price = float(take_profit.get("priceBound") or take_profit.get("price"))
                        except (ValueError, TypeError):
                            pass

                    direction = "BUY" if units > 0 else "SELL"
                    quantity = abs(units)

                    if not trade_id or not instrument or quantity <= 0:
                        continue

                    trade = TradeInfo(
                        trade_id=trade_id,
                        instrument=instrument,
                        direction=direction,
                        quantity=quantity,
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                    )
                    trades.append(trade)

                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse trade data: {e}. Skipping trade.")
                    continue

            logger.debug(f"Fetched {len(trades)} trades")
            return trades

        except OandaApiError as e:
            logger.error(f"OANDA API error fetching trades: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            raise

    def get_account_summary(self) -> AccountSummary:
        """Get account-level summary (NAV, balance, margin, unrealized PnL).

        Returns:
            AccountSummary object.

        Raises:
            Exception: If fetch fails.
        """
        try:
            logger.debug("Fetching account summary")
            response = self._client.get_account_summary()

            # OANDA account summary format: {account: {balance, nav, unrealizedPL, marginUsed, ...}}
            account_data = response.get("account", {})

            nav = float(account_data.get("nav", 0.0) or 0.0)
            balance = float(account_data.get("balance", 0.0) or 0.0)
            unrealized_pnl = float(account_data.get("unrealizedPL", 0.0) or 0.0)
            margin_used = float(account_data.get("marginUsed", 0.0) or 0.0)

            summary = AccountSummary(
                nav=nav,
                balance=balance,
                unrealized_pnl=unrealized_pnl,
                margin_used=margin_used,
            )

            logger.debug(
                f"Account summary: nav={nav}, balance={balance}, "
                f"unrealized_pnl={unrealized_pnl}, margin_used={margin_used}"
            )
            return summary

        except OandaApiError as e:
            logger.error(f"OANDA API error fetching account summary: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
            raise
