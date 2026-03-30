"""Unit tests for OandaBroker implementation.

Tests verify that OandaBroker correctly wraps OandaPracticeClient
and provides proper BrokerClient interface compliance.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from src.brokers.oanda import OandaBroker
from src.brokers.instrument import Instrument
from src.brokers.types import (
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
    AccountSummary,
)
from src.utils.oanda_practice import OandaApiError, OandaPracticeClient


class TestOandaBrokerInit:
    """Test OandaBroker initialization."""

    def test_init_with_client(self):
        """Test initialization with explicit client."""
        mock_client = Mock(spec=OandaPracticeClient)
        broker = OandaBroker(client=mock_client)
        assert broker._client is mock_client

    def test_init_from_env(self):
        """Test initialization from environment variables."""
        with patch.object(OandaPracticeClient, "from_env") as mock_from_env:
            mock_client = Mock(spec=OandaPracticeClient)
            mock_from_env.return_value = mock_client

            broker = OandaBroker()
            assert broker._client is mock_client
            mock_from_env.assert_called_once()

    def test_from_env_classmethod(self):
        """Test from_env classmethod."""
        with patch.object(OandaPracticeClient, "from_env") as mock_from_env:
            mock_client = Mock(spec=OandaPracticeClient)
            mock_from_env.return_value = mock_client

            broker = OandaBroker.from_env()
            assert isinstance(broker, OandaBroker)
            assert broker._client is mock_client


class TestOandaBrokerConnection:
    """Test connection methods (should be no-ops for REST API)."""

    def test_connect_noop(self):
        """Test that connect is a no-op."""
        mock_client = Mock(spec=OandaPracticeClient)
        broker = OandaBroker(client=mock_client)
        broker.connect()  # Should not raise

    def test_disconnect_noop(self):
        """Test that disconnect is a no-op."""
        mock_client = Mock(spec=OandaPracticeClient)
        broker = OandaBroker(client=mock_client)
        broker.disconnect()  # Should not raise


class TestFetchCandles:
    """Test fetch_candles method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    @pytest.fixture
    def instrument_eur_usd(self):
        """Create EUR_USD instrument."""
        return Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=10.0,
        )

    def test_fetch_candles_success(self, broker, instrument_eur_usd):
        """Test successful candle fetch."""
        # Mock OANDA response
        broker._client.get_candles.return_value = {
            "candles": [
                {
                    "time": "2026-03-29T10:00:00Z",
                    "mid": {"o": 1.0850, "h": 1.0860, "l": 1.0840, "c": 1.0855},
                    "volume": 1000.0,
                },
                {
                    "time": "2026-03-29T11:00:00Z",
                    "mid": {"o": 1.0855, "h": 1.0870, "l": 1.0850, "c": 1.0865},
                    "volume": 1200.0,
                },
            ]
        }

        candles = broker.fetch_candles(instrument_eur_usd, "H1", 2)

        assert len(candles) == 2
        assert isinstance(candles[0], CandleData)
        assert candles[0].open == 1.0850
        assert candles[0].high == 1.0860
        assert candles[0].low == 1.0840
        assert candles[0].close == 1.0855
        assert candles[0].volume == 1000.0
        assert candles[1].close == 1.0865

    def test_fetch_candles_empty(self, broker, instrument_eur_usd):
        """Test fetch with no candles returned."""
        broker._client.get_candles.return_value = {"candles": []}
        candles = broker.fetch_candles(instrument_eur_usd, "H1", 100)
        assert candles == []

    def test_fetch_candles_invalid_data(self, broker, instrument_eur_usd):
        """Test fetch with malformed candle data (should skip invalid candles)."""
        broker._client.get_candles.return_value = {
            "candles": [
                {
                    "time": "2026-03-29T10:00:00Z",
                    "mid": {"o": 1.0850, "h": 1.0860, "l": 1.0840, "c": 1.0855},
                    "volume": 1000.0,
                },
                # Malformed: high < low (should be skipped)
                {
                    "time": "2026-03-29T11:00:00Z",
                    "mid": {"o": 1.0855, "h": 1.0840, "l": 1.0850, "c": 1.0865},
                    "volume": 1200.0,
                },
            ]
        }

        candles = broker.fetch_candles(instrument_eur_usd, "H1", 2)
        # Should only include valid candle
        assert len(candles) == 1

    def test_fetch_candles_api_error(self, broker, instrument_eur_usd):
        """Test fetch with OANDA API error."""
        broker._client.get_candles.side_effect = OandaApiError(
            500, "Server error"
        )

        with pytest.raises(OandaApiError):
            broker.fetch_candles(instrument_eur_usd, "H1", 100)

    def test_fetch_candles_generic_error(self, broker, instrument_eur_usd):
        """Test fetch with generic error."""
        broker._client.get_candles.side_effect = Exception("Connection error")

        with pytest.raises(Exception):
            broker.fetch_candles(instrument_eur_usd, "H1", 100)


class TestPlaceOrder:
    """Test place_order method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    @pytest.fixture
    def instrument_eur_usd(self):
        """Create EUR_USD instrument."""
        return Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=10.0,
        )

    def test_place_order_buy_success(self, broker, instrument_eur_usd):
        """Test successful BUY order placement."""
        broker._client.create_market_order.return_value = {
            "orderFillTransaction": {
                "price": "1.0900",
                "tradeOpened": {"tradeID": "12345"},
            }
        }

        result = broker.place_order(
            instrument_eur_usd,
            "BUY",
            1000.0,
            sl_price=1.0850,
            tp_price=1.0950,
            entry_price=1.0900,
        )

        assert result.status == "FILLED"
        assert result.trade_id == "12345"
        assert result.fill_price == 1.09
        assert result.error_message is None

        # Verify client was called with correct units (positive for BUY)
        broker._client.create_market_order.assert_called_once()
        call_kwargs = broker._client.create_market_order.call_args[1]
        assert call_kwargs["units"] == 1000
        assert call_kwargs["stop_loss_price"] == 1.0850
        assert call_kwargs["take_profit_price"] == 1.0950

    def test_place_order_sell_success(self, broker, instrument_eur_usd):
        """Test successful SELL order placement."""
        broker._client.create_market_order.return_value = {
            "orderFillTransaction": {
                "price": "1.0900",
                "tradeOpened": {"tradeID": "12346"},
            }
        }

        result = broker.place_order(
            instrument_eur_usd,
            "SELL",
            1000.0,
            sl_price=1.0950,
            tp_price=1.0850,
            entry_price=1.0900,
        )

        assert result.status == "FILLED"
        assert result.trade_id == "12346"

        # Verify client was called with negative units for SELL
        call_kwargs = broker._client.create_market_order.call_args[1]
        assert call_kwargs["units"] == -1000

    def test_place_order_invalid_direction(self, broker, instrument_eur_usd):
        """Test order with invalid direction."""
        result = broker.place_order(
            instrument_eur_usd,
            "INVALID",
            1000.0,
            sl_price=1.0850,
            tp_price=1.0950,
            entry_price=1.0900,
        )

        assert result.status == "REJECTED"
        assert result.error_message is not None
        assert "Invalid direction" in result.error_message

    def test_place_order_api_error(self, broker, instrument_eur_usd):
        """Test order placement with OANDA API error."""
        broker._client.create_market_order.side_effect = OandaApiError(
            400, "Invalid request"
        )

        result = broker.place_order(
            instrument_eur_usd,
            "BUY",
            1000.0,
            sl_price=1.0850,
            tp_price=1.0950,
            entry_price=1.0900,
        )

        assert result.status == "REJECTED"
        assert result.error_message is not None
        assert "OANDA API error" in result.error_message

    def test_place_order_malformed_response(self, broker, instrument_eur_usd):
        """Test order placement with malformed response."""
        broker._client.create_market_order.return_value = {
            "orderFillTransaction": {}  # Missing tradeOpened
        }

        result = broker.place_order(
            instrument_eur_usd,
            "BUY",
            1000.0,
            sl_price=1.0850,
            tp_price=1.0950,
            entry_price=1.0900,
        )

        assert result.status == "REJECTED"
        assert "No trade ID" in result.error_message


class TestCloseTrade:
    """Test close_trade method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    def test_close_trade_success(self, broker):
        """Test successful trade close."""
        broker._client.close_trade.return_value = {
            "orderFillTransaction": {"price": "1.0910"}
        }

        result = broker.close_trade("12345")
        assert result is True
        broker._client.close_trade.assert_called_once_with(trade_id="12345")

    def test_close_trade_not_found(self, broker):
        """Test closing non-existent trade."""
        broker._client.close_trade.return_value = {}

        result = broker.close_trade("99999")
        assert result is False

    def test_close_trade_empty_id(self, broker):
        """Test close_trade with empty trade_id."""
        result = broker.close_trade("")
        assert result is False

    def test_close_trade_api_error(self, broker):
        """Test close_trade with API error."""
        broker._client.close_trade.side_effect = OandaApiError(404, "Not found")

        with pytest.raises(OandaApiError):
            broker.close_trade("12345")


class TestGetNav:
    """Test get_nav method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    def test_get_nav_success(self, broker):
        """Test successful NAV fetch."""
        broker._client.get_account_summary.return_value = {
            "account": {"nav": 50000.0}
        }

        nav = broker.get_nav()
        assert nav == 50000.0

    def test_get_nav_api_error(self, broker):
        """Test NAV fetch with API error."""
        broker._client.get_account_summary.side_effect = OandaApiError(
            500, "Server error"
        )

        with pytest.raises(OandaApiError):
            broker.get_nav()


class TestGetOpenPositions:
    """Test get_open_positions method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    def test_get_open_positions_empty(self, broker):
        """Test with no open positions."""
        broker._client.get_open_positions.return_value = {"positions": []}

        positions = broker.get_open_positions()
        assert positions == []

    def test_get_open_positions_with_long_and_short(self, broker):
        """Test with both long and short positions."""
        broker._client.get_open_positions.return_value = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {
                        "units": 1000.0,
                        "averagePrice": 1.0850,
                        "unrealizedPL": 100.0,
                    },
                    "short": {
                        "units": -500.0,
                        "averagePrice": 1.0900,
                        "unrealizedPL": -50.0,
                    },
                }
            ]
        }

        positions = broker.get_open_positions()

        assert len(positions) == 2
        # Check long position
        long_pos = [p for p in positions if p.direction == "BUY"][0]
        assert long_pos.instrument == "EUR_USD"
        assert long_pos.quantity == 1000.0
        assert long_pos.avg_price == 1.0850
        assert long_pos.unrealized_pnl == 100.0

        # Check short position
        short_pos = [p for p in positions if p.direction == "SELL"][0]
        assert short_pos.instrument == "EUR_USD"
        assert short_pos.quantity == 500.0  # Absolute value
        assert short_pos.avg_price == 1.0900
        assert short_pos.unrealized_pnl == -50.0

    def test_get_open_positions_api_error(self, broker):
        """Test with API error."""
        broker._client.get_open_positions.side_effect = OandaApiError(
            500, "Server error"
        )

        with pytest.raises(OandaApiError):
            broker.get_open_positions()


class TestGetPriceQuote:
    """Test get_price_quote method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    @pytest.fixture
    def instrument_eur_usd(self):
        """Create EUR_USD instrument."""
        return Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR_USD",
            pip_value=10.0,
        )

    def test_get_price_quote_success(self, broker, instrument_eur_usd):
        """Test successful price quote fetch."""
        broker._client.get_price_quote.return_value = {
            "bid": 1.0895,
            "ask": 1.0905,
        }

        quote = broker.get_price_quote(instrument_eur_usd)

        assert quote["bid"] == 1.0895
        assert quote["ask"] == 1.0905
        broker._client.get_price_quote.assert_called_once_with(
            instrument="EUR_USD"
        )

    def test_get_price_quote_api_error(self, broker, instrument_eur_usd):
        """Test price quote fetch with API error."""
        broker._client.get_price_quote.side_effect = ValueError(
            "Missing bid/ask"
        )

        with pytest.raises(ValueError):
            broker.get_price_quote(instrument_eur_usd)


class TestGetTrades:
    """Test get_trades method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    def test_get_trades_empty(self, broker):
        """Test with no trades."""
        broker._client.get_trades.return_value = {"trades": []}

        trades = broker.get_trades()
        assert trades == []

    def test_get_trades_with_sl_tp(self, broker):
        """Test with trades including SL and TP."""
        broker._client.get_trades.return_value = {
            "trades": [
                {
                    "id": "12345",
                    "instrument": "EUR_USD",
                    "initialUnits": 1000.0,
                    "price": 1.0850,
                    "stopLossOrder": {"price": 1.0800},
                    "takeProfitOrder": {"price": 1.0950},
                },
                {
                    "id": "12346",
                    "instrument": "GBP_USD",
                    "initialUnits": -500.0,
                    "price": 1.2650,
                    "stopLossOrder": None,
                    "takeProfitOrder": None,
                },
            ]
        }

        trades = broker.get_trades()

        assert len(trades) == 2
        assert trades[0].trade_id == "12345"
        assert trades[0].direction == "BUY"
        assert trades[0].sl_price == 1.0800
        assert trades[0].tp_price == 1.0950

        assert trades[1].trade_id == "12346"
        assert trades[1].direction == "SELL"
        assert trades[1].quantity == 500.0

    def test_get_trades_api_error(self, broker):
        """Test trades fetch with API error."""
        broker._client.get_trades.side_effect = OandaApiError(
            500, "Server error"
        )

        with pytest.raises(OandaApiError):
            broker.get_trades()


class TestGetAccountSummary:
    """Test get_account_summary method."""

    @pytest.fixture
    def broker(self):
        """Create broker with mock client."""
        mock_client = Mock(spec=OandaPracticeClient)
        return OandaBroker(client=mock_client)

    def test_get_account_summary_success(self, broker):
        """Test successful account summary fetch."""
        broker._client.get_account_summary.return_value = {
            "account": {
                "nav": 50000.0,
                "balance": 50000.0,
                "unrealizedPL": 1000.0,
                "marginUsed": 5000.0,
            }
        }

        summary = broker.get_account_summary()

        assert isinstance(summary, AccountSummary)
        assert summary.nav == 50000.0
        assert summary.balance == 50000.0
        assert summary.unrealized_pnl == 1000.0
        assert summary.margin_used == 5000.0

    def test_get_account_summary_api_error(self, broker):
        """Test account summary fetch with API error."""
        broker._client.get_account_summary.side_effect = OandaApiError(
            500, "Server error"
        )

        with pytest.raises(OandaApiError):
            broker.get_account_summary()


class TestBrokerClientInterfaceCompliance:
    """Verify OandaBroker implements all BrokerClient abstract methods."""

    def test_all_abstract_methods_implemented(self):
        """Verify all BrokerClient abstract methods are implemented."""
        mock_client = Mock(spec=OandaPracticeClient)
        broker = OandaBroker(client=mock_client)

        # Verify all required methods exist and are callable
        assert callable(broker.connect)
        assert callable(broker.disconnect)
        assert callable(broker.fetch_candles)
        assert callable(broker.place_order)
        assert callable(broker.close_trade)
        assert callable(broker.get_nav)
        assert callable(broker.get_open_positions)
        assert callable(broker.get_price_quote)
        assert callable(broker.get_trades)
        assert callable(broker.get_account_summary)

    def test_broker_is_instance_of_broker_client(self):
        """Verify OandaBroker is instance of BrokerClient."""
        from src.brokers.base import BrokerClient

        mock_client = Mock(spec=OandaPracticeClient)
        broker = OandaBroker(client=mock_client)

        assert isinstance(broker, BrokerClient)
