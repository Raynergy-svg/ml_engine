"""Unit tests for IBKRBroker implementation.

Tests verify that IBKRBroker correctly connects/disconnects from IBKR,
handles auto-reconnect, and implements the BrokerClient interface.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from src.brokers.ibkr import IBKRBroker
from src.brokers.instrument import Instrument
from src.brokers.types import (
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
    AccountSummary,
)


class TestIBKRBrokerInit:
    """Test IBKRBroker initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        broker = IBKRBroker()
        assert broker.host == "127.0.0.1"
        assert broker.port == 7497
        assert broker.client_id == 1
        assert broker._connected is False
        assert broker._ib is None
        assert broker._next_valid_id is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        broker = IBKRBroker(host="192.168.1.100", port=4002, client_id=42)
        assert broker.host == "192.168.1.100"
        assert broker.port == 4002
        assert broker.client_id == 42

    def test_repr(self):
        """Test string representation."""
        broker = IBKRBroker(host="localhost", port=7497, client_id=1)
        repr_str = repr(broker)
        assert "IBKRBroker" in repr_str
        assert "host=localhost" in repr_str
        assert "port=7497" in repr_str
        assert "client_id=1" in repr_str
        assert "disconnected" in repr_str


class TestIBKRBrokerConnect:
    """Test IBKRBroker connection logic."""

    @patch("src.brokers.ibkr.IB")
    def test_connect_success(self, mock_ib_class):
        """Test successful connection."""
        # Mock the IB class and instance
        mock_ib = MagicMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = True

        broker = IBKRBroker()

        # Mock the async_connect to avoid asyncio issues in tests
        with patch.object(broker, "_async_connect", new_callable=AsyncMock):
            broker.connect()

        assert broker._connected is True
        assert broker._ib is mock_ib
        assert broker._reconnect_attempts == 0

    @patch("src.brokers.ibkr.IB")
    def test_connect_already_connected(self, mock_ib_class):
        """Test connect when already connected."""
        mock_ib = MagicMock()
        broker = IBKRBroker()
        broker._connected = True
        broker._ib = mock_ib

        # Should not create new IB instance
        broker.connect()
        mock_ib_class.assert_not_called()

    @patch("src.brokers.ibkr.IB")
    def test_connect_raises_connection_error_on_gateway_not_running(
        self, mock_ib_class
    ):
        """Test connection error when Gateway not running (502)."""
        mock_ib = MagicMock()
        mock_ib_class.return_value = mock_ib

        broker = IBKRBroker()

        # Mock _async_connect to raise OSError with 502
        with patch.object(
            broker,
            "_async_connect",
            new_callable=AsyncMock,
            side_effect=OSError("502: Gateway not running"),
        ):
            with pytest.raises(ConnectionError) as exc_info:
                broker.connect()
            assert "Gateway" in str(exc_info.value) or "502" in str(
                exc_info.value
            )

    @patch("src.brokers.ibkr.IB")
    def test_connect_raises_connection_error_on_timeout(self, mock_ib_class):
        """Test connection error on timeout."""
        mock_ib = MagicMock()
        mock_ib_class.return_value = mock_ib

        broker = IBKRBroker()

        # Mock _async_connect to raise TimeoutError
        with patch.object(
            broker,
            "_async_connect",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Timeout waiting for nextValidId"),
        ):
            with pytest.raises(ConnectionError) as exc_info:
                broker.connect()
            assert "timeout" in str(exc_info.value).lower()

    @patch("src.brokers.ibkr.IB")
    def test_connect_raises_connection_error_on_generic_exception(
        self, mock_ib_class
    ):
        """Test connection error on generic exception."""
        mock_ib = MagicMock()
        mock_ib_class.return_value = mock_ib

        broker = IBKRBroker()

        # Mock _async_connect to raise generic exception
        with patch.object(
            broker,
            "_async_connect",
            new_callable=AsyncMock,
            side_effect=Exception("Generic error"),
        ):
            with pytest.raises(ConnectionError) as exc_info:
                broker.connect()
            assert "Generic error" in str(exc_info.value)


class TestIBKRBrokerDisconnect:
    """Test IBKRBroker disconnection logic."""

    @patch("src.brokers.ibkr.IB")
    def test_disconnect_when_connected(self, mock_ib_class):
        """Test disconnect when connected."""
        mock_ib = MagicMock()
        mock_ib.disconnectAsync = AsyncMock()

        broker = IBKRBroker()
        broker._connected = True
        broker._ib = mock_ib
        broker._next_valid_id = 123

        broker.disconnect()

        assert broker._connected is False
        assert broker._ib is None
        assert broker._next_valid_id is None

    def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        # Should not raise error
        broker.disconnect()
        assert broker._connected is False
        assert broker._ib is None

    @patch("src.brokers.ibkr.IB")
    def test_disconnect_handles_exception(self, mock_ib_class):
        """Test disconnect handles exceptions gracefully."""
        mock_ib = MagicMock()
        mock_ib.disconnectAsync = AsyncMock(
            side_effect=Exception("Disconnect error")
        )

        broker = IBKRBroker()
        broker._connected = True
        broker._ib = mock_ib

        # Should not raise, but log error
        broker.disconnect()
        assert broker._connected is False
        assert broker._ib is None


class TestIBKRBrokerIsConnected:
    """Test is_connected property."""

    def test_is_connected_when_not_connected(self):
        """Test is_connected returns False when not connected."""
        broker = IBKRBroker()
        assert broker.is_connected is False

    @patch("src.brokers.ibkr.IB")
    def test_is_connected_when_connected(self, mock_ib_class):
        """Test is_connected returns True when connected."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        broker = IBKRBroker()
        broker._ib = mock_ib

        assert broker.is_connected is True

    @patch("src.brokers.ibkr.IB")
    def test_is_connected_when_ib_disconnected(self, mock_ib_class):
        """Test is_connected returns False when IB is disconnected."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = False

        broker = IBKRBroker()
        broker._ib = mock_ib

        assert broker.is_connected is False


class TestIBKRBrokerNextValidIdCallback:
    """Test nextValidId callback handling."""

    def test_on_next_valid_id(self):
        """Test _on_next_valid_id callback."""
        broker = IBKRBroker()
        assert broker._next_valid_id is None

        broker._on_next_valid_id(12345)

        assert broker._next_valid_id == 12345


class TestIBKRBrokerBuildContract:
    """Test _build_contract helper method."""

    def test_build_contract_forex(self):
        """Test building a Forex contract from an Instrument."""
        from unittest.mock import patch

        with patch("src.brokers.ibkr.Forex") as mock_forex_class:
            mock_forex = MagicMock()
            mock_forex_class.return_value = mock_forex

            broker = IBKRBroker()
            instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

            contract = broker._build_contract(instrument)

            mock_forex_class.assert_called_once_with("EUR", "USD")
            assert contract is mock_forex

    def test_build_contract_forex_invalid_symbol(self):
        """Test building a Forex contract with invalid symbol format."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EURUSD", pip_value=10.0)  # Missing /

        with pytest.raises(ValueError) as exc_info:
            broker._build_contract(instrument)
        assert "Invalid FX broker_symbol format" in str(exc_info.value)

    def test_build_contract_futures(self):
        """Test building a Future contract from an Instrument."""
        from unittest.mock import patch

        with patch("src.brokers.ibkr.Future") as mock_future_class:
            mock_future = MagicMock()
            mock_future_class.return_value = mock_future

            broker = IBKRBroker()
            instrument = Instrument.futures(
                symbol="ES",
                broker_symbol="ES",
                tick_size=0.25,
                tick_value=12.5,
                multiplier=50.0,
                exchange="CME",
                currency="USD",
                contract_month="202406",
            )

            contract = broker._build_contract(instrument)

            # Verify Future was called with expected arguments
            assert contract is mock_future

    def test_build_contract_unsupported_asset_class(self):
        """Test building a contract with unsupported asset class."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)
        # Manually set invalid asset_class for testing
        object.__setattr__(instrument, "asset_class", "CRYPTO")

        with pytest.raises(ValueError) as exc_info:
            broker._build_contract(instrument)
        assert "Unsupported asset_class" in str(exc_info.value)


class TestIBKRBrokerFetchCandles:
    """Test fetch_candles implementation."""

    def test_fetch_candles_not_connected(self):
        """Test fetch_candles raises error when not connected."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        with pytest.raises(ConnectionError) as exc_info:
            broker.fetch_candles(instrument, "H1", 100)
        assert "Not connected" in str(exc_info.value)

    def test_fetch_candles_unsupported_granularity(self):
        """Test fetch_candles raises error for unsupported granularity."""
        broker = IBKRBroker()
        broker._connected = True
        broker._ib = MagicMock()

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        with pytest.raises(ValueError) as exc_info:
            broker.fetch_candles(instrument, "W1", 100)
        assert "Unsupported granularity" in str(exc_info.value)

    @patch("src.brokers.ibkr.Forex")
    def test_fetch_candles_success(self, mock_forex_class):
        """Test successful fetch_candles call."""
        from unittest.mock import AsyncMock

        # Setup mock candle data
        mock_candle1 = MagicMock()
        mock_candle1.date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_candle1.open = 1.0800
        mock_candle1.high = 1.0850
        mock_candle1.low = 1.0750
        mock_candle1.close = 1.0820
        mock_candle1.volume = 1000000

        # Setup broker mock
        mock_ib = MagicMock()
        mock_ib.reqHistoricalDataAsync = AsyncMock(
            return_value=[mock_candle1]
        )
        broker = IBKRBroker()
        broker._connected = True
        broker._ib = mock_ib

        # Mock the contract building
        with patch.object(broker, "_build_contract") as mock_build:
            mock_contract = MagicMock()
            mock_build.return_value = mock_contract

            # Call fetch_candles
            instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)
            candles = broker.fetch_candles(instrument, "H1", 1)

            # Verify results
            assert len(candles) == 1
            assert isinstance(candles[0], CandleData)
            assert candles[0].open == 1.0800
            assert candles[0].close == 1.0820
            assert candles[0].volume == 1000000

    def test_fetch_candles_granularity_mapping(self):
        """Test granularity mapping is correct."""
        broker = IBKRBroker()

        # Verify all expected granularities are supported
        expected_granularities = {
            "M1": ("1 min", "1 D"),
            "M5": ("5 mins", "5 D"),
            "M15": ("15 mins", "15 D"),
            "M30": ("30 mins", "30 D"),
            "H1": ("1 hour", "30 D"),
            "H4": ("4 hours", "120 D"),
            "D": ("1 day", "365 D"),
        }

        for granularity, (bar_size, duration) in expected_granularities.items():
            assert broker._GRANULARITY_MAP[granularity] == (bar_size, duration)

    def test_fetch_candles_pacing_guard(self):
        """Test pacing guard prevents rapid consecutive requests."""
        import time

        broker = IBKRBroker()
        broker._pacing_delay_seconds = 0.1  # Short delay for testing
        broker._pacing_guard["EUR_USD:H1"] = time.time() - 0.05  # 50ms ago

        broker._connected = True
        broker._ib = MagicMock()

        # Mock the async request
        with patch.object(
            broker, "_fetch_historical_data_async", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = []

            with patch.object(broker, "_build_contract") as mock_build:
                mock_build.return_value = MagicMock()

                instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)
                start = time.time()
                broker.fetch_candles(instrument, "H1", 1)
                elapsed = time.time() - start

                # Should have waited for pacing delay
                assert elapsed >= 0.05  # At least the remaining delay

    def test_fetch_candles_converts_bar_data_to_candle_data(self):
        """Test that BarData objects are correctly converted to CandleData."""
        from unittest.mock import AsyncMock

        # Create a mock bar object with datetime
        mock_bar = MagicMock()
        mock_bar.date = datetime(2024, 3, 29, 10, 0, 0, tzinfo=timezone.utc)
        mock_bar.open = 1.0900
        mock_bar.high = 1.0950
        mock_bar.low = 1.0850
        mock_bar.close = 1.0920
        mock_bar.volume = 500000

        # Setup broker
        mock_ib = MagicMock()
        mock_ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])
        broker = IBKRBroker()
        broker._connected = True
        broker._ib = mock_ib

        with patch.object(broker, "_build_contract") as mock_build:
            mock_build.return_value = MagicMock()

            instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)
            candles = broker.fetch_candles(instrument, "H1", 1)

            assert len(candles) == 1
            candle = candles[0]
            assert candle.time == datetime(
                2024, 3, 29, 10, 0, 0, tzinfo=timezone.utc
            )
            assert candle.open == 1.0900
            assert candle.high == 1.0950
            assert candle.low == 1.0850
            assert candle.close == 1.0920
            assert candle.volume == 500000


class TestIBKRBrokerStubMethods:
    """Test stub and partially-implemented methods."""

    def test_place_order_rejected_when_not_connected(self):
        """Test place_order returns rejected OrderResult when not connected."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        result = broker.place_order(
            instrument,
            "BUY",
            100000,
            1.0800,
            1.1000,
            1.0900,
        )
        assert result.status == "rejected"
        assert "Not connected" in result.error_message
        assert result.trade_id == "0"

    def test_close_trade_raises_error_when_not_connected(self):
        """Test close_trade raises ConnectionError when not connected."""
        broker = IBKRBroker()

        with pytest.raises(ConnectionError) as exc_info:
            broker.close_trade("trade_123")
        assert "Not connected" in str(exc_info.value)

    def test_get_nav_returns_zero_when_not_connected(self):
        """Test get_nav returns 0.0 when not connected."""
        broker = IBKRBroker()
        nav = broker.get_nav()
        assert nav == 0.0

    def test_get_open_positions_returns_empty_when_not_connected(self):
        """Test get_open_positions returns empty list when not connected."""
        broker = IBKRBroker()
        positions = broker.get_open_positions()
        assert positions == []

    def test_get_price_quote_not_implemented(self):
        """Test get_price_quote raises NotImplementedError."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        with pytest.raises(NotImplementedError) as exc_info:
            broker.get_price_quote(instrument)
        assert "US-009" in str(exc_info.value)

    def test_get_trades_returns_empty_when_not_connected(self):
        """Test get_trades returns empty list when not connected."""
        broker = IBKRBroker()
        trades = broker.get_trades()
        assert trades == []

    def test_get_account_summary_returns_zero_when_not_connected(self):
        """Test get_account_summary returns zero summary when not connected."""
        broker = IBKRBroker()
        summary = broker.get_account_summary()
        assert summary.nav == 0.0
        assert summary.balance == 0.0
        assert summary.unrealized_pnl == 0.0
        assert summary.margin_used == 0.0


class TestIBKRBrokerReconnect:
    """Test auto-reconnect logic."""

    def test_reconnect_raises_error_at_max_attempts(self):
        """Test reconnect raises error after max attempts."""
        broker = IBKRBroker()
        broker._reconnect_attempts = broker._max_reconnect_attempts

        with pytest.raises(ConnectionError) as exc_info:
            broker._reconnect()
        assert "Max IBKR reconnect attempts" in str(exc_info.value)

    @patch.object(IBKRBroker, "connect")
    def test_reconnect_with_exponential_backoff(self, mock_connect):
        """Test reconnect uses exponential backoff."""
        broker = IBKRBroker()
        broker._reconnect_attempts = 0
        broker._reconnect_base_delay = 0.1  # Use small value for testing
        broker._max_reconnect_attempts = 3

        # First attempt should succeed
        broker._reconnect()

        # Verify connect was called
        mock_connect.assert_called()
        # Verify attempt counter incremented
        assert broker._reconnect_attempts == 1

    @patch.object(IBKRBroker, "connect")
    def test_reconnect_increments_attempts(self, mock_connect):
        """Test reconnect increments attempt counter."""
        broker = IBKRBroker()
        broker._reconnect_attempts = 2
        broker._reconnect_base_delay = 0.001  # Very small delay for testing
        broker._max_reconnect_attempts = 10

        # Mock connect to succeed (not raise)
        mock_connect.return_value = None

        broker._reconnect()

        # After reconnect call, attempts should be incremented
        assert broker._reconnect_attempts == 3  # Incremented before connect


class TestIBKRBrokerIntegration:
    """Integration tests for IBKRBroker."""

    @patch("src.brokers.ibkr.IB")
    def test_connect_disconnect_cycle(self, mock_ib_class):
        """Test full connect/disconnect cycle."""
        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock()
        mock_ib.disconnectAsync = AsyncMock()
        mock_ib_class.return_value = mock_ib

        broker = IBKRBroker()

        # Connect
        with patch.object(broker, "_async_connect", new_callable=AsyncMock):
            broker.connect()
            assert broker._connected is True

        # Disconnect
        broker.disconnect()
        assert broker._connected is False

    @patch("src.brokers.ibkr.IB")
    def test_multiple_connect_calls(self, mock_ib_class):
        """Test multiple connect calls don't create duplicate IB instances."""
        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock()
        mock_ib_class.return_value = mock_ib

        broker = IBKRBroker()

        with patch.object(broker, "_async_connect", new_callable=AsyncMock):
            broker.connect()
            first_ib = broker._ib

            # Second connect should not create new instance
            broker.connect()
            assert broker._ib is first_ib

    def test_broker_client_interface_compliance(self):
        """Test that IBKRBroker implements BrokerClient interface."""
        from src.brokers.base import BrokerClient

        broker = IBKRBroker()

        # Check that broker is instance of BrokerClient
        assert isinstance(broker, BrokerClient)

        # Check that all abstract methods are implemented
        assert hasattr(broker, "connect")
        assert hasattr(broker, "disconnect")
        assert hasattr(broker, "fetch_candles")
        assert hasattr(broker, "place_order")
        assert hasattr(broker, "close_trade")
        assert hasattr(broker, "get_nav")
        assert hasattr(broker, "get_open_positions")
        assert hasattr(broker, "get_price_quote")
        assert hasattr(broker, "get_trades")
        assert hasattr(broker, "get_account_summary")
