"""Unit tests for IBKRBroker order execution (US-010).

Tests verify that place_order() creates bracket orders and close_trade()
cancels orders correctly.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.brokers.ibkr import IBKRBroker
from src.brokers.instrument import Instrument


class TestIBKRBrokerPlaceOrder:
    """Test place_order() bracket order creation."""

    def test_place_order_not_connected(self):
        """Test place_order returns rejected when not connected."""
        broker = IBKRBroker()
        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        result = broker.place_order(
            instrument, "BUY", 100000, 1.0800, 1.1000, 1.0900
        )

        assert result.status == "rejected"
        assert result.trade_id == "0"
        assert "Not connected" in result.error_message

    @patch("src.brokers.ibkr.IB")
    def test_place_order_no_next_valid_id(self, mock_ib_class):
        """Test place_order returns rejected when nextValidId not set."""
        broker = IBKRBroker()
        broker._ib = MagicMock()
        broker._connected = True
        broker._next_valid_id = None

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        result = broker.place_order(
            instrument, "BUY", 100000, 1.0800, 1.1000, 1.0900
        )

        assert result.status == "rejected"
        assert result.trade_id == "0"
        assert "nextValidId" in result.error_message

    @patch("src.brokers.ibkr.IB")
    def test_place_order_invalid_direction(self, mock_ib_class):
        """Test place_order rejects invalid direction."""
        broker = IBKRBroker()
        broker._ib = MagicMock()
        broker._connected = True
        broker._next_valid_id = 1000

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        result = broker.place_order(
            instrument, "INVALID", 100000, 1.0800, 1.1000, 1.0900
        )

        assert result.status == "rejected"
        assert result.trade_id == "0"
        assert "Invalid direction" in result.error_message

    @patch("src.brokers.ibkr.IB")
    def test_place_order_rr_ratio_below_minimum(self, mock_ib_class):
        """Test place_order rejects with R:R ratio < 1.2."""
        broker = IBKRBroker()
        broker._ib = MagicMock()
        broker._connected = True
        broker._next_valid_id = 1000

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        # BUY @ 1.0900, SL=1.0800 (100 pips), TP=1.0950 (50 pips) => R:R = 0.5:1
        result = broker.place_order(
            instrument, "BUY", 100000, 1.0800, 1.0950, 1.0900
        )

        assert result.status == "rejected"
        assert result.trade_id == "0"
        assert "R:R ratio" in result.error_message

    @patch("src.brokers.ibkr.IB")
    def test_place_order_success_buy(self, mock_ib_class):
        """Test successful BUY bracket order placement."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        broker._ib = mock_ib
        broker._connected = True
        broker._next_valid_id = 1000

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        mock_contract = MagicMock()
        with patch.object(broker, "_build_contract", return_value=mock_contract):
            result = broker.place_order(
                instrument, "BUY", 100000, 1.0800, 1.1100, 1.0900
            )

        assert result.status == "PENDING"
        assert result.trade_id == "1000"
        assert result.fill_price == 1.0900

        # Verify three orders placed
        assert mock_ib.placeOrder.call_count == 3

        calls = mock_ib.placeOrder.call_args_list

        # Parent: BUY MKT, transmit=False
        _, parent_order = calls[0][0]
        assert parent_order.orderId == 1000
        assert parent_order.action == "BUY"
        assert parent_order.orderType == "MKT"
        assert parent_order.transmit is False

        # TP: SELL LMT, transmit=False
        _, tp_order = calls[1][0]
        assert tp_order.orderId == 1001
        assert tp_order.action == "SELL"
        assert tp_order.orderType == "LMT"
        assert tp_order.lmtPrice == 1.1100
        assert tp_order.transmit is False

        # SL: SELL STP, transmit=True (CRITICAL)
        _, sl_order = calls[2][0]
        assert sl_order.orderId == 1002
        assert sl_order.action == "SELL"
        assert sl_order.orderType == "STP"
        assert sl_order.auxPrice == 1.0800
        assert sl_order.transmit is True

        # Verify nextValidId incremented by 3
        assert broker._next_valid_id == 1003

    @patch("src.brokers.ibkr.IB")
    def test_place_order_success_sell(self, mock_ib_class):
        """Test successful SELL bracket order placement."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        broker._ib = mock_ib
        broker._connected = True
        broker._next_valid_id = 2000

        instrument = Instrument.fx("EUR_USD", "EUR/USD", pip_value=10.0)

        mock_contract = MagicMock()
        with patch.object(broker, "_build_contract", return_value=mock_contract):
            # SELL @ 1.1000, SL=1.1200 (200 pips), TP=1.0600 (400 pips) => R:R = 2:1
            result = broker.place_order(
                instrument, "SELL", 50000, 1.1200, 1.0600, 1.1000
            )

        assert result.status == "PENDING"
        assert result.trade_id == "2000"
        assert result.fill_price == 1.1000

        assert mock_ib.placeOrder.call_count == 3

        calls = mock_ib.placeOrder.call_args_list

        # Parent: SELL MKT
        _, parent_order = calls[0][0]
        assert parent_order.action == "SELL"
        assert parent_order.orderType == "MKT"
        assert parent_order.transmit is False

        # TP: BUY LMT (reverse direction)
        _, tp_order = calls[1][0]
        assert tp_order.action == "BUY"
        assert tp_order.orderType == "LMT"
        assert tp_order.lmtPrice == 1.0600
        assert tp_order.transmit is False

        # SL: BUY STP (reverse direction)
        _, sl_order = calls[2][0]
        assert sl_order.action == "BUY"
        assert sl_order.orderType == "STP"
        assert sl_order.auxPrice == 1.1200
        assert sl_order.transmit is True

        assert broker._next_valid_id == 2003


class TestIBKRBrokerCloseTrade:
    """Test close_trade() implementation."""

    def test_close_trade_not_connected(self):
        """Test close_trade raises error when not connected."""
        broker = IBKRBroker()

        with pytest.raises(ConnectionError) as exc_info:
            broker.close_trade("1000")
        assert "Not connected" in str(exc_info.value)

    @patch("src.brokers.ibkr.IB")
    def test_close_trade_empty_trade_id(self, mock_ib_class):
        """Test close_trade returns False for empty trade_id."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        broker._ib = mock_ib
        broker._connected = True

        result = broker.close_trade("")

        assert result is False
        mock_ib.cancelOrder.assert_not_called()

    @patch("src.brokers.ibkr.IB")
    def test_close_trade_invalid_format(self, mock_ib_class):
        """Test close_trade returns False for non-integer trade_id."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        broker._ib = mock_ib
        broker._connected = True

        result = broker.close_trade("not_integer")

        assert result is False
        mock_ib.cancelOrder.assert_not_called()

    @patch("src.brokers.ibkr.IB")
    def test_close_trade_success(self, mock_ib_class):
        """Test successful trade closure."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        broker._ib = mock_ib
        broker._connected = True

        result = broker.close_trade("1000")

        assert result is True
        mock_ib.cancelOrder.assert_called_once_with(1000)

    @patch("src.brokers.ibkr.IB")
    def test_close_trade_exception(self, mock_ib_class):
        """Test close_trade handles exceptions."""
        broker = IBKRBroker()
        mock_ib = MagicMock()
        mock_ib.cancelOrder.side_effect = Exception("API error")
        broker._ib = mock_ib
        broker._connected = True

        result = broker.close_trade("1000")

        assert result is False
