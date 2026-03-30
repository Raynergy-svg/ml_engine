"""Unit tests for IBKRBroker account and position methods (US-011).

Tests for:
- get_nav() — retrieve account NAV from ib.accountSummary()
- get_open_positions() — retrieve current positions from ib.positions()
- get_trades() — retrieve open trades from ib.openTrades()
- get_account_summary() — retrieve account summary data
- close_trade() — close a specific trade (verify it was implemented correctly)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.brokers.ibkr import IBKRBroker
from src.brokers.instrument import Instrument
from src.brokers.types import PositionInfo, TradeInfo, AccountSummary


class TestGetNAV:
    """Tests for get_nav() method."""

    def test_get_nav_when_not_connected(self):
        """Should return 0.0 when not connected to IBKR."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        nav = broker.get_nav()

        assert nav == 0.0

    def test_get_nav_success(self):
        """Should retrieve NAV from accountSummary."""
        broker = IBKRBroker()
        broker._connected = True

        # Mock _ib and accountSummary
        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Create mock summary objects
        mock_summary = MagicMock()
        mock_summary.tag = "NetLiquidation"
        mock_summary.value = "100500.50"

        mock_ib.accountSummary.return_value = [mock_summary]

        nav = broker.get_nav()

        assert nav == 100500.50
        mock_ib.accountSummary.assert_called_once()

    def test_get_nav_invalid_value(self):
        """Should return 0.0 if NAV value is non-numeric."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        mock_summary = MagicMock()
        mock_summary.tag = "NetLiquidation"
        mock_summary.value = "NOT_A_NUMBER"

        mock_ib.accountSummary.return_value = [mock_summary]

        nav = broker.get_nav()

        assert nav == 0.0

    def test_get_nav_not_found(self):
        """Should return 0.0 if NetLiquidation tag not found."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        mock_summary = MagicMock()
        mock_summary.tag = "SomeOtherTag"
        mock_summary.value = "12345"

        mock_ib.accountSummary.return_value = [mock_summary]

        nav = broker.get_nav()

        assert nav == 0.0

    def test_get_nav_exception_handling(self):
        """Should return 0.0 on exception."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.accountSummary.side_effect = Exception("Test error")

        nav = broker.get_nav()

        assert nav == 0.0


class TestGetOpenPositions:
    """Tests for get_open_positions() method."""

    def test_get_open_positions_when_not_connected(self):
        """Should return empty list when not connected."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        positions = broker.get_open_positions()

        assert positions == []

    def test_get_open_positions_empty(self):
        """Should return empty list when no positions open."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.positions.return_value = []

        positions = broker.get_open_positions()

        assert positions == []

    def test_get_open_positions_single_long(self):
        """Should return BUY position when position is positive."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Mock position object
        mock_position = MagicMock()
        mock_position.contract.symbol = "EUR_USD"
        mock_position.position = 10.0
        mock_position.avgCost = 1.0950

        mock_ib.positions.return_value = [mock_position]

        positions = broker.get_open_positions()

        assert len(positions) == 1
        assert positions[0].instrument == "EUR_USD"
        assert positions[0].direction == "BUY"
        assert positions[0].quantity == 10.0
        assert positions[0].avg_price == 1.0950

    def test_get_open_positions_single_short(self):
        """Should return SELL position when position is negative."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Mock position object
        mock_position = MagicMock()
        mock_position.contract.symbol = "GBP_USD"
        mock_position.position = -5.0
        mock_position.avgCost = 1.2750

        mock_ib.positions.return_value = [mock_position]

        positions = broker.get_open_positions()

        assert len(positions) == 1
        assert positions[0].instrument == "GBP_USD"
        assert positions[0].direction == "SELL"
        assert positions[0].quantity == 5.0
        assert positions[0].avg_price == 1.2750

    def test_get_open_positions_multiple(self):
        """Should return multiple positions."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        mock_pos1 = MagicMock()
        mock_pos1.contract.symbol = "EUR_USD"
        mock_pos1.position = 10.0
        mock_pos1.avgCost = 1.0950

        mock_pos2 = MagicMock()
        mock_pos2.contract.symbol = "GBP_USD"
        mock_pos2.position = -5.0
        mock_pos2.avgCost = 1.2750

        mock_ib.positions.return_value = [mock_pos1, mock_pos2]

        positions = broker.get_open_positions()

        assert len(positions) == 2
        assert positions[0].direction == "BUY"
        assert positions[1].direction == "SELL"

    def test_get_open_positions_exception_handling(self):
        """Should return empty list on exception."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.positions.side_effect = Exception("Test error")

        positions = broker.get_open_positions()

        assert positions == []


class TestGetTrades:
    """Tests for get_trades() method."""

    def test_get_trades_when_not_connected(self):
        """Should return empty list when not connected."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        trades = broker.get_trades()

        assert trades == []

    def test_get_trades_empty(self):
        """Should return empty list when no trades open."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.openTrades.return_value = []

        trades = broker.get_trades()

        assert trades == []

    def test_get_trades_single_buy(self):
        """Should return BUY trade info."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Mock trade object
        mock_trade = MagicMock()
        mock_trade.order.orderId = 123
        mock_trade.order.action = "BUY"
        mock_trade.order.totalQuantity = 10.0
        mock_trade.order.lmtPrice = 1.0950
        mock_trade.contract.symbol = "EUR_USD"

        mock_ib.openTrades.return_value = [mock_trade]

        trades = broker.get_trades()

        assert len(trades) == 1
        assert trades[0].trade_id == "123"
        assert trades[0].instrument == "EUR_USD"
        assert trades[0].direction == "BUY"
        assert trades[0].quantity == 10.0
        assert trades[0].entry_price == 1.0950

    def test_get_trades_single_sell(self):
        """Should return SELL trade info."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        mock_trade = MagicMock()
        mock_trade.order.orderId = 124
        mock_trade.order.action = "SELL"
        mock_trade.order.totalQuantity = 5.0
        mock_trade.order.lmtPrice = 1.2750
        mock_trade.contract.symbol = "GBP_USD"

        mock_ib.openTrades.return_value = [mock_trade]

        trades = broker.get_trades()

        assert len(trades) == 1
        assert trades[0].trade_id == "124"
        assert trades[0].direction == "SELL"
        assert trades[0].quantity == 5.0

    def test_get_trades_multiple(self):
        """Should return multiple trades."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        mock_trade1 = MagicMock()
        mock_trade1.order.orderId = 123
        mock_trade1.order.action = "BUY"
        mock_trade1.order.totalQuantity = 10.0
        mock_trade1.order.lmtPrice = 1.0950
        mock_trade1.contract.symbol = "EUR_USD"

        mock_trade2 = MagicMock()
        mock_trade2.order.orderId = 124
        mock_trade2.order.action = "SELL"
        mock_trade2.order.totalQuantity = 5.0
        mock_trade2.order.lmtPrice = 1.2750
        mock_trade2.contract.symbol = "GBP_USD"

        mock_ib.openTrades.return_value = [mock_trade1, mock_trade2]

        trades = broker.get_trades()

        assert len(trades) == 2
        assert trades[0].trade_id == "123"
        assert trades[1].trade_id == "124"

    def test_get_trades_exception_handling(self):
        """Should return empty list on exception."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.openTrades.side_effect = Exception("Test error")

        trades = broker.get_trades()

        assert trades == []


class TestGetAccountSummary:
    """Tests for get_account_summary() method."""

    def test_get_account_summary_when_not_connected(self):
        """Should return zero summary when not connected."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        summary = broker.get_account_summary()

        assert isinstance(summary, AccountSummary)
        assert summary.nav == 0.0
        assert summary.balance == 0.0
        assert summary.unrealized_pnl == 0.0
        assert summary.margin_used == 0.0

    def test_get_account_summary_success(self):
        """Should retrieve all account summary fields."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Create mock summary objects
        summaries = []
        tags_values = [
            ("NetLiquidation", "100500.50"),
            ("TotalCashValue", "50000.25"),
            ("UnrealizedPnL", "500.25"),
            ("InitMarginReq", "10000.00"),
        ]

        for tag, value in tags_values:
            mock_summary = MagicMock()
            mock_summary.tag = tag
            mock_summary.value = value
            summaries.append(mock_summary)

        mock_ib.accountSummary.return_value = summaries

        summary = broker.get_account_summary()

        assert summary.nav == 100500.50
        assert summary.balance == 50000.25
        assert summary.unrealized_pnl == 500.25
        assert summary.margin_used == 10000.00

    def test_get_account_summary_partial_fields(self):
        """Should handle missing fields gracefully."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        # Only provide NAV
        mock_summary = MagicMock()
        mock_summary.tag = "NetLiquidation"
        mock_summary.value = "100500.50"

        mock_ib.accountSummary.return_value = [mock_summary]

        summary = broker.get_account_summary()

        assert summary.nav == 100500.50
        assert summary.balance == 0.0
        assert summary.unrealized_pnl == 0.0
        assert summary.margin_used == 0.0

    def test_get_account_summary_invalid_values(self):
        """Should skip invalid values and continue."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        summaries = []

        # Invalid value
        mock_summary1 = MagicMock()
        mock_summary1.tag = "NetLiquidation"
        mock_summary1.value = "NOT_A_NUMBER"
        summaries.append(mock_summary1)

        # Valid value
        mock_summary2 = MagicMock()
        mock_summary2.tag = "TotalCashValue"
        mock_summary2.value = "50000.25"
        summaries.append(mock_summary2)

        mock_ib.accountSummary.return_value = summaries

        summary = broker.get_account_summary()

        assert summary.nav == 0.0
        assert summary.balance == 50000.25

    def test_get_account_summary_exception_handling(self):
        """Should return zero summary on exception."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.accountSummary.side_effect = Exception("Test error")

        summary = broker.get_account_summary()

        assert summary.nav == 0.0
        assert summary.balance == 0.0


class TestCloseTrade:
    """Tests for close_trade() method (US-010 implementation)."""

    def test_close_trade_not_connected(self):
        """Should raise ConnectionError when not connected."""
        broker = IBKRBroker()
        broker._connected = False
        broker._ib = None

        with pytest.raises(ConnectionError):
            broker.close_trade("123")

    def test_close_trade_empty_trade_id(self):
        """Should return False for empty trade_id."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        result = broker.close_trade("")

        assert result is False

    def test_close_trade_invalid_order_id(self):
        """Should return False for non-integer trade_id."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        result = broker.close_trade("not_an_id")

        assert result is False

    def test_close_trade_success(self):
        """Should successfully cancel an order."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib

        result = broker.close_trade("123")

        assert result is True
        mock_ib.cancelOrder.assert_called_once_with(123)

    def test_close_trade_exception(self):
        """Should return False on exception."""
        broker = IBKRBroker()
        broker._connected = True

        mock_ib = MagicMock()
        broker._ib = mock_ib
        mock_ib.cancelOrder.side_effect = Exception("Test error")

        result = broker.close_trade("123")

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
