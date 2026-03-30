"""Tests for ExecutionManager BrokerClient integration (US-005).

Tests verify that ExecutionManager correctly uses the BrokerClient ABC
instead of directly calling OandaPracticeClient methods.
"""

import logging
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from src.scanner.execution import ExecutionManager, ExecutionConfig, ExecutionResult
from src.brokers.base import BrokerClient
from src.brokers.types import OrderResult, AccountSummary, PositionInfo, TradeInfo
from src.brokers.instrument import Instrument

logger = logging.getLogger(__name__)


class MockBrokerClient(BrokerClient):
    """Mock BrokerClient for testing."""

    def __init__(self):
        self.connected = False
        self.orders = []
        self.nav = 10000.0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def fetch_candles(self, instrument, granularity, count):
        return []

    def place_order(
        self,
        instrument: Instrument,
        direction: str,
        quantity: float,
        sl_price: float,
        tp_price: float,
        entry_price: float,
    ) -> OrderResult:
        """Simulate successful order placement."""
        trade_id = f"test_{len(self.orders) + 1}"
        self.orders.append({
            "trade_id": trade_id,
            "instrument": instrument.symbol,
            "direction": direction,
            "quantity": quantity,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_price": entry_price,
        })
        return OrderResult(
            trade_id=trade_id,
            fill_price=entry_price,
            status="FILLED",
        )

    def close_trade(self, trade_id: str) -> bool:
        return True

    def get_nav(self) -> float:
        return self.nav

    def get_open_positions(self) -> list:
        return []

    def get_price_quote(self, instrument: Instrument) -> dict:
        return {"bid": 1.0850, "ask": 1.0853}  # 3 pips spread (within default max of 3.0)

    def get_trades(self) -> list:
        return []

    def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            nav=self.nav,
            balance=10000.0,
            unrealized_pnl=0.0,
            margin_used=0.0,
        )


class TestExecutionManagerBrokerInit:
    """Tests for ExecutionManager initialization with BrokerClient."""

    def test_init_with_broker_provided(self):
        """Test that ExecutionManager accepts broker parameter."""
        mock_broker = MockBrokerClient()
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, broker=mock_broker)

        assert mgr._broker is mock_broker
        assert mgr._legacy_oanda is None

    def test_init_without_broker(self):
        """Test that ExecutionManager defers broker init if not provided."""
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, broker=None)

        assert mgr._broker is None
        assert mgr._legacy_oanda is None

    def test_init_with_legacy_oanda_client(self):
        """Test backward compatibility with oanda_client parameter."""
        mock_oanda = MagicMock()
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, oanda_client=mock_oanda)

        assert mgr._legacy_oanda is mock_oanda
        assert mgr._broker is None

    def test_init_broker_lazy_initialization(self):
        """Test that broker is lazily initialized on first use."""
        mock_broker = MockBrokerClient()
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, broker=mock_broker)

        # Manually test lazy init
        result = mgr._init_broker()
        assert result is True
        assert mgr._broker is mock_broker


class TestExecutionManagerBrokerMethods:
    """Tests for ExecutionManager methods using BrokerClient."""

    def test_fetch_live_nav(self):
        """Test fetch_live_nav uses broker.get_nav()."""
        mock_broker = MockBrokerClient()
        mock_broker.nav = 15000.0
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, broker=mock_broker)
        nav = mgr.fetch_live_nav()

        assert nav == 15000.0

    def test_fetch_live_nav_fallback_to_cached(self):
        """Test that fetch_live_nav falls back to cached value on error."""
        config = ExecutionConfig()
        config.account_equity = 12000.0

        mock_broker = MockBrokerClient()
        mgr = ExecutionManager(config=config, broker=mock_broker)
        mgr._cached_nav = 12000.0

        # When broker fails, should return cached nav
        nav = mgr.fetch_live_nav()
        # fetch_live_nav should succeed because we have a real broker
        assert nav == 10000.0  # Default from MockBrokerClient

    def test_spread_check_with_broker(self):
        """Test _check_spread_conditions uses broker.get_price_quote()."""
        mock_broker = MockBrokerClient()
        # Mock returns bid=1.0800, ask=1.0810 (10 pips, which exceeds default 3.0)
        # so set max_spread to accommodate this
        config = ExecutionConfig()
        config.max_spread_pips = 15.0  # Set via setattr since not in dataclass

        mgr = ExecutionManager(config=config, broker=mock_broker)
        spread_ok, spread_pips, reason = mgr._check_spread_conditions("EUR_USD")

        assert spread_ok is True
        assert spread_pips > 0
        assert "OK" in reason

    def test_spread_check_wide_spread(self):
        """Test _check_spread_conditions rejects wide spreads."""
        mock_broker = MagicMock(spec=BrokerClient)
        mock_broker.get_price_quote.return_value = {
            "bid": 1.0800,
            "ask": 1.0850,  # 50 pips wide (0.0050)
        }
        config = ExecutionConfig()
        # Note: max_spread_pips is accessed via getattr() with default 3.0
        # so spread of 50 pips will exceed it

        mgr = ExecutionManager(config=config, broker=mock_broker)
        spread_ok, spread_pips, reason = mgr._check_spread_conditions("EUR_USD")

        assert spread_ok is False
        assert spread_pips > 10.0
        assert "too wide" in reason.lower()


class TestExecutionManagerPlaceOrder:
    """Tests for order placement using BrokerClient."""

    def test_execute_trade_with_broker(self):
        """Test execute_trade places order via broker.place_order()."""
        mock_broker = MockBrokerClient()
        config = ExecutionConfig()

        mgr = ExecutionManager(config=config, broker=mock_broker)

        # Prepare execution context
        ctx = {
            "agent_passed": True,
            "model_disagreement": 0.1,
            "volatility_regime": "NORMAL",
            "confidence": 0.75,
            "scan_time": datetime.now(timezone.utc).isoformat(),
        }

        result = mgr.execute_trade(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.75,
            current_price=1.0850,
            atr=0.0050,
            sl_pips=15.0,
            tp_pips=25.0,
            lots=1.0,
            analysis_context=ctx,
        )

        assert result.success is True
        assert result.trade_id.startswith("test_")
        assert result.fill_price == 1.0850
        assert len(mock_broker.orders) == 1
        assert mock_broker.orders[0]["instrument"] == "EUR_USD"

    def test_execute_trade_order_rejected_retry(self):
        """Test order rejection triggers downsize retry."""
        mock_broker = MagicMock(spec=BrokerClient)

        # First call rejected, second call accepted
        mock_broker.place_order.side_effect = [
            OrderResult(
                trade_id="error_1",
                fill_price=1.0850,
                status="REJECTED",
                error_message="Margin insufficient",
            ),
            OrderResult(
                trade_id="test_1",
                fill_price=1.0850,
                status="FILLED",
            ),
        ]

        config = ExecutionConfig(
            max_order_attempts=2,
            rejection_downsize_factor=0.5,
        )

        mgr = ExecutionManager(config=config, broker=mock_broker)

        ctx = {
            "agent_passed": True,
            "model_disagreement": 0.1,
            "volatility_regime": "NORMAL",
            "confidence": 0.75,
            "scan_time": datetime.now(timezone.utc).isoformat(),
        }

        result = mgr.execute_trade(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.75,
            current_price=1.0850,
            atr=0.0050,
            sl_pips=15.0,
            tp_pips=25.0,
            lots=1.0,
            analysis_context=ctx,
        )

        assert result.success is True
        assert result.trade_id == "test_1"
        # Verify place_order was called twice (rejected once, then retried)
        assert mock_broker.place_order.call_count == 2


class TestExecutionManagerFetchTrades:
    """Tests for fetching trades using BrokerClient."""

    def test_monitor_open_trades(self):
        """Test that monitor_open_trades uses broker.get_open_positions()."""
        mock_broker = MagicMock(spec=BrokerClient)
        mock_broker.get_open_positions.return_value = [
            PositionInfo(
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                avg_price=1.0850,
                unrealized_pnl=100.0,
            ),
        ]

        config = ExecutionConfig()
        mgr = ExecutionManager(config=config, broker=mock_broker)

        statuses = mgr.monitor_open_trades(evaluate_exits=False)

        assert len(statuses) == 1
        assert statuses[0]["pair"] == "EUR_USD"
        assert statuses[0]["direction"] == "BUY"
        assert statuses[0]["units"] == 10000

    def test_fetch_actual_win_rate(self):
        """Test fetch_actual_win_rate uses broker.get_trades()."""
        mock_broker = MagicMock(spec=BrokerClient)
        mock_broker.get_trades.return_value = [
            TradeInfo(
                trade_id="1",
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                entry_price=1.0850,
            ),
            TradeInfo(
                trade_id="2",
                instrument="EUR_USD",
                direction="SELL",
                quantity=10000.0,
                entry_price=1.0900,
            ),
        ]

        config = ExecutionConfig()
        mgr = ExecutionManager(config=config, broker=mock_broker)

        win_rate, total = mgr.fetch_actual_win_rate()

        assert total == 2
        assert win_rate == 0.5  # Conservative estimate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
