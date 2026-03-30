"""Unit tests for broker abstraction layer.

Tests cover:
1. Instrument model validation (FX and FUTURES)
2. Type validation (CandleData, OrderResult, PositionInfo, TradeInfo, AccountSummary)
3. BrokerClient ABC instantiation check
"""

import unittest
from datetime import datetime, timezone

from src.brokers import Instrument, BrokerClient
from src.brokers.types import (
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
    AccountSummary,
)


class TestInstrumentFX(unittest.TestCase):
    """Test FX instrument creation and validation."""

    def test_fx_instrument_creation_valid(self):
        """Test creating a valid FX instrument."""
        inst = Instrument.fx(
            symbol="EUR_USD",
            broker_symbol="EUR/USD",
            pip_value=0.0001,
            price_precision=5,
        )
        self.assertEqual(inst.symbol, "EUR_USD")
        self.assertEqual(inst.broker_symbol, "EUR/USD")
        self.assertEqual(inst.asset_class, "FX")
        self.assertEqual(inst.pip_value, 0.0001)
        self.assertEqual(inst.price_precision, 5)
        self.assertIsNone(inst.tick_size)
        self.assertIsNone(inst.tick_value)
        self.assertIsNone(inst.multiplier)

    def test_fx_instrument_missing_pip_value(self):
        """Test that FX instruments require pip_value."""
        with self.assertRaisesRegex(ValueError, "must define pip_value"):
            Instrument(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                asset_class="FX",
                pip_value=None,  # Invalid for FX
                price_precision=5,
                margin_requirement=1.0,
                exchange="OANDA",
                currency="USD",
            )

    def test_fx_instrument_negative_pip_value(self):
        """Test that FX pip_value must be positive."""
        with self.assertRaisesRegex(ValueError, "must be positive"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                pip_value=-0.0001,
            )

    def test_fx_instrument_factory_defaults(self):
        """Test FX factory method with default parameters."""
        inst = Instrument.fx(
            symbol="GBP_USD",
            broker_symbol="GBP/USD",
            pip_value=0.0001,
        )
        self.assertEqual(inst.exchange, "OANDA")
        self.assertEqual(inst.currency, "USD")
        self.assertEqual(inst.margin_requirement, 1.0)
        self.assertEqual(inst.price_precision, 5)


class TestInstrumentFutures(unittest.TestCase):
    """Test FUTURES instrument creation and validation."""

    def test_futures_instrument_creation_valid(self):
        """Test creating a valid FUTURES instrument."""
        inst = Instrument.futures(
            symbol="ES",
            broker_symbol="ESZ23",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            contract_month="202312",
        )
        self.assertEqual(inst.symbol, "ES")
        self.assertEqual(inst.broker_symbol, "ESZ23")
        self.assertEqual(inst.asset_class, "FUTURES")
        self.assertEqual(inst.tick_size, 0.25)
        self.assertEqual(inst.tick_value, 12.5)
        self.assertEqual(inst.multiplier, 50)
        self.assertEqual(inst.contract_month, "202312")
        self.assertIsNone(inst.pip_value)

    def test_futures_instrument_missing_tick_size(self):
        """Test that FUTURES instruments require tick_size."""
        with self.assertRaisesRegex(ValueError, "must define tick_size"):
            Instrument(
                symbol="ES",
                broker_symbol="ESZ23",
                asset_class="FUTURES",
                tick_size=None,  # Invalid for FUTURES
                tick_value=12.5,
                multiplier=50,
                price_precision=2,
                margin_requirement=10.0,
                exchange="CME",
                currency="USD",
            )

    def test_futures_instrument_missing_tick_value(self):
        """Test that FUTURES instruments require tick_value."""
        with self.assertRaisesRegex(ValueError, "must define tick_value"):
            Instrument(
                symbol="ES",
                broker_symbol="ESZ23",
                asset_class="FUTURES",
                tick_size=0.25,
                tick_value=None,  # Invalid for FUTURES
                multiplier=50,
                price_precision=2,
                margin_requirement=10.0,
                exchange="CME",
                currency="USD",
            )

    def test_futures_instrument_missing_multiplier(self):
        """Test that FUTURES instruments require multiplier."""
        with self.assertRaisesRegex(ValueError, "must define multiplier"):
            Instrument(
                symbol="ES",
                broker_symbol="ESZ23",
                asset_class="FUTURES",
                tick_size=0.25,
                tick_value=12.5,
                multiplier=None,  # Invalid for FUTURES
                price_precision=2,
                margin_requirement=10.0,
                exchange="CME",
                currency="USD",
            )

    def test_futures_instrument_factory_defaults(self):
        """Test FUTURES factory method with default parameters."""
        inst = Instrument.futures(
            symbol="NQ",
            broker_symbol="NQZ23",
            tick_size=0.25,
            tick_value=5.0,
            multiplier=100,
        )
        self.assertEqual(inst.exchange, "CME")
        self.assertEqual(inst.currency, "USD")
        self.assertEqual(inst.margin_requirement, 10.0)
        self.assertEqual(inst.price_precision, 2)
        self.assertIsNone(inst.contract_month)


class TestInstrumentValidation(unittest.TestCase):
    """Test shared instrument validation rules."""

    def test_instrument_empty_symbol(self):
        """Test that symbol cannot be empty."""
        with self.assertRaisesRegex(ValueError, "symbol cannot be empty"):
            Instrument.fx(
                symbol="",
                broker_symbol="EUR/USD",
                pip_value=0.0001,
            )

    def test_instrument_empty_broker_symbol(self):
        """Test that broker_symbol cannot be empty."""
        with self.assertRaisesRegex(ValueError, "broker_symbol cannot be empty"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="",
                pip_value=0.0001,
            )

    def test_instrument_negative_price_precision(self):
        """Test that price_precision cannot be negative."""
        with self.assertRaisesRegex(ValueError, "price_precision cannot be negative"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                pip_value=0.0001,
                price_precision=-1,
            )

    def test_instrument_negative_margin_requirement(self):
        """Test that margin_requirement cannot be negative."""
        with self.assertRaisesRegex(ValueError, "margin_requirement cannot be negative"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                pip_value=0.0001,
                margin_requirement=-1.0,
            )

    def test_instrument_empty_exchange(self):
        """Test that exchange cannot be empty."""
        with self.assertRaisesRegex(ValueError, "exchange cannot be empty"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                pip_value=0.0001,
                exchange="",
            )

    def test_instrument_empty_currency(self):
        """Test that currency cannot be empty."""
        with self.assertRaisesRegex(ValueError, "currency cannot be empty"):
            Instrument.fx(
                symbol="EUR_USD",
                broker_symbol="EUR/USD",
                pip_value=0.0001,
                currency="",
            )

    def test_instrument_invalid_asset_class(self):
        """Test that asset_class must be FX or FUTURES."""
        with self.assertRaisesRegex(ValueError, "Invalid asset_class"):
            Instrument(
                symbol="TEST",
                broker_symbol="TEST",
                asset_class="STOCKS",  # Invalid
                pip_value=0.0001,
                price_precision=5,
                margin_requirement=1.0,
                exchange="NYSE",
                currency="USD",
            )


class TestCandleData(unittest.TestCase):
    """Test CandleData type validation."""

    def test_candle_data_valid(self):
        """Test creating valid candle data."""
        now = datetime.now(timezone.utc)
        candle = CandleData(
            time=now,
            open=1.1050,
            high=1.1100,
            low=1.1000,
            close=1.1075,
            volume=1000.0,
        )
        self.assertEqual(candle.time, now)
        self.assertEqual(candle.open, 1.1050)
        self.assertEqual(candle.high, 1.1100)
        self.assertEqual(candle.low, 1.1000)
        self.assertEqual(candle.close, 1.1075)
        self.assertEqual(candle.volume, 1000.0)

    def test_candle_data_high_less_than_low(self):
        """Test that candle high must be >= low."""
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "high.*<.*low"):
            CandleData(
                time=now,
                open=1.1050,
                high=1.0900,
                low=1.1000,
                close=1.1075,
                volume=1000.0,
            )

    def test_candle_data_negative_price(self):
        """Test that candle prices cannot be negative."""
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "prices cannot be negative"):
            CandleData(
                time=now,
                open=-1.1050,
                high=1.1100,
                low=1.1000,
                close=1.1075,
                volume=1000.0,
            )

    def test_candle_data_negative_volume(self):
        """Test that candle volume cannot be negative."""
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "volume cannot be negative"):
            CandleData(
                time=now,
                open=1.1050,
                high=1.1100,
                low=1.1000,
                close=1.1075,
                volume=-100.0,
            )


class TestOrderResult(unittest.TestCase):
    """Test OrderResult type validation."""

    def test_order_result_valid(self):
        """Test creating valid order result."""
        result = OrderResult(
            trade_id="123456",
            fill_price=1.1050,
            status="FILLED",
        )
        self.assertEqual(result.trade_id, "123456")
        self.assertEqual(result.fill_price, 1.1050)
        self.assertEqual(result.status, "FILLED")
        self.assertIsNone(result.error_message)

    def test_order_result_with_error(self):
        """Test order result with error message."""
        result = OrderResult(
            trade_id="123456",
            fill_price=1.1050,
            status="REJECTED",
            error_message="Insufficient margin",
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(result.error_message, "Insufficient margin")

    def test_order_result_negative_fill_price(self):
        """Test that fill_price cannot be negative."""
        with self.assertRaisesRegex(ValueError, "Fill price cannot be negative"):
            OrderResult(
                trade_id="123456",
                fill_price=-1.1050,
                status="FILLED",
            )

    def test_order_result_empty_trade_id(self):
        """Test that trade_id cannot be empty."""
        with self.assertRaisesRegex(ValueError, "trade_id cannot be empty"):
            OrderResult(
                trade_id="",
                fill_price=1.1050,
                status="FILLED",
            )

    def test_order_result_empty_status(self):
        """Test that status cannot be empty."""
        with self.assertRaisesRegex(ValueError, "status cannot be empty"):
            OrderResult(
                trade_id="123456",
                fill_price=1.1050,
                status="",
            )


class TestPositionInfo(unittest.TestCase):
    """Test PositionInfo type validation."""

    def test_position_info_valid_buy(self):
        """Test creating valid BUY position."""
        pos = PositionInfo(
            instrument="EUR_USD",
            direction="BUY",
            quantity=10000.0,
            avg_price=1.1050,
            unrealized_pnl=150.0,
        )
        self.assertEqual(pos.instrument, "EUR_USD")
        self.assertEqual(pos.direction, "BUY")
        self.assertEqual(pos.quantity, 10000.0)
        self.assertEqual(pos.avg_price, 1.1050)
        self.assertEqual(pos.unrealized_pnl, 150.0)

    def test_position_info_valid_sell(self):
        """Test creating valid SELL position."""
        pos = PositionInfo(
            instrument="GBP_USD",
            direction="SELL",
            quantity=5000.0,
            avg_price=1.2700,
            unrealized_pnl=-50.0,
        )
        self.assertEqual(pos.direction, "SELL")
        self.assertEqual(pos.unrealized_pnl, -50.0)

    def test_position_info_invalid_direction(self):
        """Test that direction must be BUY or SELL."""
        with self.assertRaisesRegex(ValueError, "Invalid direction"):
            PositionInfo(
                instrument="EUR_USD",
                direction="LONG",
                quantity=10000.0,
                avg_price=1.1050,
                unrealized_pnl=0.0,
            )

    def test_position_info_empty_instrument(self):
        """Test that instrument cannot be empty."""
        with self.assertRaisesRegex(ValueError, "instrument cannot be empty"):
            PositionInfo(
                instrument="",
                direction="BUY",
                quantity=10000.0,
                avg_price=1.1050,
                unrealized_pnl=0.0,
            )

    def test_position_info_zero_quantity(self):
        """Test that quantity must be positive."""
        with self.assertRaisesRegex(ValueError, "quantity must be positive"):
            PositionInfo(
                instrument="EUR_USD",
                direction="BUY",
                quantity=0.0,
                avg_price=1.1050,
                unrealized_pnl=0.0,
            )

    def test_position_info_negative_avg_price(self):
        """Test that avg_price cannot be negative."""
        with self.assertRaisesRegex(ValueError, "avg_price cannot be negative"):
            PositionInfo(
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                avg_price=-1.1050,
                unrealized_pnl=0.0,
            )


class TestTradeInfo(unittest.TestCase):
    """Test TradeInfo type validation."""

    def test_trade_info_valid_complete(self):
        """Test creating valid trade with SL and TP."""
        trade = TradeInfo(
            trade_id="T123",
            instrument="EUR_USD",
            direction="BUY",
            quantity=10000.0,
            entry_price=1.1050,
            sl_price=1.1000,
            tp_price=1.1150,
        )
        self.assertEqual(trade.trade_id, "T123")
        self.assertEqual(trade.sl_price, 1.1000)
        self.assertEqual(trade.tp_price, 1.1150)

    def test_trade_info_valid_no_sl_tp(self):
        """Test creating trade without SL/TP."""
        trade = TradeInfo(
            trade_id="T123",
            instrument="EUR_USD",
            direction="SELL",
            quantity=5000.0,
            entry_price=1.1050,
        )
        self.assertIsNone(trade.sl_price)
        self.assertIsNone(trade.tp_price)

    def test_trade_info_empty_trade_id(self):
        """Test that trade_id cannot be empty."""
        with self.assertRaisesRegex(ValueError, "trade_id cannot be empty"):
            TradeInfo(
                trade_id="",
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                entry_price=1.1050,
            )

    def test_trade_info_negative_sl_price(self):
        """Test that sl_price cannot be negative."""
        with self.assertRaisesRegex(ValueError, "sl_price cannot be negative"):
            TradeInfo(
                trade_id="T123",
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                entry_price=1.1050,
                sl_price=-1.1000,
            )

    def test_trade_info_negative_tp_price(self):
        """Test that tp_price cannot be negative."""
        with self.assertRaisesRegex(ValueError, "tp_price cannot be negative"):
            TradeInfo(
                trade_id="T123",
                instrument="EUR_USD",
                direction="BUY",
                quantity=10000.0,
                entry_price=1.1050,
                tp_price=-1.1150,
            )


class TestAccountSummary(unittest.TestCase):
    """Test AccountSummary type validation."""

    def test_account_summary_valid(self):
        """Test creating valid account summary."""
        summary = AccountSummary(
            nav=100000.0,
            balance=100000.0,
            unrealized_pnl=500.0,
            margin_used=5000.0,
        )
        self.assertEqual(summary.nav, 100000.0)
        self.assertEqual(summary.balance, 100000.0)
        self.assertEqual(summary.unrealized_pnl, 500.0)
        self.assertEqual(summary.margin_used, 5000.0)

    def test_account_summary_negative_nav(self):
        """Test that nav cannot be negative."""
        with self.assertRaisesRegex(ValueError, "nav cannot be negative"):
            AccountSummary(
                nav=-100000.0,
                balance=100000.0,
                unrealized_pnl=0.0,
                margin_used=0.0,
            )

    def test_account_summary_negative_margin_used(self):
        """Test that margin_used cannot be negative."""
        with self.assertRaisesRegex(ValueError, "margin_used cannot be negative"):
            AccountSummary(
                nav=100000.0,
                balance=100000.0,
                unrealized_pnl=0.0,
                margin_used=-5000.0,
            )


class TestBrokerClientABC(unittest.TestCase):
    """Test BrokerClient abstract base class."""

    def test_broker_client_cannot_instantiate(self):
        """Test that BrokerClient cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BrokerClient()

    def test_broker_client_subclass_requires_all_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteBrokerClient(BrokerClient):
            def connect(self) -> None:
                pass

        with self.assertRaises(TypeError):
            IncompleteBrokerClient()


if __name__ == "__main__":
    unittest.main()
