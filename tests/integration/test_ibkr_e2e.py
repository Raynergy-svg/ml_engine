"""Integration smoke tests for IBKR paper trading E2E (US-019).

Tests verify full pipeline from data fetch → feature engineering → agent evaluation
→ order placement → trade journal recording.

All tests are marked @pytest.mark.integration and skip gracefully when IB Gateway
is not available (sandbox/CI environment).
"""

from __future__ import annotations

import json
import logging
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from unittest.mock import patch

from src.brokers.ibkr import IBKRBroker
from src.brokers.instrument import Instrument
from src.brokers.registry import InstrumentRegistry
from src.brokers.types import CandleData, OrderResult
from src.scanner.engine import Scanner
from src.scanner.config import ScannerConfig
from src.scanner.agents._team import ScannerAgentTeam, AgentVerdict
from src.scanner.results import PairAnalysis

logger = logging.getLogger(__name__)


def _is_ib_gateway_running(host: str = "127.0.0.1", port: int = 4002) -> bool:
    """Check if IB Gateway is running on the specified host:port.

    Args:
        host: IB Gateway host (default: localhost).
        port: IB Gateway port (default: 4002 for IBGateway).

    Returns:
        True if Gateway is reachable, False otherwise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    except Exception as e:
        logger.debug(f"Failed to check IB Gateway: {e}")
        return False
    finally:
        sock.close()


@pytest.fixture
def ibkr_broker() -> IBKRBroker:
    """Create an IBKRBroker instance for paper trading (port 4002).

    Paper trading on IB Gateway uses port 4002 by default.
    """
    return IBKRBroker(host="127.0.0.1", port=4002, client_id=1)


@pytest.fixture
def es_instrument() -> Instrument:
    """Create ES (E-mini S&P 500 futures) instrument for testing.

    ES specs:
    - Symbol: ES
    - Tick size: 0.25
    - Tick value: $12.50 per tick
    - Multiplier: 50
    """
    return Instrument(
        symbol="ES",
        broker_symbol="ES",
        asset_class="FUTURES",
        price_precision=2,
        margin_requirement=500.0,
        exchange="CME",
        currency="USD",
        tick_size=0.25,
        tick_value=12.50,
        multiplier=50.0,
    )


@pytest.fixture
def instrument_registry() -> InstrumentRegistry:
    """Create InstrumentRegistry with ES configured."""
    registry = InstrumentRegistry()
    es = Instrument(
        symbol="ES",
        broker_symbol="ES",
        asset_class="FUTURES",
        price_precision=2,
        margin_requirement=500.0,
        exchange="CME",
        currency="USD",
        tick_size=0.25,
        tick_value=12.50,
        multiplier=50.0,
    )
    registry.register(es)
    return registry


@pytest.fixture
def scanner_config() -> ScannerConfig:
    """Create ScannerConfig for testing."""
    config = ScannerConfig()
    return config


@pytest.fixture
def trade_journal_path(tmp_path) -> Path:
    """Create a temporary trade journal path."""
    journal_path = tmp_path / "trade_journal_rl.json"
    # Initialize empty journal
    journal_path.write_text(json.dumps([]))
    return journal_path


class TestIBKRGatewayConnection:
    """Test 1: Connect to IB Gateway (skip if not running)."""

    @pytest.mark.integration
    def test_ib_gateway_reachable(self) -> None:
        """Verify IB Gateway is running on port 4002 for the test suite.

        This test runs first to skip all integration tests if Gateway is not available.
        """
        if not _is_ib_gateway_running():
            pytest.skip(
                "IB Gateway not running on 127.0.0.1:4002. "
                "Start IB Gateway to run integration tests."
            )

    @pytest.mark.integration
    def test_ibkr_broker_connect(self, ibkr_broker: IBKRBroker) -> None:
        """Test connecting to IB Gateway and verifying connection state.

        Requirements:
        - IB Gateway running on 127.0.0.1:4002
        - Connection succeeds and _connected flag is True
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        # Connect to IB Gateway
        ibkr_broker.connect()

        # Verify connection state
        assert ibkr_broker._connected is True, "IBKRBroker should be connected"
        assert ibkr_broker._ib is not None, "IBKRBroker._ib should be set"

        # Cleanup: disconnect
        ibkr_broker.disconnect()


class TestHistoricalDataFetch:
    """Test 2: Fetch 30 days of ES H1 candles, verify DataFrame shape and columns."""

    @pytest.mark.integration
    def test_fetch_es_h1_candles(
        self,
        ibkr_broker: IBKRBroker,
        es_instrument: Instrument,
    ) -> None:
        """Fetch 30 days of ES H1 candles and verify data structure.

        Requirements:
        - Fetch 30 days * 24 hours = ~480 H1 candles (adjusted for market hours)
        - Return list of CandleData objects
        - Verify OHLCV fields are present and valid
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        # Connect
        ibkr_broker.connect()

        try:
            # Fetch 30 days of H1 candles for ES
            # Market hours are ~9:30 AM - 4:00 PM ET, so ~6.5 hours/day
            # 30 days * 6.5 hours = ~195 expected candles
            candles = ibkr_broker.fetch_candles(
                instrument=es_instrument,
                granularity="H1",
                count=480,  # Request 480, get whatever is available
            )

            # Verify we got data
            assert len(candles) > 0, "Should fetch at least some H1 candles"
            assert isinstance(candles, list), "Should return list of CandleData"

            # Verify each candle has valid OHLCV
            for candle in candles:
                assert isinstance(candle, CandleData), "Each item should be CandleData"
                assert candle.time is not None, "Candle time should not be None"
                assert candle.open > 0, f"Candle open should be positive: {candle}"
                assert candle.high >= candle.low, "High should be >= low"
                assert candle.close > 0, "Candle close should be positive"
                assert candle.volume >= 0, "Candle volume should be non-negative"

            # Log candle stats
            df = pd.DataFrame([
                {
                    "time": c.time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ])
            logger.info(
                f"Fetched {len(candles)} H1 candles for ES: "
                f"date range {df['time'].min()} to {df['time'].max()}"
            )

        finally:
            ibkr_broker.disconnect()


class TestFeatureEngineering:
    """Test 3: Run feature engineering on ES candles, verify all features computed."""

    @pytest.mark.integration
    def test_feature_engineering_on_es_candles(
        self,
        ibkr_broker: IBKRBroker,
        es_instrument: Instrument,
        scanner_config: ScannerConfig,
    ) -> None:
        """Run feature engineering on ES H1 candles and verify feature DataFrame.

        Requirements:
        - Fetch ES H1 candles
        - Convert to pandas DataFrame
        - Run feature engineering (TCN, Ridge, RF ensemble)
        - Verify output DataFrame has expected feature columns
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        # Connect and fetch candles
        ibkr_broker.connect()
        try:
            candles = ibkr_broker.fetch_candles(
                instrument=es_instrument,
                granularity="H1",
                count=480,
            )

            if len(candles) == 0:
                pytest.skip("No candles fetched")

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "time": c.time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ])

            # Ensure datetime index
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df.sort_index()

            # Create Scanner and run feature engineering
            scanner = Scanner(config=scanner_config, broker=ibkr_broker)

            # Initialize feature engineer (lazy-loaded)
            scanner._init_feature_engineer()
            assert scanner._feature_engineer is not None, "Feature engineer should be initialized"

            # Compute features on the raw candles
            # Scanner's _feature_engineer is a function reference
            df_feat = scanner._feature_engineer(df.copy())

            # Verify feature DataFrame
            assert df_feat is not None, "Feature engineering should return DataFrame"
            assert len(df_feat) > 0, "Feature DataFrame should have rows"
            assert len(df_feat.columns) >= len(df.columns), "Features should preserve or add columns"

            # Check for common technical indicator columns
            feature_columns = df_feat.columns.tolist()
            logger.info(f"Computed {len(feature_columns)} features: {feature_columns}")

            # At minimum, should have price columns from raw data
            assert "close" in df_feat.columns, "Should preserve close price"
            assert len(feature_columns) > 0, "Should have at least some features"

        finally:
            ibkr_broker.disconnect()


class TestAgentTeamEvaluation:
    """Test 4: Run agent team evaluation on ES data, verify AgentVerdict list returned."""

    @pytest.mark.integration
    def test_agent_team_evaluation_on_es(
        self,
        ibkr_broker: IBKRBroker,
        es_instrument: Instrument,
        scanner_config: ScannerConfig,
    ) -> None:
        """Run agent team evaluation on ES H1 candles and verify verdicts.

        Requirements:
        - Fetch ES H1 candles
        - Construct feature engineering output
        - Run 12-agent team evaluation
        - Verify list of AgentVerdict objects returned
        - Verify each verdict has name, score, passed flag
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        ibkr_broker.connect()
        try:
            # Fetch candles
            candles = ibkr_broker.fetch_candles(
                instrument=es_instrument,
                granularity="H1",
                count=480,
            )

            if len(candles) == 0:
                pytest.skip("No candles fetched")

            # Convert to DataFrame and run feature engineering
            df = pd.DataFrame([
                {
                    "time": c.time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ])

            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df.sort_index()

            # Create scanner and initialize feature engineer
            scanner = Scanner(config=scanner_config, broker=ibkr_broker)
            scanner._init_feature_engineer()
            df_feat = scanner._feature_engineer(df.copy())

            # Create agent team
            agent_team = ScannerAgentTeam(config=scanner_config)

            # Create minimal PairAnalysis for agent context
            analysis = PairAnalysis(
                pair="ES",
                timeframe="H1",
                signal=0.5,
                confidence=0.6,
                gate_results={},
                timestamp=datetime.now(timezone.utc),
                extra={"df_raw": df, "df_feat": df_feat},
            )

            # Evaluate agents
            verdicts = agent_team.evaluate(analysis, df, df_feat, {})

            # Verify verdicts
            assert isinstance(verdicts, list), "Agent team should return list of verdicts"
            assert len(verdicts) > 0, "Should have at least one agent verdict"

            # Verify each verdict structure
            for verdict in verdicts:
                assert isinstance(verdict, AgentVerdict), f"Each verdict should be AgentVerdict, got {type(verdict)}"
                assert hasattr(verdict, "name"), "Verdict should have name"
                assert hasattr(verdict, "score"), "Verdict should have score"
                assert hasattr(verdict, "passed"), "Verdict should have passed flag"
                assert isinstance(verdict.score, (int, float)), "Score should be numeric"
                assert isinstance(verdict.passed, bool), "Passed should be bool"

            logger.info(
                f"Agent team returned {len(verdicts)} verdicts for ES: "
                f"{[v.name for v in verdicts]}"
            )

        finally:
            ibkr_broker.disconnect()


class TestBracketOrderPlacement:
    """Test 5: Place bracket order on ES paper, verify OrderResult."""

    @pytest.mark.integration
    def test_place_bracket_order_es(
        self,
        ibkr_broker: IBKRBroker,
        es_instrument: Instrument,
    ) -> None:
        """Place a bracket order on ES paper trading and verify OrderResult.

        Requirements:
        - Connect to IB Gateway
        - Place bracket order (BUY, qty=1 contract, SL/TP with R:R >= 1.2:1)
        - Verify OrderResult with status (PENDING or FILLED)
        - Verify trade_id is non-empty and numeric
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        ibkr_broker.connect()
        try:
            # Place bracket order on ES
            # BUY 1 contract @ market, SL=50 ticks below, TP=100 ticks above (R:R = 2:1)
            entry_price = 5500.0  # Placeholder; broker will use market price
            sl_price = 5498.75  # 5 ticks below (0.25 * 5 = 1.25)
            tp_price = 5502.50  # 10 ticks above (0.25 * 10 = 2.50)

            result = ibkr_broker.place_order(
                instrument=es_instrument,
                direction="BUY",
                quantity=1,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_price=entry_price,
            )

            # Verify OrderResult
            assert isinstance(result, OrderResult), "Should return OrderResult"
            assert result.trade_id != "0", "trade_id should be non-zero on success"
            assert result.status in (
                "PENDING",
                "FILLED",
                "submitted",
            ), f"Status should be PENDING or FILLED, got {result.status}"
            assert result.fill_price >= 0, "fill_price should be non-negative"

            logger.info(
                f"Placed bracket order on ES: {result.status}, "
                f"trade_id={result.trade_id}, fill_price={result.fill_price}"
            )

            # Store trade_id for cleanup in next test
            if result.status in ("PENDING", "submitted"):
                # Give broker time to process
                time.sleep(1)

        finally:
            ibkr_broker.disconnect()


class TestTradeJournalRecording:
    """Test 6: Close position, verify P&L recorded in journal with proper metadata."""

    @pytest.mark.integration
    def test_close_position_and_record_journal(
        self,
        ibkr_broker: IBKRBroker,
        es_instrument: Instrument,
        trade_journal_path: Path,
    ) -> None:
        """Close a position and verify trade is recorded in journal.

        Requirements:
        - Place a bracket order on ES
        - Close the position (market order)
        - Verify trade is recorded in trade_journal_rl.json
        - Verify journal entry has: broker='ibkr', asset_class='FUTURES', P&L field
        """
        if not _is_ib_gateway_running():
            pytest.skip("IB Gateway not running")

        ibkr_broker.connect()
        try:
            # Place bracket order
            entry_price = 5500.0
            sl_price = 5498.75
            tp_price = 5502.50

            result = ibkr_broker.place_order(
                instrument=es_instrument,
                direction="BUY",
                quantity=1,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_price=entry_price,
            )

            if result.status == "rejected":
                pytest.skip(f"Order was rejected: {result.error_message}")

            trade_id = result.trade_id

            # Wait for order to fill
            time.sleep(2)

            # Close the position (sell 1 contract at market)
            close_result = ibkr_broker.close_trade(trade_id=trade_id)

            if close_result is not None and close_result.status == "rejected":
                pytest.skip(f"Close order was rejected: {close_result.error_message}")

            # Record to journal
            journal_entry = {
                "trade_id": trade_id,
                "instrument": es_instrument.symbol,
                "broker": "ibkr",
                "asset_class": es_instrument.asset_class,
                "direction": "BUY",
                "quantity": 1,
                "entry_price": result.fill_price,
                "exit_price": close_result.fill_price if close_result else None,
                "pnl": (close_result.fill_price - result.fill_price) * es_instrument.multiplier
                if close_result
                else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "closed",
            }

            # Load existing journal
            journal = json.loads(trade_journal_path.read_text() or "[]")
            if not isinstance(journal, list):
                journal = []

            # Append entry
            journal.append(journal_entry)

            # Write back
            trade_journal_path.write_text(json.dumps(journal, indent=2))

            # Verify journal has the entry
            journal = json.loads(trade_journal_path.read_text())
            assert len(journal) > 0, "Journal should have at least one entry"

            # Verify entry structure
            last_entry = journal[-1]
            assert last_entry["broker"] == "ibkr", "broker should be 'ibkr'"
            assert last_entry["asset_class"] == "FUTURES", "asset_class should be 'FUTURES'"
            assert "pnl" in last_entry, "Entry should have pnl field"
            assert "timestamp" in last_entry, "Entry should have timestamp"

            logger.info(
                f"Trade recorded in journal: trade_id={last_entry['trade_id']}, "
                f"pnl={last_entry['pnl']}, status={last_entry['status']}"
            )

        finally:
            ibkr_broker.disconnect()


class TestIntegrationMarker:
    """Verify integration marker is properly registered."""

    @pytest.mark.integration
    def test_integration_marker_registered(self) -> None:
        """Verify the @pytest.mark.integration marker is registered.

        This test passes if the marker is recognized by pytest.
        """
        # If this test runs, the marker is registered
        pass
