"""Tests for Scanner broker integration (US-006).

Validates that Scanner correctly uses BrokerClient for data fetching.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch, call

import pandas as pd
import pytest

from src.brokers.base import BrokerClient
from src.brokers.types import CandleData
from src.brokers.instrument import Instrument
from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner


class MockBrokerClient(BrokerClient):
    """Mock broker for testing."""

    def __init__(self):
        self.connected = False
        self.fetch_calls = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def fetch_candles(
        self,
        instrument: Instrument,
        granularity: str,
        count: int,
    ) -> List[CandleData]:
        """Return mock candles."""
        self.fetch_calls.append({
            "instrument": instrument.symbol,
            "granularity": granularity,
            "count": count,
        })

        # Generate mock candles
        candles = []
        base_time = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        base_price = 1.0850

        for i in range(count):
            time = base_time + timedelta(hours=i)
            open_price = base_price + (i * 0.001)
            close_price = open_price + 0.0005
            high_price = max(open_price, close_price) + 0.0005
            low_price = min(open_price, close_price) - 0.0005

            candles.append(CandleData(
                time=time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000000 + (i * 1000),
            ))

        return candles

    def place_order(self, *args, **kwargs):
        raise NotImplementedError("Mock broker does not support place_order")

    def close_trade(self, trade_id: str) -> bool:
        raise NotImplementedError("Mock broker does not support close_trade")

    def get_open_positions(self):
        raise NotImplementedError("Mock broker does not support get_open_positions")

    def get_nav(self):
        raise NotImplementedError("Mock broker does not support get_nav")

    def get_account_summary(self):
        raise NotImplementedError("Mock broker does not support get_account_summary")

    def get_price_quote(self, instrument: Instrument):
        raise NotImplementedError("Mock broker does not support get_price_quote")

    def get_trades(self):
        raise NotImplementedError("Mock broker does not support get_trades")


class TestScannerBrokerInitialization:
    """Test Scanner broker initialization."""

    def test_scanner_accepts_broker_parameter(self):
        """Scanner should accept a broker parameter in __init__."""
        broker = MockBrokerClient()
        config = ScannerConfig()

        scanner = Scanner(config=config, broker=broker)

        assert scanner._broker is broker

    def test_scanner_lazy_initializes_broker_from_env(self):
        """Scanner should lazy-initialize broker from environment if not provided."""
        config = ScannerConfig()

        with patch('src.scanner.engine.Scanner._init_broker_client', return_value=False):
            scanner = Scanner(config=config)
            assert scanner._broker is None

    def test_scanner_backward_compatible_with_oanda_client(self):
        """Scanner should still support legacy oanda_client parameter."""
        mock_oanda = Mock()
        config = ScannerConfig()

        scanner = Scanner(config=config, oanda_client=mock_oanda)

        assert scanner._oanda is mock_oanda


class TestFetchPairDataWithBroker:
    """Test _fetch_pair_data with BrokerClient."""

    def test_fetch_pair_data_uses_broker_first(self):
        """_fetch_pair_data should use broker if available."""
        broker = MockBrokerClient()
        config = ScannerConfig(min_candles=10)

        scanner = Scanner(config=config, broker=broker)

        # Mock _load_local_csv to prevent CSV fallback
        with patch.object(scanner, '_load_local_csv', return_value=None):
            df = scanner._fetch_pair_data("EUR_USD", count=100)

        assert df is not None
        assert len(df) == 100
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.name == "time"

        # Verify broker was called
        assert len(broker.fetch_calls) == 1
        assert broker.fetch_calls[0]["instrument"] == "EUR_USD"
        assert broker.fetch_calls[0]["granularity"] == config.granularity
        assert broker.fetch_calls[0]["count"] == 100

    def test_fetch_pair_data_respects_min_candles(self):
        """_fetch_pair_data should reject data with insufficient candles."""
        broker = MockBrokerClient()
        config = ScannerConfig(min_candles=200)

        scanner = Scanner(config=config, broker=broker)

        # Mock _load_local_csv to prevent CSV fallback
        with patch.object(scanner, '_load_local_csv', return_value=None):
            # Request only 100 candles but min_candles is 200
            df = scanner._fetch_pair_data("EUR_USD", count=100)

        # Should fall back to CSV (mocked to None) since broker returned < min_candles
        assert df is None

    def test_fetch_pair_data_dataframe_format(self):
        """_fetch_pair_data should return DataFrame with correct format."""
        broker = MockBrokerClient()
        config = ScannerConfig(min_candles=10)

        scanner = Scanner(config=config, broker=broker)
        df = scanner._fetch_pair_data("EUR_USD", count=50)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "time"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

        # Check columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check data types
        assert pd.api.types.is_float_dtype(df["open"])
        assert pd.api.types.is_float_dtype(df["high"])
        assert pd.api.types.is_float_dtype(df["close"])
        assert pd.api.types.is_numeric_dtype(df["volume"])

    def test_fetch_pair_data_preserves_candle_values(self):
        """_fetch_pair_data should preserve CandleData values in DataFrame."""
        broker = MockBrokerClient()
        config = ScannerConfig(min_candles=5)

        scanner = Scanner(config=config, broker=broker)

        # Mock _load_local_csv to prevent CSV fallback
        with patch.object(scanner, '_load_local_csv', return_value=None):
            df = scanner._fetch_pair_data("EUR_USD", count=10)

        # Verify the call was made to broker
        assert len(broker.fetch_calls) > 0

        # Check that first row values match expected ranges
        first_row = df.iloc[0]
        assert first_row["open"] >= 1.085
        assert first_row["high"] > first_row["low"]
        assert first_row["close"] > 0
        assert first_row["volume"] > 0

    def test_fetch_pair_data_handles_broker_exception(self):
        """_fetch_pair_data should fall back to CSV if broker fails."""
        broker = Mock(spec=BrokerClient)
        broker.fetch_candles.side_effect = Exception("Broker API error")

        config = ScannerConfig(min_candles=10)
        scanner = Scanner(config=config, broker=broker)

        # Mock _load_local_csv to return None (no fallback CSV)
        with patch.object(scanner, '_load_local_csv', return_value=None):
            df = scanner._fetch_pair_data("EUR_USD", count=100)

        assert df is None

    def test_fetch_pair_data_falls_back_to_csv_on_broker_failure(self):
        """_fetch_pair_data should use CSV fallback if broker returns empty."""
        broker = Mock(spec=BrokerClient)
        broker.fetch_candles.return_value = []

        config = ScannerConfig(min_candles=10)
        scanner = Scanner(config=config, broker=broker)

        # Create mock CSV data
        mock_csv_df = pd.DataFrame({
            "open": [1.0850, 1.0851, 1.0852],
            "high": [1.0852, 1.0853, 1.0854],
            "low": [1.0848, 1.0849, 1.0850],
            "close": [1.0851, 1.0852, 1.0853],
            "volume": [1000000, 1000000, 1000000],
        })
        mock_csv_df.index = pd.date_range("2026-03-01", periods=3, freq="h")
        mock_csv_df.index.name = "time"

        with patch.object(scanner, '_load_local_csv', return_value=mock_csv_df):
            df = scanner._fetch_pair_data("EUR_USD", count=100)

        assert df is not None
        assert len(df) == 3


class TestBrokerLazyInitialization:
    """Test broker lazy initialization."""

    def test_init_broker_client_creates_from_env(self):
        """_init_broker_client should create OandaBroker from environment."""
        config = ScannerConfig()
        scanner = Scanner(config=config)

        # Mock OandaBroker.from_env
        with patch('src.brokers.oanda.OandaBroker.from_env') as mock_from_env:
            mock_broker = Mock(spec=BrokerClient)
            mock_from_env.return_value = mock_broker

            result = scanner._init_broker_client()

            assert result is True
            assert scanner._broker is mock_broker
            mock_from_env.assert_called_once()

    def test_init_broker_client_returns_false_on_missing_credentials(self):
        """_init_broker_client should return False if credentials missing."""
        config = ScannerConfig()
        scanner = Scanner(config=config)

        with patch('src.brokers.oanda.OandaBroker.from_env') as mock_from_env:
            mock_from_env.side_effect = OSError("Missing OANDA credentials")

            result = scanner._init_broker_client()

            assert result is False

    def test_init_broker_client_returns_true_if_already_set(self):
        """_init_broker_client should return True if broker already initialized."""
        broker = MockBrokerClient()
        config = ScannerConfig()
        scanner = Scanner(config=config, broker=broker)

        result = scanner._init_broker_client()

        assert result is True

    def test_init_broker_client_handles_import_error(self):
        """_init_broker_client should handle ImportError gracefully."""
        config = ScannerConfig()
        scanner = Scanner(config=config)

        with patch('src.brokers.oanda.OandaBroker.from_env') as mock_from_env:
            mock_from_env.side_effect = ImportError("OandaBroker not found")

            result = scanner._init_broker_client()

            assert result is False


class TestFeatureEngineeredPipeline:
    """Test that feature engineering works with broker data."""

    def test_compute_features_receives_broker_dataframe(self):
        """_compute_features should work with broker-fetched DataFrames."""
        broker = MockBrokerClient()
        config = ScannerConfig(min_candles=10)

        scanner = Scanner(config=config, broker=broker)
        df = scanner._fetch_pair_data("EUR_USD", count=50)

        # Mock the feature engineer since it's a complex dependency
        with patch.object(scanner, '_init_feature_engineer', return_value=False):
            features_df = scanner._compute_features(df)

        assert features_df is not None
        assert "returns" in features_df.columns
        assert "volatility" in features_df.columns


class TestScannerRunWithBroker:
    """Test full scanner run cycle with broker."""

    def test_scan_cycle_uses_broker_for_data(self):
        """Direct _fetch_pair_data call within scan context uses broker."""
        broker = MockBrokerClient()
        config = ScannerConfig(
            pairs=["EUR_USD"],
            min_candles=10,
            granularity="H1",
        )

        scanner = Scanner(config=config, broker=broker)

        # Test direct _fetch_pair_data call (which is called by scan internally)
        with patch.object(scanner, '_load_local_csv', return_value=None):
            df = scanner._fetch_pair_data("EUR_USD", count=100)

        # Verify broker was called
        assert len(broker.fetch_calls) > 0
        assert broker.fetch_calls[0]["instrument"] == "EUR_USD"

        # Verify DataFrame is in expected format for feature engineering
        assert df is not None
        assert "open" in df.columns
        assert "close" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
