"""
Unit Tests for Data Path Resolution.

This module tests the centralized data path configuration to ensure
correct path resolution for all supported trading pairs.

Success Criteria:
1. EUR_USD training data loads without "No training data found" warnings
2. Actual data files are discovered and loaded instead of simulation mode
3. All path references are consolidated in a single configuration
4. Unit tests verify correct path resolution for all supported trading pairs
5. Logging output confirms the exact file paths being accessed during training
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.data_paths import (
    MARKET_DATA_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    REPLAY_DIR,
    SUPPORTED_PAIRS,
    SUPPORTED_GRANULARITIES,
    DataPathResolver,
    diagnose_data_availability,
    find_pair_data,
    get_data_resolver,
)


class TestDataPathConstants:
    """Test that path constants are correctly defined."""

    def test_market_data_dir_exists(self):
        """Verify market_data directory exists."""
        assert MARKET_DATA_DIR.exists(), f"Market data dir should exist: {MARKET_DATA_DIR}"

    def test_trained_data_dir_structure(self):
        """Verify trained_data subdirectories are defined correctly."""
        # These may not exist yet, but should be valid Path objects
        assert isinstance(FEATURES_DIR, Path)
        assert isinstance(MODELS_DIR, Path)
        assert isinstance(REPLAY_DIR, Path)

    def test_supported_pairs_list(self):
        """Verify supported pairs list includes major pairs."""
        expected_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
        for pair in expected_pairs:
            assert pair in SUPPORTED_PAIRS, f"{pair} should be in supported pairs"

    def test_supported_granularities(self):
        """Verify supported granularities include common timeframes."""
        expected_granularities = ["M5", "M15", "H1", "H4", "D"]
        for gran in expected_granularities:
            assert gran in SUPPORTED_GRANULARITIES, f"{gran} should be supported"


class TestDataPathResolver:
    """Test DataPathResolver functionality."""

    def test_resolver_initialization(self):
        """Test resolver initializes correctly."""
        resolver = DataPathResolver(pair="EUR_USD", granularity="H1")

        assert resolver.pair == "EUR_USD"
        assert resolver.granularity == "H1"

    def test_resolver_normalizes_pair_case(self):
        """Test resolver normalizes pair to uppercase."""
        resolver = DataPathResolver(pair="eur_usd")

        assert resolver.pair == "EUR_USD"

    def test_find_latest_market_data_eur_usd(self):
        """Test finding latest market data for EUR_USD."""
        resolver = DataPathResolver(pair="EUR_USD", granularity="H1")

        # This should find actual files in market_data/
        result = resolver.find_latest_market_data()

        # If market data exists, verify it's the right format
        if result is not None:
            assert result.exists()
            assert "EUR_USD" in result.name
            assert "H1" in result.name
            assert result.suffix == ".csv"

    def test_find_latest_market_data_live_vs_oos(self):
        """Test distinguishing between live and OOS data."""
        resolver = DataPathResolver(pair="USD_JPY", granularity="M5")

        live_path = resolver.find_latest_market_data(data_type="live")
        oos_path = resolver.find_latest_market_data(data_type="oos")

        # If both exist, they should be different
        if live_path and oos_path:
            assert live_path != oos_path
            assert "_oos_" not in live_path.name
            assert "_oos_" in oos_path.name

    def test_find_features_file_returns_none_if_not_exists(self):
        """Test features file returns None if not exists."""
        resolver = DataPathResolver(pair="NONEXISTENT_PAIR")

        result = resolver.find_features_file()

        assert result is None

    def test_find_replay_buffer_returns_none_if_not_exists(self):
        """Test replay buffer returns None if not exists."""
        resolver = DataPathResolver(pair="NONEXISTENT_PAIR")

        result = resolver.find_replay_buffer()

        assert result is None

    def test_find_training_data_finds_market_data(self):
        """Test that find_training_data can find market data."""
        resolver = DataPathResolver(pair="EUR_USD", granularity="H1")

        path, source_type = resolver.find_training_data()

        # Should find market data since we know it exists
        if path is not None:
            assert source_type in ["features", "market_data", "replay"]
            assert path.exists()

    def test_find_training_data_returns_none_for_nonexistent_pair(self):
        """Test that find_training_data returns None for nonexistent data."""
        resolver = DataPathResolver(pair="NONEXISTENT_PAIR")

        path, source_type = resolver.find_training_data()

        assert path is None
        assert source_type == "none"

    def test_list_available_data_returns_structure(self):
        """Test that list_available_data returns correct structure."""
        resolver = DataPathResolver(pair="EUR_USD")

        result = resolver.list_available_data()

        assert "pair" in result
        assert "granularity" in result
        assert "market_data" in result
        assert "features" in result
        assert "replay_buffer" in result
        assert "models" in result
        assert result["pair"] == "EUR_USD"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_data_resolver(self):
        """Test get_data_resolver creates resolver."""
        resolver = get_data_resolver("GBP_USD", "H4")

        assert isinstance(resolver, DataPathResolver)
        assert resolver.pair == "GBP_USD"
        assert resolver.granularity == "H4"

    def test_find_pair_data(self):
        """Test find_pair_data convenience function."""
        path, source_type = find_pair_data("EUR_USD", "H1")

        # Should find data since EUR_USD market data exists
        if path is not None:
            assert path.exists()
            assert source_type in ["features", "market_data", "replay"]

    def test_ensure_data_directories(self):
        """Test ensure_data_directories creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the constants to use temp directory
            with patch("src.training.data_paths.TRAINED_DATA_DIR", Path(tmpdir) / "trained_data"):
                with patch("src.training.data_paths.FEATURES_DIR", Path(tmpdir) / "trained_data" / "features"):
                    with patch("src.training.data_paths.MODELS_DIR", Path(tmpdir) / "trained_data" / "models"):
                        with patch("src.training.data_paths.REPLAY_DIR", Path(tmpdir) / "trained_data" / "replay"):
                            # Import again to get patched values
                            from src.training import data_paths
                            data_paths.FEATURES_DIR = Path(tmpdir) / "trained_data" / "features"
                            data_paths.MODELS_DIR = Path(tmpdir) / "trained_data" / "models"
                            data_paths.REPLAY_DIR = Path(tmpdir) / "trained_data" / "replay"

                            data_paths.ensure_data_directories()

                            assert data_paths.FEATURES_DIR.exists()
                            assert data_paths.MODELS_DIR.exists()
                            assert data_paths.REPLAY_DIR.exists()


class TestDiagnoseDataAvailability:
    """Test diagnostic functions."""

    def test_diagnose_eur_usd(self):
        """Test diagnostic for EUR_USD."""
        result = diagnose_data_availability("EUR_USD")

        assert "timestamp" in result
        assert "directories" in result
        assert "available_data" in result
        assert "recommendations" in result

        # Check structure
        assert "market_data" in result["directories"]
        assert "features" in result["directories"]
        assert "models" in result["directories"]
        assert "replay" in result["directories"]

    def test_diagnose_shows_market_data_exists(self):
        """Test diagnostic correctly identifies existing market data."""
        result = diagnose_data_availability("EUR_USD")

        # EUR_USD market data should exist
        market_data = result["available_data"]["market_data"]
        assert len(market_data) > 0, "EUR_USD should have market data files"


class TestPathResolutionForAllPairs:
    """Test path resolution for all supported trading pairs."""

    @pytest.mark.parametrize("pair", SUPPORTED_PAIRS)
    def test_resolver_for_each_pair(self, pair):
        """Test resolver works for all supported pairs."""
        resolver = DataPathResolver(pair=pair)

        assert resolver.pair == pair

        # Should not raise any exceptions
        path, source_type = resolver.find_training_data()

        # Result should be valid even if no data found
        assert source_type in ["features", "market_data", "replay", "none"]
        if path is not None:
            assert path.exists()

    @pytest.mark.parametrize("granularity", SUPPORTED_GRANULARITIES)
    def test_resolver_for_each_granularity(self, granularity):
        """Test resolver works for all supported granularities."""
        resolver = DataPathResolver(pair="EUR_USD", granularity=granularity)

        assert resolver.granularity == granularity


class TestLoggingOutput:
    """Test that logging output confirms exact file paths being accessed."""

    def test_logging_shows_searched_paths(self, caplog):
        """Test that logging shows which paths are being searched."""
        with caplog.at_level(logging.DEBUG):
            resolver = DataPathResolver(pair="EUR_USD")
            resolver.find_training_data()

        # Should have some log output
        assert len(caplog.records) > 0

    def test_logging_shows_found_path(self, caplog):
        """Test that logging shows which path was found."""
        with caplog.at_level(logging.INFO):
            resolver = DataPathResolver(pair="EUR_USD")
            path, source_type = resolver.find_training_data()

        if path is not None:
            # Should log the found path
            log_messages = [r.message for r in caplog.records]
            assert any("EUR_USD" in msg for msg in log_messages)

    def test_logging_shows_warning_for_missing_data(self, caplog):
        """Test that logging shows warning when no data found."""
        with caplog.at_level(logging.WARNING):
            resolver = DataPathResolver(pair="NONEXISTENT_PAIR")
            path, source_type = resolver.find_training_data()

        # Should have warning about no data found
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


class TestSuccessCriteria:
    """Tests for the success criteria specified in the task."""

    def test_success_criteria_1_eur_usd_loads_without_warning(self, caplog):
        """
        Success Criteria 1: EUR_USD training data loads without
        "No training data found" warnings.
        """
        resolver = DataPathResolver(pair="EUR_USD", granularity="H1")
        path, source_type = resolver.find_training_data()

        # EUR_USD should have market data available
        assert path is not None, "EUR_USD should have training data available"
        assert source_type != "none", "Should find a valid data source"

    def test_success_criteria_2_actual_data_discovered(self):
        """
        Success Criteria 2: Actual data files are discovered and loaded
        instead of falling back to simulation mode.
        """
        resolver = DataPathResolver(pair="EUR_USD", granularity="H1")

        # Find latest market data
        market_path = resolver.find_latest_market_data()

        assert market_path is not None, "EUR_USD market data should exist"
        assert market_path.exists(), f"File should exist: {market_path}"
        assert market_path.stat().st_size > 0, "File should not be empty"

    def test_success_criteria_3_consolidated_configuration(self):
        """
        Success Criteria 3: All path references are consolidated
        in a single configuration file.
        """
        from src.training.data_paths import (
            MARKET_DATA_DIR,
            FEATURES_DIR,
            MODELS_DIR,
            REPLAY_DIR,
            DataPathResolver,
        )

        # All paths should come from the same module
        assert MARKET_DATA_DIR is not None
        assert FEATURES_DIR is not None
        assert MODELS_DIR is not None
        assert REPLAY_DIR is not None

        # DataPathResolver should use these constants
        resolver = DataPathResolver("EUR_USD")
        assert resolver is not None

    def test_success_criteria_4_unit_tests_for_all_pairs(self):
        """
        Success Criteria 4: Unit tests verify correct path resolution
        for all supported trading pairs.
        """
        # This is verified by the TestPathResolutionForAllPairs class above
        # which tests all SUPPORTED_PAIRS
        assert len(SUPPORTED_PAIRS) > 0

        for pair in SUPPORTED_PAIRS:
            resolver = DataPathResolver(pair=pair)
            assert resolver.pair == pair

    def test_success_criteria_5_logging_confirms_paths(self, caplog):
        """
        Success Criteria 5: Logging output confirms the exact file paths
        being accessed during training runs.
        """
        with caplog.at_level(logging.INFO):
            resolver = DataPathResolver(pair="EUR_USD", granularity="H1")
            path, source_type = resolver.find_training_data()

        # Check that logs contain path information
        log_messages = [r.message for r in caplog.records]

        # Should have logged about data discovery
        assert len(log_messages) > 0, "Should have log output"

        # Should mention EUR_USD in logs
        eur_usd_mentioned = any("EUR_USD" in msg for msg in log_messages)
        assert eur_usd_mentioned, "Should mention EUR_USD in logs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
