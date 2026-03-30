"""Tests for US-016: Roll Calendar integration in Scanner loop.

Tests verify that:
1. RollCalendar is initialized in Scanner
2. Roll checks happen before data fetch for FUTURES
3. Expired contracts are blocked
4. Roll warnings are logged
5. FX instruments skip roll checks
6. Non-supported futures symbols are skipped gracefully
"""

import unittest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.scanner.engine import Scanner
from src.scanner.config import ScannerConfig
from src.brokers.instrument import Instrument
from src.brokers.registry import InstrumentRegistry
from src.brokers.roll_calendar import RollCalendar


class TestScannerRollCalendarInit(unittest.TestCase):
    """Test RollCalendar initialization in Scanner."""

    def test_roll_calendar_initializes(self):
        """RollCalendar should initialize and be available."""
        config = ScannerConfig()
        scanner = Scanner(config=config)

        self.assertIsNotNone(scanner._roll_calendar)
        self.assertIsInstance(scanner._roll_calendar, RollCalendar)

    def test_roll_calendar_survives_import_error(self):
        """Scanner should continue if RollCalendar import fails."""
        config = ScannerConfig()

        with patch("src.brokers.roll_calendar.RollCalendar", side_effect=ImportError("Fail")):
            scanner = Scanner(config=config)
            # Should not raise; _roll_calendar may be None or initialized
            self.assertIsNotNone(scanner)


class TestRollCalendarChecks(unittest.TestCase):
    """Test roll calendar checks in _scan_pair."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ScannerConfig(non_interactive=True)
        self.scanner = Scanner(config=self.config)
        # Mock session timing to always allow trading (tests may run on weekends)
        self.scanner.config.get_trading_session_status = Mock(
            return_value={"session_allowed": True, "message": "mocked for test"}
        )

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.scanner.engine.Scanner._compute_features")
    @patch("src.brokers.registry.get_registry")
    def test_fx_pair_skips_roll_check(self, mock_get_registry, mock_compute_features, mock_fetch):
        """FX instruments should skip roll calendar checks entirely."""
        # Create FX instrument with a symbol not in blocked_pairs
        fx_instr = Instrument.fx(
            symbol="GBP_USD",
            broker_symbol="GBP/USD",
            pip_value=0.0001,
        )

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_optional.return_value = fx_instr
        mock_get_registry.return_value = mock_registry

        # Mock data
        mock_fetch.return_value = None  # Simulates no data

        # Scan should skip roll check and go straight to data fetch attempt
        result = self.scanner._scan_pair("GBP_USD")

        # Should return error about no data, not about roll
        self.assertIsNotNone(result)
        self.assertEqual(result.pair, "GBP_USD")
        self.assertEqual(result.direction, "HOLD")
        self.assertIn("No data", result.error or "")

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.brokers.registry.get_registry")
    def test_expired_contract_blocks_scan(self, mock_get_registry, mock_fetch):
        """Expired contract (days_until_roll <= 0) should block scanning."""
        # Create FUTURES instrument
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_optional.return_value = fut_instr
        mock_get_registry.return_value = mock_registry

        # Mock roll calendar to return expired (days_until_roll <= 0)
        self.scanner._roll_calendar.get_active_contract = Mock(return_value="202203")
        self.scanner._roll_calendar.should_roll = Mock(return_value=False)
        self.scanner._roll_calendar.days_until_roll = Mock(return_value=-5)

        # Scan should return early with expiry error
        result = self.scanner._scan_pair("ES")

        # Should NOT call fetch_pair_data
        mock_fetch.assert_not_called()

        # Should return error about contract expiry
        self.assertIsNotNone(result)
        self.assertEqual(result.pair, "ES")
        self.assertEqual(result.direction, "HOLD")
        self.assertIn("expired", result.error.lower() or "")

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.brokers.registry.get_registry")
    def test_roll_window_logs_warning(self, mock_get_registry, mock_fetch):
        """Entering roll window should log warning."""
        # Create FUTURES instrument
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_optional.return_value = fut_instr
        mock_get_registry.return_value = mock_registry

        # Mock roll calendar to return within roll window
        self.scanner._roll_calendar.get_active_contract = Mock(return_value="202203")
        self.scanner._roll_calendar.should_roll = Mock(return_value=True)
        self.scanner._roll_calendar.days_until_roll = Mock(return_value=2)

        # Mock fetch to return None (test should pass through to fetch even with warning)
        mock_fetch.return_value = None

        with patch("src.scanner.engine.logger") as mock_logger:
            result = self.scanner._scan_pair("ES")

            # Should have logged warning about roll window
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            self.assertTrue(any("roll window" in str(c).lower() for c in warning_calls))

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.brokers.registry.get_registry")
    def test_active_contract_logs_info(self, mock_get_registry, mock_fetch):
        """Active contract should log debug info."""
        # Create FUTURES instrument
        fut_instr = Instrument.futures(
            symbol="ES",
            broker_symbol="ES",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_optional.return_value = fut_instr
        mock_get_registry.return_value = mock_registry

        # Mock roll calendar with normal state (not in roll window, not expired)
        self.scanner._roll_calendar.get_active_contract = Mock(return_value="202203")
        self.scanner._roll_calendar.should_roll = Mock(return_value=False)
        self.scanner._roll_calendar.days_until_roll = Mock(return_value=15)

        # Mock fetch to return None
        mock_fetch.return_value = None

        with patch("src.scanner.engine.logger") as mock_logger:
            result = self.scanner._scan_pair("ES")

            # Should have logged debug info about active contract
            mock_logger.debug.assert_called()
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            self.assertTrue(
                any("active contract" in str(c).lower() for c in debug_calls),
                f"Expected 'active contract' in debug calls: {debug_calls}"
            )

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.brokers.registry.get_registry")
    def test_unsupported_futures_symbol_handled(self, mock_get_registry, mock_fetch):
        """Unsupported futures symbol should be handled gracefully."""
        # Create a FUTURES instrument not in roll calendar (e.g., custom symbol)
        fut_instr = Instrument.futures(
            symbol="XX_UNKNOWN",
            broker_symbol="XX",
            tick_size=0.25,
            tick_value=12.50,
            multiplier=50.0,
        )

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_optional.return_value = fut_instr
        mock_get_registry.return_value = mock_registry

        # Mock roll calendar to raise ValueError for unknown symbol
        self.scanner._roll_calendar.get_active_contract = Mock(
            side_effect=ValueError("Unsupported symbol: XX_UNKNOWN")
        )

        # Mock fetch to return None
        mock_fetch.return_value = None

        with patch("src.scanner.engine.logger") as mock_logger:
            result = self.scanner._scan_pair("XX_UNKNOWN")

            # Should log debug about symbol not in roll calendar
            mock_logger.debug.assert_called()
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            self.assertTrue(
                any("not in roll calendar" in str(c).lower() for c in debug_calls),
                f"Expected 'not in roll calendar' in debug calls: {debug_calls}"
            )

            # Should still proceed to fetch
            mock_fetch.assert_called_once()

    @patch("src.scanner.engine.Scanner._fetch_pair_data")
    @patch("src.brokers.registry.get_registry")
    def test_registry_lookup_error_handled(self, mock_get_registry, mock_fetch):
        """Registry lookup error should not block scanning."""
        # Mock registry to return None (not found)
        mock_registry = Mock()
        mock_registry.get_optional.return_value = None
        mock_get_registry.return_value = mock_registry

        # Mock fetch to return None
        mock_fetch.return_value = None

        with patch("src.scanner.engine.logger") as mock_logger:
            result = self.scanner._scan_pair("UNKNOWN_PAIR")

            # Should proceed to fetch (no instrument = can't check asset class)
            mock_fetch.assert_called_once()


class TestRollCalendarRealAPI(unittest.TestCase):
    """Test real RollCalendar API with Scanner."""

    def test_es_contract_expiry(self):
        """Test ES contract expiration logic."""
        rc = RollCalendar()

        # Get active contract
        active = rc.get_active_contract("ES")
        self.assertIsNotNone(active)
        self.assertRegex(active, r"^\d{6}$")  # YYYYMM format

        # Days until roll should be >= 0 for active contract
        days = rc.days_until_roll("ES")
        self.assertGreaterEqual(days, -1)  # Can be -1 on day after expiry

        # should_roll depends on days_until_roll
        should_roll = rc.should_roll("ES")
        self.assertIsInstance(should_roll, bool)

    def test_cl_contract_expiry(self):
        """Test CL contract expiration logic."""
        rc = RollCalendar()

        # Get active contract
        active = rc.get_active_contract("CL")
        self.assertIsNotNone(active)
        self.assertRegex(active, r"^\d{6}$")

        # Days until roll
        days = rc.days_until_roll("CL")
        self.assertIsInstance(days, int)

    def test_unsupported_symbol_raises(self):
        """Unsupported symbol should raise ValueError."""
        rc = RollCalendar()

        with self.assertRaises(ValueError):
            rc.get_active_contract("UNKNOWN_SYMBOL")


if __name__ == "__main__":
    unittest.main()
