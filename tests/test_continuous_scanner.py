"""Phase 36 (US-231): Unit tests for ContinuousScanner lifecycle methods.

Tests shutdown sequence, config handling, scan counting, signal setup,
correlation filtering, and scan cycle logging.
Uses mock-based approach — no OANDA or live market dependencies.
"""

import json
import os
import signal
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.scanner.automation.continuous import ContinuousConfig, ContinuousScanner


# ---------------------------------------------------------------------------
# Helper: create a minimal mock scanner
# ---------------------------------------------------------------------------
def _mock_scanner(**config_overrides):
    """Create a mock Scanner object for ContinuousScanner."""
    scanner = MagicMock()
    scanner.config = MagicMock()
    scanner.config.enable_module_activation = False
    scanner.config.enable_memory_manager = False
    scanner._memory_manager = None
    for k, v in config_overrides.items():
        setattr(scanner.config, k, v)
    return scanner


# ---------------------------------------------------------------------------
# Tests: ContinuousConfig
# ---------------------------------------------------------------------------
class TestContinuousConfig:
    def test_default_interval(self):
        cfg = ContinuousConfig()
        assert cfg.interval_minutes == 5

    def test_default_auto_execute_false(self):
        cfg = ContinuousConfig()
        assert cfg.auto_execute is False

    def test_default_top_n(self):
        cfg = ContinuousConfig()
        assert cfg.top_n == 5

    def test_default_max_scans_unlimited(self):
        cfg = ContinuousConfig()
        assert cfg.max_scans is None

    def test_custom_values(self):
        cfg = ContinuousConfig()
        cfg.interval_minutes = 10
        cfg.auto_execute = True
        cfg.max_scans = 50
        assert cfg.interval_minutes == 10
        assert cfg.auto_execute is True
        assert cfg.max_scans == 50


# ---------------------------------------------------------------------------
# Tests: ContinuousScanner init and lifecycle
# ---------------------------------------------------------------------------
class TestContinuousScannerInit:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_init_default_config(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        assert cs._scan_count == 0
        assert cs._running is False
        assert cs.scanner is scanner

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_init_custom_config(self, mock_maint):
        scanner = _mock_scanner()
        cfg = ContinuousConfig()
        cfg.interval_minutes = 15
        cs = ContinuousScanner(scanner, config=cfg)
        assert cs.config.interval_minutes == 15

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_stop_sets_running_false(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        cs._running = True
        cs.stop()
        assert cs._running is False


# ---------------------------------------------------------------------------
# Tests: Signal handler
# ---------------------------------------------------------------------------
class TestSignalHandler:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_signal_handler_stops_scanner(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        cs._running = True
        cs._setup_signal_handler()
        # Simulate SIGINT
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert cs._running is False


# ---------------------------------------------------------------------------
# Tests: Scan cycle logging
# ---------------------------------------------------------------------------
class TestScanCycleLogging:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_log_scan_cycle_no_crash(self, mock_maint):
        """_log_scan_cycle should not crash even with mock data."""
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        cs._scan_count = 5

        mock_result = MagicMock()
        mock_analysis = MagicMock()
        mock_analysis.is_tradeable = True
        mock_analysis.pair = "EUR_USD"
        mock_analysis.overall_score = 0.85
        mock_result.analyses = [mock_analysis]
        mock_result.model_type = "ensemble"

        tmpdir = tempfile.mkdtemp()
        log_file = os.path.join(tmpdir, "trained_data", "scan_cycle_log.jsonl")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Patch pathlib.Path inside the method's local import
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            cs._log_scan_cycle(mock_result, auto_execute=False)
        finally:
            os.chdir(orig_cwd)

        # Verify file was written
        assert os.path.exists(log_file)
        with open(log_file) as f:
            record = json.loads(f.readline())
        assert record["scan_number"] == 5
        assert record["tradeable_count"] == 1


# ---------------------------------------------------------------------------
# Tests: Pair rotation
# ---------------------------------------------------------------------------
class TestPairRotation:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_no_rotation_without_optimizer(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        cs._portfolio_optimizer = None
        # Should return without error
        cs._apply_pair_rotation()

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_rotation_checks_should_rotate(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        mock_optimizer = MagicMock()
        mock_optimizer.should_rotate.return_value = False
        cs._portfolio_optimizer = mock_optimizer
        cs._scan_count = 3

        cs._apply_pair_rotation()
        mock_optimizer.should_rotate.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# Tests: Correlation filter
# ---------------------------------------------------------------------------
class TestCorrelationFilter:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_filter_returns_list(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        # With empty tradeable list, should return empty
        result = cs._filter_correlated_exposure([])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Direction extraction
# ---------------------------------------------------------------------------
class TestDirectionExtraction:
    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_get_direction_long_from_predicted(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        analysis = MagicMock()
        analysis.predicted_direction = "LONG"
        assert cs._get_direction_from_analysis(analysis) == "LONG"

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_get_direction_short_from_predicted(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        analysis = MagicMock()
        analysis.predicted_direction = "SHORT"
        assert cs._get_direction_from_analysis(analysis) == "SHORT"

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_get_direction_hold(self, mock_maint):
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        analysis = MagicMock()
        analysis.predicted_direction = "HOLD"
        assert cs._get_direction_from_analysis(analysis) == "HOLD"

    @patch("src.scanner.automation.continuous.IdleMaintenance", create=True)
    def test_get_direction_from_score_fallback(self, mock_maint):
        """When predicted_direction is missing, infer from overall_score."""
        scanner = _mock_scanner()
        cs = ContinuousScanner(scanner)
        analysis = MagicMock(spec=[])  # No predicted_direction attribute
        analysis.overall_score = 0.7
        result = cs._get_direction_from_analysis(analysis)
        assert result == "LONG"
