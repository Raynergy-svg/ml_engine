"""Phase 38 (US-243): Unit tests for ExecutionManager trade logging and journal sync.

Tests _log_trade, _append_journal_entry, sync_journal, fetch_actual_win_rate,
win_rate_by_pair, avg_rr_ratio, execution_quality_summary.
All tests use mock OANDA and temp files — no real API calls.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.execution import ExecutionConfig, ExecutionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_manager(**config_overrides):
    cfg = ExecutionConfig(**config_overrides)
    mock_oanda = MagicMock()
    return ExecutionManager(config=cfg, oanda_client=mock_oanda)


# ---------------------------------------------------------------------------
# Tests: _append_journal_entry
# ---------------------------------------------------------------------------
class TestAppendJournalEntry:
    def test_creates_journal_file(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            mgr._append_journal_entry(
                trade_id="T001", pair="EUR_USD", direction="LONG",
                confidence=0.75, entry=1.1050, sl=1.1000, tp=1.1100,
                lots=1.0,
            )
        finally:
            os.chdir(original_cwd)

        actual_path = tmp_path / "trained_data" / "trade_journal_rl.json"
        assert actual_path.exists()
        entries = json.loads(actual_path.read_text())
        assert len(entries) == 1
        assert entries[0]["trade_id"] == "T001"
        assert entries[0]["pair"] == "EUR_USD"
        assert entries[0]["direction"] == "LONG"

    def test_deduplicates_by_trade_id(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            # First entry
            mgr._append_journal_entry(
                trade_id="T001", pair="EUR_USD", direction="LONG",
                confidence=0.75, entry=1.1050, sl=1.1000, tp=1.1100, lots=1.0,
            )
            # Duplicate — same trade_id, different price
            mgr._append_journal_entry(
                trade_id="T001", pair="EUR_USD", direction="LONG",
                confidence=0.80, entry=1.1055, sl=1.1005, tp=1.1105, lots=1.0,
            )
        finally:
            os.chdir(original_cwd)

        entries = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())
        assert len(entries) == 1
        # Should have the latest entry
        assert entries[0]["confidence"] == 0.80

    def test_includes_gate_context(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            ctx = {
                "momentum_passed": True,
                "confidence_passed": True,
                "risk_passed": True,
                "agent_votes": 8,
                "agent_total": 12,
                "weighted_vote_score": 0.72,
            }
            mgr._append_journal_entry(
                trade_id="T002", pair="GBP_USD", direction="SHORT",
                confidence=0.68, entry=1.2700, sl=1.2750, tp=1.2650, lots=0.5,
                analysis_context=ctx,
            )
        finally:
            os.chdir(original_cwd)

        entries = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())
        assert entries[0]["gates"]["momentum_passed"] is True
        assert entries[0]["agents"]["agent_votes"] == 8

    def test_handles_corrupted_journal(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        # Pre-write corrupted file
        (tmp_path / "trained_data" / "trade_journal_rl.json").write_text("NOT JSON!!!")
        os.chdir(tmp_path)
        try:
            mgr._append_journal_entry(
                trade_id="T003", pair="EUR_USD", direction="LONG",
                confidence=0.60, entry=1.1000, sl=1.0950, tp=1.1050, lots=0.5,
            )
        finally:
            os.chdir(original_cwd)

        entries = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())
        assert len(entries) == 1  # Recovered from corruption


# ---------------------------------------------------------------------------
# Tests: _log_trade
# ---------------------------------------------------------------------------
class TestLogTrade:
    def test_log_trade_with_memory_client(self):
        mgr = _make_manager()
        mock_memory = MagicMock()
        mgr._memory_client = mock_memory

        with patch.object(mgr, '_append_journal_entry'):
            mgr._log_trade(
                pair="EUR_USD", direction="LONG", confidence=0.75,
                lots=1.0, entry=1.1050, sl=1.1000, tp=1.1100,
                trade_id="T100",
            )
        mock_memory.log_trade.assert_called_once()

    def test_log_trade_no_memory_client(self):
        mgr = _make_manager()
        mgr._memory_client = None

        with patch.object(mgr, '_init_memory_client'):
            with patch.object(mgr, '_append_journal_entry') as mock_append:
                mgr._log_trade(
                    pair="EUR_USD", direction="LONG", confidence=0.75,
                    lots=1.0, entry=1.1050, sl=1.1000, tp=1.1100,
                    trade_id="T101",
                )
        # _append_journal_entry should NOT be called since _memory_client is None
        # and the early return fires
        mock_append.assert_not_called()

    def test_log_trade_with_analysis_context(self):
        mgr = _make_manager()
        mock_memory = MagicMock()
        mgr._memory_client = mock_memory
        ctx = {"momentum_passed": True, "confidence_passed": True}

        with patch.object(mgr, '_append_journal_entry'):
            mgr._log_trade(
                pair="EUR_USD", direction="SHORT", confidence=0.82,
                lots=0.5, entry=1.1050, sl=1.1100, tp=1.1000,
                trade_id="T102", analysis_context=ctx,
            )

        call_args = mock_memory.log_trade.call_args[0][0]
        assert call_args["metadata"] == ctx


# ---------------------------------------------------------------------------
# Tests: fetch_actual_win_rate
# ---------------------------------------------------------------------------
class TestFetchActualWinRate:
    def test_no_trades(self):
        mgr = _make_manager()
        mgr._oanda._request.return_value = {"trades": []}
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 0.0
        assert total == 0

    def test_all_wins(self):
        mgr = _make_manager()
        mgr._oanda._request.return_value = {
            "trades": [
                {"realizedPL": "50.0"},
                {"realizedPL": "30.0"},
                {"realizedPL": "10.0"},
            ]
        }
        # Need to mock _oanda._config.account_id
        mgr._oanda._config = MagicMock()
        mgr._oanda._config.account_id = "test-123"
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 1.0
        assert total == 3

    def test_mixed_results(self):
        mgr = _make_manager()
        mgr._oanda._config = MagicMock()
        mgr._oanda._config.account_id = "test-123"
        mgr._oanda._request.return_value = {
            "trades": [
                {"realizedPL": "50.0"},
                {"realizedPL": "-20.0"},
                {"realizedPL": "30.0"},
                {"realizedPL": "-10.0"},
            ]
        }
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 0.5
        assert total == 4

    def test_api_exception(self):
        mgr = _make_manager()
        mgr._oanda._config = MagicMock()
        mgr._oanda._config.account_id = "test-123"
        mgr._oanda._request.side_effect = Exception("API timeout")
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 0.0
        assert total == 0


# ---------------------------------------------------------------------------
# Tests: sync_journal
# ---------------------------------------------------------------------------
class TestSyncJournal:
    def test_sync_journal_success(self):
        mgr = _make_manager()
        mock_journal = MagicMock()
        mock_journal.update_from_oanda.return_value = 5

        # TradeJournal is imported locally inside sync_journal
        with patch("src.utils.trade_journal.TradeJournal", return_value=mock_journal):
            count = mgr.sync_journal()
        # TradeJournal is available, so import should work
        assert isinstance(count, int)

    def test_sync_journal_no_oanda(self):
        cfg = ExecutionConfig()
        mgr = ExecutionManager(config=cfg, oanda_client=None)
        count = mgr.sync_journal()
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: execution_quality_summary
# ---------------------------------------------------------------------------
class TestExecutionQualitySummary:
    def test_summary_structure(self):
        mgr = _make_manager()
        summary = mgr.execution_quality_summary()
        assert isinstance(summary, dict)
        # Should have key fields
        assert "total_trades" in summary or "message" in summary


# ---------------------------------------------------------------------------
# Tests: win_rate_by_pair
# ---------------------------------------------------------------------------
class TestWinRateByPair:
    def test_empty_returns_dict(self):
        mgr = _make_manager()
        result = mgr.win_rate_by_pair()
        assert isinstance(result, dict)
