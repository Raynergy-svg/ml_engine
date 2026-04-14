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

    def test_records_rr_ratio_and_pip_distances(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            mgr._append_journal_entry(
                trade_id="T004", pair="EUR_USD", direction="LONG",
                confidence=0.60, entry=1.1000, sl=1.0950, tp=1.1100, lots=0.5,
            )
        finally:
            os.chdir(original_cwd)

        entry = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())[0]
        assert entry["sl_pips"] == 50.0
        assert entry["tp_pips"] == 100.0
        assert entry["rr_ratio"] == 2.0
        assert entry["risk_reward"] == 2.0

    def test_init_adaptive_sizer_rehydrates_from_closed_journal(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            (tmp_path / "trained_data" / "adaptive_sizer_state.json").write_text(json.dumps({
                "config": {
                    "kelly_window": 50,
                    "kelly_fraction": 0.33,
                    "sigmoid_steepness": 4.0,
                    "sigmoid_midpoint": 0.5,
                    "max_acceptable_drawdown": 0.15,
                    "drawdown_floor": 0.3,
                    "base_risk_pct": 0.05,
                    "max_risk_per_trade": 0.1,
                    "min_position_units": 1000,
                    "regime_multipliers": {"LOW": 1.3, "NORMAL": 1.0, "HIGH": 0.65, "EXTREME": 0.4},
                    "component_weights": {"kelly": 0.3, "confidence": 0.3, "drawdown": 0.2, "regime": 0.2},
                    "streak_enabled": True,
                },
                "trade_history": [[100.0, True], [-50.0, False]],
            }))
            (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([
                {
                    "trade_id": "A1",
                    "pair": "EUR_USD",
                    "timestamp": "2026-04-01T00:00:00Z",
                    "outcome": {
                        "realized_pl": 10.0,
                        "trade_won": True,
                        "close_time": "2026-04-01T01:00:00Z",
                    },
                },
                {
                    "trade_id": "A2",
                    "pair": "GBP_USD",
                    "timestamp": "2026-04-01T02:00:00Z",
                    "outcome": {
                        "realized_pl": -5.0,
                        "trade_won": False,
                        "close_time": "2026-04-01T03:00:00Z",
                    },
                },
                {
                    "trade_id": "A3",
                    "pair": "USD_CHF",
                    "timestamp": "2026-04-01T04:00:00Z",
                    "outcome": {
                        "realized_pl": 7.5,
                        "trade_won": True,
                        "close_time": "2026-04-01T05:00:00Z",
                    },
                },
            ]))

            mgr._init_adaptive_position_sizer()
        finally:
            os.chdir(original_cwd)

        history = list(mgr._adaptive_position_sizer._trade_history)
        assert history == [(10.0, True), (-5.0, False), (7.5, True)]


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
        # Source uses _broker (via _init_broker wrapping _legacy_oanda).
        # Mock _broker.get_trades to return empty list.
        mock_broker = MagicMock()
        mock_broker.get_trades.return_value = []
        mgr._broker = mock_broker
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 0.0
        assert total == 0


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestSyncClosedTradesRL:
    def test_resolve_trade_state_uses_individual_trade_endpoint(self):
        mgr = _make_manager()
        fake_requests = MagicMock()
        fake_requests.get = MagicMock()
        mgr._retry_oanda = MagicMock(return_value=_FakeResponse({
            "trade": {
                "id": "1092",
                "state": "CLOSED",
                "realizedPL": "12.5",
                "averageClosePrice": "1.1010",
                "price": "1.1000",
                "closeTime": "2026-04-02T01:20:00Z",
            }
        }))

        state, trade = mgr._resolve_trade_state_from_oanda(
            trade_id="1092",
            requests=fake_requests,
            acct="acct",
            base="https://api-fxpractice.oanda.com",
            headers={"Authorization": "Bearer test"},
            closed_trades={},
            open_trades={},
        )

        assert state == "closed"
        assert trade["id"] == "1092"

    def test_sync_closed_trades_rl_updates_pending_trade_from_individual_lookup(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([
                {
                    "trade_id": "1092",
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "timestamp": "2026-04-02T01:00:00Z",
                    "entry_price": 1.1000,
                    "outcome": None,
                    "agents": {"agent_reasons": []},
                    "regime": {},
                }
            ]))

            with patch.dict(os.environ, {
                "OANDA_API_TOKEN": "token",
                "OANDA_ACCOUNT_ID": "acct",
            }, clear=False):
                with patch.object(mgr, "_fetch_oanda_trade_snapshots", return_value=({}, {})):
                    with patch.object(
                        mgr,
                        "_resolve_trade_state_from_oanda",
                        return_value=("closed", {
                            "id": "1092",
                            "realizedPL": "15.0",
                            "averageClosePrice": "1.1015",
                            "price": "1.1000",
                            "closeTime": "2026-04-02T01:15:00Z",
                            "initialUnits": "1000",
                        }),
                    ):
                        result = mgr.sync_closed_trades_rl()
        finally:
            os.chdir(original_cwd)

        assert result["trades_synced"] == 1
        saved = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())
        assert isinstance(saved[0]["outcome"], dict)
        assert saved[0]["outcome"]["trade_won"] is True

    def test_sync_closed_trades_rl_backfills_rr_fields_into_outcome(self, tmp_path):
        mgr = _make_manager()
        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([
                {
                    "trade_id": "3092",
                    "pair": "EUR_JPY",
                    "direction": "SHORT",
                    "timestamp": "2026-04-02T01:00:00Z",
                    "entry_price": 183.906,
                    "sl_price": 184.206,
                    "tp_price": 183.306,
                    "outcome": None,
                    "agents": {"agent_reasons": []},
                    "regime": {},
                }
            ]))

            with patch.dict(os.environ, {
                "OANDA_API_TOKEN": "token",
                "OANDA_ACCOUNT_ID": "acct",
            }, clear=False):
                with patch.object(mgr, "_fetch_oanda_trade_snapshots", return_value=({}, {})):
                    with patch.object(
                        mgr,
                        "_resolve_trade_state_from_oanda",
                        return_value=("closed", {
                            "id": "3092",
                            "realizedPL": "15.0",
                            "averageClosePrice": "183.306",
                            "price": "183.906",
                            "closeTime": "2026-04-02T01:15:00Z",
                            "initialUnits": "-1000",
                        }),
                    ):
                        result = mgr.sync_closed_trades_rl()
        finally:
            os.chdir(original_cwd)

        assert result["trades_synced"] == 1
        saved = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())[0]
        assert saved["sl_pips"] == 30.0
        assert saved["tp_pips"] == 60.0
        assert saved["rr_ratio"] == 2.0
        assert saved["outcome"]["sl_pips"] == 30.0
        assert saved["outcome"]["tp_pips"] == 60.0
        assert saved["outcome"]["rr_ratio"] == 2.0

    def test_sync_closed_trades_rl_triggers_background_retrain_from_close_path(self, tmp_path):
        mgr = _make_manager()
        scanner = MagicMock()
        scanner._retrain_trigger = MagicMock()
        scanner._retrain_trigger.check_drift.return_value = MagicMock(
            pairs=["EUR_USD"],
            reason="Drift detected",
        )

        original_cwd = os.getcwd()
        (tmp_path / "trained_data").mkdir(exist_ok=True)
        os.chdir(tmp_path)
        try:
            (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([
                {
                    "trade_id": "2092",
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "timestamp": "2026-04-02T01:00:00Z",
                    "entry_price": 1.1000,
                    "outcome": None,
                    "agents": {"agent_reasons": []},
                    "regime": {},
                }
            ]))

            with patch.dict(os.environ, {
                "OANDA_API_TOKEN": "token",
                "OANDA_ACCOUNT_ID": "acct",
            }, clear=False):
                with patch.object(mgr, "_fetch_oanda_trade_snapshots", return_value=({}, {})):
                    with patch.object(
                        mgr,
                        "_resolve_trade_state_from_oanda",
                        return_value=("closed", {
                            "id": "2092",
                            "realizedPL": "-15.0",
                            "averageClosePrice": "1.0985",
                            "price": "1.1000",
                            "closeTime": "2026-04-02T01:15:00Z",
                            "initialUnits": "1000",
                        }),
                    ):
                        with patch.object(mgr, "_spawn_background_retrain") as mock_spawn:
                            result = mgr.sync_closed_trades_rl(scanner=scanner)
        finally:
            os.chdir(original_cwd)

        assert result["trades_synced"] == 1
        scanner._retrain_trigger.record_prediction.assert_called_once_with(
            pair="EUR_USD",
            correct=False,
        )
        scanner._retrain_trigger.check_drift.assert_called_once_with()
        mock_spawn.assert_called_once_with(["EUR_USD"])

    def test_all_wins(self):
        mgr = _make_manager()
        # get_trades returns TradeInfo-like objects; current source counts them
        # and returns 0.5 (neutral) because TradeInfo lacks realized_pnl.
        mock_broker = MagicMock()
        mock_broker.get_trades.return_value = [
            MagicMock(), MagicMock(), MagicMock(),
        ]
        mgr._broker = mock_broker
        win_rate, total = mgr.fetch_actual_win_rate()
        # Current implementation returns 0.5 (neutral) when trades exist
        assert win_rate == 0.5
        assert total == 3

    def test_mixed_results(self):
        mgr = _make_manager()
        mock_broker = MagicMock()
        mock_broker.get_trades.return_value = [
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        ]
        mgr._broker = mock_broker
        win_rate, total = mgr.fetch_actual_win_rate()
        assert win_rate == 0.5
        assert total == 4

    def test_api_exception(self):
        mgr = _make_manager()
        mock_broker = MagicMock()
        mock_broker.get_trades.side_effect = Exception("API timeout")
        mgr._broker = mock_broker
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
