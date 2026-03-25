"""Phase 37 (US-236): Unit tests for TradeJournal core operations.

Tests log_trade, get_statistics, get_recent_trades, _load/_save,
clear_journal, TradeEntry dataclass, and JSON serialization roundtrip.
No OANDA dependency — all tests use temp files.
"""

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.utils.trade_journal import (
    TradeEntry,
    TradeJournal,
    LiveTradeInfo,
    AccountSummary,
    ClosedTradeInfo,
    format_journal_stats,
)


# ---------------------------------------------------------------------------
# Helper: create a trade entry with minimal required fields
# ---------------------------------------------------------------------------
def _make_trade_entry(
    trade_id="T001",
    instrument="EUR_USD",
    direction="long",
    status="open",
    pnl=None,
    pnl_pips=None,
    slippage_pips=0.5,
    **overrides,
):
    defaults = dict(
        trade_id=trade_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        instrument=instrument,
        direction=direction,
        entry_price=1.1050,
        expected_price=1.1050,
        units=100000,
        lots=1.0,
        stop_loss=1.1000,
        take_profit=1.1100,
        trailing_stop=0.0,
        atr=0.0015,
        tcn_probability=0.75,
        ridge_confidence=65.0,
        xgb_momentum=0.5,
        rf_drawdown_pips=10.0,
        slippage_pips=slippage_pips,
        fill_price=1.1050,
        status=status,
        pnl=pnl,
        pnl_pips=pnl_pips,
    )
    defaults.update(overrides)
    return TradeEntry(**defaults)


def _make_journal(trades=None):
    """Create a TradeJournal backed by a temp file, optionally pre-loaded."""
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "test_journal.json"

    if trades is not None:
        data = {
            "trades": [asdict(t) for t in trades],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return TradeJournal(path=path)


# ---------------------------------------------------------------------------
# Tests: TradeEntry dataclass
# ---------------------------------------------------------------------------
class TestTradeEntryDataclass:
    def test_default_status_is_open(self):
        te = _make_trade_entry()
        assert te.status == "open"

    def test_default_slippage_zero(self):
        te = _make_trade_entry(slippage_pips=0.0)
        assert te.slippage_pips == 0.0

    def test_asdict_roundtrip(self):
        te = _make_trade_entry()
        d = asdict(te)
        te2 = TradeEntry(**d)
        assert te2.trade_id == te.trade_id
        assert te2.instrument == te.instrument
        assert te2.direction == te.direction

    def test_json_serializable(self):
        te = _make_trade_entry()
        serialized = json.dumps(asdict(te))
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["trade_id"] == "T001"


# ---------------------------------------------------------------------------
# Tests: TradeJournal _load / _save
# ---------------------------------------------------------------------------
class TestJournalLoadSave:
    def test_empty_journal_on_missing_file(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "nonexistent.json"
        journal = TradeJournal(path=path)
        assert len(journal.trades) == 0

    def test_save_creates_file(self):
        journal = _make_journal()
        journal.trades = [_make_trade_entry()]
        journal._save()
        assert journal.path.exists()

    def test_save_load_roundtrip(self):
        journal = _make_journal()
        te = _make_trade_entry(trade_id="RT001")
        journal.trades = [te]
        journal._save()

        # Re-load
        journal2 = TradeJournal(path=journal.path)
        assert len(journal2.trades) == 1
        assert journal2.trades[0].trade_id == "RT001"

    def test_load_corrupted_json_recovers(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "bad.json"
        path.write_text("{ not valid json !!!")
        journal = TradeJournal(path=path)
        assert len(journal.trades) == 0


# ---------------------------------------------------------------------------
# Tests: log_trade
# ---------------------------------------------------------------------------
class TestLogTrade:
    def test_log_trade_adds_entry(self):
        journal = _make_journal()
        entry = journal.log_trade(
            trade_id="LT001",
            instrument="EUR_USD",
            direction="long",
            units=100000,
            lots=1.0,
            expected_price=1.1050,
            fill_price=1.1052,
            stop_loss=1.1000,
            take_profit=1.1100,
            trailing_stop=0.0,
            atr=0.0015,
            tcn_probability=0.72,
            ridge_confidence=60.0,
            xgb_momentum=0.5,
            rf_drawdown_pips=15.0,
        )
        assert len(journal.trades) == 1
        assert entry.trade_id == "LT001"
        assert entry.status == "open"

    def test_log_trade_calculates_slippage(self):
        journal = _make_journal()
        entry = journal.log_trade(
            trade_id="LT002",
            instrument="EUR_USD",
            direction="long",
            units=100000,
            lots=1.0,
            expected_price=1.1050,
            fill_price=1.1052,
            stop_loss=1.1000,
            take_profit=1.1100,
            trailing_stop=0.0,
            atr=0.0015,
            tcn_probability=0.72,
            ridge_confidence=60.0,
            xgb_momentum=0.5,
            rf_drawdown_pips=15.0,
        )
        # Slippage = (1.1052 - 1.1050) * 10000 = 2.0 pips
        assert abs(entry.slippage_pips - 2.0) < 0.01

    def test_log_trade_jpy_slippage(self):
        journal = _make_journal()
        entry = journal.log_trade(
            trade_id="LT003",
            instrument="USD_JPY",
            direction="long",
            units=100000,
            lots=1.0,
            expected_price=150.500,
            fill_price=150.520,
            stop_loss=150.000,
            take_profit=151.000,
            trailing_stop=0.0,
            atr=0.15,
            tcn_probability=0.72,
            ridge_confidence=60.0,
            xgb_momentum=0.5,
            rf_drawdown_pips=15.0,
        )
        # JPY: (150.520 - 150.500) * 100 = 2.0 pips
        assert abs(entry.slippage_pips - 2.0) < 0.01

    def test_log_trade_persists_to_disk(self):
        journal = _make_journal()
        journal.log_trade(
            trade_id="LT004",
            instrument="GBP_USD",
            direction="short",
            units=50000,
            lots=0.5,
            expected_price=1.2700,
            fill_price=1.2700,
            stop_loss=1.2750,
            take_profit=1.2650,
            trailing_stop=0.0,
            atr=0.0020,
            tcn_probability=0.68,
            ridge_confidence=55.0,
            xgb_momentum=-0.3,
            rf_drawdown_pips=20.0,
        )
        # Reload from disk
        journal2 = TradeJournal(path=journal.path)
        assert len(journal2.trades) == 1
        assert journal2.trades[0].trade_id == "LT004"


# ---------------------------------------------------------------------------
# Tests: get_statistics
# ---------------------------------------------------------------------------
class TestGetStatistics:
    def test_empty_journal_stats(self):
        journal = _make_journal()
        stats = journal.get_statistics()
        assert stats["total_trades"] == 0
        assert stats["open_trades"] == 0
        assert "message" in stats

    def test_all_open_trades_stats(self):
        trades = [_make_trade_entry(trade_id=f"T{i}", status="open") for i in range(3)]
        journal = _make_journal(trades=trades)
        stats = journal.get_statistics()
        assert stats["total_trades"] == 3
        assert stats["open_trades"] == 3
        assert stats["closed_trades"] == 0

    def test_mixed_win_loss_stats(self):
        trades = [
            _make_trade_entry(trade_id="W1", status="win", pnl=50.0, pnl_pips=25.0,
                              exit_reason="tp", instrument="EUR_USD"),
            _make_trade_entry(trade_id="W2", status="win", pnl=30.0, pnl_pips=15.0,
                              exit_reason="tp", instrument="EUR_USD"),
            _make_trade_entry(trade_id="L1", status="loss", pnl=-20.0, pnl_pips=-10.0,
                              exit_reason="sl", instrument="GBP_USD"),
            _make_trade_entry(trade_id="O1", status="open"),
        ]
        journal = _make_journal(trades=trades)
        stats = journal.get_statistics()

        assert stats["total_trades"] == 4
        assert stats["open_trades"] == 1
        assert stats["closed_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert abs(stats["win_rate"] - 2 / 3) < 0.01
        assert abs(stats["total_pnl"] - 60.0) < 0.01
        assert stats["profit_factor"] > 0

    def test_all_wins_profit_factor_infinite(self):
        trades = [
            _make_trade_entry(trade_id="W1", status="win", pnl=50.0, pnl_pips=25.0),
        ]
        journal = _make_journal(trades=trades)
        stats = journal.get_statistics()
        assert stats["profit_factor"] == float("inf")

    def test_exit_reasons_breakdown(self):
        trades = [
            _make_trade_entry(trade_id="T1", status="win", pnl=10.0, pnl_pips=5.0, exit_reason="tp"),
            _make_trade_entry(trade_id="T2", status="loss", pnl=-10.0, pnl_pips=-5.0, exit_reason="sl"),
            _make_trade_entry(trade_id="T3", status="win", pnl=15.0, pnl_pips=7.0, exit_reason="tp"),
        ]
        journal = _make_journal(trades=trades)
        stats = journal.get_statistics()
        assert stats["exit_reasons"]["tp"] == 2
        assert stats["exit_reasons"]["sl"] == 1

    def test_by_instrument_breakdown(self):
        trades = [
            _make_trade_entry(trade_id="T1", status="win", pnl=10.0, pnl_pips=5.0, instrument="EUR_USD"),
            _make_trade_entry(trade_id="T2", status="loss", pnl=-5.0, pnl_pips=-2.5, instrument="EUR_USD"),
            _make_trade_entry(trade_id="T3", status="win", pnl=20.0, pnl_pips=10.0, instrument="GBP_USD"),
        ]
        journal = _make_journal(trades=trades)
        stats = journal.get_statistics()
        assert "EUR_USD" in stats["by_instrument"]
        assert stats["by_instrument"]["EUR_USD"]["wins"] == 1
        assert stats["by_instrument"]["EUR_USD"]["losses"] == 1


# ---------------------------------------------------------------------------
# Tests: get_recent_trades
# ---------------------------------------------------------------------------
class TestGetRecentTrades:
    def test_empty_journal(self):
        journal = _make_journal()
        assert journal.get_recent_trades() == []

    def test_returns_n_most_recent(self):
        trades = [_make_trade_entry(trade_id=f"T{i}") for i in range(20)]
        journal = _make_journal(trades=trades)
        recent = journal.get_recent_trades(n=5)
        assert len(recent) == 5

    def test_returns_all_when_fewer_than_n(self):
        trades = [_make_trade_entry(trade_id=f"T{i}") for i in range(3)]
        journal = _make_journal(trades=trades)
        recent = journal.get_recent_trades(n=10)
        assert len(recent) == 3


# ---------------------------------------------------------------------------
# Tests: clear_journal
# ---------------------------------------------------------------------------
class TestClearJournal:
    def test_clear_removes_all_trades(self):
        trades = [_make_trade_entry(trade_id=f"T{i}") for i in range(5)]
        journal = _make_journal(trades=trades)
        assert len(journal.trades) == 5
        journal.clear_journal()
        assert len(journal.trades) == 0

    def test_clear_persists(self):
        trades = [_make_trade_entry()]
        journal = _make_journal(trades=trades)
        journal.clear_journal()
        journal2 = TradeJournal(path=journal.path)
        assert len(journal2.trades) == 0


# ---------------------------------------------------------------------------
# Tests: format_journal_stats
# ---------------------------------------------------------------------------
class TestFormatJournalStats:
    def test_format_no_closed_trades(self):
        stats = {"message": "No closed trades yet", "open_trades": 2, "total_trades": 2}
        output = format_journal_stats(stats)
        assert "No closed trades yet" in output

    def test_format_with_data(self):
        stats = {
            "total_trades": 10,
            "open_trades": 2,
            "closed_trades": 8,
            "wins": 5,
            "losses": 3,
            "win_rate": 0.625,
            "total_pnl": 150.0,
            "total_pips": 75.0,
            "avg_win": 50.0,
            "avg_loss": 25.0,
            "avg_win_pips": 25.0,
            "avg_loss_pips": 12.5,
            "profit_factor": 2.0,
            "avg_slippage_pips": 0.5,
            "exit_reasons": {"tp": 4, "sl": 3, "ts": 1},
            "by_instrument": {"EUR_USD": {"wins": 3, "losses": 2, "pnl": 100.0}},
        }
        output = format_journal_stats(stats)
        assert "62.5%" in output
        assert "$150.00" in output
        assert "EUR_USD" in output


# ---------------------------------------------------------------------------
# Tests: Dataclass helpers
# ---------------------------------------------------------------------------
class TestDataclassHelpers:
    def test_live_trade_info_defaults(self):
        lti = LiveTradeInfo(
            trade_id="LT1", instrument="EUR_USD", direction="long",
            units=100000, entry_price=1.1050, current_price=1.1060,
            unrealized_pnl=10.0,
        )
        assert lti.stop_loss is None
        assert lti.take_profit is None

    def test_account_summary_defaults(self):
        acct = AccountSummary(
            balance=100000.0, nav=100050.0, unrealized_pnl=50.0,
            realized_pnl=200.0, margin_used=1000.0, margin_available=99000.0,
            open_trade_count=1, open_position_count=1,
        )
        assert acct.currency == "USD"

    def test_closed_trade_info_defaults(self):
        cti = ClosedTradeInfo(
            trade_id="CT1", instrument="EUR_USD", direction="long",
            units=100000, entry_price=1.1050, exit_price=1.1100,
            realized_pnl=50.0, open_time="2026-01-01T00:00:00Z",
            close_time="2026-01-01T12:00:00Z",
        )
        assert cti.financing == 0.0
