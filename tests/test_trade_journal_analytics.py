"""Phase 37 (US-237): Unit tests for TradeJournal analytics.

Tests SlippageAnalyzer (analyze_slippage, _std_dev, _get_lot_bucket,
_generate_recommendations) and get_kelly_adjusted_stats.
Edge cases: empty journal, single trade, all wins, all losses.
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
    SlippageAnalyzer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_trade(
    trade_id="T001",
    instrument="EUR_USD",
    direction="long",
    lots=1.0,
    slippage_pips=0.5,
    status="win",
    pnl=50.0,
    pnl_pips=25.0,
    exit_reason="tp",
    **overrides,
):
    defaults = dict(
        trade_id=trade_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        instrument=instrument,
        direction=direction,
        entry_price=1.1050,
        expected_price=1.1050,
        units=int(lots * 100000),
        lots=lots,
        stop_loss=1.1000,
        take_profit=1.1100,
        trailing_stop=0.0,
        atr=0.0015,
        tcn_probability=0.72,
        ridge_confidence=60.0,
        xgb_momentum=0.5,
        rf_drawdown_pips=15.0,
        slippage_pips=slippage_pips,
        fill_price=1.1050,
        status=status,
        pnl=pnl,
        pnl_pips=pnl_pips,
        exit_reason=exit_reason,
    )
    defaults.update(overrides)
    return TradeEntry(**defaults)


def _make_journal_with_trades(trades):
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "test_journal.json"
    data = {
        "trades": [asdict(t) for t in trades],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return TradeJournal(path=path)


# ---------------------------------------------------------------------------
# Tests: SlippageAnalyzer._std_dev
# ---------------------------------------------------------------------------
class TestStdDev:
    def test_single_value_returns_zero(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._std_dev([5.0]) == 0.0

    def test_empty_returns_zero(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._std_dev([]) == 0.0

    def test_identical_values_zero_std(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._std_dev([3.0, 3.0, 3.0]) == 0.0

    def test_known_std_dev(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        # [2, 4, 4, 4, 5, 5, 7, 9] — population std dev ≈ 2.0
        result = analyzer._std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(result - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: SlippageAnalyzer._get_lot_bucket
# ---------------------------------------------------------------------------
class TestGetLotBucket:
    def test_micro_lot(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._get_lot_bucket(0.1) == "0-1 lots"

    def test_standard_lot(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._get_lot_bucket(1.0) == "0-1 lots"

    def test_medium_lot(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._get_lot_bucket(2.5) == "1-3 lots"

    def test_large_lot(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._get_lot_bucket(7.0) == "5-10 lots"

    def test_very_large_lot(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        assert analyzer._get_lot_bucket(15.0) == "10+ lots"


# ---------------------------------------------------------------------------
# Tests: SlippageAnalyzer.analyze_slippage
# ---------------------------------------------------------------------------
class TestAnalyzeSlippage:
    def test_empty_journal(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "empty.json"
        journal = TradeJournal(path=path)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()
        assert "message" in result

    def test_single_trade_analysis(self):
        trades = [_make_trade(slippage_pips=1.5)]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert result["summary"]["total_trades"] == 1
        assert abs(result["summary"]["avg_slippage_pips"] - 1.5) < 0.01
        assert result["summary"]["max_slippage_pips"] == 1.5

    def test_multiple_trades_stats(self):
        trades = [
            _make_trade(trade_id="T1", slippage_pips=1.0, lots=0.5),
            _make_trade(trade_id="T2", slippage_pips=2.0, lots=1.0),
            _make_trade(trade_id="T3", slippage_pips=3.0, lots=2.0),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert result["summary"]["total_trades"] == 3
        assert abs(result["summary"]["avg_slippage_pips"] - 2.0) < 0.01
        assert result["summary"]["max_slippage_pips"] == 3.0
        assert result["summary"]["min_slippage_pips"] == 1.0

    def test_by_lot_size_breakdown(self):
        trades = [
            _make_trade(trade_id="T1", slippage_pips=0.5, lots=0.5),
            _make_trade(trade_id="T2", slippage_pips=1.5, lots=2.0),
            _make_trade(trade_id="T3", slippage_pips=3.0, lots=12.0),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert "0-1 lots" in result["by_lot_size"]
        assert "1-3 lots" in result["by_lot_size"]
        assert "10+ lots" in result["by_lot_size"]

    def test_by_instrument_breakdown(self):
        trades = [
            _make_trade(trade_id="T1", instrument="EUR_USD", slippage_pips=1.0),
            _make_trade(trade_id="T2", instrument="GBP_USD", slippage_pips=2.0),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert "EUR_USD" in result["by_instrument"]
        assert "GBP_USD" in result["by_instrument"]

    def test_cost_impact_calculated(self):
        trades = [_make_trade(slippage_pips=1.0, lots=1.0)]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert "cost_impact" in result
        assert result["cost_impact"]["total_slippage_cost_usd"] > 0
        assert result["cost_impact"]["projected_annual_cost_usd"] > 0

    def test_recommendations_present(self):
        trades = [_make_trade(slippage_pips=1.0)]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.analyze_slippage()

        assert "recommendations" in result
        assert len(result["recommendations"]) >= 1


# ---------------------------------------------------------------------------
# Tests: _generate_recommendations
# ---------------------------------------------------------------------------
class TestGenerateRecommendations:
    def test_high_slippage_warning(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        recs = analyzer._generate_recommendations(
            avg_slippage=4.0, by_lots={}, by_hour={},
        )
        assert any("HIGH SLIPPAGE" in r for r in recs)

    def test_moderate_slippage_warning(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        recs = analyzer._generate_recommendations(
            avg_slippage=2.5, by_lots={}, by_hour={},
        )
        assert any("MODERATE SLIPPAGE" in r for r in recs)

    def test_good_slippage(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        recs = analyzer._generate_recommendations(
            avg_slippage=0.5, by_lots={}, by_hour={},
        )
        assert any("GOOD SLIPPAGE" in r for r in recs)

    def test_large_lot_split_recommendation(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        recs = analyzer._generate_recommendations(
            avg_slippage=1.0,
            by_lots={"10+ lots": [3.0, 4.0, 5.0]},  # Avg 4.0, much higher than 1.0
            by_hour={},
        )
        assert any("SPLIT ORDERS" in r for r in recs)

    def test_best_hour_recommendation(self):
        journal = _make_journal_with_trades([_make_trade()])
        analyzer = SlippageAnalyzer(journal)
        recs = analyzer._generate_recommendations(
            avg_slippage=1.0,
            by_lots={},
            by_hour={8: [0.5, 0.3], 14: [2.0, 3.0]},
        )
        assert any("BEST HOUR" in r for r in recs)


# ---------------------------------------------------------------------------
# Tests: get_kelly_adjusted_stats
# ---------------------------------------------------------------------------
class TestKellyAdjustedStats:
    def test_empty_journal_returns_message(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "empty.json"
        journal = TradeJournal(path=path)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.get_kelly_adjusted_stats()
        assert "message" in result

    def test_kelly_with_trades(self):
        trades = [
            _make_trade(trade_id="W1", status="win", pnl=50.0, pnl_pips=25.0, slippage_pips=0.5),
            _make_trade(trade_id="W2", status="win", pnl=30.0, pnl_pips=15.0, slippage_pips=0.3),
            _make_trade(trade_id="L1", status="loss", pnl=-20.0, pnl_pips=-10.0, slippage_pips=0.4),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.get_kelly_adjusted_stats()

        assert "win_rate" in result
        assert "raw_stats" in result
        assert "slippage_adjusted" in result
        assert "impact" in result
        assert result["win_rate"] > 0
        assert result["raw_stats"]["kelly_fraction"] >= 0
        assert result["slippage_adjusted"]["kelly_fraction"] >= 0

    def test_kelly_slippage_reduces_edge(self):
        """Higher slippage should reduce Kelly fraction."""
        trades = [
            _make_trade(trade_id="W1", status="win", pnl=100.0, pnl_pips=50.0, slippage_pips=5.0),
            _make_trade(trade_id="W2", status="win", pnl=80.0, pnl_pips=40.0, slippage_pips=5.0),
            _make_trade(trade_id="L1", status="loss", pnl=-40.0, pnl_pips=-20.0, slippage_pips=5.0),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.get_kelly_adjusted_stats()

        # Slippage-adjusted Kelly should be <= raw Kelly
        assert result["slippage_adjusted"]["kelly_fraction"] <= result["raw_stats"]["kelly_fraction"]

    def test_kelly_impact_message(self):
        trades = [
            _make_trade(trade_id="W1", status="win", pnl=50.0, pnl_pips=25.0, slippage_pips=1.0),
            _make_trade(trade_id="L1", status="loss", pnl=-20.0, pnl_pips=-10.0, slippage_pips=1.0),
        ]
        journal = _make_journal_with_trades(trades)
        analyzer = SlippageAnalyzer(journal)
        result = analyzer.get_kelly_adjusted_stats()

        assert "message" in result["impact"]
        assert "Slippage reduces" in result["impact"]["message"]
