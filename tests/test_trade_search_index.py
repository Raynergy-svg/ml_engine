"""Tests for src/tui/trade_search_index.py — FTS5 trade journal search."""
from __future__ import annotations

import json

import pytest

from src.tui.trade_search_index import (
    TradeSearchIndex,
    _entry_to_corpus,
    _outcome_label,
    _to_fts_query,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _mk_entry(**over: object) -> dict:
    base = {
        "trade_id": "100",
        "pair": "EUR_USD",
        "direction": "LONG",
        "broker": "oanda",
        "asset_class": "FX",
        "timestamp": "2026-04-15T10:00:00+00:00",
        "regime": {"volatility_regime": "NORMAL", "atr_pips": 12.0},
        "gates": {"gate_summary": "3/3"},
        "outcome": {"trade_won": True, "realized_pl": 150.0},
        "agents": {
            "agent_reasons": [
                {"name": "trend", "reason": "ADX above threshold", "reason_code": "trend_ok"},
                {"name": "risk_sentinel", "reason": "drawdown within budget", "reason_code": "risk_ok"},
            ],
        },
    }
    base.update(over)
    return base


@pytest.fixture
def trade_corpus() -> list[dict]:
    """A small, varied set of trades covering all searchable axes."""
    return [
        _mk_entry(trade_id="100", pair="EUR_USD", direction="LONG"),
        _mk_entry(
            trade_id="101",
            pair="EUR_AUD",
            direction="SHORT",
            timestamp="2026-04-15T11:00:00+00:00",
            outcome={"trade_won": False, "realized_pl": -82.0},
            regime={"volatility_regime": "LOW", "atr_pips": 4.0},
            agents={"agent_reasons": [{"name": "trend", "reason_code": "no_trend"}]},
        ),
        _mk_entry(
            trade_id="1220",
            pair="EUR_AUD",
            direction="LONG",
            timestamp="2026-04-15T15:00:00+00:00",
            outcome={"trade_won": False, "realized_pl": -260.0},
            regime={"volatility_regime": "LOW", "atr_pips": 5.5},
            agents={"agent_reasons": [{"name": "trend", "reason_code": "no_trend"}]},
        ),
        _mk_entry(
            trade_id="102",
            pair="USD_CAD",
            timestamp="2026-04-15T12:00:00+00:00",
            outcome=None,  # open trade
            agents={"agent_reasons": []},
        ),
    ]


# ── Outcome / corpus helpers ──────────────────────────────────────────────


class TestHelpers:

    def test_outcome_label_win(self):
        assert _outcome_label({"outcome": {"trade_won": True, "realized_pl": 1}}) == "win"

    def test_outcome_label_loss(self):
        assert _outcome_label({"outcome": {"trade_won": False, "realized_pl": -1}}) == "loss"

    def test_outcome_label_open(self):
        assert _outcome_label({}) == "open"
        assert _outcome_label({"outcome": None}) == "open"

    def test_corpus_lowercases(self):
        corpus = _entry_to_corpus(_mk_entry(pair="EUR_USD"))
        assert "eur_usd" in corpus
        assert "EUR_USD" not in corpus

    def test_corpus_includes_agent_text(self):
        corpus = _entry_to_corpus(_mk_entry())
        assert "trend" in corpus
        assert "drawdown" in corpus


class TestFTSQueryBuilder:

    def test_simple_query_becomes_prefix_and(self):
        assert _to_fts_query("eur loss") == "eur* AND loss*"

    def test_special_chars_stripped(self):
        # Lone quote/paren must not pass through to FTS5.
        assert '"' not in _to_fts_query('"eur"')
        assert "(" not in _to_fts_query("(eur)")

    def test_empty_query_returns_sentinel(self):
        # Sentinel ensures the FTS execute() path won't accidentally match all
        # rows for whitespace-only / pure-special input.
        assert _to_fts_query("()") == "zzz_no_match_zzz"


# ── Index behavior ────────────────────────────────────────────────────────


class TestIndex:

    def test_empty_index_returns_empty(self):
        idx = TradeSearchIndex()
        assert len(idx) == 0
        assert idx.search("eur") == []
        assert idx.search("") == []

    def test_build_from_entries(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        assert len(idx) == 4

    def test_empty_query_returns_newest_first(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("", limit=10)
        assert [e["trade_id"] for e in out] == ["1220", "102", "101", "100"]

    def test_empty_query_respects_limit(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        assert len(idx.search("", limit=2)) == 2

    def test_pair_query(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("eur_aud")
        assert {e["trade_id"] for e in out} == {"101", "1220"}

    def test_outcome_query(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        wins = idx.search("win")
        losses = idx.search("loss")
        assert {e["trade_id"] for e in wins} == {"100"}
        assert {e["trade_id"] for e in losses} == {"101", "1220"}

    def test_combined_query_is_AND(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("eur_aud loss")
        assert {e["trade_id"] for e in out} == {"101", "1220"}

    def test_trade_id_query(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("1220")
        assert [e["trade_id"] for e in out] == ["1220"]

    def test_regime_query(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("low")
        assert {e["trade_id"] for e in out} == {"101", "1220"}

    def test_agent_reason_query(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        out = idx.search("drawdown")
        # Only trade 100 has the 'drawdown within budget' reason text.
        assert [e["trade_id"] for e in out] == ["100"]

    def test_unknown_token_returns_empty(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        assert idx.search("zzzzz-no-such-token") == []

    def test_malformed_fts_falls_back_via_like(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        # Pure-special input would otherwise raise sqlite3.OperationalError.
        out = idx.search("()")
        assert out == []  # neither FTS nor LIKE match the sentinel

    def test_rebuild_replaces_contents(self, trade_corpus):
        idx = TradeSearchIndex.from_entries(trade_corpus)
        idx.build([_mk_entry(trade_id="999", pair="GBP_USD")])
        assert len(idx) == 1
        assert [e["trade_id"] for e in idx.search("gbp")] == ["999"]
        # Old entries no longer indexed:
        assert idx.search("1220") == []

    def test_skips_non_dict_entries(self):
        idx = TradeSearchIndex.from_entries([_mk_entry(), "garbage", None, 42])
        assert len(idx) == 1

    def test_context_manager_closes(self, trade_corpus):
        with TradeSearchIndex.from_entries(trade_corpus) as idx:
            assert len(idx) == 4
        assert len(idx) == 0


class TestIndexFromFile:

    def test_loads_valid_journal(self, tmp_path, trade_corpus):
        path = tmp_path / "journal.json"
        path.write_text(json.dumps(trade_corpus), encoding="utf-8")
        idx = TradeSearchIndex.from_journal_file(path)
        assert len(idx) == 4

    def test_missing_file_returns_empty_index(self, tmp_path):
        idx = TradeSearchIndex.from_journal_file(tmp_path / "nope.json")
        assert len(idx) == 0
        # And search is safe:
        assert idx.search("eur") == []

    def test_corrupt_json_returns_empty_index(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        idx = TradeSearchIndex.from_journal_file(path)
        assert len(idx) == 0

    def test_wrong_top_level_type_returns_empty(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text('{"trade_id": "1"}', encoding="utf-8")
        idx = TradeSearchIndex.from_journal_file(path)
        assert len(idx) == 0
