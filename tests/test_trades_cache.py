"""Tier 2 T9: Two-tier trade journal cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tui.cache.trades_cache import TradesCache, TradeRow


def _seed_journal(path: Path, trades: list[dict]) -> None:
    path.write_text(json.dumps({"trades": trades}))


def test_initial_sync_builds_index(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [
        {"id": 1, "pair": "EUR_USD", "direction": "LONG", "pnl": 50.0,
         "opened_at": "2026-05-01T10:00:00", "closed_at": "2026-05-01T11:00:00",
         "outcome": "TP"},
        {"id": 2, "pair": "GBP_USD", "direction": "SHORT", "pnl": -23.0,
         "opened_at": "2026-05-01T12:00:00", "closed_at": "2026-05-01T13:00:00",
         "outcome": "SL"},
    ])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    rows = c.rows()
    assert len(rows) == 2
    assert any(r.pair == "EUR_USD" for r in rows)
    assert any(r.outcome == "SL" for r in rows)


def test_second_sync_reads_from_cache_unless_journal_grew(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    assert c.sync_count == 1
    c.sync()
    assert c.sync_count == 1


def test_journal_growth_triggers_incremental_sync(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    _seed_journal(j, [
        {"id": 1, "pair": "EUR_USD", "direction": "LONG",
         "pnl": 50.0, "opened_at": "x", "closed_at": "y", "outcome": "TP"},
        {"id": 2, "pair": "GBP_USD", "direction": "SHORT",
         "pnl": -23.0, "opened_at": "a", "closed_at": "b", "outcome": "SL"},
    ])
    c.sync()
    rows = c.rows()
    assert len(rows) == 2


def test_corrupt_cache_falls_back_to_full_rebuild(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    cache.write_text("not valid json")
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    assert len(c.rows()) == 1
