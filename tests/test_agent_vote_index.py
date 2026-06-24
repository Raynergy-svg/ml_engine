"""Tests for src/tui/agent_vote_index.py (Bucket D2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.tui.agent_vote_index import AgentVoteIndex, _extract_rows


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_entry(
    trade_id: str,
    *,
    regime: str = "NORMAL",
    outcome: str = "win",
    pnl: float = 100.0,
    closed_ts: str = "2026-05-01T12:00:00Z",
    reasons: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal trade-journal entry matching the production shape."""
    if reasons is None:
        reasons = [
            {"name": "trend", "score": 0.7, "passed": True, "weight": 1.0,
             "reason": "trend supportive", "reason_code": "trend_support"},
            {"name": "mean_reversion", "score": 0.4, "passed": False, "weight": 0.9,
             "reason": "RSI neutral chop", "reason_code": "mean_reversion_neutral"},
        ]
    if outcome == "open":
        outcome_blob = None
    else:
        outcome_blob = {
            "realized_pl": pnl,
            "trade_won": outcome == "win",
            "close_time": closed_ts,
        }
    entry = {
        "trade_id": trade_id,
        "timestamp": closed_ts,
        "regime": {"volatility_regime": regime},
        "agents": {"agent_reasons": reasons},
    }
    if outcome_blob is not None:
        entry["outcome"] = outcome_blob
    return entry


@pytest.fixture
def journal_path(tmp_path):
    return tmp_path / "trade_journal_rl.json"


@pytest.fixture
def index_db(tmp_path):
    return tmp_path / "agent_vote_index.db"


def _write_journal(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------

def test_sync_from_empty_journal(journal_path, index_db):
    """Empty list -> 0 rows, no exception."""
    _write_journal(journal_path, [])
    with AgentVoteIndex.open(db_path=index_db) as idx:
        result = idx.sync_from_journal(journal_path=journal_path)
        assert result == {"added": 0, "updated": 0, "skipped": 0}
        assert idx.query_votes() == []


def test_sync_from_missing_journal(tmp_path, index_db):
    """Missing journal -> empty result, no exception."""
    with AgentVoteIndex.open(db_path=index_db) as idx:
        result = idx.sync_from_journal(journal_path=tmp_path / "does_not_exist.json")
        assert result["added"] == 0


def test_sync_corrupt_journal_falls_back_gracefully(tmp_path, journal_path, index_db, caplog):
    """Malformed JSON -> warn + empty result, no exception."""
    journal_path.write_text("{not valid json", encoding="utf-8")
    with AgentVoteIndex.open(db_path=index_db) as idx:
        result = idx.sync_from_journal(journal_path=journal_path)
        assert result["added"] == 0


def test_sync_non_list_root_falls_back(journal_path, index_db):
    """Journal root is dict (wrong shape) -> warn + empty result."""
    journal_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with AgentVoteIndex.open(db_path=index_db) as idx:
        result = idx.sync_from_journal(journal_path=journal_path)
        assert result["added"] == 0


def test_sync_upsert_and_watermark(journal_path, index_db):
    """Initial sync ingests all rows; re-running same journal is a no-op via watermark."""
    entries = [
        _make_entry("1000", outcome="win", pnl=50.0),
        _make_entry("1001", outcome="loss", pnl=-75.0),
    ]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        r1 = idx.sync_from_journal(journal_path=journal_path)
        # 2 trades × 2 reasons each = 4 rows
        assert r1["added"] == 4

        # Re-syncing the same journal should advance watermark only (no new rows).
        r2 = idx.sync_from_journal(journal_path=journal_path)
        assert r2["added"] == 0

        # Append a new trade -> only its rows are added.
        entries.append(_make_entry("1002", outcome="loss", pnl=-30.0))
        _write_journal(journal_path, entries)
        r3 = idx.sync_from_journal(journal_path=journal_path)
        assert r3["added"] == 2


def test_sync_watermark_mismatch_triggers_rebuild(journal_path, index_db):
    """If the trade at the watermark anchor changes, full rebuild kicks in."""
    entries = [_make_entry("orig_1"), _make_entry("orig_2")]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        assert len(idx.query_votes()) == 4  # 2 trades × 2 reasons

        # Rewrite the journal with DIFFERENT trade_ids — watermark anchor no longer matches.
        new_entries = [_make_entry("new_1"), _make_entry("new_2"), _make_entry("new_3")]
        _write_journal(journal_path, new_entries)

        idx.sync_from_journal(journal_path=journal_path)
        rows = idx.query_votes(limit=100)
        names = {r["trade_id"] for r in rows}
        assert names == {"new_1", "new_2", "new_3"}, f"got {names}"


# ---------------------------------------------------------------------------
# Query filters
# ---------------------------------------------------------------------------

def test_query_filters_compose(journal_path, index_db):
    """Multiple filters AND together; result respects all."""
    base_ts = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    entries = [
        # Trade A: LOW regime, loss, MR failed
        _make_entry("A", regime="LOW", outcome="loss", pnl=-100,
                    closed_ts=base_ts.isoformat()),
        # Trade B: NORMAL regime, win, MR passed
        _make_entry("B", regime="NORMAL", outcome="win", pnl=50,
                    closed_ts=(base_ts + timedelta(hours=1)).isoformat(),
                    reasons=[
                        {"name": "trend", "score": 0.7, "passed": True, "weight": 1.0,
                         "reason": "trend supportive", "reason_code": "trend_support"},
                        {"name": "mean_reversion", "score": 0.7, "passed": True, "weight": 0.9,
                         "reason": "RSI supportive", "reason_code": "mean_reversion_support"},
                    ]),
        # Trade C: LOW regime, loss, MR failed (target row)
        _make_entry("C", regime="LOW", outcome="loss", pnl=-200,
                    closed_ts=(base_ts + timedelta(hours=2)).isoformat()),
    ]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        rows = idx.query_votes(
            agent_name="mean_reversion",
            passed=0,
            outcome="loss",
            regime="LOW",
        )
        trade_ids = sorted(r["trade_id"] for r in rows)
        assert trade_ids == ["A", "C"]


def test_query_since_filter(journal_path, index_db):
    """closed_ts_iso >= since filters out older rows."""
    old_ts = "2026-04-01T12:00:00Z"
    new_ts = "2026-05-10T12:00:00Z"
    entries = [
        _make_entry("old", closed_ts=old_ts),
        _make_entry("new", closed_ts=new_ts),
    ]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        rows = idx.query_votes(since="2026-05-01T00:00:00Z")
        trade_ids = {r["trade_id"] for r in rows}
        assert trade_ids == {"new"}


def test_query_fts_match(journal_path, index_db):
    """FTS5 MATCH against reason_text returns matching rows."""
    entries = [
        _make_entry("X", reasons=[
            {"name": "execution_quality", "score": 0.3, "passed": False, "weight": 1.0,
             "reason": "spread too wide for drawdown profile", "reason_code": "execution_block"}
        ]),
        _make_entry("Y", reasons=[
            {"name": "execution_quality", "score": 0.8, "passed": True, "weight": 1.0,
             "reason": "execution solid", "reason_code": "execution_ok"}
        ]),
    ]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        rows = idx.query_votes(reason_fts="drawdown")
        assert len(rows) == 1
        assert rows[0]["trade_id"] == "X"


def test_query_fts_malformed_falls_back_to_like(journal_path, index_db):
    """A malformed FTS expression falls back to a LIKE search; never raises."""
    entries = [_make_entry("Z", reasons=[
        {"name": "trend", "score": 0.7, "passed": True, "weight": 1.0,
         "reason": "trend in chop territory", "reason_code": "trend_chop"}
    ])]
    _write_journal(journal_path, entries)

    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        # ":" and "(" alone would be invalid FTS5 operators; sanitizer drops them.
        rows = idx.query_votes(reason_fts="chop:")
        assert len(rows) == 1


def test_query_passed_filter_distinct_yes_no(journal_path, index_db):
    """passed=1 returns only YES votes; passed=0 returns only NO votes."""
    _write_journal(journal_path, [_make_entry("1")])
    with AgentVoteIndex.open(db_path=index_db) as idx:
        idx.sync_from_journal(journal_path=journal_path)
        yes = idx.query_votes(passed=1)
        no = idx.query_votes(passed=0)
        assert {r["agent_name"] for r in yes} == {"trend"}
        assert {r["agent_name"] for r in no} == {"mean_reversion"}


# ---------------------------------------------------------------------------
# _extract_rows unit tests
# ---------------------------------------------------------------------------

def test_extract_rows_handles_missing_outcome():
    """Open trades (no outcome dict) get outcome='open' and pnl=None."""
    entry = _make_entry("open_trade", outcome="open")
    rows = _extract_rows(entry)
    assert len(rows) == 2
    for row in rows:
        assert row[7] == "open"   # outcome column
        assert row[8] is None     # pnl_usd column


def test_extract_rows_skips_non_dict_reasons():
    """Garbage entries in agent_reasons list are tolerated."""
    entry = _make_entry("mixed", reasons=[
        "garbage",
        {"name": "trend", "score": 0.5, "passed": True, "weight": 1.0, "reason": "ok"},
        None,
    ])
    rows = _extract_rows(entry)
    assert len(rows) == 1
    assert rows[0][1] == "trend"


def test_extract_rows_sub_dollar_non_win_is_loss():
    """Unified convention (2026-06-24): a sub-$1 non-win is 'loss', not
    'breakeven' — agrees with trade_search_index + the trades modal."""
    entry = _make_entry("be", outcome="loss", pnl=0.5)
    # trade_won is False and |pnl|<1 -> 'loss' under the unified convention.
    entry["outcome"]["trade_won"] = False
    rows = _extract_rows(entry)
    assert all(r[7] == "loss" for r in rows)
