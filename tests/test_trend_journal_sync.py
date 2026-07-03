"""Tests for the trend-lane -> RL journal sync gap (2026-07-02 audit).

Root cause: ``com.buddy.trend`` (scripts/run_oanda_trend.py -> run_oanda_trend_cycle)
places real OANDA practice orders but never calls ExecutionManager.execute_trade(),
so no entry is ever written to trade_journal_rl.json for trend-lane trades. Since
OutcomeBackfill only PATCHES existing journal entries (it looks up by trade_id and
skips anything not already present), the missing open-side write means outcomes can
never be recovered downstream either.

The trend lane is explicitly non-directional (price vs SMA, no 15-agent consensus),
so entries it creates must be tagged lane="trend" / rl_eligible=False and must NEVER
flow into ScannerAgentTeam.update_weights_from_outcome() — there are no agent verdicts
to reward or penalize.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.scanner.automation.trend_journal_sync import TrendJournalSync


def _open_fill(
    tx_id: str, trade_id: str, instrument: str, units: str, price: str, time: str,
) -> Dict[str, Any]:
    return {
        "id": tx_id,
        "type": "ORDER_FILL",
        "instrument": instrument,
        "units": units,
        "price": price,
        "time": time,
        "reason": "MARKET_ORDER",
        "tradeOpened": {"tradeID": trade_id, "units": units, "price": price},
    }


def _close_fill(
    tx_id: str, trade_id: str, realized_pl: str, time: str,
    reason: str = "STOP_LOSS_ORDER",
) -> Dict[str, Any]:
    return {
        "id": tx_id,
        "type": "ORDER_FILL",
        "time": time,
        "reason": reason,
        "tradesClosed": [{"tradeID": trade_id, "realizedPL": realized_pl}],
    }


def _make_sync(
    tmp_path: Path,
    pages: List[Tuple[List[Dict[str, Any]], Optional[str]]],
    journal: Optional[List[Dict[str, Any]]] = None,
    cursor_seed: Optional[str] = None,
) -> Tuple[TrendJournalSync, Path, Path]:
    journal_path = tmp_path / "trade_journal_rl.json"
    cursor_path = tmp_path / "trend_journal_sync_cursor.json"
    journal_path.write_text(json.dumps(journal if journal is not None else []))
    if cursor_seed is not None:
        cursor_path.write_text(json.dumps({"last_processed_tx_id": cursor_seed}))

    page_iter = iter(pages)

    def _fetcher(since_id: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            return next(page_iter)
        except StopIteration:
            return [], None

    sync = TrendJournalSync(
        journal_path=journal_path,
        cursor_path=cursor_path,
        transactions_fetcher=_fetcher,
    )
    return sync, journal_path, cursor_path


def test_creates_journal_entry_for_trend_lane_open(tmp_path: Path) -> None:
    """FAILING-TEST-FIRST: a trend-lane ORDER_FILL open must produce a journal
    entry — this is the exact gap that let 57 real closes go unsynced."""
    page = [
        _open_fill("1329", "1329", "USD_JPY", "78", "161.842",
                   "2026-06-25T13:05:17.241991175Z"),
    ]
    sync, journal_path, _ = _make_sync(tmp_path, [(page, None)])

    result = sync.run_once()

    assert result.opened == 1
    written = json.loads(journal_path.read_text())
    assert len(written) == 1
    entry = written[0]
    assert entry["trade_id"] == "1329"
    assert entry["pair"] == "USD_JPY"
    assert entry["lane"] == "trend"
    assert entry["rl_eligible"] is False
    assert entry["agents"]["agent_votes"] is None
    assert entry["outcome"] is None
    assert entry["entry_price"] == pytest.approx(161.842)


def test_patches_outcome_on_close_without_fabricating_agent_votes(tmp_path: Path) -> None:
    """Close-side sync must patch outcome/realized_pl, but must NOT invent
    agent_verdicts — the RL weight update path must never see this trade."""
    open_page = [
        _open_fill("1329", "1329", "USD_JPY", "78", "161.842",
                   "2026-06-25T13:05:17.241991175Z"),
    ]
    sync, journal_path, cursor_path = _make_sync(tmp_path, [(open_page, None)])
    sync.run_once()

    close_page = [
        _close_fill("1400", "1329", "-42.50", "2026-06-26T09:00:00.000000000Z"),
    ]
    sync2 = TrendJournalSync(
        journal_path=journal_path,
        cursor_path=cursor_path,
        transactions_fetcher=lambda since_id: (close_page, None) if since_id == "1329" else ([], None),
    )
    result = sync2.run_once()

    assert result.closed == 1
    written = json.loads(journal_path.read_text())
    entry = written[0]
    assert entry["outcome"] == "loss"
    assert entry["realized_pl"] == pytest.approx(-42.5)
    assert entry["lane"] == "trend"
    assert entry["rl_eligible"] is False


def test_never_calls_rl_weight_update_for_trend_lane(tmp_path: Path, monkeypatch) -> None:
    """Hard guard: TrendJournalSync must not import/call
    ScannerAgentTeam.update_weights_from_outcome under any code path — trend-lane
    trades have no agent verdicts and must never mutate agent_weights.json."""
    import src.scanner.agents as agents_pkg

    called = {"n": 0}
    original = agents_pkg.ScannerAgentTeam.update_weights_from_outcome

    def _spy(self, *a, **kw):
        called["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(agents_pkg.ScannerAgentTeam, "update_weights_from_outcome", _spy)

    pages = [
        _open_fill("1329", "1329", "USD_JPY", "78", "161.842", "2026-06-25T13:05:17Z"),
        _close_fill("1400", "1329", "88.0", "2026-06-26T09:00:00Z", reason="TAKE_PROFIT_ORDER"),
    ]
    sync, _, _ = _make_sync(tmp_path, [(pages, None)])
    sync.run_once()

    assert called["n"] == 0


def test_does_not_duplicate_entries_already_written_by_scanner(tmp_path: Path) -> None:
    """If a trade_id already exists in the journal (written by the real scanner
    ExecutionManager path), TrendJournalSync must not overwrite/duplicate it."""
    existing = [{
        "trade_id": "1329",
        "pair": "USD_JPY",
        "direction": "LONG",
        "lane": "scanner",
        "outcome": None,
    }]
    page = [_open_fill("1329", "1329", "USD_JPY", "78", "161.842", "2026-06-25T13:05:17Z")]
    sync, journal_path, _ = _make_sync(tmp_path, [(page, None)], journal=existing)

    result = sync.run_once()

    assert result.opened == 0
    written = json.loads(journal_path.read_text())
    assert len(written) == 1
    assert written[0]["lane"] == "scanner"  # untouched


def test_cursor_advances_and_is_idempotent(tmp_path: Path) -> None:
    page = [_open_fill("1329", "1329", "USD_JPY", "78", "161.842", "2026-06-25T13:05:17Z")]
    sync, journal_path, cursor_path = _make_sync(tmp_path, [(page, None)])

    result1 = sync.run_once()
    assert result1.opened == 1
    cursor_data = json.loads(cursor_path.read_text())
    assert cursor_data["last_processed_tx_id"] == "1329"

    # Second run with no new pages (fetcher exhausted) must not duplicate.
    sync2 = TrendJournalSync(
        journal_path=journal_path,
        cursor_path=cursor_path,
        transactions_fetcher=lambda since_id: ([], None),
    )
    result2 = sync2.run_once()
    assert result2.opened == 0
    written = json.loads(journal_path.read_text())
    assert len(written) == 1
