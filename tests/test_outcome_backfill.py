"""Tests for US-605: OutcomeBackfill catch-up pass."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.scanner.automation.outcome_backfill import OutcomeBackfill


def _journal_entry(trade_id: str, pair: str = "EUR_USD") -> Dict[str, Any]:
    return {
        "trade_id": trade_id,
        "pair": pair,
        "direction": "LONG",
        "entry_price": 1.1000,
        "sl_price": 1.0950,
        "tp_price": 1.1100,
    }


def _make_backfill(
    tmp_path: Path,
    journal: List[Dict[str, Any]],
    pages: List[Tuple[List[Dict[str, Any]], Optional[str]]],
    cursor_seed: Optional[str] = None,
) -> Tuple[OutcomeBackfill, Path, Path, List[str]]:
    journal_path = tmp_path / "trade_journal_rl.json"
    cursor_path = tmp_path / "last_tx_id.json"
    journal_path.write_text(json.dumps(journal))
    if cursor_seed is not None:
        cursor_path.write_text(json.dumps({"last_processed_tx_id": cursor_seed}))

    calls: List[str] = []
    page_iter = iter(pages)

    def _fetcher(since_id: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        calls.append(since_id)
        try:
            return next(page_iter)
        except StopIteration:
            return [], None

    bf = OutcomeBackfill(
        journal_path=journal_path,
        cursor_path=cursor_path,
        transactions_fetcher=_fetcher,
    )
    return bf, journal_path, cursor_path, calls


def test_matches_by_trade_id_and_patches_outcome(tmp_path: Path) -> None:
    journal = [_journal_entry("1234"), _journal_entry("5678", "USD_CAD")]
    tx_page = [
        {
            "id": "100",
            "type": "ORDER_FILL",
            "reason": "STOP_LOSS_ORDER",
            "time": "2026-04-24T18:00:00.000000000Z",
            "tradesClosed": [{"tradeID": "1234", "realizedPL": "-42.5"}],
        },
        {
            "id": "101",
            "type": "ORDER_FILL",
            "reason": "TAKE_PROFIT_ORDER",
            "time": "2026-04-24T19:00:00.000000000Z",
            "tradesClosed": [{"tradeID": "5678", "realizedPL": "88.0"}],
        },
    ]
    bf, journal_path, _, _ = _make_backfill(tmp_path, journal, [(tx_page, None)])

    result = bf.run_once()

    assert result.matched == 2
    assert result.unmatched_trade_ids == []
    assert result.error is None
    written = json.loads(journal_path.read_text())
    by_id = {e["trade_id"]: e for e in written}

    assert by_id["1234"]["outcome"] == "loss"
    assert by_id["1234"]["realized_pl"] == pytest.approx(-42.5)
    assert by_id["1234"]["close_time"] == "2026-04-24T18:00:00.000000000Z"
    assert by_id["1234"]["close_reason"] == "SL"

    assert by_id["5678"]["outcome"] == "win"
    assert by_id["5678"]["realized_pl"] == pytest.approx(88.0)
    assert by_id["5678"]["close_reason"] == "TP"


def test_handles_pagination(tmp_path: Path) -> None:
    journal = [_journal_entry("A1"), _journal_entry("B2"), _journal_entry("C3")]
    page1 = [
        {
            "id": "10",
            "type": "ORDER_FILL",
            "reason": "STOP_LOSS_ORDER",
            "time": "t1",
            "tradesClosed": [{"tradeID": "A1", "realizedPL": "-10"}],
        }
    ]
    page2 = [
        {
            "id": "20",
            "type": "ORDER_FILL",
            "reason": "TAKE_PROFIT_ORDER",
            "time": "t2",
            "tradesClosed": [{"tradeID": "B2", "realizedPL": "15"}],
        }
    ]
    page3 = [
        {
            "id": "30",
            "type": "ORDER_FILL",
            "reason": "MARKET_ORDER_TRADE_CLOSE",
            "time": "t3",
            "tradesClosed": [{"tradeID": "C3", "realizedPL": "0.25"}],
        }
    ]
    pages = [(page1, "10"), (page2, "20"), (page3, None)]
    bf, journal_path, cursor_path, calls = _make_backfill(tmp_path, journal, pages, cursor_seed="0")

    result = bf.run_once()

    assert result.matched == 3
    assert result.pages == 3
    # Pagination must pass the cursor from each page into the next fetch call.
    assert calls == ["0", "10", "20"]
    written = json.loads(journal_path.read_text())
    by_id = {e["trade_id"]: e for e in written}
    assert by_id["C3"]["close_reason"] == "manual"
    assert by_id["C3"]["outcome"] == "win"

    cursor = json.loads(cursor_path.read_text())
    assert cursor["last_processed_tx_id"] == "30"


def test_unknown_trade_id_handled_gracefully(tmp_path: Path) -> None:
    journal = [_journal_entry("KNOWN")]
    tx_page = [
        {
            "id": "500",
            "type": "ORDER_FILL",
            "reason": "STOP_LOSS_ORDER",
            "time": "ts",
            "tradesClosed": [
                {"tradeID": "UNKNOWN_1", "realizedPL": "-12"},
                {"tradeID": "KNOWN", "realizedPL": "-5"},
                {"tradeID": "UNKNOWN_2", "realizedPL": "-7"},
            ],
        }
    ]
    bf, journal_path, _, _ = _make_backfill(tmp_path, journal, [(tx_page, None)])

    result = bf.run_once()

    assert result.matched == 1
    assert sorted(result.unmatched_trade_ids) == ["UNKNOWN_1", "UNKNOWN_2"]
    assert result.error is None

    written = json.loads(journal_path.read_text())
    assert len(written) == 1
    assert written[0]["trade_id"] == "KNOWN"
    assert written[0]["outcome"] == "loss"


def test_persists_txid_cursor(tmp_path: Path) -> None:
    journal = [_journal_entry("T1")]
    tx_page = [
        {
            "id": "777",
            "type": "ORDER_FILL",
            "reason": "TAKE_PROFIT_ORDER",
            "time": "ts",
            "tradesClosed": [{"tradeID": "T1", "realizedPL": "100"}],
        },
        # Non-fill transaction should still advance the cursor.
        {"id": "800", "type": "DAILY_FINANCING"},
    ]
    bf, _, cursor_path, calls = _make_backfill(
        tmp_path, journal, [(tx_page, None)], cursor_seed="42"
    )

    result = bf.run_once()

    # Fetcher was seeded with persisted cursor.
    assert calls[0] == "42"
    assert result.last_tx_id == "800"

    cursor_blob = json.loads(cursor_path.read_text())
    assert cursor_blob["last_processed_tx_id"] == "800"
    assert "updated_at" in cursor_blob

    # Second run with no new transactions must not regress the cursor.
    bf2, _, _, _ = _make_backfill(
        tmp_path, journal, [([], None)], cursor_seed="800"
    )
    result2 = bf2.run_once()
    assert result2.last_tx_id == "800"
    assert result2.matched == 0
