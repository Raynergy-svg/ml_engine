"""Trend-lane journal ENRICHMENT tests (P0, docs/training-architecture-audit-2026-07-03.md).

Only 18/205 journal records carried learning-usable context; trend records are
nulls-by-design for agents/gates but CAN cheaply carry market context available
at sync time. These tests pin down:

  * open-side enrichment from the open ORDER_FILL itself (top-of-book bid/ask/
    spread, halfSpreadCost, accountBalance, requestedUnits) — None when absent,
    never fabricated;
  * atr_stop_distance joined from trained_data/oanda/risk_state.json (the
    entry-anchored r_distance the trend lane persists) — None when the file or
    trade is missing;
  * close-side enrichment (exit_price, close spread, financing, raw reason);
  * tradeReduced partial closes: accumulated idempotently by fill tx id, with
    realized_pl_total = partials + final fill and outcome derived from TOTAL;
  * rl_eligible stays False; agent votes stay null (never fabricated);
  * corrupt / wrong-shape journal is never clobbered (financial artifact).

No mocks (repo hard rule): real TrendJournalSync, real disk via tmp_path, and a
plain data-injection fetcher function (same convention as
tests/test_trend_journal_sync.py — the constructor takes a fetcher callable by
design).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.scanner.automation.trend_journal_sync import TrendJournalSync


# --------------------------------------------------------------------------- #
# Realistic OANDA transaction builders (field names match a real practice
# account's transactions.jsonl — see trained_data/oanda/transactions.jsonl).
# --------------------------------------------------------------------------- #

def _full_price(bid: str, ask: str) -> Dict[str, Any]:
    return {
        "bids": [{"liquidity": "500000", "price": bid}],
        "asks": [{"liquidity": "500000", "price": ask}],
        "closeoutBid": bid,
        "closeoutAsk": ask,
    }


def _open_fill(
    tx_id: str,
    trade_id: str,
    instrument: str = "USD_JPY",
    units: str = "78",
    price: str = "161.842",
    time: str = "2026-06-25T13:05:17.241991175Z",
    *,
    rich: bool = True,
) -> Dict[str, Any]:
    tx: Dict[str, Any] = {
        "id": tx_id,
        "type": "ORDER_FILL",
        "instrument": instrument,
        "units": units,
        "price": price,
        "time": time,
        "reason": "MARKET_ORDER",
        "tradeOpened": {"tradeID": trade_id, "units": units, "price": price},
    }
    if rich:
        tx.update({
            "fullPrice": _full_price("161.830", "161.845"),
            "halfSpreadCost": "0.3651",
            "accountBalance": "100409.8508",
            "requestedUnits": "80",
        })
    return tx


def _close_fill(
    tx_id: str,
    trade_id: str,
    realized_pl: str = "-42.50",
    time: str = "2026-06-26T09:00:00.000000000Z",
    reason: str = "STOP_LOSS_ORDER",
    *,
    rich: bool = True,
) -> Dict[str, Any]:
    tc: Dict[str, Any] = {"tradeID": trade_id, "realizedPL": realized_pl}
    tx: Dict[str, Any] = {
        "id": tx_id,
        "type": "ORDER_FILL",
        "time": time,
        "reason": reason,
        "tradesClosed": [tc],
    }
    if rich:
        tc.update({
            "price": "161.310",
            "halfSpreadCost": "0.2990",
            "financing": "-1.2345",
        })
        tx["fullPrice"] = _full_price("161.302", "161.318")
    return tx


def _reduce_fill(
    tx_id: str,
    trade_id: str,
    realized_pl: str,
    time: str = "2026-06-25T18:00:00.000000000Z",
) -> Dict[str, Any]:
    return {
        "id": tx_id,
        "type": "ORDER_FILL",
        "time": time,
        "reason": "MARKET_ORDER",
        "tradeReduced": {"tradeID": trade_id, "realizedPL": realized_pl},
    }


def _make_sync(
    tmp_path: Path,
    pages: List[Tuple[List[Dict[str, Any]], Optional[str]]],
    journal: Optional[List[Dict[str, Any]]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
) -> Tuple[TrendJournalSync, Path, Path]:
    journal_path = tmp_path / "trade_journal_rl.json"
    cursor_path = tmp_path / "cursor.json"
    risk_state_path = tmp_path / "risk_state.json"
    journal_path.write_text(json.dumps(journal if journal is not None else []))
    if risk_state is not None:
        risk_state_path.write_text(json.dumps(risk_state))

    page_iter = iter(pages)

    def _fetcher(since_id: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            return next(page_iter)
        except StopIteration:
            return [], None

    sync = TrendJournalSync(
        journal_path=journal_path,
        cursor_path=cursor_path,
        risk_state_path=risk_state_path,
        transactions_fetcher=_fetcher,
    )
    return sync, journal_path, cursor_path


def _reopen(
    journal_path: Path,
    cursor_path: Path,
    pages: List[Tuple[List[Dict[str, Any]], Optional[str]]],
    risk_state_path: Optional[Path] = None,
) -> TrendJournalSync:
    page_iter = iter(pages)

    def _fetcher(since_id: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            return next(page_iter)
        except StopIteration:
            return [], None

    return TrendJournalSync(
        journal_path=journal_path,
        cursor_path=cursor_path,
        risk_state_path=risk_state_path or (journal_path.parent / "risk_state.json"),
        transactions_fetcher=_fetcher,
    )


# --------------------------------------------------------------------------- #
# Open-side enrichment
# --------------------------------------------------------------------------- #

def test_open_entry_carries_market_context_from_the_fill(tmp_path: Path) -> None:
    sync, journal_path, _ = _make_sync(tmp_path, [([_open_fill("1329", "1329")], None)])
    result = sync.run_once()

    assert result.opened == 1 and result.error is None
    entry = json.loads(journal_path.read_text())[0]
    assert entry["entry_bid"] == pytest.approx(161.830)
    assert entry["entry_ask"] == pytest.approx(161.845)
    assert entry["entry_spread"] == pytest.approx(0.015)
    assert entry["entry_half_spread_cost"] == pytest.approx(0.3651)
    assert entry["account_balance"] == pytest.approx(100409.8508)
    assert entry["requested_units"] == pytest.approx(80.0)
    # Baseline fields the audit asked to confirm are present.
    assert entry["units"] == pytest.approx(78.0)
    assert entry["entry_price"] == pytest.approx(161.842)


def test_enrichment_is_null_when_fill_lacks_fields_never_fabricated(tmp_path: Path) -> None:
    sync, journal_path, _ = _make_sync(
        tmp_path, [([_open_fill("1329", "1329", rich=False)], None)],
    )
    result = sync.run_once()

    assert result.opened == 1 and result.error is None
    entry = json.loads(journal_path.read_text())[0]
    for f in ("entry_bid", "entry_ask", "entry_spread", "entry_half_spread_cost",
              "account_balance", "requested_units", "atr_stop_distance"):
        assert entry[f] is None, f"{f} must be None when unsourced, got {entry[f]!r}"


def test_regime_is_dict_shaped_with_honestly_null_classification(tmp_path: Path) -> None:
    """Consumers do `entry.get("regime", {}).get("volatility_regime")`
    (execution.py sync_closed_trades_rl outcome_data) — a bare None regime
    raises AttributeError there. Dict shape with a null classification is both
    honest (no classifier ran) and consumer-safe."""
    sync, journal_path, _ = _make_sync(tmp_path, [([_open_fill("1329", "1329")], None)])
    sync.run_once()

    entry = json.loads(journal_path.read_text())[0]
    assert isinstance(entry["regime"], dict)
    assert entry["regime"]["volatility_regime"] is None
    # The consumer access pattern must not raise:
    assert entry.get("regime", {}).get("volatility_regime") is None


def test_atr_stop_distance_joined_from_real_risk_state_file(tmp_path: Path) -> None:
    risk_state = {
        "1329": {"instrument": "USD_JPY", "entry_price": 161.842, "r_distance": 1.734},
    }
    sync, journal_path, _ = _make_sync(
        tmp_path, [([_open_fill("1329", "1329")], None)], risk_state=risk_state,
    )
    sync.run_once()

    entry = json.loads(journal_path.read_text())[0]
    assert entry["atr_stop_distance"] == pytest.approx(1.734)


def test_atr_stop_distance_null_when_risk_state_missing(tmp_path: Path) -> None:
    sync, journal_path, _ = _make_sync(tmp_path, [([_open_fill("1329", "1329")], None)])
    sync.run_once()
    assert json.loads(journal_path.read_text())[0]["atr_stop_distance"] is None


def test_rl_contract_unchanged_on_enriched_entries(tmp_path: Path) -> None:
    """Enrichment must not loosen the RL contract: rl_eligible stays False,
    agent votes stay null, confidence stays null (never fabricated)."""
    pages = [([_open_fill("1329", "1329"), _close_fill("1400", "1329")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    sync.run_once()

    entry = json.loads(journal_path.read_text())[0]
    assert entry["rl_eligible"] is False
    assert entry["lane"] == "trend"
    assert entry["agents"]["agent_votes"] is None
    assert entry["confidence"] is None
    assert entry["gates"] is None


# --------------------------------------------------------------------------- #
# Close-side enrichment
# --------------------------------------------------------------------------- #

def test_close_patch_carries_spread_financing_and_raw_reason(tmp_path: Path) -> None:
    pages = [([_open_fill("1329", "1329"), _close_fill("1400", "1329")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    result = sync.run_once()

    assert result.closed == 1
    entry = json.loads(journal_path.read_text())[0]
    assert entry["outcome"] == "loss"
    assert entry["realized_pl"] == pytest.approx(-42.5)
    assert entry["realized_pl_total"] == pytest.approx(-42.5)
    assert entry["exit_price"] == pytest.approx(161.310)
    assert entry["close_bid"] == pytest.approx(161.302)
    assert entry["close_ask"] == pytest.approx(161.318)
    assert entry["close_spread"] == pytest.approx(0.016)
    assert entry["close_half_spread_cost"] == pytest.approx(0.2990)
    assert entry["financing"] == pytest.approx(-1.2345)
    assert entry["close_reason"] == "SL"
    assert entry["close_reason_raw"] == "STOP_LOSS_ORDER"


def test_close_enrichment_null_when_close_fill_is_sparse(tmp_path: Path) -> None:
    pages = [([
        _open_fill("1329", "1329"),
        _close_fill("1400", "1329", rich=False),
    ], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    sync.run_once()

    entry = json.loads(journal_path.read_text())[0]
    for f in ("exit_price", "close_bid", "close_ask", "close_spread",
              "close_half_spread_cost", "financing"):
        assert entry[f] is None, f"{f} must be None when unsourced, got {entry[f]!r}"
    # Core closure schema unchanged.
    assert entry["outcome"] == "loss"
    assert entry["realized_pl"] == pytest.approx(-42.5)


def test_close_patches_entries_created_before_enrichment_shipped(tmp_path: Path) -> None:
    """179 pre-enrichment trend entries exist on disk (regime=None, no
    enrichment keys). A close arriving now must patch them without error."""
    old_schema_entry = {
        "trade_id": "1329", "lane": "trend", "rl_eligible": False,
        "timestamp": "2026-06-25T13:05:17Z", "pair": "USD_JPY",
        "direction": "LONG", "entry_price": 161.842, "units": 78.0,
        "confidence": None, "gates": None,
        "agents": {"agent_votes": None, "note": "trend lane"},
        "regime": None, "outcome": None, "source": "trend_journal_sync",
    }
    pages = [([_close_fill("1400", "1329")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages, journal=[old_schema_entry])
    result = sync.run_once()

    assert result.closed == 1 and result.error is None
    entry = json.loads(journal_path.read_text())[0]
    assert entry["outcome"] == "loss"
    assert entry["exit_price"] == pytest.approx(161.310)


# --------------------------------------------------------------------------- #
# Partial closes (tradeReduced)
# --------------------------------------------------------------------------- #

def test_partial_closes_accumulate_into_realized_pl_total(tmp_path: Path) -> None:
    """Trend-lane rebalances reduce positions partially; those realizedPL
    amounts never appear in the final tradesClosed fill. outcome must reflect
    the TOTAL: partial +100.0 then final -40.0 => win, not loss."""
    pages = [([
        _open_fill("1329", "1329"),
        _reduce_fill("1350", "1329", "100.0"),
        _close_fill("1400", "1329", realized_pl="-40.0"),
    ], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    result = sync.run_once()

    assert result.partials == 1 and result.closed == 1
    entry = json.loads(journal_path.read_text())[0]
    assert entry["partial_realized_pl"] == pytest.approx(100.0)
    assert entry["realized_pl"] == pytest.approx(-40.0)  # final fill only (schema compat)
    assert entry["realized_pl_total"] == pytest.approx(60.0)
    assert entry["outcome"] == "win"
    assert entry["partial_close_tx_ids"] == ["1350"]


def test_partial_close_accumulation_is_idempotent_by_tx_id(tmp_path: Path) -> None:
    """Re-fetching the same transaction window (e.g. after a cursor-write
    failure) must never double-count a partial close."""
    open_and_partial = [
        _open_fill("1329", "1329"),
        _reduce_fill("1350", "1329", "100.0"),
    ]
    sync, journal_path, cursor_path = _make_sync(tmp_path, [(open_and_partial, None)])
    sync.run_once()

    # Simulate cursor rollback -> same window re-fetched by a fresh instance.
    cursor_path.write_text(json.dumps({"last_processed_tx_id": "0"}))
    sync2 = _reopen(journal_path, cursor_path, [(open_and_partial, None)])
    result2 = sync2.run_once()

    assert result2.partials == 0
    entry = json.loads(journal_path.read_text())[0]
    assert entry["partial_realized_pl"] == pytest.approx(100.0)
    assert entry["partial_close_tx_ids"] == ["1350"]
    assert len(json.loads(journal_path.read_text())) == 1  # no duplicate entry


def test_partial_close_never_touches_non_trend_entries(tmp_path: Path) -> None:
    scanner_entry = {"trade_id": "1329", "lane": "scanner", "outcome": None,
                     "pair": "USD_JPY"}
    pages = [([_reduce_fill("1350", "1329", "100.0")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages, journal=[scanner_entry])
    result = sync.run_once()

    assert result.partials == 0
    written = json.loads(journal_path.read_text())
    assert written == [scanner_entry]  # byte-for-byte untouched


# --------------------------------------------------------------------------- #
# Dedup / never-create-on-close / journal safety
# --------------------------------------------------------------------------- #

def test_one_close_never_creates_an_entry(tmp_path: Path) -> None:
    pages = [([_close_fill("1400", "9999")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    result = sync.run_once()

    assert result.closed == 0
    assert result.unmatched_close_trade_ids == ["9999"]
    assert json.loads(journal_path.read_text()) == []


def test_rerun_over_same_window_creates_no_duplicate_entries(tmp_path: Path) -> None:
    page = [_open_fill("1329", "1329"), _close_fill("1400", "1329")]
    sync, journal_path, cursor_path = _make_sync(tmp_path, [(page, None)])
    sync.run_once()

    cursor_path.write_text(json.dumps({"last_processed_tx_id": "0"}))
    sync2 = _reopen(journal_path, cursor_path, [(page, None)])
    result2 = sync2.run_once()

    assert result2.opened == 0
    assert result2.skipped_existing == ["1329"]
    written = json.loads(journal_path.read_text())
    assert len(written) == 1
    assert written[0]["realized_pl_total"] == pytest.approx(-42.5)  # unchanged


def test_corrupt_journal_is_surfaced_not_clobbered(tmp_path: Path) -> None:
    sync, journal_path, cursor_path = _make_sync(
        tmp_path, [([_open_fill("1329", "1329")], None)],
    )
    journal_path.write_text("{ this is not json")
    before = journal_path.read_text()

    result = sync.run_once()

    assert result.error is not None
    assert journal_path.read_text() == before  # file untouched
    assert not cursor_path.exists()  # cursor NOT advanced past unjournaled txs


def test_wrong_shape_journal_is_refused_not_replaced(tmp_path: Path) -> None:
    """A dict-shaped (parseable but wrong) journal must not be silently
    replaced with only this run's trend entries."""
    sync, journal_path, cursor_path = _make_sync(
        tmp_path, [([_open_fill("1329", "1329")], None)],
    )
    journal_path.write_text(json.dumps({"trades": ["not", "a", "list"]}))
    before = journal_path.read_text()

    result = sync.run_once()

    assert result.error is not None and "not a list" in result.error
    assert journal_path.read_text() == before
    assert not cursor_path.exists()


def test_journal_write_is_atomic_and_leaves_no_tmp(tmp_path: Path) -> None:
    pages = [([_open_fill("1329", "1329"), _close_fill("1400", "1329")], None)]
    sync, journal_path, _ = _make_sync(tmp_path, pages)
    sync.run_once()

    # Valid JSON on disk, no orphaned tmp file from the atomic write.
    json.loads(journal_path.read_text())
    assert not journal_path.with_suffix(journal_path.suffix + ".tmp").exists()


def test_second_write_creates_bak_of_previous_version(tmp_path: Path) -> None:
    sync, journal_path, cursor_path = _make_sync(
        tmp_path, [([_open_fill("1329", "1329")], None)],
    )
    sync.run_once()
    first_version = journal_path.read_text()

    sync2 = _reopen(journal_path, cursor_path, [([_close_fill("1400", "1329")], None)])
    sync2.run_once()

    bak = journal_path.with_suffix(journal_path.suffix + ".bak")
    assert bak.exists()
    assert bak.read_text() == first_version
