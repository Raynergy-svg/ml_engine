"""US-009 — Tests for :mod:`src.equity.corporate_actions`.

No mocks. Real :class:`CorporateActionHandler` against a real on-disk
SHIP_GATE.json (written under ``tmp_path``), real :class:`BookState`,
real :class:`UniverseSnapshot` from :mod:`src.equity.universe`.

The PRD acceptance criteria drive the test layout:

* **4:1 split is NOT a -75% drawdown.** A 100-share, $400 stock split 4:1
  must leave NAV intact (400 shares × $100), with the broker quote and the
  back-adjusted price agreeing post-action.
* **Cash dividend** banks shares × cps in cash; broker raw price drops by
  the dividend (matching the back-adjusted return on ex-date).
* **Delisting / M&A cash-out** removes the position, banks ``shares ×
  payout`` (legitimately zero), and reports the ticker for the universe
  drop. A test asserts :func:`drop_delisted_from_universe` strips the
  ticker on/after the effective date and that the universe hash rotates.
* **Halt on divergence.** Feed the handler a broker raw price that does
  not match ``adjusted / cumulative_adjustment`` beyond tolerance and
  assert ``halted=True`` with a populated ``halt_reason``.
* **Ship-gate guard.** Missing, malformed, ``gate_pass=False``, and hash-
  mismatch SHIP_GATE.json all raise.
* **Atomic writes.** ``write_book`` round-trips through ``tmp + replace``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.equity import EQUITY_PIPELINE_VERSION
from src.equity.corporate_actions import (
    ACTION_CASH_DIV,
    ACTION_DELIST,
    ACTION_MA_CASH,
    ACTION_SPLIT,
    BookState,
    CorporateAction,
    CorporateActionConfig,
    CorporateActionError,
    CorporateActionHandler,
    CorporateActionResult,
    DivergenceAlert,
    Position,
    drop_delisted_from_universe,
)
from src.equity.universe import (
    UniverseRules,
    UniverseSnapshot,
    compute_universe_hash,
)


UNIVERSE_HASH = "abc12345" * 8  # 64-char placeholder
OTHER_HASH = "fedc6789" * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
    malformed: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if malformed:
        path.write_text("{not valid json")
        return path
    payload = {
        "gate_pass": bool(gate_pass),
        "net_sharpe": 0.55,
        "max_dd": 0.20,
        "positive_years": 11,
        "total_years": 16,
        "universe_hash": universe_hash,
        "asof": "2026-06-19",
        "recommendation": "PASS — US-009 test fixture",
        "criteria_source": "test",
        "thresholds": {},
        "pipeline_version": EQUITY_PIPELINE_VERSION,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _book(**positions_overrides) -> BookState:
    """Default book: $50k cash + AAPL@$400×100 + MSFT@$200×100 = $110k NAV."""
    positions = {
        "AAPL": Position("AAPL", shares=100.0, avg_cost=350.0, last_price=400.0),
        "MSFT": Position("MSFT", shares=100.0, avg_cost=180.0, last_price=200.0),
    }
    positions.update(positions_overrides)
    return BookState(
        asof="2026-06-19",
        cash=50_000.0,
        peak_nav=110_000.0,
        positions=positions,
    )


def _handler(tmp_path: Path) -> CorporateActionHandler:
    ship_gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(ship_gate)
    config = CorporateActionConfig()
    return CorporateActionHandler.create(
        config=config,
        ship_gate_path=ship_gate,
        universe_hash=UNIVERSE_HASH,
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_config_defaults_valid() -> None:
    cfg = CorporateActionConfig()
    assert 0 < cfg.divergence_tolerance < 1.0
    assert cfg.price_min > 0
    assert cfg.halt_on_divergence is True


@pytest.mark.parametrize(
    "tol", [0.0, -0.01, 1.0, 1.5, float("nan")]
)
def test_config_rejects_bad_tolerance(tol: float) -> None:
    with pytest.raises(CorporateActionError):
        CorporateActionConfig(divergence_tolerance=tol)


def test_config_rejects_nonpositive_price_min() -> None:
    with pytest.raises(CorporateActionError):
        CorporateActionConfig(price_min=0.0)
    with pytest.raises(CorporateActionError):
        CorporateActionConfig(price_min=-1.0)


# ---------------------------------------------------------------------------
# CorporateAction validation + round-trip
# ---------------------------------------------------------------------------


def test_action_split_valid() -> None:
    a = CorporateAction("AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0)
    assert a.ticker == "AAPL"
    assert a.kind == ACTION_SPLIT
    assert a.ratio == pytest.approx(4.0)
    assert a.asof_ts == pd.Timestamp("2026-06-19", tz="UTC")


def test_action_rejects_unknown_kind() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction("AAPL", "2026-06-19", "spinoff")


def test_action_rejects_empty_ticker() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction("", "2026-06-19", ACTION_SPLIT, ratio=4.0)


def test_action_rejects_bad_split_ratio() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction("AAPL", "2026-06-19", ACTION_SPLIT, ratio=0.0)
    with pytest.raises(CorporateActionError):
        CorporateAction(
            "AAPL", "2026-06-19", ACTION_SPLIT, ratio=float("inf")
        )


def test_action_rejects_negative_cash_div() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction(
            "AAPL",
            "2026-06-19",
            ACTION_CASH_DIV,
            cash_per_share=-0.5,
        )


def test_action_rejects_negative_payout() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction(
            "AAPL",
            "2026-06-19",
            ACTION_DELIST,
            cash_payout=-1.0,
        )


def test_action_rejects_bad_asof() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction("AAPL", "not-a-date", ACTION_SPLIT, ratio=4.0)


def test_action_dict_round_trip() -> None:
    a = CorporateAction(
        "AAPL",
        "2026-06-19",
        ACTION_CASH_DIV,
        cash_per_share=0.50,
        note="quarterly",
    )
    payload = a.to_dict()
    restored = CorporateAction.from_dict(payload)
    assert restored == a


def test_action_from_dict_rejects_bad_payload() -> None:
    with pytest.raises(CorporateActionError):
        CorporateAction.from_dict({"ticker": "AAPL"})  # missing asof/kind
    with pytest.raises(CorporateActionError):
        CorporateAction.from_dict("not-a-mapping")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Position + BookState
# ---------------------------------------------------------------------------


def test_position_market_value() -> None:
    p = Position("AAPL", shares=100.0, avg_cost=350.0, last_price=400.0)
    assert p.market_value == pytest.approx(40_000.0)


def test_position_rejects_nonfinite() -> None:
    with pytest.raises(CorporateActionError):
        Position("AAPL", shares=float("nan"), avg_cost=1, last_price=1)
    with pytest.raises(CorporateActionError):
        Position("AAPL", shares=1, avg_cost=float("inf"), last_price=1)


def test_book_nav_and_drawdown() -> None:
    book = _book()
    # 50k + 100*400 + 100*200 = 110_000
    assert book.nav == pytest.approx(110_000.0)
    assert book.drawdown() == pytest.approx(0.0)


def test_book_drawdown_after_loss() -> None:
    book = _book()
    book.peak_nav = 200_000.0
    # NAV 110k / peak 200k => dd = 0.45
    assert book.drawdown() == pytest.approx(0.45)


def test_book_dict_round_trip() -> None:
    book = _book()
    payload = book.to_dict()
    restored = BookState.from_dict(payload)
    assert restored.nav == pytest.approx(book.nav)
    assert set(restored.positions) == set(book.positions)


def test_book_rejects_bad_payload() -> None:
    with pytest.raises(CorporateActionError):
        BookState.from_dict("nope")  # type: ignore[arg-type]
    with pytest.raises(CorporateActionError):
        BookState.from_dict({"asof": "2026-06-19", "cash": 0.0})


# ---------------------------------------------------------------------------
# Ship-gate guard
# ---------------------------------------------------------------------------


def test_create_succeeds_with_valid_ship_gate(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    assert handler.universe_hash == UNIVERSE_HASH


def test_create_blocks_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(CorporateActionError, match="not found"):
        CorporateActionHandler.create(
            CorporateActionConfig(),
            ship_gate_path=missing,
            universe_hash=UNIVERSE_HASH,
        )


def test_create_blocks_when_gate_pass_false(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(path, gate_pass=False)
    with pytest.raises(CorporateActionError, match="gate_pass"):
        CorporateActionHandler.create(
            CorporateActionConfig(),
            ship_gate_path=path,
            universe_hash=UNIVERSE_HASH,
        )


def test_create_blocks_when_malformed(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(path, malformed=True)
    with pytest.raises(CorporateActionError, match="cannot read"):
        CorporateActionHandler.create(
            CorporateActionConfig(),
            ship_gate_path=path,
            universe_hash=UNIVERSE_HASH,
        )


def test_create_blocks_when_hash_mismatched(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(path, universe_hash=OTHER_HASH)
    with pytest.raises(CorporateActionError, match="universe_hash mismatch"):
        CorporateActionHandler.create(
            CorporateActionConfig(),
            ship_gate_path=path,
            universe_hash=UNIVERSE_HASH,
        )


def test_create_rejects_blank_universe_hash(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(path)
    with pytest.raises(CorporateActionError, match="non-empty"):
        CorporateActionHandler.create(
            CorporateActionConfig(),
            ship_gate_path=path,
            universe_hash="",
        )


def test_create_rejects_bad_config_type(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(path)
    with pytest.raises(CorporateActionError, match="CorporateActionConfig"):
        CorporateActionHandler.create(
            config="cfg",  # type: ignore[arg-type]
            ship_gate_path=path,
            universe_hash=UNIVERSE_HASH,
        )


# ---------------------------------------------------------------------------
# 4:1 split — NAV invariant, no drawdown
# ---------------------------------------------------------------------------


def test_split_4_to_1_preserves_nav(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    nav_before = book.nav

    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    new_book, result = handler.reconcile(book, [split], asof)

    # 400 shares × $100 (= 100 × $400 pre-split) — NAV invariant.
    aapl = new_book.positions["AAPL"]
    assert aapl.shares == pytest.approx(400.0)
    assert aapl.last_price == pytest.approx(100.0)
    assert aapl.avg_cost == pytest.approx(350.0 / 4.0)

    assert new_book.nav == pytest.approx(nav_before)
    # Crucially, drawdown is unchanged — NOT 75%.
    assert new_book.drawdown() == pytest.approx(0.0)
    assert new_book.drawdown() < 0.01

    assert result.nav_before == pytest.approx(nav_before)
    assert result.nav_after == pytest.approx(nav_before)
    assert result.cash_received == pytest.approx(0.0)
    assert result.delisted_tickers == []
    assert result.halted is False
    assert result.actions_applied == [split]


def test_split_4_to_1_does_not_register_as_drawdown(
    tmp_path: Path,
) -> None:
    """The AC headline assertion, phrased exactly the PRD way."""
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    new_book, _ = handler.reconcile(book, [split], asof)
    # A naive (last_price / yesterday_price - 1) check on AAPL alone would
    # produce -75%; the corrected book has no drawdown.
    assert new_book.drawdown() < 0.0001


def test_reverse_split_2_to_1(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "MSFT", "2026-06-19", ACTION_SPLIT, ratio=0.5
    )
    new_book, _ = handler.reconcile(book, [split], asof)
    msft = new_book.positions["MSFT"]
    assert msft.shares == pytest.approx(50.0)
    assert msft.last_price == pytest.approx(400.0)
    assert new_book.nav == pytest.approx(110_000.0)


# ---------------------------------------------------------------------------
# Cash dividend
# ---------------------------------------------------------------------------


def test_cash_dividend_banks_cash_and_drops_quote(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    nav_before = book.nav
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    div = CorporateAction(
        "AAPL", "2026-06-19", ACTION_CASH_DIV, cash_per_share=2.0
    )
    new_book, result = handler.reconcile(book, [div], asof)
    aapl = new_book.positions["AAPL"]

    # Shares & cost unchanged; broker quote drops by the dividend.
    assert aapl.shares == pytest.approx(100.0)
    assert aapl.avg_cost == pytest.approx(350.0)
    assert aapl.last_price == pytest.approx(398.0)
    # Cash up by 100 * $2 = $200.
    assert new_book.cash == pytest.approx(50_200.0)
    # NAV preserved (the divs go into cash to balance the price drop).
    assert new_book.nav == pytest.approx(nav_before)
    assert result.cash_received == pytest.approx(200.0)


def test_zero_dividend_is_noop(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    div = CorporateAction(
        "AAPL", "2026-06-19", ACTION_CASH_DIV, cash_per_share=0.0
    )
    new_book, result = handler.reconcile(book, [div], asof)
    assert new_book.cash == pytest.approx(50_000.0)
    assert new_book.positions["AAPL"].last_price == pytest.approx(400.0)
    assert result.cash_received == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Delisting / M&A
# ---------------------------------------------------------------------------


def test_delisting_with_cash_payout(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    delist = CorporateAction(
        "MSFT",
        "2026-06-19",
        ACTION_DELIST,
        cash_payout=180.0,
    )
    new_book, result = handler.reconcile(book, [delist], asof)
    assert "MSFT" not in new_book.positions
    # 100 shares × $180 = $18k payout.
    assert new_book.cash == pytest.approx(50_000.0 + 18_000.0)
    # Pre-delisting MSFT mv was 100 × $200 = $20k; payout $18k -> NAV -$2k.
    assert new_book.nav == pytest.approx(110_000.0 - 2_000.0)
    assert result.delisted_tickers == ["MSFT"]
    assert result.cash_received == pytest.approx(18_000.0)


def test_delisting_zero_recovery(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    delist = CorporateAction(
        "AAPL",
        "2026-06-19",
        ACTION_DELIST,
        cash_payout=0.0,
    )
    new_book, result = handler.reconcile(book, [delist], asof)
    assert "AAPL" not in new_book.positions
    assert new_book.cash == pytest.approx(50_000.0)
    # Lost the entire $40k AAPL leg.
    assert new_book.nav == pytest.approx(110_000.0 - 40_000.0)
    assert result.cash_received == pytest.approx(0.0)


def test_ma_cash_clears_position(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    ma = CorporateAction(
        "MSFT",
        "2026-06-19",
        ACTION_MA_CASH,
        cash_payout=205.0,
        note="acquired by ACME",
    )
    new_book, result = handler.reconcile(book, [ma], asof)
    assert "MSFT" not in new_book.positions
    assert new_book.cash == pytest.approx(50_000.0 + 20_500.0)
    assert result.delisted_tickers == ["MSFT"]


def test_action_for_unheld_ticker_is_skipped(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "TSLA",  # not in book
        "2026-06-19",
        ACTION_SPLIT,
        ratio=3.0,
    )
    new_book, result = handler.reconcile(book, [split], asof)
    assert result.actions_applied == []
    assert result.actions_skipped == [split]
    # Book identical apart from asof.
    assert new_book.nav == pytest.approx(book.nav)


# ---------------------------------------------------------------------------
# Divergence detection + HALT
# ---------------------------------------------------------------------------


def test_divergence_within_tolerance_does_not_halt(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")

    # 1.0 cumulative adjustment + matching prices -> zero divergence.
    raw = {"AAPL": 400.0, "MSFT": 200.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    cum = {"AAPL": 1.0, "MSFT": 1.0}
    new_book, result = handler.reconcile(
        book,
        [],
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    assert result.divergences == []
    assert result.halted is False
    assert new_book.halted is False


def test_divergence_outside_tolerance_halts(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")

    # Broker says AAPL = $100 but the data side says adjusted 400 with
    # cum_adj 1.0 -> expected raw 400. 75% mismatch -> halt.
    raw = {"AAPL": 100.0, "MSFT": 200.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    cum = {"AAPL": 1.0, "MSFT": 1.0}
    new_book, result = handler.reconcile(
        book,
        [],
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    assert result.halted is True
    assert result.halt_reason is not None
    assert "AAPL" in result.halt_reason
    assert new_book.halted is True
    assert new_book.halt_reason == result.halt_reason
    # And the alert captures the right numbers.
    aapl_alert = next(d for d in result.divergences if d.ticker == "AAPL")
    assert aapl_alert.expected_raw_price == pytest.approx(400.0)
    assert aapl_alert.raw_price == pytest.approx(100.0)
    assert aapl_alert.relative_error == pytest.approx(0.75, abs=1e-6)


def test_split_correctly_applied_no_divergence(tmp_path: Path) -> None:
    """A 4:1 split, then back-adjusted feed catches up — no halt."""
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")

    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    # Post-split: broker quote = $100, data feed adjusted = $400, cum=4.0.
    # Expected raw = 400 / 4 = 100  -> matches broker  -> no divergence.
    raw = {"AAPL": 100.0, "MSFT": 200.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    cum = {"AAPL": 4.0, "MSFT": 1.0}
    new_book, result = handler.reconcile(
        book,
        [split],
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    assert result.halted is False
    assert result.divergences == []
    assert new_book.positions["AAPL"].shares == pytest.approx(400.0)
    assert new_book.positions["AAPL"].last_price == pytest.approx(100.0)


def test_halt_disabled_records_divergence_without_halting(
    tmp_path: Path,
) -> None:
    ship_gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(ship_gate)
    config = CorporateActionConfig(halt_on_divergence=False)
    handler = CorporateActionHandler.create(
        config=config,
        ship_gate_path=ship_gate,
        universe_hash=UNIVERSE_HASH,
    )
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    raw = {"AAPL": 100.0, "MSFT": 200.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    cum = {"AAPL": 1.0, "MSFT": 1.0}
    new_book, result = handler.reconcile(
        book,
        [],
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    assert result.divergences  # we still saw the skew
    assert result.halted is False
    assert new_book.halted is False


def test_divergence_check_rejects_nan_adjusted(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    raw = {"AAPL": 400.0}
    adj = {"AAPL": float("nan")}
    cum = {"AAPL": 1.0}
    with pytest.raises(CorporateActionError, match="adjusted_prices"):
        handler.reconcile(
            book,
            [],
            asof,
            raw_prices=raw,
            adjusted_prices=adj,
            cumulative_adjustment=cum,
        )


def test_divergence_check_rejects_zero_cum_adj(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    raw = {"AAPL": 400.0}
    adj = {"AAPL": 400.0}
    cum = {"AAPL": 0.0}
    with pytest.raises(CorporateActionError, match="cumulative_adjustment"):
        handler.reconcile(
            book,
            [],
            asof,
            raw_prices=raw,
            adjusted_prices=adj,
            cumulative_adjustment=cum,
        )


# ---------------------------------------------------------------------------
# Reconcile-time integrity
# ---------------------------------------------------------------------------


def test_reconcile_rejects_action_with_wrong_asof(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-20", ACTION_SPLIT, ratio=4.0
    )
    with pytest.raises(CorporateActionError, match="!= reconcile asof"):
        handler.reconcile(book, [split], asof)


def test_reconcile_rejects_non_action(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    with pytest.raises(CorporateActionError, match="CorporateAction"):
        handler.reconcile(book, [{"kind": "split"}], asof)  # type: ignore[list-item]  # noqa: E501


def test_reconcile_does_not_mutate_caller_book(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    cash_before = book.cash
    aapl_shares_before = book.positions["AAPL"].shares
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    new_book, _ = handler.reconcile(book, [split], asof)
    # Original instance untouched.
    assert book.cash == pytest.approx(cash_before)
    assert book.positions["AAPL"].shares == pytest.approx(aapl_shares_before)
    # New instance carries the split.
    assert new_book.positions["AAPL"].shares == pytest.approx(400.0)


def test_reconcile_rejects_bad_raw_price(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    raw = {"AAPL": -1.0, "MSFT": 200.0}
    with pytest.raises(CorporateActionError, match="raw_prices"):
        handler.reconcile(book, [], asof, raw_prices=raw)


# ---------------------------------------------------------------------------
# Universe drop helper
# ---------------------------------------------------------------------------


def _toy_universe() -> UniverseSnapshot:
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            ["2026-06-18", "2026-06-19", "2026-06-20"], utc=True
        ),
        name="date",
    )
    tickers = ["AAPL", "MSFT", "OLD"]
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(),
        built_at="2026-06-19T00:00:00+00:00",
        pipeline_version=EQUITY_PIPELINE_VERSION,
        universe_hash=compute_universe_hash(membership),
    )


def test_drop_delisted_zeroes_membership_on_and_after(
    tmp_path: Path,
) -> None:
    snap = _toy_universe()
    new_snap = drop_delisted_from_universe(
        snap,
        delistings=[("OLD", pd.Timestamp("2026-06-19", tz="UTC"))],
    )
    members_before = new_snap.members_on(pd.Timestamp("2026-06-18", tz="UTC"))
    members_on = new_snap.members_on(pd.Timestamp("2026-06-19", tz="UTC"))
    members_after = new_snap.members_on(pd.Timestamp("2026-06-20", tz="UTC"))
    assert "OLD" in members_before
    assert "OLD" not in members_on
    assert "OLD" not in members_after
    # Hash must rotate so any downstream consumer sees a fresh universe.
    assert new_snap.universe_hash != snap.universe_hash


def test_drop_delisted_no_op_for_unknown_ticker(tmp_path: Path) -> None:
    snap = _toy_universe()
    new_snap = drop_delisted_from_universe(
        snap,
        delistings=[("UNKNOWN", pd.Timestamp("2026-06-19", tz="UTC"))],
    )
    # No change -> same instance returned (hash unchanged).
    assert new_snap is snap


def test_drop_delisted_rejects_bad_input() -> None:
    with pytest.raises(CorporateActionError, match="UniverseSnapshot"):
        drop_delisted_from_universe(
            "not-a-snapshot",  # type: ignore[arg-type]
            delistings=[],
        )


# ---------------------------------------------------------------------------
# Atomic state writes
# ---------------------------------------------------------------------------


def test_write_book_round_trips(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    new_book, _ = handler.reconcile(book, [split], asof)

    state_path = tmp_path / "subdir" / "book_state.json"
    written = handler.write_book(new_book, state_path)
    assert written == state_path
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    restored = BookState.from_dict(payload)
    assert restored.nav == pytest.approx(new_book.nav)
    assert restored.positions["AAPL"].shares == pytest.approx(400.0)
    # No stray .tmp files left behind.
    assert [p.name for p in state_path.parent.glob("*.tmp")] == []


def test_write_result_round_trips(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    div = CorporateAction(
        "AAPL", "2026-06-19", ACTION_CASH_DIV, cash_per_share=2.0
    )
    _, result = handler.reconcile(book, [div], asof)
    rpath = tmp_path / "result.json"
    handler.write_result(result, rpath)
    payload = json.loads(rpath.read_text(encoding="utf-8"))
    assert payload["cash_received"] == pytest.approx(200.0)
    assert payload["actions_applied"][0]["kind"] == ACTION_CASH_DIV


# ---------------------------------------------------------------------------
# End-to-end PRD scenario
# ---------------------------------------------------------------------------


def test_synthetic_split_dividend_delisting_scenario(
    tmp_path: Path,
) -> None:
    """One reconcile cycle drives one of each action kind + divergence-clean.

    Asserts:
    * 4:1 split on AAPL: shares×4, price÷4, NAV invariant.
    * $1 dividend on MSFT: cash += 100, price drops to $199.
    * Delisting on OLD with $5 payout: position cleared, cash += 50.
    * No divergence flag, no halt.
    """
    handler = _handler(tmp_path)
    book = _book(
        OLD=Position("OLD", shares=10.0, avg_cost=4.0, last_price=5.0),
    )
    nav_before = book.nav

    asof = pd.Timestamp("2026-06-19", tz="UTC")
    actions = [
        CorporateAction(
            "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
        ),
        CorporateAction(
            "MSFT", "2026-06-19", ACTION_CASH_DIV, cash_per_share=1.0
        ),
        CorporateAction(
            "OLD",
            "2026-06-19",
            ACTION_DELIST,
            cash_payout=5.0,
        ),
    ]
    # Post-action broker quotes (back-adjusted feed):
    raw = {"AAPL": 100.0, "MSFT": 199.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    # cum_adj for AAPL = 4 (split); MSFT = 200 / 199 ≈ 1.005025 (div).
    cum = {"AAPL": 4.0, "MSFT": 200.0 / 199.0}

    new_book, result = handler.reconcile(
        book,
        actions,
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    # Split + dividend invariants.
    assert new_book.positions["AAPL"].shares == pytest.approx(400.0)
    assert new_book.positions["AAPL"].last_price == pytest.approx(100.0)
    assert new_book.positions["MSFT"].last_price == pytest.approx(199.0)
    # OLD position removed.
    assert "OLD" not in new_book.positions
    # Each leg is NAV-neutral:
    #   AAPL split — shares×4 cancels price÷4.
    #   MSFT div   — +$100 cash cancels -$100 mv.
    #   OLD delist — +$50 cash cancels -$50 mv.
    # NAV is unchanged from nav_before.
    assert new_book.nav == pytest.approx(nav_before)
    assert result.cash_received == pytest.approx(100.0 + 50.0)
    assert result.delisted_tickers == ["OLD"]
    assert result.halted is False
    assert result.divergences == []
    assert {a.ticker for a in result.actions_applied} == {
        "AAPL",
        "MSFT",
        "OLD",
    }


def test_divergence_alert_dict_shape(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    raw = {"AAPL": 100.0, "MSFT": 200.0}
    adj = {"AAPL": 400.0, "MSFT": 200.0}
    cum = {"AAPL": 1.0, "MSFT": 1.0}
    _, result = handler.reconcile(
        book,
        [],
        asof,
        raw_prices=raw,
        adjusted_prices=adj,
        cumulative_adjustment=cum,
    )
    alert = next(d for d in result.divergences if d.ticker == "AAPL")
    assert isinstance(alert, DivergenceAlert)
    payload = alert.to_dict()
    assert payload["ticker"] == "AAPL"
    assert payload["raw_price"] == pytest.approx(100.0)
    assert payload["expected_raw_price"] == pytest.approx(400.0)
    assert payload["reason"] == "raw_vs_adjusted_mismatch"


def test_result_dict_shape(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    book = _book()
    asof = pd.Timestamp("2026-06-19", tz="UTC")
    split = CorporateAction(
        "AAPL", "2026-06-19", ACTION_SPLIT, ratio=4.0
    )
    _, result = handler.reconcile(book, [split], asof)
    assert isinstance(result, CorporateActionResult)
    payload = result.to_dict()
    assert payload["asof"] == "2026-06-19"
    assert payload["actions_applied"][0]["kind"] == ACTION_SPLIT
    assert payload["halted"] is False
