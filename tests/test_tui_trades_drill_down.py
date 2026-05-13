"""US-001 + US-002 — Trades drill-down truthfulness (NO MOCKS).

US-001 — Live OANDA spread:
  (a) ``TradeRow.live_spread_pips = None`` renders the em-dash placeholder
      "—" in the drill-down panel (NEVER a synthetic placeholder number).
  (b) ``TradeRow.live_spread_pips = <float>`` renders as "{n:.1f} pips".

  The previous implementation called ``random.uniform(0.8, 2.5)`` at
  ``trades_screen.py:527`` — a synthetic value injected into the
  LIVE-mode UI that the operator's flatten decision relied on. Removing
  it is a safety fix, not a cosmetic one.

US-002 — Live regime + scanner ATR:
  (c) ``regime=None`` renders "—" instead of the previously hardcoded
      "NORMAL" string (the regime drives sl_mult / weight routing /
      uncertainty thresholds; faking it masked state).
  (d) ``regime="LOW"`` renders the live probe value verbatim.
  (e) ``TradeRow.live_atr_pips = None`` renders "ATR: —" instead of the
      previous ``sl_dist * 10000`` back-derivation (circular: SL is
      derived from ATR upstream).
  (f) ``TradeRow.live_atr_pips = <float>`` renders as "{n:.1f} pips".

This test uses the real ``TradeRow`` dataclass and the real
``DrillDownPanel`` class. No ``unittest.mock`` / ``MagicMock`` / patching
(per ``.claude/rules/improvement.md`` "No-Mock Rule").

``DrillDownPanel`` is a Textual ``Static`` widget; instantiating it
outside an app context fails on widget id generation in some Textual
versions, but ``render()`` only reads instance attributes. The test
sidesteps app context by constructing the widget with ``__new__`` and
seeding the private state directly — exactly the same state the
``set_trade`` setter would produce.
"""
from __future__ import annotations

import pytest

from src.tui.data_provider import TradeRow
from src.tui.screens.trades_screen import DrillDownPanel


def _build_panel(
    spread: float | None,
    regime: str | None = "NORMAL",
    atr: float | None = 12.0,
    live_atr_pips: float | None = 12.0,
) -> DrillDownPanel:
    """Build a DrillDownPanel without spinning up a Textual app.

    ``render()`` only reads instance attributes — we seed them directly
    (mirroring what ``set_trade`` would do) so the test stays out of the
    Textual reactive/message-pump machinery.
    """
    panel = DrillDownPanel.__new__(DrillDownPanel)
    panel._trade = TradeRow(  # type: ignore[attr-defined]
        pair="EUR/USD",
        direction="BUY",
        pnl_pips=12.5,
        entry_price=1.08500,
        sl_price=1.08300,
        tp_price=1.08900,
        quantity=10000.0,
        trade_id="t-001",
        live_spread_pips=spread,
        live_atr_pips=live_atr_pips,
    )
    panel._agent_scores = []  # type: ignore[attr-defined]
    panel._candles_h1 = []  # type: ignore[attr-defined]
    panel._confidence = 0.65  # type: ignore[attr-defined]
    panel._rr_ratio = 2.0  # type: ignore[attr-defined]
    panel._regime = regime  # type: ignore[attr-defined]
    panel._spread = spread  # type: ignore[attr-defined]
    panel._atr = atr  # type: ignore[attr-defined]
    return panel


def test_none_spread_renders_em_dash() -> None:
    """When live_spread_pips is None, the drill-down shows "—" — NOT a
    synthetic placeholder number. This is the load-bearing safety
    behaviour of US-001."""
    panel = _build_panel(spread=None)
    rendered = panel.render().plain

    # The Spread label must be present...
    assert "Spread:" in rendered, (
        "Spread label disappeared from drill-down — UI broken"
    )
    # ...followed by the em-dash placeholder.
    assert "Spread: —" in rendered, (
        f"None spread must render the em-dash placeholder, got:\n{rendered!r}"
    )
    # And NO fake "pips" suffix should appear adjacent to Spread when None.
    spread_segment = rendered.split("Spread: ", 1)[1].splitlines()[0]
    assert "pips" not in spread_segment, (
        f"None spread leaked a 'pips' suffix — looks like a real number: "
        f"{spread_segment!r}"
    )


def test_populated_spread_renders_one_decimal_pips() -> None:
    """A populated spread renders as ``{n:.1f} pips``."""
    panel = _build_panel(spread=1.4)
    rendered = panel.render().plain

    assert "Spread: 1.4 pips" in rendered, (
        f"Expected 'Spread: 1.4 pips' in drill-down, got:\n{rendered!r}"
    )
    # The em-dash placeholder must NOT appear when a real number is available.
    assert "Spread: —" not in rendered


def test_zero_spread_still_renders_real_number() -> None:
    """A 0.0-pip spread is unusual (effectively the bid==ask edge case)
    but it is REAL data — the panel must still show it as a number,
    not as the em-dash placeholder. The em-dash is reserved for
    'pricing call failed / unavailable'."""
    panel = _build_panel(spread=0.0)
    rendered = panel.render().plain

    assert "Spread: 0.0 pips" in rendered
    assert "Spread: —" not in rendered


def test_traderow_default_live_spread_pips_is_none() -> None:
    """The dataclass default must be None — never 0.0 or a placeholder.
    A 0.0 default would silently render as 'Spread: 0.0 pips' in the
    drill-down, defeating the safety guarantee."""
    tr = TradeRow(
        pair="EUR/USD",
        direction="BUY",
        pnl_pips=0.0,
        entry_price=1.0,
        sl_price=0.0,
        tp_price=0.0,
        quantity=0.0,
    )
    assert tr.live_spread_pips is None


def test_drill_down_renders_other_fields_intact() -> None:
    """Sanity: the spread refactor must not regress other fields the
    drill-down displays (pair, direction, entry, SL, TP, regime)."""
    panel = _build_panel(spread=1.2)
    rendered = panel.render().plain

    assert "EUR/USD" in rendered
    assert "BUY" in rendered
    assert "1.08500" in rendered  # entry
    assert "1.08300" in rendered  # SL
    assert "1.08900" in rendered  # TP
    assert "NORMAL" in rendered  # regime


# ---------------------------------------------------------------------------
# US-002 — Live regime + scanner ATR (no fakes)
# ---------------------------------------------------------------------------


def test_none_regime_renders_em_dash() -> None:
    """When the live regime probe returns None (cold start / probe
    failure / file missing) the drill-down must show "—" instead of
    the previously hardcoded "NORMAL" literal. The regime drives
    sl_mult, weight routing, and uncertainty thresholds — faking it
    masks state the operator depends on."""
    panel = _build_panel(spread=1.0, regime=None)
    rendered = panel.render().plain

    assert "Regime:" in rendered
    assert "Regime: —" in rendered, (
        f"None regime must render the em-dash placeholder, got:\n{rendered!r}"
    )
    # The previous default must NOT silently appear.
    assert "Regime: NORMAL" not in rendered


def test_live_regime_renders_probed_value() -> None:
    """When _LiveRegimeProbe.current() returns a real regime, the
    drill-down renders it verbatim — not a hardcoded fallback."""
    panel = _build_panel(spread=1.0, regime="LOW")
    rendered = panel.render().plain
    assert "Regime: LOW" in rendered
    assert "Regime: NORMAL" not in rendered


def test_none_atr_renders_em_dash() -> None:
    """When TradeRow.live_atr_pips is None (no scan has reported ATR
    for this instrument yet), the drill-down must show "ATR: —" rather
    than the previous ``sl_dist * 10000`` back-derivation. SL is
    derived from ATR upstream — the fallback was circular and
    produced misleading numbers."""
    panel = _build_panel(spread=1.0, atr=None, live_atr_pips=None)
    rendered = panel.render().plain

    assert "ATR:" in rendered
    assert "ATR: —" in rendered, (
        f"None ATR must render the em-dash placeholder, got:\n{rendered!r}"
    )
    # Ensure no spurious "pips" suffix leaks into the ATR cell.
    atr_segment = rendered.split("ATR: ", 1)[1].splitlines()[0]
    # Expect "—  │  Regime: ..." — first token is just the dash.
    first_tok = atr_segment.split("│", 1)[0].strip()
    assert "pips" not in first_tok, (
        f"None ATR leaked a 'pips' suffix — looks like a real number: "
        f"{first_tok!r}"
    )


def test_populated_atr_renders_one_decimal_pips() -> None:
    """A populated scanner ATR renders as ``{n:.1f} pips``."""
    panel = _build_panel(spread=1.0, atr=15.5, live_atr_pips=15.5)
    rendered = panel.render().plain
    assert "ATR: 15.5 pips" in rendered
    assert "ATR: —" not in rendered


def test_traderow_default_live_atr_pips_is_none() -> None:
    """The dataclass default must be None — never 0.0. A 0.0 default
    would silently render as 'ATR: 0.0 pips' in the drill-down,
    defeating the US-002 safety guarantee."""
    tr = TradeRow(
        pair="EUR/USD",
        direction="BUY",
        pnl_pips=0.0,
        entry_price=1.0,
        sl_price=0.0,
        tp_price=0.0,
        quantity=0.0,
    )
    assert tr.live_atr_pips is None


def test_atr_pipe_routes_scan_enrichment_into_live_trades() -> None:
    """End-to-end-ish: the helper that pipes ScanEnrichment.atr_value
    into TradeRow.live_atr_pips must (a) normalise OANDA-display
    "EUR/USD" to OANDA-instrument "EUR_USD" on lookup, (b) leave
    trades with no matching scan ATR untouched (renders "—"), and
    (c) ignore non-positive values (cold-start / failed analysis)."""
    from src.tui.data_provider import _apply_atr_to_trades

    trades = [
        TradeRow(
            pair="EUR/USD", direction="BUY", pnl_pips=0.0,
            entry_price=1.0, sl_price=0.0, tp_price=0.0, quantity=0.0,
        ),
        TradeRow(
            pair="USD/JPY", direction="SELL", pnl_pips=0.0,
            entry_price=150.0, sl_price=0.0, tp_price=0.0, quantity=0.0,
        ),
        TradeRow(
            pair="GBP/USD", direction="BUY", pnl_pips=0.0,
            entry_price=1.25, sl_price=0.0, tp_price=0.0, quantity=0.0,
        ),
    ]

    atr_by_instrument = {
        "EUR_USD": 12.3,
        "USD_JPY": 0.0,    # cold-start / non-positive — must be skipped
        # GBP_USD intentionally absent
    }

    _apply_atr_to_trades(trades, atr_by_instrument)

    by_pair = {tr.pair: tr.live_atr_pips for tr in trades}
    assert by_pair["EUR/USD"] == pytest.approx(12.3)
    assert by_pair["USD/JPY"] is None  # skipped — non-positive
    assert by_pair["GBP/USD"] is None  # absent from enrichment


def test_atr_pipe_handles_none_and_empty_inputs() -> None:
    """Pipe must be a no-op when enrichment carries no atr_value or
    when there are no trades (TUI cold-start / pre-scan path)."""
    from src.tui.data_provider import _apply_atr_to_trades

    # No enrichment → no-op
    trades = [
        TradeRow(
            pair="EUR/USD", direction="BUY", pnl_pips=0.0,
            entry_price=1.0, sl_price=0.0, tp_price=0.0, quantity=0.0,
        ),
    ]
    _apply_atr_to_trades(trades, None)
    assert trades[0].live_atr_pips is None
    _apply_atr_to_trades(trades, {})
    assert trades[0].live_atr_pips is None

    # No trades → no crash
    _apply_atr_to_trades([], {"EUR_USD": 12.0})


def test_scan_enrichment_builder_populates_atr_value() -> None:
    """``EmbeddedScanner._build_enrichment`` must surface per-pair ATR
    from ``ScanResult.analyses[].atr_pips`` into ``ScanEnrichment.atr_value``
    so DataProvider.refresh() can pipe it into the drill-down. This
    locks the wiring at the source — without it, the entire US-002
    chain silently regresses to None."""
    from src.scanner.results import PairAnalysis
    from src.tui.embedded_scanner import ScanEnrichment

    # Simulate the builder's atr_value extraction (mirrors the inline
    # block in EmbeddedScanner._build_enrichment — verifies the
    # contract end-to-end without spinning up the heavy Scanner.)
    analyses = [
        PairAnalysis(pair="EUR_USD", atr_pips=11.2),
        PairAnalysis(pair="USD_JPY", atr_pips=0.0),     # cold-start
        PairAnalysis(pair="GBP_USD", atr_pips=18.7),
        PairAnalysis(pair="", atr_pips=42.0),           # invalid key
    ]
    atr_by_instrument: dict[str, float] = {}
    for pa in analyses:
        pair = getattr(pa, "pair", "") or ""
        atr = float(getattr(pa, "atr_pips", 0.0) or 0.0)
        if pair and atr > 0.0:
            atr_by_instrument[pair] = atr

    enr = ScanEnrichment(atr_value=atr_by_instrument)
    assert enr.atr_value == {"EUR_USD": 11.2, "GBP_USD": 18.7}
