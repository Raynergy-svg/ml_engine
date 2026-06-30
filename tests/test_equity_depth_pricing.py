"""US-014 — Depth-aware execution pricing tests.

Hard rules (matching the rest of the equity stack):

* NO ``unittest.mock`` / ``MagicMock`` / ``patch`` — real classes only.
* All disk I/O routes through ``tmp_path``.
* Atomic-write smoke: temp file never lingers after a successful write.
* Ship-gate guard fails closed: missing / malformed / false /
  mismatched ``universe_hash``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.equity.depth_pricing import (
    DepthAwareSizer,
    DepthPricingError,
    L2OrderBook,
    OrderBookEntry,
    SNAPSHOT_VERSION,
    book_to_snapshot,
    enforce_ship_gate,
    get_price_for_volume,
    get_vwap_for_volume,
    make_book,
    read_snapshot,
    snapshot_to_book,
    write_snapshot,
)


UNIVERSE_HASH = "deadbeef" * 8  # 64 hex chars — looks like a real SHA-256


# ---------------------------------------------------------------------------
# Synthetic L2 book fixtures
# ---------------------------------------------------------------------------


def _synthetic_book(symbol: str = "AAPL") -> L2OrderBook:
    """A small but realistic L2 book for the canonical math tests.

    Asks (ascending):  100.00 x 100,  100.05 x 200,  100.10 x 500
    Bids (descending):  99.95 x 150,   99.90 x 250,   99.80 x 400
    """
    asks = [
        OrderBookEntry(price=100.00, size=100.0),
        OrderBookEntry(price=100.05, size=200.0),
        OrderBookEntry(price=100.10, size=500.0),
    ]
    bids = [
        OrderBookEntry(price=99.95, size=150.0),
        OrderBookEntry(price=99.90, size=250.0),
        OrderBookEntry(price=99.80, size=400.0),
    ]
    return L2OrderBook(symbol=symbol, bids=bids, asks=asks)


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
    extra: dict | None = None,
) -> Path:
    payload = {
        "gate_pass": gate_pass,
        "universe_hash": universe_hash,
        "net_sharpe": 0.92,
        "max_dd": 0.229,
        "positive_years": 12,
        "asof": "2026-06-19",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# OrderBookEntry / L2OrderBook invariants
# ---------------------------------------------------------------------------


def test_entry_rejects_nonpositive_price():
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=0.0, size=1.0)
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=-1.0, size=1.0)


def test_entry_rejects_nonpositive_size():
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=100.0, size=0.0)
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=100.0, size=-5.0)


def test_entry_rejects_nonfinite():
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=float("nan"), size=1.0)
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=float("inf"), size=1.0)
    with pytest.raises(DepthPricingError):
        OrderBookEntry(price=100.0, size=float("nan"))


def test_book_rejects_blank_symbol():
    with pytest.raises(DepthPricingError):
        L2OrderBook(symbol="", bids=[], asks=[])


def test_book_rejects_non_descending_bids():
    with pytest.raises(DepthPricingError):
        L2OrderBook(
            symbol="AAPL",
            bids=[
                OrderBookEntry(price=99.0, size=10.0),
                OrderBookEntry(price=99.5, size=10.0),  # wrong order
            ],
            asks=[],
        )


def test_book_rejects_non_ascending_asks():
    with pytest.raises(DepthPricingError):
        L2OrderBook(
            symbol="AAPL",
            bids=[],
            asks=[
                OrderBookEntry(price=100.5, size=10.0),
                OrderBookEntry(price=100.0, size=10.0),  # wrong order
            ],
        )


def test_book_rejects_crossed_book():
    with pytest.raises(DepthPricingError):
        L2OrderBook(
            symbol="AAPL",
            bids=[OrderBookEntry(price=100.10, size=10.0)],
            asks=[OrderBookEntry(price=100.00, size=10.0)],
        )


def test_book_accepts_empty_sides():
    # Pre-open / halted books legitimately have one or both sides empty.
    book = L2OrderBook(symbol="AAPL")
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid_price() is None
    assert book.total_size(is_buy=True) == 0.0


def test_make_book_resorts_inputs():
    # Caller can pass either side in any order; make_book normalises.
    book = make_book(
        "AAPL",
        bids=[(99.80, 400.0), (99.95, 150.0), (99.90, 250.0)],
        asks=[(100.10, 500.0), (100.00, 100.0), (100.05, 200.0)],
    )
    assert [b.price for b in book.bids] == [99.95, 99.90, 99.80]
    assert [a.price for a in book.asks] == [100.00, 100.05, 100.10]


# ---------------------------------------------------------------------------
# Math: get_price_for_volume
# ---------------------------------------------------------------------------


def test_price_for_volume_buy_within_top_level():
    """Buy 50 shares: fully fills inside the top ask (100.00)."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=True, volume=50.0)
    assert result.query_volume == 50.0
    assert result.result_volume == 50.0
    assert result.result_price == pytest.approx(100.00)


def test_price_for_volume_buy_exactly_top_level():
    """Buy exactly 100 shares: consumes top ask exactly, last touched = 100.00."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=True, volume=100.0)
    assert result.result_volume == pytest.approx(100.0)
    assert result.result_price == pytest.approx(100.00)


def test_price_for_volume_buy_walks_two_levels():
    """Buy 150 shares: 100 @ 100.00 + 50 @ 100.05, worst price = 100.05."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=True, volume=150.0)
    assert result.result_volume == pytest.approx(150.0)
    assert result.result_price == pytest.approx(100.05)


def test_price_for_volume_buy_walks_all_levels():
    """Buy 800 shares (full book): worst price = 100.10."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=True, volume=800.0)
    assert result.result_volume == pytest.approx(800.0)
    assert result.result_price == pytest.approx(100.10)


def test_price_for_volume_sell_walks_bids():
    """Sell 200 shares: 150 @ 99.95 + 50 @ 99.90, worst price = 99.90."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=False, volume=200.0)
    assert result.result_volume == pytest.approx(200.0)
    assert result.result_price == pytest.approx(99.90)


def test_price_for_volume_book_too_thin_returns_partial():
    """Asking for more than the book holds returns the partial fill + total."""
    book = _synthetic_book()
    result = get_price_for_volume(book, is_buy=True, volume=10_000.0)
    total_ask_size = 100.0 + 200.0 + 500.0
    assert result.result_volume == pytest.approx(total_ask_size)
    assert result.result_price == pytest.approx(100.10)  # deepest touched


def test_price_for_volume_empty_side_returns_zero():
    book = L2OrderBook(symbol="AAPL", bids=[], asks=[])
    result = get_price_for_volume(book, is_buy=True, volume=10.0)
    assert result.result_volume == 0.0
    assert result.result_price == 0.0


def test_price_for_volume_rejects_nonpositive_volume():
    book = _synthetic_book()
    with pytest.raises(DepthPricingError):
        get_price_for_volume(book, is_buy=True, volume=0.0)
    with pytest.raises(DepthPricingError):
        get_price_for_volume(book, is_buy=True, volume=-5.0)
    with pytest.raises(DepthPricingError):
        get_price_for_volume(book, is_buy=True, volume=float("nan"))


def test_price_for_volume_rejects_non_book():
    with pytest.raises(DepthPricingError):
        get_price_for_volume("not-a-book", is_buy=True, volume=10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Math: get_vwap_for_volume
# ---------------------------------------------------------------------------


def test_vwap_for_volume_within_top_level():
    """Inside the top level VWAP == top price exactly."""
    book = _synthetic_book()
    result = get_vwap_for_volume(book, is_buy=True, volume=50.0)
    assert result.result_volume == 50.0
    assert result.result_price == pytest.approx(100.00)


def test_vwap_for_volume_two_levels_exact_arithmetic():
    """Buy 150 = (100 @ 100.00 + 50 @ 100.05) / 150 = 100.016666..."""
    book = _synthetic_book()
    result = get_vwap_for_volume(book, is_buy=True, volume=150.0)
    expected = (100 * 100.00 + 50 * 100.05) / 150
    assert result.result_volume == pytest.approx(150.0)
    assert result.result_price == pytest.approx(expected)
    # Sanity: VWAP is strictly between best-ask and worst touched price.
    assert 100.00 < result.result_price < 100.05


def test_vwap_for_volume_full_book_buy():
    """All 800 ask shares: VWAP = (100*100 + 200*100.05 + 500*100.10)/800."""
    book = _synthetic_book()
    result = get_vwap_for_volume(book, is_buy=True, volume=800.0)
    expected = (100 * 100.00 + 200 * 100.05 + 500 * 100.10) / 800
    assert result.result_volume == pytest.approx(800.0)
    assert result.result_price == pytest.approx(expected)


def test_vwap_for_volume_full_book_sell():
    """All 800 bid shares: VWAP = (150*99.95 + 250*99.90 + 400*99.80)/800."""
    book = _synthetic_book()
    result = get_vwap_for_volume(book, is_buy=False, volume=800.0)
    expected = (150 * 99.95 + 250 * 99.90 + 400 * 99.80) / 800
    assert result.result_volume == pytest.approx(800.0)
    assert result.result_price == pytest.approx(expected)


def test_vwap_for_volume_book_too_thin_returns_partial_vwap():
    """If book has only 800 shares but we want 5000, VWAP is over the 800."""
    book = _synthetic_book()
    result = get_vwap_for_volume(book, is_buy=True, volume=5000.0)
    expected = (100 * 100.00 + 200 * 100.05 + 500 * 100.10) / 800
    assert result.result_volume == pytest.approx(800.0)
    assert result.result_price == pytest.approx(expected)


def test_vwap_for_volume_empty_side_returns_zero():
    book = L2OrderBook(symbol="AAPL", bids=[], asks=[])
    result = get_vwap_for_volume(book, is_buy=False, volume=10.0)
    assert result.result_volume == 0.0
    assert result.result_price == 0.0


def test_vwap_rejects_nonpositive_volume():
    book = _synthetic_book()
    with pytest.raises(DepthPricingError):
        get_vwap_for_volume(book, is_buy=True, volume=0.0)
    with pytest.raises(DepthPricingError):
        get_vwap_for_volume(book, is_buy=True, volume=float("inf"))


def test_vwap_is_never_better_than_best_price():
    """Across many random fills, VWAP is on the wrong side of best-quote.

    Buy VWAP >= best_ask; Sell VWAP <= best_bid. This is the load-bearing
    contract for execution sizing — if it ever inverts, we've crossed
    sign somewhere.
    """
    book = _synthetic_book()
    best_ask = book.best_ask()
    best_bid = book.best_bid()
    for volume in (10.0, 50.0, 100.0, 150.0, 300.0, 700.0):
        buy = get_vwap_for_volume(book, is_buy=True, volume=volume)
        sell = get_vwap_for_volume(book, is_buy=False, volume=volume)
        assert buy.result_price >= best_ask - 1e-12
        assert sell.result_price <= best_bid + 1e-12


def test_price_and_vwap_consistency_single_level():
    """When fill is entirely within one level, VWAP == price-for-volume."""
    book = _synthetic_book()
    p = get_price_for_volume(book, is_buy=True, volume=50.0)
    v = get_vwap_for_volume(book, is_buy=True, volume=50.0)
    assert p.result_price == pytest.approx(v.result_price)


def test_price_for_volume_is_always_worst_than_vwap_multi_level():
    """Multi-level fills: limit-equivalent worst >= VWAP."""
    book = _synthetic_book()
    for volume in (150.0, 300.0, 800.0):
        p = get_price_for_volume(book, is_buy=True, volume=volume)
        v = get_vwap_for_volume(book, is_buy=True, volume=volume)
        assert p.result_price >= v.result_price


# ---------------------------------------------------------------------------
# Ship-gate guard
# ---------------------------------------------------------------------------


def test_ship_gate_passes_when_valid(tmp_path):
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json")
    enforce_ship_gate(gate, UNIVERSE_HASH)  # must not raise


def test_ship_gate_refuses_when_missing(tmp_path):
    missing = tmp_path / "SHIP_GATE.json"
    with pytest.raises(DepthPricingError, match="ship gate not found"):
        enforce_ship_gate(missing, UNIVERSE_HASH)


def test_ship_gate_refuses_when_gate_pass_false(tmp_path):
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json", gate_pass=False)
    with pytest.raises(DepthPricingError, match="gate_pass="):
        enforce_ship_gate(gate, UNIVERSE_HASH)


def test_ship_gate_refuses_when_gate_pass_truthy_not_true(tmp_path):
    # gate_pass must be literal True — "true" string is not enough.
    gate = _write_ship_gate(
        tmp_path / "SHIP_GATE.json",
        extra={"gate_pass": "true"},
    )
    with pytest.raises(DepthPricingError, match="gate_pass="):
        enforce_ship_gate(gate, UNIVERSE_HASH)


def test_ship_gate_refuses_when_universe_hash_blank(tmp_path):
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json", universe_hash="")
    with pytest.raises(DepthPricingError, match="universe_hash missing/blank"):
        enforce_ship_gate(gate, UNIVERSE_HASH)


def test_ship_gate_refuses_when_universe_hash_mismatch(tmp_path):
    gate = _write_ship_gate(
        tmp_path / "SHIP_GATE.json",
        universe_hash="abc123" + ("0" * 58),
    )
    with pytest.raises(DepthPricingError, match="universe_hash mismatch"):
        enforce_ship_gate(gate, UNIVERSE_HASH)


def test_ship_gate_refuses_when_payload_not_object(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(DepthPricingError, match="is not a JSON object"):
        enforce_ship_gate(gate, UNIVERSE_HASH)


def test_ship_gate_refuses_when_unreadable_json(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text("{not valid json")
    with pytest.raises(DepthPricingError, match="cannot read"):
        enforce_ship_gate(gate, UNIVERSE_HASH)


# ---------------------------------------------------------------------------
# DepthAwareSizer (live entry point)
# ---------------------------------------------------------------------------


def test_sizer_constructs_when_gate_valid(tmp_path):
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json")
    sizer = DepthAwareSizer(UNIVERSE_HASH, ship_gate_path=gate)
    assert sizer.universe_hash == UNIVERSE_HASH


def test_sizer_refuses_when_gate_missing(tmp_path):
    missing = tmp_path / "SHIP_GATE.json"
    with pytest.raises(DepthPricingError):
        DepthAwareSizer(UNIVERSE_HASH, ship_gate_path=missing)


def test_sizer_refuses_when_gate_universe_mismatch(tmp_path):
    gate = _write_ship_gate(
        tmp_path / "SHIP_GATE.json",
        universe_hash="other-universe-hash-0000000000000000000000000000000000000000",
    )
    with pytest.raises(DepthPricingError, match="universe_hash mismatch"):
        DepthAwareSizer(UNIVERSE_HASH, ship_gate_path=gate)


def test_sizer_refuses_pricing_when_gate_invalidated_mid_session(tmp_path):
    """If the gate file is rewritten to fail between construction and a
    pricing call, the next pricing call must refuse — matching the
    rebalance/kill-switch re-enforce discipline."""
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json")
    sizer = DepthAwareSizer(UNIVERSE_HASH, ship_gate_path=gate)
    book = _synthetic_book()
    # First call succeeds.
    sizer.vwap_for_volume(book, is_buy=True, volume=50.0)
    # Operator pulls the gate.
    _write_ship_gate(gate, gate_pass=False)
    with pytest.raises(DepthPricingError, match="gate_pass="):
        sizer.vwap_for_volume(book, is_buy=True, volume=50.0)
    with pytest.raises(DepthPricingError, match="gate_pass="):
        sizer.price_for_volume(book, is_buy=False, volume=10.0)


def test_sizer_pricing_matches_module_function(tmp_path):
    """The sizer is a thin wrapper — results must be bit-identical."""
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json")
    sizer = DepthAwareSizer(UNIVERSE_HASH, ship_gate_path=gate)
    book = _synthetic_book()

    direct = get_vwap_for_volume(book, is_buy=True, volume=300.0)
    via_sizer = sizer.vwap_for_volume(book, is_buy=True, volume=300.0)
    assert direct == via_sizer

    direct_p = get_price_for_volume(book, is_buy=False, volume=300.0)
    via_p = sizer.price_for_volume(book, is_buy=False, volume=300.0)
    assert direct_p == via_p


def test_sizer_rejects_blank_universe_hash(tmp_path):
    gate = _write_ship_gate(tmp_path / "SHIP_GATE.json")
    with pytest.raises(DepthPricingError, match="universe_hash"):
        DepthAwareSizer("", ship_gate_path=gate)


# ---------------------------------------------------------------------------
# Snapshot round-trip (atomic write discipline)
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip_preserves_book(tmp_path):
    book = _synthetic_book()
    path = tmp_path / "snap.json"
    write_snapshot(book, path)

    loaded = read_snapshot(path)
    assert loaded.symbol == book.symbol
    assert [(e.price, e.size) for e in loaded.bids] == [
        (e.price, e.size) for e in book.bids
    ]
    assert [(e.price, e.size) for e in loaded.asks] == [
        (e.price, e.size) for e in book.asks
    ]


def test_snapshot_write_leaves_no_tmp_file(tmp_path):
    """Atomic write discipline: temp sibling must not linger on success."""
    book = _synthetic_book()
    path = tmp_path / "snap.json"
    write_snapshot(book, path)
    leftover = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftover == [], f"tmp files leaked: {leftover}"


def test_snapshot_write_atomic_replaces_existing(tmp_path):
    """An existing snapshot is replaced, not appended/corrupted."""
    book1 = _synthetic_book("AAPL")
    path = tmp_path / "snap.json"
    write_snapshot(book1, path)

    book2 = make_book(
        "MSFT",
        bids=[(420.00, 50.0)],
        asks=[(420.50, 75.0)],
    )
    write_snapshot(book2, path)

    loaded = read_snapshot(path)
    assert loaded.symbol == "MSFT"
    assert loaded.bids[0].price == 420.00
    assert loaded.asks[0].price == 420.50


def test_snapshot_payload_includes_version(tmp_path):
    book = _synthetic_book()
    path = tmp_path / "snap.json"
    write_snapshot(book, path)
    payload = json.loads(path.read_text())
    assert payload["version"] == SNAPSHOT_VERSION


def test_snapshot_to_book_refuses_wrong_version():
    with pytest.raises(DepthPricingError, match="version mismatch"):
        snapshot_to_book(
            {"version": SNAPSHOT_VERSION + 99, "symbol": "AAPL",
             "bids": [], "asks": []}
        )


def test_snapshot_to_book_refuses_missing_symbol():
    with pytest.raises(DepthPricingError, match="symbol"):
        snapshot_to_book(
            {"version": SNAPSHOT_VERSION, "symbol": "", "bids": [], "asks": []}
        )


def test_snapshot_to_book_refuses_malformed_level():
    payload = book_to_snapshot(_synthetic_book())
    payload["bids"][0] = {"price": "not-a-number", "size": 10.0}
    with pytest.raises(DepthPricingError, match="invalid price/size"):
        snapshot_to_book(payload)


def test_read_snapshot_refuses_missing_file(tmp_path):
    with pytest.raises(DepthPricingError, match="snapshot not found"):
        read_snapshot(tmp_path / "nope.json")


def test_read_snapshot_refuses_corrupt_json(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text("{ this is not json")
    with pytest.raises(DepthPricingError, match="unreadable"):
        read_snapshot(path)


def test_read_snapshot_refuses_non_object_payload(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(DepthPricingError):
        read_snapshot(path)


# ---------------------------------------------------------------------------
# Attribution surface (NOTICE / Apache-2.0)
# ---------------------------------------------------------------------------


def test_module_docstring_carries_attribution():
    """Acceptance criterion: Apache-2.0 attribution header + NOTICE block.

    We assert on the file source rather than __doc__ because runtime
    docstring trimming would silently let attribution rot.
    """
    src = Path("src/equity/depth_pricing.py").read_text(encoding="utf-8")
    assert "Apache" in src
    assert "NOTICE" in src
    assert "hummingbot" in src.lower()
    assert "order_book.pyx" in src


def test_root_notice_file_attributes_hummingbot():
    """The repo-root NOTICE file must call out the hummingbot derivation
    so the Apache-2.0 attribution is discoverable without grepping
    individual source files."""
    notice = Path("NOTICE")
    assert notice.exists(), "NOTICE file missing at repo root"
    text = notice.read_text(encoding="utf-8")
    assert "hummingbot" in text.lower()
    assert "Apache" in text
    assert "order_book.pyx" in text


# ---------------------------------------------------------------------------
# No-bare-except / no-mock self-check (matches kill_switch / rebalance)
# ---------------------------------------------------------------------------


def test_module_has_no_bare_except():
    src = Path("src/equity/depth_pricing.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("except:"), (
            f"bare 'except:' in depth_pricing.py: {line!r}"
        )
        assert "except Exception: pass" not in stripped, (
            f"silent 'except Exception: pass' in depth_pricing.py: {line!r}"
        )


def test_test_file_uses_no_mocks():
    """This test asserts on the test file itself — by reading the source
    rather than its own docstring so a literal `unittest.mock` in a
    docstring would still be caught."""
    src = Path("tests/test_equity_depth_pricing.py").read_text(encoding="utf-8")
    forbidden = (
        "from unittest.mock",
        "import unittest.mock",
        "MagicMock(",
        "mock.patch(",
        "@patch(",
    )
    for needle in forbidden:
        # The strings in this tuple are themselves needles; if any appear
        # OUTSIDE this `forbidden = (...)` literal, the test file is using
        # mocks. We count occurrences and require exactly one (the
        # literal here in the assertion list).
        count = src.count(needle)
        assert count <= 1, (
            f"test_equity_depth_pricing.py appears to use {needle!r} "
            f"({count} occurrences — should be at most 1, in the forbidden "
            f"list itself)"
        )


# ---------------------------------------------------------------------------
# Sanity: math.isfinite checks behave under nan/inf
# ---------------------------------------------------------------------------


def test_make_book_propagates_invariants():
    with pytest.raises(DepthPricingError):
        make_book("AAPL", bids=[(math.nan, 1.0)], asks=[])
    with pytest.raises(DepthPricingError):
        make_book("AAPL", bids=[], asks=[(100.0, math.inf)])
