"""US-016 — Shadow fill simulator tests.

No mocks. Drives real :class:`SlippageModel`,
:class:`ShadowFillSimulator`, real :class:`EquityOrder`, real
:class:`L2OrderBook`, real ``tmp_path`` disk for both the SHIP_GATE
fixture and the simulator ledger.

Acceptance criteria covered:

* :func:`enforce_ship_gate` rejects missing / unreadable / non-dict /
  ``gate_pass != True`` / missing-hash / mismatched-hash payloads
  (mirrors the US-012/US-013/US-014/US-015 guard tests).
* Determinism: the same (order, book, model) returns a bytewise
  identical :class:`SimulatedFill`.
* L2 simulator honors participation cap, book thinness, full / partial /
  failed states.
* Quote simulator pays half-spread + slippage on BUY and SELL.
* :class:`ShadowFillSimulator` writes its ledger atomically (mkstemp +
  os.replace), rotates oldest-first when it overflows, and re-enforces
  the gate on every public call (an invalidated gate mid-session fails
  closed at the NEXT fill).
* The on-disk source ``src/equity/shadow_fills.py`` contains zero bare
  ``except:`` clauses (greps the file).
* The test file imports zero ``unittest.mock`` symbols.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pytest

from src.equity.depth_pricing import L2OrderBook, make_book
from src.equity.order_lifecycle import EquityOrder, OrderState
from src.equity.shadow_fills import (
    LEDGER_VERSION,
    ShadowFillError,
    ShadowFillSimulator,
    SimulatedFill,
    SlippageModel,
    enforce_ship_gate,
    simulate_fill_from_book,
    simulate_fill_from_quote,
)


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"
ALT_HASH = "0000000000000000000000000000000000000000000000000000000000000001"


# ---------------------------------------------------------------------------
# Real-disk fixtures (same convention as the other equity test suites).
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".shadow_fills_test_", suffix=".json", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _passing_payload(universe_hash: str = VALID_HASH) -> dict:
    return {
        "asof": "2026-05-29",
        "criteria_source": "test",
        "gate_pass": True,
        "max_dd": 0.229,
        "net_sharpe": 0.921,
        "pipeline_version": "test-v1",
        "positive_years": 13,
        "recommendation": "PASS",
        "thresholds": {
            "max_drawdown": 0.25,
            "min_net_sharpe": 0.4,
            "min_positive_years": 6.0,
            "min_total_years": 10.0,
        },
        "total_years": 17,
        "universe_hash": universe_hash,
    }


def _ship_gate(tmp_path: Path, hash_: str = VALID_HASH) -> Path:
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, _passing_payload(hash_))
    return path


def _order(qty: int = 100, side: str = "BUY", cid: str = "ord-1") -> EquityOrder:
    return EquityOrder(
        client_order_id=cid,
        ticker="AAPL",
        side=side,
        quantity=qty,
    )


def _book(symbol: str = "AAPL") -> L2OrderBook:
    return make_book(
        symbol,
        bids=[(99.95, 200), (99.90, 500), (99.80, 1_000)],
        asks=[(100.05, 200), (100.10, 500), (100.20, 1_000)],
    )


# ---------------------------------------------------------------------------
# Ship-gate guard (AC #3).
# ---------------------------------------------------------------------------


def test_ship_gate_passes_for_matching_hash(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    # Does not raise.
    enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_missing(tmp_path: Path) -> None:
    gate = tmp_path / "no_such.json"
    with pytest.raises(ShadowFillError, match="ship gate not found"):
        enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_unreadable_json(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text("not json {{", encoding="utf-8")
    with pytest.raises(ShadowFillError, match="cannot read"):
        enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_non_object_payload(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ShadowFillError, match="not a JSON object"):
        enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_gate_pass_false(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowFillError, match="gate_pass="):
        enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_gate_pass_truthy_but_not_true(tmp_path: Path) -> None:
    # Must be exactly True; 1 / "true" / "yes" all rejected.
    for truthy in (1, "true", "yes"):
        gate = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = truthy
        _atomic_write_json(gate, payload)
        with pytest.raises(ShadowFillError, match="gate_pass="):
            enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_hash_missing(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["universe_hash"] = ""
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowFillError, match="universe_hash missing/blank"):
        enforce_ship_gate(gate, VALID_HASH)


def test_ship_gate_refuses_when_hash_mismatch(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path, hash_=ALT_HASH)
    with pytest.raises(ShadowFillError, match="universe_hash mismatch"):
        enforce_ship_gate(gate, VALID_HASH)


def test_simulator_constructor_refuses_when_gate_missing(tmp_path: Path) -> None:
    with pytest.raises(ShadowFillError, match="ship gate not found"):
        ShadowFillSimulator(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            ship_gate_path=tmp_path / "no_such.json",
        )


def test_simulator_constructor_refuses_when_gate_pass_false(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowFillError, match="gate_pass="):
        ShadowFillSimulator(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            ship_gate_path=gate,
        )


def test_simulator_constructor_refuses_when_hash_mismatch(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path, hash_=ALT_HASH)
    with pytest.raises(ShadowFillError, match="universe_hash mismatch"):
        ShadowFillSimulator(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            ship_gate_path=gate,
        )


def test_simulator_reenforces_gate_on_every_call(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    sim = ShadowFillSimulator(
        universe_hash=VALID_HASH,
        ledger_path=tmp_path / "ledger.json",
        ship_gate_path=gate,
    )
    # One healthy call.
    sim.fill_from_book(_order(), _book())
    # Flip the gate mid-session — the simulator must refuse the next call.
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowFillError, match="gate_pass="):
        sim.fill_from_book(_order(cid="ord-2"), _book())


# ---------------------------------------------------------------------------
# Determinism (AC #2).
# ---------------------------------------------------------------------------


def test_simulate_fill_from_book_is_deterministic() -> None:
    order = _order(qty=300)
    book = _book()
    model = SlippageModel(extra_slippage_bps=1.5, participation_rate=1.0)

    a = simulate_fill_from_book(order, book, model)
    b = simulate_fill_from_book(order, book, model)
    c = simulate_fill_from_book(order, book, model)

    assert a == b == c, "shadow fill must be deterministic for fixed input"
    assert a.to_dict() == b.to_dict() == c.to_dict()


def test_simulate_fill_from_quote_is_deterministic() -> None:
    order = _order(qty=50, side="SELL")
    model = SlippageModel(
        extra_slippage_bps=2.0, fixed_spread_bps=3.0,
    )
    a = simulate_fill_from_quote(order, reference_price=250.0, model=model)
    b = simulate_fill_from_quote(order, reference_price=250.0, model=model)
    assert a == b


def test_simulate_fill_from_book_returns_known_values_for_fixed_input() -> None:
    # A hand-checked fixed-input fill: BUY 200 shares against a book whose
    # top ask is exactly 200 @ 100.05. With participation=1.0 and zero
    # extra slippage, VWAP == 100.05 and slippage_bps == 10 bps relative
    # to the 100.00 mid.
    order = _order(qty=200, side="BUY", cid="fixed-1")
    book = _book()
    model = SlippageModel(extra_slippage_bps=0.0, participation_rate=1.0)

    fill = simulate_fill_from_book(order, book, model)

    assert fill.fill_quantity == 200
    assert fill.order_state == OrderState.FILLED.value
    assert fill.reason == "clean_full_fill"
    assert fill.fill_price == pytest.approx(100.05, rel=1e-12)
    assert fill.reference_price == pytest.approx(100.00, rel=1e-12)
    assert fill.slippage_bps == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# L2 simulator behavior (AC #1: order + L2 input -> fill with slippage).
# ---------------------------------------------------------------------------


def test_full_fill_marks_filled_state() -> None:
    order = _order(qty=100, side="BUY")
    book = _book()
    model = SlippageModel()
    fill = simulate_fill_from_book(order, book, model)
    assert fill.fill_quantity == 100
    assert fill.order_state == OrderState.FILLED.value
    assert fill.fill_price > fill.reference_price  # BUY pays up


def test_sell_pays_down_from_midprice() -> None:
    order = _order(qty=100, side="SELL")
    book = _book()
    model = SlippageModel()
    fill = simulate_fill_from_book(order, book, model)
    assert fill.fill_quantity == 100
    assert fill.order_state == OrderState.FILLED.value
    assert fill.fill_price < fill.reference_price  # SELL receives less
    assert fill.slippage_bps > 0  # signed bps is positive on SELL too


def test_partial_fill_when_book_too_thin() -> None:
    # Asking for 5_000 shares against a book with 1_700 visible on the
    # ask side -> PARTIAL fill at the visible depth.
    order = _order(qty=5_000)
    book = _book()
    model = SlippageModel(participation_rate=1.0)
    fill = simulate_fill_from_book(order, book, model)
    assert fill.order_state == OrderState.PARTIAL.value
    assert fill.fill_quantity == 1_700
    assert fill.reason == "book_too_thin"


def test_partial_fill_capped_by_participation() -> None:
    # Visible ask depth = 1_700; 25% cap => 425 shares max.
    order = _order(qty=1_000)
    book = _book()
    model = SlippageModel(participation_rate=0.25)
    fill = simulate_fill_from_book(order, book, model)
    assert fill.fill_quantity == 425
    assert fill.order_state == OrderState.PARTIAL.value
    assert fill.reason == "capped_by_participation"


def test_failed_when_empty_side() -> None:
    book = make_book("AAPL", bids=[(99.95, 200)], asks=[])
    order = _order(qty=100, side="BUY")
    model = SlippageModel()
    # No two-sided book: mid_price is None -> FAILED "no_two_sided_book".
    fill = simulate_fill_from_book(order, book, model)
    assert fill.fill_quantity == 0
    assert fill.order_state == OrderState.FAILED.value
    assert fill.reason == "no_two_sided_book"


def test_failed_when_book_exists_but_participation_caps_to_zero() -> None:
    # Tiny visible depth + tiny participation rate -> floor(depth * rate)
    # == 0, which surfaces FAILED with "capped_by_participation_to_zero".
    book = make_book("AAPL", bids=[(99.0, 1)], asks=[(101.0, 1)])
    order = _order(qty=1, side="BUY")
    model = SlippageModel(participation_rate=0.01)
    fill = simulate_fill_from_book(order, book, model)
    assert fill.order_state == OrderState.FAILED.value
    assert fill.reason == "capped_by_participation_to_zero"


# ---------------------------------------------------------------------------
# Quote simulator behavior.
# ---------------------------------------------------------------------------


def test_quote_buy_pays_half_spread_plus_slippage() -> None:
    order = _order(qty=10, side="BUY")
    model = SlippageModel(
        fixed_spread_bps=2.0, extra_slippage_bps=1.0,
    )
    fill = simulate_fill_from_quote(order, reference_price=100.0, model=model)
    assert fill.order_state == OrderState.FILLED.value
    assert fill.fill_price == pytest.approx(100.0 * 1.0003)
    assert fill.slippage_bps == pytest.approx(3.0)


def test_quote_sell_receives_below_reference() -> None:
    order = _order(qty=10, side="SELL")
    model = SlippageModel(
        fixed_spread_bps=2.0, extra_slippage_bps=1.0,
    )
    fill = simulate_fill_from_quote(order, reference_price=100.0, model=model)
    assert fill.order_state == OrderState.FILLED.value
    assert fill.fill_price == pytest.approx(100.0 * 0.9997)
    assert fill.slippage_bps == pytest.approx(3.0)


def test_quote_rejects_invalid_reference_price() -> None:
    order = _order(qty=10)
    model = SlippageModel()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ShadowFillError, match="reference_price"):
            simulate_fill_from_quote(order, reference_price=bad, model=model)


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_slippage_model_rejects_invalid_values() -> None:
    with pytest.raises(ShadowFillError):
        SlippageModel(extra_slippage_bps=-1.0)
    with pytest.raises(ShadowFillError):
        SlippageModel(participation_rate=0.0)
    with pytest.raises(ShadowFillError):
        SlippageModel(participation_rate=1.5)
    with pytest.raises(ShadowFillError):
        SlippageModel(fixed_spread_bps=float("nan"))


def test_simulator_rejects_non_order_input(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    sim = ShadowFillSimulator(
        universe_hash=VALID_HASH,
        ledger_path=tmp_path / "ledger.json",
        ship_gate_path=gate,
    )
    with pytest.raises(ShadowFillError, match="EquityOrder"):
        sim.fill_from_book(object(), _book())  # type: ignore[arg-type]
    with pytest.raises(ShadowFillError, match="L2OrderBook"):
        sim.fill_from_book(_order(), object())  # type: ignore[arg-type]


def test_simulate_from_book_rejects_negative_quantity() -> None:
    book = _book()
    model = SlippageModel()
    with pytest.raises(ShadowFillError, match="quantity"):
        simulate_fill_from_book(
            EquityOrder(
                client_order_id="bad", ticker="AAPL", side="BUY", quantity=-1,
            ),
            book,
            model,
        )


# ---------------------------------------------------------------------------
# Atomic ledger persistence + rotation (AC #7).
# ---------------------------------------------------------------------------


def test_ledger_is_written_atomically(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    ledger = tmp_path / "ledger.json"
    sim = ShadowFillSimulator(
        universe_hash=VALID_HASH,
        ledger_path=ledger,
        ship_gate_path=gate,
    )
    sim.fill_from_book(_order(), _book())
    sim.fill_from_book(_order(cid="ord-2", side="SELL"), _book())

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["version"] == LEDGER_VERSION
    assert payload["universe_hash"] == VALID_HASH
    assert isinstance(payload["entries"], list)
    assert len(payload["entries"]) == 2
    # No stray tempfiles left behind (mkstemp + os.replace contract).
    stragglers = [
        p for p in ledger.parent.iterdir()
        if p.name.endswith(".tmp") and p.name.startswith(".")
    ]
    assert stragglers == []


def test_ledger_rotates_oldest_first_when_full(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    sim = ShadowFillSimulator(
        universe_hash=VALID_HASH,
        ledger_path=tmp_path / "ledger.json",
        ship_gate_path=gate,
        max_ledger_entries=3,
    )
    for i in range(5):
        sim.fill_from_book(_order(cid=f"ord-{i}"), _book())

    entries = sim.entries()
    assert len(entries) == 3
    # Oldest 2 evicted; we keep ord-2 .. ord-4 in order.
    assert [e["client_order_id"] for e in entries] == ["ord-2", "ord-3", "ord-4"]


def test_ledger_round_trip_via_disk(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    ledger = tmp_path / "ledger.json"
    sim = ShadowFillSimulator(
        universe_hash=VALID_HASH,
        ledger_path=ledger,
        ship_gate_path=gate,
    )
    sim.fill_from_book(_order(cid="ord-1"), _book())
    sim.fill_from_quote(_order(cid="ord-2", side="SELL"), 100.0)

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    cids = [e["client_order_id"] for e in payload["entries"]]
    assert cids == ["ord-1", "ord-2"]
    # SimulatedFill JSON round-trips through dataclass asdict.
    sf = SimulatedFill(**payload["entries"][0])
    assert sf.client_order_id == "ord-1"


# ---------------------------------------------------------------------------
# Source-level guards (no bare except, no mocks).
# ---------------------------------------------------------------------------


def test_module_source_has_no_bare_except() -> None:
    source = Path("src/equity/shadow_fills.py").read_text(encoding="utf-8")
    # Strip comments + docstrings before matching.
    stripped = re.sub(r"#.*", "", source)
    stripped = re.sub(r"\"\"\".*?\"\"\"", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"'''.*?'''", "", stripped, flags=re.DOTALL)
    assert not re.search(r"^\s*except\s*:", stripped, re.MULTILINE), (
        "src/equity/shadow_fills.py contains a bare except clause"
    )
    assert not re.search(
        r"except\s+Exception\s*:\s*pass", stripped
    ), "src/equity/shadow_fills.py contains a silent Exception swallow"


def test_test_file_has_no_mock_imports() -> None:
    # The forbidden-tokens self-check looks only at import lines so the
    # test body can legally describe the no-mock contract in prose.
    source = Path(__file__).read_text(encoding="utf-8")
    import_lines = [
        ln for ln in source.splitlines()
        if ln.startswith("import ") or ln.startswith("from ")
    ]
    joined = "\n".join(import_lines)
    assert "unittest.mock" not in joined, (
        "test file imports from unittest.mock — forbidden by no-mock policy"
    )
    assert "import mock" not in joined
    assert "from mock" not in joined
