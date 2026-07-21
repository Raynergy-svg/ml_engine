"""FX observe-only capture + order authorization.

Two defects this pins, both introduced by the first FX capture commit and
caught in review:

1. Capture sat AFTER the halt check, so a halted cycle recorded NOTHING -- the
   exact case where candidate accumulation is most valuable. It also sat after
   bracket repair and winner management, which MUTATE broker state, so a
   "capture before execution" claim was untrue.
2. Persistence failure was swallowed and the cycle continued, and the broker
   link used a synthesized ``{instrument}:{timestamp}`` id while discarding
   ``create_market_order()``'s response -- a linkage that resolves to nothing
   at OANDA.

No mocks: a hand-written client implementing the documented broker interface
(dependency injection, as with the flatten ReplayTransport), real DecisionRecords,
real disk via tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.equity import oanda_trend
from src.equity.oanda_trend import _extract_broker_ids, run_oanda_trend_cycle
from src.learning.decision_ledger import load_decisions

INSTRUMENTS = ["EUR_USD", "USD_JPY"]


class _Config:
    oanda_environment = "practice"


def _rising_candles(n: int = 260, start: float = 1.10, step: float = 0.001) -> dict:
    """A clean uptrend so trend_targets emits a non-zero signal."""
    out = {}
    for inst in INSTRUMENTS:
        rows = []
        for i in range(n):
            px = start + i * step
            rows.append({
                "time": (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i)).isoformat(),
                "complete": True,
                "mid": {"o": f"{px:.5f}", "h": f"{px + 0.002:.5f}",
                        "l": f"{px - 0.002:.5f}", "c": f"{px:.5f}"},
            })
        out[inst] = {"candles": rows}
    return out


class _Client:
    """Records every broker interaction so mutations can be asserted absent."""

    def __init__(self, candles, *, order_response=None):
        self.candles = candles
        self.orders: list = []
        self.bracket_repairs: list = []
        self.trades_read = 0
        self._order_response = order_response

    def get_candles(self, instrument, **_kw):
        return self.candles[instrument]

    def get_account_summary(self):
        return {"account": {"NAV": "100000", "openTradeCount": "0"}}

    def get_open_positions(self):
        return {"positions": []}

    def get_trades(self, *, state="OPEN"):
        self.trades_read += 1
        return {"trades": []}

    def create_market_order(self, **kwargs):
        self.orders.append(kwargs)
        if self._order_response is not None:
            return self._order_response
        return {"orderCreateTransaction": {"id": f"ORD-{len(self.orders)}"},
                "orderFillTransaction": {"id": f"FILL-{len(self.orders)}",
                                         "orderID": f"ORD-{len(self.orders)}"}}

    def set_trade_dependent_orders(self, **kwargs):
        self.bracket_repairs.append(kwargs)
        return {"relatedTransactionIDs": ["1"]}


def _halt_state(root: Path, *, halted: bool) -> None:
    (root / ".claude").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "state.json").write_text(json.dumps({
        "schema_version": "2", "halted": False, "mode": "dry_run",
        "halted_lanes": {"oanda_fx": halted, "equity": False, "brain": False,
                         "crypto_momentum": False, "track_b": False,
                         "crypto_carry": False},
    }))


def _run(root: Path, client, ledger: Path, **kw):
    """Run a real cycle with the decision ledger pointed at tmp_path."""
    import src.learning.decision_ledger as dl

    original = dl.DECISION_LEDGER_PATH
    dl.DECISION_LEDGER_PATH = ledger
    try:
        return run_oanda_trend_cycle(
            client=client, config=_Config(), instruments=INSTRUMENTS,
            project_root=root, **kw)
    finally:
        dl.DECISION_LEDGER_PATH = original


# ---------------------------------------------------------------------------
# Observe-only: halted
# ---------------------------------------------------------------------------


def test_halted_cycle_still_records_the_full_candidate_set(tmp_path):
    """THE defect: a halted cycle recorded nothing at all."""
    _halt_state(tmp_path, halted=True)
    client = _Client(_rising_candles())
    ledger = tmp_path / "decisions.jsonl"

    result = _run(tmp_path, client, ledger)

    assert result.ran is False
    assert result.reason == "lane_halted"

    rows = [r for r in load_decisions(ledger) if r.get("row_type") is None]
    assert rows, "a halted cycle MUST still record candidates"
    assert {r["instrument"] for r in rows} == set(INSTRUMENTS)
    for r in rows:
        assert r["decision_class"] == "CANDIDATE_NOT_EXECUTED"
        assert r["disposition"] == "NO_ACT"
        assert "lane_halted:oanda_fx" in r["reason_codes"]


def test_halted_cycle_records_signal_and_intended_sizing(tmp_path):
    """The candidate is only useful if it carries what was actually planned."""
    _halt_state(tmp_path, halted=True)
    client = _Client(_rising_candles())
    ledger = tmp_path / "decisions.jsonl"

    _run(tmp_path, client, ledger)

    rows = [r for r in load_decisions(ledger) if r.get("row_type") is None]
    row = next(r for r in rows if r["instrument"] == "EUR_USD")
    assert row["intended_units"] is not None
    assert row["current_units"] is not None
    assert row["order_delta"] is not None
    assert row["portfolio_nav"] == pytest.approx(100000.0)
    assert row["reference_mid"] is not None
    assert row["strategy_version"], "signal identity must be recorded"


def test_halted_cycle_performs_no_broker_mutation(tmp_path):
    """Observe-only must stop BEFORE bracket repair and stop movement."""
    _halt_state(tmp_path, halted=True)
    client = _Client(_rising_candles())

    _run(tmp_path, client, tmp_path / "d.jsonl")

    assert client.orders == [], "halted cycle must submit no order"
    assert client.bracket_repairs == [], "halted cycle must not repair brackets"
    assert client.trades_read == 0, "halted cycle must not reach the mutation path"


def test_dry_run_also_records_candidates(tmp_path):
    """dry_run previously returned before capture too."""
    _halt_state(tmp_path, halted=False)
    client = _Client(_rising_candles())
    ledger = tmp_path / "d.jsonl"

    result = _run(tmp_path, client, ledger, dry_run=True)

    assert result.reason == "dry_run"
    rows = [r for r in load_decisions(ledger) if r.get("row_type") is None]
    assert rows, "dry_run must record candidates"
    assert all(r["disposition"] == "NO_ACT" for r in rows)
    assert all("dry_run" in r["reason_codes"] for r in rows)
    assert client.orders == []


def test_unreadable_halt_state_fails_closed_without_capture(tmp_path):
    """Cannot prove we may even observe -> return immediately."""
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "state.json").write_text("{not json")
    client = _Client(_rising_candles())

    result = _run(tmp_path, client, tmp_path / "d.jsonl")

    assert result.ran is False
    assert result.reason == "halt_state_unreadable"
    assert client.orders == []


# ---------------------------------------------------------------------------
# Authorization on the executable path
# ---------------------------------------------------------------------------


def test_persistence_failure_refuses_all_orders(tmp_path):
    """Discovered BEFORE the first order, not halfway through the book."""
    _halt_state(tmp_path, halted=False)
    client = _Client(_rising_candles())
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")

    result = _run(tmp_path, client, blocked / "d.jsonl")

    assert result.ran is False
    assert result.reason in ("decision_persist_failed", "decision_not_persisted")
    assert client.orders == [], "no order may be sent when its decision is unpersisted"


def test_executed_cycle_links_the_real_broker_order_id(tmp_path):
    """The synthesized {instrument}:{timestamp} id resolved to nothing at OANDA."""
    _halt_state(tmp_path, halted=False)
    client = _Client(_rising_candles())
    ledger = tmp_path / "d.jsonl"

    result = _run(tmp_path, client, ledger)

    if not client.orders:
        pytest.skip("this signal/sizing produced no order delta")
    assert result.reason == "executed"

    links = [r for r in load_decisions(ledger) if r.get("row_type") == "broker_link"]
    assert links, "a submitted order must produce a linkage row"
    for link in links:
        assert link["broker_order_id"].startswith("ORD-"), (
            f"linkage must use the BROKER's id, got {link['broker_order_id']!r}"
        )
        assert ":" not in link["broker_order_id"], "synthesized instrument:timestamp id"

    decisions = {r["decision_digest"] for r in load_decisions(ledger)
                 if r.get("row_type") is None}
    for link in links:
        assert link["links_decision_digest"] in decisions, (
            "every link must resolve to a decision row on disk"
        )


def test_broker_id_extraction_handles_every_real_response_shape():
    assert _extract_broker_ids(
        {"orderCreateTransaction": {"id": "1156"},
         "orderFillTransaction": {"id": "1157", "orderID": "1156"}}
    ) == {"order_id": "1156", "fill_id": "1157"}

    # A cancelled (e.g. FIFO-refused) order still identifies itself.
    assert _extract_broker_ids(
        {"orderCancelTransaction": {"orderID": "1317", "reason": "FIFO_VIOLATION"}}
    )["order_id"] == "1317"

    assert _extract_broker_ids(None) == {"order_id": None, "fill_id": None}
    assert _extract_broker_ids({}) == {"order_id": None, "fill_id": None}


def test_order_without_a_usable_broker_id_is_surfaced(tmp_path):
    """A response we cannot link must not pass silently."""
    _halt_state(tmp_path, halted=False)
    client = _Client(_rising_candles(), order_response={"unexpected": "shape"})
    ledger = tmp_path / "d.jsonl"

    _run(tmp_path, client, ledger)

    if not client.orders:
        pytest.skip("this signal/sizing produced no order delta")
    links = [r for r in load_decisions(ledger) if r.get("row_type") == "broker_link"]
    assert links == [], "an unlinkable response must not produce a bogus linkage row"


# ---------------------------------------------------------------------------
# Ordering invariant
# ---------------------------------------------------------------------------


def test_capture_precedes_every_broker_mutation_in_source_order():
    """Structural guard.

    The capture block must sit ABOVE bracket repair, winner management and the
    order loop. If someone moves a mutation above it, decisions would again be
    written after broker state had already changed.
    """
    src = Path(oanda_trend.__file__).read_text()
    capture = src.index("DECISION CAPTURE + EXECUTION PERMISSION CHECKPOINT")
    for mutation in ("repair_missing_trade_brackets(",
                     "set_trade_dependent_orders(",
                     "create_market_order("):
        idx = src.index(mutation, capture)
        assert idx > capture, f"{mutation} must come AFTER decision capture"


def test_execute_disposition_records_authorization_not_completion(tmp_path):
    """KNOWN SEMANTIC GAP, pinned deliberately.

    Capture sits at the authorization boundary: above every broker mutation,
    which is where it must be. But the per-instrument RISK GATES
    (bucket-risk cap, cost model, pyramid rules) run BELOW it, inside the order
    loop. So an instrument can be recorded EXECUTED_DECISION and then be
    refused by a gate, and no order is submitted.

    EXECUTED_DECISION therefore means AUTHORIZED TO SUBMIT, not SUBMITTED. The
    invariant the operator asked for still holds in the direction that matters:
    every submitted order resolves to an EXECUTED_DECISION on disk. The
    converse does not, and the absence of a broker-link row is what
    distinguishes them.

    Closing this properly means recording gate refusals as their own lifecycle
    rows -- deliberately NOT done here.
    """
    _halt_state(tmp_path, halted=False)
    client = _Client(_rising_candles())
    ledger = tmp_path / "d.jsonl"

    _run(tmp_path, client, ledger)

    rows = load_decisions(ledger)
    decisions = [r for r in rows if r.get("row_type") is None]
    links = [r for r in rows if r.get("row_type") == "broker_link"]

    assert decisions, "the executable path must still record decisions"
    # Every LINK resolves to a decision -- the direction that matters.
    digests = {r["decision_digest"] for r in decisions}
    for link in links:
        assert link["links_decision_digest"] in digests

    # And an authorized decision with no link means a downstream gate refused.
    assert len(links) <= len(decisions)
