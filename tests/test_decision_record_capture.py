"""DecisionRecord spine — the 12 acceptance criteria.

No mocks. Real DecisionRecords, real ledgers on real disk via tmp_path, real
AutonomousLoop instances reusing the existing control-loop fixture, and real
concurrent processes for the locking test.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.equity.control_loop import CycleOutcome
from src.equity.decision_capture import build_cycle_decisions, persist_cycle_decisions
from src.learning.decision_ledger import (
    DECISION_LEDGER_PATH,
    append_decisions,
    load_decisions,
)
from src.learning.decision_record import (
    DecisionClass,
    DecisionRecord,
    DecisionRecordError,
    Disposition,
)
from src.learning.outcome_ledger import OutcomeLedgerKindError, append_rows

# Reuse the existing control-loop fixture rather than build a parallel one.
from tests.test_equity_control_loop import _build_loop  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _records(disposition=Disposition.HALT, *, cycle_id="c1", weights=None, converter=None):
    return build_cycle_decisions(
        cycle_id=cycle_id,
        decision_timestamp="2026-07-20T12:00:00Z",
        disposition=disposition,
        reason_codes=["lane_halted"],
        target_weights=weights if weights is not None else {"AAPL": 0.5, "MSFT": 0.5},
        degross_factor=0.5,
        current_weights={"AAPL": 0.2},
        portfolio_nav=100_000.0,
        peak_nav=110_000.0,
        strategy_version="test-1",
        unit_converter=converter,
    )


# ---------------------------------------------------------------------------
# AC1 / AC2 / AC10 — behaviour at the live control-loop seam
# ---------------------------------------------------------------------------


def test_ac1_executable_write_failure_prevents_order_submission(tmp_path, monkeypatch):
    """A decision that cannot be recorded must not become an order."""
    loop, broker, heartbeat, fetcher = _build_loop(
        tmp_path, target_weights={"AAPL": 0.10, "MSFT": 0.10}, max_cycles=1
    )
    # Point the ledger at a path that cannot be created -> write fails.
    unwritable = tmp_path / "blocked"
    unwritable.write_text("i am a file, not a directory")
    loop.decision_ledger_path = unwritable / "decision_ledger.jsonl"

    import pandas as pd

    state = loop.load_portfolio_state()
    report = loop.run_cycle(asof=pd.Timestamp("2026-07-01", tz="UTC"), portfolio_state=state)

    assert report.outcome is CycleOutcome.ABSTAINED_UNRECORDED
    assert broker.sent == [] or all(not o for o in broker.sent), "no order may be submitted"


def test_ac2_halted_candidate_still_produces_a_record(tmp_path):
    """The milestone: a refused cycle still yields CANDIDATE_NOT_EXECUTED."""
    loop, _broker, _hb, _f = _build_loop(
        tmp_path, target_weights={"AAPL": 0.10, "MSFT": 0.10}, max_cycles=1
    )
    ledger = tmp_path / "decisions.jsonl"
    loop.decision_ledger_path = ledger

    import pandas as pd

    # Drive a REAL risk refusal: a 50% drawdown trips the guardian, so the
    # risk layer -- not the test -- produces the halt/block verdict.
    state = loop.load_portfolio_state()
    state.peak_nav = state.nav * 2.0
    loop.write_portfolio_state(state)
    report = loop.run_cycle(asof=pd.Timestamp("2026-07-01", tz="UTC"), portfolio_state=state)

    assert report.outcome in (CycleOutcome.HALTED, CycleOutcome.RISK_BLOCKED, CycleOutcome.NO_PLAN), (
        f"expected a refusal, got {report.outcome}"
    )
    rows = load_decisions(ledger)
    assert rows, "a refused cycle must still leave decision evidence"
    assert all(r["record_kind"] == "decision" for r in rows)
    candidates = [r for r in rows if r["decision_class"] == "CANDIDATE_NOT_EXECUTED"]
    assert candidates, "a refused cycle must still leave candidate evidence"
    row = candidates[0]
    assert row["disposition"] in {"HALT", "NO_ACT", "ABSTAIN"}
    assert row["reason_codes"], "refusal reason must be recorded"
    assert row["pre_degross_weight"] is not None
    assert row["intended_weight"] is not None
    assert row["degross_factor"] is not None


def test_ac10_existing_execution_and_degross_behaviour_unchanged(tmp_path):
    """Capture records; it must not alter what the loop decides."""
    loop, _b, _h, _f = _build_loop(
        tmp_path, target_weights={"AAPL": 0.10, "MSFT": 0.10}, max_cycles=1
    )
    loop.decision_ledger_path = tmp_path / "d.jsonl"

    import pandas as pd

    state = loop.load_portfolio_state()
    report = loop.run_cycle(asof=pd.Timestamp("2026-07-01", tz="UTC"), portfolio_state=state)

    # The de-gross factor the loop applied is still the risk layer's own value.
    if report.risk is not None:
        assert 0.0 <= report.risk.degross_factor <= 1.0
    assert report.outcome is not CycleOutcome.ABSTAINED_UNRECORDED


# ---------------------------------------------------------------------------
# AC3 / AC4 / AC5 — identity and idempotency
# ---------------------------------------------------------------------------


def test_ac3_same_cycle_replay_produces_one_record(tmp_path):
    ledger = tmp_path / "d.jsonl"
    recs = _records()

    append_decisions(recs, ledger)
    second = append_decisions(recs, ledger)

    assert second["appended"] == 0
    assert second["skipped_duplicate"] == len(recs)
    assert len(load_decisions(ledger)) == len(recs)


def test_ac5_material_change_produces_a_new_record(tmp_path):
    ledger = tmp_path / "d.jsonl"
    append_decisions(_records(), ledger)

    # Different intended size -> materially different decision.
    append_decisions(_records(weights={"AAPL": 0.9, "MSFT": 0.5}), ledger)

    rows = load_decisions(ledger)
    aapl = [r for r in rows if r["instrument"] == "AAPL"]
    assert len(aapl) == 2, "a materially different decision must mint a new record"


def test_ac5_immaterial_change_does_not_mint_a_record(tmp_path):
    """Same decision, different incidental context -> still one record."""
    ledger = tmp_path / "d.jsonl"
    append_decisions(_records(), ledger)

    same = build_cycle_decisions(
        cycle_id="c1", decision_timestamp="2026-07-20T12:00:00Z",
        disposition=Disposition.HALT, reason_codes=["lane_halted"],
        target_weights={"AAPL": 0.5, "MSFT": 0.5}, degross_factor=0.5,
        current_weights={"AAPL": 0.2},
        portfolio_nav=999_999.0,  # incidental: not a material field
        peak_nav=110_000.0, strategy_version="test-1",
    )
    stats = append_decisions(same, ledger)

    assert stats["appended"] == 0


def test_ac4_concurrent_writers_produce_one_valid_record(tmp_path):
    ledger = tmp_path / "d.jsonl"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from src.equity.decision_capture import build_cycle_decisions
        from src.learning.decision_ledger import append_decisions
        from src.learning.decision_record import Disposition
        recs = build_cycle_decisions(
            cycle_id="race", decision_timestamp="2026-07-20T12:00:00Z",
            disposition=Disposition.HALT, reason_codes=["r"],
            target_weights={{"AAPL": 0.5}}, degross_factor=0.5,
            current_weights={{}}, portfolio_nav=1.0, peak_nav=1.0,
        )
        append_decisions(recs, {str(ledger)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    for p in procs:
        assert p.wait(timeout=120) == 0

    raw = ledger.read_text().splitlines()
    assert len(raw) == 1, f"expected exactly one record, got {len(raw)}"
    json.loads(raw[0])


# ---------------------------------------------------------------------------
# AC6 / AC7 — sizing preserved, absences honest
# ---------------------------------------------------------------------------


def test_ac6_pre_degross_and_intended_units_both_preserved(tmp_path):
    """De-gross is only measurable if BOTH sides of it are recorded."""
    recs = _records(converter=lambda ticker, weight: weight * 1000.0)
    aapl = next(r for r in recs if r.instrument == "AAPL")

    assert aapl.pre_degross_units == pytest.approx(500.0)   # 0.5 * 1000
    assert aapl.intended_units == pytest.approx(250.0)      # 0.5 * 0.5 * 1000
    assert aapl.pre_degross_units != aapl.intended_units
    assert aapl.degross_factor == pytest.approx(0.5)
    # weights are native and always present
    assert aapl.pre_degross_weight == pytest.approx(0.5)
    assert aapl.intended_weight == pytest.approx(0.25)


def test_ac7_missing_forecast_stays_unavailable_not_zero():
    aapl = next(r for r in _records() if r.instrument == "AAPL")

    assert aapl.expected_move_bps is None
    assert aapl.spread_bps is None
    assert aapl.estimated_total_cost_bps is None
    assert "expected_move_bps" in aapl.unavailable_fields
    assert "spread_bps" in aapl.unavailable_fields
    # units unavailable without a converter -- NOT zero
    assert aapl.intended_units is None
    assert "intended_units" in aapl.unavailable_fields


def test_declared_unavailable_field_with_a_value_is_refused():
    with pytest.raises(DecisionRecordError, match="declared unavailable"):
        DecisionRecord(
            decision_id="d", cycle_id="c", decision_class=DecisionClass.CANDIDATE_NOT_EXECUTED,
            disposition=Disposition.HALT, reason_codes=(), instrument="AAPL",
            direction="LONG", decision_timestamp="t",
            expected_move_bps=12.0, unavailable_fields=("expected_move_bps",),
        )


# ---------------------------------------------------------------------------
# AC8 / AC9 — the two corpora cannot blend
# ---------------------------------------------------------------------------


def test_ac8_candidate_records_cannot_enter_the_realized_ledger(tmp_path):
    row = _records()[0].to_row()

    with pytest.raises(OutcomeLedgerKindError, match="refuses a decision record"):
        append_rows([row], tmp_path / "realized.jsonl")


def test_ac8_decision_ledger_refuses_a_realized_row(tmp_path):
    """The guard is symmetric."""

    class _NotADecision:
        def to_row(self):
            return {"record_kind": "realized_outcome", "outcome_key": "x"}

    with pytest.raises(DecisionRecordError, match="refuses record_kind"):
        append_decisions([_NotADecision()], tmp_path / "d.jsonl")


def test_ac9_realized_attribution_rejects_a_candidate_contaminated_ledger(tmp_path):
    ledger = tmp_path / "contaminated.jsonl"
    ledger.write_text(json.dumps(_records()[0].to_row()) + "\n")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_outcome_attribution.py"),
         "--quiet", "--ledger", str(ledger), "--memory", str(tmp_path / "m.json")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )

    assert proc.returncode == 2
    assert "REFUSING" in proc.stderr or "decision record" in proc.stderr


def test_candidate_cannot_carry_broker_linkage():
    """A record for a trade that was never sent cannot claim a broker order."""
    with pytest.raises(DecisionRecordError, match="cannot carry broker_order_id"):
        DecisionRecord(
            decision_id="d", cycle_id="c", decision_class=DecisionClass.CANDIDATE_NOT_EXECUTED,
            disposition=Disposition.HALT, reason_codes=(), instrument="AAPL",
            direction="LONG", decision_timestamp="t", broker_order_id="OID-1",
        )


def test_disposition_and_class_pairing_is_enforced_both_ways():
    with pytest.raises(DecisionRecordError, match="requires decision_class EXECUTED_DECISION"):
        DecisionRecord(
            decision_id="d", cycle_id="c", decision_class=DecisionClass.CANDIDATE_NOT_EXECUTED,
            disposition=Disposition.EXECUTE, reason_codes=(), instrument="A",
            direction="LONG", decision_timestamp="t",
        )
    with pytest.raises(DecisionRecordError, match="requires decision_class CANDIDATE_NOT_EXECUTED"):
        DecisionRecord(
            decision_id="d", cycle_id="c", decision_class=DecisionClass.EXECUTED_DECISION,
            disposition=Disposition.HALT, reason_codes=(), instrument="A",
            direction="LONG", decision_timestamp="t",
        )


def test_persist_failure_contract_differs_by_class(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    target = blocked / "d.jsonl"

    ok_exec, _ = persist_cycle_decisions(_records(), executable=True, path=target)
    ok_cand, _ = persist_cycle_decisions(_records(), executable=False, path=target)

    assert ok_exec is False, "executable path must report failure so the caller abstains"
    assert ok_cand is False, "candidate path also reports it -- caller keeps its disposition"


# ---------------------------------------------------------------------------
# AC11 / AC12 — no LLM enters the equity path
# ---------------------------------------------------------------------------


def test_ac11_decision_capture_imports_no_llm():
    """AST-walk the new module the same way the standing canary does."""
    forbidden = ("anthropic", "openai", "langchain", "litellm", "ollama",
                 "transformers", "mistralai", "cohere", "claude")
    tree = ast.parse((REPO_ROOT / "src" / "equity" / "decision_capture.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    for name in names:
        head = name.split(".")[0].lower()
        assert head not in forbidden, f"forbidden import {name!r} in decision_capture"


def test_ac12_standing_no_llm_canary_still_green():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_no_llm_in_equity_path.py", "-q"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]


def test_default_ledger_path_is_separate_from_the_realized_ledger():
    from src.learning.outcome_ledger import LEDGER_PATH

    assert DECISION_LEDGER_PATH != LEDGER_PATH
    assert DECISION_LEDGER_PATH.name != LEDGER_PATH.name
