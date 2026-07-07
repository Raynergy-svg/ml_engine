"""No-mock tests for src.brain_loop.cycle — the per-invocation orchestrator.

Wires monitor -> derisk (breach path) and hypothesis -> gate_runner ->
promotion (research path), on real tmp_path state, with the real StateEngine,
real decide_cycle, and a trivial real subprocess standing in for the harness.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.brain_loop.cycle import run_cycle
from src.brain_loop.hypothesis_registry import read_ledger
from src.brain_loop.promotion import list_requests
from src.scanner.automation.state_engine import StateEngine
from src.scanner.config import ScannerConfig


def _paths(tmp_path):
    return {
        "project_root": tmp_path,
        "state_path": tmp_path / ".claude" / "state.json",
        "ledger_path": tmp_path / "hypotheses.jsonl",
        "requests_dir": tmp_path / "promotion_requests",
    }


def _write_ship_gate(project_root) -> None:
    sg_dir = project_root / "trained_data" / "backtests"
    sg_dir.mkdir(parents=True, exist_ok=True)
    (sg_dir / "SHIP_GATE.json").write_text(json.dumps({"gate_pass": True, "universe_hash": "abc"}))


_HYPOTHESIS = {
    "hypothesis_id": "h-1",
    "name": "test-hypothesis",
    "mechanism": "synthetic, for orchestrator wiring only",
    "novelty_justification": "n/a — unit test",
    "params": {"lookback": 10},
    "harness_cmd": [sys.executable, "-c", "print('ran')"],
}


def _hypothesis_with_real_harness(project_root) -> dict:
    """gate_runner.run_backtest now refuses `-c` inline code (harness_cmd[1] must be
    a real .py file under scripts/) — tests that actually reach run_and_gate need a
    real script on disk, not the bare _HYPOTHESIS constant."""
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / "fake_harness.py"
    script.write_text("print('ran')")
    return {**_HYPOTHESIS, "harness_cmd": [sys.executable, str(script)]}


def test_already_halted_short_circuits_everything(tmp_path):
    p = _paths(tmp_path)
    StateEngine(p["state_path"]).set_halted(True)

    result = run_cycle(
        project_root=p["project_root"], config=ScannerConfig(), ledger_path=p["ledger_path"],
        requests_dir=p["requests_dir"], state_path=p["state_path"], hypothesis=_HYPOTHESIS,
    )

    assert result.halted is True
    assert result.hypothesis_result is None
    assert read_ledger(p["ledger_path"]) == []  # no hypothesis work happened


def test_breach_triggers_derisk_halt_and_skips_hypothesis(tmp_path):
    p = _paths(tmp_path)
    StateEngine(p["state_path"]).set_halted(False)
    portfolio_dir = p["project_root"] / "trained_data" / "equity"
    portfolio_dir.mkdir(parents=True)
    (portfolio_dir / "portfolio_state.json").write_text(
        json.dumps({"nav": 70.0, "peak_nav": 100.0})
    )

    result = run_cycle(
        project_root=p["project_root"], config=ScannerConfig(), ledger_path=p["ledger_path"],
        requests_dir=p["requests_dir"], state_path=p["state_path"], hypothesis=_HYPOTHESIS,
        data_asof=pd.Timestamp.now(tz="UTC"),
    )

    assert result.halted is True
    assert result.breach_derisked is True
    assert StateEngine(p["state_path"]).get_halted() is True
    assert result.hypothesis_result is None
    assert read_ledger(p["ledger_path"]) == []


def test_passing_hypothesis_registers_gates_and_proposes_shadow(tmp_path):
    p = _paths(tmp_path)
    StateEngine(p["state_path"]).set_halted(False)
    _write_ship_gate(p["project_root"])

    result = run_cycle(
        project_root=p["project_root"], config=ScannerConfig(), ledger_path=p["ledger_path"],
        requests_dir=p["requests_dir"], state_path=p["state_path"],
        hypothesis=_hypothesis_with_real_harness(p["project_root"]),
        data_asof=pd.Timestamp.now(tz="UTC"),
    )

    assert result.halted is False
    records = read_ledger(p["ledger_path"])
    kinds = [r["kind"] for r in records]
    assert kinds == ["register", "gate_verdict"]
    assert records[-1]["decision"] == "continue"

    requests = list_requests(p["requests_dir"])
    assert len(requests) == 1
    assert requests[0]["status"] == "APPROVED_SHADOW"
    assert requests[0]["target"] == "shadow"


def test_failing_hypothesis_registers_and_gates_but_never_proposes_promotion(tmp_path):
    p = _paths(tmp_path)
    StateEngine(p["state_path"]).set_halted(False)
    # No SHIP_GATE.json -> NO_ACT.
    result = run_cycle(
        project_root=p["project_root"], config=ScannerConfig(), ledger_path=p["ledger_path"],
        requests_dir=p["requests_dir"], state_path=p["state_path"],
        hypothesis=_hypothesis_with_real_harness(p["project_root"]),
        data_asof=pd.Timestamp.now(tz="UTC"),
    )

    assert result.halted is False
    records = read_ledger(p["ledger_path"])
    assert records[-1]["kind"] == "gate_verdict"
    assert records[-1]["decision"] == "no_act"
    assert list_requests(p["requests_dir"]) == []


def test_monitor_only_cycle_with_no_hypothesis_touches_no_ledger(tmp_path):
    p = _paths(tmp_path)
    StateEngine(p["state_path"]).set_halted(False)
    _write_ship_gate(p["project_root"])

    result = run_cycle(
        project_root=p["project_root"], config=ScannerConfig(), ledger_path=p["ledger_path"],
        requests_dir=p["requests_dir"], state_path=p["state_path"], hypothesis=None,
        data_asof=pd.Timestamp.now(tz="UTC"),
    )

    assert result.halted is False
    assert result.hypothesis_result is None
    assert read_ledger(p["ledger_path"]) == []
    assert result.rails["decision"] == "continue"


def test_never_promotes_live_autonomously():
    """Structural proof: run_cycle's implementation never calls
    propose_promotion with target='live' anywhere in its source."""
    import inspect

    from src.brain_loop import cycle

    source = inspect.getsource(cycle)
    assert 'target="live"' not in source
    assert "target='live'" not in source
