"""No-mock tests for src.brain_loop.gate_runner.

Real ScannerConfig, real decide_cycle() (from src.equity.decision_gate), real
subprocess execution (trivial commands — the actual backtest harness is far too
slow for a unit test; gate_runner is a thin wrapper and its own correctness
does not depend on which command it shells out to).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from src.brain_loop.gate_runner import evaluate_gate, run_and_gate, run_backtest
from src.brain_loop.hypothesis_registry import read_ledger, register_hypothesis
from src.equity.decision_gate import EquityDecision
from src.scanner.config import ScannerConfig


def _write_state(project_root, halted: bool) -> None:
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "state.json").write_text(json.dumps({"halted": halted}))


def test_evaluate_gate_refuses_when_halted(tmp_path):
    _write_state(tmp_path, halted=True)
    decision = evaluate_gate(config=ScannerConfig(), project_root=tmp_path)
    assert decision.decision is EquityDecision.REFUSE


def test_evaluate_gate_no_act_when_ship_gate_missing(tmp_path):
    _write_state(tmp_path, halted=False)
    decision = evaluate_gate(config=ScannerConfig(), project_root=tmp_path)
    assert decision.decision is EquityDecision.NO_ACT


def test_evaluate_gate_continue_when_ship_gate_passes(tmp_path):
    _write_state(tmp_path, halted=False)
    sg_dir = tmp_path / "trained_data" / "backtests"
    sg_dir.mkdir(parents=True)
    (sg_dir / "SHIP_GATE.json").write_text(json.dumps({"gate_pass": True, "universe_hash": "abc"}))
    decision = evaluate_gate(
        config=ScannerConfig(), project_root=tmp_path, data_asof=pd.Timestamp.now(tz="UTC")
    )
    assert decision.decision is EquityDecision.CONTINUE


def test_run_backtest_executes_real_subprocess(tmp_path):
    result = run_backtest([sys.executable, "-c", "print('hello-brain-loop')"], cwd=tmp_path)
    assert result.returncode == 0
    assert "hello-brain-loop" in result.stdout


def test_run_backtest_times_out_on_a_hanging_process(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_backtest(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            timeout=0.2,
        )


def test_run_and_gate_records_pass_verdict(tmp_path):
    _write_state(tmp_path, halted=False)
    sg_dir = tmp_path / "trained_data" / "backtests"
    sg_dir.mkdir(parents=True)
    (sg_dir / "SHIP_GATE.json").write_text(json.dumps({"gate_pass": True, "universe_hash": "abc"}))

    ledger_path = tmp_path / "hypotheses.jsonl"
    register_hypothesis(
        ledger_path, hypothesis_id="h-1", name="x", mechanism="m",
        novelty_justification="n", params={"a": 1},
    )

    summary = run_and_gate(
        hypothesis_id="h-1",
        ledger_path=ledger_path,
        harness_cmd=[sys.executable, "-c", "print('ran')"],
        config=ScannerConfig(),
        project_root=tmp_path,
        data_asof=pd.Timestamp.now(tz="UTC"),
    )
    assert summary["harness_returncode"] == 0
    assert summary["decision"]["decision"] == "continue"

    records = read_ledger(ledger_path)
    verdicts = [r for r in records if r["kind"] == "gate_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["decision"] == "continue"


def test_run_and_gate_records_fail_verdict_when_ship_gate_missing(tmp_path):
    _write_state(tmp_path, halted=False)
    ledger_path = tmp_path / "hypotheses.jsonl"
    register_hypothesis(
        ledger_path, hypothesis_id="h-1", name="x", mechanism="m",
        novelty_justification="n", params={"a": 1},
    )

    summary = run_and_gate(
        hypothesis_id="h-1",
        ledger_path=ledger_path,
        harness_cmd=[sys.executable, "-c", "print('ran')"],
        config=ScannerConfig(),
        project_root=tmp_path,
    )
    assert summary["decision"]["decision"] == "no_act"


def test_run_and_gate_refuses_unregistered_hypothesis(tmp_path):
    _write_state(tmp_path, halted=False)
    ledger_path = tmp_path / "hypotheses.jsonl"
    from src.brain_loop.hypothesis_registry import HypothesisNotRegistered

    with pytest.raises(HypothesisNotRegistered):
        run_and_gate(
            hypothesis_id="never-registered",
            ledger_path=ledger_path,
            harness_cmd=[sys.executable, "-c", "print('ran')"],
            config=ScannerConfig(),
            project_root=tmp_path,
        )
