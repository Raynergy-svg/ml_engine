"""Pillar 3 — objective per-cycle decision gate for the equity harvester.

`decide_cycle` re-derives CONTINUE / ABSTAIN / NO_ACT / HALT / REFUSE purely
from disk — global halt, ship-gate status, data freshness, portfolio drawdown —
so the harvester's per-cycle action is driven by observable reality, never by
the runner's self-report. Fail-closed, most-severe-first precedence.

No mocks: real `StateEngine`, real `PortfolioState`, real `SHIP_GATE.json`, real
disk via `tmp_path`. Each test is a synthetic case proving one branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.equity.decision_gate import CycleDecision, EquityDecision, decide_cycle
from src.equity.risk_agents import PortfolioState
from src.scanner.automation.state_engine import StateEngine
from src.scanner.config import ScannerConfig

UHASH = "deadbeef" * 8
_GATE_REL = "trained_data/backtests/SHIP_GATE.json"
_PSTATE_REL = "trained_data/equity/portfolio_state.json"
NOW = pd.Timestamp("2026-06-24T00:00:00Z")


def _write_ship_gate(root: Path, *, gate_pass: bool = True, universe_hash: str = UHASH):
    path = root / _GATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gate_pass": bool(gate_pass),
                "net_sharpe": 0.55,
                "max_dd": 0.20,
                "positive_years": 11,
                "total_years": 16,
                "universe_hash": universe_hash,
                "asof": "2026-06-19",
            },
            sort_keys=True,
        )
    )


def _write_portfolio(root: Path, *, nav: float, peak_nav: float):
    path = root / _PSTATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PortfolioState(nav=nav, peak_nav=peak_nav).to_dict()))


def _prep(tmp_path: Path, monkeypatch, *, halted: bool):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir(exist_ok=True)
    StateEngine().set_halted(halted)
    return ScannerConfig()


def _decide(tmp_path, cfg, **kw):
    return decide_cycle(
        config=cfg, project_root=tmp_path, now=NOW, snapshot_universe_hash=UHASH, **kw
    )


def test_continue_when_all_healthy(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path)  # no portfolio_state -> no positions -> no dd risk
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert isinstance(res, CycleDecision)
    assert res.decision is EquityDecision.CONTINUE
    assert res.actionable is True


def test_refuse_when_global_halt(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=True)
    _write_ship_gate(tmp_path)
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.REFUSE
    assert res.actionable is False
    assert any("halt" in r for r in res.reasons)


def test_halt_when_drawdown_breach(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path)
    _write_portfolio(tmp_path, nav=75.0, peak_nav=100.0)  # 25% dd >= dd_hard 0.20
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.HALT
    assert res.actionable is False


def test_no_act_when_ship_gate_fails(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path, gate_pass=False)
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.NO_ACT


def test_no_act_when_universe_hash_mismatch(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path, universe_hash="f" * 64)
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.NO_ACT


def test_no_act_when_ship_gate_missing(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    res = _decide(tmp_path, cfg, data_asof=NOW)  # no SHIP_GATE.json on disk
    assert res.decision is EquityDecision.NO_ACT


def test_abstain_when_data_stale(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path)
    stale = NOW - pd.Timedelta(days=30)
    res = _decide(tmp_path, cfg, data_asof=stale)
    assert res.decision is EquityDecision.ABSTAIN


def test_refuse_when_state_json_corrupt(tmp_path: Path, monkeypatch):
    """HIGH-defect fix: a corrupt halt file must fail CLOSED (REFUSE), not open."""
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    (tmp_path / ".claude" / "state.json").write_text("{ broken json")
    _write_ship_gate(tmp_path)
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.REFUSE


def test_halt_when_drawdown_exactly_at_threshold(tmp_path: Path, monkeypatch):
    """LOW-defect fix: an exact 20% drawdown must breach dd_hard (>=, not float >)."""
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path)
    _write_portfolio(tmp_path, nav=80.0, peak_nav=100.0)  # exactly 0.20
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.HALT


def test_abstain_when_no_data_asof(tmp_path: Path, monkeypatch):
    """MEDIUM-defect fix: absent/unknown data freshness fails closed (ABSTAIN)."""
    cfg = _prep(tmp_path, monkeypatch, halted=False)
    _write_ship_gate(tmp_path)
    res = _decide(tmp_path, cfg, data_asof=None)
    assert res.decision is EquityDecision.ABSTAIN


def test_global_halt_takes_precedence_over_drawdown(tmp_path: Path, monkeypatch):
    cfg = _prep(tmp_path, monkeypatch, halted=True)
    _write_ship_gate(tmp_path)
    _write_portfolio(tmp_path, nav=50.0, peak_nav=100.0)  # also a dd breach
    res = _decide(tmp_path, cfg, data_asof=NOW)
    assert res.decision is EquityDecision.REFUSE  # most-severe wins
