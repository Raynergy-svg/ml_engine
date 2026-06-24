"""N4 — shadow-only equity-harvester runner (2026-06-24).

Wires the orphaned harvester library into a runnable SHADOW tick:
``UniverseSnapshot`` + prices -> ``HarvesterStrategy`` -> target weights ->
``RebalanceScheduler.plan_and_persist`` -> shadow ``execute_plan``. The tick
writes ``trained_data/equity/rebalance_state.json`` so the TUI "HARVESTER
REBALANCE PLAN" panel (``DataProvider.get_rebalance_status``) renders a real
plan — with NO broker and NO real orders.

No mocks: real ``HarvesterStrategy``, real ``RebalanceScheduler``, real
``UniverseSnapshot``, real ``StateEngine``, real ``DataProvider``, real disk via
``tmp_path``. Reproduce-then-resolve: the disabled/halted refusals (tests 1-2)
only hold once the runner's fail-closed guards exist — they FAIL without the
runner. The happy-path + shadow-fill assertions (tests 3-4) prove the TUI lights
up end-to-end in shadow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.equity.cycle_ledger import (
    LEDGER_PATH_DEFAULT,
    read_ledger,
    verify_chain,
)
from src.equity.runner import ShadowRunResult, run_shadow_rebalance
from src.equity.universe import UniverseRules, UniverseSnapshot
from src.scanner.automation.state_engine import StateEngine
from src.scanner.config import ScannerConfig
from src.tui.data_provider import DataProvider

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
UHASH = "deadbeef" * 8
_STATE_REL = "trained_data/equity/rebalance_state.json"
_GATE_REL = "trained_data/backtests/SHIP_GATE.json"


def _panels(seed: int = 7, n_days: int = 90):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n_days, tz="UTC").normalize()
    membership = pd.DataFrame(
        np.ones((n_days, len(TICKERS)), dtype=bool), index=dates, columns=TICKERS
    )
    rets = rng.normal(0.0, 0.008, size=(n_days, len(TICKERS)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=TICKERS
    )
    return membership, prices


def _snapshot(membership: pd.DataFrame) -> UniverseSnapshot:
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(),
        built_at="2026-06-19T00:00:00Z",
        pipeline_version="2026-06-18-eq1",
        universe_hash=UHASH,
    )


def _write_ship_gate(root: Path, *, gate_pass: bool = True) -> Path:
    path = root / _GATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": bool(gate_pass),
        "net_sharpe": 0.55,
        "max_dd": 0.20,
        "positive_years": 11,
        "total_years": 16,
        "universe_hash": UHASH,
        "asof": "2026-06-19",
        "recommendation": "PASS — test fixture",
        "criteria_source": "test",
        "thresholds": {},
        "pipeline_version": "2026-06-18-eq1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _prepare(tmp_path: Path, monkeypatch, *, enabled: bool, halted: bool):
    """Real fixture: chdir tmp, real StateEngine halt state, real ship gate."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir(exist_ok=True)
    StateEngine().set_halted(halted)
    _write_ship_gate(tmp_path)
    membership, prices = _panels()
    snapshot = _snapshot(membership)
    cfg = ScannerConfig()
    cfg.enable_equity_harvester = enabled
    asof = membership.index[-1]
    return cfg, snapshot, prices, asof


def test_runner_refuses_when_disabled(tmp_path: Path, monkeypatch):
    """Master flag off -> refuse, write no state (fail-closed)."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=False, halted=False
    )
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path
    )
    assert isinstance(res, ShadowRunResult)
    assert res.ran is False
    assert res.reason == "disabled"
    assert not (tmp_path / _STATE_REL).exists()


def test_runner_refuses_when_globally_halted(tmp_path: Path, monkeypatch):
    """state.json halted=True -> refuse cycles (mirrors FX halt discipline)."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=True
    )
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path
    )
    assert res.ran is False
    assert res.reason == "refuse"  # decision gate REFUSE on global halt
    assert not (tmp_path / _STATE_REL).exists()


def test_runner_writes_plan_that_tui_renders(tmp_path: Path, monkeypatch):
    """Enabled + not halted -> writes a real plan the TUI panel reads."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,  # data is current as of asof -> decision gate CONTINUE
    )
    assert res.ran is True
    assert res.reason == "executed"
    assert (tmp_path / _STATE_REL).exists()

    # TUI closure: the panel's own reader now returns a real, non-empty plan.
    status = DataProvider(project_root=str(tmp_path)).get_rebalance_status()
    assert status["available"] is True
    assert status["active_plan"] is not None
    assert len(status["active_plan"]["orders"]) > 0
    assert status["active_plan"]["target_weights"]  # non-empty target book


def test_runner_is_reinvocable_finalizes_prior_plan(tmp_path: Path, monkeypatch):
    """A repeated tick (the shadow "loop") must not crash on the prior plan."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    res1 = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res1.ran is True
    # Second invocation against the same state must succeed (re-plan), not raise
    # RebalanceError on the still-active prior plan.
    res2 = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res2.ran is True
    assert res2.reason == "executed"


def test_runner_orders_filled_in_shadow_no_broker(tmp_path: Path, monkeypatch):
    """Shadow fills mark orders FILLED with no broker contact."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res.plan is not None
    assert len(res.plan.orders) > 0
    assert all(o.is_filled for o in res.plan.orders)


def test_runner_abstains_on_stale_data(tmp_path: Path, monkeypatch):
    """Wired to the decision gate: stale market data -> ABSTAIN, no plan."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    # No `now` -> real now (2026-06-24) is >7d past the synthetic data's last bar.
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path
    )
    assert res.ran is False
    assert res.reason == "abstain"
    assert not (tmp_path / _STATE_REL).exists()


def test_runner_records_each_cycle_to_tamper_evident_ledger(tmp_path: Path, monkeypatch):
    """Pillar 2: every tick (actionable or not) appends one chained ledger record."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    ledger = tmp_path / LEDGER_PATH_DEFAULT

    # Actionable tick -> a CONTINUE record with order count + weight fingerprint.
    run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    recs = read_ledger(ledger)
    assert len(recs) == 1
    assert recs[0].decision == "continue"
    assert recs[0].actionable is True
    assert recs[0].n_orders > 0
    assert len(recs[0].target_weight_hash) == 64

    # Non-actionable tick (stale data) -> a second ABSTAIN record, still chained.
    run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path
    )
    recs = read_ledger(ledger)
    assert len(recs) == 2
    assert recs[1].decision == "abstain"
    assert recs[1].actionable is False
    ok, broken = verify_chain(ledger)
    assert ok is True and broken is None


def test_runner_tick_survives_corrupt_ledger_non_blocking(tmp_path: Path, monkeypatch):
    """D-1: a corrupt ledger must NOT crash a tick whose plan already executed."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    ledger = tmp_path / LEDGER_PATH_DEFAULT
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{ corrupt tail line\n")  # poison the ledger tail
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res.ran is True  # plan executed; ledger append failure is non-blocking
    assert res.reason == "executed"


def test_runner_abstains_on_empty_price_frame(tmp_path: Path, monkeypatch):
    """MEDIUM-defect fix: an empty price frame must ABSTAIN, not crash the gate."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    empty = prices.iloc[0:0]  # zero rows, same columns -> no data_asof
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=empty, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res.ran is False
    assert res.reason == "abstain"
    assert not (tmp_path / _STATE_REL).exists()


def test_runner_no_acts_on_ship_gate_fail(tmp_path: Path, monkeypatch):
    """Wired to the decision gate: failing ship gate -> NO_ACT (no raise, no plan)."""
    cfg, snap, prices, asof = _prepare(
        tmp_path, monkeypatch, enabled=True, halted=False
    )
    _write_ship_gate(tmp_path, gate_pass=False)  # overwrite the passing fixture gate
    res = run_shadow_rebalance(
        config=cfg, snapshot=snap, prices=prices, asof=asof, project_root=tmp_path,
        now=asof,
    )
    assert res.ran is False
    assert res.reason == "no_act"
    assert not (tmp_path / _STATE_REL).exists()
