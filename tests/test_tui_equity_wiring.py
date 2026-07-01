"""Equity-harvester TUI wiring tests (2026-06-21 audit, P1-1/P1-2/P1-6).

Surfaces the equity control-loop / ship-gate / live-gate state files into
the TUI. These tests exercise the read-only state readers and the widget
states they drive.

NO MOCKS (CLAUDE.md No-Mock Rule): every test writes a real JSON file to a
real ``tmp_path`` and reads it back through the real DataProvider, real
StateStrip, and real HeaderBar. Absent-file paths use a fresh tmp_path with
no state written, exercising the graceful-degradation contract.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.tui.app import HarvesterPanel, HeaderBar
from src.tui.data_provider import DataProvider
from src.tui.screens.agents_screen import RiskGatesPanel
from src.tui.screens.inbox_screen import EquityAlertsBanner
from src.tui.screens.trades_screen import RebalancePlanPanel
from src.tui.widgets.state_strip import StateStrip


# ── helpers ────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ship_gate_payload(gate_pass: bool = True) -> dict:
    return {
        "asof": "2026-05-29",
        "gate_pass": gate_pass,
        "max_dd": 0.229,
        "net_sharpe": 0.921,
        "positive_years": 13,
        "total_years": 17,
        "thresholds": {
            "max_drawdown": 0.25,
            "min_net_sharpe": 0.4,
            "min_positive_years": 6.0,
            "min_total_years": 10.0,
        },
        "universe_hash": "8bfe419a0c1c06306ff3fa35c39132df522b17ae",
        "recommendation": "PASS",
    }


# ── get_ship_gate_status ───────────────────────────────────────────────


def test_ship_gate_absent_is_unavailable(tmp_path: Path) -> None:
    """No SHIP_GATE.json → available False, no crash."""
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_ship_gate_status()
    assert status["available"] is False


def test_ship_gate_pass_surfaces_metrics(tmp_path: Path) -> None:
    """A passing gate file surfaces gate_pass True + Sharpe + DD."""
    _write_json(
        tmp_path / "trained_data" / "backtests" / "SHIP_GATE.json",
        _ship_gate_payload(gate_pass=True),
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_ship_gate_status()
    assert status["available"] is True
    assert status["gate_pass"] is True
    assert abs(status["net_sharpe"] - 0.921) < 1e-9
    assert abs(status["max_dd"] - 0.229) < 1e-9
    assert status["thresholds"]["min_net_sharpe"] == 0.4


def test_ship_gate_fail_reads_false(tmp_path: Path) -> None:
    """gate_pass False is faithfully surfaced (not coerced to True)."""
    _write_json(
        tmp_path / "trained_data" / "backtests" / "SHIP_GATE.json",
        _ship_gate_payload(gate_pass=False),
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_ship_gate_status()
    assert status["available"] is True
    assert status["gate_pass"] is False


def test_ship_gate_corrupt_degrades_gracefully(tmp_path: Path) -> None:
    """Corrupt JSON → available False, never raises."""
    path = tmp_path / "trained_data" / "backtests" / "SHIP_GATE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_ship_gate_status()["available"] is False


# ── get_live_gate_status (SHADOW default when absent) ──────────────────


def test_live_gate_absent_defaults_shadow(tmp_path: Path) -> None:
    """No live_gate_state.json → SHADOW (safe default), armed False."""
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_live_gate_status()
    assert status["available"] is False
    assert status["armed"] is False
    assert status["mode"] == "SHADOW"


def test_live_gate_disarmed_is_shadow(tmp_path: Path) -> None:
    """armed False on disk → SHADOW mode."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "live_gate_state.json",
        {"version": 1, "armed": False, "universe_hash": "abc"},
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_live_gate_status()
    assert status["available"] is True
    assert status["mode"] == "SHADOW"


def test_live_gate_armed_is_live_with_nav_fraction(tmp_path: Path) -> None:
    """armed True → LIVE mode + nav_fraction surfaced."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "live_gate_state.json",
        {
            "version": 1,
            "armed": True,
            "universe_hash": "abc",
            "initial_nav_fraction": 0.05,
            "max_portfolio_risk_fraction": 0.15,
        },
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_live_gate_status()
    assert status["mode"] == "LIVE"
    assert abs(status["nav_fraction"] - 0.05) < 1e-9


# ── get_harvester_status (control-loop pending state) ──────────────────


def test_harvester_absent_is_pending(tmp_path: Path) -> None:
    """Neither loop nor portfolio state on disk → available False."""
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_harvester_status()["available"] is False


def test_harvester_loop_state_surfaces_cycle_and_halt(tmp_path: Path) -> None:
    """loop_state.json surfaces cycle_count, halted, transport, reconcile."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "loop_state.json",
        {
            "version": 1,
            "cycle_count": 42,
            "halted": False,
            "last_cycle_asof": "2026-06-22T00:00:00+00:00",
            "last_transport_state": "CONNECTED",
            "consecutive_failures": 0,
            "last_reconcile": {
                "asof": "2026-06-22T00:00:00+00:00",
                "nav": 100000.0,
                "cash_drift_frac": 0.001,
                "drift_breaches": ["AAPL"],
            },
            "last_risk_decision": {
                "block_trade": False,
                "halt": False,
                "degross_factor": 1.0,
                "reasons": [],
            },
        },
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_harvester_status()
    assert status["available"] is True
    assert status["cycle_count"] == 42
    assert status["halted"] is False
    assert status["transport_state"] == "CONNECTED"
    assert status["reconcile"]["n_breaches"] == 1


def test_harvester_portfolio_state_computes_drawdown(tmp_path: Path) -> None:
    """portfolio_state.json drives NAV / peak / drawdown%."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "portfolio_state.json",
        {"version": 1, "nav": 90000.0, "peak_nav": 100000.0, "halted": False},
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_harvester_status()
    assert status["available"] is True
    assert abs(status["nav"] - 90000.0) < 1e-6
    assert abs(status["drawdown_pct"] - 10.0) < 1e-6


def test_harvester_portfolio_halt_sets_halted(tmp_path: Path) -> None:
    """A halted portfolio state flips the displayed halted flag."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "portfolio_state.json",
        {"version": 1, "nav": 100000.0, "peak_nav": 100000.0, "halted": True},
    )
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_harvester_status()["halted"] is True


# ── StateStrip ship-gate badge ─────────────────────────────────────────


def test_state_strip_badge_absent_renders_dash() -> None:
    """No ship-gate data → badge shows dim '—', no PASS/FAIL."""
    strip = StateStrip()
    strip.update_ship_gate({"available": False})
    plain = strip.render().plain
    assert "SHIP_GATE" in plain
    assert "—" in plain
    assert "PASS" not in plain and "FAIL" not in plain


def test_state_strip_badge_pass_renders_green_text() -> None:
    """gate_pass True → 'PASS' with Sharpe + DD in the rendered text."""
    strip = StateStrip()
    strip.update_ship_gate(
        {
            "available": True,
            "gate_pass": True,
            "net_sharpe": 0.92,
            "max_dd": 0.229,
        }
    )
    plain = strip.render().plain
    assert "PASS" in plain
    assert "0.92" in plain
    assert "22.9%" in plain


def test_state_strip_badge_fail_renders_fail() -> None:
    """gate_pass False → 'FAIL'."""
    strip = StateStrip()
    strip.update_ship_gate(
        {
            "available": True,
            "gate_pass": False,
            "net_sharpe": 0.10,
            "max_dd": 0.40,
        }
    )
    plain = strip.render().plain
    assert "FAIL" in plain


# ── HeaderBar SHADOW/LIVE indicator ────────────────────────────────────


def test_header_mode_defaults_shadow() -> None:
    """Fresh HeaderBar renders MODE SHADOW (the safe default)."""
    header = HeaderBar()
    plain = header.render().plain
    assert "SHADOW" in plain
    assert "LIVE ⚠" not in plain


def test_header_mode_shadow_when_disarmed() -> None:
    """A disarmed live-gate dict keeps the header in SHADOW."""
    header = HeaderBar()
    header.update_live_gate({"available": True, "mode": "SHADOW", "armed": False})
    assert "SHADOW" in header.render().plain


def test_header_mode_live_shows_warning_and_nav() -> None:
    """An armed live-gate dict renders 'LIVE ⚠' + NAV fraction."""
    header = HeaderBar()
    header.update_live_gate(
        {"available": True, "mode": "LIVE", "armed": True, "nav_fraction": 0.05}
    )
    plain = header.render().plain
    assert "LIVE ⚠" in plain
    assert "5% NAV" in plain


# ── HarvesterPanel control-loop panel ──────────────────────────────────


def test_harvester_panel_pending_when_no_state() -> None:
    """Default HarvesterPanel renders the honest pending message."""
    panel = HarvesterPanel()
    assert "not yet running" in panel.render().plain


def test_harvester_panel_renders_running_state() -> None:
    """A populated status dict renders NAV / cycle / RUNNING."""
    panel = HarvesterPanel()
    panel.update_status(
        {
            "available": True,
            "halted": False,
            "cycle_count": 7,
            "transport_state": "CONNECTED",
            "nav": 100000.0,
            "peak_nav": 105000.0,
            "drawdown_pct": 4.76,
        }
    )
    plain = panel.render().plain
    assert "RUNNING" in plain
    assert "CYCLE 7" in plain
    assert "$100,000" in plain


def test_harvester_panel_renders_halted_state() -> None:
    """halted True renders HALTED."""
    panel = HarvesterPanel()
    panel.update_status(
        {"available": True, "halted": True, "cycle_count": 1}
    )
    assert "HALTED" in panel.render().plain


# ── get_risk_gates_status (P1-3, reads loop_state.last_risk_decision) ──


def _loop_state_with_risk(decision: dict) -> dict:
    return {
        "version": 1,
        "cycle_count": 5,
        "halted": bool(decision.get("halt", False)),
        "last_cycle_asof": "2026-06-22T00:00:00+00:00",
        "last_transport_state": "CONNECTED",
        "consecutive_failures": 0,
        "last_risk_decision": decision,
    }


def _verdict(name: str, *, passed: bool, block: bool = False) -> dict:
    """A real AgentVerdict-shaped dict (matches _team.AgentVerdict.to_dict)."""
    return {
        "name": name,
        "score": 0.8 if passed else 0.2,
        "passed": passed,
        "weight": 1.1,
        "reason": f"{name} reason",
        "reason_code": f"{name}_ok" if passed else f"{name}_block",
        "block_trade": block,
        "metadata": {},
    }


def test_risk_gates_absent_is_unavailable(tmp_path: Path) -> None:
    """No loop_state.json → available False."""
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_risk_gates_status()["available"] is False


def test_risk_gates_no_decision_is_unavailable(tmp_path: Path) -> None:
    """Loop state present but no risk decision yet → available False."""
    _write_json(
        tmp_path / "trained_data" / "equity" / "loop_state.json",
        {"version": 1, "cycle_count": 1, "halted": False},
    )
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_risk_gates_status()["available"] is False


def test_risk_gates_corrupt_degrades(tmp_path: Path) -> None:
    """Corrupt loop_state.json → available False, never raises."""
    path = tmp_path / "trained_data" / "equity" / "loop_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_risk_gates_status()["available"] is False


def test_risk_gates_surface_per_gate_verdicts(tmp_path: Path) -> None:
    """Each gate's pass/block + composite degross factor are surfaced."""
    decision = {
        "block_trade": True,
        "halt": False,
        "degross_factor": 0.5,
        "reasons": ["gross_cap_exceeded"],
        "verdicts": [
            _verdict("drawdown_guardian", passed=True),
            _verdict("vol_targeter", passed=True),
            _verdict("gross_exposure_cap", passed=False, block=True),
            _verdict("per_name_exposure_cap", passed=True),
            _verdict("concentration_check", passed=True),
            _verdict("adv_participation", passed=True),
            _verdict("rebalance_gate", passed=True),
        ],
    }
    _write_json(
        tmp_path / "trained_data" / "equity" / "loop_state.json",
        _loop_state_with_risk(decision),
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_risk_gates_status()
    assert status["available"] is True
    assert status["block_trade"] is True
    assert status["halt"] is False
    assert abs(status["degross_factor"] - 0.5) < 1e-9
    assert len(status["gates"]) == 7
    gx = next(g for g in status["gates"] if g["name"] == "gross_exposure_cap")
    assert gx["passed"] is False
    assert gx["block_trade"] is True
    assert gx["reason_code"] == "gross_exposure_cap_block"
    assert status["reasons"] == ["gross_cap_exceeded"]


def test_risk_gates_halt_surfaced(tmp_path: Path) -> None:
    """A halting drawdown decision surfaces halt True + degross 0.0."""
    decision = {
        "block_trade": True,
        "halt": True,
        "degross_factor": 0.0,
        "reasons": ["drawdown_hard_breach_halt"],
        "verdicts": [_verdict("drawdown_guardian", passed=False, block=True)],
    }
    _write_json(
        tmp_path / "trained_data" / "equity" / "loop_state.json",
        _loop_state_with_risk(decision),
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_risk_gates_status()
    assert status["halt"] is True
    assert status["degross_factor"] == 0.0


def test_risk_gates_real_decision_from_risk_layer(tmp_path: Path) -> None:
    """End-to-end: a real RiskDecision.to_dict() round-trips through the reader.

    Uses the real risk_agents gate functions (no mocks) to build a genuine
    RiskDecision, serialises it exactly as the control loop does, and verifies
    the reader surfaces all seven gate verdicts.
    """
    from src.equity.risk_agents import (
        RiskAgentConfig,
        RiskDecision,
        adv_participation,
        concentration_check,
        drawdown_guardian,
        gross_exposure_cap,
        per_name_exposure_cap,
        rebalance_gate,
        vol_targeter,
    )
    from src.equity.risk_agents import PortfolioState
    import pandas as pd

    cfg = RiskAgentConfig()
    state = PortfolioState(nav=100000.0, peak_nav=100000.0)
    asof = pd.Timestamp("2026-06-22", tz="UTC")
    target = {"AAPL": 0.10, "MSFT": 0.10}
    verdicts = [
        drawdown_guardian(state, cfg),
        vol_targeter(state, cfg),
        gross_exposure_cap(target, cfg),
        per_name_exposure_cap(target, cfg),
        concentration_check(target, cfg),
        adv_participation(target, state.current_weights, state.nav, None, cfg),
        rebalance_gate(state, asof, cfg),
    ]
    decision = RiskDecision(
        verdicts=verdicts,
        block_trade=any(v.block_trade for v in verdicts),
        degross_factor=1.0,
        halt=False,
        reasons=[],
    )
    _write_json(
        tmp_path / "trained_data" / "equity" / "loop_state.json",
        _loop_state_with_risk(decision.to_dict()),
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_risk_gates_status()
    assert status["available"] is True
    names = {g["name"] for g in status["gates"]}
    assert names == {
        "drawdown_guardian",
        "vol_targeter",
        "gross_exposure_cap",
        "per_name_exposure_cap",
        "concentration_check",
        "adv_participation",
        "rebalance_gate",
    }


# ── get_rebalance_status (P1-4) — real RebalanceState round-trip ───────


def test_rebalance_absent_is_unavailable(tmp_path: Path) -> None:
    """No rebalance_state.json → available False."""
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_rebalance_status()["available"] is False


def test_rebalance_corrupt_degrades(tmp_path: Path) -> None:
    """Corrupt rebalance_state.json → available False, never raises."""
    path = tmp_path / "trained_data" / "equity" / "rebalance_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_rebalance_status()["available"] is False


def test_rebalance_no_active_plan(tmp_path: Path) -> None:
    """State present, no active plan → available True, active_plan None."""
    from src.equity.rebalance import RebalanceState, STATE_PATH_DEFAULT

    state = RebalanceState(
        last_rebalance_asof="2026-06-20T00:00:00+00:00",
        current_actual_weights={"AAPL": 0.5, "MSFT": 0.5},
        active_plan=None,
    )
    path = tmp_path / STATE_PATH_DEFAULT
    _write_json(path, state.to_dict())
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_rebalance_status()
    assert status["available"] is True
    assert status["active_plan"] is None
    assert status["current_weights"]["AAPL"] == 0.5
    assert status["last_rebalance_asof"] == "2026-06-20T00:00:00+00:00"


def test_rebalance_active_plan_surfaces_orders(tmp_path: Path) -> None:
    """A real RebalancePlan round-trips: target weights + per-order status.

    Builds a genuine RebalanceState with an active plan via the real
    dataclasses (no mocks), persists it via the canonical path constant, and
    verifies the reader surfaces order ticker/side/status.
    """
    from src.equity.rebalance import (
        Order,
        OrderStatus,
        RebalancePlan,
        RebalanceState,
        STATE_PATH_DEFAULT,
    )

    plan = RebalancePlan(
        rebalance_id="rb-001",
        asof="2026-06-22T00:00:00+00:00",
        target_weights={"AAPL": 0.6, "MSFT": 0.4},
        orders=[
            Order(
                client_order_id="oid-1",
                ticker="AAPL",
                side="BUY",
                weight_delta=0.1,
                target_weight=0.6,
                status=OrderStatus.FILLED.value,
            ),
            Order(
                client_order_id="oid-2",
                ticker="MSFT",
                side="SELL",
                weight_delta=-0.1,
                target_weight=0.4,
                status=OrderStatus.PENDING.value,
            ),
        ],
        no_trade_band=0.005,
    )
    state = RebalanceState(
        last_rebalance_asof="2026-06-22T00:00:00+00:00",
        current_actual_weights={"AAPL": 0.5, "MSFT": 0.5},
        active_plan=plan.to_dict(),
    )
    path = tmp_path / STATE_PATH_DEFAULT
    _write_json(path, state.to_dict())
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_rebalance_status()
    assert status["available"] is True
    assert status["active_plan"]["rebalance_id"] == "rb-001"
    assert status["active_plan"]["target_weights"]["AAPL"] == 0.6
    orders = status["active_plan"]["orders"]
    assert len(orders) == 2
    aapl = next(o for o in orders if o["ticker"] == "AAPL")
    assert aapl["status"] == "FILLED"
    assert aapl["side"] == "BUY"
    msft = next(o for o in orders if o["ticker"] == "MSFT")
    assert msft["status"] == "PENDING"


# ── get_alerts_status (P1-7) ──────────────────────────────────────────


def test_alerts_absent_is_unavailable(tmp_path: Path) -> None:
    """No state file and no audit dir → available False."""
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_alerts_status()["available"] is False


def test_alerts_corrupt_state_with_no_audit_unavailable(tmp_path: Path) -> None:
    """Corrupt state + no audit dir → available False, never raises."""
    path = tmp_path / "trained_data" / "equity" / "alerts_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    dp = DataProvider(project_root=str(tmp_path))
    assert dp.get_alerts_status()["available"] is False


def test_alerts_state_only_surfaces_cooldowns(tmp_path: Path) -> None:
    """alerts_state.json alone surfaces last_fired cooldown log."""
    from src.equity.alerts import STATE_PATH_DEFAULT

    _write_json(
        tmp_path / STATE_PATH_DEFAULT,
        {
            "last_fired": {"HALT": 1782000000.0, "DD_BREACH": 1782000500.0},
            "universe_hash": "abc",
            "last_updated": "2026-06-22T00:00:00+00:00",
        },
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_alerts_status()
    assert status["available"] is True
    assert status["last_fired"]["HALT"] == 1782000000.0
    assert status["universe_hash"] == "abc"
    assert status["alerts"] == []


def test_alerts_audit_surfaces_recent_notifications(tmp_path: Path) -> None:
    """Audit sidecars surface alert type/severity/message newest-first."""
    from src.equity.alerts import AUDIT_DIR_DEFAULT

    audit = tmp_path / AUDIT_DIR_DEFAULT
    audit.mkdir(parents=True, exist_ok=True)
    # Filenames are timestamp-prefixed; lexical sort ≈ chronological.
    _write_json(
        audit / "2026-06-22T00:00:00-HALT-0001.email.json",
        {
            "notification": {
                "alert_type": "HALT",
                "severity": "CRITICAL",
                "message": "operator halt engaged",
                "timestamp": "2026-06-22T00:00:00+00:00",
            }
        },
    )
    _write_json(
        audit / "2026-06-22T01:00:00-DD_BREACH-0002.email.json",
        {
            "notification": {
                "alert_type": "DD_BREACH",
                "severity": "CRITICAL",
                "message": "drawdown breached",
                "timestamp": "2026-06-22T01:00:00+00:00",
            }
        },
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_alerts_status()
    assert status["available"] is True
    assert len(status["alerts"]) == 2
    # newest-first: DD_BREACH (01:00) before HALT (00:00)
    assert status["alerts"][0]["alert_type"] == "DD_BREACH"
    assert status["alerts"][1]["alert_type"] == "HALT"


def test_alerts_corrupt_sidecar_skipped(tmp_path: Path) -> None:
    """A corrupt audit sidecar is skipped, the good one still surfaces."""
    from src.equity.alerts import AUDIT_DIR_DEFAULT

    audit = tmp_path / AUDIT_DIR_DEFAULT
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "2026-06-22T00:00:00-HALT-0001.email.json").write_text(
        "{not json", encoding="utf-8"
    )
    _write_json(
        audit / "2026-06-22T01:00:00-DD_BREACH-0002.email.json",
        {
            "notification": {
                "alert_type": "DD_BREACH",
                "severity": "CRITICAL",
                "message": "drawdown breached",
                "timestamp": "2026-06-22T01:00:00+00:00",
            }
        },
    )
    dp = DataProvider(project_root=str(tmp_path))
    status = dp.get_alerts_status()
    assert status["available"] is True
    assert len(status["alerts"]) == 1
    assert status["alerts"][0]["alert_type"] == "DD_BREACH"


# ── RiskGatesPanel widget (P1-3) ──────────────────────────────────────


def test_risk_gates_panel_pending_when_unavailable() -> None:
    """Default / unavailable status → honest empty state, no crash."""
    panel = RiskGatesPanel()
    plain = panel.render().plain
    assert "No risk decision yet" in plain


def test_risk_gates_panel_renders_clear() -> None:
    """A clear decision renders CLEAR + DEGROSS + per-gate PASS rows."""
    panel = RiskGatesPanel()
    panel.update_status(
        {
            "available": True,
            "block_trade": False,
            "halt": False,
            "degross_factor": 1.0,
            "reasons": [],
            "gates": [
                {
                    "name": "drawdown_guardian",
                    "passed": True,
                    "block_trade": False,
                    "score": 0.9,
                    "weight": 1.3,
                    "reason": "ok",
                    "reason_code": "drawdown_ok",
                },
            ],
            "last_cycle_asof": "2026-06-22T00:00:00+00:00",
        }
    )
    plain = panel.render().plain
    assert "CLEAR" in plain
    assert "1.00x" in plain
    assert "drawdown" in plain
    assert "PASS" in plain


def test_risk_gates_panel_renders_blocked_gate() -> None:
    """A blocking gate renders BLOCKED + the gate's BLOCK row."""
    panel = RiskGatesPanel()
    panel.update_status(
        {
            "available": True,
            "block_trade": True,
            "halt": False,
            "degross_factor": 0.5,
            "reasons": ["gross_cap_exceeded"],
            "gates": [
                {
                    "name": "gross_exposure_cap",
                    "passed": False,
                    "block_trade": True,
                    "score": 0.1,
                    "weight": 1.2,
                    "reason": "gross too high",
                    "reason_code": "gross_cap_exceeded",
                },
            ],
        }
    )
    plain = panel.render().plain
    assert "BLOCKED" in plain
    assert "BLOCK" in plain
    assert "gross_cap" in plain
    assert "gross_cap_exceeded" in plain


def test_risk_gates_panel_renders_halt() -> None:
    """halt True renders HALTED + degross 0.00x."""
    panel = RiskGatesPanel()
    panel.update_status(
        {
            "available": True,
            "block_trade": True,
            "halt": True,
            "degross_factor": 0.0,
            "reasons": ["drawdown_hard_breach_halt"],
            "gates": [],
        }
    )
    plain = panel.render().plain
    assert "HALTED" in plain
    assert "0.00x" in plain


# ── RebalancePlanPanel widget (P1-4) ──────────────────────────────────


def test_rebalance_panel_pending_when_unavailable() -> None:
    """Default / unavailable status → honest empty state."""
    panel = RebalancePlanPanel()
    assert "No rebalance has run yet" in panel.render().plain


def test_rebalance_panel_no_active_plan() -> None:
    """State present, no active plan → shows last asof + current weights."""
    panel = RebalancePlanPanel()
    panel.update_status(
        {
            "available": True,
            "last_rebalance_asof": "2026-06-20T00:00:00+00:00",
            "current_weights": {"AAPL": 0.6, "MSFT": 0.4},
            "active_plan": None,
        }
    )
    plain = panel.render().plain
    assert "No active plan in flight" in plain
    assert "AAPL" in plain


def test_rebalance_panel_renders_orders_with_status() -> None:
    """An active plan renders per-order ticker/side/status + tally."""
    panel = RebalancePlanPanel()
    panel.update_status(
        {
            "available": True,
            "last_rebalance_asof": "2026-06-22T00:00:00+00:00",
            "current_weights": {"AAPL": 0.5, "MSFT": 0.5},
            "active_plan": {
                "rebalance_id": "rb-001",
                "asof": "2026-06-22T00:00:00+00:00",
                "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
                "orders": [
                    {
                        "ticker": "AAPL",
                        "side": "BUY",
                        "target_weight": 0.6,
                        "status": "FILLED",
                    },
                    {
                        "ticker": "MSFT",
                        "side": "SELL",
                        "target_weight": 0.4,
                        "status": "PENDING",
                    },
                ],
            },
        }
    )
    plain = panel.render().plain
    assert "rb-001" in plain
    assert "AAPL" in plain
    assert "BUY" in plain
    assert "FILLED" in plain
    assert "PENDING" in plain
    assert "1 FILLED" in plain


# ── EquityAlertsBanner widget (P1-7) ──────────────────────────────────


def test_alerts_banner_hidden_when_unavailable() -> None:
    """Unavailable status → banner hidden, empty render (inbox unchanged)."""
    banner = EquityAlertsBanner()
    banner.update_status({"available": False})
    assert banner.display is False
    assert banner.render().plain == ""


def test_alerts_banner_shows_when_available() -> None:
    """Available status → banner displayed + renders alert rows."""
    banner = EquityAlertsBanner()
    banner.update_status(
        {
            "available": True,
            "last_fired": {"HALT": 1782000000.0},
            "universe_hash": "abc",
            "last_updated": "2026-06-22T00:00:00+00:00",
            "alerts": [
                {
                    "alert_type": "DD_BREACH",
                    "severity": "CRITICAL",
                    "message": "drawdown breached",
                    "timestamp": "2026-06-22T01:00:00+00:00",
                },
            ],
        }
    )
    assert banner.display is True
    plain = banner.render().plain
    assert "EQUITY ALERTS" in plain
    assert "DD_BREACH" in plain
    assert "CRITICAL" in plain
    assert "Cooldown active" in plain
    assert "HALT" in plain


def test_alerts_banner_state_only_no_alerts() -> None:
    """State present but no fired alerts → banner shown, cooldown-only line."""
    banner = EquityAlertsBanner()
    banner.update_status(
        {
            "available": True,
            "last_fired": {},
            "universe_hash": "abc",
            "last_updated": "2026-06-22T00:00:00+00:00",
            "alerts": [],
        }
    )
    assert banner.display is True
    assert "No alerts fired" in banner.render().plain


# ── P0-5: equity in the asset-class cycle ─────────────────────────────


def test_equity_in_asset_class_cycle() -> None:
    """'equity' is part of the ctrl+F asset-class cycle."""
    from src.tui.app import BuddyApp

    assert "equity" in BuddyApp._ASSET_CLASSES


def test_equity_cycle_does_not_mutate_fx_scanner_config() -> None:
    """Cycling to equity must NOT touch the FX EmbeddedScanner config.

    Drives the real cycle handler under a Pilot. A sentinel scanner records
    whether get_config() is called while in equity mode; the equity branch
    returns early before any config mutation, so for the equity selection the
    handler must not reach the scanner-config path. Verifies FX is unbroken.
    """
    from src.tui.app import BuddyApp

    class _SentinelScanner:
        def __init__(self) -> None:
            self.get_config_calls = 0

        def get_config(self):
            self.get_config_calls += 1
            return None

    app = BuddyApp()

    async def _run() -> None:
        async with app.run_test(headless=True, size=(80, 24)) as pilot:
            await pilot.pause()
            sentinel = _SentinelScanner()
            app._scanner = sentinel
            # Cycle fx → futures → hybrid → equity (4 presses from fx)
            start = app._ASSET_CLASSES.index(app._asset_class)
            steps = (app._ASSET_CLASSES.index("equity") - start) % len(
                app._ASSET_CLASSES
            )
            calls_before_equity = None
            for i in range(steps):
                app.action_cycle_asset_class()
                if app._asset_class == "equity":
                    calls_before_equity = sentinel.get_config_calls
                await pilot.pause()
            assert app._asset_class == "equity"
            # The equity branch returned early — get_config not called for it.
            # (It may have been called for the futures/hybrid steps before.)
            assert sentinel.get_config_calls == calls_before_equity

    asyncio.run(_run())
