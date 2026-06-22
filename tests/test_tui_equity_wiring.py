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

import json
from pathlib import Path

from src.tui.app import HarvesterPanel, HeaderBar
from src.tui.data_provider import DataProvider
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
