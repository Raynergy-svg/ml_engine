"""No-mock tests for the mechanical ship gate (US-007)."""

from __future__ import annotations

from src.factor.ship_gate import evaluate_gate


def _passing_report():
    return {
        "net_sharpe": 0.55,
        "positive_years": 7,
        "total_years": 11,
        "max_drawdown": 0.18,
        "walk_forward": True,
    }


def test_gate_passes_when_all_met():
    v = evaluate_gate(_passing_report())
    assert v.passed is True
    assert "PASS" in v.summary


def test_gate_fails_on_low_sharpe():
    r = _passing_report()
    r["net_sharpe"] = 0.30
    v = evaluate_gate(r)
    assert v.passed is False
    assert "net_sharpe" in v.summary


def test_gate_fails_on_drawdown():
    r = _passing_report()
    r["max_drawdown"] = 0.40
    v = evaluate_gate(r)
    assert v.passed is False
    assert v.criteria["max_drawdown"]["passed"] is False


def test_gate_requires_walk_forward():
    r = _passing_report()
    r["walk_forward"] = False
    assert evaluate_gate(r).passed is False
