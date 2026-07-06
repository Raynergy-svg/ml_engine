"""No-mock tests for src.brain_loop.derisk — the ONLY brain_loop module allowed
to touch live-affecting state.

Load-bearing guarantees:
  * halt() only ever sets halted=True (no ``value`` parameter exists at all —
    it structurally cannot be used to unhalt).
  * halt() reuses the real StateEngine, verified via a real disk read-back.
  * tighten_risk_config() only ever REDUCES risk and stays within the existing
    clamp_risk_pct bounds — it cannot increase risk even if misused.
  * every de-risk action is audited with a mandatory actor (TypeError if
    omitted), mirroring dashboard/server/control_safety.py's audit() contract.
"""
from __future__ import annotations

import json

import pytest

from src.brain_loop.derisk import apply_risk_tightening, halt, tighten_risk_config
from src.equity.trend_risk_gates import MAX_RISK_PCT, MIN_RISK_PCT
from src.scanner.automation.state_engine import StateEngine


def test_halt_flips_real_state_via_real_state_engine(tmp_path):
    state_path = tmp_path / ".claude" / "state.json"
    engine = StateEngine(state_path)
    assert engine.get_halted() is False  # default state

    halt(state_path, reason="drawdown_breach:0.30", actor="brain_loop")

    # Independent read-back through a FRESH StateEngine instance (no shared cache).
    assert StateEngine(state_path).get_halted() is True


def test_halt_records_last_actor(tmp_path):
    state_path = tmp_path / ".claude" / "state.json"
    halt(state_path, reason="cost_anomaly", actor="brain_loop")
    state = StateEngine(state_path).load_state()
    assert state["last_actor"] == "brain_loop"


def test_halt_requires_actor_keyword(tmp_path):
    state_path = tmp_path / ".claude" / "state.json"
    with pytest.raises(TypeError):
        halt(state_path, reason="x")  # actor omitted -> must fail, not silently default


def test_halt_has_no_value_parameter_at_all():
    """Structural proof, not policy: halt() cannot be called to unhalt because
    no such parameter exists on the function."""
    import inspect

    sig = inspect.signature(halt)
    assert "value" not in sig.parameters


def test_halt_writes_an_audit_entry(tmp_path):
    state_path = tmp_path / ".claude" / "state.json"
    audit_path = tmp_path / ".claude" / "brain_loop_audit.jsonl"
    halt(state_path, reason="drawdown_breach:0.30", actor="brain_loop", audit_path=audit_path)

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["actor"] == "brain_loop"
    assert entry["action"] == "halt"
    assert entry["reason"] == "drawdown_breach:0.30"


def test_tighten_risk_config_only_ever_reduces(tmp_path):
    tightened = tighten_risk_config(0.02, delta=0.005)
    assert tightened < 0.02
    assert MIN_RISK_PCT <= tightened <= MAX_RISK_PCT


def test_tighten_risk_config_cannot_increase_even_with_negative_delta():
    """A negative delta must not sneak an increase — the function subtracts
    abs(delta), never adds."""
    tightened = tighten_risk_config(0.02, delta=-0.01)
    assert tightened <= 0.02


def test_tighten_risk_config_floors_at_min_risk_pct():
    tightened = tighten_risk_config(MIN_RISK_PCT, delta=0.5)
    assert tightened == MIN_RISK_PCT


def test_tighten_risk_config_nan_delta_never_increases_risk():
    """Security-review finding (2026-07-01): NaN comparisons are always False
    under IEEE 754, so `reduced <= 0` silently passes NaN through to
    clamp_risk_pct, which maps non-finite input to DEFAULT_RISK_PCT (0.01) —
    HIGHER than a current_risk_pct sitting at MIN_RISK_PCT (0.005). A NaN
    delta must never be able to increase risk."""
    tightened = tighten_risk_config(MIN_RISK_PCT, delta=float("nan"))
    assert tightened <= MIN_RISK_PCT


def test_tighten_risk_config_nan_current_risk_pct_never_increases_risk():
    tightened = tighten_risk_config(float("nan"), delta=0.01)
    assert tightened == MIN_RISK_PCT


def test_apply_risk_tightening_persists_atomically(tmp_path):
    overrides_path = tmp_path / "risk_overrides.json"
    result = apply_risk_tightening(
        overrides_path, delta=0.005, actor="brain_loop", current_risk_pct=0.02
    )
    assert result["risk_pct"] < 0.02
    on_disk = json.loads(overrides_path.read_text())
    assert on_disk["risk_pct"] == result["risk_pct"]
    assert on_disk["_actor"] == "brain_loop"


def test_apply_risk_tightening_is_monotonic_across_calls(tmp_path):
    overrides_path = tmp_path / "risk_overrides.json"
    first = apply_risk_tightening(
        overrides_path, delta=0.002, actor="brain_loop", current_risk_pct=0.03
    )
    second = apply_risk_tightening(overrides_path, delta=0.002, actor="brain_loop")
    assert second["risk_pct"] <= first["risk_pct"]
