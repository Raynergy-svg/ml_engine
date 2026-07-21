"""Per-lane halt control — reproduce-then-resolve tests.

Extends the single global ``halted`` boolean in ``.claude/state.json`` with an
optional ``halted_lanes`` dict so the FX scanner (``oanda_fx``), the equity
harvester (``equity``), and the brain loop (``brain``) can be halted
independently once the legacy global flag is False. Covers, in order:

  1. equity can run while oanda_fx stays halted, and vice versa
     (``src.scanner.automation.state_engine.StateEngine.get_halted`` AND
     ``src.equity.decision_gate._lane_halted`` — the two lane-aware readers).
  2. missing/corrupt state fails CLOSED for every lane.
  3. the legacy global ``halted=True`` still halts everything, regardless of
     any lane override.
  4. unknown lane names are rejected (typo/orphan-key protection).

No mocks: real ``StateEngine``, real disk via ``tmp_path``, real
``decision_gate._lane_halted`` reading real JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.equity.decision_gate import _lane_halted
from src.scanner.automation.state_engine import KNOWN_LANES, StateEngine


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "state.json"


# ── (a) independent lane control ──────────────────────────────────────────

def test_equity_can_run_while_oanda_fx_stays_halted(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    engine.set_halted(False)  # global unhalt — cascades all lanes to False
    engine.set_halted(True, lane="oanda_fx")  # re-halt ONLY the FX lane

    assert engine.get_halted(lane="oanda_fx") is True
    assert engine.get_halted(lane="equity") is False
    assert engine.get_halted(lane="brain") is False

    halted, readable, reason = _lane_halted(tmp_path, "equity")
    assert (halted, readable) == (False, True), reason
    fx_halted, fx_readable, fx_reason = _lane_halted(tmp_path, "oanda_fx")
    assert (fx_halted, fx_readable) == (True, True)
    assert fx_reason == "lane_halted:oanda_fx"


def test_oanda_fx_can_run_while_equity_stays_halted(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    engine.set_halted(False)
    engine.set_halted(True, lane="equity")

    assert engine.get_halted(lane="equity") is True
    assert engine.get_halted(lane="oanda_fx") is False
    assert engine.get_halted(lane="brain") is False

    halted, readable, _reason = _lane_halted(tmp_path, "oanda_fx")
    assert (halted, readable) == (False, True)
    eq_halted, eq_readable, eq_reason = _lane_halted(tmp_path, "equity")
    assert (eq_halted, eq_readable) == (True, True)
    assert eq_reason == "lane_halted:equity"


def test_unhalting_one_lane_does_not_touch_the_others(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    engine.set_halted(False)
    engine.set_halted(True, lane="oanda_fx")
    engine.set_halted(True, lane="brain")
    # Unhalt only equity — the other two must remain exactly as set.
    engine.set_halted(False, lane="equity")

    assert engine.get_halted(lane="equity") is False
    assert engine.get_halted(lane="oanda_fx") is True
    assert engine.get_halted(lane="brain") is True


# ── (b) missing/corrupt state fails CLOSED for every lane ─────────────────

@pytest.mark.parametrize("lane", KNOWN_LANES)
def test_decision_gate_lane_halted_fails_closed_on_absent_file(tmp_path: Path, lane: str):
    """No .claude/state.json at all -> every lane REFUSEs (never assumed safe)."""
    halted, readable, reason = _lane_halted(tmp_path, lane)
    assert halted is True
    assert readable is True
    assert reason == "lane_not_configured"


@pytest.mark.parametrize("lane", KNOWN_LANES)
def test_decision_gate_lane_halted_fails_closed_on_corrupt_file(tmp_path: Path, lane: str):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "state.json").write_text("{not valid json")

    halted, readable, reason = _lane_halted(tmp_path, lane)
    assert halted is True
    assert readable is False
    assert reason == "state_unreadable"


@pytest.mark.parametrize("lane", KNOWN_LANES)
def test_decision_gate_lane_halted_fails_closed_when_lane_missing_from_dict(
    tmp_path: Path, lane: str
):
    """halted_lanes exists (global unhalted) but omits THIS lane -> fail closed."""
    (tmp_path / ".claude").mkdir()
    other_lanes = {ln: False for ln in KNOWN_LANES if ln != lane}
    (tmp_path / ".claude" / "state.json").write_text(
        json.dumps({"halted": False, "halted_lanes": other_lanes})
    )

    halted, readable, reason = _lane_halted(tmp_path, lane)
    assert halted is True
    assert readable is True
    assert reason == "lane_not_configured"


@pytest.mark.parametrize("lane", KNOWN_LANES)
def test_state_engine_lane_fails_closed_when_lane_missing_from_configured_dict(
    tmp_path: Path, lane: str
):
    """StateEngine's own fail-closed guarantee: once halted_lanes is in use at
    all, an unlisted lane is never assumed safe — even though a TOTALLY
    unconfigured/corrupt file defers to the (also fail-closed-capable) global
    flag for backward compatibility (see get_halted's docstring)."""
    engine = StateEngine(_state_path(tmp_path))
    other_lanes = {ln: False for ln in KNOWN_LANES if ln != lane}
    _state_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _state_path(tmp_path).write_text(
        json.dumps({"halted": False, "halted_lanes": other_lanes, "schema_version": "2"})
    )
    assert engine.get_halted(lane=lane) is True


# ── (c) legacy global halted=True still halts everything ──────────────────

def test_legacy_global_halt_wins_over_any_lane_override(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    engine.set_halted(False)
    engine.set_halted(False, lane="oanda_fx")
    engine.set_halted(False, lane="equity")
    engine.set_halted(False, lane="brain")
    for lane in KNOWN_LANES:
        assert engine.get_halted(lane=lane) is False

    # Flip the GLOBAL flag back on (mirrors flatten_all / auto-halt / derisk —
    # every existing autonomous set_halted(True) call site is unmodified and
    # goes through this exact global path).
    engine.set_halted(True)
    for lane in KNOWN_LANES:
        assert engine.get_halted(lane=lane) is True, f"lane={lane} must be halted"

    for lane in KNOWN_LANES:
        halted, readable, reason = _lane_halted(tmp_path, lane)
        assert (halted, readable, reason) == (True, True, "global_halt")


def test_legacy_zero_arg_get_halted_and_set_halted_unchanged(tmp_path: Path):
    """The original, pre-per-lane call signature — no callers anywhere in the
    codebase need to change to stay correct."""
    engine = StateEngine(_state_path(tmp_path))
    assert engine.get_halted() is False
    engine.set_halted(True)
    assert engine.get_halted() is True
    engine.set_halted(False)
    assert engine.get_halted() is False


# ── (d) unknown lane names are rejected ────────────────────────────────────

def test_unknown_lane_name_rejected_on_read_and_write(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    with pytest.raises(ValueError):
        engine.get_halted(lane="crypto")
    with pytest.raises(ValueError):
        engine.set_halted(True, lane="crypto")


def test_get_lane_status_reflects_every_known_lane(tmp_path: Path):
    engine = StateEngine(_state_path(tmp_path))
    engine.set_halted(False)
    engine.set_halted(True, lane="oanda_fx")
    status = engine.get_lane_status()
    # Pre-existing gap fixed in passing (2026-07-07): this assertion hardcoded a
    # 3-lane dict from before crypto_momentum/track_b/crypto_carry were added to
    # KNOWN_LANES and was already stale (failing on this line) prior to this
    # session's crypto_carry addition -- update to the full known-lanes set
    # rather than leaving a silently-stale assertion.
    assert status == {"oanda_fx": True, "equity": False, "brain": False,
                      "crypto_momentum": False, "track_b": False, "crypto_carry": False}


# ── constructor-bound lane (used by src/scanner/execution.py so its guard
#    calls keep the literal `get_halted()` / `set_halted(True)` no-arg forms
#    the risk_monitor.sh L-004 guard-presence check greps for) ─────────────

def test_constructor_bound_lane_no_arg_calls_match_explicit_lane_calls(tmp_path: Path):
    path = _state_path(tmp_path)
    StateEngine(path).set_halted(False)
    StateEngine(path).set_halted(True, lane="oanda_fx")

    scoped = StateEngine(path, lane="oanda_fx")
    assert scoped.get_halted() is True  # no-arg call reads the bound lane
    assert StateEngine(path, lane="equity").get_halted() is False


def test_constructor_bound_lane_no_arg_set_halted_writes_only_that_lane(tmp_path: Path):
    path = _state_path(tmp_path)
    StateEngine(path).set_halted(False)
    StateEngine(path, lane="oanda_fx").set_halted(True)  # no-arg write, bound lane

    engine = StateEngine(path)
    assert engine.get_halted(lane="oanda_fx") is True
    assert engine.get_halted(lane="equity") is False
    assert engine.get_halted(lane="brain") is False


def test_explicit_lane_argument_overrides_constructor_bound_lane(tmp_path: Path):
    path = _state_path(tmp_path)
    StateEngine(path).set_halted(False)
    scoped = StateEngine(path, lane="oanda_fx")
    scoped.set_halted(True, lane="equity")  # explicit lane wins over the bound one

    assert StateEngine(path).get_halted(lane="equity") is True
    assert StateEngine(path).get_halted(lane="oanda_fx") is False


def test_constructor_rejects_unknown_bound_lane(tmp_path: Path):
    with pytest.raises(ValueError):
        StateEngine(_state_path(tmp_path), lane="crypto")
