"""No-mock tests for the AXIOM control immutables. Real ScannerConfig, real disk.

Proves a crafted request cannot reach live / over-leverage / unknown actions / smuggled
environment overrides — the structural guarantees behind the (disabled) control surface.
"""
import json

import pytest

from dashboard.server import control_safety as cs


def test_assert_practice_returns_practice():
    assert cs.assert_practice() == "practice"


def test_halt_allowed_no_params_regardless_of_arm_state():
    # halt is risk-REDUCING and must never be gated by arm state.
    assert cs.enforce("halt", {}) == {}


def test_unknown_action_denied():
    with pytest.raises(cs.ControlDenied):
        cs.enforce("promote_model", {})
    with pytest.raises(cs.ControlDenied):
        cs.enforce("flip_to_live", {})


def test_smuggled_environment_override_denied():
    for bad in ("environment", "env", "url", "base_url", "account", "account_id", "oanda_environment"):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("halt", {bad: "live"})


def test_leverage_clamped_to_cap(tmp_path, monkeypatch):
    # set_gross_leverage now requires ARM (investigation finding, 2026-07-01) —
    # this test is about the leverage-clamp validation itself, so arm first.
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.set_arm_state(True, actor="test")
    assert cs.enforce("set_gross_leverage", {"gross_leverage": 5})["gross_leverage"] == 5.0
    assert cs.enforce("set_gross_leverage", {"gross_leverage": cs.LEVERAGE_CAP})["gross_leverage"] == cs.LEVERAGE_CAP
    for bad in (999, cs.LEVERAGE_CAP + 0.01, -1, float("inf"), float("nan"), "abc", None):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("set_gross_leverage", {"gross_leverage": bad})


def test_loop_whitelist(tmp_path, monkeypatch):
    # start_loop now requires ARM; stop_loop never does (risk-reducing).
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.set_arm_state(True, actor="test")
    assert cs.enforce("start_loop", {"loop": "trend"})["loop"] == "trend"
    assert cs.enforce("stop_loop", {"loop": "tier7"})["loop"] == "tier7"
    for bad in ("evil", "rm -rf", "", None):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("start_loop", {"loop": bad})


def test_unhalt_eligibility_is_deterministic_from_disk():
    # Returns a checks dict when eligible, or raises ControlDenied with reasons.
    # Either way it must be practice-pinned and surface the 3 signals.
    try:
        checks = cs.assert_unhalt_eligible()
        assert "drawdown_pct" in checks and "gates_green" in checks and "oldest_model_age_days" in checks
    except cs.ControlDenied as exc:
        assert "unhalt blocked" in str(exc)


def test_set_override_clamps_and_writes(tmp_path):
    cs.OVERRIDES_PATH = tmp_path / "control_overrides.json"  # real path, not a mock
    cs.set_override("gross_leverage", cs.validate_leverage(5))
    import json
    saved = json.loads(cs.OVERRIDES_PATH.read_text())
    assert saved["gross_leverage"] == 5.0 and saved["_source"] == "axiom_control"
    assert cs.read_overrides()["gross_leverage"] == 5.0
    with pytest.raises(cs.ControlDenied):
        cs.set_override("evil_key", 1)


def test_read_overrides_missing_or_corrupt_returns_empty(tmp_path):
    cs.OVERRIDES_PATH = tmp_path / "missing.json"
    assert cs.read_overrides() == {}
    cs.OVERRIDES_PATH = tmp_path / "corrupt.json"
    cs.OVERRIDES_PATH.write_text("{bad", encoding="utf-8")
    assert cs.read_overrides() == {}


def test_audit_appends_real_line(tmp_path):
    cs.AUDIT_PATH = tmp_path / "control_audit.jsonl"   # real path, not a mock
    cs.audit({"action": "halt", "allowed": True, "reason": "test", "result": "halted"},
             actor="test-actor")
    lines = cs.AUDIT_PATH.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "halt" and rec["allowed"] is True and "ts" in rec
    assert rec["actor"] == "test-actor"


def test_audit_requires_actor_structurally():
    """actor is a required keyword arg — omitting it is a TypeError, not a silent
    gap (investigation finding, 2026-07-01: attribution was permanently absent
    because it was never a required field)."""
    with pytest.raises(TypeError):
        cs.audit({"action": "halt", "allowed": True})  # no actor= -> must fail


# --------------------------------------------------------------------------- #
# ARM lockdown (investigation finding, 2026-07-01): a real flatten + repeated
# leverage swings fired against the LIVE practice account during an AXIOM
# dashboard QA/test pass. Every armed-required action must be a structural
# no-op while disarmed (the default), regardless of how the request got there.
# --------------------------------------------------------------------------- #
def test_default_disarmed_blocks_every_armed_required_action(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")  # no file -> disarmed
    assert cs.is_armed() is False
    for action, params in (("flatten", {}), ("set_gross_leverage", {"gross_leverage": 3.0}),
                           ("start_loop", {"loop": "trend"}), ("unhalt", {})):
        with pytest.raises(cs.ControlDenied, match="DISARMED"):
            cs.enforce(action, params)


def test_halt_stop_loop_arm_disarm_never_require_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")  # disarmed
    assert cs.enforce("halt", {}) == {}
    assert cs.enforce("stop_loop", {"loop": "trend"})["loop"] == "trend"


# --------------------------------------------------------------------------- #
# Per-lane halt/unhalt (2026-07-02): an optional, validated ``lane`` param on
# halt/unhalt only — ARM gating and the eligibility gate are untouched.
# --------------------------------------------------------------------------- #
def test_halt_accepts_optional_valid_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")  # disarmed; halt never needs arm
    assert cs.enforce("halt", {"lane": "oanda_fx"}) == {"lane": "oanda_fx"}
    assert cs.enforce("halt", {}) == {}  # lane remains fully optional


def test_halt_rejects_unknown_lane():
    with pytest.raises(cs.ControlDenied):
        cs.enforce("halt", {"lane": "crypto"})


def test_halt_rejects_extra_params_alongside_lane():
    with pytest.raises(cs.ControlDenied):
        cs.enforce("halt", {"lane": "oanda_fx", "extra": "nope"})


def test_unhalt_with_lane_still_requires_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")  # disarmed
    with pytest.raises(cs.ControlDenied, match="DISARMED"):
        cs.enforce("unhalt", {"lane": "equity"})
    cs.set_arm_state(True, actor="test")
    assert cs.enforce("unhalt", {"lane": "equity"}) == {"lane": "equity"}
    assert cs.enforce("arm", {}) == {}
    assert cs.enforce("disarm", {}) == {}


def test_arm_then_armed_required_actions_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.set_arm_state(True, actor="operator")
    assert cs.is_armed() is True
    assert cs.enforce("set_gross_leverage", {"gross_leverage": 3.0})["gross_leverage"] == 3.0
    assert cs.enforce("start_loop", {"loop": "trend"})["loop"] == "trend"
    # flatten/unhalt still take no params, but no longer raise for arm reasons:
    assert cs.enforce("flatten", {}) == {}


def test_disarm_immediately_blocks_again(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.set_arm_state(True, actor="operator")
    assert cs.is_armed() is True
    cs.set_arm_state(False, actor="operator")
    assert cs.is_armed() is False
    with pytest.raises(cs.ControlDenied, match="DISARMED"):
        cs.enforce("flatten", {})


def test_arm_expires_after_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    import time
    state = {"armed": True, "armed_at": time.time() - (cs.ARM_TTL_SECONDS + 5), "actor": "operator"}
    cs.ARM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cs.ARM_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    assert cs.is_armed() is False   # stale arm from a forgotten session must NOT authorize
    with pytest.raises(cs.ControlDenied, match="DISARMED"):
        cs.enforce("flatten", {})


def test_is_armed_fails_closed_on_corrupt_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.ARM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cs.ARM_STATE_PATH.write_text("{not valid json", encoding="utf-8")
    assert cs.is_armed() is False  # corrupt -> disarmed, never crash


def test_is_armed_fails_closed_on_missing_armed_at(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    cs.ARM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cs.ARM_STATE_PATH.write_text(json.dumps({"armed": True}), encoding="utf-8")  # no armed_at
    assert cs.is_armed() is False


def test_set_arm_state_writes_real_atomic_state(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    state = cs.set_arm_state(True, actor="operator-session-42")
    assert state["armed"] is True and state["actor"] == "operator-session-42"
    on_disk = json.loads(cs.ARM_STATE_PATH.read_text())
    assert on_disk["armed"] is True
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())  # atomic, no leftover tmp


# --------------------------------------------------------------------------- #
# S-A1: unhalt eligibility must FAIL-CLOSED — unverifiable rails BLOCK, never pass.
# Real files under a tmp REPO_ROOT (monkeypatched path constant — no mocks).
# --------------------------------------------------------------------------- #
def _build_repo(tmp_path, *, nav=100_000.0, peak=100_500.0, verdict_checks=None,
                write_peak=True, write_verdict=True, verdict_age_s=0.0):
    import os
    import time
    (tmp_path / "trained_data" / "oanda").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "loop").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trained_data" / "oanda" / "account_state.json").write_text(
        json.dumps({"nav": nav}), encoding="utf-8")
    if write_peak:
        (tmp_path / "trained_data" / "oanda" / "peak_nav.json").write_text(
            json.dumps({"peak_nav": peak}), encoding="utf-8")
    if write_verdict:
        vp = tmp_path / ".claude" / "loop" / "verdict.json"
        checks = verdict_checks if verdict_checks is not None else [{"name": "practice", "ok": True}]
        vp.write_text(json.dumps({"checks": checks}), encoding="utf-8")
        if verdict_age_s:
            old = time.time() - verdict_age_s
            os.utime(vp, (old, old))
    return tmp_path


def test_unhalt_eligible_when_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "REPO_ROOT", _build_repo(tmp_path))
    checks = cs.assert_unhalt_eligible()  # GREEN+fresh verdict, valid peak/nav -> eligible
    assert checks["gates_green"] is True and checks["drawdown_pct"] is not None


def test_unhalt_blocked_when_verdict_missing(tmp_path, monkeypatch):
    # The core S-A1 bug: a missing verdict used to pass (gates_green:False, no reason).
    monkeypatch.setattr(cs, "REPO_ROOT", _build_repo(tmp_path, write_verdict=False))
    with pytest.raises(cs.ControlDenied, match="verdict missing/empty"):
        cs.assert_unhalt_eligible()


def test_unhalt_blocked_when_peak_nav_missing(tmp_path, monkeypatch):
    # Missing peak_nav used to silently skip the entire drawdown rail.
    monkeypatch.setattr(cs, "REPO_ROOT", _build_repo(tmp_path, write_peak=False))
    with pytest.raises(cs.ControlDenied, match="drawdown rail UNVERIFIABLE"):
        cs.assert_unhalt_eligible()


def test_unhalt_blocked_when_verdict_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "REPO_ROOT",
                        _build_repo(tmp_path, verdict_age_s=cs.UNHALT_VERDICT_MAX_AGE_S + 3600))
    with pytest.raises(cs.ControlDenied, match="stale"):
        cs.assert_unhalt_eligible()


def test_unhalt_blocked_when_drawdown_breaches_rail(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "REPO_ROOT",
                        _build_repo(tmp_path, nav=70_000.0, peak=100_000.0))  # 30% dd
    with pytest.raises(cs.ControlDenied, match="drawdown"):
        cs.assert_unhalt_eligible()
