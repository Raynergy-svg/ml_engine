"""No-mock tests for the AXIOM control immutables. Real ScannerConfig, real disk.

Proves a crafted request cannot reach live / over-leverage / unknown actions / smuggled
environment overrides — the structural guarantees behind the (disabled) control surface.
"""
import json

import pytest

from dashboard.server import control_safety as cs


def test_assert_practice_returns_practice():
    assert cs.assert_practice() == "practice"


def test_halt_unhalt_allowed_no_params():
    assert cs.enforce("halt", {}) == {}
    assert cs.enforce("unhalt", {}) == {}


def test_unknown_action_denied():
    with pytest.raises(cs.ControlDenied):
        cs.enforce("promote_model", {})
    with pytest.raises(cs.ControlDenied):
        cs.enforce("flip_to_live", {})


def test_smuggled_environment_override_denied():
    for bad in ("environment", "env", "url", "base_url", "account", "account_id", "oanda_environment"):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("halt", {bad: "live"})


def test_leverage_clamped_to_cap():
    assert cs.enforce("set_gross_leverage", {"gross_leverage": 5})["gross_leverage"] == 5.0
    assert cs.enforce("set_gross_leverage", {"gross_leverage": cs.LEVERAGE_CAP})["gross_leverage"] == cs.LEVERAGE_CAP
    for bad in (999, cs.LEVERAGE_CAP + 0.01, -1, float("inf"), float("nan"), "abc", None):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("set_gross_leverage", {"gross_leverage": bad})


def test_loop_whitelist():
    assert cs.enforce("start_loop", {"loop": "trend"})["loop"] == "trend"
    assert cs.enforce("stop_loop", {"loop": "tier7"})["loop"] == "tier7"
    for bad in ("evil", "rm -rf", "", None):
        with pytest.raises(cs.ControlDenied):
            cs.enforce("start_loop", {"loop": bad})


def test_audit_appends_real_line(tmp_path):
    cs.AUDIT_PATH = tmp_path / "control_audit.jsonl"   # real path, not a mock
    cs.audit({"action": "halt", "allowed": True, "reason": "test", "result": "halted"})
    lines = cs.AUDIT_PATH.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "halt" and rec["allowed"] is True and "ts" in rec
