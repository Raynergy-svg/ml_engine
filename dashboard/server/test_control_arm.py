"""Reproduce-then-resolve: AXIOM control ARM lockdown + actor attribution
(2026-07-01 investigation — a real flatten + repeated leverage swings fired
against the live practice account during a dashboard QA/test pass).

HTTP-level tests (real FastAPI app + real control_safety, no mocks) proving:
  1. A DISARMED dashboard cannot place any live-order control action — the
     endpoint refuses (403) and the attempt is audited, never reaching OANDA.
  2. halt/stop_loop/arm/disarm are NEVER gated by arm state (risk-reducing).
  3. Arming, then acting, works; disarming immediately blocks again.
  4. Every control action's audit entry carries a real actor (header or IP).
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.server import control
from dashboard.server import control_safety as cs


@pytest.fixture
def client(tmp_path, monkeypatch):
    # chdir into tmp so cwd-relative writers reached through the control layer
    # (StateEngine -> .claude/state.json on halt/unhalt) hit tmp_path, never
    # the repo's live operator state. Caught by the root-conftest pollution
    # guard (2026-07-18): these tests were flipping the real state.json halted
    # flag on every run — the exact Hard NO #2 surface the guard protects.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir(exist_ok=True)
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    app = FastAPI()
    app.include_router(control.router)
    return TestClient(app)


def _last_audit_entry(tmp_path):
    lines = (tmp_path / "control_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_disarmed_flatten_refused_and_never_reaches_oanda(client, tmp_path):
    resp = client.post("/api/control/flatten", headers={"x-axiom-confirm": "flatten"})
    assert resp.status_code == 403
    assert "DISARMED" in resp.json()["detail"]
    entry = _last_audit_entry(tmp_path)
    assert entry["action"] == "flatten" and entry["allowed"] is False
    assert "actor" in entry  # attribution present even on a denial


def test_disarmed_set_gross_leverage_refused(client):
    resp = client.post("/api/control/set_gross_leverage",
                       headers={"x-axiom-confirm": "set_gross_leverage"},
                       json={"params": {"gross_leverage": 7.0}})
    assert resp.status_code == 403
    assert "DISARMED" in resp.json()["detail"]


def test_disarmed_start_loop_refused(client):
    resp = client.post("/api/control/start_loop",
                       headers={"x-axiom-confirm": "start_loop"},
                       json={"params": {"loop": "trend"}})
    assert resp.status_code == 403
    assert "DISARMED" in resp.json()["detail"]


def test_disarmed_unhalt_refused(client):
    resp = client.post("/api/control/unhalt", headers={"x-axiom-confirm": "unhalt"})
    assert resp.status_code == 403
    assert "DISARMED" in resp.json()["detail"]


def test_halt_stop_loop_never_require_arm(client):
    r1 = client.post("/api/control/halt", headers={"x-axiom-confirm": "halt"})
    assert r1.status_code == 200 and r1.json()["ok"] is True
    r2 = client.post("/api/control/stop_loop", headers={"x-axiom-confirm": "stop_loop"},
                     json={"params": {"loop": "trend"}})
    assert r2.status_code == 200 and r2.json()["ok"] is True


def test_arm_then_set_gross_leverage_succeeds_then_disarm_blocks_again(client, tmp_path):
    arm_resp = client.post("/api/control/arm", headers={"x-axiom-confirm": "arm"})
    assert arm_resp.status_code == 200
    assert arm_resp.json()["state"]["armed"] is True

    lev_resp = client.post("/api/control/set_gross_leverage",
                           headers={"x-axiom-confirm": "set_gross_leverage"},
                           json={"params": {"gross_leverage": 5.0}})
    assert lev_resp.status_code == 200
    assert lev_resp.json()["gross_leverage"] == 5.0

    disarm_resp = client.post("/api/control/disarm", headers={"x-axiom-confirm": "disarm"})
    assert disarm_resp.status_code == 200
    assert disarm_resp.json()["state"]["armed"] is False

    blocked = client.post("/api/control/set_gross_leverage",
                          headers={"x-axiom-confirm": "set_gross_leverage"},
                          json={"params": {"gross_leverage": 3.0}})
    assert blocked.status_code == 403
    assert "DISARMED" in blocked.json()["detail"]


def test_actor_header_recorded_in_audit(client, tmp_path):
    client.post("/api/control/halt", headers={"x-axiom-confirm": "halt", "x-axiom-actor": "dashboard-session-abc123"})
    entry = _last_audit_entry(tmp_path)
    assert entry["actor"] == "dashboard-session-abc123"


def test_actor_falls_back_to_client_ip_when_no_header(client, tmp_path):
    client.post("/api/control/halt", headers={"x-axiom-confirm": "halt"})  # no x-axiom-actor
    entry = _last_audit_entry(tmp_path)
    assert entry["actor"] != "unknown"  # TestClient supplies a client host
    assert entry["actor"]  # non-empty, some attribution recorded


def test_state_reports_arm_status(client):
    resp = client.get("/api/control/state")
    assert resp.json()["armed"] is False
    client.post("/api/control/arm", headers={"x-axiom-confirm": "arm"})
    resp2 = client.get("/api/control/state")
    assert resp2.json()["armed"] is True
    assert resp2.json()["arm_expires_at"] is not None
