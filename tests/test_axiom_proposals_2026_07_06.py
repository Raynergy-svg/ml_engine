"""AXIOM Activity panel — Accept/Deny endpoints (reproduce-then-resolve).

Proves, against real classes, real (tmp) disk, and a real FastAPI TestClient
mounted with ONLY dashboard/server/axiom_proposals.py's router — no
unittest.mock:
  (a) Deny never executes anything; it only records a disposition, and the
      proposal moves from pending to resolved on the next read.
  (b) Accept dispatches to the SAME existing safeguarded path each action
      type already has -- unhalt_lane still hits the ARM gate + eligibility
      check; increase_gross_leverage refuses a non-increase; config-adjuster
      actions go through real schema validation + the two-stage approve flow.
  (c) A missing/incorrect X-AXIOM-Confirm header refuses BEFORE any handler runs.
  (d) arm_live_gate / promote_model are honestly NOT wired -- 501, not a
      silent no-op and not a half-verified attempt.
  (e) proposal_id is stable (same action+params -> same id) and
      read_mind_window() dedupes + joins disposition correctly.
"""
from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.server import control_safety as cs
from src.agent_runtime import loop as agent_loop
from src.agent_runtime import policy
from src.agent_runtime import proposal_store
from src.scanner.automation import adjustment_approver as approver_mod
from src.scanner.automation import config_adjuster as adjuster_mod
from src.scanner.automation import state_engine as state_engine_mod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    monkeypatch.setattr(cs, "ARM_STATE_PATH", tmp_path / "control_arm_state.json")
    monkeypatch.setattr(state_engine_mod, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(adjuster_mod, "_ADJUSTMENTS_PATH", tmp_path / "config_adjustments.json")
    monkeypatch.setattr(adjuster_mod, "_PENDING_PATH", tmp_path / "pending_adjustments.json")
    monkeypatch.setattr(approver_mod, "_PENDING_PATH", tmp_path / "pending_adjustments.json")
    monkeypatch.setattr(approver_mod, "_APPROVED_PATH", tmp_path / "config_adjustments.json")
    monkeypatch.setattr(approver_mod, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(proposal_store, "DISPOSITIONS_PATH", tmp_path / "proposal_dispositions.json")
    monkeypatch.setattr(agent_loop, "CYCLES_PATH", tmp_path / "loop_cycles.jsonl")

    from dashboard.server import axiom_proposals

    app = FastAPI()
    app.include_router(axiom_proposals.router)
    return TestClient(app)


def _write_cycle(tmp_path, *, action, params, rationale="test rationale", cycle_id="cycle-test1"):
    proposal_id = policy.compute_proposal_id(action, params)
    cycle = {
        "cycle_id": cycle_id, "ts": "2026-07-06T12:00:00+00:00",
        "outcomes": [{
            "action": action, "tier": "escalation", "executed": False, "shadow": False,
            "proposal": True, "denied": False, "proposal_id": proposal_id,
            "detail": {"action": action, "rationale": rationale, "params": params},
        }],
    }
    path = tmp_path / "loop_cycles.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(cycle) + "\n")
    return proposal_id


# ---------------------------------------------------------------------------
# (a) deny never executes; moves pending -> resolved
# ---------------------------------------------------------------------------


def test_deny_records_disposition_and_never_executes(client, tmp_path):
    proposal_id = _write_cycle(tmp_path, action="unhalt_lane", params={"lane": "oanda_fx"})

    resp = client.post(f"/api/axiom_proposals/{proposal_id}/deny")

    assert resp.status_code == 200
    assert resp.json()["disposition"]["status"] == "denied"
    # never touched state.json
    assert not (tmp_path / "state.json").exists()

    listed = client.get("/api/axiom_proposals").json()
    pending_ids = [p["proposal_id"] for p in listed["pending"]]
    resolved_ids = [p["proposal_id"] for p in listed["resolved"]]
    assert proposal_id not in pending_ids
    assert proposal_id in resolved_ids


def test_deny_unknown_proposal_id_404s(client):
    resp = client.post("/api/axiom_proposals/does-not-exist/deny")
    assert resp.status_code == 404


def test_concurrent_record_disposition_does_not_lose_decisions(tmp_path):
    """Real threads writing distinct operator decisions concurrently must all
    survive -- the old unlocked load-modify-save could let one overlapping
    accept/deny request silently clobber another's freshly-recorded
    disposition (a lost operator decision, not just a lost log line)."""
    path = tmp_path / "proposal_dispositions.json"
    n_writers = 12
    errors = []

    def _write(i: int) -> None:
        try:
            proposal_store.record_disposition(
                f"proposal-{i}", status="denied", actor="test-operator",
                reason="concurrency test", path=path,
            )
        except Exception as exc:  # noqa: BLE001 - collected below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected exceptions from concurrent writers: {errors}"
    dispositions = proposal_store.list_dispositions(path=path)
    assert len(dispositions) == n_writers, "a concurrent disposition write was silently dropped"
    assert set(dispositions) == {f"proposal-{i}" for i in range(n_writers)}


# ---------------------------------------------------------------------------
# (c) confirm header gate
# ---------------------------------------------------------------------------


def test_accept_without_confirm_header_is_refused_before_any_handler_runs(client, tmp_path):
    proposal_id = _write_cycle(tmp_path, action="unhalt_lane", params={"lane": "oanda_fx"})

    resp = client.post(f"/api/axiom_proposals/{proposal_id}/accept")

    assert resp.status_code == 403
    assert not (tmp_path / "state.json").exists()  # nothing was touched
    assert proposal_store.get_disposition(proposal_id, path=tmp_path / "proposal_dispositions.json") is None


def test_accept_with_wrong_confirm_header_is_refused(client, tmp_path):
    proposal_id = _write_cycle(tmp_path, action="unhalt_lane", params={"lane": "oanda_fx"})

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "some_other_action"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# (b) accept dispatches to the SAME existing safeguarded path
# ---------------------------------------------------------------------------


def test_accept_unhalt_lane_still_requires_arm_even_via_activity_panel(client, tmp_path):
    """The whole point: Accept must NOT bypass the ARM gate unhalt already
    requires. Controls are disarmed by default (no arm state file yet)."""
    proposal_id = _write_cycle(tmp_path, action="unhalt_lane", params={"lane": "oanda_fx"})

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "unhalt_lane"},
    )

    assert resp.status_code == 403
    assert "DISARMED" in resp.json()["detail"]
    disposition = proposal_store.get_disposition(proposal_id, path=tmp_path / "proposal_dispositions.json")
    assert disposition["status"] == "denied"
    # state.json was never touched -- the ARM check runs before any StateEngine call
    assert not (tmp_path / "state.json").exists()


def test_accept_increase_gross_leverage_refuses_a_non_increase(client, tmp_path):
    cs.set_override("gross_leverage", 5.0)  # current = 5.0
    proposal_id = _write_cycle(
        tmp_path, action="increase_gross_leverage", params={"new_value": 3.0},  # DECREASE, mislabeled
    )

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "increase_gross_leverage"},
    )

    assert resp.status_code == 403
    assert "strictly greater" in resp.json()["detail"]
    assert cs.read_overrides()["gross_leverage"] == 5.0  # unchanged


def test_accept_increase_gross_leverage_succeeds_when_armed_and_actually_increasing(client, tmp_path):
    cs.set_override("gross_leverage", 2.0)
    cs.set_arm_state(True, "test-operator")
    proposal_id = _write_cycle(tmp_path, action="increase_gross_leverage", params={"new_value": 4.0})

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "increase_gross_leverage"},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["gross_leverage"] == 4.0
    assert cs.read_overrides()["gross_leverage"] == 4.0
    disposition = proposal_store.get_disposition(proposal_id, path=tmp_path / "proposal_dispositions.json")
    assert disposition["status"] == "accepted"


def test_accept_enable_new_exposure_goes_through_real_config_adjuster_and_approver(client, tmp_path):
    proposal_id = _write_cycle(
        tmp_path, action="enable_new_exposure",
        params={"key": "min_confidence", "value": 55.0, "reason": "test"},
    )

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "enable_new_exposure"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()["result"]
    assert body["result"] == "config_adjustment_approved"
    approved = json.loads((tmp_path / "config_adjustments.json").read_text())
    assert any(a.get("key") == "min_confidence" for a in approved.get("history", approved if isinstance(approved, list) else []))


def test_accept_change_strategy_or_code_refuses_an_unknown_config_key(client, tmp_path):
    proposal_id = _write_cycle(
        tmp_path, action="change_strategy_or_code",
        params={"key": "totally_made_up_field_xyz", "value": 1, "reason": "test"},
    )

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": "change_strategy_or_code"},
    )

    assert resp.status_code == 403
    assert not (tmp_path / "config_adjustments.json").exists() or json.loads(
        (tmp_path / "config_adjustments.json").read_text()
    ) in ({}, [], {"history": []})


# ---------------------------------------------------------------------------
# (d) arm_live_gate / promote_model deliberately not wired
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["arm_live_gate", "promote_model"])
def test_not_yet_wired_actions_return_501_not_a_silent_noop(client, tmp_path, action):
    proposal_id = _write_cycle(tmp_path, action=action, params={})

    resp = client.post(
        f"/api/axiom_proposals/{proposal_id}/accept",
        headers={"X-AXIOM-Confirm": action},
    )

    assert resp.status_code == 501
    assert proposal_store.get_disposition(proposal_id, path=tmp_path / "proposal_dispositions.json") is None


# ---------------------------------------------------------------------------
# (e) proposal_id stability + read_mind_window dedup/join
# ---------------------------------------------------------------------------


def test_proposal_id_is_stable_for_identical_action_and_params():
    a = policy.compute_proposal_id("unhalt_lane", {"lane": "oanda_fx"})
    b = policy.compute_proposal_id("unhalt_lane", {"lane": "oanda_fx"})
    c = policy.compute_proposal_id("unhalt_lane", {"lane": "equity"})
    assert a == b
    assert a != c


def test_read_mind_window_dedupes_repeated_proposal_across_cycles_and_joins_disposition(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, "CYCLES_PATH", tmp_path / "loop_cycles.jsonl")
    monkeypatch.setattr(proposal_store, "DISPOSITIONS_PATH", tmp_path / "proposal_dispositions.json")
    autonomy_path = tmp_path / "loop_autonomy.json"

    # same proposal re-surfaces in 3 separate cycles (the LLM keeps re-noticing
    # the same unresolved defect) -- must appear ONCE in pending_operator_proposals.
    pid = None
    for i in range(3):
        pid = _write_cycle(
            tmp_path, action="unhalt_lane", params={"lane": "oanda_fx"}, cycle_id=f"cycle-{i}",
        )

    window = agent_loop.read_mind_window(
        cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=autonomy_path,
    )
    assert len(window["pending_operator_proposals"]) == 1
    assert window["pending_operator_proposals"][0]["proposal_id"] == pid
    assert window["resolved_proposals"] == []

    proposal_store.record_disposition(pid, status="denied", actor="test-operator")

    window2 = agent_loop.read_mind_window(
        cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=autonomy_path,
    )
    assert window2["pending_operator_proposals"] == []
    assert len(window2["resolved_proposals"]) == 1
    assert window2["resolved_proposals"][0]["disposition"]["status"] == "denied"
