"""HTTP boundary tests for the governed Phase-M training routes."""

from __future__ import annotations

import subprocess
from decimal import Decimal

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI

from dashboard.server.training_api import _completion_fields, router
from dashboard.server.training_identity import require_runtime_git_commit


def _client():
    app = FastAPI()
    app.include_router(router)
    return testclient.TestClient(app)


def _credential():
    return {
        "action": "run",
        "subject": "a" * 64,
        "nonce": "single-use-nonce",
        "issued_at": "2026-07-14T12:00:00Z",
        "expires_at": "2026-07-14T12:01:00Z",
        "key_id": "ed25519:" + "b" * 32,
        "signature_b64": "not-authority",
    }


def test_read_cockpit_is_mounted_and_truthful():
    response = _client().get("/api/axiom_training")
    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert "data" in payload
    assert "evidence" in payload
    assert payload["controls"]["operator_private_key_on_server"] is False


def test_run_requires_structured_signed_body_not_legacy_confirm_header():
    response = _client().post(
        "/api/axiom_training/run",
        headers={"x-axiom-confirm": "run"},
        json={"request": {"dataset_sha256": "a" * 64, "lane": "risk_target", "program": "risk_target_baseline"}},
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "credential" for error in response.json()["detail"])


def test_run_route_declares_accepted_status_for_queued_work():
    route = next(route for route in router.routes if route.path == "/api/axiom_training/run")
    assert route.status_code == 202


def test_run_is_fail_closed_when_governed_controls_are_disabled(monkeypatch):
    monkeypatch.delenv("AXIOM_CONTROL_ENABLED", raising=False)
    response = _client().post(
        "/api/axiom_training/run",
        json={
            "credential": _credential(),
            "request": {
                "dataset_sha256": "a" * 64,
                "lane": "risk_target",
                "program": "risk_target_baseline",
            },
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "AXIOM governed controls are disabled"


def test_run_requires_explicit_training_program():
    response = _client().post(
        "/api/axiom_training/run",
        json={
            "credential": _credential(),
            "request": {"dataset_sha256": "a" * 64, "lane": "risk_target"},
        },
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "program" for error in response.json()["detail"])


def test_promotion_rejects_extra_fields_before_control_plane():
    body = {
        "credential": {**_credential(), "action": "promote"},
        "disposition": {},
        "bypass_transition_policy": True,
    }
    response = _client().post("/api/axiom_training/promote", json=body)
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])


def test_p2_program_cannot_enter_the_baseline_run_route():
    response = _client().post(
        "/api/axiom_training/run",
        json={
            "credential": _credential(),
            "request": {
                "dataset_sha256": "a" * 64,
                "lane": "risk_target",
                "program": "risk_target_p2",
            },
        },
    )
    assert response.status_code == 422


def test_runtime_git_identity_refuses_dirty_training_source(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "src" / "evidence" / "marker.py"
    source.parent.mkdir(parents=True)
    source.write_text("IDENTITY = 1\n", encoding="utf-8")
    for args in (
        ("init",),
        ("config", "user.email", "axiom-test@example.invalid"),
        ("config", "user.name", "AXIOM Test"),
        ("add", "."),
        ("commit", "-m", "frozen training source"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    commit = require_runtime_git_commit(repo)
    assert len(commit) == 40

    source.write_text("IDENTITY = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        require_runtime_git_commit(repo)


def test_completion_fields_report_exact_identity_and_local_cost():
    result = {
        "job_manifest_digest": "a" * 64,
        "job_manifest": {
            "git_commit": "b" * 40,
            "container_digest": "c" * 64,
            "configuration_digest": "d" * 64,
            "feature_pipeline_version": "risk-target-v1",
            "resource_class": "cpu-small",
            "assigned_unit": "risk_target/all_heads/fold_0",
            "trial_budget": 1,
        },
        "outcomes": {},
    }
    fields = _completion_fields(result, 1800.0, Decimal("0.40"))
    assert fields["git_commit"] == "b" * 40
    assert fields["cost"] == 0.2
    assert fields["cost_currency"] == "USD"
    assert fields["cost_basis"] == "configured_local_cpu_hourly_rate"
