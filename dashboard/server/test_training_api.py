"""HTTP boundary tests for the governed Phase-M training routes."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI

from dashboard.server.training_api import router


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
        json={"request": {"dataset_sha256": "a" * 64, "lane": "risk_target"}},
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "credential" for error in response.json()["detail"])


def test_run_is_fail_closed_when_governed_controls_are_disabled(monkeypatch):
    monkeypatch.delenv("AXIOM_CONTROL_ENABLED", raising=False)
    response = _client().post(
        "/api/axiom_training/run",
        json={
            "credential": _credential(),
            "request": {"dataset_sha256": "a" * 64, "lane": "risk_target"},
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "AXIOM governed controls are disabled"


def test_promotion_rejects_extra_fields_before_control_plane():
    body = {
        "credential": {**_credential(), "action": "promote"},
        "disposition": {},
        "bypass_transition_policy": True,
    }
    response = _client().post("/api/axiom_training/promote", json=body)
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])
