from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dashboard.server import data_sources as ds
from dashboard.server.app import app
from src.axiom_operator import session as op_session
from src.axiom_operator.policy import PolicyDecision
from src.axiom_operator.runner import OperatorRunResult


def test_read_axiom_operator_empty_state(monkeypatch, tmp_path: Path):
    session_path = tmp_path / "operator_session.json"
    decisions_path = tmp_path / "operator_decisions.jsonl"

    def snapshot():
        return op_session.read_operator_snapshot(
            session_path=session_path,
            decisions_path=decisions_path,
        )

    monkeypatch.setattr(ds, "read_operator_snapshot", snapshot, raising=False)
    out = op_session.read_operator_snapshot(session_path=session_path, decisions_path=decisions_path)

    assert out["has_run"] is False
    assert out["session"]["status"] == "idle"
    assert out["session"]["provider"] == "claude_subscription"
    assert out["recent_decisions"] == []


def test_axiom_operator_api_reads_session(monkeypatch, tmp_path: Path):
    session_path = tmp_path / "operator_session.json"
    decisions_path = tmp_path / "operator_decisions.jsonl"
    session_path.write_text(json.dumps({
        "session_id": "test",
        "status": "held",
        "provider": "claude_subscription",
        "epoch": 3,
        "blocked_reasons": ["subscription cooldown"],
        "open_incidents": [],
        "last_tools": [],
    }), encoding="utf-8")
    decisions_path.write_text(json.dumps({
        "ts": "2026-07-04T00:00:00+00:00",
        "epoch": 3,
        "status": "held",
        "action": "observe",
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        ds,
        "read_axiom_operator",
        lambda: op_session.read_operator_snapshot(session_path=session_path, decisions_path=decisions_path),
    )

    client = TestClient(app)
    resp = client.get("/api/axiom_operator")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_run"] is True
    assert body["session"]["status"] == "held"
    assert body["last_decision"]["action"] == "observe"


def test_axiom_operator_run_endpoint_invokes_epoch(monkeypatch):
    calls = {"n": 0}

    def fake_run_once(self, *, observation=None, timeout_seconds=120, allow_api_key=None):
        calls["n"] += 1
        assert observation["source"] == "dashboard"
        return OperatorRunResult(
            ok=True,
            status="acted",
            epoch=4,
            action="recheck",
            policy=PolicyDecision("recheck", True, "safe_auto", False, "defensive operator action"),
            session={"status": "acted", "epoch": 4},
            decision={"status": "acted", "action": "recheck", "epoch": 4},
        )

    monkeypatch.setattr("src.axiom_operator.runner.AxiomOperator.run_once", fake_run_once)

    client = TestClient(app)
    resp = client.post("/api/axiom_operator/run")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["status"] == "acted"
    assert body["action"] == "recheck"
    assert calls["n"] == 1
