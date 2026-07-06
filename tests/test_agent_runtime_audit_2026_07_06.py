"""AXIOM Agent Runtime — audit extends the existing control-audit log (real disk)."""
from __future__ import annotations

import json

from dashboard.server import control_safety as cs
from src.agent_runtime import audit as art_audit


def test_record_appends_to_the_shared_control_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")

    art_audit.record(
        action="halt_lane",
        tier="deescalation",
        actor="test-agent",
        allowed=True,
        outcome="executed",
        reason="preflight ok",
        params={"lane": "equity"},
        result={"result": "halted"},
    )

    assert cs.AUDIT_PATH.exists()
    rows = [json.loads(ln) for ln in cs.AUDIT_PATH.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "agent_runtime"
    assert row["actor"] == "test-agent"
    assert row["tier"] == "deescalation"
    assert row["allowed"] is True
    assert row["outcome"] == "executed"
    assert "ts" in row  # stamped by the shared cs.audit() writer


def test_record_and_dashboard_entries_share_one_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")

    cs.audit({"action": "halt", "allowed": True, "result": "halted"}, actor="dashboard-user")
    art_audit.record(
        action="halt_lane", tier="deescalation", actor="agent-loop",
        allowed=True, outcome="executed",
    )

    rows = [json.loads(ln) for ln in cs.AUDIT_PATH.read_text().splitlines() if ln.strip()]
    assert len(rows) == 2
    assert rows[0]["actor"] == "dashboard-user"
    assert "source" not in rows[0]  # pre-existing dashboard entries are untouched/unaffected
    assert rows[1]["source"] == "agent_runtime"


def test_read_recent_returns_only_agent_runtime_entries_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")

    cs.audit({"action": "halt", "allowed": True, "result": "halted"}, actor="dashboard-user")
    art_audit.record(action="a1", tier="operational", actor="x", allowed=True, outcome="executed")
    art_audit.record(action="a2", tier="operational", actor="x", allowed=True, outcome="executed")

    rows = art_audit.read_recent(limit=10)
    assert [r["action"] for r in rows] == ["a2", "a1"]
    assert all(r["source"] == "agent_runtime" for r in rows)


def test_read_recent_on_missing_file_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "does_not_exist.jsonl")
    assert art_audit.read_recent() == []
