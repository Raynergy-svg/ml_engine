from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from cli.model_management import model_status
from memory_client import MLEngineMemory


def test_model_status_decisions_prints_runtime_decisions(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    memory = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory")
    now = datetime.now(timezone.utc)
    memory.update_runtime_state(
        "buddy_runtime",
        {
            "last_trade_execution_decision": {
                "timestamp": (now - timedelta(minutes=12)).isoformat(),
                "instrument": "EUR_USD",
                "direction": "long",
                "decision_type": "dry_run_signal",
                "reason": "All model gates passed and intelligent review approved",
            },
            "last_no_trade_decision": {
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "instrument": "GBP_USD",
                "decision_type": "intelligent_reject",
                "reason": "event risk",
                "risk_flags": ["economic_event"],
            },
        },
    )

    model_status("config.yaml", decisions=True)

    out = capsys.readouterr().out.lower()
    assert "last trade execution decision" in out
    assert "recorded 12m ago" in out
    assert "all model gates passed and intelligent review approved" in out
    assert "last no-trade decision" in out
    assert "recorded 2h ago" in out
    assert "event risk" in out
    assert "economic_event" in out
