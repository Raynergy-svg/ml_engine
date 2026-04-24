"""Diagnostics screen live wiring tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.tui.data_provider import DashboardSnapshot
from src.tui.screens.diagnostics_screen import DiagnosticsScreen


def test_live_log_entries_read_standard_logs_jsonl_and_scan_report(tmp_path: Path):
    root = tmp_path
    (root / "logs").mkdir()
    (root / "trained_data").mkdir()

    (root / "logs" / "buddy_tui.log").write_text(
        "2026-04-24 10:00:00 - src.tui.data_provider - WARNING - Account refresh failed\n",
        encoding="utf-8",
    )
    (root / "logs" / "reflection_log.jsonl").write_text(
        json.dumps({
            "ts": "2026-04-24T10:01:00+00:00",
            "trade_id": "SELFHEAL-C1",
            "success": False,
            "error": "claude rate-limited",
        }) + "\n",
        encoding="utf-8",
    )
    (root / "trained_data" / "scan_diagnostics_report.json").write_text(
        json.dumps({
            "timestamp": "2026-04-24T10:02:00+00:00",
            "stalled": True,
            "diagnostic_summary": "STALLED: no tradeable pair",
        }),
        encoding="utf-8",
    )

    screen = DiagnosticsScreen(project_root=str(root), live=True)
    entries = screen._load_live_log_entries()

    assert any(e["sev"] == "WARN" and "Account refresh failed" in e["msg"] for e in entries)
    assert any(e["sev"] == "ERR" and "claude rate-limited" in e["msg"] for e in entries)
    assert any(e["sev"] == "WARN" and "STALLED" in e["msg"] for e in entries)


def test_maintenance_tasks_use_real_artifacts(tmp_path: Path):
    root = tmp_path
    (root / ".claude").mkdir()
    (root / "trained_data" / "models" / "weight_snapshots").mkdir(parents=True)

    (root / ".claude" / "learnings.md").write_text("- learning\n", encoding="utf-8")
    (root / ".claude" / "config_adjustments.json").write_text("{}", encoding="utf-8")
    (root / "trained_data" / "health_registry_log.json").write_text(
        json.dumps({"cycles_rejected": 0, "module_states": {}}),
        encoding="utf-8",
    )
    (root / "trained_data" / "scan_diagnostics_report.json").write_text(
        json.dumps({"stalled": False}),
        encoding="utf-8",
    )
    (root / "trained_data" / "scan_diagnostics_state.json").write_text(
        json.dumps({"total_scans": 3}),
        encoding="utf-8",
    )
    (root / "trained_data" / ".autonomous_retrain_state.json").write_text(
        json.dumps({
            "last_completed_at": datetime.now(timezone.utc).timestamp(),
            "last_status": "success",
        }),
        encoding="utf-8",
    )
    (root / "trained_data" / "models" / "weight_snapshots" / "1.json").write_text(
        "{}",
        encoding="utf-8",
    )

    screen = DiagnosticsScreen(project_root=str(root), live=True)
    tasks = screen._build_maintenance_tasks(datetime.now(timezone.utc), DashboardSnapshot(scanner_ready=True))
    by_name = {task["name"]: task for task in tasks}

    assert by_name["Model Retraining"]["status"] == "OK"
    assert by_name["State Validation"]["status"] == "OK"
    assert by_name["Scan Diagnostics"]["status"] == "OK"
    assert "Weight Backup" in by_name


def test_snapshot_model_detail_uses_model_file_age(tmp_path: Path):
    root = tmp_path
    model_dir = root / "trained_data" / "models" / "joint"
    model_dir.mkdir(parents=True)
    (model_dir / "ridge_confidence.pkl").write_text("model", encoding="utf-8")

    screen = DiagnosticsScreen(project_root=str(root), live=True)
    age = screen._model_age_for(
        "confidence_ridge",
        freshness={},
        models_dir=root / "trained_data" / "models",
    )

    assert age.endswith("ago") or age == "0m ago"
