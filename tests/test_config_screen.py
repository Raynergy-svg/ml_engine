import json

from src.tui.screens.config_screen import (
    ConfigScreen,
    _AGENT_TOGGLES,
    _GATE_FIELDS,
    _load_adjustment_rows,
)


def test_config_catalog_exposes_tier7_controls():
    numeric_fields = {field for field, _label, _default in _GATE_FIELDS}
    toggle_fields = {field for field, _label in _AGENT_TOGGLES}

    assert {
        "sub_inference_min_confidence",
        "disagreement_hard_floor",
        "staleness_uncertainty_threshold",
        "daily_trade_limit",
        "max_spread_pips",
    } <= numeric_fields
    assert {
        "use_rl_sizer",
        "use_rl_gates",
        "enable_model_routing",
        "enable_smart_execution",
        "enable_execution_routing",
        "enable_health_registry",
    } <= toggle_fields


def test_save_staged_changes_queues_pending_proposals(tmp_path):
    screen = ConfigScreen(project_root=str(tmp_path))
    screen._active_profile = "smart"
    screen._current_config = {
        "min_confidence": 42.0,
        "enable_execution": True,
    }
    screen._staged_changes = {
        "min_confidence": (42.0, 45.0),
        "enable_execution": (True, False),
    }

    screen._save_staged_changes()

    pending_path = tmp_path / ".claude" / "pending_adjustments.json"
    approved_path = tmp_path / ".claude" / "config_adjustments.json"
    data = json.loads(pending_path.read_text())
    proposals = data["proposals"]

    assert approved_path.exists() is False
    assert {proposal["key"] for proposal in proposals} == {"min_confidence", "enable_execution"}
    assert {proposal["source"] for proposal in proposals} == {"tui_config"}
    assert {proposal["status"] for proposal in proposals} == {"pending"}
    assert screen._staged_changes == {}


def test_adjustment_rows_merge_pending_and_approved_history(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "pending_adjustments.json").write_text(json.dumps({
        "proposals": [{
            "timestamp": "2026-04-24T02:00:00+00:00",
            "key": "min_confidence",
            "current_value": 42.0,
            "proposed_value": 45.0,
            "source": "tui_config",
            "status": "pending",
        }]
    }))
    (claude_dir / "config_adjustments.json").write_text(json.dumps({
        "history": [{
            "timestamp": "2026-04-24T01:00:00+00:00",
            "key": "risk_per_trade_pct",
            "old_value": 0.05,
            "new_value": 0.04,
            "source": "inbox",
        }]
    }))

    rows = _load_adjustment_rows(tmp_path)

    assert [row["field"] for row in rows] == ["min_confidence", "risk_per_trade_pct"]
    assert [row["status"] for row in rows] == ["pending", "approved"]
    assert rows[0]["new"] == 45.0
    assert rows[1]["old"] == 0.05
