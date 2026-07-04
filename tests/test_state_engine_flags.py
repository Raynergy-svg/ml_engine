"""Tests for StateEngine supervisor flags — US-501 schema v2."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.scanner.automation.state_engine import StateEngine, SCHEMA_VERSION


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


# ── get / set halted ─────────────────────────────────────────────────

def test_get_halted_default(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    assert engine.get_halted() is False


def test_set_get_halted_true(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_halted(True)
    assert engine.get_halted() is True


def test_set_get_halted_false(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_halted(True)
    engine.set_halted(False)
    assert engine.get_halted() is False


# ── get_halted_strict (fail-closed, 2026-07-02 operator decision) ────

def test_get_halted_strict_requires_a_lane(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    with pytest.raises(ValueError):
        engine.get_halted_strict()


def test_get_halted_strict_rejects_unknown_lane(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    with pytest.raises(ValueError):
        engine.get_halted_strict(lane="not_a_real_lane")


def test_get_halted_strict_missing_file_fails_closed(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert not tmp_state.exists()
    assert engine.get_halted_strict() is True


def test_get_halted_strict_corrupt_json_fails_closed(tmp_state: Path) -> None:
    tmp_state.write_text("{not valid json")
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_non_dict_payload_fails_closed(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps([1, 2, 3]))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_missing_halted_lanes_fails_closed(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": False}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_missing_lane_entry_fails_closed(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": False, "halted_lanes": {"equity": False}}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_non_bool_lane_entry_fails_closed(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": False, "halted_lanes": {"oanda_fx": "nope"}}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_explicit_false_is_not_halted(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": False, "halted_lanes": {"oanda_fx": False}}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is False


def test_get_halted_strict_explicit_true_is_halted(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": False, "halted_lanes": {"oanda_fx": True}}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_global_halt_overrides_lane_false(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({"halted": True, "halted_lanes": {"oanda_fx": False}}))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict() is True


def test_get_halted_strict_explicit_lane_arg_overrides_bound_lane(tmp_state: Path) -> None:
    tmp_state.write_text(json.dumps({
        "halted": False,
        "halted_lanes": {"oanda_fx": True, "equity": False},
    }))
    engine = StateEngine(state_path=tmp_state, lane="oanda_fx")
    assert engine.get_halted_strict(lane="equity") is False


# ── get / set paused ─────────────────────────────────────────────────

def test_get_paused_default(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    assert engine.get_paused() is False


def test_set_get_paused_true(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_paused(True)
    assert engine.get_paused() is True


def test_set_get_paused_roundtrip(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_paused(True)
    engine.set_paused(False)
    assert engine.get_paused() is False


# ── get / set mode ───────────────────────────────────────────────────

def test_get_mode_default(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    assert engine.get_mode() == "dry_run"


def test_set_mode_live(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_mode("live")
    assert engine.get_mode() == "live"


def test_set_mode_dry_run(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    engine.set_mode("live")
    engine.set_mode("dry_run")
    assert engine.get_mode() == "dry_run"


def test_set_mode_invalid_raises(tmp_state: Path) -> None:
    engine = StateEngine(state_path=tmp_state)
    with pytest.raises(ValueError, match="Invalid mode"):
        engine.set_mode("paper")


# ── persistence across instantiation ─────────────────────────────────

def test_persistence_across_instantiation(tmp_state: Path) -> None:
    engine1 = StateEngine(state_path=tmp_state)
    engine1.set_halted(True)
    engine1.set_paused(True)
    engine1.set_mode("live")

    engine2 = StateEngine(state_path=tmp_state)
    assert engine2.get_halted() is True
    assert engine2.get_paused() is True
    assert engine2.get_mode() == "live"


def test_flags_survive_other_updates(tmp_state: Path) -> None:
    """Setting flags multiple times does not lose unrelated state."""
    engine = StateEngine(state_path=tmp_state)
    engine.set_halted(True)
    engine.set_mode("live")
    engine.set_paused(True)

    engine2 = StateEngine(state_path=tmp_state)
    assert engine2.get_halted() is True
    assert engine2.get_mode() == "live"
    assert engine2.get_paused() is True


# ── schema migration ─────────────────────────────────────────────────

def test_schema_migration_from_v1(tmp_state: Path) -> None:
    """v1 state (no schema_version) is migrated to v2 with defaults."""
    v1: dict = {
        "goal": "continuous_scan",
        "status": "ready",
        "done": ["US-002"],
        "next": "boot TUI",
        "open_questions": [],
        "last_updated": "2026-01-01T00:00:00+00:00",
    }
    tmp_state.write_text(json.dumps(v1))

    engine = StateEngine(state_path=tmp_state)
    assert engine.get_halted() is False
    assert engine.get_paused() is False
    assert engine.get_mode() == "dry_run"

    # Original v1 data preserved
    state = engine.load_state()
    assert state["goal"] == "continuous_scan"
    assert state["done"] == ["US-002"]
    assert state["schema_version"] == SCHEMA_VERSION


def test_schema_migration_writes_version(tmp_state: Path) -> None:
    """Migration persists schema_version=2 on next write."""
    v1: dict = {"goal": "test", "status": "ready", "done": [], "next": "", "open_questions": []}
    tmp_state.write_text(json.dumps(v1))

    engine = StateEngine(state_path=tmp_state)
    engine.set_halted(True)  # triggers a write

    saved = json.loads(tmp_state.read_text())
    assert saved["schema_version"] == SCHEMA_VERSION


def test_v2_state_not_re_migrated(tmp_state: Path) -> None:
    """If schema_version==2 is already present, no data loss on repeated loads."""
    engine1 = StateEngine(state_path=tmp_state)
    engine1.set_mode("live")

    # Load twice — should not overwrite mode back to dry_run
    engine2 = StateEngine(state_path=tmp_state)
    assert engine2.get_mode() == "live"


# ── concurrent writes ─────────────────────────────────────────────────

def test_concurrent_writes_dont_corrupt(tmp_state: Path) -> None:
    """10 threads × 100 flag toggles — file stays valid JSON."""
    engine = StateEngine(state_path=tmp_state)
    engine.set_halted(False)  # ensure file exists before threads start

    errors: list[str] = []

    def toggle_halted() -> None:
        for _ in range(50):
            try:
                engine.set_halted(True)
                engine.set_halted(False)
            except Exception as exc:
                errors.append(str(exc))

    def toggle_mode() -> None:
        for _ in range(50):
            try:
                engine.set_mode("live")
                engine.set_mode("dry_run")
            except Exception as exc:
                errors.append(str(exc))

    threads = [threading.Thread(target=toggle_halted) for _ in range(5)]
    threads += [threading.Thread(target=toggle_mode) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent writes: {errors}"

    data = json.loads(tmp_state.read_text())
    assert "halted" in data
    assert "mode" in data
    assert data["schema_version"] == SCHEMA_VERSION
