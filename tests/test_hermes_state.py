"""HermesState persistence — load/save round-trip + defaults."""
from __future__ import annotations

from pathlib import Path

from src.scanner.automation.hermes_state import HermesState


def test_default_state_fields(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    s = HermesState.from_path(p)
    assert s.last_silent_at_iso is None
    assert s.last_alert_keys == {}
    assert s.last_job_failure_keys == {}
    assert s.last_brief_at_iso is None
    assert s.cycle_count_at_last_brief is None


def test_save_then_load_round_trip(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    s = HermesState(
        last_silent_at_iso="2026-05-12T07:00:00+00:00",
        last_alert_keys={"consecutive_losses": "2026-05-12T07:30:00+00:00"},
        last_job_failure_keys={"nightly_audit": "2026-05-12T22:00:00+00:00"},
        last_brief_at_iso="2026-05-12T07:00:00+00:00",
        cycle_count_at_last_brief=12345,
    )
    s.save_to(p)
    s2 = HermesState.from_path(p)
    assert s2 == s


def test_missing_file_returns_defaults(tmp_path: Path):
    p = tmp_path / "does_not_exist.json"
    s = HermesState.from_path(p)
    assert s == HermesState()


def test_corrupt_file_returns_defaults(tmp_path: Path):
    p = tmp_path / "hermes_state.json"
    p.write_text("not valid json {{{")
    s = HermesState.from_path(p)
    assert s == HermesState()


def test_partial_file_loads_missing_fields_as_defaults(tmp_path: Path):
    """Forward-compatibility: old state files load with new fields defaulted."""
    p = tmp_path / "hermes_state.json"
    p.write_text('{"last_brief_at_iso": "2026-05-10T07:00:00+00:00"}')
    s = HermesState.from_path(p)
    assert s.last_brief_at_iso == "2026-05-10T07:00:00+00:00"
    assert s.last_alert_keys == {}
    assert s.cycle_count_at_last_brief is None
