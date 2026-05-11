"""Regression tests for DataProvider supervisor-flag honesty.

The TUI's DashboardSnapshot used to read .claude/state.json directly via
json.loads(), bypassing StateEngine's schema migration and atomic-read lock.
A pre-v2 state file (no `scanner_paused`/`mode` keys) silently surfaced as
False/"dry_run" defaults — a lying counter.

These tests use REAL StateEngine + REAL tmp_path disk. No mocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.state_engine import StateEngine
from src.tui.data_provider import DataProvider


def _seed_state(project_root: Path, **overrides) -> None:
    """Use StateEngine itself to seed a v2 state.json under project_root."""
    state_path = project_root / ".claude" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    se = StateEngine(state_path=state_path)
    if "halted" in overrides:
        se.set_halted(bool(overrides["halted"]))
    if "paused" in overrides:
        se.set_paused(bool(overrides["paused"]))
    if "mode" in overrides:
        se.set_mode(str(overrides["mode"]))


def test_supervisor_flags_reflect_ground_truth_paused(tmp_path: Path) -> None:
    """Flipping scanner_paused via real StateEngine surfaces in DashboardSnapshot."""
    _seed_state(tmp_path, paused=True)
    provider = DataProvider(project_root=str(tmp_path))
    snap = provider._refresh_system(provider.snapshot)
    assert snap.scanner_paused is True


def test_supervisor_flags_reflect_ground_truth_halted(tmp_path: Path) -> None:
    """Flipping halted via real StateEngine surfaces in DashboardSnapshot."""
    _seed_state(tmp_path, halted=True)
    provider = DataProvider(project_root=str(tmp_path))
    snap = provider._refresh_system(provider.snapshot)
    assert snap.halted is True


def test_supervisor_flags_reflect_ground_truth_mode_live(tmp_path: Path) -> None:
    """Flipping mode to 'live' via real StateEngine surfaces in DashboardSnapshot."""
    _seed_state(tmp_path, mode="live")
    provider = DataProvider(project_root=str(tmp_path))
    snap = provider._refresh_system(provider.snapshot)
    assert snap.mode == "live"


def test_supervisor_flags_default_when_no_state_file(tmp_path: Path) -> None:
    """No state.json on disk → defaults from DataProvider dataclass + StateEngine."""
    provider = DataProvider(project_root=str(tmp_path))
    snap = provider._refresh_system(provider.snapshot)
    # StateEngine returns dict(_DEFAULT_STATE) when file missing — all-defaults
    assert snap.scanner_paused is False
    assert snap.halted is False
    assert snap.mode == "dry_run"


def test_supervisor_flags_schema_v1_file_is_migrated_not_lying(tmp_path: Path) -> None:
    """A pre-v2 state file (no scanner_paused/mode keys) must be migrated,
    not silently surface as default values from a dict.get() miss.
    """
    state_path = tmp_path / ".claude" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a v1-shape file deliberately missing v2 keys
    state_path.write_text(json.dumps({
        "goal": "legacy",
        "status": "ready",
        "schema_version": "1",
    }))
    provider = DataProvider(project_root=str(tmp_path))
    snap = provider._refresh_system(provider.snapshot)
    # After migration, keys exist and reflect defaults honestly
    assert snap.scanner_paused is False
    assert snap.halted is False
    assert snap.mode == "dry_run"
    # Verify migration actually wrote v2 keys to disk
    migrated = json.loads(state_path.read_text())
    assert migrated.get("schema_version") == "2"
    assert "scanner_paused" in migrated
    assert "halted" in migrated
    assert "mode" in migrated


def test_supervisor_flags_roundtrip_paused_then_unpaused(tmp_path: Path) -> None:
    """Toggling scanner_paused True→False via StateEngine is reflected in snapshot."""
    _seed_state(tmp_path, paused=True)
    provider = DataProvider(project_root=str(tmp_path))
    snap1 = provider._refresh_system(provider.snapshot)
    assert snap1.scanner_paused is True

    # Operator un-pauses via real StateEngine (same code path as TUI Space key)
    StateEngine(state_path=tmp_path / ".claude" / "state.json").set_paused(False)
    snap2 = provider._refresh_system(provider.snapshot)
    assert snap2.scanner_paused is False


# ── Risk-cache fallback (refresh() previous-snapshot preservation) ───────
# When ScanEnrichment fields portfolio_risk_pct / drawdown_pct / max_drawdown_pct
# carry the -1.0 sentinel ("scanner couldn't compute"), refresh() must preserve
# the last valid value from self._snapshot instead of overwriting with 0.0/NaN.


def test_refresh_preserves_prior_risk_when_enrichment_sentinel(tmp_path: Path) -> None:
    """Invalid (-1.0 sentinel) enrichment risk fields keep prior snapshot values."""
    from src.tui.embedded_scanner import ScanEnrichment

    provider = DataProvider(project_root=str(tmp_path))
    # Seed last-good values into the live snapshot the way a prior refresh would
    with provider._lock:
        provider._snapshot.portfolio_risk_pct = 5.20
        provider._snapshot.drawdown_pct = 1.30
        provider._snapshot.max_drawdown_pct = 3.70

    # Scanner couldn't compute risk this cycle — sentinel -1.0 for all three
    provider.apply_scan_enrichment(ScanEnrichment(
        portfolio_risk_pct=-1.0,
        drawdown_pct=-1.0,
        max_drawdown_pct=-1.0,
    ))
    snap = provider.refresh()

    assert snap.portfolio_risk_pct == 5.20
    assert snap.drawdown_pct == 1.30
    assert snap.max_drawdown_pct == 3.70


def test_refresh_overwrites_prior_risk_when_enrichment_valid(tmp_path: Path) -> None:
    """Valid enrichment risk fields replace prior snapshot values (no stickiness)."""
    from src.tui.embedded_scanner import ScanEnrichment

    provider = DataProvider(project_root=str(tmp_path))
    with provider._lock:
        provider._snapshot.portfolio_risk_pct = 5.20
        provider._snapshot.drawdown_pct = 1.30
        provider._snapshot.max_drawdown_pct = 3.70

    provider.apply_scan_enrichment(ScanEnrichment(
        portfolio_risk_pct=7.10,
        drawdown_pct=2.40,
        max_drawdown_pct=4.50,
    ))
    snap = provider.refresh()

    assert snap.portfolio_risk_pct == 7.10
    assert snap.drawdown_pct == 2.40
    assert snap.max_drawdown_pct == 4.50


def test_refresh_first_call_with_sentinel_uses_dataclass_defaults(tmp_path: Path) -> None:
    """First refresh() with sentinel enrichment falls back to DashboardSnapshot defaults (0.0)."""
    from src.tui.embedded_scanner import ScanEnrichment

    provider = DataProvider(project_root=str(tmp_path))
    # No prior values seeded — _snapshot is fresh DashboardSnapshot() from __init__

    provider.apply_scan_enrichment(ScanEnrichment(
        portfolio_risk_pct=-1.0,
        drawdown_pct=-1.0,
        max_drawdown_pct=-1.0,
    ))
    snap = provider.refresh()

    # Honest: no data yet, so 0.0 default (NOT misleading because there IS no
    # prior valid value to lie about preserving). This is the correct semantics
    # on cold boot.
    assert snap.portfolio_risk_pct == 0.0
    assert snap.drawdown_pct == 0.0
    assert snap.max_drawdown_pct == 0.0
