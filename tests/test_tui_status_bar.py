from __future__ import annotations

from types import SimpleNamespace

from src.tui.app import _scanner_alive_from_sources, SystemHealthBar
from src.tui.data_provider import DataProvider, DashboardSnapshot


def test_system_health_bar_shows_halted_not_green_latency() -> None:
    bar = SystemHealthBar()
    snap = DashboardSnapshot(scanner_ready=True, halted=True, scan_duration_ms=0)

    bar.update_from_snapshot(snap)

    rendered = bar.render()
    scanner_segment = rendered.plain.split("│", 1)[0]
    assert "HALTED" in scanner_segment
    assert "0ms" not in scanner_segment


def test_heartbeat_treats_embedded_scanner_ready_as_alive() -> None:
    snap = DashboardSnapshot(scanner_ready=True, scan_cycle_count=0)
    scanner = SimpleNamespace(is_ready=True, _running=True)

    assert _scanner_alive_from_sources(snap, scanner) is True


def test_state_file_cycle_count_does_not_imply_scanner_ready(tmp_path) -> None:
    state_dir = tmp_path / ".claude"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        '{"scan_cycle_count": 42, "halted": true, "mode": "live"}',
        encoding="utf-8",
    )

    snap = DataProvider(project_root=str(tmp_path)).refresh()

    assert snap.scan_cycle_count == 42
    assert snap.scanner_ready is False
    assert snap.halted is True
