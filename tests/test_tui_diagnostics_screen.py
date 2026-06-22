"""US-009 + P0-1 (2026-06-21 equity-wiring audit): DiagnosticsScreen tests.

Covers:
  AC-1  Demo mode renders an HONEST empty state from real log sources — NOT
        hardcoded fake FX seed lines. Pre-2026-06-21 the screen seeded fake
        FX entries ("OANDA connection established", "EUR/GBP spread elevated")
        in demo mode; that fiction was removed when the equity harvester
        became the live stack. With an empty tmp_path repo root there are no
        real log sources, so the screen shows the "No live diagnostic log
        entries found yet" honest empty state and NO fake FX strings.
  AC-2  psutil ImportError path mounts a Static(.error-banner) inside the
        SystemVitalsPanel — no random.uniform fake data.

No mocks per CLAUDE.md No-Mock Rule.
sys.modules['psutil'] = None is the permitted import-boundary monkeypatch:
CPython raises ModuleNotFoundError (subclass of ImportError) whenever
'import psutil' is encountered while sys.modules has None for that key.
The same technique is used in test_tui_jobs_screen_error.py (US-008).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static

from src.tui.screens.diagnostics_screen import DiagnosticsScreen, SystemVitalsPanel

_TUI_THEME_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"
)

_SENTINEL = object()


def _block_import(key: str) -> object:
    saved = sys.modules.get(key, _SENTINEL)
    sys.modules[key] = None  # type: ignore[assignment]
    return saved


def _restore_import(key: str, saved: object) -> None:
    if saved is _SENTINEL:
        sys.modules.pop(key, None)
    else:
        sys.modules[key] = saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Minimal host app
# ---------------------------------------------------------------------------


class _DiagApp(App):
    """Wraps DiagnosticsScreen in the smallest possible App for testing."""

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        self._root = project_root
        self.diag: DiagnosticsScreen | None = None

    def compose(self) -> ComposeResult:
        self.diag = DiagnosticsScreen(
            project_root=str(self._root) if self._root else None,
            live=False,
        )
        yield self.diag


# ---------------------------------------------------------------------------
# AC-1 helpers — log messages after demo-mode mount
# ---------------------------------------------------------------------------


def _collect_log_msgs(tmp_path: Path) -> list[str]:
    """Run the app briefly in demo mode and return rendered log message strings.

    project_root is an empty tmp_path, so there are no real log sources on
    disk — the screen must show its honest empty state, NOT fake FX seeds.
    """
    app = _DiagApp(project_root=tmp_path)

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)

    asyncio.run(_run())
    assert app.diag is not None
    return [e["msg"] for e in app.diag._error_log_entries]


# ---------------------------------------------------------------------------
# AC-1 (P0-1): demo mode shows honest empty state, NOT fake FX seed lines
# ---------------------------------------------------------------------------


def test_demo_mode_shows_honest_empty_state(tmp_path: Path) -> None:
    """With no real log sources, the screen shows the empty-state message."""
    msgs = _collect_log_msgs(tmp_path)
    assert any("No live diagnostic log entries found yet" in m for m in msgs), (
        f"Expected honest empty-state message; got: {msgs}"
    )


def test_demo_mode_no_fake_oanda_line(tmp_path: Path) -> None:
    """The fake 'OANDA connection established' seed line is gone."""
    msgs = _collect_log_msgs(tmp_path)
    fake = [m for m in msgs if "OANDA connection established" in m]
    assert not fake, f"Fake OANDA seed line still present: {fake}"


def test_demo_mode_no_fake_fx_spread_line(tmp_path: Path) -> None:
    """The fake 'EUR/GBP spread elevated' seed line is gone."""
    msgs = _collect_log_msgs(tmp_path)
    fake = [m for m in msgs if "spread elevated" in m]
    assert not fake, f"Fake FX spread seed line still present: {fake}"


def test_demo_mode_no_fx_pair_seed_lines(tmp_path: Path) -> None:
    """No hardcoded FX-pair fiction (EUR/GBP, GBP/JPY) in the rendered log."""
    msgs = _collect_log_msgs(tmp_path)
    fake = [m for m in msgs if "EUR/GBP" in m or "GBP/JPY" in m]
    assert not fake, f"Hardcoded FX-pair seed lines still present: {fake}"


# ---------------------------------------------------------------------------
# AC-2: psutil missing → error-banner Static, no random data
# ---------------------------------------------------------------------------


def test_psutil_missing_mounts_error_banner(tmp_path: Path) -> None:
    """SystemVitalsPanel shows a .error-banner Static when psutil is absent."""
    banner_count = 0
    banner_text = ""

    app = _DiagApp(project_root=tmp_path)
    saved = _block_import("psutil")
    try:
        async def _run() -> None:
            nonlocal banner_count, banner_text
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.pause(0.3)
                assert app.diag is not None
                panel = app.diag.query_one("#diag-vitals", SystemVitalsPanel)
                banners = panel.query(".error-banner")
                banner_count = len(banners)
                if banner_count:
                    banner_text = str(banners.first(Static).render())

        asyncio.run(_run())
    finally:
        _restore_import("psutil", saved)

    assert banner_count == 1, (
        f"Expected exactly 1 .error-banner inside SystemVitalsPanel; got {banner_count}"
    )
    assert "psutil missing" in banner_text, (
        f"Banner text should contain 'psutil missing'; got: {banner_text!r}"
    )
    assert "pip install psutil" in banner_text, (
        f"Banner text should contain install hint; got: {banner_text!r}"
    )


def test_psutil_missing_sets_error_flag(tmp_path: Path) -> None:
    """_psutil_error_shown flag is True after ImportError fires."""
    flag_value: list[bool] = []

    app = _DiagApp(project_root=tmp_path)
    saved = _block_import("psutil")
    try:
        async def _run() -> None:
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.pause(0.3)
                assert app.diag is not None
                panel = app.diag.query_one("#diag-vitals", SystemVitalsPanel)
                flag_value.append(panel._psutil_error_shown)

        asyncio.run(_run())
    finally:
        _restore_import("psutil", saved)

    assert flag_value == [True], (
        f"Expected _psutil_error_shown=True after ImportError; got {flag_value}"
    )


def test_psutil_missing_show_psutil_error_idempotent(tmp_path: Path) -> None:
    """show_psutil_error() called twice mounts only one banner."""
    banner_count = 0

    app = _DiagApp(project_root=tmp_path)
    saved = _block_import("psutil")
    try:
        async def _run() -> None:
            nonlocal banner_count
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.pause(0.3)
                assert app.diag is not None
                panel = app.diag.query_one("#diag-vitals", SystemVitalsPanel)
                # Manually call again to test idempotency guard
                panel.show_psutil_error()
                await pilot.pause(0.1)
                banner_count = len(panel.query(".error-banner"))

        asyncio.run(_run())
    finally:
        _restore_import("psutil", saved)

    assert banner_count == 1, (
        f"Double call to show_psutil_error() should still yield 1 banner; got {banner_count}"
    )
