"""US-009: DiagnosticsScreen correctness tests.

Covers:
  AC-1  _seed_error_log contains current ML stack strings (Transformer /
        LightGBM / XGBoost) and does NOT contain the stale 'TCN + Ridge + RF'.
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

import pytest
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
# AC-1 helpers — seed strings after demo-mode mount
# ---------------------------------------------------------------------------


def _collect_seed_msgs(tmp_path: Path) -> list[str]:
    """Run the app briefly in demo mode and return seed log message strings."""
    app = _DiagApp(project_root=tmp_path)

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)

    asyncio.run(_run())
    assert app.diag is not None
    return [e["msg"] for e in app.diag._error_log_entries]


# ---------------------------------------------------------------------------
# AC-1: seed strings name current ML stack
# ---------------------------------------------------------------------------


def test_seed_log_contains_transformer(tmp_path: Path) -> None:
    """Seed entry names Transformer (direction head)."""
    msgs = _collect_seed_msgs(tmp_path)
    assert any("Transformer" in m for m in msgs), (
        f"Expected 'Transformer' in seed log; got: {msgs}"
    )


def test_seed_log_contains_lightgbm(tmp_path: Path) -> None:
    """Seed entry names LightGBM (confidence/momentum/risk heads)."""
    msgs = _collect_seed_msgs(tmp_path)
    assert any("LightGBM" in m for m in msgs), (
        f"Expected 'LightGBM' in seed log; got: {msgs}"
    )


def test_seed_log_contains_xgboost(tmp_path: Path) -> None:
    """Seed entry names XGBoost (meta-labeler)."""
    msgs = _collect_seed_msgs(tmp_path)
    assert any("XGBoost" in m for m in msgs), (
        f"Expected 'XGBoost' in seed log; got: {msgs}"
    )


def test_seed_log_no_stale_tcn_ridge_rf(tmp_path: Path) -> None:
    """The stale 'TCN + Ridge + RF' string is absent from all seed entries."""
    msgs = _collect_seed_msgs(tmp_path)
    stale = [m for m in msgs if "TCN" in m and "Ridge" in m and "RF" in m]
    assert not stale, (
        f"Stale 'TCN + Ridge + RF' string found in seed entries: {stale}"
    )


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
