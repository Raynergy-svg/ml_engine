"""US-008 F9 Jobs screen error-handling tests.

Tests that _open_jobs_screen emits a visible toast (self.notify) and
brain-feed line when it encounters ImportError, OSError, or ValueError,
rather than silently swallowing the exception.

Per CLAUDE.md No-Mock Rule: NO unittest.mock / MagicMock / patch.
The ImportError path is tested by inserting None into sys.modules for
the jobs_screen module key — this is a real sys.modules mutation (the
standard CPython "block import" sentinel, not a mock object).  When
sys.modules[key] is None, CPython's import machinery raises
ModuleNotFoundError (a subclass of ImportError) before any module code
runs.  The import path is the system boundary, so this is the AC-permitted
exception to the no-mock rule.

notify() and _write_brain() captures are done via a real BuddyApp subclass
override (no unittest.mock).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from src.tui.app import BuddyApp

_TUI_THEME_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"
)

_JOBS_SCREEN_MODULE = "src.tui.screens.jobs_screen"
_JOBS_REGISTRY_MODULE = "src.scanner.automation.scheduled_jobs"


class _ObservingApp(BuddyApp):
    """Real BuddyApp subclass that captures notify() and _write_brain() calls.

    Both are real method overrides — no unittest.mock involved.
    CSS_PATH is repointed at the real theme.tcss because Textual resolves
    relative CSS paths against the *defining module's* directory.
    """

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self) -> None:
        super().__init__(live=False)
        self.notified: List[Tuple[str, str]] = []  # (message, severity)
        self.brain_lines: List[str] = []

    def notify(self, message: str, **kwargs: Any) -> Any:  # type: ignore[override]
        severity = str(kwargs.get("severity", "information"))
        self.notified.append((message, severity))
        return super().notify(message, **kwargs)

    def _write_brain(self, markup: str) -> None:  # type: ignore[override]
        self.brain_lines.append(markup)
        try:
            super()._write_brain(markup)
        except Exception:
            pass


def _block_import(module_key: str) -> Any:
    """Set sys.modules[module_key] = None and return the previous value."""
    saved = sys.modules.get(module_key, _SENTINEL)
    sys.modules[module_key] = None  # type: ignore[assignment]
    return saved


def _restore_import(module_key: str, saved: Any) -> None:
    if saved is _SENTINEL:
        sys.modules.pop(module_key, None)
    else:
        sys.modules[module_key] = saved


_SENTINEL = object()


# ---------------------------------------------------------------------------
# AC-1 + AC-2: ImportError path → warning toast + brain line
# ---------------------------------------------------------------------------


def test_import_error_shows_warning_toast() -> None:
    """F9 with a broken jobs_screen import emits a warning toast."""
    app = _ObservingApp()
    saved = _block_import(_JOBS_SCREEN_MODULE)
    try:
        async def _run() -> None:
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.press("f9")
                await pilot.pause(0.3)

        asyncio.run(_run())
    finally:
        _restore_import(_JOBS_SCREEN_MODULE, saved)

    warning_toasts = [(m, s) for m, s in app.notified if s == "warning"]
    assert warning_toasts, (
        f"Expected at least one warning toast; got notified={app.notified}"
    )
    # The toast message must reference the failure
    messages = [m for m, _ in warning_toasts]
    assert any("unavailable" in m.lower() or "import" in m.lower() for m in messages), (
        f"Toast message should name the cause; got messages={messages}"
    )


def test_import_error_writes_brain_line() -> None:
    """F9 with a broken jobs_screen import writes a brain-feed line."""
    app = _ObservingApp()
    saved = _block_import(_JOBS_SCREEN_MODULE)
    try:
        async def _run() -> None:
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.press("f9")
                await pilot.pause(0.3)

        asyncio.run(_run())
    finally:
        _restore_import(_JOBS_SCREEN_MODULE, saved)

    assert app.brain_lines, (
        "Expected at least one brain-feed line after ImportError; got none"
    )
    assert any("F9" in line or "Jobs" in line or "import" in line.lower()
               for line in app.brain_lines), (
        f"Brain line should reference F9/Jobs/import; got={app.brain_lines}"
    )


# ---------------------------------------------------------------------------
# AC-1 + AC-2: OSError path — tested via ScheduledJobsRegistry subclass
# ---------------------------------------------------------------------------


def test_os_error_shows_warning_toast() -> None:
    """F9 when registry.load() raises OSError emits a warning toast.

    Injects a broken ScheduledJobsRegistry into sys.modules by substituting
    a real module-level object that replaces the class with one that raises
    OSError on load().  This avoids blocking the module entirely (which
    would hit the ImportError branch instead of the OSError branch) while
    still using no unittest.mock — the replacement is a real class definition.
    """
    import types

    class _BrokenRegistry:
        def __init__(self) -> None:
            pass

        def load(self) -> None:
            raise OSError("synthetic OS error for test")

    fake_mod = types.ModuleType(_JOBS_REGISTRY_MODULE)
    fake_mod.ScheduledJobsRegistry = _BrokenRegistry  # type: ignore[attr-defined]

    # JobsScreen must be importable so we only hit the OSError branch
    saved_registry = _block_import(_JOBS_REGISTRY_MODULE)
    sys.modules[_JOBS_REGISTRY_MODULE] = fake_mod

    app = _ObservingApp()
    try:
        async def _run() -> None:
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.press("f9")
                await pilot.pause(0.3)

        asyncio.run(_run())
    finally:
        _restore_import(_JOBS_REGISTRY_MODULE, saved_registry)

    warning_toasts = [(m, s) for m, s in app.notified if s == "warning"]
    assert warning_toasts, (
        f"Expected a warning toast for OSError; got notified={app.notified}"
    )


def test_value_error_shows_warning_toast() -> None:
    """F9 when registry.load() raises ValueError emits a warning toast."""
    import types

    class _BrokenRegistry:
        def __init__(self) -> None:
            pass

        def load(self) -> None:
            raise ValueError("synthetic value error for test")

    fake_mod = types.ModuleType(_JOBS_REGISTRY_MODULE)
    fake_mod.ScheduledJobsRegistry = _BrokenRegistry  # type: ignore[attr-defined]

    saved_registry = _block_import(_JOBS_REGISTRY_MODULE)
    sys.modules[_JOBS_REGISTRY_MODULE] = fake_mod

    app = _ObservingApp()
    try:
        async def _run() -> None:
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                await pilot.press("f9")
                await pilot.pause(0.3)

        asyncio.run(_run())
    finally:
        _restore_import(_JOBS_REGISTRY_MODULE, saved_registry)

    warning_toasts = [(m, s) for m, s in app.notified if s == "warning"]
    assert warning_toasts, (
        f"Expected a warning toast for ValueError; got notified={app.notified}"
    )
