"""Mythos audit 2026-04-30 — unified debug log.

Operator's verification gap: "we need to make a debug log where
everything that buddy is doing, background and foreground, is there
neatly for you to read.. to bridge the gap between what i see and
what you read."

Pre-fix: Python's `logging.getLogger().info(...)` calls fall through
to stderr by default. In the TUI runtime, stderr is captured by
Textual's redirect into logs/buddy_tui.stderr.log — but that file
is full of Textual escape sequences, making grep useless. The
operator could see logs scroll past on the TUI brain feed but they
existed nowhere on disk in plain text.

Fix: ensure_runtime_env() now attaches a RotatingFileHandler to the
root logger that writes EVERY log line (DEBUG and above) to
logs/buddy_debug.log with a structured format. Plain text,
grep-friendly, rotated at 50MB / 3 backups (200MB ceiling).

Tests pin:
  * Handler attached on first call
  * Idempotent — subsequent calls don't stack handlers (matters for
    os.execv-based safe-restart)
  * Format follows the documented schema
  * Failures don't break the bootstrap
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


def _clean_handlers():
    """Strip any prior buddy unified handlers so each test starts clean."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_buddy_unified_marker", False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


@pytest.fixture
def isolated_root(tmp_path):
    _clean_handlers()
    yield tmp_path
    _clean_handlers()


def test_first_call_attaches_handler(isolated_root):
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)

    root = logging.getLogger()
    markers = [h for h in root.handlers if getattr(h, "_buddy_unified_marker", False)]
    assert len(markers) == 1


def test_log_lines_written_to_disk(isolated_root):
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)

    logger = logging.getLogger("test.unified")
    logger.info("verification_test_marker")
    # Force flush
    for h in logging.getLogger().handlers:
        if getattr(h, "_buddy_unified_marker", False):
            h.flush()

    log_file = isolated_root / "logs" / "buddy_debug.log"
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "verification_test_marker" in contents
    assert "test.unified" in contents
    assert "INFO" in contents


def test_log_format_includes_timestamp_level_module(isolated_root):
    """Verifies the format `%(asctime)s [%(levelname)5s] %(name)s: %(message)s`
    so operator + audit can grep by any field."""
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)

    logger = logging.getLogger("src.scanner.engine")
    logger.warning("test_warning_for_format_check")
    for h in logging.getLogger().handlers:
        if getattr(h, "_buddy_unified_marker", False):
            h.flush()

    log_file = isolated_root / "logs" / "buddy_debug.log"
    contents = log_file.read_text(encoding="utf-8")
    # Find our test line
    target = next(
        (line for line in contents.splitlines() if "test_warning_for_format_check" in line),
        None,
    )
    assert target is not None
    # Format check: `<iso-ish timestamp> [<5char level>] <name>: <msg>`
    assert "[ WARN" in target or "[WARNING" in target  # 5-char field
    assert "src.scanner.engine" in target
    assert ":" in target
    # Date prefix YYYY-MM-DD
    assert target.startswith("20")


def test_idempotent_does_not_stack_handlers(isolated_root):
    """os.execv-based safe-restart re-runs ensure_runtime_env. We
    must not accumulate handlers — each restart should re-use the
    existing one."""
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)
    _configure_unified_debug_log(isolated_root)
    _configure_unified_debug_log(isolated_root)

    root = logging.getLogger()
    markers = [h for h in root.handlers if getattr(h, "_buddy_unified_marker", False)]
    assert len(markers) == 1, (
        f"Expected 1 unified handler after 3 calls, got {len(markers)}"
    )


def test_capture_includes_warning_and_error_and_debug(isolated_root):
    """Root logger.setLevel(DEBUG) ensures all severities reach the
    handler. Lower-severity messages must not be filtered."""
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)

    logger = logging.getLogger("test.severity")
    logger.debug("dbg_marker")
    logger.info("info_marker")
    logger.warning("warn_marker")
    logger.error("err_marker")
    for h in logging.getLogger().handlers:
        if getattr(h, "_buddy_unified_marker", False):
            h.flush()

    log_file = isolated_root / "logs" / "buddy_debug.log"
    contents = log_file.read_text(encoding="utf-8")
    for marker in ("dbg_marker", "info_marker", "warn_marker", "err_marker"):
        assert marker in contents, f"Severity test failed: {marker} missing"


def test_mkdir_failure_does_not_raise(monkeypatch):
    """If logs/ dir is read-only, bootstrap should swallow the error
    rather than break TUI startup."""
    from src.bootstrap.env import _configure_unified_debug_log
    _clean_handlers()

    bad_root = Path("/dev/null")  # mkdir under /dev/null fails
    # Must not raise.
    _configure_unified_debug_log(bad_root)


def test_log_path_under_project_root_logs_dir(isolated_root):
    """Operator's `tail -f logs/buddy_debug.log` must find the file
    under the project's logs/ directory, not somewhere unexpected."""
    from src.bootstrap.env import _configure_unified_debug_log
    _configure_unified_debug_log(isolated_root)
    logger = logging.getLogger("test.path_check")
    logger.info("path_check_marker")
    for h in logging.getLogger().handlers:
        if getattr(h, "_buddy_unified_marker", False):
            h.flush()

    expected = isolated_root / "logs" / "buddy_debug.log"
    assert expected.exists()
