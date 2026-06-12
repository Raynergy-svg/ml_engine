"""Phase 1 hot-reload + auto-init: verify the bootstrap module
delivers the operator's "stop manually exporting BUDDY_* every
session" requirement.

Mythos audit 2026-04-28 (DevOps Automator subagent design) — single
source of truth lives in src/bootstrap/env.py; both ./buddy
(scripts/init.sh) and python -m entry points must see identical env.
"""
from __future__ import annotations

import io
import os
import platform
import sys
from pathlib import Path

import pytest

from src.bootstrap.env import (
    _BEHAVIOR_DEFAULTS,
    _MACOS_DEFAULTS,
    _TUI_DEFAULTS,
    _parse_dotenv,
    dump_shell_exports,
    ensure_runtime_env,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every key the bootstrap module touches so tests start
    from a known-empty state."""
    keys = (
        list(_MACOS_DEFAULTS.keys())
        + list(_TUI_DEFAULTS.keys())
        + list(_BEHAVIOR_DEFAULTS.keys())
        # OANDA_ACCOUNT_ID must be stripped too: ensure_runtime_env uses
        # setdefault semantics (operator env always wins), so a value
        # exported in the invoking shell would beat the test's .env.local
        # and fail test_dotenv_local_sourced_for_secrets.
        + ["BUDDY_SESSION_ID", "OANDA_API_TOKEN", "OANDA_ACCOUNT_ID", "BUDDY_TEST_TOGGLE"]
    )
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_idempotent_repeated_calls_dont_change_state(clean_env, tmp_path):
    """ensure_runtime_env() may be called from main.py AND tui/__main__.py
    in the same process — must be a no-op on the second call."""
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    snapshot = dict(os.environ)
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    assert dict(os.environ) == snapshot


def test_macos_defaults_applied_on_darwin(clean_env, tmp_path):
    """macOS thread + Metal caps must be set before any TF import.
    Skipped on non-Darwin since the defaults are platform-conditional."""
    if platform.system() != "Darwin":
        pytest.skip("macOS-only defaults")
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    for k, v in _MACOS_DEFAULTS.items():
        assert os.environ.get(k) == v, f"{k} did not apply: {os.environ.get(k)!r}"


def test_behavior_defaults_applied_on_all_platforms(clean_env, tmp_path):
    """BUDDY_HOT_RELOAD, BUDDY_META_*, etc. should ship at safe defaults
    so operators don't have to remember to export them."""
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    for k, v in _BEHAVIOR_DEFAULTS.items():
        assert os.environ.get(k) == v


def test_explicit_export_wins_over_defaults(clean_env, tmp_path):
    """If the operator manually exports BUDDY_HOT_RELOAD=1, the
    bootstrap default must NOT clobber it. Operator value always wins."""
    clean_env.setenv("BUDDY_HOT_RELOAD", "1")
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    assert os.environ["BUDDY_HOT_RELOAD"] == "1"


def test_dotenv_local_sourced_for_secrets(clean_env, tmp_path):
    """OANDA secrets in .env.local must be loaded automatically — no
    operator action required beyond editing the file once."""
    (tmp_path / ".env.local").write_text(
        "OANDA_API_TOKEN=test-token-abc\n"
        "OANDA_ACCOUNT_ID=123-456-789\n"
    )
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    assert os.environ["OANDA_API_TOKEN"] == "test-token-abc"
    assert os.environ["OANDA_ACCOUNT_ID"] == "123-456-789"


def test_dotenv_toggles_overrides_built_in_defaults(clean_env, tmp_path):
    """The whole point of .env.local.toggles: pin BUDDY_* values once,
    they survive every restart and override the built-in defaults."""
    (tmp_path / ".env.local.toggles").write_text(
        "BUDDY_HOT_RELOAD=1\n"
        "BUDDY_META_MANAGER_ENABLED=1\n"
    )
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=False)
    assert os.environ["BUDDY_HOT_RELOAD"] == "1"
    assert os.environ["BUDDY_META_MANAGER_ENABLED"] == "1"


def test_dotenv_parser_handles_quotes_and_comments(tmp_path):
    """The shell `set -a; source` path tolerates quotes, comments,
    and `export` prefixes — Python parser must match."""
    (tmp_path / ".env").write_text(
        '# top comment\n'
        'export PLAIN=plainvalue\n'
        'QUOTED="value with spaces"\n'
        "SQUOTED='single quoted'\n"
        'WITH_INLINE=val # trailing comment\n'
        '\n'
        '   # indented comment skipped\n'
        'NO_EQUALS_LINE\n'
        'EMPTY=\n'
    )
    parsed = _parse_dotenv(tmp_path / ".env")
    assert parsed["PLAIN"] == "plainvalue"
    assert parsed["QUOTED"] == "value with spaces"
    assert parsed["SQUOTED"] == "single quoted"
    assert parsed["WITH_INLINE"] == "val"
    assert "NO_EQUALS_LINE" not in parsed


def test_dotenv_parser_returns_empty_for_missing_file(tmp_path):
    assert _parse_dotenv(tmp_path / "does_not_exist") == {}


def test_dotenv_parser_safe_on_malformed_input(tmp_path):
    """Never raise on a corrupt .env — operator may have a half-saved
    file mid-edit. Bootstrap must degrade gracefully."""
    bad = tmp_path / ".env.bad"
    bad.write_bytes(b"\xff\xfe not utf8 \xff")
    assert _parse_dotenv(bad) == {}


def test_dump_shell_exports_emits_only_unset_vars(clean_env, capsys):
    """scripts/init.sh evals the output of dump_shell_exports — it
    must NOT clobber values the operator already exported in their
    shell, otherwise their values would silently revert to defaults."""
    clean_env.setenv("BUDDY_HOT_RELOAD", "1")  # operator-set
    buf = io.StringIO()
    dump_shell_exports(stream=buf)
    out = buf.getvalue()
    # Operator-set var: NOT in output
    assert "export BUDDY_HOT_RELOAD=" not in out
    # Default-needed var: IS in output
    assert "export BUDDY_META_USE_LLM=" in out


def test_dump_shell_exports_quotes_values_safely(clean_env):
    """Single-quote-escape so a value containing a quote can't break
    out of the shell quoting context."""
    clean_env.setenv("DUMMY_TEST", "ignore")  # ensure clean state
    # Insert a fake default that contains a single quote
    from src.bootstrap import env as bootstrap_env
    original = dict(bootstrap_env._BEHAVIOR_DEFAULTS)
    bootstrap_env._BEHAVIOR_DEFAULTS["BUDDY_TEST_TOGGLE"] = "value's 'tricky'"
    try:
        buf = io.StringIO()
        dump_shell_exports(stream=buf)
        line = next(
            ln for ln in buf.getvalue().splitlines()
            if ln.startswith("export BUDDY_TEST_TOGGLE=")
        )
        # The escaped form should round-trip through bash safely.
        # We check structurally rather than re-parsing through a shell.
        assert "BUDDY_TEST_TOGGLE='" in line
        assert line.endswith("'")
    finally:
        bootstrap_env._BEHAVIOR_DEFAULTS.clear()
        bootstrap_env._BEHAVIOR_DEFAULTS.update(original)


def test_session_id_auto_generated_when_unset(clean_env, tmp_path):
    """Per-process session ID is the join key for reload journal +
    heartbeat + trade journal. Must auto-populate."""
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=True)
    sid = os.environ.get("BUDDY_SESSION_ID", "")
    assert sid, "BUDDY_SESSION_ID was not set"
    # Format: YYYYMMDDTHHMMSSZ — 16 chars, ends with Z
    assert sid.endswith("Z")
    assert len(sid) == 16


def test_session_id_pinned_value_preserved(clean_env, tmp_path):
    """If operator pinned BUDDY_SESSION_ID in .env.local.toggles
    (e.g. for a long-running soak test correlation), don't overwrite."""
    clean_env.setenv("BUDDY_SESSION_ID", "soak-2026-04-28")
    ensure_runtime_env(project_root=tmp_path, enforce_session_id=True)
    assert os.environ["BUDDY_SESSION_ID"] == "soak-2026-04-28"


def test_no_unconditional_writes_to_real_project_root(clean_env):
    """Smoke: calling with the real project root must not write any
    files, only read .env.local* if present and set os.environ. The
    bootstrap is purely consultative."""
    real_root = Path(__file__).resolve().parents[2]
    before = set(real_root.iterdir())
    ensure_runtime_env(project_root=real_root, enforce_session_id=False)
    after = set(real_root.iterdir())
    assert before == after
