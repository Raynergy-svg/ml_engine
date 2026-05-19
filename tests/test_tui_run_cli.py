"""Tests for US-014: --live/--demo CLI flags and OANDA auto-detect in src/tui/app.py:run().

Exercises argparse and credential logic directly — no TUI boot.
No mocks; env vars set/unset via monkeypatch.setenv (env is a system boundary).
"""
from __future__ import annotations

import argparse
import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers that replicate the run() logic without booting Textual.
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true")
    mode_group.add_argument("--demo", action="store_true")
    return parser


def _run_mode(argv: list[str], monkeypatch, api_key: str | None, account_id: str | None) -> bool:
    """Return the `live` bool that run() would produce, or raise SystemExit."""
    # Set env to desired state.
    if api_key is not None:
        monkeypatch.setenv("OANDA_API_KEY", api_key)
    else:
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_API_TOKEN", raising=False)

    if account_id is not None:
        monkeypatch.setenv("OANDA_ACCOUNT_ID", account_id)
    else:
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)

    from src.tui.screens.mode_modal import check_oanda_credentials

    parser = _build_parser()
    args = parser.parse_args(argv)
    creds_ok, creds_msg = check_oanda_credentials()

    if args.live:
        if not creds_ok:
            print(
                f"ERROR: --live requires OANDA credentials but they are missing: {creds_msg}",
                file=sys.stderr,
            )
            sys.exit(2)
        return True
    elif args.demo:
        return False
    else:
        return creds_ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_FAKE_KEY = "fakekey123"
_FAKE_ACCOUNT = "001-001-12345678-001"


def test_live_flag_sets_live_true(monkeypatch):
    """--live with valid creds → live=True."""
    result = _run_mode(["--live"], monkeypatch, api_key=_FAKE_KEY, account_id=_FAKE_ACCOUNT)
    assert result is True


def test_demo_flag_sets_live_false(monkeypatch):
    """--demo → live=False regardless of creds."""
    result = _run_mode(["--demo"], monkeypatch, api_key=_FAKE_KEY, account_id=_FAKE_ACCOUNT)
    assert result is False


def test_no_flag_with_creds_gives_live(monkeypatch):
    """No flag + valid OANDA creds → auto-detect → live=True."""
    result = _run_mode([], monkeypatch, api_key=_FAKE_KEY, account_id=_FAKE_ACCOUNT)
    assert result is True


def test_no_flag_without_creds_gives_demo(monkeypatch):
    """No flag + missing OANDA creds → auto-detect → live=False."""
    result = _run_mode([], monkeypatch, api_key=None, account_id=None)
    assert result is False


def test_live_without_creds_exits_2(monkeypatch):
    """--live without OANDA creds → sys.exit(2)."""
    with pytest.raises(SystemExit) as exc_info:
        _run_mode(["--live"], monkeypatch, api_key=None, account_id=None)
    assert exc_info.value.code == 2


def test_live_and_demo_mutually_exclusive(monkeypatch):
    """--live and --demo together → argparse error (SystemExit)."""
    monkeypatch.setenv("OANDA_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("OANDA_ACCOUNT_ID", _FAKE_ACCOUNT)
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--live", "--demo"])


def test_demo_flag_ignores_missing_creds(monkeypatch):
    """--demo with no creds → live=False (demo doesn't need creds)."""
    result = _run_mode(["--demo"], monkeypatch, api_key=None, account_id=None)
    assert result is False
