"""Tier 1 T6: Tests for inline error banner surface."""
from __future__ import annotations

import pytest

from src.tui.embedded_scanner import EmbeddedScanner


def _capture_brain():
    msgs: list[str] = []
    def cb(line: str) -> None:
        msgs.append(str(line))
    return cb, msgs


def test_error_banner_field_exists_and_starts_none():
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)
    assert hasattr(es, "error_banner")
    assert es.error_banner is None


def test_error_banner_set_on_non_gate_exception(monkeypatch):
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)

    def boom(*a, **kw):
        raise FileNotFoundError("model file missing: foo.pkl")
    monkeypatch.setattr(es, "_init_scanner", boom, raising=False)
    try:
        es.run_one_cycle()
    except Exception:
        pass
    assert es.error_banner is not None
    assert "model file missing" in es.error_banner or "foo.pkl" in es.error_banner


def test_error_banner_clearable():
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)
    es.error_banner = "something broke"
    es.dismiss_error_banner()
    assert es.error_banner is None
