"""Tier 1 T4: Tests for the transient phase indicator widget."""
from __future__ import annotations

from src.tui.widgets.phase_indicator import PhaseState


def test_default_is_idle():
    p = PhaseState()
    assert p.phase == "idle"
    assert p.detail == ""


def test_set_phase_updates_both_fields():
    p = PhaseState()
    p.set("scanning", "EUR_USD")
    assert p.phase == "scanning"
    assert p.detail == "EUR_USD"


def test_clear_resets_to_idle():
    p = PhaseState()
    p.set("scanning", "EUR_USD")
    p.clear()
    assert p.phase == "idle"
    assert p.detail == ""


def test_format_idle_returns_dim_placeholder():
    p = PhaseState()
    s = p.format()
    assert "idle" in s.lower() or s.strip() == "" or "—" in s


def test_format_active_includes_phase_and_detail():
    p = PhaseState(phase="gate-check", detail="agent 7/15 devil_advocate")
    s = p.format()
    assert "gate-check" in s.lower()
    assert "devil_advocate" in s
