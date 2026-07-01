"""Cluster-2 (A3): MTFConfluencePanel decorative width clamping.

The panel's CONFLUENCE divider (54 dashes) and progress bar (25 blocks) were
hardcoded full-width and hard-overflowed the 1fr Overview column on an 80-col
terminal. _decor_widths() now derives both from the panel's content width so
the decorative line and bar never exceed a narrow column; full-width terminals
are unchanged. (Data-row sparkline responsiveness is a separate redesign.)

No mocks: pure static-method contract test.
"""
from __future__ import annotations

from src.tui.app import MTFConfluencePanel


def test_decor_widths_unchanged_on_wide_columns():
    # Wide allocations keep the original full-size decoration.
    assert MTFConfluencePanel._decor_widths(120) == (54, 25)
    assert MTFConfluencePanel._decor_widths(80) == (54, 25)


def test_decor_widths_clamp_on_narrow_column():
    # ~26-col 1fr column at an 80-col terminal: both shrink to fit.
    divider, bar = MTFConfluencePanel._decor_widths(26)
    assert divider <= 26 and bar <= 26
    assert (divider, bar) == (22, 8)


def test_decor_widths_safe_fallback_when_size_unknown():
    # Unmounted/first-paint (size 0) falls back near full width, never tiny.
    divider, bar = MTFConfluencePanel._decor_widths(0)
    assert divider >= 50 and bar == 25
