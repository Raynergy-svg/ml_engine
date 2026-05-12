"""Tier 1 T3: Tests for cumulative work-unit counter."""
from __future__ import annotations

from src.tui.widgets.stats_bar import ScanCounters


def test_counters_default_zero():
    c = ScanCounters()
    assert c.cycles == 0
    assert c.pairs_scanned == 0
    assert c.gates_checked == 0
    assert c.trades_executed == 0


def test_increment_cycle_only():
    c = ScanCounters()
    c.bump_cycle()
    assert c.cycles == 1
    assert c.pairs_scanned == 0


def test_increment_pair_and_gate_and_trade():
    c = ScanCounters()
    c.bump_pair(3)
    c.bump_gates_checked(8)
    c.bump_trade(2)
    assert c.pairs_scanned == 3
    assert c.gates_checked == 8
    assert c.trades_executed == 2


def test_format_compact():
    c = ScanCounters(cycles=42, pairs_scanned=100, gates_checked=560, trades_executed=7)
    s = c.format_compact()
    assert "42" in s
    assert "100" in s
    assert "7" in s


def test_format_detailed_has_all_fields():
    c = ScanCounters(cycles=42, pairs_scanned=100, gates_checked=560, trades_executed=7)
    s = c.format_detailed()
    assert "cycles" in s.lower()
    assert "pairs" in s.lower()
    assert "gates" in s.lower()
    assert "trades" in s.lower()
