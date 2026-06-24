"""Cluster-2 (A2): demo-mode brain log must start at the meta-ledger EOF.

In demo mode _tick_brain tails .claude/meta/changes.jsonl from
self._brain_ledger_pos. That field was initialised to 0, so the first 0.8s
tick replayed the ENTIRE historical ledger into the brain log (the live path
already avoided this via _initial_meta_ledger_offset, which starts at EOF).

These tests pin the helper's EOF contract and that __init__ wires it.
No mocks: real module path + real disk.
"""
from __future__ import annotations

from pathlib import Path

import src.tui.app as app_mod
from src.tui.app import BuddyApp


def _ledger_path() -> Path:
    return (
        Path(app_mod.__file__).resolve().parent.parent.parent
        / ".claude" / "meta" / "changes.jsonl"
    )


def test_initial_brain_ledger_pos_returns_eof():
    """The start position equals the ledger's current size (EOF), or 0 if
    the ledger is absent — never 0 while the file has content (which would
    replay history)."""
    # Method ignores self; call it unbound to avoid constructing the app.
    pos = BuddyApp._initial_brain_ledger_pos(object())  # type: ignore[arg-type]
    ledger = _ledger_path()
    expected = ledger.stat().st_size if ledger.exists() else 0
    assert pos == expected
    assert pos >= 0


def test_init_wires_brain_ledger_pos_to_helper():
    """__init__ must seed _brain_ledger_pos from the EOF helper, not 0."""
    import inspect
    src = inspect.getsource(BuddyApp.__init__)
    assert "self._brain_ledger_pos: int = self._initial_brain_ledger_pos()" in src
    assert "self._brain_ledger_pos: int = 0" not in src
