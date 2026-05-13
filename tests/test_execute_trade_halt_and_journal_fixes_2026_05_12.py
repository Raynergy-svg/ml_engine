"""Regression tests for the orphan-trade + halt-bypass fixes (2026-05-12).

Two bugs surfaced in the post-mortem of the May 12 incident:

1. **Orphan-trade bug** — `_log_trade` early-returned at
   ``if self._memory_client is None: return`` when MLEngineMemory failed to
   import, skipping the subsequent ``_append_journal_entry`` call. Result:
   OANDA filled the order; nothing landed in ``trade_journal_rl.json``.
   Seven trades over April 15 – May 12 (1207/1211/1215/1277/1281/1287/1300)
   hit this path and required manual OANDA-reconciliation recovery.

2. **Halt-bypass bug** — ``ExecutionManager.execute_trade`` had zero
   ``StateEngine.get_halted()`` check. The TUI runtime path
   (embedded_scanner → Scanner.execute_trades → ExecutionManager.execute_trades
   → execute_trade) skips ``submit_trade``, the only other halt-aware entry.
   ``EmbeddedScanner.run_one_cycle``'s halt early-return catches halts set
   BEFORE the cycle but not mid-cycle. Trade 1306 (2026-05-12T20:39 UTC)
   opened while ``halted=True``.

No mocks: real ExecutionManager, real StateEngine, real disk via tmp_path.
The memory_client is set to None to simulate the import-fail bug condition
(which is the actual production scenario).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.execution import ExecutionConfig, ExecutionManager
from src.scanner.automation.state_engine import StateEngine


# ---------------------------------------------------------------------- Fix 1


def test_log_trade_writes_journal_when_memory_client_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Orphan-trade fix: journal write must fire even when memory_client is None.

    Pre-fix behavior: _log_trade returned early at line 3700, the journal
    append at the original line 3725 never ran, and OANDA-filled trades
    became orphans (no journal record).
    """
    monkeypatch.chdir(tmp_path)
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal.parent.mkdir(parents=True)
    journal.write_text("[]")

    mgr = ExecutionManager(config=ExecutionConfig())
    # Force the bug condition — the actual production failure mode where
    # MLEngineMemory import fails (chroma lock / import state).
    mgr._memory_client = None
    # Patch _init_memory_client to keep it None — otherwise it tries to
    # initialize (and likely fails the import the same way production does,
    # but we don't want that noise in the test).
    mgr._init_memory_client = lambda: None  # type: ignore[method-assign]

    mgr._log_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.7,
        lots=1.0,
        entry=1.0800,
        sl=1.0790,
        tp=1.0820,
        trade_id="TEST_ORPHAN_FIX_1",
        analysis_context=None,
    )

    # The journal MUST have the entry now — not gated on memory_client.
    entries = json.loads(journal.read_text())
    assert len(entries) == 1
    assert entries[0]["trade_id"] == "TEST_ORPHAN_FIX_1"
    assert entries[0]["pair"] == "EUR_USD"


def test_log_trade_writes_journal_when_memory_client_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Belt-and-suspenders: even if memory_client.log_trade raises mid-call,
    the journal write must still fire because it sits AFTER the try/except.
    """
    monkeypatch.chdir(tmp_path)
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal.parent.mkdir(parents=True)
    journal.write_text("[]")

    class _ExplodingMemoryClient:
        def log_trade(self, record):  # noqa: ARG002
            raise RuntimeError("simulated chroma write failure")

    mgr = ExecutionManager(config=ExecutionConfig())
    mgr._memory_client = _ExplodingMemoryClient()
    mgr._init_memory_client = lambda: None  # type: ignore[method-assign]

    mgr._log_trade(
        pair="GBP_USD",
        direction="SHORT",
        confidence=0.65,
        lots=2.0,
        entry=1.2500,
        sl=1.2520,
        tp=1.2460,
        trade_id="TEST_ORPHAN_FIX_2",
        analysis_context={"note": "memory client will raise"},
    )

    entries = json.loads(journal.read_text())
    assert len(entries) == 1
    assert entries[0]["trade_id"] == "TEST_ORPHAN_FIX_2"


# ---------------------------------------------------------------------- Fix 2


def test_execute_trade_blocks_when_halted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Halt-bypass fix: execute_trade must short-circuit when state.halted=True.

    Pre-fix: TUI runtime path called execute_trade directly (bypassing
    submit_trade), and execute_trade had zero halt-awareness — orders
    opened mid-cycle even when halted.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data").mkdir()

    # Set halt via real StateEngine — same code path the auto-halt loop uses.
    se = StateEngine()
    se.set_halted(True)
    assert se.get_halted() is True  # verify the fixture

    mgr = ExecutionManager(config=ExecutionConfig())

    result = mgr.execute_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.75,
        current_price=1.0800,
        atr=0.0010,
    )

    assert result.success is False
    assert "halted" in (result.error or "").lower()


def test_execute_trade_halt_check_does_not_raise_when_state_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Defensive: if the halt-check itself fails (corrupt state.json, missing
    StateEngine module, etc.), execute_trade must NOT raise out of the guard.
    It logs and proceeds — the circuit breaker downstream still catches obvious
    failures.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    # Write corrupt state.json — load_state catches JSON errors and returns
    # _DEFAULT_STATE (halted=False), so this exercises the no-halt path.
    (tmp_path / ".claude" / "state.json").write_text("{not valid json")
    (tmp_path / "trained_data").mkdir()

    mgr = ExecutionManager(config=ExecutionConfig())

    # Should NOT raise out of the halt guard. May still fail downstream
    # because there's no broker — but the failure must not be the halt guard.
    result = mgr.execute_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.75,
        current_price=1.0800,
        atr=0.0010,
    )
    # The halt guard is non-raising — either trade proceeds past it (and
    # fails for a different reason downstream), or returns a non-halt error.
    assert "halted" not in (result.error or "").lower()


def test_execute_trade_proceeds_past_halt_guard_when_not_halted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression: with halted=False, execute_trade must NOT short-circuit
    on the halt guard. It may still fail later (no broker, circuit breaker,
    ATR pre-check) — but the failure must not be the halt guard.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data").mkdir()

    se = StateEngine()
    se.set_halted(False)
    assert se.get_halted() is False

    mgr = ExecutionManager(config=ExecutionConfig())

    result = mgr.execute_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.75,
        current_price=1.0800,
        atr=0.0010,
    )
    # Must NOT be blocked by halt guard.
    assert "halted" not in (result.error or "").lower()
