"""
Integration tests for US-511: Pre-trade veto window.

Tests:
  1. Veto during window → no OANDA submit, journal has vetoed_by='operator'
  2. Window expires without veto → OANDA submit called
  3. Performance: 0ms window adds <5ms p99 overhead over 1000 cycles
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scanner.automation.event_bus import EventBus, get_event_bus, reset_event_bus


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_mock_execution_manager(veto_ms: int = 0, config_pair_overrides: Dict[str, int] | None = None):
    """Build a minimal ExecutionManager mock with veto config attached."""
    from src.scanner.execution import ExecutionManager

    cfg = MagicMock()
    cfg.veto_window_ms_default = veto_ms
    cfg.veto_window_ms_per_pair = config_pair_overrides or {}

    em = MagicMock(spec=ExecutionManager)
    em.config = cfg
    # Wire submit_trade to the real implementation bound to this mock
    em.submit_trade = ExecutionManager.submit_trade.__get__(em, ExecutionManager)
    return em


def _call_submit(em, pair="EUR_USD", direction="BUY", confidence=0.75,
                 current_price=1.0850, atr=0.0012, analysis_context=None):
    return em.submit_trade(
        pair=pair,
        direction=direction,
        confidence=confidence,
        current_price=current_price,
        atr=atr,
        analysis_context=analysis_context or {},
    )


# ─────────────────────────────────────────────────────────────────────
# Test 1: Veto during window → no execute_trade, journal has vetoed_by
# ─────────────────────────────────────────────────────────────────────

def test_veto_during_window_cancels_execution(tmp_path):
    """Inject synthetic signal with 3s window, send veto during window.
    Assert: no OANDA submit, trade_journal_rl.json has vetoed_by='operator'."""
    reset_event_bus()
    bus = get_event_bus()

    # Patch journal path to tmp_path
    journal_path = tmp_path / "trade_journal_rl.json"

    em = _make_mock_execution_manager(veto_ms=3000)
    em.execute_trade = MagicMock()  # should NOT be called

    state_engine_mock = MagicMock()
    state_engine_mock.get_paused.return_value = False
    state_engine_mock.get_mode.return_value = "live"

    collected_signal_ids: list = []

    def _spy_pending(evt):
        if evt.get("event_type") == "signal.pending":
            sig_id = evt["payload"].get("signal_id", "")
            if sig_id:
                collected_signal_ids.append(sig_id)
                # Simulate operator pressing A after ~200ms
                def _veto():
                    time.sleep(0.2)
                    bus.publish("control.signal_veto", {
                        "signal_id": sig_id,
                        "vetoed_by": "operator",
                    })
                threading.Thread(target=_veto, daemon=True).start()

    # Lightweight replay-based spy: subscribe in background thread
    import asyncio

    async def _async_spy():
        async for evt in bus.subscribe():
            _spy_pending(evt)

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    asyncio.run_coroutine_threadsafe(_async_spy(), loop)
    time.sleep(0.05)  # let subscriber register

    with (
        patch("src.scanner.automation.state_engine.StateEngine", return_value=state_engine_mock),
        patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus),
    ):
        # Redirect the journal to tmp_path by chdir'ing
        _cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = _call_submit(em)
        finally:
            os.chdir(_cwd)

    loop.call_soon_threadsafe(loop.stop)

    assert result["status"] == "vetoed", f"Expected vetoed, got: {result}"
    assert result.get("vetoed_by") == "operator"
    assert "signal_id" in result
    em.execute_trade.assert_not_called()

    # Journal check — since we chdir'd to tmp_path, journal is at tmp_path/trained_data/
    jpath = tmp_path / "trained_data" / "trade_journal_rl.json"
    assert jpath.exists(), f"Journal file not found at {jpath}"
    entries = json.loads(jpath.read_text())
    vetoed = [e for e in entries if e.get("vetoed_by") == "operator"]
    assert vetoed, "No vetoed_by='operator' entry found in journal"
    assert vetoed[-1]["pair"] == "EUR_USD"
    assert vetoed[-1]["direction"] == "BUY"
    assert vetoed[-1]["outcome"] == "vetoed"


# ─────────────────────────────────────────────────────────────────────
# Test 2: Window expires without veto → execute_trade called
# ─────────────────────────────────────────────────────────────────────

def test_window_expiry_proceeds_to_execution():
    """Window expires without veto → OANDA submit (execute_trade) is called."""
    reset_event_bus()
    bus = get_event_bus()

    em = _make_mock_execution_manager(veto_ms=300)  # 300ms window for speed

    exec_result = MagicMock()
    exec_result.success = True
    exec_result.trade_id = "T-9999"
    em.execute_trade = MagicMock(return_value=exec_result)

    state_engine_mock = MagicMock()
    state_engine_mock.get_paused.return_value = False
    state_engine_mock.get_mode.return_value = "live"

    with (
        patch("src.scanner.automation.state_engine.StateEngine", return_value=state_engine_mock),
        patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus),
    ):
        result = _call_submit(em)

    assert result["status"] == "executed", f"Expected executed, got: {result}"
    em.execute_trade.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Test 3: Per-pair override takes precedence over default
# ─────────────────────────────────────────────────────────────────────

def test_per_pair_override_takes_precedence():
    """veto_window_ms_per_pair[pair] overrides veto_window_ms_default."""
    reset_event_bus()
    bus = get_event_bus()

    # default=500ms, but EUR_USD overridden to 0 → should skip veto window entirely
    em = _make_mock_execution_manager(veto_ms=500, config_pair_overrides={"EUR_USD": 0})

    exec_result = MagicMock()
    exec_result.success = True
    exec_result.trade_id = "T-0001"
    em.execute_trade = MagicMock(return_value=exec_result)

    state_engine_mock = MagicMock()
    state_engine_mock.get_paused.return_value = False
    state_engine_mock.get_mode.return_value = "live"

    start = time.monotonic()
    with (
        patch("src.scanner.automation.state_engine.StateEngine", return_value=state_engine_mock),
        patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus),
    ):
        result = _call_submit(em, pair="EUR_USD")
    elapsed_ms = (time.monotonic() - start) * 1000

    assert result["status"] == "executed"
    # Should not wait 500ms — should be instant (0ms override)
    assert elapsed_ms < 100, f"With 0ms override expected <100ms, got {elapsed_ms:.1f}ms"


# ─────────────────────────────────────────────────────────────────────
# Test 4: signal.pending payload completeness
# ─────────────────────────────────────────────────────────────────────

def test_signal_pending_payload_fields():
    """signal.pending must include all required fields per AC."""
    reset_event_bus()
    bus = get_event_bus()

    captured: list = []

    import asyncio

    async def _capture():
        async for evt in bus.subscribe():
            if evt.get("event_type") == "signal.pending":
                captured.append(evt["payload"])
                return  # stop after first

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    fut = asyncio.run_coroutine_threadsafe(_capture(), loop)
    time.sleep(0.05)

    em = _make_mock_execution_manager(veto_ms=200)

    exec_result = MagicMock()
    exec_result.success = True
    exec_result.trade_id = "T-X"
    em.execute_trade = MagicMock(return_value=exec_result)

    state_engine_mock = MagicMock()
    state_engine_mock.get_paused.return_value = False
    state_engine_mock.get_mode.return_value = "live"

    with (
        patch("src.scanner.automation.state_engine.StateEngine", return_value=state_engine_mock),
        patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus),
    ):
        _call_submit(em, pair="GBP_USD", direction="SELL")

    loop.call_soon_threadsafe(loop.stop)

    required_fields = {"signal_id", "pair", "direction", "entry", "sl", "tp",
                       "weighted_vote_score", "window_ms", "deadline_iso"}
    assert captured, "No signal.pending event captured"
    payload = captured[0]
    missing = required_fields - set(payload.keys())
    assert not missing, f"signal.pending missing fields: {missing}"
    assert payload["pair"] == "GBP_USD"
    assert payload["direction"] == "SELL"
    assert payload["window_ms"] == 200


# ─────────────────────────────────────────────────────────────────────
# Test 5: Performance benchmark — 0ms window adds <5ms p99 over 1000 cycles
# ─────────────────────────────────────────────────────────────────────

def test_zero_ms_window_overhead_p99():
    """0ms veto window must add <5ms overhead to execution path (p99 over 1000 cycles)."""
    reset_event_bus()
    bus = get_event_bus()

    em = _make_mock_execution_manager(veto_ms=0)

    exec_result = MagicMock()
    exec_result.success = True
    exec_result.trade_id = "T-PERF"
    em.execute_trade = MagicMock(return_value=exec_result)

    state_engine_mock = MagicMock()
    state_engine_mock.get_paused.return_value = False
    state_engine_mock.get_mode.return_value = "live"

    latencies_ms: List[float] = []

    with (
        patch("src.scanner.automation.state_engine.StateEngine", return_value=state_engine_mock),
        patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus),
    ):
        for _ in range(1000):
            start = time.monotonic()
            _call_submit(em)
            latencies_ms.append((time.monotonic() - start) * 1000)

    latencies_ms.sort()
    p99 = latencies_ms[int(0.99 * len(latencies_ms))]
    print(f"\n[perf] 0ms veto window p99 overhead: {p99:.2f}ms (median: {latencies_ms[500]:.2f}ms)")
    assert p99 < 5.0, f"p99 latency {p99:.2f}ms exceeds 5ms budget"


# ─────────────────────────────────────────────────────────────────────
# Test 6: EventBus veto registry — register/trigger/cleanup
# ─────────────────────────────────────────────────────────────────────

def test_event_bus_veto_registry():
    """register_veto_wait sets the event when matching control.signal_veto is published."""
    reset_event_bus()
    bus = get_event_bus()

    sig_id = str(uuid.uuid4())
    evt = bus.register_veto_wait(sig_id)
    assert not evt.is_set()

    bus.publish("control.signal_veto", {"signal_id": sig_id, "vetoed_by": "operator"})
    assert evt.is_set()

    # Registry entry should be cleaned up after trigger
    with bus._lock:
        assert sig_id not in bus._veto_events


def test_event_bus_veto_wrong_signal_id_does_not_fire():
    """control.signal_veto with non-matching signal_id leaves other events untouched."""
    reset_event_bus()
    bus = get_event_bus()

    sig_id_a = str(uuid.uuid4())
    sig_id_b = str(uuid.uuid4())
    evt_a = bus.register_veto_wait(sig_id_a)

    bus.publish("control.signal_veto", {"signal_id": sig_id_b, "vetoed_by": "operator"})
    assert not evt_a.is_set(), "Wrong signal_id should not fire evt_a"

    bus.cancel_veto_wait(sig_id_a)


# ─────────────────────────────────────────────────────────────────────
# Test 7: ScannerConfig fields present with correct defaults
# ─────────────────────────────────────────────────────────────────────

def test_scanner_config_veto_fields():
    """ScannerConfig must have veto_window_ms_default and veto_window_ms_per_pair."""
    from src.scanner.config import ScannerConfig
    cfg = ScannerConfig()
    assert hasattr(cfg, "veto_window_ms_default"), "veto_window_ms_default missing"
    assert hasattr(cfg, "veto_window_ms_per_pair"), "veto_window_ms_per_pair missing"
    assert cfg.veto_window_ms_default == 0
    assert isinstance(cfg.veto_window_ms_per_pair, dict)
    assert cfg.veto_window_ms_per_pair == {}


def test_scanner_config_veto_fields_in_smart_profile():
    """apply_profile('smart') must propagate veto window fields without error."""
    from src.scanner.config import ScannerConfig
    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    # smart profile sets veto_window_ms_default=0 explicitly
    assert cfg.veto_window_ms_default == 0
    assert isinstance(cfg.veto_window_ms_per_pair, dict)
