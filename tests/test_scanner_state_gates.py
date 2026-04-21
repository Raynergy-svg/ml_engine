"""US-502: Unit tests for scanner state gates (halted, paused, mode)."""

from __future__ import annotations

import json
import builtins
import io
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


# ── helpers ──────────────────────────────────────────────────────────

def _mock_state_engine(halted=False, paused=False, mode="dry_run"):
    """Return a MagicMock StateEngine with the given flag values."""
    se = MagicMock()
    se.get_halted.return_value = halted
    se.get_paused.return_value = paused
    se.get_mode.return_value = mode
    return se


def _mock_bus():
    published = []
    bus = MagicMock()
    bus.publish.side_effect = lambda et, p: published.append((et, p))
    return bus, published


# ── ContinuousScanner._run_smart_loop — halted gate ──────────────────

class TestRunSmartLoopHaltedGate:
    def test_halted_true_short_circuits(self):
        """When halted=True, _run_smart_loop returns without creating ExecutionManager."""
        from src.scanner.automation.continuous import ContinuousScanner
        from src.scanner.automation.event_bus import reset_event_bus

        reset_event_bus()
        se = _mock_state_engine(halted=True)
        bus, published = _mock_bus()

        scanner_mock = MagicMock()
        scanner_mock.config = MagicMock()
        cs = ContinuousScanner(scanner_mock)

        # StateEngine imported locally inside the method — patch the source module
        with patch("src.scanner.automation.state_engine.StateEngine", return_value=se), \
             patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus), \
             patch("src.scanner.execution.ExecutionManager") as em_cls:
            cs._run_smart_loop()
            em_cls.assert_not_called()

        assert any(et == "control.kill" for et, _ in published), \
            "Expected control.kill event when halted"

    def test_halted_false_proceeds(self):
        """When halted=False, _run_smart_loop proceeds past the gate."""
        from src.scanner.automation.continuous import ContinuousScanner

        se = _mock_state_engine(halted=False)
        bus, _ = _mock_bus()

        scanner_mock = MagicMock()
        scanner_mock.config = MagicMock()
        cs = ContinuousScanner(scanner_mock)

        with patch("src.scanner.automation.state_engine.StateEngine", return_value=se), \
             patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus), \
             patch("src.scanner.execution.ExecutionManager") as em_cls:
            em_instance = MagicMock()
            em_instance.monitor_open_trades.return_value = []
            em_instance.apply_drawdown_guardian.return_value = []
            em_cls.return_value = em_instance
            try:
                cs._run_smart_loop()
            except Exception:
                pass
            em_cls.assert_called()


# ── ExecutionManager.submit_trade — paused gate ──────────────────────

class TestSubmitTradePausedGate:
    def _call_submit(self, se, bus):
        from src.scanner.execution import ExecutionManager
        em = ExecutionManager()
        # Patch inside the method's local import scope
        with patch("src.scanner.automation.state_engine.StateEngine", return_value=se), \
             patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus), \
             patch.object(em, "execute_trade") as mock_exec:
            result = em.submit_trade(
                pair="EUR_USD", direction="LONG", confidence=0.75,
                current_price=1.1, atr=0.001,
            )
        return result, mock_exec

    def test_paused_returns_paused_status(self):
        se = _mock_state_engine(paused=True, mode="live")
        bus, published = _mock_bus()
        result, _ = self._call_submit(se, bus)
        assert result["status"] == "paused"
        rejected = [p for et, p in published if et == "signal.rejected"]
        assert rejected, "Expected signal.rejected event"
        assert rejected[0]["reason"] == "paused"

    def test_paused_does_not_call_execute_trade(self):
        se = _mock_state_engine(paused=True, mode="live")
        bus, _ = _mock_bus()
        _, mock_exec = self._call_submit(se, bus)
        mock_exec.assert_not_called()


# ── ExecutionManager.submit_trade — dry_run mode ─────────────────────

class TestSubmitTradeDryRunMode:
    def _call_submit_dry_run(self):
        from src.scanner.execution import ExecutionManager, ExecutionResult
        se = _mock_state_engine(paused=False, mode="dry_run")
        bus, published = _mock_bus()
        em = ExecutionManager()

        open_calls = []
        original_open = builtins.open

        def fake_open(path, mode="r", **kw):
            if "dry_run_journal" in str(path) and "a" in mode:
                buf = io.StringIO()
                open_calls.append(buf)
                # Return a context manager wrapping the StringIO
                class _CM:
                    def __enter__(self_): return buf
                    def __exit__(self_, *a): pass
                return _CM()
            return original_open(path, mode, **kw)

        with patch("src.scanner.automation.state_engine.StateEngine", return_value=se), \
             patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus), \
             patch.object(em, "execute_trade") as mock_exec, \
             patch("builtins.open", side_effect=fake_open):
            result = em.submit_trade(
                pair="EUR_USD", direction="LONG", confidence=0.75,
                current_price=1.1, atr=0.001,
            )
        return result, mock_exec, open_calls, published

    def test_dry_run_no_oanda_call(self):
        result, mock_exec, _, _ = self._call_submit_dry_run()
        mock_exec.assert_not_called()
        assert result["status"] == "dry_run"

    def test_dry_run_emits_signal_executed(self):
        _, _, _, published = self._call_submit_dry_run()
        executed = [p for et, p in published if et == "signal.executed"]
        assert executed, "Expected signal.executed event for dry_run"
        assert executed[0]["mode"] == "dry_run"


# ── ExecutionManager.submit_trade — live mode ────────────────────────

class TestSubmitTradeLiveMode:
    def test_live_calls_execute_trade(self):
        from src.scanner.execution import ExecutionManager, ExecutionResult

        se = _mock_state_engine(paused=False, mode="live")
        bus, published = _mock_bus()
        em = ExecutionManager()
        mock_result = ExecutionResult(success=True, trade_id="T123")

        with patch("src.scanner.automation.state_engine.StateEngine", return_value=se), \
             patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus), \
             patch.object(em, "execute_trade", return_value=mock_result) as mock_exec:
            result = em.submit_trade(
                pair="GBP_USD", direction="LONG", confidence=0.8,
                current_price=1.25, atr=0.0015,
            )
            mock_exec.assert_called_once()

        assert result["status"] == "executed"
        executed = [p for et, p in published if et == "signal.executed"]
        assert executed, "Expected signal.executed event on live submit"
        assert executed[0]["mode"] == "live"
