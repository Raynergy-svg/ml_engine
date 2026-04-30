"""Mythos audit 2026-04-30 — auto-halt on consecutive-loss streak.

Operator complaint (verbatim): "make sure that the scanner doesn't
just continuously scan make sure it actually stops. And if it loses".

Pre-fix: AlertManager.check_consecutive_losses raised an alert at the
default threshold of 3, and the engine logged it as a WARNING — but
nothing acted on it. Disk evidence (.claude/alert_state.json) showed
67 consecutive_losses alerts accumulated, with the most recent one
reporting 16 consecutive losses against threshold=3. Bot kept
scanning + losing.

This test pins the new behavior: when consecutive losses reach
auto_halt_consecutive_loss_threshold (default 5), the engine sets
StateEngine.halted=True. Operator must manually un-halt to resume.

We test the helper directly (Scanner._maybe_auto_halt_on_loss_streak)
because spinning up a full Scanner instance requires model files +
OANDA mocking. The helper has only two external dependencies
(StateEngine + meta_manager.route_incident), both easy to mock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _FakeAlert:
    alert_type: str
    severity: str
    message: str
    value: float
    threshold: float


def _make_engine(threshold: int = 5):
    """Build a bare Scanner instance with just the auto-halt helper accessible."""
    from src.scanner.engine import Scanner
    eng = Scanner.__new__(Scanner)
    eng.config = MagicMock()
    eng.config.auto_halt_consecutive_loss_threshold = threshold
    return eng


def test_below_threshold_does_not_halt():
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "WARNING", "3 consecutive losses", 3.0, 3.0)]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        eng._maybe_auto_halt_on_loss_streak(alerts)
    MockSE.assert_not_called()


def test_at_threshold_halts():
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "5 consecutive losses", 5.0, 3.0)]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        instance = MockSE.return_value
        instance.get_halted.return_value = False
        eng._maybe_auto_halt_on_loss_streak(alerts)
    instance.set_halted.assert_called_once_with(True)


def test_above_threshold_halts():
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "16 consecutive losses", 16.0, 3.0)]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        instance = MockSE.return_value
        instance.get_halted.return_value = False
        eng._maybe_auto_halt_on_loss_streak(alerts)
    instance.set_halted.assert_called_once_with(True)


def test_already_halted_does_not_double_act():
    """Idempotent: when state is already halted, don't re-fire halt
    or re-route to meta on every cycle."""
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "16 losses", 16.0, 3.0)]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        instance = MockSE.return_value
        instance.get_halted.return_value = True
        with patch("src.scanner.automation.meta_manager.route_incident") as MockRoute:
            eng._maybe_auto_halt_on_loss_streak(alerts)
    instance.set_halted.assert_not_called()
    MockRoute.assert_not_called()


def test_no_consecutive_losses_alert_no_halt():
    """Drawdown or weight-instability alerts must not trigger the halt."""
    eng = _make_engine(threshold=5)
    alerts = [
        _FakeAlert("drawdown", "WARNING", "5% drawdown", 0.05, 0.04),
        _FakeAlert("weight_instability", "WARNING", "delta", 0.3, 0.2),
    ]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        eng._maybe_auto_halt_on_loss_streak(alerts)
    MockSE.assert_not_called()


def test_threshold_zero_disables_halt():
    """Operator can disable auto-halt by setting threshold to 0."""
    eng = _make_engine(threshold=0)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "100 losses", 100.0, 3.0)]
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        eng._maybe_auto_halt_on_loss_streak(alerts)
    MockSE.assert_not_called()


def test_routes_incident_to_meta_with_recommendations():
    """The reasoning layer the operator asked for: when halt fires,
    route an incident to the meta-pipeline carrying recommended
    actions for the deterministic surgeon."""
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "16 losses", 16.0, 3.0)]
    captured = []

    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        instance = MockSE.return_value
        instance.get_halted.return_value = False
        with patch(
            "src.scanner.automation.meta_manager.is_enabled", return_value=True
        ), patch(
            "src.scanner.automation.meta_manager.route_incident",
            lambda inc: (captured.append(inc) or True),
        ):
            eng._maybe_auto_halt_on_loss_streak(alerts)

    assert captured, "Expected an incident to be routed to meta-pipeline"
    inc = captured[0]
    assert inc["kind"] == "auto_halt_loss_streak"
    assert inc["signal"]["current_value"] == 16
    assert inc["signal"]["threshold"] == 5
    actions = inc["diag"]["recommended_actions"]
    assert any("reset_gate_threshold_to_default" in a for a in actions), (
        "Expected gate-reset action so deterministic surgeon can act"
    )


def test_route_failure_does_not_undo_halt():
    """If the meta routing call raises, the halt itself stays in
    effect — meta is observability, halt is the safety action."""
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "16 losses", 16.0, 3.0)]

    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        instance = MockSE.return_value
        instance.get_halted.return_value = False
        with patch(
            "src.scanner.automation.meta_manager.is_enabled", return_value=True
        ), patch(
            "src.scanner.automation.meta_manager.route_incident",
            side_effect=RuntimeError("meta down"),
        ):
            eng._maybe_auto_halt_on_loss_streak(alerts)

    instance.set_halted.assert_called_once_with(True)


def test_state_engine_failure_does_not_raise():
    """Defensive: if StateEngine itself fails to instantiate or write,
    don't propagate the exception out of the alert handler — the
    scan loop must keep running even with degraded state I/O."""
    eng = _make_engine(threshold=5)
    alerts = [_FakeAlert("consecutive_losses", "CRITICAL", "16 losses", 16.0, 3.0)]
    with patch(
        "src.scanner.automation.state_engine.StateEngine",
        side_effect=OSError("disk full"),
    ):
        # Must not raise.
        eng._maybe_auto_halt_on_loss_streak(alerts)


def test_malformed_alert_value_does_not_raise():
    """If alert.value isn't numeric (corrupted alert serialization),
    skip the alert rather than crashing the scan loop."""
    eng = _make_engine(threshold=5)
    bad_alert = _FakeAlert("consecutive_losses", "CRITICAL", "msg", "not_a_number", 3.0)
    with patch("src.scanner.automation.state_engine.StateEngine") as MockSE:
        eng._maybe_auto_halt_on_loss_streak([bad_alert])
    MockSE.assert_not_called()
