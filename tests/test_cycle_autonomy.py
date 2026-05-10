"""Tests for CycleAutonomyTriggers — the module that fires Claude on
cycle events (periodic, self-heal, rejection streak) so buddy isn't
silent whenever no trade closes."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Ensure the three feature flags are in their default-enabled state."""
    for key in (
        "BUDDY_DISABLE_PERIODIC_REFLECTION",
        "BUDDY_DISABLE_SELF_HEAL_REFLECTION",
        "BUDDY_DISABLE_REJECTION_REFLECTION",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_triggers(**kwargs):
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    brain = MagicMock()
    kwargs.setdefault("brain_callback", brain)
    t = CycleAutonomyTriggers(**kwargs)
    # Stash brain for assertions
    t._test_brain = brain
    return t


def _fake_scan_result(tradeable_count=0, analyses_count=0):
    return SimpleNamespace(
        tradeable=[object()] * tradeable_count,
        analyses=[object()] * analyses_count,
    )


# ── Periodic ───────────────────────────────────────────────────────────


def test_periodic_fires_every_n_cycles():
    t = _make_triggers(periodic_every_n=3)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(
        (trade_id, mode)
    )
    # Force-skip the self-heal check so we isolate periodic behavior
    t._should_fire_self_heal = lambda: False
    # 2026-05-10 (Angle 3'): periodic now short-circuits when _pending_diag
    # is null. To exercise the heavy-path schedule (which is what this test
    # is about), populate a structured diag so the short-circuit doesn't
    # fire. Tests for the short-circuit itself live in
    # test_cycle_autonomy_null_diag.py.
    t._pending_diag = {"recommended_actions": ["noop_for_test"]}

    for cycle in range(1, 10):
        t.on_cycle_complete(
            scan_count=cycle,
            scan_result=_fake_scan_result(0, 5),
            trades_executed=0,
            tradeable_count=0,
        )

    # Should fire at cycles 3, 6, 9 = 3 times
    periodic_calls = [c for c in invoked if c[0].startswith("PERIODIC-")]
    assert len(periodic_calls) == 3
    # All lightweight mode (not deep)
    assert all(c[1] == "lightweight" for c in periodic_calls)


def test_periodic_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BUDDY_DISABLE_PERIODIC_REFLECTION", "1")
    t = _make_triggers(periodic_every_n=1)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)
    t._should_fire_self_heal = lambda: False

    for cycle in range(1, 5):
        t.on_cycle_complete(
            scan_count=cycle,
            scan_result=_fake_scan_result(),
            trades_executed=0,
            tradeable_count=0,
        )
    periodic = [c for c in invoked if c.startswith("PERIODIC-")]
    assert len(periodic) == 0


# ── Rejection streak ───────────────────────────────────────────────────


def test_rejection_fires_after_streak_threshold():
    t = _make_triggers(periodic_every_n=999, rejection_streak_threshold=3)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(
        (trade_id, mode)
    )
    t._should_fire_self_heal = lambda: False
    # 2026-05-10 (Angle 3'): rejection now short-circuits when _pending_diag
    # is null. Populate a structured diag so the heavy-path scheduling
    # logic still runs; null-diag short-circuit is covered separately.
    t._pending_diag = {"recommended_actions": ["noop_for_test"]}

    # 3 consecutive cycles with tradeable but zero executed → fires
    for cycle in range(1, 4):
        t.on_cycle_complete(
            scan_count=cycle,
            scan_result=_fake_scan_result(tradeable_count=2),
            trades_executed=0,
            tradeable_count=2,
        )

    rejection_calls = [c for c in invoked if c[0].startswith("REJECTION-")]
    assert len(rejection_calls) == 1, f"expected 1 rejection fire, got {rejection_calls}"


def test_rejection_streak_resets_on_execution():
    t = _make_triggers(periodic_every_n=999, rejection_streak_threshold=3)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)
    t._should_fire_self_heal = lambda: False

    # 2 rejections, then a successful execution, then 2 more rejections
    # → streak counter reset → NO rejection fire (never reached 3 in a row)
    for trades in [0, 0, 1, 0, 0]:
        t.on_cycle_complete(
            scan_count=1,
            scan_result=_fake_scan_result(tradeable_count=2),
            trades_executed=trades,
            tradeable_count=2,
        )
    rejection_calls = [c for c in invoked if c.startswith("REJECTION-")]
    assert len(rejection_calls) == 0


def test_rejection_ignores_dry_scans():
    """No tradeable = no rejection. Streak only counts when gates PASSED but
    execution filters stopped the trade."""
    t = _make_triggers(periodic_every_n=999, rejection_streak_threshold=3)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)
    t._should_fire_self_heal = lambda: False

    # 5 cycles with no tradeable setups at all (dry scans, not rejections)
    for cycle in range(1, 6):
        t.on_cycle_complete(
            scan_count=cycle,
            scan_result=_fake_scan_result(tradeable_count=0),
            trades_executed=0,
            tradeable_count=0,
        )
    rejection_calls = [c for c in invoked if c.startswith("REJECTION-")]
    assert len(rejection_calls) == 0


# ── Self-heal ──────────────────────────────────────────────────────────


def test_self_heal_fires_on_degraded_diagnosis():
    t = _make_triggers(periodic_every_n=999)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(
        (trade_id, mode)
    )

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics"
    ) as MockDiag:
        MockDiag.return_value.run.return_value = {
            "status": "DEGRADED",
            "issues": [
                {"check": "gate_health", "severity": "warning", "detail": "low pass rate"},
            ],
            "recommended_actions": ["lower weighted_vote_threshold"],
        }
        t.on_cycle_complete(
            scan_count=1,
            scan_result=_fake_scan_result(),
            trades_executed=0,
            tradeable_count=0,
        )

    self_heal_calls = [c for c in invoked if c[0].startswith("SELFHEAL-")]
    assert len(self_heal_calls) == 1
    assert self_heal_calls[0][1] == "deep"  # self-heal is always deep mode


def test_self_heal_deduplicates_same_issue():
    """Same issue across cycles should not keep spawning Claude. The signature
    tracking ensures one spawn per distinct issue set."""
    t = _make_triggers(periodic_every_n=999)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)

    same_diag = {
        "status": "DEGRADED",
        "issues": [
            {"check": "gate_health", "severity": "warning", "detail": "low pass rate"},
        ],
        "recommended_actions": [],
    }
    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics"
    ) as MockDiag:
        MockDiag.return_value.run.return_value = same_diag
        for cycle in range(1, 6):
            t.on_cycle_complete(
                scan_count=cycle,
                scan_result=_fake_scan_result(),
                trades_executed=0,
                tradeable_count=0,
            )
    self_heal_calls = [c for c in invoked if c.startswith("SELFHEAL-")]
    assert len(self_heal_calls) == 1


def test_self_heal_fires_again_when_new_issue_appears():
    t = _make_triggers(periodic_every_n=999)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)

    first = {
        "status": "DEGRADED",
        "issues": [{"check": "gate_health", "severity": "warning", "detail": "x"}],
    }
    second = {
        "status": "CRITICAL",
        "issues": [
            {"check": "gate_health", "severity": "warning", "detail": "x"},
            {"check": "agent_drift", "severity": "critical", "detail": "y"},
        ],
    }
    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics"
    ) as MockDiag:
        MockDiag.return_value.run.side_effect = [first, first, second]
        for cycle in range(1, 4):
            t.on_cycle_complete(
                scan_count=cycle,
                scan_result=_fake_scan_result(),
                trades_executed=0,
                tradeable_count=0,
            )
    # First fire on cycle 1, dedup on 2, second fire on 3 (new issue)
    self_heal_calls = [c for c in invoked if c.startswith("SELFHEAL-")]
    assert len(self_heal_calls) == 2


def test_self_heal_healthy_never_fires():
    t = _make_triggers(periodic_every_n=999)
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics"
    ) as MockDiag:
        MockDiag.return_value.run.return_value = {
            "status": "HEALTHY",
            "issues": [],
        }
        for cycle in range(1, 5):
            t.on_cycle_complete(
                scan_count=cycle,
                scan_result=_fake_scan_result(),
                trades_executed=0,
                tradeable_count=0,
            )
    self_heal_calls = [c for c in invoked if c.startswith("SELFHEAL-")]
    assert len(self_heal_calls) == 0


# ── Priority: self-heal > rejection > periodic ─────────────────────────


def test_self_heal_preempts_periodic_on_same_cycle():
    t = _make_triggers(periodic_every_n=1)  # periodic would fire every cycle
    invoked = []
    t._invoke = lambda prompt, trade_id, mode, timeout: invoked.append(trade_id)

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics"
    ) as MockDiag:
        MockDiag.return_value.run.return_value = {
            "status": "DEGRADED",
            "issues": [{"check": "x", "severity": "warning", "detail": "y"}],
        }
        t.on_cycle_complete(
            scan_count=1,
            scan_result=_fake_scan_result(),
            trades_executed=0,
            tradeable_count=0,
        )
    assert len(invoked) == 1
    assert invoked[0].startswith("SELFHEAL-")  # not PERIODIC
