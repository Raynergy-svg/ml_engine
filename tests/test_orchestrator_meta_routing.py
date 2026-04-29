"""Mythos audit 2026-04-29 — orchestrator routes per-cycle diagnostics
through the meta-pipeline (Phase 2 step A).

Bug pattern this closes:
    Pre-fix the orchestrator's _post_trade_diagnostics_dispatch ran
    every cycle but called SelfHeal directly and never invoked
    route_incident. The meta-pipeline therefore had no per-cycle hook;
    only cycle_autonomy fired it (rare, debounced on signature
    change). In production today (04-29) the .claude/meta/changes/
    directory is empty since 10:47 AM — 12+ hours of dead silence —
    despite the orchestrator running ~6 cycles/min. The architecture
    existed but was structurally orphaned.

Acceptance:
    * When enable_meta_manager=True AND status!=HEALTHY AND
      recommended_actions present, the orchestrator MUST call
      route_incident with the diag attached.
    * Dedup: identical (status, actions) signature on the next cycle
      MUST NOT call route_incident again (G8 fallback exists at intake
      but skipping the call avoids a changes_dir scan).
    * SIG change MUST re-route.
    * When enable_meta_manager=False, route_incident MUST NOT be called
    * When status=HEALTHY or actions empty, route_incident MUST NOT be called
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.scanner.automation.orchestrator import Orchestrator


def _make_orch() -> Orchestrator:
    """Construct an Orchestrator with the bare minimum the dispatch
    method needs. The full Orchestrator wires many subsystems; this
    test focuses on _maybe_route_to_meta, which is path-isolated."""
    orch = Orchestrator.__new__(Orchestrator)
    return orch


def _diag(status: str = "CRITICAL", actions: List[str] = None) -> Dict[str, Any]:
    if actions is None:
        actions = ["reset_gate_threshold_to_default:min_confidence"]
    return {
        "status": status,
        "issues": [{"check": "quiet_streak", "severity": status}],
        "recommended_actions": list(actions),
    }


@pytest.fixture
def meta_enabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "1")


@pytest.fixture
def meta_disabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "0")


def test_routes_to_meta_when_enabled(meta_enabled):
    """Happy path: meta on, CRITICAL diag with actions → route_incident
    called once with the diag attached."""
    orch = _make_orch()
    captured: List[dict] = []

    def fake_route(incident):
        captured.append(incident)
        return True

    with patch(
        "src.scanner.automation.meta_manager.route_incident", fake_route
    ):
        orch._maybe_route_to_meta(_diag(), heal_result={"status": "applied"})

    assert len(captured) == 1
    inc = captured[0]
    assert inc["kind"] == "self_heal"
    assert inc["source"] == "orchestrator_post_trade_diagnostics"
    assert inc["diag"]["status"] == "CRITICAL"
    assert inc["heal"] == {"status": "applied"}


def test_dedup_skips_identical_signature(meta_enabled):
    """Per-cycle: when the diagnostic returns the same actions twice in
    a row, only the first cycle should route. This avoids creating a
    fresh ChangePackage every 10s during an unresolved CRITICAL."""
    orch = _make_orch()
    captured: List[dict] = []

    with patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        orch._maybe_route_to_meta(_diag(), None)
        orch._maybe_route_to_meta(_diag(), None)
        orch._maybe_route_to_meta(_diag(), None)

    assert len(captured) == 1, f"Expected 1 routing call, got {len(captured)}"


def test_signature_change_re_routes(meta_enabled):
    """When the action list changes, route_incident should fire again
    so the new diagnostic state gets its own ChangePackage."""
    orch = _make_orch()
    captured: List[dict] = []

    with patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        orch._maybe_route_to_meta(
            _diag(actions=["reset_gate_threshold_to_default:min_confidence"]),
            None,
        )
        orch._maybe_route_to_meta(
            _diag(actions=["reset_gate_threshold_to_default:min_momentum"]),
            None,
        )

    assert len(captured) == 2


def test_status_change_re_routes(meta_enabled):
    """Same actions but status flipped from DEGRADED → CRITICAL is a
    new incident — must route."""
    orch = _make_orch()
    captured: List[dict] = []

    with patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        orch._maybe_route_to_meta(_diag(status="DEGRADED"), None)
        orch._maybe_route_to_meta(_diag(status="CRITICAL"), None)

    assert len(captured) == 2


def test_does_not_route_when_meta_disabled(meta_disabled):
    """Operator hasn't opted into meta — orchestrator must respect the
    flag and skip routing entirely (no changes_dir scan, no G8 dedup)."""
    orch = _make_orch()
    captured: List[dict] = []

    with patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        orch._maybe_route_to_meta(_diag(), None)

    assert captured == []


def test_does_not_route_on_empty_actions(meta_enabled):
    """A non-HEALTHY status with no recommended_actions has nothing
    actionable — emitting a meta package would be noise."""
    orch = _make_orch()
    captured: List[dict] = []

    with patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        orch._maybe_route_to_meta(_diag(actions=[]), None)

    assert captured == []


def test_route_failure_does_not_raise(meta_enabled):
    """If route_incident raises (network glitch, corrupt changes_dir),
    the orchestrator MUST keep scanning — meta is observability,
    never load-bearing on the trade flow."""
    orch = _make_orch()

    def boom(incident):
        raise RuntimeError("simulated meta failure")

    with patch(
        "src.scanner.automation.meta_manager.route_incident", boom
    ):
        # Must not raise.
        orch._maybe_route_to_meta(_diag(), None)


def test_meta_module_import_failure_swallowed():
    """When the meta module isn't importable at all (broken install,
    partial deploy), orchestrator gracefully no-ops."""
    orch = _make_orch()

    # We simulate ImportError by mocking the module-level import
    # via sys.modules manipulation isn't ideal; the simpler path is
    # to assert the function handles the ImportError branch by
    # patching the import path. Use pytest's monkeypatch.
    import sys
    saved = sys.modules.get("src.scanner.automation.meta_manager")
    try:
        sys.modules["src.scanner.automation.meta_manager"] = None  # type: ignore[assignment]
        # Must not raise.
        orch._maybe_route_to_meta(_diag(), None)
    finally:
        if saved is not None:
            sys.modules["src.scanner.automation.meta_manager"] = saved
        else:
            sys.modules.pop("src.scanner.automation.meta_manager", None)
