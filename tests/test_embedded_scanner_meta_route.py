"""Mythos audit 2026-04-30 — methodology fix for commit f070d39.

f070d39 wired Orchestrator._post_trade_diagnostics_dispatch to call
route_incident per cycle. Unit tests on Orchestrator passed; class
worked correctly in isolation. But integration verification was
skipped: a single grep proved EmbeddedScanner doesn't instantiate
Orchestrator. The TUI scan path never touches Orchestrator.run_cycle,
so the per-cycle meta routing was dead code in the live system.

This commit re-wires the same routing to EmbeddedScanner._run_one_scan
where the actual TUI scan loop lives. These tests pin:
  * meta routing fires per cycle when status != HEALTHY + actions present
  * dedup throttle on identical (status, actions) signature
  * disabled-meta short-circuit (no routing, no I/O)
  * exception isolation (route failures don't crash the scan loop)
  * empty/missing diag tolerated
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


def _make_embedded():
    """Build a bare EmbeddedScanner instance for unit testing the
    meta-route helper. Bypasses __init__ which has heavy deps."""
    from src.tui.embedded_scanner import EmbeddedScanner
    es = EmbeddedScanner.__new__(EmbeddedScanner)
    return es


def _diag(status: str = "CRITICAL", actions: list = None) -> Dict[str, Any]:
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


def test_routes_to_meta_when_enabled(meta_enabled):
    es = _make_embedded()
    captured = []
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = _diag()

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 1
    inc = captured[0]
    assert inc["kind"] == "self_heal"
    assert inc["source"] == "tui_embedded_scanner_per_cycle"
    assert inc["diag"]["status"] == "CRITICAL"


def test_dedup_skips_identical_signature(meta_enabled):
    es = _make_embedded()
    captured = []
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = _diag()

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()
        es._maybe_route_to_meta_per_cycle()
        es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 1, (
        f"Identical-signature cycles should dedup. captured={len(captured)}"
    )


def test_signature_change_re_routes(meta_enabled):
    es = _make_embedded()
    captured = []

    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.side_effect = [
        _diag(actions=["reset_gate_threshold_to_default:min_confidence"]),
        _diag(actions=["reset_gate_threshold_to_default:min_momentum"]),
    ]

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()
        es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 2


def test_does_not_route_when_meta_disabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "0")
    es = _make_embedded()
    captured = []

    # PostTradeDiagnostics should not even be called when meta disabled —
    # short-circuit avoids the diagnostic I/O cost.
    diag_called = []
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.side_effect = lambda: (
        diag_called.append(1) or _diag()
    )

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()

    assert captured == []
    assert diag_called == [], (
        "Diagnostic should not run when meta is disabled"
    )


def test_does_not_route_on_healthy_status(meta_enabled):
    es = _make_embedded()
    captured = []
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = _diag(status="HEALTHY", actions=[])

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()

    assert captured == []


def test_does_not_route_on_empty_actions(meta_enabled):
    es = _make_embedded()
    captured = []
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = _diag(actions=[])

    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()

    assert captured == []


def test_diagnostic_failure_does_not_raise(meta_enabled):
    es = _make_embedded()
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.side_effect = RuntimeError("diag broke")
    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ):
        # Must not raise.
        es._maybe_route_to_meta_per_cycle()


def test_route_failure_does_not_raise(meta_enabled):
    es = _make_embedded()
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = _diag()
    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        side_effect=RuntimeError("meta down"),
    ):
        # Must not raise.
        es._maybe_route_to_meta_per_cycle()


def test_non_dict_diag_tolerated(meta_enabled):
    es = _make_embedded()
    fake_diag_cls = MagicMock()
    fake_diag_cls.return_value.run.return_value = "not a dict"
    captured = []
    with patch(
        "src.scanner.feedback.diagnostics.PostTradeDiagnostics",
        fake_diag_cls,
    ), patch(
        "src.scanner.automation.meta_manager.route_incident",
        lambda inc: (captured.append(inc) or True),
    ):
        es._maybe_route_to_meta_per_cycle()
    assert captured == []
