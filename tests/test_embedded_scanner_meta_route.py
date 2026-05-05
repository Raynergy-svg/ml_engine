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

from typing import Any, Dict
from types import SimpleNamespace

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


class _DiagRunner:
    values: list[Any] = []
    calls: int = 0

    def run(self) -> Any:
        type(self).calls += 1
        if not type(self).values:
            return _diag()
        value = type(self).values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class _DrainManager:
    def __init__(self, counters: dict[str, int] | None = None) -> None:
        self.counters = counters or {"approved": 1, "deployed": 0}
        self.calls: list[int] = []

    def drain(self, current_cycle: int) -> dict[str, int]:
        self.calls.append(current_cycle)
        return self.counters


@pytest.fixture
def meta_enabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "1")


def test_routes_to_meta_when_enabled(meta_enabled, monkeypatch):
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [_diag()]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 1
    inc = captured[0]
    assert inc["kind"] == "self_heal"
    assert inc["source"] == "tui_embedded_scanner_per_cycle"
    assert inc["diag"]["status"] == "CRITICAL"


def test_sync_meta_env_from_config_enables_smart_profile(monkeypatch):
    """TUI live path must export config flags because it bypasses Orchestrator."""
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "0")
    monkeypatch.setenv("BUDDY_META_USE_LLM", "1")
    es = _make_embedded()

    es._sync_meta_env_from_config(SimpleNamespace(
        enable_meta_manager=True,
        meta_manager_use_llm=False,
    ))

    assert __import__("os").environ["BUDDY_META_MANAGER_ENABLED"] == "1"
    assert __import__("os").environ["BUDDY_META_USE_LLM"] == "0"


def test_sync_meta_env_from_config_disables_when_profile_off(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "1")
    es = _make_embedded()

    es._sync_meta_env_from_config(SimpleNamespace(
        enable_meta_manager=False,
        meta_manager_use_llm=False,
    ))

    assert __import__("os").environ["BUDDY_META_MANAGER_ENABLED"] == "0"


def test_resolve_scan_pairs_uses_active_instruments_for_asset_switch() -> None:
    es = _make_embedded()
    es._config = SimpleNamespace(
        pairs=[],
        default_pairs=["EUR_USD"],
        active_instruments=["ES", "NQ"],
        blocked_pairs=["NQ"],
    )

    assert es._resolve_scan_pairs() == ["ES"]


def test_dedup_skips_identical_signature(meta_enabled, monkeypatch):
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [_diag(), _diag(), _diag()]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()
    es._maybe_route_to_meta_per_cycle()
    es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 1, (
        f"Identical-signature cycles should dedup. captured={len(captured)}"
    )


def test_signature_change_re_routes(meta_enabled, monkeypatch):
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [
        _diag(actions=["reset_gate_threshold_to_default:min_confidence"]),
        _diag(actions=["reset_gate_threshold_to_default:min_momentum"]),
    ]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()
    es._maybe_route_to_meta_per_cycle()

    assert len(captured) == 2


def test_does_not_route_when_meta_disabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "0")
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [_diag()]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()

    assert captured == []
    assert _DiagRunner.calls == 0, (
        "Diagnostic should not run when meta is disabled"
    )


def test_drains_meta_pipeline_when_enabled(meta_enabled, monkeypatch):
    es = _make_embedded()
    es._scan_count = 7
    manager = _DrainManager()

    from src.scanner.automation import meta_manager as _mm

    monkeypatch.setattr(_mm, "_PRODUCTION_MGR", None)
    monkeypatch.setattr(_mm, "_build_production_manager", lambda: manager)

    es._maybe_drain_meta_pipeline()

    assert _mm._PRODUCTION_MGR is manager
    assert manager.calls == [7]


def test_does_not_drain_meta_pipeline_when_disabled(monkeypatch):
    monkeypatch.setenv("BUDDY_META_MANAGER_ENABLED", "0")
    es = _make_embedded()
    es._scan_count = 3

    from src.scanner.automation import meta_manager as _mm

    build_calls = []
    monkeypatch.setattr(_mm, "_PRODUCTION_MGR", None)
    monkeypatch.setattr(_mm, "_build_production_manager", lambda: build_calls.append(1))

    es._maybe_drain_meta_pipeline()

    assert build_calls == []


def test_does_not_route_on_healthy_status(meta_enabled, monkeypatch):
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [_diag(status="HEALTHY", actions=[])]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()

    assert captured == []


def test_does_not_route_on_empty_actions(meta_enabled, monkeypatch):
    es = _make_embedded()
    captured = []
    _DiagRunner.values = [_diag(actions=[])]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()

    assert captured == []


def test_diagnostic_failure_does_not_raise(meta_enabled, monkeypatch):
    es = _make_embedded()
    _DiagRunner.values = [RuntimeError("diag broke")]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)

    # Must not raise.
    es._maybe_route_to_meta_per_cycle()


def test_route_failure_does_not_raise(meta_enabled, monkeypatch):
    es = _make_embedded()
    _DiagRunner.values = [_diag()]
    _DiagRunner.calls = 0

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)

    def fail_route(_inc):
        raise RuntimeError("meta down")

    monkeypatch.setattr(meta_manager, "route_incident", fail_route)

    # Must not raise.
    es._maybe_route_to_meta_per_cycle()


def test_non_dict_diag_tolerated(meta_enabled, monkeypatch):
    es = _make_embedded()
    _DiagRunner.values = ["not a dict"]
    _DiagRunner.calls = 0
    captured = []

    from src.scanner.feedback import diagnostics
    from src.scanner.automation import meta_manager

    monkeypatch.setattr(diagnostics, "PostTradeDiagnostics", _DiagRunner)
    monkeypatch.setattr(meta_manager, "route_incident", lambda inc: (captured.append(inc) or True))

    es._maybe_route_to_meta_per_cycle()
    assert captured == []
