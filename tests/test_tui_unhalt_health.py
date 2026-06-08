from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.scanner.automation.state_engine import StateEngine
from src.tui.app import BuddyApp
from src.tui.data_provider import DashboardSnapshot

# SPEC (pending OPERATOR DECISION — do not flip without sign-off): this asserts
# action_unhalt() should clear both halted AND paused while only WARNING on
# CRITICAL model freshness, instead of hard-blocking. Today _evaluate_unhalt_health
# hard-blocks on `scanner is paused`, incomplete models, and STALE/CRITICAL
# freshness (src/tui/app.py ~2462/2488/2492). Softening a halt-safety gate
# conflicts with the documented "Halt > break" invariant and the staleness
# trading rules. The nuance the operator must rule on: the test's CRITICAL is
# count-driven (2/5 models) with age=0.0 (fresh) — arguably benign — but a blanket
# demotion would also wave through genuinely stale (old) models. No implementation
# exists in any stash; it was never written.
pytestmark = pytest.mark.xfail(
    reason="pending operator decision: softening the unhalt halt-safety gate "
    "(clear-pause + warn-not-block on model freshness) conflicts with the "
    "Halt > break invariant; no implementation exists",
    strict=False,
)


def test_unhalt_allows_paused_and_model_warnings(monkeypatch, tmp_path):
    """Unhalt should clear pause and warn on model gaps, not hard-block."""
    import src.scanner.automation.state_engine as state_mod
    import src.tui.app as app_mod

    state_path = tmp_path / ".claude" / "state.json"
    heartbeat_path = tmp_path / ".claude" / "heartbeat.json"
    heartbeat_path.parent.mkdir(parents=True)
    heartbeat_path.write_text(json.dumps({
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "scanner_alive": True,
    }))

    monkeypatch.setattr(app_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path)

    engine = StateEngine()
    engine.set_halted(True)
    engine.set_paused(True)

    app = BuddyApp.__new__(BuddyApp)
    app._live = True
    app._provider = SimpleNamespace(
        snapshot=DashboardSnapshot(oanda_connected=True, scanner_paused=True)
    )
    app._scanner = SimpleNamespace(
        is_ready=True,
        get_model_health=lambda: {
            "count": 2,
            "total": 5,
            "freshness": {"status": "CRITICAL", "oldest_age_days": 0.0},
        },
    )
    app._write_brain = lambda *a, **k: None
    app._append_control_log = lambda *a, **k: None
    app.notify = lambda *a, **k: None

    app.action_unhalt()

    assert engine.get_halted() is False
    assert engine.get_paused() is False
