from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from src.scanner.automation.state_engine import StateEngine
from src.tui.app import BuddyApp
from src.tui.data_provider import DashboardSnapshot


def _build_halted_app(monkeypatch, tmp_path, *, freshness):
    """Wire a halted+paused BuddyApp whose scanner reports the given freshness.

    Returns (app, engine) where engine reads/writes the monkeypatched
    tmp state.json shared with app.action_unhalt()'s own StateEngine.
    """
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
            "freshness": freshness,
        },
    )
    app._write_brain = lambda *a, **k: None
    app._append_control_log = lambda *a, **k: None
    app.notify = lambda *a, **k: None
    return app, engine


def test_unhalt_allows_paused_and_model_warnings(monkeypatch, tmp_path):
    """Unhalt clears pause and only WARNS on count-driven CRITICAL freshness.

    count=2/5 with oldest_age_days=0.0 → freshness is CRITICAL because per-pair
    coverage is incomplete, but the artifacts present are fresh. This is a
    warning, not a safety blocker — unhalt should resume.
    """
    app, engine = _build_halted_app(
        monkeypatch,
        tmp_path,
        freshness={"status": "CRITICAL", "oldest_age_days": 0.0},
    )

    app.action_unhalt()

    assert engine.get_halted() is False
    assert engine.get_paused() is False


def test_unhalt_still_blocks_on_genuinely_stale_models(monkeypatch, tmp_path):
    """Unhalt MUST still hard-block when models are genuinely stale (age > 7d).

    Same shape as the warning case, but oldest_age_days=30.0 — this is real
    staleness, not just incomplete coverage. Halt > break: the gate must keep
    the scanner halted (and paused) rather than resume on stale models.
    """
    app, engine = _build_halted_app(
        monkeypatch,
        tmp_path,
        freshness={"status": "CRITICAL", "oldest_age_days": 30.0},
    )

    app.action_unhalt()

    assert engine.get_halted() is True   # blocked — still halted
    assert engine.get_paused() is True   # block returns before clearing pause


def test_unhalt_blocks_when_freshness_age_unknown(monkeypatch, tmp_path):
    """STALE/CRITICAL with no age is treated conservatively as a hard block."""
    app, engine = _build_halted_app(
        monkeypatch,
        tmp_path,
        freshness={"status": "STALE"},  # no oldest_age_days
    )

    app.action_unhalt()

    assert engine.get_halted() is True
