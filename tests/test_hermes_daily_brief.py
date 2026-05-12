"""Hermes daily brief — fixed-schema composition."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.hermes_daily_brief import BriefContext, run_once


UTC = timezone.utc
NOW = datetime(2026, 5, 13, 7, 0, 0, tzinfo=UTC)
MIDNIGHT_TODAY = datetime(2026, 5, 13, 0, 0, tzinfo=UTC)


def _seed_minimal(root: Path) -> None:
    """Plant minimal inputs for a clean brief composition."""
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "heartbeat.json").write_text(json.dumps({
        "scanner_alive": True,
        "ts_iso": (NOW - timedelta(seconds=2)).isoformat(),
        "cycle_count": 12500,
        "pid": 1234,
        "mode": "live",
    }))
    (claude / "state.json").write_text(json.dumps({
        "halted": False, "mode": "live",
    }))
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [],
    }))
    (root / "trained_data").mkdir(parents=True, exist_ok=True)
    (root / "trained_data" / "trade_journal_rl.json").write_text(json.dumps([]))
    (root / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({}))
    (root / "trained_data" / "models").mkdir(parents=True, exist_ok=True)


def _ctx(root: Path) -> BriefContext:
    return BriefContext(repo_root=root, now=NOW)


def test_minimal_brief_composes_with_nominal_notable(tmp_path: Path):
    _seed_minimal(tmp_path)
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "### 07:00Z — brief" in digest
    assert "halted: false" in digest
    assert "mode: live" in digest
    assert "all systems nominal" in digest


def test_cycle_count_delta_uses_last_brief_state(tmp_path: Path):
    _seed_minimal(tmp_path)
    state_path = tmp_path / ".claude" / "hermes_state.json"
    state_path.write_text(json.dumps({
        "last_brief_at_iso": (NOW - timedelta(days=1)).isoformat(),
        "cycle_count_at_last_brief": 12400,
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "cycles_today: 100" in digest


def test_first_brief_reports_unknown_cycles_today(tmp_path: Path):
    _seed_minimal(tmp_path)
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "cycles_today: unknown" in digest


def test_brief_persists_cycle_count_for_next_run(tmp_path: Path):
    _seed_minimal(tmp_path)
    run_once(_ctx(tmp_path))
    state = json.loads((tmp_path / ".claude" / "hermes_state.json").read_text())
    assert state["cycle_count_at_last_brief"] == 12500
    assert state["last_brief_at_iso"] == NOW.isoformat()


def test_trades_24h_counts_only_today_closes(tmp_path: Path):
    _seed_minimal(tmp_path)
    trades = [
        {"pair": "EUR_USD", "direction": "LONG", "pnl": 50.0,
         "close_time": (MIDNIGHT_TODAY - timedelta(hours=2)).isoformat()},
        {"pair": "GBP_USD", "direction": "SHORT", "pnl": -23.0,
         "close_time": (MIDNIGHT_TODAY + timedelta(hours=3)).isoformat()},
        {"pair": "USD_JPY", "direction": "LONG", "pnl": 12.5,
         "close_time": (MIDNIGHT_TODAY + timedelta(hours=5)).isoformat()},
    ]
    (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps(trades))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "trades_24h: 2 trades" in digest
    assert "-10.50" in digest


def test_notable_priority_halted_loss_streak(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / ".claude" / "state.json").write_text(json.dumps({
        "halted": True, "mode": "live",
    }))
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps({
        "active_alerts": [{
            "alert_type": "consecutive_losses",
            "value": 5.0, "threshold": 3.0,
            "acknowledged": False,
            "message": "5 consecutive losses",
        }],
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "halted on loss streak" in digest


def test_notable_priority_halted_alone(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / ".claude" / "state.json").write_text(json.dumps({
        "halted": True, "mode": "live",
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "halted; operator un-halt required to resume" in digest


def test_notable_priority_job_failure(tmp_path: Path):
    _seed_minimal(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "active",
            "last_status": "failure",
            "last_error": "exit 1: out of disk",
            "last_status_at": (NOW - timedelta(hours=2)).isoformat(),
            "run_count": 7,
        },
    }))
    run_once(_ctx(tmp_path))
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "scheduled job nightly_audit failed" in digest


def test_alerts_24h_counts_watch_entries(tmp_path: Path):
    _seed_minimal(tmp_path)
    digest_path = tmp_path / ".claude" / "brain" / "hermes_watchdog.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(
        "# Hermes Watchdog Digest\n\n"
        "## 2026-05-12\n\n"
        "### 14:30Z — watch\n- trigger: consecutive_losses\n\n"
        "### 18:00Z — watch\n- trigger: drawdown\n\n"
        "## 2026-05-11\n\n"
        "### 23:00Z — watch\n- trigger: stale_models\n\n"
    )
    run_once(_ctx(tmp_path))
    digest = digest_path.read_text()
    assert "alerts_24h: 2 watch entries" in digest
