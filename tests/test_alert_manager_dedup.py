"""Regression guard: AlertManager dedupes by alert_type instead of appending.

Mythos audit 2026-05-05 — operator-visible bug: 73 duplicate
`consecutive_losses` entries accumulated in active_alerts because every
`_fire_alert` append'd unconditionally. The defensive agents
(devil_advocate=1.30, risk_sentinel=1.25, uncertainty=1.10 — combined
weight 3.65 of 6.30 total) read these as "active loss streak in
progress" → voted NO on every setup → only conf >= 0.72 could clear
the WVS gate. Pairs with conf 0.40-0.69 (genuine setups) blocked
solely on agent_passed=False.

Disk evidence (.claude/alert_state.json before fix):
  active alerts: 75 (73 consecutive_losses dups, 1 win_rate_drop, 1 drawdown)
  oldest entry timestamp: 2026-03-23 (over a month old)
  newest entry timestamp: 2026-05-05 (current cycle)

NO MOCKS. Real AlertManager, real disk via tmp_path, real Alert objects
fired via the public dispatch path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.alert_manager import AlertManager


@pytest.fixture
def alert_mgr(tmp_path: Path) -> AlertManager:
    """Real AlertManager rooted at tmp_path."""
    state = tmp_path / "alert_state.json"
    log = tmp_path / "alerts.jsonl"
    return AlertManager(state_path=state, alert_log_path=log)


def _losing_trades(n: int) -> list:
    """check_consecutive_losses expects {outcome: {trade_won: bool}} entries."""
    return [{"outcome": {"trade_won": False, "pnl_pips": -10.0}} for _ in range(n)]


def _zero_cooldowns(mgr: AlertManager) -> None:
    """Bypass cooldown gate so repeated test fires aren't suppressed."""
    for cfg in mgr._configs.values():
        cfg["cooldown_minutes"] = 0


def _fire_via_check(mgr: AlertManager, n: int) -> None:
    """Run the production dispatch (check + _fire_alert)."""
    alert = mgr.check_consecutive_losses(_losing_trades(n))
    if alert is not None:
        mgr._fire_alert(alert)


def test_repeated_fire_replaces_instead_of_appending(alert_mgr: AlertManager) -> None:
    """Fire the same alert_type 5 times → exactly 1 active alert,
    holding the latest value (not 5 stacked entries).

    Pre-fix: each fire appended → 5 active alerts → operator panel
    showed phantom history.
    """
    _zero_cooldowns(alert_mgr)

    for n in (3, 5, 8, 12, 16):
        _fire_via_check(alert_mgr, n)

    active = alert_mgr.get_active_alerts()
    losses = [a for a in active if a.alert_type == "consecutive_losses"]

    assert len(losses) == 1, (
        f"Expected exactly 1 consecutive_losses alert (latest value wins), "
        f"got {len(losses)}. Pre-fix dedup bug would yield 5."
    )
    assert losses[0].value == 16.0, (
        f"Expected latest value 16, got {losses[0].value}. The replacement "
        f"isn't picking up the most-recent fire."
    )


def test_acknowledged_alerts_are_preserved_for_audit(
    alert_mgr: AlertManager,
) -> None:
    """Acknowledging an alert keeps it in the list (audit trail) but
    a subsequent fire of the same type should APPEND a new active row,
    not replace the acknowledged one.
    """
    _zero_cooldowns(alert_mgr)

    _fire_via_check(alert_mgr, 3)
    alert_mgr.acknowledge("consecutive_losses")
    _fire_via_check(alert_mgr, 8)

    losses = [
        a for a in alert_mgr._active_alerts
        if a.alert_type == "consecutive_losses"
    ]
    assert len(losses) == 2, (
        f"Expected 1 acknowledged + 1 fresh = 2 total in list, got {len(losses)}"
    )
    ack = [a for a in losses if a.acknowledged]
    unack = [a for a in losses if not a.acknowledged]
    assert len(ack) == 1 and ack[0].value == 3.0
    assert len(unack) == 1 and unack[0].value == 8.0


def test_different_alert_types_coexist(alert_mgr: AlertManager) -> None:
    """Different alert types must coexist without replacing each other."""
    _zero_cooldowns(alert_mgr)

    losses_alert = alert_mgr.check_consecutive_losses(_losing_trades(5))
    if losses_alert is not None:
        alert_mgr._fire_alert(losses_alert)
    dd_alert = alert_mgr.check_drawdown(current_nav=85_000, peak_nav=100_000)
    if dd_alert is not None:
        alert_mgr._fire_alert(dd_alert)

    active = alert_mgr.get_active_alerts()
    types = {a.alert_type for a in active}
    assert "consecutive_losses" in types
    assert "drawdown" in types


def test_disk_state_persists_dedup(alert_mgr: AlertManager, tmp_path: Path) -> None:
    """On-disk alert_state.json must reflect the dedup so a process
    restart picks up exactly one active alert per type instead of
    re-loading a polluted historical list.
    """
    _zero_cooldowns(alert_mgr)

    for n in (3, 5, 16):
        _fire_via_check(alert_mgr, n)

    state_path = tmp_path / "alert_state.json"
    assert state_path.exists(), "alert_state.json wasn't persisted"
    blob = json.loads(state_path.read_text())
    active = blob.get("active_alerts", [])
    losses = [a for a in active if a.get("alert_type") == "consecutive_losses"]
    assert len(losses) == 1, (
        f"Disk state has {len(losses)} consecutive_losses entries; expected 1"
    )
    assert losses[0].get("value") == 16.0
