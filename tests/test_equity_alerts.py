"""US-018 — Out-of-band alerting tests.

No mocks. Drives real :class:`HarvesterAlertManager`, real
:class:`FileNotificationChannel`, real :class:`EmailNotificationChannel`
(in dry-run mode with a real audit dir), real
:class:`SlackWebhookChannel` (in dry-run mode with a real audit dir),
real ``tmp_path`` disk for SHIP_GATE.json, alert-state JSON, and the
outbox.

Acceptance criteria covered:

* :func:`enforce_ship_gate` rejects missing / unreadable / non-dict /
  ``gate_pass != True`` / missing-hash / mismatched-hash payloads
  (mirrors the US-012..US-016 guard tests).
* Simulated breach (DD breach, broker disconnect, halt, stale data,
  rebalance failure, position drift) emits a notification payload
  through every configured channel.
* The harvester alert vocabulary is exactly the six harvester
  conditions — the directional ``consecutive_losses`` /
  ``win_rate_drop`` types are absent.
* Cooldown debouncing — same alert twice within the cooldown window
  fires once. ``force=True`` bypasses the cooldown.
* Atomic state writes (``tmp + os.replace``) — alert-state JSON
  remains valid across re-construction with prior cooldown state.
* The on-disk source ``src/equity/alerts.py`` contains zero bare
  ``except:`` clauses (greps the file).
* The test file imports zero ``unittest.mock`` symbols.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from src.equity.alerts import (
    ALERT_BROKER_DISCONNECT,
    ALERT_DD_BREACH,
    ALERT_HALT,
    ALERT_POSITION_DRIFT,
    ALERT_REBALANCE_FAILED,
    ALERT_STALE_DATA,
    AlertError,
    ChannelError,
    EmailConfig,
    EmailNotificationChannel,
    FileNotificationChannel,
    HARVESTER_ALERT_TYPES,
    HarvesterAlertManager,
    Notification,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    SlackWebhookChannel,
    SlackWebhookConfig,
    enforce_ship_gate,
)


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"
ALT_HASH = "0000000000000000000000000000000000000000000000000000000000000001"


# ---------------------------------------------------------------------------
# Real-disk fixtures (same convention as the other equity test suites).
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".alerts_test_", suffix=".json", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _passing_payload(universe_hash: str = VALID_HASH) -> dict:
    return {
        "asof": "2026-05-29",
        "criteria_source": "test",
        "gate_pass": True,
        "max_dd": 0.229,
        "net_sharpe": 0.921,
        "pipeline_version": "test-v1",
        "positive_years": 13,
        "recommendation": "PASS",
        "thresholds": {
            "max_drawdown": 0.25,
            "min_net_sharpe": 0.4,
            "min_positive_years": 6.0,
            "min_total_years": 10.0,
        },
        "total_years": 17,
        "universe_hash": universe_hash,
    }


def _ship_gate(tmp_path: Path, hash_: str = VALID_HASH) -> Path:
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, _passing_payload(hash_))
    return path


def _build_manager(
    tmp_path: Path,
    *,
    cooldowns: dict | None = None,
    universe_hash: str = VALID_HASH,
) -> tuple[HarvesterAlertManager, FileNotificationChannel, Path]:
    gate = _ship_gate(tmp_path, universe_hash)
    outbox = tmp_path / "outbox"
    file_channel = FileNotificationChannel(outbox)
    state_path = tmp_path / "alert_state.json"
    mgr = HarvesterAlertManager(
        channels=[file_channel],
        ship_gate_path=gate,
        universe_hash=universe_hash,
        state_path=state_path,
        cooldowns=cooldowns,
    )
    return mgr, file_channel, outbox


def _read_outbox(outbox: Path) -> List[dict]:
    files = sorted(outbox.glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


# ---------------------------------------------------------------------------
# Ship-gate guard tests.
# ---------------------------------------------------------------------------


def test_ship_gate_passes_for_matching_hash(tmp_path: Path) -> None:
    enforce_ship_gate(_ship_gate(tmp_path), VALID_HASH)


def test_ship_gate_refuses_when_missing(tmp_path: Path) -> None:
    with pytest.raises(AlertError, match="ship gate not found"):
        enforce_ship_gate(tmp_path / "SHIP_GATE.json", VALID_HASH)


def test_ship_gate_refuses_when_not_json(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(AlertError, match="cannot read"):
        enforce_ship_gate(path, VALID_HASH)


def test_ship_gate_refuses_when_payload_not_object(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, {"_": "_"})  # written as object then overwritten
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(AlertError, match="not a JSON object"):
        enforce_ship_gate(path, VALID_HASH)


def test_ship_gate_refuses_when_gate_pass_false(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(path, payload)
    with pytest.raises(AlertError, match="gate_pass="):
        enforce_ship_gate(path, VALID_HASH)


def test_ship_gate_refuses_when_gate_pass_truthy_but_not_true(
    tmp_path: Path,
) -> None:
    for truthy in (1, "true", "True", [True], {"x": 1}):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = truthy
        _atomic_write_json(path, payload)
        with pytest.raises(AlertError, match="gate_pass="):
            enforce_ship_gate(path, VALID_HASH)


def test_ship_gate_refuses_when_hash_missing(tmp_path: Path) -> None:
    path = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload.pop("universe_hash")
    _atomic_write_json(path, payload)
    with pytest.raises(AlertError, match="universe_hash missing"):
        enforce_ship_gate(path, VALID_HASH)


def test_ship_gate_refuses_when_hash_mismatched(tmp_path: Path) -> None:
    path = _ship_gate(tmp_path, VALID_HASH)
    with pytest.raises(AlertError, match="universe_hash mismatch"):
        enforce_ship_gate(path, ALT_HASH)


# ---------------------------------------------------------------------------
# Constructor refusals.
# ---------------------------------------------------------------------------


def test_manager_refuses_to_arm_without_passing_gate(tmp_path: Path) -> None:
    # gate file doesn't exist
    outbox = tmp_path / "outbox"
    with pytest.raises(AlertError, match="ship gate not found"):
        HarvesterAlertManager(
            channels=[FileNotificationChannel(outbox)],
            ship_gate_path=tmp_path / "SHIP_GATE.json",
            universe_hash=VALID_HASH,
            state_path=tmp_path / "alert_state.json",
        )


def test_manager_refuses_to_arm_with_failing_gate(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(AlertError, match="gate_pass="):
        HarvesterAlertManager(
            channels=[FileNotificationChannel(tmp_path / "outbox")],
            ship_gate_path=gate,
            universe_hash=VALID_HASH,
            state_path=tmp_path / "alert_state.json",
        )


def test_manager_refuses_to_arm_with_mismatched_hash(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path, VALID_HASH)
    with pytest.raises(AlertError, match="universe_hash mismatch"):
        HarvesterAlertManager(
            channels=[FileNotificationChannel(tmp_path / "outbox")],
            ship_gate_path=gate,
            universe_hash=ALT_HASH,
            state_path=tmp_path / "alert_state.json",
        )


def test_manager_refuses_no_channels(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    with pytest.raises(AlertError, match="at least one channel"):
        HarvesterAlertManager(
            channels=[],
            ship_gate_path=gate,
            universe_hash=VALID_HASH,
            state_path=tmp_path / "alert_state.json",
        )


def test_manager_refuses_unknown_alert_type_in_cooldowns(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    with pytest.raises(AlertError, match="unknown alert type"):
        HarvesterAlertManager(
            channels=[FileNotificationChannel(tmp_path / "outbox")],
            ship_gate_path=gate,
            universe_hash=VALID_HASH,
            state_path=tmp_path / "alert_state.json",
            cooldowns={"consecutive_losses": 60.0},
        )


# ---------------------------------------------------------------------------
# Vocabulary — the harvester conditions REPLACE the directional ones.
# ---------------------------------------------------------------------------


def test_harvester_alert_vocabulary_is_exactly_the_six_conditions() -> None:
    assert HARVESTER_ALERT_TYPES == frozenset(
        {
            ALERT_HALT,
            ALERT_BROKER_DISCONNECT,
            ALERT_DD_BREACH,
            ALERT_REBALANCE_FAILED,
            ALERT_STALE_DATA,
            ALERT_POSITION_DRIFT,
        }
    )
    assert "consecutive_losses" not in HARVESTER_ALERT_TYPES
    assert "win_rate_drop" not in HARVESTER_ALERT_TYPES


def test_manager_notify_refuses_directional_alert_types(tmp_path: Path) -> None:
    mgr, _, _ = _build_manager(tmp_path)
    with pytest.raises(AlertError, match="unknown alert type"):
        mgr.notify(
            "consecutive_losses",
            SEVERITY_WARNING,
            "this should not fire",
        )
    with pytest.raises(AlertError, match="unknown alert type"):
        mgr.notify(
            "win_rate_drop",
            SEVERITY_WARNING,
            "this should not fire either",
        )


# ---------------------------------------------------------------------------
# Simulated breaches — each must emit a payload through the channel.
# ---------------------------------------------------------------------------


def test_dd_breach_emits_notification_payload(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    notification = mgr.check_dd_breach(current_dd=0.27, threshold=0.20)

    assert notification is not None
    assert notification.alert_type == ALERT_DD_BREACH
    assert notification.severity == SEVERITY_CRITICAL
    assert notification.universe_hash == VALID_HASH
    assert "0.20" not in notification.message  # we format as percent
    assert "27" in notification.message
    assert notification.payload["current_dd"] == pytest.approx(0.27)
    assert notification.payload["threshold"] == pytest.approx(0.20)

    # And the channel wrote the payload to disk.
    posts = _read_outbox(outbox)
    assert len(posts) == 1
    record = posts[0]
    assert record["alert_type"] == ALERT_DD_BREACH
    assert record["universe_hash"] == VALID_HASH
    assert record["payload"]["current_dd"] == pytest.approx(0.27)


def test_dd_breach_below_threshold_does_not_fire(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    assert mgr.check_dd_breach(current_dd=0.10, threshold=0.20) is None
    assert _read_outbox(outbox) == []


def test_halt_fires_only_when_halted(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    assert mgr.check_halt(halted=False) is None
    assert _read_outbox(outbox) == []

    notif = mgr.check_halt(halted=True, reason="operator pulled the cord")
    assert notif is not None
    assert notif.alert_type == ALERT_HALT
    assert notif.severity == SEVERITY_CRITICAL
    assert "operator pulled the cord" in notif.message

    posts = _read_outbox(outbox)
    assert len(posts) == 1
    assert posts[0]["alert_type"] == ALERT_HALT


def test_broker_disconnect_fires_only_when_disconnected(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    assert mgr.check_broker_disconnect(connected=True) is None
    notif = mgr.check_broker_disconnect(connected=False, broker_name="ibkr")
    assert notif is not None
    assert notif.alert_type == ALERT_BROKER_DISCONNECT
    assert "ibkr" in notif.message
    assert _read_outbox(outbox)[0]["payload"]["broker_name"] == "ibkr"


def test_rebalance_failed_always_fires_and_bypasses_cooldown(
    tmp_path: Path,
) -> None:
    mgr, _, outbox = _build_manager(
        tmp_path, cooldowns={ALERT_REBALANCE_FAILED: 86400}
    )
    n1 = mgr.check_rebalance_failed("broker returned timeout for AAPL")
    n2 = mgr.check_rebalance_failed("subsequent failure during retry")
    assert n1 is not None and n2 is not None
    posts = _read_outbox(outbox)
    assert len(posts) == 2
    assert "AAPL" in posts[0]["message"]
    assert "retry" in posts[1]["message"]


def test_rebalance_failed_refuses_empty_string(tmp_path: Path) -> None:
    mgr, _, _ = _build_manager(tmp_path)
    with pytest.raises(AlertError, match="non-empty error string"):
        mgr.check_rebalance_failed("   ")


def test_stale_data_fires_when_age_exceeds_threshold(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    fresh = now - timedelta(minutes=4)
    stale = now - timedelta(minutes=12)

    assert mgr.check_stale_data(fresh, now, threshold_minutes=5.0) is None
    notif = mgr.check_stale_data(stale, now, threshold_minutes=5.0)
    assert notif is not None
    assert notif.alert_type == ALERT_STALE_DATA
    assert "12.0" in notif.message or "12 " in notif.message
    posts = _read_outbox(outbox)
    assert len(posts) == 1
    assert posts[0]["payload"]["threshold_minutes"] == pytest.approx(5.0)


def test_stale_data_refuses_mixed_tz(tmp_path: Path) -> None:
    mgr, _, _ = _build_manager(tmp_path)
    naive = datetime(2026, 6, 21, 12, 0, 0)
    aware = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(AlertError, match="tz-aware or both naive"):
        mgr.check_stale_data(naive, aware, threshold_minutes=5.0)


def test_position_drift_fires_when_max_drift_exceeds_threshold(
    tmp_path: Path,
) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    target = {"AAPL": 0.5, "MSFT": 0.5}
    actual_ok = {"AAPL": 0.51, "MSFT": 0.49}
    actual_bad = {"AAPL": 0.65, "MSFT": 0.35}

    assert (
        mgr.check_position_drift(actual_ok, target, threshold=0.05) is None
    )
    notif = mgr.check_position_drift(actual_bad, target, threshold=0.05)
    assert notif is not None
    assert notif.alert_type == ALERT_POSITION_DRIFT
    assert notif.severity == SEVERITY_WARNING
    assert notif.payload["worst_ticker"] in {"AAPL", "MSFT"}
    posts = _read_outbox(outbox)
    assert len(posts) == 1
    assert posts[0]["payload"]["max_drift"] == pytest.approx(0.15)


def test_position_drift_handles_missing_tickers_on_either_side(
    tmp_path: Path,
) -> None:
    mgr, _, outbox = _build_manager(tmp_path)
    # GOOG appears in target but not actual — drift is the full target weight.
    target = {"AAPL": 0.5, "GOOG": 0.5}
    actual = {"AAPL": 0.5}
    notif = mgr.check_position_drift(actual, target, threshold=0.10)
    assert notif is not None
    assert notif.payload["worst_ticker"] == "GOOG"
    assert notif.payload["max_drift"] == pytest.approx(0.5)
    posts = _read_outbox(outbox)
    assert len(posts) == 1


# ---------------------------------------------------------------------------
# Cooldown debouncing.
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_second_fire_within_window(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(
        tmp_path, cooldowns={ALERT_DD_BREACH: 300.0}
    )
    now = 1_000_000.0
    n1 = mgr.notify(
        ALERT_DD_BREACH,
        SEVERITY_CRITICAL,
        "first breach",
        {"current_dd": 0.21},
        now=now,
    )
    n2 = mgr.notify(
        ALERT_DD_BREACH,
        SEVERITY_CRITICAL,
        "second breach inside cooldown",
        {"current_dd": 0.22},
        now=now + 60.0,
    )
    assert n1 is not None
    assert n2 is None
    assert len(_read_outbox(outbox)) == 1


def test_cooldown_allows_second_fire_after_window(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(
        tmp_path, cooldowns={ALERT_DD_BREACH: 60.0}
    )
    now = 2_000_000.0
    mgr.notify(
        ALERT_DD_BREACH,
        SEVERITY_CRITICAL,
        "first",
        {"current_dd": 0.30},
        now=now,
    )
    second = mgr.notify(
        ALERT_DD_BREACH,
        SEVERITY_CRITICAL,
        "second past cooldown",
        {"current_dd": 0.32},
        now=now + 120.0,
    )
    assert second is not None
    assert len(_read_outbox(outbox)) == 2


def test_cooldown_force_overrides(tmp_path: Path) -> None:
    mgr, _, outbox = _build_manager(
        tmp_path, cooldowns={ALERT_REBALANCE_FAILED: 86400.0}
    )
    now = 3_000_000.0
    a = mgr.notify(
        ALERT_REBALANCE_FAILED,
        SEVERITY_CRITICAL,
        "first failure",
        now=now,
    )
    b = mgr.notify(
        ALERT_REBALANCE_FAILED,
        SEVERITY_CRITICAL,
        "second failure",
        now=now + 10.0,
        force=True,
    )
    assert a is not None and b is not None
    assert len(_read_outbox(outbox)) == 2


# ---------------------------------------------------------------------------
# Atomic state writes — state survives re-construction.
# ---------------------------------------------------------------------------


def test_state_persists_across_manager_reconstruction(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    state_path = tmp_path / "alert_state.json"
    outbox1 = tmp_path / "outbox1"
    outbox2 = tmp_path / "outbox2"
    cooldowns = {ALERT_DD_BREACH: 600.0}

    mgr1 = HarvesterAlertManager(
        channels=[FileNotificationChannel(outbox1)],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=state_path,
        cooldowns=cooldowns,
    )
    now = 5_000_000.0
    mgr1.notify(
        ALERT_DD_BREACH, SEVERITY_CRITICAL, "first", {"current_dd": 0.3}, now=now
    )
    assert state_path.exists()
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["universe_hash"] == VALID_HASH
    assert state_payload["last_fired"][ALERT_DD_BREACH] == pytest.approx(now)

    # Rebuild — should load the cooldown timestamp and suppress immediately.
    mgr2 = HarvesterAlertManager(
        channels=[FileNotificationChannel(outbox2)],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=state_path,
        cooldowns=cooldowns,
    )
    suppressed = mgr2.notify(
        ALERT_DD_BREACH,
        SEVERITY_CRITICAL,
        "second within window",
        {"current_dd": 0.31},
        now=now + 30.0,
    )
    assert suppressed is None
    assert _read_outbox(outbox2) == []


def test_state_corruption_is_handled_gracefully(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    state_path = tmp_path / "alert_state.json"
    state_path.write_text("not valid json", encoding="utf-8")

    # Constructor must not crash on a corrupt state file.
    mgr = HarvesterAlertManager(
        channels=[FileNotificationChannel(tmp_path / "outbox")],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=state_path,
    )
    notif = mgr.check_dd_breach(current_dd=0.30, threshold=0.20)
    assert notif is not None


# ---------------------------------------------------------------------------
# Mid-run ship-gate invalidation — next call fails closed.
# ---------------------------------------------------------------------------


def test_notify_re_checks_ship_gate_on_every_call(tmp_path: Path) -> None:
    mgr, _, _ = _build_manager(tmp_path)
    # Invalidate the gate on disk by flipping gate_pass to False.
    bad = _passing_payload()
    bad["gate_pass"] = False
    _atomic_write_json(mgr.ship_gate_path, bad)
    with pytest.raises(AlertError, match="gate_pass="):
        mgr.check_dd_breach(current_dd=0.30, threshold=0.20)


def test_notify_re_checks_universe_hash_on_every_call(tmp_path: Path) -> None:
    mgr, _, _ = _build_manager(tmp_path)
    # Rewrite the gate with a different hash — must fail closed.
    bad = _passing_payload(ALT_HASH)
    _atomic_write_json(mgr.ship_gate_path, bad)
    with pytest.raises(AlertError, match="universe_hash mismatch"):
        mgr.check_halt(halted=True, reason="should refuse")


# ---------------------------------------------------------------------------
# Email + Slack channels in dry-run mode — real audit files on disk.
# ---------------------------------------------------------------------------


def test_email_channel_writes_audit_record_in_dry_run(tmp_path: Path) -> None:
    audit_dir = tmp_path / "email-audit"
    cfg = EmailConfig(
        smtp_host="smtp.example.invalid",
        smtp_port=587,
        from_addr="bot@example.invalid",
        to_addrs=["ops@example.invalid"],
        use_starttls=True,
        dry_run=True,
        audit_dir=audit_dir,
    )
    ch = EmailNotificationChannel(cfg)
    gate = _ship_gate(tmp_path)
    mgr = HarvesterAlertManager(
        channels=[ch],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=tmp_path / "state.json",
    )
    notif = mgr.check_dd_breach(current_dd=0.30, threshold=0.20)
    assert notif is not None

    audits = sorted(audit_dir.glob("*.email.json"))
    assert len(audits) == 1
    record = json.loads(audits[0].read_text(encoding="utf-8"))
    assert record["notification"]["alert_type"] == ALERT_DD_BREACH
    assert record["subject"].startswith("[CRITICAL] equity-harvester: ")
    assert record["from"] == "bot@example.invalid"
    assert record["to"] == ["ops@example.invalid"]
    assert "current_dd" in record["body"]
    assert record["dry_run"] is True


def test_email_channel_requires_recipient() -> None:
    cfg = EmailConfig(
        smtp_host="smtp.example.invalid",
        smtp_port=587,
        from_addr="bot@example.invalid",
        to_addrs=[],
        dry_run=True,
    )
    with pytest.raises(AlertError, match="at least one recipient"):
        EmailNotificationChannel(cfg)


def test_slack_channel_writes_audit_record_in_dry_run(tmp_path: Path) -> None:
    audit_dir = tmp_path / "slack-audit"
    cfg = SlackWebhookConfig(
        webhook_url="https://hooks.slack.invalid/services/T0/B0/abc",
        channel="#alerts",
        dry_run=True,
        audit_dir=audit_dir,
    )
    ch = SlackWebhookChannel(cfg)
    gate = _ship_gate(tmp_path)
    mgr = HarvesterAlertManager(
        channels=[ch],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=tmp_path / "state.json",
    )
    notif = mgr.check_broker_disconnect(connected=False, broker_name="ibkr")
    assert notif is not None

    audits = sorted(audit_dir.glob("*.slack.json"))
    assert len(audits) == 1
    record = json.loads(audits[0].read_text(encoding="utf-8"))
    assert record["notification"]["alert_type"] == ALERT_BROKER_DISCONNECT
    assert record["slack_payload"]["channel"] == "#alerts"
    assert "BROKER_DISCONNECT" in record["slack_payload"]["text"]
    assert record["dry_run"] is True


def test_slack_channel_requires_webhook_url() -> None:
    with pytest.raises(AlertError, match="webhook_url"):
        SlackWebhookChannel(
            SlackWebhookConfig(webhook_url="", dry_run=True)
        )


# ---------------------------------------------------------------------------
# Channel-failure resilience — one channel failing doesn't kill the others.
# ---------------------------------------------------------------------------


class _FailingChannel:
    """Real (not mocked) channel that always raises on send.

    Concrete class — defined in the test file (not the production tree) and
    not a unittest.mock object. Used to verify that the manager records the
    failure without taking down the rest of the fan-out.
    """

    def __init__(self) -> None:
        self.calls = 0

    def send(self, notification: Notification) -> None:
        self.calls += 1
        raise ChannelError("synthetic channel failure for test")


def test_one_failing_channel_does_not_block_others(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    outbox = tmp_path / "outbox"
    file_ch = FileNotificationChannel(outbox)
    failing_ch = _FailingChannel()
    mgr = HarvesterAlertManager(
        channels=[failing_ch, file_ch],
        ship_gate_path=gate,
        universe_hash=VALID_HASH,
        state_path=tmp_path / "state.json",
    )
    notif = mgr.check_dd_breach(current_dd=0.30, threshold=0.20)
    assert notif is not None
    # The failing channel was called…
    assert failing_ch.calls == 1
    # …but the file channel still delivered the payload.
    assert len(_read_outbox(outbox)) == 1


# ---------------------------------------------------------------------------
# Source-tree hygiene.
# ---------------------------------------------------------------------------


def test_alerts_source_has_no_bare_except() -> None:
    source = (
        Path(__file__).resolve().parent.parent / "src" / "equity" / "alerts.py"
    )
    text = source.read_text(encoding="utf-8")
    # Strip docstrings/comments isn't necessary — we just look for `except:`
    # with optional whitespace immediately followed by colon.
    assert not re.search(r"^\s*except\s*:", text, flags=re.MULTILINE), (
        "src/equity/alerts.py contains a bare except clause"
    )


def test_test_file_imports_no_unittest_mock() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    # Mention in the module docstring / this assertion's literal strings is
    # allowed; an actual import / live use is not.
    assert not re.search(r"^\s*from\s+unittest\.mock\s+import\b", text, re.M)
    assert not re.search(r"^\s*import\s+unittest\.mock\b", text, re.M)
    assert not re.search(r"^\s*from\s+mock\s+import\b", text, re.M)
    assert not re.search(r"^\s*import\s+mock\b", text, re.M)
    # No `@patch(...)` decorator usage at column-0 or indented.
    assert not re.search(r"^\s*@patch\(", text, re.M)
